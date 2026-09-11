import { serializeCaseCanonical } from "@/lib/correction-rag/format-case";
import { searchSimilarCorrections } from "@/lib/correction-rag/store";
import { createLlmClient } from "@/lib/llm";
import { loadCurriculum } from "@/lib/prompts/load-curriculum";
import {
  STAGE_TITLES,
  diagnosticCriticSystemPrompt,
  mimicConflictSystemPrompt,
  stage1SystemPrompt,
  stage2SystemPrompt,
  stage3SystemPrompt,
  stage4SystemPrompt,
} from "@/lib/prompts/agent-stages";
import {
  TAXONOMY_VERSION,
  type ClinicalCase,
  finalDiagnosisSchema,
  type FinalDiagnosis,
} from "@/lib/schemas/clinical-case";
import {
  abstentionResult,
  runClinicalGate,
  verifyDiagnosisProposal,
} from "@/lib/endodontic-agent/clinical-verifier";
import { isCorrectionRagEnabled } from "@/lib/features";

export type DiagnosisPipelineOptions = {
  skipRag?: boolean;
};

export type DiagnosisPipelineResult = {
  final: FinalDiagnosis;
  statusMessage: string;
  gateStatus: FinalDiagnosis["status"];
  ragMatchCount: number;
};

export type StreamEvent =
  | { type: "status"; message: string }
  | { type: "rag"; matchCount: number; contextBlock: string }
  | { type: "stage"; stage: 1 | 2 | 3 | 4; title: string; content: string }
  | {
      type: "evidence";
      evidenceFor: FinalDiagnosis["evidenceFor"];
      evidenceAgainst: FinalDiagnosis["evidenceAgainst"];
      conflicts: string[];
      missingRequiredData: string[];
      recommendedNextTests: string[];
    }
  | { type: "final"; result: FinalDiagnosis }
  | { type: "error"; message: string };

function ragTopK(): number {
  const n = parseInt(process.env.RAG_TOP_K ?? "5", 10);
  if (Number.isNaN(n)) return 5;
  return Math.min(20, Math.max(1, n));
}

function ragMinScore(): number {
  const n = Number.parseFloat(process.env.RAG_MIN_SCORE ?? "0.78");
  return Number.isFinite(n) ? n : 0.78;
}

function formatRagBlock(
  hits: Array<{
    score: number;
    caseCanonical: string;
    agentPulpal: string;
    agentApical: string;
    correctedPulpal: string;
    correctedApical: string;
    reasoning: string;
    misunderstood: string | null;
  }>,
): string {
  const filtered = hits.filter((h) => h.score >= ragMinScore());
  if (!filtered.length) return "";
  const parts = filtered.map((h, i) => {
    const mu = h.misunderstood?.trim()
      ? `\nWhat was misunderstood: ${h.misunderstood.trim()}`
      : "";
    return [
      `--- Correction ${i + 1} (similarity ${h.score.toFixed(3)}; dual-reviewed approved) ---`,
      "Case:",
      h.caseCanonical,
      "",
      `Agent said: Pulpal="${h.agentPulpal}", Apical="${h.agentApical}"`,
      `Clinician corrected to: Pulpal="${h.correctedPulpal}", Apical="${h.correctedApical}"`,
      "",
      "Clinician reasoning (do not repeat this error pattern unless clearly justified by new findings):",
      h.reasoning,
      mu,
    ].join("\n");
  });
  return [
    "## Clinician corrections from similar cases",
    "These are dual-reviewed, approved, PHI-free teaching examples at the current taxonomy.",
    "Unreviewed corrections are evaluation cases only and are never retrieved here.",
    "Weight them as soft priors. Curriculum and raw inputs override corrections on conflicts.",
    "",
    ...parts,
  ].join("\n");
}

function parseModelJson(raw: string): unknown {
  let s = raw.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```(?:json)?\s*\n?/i, "");
    const end = s.lastIndexOf("```");
    if (end !== -1) s = s.slice(0, end).trim();
  }
  return JSON.parse(s);
}

function emitEvidence(gate: ReturnType<typeof runClinicalGate>, extra?: Partial<FinalDiagnosis>): Extract<StreamEvent, { type: "evidence" }> {
  return {
    type: "evidence",
    evidenceFor: extra?.evidenceFor ?? gate.evidenceFor,
    evidenceAgainst: extra?.evidenceAgainst ?? gate.evidenceAgainst,
    conflicts: extra?.conflicts ?? gate.conflicts,
    missingRequiredData: extra?.missingRequiredData ?? gate.missingRequiredData,
    recommendedNextTests: extra?.recommendedNextTests ?? gate.recommendedNextTests,
  };
}

/**
 * Agentic multi-stage diagnosis pipeline:
 * Stage 0 gate → RAG (approved-only) → Stage 1 interpretation →
 * independent pulpal / apical / mimic → synthesis → diagnostic critic →
 * deterministic verifier.
 */
export async function* runDiagnosisPipeline(
  clinicalCaseInput: ClinicalCase,
  options: DiagnosisPipelineOptions = {},
): AsyncGenerator<StreamEvent> {
  try {
    const clinicalCase = clinicalCaseInput;
    yield { type: "status", message: "Running Stage 0 scope/validity gate…" };

    const gate = runClinicalGate(clinicalCase);
    if (gate.status !== "diagnosable") {
      const final = abstentionResult(gate);
      yield { type: "status", message: `Stage 0 gate: ${gate.status}. Abstaining.` };
      yield emitEvidence(gate, final);
      yield { type: "final", result: final };
      return;
    }

    const llm = createLlmClient();
    const curriculum = loadCurriculum();
    const caseBlock = serializeCaseCanonical(clinicalCase);
    const caseJson = JSON.stringify(clinicalCase, null, 2);

    let ragContext = "";
    let ragMatchCount = 0;
    if (!options.skipRag && isCorrectionRagEnabled()) {
      try {
        const qVec = await llm.embedText(caseBlock);
        // Governance: approved + current taxonomy + ≥2 reviewers + no PHI
        const hits = searchSimilarCorrections(qVec, ragTopK(), {
          approvalStatus: "approved",
          taxonomyVersion: TAXONOMY_VERSION,
          minReviewerCount: 2,
          allowPhi: false,
        });
        ragMatchCount = hits.length;
        ragContext = formatRagBlock(hits);
        yield {
          type: "rag",
          matchCount: ragMatchCount,
          contextBlock: ragContext,
        };
      } catch {
        yield { type: "rag", matchCount: 0, contextBlock: "" };
      }
    } else {
      yield {
        type: "rag",
        matchCount: 0,
        contextBlock: "",
      };
    }

    yield { type: "status", message: "Stage 1 — interpreting tests…" };
    const stage1Content = await llm.chatComplete([
      {
        role: "system",
        content: stage1SystemPrompt(curriculum, ragContext),
      },
      {
        role: "user",
        content: `Clinical case (structured JSON):\n${caseJson}\n\nHuman-readable summary:\n${caseBlock}`,
      },
    ]);
    yield {
      type: "stage",
      stage: 1,
      title: STAGE_TITLES[1],
      content: stage1Content,
    };

    // Independent specialists after interpretation — no pulpal→apical anchoring.
    yield {
      type: "status",
      message: "Running independent pulpal, apical, and mimic/conflict agents…",
    };
    const sharedEvidenceUser = [
      `Case JSON:\n${caseJson}`,
      `Summary:\n${caseBlock}`,
      `Stage 1 analysis:\n${stage1Content}`,
      "Imaging alone cannot establish pulp vitality.",
    ].join("\n\n");

    const [stage2Content, stage3Content, mimicContent] = await Promise.all([
      llm.chatComplete([
        { role: "system", content: stage2SystemPrompt() },
        { role: "user", content: sharedEvidenceUser },
      ]),
      llm.chatComplete([
        { role: "system", content: stage3SystemPrompt() },
        { role: "user", content: sharedEvidenceUser },
      ]),
      llm.chatComplete([
        { role: "system", content: mimicConflictSystemPrompt() },
        { role: "user", content: sharedEvidenceUser },
      ]),
    ]);

    yield {
      type: "stage",
      stage: 2,
      title: STAGE_TITLES[2],
      content: stage2Content,
    };
    yield {
      type: "stage",
      stage: 3,
      title: STAGE_TITLES[3],
      content: `${stage3Content}\n\n--- Mimic/conflict (independent) ---\n${mimicContent}`,
    };

    yield { type: "status", message: "Stage 4 — synthesis (coherence check)…" };
    const stage4System = stage4SystemPrompt(curriculum, ragContext);
    const stage4User = [
      `Case JSON:\n${caseJson}`,
      `Summary:\n${caseBlock}`,
      `Stage 1:\n${stage1Content}`,
      `Independent pulpal agent:\n${stage2Content}`,
      `Independent apical agent:\n${stage3Content}`,
      `Independent mimic/conflict agent:\n${mimicContent}`,
      "Gate warnings:",
      ...gate.warnings,
      "Synthesize only after checking coherence among independent agents.",
      "Produce the JSON object now.",
    ].join("\n\n");

    let raw = await llm.chatComplete(
      [
        { role: "system", content: stage4System },
        { role: "user", content: stage4User },
      ],
      { responseFormatJson: true },
    );

    let attempt = finalDiagnosisSchema.safeParse(parseModelJson(raw));
    if (!attempt.success) {
      raw = await llm.chatComplete(
        [
          { role: "system", content: stage4System },
          { role: "user", content: stage4User },
          {
            role: "user",
            content: `Your previous output failed validation: ${attempt.error.message}\nReturn corrected JSON ONLY.`,
          },
        ],
        { responseFormatJson: true },
      );
      attempt = finalDiagnosisSchema.safeParse(parseModelJson(raw));
    }
    if (!attempt.success) {
      yield {
        type: "error",
        message: `Final JSON failed schema: ${attempt.error.message}`,
      };
      return;
    }

    let proposal = attempt.data;
    proposal = finalDiagnosisSchema.parse({
      ...proposal,
      taxonomyVersion: TAXONOMY_VERSION,
      clinicianConfirmationRequired: true,
      evidenceFor: [...gate.evidenceFor, ...proposal.evidenceFor],
      evidenceAgainst: [...gate.evidenceAgainst, ...proposal.evidenceAgainst],
      conflicts: [...new Set([...gate.conflicts, ...proposal.conflicts])],
      missingRequiredData: [
        ...new Set([
          ...gate.missingRequiredData,
          ...proposal.missingRequiredData,
        ]),
      ],
      recommendedNextTests: [
        ...new Set([
          ...gate.recommendedNextTests,
          ...proposal.recommendedNextTests,
        ]),
      ],
      warnings: [
        ...(proposal.warnings ?? []),
        ...gate.warnings,
        "Clinician confirmation required. Educational decision support only.",
      ],
    });

    yield {
      type: "stage",
      stage: 4,
      title: STAGE_TITLES[4],
      content: JSON.stringify(proposal, null, 2),
    };

    // Diagnostic critic between synthesis and deterministic verifier
    yield { type: "status", message: "Diagnostic critic reviewing synthesis…" };
    const criticRaw = await llm.chatComplete([
      { role: "system", content: diagnosticCriticSystemPrompt() },
      {
        role: "user",
        content: [
          `Stage 1:\n${stage1Content}`,
          `Pulpal:\n${stage2Content}`,
          `Apical:\n${stage3Content}`,
          `Mimic/conflict:\n${mimicContent}`,
          `Synthesizer JSON:\n${JSON.stringify(proposal, null, 2)}`,
        ].join("\n\n"),
      },
    ]);
    const criticContradicted = /^Critic:\s*contradicted/im.test(criticRaw.trim());
    if (criticContradicted) {
      const rejected = abstentionResult({
        ...gate,
        status: "conflicting_data",
        conflicts: [
          ...gate.conflicts,
          `Diagnostic critic contradicted synthesis: ${criticRaw.slice(0, 400)}`,
        ],
      });
      const final: FinalDiagnosis = {
        ...rejected,
        evidenceFor: proposal.evidenceFor,
        evidenceAgainst: [
          ...proposal.evidenceAgainst,
          {
            claim: "Diagnostic critic contradicted synthesizer coherence",
            source: "verifier",
          },
        ],
        biologicalJustification: proposal.biologicalJustification,
        warnings: [
          ...(proposal.warnings ?? []),
          "Diagnostic critic rejected proposal before deterministic verifier.",
        ],
      };
      yield { type: "status", message: "Diagnostic critic contradicted synthesis." };
      yield emitEvidence(gate, final);
      yield { type: "final", result: final };
      return;
    }

    const verification = verifyDiagnosisProposal(clinicalCase, gate, {
      status: proposal.status,
      pulpalDiagnosis: proposal.pulpalDiagnosis,
      apicalDiagnosis: proposal.apicalDiagnosis,
    });

    if (!verification.ok) {
      const rejected = abstentionResult({
        ...gate,
        status: "conflicting_data",
        conflicts: [...gate.conflicts, verification.reason],
      });
      const final: FinalDiagnosis = {
        ...rejected,
        evidenceFor: proposal.evidenceFor,
        evidenceAgainst: [
          ...proposal.evidenceAgainst,
          { claim: verification.reason, source: "verifier" },
        ],
        biologicalJustification: proposal.biologicalJustification,
        warnings: [
          ...(proposal.warnings ?? []),
          `Verifier rejected proposal: ${verification.reason}`,
        ],
      };
      yield { type: "status", message: `Verifier rejected: ${verification.reason}` };
      yield emitEvidence(gate, final);
      yield { type: "final", result: final };
      return;
    }

    if (proposal.status !== "diagnosable") {
      const final: FinalDiagnosis = {
        ...proposal,
        pulpalDiagnosis: null,
        apicalDiagnosis: null,
        finalDiagnosisLine: null,
        clinicianConfirmationRequired: true,
      };
      yield emitEvidence(gate, final);
      yield { type: "final", result: final };
      return;
    }

    const final: FinalDiagnosis = {
      ...proposal,
      status: "diagnosable",
      finalDiagnosisLine:
        proposal.finalDiagnosisLine ??
        `${proposal.pulpalDiagnosis} with ${proposal.apicalDiagnosis}`,
      clinicianConfirmationRequired: true,
      warnings: [
        ...(proposal.warnings ?? []),
        "Diagnostic critic: passed (or abstained without contradiction).",
      ],
    };

    yield { type: "status", message: "Diagnosis complete — clinician confirmation required." };
    yield emitEvidence(gate, final);
    yield { type: "final", result: final };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Diagnosis failed",
    };
  }
}

/** Collect the final result from the streaming pipeline (batch / tests). */
export async function runDiagnosisOnce(
  clinicalCase: ClinicalCase,
  options: DiagnosisPipelineOptions = {},
): Promise<{
  final: FinalDiagnosis;
  statusMessage: string;
  gateStatus: FinalDiagnosis["status"];
  ragMatchCount: number;
}> {
  let final: FinalDiagnosis | null = null;
  let statusMessage = "";
  let ragMatchCount = 0;
  for await (const ev of runDiagnosisPipeline(clinicalCase, options)) {
    if (ev.type === "status") statusMessage = ev.message;
    if (ev.type === "rag") ragMatchCount = ev.matchCount;
    if (ev.type === "final") final = ev.result;
    if (ev.type === "error") throw new Error(ev.message);
  }
  if (!final) throw new Error("Pipeline ended without a final diagnosis");
  return {
    final,
    statusMessage,
    gateStatus: final.status,
    ragMatchCount,
  };
}

/** Embed the full correction document for storage (RAG ingest). */
export async function embedCorrectionDocument(doc: string): Promise<number[]> {
  return createLlmClient().embedText(doc);
}
