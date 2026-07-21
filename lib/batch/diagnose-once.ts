import type { ClinicalCase } from "@/lib/schemas/clinical-case";
import {
  APICAL_DIAGNOSES,
  clinicalCaseSchema,
  finalDiagnosisSchema,
  PULPAL_DIAGNOSES,
  TAXONOMY_VERSION,
  type FinalDiagnosis,
} from "@/lib/schemas/clinical-case";
import { createLlmClient } from "@/lib/llm";
import { getDb } from "@/lib/correction-rag/db";
import { serializeCaseCanonical } from "@/lib/correction-rag/format-case";
import { searchSimilarCorrections } from "@/lib/correction-rag/search";
import { loadCurriculum } from "@/lib/prompts/load-curriculum";
import {
  abstentionResult,
  runClinicalGate,
  verifyDiagnosisProposal,
} from "@/lib/endodontic-agent/clinical-verifier";
import { isCorrectionRagEnabled, isCtModuleEnabled } from "@/lib/features";
import { loadCtSupport } from "@/lib/ct-analysis/store";
import { z } from "zod";

export type DiagnosisPipelineOptions = {
  skipRag?: boolean;
};

export type DiagnosisPipelineResult = {
  final: FinalDiagnosis;
  statusMessage: string;
  gateStatus: FinalDiagnosis["status"];
  ragMatchCount: number;
};

function ragTopK(): number {
  const n = parseInt(process.env.RAG_TOP_K ?? "5", 10);
  if (Number.isNaN(n)) return 5;
  return Math.min(20, Math.max(1, n));
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
  if (!hits.length) return "";
  const threshold = Number.parseFloat(process.env.RAG_MIN_SCORE ?? "0.78");
  const filtered = hits.filter((h) => h.score >= threshold);
  if (!filtered.length) return "";
  return [
    "## Approved correction library (soft prior only — cannot override prerequisites)",
    ...filtered.map((h, i) =>
      [
        `--- Memory ${i + 1} (similarity ${h.score.toFixed(3)}) ---`,
        h.caseCanonical,
        `Agent: ${h.agentPulpal} / ${h.agentApical}`,
        `Adjudicated: ${h.correctedPulpal} / ${h.correctedApical}`,
        `Reasoning: ${h.reasoning}`,
        h.misunderstood ? `Misunderstood: ${h.misunderstood}` : "",
      ]
        .filter(Boolean)
        .join("\n"),
    ),
  ].join("\n\n");
}

const candidateSchema = z.object({
  pulpalDiagnosis: z.enum(PULPAL_DIAGNOSES),
  apicalDiagnosis: z.enum(APICAL_DIAGNOSES),
  evidenceFor: z
    .array(
      z.object({
        claim: z.string(),
        source: z.enum([
          "history",
          "visual",
          "sensibility",
          "mechanical",
          "imaging",
          "periodontal",
          "confounder",
          "ct_support",
          "curriculum",
          "verifier",
        ]),
      }),
    )
    .max(40),
  evidenceAgainst: z
    .array(
      z.object({
        claim: z.string(),
        source: z.enum([
          "history",
          "visual",
          "sensibility",
          "mechanical",
          "imaging",
          "periodontal",
          "confounder",
          "ct_support",
          "curriculum",
          "verifier",
        ]),
      }),
    )
    .max(40)
    .default([]),
  conflicts: z.array(z.string()).max(30).default([]),
  biologicalJustification: z.string(),
  warnings: z.array(z.string()).max(30).optional(),
});

function attachCtSupport(clinicalCase: ClinicalCase): ClinicalCase {
  if (!isCtModuleEnabled() || !clinicalCase.ctAnalysisId) {
    const { ctSupport: _drop, ...rest } = clinicalCase;
    return { ...rest, ctSupport: undefined };
  }
  const stored = loadCtSupport(clinicalCase.ctAnalysisId);
  if (!stored) {
    throw new Error(
      "ctAnalysisId was not found server-side. Re-run CT analysis or clear the CT field.",
    );
  }
  return { ...clinicalCase, ctSupport: stored };
}

export async function runDiagnosisOnce(
  clinicalCaseInput: ClinicalCase,
  options: DiagnosisPipelineOptions = {},
): Promise<DiagnosisPipelineResult> {
  const clinicalCase = attachCtSupport(clinicalCaseInput);
  const gate = runClinicalGate(clinicalCase);

  if (gate.status !== "diagnosable") {
    return {
      final: abstentionResult(gate),
      statusMessage: `Stage 0 gate: ${gate.status}. No AAE enums assigned.`,
      gateStatus: gate.status,
      ragMatchCount: 0,
    };
  }

  const llm = createLlmClient();
  const curriculum = loadCurriculum();
  const caseBlock = serializeCaseCanonical(clinicalCase);
  const caseJson = JSON.stringify(clinicalCase, null, 2);

  let ragContext = "";
  let ragMatchCount = 0;
  if (!options.skipRag && isCorrectionRagEnabled()) {
    try {
      const db = getDb();
      const qVec = await llm.embedText(caseBlock);
      const hits = searchSimilarCorrections(db, qVec, ragTopK());
      ragMatchCount = hits.length;
      ragContext = formatRagBlock(hits);
    } catch {
      /* RAG optional */
    }
  }

  const ctNote = clinicalCase.ctSupport
    ? [
        "Optional CT support is present as RESEARCH CONTEXT only.",
        "It must never establish pulp vitality, endodontic origin, abscess type, or force any apical enum.",
        "candidateLowAttenuationRegion is a candidate region, not a diagnosis.",
        "canalLengthEstimateMm is an anatomical estimate only — never a clinical working length.",
      ].join(" ")
    : "No CT support module data.";

  const system = [
    "You propose AAE 2009 pulpal and apical diagnoses for an untreated permanent tooth.",
    "A deterministic verifier will accept or reject your proposal. Prefer honesty about uncertainty.",
    "",
    curriculum,
    "",
    "## Hard constraints (non-negotiable)",
    "- Do NOT force Pulp Necrosis from Cold− and EPT− alone. Necrosis requires valid repeated testing relative to controls plus supporting findings, without unresolved false-negative confounders.",
    "- Do NOT use linger duration alone to separate Symptomatic vs Asymptomatic Irreversible Pulpitis. AIP depends on asymptomatic deep caries/exposure/trauma patterns.",
    "- Acute Apical Abscess requires rapid-onset swelling with purulence/fluctuance signs.",
    "- Chronic Apical Abscess requires a sinus tract (drainage).",
    "- Imaging/CT alone never establishes endodontic origin.",
    "- Tooth Slooth and transillumination are structural/crack clues, not pulp sensibility.",
    `- ${ctNote}`,
    "",
    ragContext || "(No correction-library memories injected.)",
    "",
    "Return ONLY JSON with keys:",
    `pulpalDiagnosis (one of ${JSON.stringify([...PULPAL_DIAGNOSES])})`,
    `apicalDiagnosis (one of ${JSON.stringify([...APICAL_DIAGNOSES])})`,
    "evidenceFor, evidenceAgainst (arrays of {claim, source})",
    "conflicts (string array)",
    "biologicalJustification (2–6 sentences)",
    "warnings (optional string array)",
  ].join("\n");

  let raw = await llm.chatComplete(
    [
      { role: "system", content: system },
      {
        role: "user",
        content: `Case JSON:\n${caseJson}\n\nSummary:\n${caseBlock}\n\nPropose diagnoses now.`,
      },
    ],
    { responseFormatJson: true },
  );

  let parsed = candidateSchema.safeParse(parseModelJson(raw));
  if (!parsed.success) {
    raw = await llm.chatComplete(
      [
        { role: "system", content: system },
        {
          role: "user",
          content: `Case JSON:\n${caseJson}\n\nPrior invalid JSON error: ${parsed.error.message}\nReturn corrected JSON ONLY.`,
        },
      ],
      { responseFormatJson: true },
    );
    parsed = candidateSchema.safeParse(parseModelJson(raw));
  }
  if (!parsed.success) {
    throw new Error(`Candidate JSON failed schema: ${parsed.error.message}`);
  }

  const proposal = parsed.data;
  const verification = verifyDiagnosisProposal(clinicalCase, gate, {
    pulpalDiagnosis: proposal.pulpalDiagnosis,
    apicalDiagnosis: proposal.apicalDiagnosis,
  });

  if (!verification.ok) {
    // Deterministic abstention on verifier rejection — do not let Stage 4 "override".
    return {
      final: {
        ...abstentionResult({
          ...gate,
          status: "conflicting_data",
          conflicts: [...gate.conflicts, verification.reason],
        }),
        evidenceFor: [...gate.evidenceFor, ...proposal.evidenceFor],
        evidenceAgainst: [
          ...gate.evidenceAgainst,
          ...proposal.evidenceAgainst,
          { claim: verification.reason, source: "verifier" },
        ],
        biologicalJustification: proposal.biologicalJustification,
      },
      statusMessage: `Verifier rejected proposal: ${verification.reason}`,
      gateStatus: "conflicting_data",
      ragMatchCount,
    };
  }

  const final: FinalDiagnosis = finalDiagnosisSchema.parse({
    taxonomyVersion: TAXONOMY_VERSION,
    status: "diagnosable",
    pulpalDiagnosis: proposal.pulpalDiagnosis,
    apicalDiagnosis: proposal.apicalDiagnosis,
    finalDiagnosisLine: `${proposal.pulpalDiagnosis} with ${proposal.apicalDiagnosis}`,
    evidenceFor: [...gate.evidenceFor, ...proposal.evidenceFor],
    evidenceAgainst: [...gate.evidenceAgainst, ...proposal.evidenceAgainst],
    conflicts: [...gate.conflicts, ...proposal.conflicts],
    missingRequiredData: [],
    recommendedNextTests: gate.recommendedNextTests,
    biologicalJustification: proposal.biologicalJustification,
    warnings: [
      ...(proposal.warnings ?? []),
      ...gate.warnings,
      "Clinician confirmation required. Educational decision support only.",
    ],
    clinicianConfirmationRequired: true,
  });

  return {
    final,
    statusMessage: "Gate passed; proposal verified.",
    gateStatus: "diagnosable",
    ragMatchCount,
  };
}

const extractionResultSchema = z.object({
  eligible: z.boolean(),
  ineligibleReason: z.string().optional(),
  case: clinicalCaseSchema.optional(),
});

export type RowExtraction = z.infer<typeof extractionResultSchema>;

const EXTRACTION_PROMPT = `Convert chart text into the ClinicalCase JSON used by this endodontic MVP (taxonomy AAE_2009, untreated permanent teeth only).

Use exact enum strings from the schema. Never convert missing findings to "absent" — use unknown / not_performed / not_assessed.
Set eligible:false for implant-only rows, previously treated teeth, primary teeth, or insufficient endodontic data.
Tooth Slooth and transillumination are structural, not sensibility tests.`;

export async function extractClinicalCaseFromRowText(
  rowText: string,
): Promise<RowExtraction> {
  const llm = createLlmClient();
  const raw = await llm.chatComplete(
    [
      { role: "system", content: EXTRACTION_PROMPT },
      {
        role: "user",
        content: `Extract structured findings from this chart row:\n\n${rowText}`,
      },
    ],
    { responseFormatJson: true },
  );
  const parsed = extractionResultSchema.safeParse(parseModelJson(raw));
  if (!parsed.success) {
    return {
      eligible: false,
      ineligibleReason: `Extraction failed: ${parsed.error.message}`,
    };
  }
  if (parsed.data.eligible && !parsed.data.case) {
    return {
      eligible: false,
      ineligibleReason: "Marked eligible but no case object returned",
    };
  }
  return parsed.data;
}

export async function embedCorrectionDocument(doc: string): Promise<number[]> {
  return createLlmClient().embedText(doc);
}
