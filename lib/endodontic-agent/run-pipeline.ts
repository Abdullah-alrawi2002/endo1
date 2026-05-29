import { getDb } from "@/lib/correction-rag/db";
import { serializeCaseCanonical } from "@/lib/correction-rag/format-case";
import { searchSimilarCorrections } from "@/lib/correction-rag/search";
import { createLlmClient } from "@/lib/llm";
import { loadCurriculum } from "@/lib/prompts/load-curriculum";
import {
  APICAL_DIAGNOSES,
  type ClinicalCase,
  finalDiagnosisSchema,
  type FinalDiagnosis,
  PULPAL_DIAGNOSES,
} from "@/lib/schemas/clinical-case";

export type StreamEvent =
  | { type: "rag"; matchCount: number; contextBlock: string }
  | { type: "stage"; stage: 1 | 2 | 3 | 4; title: string; content: string }
  | { type: "final"; result: FinalDiagnosis }
  | { type: "error"; message: string };

function ragTopK(): number {
  const n = parseInt(process.env.RAG_TOP_K ?? "5", 10);
  if (Number.isNaN(n)) return 5;
  return Math.min(20, Math.max(1, n));
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
  const parts = hits.map((h, i) => {
    const mu = h.misunderstood?.trim()
      ? `\nWhat was misunderstood: ${h.misunderstood.trim()}`
      : "";
    return [
      `--- Correction ${i + 1} (similarity ${h.score.toFixed(3)}) ---`,
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
    "These are institutional teaching examples. Weight them heavily in your reasoning.",
    "If any correction conflicts with immutable test logic (e.g., Cold negative + EPT negative still implying vital pulp), the curriculum and raw inputs override the correction—state that conflict explicitly.",
    "",
    ...parts,
  ].join("\n");
}

/** Strip optional ```json fences (Claude / some Gemini outputs). */
function parseModelJson(raw: string): unknown {
  let s = raw.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```(?:json)?\s*\n?/i, "");
    const end = s.lastIndexOf("```");
    if (end !== -1) s = s.slice(0, end).trim();
  }
  return JSON.parse(s);
}

const STAGE4_JSON_INSTRUCTION = `Return ONLY a JSON object with these exact keys:
- "pulpalDiagnosis": one of ${JSON.stringify([...PULPAL_DIAGNOSES])}
- "apicalDiagnosis": one of ${JSON.stringify([...APICAL_DIAGNOSES])}
- "finalDiagnosisLine": a single line combining pulpal + apical (student-style wording)
- "biologicalJustification": 2–6 sentences explaining mechanisms at dental-student depth
- "warnings": optional string array (may be empty or omitted)

The pulpal and apical strings must match exactly (including capitalization).`;

export async function* runDiagnosisPipeline(
  clinicalCase: ClinicalCase,
): AsyncGenerator<StreamEvent> {
  const llm = createLlmClient();
  const curriculum = loadCurriculum();
  const caseBlock = serializeCaseCanonical(clinicalCase);
  const caseJson = JSON.stringify(clinicalCase, null, 2);

  let ragContext = "";
  try {
    const db = getDb();
    const qVec = await llm.embedText(caseBlock);
    const hits = searchSimilarCorrections(db, qVec, ragTopK());
    ragContext = formatRagBlock(hits);
    yield {
      type: "rag",
      matchCount: hits.length,
      contextBlock: ragContext,
    };
  } catch {
    yield { type: "rag", matchCount: 0, contextBlock: "" };
  }

  const stage1System = [
    "You are an expert endodontic educator AI. Follow the curriculum exactly.",
    "",
    curriculum,
    "",
    ragContext || "(No similar stored corrections were retrieved.)",
    "",
    "Stage 1 task — Pulpal + test biology framing:",
    "1) Summarize Cold and EPT and what they imply biologically.",
    "2) Summarize Percussion, Palpation, and PARL and what they imply outside the root.",
    "3) Note any conflicts, missing optional information, or caveats.",
    "Do not output the final pulpal/apical labels yet (those come in later stages).",
  ].join("\n");

  let stage1Content: string;
  try {
    stage1Content = await llm.chatComplete([
      { role: "system", content: stage1System },
      {
        role: "user",
        content: `Clinical case (structured JSON):\n${caseJson}\n\nHuman-readable summary:\n${caseBlock}`,
      },
    ]);
    yield {
      type: "stage",
      stage: 1,
      title: "Stage 1 — Test interpretation",
      content: stage1Content,
    };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Stage 1 failed",
    };
    return;
  }

  const stage2System = [
    "You are the pulpal diagnostician.",
    `You MUST choose exactly ONE pulpal diagnosis from this list (exact spelling): ${PULPAL_DIAGNOSES.join(" | ")}`,
    "Use the curriculum pulpal rules. Respect hard logic: when EPT was performed, if clinical.cold is negative AND clinical.ept is negative, pulpal must be Pulp Necrosis unless explicit user notes document a false-negative scenario.",
    "Output format: first line exactly: Pulpal Diagnosis: <one of the allowed strings>",
    "Then 4–10 sentences explaining your reasoning with citations to cold/EPT and optional fields.",
  ].join("\n");

  let stage2Content: string;
  try {
    stage2Content = await llm.chatComplete([
      { role: "system", content: stage2System },
      {
        role: "user",
        content: `Case JSON:\n${caseJson}\n\nSummary:\n${caseBlock}\n\nStage 1 analysis:\n${stage1Content}`,
      },
    ]);
    yield {
      type: "stage",
      stage: 2,
      title: "Stage 2 — Pulpal diagnosis",
      content: stage2Content,
    };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Stage 2 failed",
    };
    return;
  }

  const stage3System = [
    "You are the apical diagnostician.",
    `You MUST choose exactly ONE apical diagnosis from this list (exact spelling): ${APICAL_DIAGNOSES.join(" | ")}`,
    "Use percussion, palpation, PARL, and pulpal context. Apply SAP vs AAA and AAP vs CAA rules from the curriculum.",
    "Output format: first line exactly: Apical Diagnosis: <one of the allowed strings>",
    "Then 4–10 sentences explaining your reasoning.",
  ].join("\n");

  let stage3Content: string;
  try {
    stage3Content = await llm.chatComplete([
      { role: "system", content: stage3System },
      {
        role: "user",
        content: `Case JSON:\n${caseJson}\n\nSummary:\n${caseBlock}\n\nStage 1:\n${stage1Content}\n\nStage 2:\n${stage2Content}`,
      },
    ]);
    yield {
      type: "stage",
      stage: 3,
      title: "Stage 3 — Apical diagnosis",
      content: stage3Content,
    };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Stage 3 failed",
    };
    return;
  }

  const stage4System = [
    "You are the synthesizer and verifier.",
    curriculum,
    "",
    ragContext ? `${ragContext}\n` : "",
    STAGE4_JSON_INSTRUCTION,
    "",
    "Verify internal consistency with the raw tests:",
    "- When clinical.ept is not 'not_performed', if clinical.cold is negative AND clinical.ept is negative, pulpalDiagnosis must be Pulp Necrosis (unless explicit documented false-negative in notes).",
    "- Ensure apicalDiagnosis matches percussion/palpation and imaging findings (periapical RL, widened PDL, J-shaped RL, resorption, visual swelling/sinus tract).",
    "If you must override Stage 2 or Stage 3 to fix inconsistency, do so and explain briefly inside biologicalJustification or warnings.",
  ].join("\n");

  const stage4User = [
    `Case JSON:\n${caseJson}`,
    `Summary:\n${caseBlock}`,
    `Stage 1:\n${stage1Content}`,
    `Stage 2:\n${stage2Content}`,
    `Stage 3:\n${stage3Content}`,
    "Produce the JSON object now.",
  ].join("\n\n");

  try {
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
    const parsed = attempt.data;
    yield {
      type: "stage",
      stage: 4,
      title: "Stage 4 — Synthesis (structured)",
      content: JSON.stringify(parsed, null, 2),
    };
    yield { type: "final", result: parsed };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Stage 4 failed",
    };
  }
}

/** Embed the full correction document for storage (RAG ingest). */
export async function embedCorrectionDocument(doc: string): Promise<number[]> {
  return createLlmClient().embedText(doc);
}
