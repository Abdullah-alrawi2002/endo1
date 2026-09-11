/**
 * Batch helpers — diagnosis runs through the same multi-stage agent pipeline.
 */
export {
  embedCorrectionDocument,
  runDiagnosisOnce,
  type DiagnosisPipelineOptions,
  type DiagnosisPipelineResult,
} from "@/lib/endodontic-agent/run-pipeline";

import { z } from "zod";
import { createLlmClient } from "@/lib/llm";
import { clinicalCaseSchema } from "@/lib/schemas/clinical-case";

function parseModelJson(raw: string): unknown {
  let s = raw.trim();
  if (s.startsWith("```")) {
    s = s.replace(/^```(?:json)?\s*\n?/i, "");
    const end = s.lastIndexOf("```");
    if (end !== -1) s = s.slice(0, end).trim();
  }
  return JSON.parse(s);
}

const extractionResultSchema = z.object({
  eligible: z.boolean(),
  ineligibleReason: z.string().optional(),
  case: clinicalCaseSchema.optional(),
});

export type RowExtraction = z.infer<typeof extractionResultSchema>;

const EXTRACTION_PROMPT = `Convert chart text into the ClinicalCase JSON used by this endodontic agent (taxonomy AAE_2009, untreated permanent teeth only).

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
