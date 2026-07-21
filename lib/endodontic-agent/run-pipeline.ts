import type { ClinicalCase, FinalDiagnosis } from "@/lib/schemas/clinical-case";
import {
  embedCorrectionDocument,
  runDiagnosisOnce,
  type DiagnosisPipelineOptions,
} from "@/lib/batch/diagnose-once";

export type StreamEvent =
  | { type: "status"; message: string }
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

/**
 * Streams status + evidence + final result only.
 * Does not stream provisional chain-of-thought or tentative diagnoses.
 */
export async function* runDiagnosisPipeline(
  clinicalCase: ClinicalCase,
  options: DiagnosisPipelineOptions = {},
): AsyncGenerator<StreamEvent> {
  try {
    yield { type: "status", message: "Running Stage 0 scope/validity gate…" };
    const result = await runDiagnosisOnce(clinicalCase, options);
    yield { type: "status", message: result.statusMessage };
    yield {
      type: "evidence",
      evidenceFor: result.final.evidenceFor,
      evidenceAgainst: result.final.evidenceAgainst,
      conflicts: result.final.conflicts,
      missingRequiredData: result.final.missingRequiredData,
      recommendedNextTests: result.final.recommendedNextTests,
    };
    yield { type: "final", result: result.final };
  } catch (e) {
    yield {
      type: "error",
      message: e instanceof Error ? e.message : "Diagnosis failed",
    };
  }
}

export { embedCorrectionDocument, runDiagnosisOnce };
