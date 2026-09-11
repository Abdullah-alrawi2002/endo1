/**
 * Feature flags for the Endodontic Diagnostic Agent.
 * Correction RAG is ON by default.
 */
export function isCorrectionRagEnabled(): boolean {
  const v = process.env.ENABLE_CORRECTION_RAG?.trim().toLowerCase();
  if (v === "false" || v === "0" || v === "no") return false;
  return true;
}

export function publicFeatureFlags() {
  return {
    taxonomyVersion: "AAE_2009" as const,
    correctionRagEnabled: isCorrectionRagEnabled(),
    mvpMode: "clinical_agentic" as const,
  };
}
