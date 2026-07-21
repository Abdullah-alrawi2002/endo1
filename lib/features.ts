/**
 * Feature flags for the clinical-first MVP.
 * CT support and correction RAG are experimental and OFF by default.
 */
export function isCtModuleEnabled(): boolean {
  return process.env.ENABLE_CT_MODULE?.trim().toLowerCase() === "true";
}

export function isCorrectionRagEnabled(): boolean {
  return process.env.ENABLE_CORRECTION_RAG?.trim().toLowerCase() === "true";
}

export function publicFeatureFlags() {
  return {
    taxonomyVersion: "AAE_2009" as const,
    ctModuleEnabled: isCtModuleEnabled(),
    correctionRagEnabled: isCorrectionRagEnabled(),
    mvpMode: "clinical_only" as const,
  };
}
