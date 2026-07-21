/**
 * Feature flags.
 * Correction RAG is ON by default (core plan feature); set ENABLE_CORRECTION_RAG=false to disable.
 * CT support remains experimental and OFF unless ENABLE_CT_MODULE=true.
 */
export function isCtModuleEnabled(): boolean {
  return process.env.ENABLE_CT_MODULE?.trim().toLowerCase() === "true";
}

export function isCorrectionRagEnabled(): boolean {
  const v = process.env.ENABLE_CORRECTION_RAG?.trim().toLowerCase();
  if (v === "false" || v === "0" || v === "no") return false;
  return true;
}

export function publicFeatureFlags() {
  return {
    taxonomyVersion: "AAE_2009" as const,
    ctModuleEnabled: isCtModuleEnabled(),
    correctionRagEnabled: isCorrectionRagEnabled(),
    mvpMode: "clinical_agentic" as const,
  };
}
