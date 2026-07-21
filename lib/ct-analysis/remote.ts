import { ctSupportSchema, type CTSupport } from "@/lib/schemas/clinical-case";

function sidecarUrl(): string | null {
  const url = process.env.CT_SIDECAR_URL?.trim().replace(/\/$/, "");
  return url || null;
}

/** Fetch a stored CT analysis from the hosted sidecar (Vercel + Railway setup). */
export async function fetchCtSupportFromSidecar(
  analysisId: string,
): Promise<CTSupport | null> {
  const base = sidecarUrl();
  if (!base) return null;
  try {
    const response = await fetch(`${base}/analysis/${encodeURIComponent(analysisId)}`, {
      cache: "no-store",
    });
    if (!response.ok) return null;
    const body: unknown = await response.json();
    const parsed = ctSupportSchema.safeParse(body);
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

export function publicCtUploadUrl(): string | null {
  if (process.env.ENABLE_CT_MODULE?.trim().toLowerCase() !== "true") {
    return null;
  }
  return sidecarUrl();
}
