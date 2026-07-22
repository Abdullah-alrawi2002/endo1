import { ctSupportSchema, type CTSupport } from "@/lib/schemas/clinical-case";
import { isCtModuleEnabled } from "@/lib/features";

function sidecarUrl(): string | null {
  const url = process.env.CT_SIDECAR_URL?.trim().replace(/\/$/, "");
  return url || null;
}

function isInternalSidecar(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return (
      host === "127.0.0.1" ||
      host === "localhost" ||
      host === "ct-sidecar" ||
      host.endsWith(".internal")
    );
  } catch {
    return true;
  }
}

/** Fetch a stored CT analysis from the CT sidecar. */
export async function fetchCtSupportFromSidecar(
  analysisId: string,
): Promise<CTSupport | null> {
  const base = sidecarUrl();
  if (!base) return null;
  try {
    const response = await fetch(
      `${base}/analysis/${encodeURIComponent(analysisId)}`,
      { cache: "no-store" },
    );
    if (!response.ok) return null;
    const body: unknown = await response.json();
    const parsed = ctSupportSchema.safeParse(body);
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

/**
 * Public browser upload target.
 * - External sidecar URL → browser uploads directly (Vercel + separate CT host).
 * - Internal sidecar (all-in-one Docker) → null; browser uses same-origin /api/analyze-ct.
 */
export function publicCtUploadUrl(): string | null {
  if (!isCtModuleEnabled()) return null;
  const url = sidecarUrl();
  if (!url) return null;
  if (isInternalSidecar(url)) return null;
  return url;
}

export function getSidecarBaseUrl(): string | null {
  return sidecarUrl();
}
