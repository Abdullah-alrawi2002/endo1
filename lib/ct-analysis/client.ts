import { z } from "zod";
import { ctSupportSchema, type CTSupport } from "@/lib/schemas/clinical-case";

function sidecarUrl(): string {
  return (
    process.env.CT_SIDECAR_URL?.trim().replace(/\/$/, "") ||
    "http://127.0.0.1:8000"
  );
}

/** Sidecar returns a research support payload (no diagnostic lesion flags). */
const sidecarResponseSchema = ctSupportSchema
  .omit({ analysisId: true })
  .extend({
    analysisId: z.string().max(120).optional(),
  });

export async function requestCtAnalysis(
  form: FormData,
): Promise<Omit<CTSupport, "analysisId"> & { analysisId?: string }> {
  const timeoutMs = Number.parseInt(
    process.env.CT_ANALYSIS_TIMEOUT_MS ?? "1800000",
    10,
  );
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${sidecarUrl()}/analyze`, {
      method: "POST",
      body: form,
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("CT sidecar timed out while processing the scan");
    }
    throw new Error(
      `Could not reach CT sidecar at ${sidecarUrl()}: ${
        error instanceof Error ? error.message : "connection failed"
      }`,
    );
  } finally {
    clearTimeout(timer);
  }

  const body = (await response.json().catch(() => null)) as
    | { detail?: string }
    | null;
  if (!response.ok) {
    throw new Error(body?.detail || `CT sidecar returned HTTP ${response.status}`);
  }
  const parsed = sidecarResponseSchema.safeParse(body);
  if (!parsed.success) {
    throw new Error(
      `CT sidecar returned an invalid payload: ${parsed.error.message}`,
    );
  }
  return parsed.data;
}
