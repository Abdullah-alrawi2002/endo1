import { mkdirSync, readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import {
  ctSupportSchema,
  type CTSupport,
} from "@/lib/schemas/clinical-case";
import { fetchCtSupportFromSidecar } from "@/lib/ct-analysis/remote";

function storeDir(): string {
  if (process.env.VERCEL) {
    return join("/tmp", "ct-analyses");
  }
  const configured = process.env.CT_ANALYSIS_STORE_PATH?.trim();
  const dir = configured || join(/*turbopackIgnore: true*/ process.cwd(), "data", "ct-analyses");
  mkdirSync(dir, { recursive: true });
  return dir;
}

export function saveCtSupport(
  analysis: Omit<CTSupport, "analysisId"> & { analysisId?: string },
): CTSupport {
  const analysisId = analysis.analysisId ?? randomUUID();
  const full = ctSupportSchema.parse({ ...analysis, analysisId });
  const path = join(storeDir(), `${analysisId}.json`);
  try {
    writeFileSync(path, JSON.stringify(full, null, 2), "utf8");
  } catch {
    /* On serverless, local cache is best-effort; sidecar is source of truth. */
  }
  return full;
}

export async function loadCtSupport(analysisId: string): Promise<CTSupport | null> {
  const path = join(storeDir(), `${analysisId}.json`);
  if (existsSync(path)) {
    try {
      const parsed = ctSupportSchema.safeParse(
        JSON.parse(readFileSync(path, "utf8")),
      );
      if (parsed.success) return parsed.data;
    } catch {
      /* fall through */
    }
  }
  return fetchCtSupportFromSidecar(analysisId);
}
