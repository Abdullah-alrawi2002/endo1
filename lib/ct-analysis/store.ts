import { mkdirSync, readFileSync, writeFileSync, existsSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import {
  ctSupportSchema,
  type CTSupport,
} from "@/lib/schemas/clinical-case";

function storeDir(): string {
  const configured = process.env.CT_ANALYSIS_STORE_PATH?.trim();
  const dir = configured || join(process.cwd(), "data", "ct-analyses");
  mkdirSync(dir, { recursive: true });
  return dir;
}

export function saveCtSupport(analysis: Omit<CTSupport, "analysisId"> & { analysisId?: string }): CTSupport {
  const analysisId = analysis.analysisId ?? randomUUID();
  const full = ctSupportSchema.parse({ ...analysis, analysisId });
  const path = join(storeDir(), `${analysisId}.json`);
  writeFileSync(path, JSON.stringify(full, null, 2), "utf8");
  return full;
}

export function loadCtSupport(analysisId: string): CTSupport | null {
  const path = join(storeDir(), `${analysisId}.json`);
  if (!existsSync(path)) return null;
  try {
    const parsed = ctSupportSchema.safeParse(JSON.parse(readFileSync(path, "utf8")));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}
