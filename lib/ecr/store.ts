import { mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import {
  ecrJobSchema,
  type EcrJob,
  type EcrResult,
} from "@/lib/ecr/schemas";

function storeDir(): string {
  if (process.env.VERCEL) {
    return join("/tmp", "ecr-analyses");
  }
  const configured = process.env.ECR_ANALYSIS_STORE_PATH?.trim();
  if (configured) return configured;
  const dir = join(
    /*turbopackIgnore: true*/ process.cwd(),
    "data",
    "ecr-analyses",
  );
  mkdirSync(dir, { recursive: true });
  return dir;
}

function pathFor(id: string): string {
  return join(storeDir(), `${id}.json`);
}

export function createEcrJob(partial?: Partial<EcrJob>): EcrJob {
  const now = Date.now();
  const job: EcrJob = ecrJobSchema.parse({
    analysisId: partial?.analysisId ?? randomUUID(),
    jobStatus: partial?.jobStatus ?? "queued",
    createdAt: now,
    updatedAt: now,
    error: null,
    result: null,
    audit: [
      {
        at: now,
        actor: "system",
        action: "job_created",
      },
    ],
    ...partial,
  });
  saveEcrJob(job);
  return job;
}

export function saveEcrJob(job: EcrJob): void {
  const parsed = ecrJobSchema.parse({ ...job, updatedAt: Date.now() });
  writeFileSync(pathFor(parsed.analysisId), JSON.stringify(parsed, null, 2), "utf8");
}

export function loadEcrJob(analysisId: string): EcrJob | null {
  const path = pathFor(analysisId);
  if (!existsSync(path)) return null;
  try {
    return ecrJobSchema.parse(JSON.parse(readFileSync(path, "utf8")));
  } catch {
    return null;
  }
}

export function appendAudit(
  job: EcrJob,
  entry: { actor: string; action: string; detail?: string },
): EcrJob {
  const next: EcrJob = {
    ...job,
    updatedAt: Date.now(),
    audit: [
      ...job.audit,
      { at: Date.now(), actor: entry.actor, action: entry.action, detail: entry.detail },
    ].slice(-500),
  };
  saveEcrJob(next);
  return next;
}

export function listRecentEcrJobs(limit = 30): EcrJob[] {
  const dir = storeDir();
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      try {
        return ecrJobSchema.parse(
          JSON.parse(readFileSync(join(dir, f), "utf8")),
        );
      } catch {
        return null;
      }
    })
    .filter((j): j is EcrJob => j !== null)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, limit);
}

export function setEcrResult(job: EcrJob, result: EcrResult, status: EcrJob["jobStatus"]): EcrJob {
  const next: EcrJob = {
    ...job,
    jobStatus: status,
    result,
    updatedAt: Date.now(),
    audit: [
      ...job.audit,
      {
        at: Date.now(),
        actor: "system",
        action: "result_written",
        detail: `status=${status}; patel=${result.patel.code ?? "null"}`,
      },
    ].slice(-500),
  };
  saveEcrJob(next);
  return next;
}
