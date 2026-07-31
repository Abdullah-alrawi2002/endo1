import { NextResponse } from "next/server";
import { loadEcrJob } from "@/lib/ecr/store";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

/** GET /api/v1/ecr/analyses/{id} */
export async function GET(_req: Request, ctx: Ctx) {
  const { id } = await ctx.params;
  const job = loadEcrJob(id);
  if (!job) {
    return NextResponse.json({ error: "Analysis not found" }, { status: 404 });
  }
  return NextResponse.json({
    analysisId: job.analysisId,
    jobStatus: job.jobStatus,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    error: job.error ?? null,
    result: job.result,
    patelNetwork: job.patelNetwork ?? null,
    provisionalPlan: job.provisionalPlan ?? null,
    evidenceHash: job.evidenceHash ?? null,
    audit: job.audit,
  });
}
