import { NextResponse } from "next/server";
import { appendAudit, loadEcrJob, setEcrResult } from "@/lib/ecr/store";
import { runEcrPipeline } from "@/lib/ecr/pipeline";
import { ecrMeasurementInputSchema } from "@/lib/ecr/schemas";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

/** POST /api/v1/ecr/analyses/{id}/recalculate — re-run geometry after measurement/mask corrections. */
export async function POST(req: Request, ctx: Ctx) {
  const { id } = await ctx.params;
  const job = loadEcrJob(id);
  if (!job) {
    return NextResponse.json({ error: "Analysis not found" }, { status: 404 });
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = ecrMeasurementInputSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  appendAudit(job, {
    actor: "clinician",
    action: "recalculate_requested",
  });

  const result = runEcrPipeline(parsed.data, id);
  const updated = setEcrResult(
    job,
    result,
    result.patel.code ? "completed" : "abstained",
  );

  return NextResponse.json({
    analysisId: updated.analysisId,
    jobStatus: updated.jobStatus,
    result: updated.result,
  });
}
