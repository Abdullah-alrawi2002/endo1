import { NextResponse } from "next/server";
import { appendAudit, loadEcrJob, saveEcrJob, setEcrResult } from "@/lib/ecr/store";
import { runEcrPipeline } from "@/lib/ecr/pipeline";
import { ecrReviewSchema } from "@/lib/ecr/schemas";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

/** POST /api/v1/ecr/analyses/{id}/review — confirm, modify, abstain, or refer. */
export async function POST(req: Request, ctx: Ctx) {
  const { id } = await ctx.params;
  let job = loadEcrJob(id);
  if (!job || !job.result) {
    return NextResponse.json({ error: "Analysis not found" }, { status: 404 });
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = ecrReviewSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const { decision, reviewerId, reviewerNotes, modifiedMeasurements } = parsed.data;

  if (decision === "modify" && modifiedMeasurements) {
    const result = runEcrPipeline(modifiedMeasurements, id);
    result.review = {
      specialistConfirmationRequired: true,
      entireVolumeInterpretationRequired: true,
      decision,
      reviewerNotes: reviewerNotes ?? null,
      decidedAt: Date.now(),
      reviewerId,
    };
    job = setEcrResult(job, result, result.patel.code ? "completed" : "abstained");
  } else {
    job.result = {
      ...job.result,
      review: {
        ...job.result.review,
        decision,
        reviewerNotes: reviewerNotes ?? null,
        decidedAt: Date.now(),
        reviewerId,
      },
      updatedAt: Date.now(),
    };
    if (decision === "abstain") {
      job.jobStatus = "abstained";
      job.result.patel = { ...job.result.patel, status: "abstained", code: null };
      job.result.managementSupport = {
        ...job.result.managementSupport,
        options: [],
        definitiveTreatmentPlanAvailable: false,
      };
    }
    saveEcrJob(job);
  }

  job = appendAudit(job, {
    actor: reviewerId,
    action: `review_${decision}`,
    detail: reviewerNotes,
  });

  return NextResponse.json({
    analysisId: job.analysisId,
    jobStatus: job.jobStatus,
    result: job.result,
    notice:
      "Reviewer decision stored separately from CBCT-only algorithm output for evaluation.",
  });
}
