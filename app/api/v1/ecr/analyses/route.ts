import { NextResponse } from "next/server";
import {
  appendAudit,
  createEcrJob,
  listRecentEcrJobs,
  saveEcrJob,
  setEcrResult,
} from "@/lib/ecr/store";
import { runEcrPipeline } from "@/lib/ecr/pipeline";
import { buildIntegratedPlan } from "@/lib/integration/build-integrated-plan";
import {
  ecrAnalysisCreateSchema,
} from "@/lib/ecr/schemas";

export const runtime = "nodejs";
export const maxDuration = 120;

/** POST /api/v1/ecr/analyses — create analysis from reviewed measurements. */
export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = ecrAnalysisCreateSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  if (!parsed.data.measurements) {
    return NextResponse.json(
      {
        error:
          "measurements required — demo/fabricated measurements are disabled. Provide clinician-reviewed CBCT measurements.",
      },
      { status: 400 },
    );
  }

  let job = createEcrJob({ jobStatus: "running" });
  job = appendAudit(job, {
    actor: "api",
    action: "analysis_started",
    detail: parsed.data.seriesLabel ?? parsed.data.dicomReference ?? "measurement_input",
  });

  try {
    const measurements = parsed.data.measurements;
    measurements.toothLabel = parsed.data.toothLabel;

    const useAgentNetwork =
      process.env.ENDO_ECR_AGENT_NETWORK === "1" ||
      process.env.ENDO_ECR_AGENT_NETWORK === "true" ||
      (body as { useAgentNetwork?: boolean }).useAgentNetwork === true;

    if (useAgentNetwork) {
      const integrated = await buildIntegratedPlan({
        caseId: job.analysisId,
        measurements,
        ecrAnalysisId: job.analysisId,
      });
      // Also persist deterministic Phase-1 result for compatibility
      const result = runEcrPipeline(measurements, job.analysisId);
      result.warnings = [
        ...result.warnings,
        ...integrated.patelNetwork.warnings,
        `Agent-network code: ${integrated.patelNetwork.code ?? "null"} (decisionSource=${integrated.patelNetwork.decisionSource})`,
        `Deterministic reference (hidden from agents): ${integrated.deterministicReferenceCode ?? "null"}`,
        `Plan status: ${integrated.provisionalPlan.planStatus}`,
      ];
      // Prefer agent-network code when complete; keep deterministic components for audit
      if (
        integrated.patelNetwork.code &&
        integrated.patelNetwork.classificationStatus === "complete"
      ) {
        result.patel = {
          ...result.patel,
          code: integrated.patelNetwork.code,
          manualReviewRequired:
            result.patel.manualReviewRequired ||
            integrated.patelNetwork.requiresSpecialistReview,
        };
      } else if (integrated.patelNetwork.requiresSpecialistReview) {
        result.patel = {
          ...result.patel,
          status: "abstained",
          code: null,
          manualReviewRequired: true,
        };
      }

      const completed = setEcrResult(
        job,
        result,
        result.patel.code ? "completed" : "abstained",
      );
      saveEcrJob(
        appendAudit(completed, {
          actor: "system",
          action: "agent_network_completed",
          detail: integrated.evidenceHash,
        }),
      );

      return NextResponse.json({
        analysisId: completed.analysisId,
        jobStatus: completed.jobStatus,
        result: completed.result,
        unifiedCase: integrated.unified,
        patelNetwork: {
          ...integrated.patelNetwork,
          // Strip hidden reference from client-facing agent payload? Keep for research UI audit panel.
        },
        provisionalPlan: integrated.provisionalPlan,
        notice:
          "CBCT-derived ECR classification and treatment-planning options — not a definitive treatment plan. Agents received evidence only; deterministic classifier used as invisible verifier.",
      });
    }

    const result = runEcrPipeline(measurements, job.analysisId);
    const completed = setEcrResult(
      job,
      result,
      result.patel.code ? "completed" : "abstained",
    );

    return NextResponse.json({
      analysisId: completed.analysisId,
      jobStatus: completed.jobStatus,
      result: completed.result,
      notice:
        "CBCT-derived ECR classification and treatment-planning options — not a definitive treatment plan. Phase-1 accepts clinician-reviewed measurements; full DICOM segmentation is roadmap Phase 2–3.",
    });
  } catch (e) {
    job.jobStatus = "failed";
    job.error = e instanceof Error ? e.message : "Analysis failed";
    saveEcrJob(
      appendAudit(job, {
        actor: "system",
        action: "analysis_failed",
        detail: job.error,
      }),
    );
    return NextResponse.json({ error: job.error }, { status: 500 });
  }
}

export async function GET() {
  const jobs = listRecentEcrJobs(50).map((j) => ({
    analysisId: j.analysisId,
    jobStatus: j.jobStatus,
    updatedAt: j.updatedAt,
    patelCode: j.result?.patel.code ?? null,
    toothLabel: j.result?.target.toothLabel ?? null,
  }));
  return NextResponse.json({ analyses: jobs });
}
