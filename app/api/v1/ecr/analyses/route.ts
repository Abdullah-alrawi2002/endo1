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
import { ecrAnalysisCreateSchema } from "@/lib/ecr/schemas";

export const runtime = "nodejs";
export const maxDuration = 120;

/**
 * POST /api/v1/ecr/analyses
 * CBCT-derived measurements → Patel classification → provisional treatment plan.
 * Always runs the evidence-first agent network + treatment synthesizer unless
 * useAgentNetwork is explicitly false (deterministic-only research arm).
 */
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
          "measurements required. Enter clinician-reviewed CBCT measurements (CEJ/crest extents, circumference, lesion–canal relation). Automatic ECR segmentation is not yet enabled.",
      },
      { status: 400 },
    );
  }

  let job = createEcrJob({ jobStatus: "running" });
  job = appendAudit(job, {
    actor: "api",
    action: "analysis_started",
    detail:
      parsed.data.seriesLabel ??
      parsed.data.dicomReference ??
      "cbct_measurement_input",
  });

  try {
    const measurements = parsed.data.measurements;
    measurements.toothLabel = parsed.data.toothLabel;

    const bodyObj = body as { useAgentNetwork?: boolean };
    const useAgentNetwork = bodyObj.useAgentNetwork !== false;

    // Deterministic geometry always runs (research reference + schema-compatible result).
    let result = runEcrPipeline(measurements, job.analysisId);

    if (!useAgentNetwork) {
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
          "Deterministic Patel + conditional ESE options only (agent network disabled). Not a definitive treatment plan.",
      });
    }

    const integrated = await buildIntegratedPlan({
      caseId: job.analysisId,
      measurements,
      ecrAnalysisId: job.analysisId,
    });

    result = {
      ...result,
      warnings: [
        ...result.warnings,
        ...integrated.patelNetwork.warnings,
        `Decision source: ${integrated.patelNetwork.decisionSource}`,
        `Plan status: ${integrated.provisionalPlan.planStatus}`,
      ],
    };

    if (
      integrated.patelNetwork.code &&
      integrated.patelNetwork.classificationStatus === "complete" &&
      !integrated.patelNetwork.requiresSpecialistReview
    ) {
      result = {
        ...result,
        patel: {
          ...result.patel,
          code: integrated.patelNetwork.code,
          status: "complete",
          manualReviewRequired:
            result.patel.manualReviewRequired ||
            integrated.patelNetwork.requiresSpecialistReview,
        },
      };
    } else if (
      integrated.patelNetwork.requiresSpecialistReview ||
      integrated.patelNetwork.classificationStatus === "needs_specialist_review" ||
      integrated.patelNetwork.classificationStatus === "abstained"
    ) {
      result = {
        ...result,
        patel: {
          ...result.patel,
          status:
            integrated.patelNetwork.classificationStatus === "abstained"
              ? "abstained"
              : result.patel.status === "complete"
                ? "incomplete"
                : result.patel.status,
          code:
            integrated.patelNetwork.classificationStatus === "abstained"
              ? null
              : integrated.patelNetwork.code ?? result.patel.code,
          manualReviewRequired: true,
        },
      };
    }

    // Prefer agent-network multilabel options when plan produced candidates
    if (integrated.provisionalPlan.candidates.length) {
      const fromPlan = integrated.provisionalPlan.candidates.map((c) => ({
        option: c.strategy,
        status: c.status,
        supportingRuleIds: c.sourceRuleIds.length
          ? c.sourceRuleIds
          : ["TREATMENT-AGENT-3.0"],
        supportingCbctFeatures: c.activateIf,
        limitingCbctFeatures: c.rejectIf,
        requiresClinicalConfirmation: c.requiredClinicalConfirmations,
        evidenceVersion: c.evidenceVersion,
      }));
      result = {
        ...result,
        managementSupport: {
          ...result.managementSupport,
          options: fromPlan.length ? fromPlan : result.managementSupport.options,
        },
      };
    }

    const completed = setEcrResult(
      job,
      result,
      result.patel.code || integrated.provisionalPlan.candidates.length
        ? "completed"
        : "abstained",
      {
        patelNetwork: integrated.patelNetwork,
        provisionalPlan: integrated.provisionalPlan,
        evidenceHash: integrated.evidenceHash,
      },
    );

    saveEcrJob(
      appendAudit(completed, {
        actor: "system",
        action: "treatment_plan_generated",
        detail: `${integrated.provisionalPlan.planStatus}; primary=${integrated.provisionalPlan.primaryStrategy ?? "none"}`,
      }),
    );

    return NextResponse.json({
      analysisId: completed.analysisId,
      jobStatus: completed.jobStatus,
      result: completed.result,
      patelNetwork: integrated.patelNetwork,
      provisionalPlan: integrated.provisionalPlan,
      unifiedCase: integrated.unified,
      evidenceHash: integrated.evidenceHash,
      notice:
        "CBCT-derived Patel classification and provisional treatment-planning options for external cervical resorption — not a definitive treatment plan. Specialist confirmation of the entire volume is required.",
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
    planStatus:
      j.provisionalPlan &&
      typeof j.provisionalPlan === "object" &&
      j.provisionalPlan !== null &&
      "planStatus" in j.provisionalPlan
        ? (j.provisionalPlan as { planStatus: string }).planStatus
        : null,
    toothLabel: j.result?.target.toothLabel ?? null,
  }));
  return NextResponse.json({ analyses: jobs });
}
