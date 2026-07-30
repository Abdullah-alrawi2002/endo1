import { NextResponse } from "next/server";
import {
  appendAudit,
  createEcrJob,
  listRecentEcrJobs,
  saveEcrJob,
  setEcrResult,
} from "@/lib/ecr/store";
import { runEcrPipeline } from "@/lib/ecr/pipeline";
import {
  ecrAnalysisCreateSchema,
  ecrMeasurementInputSchema,
  type EcrMeasurementInput,
} from "@/lib/ecr/schemas";

export const runtime = "nodejs";
export const maxDuration = 60;

function demoMeasurements(toothLabel: string): EcrMeasurementInput {
  return ecrMeasurementInputSchema.parse({
    toothLabel,
    rootsAffected: ["single_root"],
    mostApicalExtentMmFromCEJ: 4.8,
    rootLengthCejToApexMm: 16,
    localCrestDistanceMmFromCEJ: 2.0,
    maximumCircumferenceDegrees: 142,
    minimumLesionCanalSeparationMm: 0,
    separationUncertaintyLowerBoundMm: 0,
    lesionCanalContactOrIntersection: true,
    continuousDentineBarrierVisible: false,
    ecrDifferential: "appearance_consistent_with_ecr",
    maskReviewStatus: "clinician_reviewed",
    scan: {
      qualityStatus: "pass",
      oodStatus: "in_domain",
      nativeVoxelMm: [0.2, 0.2, 0.2],
      fovCompleteForTarget: true,
      targetStructuresVisible: {
        crownRootComplex: true,
        cejRegion: true,
        localAlveolarCrest: true,
        apex: true,
        canalBoundaryNearLesion: true,
        lesionMargins: true,
      },
      artifactWarnings: [],
      qualityGateFailures: [],
    },
    planning: {
      portalSurface: "buccal",
      portalAreaMm2: 3.1,
      portalSupracrestal: "present",
      lesionVolumeMm3: 18.4,
      lesionMaxDepthMm: 2.2,
      lesionMaxWidthMm: 3.0,
      externalAccessProxy: "favorable",
      internalAccessProxy: "possible",
      structuralContinuity: "reduced",
      furcationInvolvement: "not_applicable",
      boneCrestLossProxy: "present",
      adjacentAnatomyWarnings: [],
      existingTreatmentFindings: [],
      fractureRiskProxy: "moderate",
      rootFormForReplantation: "compatible",
    },
    portalSurface: "buccal",
    portalAreaMm2: 3.1,
    lesionVolumeMm3: 18.4,
  });
}

/** POST /api/v1/ecr/analyses — create analysis from measurements (Phase-1). */
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

  let job = createEcrJob({ jobStatus: "running" });
  job = appendAudit(job, {
    actor: "api",
    action: "analysis_started",
    detail: parsed.data.seriesLabel ?? parsed.data.dicomReference ?? "measurement_input",
  });

  try {
    const measurements =
      parsed.data.measurements ?? demoMeasurements(parsed.data.toothLabel);
    measurements.toothLabel = parsed.data.toothLabel;

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
