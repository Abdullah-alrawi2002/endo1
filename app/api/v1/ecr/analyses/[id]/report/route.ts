import { NextResponse } from "next/server";
import { loadEcrJob } from "@/lib/ecr/store";
import type { ProvisionalPlan } from "@/lib/ecr/agents/schemas";

export const runtime = "nodejs";

type Ctx = { params: Promise<{ id: string }> };

/** GET /api/v1/ecr/analyses/{id}/report */
export async function GET(req: Request, ctx: Ctx) {
  const { id } = await ctx.params;
  const job = loadEcrJob(id);
  if (!job?.result) {
    return NextResponse.json({ error: "Analysis not found" }, { status: 404 });
  }

  const { searchParams } = new URL(req.url);
  const format = searchParams.get("format") ?? "json";
  const plan = job.provisionalPlan as ProvisionalPlan | null | undefined;

  if (format === "json") {
    return NextResponse.json({
      result: job.result,
      patelNetwork: job.patelNetwork ?? null,
      provisionalPlan: plan ?? null,
      evidenceHash: job.evidenceHash ?? null,
    });
  }

  const r = job.result;
  const lines = [
    "ECR CBCT Treatment-Planning Report",
    "==================================",
    r.outputLabel,
    "",
    `Analysis ID: ${r.analysisId}`,
    `Tooth: ${r.target.toothLabel}`,
    `Scan quality: ${r.scan.qualityStatus} (OOD: ${r.scan.oodStatus})`,
    `ECR differential: ${r.ecrAssessment.status}`,
    `Mask review: ${r.ecrAssessment.maskReviewStatus}`,
    job.evidenceHash ? `Evidence hash: ${job.evidenceHash}` : "",
    "",
    "Patel classification (external cervical resorption)",
    "---------------------------------------------------",
    `Status: ${r.patel.status}`,
    `Code: ${r.patel.code ?? "(none — incomplete/abstained)"}`,
    `Height: ${r.patel.height.value ?? "indeterminate"} (borderline=${r.patel.height.borderline})`,
    `Circumference: ${r.patel.circumference.value ?? "indeterminate"} (${r.patel.circumference.maximumAngleDegrees ?? "n/a"}°)`,
    `Canal proximity: ${r.patel.canalProximity.value ?? "indeterminate"} (${r.patel.canalProximity.canalWallStatus ?? "n/a"})`,
    "Note: Patel p = probable pulpal involvement on imaging — not pulp necrosis.",
    `Manual review required: ${r.patel.manualReviewRequired}`,
    "",
  ];

  if (plan) {
    lines.push(
      "Provisional treatment plan (CBCT-based — not definitive)",
      "-------------------------------------------------------",
      `Plan status: ${plan.planStatus}`,
      `Primary strategy: ${plan.primaryStrategy ?? "(none)"}`,
      `Safety: ${plan.safetyStatus} · Critic: ${plan.criticStatus}`,
      "",
      "Candidates (overlapping alternatives preserved):",
      ...plan.candidates.map(
        (c) =>
          `- [${c.status}] ${c.strategy}\n  Objective: ${c.treatmentObjective}\n  Activate if: ${c.activateIf.join("; ") || "—"}\n  Reject if: ${c.rejectIf.join("; ") || "—"}\n  Clinical confirmations: ${c.requiredClinicalConfirmations.join("; ")}`,
      ),
      "",
      "Procedural sequence:",
      ...(plan.proceduralSequence.length
        ? plan.proceduralSequence.map((s, i) => `${i + 1}. ${s}`)
        : ["- (see candidate sequences)"]),
      "",
      "Decision checkpoints:",
      ...plan.decisionCheckpoints.map((c) => `- ${c}`),
      "",
      "Stop conditions:",
      ...plan.stopConditions.map((c) => `- ${c}`),
      "",
      "Follow-up:",
      ...plan.followUp.map((c) => `- ${c}`),
      "",
    );
  }

  lines.push(
    "Conditional management options (ESE imaging priors)",
    "---------------------------------------------------",
    `Ruleset: ${r.managementSupport.rulesetVersion}`,
    ...r.managementSupport.options.map(
      (o) =>
        `- [${o.status}] ${o.option} (rules: ${o.supportingRuleIds.join(", ")})`,
    ),
    "",
    "Cannot determine from CBCT alone:",
    ...r.managementSupport.cannotDetermineFromCbct.map((c) => `- ${c}`),
    "",
    "Warnings:",
    ...(r.warnings.length ? r.warnings.map((w) => `- ${w}`) : ["- none"]),
    "",
    `Specialist confirmation required: ${r.review.specialistConfirmationRequired}`,
    `Entire volume interpretation required: ${r.review.entireVolumeInterpretationRequired}`,
    `Reviewer decision: ${r.review.decision ?? "pending"}`,
  );

  return new NextResponse(lines.filter((l) => l !== undefined).join("\n"), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Content-Disposition": `inline; filename="ecr-plan-${id}.txt"`,
    },
  });
}
