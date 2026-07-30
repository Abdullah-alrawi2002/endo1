import { NextResponse } from "next/server";
import { loadEcrJob } from "@/lib/ecr/store";

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

  if (format === "json") {
    return NextResponse.json(job.result);
  }

  const r = job.result;
  const lines = [
    "ECR CBCT Analysis Report",
    "========================",
    r.outputLabel,
    "",
    `Analysis ID: ${r.analysisId}`,
    `Tooth: ${r.target.toothLabel}`,
    `Scan quality: ${r.scan.qualityStatus} (OOD: ${r.scan.oodStatus})`,
    `ECR differential: ${r.ecrAssessment.status}`,
    `Mask review: ${r.ecrAssessment.maskReviewStatus}`,
    "",
    "Patel classification",
    "--------------------",
    `Status: ${r.patel.status}`,
    `Code: ${r.patel.code ?? "(none — incomplete/abstained)"}`,
    `Height: ${r.patel.height.value ?? "indeterminate"} (borderline=${r.patel.height.borderline})`,
    `Circumference: ${r.patel.circumference.value ?? "indeterminate"} (${r.patel.circumference.maximumAngleDegrees ?? "n/a"}°)`,
    `Canal proximity: ${r.patel.canalProximity.value ?? "indeterminate"} (${r.patel.canalProximity.canalWallStatus ?? "n/a"})`,
    `Manual review required: ${r.patel.manualReviewRequired}`,
    "",
    "Conditional management options (not a definitive plan)",
    "------------------------------------------------------",
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
  ];

  return new NextResponse(lines.join("\n"), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Content-Disposition": `inline; filename="ecr-${id}.txt"`,
    },
  });
}
