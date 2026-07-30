import type {
  EcrMeasurementInput,
  EcrResult,
  PatelResult,
  ScanQuality,
} from "@/lib/ecr/schemas";

export type VerificationIssue = {
  code: string;
  severity: "reject" | "abstain" | "warn";
  message: string;
};

const ECR_SUPPORTED = new Set([
  "appearance_consistent_with_ecr",
]);

/**
 * Deterministic verifier — runs before treatment options.
 * Rejects inconsistent codes and CBCT-only definitive treatment claims.
 */
export function verifyEcrAnalysis(args: {
  scan: ScanQuality;
  ecrStatus: EcrMeasurementInput["ecrDifferential"];
  maskReviewStatus: EcrMeasurementInput["maskReviewStatus"];
  patel: PatelResult;
}): { ok: boolean; issues: VerificationIssue[]; mayRankOptions: boolean } {
  const issues: VerificationIssue[] = [];

  if (args.scan.qualityStatus === "fail") {
    issues.push({
      code: "quality_fail",
      severity: "abstain",
      message:
        "Scan quality fail — no ECR conclusion, Patel code, or option ranking.",
    });
  }

  if (args.scan.oodStatus === "out_of_domain") {
    issues.push({
      code: "ood",
      severity: "abstain",
      message: "Target anatomy outside validated domain.",
    });
  }

  if (!ECR_SUPPORTED.has(args.ecrStatus)) {
    issues.push({
      code: "differential_not_ecr",
      severity: "abstain",
      message: `ECR differential is '${args.ecrStatus}' — no Patel code; return differential/abstention.`,
    });
  }

  if (
    args.maskReviewStatus !== "clinician_reviewed" &&
    args.maskReviewStatus !== "clinician_corrected"
  ) {
    issues.push({
      code: "masks_unreviewed",
      severity: "warn",
      message:
        "Critical masks not clinician-reviewed — Phase-1 requires review before clinical-facing use.",
    });
  }

  const structures = args.scan.targetStructuresVisible;
  if (!structures.cejRegion || !structures.localAlveolarCrest || !structures.apex) {
    issues.push({
      code: "landmarks_unresolved",
      severity: "abstain",
      message: "CEJ, crest, or apex unresolved — height indeterminate; no complete Patel code.",
    });
  }
  if (!structures.lesionMargins) {
    issues.push({
      code: "lesion_unresolved",
      severity: "abstain",
      message: "Lesion margins unresolved — circumference/height not assessable.",
    });
  }
  if (!structures.canalBoundaryNearLesion) {
    issues.push({
      code: "canal_unresolved",
      severity: "abstain",
      message: "Canal boundary unresolved — d/p indeterminate; no complete Patel code.",
    });
  }

  if (args.patel.height.status !== "complete") {
    issues.push({
      code: "height_incomplete",
      severity: "abstain",
      message: "Height component incomplete.",
    });
  }
  if (args.patel.circumference.status !== "complete") {
    issues.push({
      code: "circumference_incomplete",
      severity: "abstain",
      message: "Circumference component incomplete.",
    });
  }
  if (args.patel.canalProximity.status !== "complete") {
    issues.push({
      code: "canal_proximity_incomplete",
      severity: "abstain",
      message: "Canal proximity component incomplete.",
    });
  }

  if (args.patel.code) {
    const expected = `${args.patel.height.value}${args.patel.circumference.value}${args.patel.canalProximity.value}`;
    if (args.patel.code !== expected) {
      issues.push({
        code: "code_mismatch",
        severity: "reject",
        message: `Code ${args.patel.code} does not match components ${expected}.`,
      });
    }
  }

  const borderlineWithoutFlag =
    (args.patel.height.borderline ||
      args.patel.circumference.borderline ||
      args.patel.canalProximity.borderline) &&
    !args.patel.manualReviewRequired;
  if (borderlineWithoutFlag) {
    issues.push({
      code: "borderline_unflagged",
      severity: "reject",
      message: "Classification near a boundary without manual-review flag.",
    });
  }

  const hasReject = issues.some((i) => i.severity === "reject");
  const hasAbstain = issues.some((i) => i.severity === "abstain");
  const mayRankOptions =
    !hasReject &&
    !hasAbstain &&
    args.patel.status === "complete" &&
    args.patel.code !== null &&
    ECR_SUPPORTED.has(args.ecrStatus) &&
    args.scan.qualityStatus !== "fail";

  return {
    ok: !hasReject && !hasAbstain,
    issues,
    mayRankOptions,
  };
}

/** Safety: never allow definitive treatment plan language from CBCT-only output. */
export function assertNoDefinitiveTreatmentPlan(
  result: Pick<EcrResult, "managementSupport">,
): VerificationIssue[] {
  const issues: VerificationIssue[] = [];
  if (result.managementSupport.definitiveTreatmentPlanAvailable !== false) {
    issues.push({
      code: "definitive_plan_forbidden",
      severity: "reject",
      message:
        "Definitive treatment plan from CBCT alone is forbidden — replace with conditional options.",
    });
  }
  return issues;
}
