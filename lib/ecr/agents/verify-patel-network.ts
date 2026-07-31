import type {
  AdjudicatorResult,
  ComponentAgentResult,
  CriticResult,
  EcrEvidencePackage,
} from "@/lib/ecr/agents/schemas";
import { calculatePatelFromMeasurements } from "@/lib/ecr/geometry";
import type { EcrMeasurementInput, PatelResult } from "@/lib/ecr/schemas";
import { assertEvidenceHash } from "@/lib/ecr/evidence/evidence-hash";
import { recomputeEvidenceHash } from "@/lib/ecr/evidence/build-evidence-package";

export type PatelNetworkVerification = {
  ok: boolean;
  deterministicCompatibility: "passed" | "failed" | "not_run";
  deterministicReference: PatelResult;
  deterministicReferenceCode: string | null;
  issues: Array<{ severity: "reject" | "warn" | "info"; message: string }>;
  requiresSpecialistReview: boolean;
  allowReevaluation: boolean;
};

/**
 * Deterministic verifier — invisible reference for research comparison.
 * Never fed back into component-agent prompts as an answer to copy.
 */
export function verifyPatelNetwork(input: {
  evidence: EcrEvidencePackage;
  measurements: EcrMeasurementInput;
  components: ComponentAgentResult[];
  adjudicator: AdjudicatorResult | null;
  critic: CriticResult | null;
  reevaluationUsed: boolean;
}): PatelNetworkVerification {
  const issues: PatelNetworkVerification["issues"] = [];
  let requiresSpecialistReview = false;
  let allowReevaluation = false;

  // Stale evidence after measurement edits
  const recomputed = recomputeEvidenceHash(input.evidence);
  try {
    assertEvidenceHash(
      input.evidence.caseEvidenceHash,
      recomputed,
      "evidence_package",
    );
  } catch (e) {
    issues.push({
      severity: "reject",
      message: e instanceof Error ? e.message : "Evidence hash mismatch",
    });
    requiresSpecialistReview = true;
  }

  for (const c of input.components) {
    if (c.caseEvidenceHash !== input.evidence.caseEvidenceHash) {
      issues.push({
        severity: "reject",
        message: `${c.agentRole} cited stale caseEvidenceHash`,
      });
      requiresSpecialistReview = true;
    }
    const unknown = c.evidenceIds.filter(
      (id) => !input.evidence.evidenceIds.includes(id) && !id.endsWith("_offline"),
    );
    if (unknown.length) {
      issues.push({
        severity: "reject",
        message: `${c.agentRole} cited nonexistent evidence IDs: ${unknown.join(", ")}`,
      });
      requiresSpecialistReview = true;
    }
  }

  // Quality / differential gates
  if (input.evidence.quality.qualityStatus === "fail") {
    issues.push({ severity: "reject", message: "Quality gate failed — abstain." });
    requiresSpecialistReview = true;
  }
  if (
    input.evidence.maskReviewStatus !== "clinician_reviewed" &&
    input.evidence.maskReviewStatus !== "clinician_corrected"
  ) {
    issues.push({
      severity: "reject",
      message: "Masks not clinician-reviewed — classification abstained.",
    });
    requiresSpecialistReview = true;
  }
  if (input.evidence.ecrDifferential !== "appearance_consistent_with_ecr") {
    issues.push({
      severity: "reject",
      message: `Non-ECR differential (${input.evidence.ecrDifferential}) — Patel network must abstain.`,
    });
    requiresSpecialistReview = true;
  }

  const deterministicReference = calculatePatelFromMeasurements(input.measurements);
  const deterministicReferenceCode = deterministicReference.code;

  let deterministicCompatibility: "passed" | "failed" | "not_run" = "not_run";

  if (input.adjudicator?.patelCode && deterministicReferenceCode) {
    if (input.adjudicator.patelCode === deterministicReferenceCode) {
      deterministicCompatibility = "passed";
    } else {
      deterministicCompatibility = "failed";
      issues.push({
        severity: "warn",
        message: `LLM code ${input.adjudicator.patelCode} incompatible with deterministic reference ${deterministicReferenceCode}`,
      });
      if (!input.reevaluationUsed) {
        allowReevaluation = true;
      } else {
        requiresSpecialistReview = true;
      }
    }
  } else if (input.adjudicator?.abstain || !input.adjudicator?.patelCode) {
    deterministicCompatibility = "not_run";
    if (!requiresSpecialistReview) {
      issues.push({
        severity: "info",
        message: "Adjudicator abstained — specialist review recommended.",
      });
      requiresSpecialistReview = true;
    }
  }

  if (input.critic?.criticStatus === "contradicted") {
    issues.push({
      severity: "warn",
      message: input.critic.contradiction ?? "Adjudicator/critic disagreement",
    });
    if (!input.reevaluationUsed) {
      allowReevaluation = true;
    } else {
      requiresSpecialistReview = true;
    }
  }

  // p must never be treated as pulp necrosis in rationales
  const necrosisLeak = [
    input.adjudicator?.conciseRationale,
    ...input.components.map((c) => c.conciseRationale),
  ].some((t) => t && /pulp necrosis/i.test(t) && /\bp\b/.test(t));
  if (necrosisLeak) {
    issues.push({
      severity: "reject",
      message: "Agent incorrectly equated Patel p with pulp necrosis.",
    });
    requiresSpecialistReview = true;
  }

  const ok =
    !issues.some((i) => i.severity === "reject") &&
    deterministicCompatibility !== "failed" &&
    input.critic?.criticStatus !== "contradicted" &&
    Boolean(input.adjudicator?.patelCode) &&
    !requiresSpecialistReview;

  return {
    ok,
    deterministicCompatibility,
    deterministicReference,
    deterministicReferenceCode,
    issues,
    requiresSpecialistReview:
      requiresSpecialistReview ||
      deterministicCompatibility === "failed" ||
      input.critic?.criticStatus === "contradicted",
    allowReevaluation,
  };
}
