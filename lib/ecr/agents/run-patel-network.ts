import {
  canalEvidenceView,
  circumferenceEvidenceView,
  heightEvidenceView,
} from "@/lib/ecr/evidence/build-evidence-package";
import {
  patelNetworkResultSchema,
  type EcrEvidencePackage,
  type PatelNetworkResult,
} from "@/lib/ecr/agents/schemas";
import { runHeightAgent } from "@/lib/ecr/agents/height-agent";
import { runCircumferenceAgent } from "@/lib/ecr/agents/circumference-agent";
import { runCanalAgent } from "@/lib/ecr/agents/canal-agent";
import { runPatelAdjudicator } from "@/lib/ecr/agents/patel-adjudicator";
import { runClassificationCritic } from "@/lib/ecr/agents/classification-critic";
import { verifyPatelNetwork } from "@/lib/ecr/agents/verify-patel-network";
import type { EcrMeasurementInput } from "@/lib/ecr/schemas";
import { shouldSkipLlm } from "@/lib/llm/structured";
import { PATEL_CIRCUMFERENCES, PATEL_CANAL_PROXIMITIES } from "@/lib/ecr/schemas";

/**
 * Deterministic outer orchestration:
 * parallel component agents → adjudicator → critic → deterministic verifier.
 * One bounded re-evaluation; persistent disagreement → needs_specialist_review.
 */
export async function runPatelNetwork(input: {
  evidence: EcrEvidencePackage;
  measurements: EcrMeasurementInput;
}): Promise<PatelNetworkResult> {
  const { evidence, measurements } = input;
  const warnings: string[] = [];

  // Independent component agents (no peer conclusions, no deterministic codes).
  const [height, circumference, canal] = await Promise.all([
    runHeightAgent(heightEvidenceView(evidence)),
    runCircumferenceAgent(circumferenceEvidenceView(evidence)),
    runCanalAgent(canalEvidenceView(evidence)),
  ]);
  let components = [height, circumference, canal];

  let adjudicator = await runPatelAdjudicator({ evidence, components });
  let critic = await runClassificationCritic({
    evidence,
    components,
    adjudicator,
  });

  let verification = verifyPatelNetwork({
    evidence,
    measurements,
    components,
    adjudicator,
    critic,
    reevaluationUsed: false,
  });

  let reevaluationUsed = false;
  if (verification.allowReevaluation && !verification.ok) {
    reevaluationUsed = true;
    warnings.push("Bounded re-evaluation triggered after verifier/critic disagreement.");
    // Re-run only adjudicator+critic with hint — components stay independent (no answer leak).
    adjudicator = await runPatelAdjudicator({
      evidence,
      components,
      reevaluationHint:
        "Prior pass disagreed with independent critic or geometry verifier. Re-check component consistency without inventing measurements.",
    });
    critic = await runClassificationCritic({
      evidence,
      components,
      adjudicator,
    });
    verification = verifyPatelNetwork({
      evidence,
      measurements,
      components,
      adjudicator,
      critic,
      reevaluationUsed: true,
    });
  }

  warnings.push(...verification.issues.map((i) => `[${i.severity}] ${i.message}`));

  const requiresSpecialistReview =
    verification.requiresSpecialistReview ||
    (reevaluationUsed && !verification.ok);

  let classificationStatus: PatelNetworkResult["classificationStatus"] = "incomplete";
  if (requiresSpecialistReview) {
    classificationStatus = "needs_specialist_review";
  } else if (
    evidence.quality.qualityStatus === "fail" ||
    evidence.ecrDifferential !== "appearance_consistent_with_ecr"
  ) {
    classificationStatus = "abstained";
  } else if (adjudicator.patelCode && verification.ok) {
    classificationStatus = "complete";
  } else if (adjudicator.abstain) {
    classificationStatus = "abstained";
  }

  const hVal =
    typeof height.conclusion === "number" ? height.conclusion : null;
  const cVal =
    typeof circumference.conclusion === "string" &&
    (PATEL_CIRCUMFERENCES as readonly string[]).includes(circumference.conclusion)
      ? (circumference.conclusion as "A" | "B" | "C" | "D")
      : null;
  const pVal =
    typeof canal.conclusion === "string" &&
    (PATEL_CANAL_PROXIMITIES as readonly string[]).includes(canal.conclusion)
      ? (canal.conclusion as "d" | "p")
      : null;

  return patelNetworkResultSchema.parse({
    classificationStatus,
    decisionSource: shouldSkipLlm()
      ? "offline_evidence_interpreter"
      : adjudicator.patelCode
        ? "llm_patel_adjudicator"
        : "abstained",
    caseEvidenceHash: evidence.caseEvidenceHash,
    height: {
      value: hVal as 1 | 2 | 3 | 4 | null,
      alternatives: height.alternatives.filter(
        (a): a is 1 | 2 | 3 | 4 => typeof a === "number",
      ),
      borderline: height.uncertainties.length > 0,
    },
    circumference: {
      value: cVal,
      maximumAngleDegrees: evidence.measurements.maximumCircumferenceDegrees,
      alternatives: circumference.alternatives.filter(
        (a): a is "A" | "B" | "C" | "D" =>
          typeof a === "string" &&
          (PATEL_CIRCUMFERENCES as readonly string[]).includes(a),
      ),
      borderline: circumference.uncertainties.length > 0,
    },
    canalProximity: {
      value: pVal,
      minimumSeparationMm: evidence.measurements.minimumLesionCanalSeparationMm,
      alternatives: canal.alternatives.filter(
        (a): a is "d" | "p" =>
          typeof a === "string" &&
          (PATEL_CANAL_PROXIMITIES as readonly string[]).includes(a),
      ),
      borderline: canal.uncertainties.length > 0,
    },
    code: requiresSpecialistReview ? adjudicator.patelCode : adjudicator.patelCode,
    criticStatus: critic.criticStatus,
    deterministicCompatibility: verification.deterministicCompatibility,
    deterministicReferenceCode: verification.deterministicReferenceCode,
    requiresSpecialistReview,
    componentAgents: components,
    adjudicator,
    critic,
    warnings,
  });
}
