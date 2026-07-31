import { randomUUID } from "crypto";
import { buildEvidencePackage } from "@/lib/ecr/evidence/build-evidence-package";
import { runPatelNetwork } from "@/lib/ecr/agents/run-patel-network";
import { runTreatmentAgentNetwork } from "@/lib/ecr/agents/run-treatment-network";
import type { EcrMeasurementInput, PatelResult } from "@/lib/ecr/schemas";
import type { FinalDiagnosis } from "@/lib/schemas/clinical-case";
import {
  assertIntegrationWalls,
  unifiedCaseSchema,
  type UnifiedCase,
} from "@/lib/schemas/unified-case";
import type { PatelNetworkResult, ProvisionalPlan } from "@/lib/ecr/agents/schemas";

export type IntegratedPlanResult = {
  unified: UnifiedCase;
  evidenceHash: string;
  patelNetwork: PatelNetworkResult;
  provisionalPlan: ProvisionalPlan;
  /** Hidden deterministic reference — for audit/research only. */
  deterministicReferenceCode: string | null;
};

/**
 * Connect clinical + ECR products into one UnifiedCase.
 * Clinical may influence RCT branch activation; cannot alter Patel code.
 */
export async function buildIntegratedPlan(input: {
  caseId?: string;
  measurements: EcrMeasurementInput;
  clinicalDiagnosis?: FinalDiagnosis | null;
  ecrAnalysisId?: string;
  clinicalEvidenceVersion?: string;
}): Promise<IntegratedPlanResult> {
  const caseId = input.caseId ?? randomUUID();
  const evidence = buildEvidencePackage({ measurements: input.measurements });
  const patelNetwork = await runPatelNetwork({
    evidence,
    measurements: input.measurements,
  });

  const patelClassification: PatelResult | undefined = patelNetwork.code
    ? {
        status:
          patelNetwork.classificationStatus === "complete"
            ? "complete"
            : "incomplete",
        height: {
          status: patelNetwork.height.value != null ? "complete" : "indeterminate",
          value: patelNetwork.height.value,
          mostApicalExtentMmFromCEJ:
            evidence.measurements.mostApicalExtentMmFromCEJ,
          relativeToLocalCrest: null,
          rootThird: null,
          borderline: patelNetwork.height.borderline,
          plausibleValues: patelNetwork.height.value
            ? [patelNetwork.height.value]
            : [],
          notes: [],
        },
        circumference: {
          status:
            patelNetwork.circumference.value != null ? "complete" : "indeterminate",
          value: patelNetwork.circumference.value,
          maximumAngleDegrees: patelNetwork.circumference.maximumAngleDegrees,
          borderline: patelNetwork.circumference.borderline,
          plausibleValues: patelNetwork.circumference.value
            ? [patelNetwork.circumference.value]
            : [],
          notes: [],
        },
        canalProximity: {
          status:
            patelNetwork.canalProximity.value != null ? "complete" : "indeterminate",
          value: patelNetwork.canalProximity.value,
          minimumSeparationMm: patelNetwork.canalProximity.minimumSeparationMm,
          lowerUncertaintyBoundMm:
            evidence.measurements.separationUncertaintyLowerBoundMm,
          canalWallStatus: null,
          borderline: patelNetwork.canalProximity.borderline,
          plausibleValues: patelNetwork.canalProximity.value
            ? [patelNetwork.canalProximity.value]
            : [],
          notes: [
            "p means probable pulpal involvement on imaging, not pulp necrosis",
          ],
        },
        code: patelNetwork.code,
        rootsAffected: evidence.targetTooth.rootsAffected,
        manualReviewRequired: patelNetwork.requiresSpecialistReview,
      }
    : undefined;

  const provisionalPlan = runTreatmentAgentNetwork({
    evidence,
    network: patelNetwork,
    measurements: input.measurements,
    clinical: input.clinicalDiagnosis,
  });

  const walls = assertIntegrationWalls({
    patelBefore: patelClassification,
    patelAfter: patelClassification,
    clinical: input.clinicalDiagnosis,
    plan: provisionalPlan,
  });

  if (walls.length) {
    provisionalPlan.warnings.push(...walls);
    provisionalPlan.planStatus = "needs_specialist_review";
  }

  const clinicalOk = input.clinicalDiagnosis?.status === "diagnosable";
  const cbctOk =
    patelNetwork.classificationStatus === "complete" &&
    !patelNetwork.requiresSpecialistReview;

  const unified = unifiedCaseSchema.parse({
    caseId,
    clinicalEvidenceVersion: input.clinicalEvidenceVersion,
    clinicalDiagnosis: input.clinicalDiagnosis ?? undefined,
    ecrAnalysisId: input.ecrAnalysisId,
    patelClassification,
    provisionalPlan,
    treatmentFeasibility: {
      clinicalDiagnosisVerified: Boolean(clinicalOk),
      cbctClassificationVerified: Boolean(cbctOk),
      pulpalDiagnosisAvailable: Boolean(input.clinicalDiagnosis?.pulpalDiagnosis),
      vitalityEstablishedByCbct: false,
      patelPUsedAsPulpNecrosis: false,
      clinicalMayAlterPatelCode: false,
      overlappingTreatmentAlternativesPreserved:
        provisionalPlan.candidates.length +
          provisionalPlan.alternativeStrategies.length >
        1,
      planStatus: provisionalPlan.planStatus,
    },
    walls: {
      clinicalCannotChangePatelCode: true,
      patelPNotPulpalEvidence: true,
      cbctCannotEstablishVitality: true,
    },
  });

  return {
    unified,
    evidenceHash: evidence.caseEvidenceHash,
    patelNetwork,
    provisionalPlan,
    deterministicReferenceCode: patelNetwork.deterministicReferenceCode,
  };
}
