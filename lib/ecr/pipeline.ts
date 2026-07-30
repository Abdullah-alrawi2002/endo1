import { randomUUID } from "crypto";
import { calculatePatelFromMeasurements } from "@/lib/ecr/geometry";
import { rankManagementOptions } from "@/lib/ecr/rules/treatment-options";
import {
  assertNoDefinitiveTreatmentPlan,
  verifyEcrAnalysis,
} from "@/lib/ecr/rules/verifier";
import {
  ECR_SCHEMA_VERSION,
  EVIDENCE_SCOPE,
  PATEL_CLASSIFICATION,
  type EcrMeasurementInput,
  type EcrResult,
  type PlanningFeatures,
  ecrResultSchema,
} from "@/lib/ecr/schemas";

function defaultPlanning(
  input: EcrMeasurementInput,
): PlanningFeatures {
  const p = input.planning ?? {};
  return {
    portalSurface: input.portalSurface ?? p.portalSurface ?? null,
    portalAreaMm2: input.portalAreaMm2 ?? p.portalAreaMm2 ?? null,
    portalSupracrestal: p.portalSupracrestal ?? null,
    lesionVolumeMm3: input.lesionVolumeMm3 ?? p.lesionVolumeMm3 ?? null,
    lesionMaxDepthMm: p.lesionMaxDepthMm ?? null,
    lesionMaxWidthMm: p.lesionMaxWidthMm ?? null,
    externalAccessProxy: p.externalAccessProxy ?? "not_assessable",
    internalAccessProxy: p.internalAccessProxy ?? "not_assessable",
    structuralContinuity: p.structuralContinuity ?? "not_assessable",
    furcationInvolvement: p.furcationInvolvement ?? "not_applicable",
    boneCrestLossProxy: p.boneCrestLossProxy ?? null,
    adjacentAnatomyWarnings: p.adjacentAnatomyWarnings ?? [],
    existingTreatmentFindings: p.existingTreatmentFindings ?? [],
    fractureRiskProxy: p.fractureRiskProxy ?? "not_assessable",
    rootFormForReplantation: p.rootFormForReplantation ?? "not_assessable",
  };
}

/**
 * Phase-1 ECR pipeline: measurements → Patel → verify → conditional options.
 * No LLM on the classification path.
 */
export function runEcrPipeline(
  input: EcrMeasurementInput,
  analysisId: string = randomUUID(),
): EcrResult {
  const now = Date.now();
  const planning = defaultPlanning(input);
  let patel = calculatePatelFromMeasurements(input);

  const verification = verifyEcrAnalysis({
    scan: input.scan,
    ecrStatus: input.ecrDifferential,
    maskReviewStatus: input.maskReviewStatus,
    patel,
  });

  const warnings = verification.issues.map((i) => `[${i.severity}] ${i.message}`);

  if (!verification.ok || !verification.mayRankOptions) {
    patel = {
      ...patel,
      status: verification.issues.some((i) => i.severity === "reject")
        ? "rejected"
        : "abstained",
      code: null,
      manualReviewRequired: true,
    };
  }

  const managementSupport = verification.mayRankOptions
    ? rankManagementOptions({ patel, planning })
    : {
        rulesetVersion: "ESE_RR_2023_CORRECTED_v1",
        outputType: "conditional_imaging_option_set" as const,
        options: [],
        definitiveTreatmentPlanAvailable: false as const,
        cannotDetermineFromCbct: [
          "pulp sensibility / pulpal diagnosis",
          "apical diagnosis",
          "symptoms and lesion activity",
          "bleeding on probing / periodontal probing depths",
          "clinical accessibility and isolatability",
          "definitive restorability and esthetic acceptability",
          "periodontal prognosis",
          "patient age, growth status, medical risk, preferences, and tolerance",
          "whether RCT, monitoring, intentional replantation, decoronation, or extraction is definitively indicated",
        ],
      };

  const safety = assertNoDefinitiveTreatmentPlan({ managementSupport });
  if (safety.length) {
    warnings.push(...safety.map((s) => `[${s.severity}] ${s.message}`));
  }

  const result: EcrResult = {
    schemaVersion: ECR_SCHEMA_VERSION,
    evidenceScope: EVIDENCE_SCOPE,
    classificationSystem: {
      name: PATEL_CLASSIFICATION.name,
      version: PATEL_CLASSIFICATION.version,
    },
    analysisId,
    scan: input.scan,
    target: {
      toothLabel: input.toothLabel,
      toothSelectionSource: "upload_form",
      rootsAffected: input.rootsAffected,
    },
    ecrAssessment: {
      status: input.ecrDifferential,
      differentialCandidates: [],
      modelVersion: "manual-or-semiauto-phase1",
      maskReviewStatus: input.maskReviewStatus,
    },
    patel,
    planningFeatures: planning,
    managementSupport,
    review: {
      specialistConfirmationRequired: true,
      entireVolumeInterpretationRequired: true,
      decision: null,
      reviewerNotes: null,
      decidedAt: null,
      reviewerId: null,
    },
    outputLabel:
      "CBCT-derived ECR classification and treatment-planning options — not a definitive treatment plan",
    warnings,
    createdAt: now,
    updatedAt: now,
  };

  return ecrResultSchema.parse(result);
}
