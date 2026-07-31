import { z } from "zod";
import {
  finalDiagnosisSchema,
  type FinalDiagnosis,
} from "@/lib/schemas/clinical-case";
import {
  patelResultSchema,
  planningFeaturesSchema,
  type PatelResult,
} from "@/lib/ecr/schemas";
import {
  provisionalPlanSchema,
  type ProvisionalPlan,
} from "@/lib/ecr/agents/schemas";

/**
 * Shared domain object connecting clinical diagnosis + ECR CBCT products.
 * Walls:
 * - Clinical diagnosis may influence RCT branch.
 * - Clinical diagnosis cannot change the Patel code.
 * - Patel p cannot enter the pulpal agent's evidence.
 * - CBCT cannot establish vitality.
 */
export const treatmentFeasibilityChecklistSchema = z.object({
  clinicalDiagnosisVerified: z.boolean(),
  cbctClassificationVerified: z.boolean(),
  pulpalDiagnosisAvailable: z.boolean(),
  vitalityEstablishedByCbct: z.literal(false),
  patelPUsedAsPulpNecrosis: z.literal(false),
  clinicalMayAlterPatelCode: z.literal(false),
  overlappingTreatmentAlternativesPreserved: z.boolean(),
  planStatus: z.enum([
    "provisional_cbct_based",
    "provisional_integrated",
    "needs_specialist_review",
    "not_generated",
  ]),
});

export const unifiedCaseSchema = z.object({
  caseId: z.string().min(1).max(120),
  clinicalEvidenceVersion: z.string().max(80).optional(),
  clinicalDiagnosis: finalDiagnosisSchema.optional(),
  ecrAnalysisId: z.string().max(120).optional(),
  patelClassification: patelResultSchema.optional(),
  treatmentFeasibility: treatmentFeasibilityChecklistSchema.optional(),
  provisionalPlan: provisionalPlanSchema.optional(),
  planningFeatures: planningFeaturesSchema.optional(),
  walls: z
    .object({
      clinicalCannotChangePatelCode: z.literal(true).default(true),
      patelPNotPulpalEvidence: z.literal(true).default(true),
      cbctCannotEstablishVitality: z.literal(true).default(true),
    })
    .default({
      clinicalCannotChangePatelCode: true,
      patelPNotPulpalEvidence: true,
      cbctCannotEstablishVitality: true,
    }),
});

export type UnifiedCase = z.infer<typeof unifiedCaseSchema>;
export type TreatmentFeasibilityChecklist = z.infer<
  typeof treatmentFeasibilityChecklistSchema
>;

export function assertIntegrationWalls(input: {
  patelBefore?: PatelResult | null;
  patelAfter?: PatelResult | null;
  clinical?: FinalDiagnosis | null;
  plan?: ProvisionalPlan | null;
}): string[] {
  const violations: string[] = [];
  if (
    input.patelBefore?.code &&
    input.patelAfter?.code &&
    input.patelBefore.code !== input.patelAfter.code &&
    input.clinical
  ) {
    violations.push(
      "Clinical diagnosis must not change Patel code (integration wall violation)",
    );
  }
  if (input.plan && input.plan.definitiveTreatmentPlanAvailable !== false) {
    violations.push("Definitive plan forbidden without full clinical+CBCT specialist confirmation");
  }
  return violations;
}
