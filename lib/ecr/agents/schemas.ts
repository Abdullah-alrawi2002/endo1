import { z } from "zod";
import {
  ACCESS_PROXY_VALUES,
  ECR_DIFFERENTIAL_STATUSES,
  OOD_STATUSES,
  PATEL_CANAL_PROXIMITIES,
  PATEL_CIRCUMFERENCES,
  PATEL_HEIGHTS,
  QUALITY_STATUSES,
  STRUCTURAL_CONTINUITY_VALUES,
  MANAGEMENT_OPTIONS,
  OPTION_STATUSES,
} from "@/lib/ecr/schemas";

export const AGENT_CONTRACT_VERSION = "3.0" as const;
export const ECR_EVIDENCE_PACKAGE_VERSION = "CBCT_EVIDENCE_3.0" as const;

export const PATEL_AGENT_ROLES = [
  "patel_height",
  "patel_circumference",
  "patel_canal",
  "patel_adjudicator",
  "classification_critic",
] as const;

export const TREATMENT_AGENT_ROLES = [
  "external_repair",
  "internal_repair",
  "preservation",
  "surgical_alternatives",
  "treatment_critic",
  "plan_synthesizer",
  "final_safety",
] as const;

export const AGENT_RESULT_STATUSES = [
  "complete",
  "indeterminate",
  "abstained",
  "rejected",
  "needs_specialist_review",
] as const;

/** Component-agent output — never includes peer conclusions or deterministic codes. */
export const componentAgentResultSchema = z.object({
  agentRole: z.enum(["patel_height", "patel_circumference", "patel_canal"]),
  agentContractVersion: z.literal(AGENT_CONTRACT_VERSION),
  promptVersion: z.string().max(80),
  modelId: z.string().max(120),
  caseEvidenceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  status: z.enum(["complete", "indeterminate", "abstained", "rejected"]),
  conclusion: z.union([
    z.literal(1),
    z.literal(2),
    z.literal(3),
    z.literal(4),
    z.enum(PATEL_CIRCUMFERENCES),
    z.enum(PATEL_CANAL_PROXIMITIES),
    z.null(),
  ]),
  alternatives: z
    .array(
      z.union([
        z.literal(1),
        z.literal(2),
        z.literal(3),
        z.literal(4),
        z.enum(PATEL_CIRCUMFERENCES),
        z.enum(PATEL_CANAL_PROXIMITIES),
      ]),
    )
    .max(4)
    .default([]),
  evidenceIds: z.array(z.string().max(120)).min(1).max(20),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string().max(400)).max(20).default([]),
  abstain: z.boolean(),
});

export const adjudicatorResultSchema = z.object({
  agentRole: z.literal("patel_adjudicator"),
  agentContractVersion: z.literal(AGENT_CONTRACT_VERSION),
  promptVersion: z.string().max(80),
  modelId: z.string().max(120),
  caseEvidenceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  status: z.enum(AGENT_RESULT_STATUSES),
  patelCode: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  height: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.null()]),
  circumference: z.enum([...PATEL_CIRCUMFERENCES]).nullable(),
  canalProximity: z.enum([...PATEL_CANAL_PROXIMITIES]).nullable(),
  evidenceIds: z.array(z.string().max(120)).max(40).default([]),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string().max(400)).max(20).default([]),
  abstain: z.boolean(),
});

export const criticResultSchema = z.object({
  agentRole: z.literal("classification_critic"),
  agentContractVersion: z.literal(AGENT_CONTRACT_VERSION),
  promptVersion: z.string().max(80),
  modelId: z.string().max(120),
  caseEvidenceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  status: z.enum(AGENT_RESULT_STATUSES),
  criticStatus: z.enum(["reproduced", "contradicted", "abstained"]),
  reconstructedCode: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  contradiction: z.string().max(1000).nullable(),
  evidenceIds: z.array(z.string().max(120)).max(40).default([]),
  conciseRationale: z.string().max(2000),
  abstain: z.boolean(),
});

export const treatmentCandidateSchema = z.object({
  strategy: z.enum(MANAGEMENT_OPTIONS),
  status: z.enum(OPTION_STATUSES),
  treatmentObjective: z.string().max(1000),
  proceduralSequence: z.array(z.string().max(400)).max(30).default([]),
  supportingEvidenceIds: z.array(z.string().max(120)).max(40).default([]),
  limitingEvidenceIds: z.array(z.string().max(120)).max(40).default([]),
  requiredClinicalConfirmations: z.array(z.string().max(300)).max(30).default([]),
  activateIf: z.array(z.string().max(300)).max(20).default([]),
  rejectIf: z.array(z.string().max(300)).max(20).default([]),
  stopConditions: z.array(z.string().max(300)).max(20).default([]),
  alternatives: z.array(z.string().max(200)).max(10).default([]),
  followUp: z.array(z.string().max(300)).max(20).default([]),
  sourceRuleIds: z.array(z.string().max(80)).max(20).default([]),
  evidenceVersion: z.string().max(80),
});

export const provisionalPlanSchema = z.object({
  planStatus: z.enum([
    "provisional_cbct_based",
    "provisional_integrated",
    "needs_specialist_review",
    "not_generated",
  ]),
  primaryStrategy: z.enum(MANAGEMENT_OPTIONS).nullable(),
  candidates: z.array(treatmentCandidateSchema).max(20),
  alternativeStrategies: z.array(treatmentCandidateSchema).max(10),
  proceduralSequence: z.array(z.string().max(400)).max(40).default([]),
  requiredClinicalConfirmations: z.array(z.string().max(300)).max(40).default([]),
  decisionCheckpoints: z.array(z.string().max(300)).max(40).default([]),
  stopConditions: z.array(z.string().max(300)).max(40).default([]),
  followUp: z.array(z.string().max(300)).max(20).default([]),
  sourceRuleIds: z.array(z.string().max(80)).max(40).default([]),
  unsupportedVariables: z.array(z.string().max(200)).max(40).default([]),
  definitiveTreatmentPlanAvailable: z.literal(false),
  caseEvidenceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  criticStatus: z.enum(["passed", "revision_requested", "rejected", "not_run"]),
  safetyStatus: z.enum(["passed", "rejected", "not_run"]),
  warnings: z.array(z.string().max(500)).max(40).default([]),
});

export const heightEvidenceViewSchema = z.object({
  caseEvidenceHash: z.string(),
  evidenceIds: z.array(z.string()),
  mostApicalExtentMmFromCEJ: z.number().nullable(),
  rootLengthCejToApexMm: z.number().nullable(),
  localCrestDistanceMmFromCEJ: z.number().nullable(),
  structuresVisible: z.object({
    cejRegion: z.boolean(),
    localAlveolarCrest: z.boolean(),
    apex: z.boolean(),
    crownRootComplex: z.boolean(),
  }),
  prohibited: z.array(z.string()),
});

export const circumferenceEvidenceViewSchema = z.object({
  caseEvidenceHash: z.string(),
  evidenceIds: z.array(z.string()),
  maximumCircumferenceDegrees: z.number().nullable(),
  circumferenceUncertaintyDegrees: z.number().nullable(),
  structuresVisible: z.object({
    lesionMargins: z.boolean(),
  }),
  prohibited: z.array(z.string()),
});

export const canalEvidenceViewSchema = z.object({
  caseEvidenceHash: z.string(),
  evidenceIds: z.array(z.string()),
  minimumLesionCanalSeparationMm: z.number().nullable(),
  separationUncertaintyLowerBoundMm: z.number().nullable(),
  lesionCanalContactOrIntersection: z.boolean().nullable(),
  continuousDentineBarrierVisible: z.boolean().nullable(),
  structuresVisible: z.object({
    canalBoundaryNearLesion: z.boolean(),
    lesionMargins: z.boolean(),
  }),
  prohibited: z.array(z.string()),
});

export const ecrEvidencePackageSchema = z.object({
  packageVersion: z.literal(ECR_EVIDENCE_PACKAGE_VERSION),
  caseEvidenceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  scope: z.literal("CBCT_ONLY"),
  quality: z.object({
    qualityStatus: z.enum(QUALITY_STATUSES),
    oodStatus: z.enum(OOD_STATUSES),
    fovCompleteForTarget: z.boolean(),
    nativeVoxelMm: z.tuple([z.number(), z.number(), z.number()]).nullable(),
    artifactWarnings: z.array(z.string()).default([]),
    qualityGateFailures: z.array(z.string()).default([]),
    targetStructuresVisible: z.object({
      crownRootComplex: z.boolean(),
      cejRegion: z.boolean(),
      localAlveolarCrest: z.boolean(),
      apex: z.boolean(),
      canalBoundaryNearLesion: z.boolean(),
      lesionMargins: z.boolean(),
    }),
  }),
  ecrDifferential: z.enum(ECR_DIFFERENTIAL_STATUSES),
  maskReviewStatus: z.enum([
    "not_started",
    "proposed",
    "clinician_reviewed",
    "clinician_corrected",
  ]),
  targetTooth: z.object({
    label: z.string(),
    rootsAffected: z.array(z.string()),
  }),
  /** Raw measurements — NOT Patel categories or codes. */
  measurements: z.object({
    mostApicalExtentMmFromCEJ: z.number().nullable(),
    rootLengthCejToApexMm: z.number().nullable(),
    localCrestDistanceMmFromCEJ: z.number().nullable(),
    maximumCircumferenceDegrees: z.number().nullable(),
    circumferenceUncertaintyDegrees: z.number().nullable(),
    minimumLesionCanalSeparationMm: z.number().nullable(),
    separationUncertaintyLowerBoundMm: z.number().nullable(),
    lesionCanalContactOrIntersection: z.boolean().nullable(),
    continuousDentineBarrierVisible: z.boolean().nullable(),
  }),
  planningFeatures: z.object({
    portalSurface: z.string().nullable(),
    portalAreaMm2: z.number().nullable(),
    externalAccessProxy: z.enum(ACCESS_PROXY_VALUES),
    internalAccessProxy: z.enum(ACCESS_PROXY_VALUES),
    structuralContinuity: z.enum(STRUCTURAL_CONTINUITY_VALUES),
    fractureRiskProxy: z.string(),
    rootFormForReplantation: z.string(),
    adjacentAnatomyWarnings: z.array(z.string()),
  }),
  evidenceIds: z.array(z.string().max(120)).min(1).max(80),
  prohibitedInferences: z.array(z.string()).default([
    "pulp_vitality",
    "pulpal_diagnosis",
    "symptoms",
    "periodontal_probing",
    "clinical_access",
    "patient_preference",
    "deterministic_patel_code",
    "peer_component_conclusion",
  ]),
});

export const patelNetworkResultSchema = z.object({
  classificationStatus: z.enum([
    "complete",
    "incomplete",
    "abstained",
    "rejected",
    "needs_specialist_review",
  ]),
  decisionSource: z.enum([
    "llm_patel_adjudicator",
    "offline_evidence_interpreter",
    "abstained",
  ]),
  caseEvidenceHash: z.string(),
  height: z.object({
    value: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.null()]),
    alternatives: z.array(z.number()).default([]),
    borderline: z.boolean(),
  }),
  circumference: z.object({
    value: z.enum([...PATEL_CIRCUMFERENCES]).nullable(),
    maximumAngleDegrees: z.number().nullable(),
    alternatives: z.array(z.string()).default([]),
    borderline: z.boolean(),
  }),
  canalProximity: z.object({
    value: z.enum([...PATEL_CANAL_PROXIMITIES]).nullable(),
    minimumSeparationMm: z.number().nullable(),
    alternatives: z.array(z.string()).default([]),
    borderline: z.boolean(),
  }),
  code: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  criticStatus: z.enum(["reproduced", "contradicted", "not_run", "abstained"]),
  deterministicCompatibility: z.enum(["passed", "failed", "not_run"]),
  /** Hidden reference only — never shown to agents. */
  deterministicReferenceCode: z.string().nullable(),
  requiresSpecialistReview: z.boolean(),
  componentAgents: z.array(componentAgentResultSchema).max(3),
  adjudicator: adjudicatorResultSchema.nullable(),
  critic: criticResultSchema.nullable(),
  warnings: z.array(z.string()).max(40).default([]),
});

export type ComponentAgentResult = z.infer<typeof componentAgentResultSchema>;
export type AdjudicatorResult = z.infer<typeof adjudicatorResultSchema>;
export type CriticResult = z.infer<typeof criticResultSchema>;
export type TreatmentCandidate = z.infer<typeof treatmentCandidateSchema>;
export type ProvisionalPlan = z.infer<typeof provisionalPlanSchema>;
export type EcrEvidencePackage = z.infer<typeof ecrEvidencePackageSchema>;
export type PatelNetworkResult = z.infer<typeof patelNetworkResultSchema>;
export type HeightEvidenceView = z.infer<typeof heightEvidenceViewSchema>;
export type CircumferenceEvidenceView = z.infer<typeof circumferenceEvidenceViewSchema>;
export type CanalEvidenceView = z.infer<typeof canalEvidenceViewSchema>;

// Keep height enum export for prompts.
void PATEL_HEIGHTS;
