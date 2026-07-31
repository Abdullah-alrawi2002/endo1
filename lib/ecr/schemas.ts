import { z } from "zod";

/** Product / schema freeze for CBCT-only ECR analysis. */
export const ECR_SCHEMA_VERSION = "ECR_CBCT_1.0" as const;
export const EVIDENCE_SCOPE = "CBCT_ONLY" as const;
export const PATEL_CLASSIFICATION = {
  name: "PATEL_ECR_3D" as const,
  version: "2018" as const,
};

export const QUALITY_STATUSES = ["pass", "conditional", "fail"] as const;
export const OOD_STATUSES = [
  "in_domain",
  "near_domain",
  "out_of_domain",
  "unknown",
] as const;

export const ECR_DIFFERENTIAL_STATUSES = [
  "appearance_consistent_with_ecr",
  "internal_resorption_more_likely",
  "caries_more_likely",
  "external_inflammatory_or_surface_resorption_more_likely",
  "replacement_resorption_or_ankylosis_more_likely",
  "perforation_or_iatrogenic_defect_more_likely",
  "fracture_or_developmental_defect_more_likely",
  "artifact_limited",
  "indeterminate",
  "no_candidate",
] as const;

export const PATEL_HEIGHTS = [1, 2, 3, 4] as const;
export const PATEL_CIRCUMFERENCES = ["A", "B", "C", "D"] as const;
export const PATEL_CANAL_PROXIMITIES = ["d", "p"] as const;

export const PATEL_COMPONENT_STATUSES = [
  "complete",
  "indeterminate",
  "not_assessable",
] as const;

export const PATEL_OVERALL_STATUSES = [
  "complete",
  "incomplete",
  "abstained",
  "rejected",
] as const;

export const ACCESS_PROXY_VALUES = [
  "favorable",
  "possible",
  "unfavorable",
  "not_assessable",
] as const;

export const STRUCTURAL_CONTINUITY_VALUES = [
  "preserved",
  "reduced",
  "severely_compromised",
  "not_assessable",
] as const;

export const TRIAD = ["absent", "present", "indeterminate", "not_applicable"] as const;

export const OPTION_STATUSES = [
  "favored_on_imaging",
  "possible",
  "not_assessable",
  "imaging_discouraged",
] as const;

export const MANAGEMENT_OPTIONS = [
  "external_repair_without_automatic_RCT",
  "external_repair_with_RCT_consideration",
  "internal_repair_with_RCT",
  "intentional_replantation_discussion",
  "periodic_review",
  "decoronation_or_removal_replacement_discussion",
  "extraction_discussion",
] as const;

export const REVIEW_DECISIONS = [
  "confirm",
  "modify",
  "abstain",
  "request_radiology_review",
] as const;

export const JOB_STATUSES = [
  "queued",
  "running",
  "awaiting_mask_review",
  "completed",
  "failed",
  "abstained",
] as const;

export const heightComponentSchema = z.object({
  status: z.enum(PATEL_COMPONENT_STATUSES),
  value: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4)]).nullable(),
  mostApicalExtentMmFromCEJ: z.number().nullable(),
  relativeToLocalCrest: z
    .enum(["coronal_to_crest", "at_crest", "apical_to_crest", "indeterminate"])
    .nullable(),
  rootThird: z
    .enum(["cej_or_coronal_to_crest", "coronal", "middle", "apical", "indeterminate"])
    .nullable(),
  borderline: z.boolean(),
  plausibleValues: z.array(z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4)])).max(2),
  notes: z.array(z.string().max(400)).max(20).default([]),
});

export const circumferenceComponentSchema = z.object({
  status: z.enum(PATEL_COMPONENT_STATUSES),
  value: z.enum(PATEL_CIRCUMFERENCES).nullable(),
  maximumAngleDegrees: z.number().min(0).max(360).nullable(),
  borderline: z.boolean(),
  plausibleValues: z.array(z.enum(PATEL_CIRCUMFERENCES)).max(2),
  sourcePlaneIndex: z.number().int().nullable().optional(),
  notes: z.array(z.string().max(400)).max(20).default([]),
});

export const canalProximityComponentSchema = z.object({
  status: z.enum(PATEL_COMPONENT_STATUSES),
  value: z.enum(PATEL_CANAL_PROXIMITIES).nullable(),
  minimumSeparationMm: z.number().nullable(),
  lowerUncertaintyBoundMm: z.number().nullable().optional(),
  canalWallStatus: z
    .enum([
      "continuous_dentine_barrier",
      "probable_contact",
      "breach_or_intersection",
      "unresolved",
    ])
    .nullable(),
  borderline: z.boolean(),
  plausibleValues: z.array(z.enum(PATEL_CANAL_PROXIMITIES)).max(2),
  notes: z.array(z.string().max(400)).max(20).default([]),
});

export const patelResultSchema = z.object({
  status: z.enum(PATEL_OVERALL_STATUSES),
  height: heightComponentSchema,
  circumference: circumferenceComponentSchema,
  canalProximity: canalProximityComponentSchema,
  /** Concatenated code such as 2Bp; null when incomplete/abstained. */
  code: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  rootsAffected: z.array(z.string().max(40)).max(10).default([]),
  manualReviewRequired: z.boolean(),
});

export const planningFeaturesSchema = z.object({
  portalSurface: z.string().max(80).nullable(),
  portalAreaMm2: z.number().nullable(),
  portalSupracrestal: z.enum(TRIAD).nullable(),
  lesionVolumeMm3: z.number().nullable(),
  lesionMaxDepthMm: z.number().nullable(),
  lesionMaxWidthMm: z.number().nullable(),
  externalAccessProxy: z.enum(ACCESS_PROXY_VALUES),
  internalAccessProxy: z.enum(ACCESS_PROXY_VALUES),
  structuralContinuity: z.enum(STRUCTURAL_CONTINUITY_VALUES),
  furcationInvolvement: z.enum(TRIAD),
  boneCrestLossProxy: z.enum(TRIAD).nullable(),
  adjacentAnatomyWarnings: z.array(z.string().max(300)).max(30).default([]),
  existingTreatmentFindings: z.array(z.string().max(200)).max(20).default([]),
  fractureRiskProxy: z
    .enum(["low", "moderate", "high", "not_assessable"])
    .default("not_assessable"),
  rootFormForReplantation: z
    .enum(["compatible", "imaging_prohibitive", "not_assessable"])
    .default("not_assessable"),
});

export const managementOptionSchema = z.object({
  option: z.enum(MANAGEMENT_OPTIONS),
  status: z.enum(OPTION_STATUSES),
  supportingRuleIds: z.array(z.string().max(80)).min(1).max(20),
  supportingCbctFeatures: z.array(z.string().max(300)).max(20).default([]),
  limitingCbctFeatures: z.array(z.string().max(300)).max(20).default([]),
  requiresClinicalConfirmation: z.array(z.string().max(300)).min(1).max(20),
  evidenceVersion: z.string().max(80),
});

export const managementSupportSchema = z.object({
  rulesetVersion: z.string().max(80),
  outputType: z.literal("conditional_imaging_option_set"),
  options: z.array(managementOptionSchema).max(20),
  definitiveTreatmentPlanAvailable: z.literal(false),
  cannotDetermineFromCbct: z.array(z.string().max(300)).max(40),
});

export const scanQualitySchema = z.object({
  qualityStatus: z.enum(QUALITY_STATUSES),
  oodStatus: z.enum(OOD_STATUSES),
  seriesChecksum: z.string().max(128).optional(),
  nativeVoxelMm: z.tuple([z.number(), z.number(), z.number()]).nullable(),
  fovCompleteForTarget: z.boolean(),
  targetStructuresVisible: z.object({
    crownRootComplex: z.boolean(),
    cejRegion: z.boolean(),
    localAlveolarCrest: z.boolean(),
    apex: z.boolean(),
    canalBoundaryNearLesion: z.boolean(),
    lesionMargins: z.boolean(),
  }),
  artifactWarnings: z.array(z.string().max(300)).max(40).default([]),
  qualityGateFailures: z.array(z.string().max(300)).max(40).default([]),
});

export const ecrAssessmentSchema = z.object({
  status: z.enum(ECR_DIFFERENTIAL_STATUSES),
  differentialCandidates: z
    .array(
      z.object({
        label: z.enum(ECR_DIFFERENTIAL_STATUSES),
        rationale: z.string().max(500),
      }),
    )
    .max(10)
    .default([]),
  modelVersion: z.string().max(80).default("manual-or-semiauto-phase1"),
  maskReviewStatus: z.enum([
    "not_started",
    "proposed",
    "clinician_reviewed",
    "clinician_corrected",
  ]),
});

export const ecrResultSchema = z.object({
  schemaVersion: z.literal(ECR_SCHEMA_VERSION),
  evidenceScope: z.literal(EVIDENCE_SCOPE),
  classificationSystem: z.object({
    name: z.literal(PATEL_CLASSIFICATION.name),
    version: z.literal(PATEL_CLASSIFICATION.version),
  }),
  analysisId: z.string().min(8).max(120),
  scan: scanQualitySchema,
  target: z.object({
    toothLabel: z.string().min(1).max(40),
    toothSelectionSource: z.enum(["viewer", "upload_form", "unknown"]),
    rootsAffected: z.array(z.string().max(40)).max(10),
  }),
  ecrAssessment: ecrAssessmentSchema,
  patel: patelResultSchema,
  planningFeatures: planningFeaturesSchema,
  managementSupport: managementSupportSchema,
  review: z.object({
    specialistConfirmationRequired: z.literal(true),
    entireVolumeInterpretationRequired: z.literal(true),
    decision: z.enum(REVIEW_DECISIONS).nullable().default(null),
    reviewerNotes: z.string().max(4000).nullable().default(null),
    decidedAt: z.number().nullable().default(null),
    reviewerId: z.string().max(200).nullable().default(null),
  }),
  outputLabel: z.literal(
    "CBCT-derived ECR classification and treatment-planning options — not a definitive treatment plan",
  ),
  warnings: z.array(z.string().max(500)).max(40).default([]),
  createdAt: z.number(),
  updatedAt: z.number(),
});

export type EcrResult = z.infer<typeof ecrResultSchema>;
export type PatelResult = z.infer<typeof patelResultSchema>;
export type ManagementOption = z.infer<typeof managementOptionSchema>;
export type ManagementSupport = z.infer<typeof managementSupportSchema>;
export type PlanningFeatures = z.infer<typeof planningFeaturesSchema>;
export type ScanQuality = z.infer<typeof scanQualitySchema>;

/**
 * Measurement inputs for Phase-1 deterministic geometry.
 * Later phases may derive these from masks; the verifier still consumes this shape.
 */
export const ecrMeasurementInputSchema = z.object({
  toothLabel: z.string().min(1).max(40),
  rootsAffected: z.array(z.string().max(40)).min(1).max(10).default(["single_root"]),
  /** Most apical lesion extent along root centerline from CEJ (mm). */
  mostApicalExtentMmFromCEJ: z.number().min(0).max(40).nullable(),
  rootLengthCejToApexMm: z.number().min(1).max(40).nullable(),
  localCrestDistanceMmFromCEJ: z.number().min(0).max(20).nullable(),
  /** Maximum circumferential union angle in degrees. */
  maximumCircumferenceDegrees: z.number().min(0).max(360).nullable(),
  /** Optional angular uncertainty for borderline circumference. */
  circumferenceUncertaintyDegrees: z.number().min(0).max(90).nullable().optional(),
  /** Minimum lesion-to-canal surface distance (mm). Null if unresolved. */
  minimumLesionCanalSeparationMm: z.number().min(0).max(10).nullable(),
  /** Lower uncertainty bound for separation; used for d/p borderline. */
  separationUncertaintyLowerBoundMm: z.number().min(-1).max(10).nullable().optional(),
  lesionCanalContactOrIntersection: z.boolean().nullable(),
  continuousDentineBarrierVisible: z.boolean().nullable(),
  ecrDifferential: z.enum(ECR_DIFFERENTIAL_STATUSES),
  maskReviewStatus: z.enum([
    "not_started",
    "proposed",
    "clinician_reviewed",
    "clinician_corrected",
  ]),
  scan: scanQualitySchema,
  planning: planningFeaturesSchema.partial().optional(),
  portalSurface: z.string().max(80).optional(),
  portalAreaMm2: z.number().optional(),
  lesionVolumeMm3: z.number().optional(),
});

export type EcrMeasurementInput = z.infer<typeof ecrMeasurementInputSchema>;

export const ecrAnalysisCreateSchema = z.object({
  toothLabel: z.string().min(1).max(40),
  seriesLabel: z.string().max(200).optional(),
  measurements: ecrMeasurementInputSchema.optional(),
  /** Opaque reference when DICOM was stored elsewhere. */
  dicomReference: z.string().max(500).optional(),
  notes: z.string().max(2000).optional(),
});

export const ecrReviewSchema = z.object({
  decision: z.enum(REVIEW_DECISIONS),
  reviewerId: z.string().min(2).max(200),
  reviewerNotes: z.string().max(4000).optional(),
  modifiedMeasurements: ecrMeasurementInputSchema.optional(),
});

export const ecrJobSchema = z.object({
  analysisId: z.string(),
  jobStatus: z.enum(JOB_STATUSES),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().nullable().optional(),
  result: ecrResultSchema.nullable(),
  /** Agent-network Patel classification (evidence-first); optional for Phase-1 jobs. */
  patelNetwork: z.unknown().nullable().optional(),
  /** Provisional overlapping treatment plan from treatment-agent network. */
  provisionalPlan: z.unknown().nullable().optional(),
  evidenceHash: z.string().nullable().optional(),
  audit: z
    .array(
      z.object({
        at: z.number(),
        actor: z.string().max(200),
        action: z.string().max(120),
        detail: z.string().max(1000).optional(),
      }),
    )
    .max(500)
    .default([]),
});

export type EcrJob = z.infer<typeof ecrJobSchema>;
