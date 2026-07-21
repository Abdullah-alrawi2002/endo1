import { z } from "zod";

/** Frozen research taxonomy for this protocol. */
export const TAXONOMY_VERSION = "AAE_2009" as const;

/**
 * AAE 2009 pulpal diagnoses used in the clinical MVP.
 * Previously Treated / Previously Initiated Therapy are excluded at intake
 * (out_of_scope) rather than diagnosed by this release.
 */
export const PULPAL_DIAGNOSES = [
  "Normal Pulp",
  "Reversible Pulpitis",
  "Symptomatic Irreversible Pulpitis",
  "Asymptomatic Irreversible Pulpitis",
  "Pulp Necrosis",
] as const;

export const APICAL_DIAGNOSES = [
  "Normal Apical Tissues",
  "Symptomatic Apical Periodontitis",
  "Asymptomatic Apical Periodontitis",
  "Acute Apical Abscess",
  "Chronic Apical Abscess",
] as const;

export const DIAGNOSTIC_STATUSES = [
  "diagnosable",
  "insufficient_data",
  "conflicting_data",
  "out_of_scope",
] as const;

export type PulpalDiagnosis = (typeof PULPAL_DIAGNOSES)[number];
export type ApicalDiagnosis = (typeof APICAL_DIAGNOSES)[number];
export type DiagnosticStatus = (typeof DIAGNOSTIC_STATUSES)[number];

export const pulpalDiagnosisSchema = z.enum(PULPAL_DIAGNOSES);
export const apicalDiagnosisSchema = z.enum(APICAL_DIAGNOSES);
export const diagnosticStatusSchema = z.enum(DIAGNOSTIC_STATUSES);

/** Present/absent findings that may also be unknown. */
export const triadFindingSchema = z.enum(["absent", "present", "unknown"]);

/**
 * Optional clinical tests — never silently coerce missing → absent.
 * `invalid` / `unable_to_test` record that an attempt failed or was unreliable.
 */
export const optionalTestSchema = z.enum([
  "not_performed",
  "positive",
  "negative",
  "unable_to_test",
  "invalid",
]);

export const imagingFindingSchema = z.enum([
  "present",
  "absent",
  "unknown",
  "not_assessed",
]);

export const severitySchema = z.enum([
  "none",
  "mild",
  "moderate",
  "severe",
  "unknown",
  "not_assessed",
]);

export const testValiditySchema = z.enum([
  "valid",
  "invalid",
  "unable_to_test",
  "not_performed",
  "unknown",
]);

export const comparativeResponseSchema = z.enum([
  "not_compared",
  "similar_to_control",
  "exaggerated_vs_control",
  "reduced_vs_control",
  "absent_vs_control_positive",
  "unknown",
]);

export const toothIdentitySchema = z.object({
  universal: z.number().int().min(1).max(32),
  fdi: z.string().max(10).optional(),
  dentition: z.enum(["permanent", "primary", "unknown"]),
  apexMaturity: z.enum(["mature", "open", "unknown"]),
});

export const treatmentHistorySchema = z.enum([
  "untreated",
  "previously_initiated",
  "previously_obturated",
  "regenerative",
  "unknown",
]);

export const symptomsSchema = z.object({
  spontaneousPain: triadFindingSchema,
  nocturnalPain: triadFindingSchema,
  posturalPain: triadFindingSchema,
  referredPain: triadFindingSchema,
  thermalPainHistory: triadFindingSchema,
  heatResponse: z.enum([
    "not_performed",
    "none",
    "mild",
    "severe",
    "relieved_by_cold",
    "unable_to_test",
    "unknown",
  ]),
  coldRelievesPain: triadFindingSchema,
});

export const visualExamSchema = z.object({
  sinusTract: triadFindingSchema,
  sinusTractTraced: triadFindingSchema,
  swelling: triadFindingSchema,
  swellingSeverity: severitySchema,
  fluctuance: triadFindingSchema,
  pus: triadFindingSchema,
  rapidOnsetSwelling: triadFindingSchema,
  fever: triadFindingSchema,
  lymphadenopathy: triadFindingSchema,
  decay: triadFindingSchema,
  deepCariesOrExposure: triadFindingSchema,
  crownPresent: triadFindingSchema,
  crackSuspected: triadFindingSchema,
  traumaSigns: triadFindingSchema,
});

export const clinicalTestsSchema = z.object({
  cold: z.enum([
    "normal",
    "exaggerated_non_lingering",
    "lingering",
    "negative",
    "unable_to_test",
    "not_performed",
  ]),
  coldLingerSeconds: z.number().positive().optional(),
  coldComparedToControl: comparativeResponseSchema,
  coldValidity: testValiditySchema,
  coldRepeated: triadFindingSchema,
  ept: optionalTestSchema,
  eptComparedToControl: comparativeResponseSchema,
  eptValidity: testValiditySchema,
  eptRepeated: triadFindingSchema,
  percussion: severitySchema,
  palpation: severitySchema,
  biting: severitySchema,
  /** Structural / crack assessment — not a pulp sensibility test. */
  toothSloothBiting: optionalTestSchema,
  /** Structural assessment — not a pulp sensibility test. */
  transillumination: optionalTestSchema,
  fluorescentLight: optionalTestSchema,
});

export const periodontalSchema = z.object({
  probingDepthsMm: z.string().max(200).optional(),
  isolatedDeepPocket: triadFindingSchema,
  mobility: z.enum(["0", "1", "2", "3", "unknown", "not_assessed"]),
  occlusionTrauma: triadFindingSchema,
});

export const imagingFindingsSchema = z.object({
  periapicalRadiolucency: imagingFindingSchema,
  jShapedPeriapicalRadiolucency: imagingFindingSchema,
  widenedPeriodontalLigament: imagingFindingSchema,
  laminaDuraLoss: imagingFindingSchema,
  parlLocationNotes: z.string().max(500).optional(),
  multiplePaViews: triadFindingSchema,
  internalResorption: imagingFindingSchema,
  externalResorption: imagingFindingSchema,
});

export const confoundersSchema = z.object({
  recentAnesthesia: triadFindingSchema,
  calcificationSuspected: triadFindingSchema,
  poorIsolation: triadFindingSchema,
  generalizedLowResponsiveness: triadFindingSchema,
  recentTrauma: triadFindingSchema,
});

/**
 * Non-diagnostic CBCT research support payload.
 * Never establishes endodontic origin or forces an apical enum.
 */
export const ctSupportSchema = z.object({
  analysisId: z.string().min(8).max(120),
  status: z.enum(["completed", "partial", "failed"]),
  targetToothUniversal: z.number().int().min(1).max(32),
  clinicianSeedProvided: z.boolean(),
  qualityGatePassed: z.boolean(),
  candidateLowAttenuationRegion: z.boolean().nullable(),
  candidateLocation: z.string().max(300).nullable(),
  candidateVolumeMm3: z.number().nullable(),
  relativeAttenuationDropPercent: z.number().nullable(),
  artifactWarnings: z.array(z.string().max(300)).max(30),
  qualityGateFailures: z.array(z.string().max(300)).max(30),
  morphologyNotes: z.string().max(4000).nullable(),
  vertucciScreen: z.string().max(120).nullable(),
  canalLengthEstimateMm: z.number().nullable(),
  canalLengthEstimateNote: z.string().max(400).nullable(),
  clinicianReviewed: z.boolean(),
  reviewRequired: z.literal(true),
  provider: z.string().max(120),
});

export type CTSupport = z.infer<typeof ctSupportSchema>;

export const clinicalCaseSchema = z.object({
  taxonomyVersion: z.literal(TAXONOMY_VERSION).default(TAXONOMY_VERSION),
  tooth: toothIdentitySchema,
  treatmentHistory: treatmentHistorySchema,
  symptoms: symptomsSchema,
  visual: visualExamSchema,
  clinical: clinicalTestsSchema,
  periodontal: periodontalSchema,
  imaging: imagingFindingsSchema,
  confounders: confoundersSchema,
  /** Opaque server-side CT analysis id when the research module is enabled. */
  ctAnalysisId: z.string().min(8).max(120).optional(),
  /** Only populated server-side after retrieving a stored CT analysis. */
  ctSupport: ctSupportSchema.optional(),
  additionalNotes: z.string().max(4000).optional(),
});

export type ClinicalCase = z.infer<typeof clinicalCaseSchema>;

export const evidenceItemSchema = z.object({
  claim: z.string().max(500),
  source: z.enum([
    "history",
    "visual",
    "sensibility",
    "mechanical",
    "imaging",
    "periodontal",
    "confounder",
    "ct_support",
    "curriculum",
    "verifier",
  ]),
});

export const finalDiagnosisSchema = z.object({
  taxonomyVersion: z.literal(TAXONOMY_VERSION),
  status: diagnosticStatusSchema,
  pulpalDiagnosis: pulpalDiagnosisSchema.nullable(),
  apicalDiagnosis: apicalDiagnosisSchema.nullable(),
  finalDiagnosisLine: z.string().nullable(),
  evidenceFor: z.array(evidenceItemSchema).max(40),
  evidenceAgainst: z.array(evidenceItemSchema).max(40),
  conflicts: z.array(z.string().max(500)).max(30),
  missingRequiredData: z.array(z.string().max(300)).max(40),
  recommendedNextTests: z.array(z.string().max(300)).max(30),
  biologicalJustification: z.string().max(8000).nullable(),
  warnings: z.array(z.string().max(500)).max(30).optional(),
  clinicianConfirmationRequired: z.literal(true),
});

export type FinalDiagnosis = z.infer<typeof finalDiagnosisSchema>;

export const ERROR_TYPES = [
  "invalid_test_interpretation",
  "mimic",
  "missing_prerequisite",
  "taxonomy_error",
  "scope_error",
  "other",
] as const;

export const correctionIngestSchema = z.object({
  case: clinicalCaseSchema,
  agentStatus: diagnosticStatusSchema,
  agentPulpal: pulpalDiagnosisSchema.nullable(),
  agentApical: apicalDiagnosisSchema.nullable(),
  adjudicatedReferenceDiagnosis: z.object({
    status: diagnosticStatusSchema,
    pulpal: pulpalDiagnosisSchema.nullable(),
    apical: apicalDiagnosisSchema.nullable(),
  }),
  correctionReasoning: z.string().min(10).max(8000),
  errorTypes: z.array(z.enum(ERROR_TYPES)).min(1).max(10),
  specialistIdentity: z.string().min(2).max(200),
  taxonomyVersion: z.literal(TAXONOMY_VERSION),
  approvalStatus: z.enum(["pending", "approved", "withdrawn"]).default("pending"),
  misunderstoodSummary: z.string().max(4000).optional(),
});

export type CorrectionIngest = z.infer<typeof correctionIngestSchema>;

export const defaultClinicalCase: ClinicalCase = {
  taxonomyVersion: TAXONOMY_VERSION,
  tooth: {
    universal: 30,
    fdi: "46",
    dentition: "permanent",
    apexMaturity: "mature",
  },
  treatmentHistory: "untreated",
  symptoms: {
    spontaneousPain: "unknown",
    nocturnalPain: "unknown",
    posturalPain: "unknown",
    referredPain: "unknown",
    thermalPainHistory: "unknown",
    heatResponse: "not_performed",
    coldRelievesPain: "unknown",
  },
  visual: {
    sinusTract: "absent",
    sinusTractTraced: "unknown",
    swelling: "absent",
    swellingSeverity: "none",
    fluctuance: "absent",
    pus: "absent",
    rapidOnsetSwelling: "absent",
    fever: "absent",
    lymphadenopathy: "absent",
    decay: "unknown",
    deepCariesOrExposure: "unknown",
    crownPresent: "unknown",
    crackSuspected: "unknown",
    traumaSigns: "absent",
  },
  clinical: {
    cold: "lingering",
    coldLingerSeconds: 30,
    coldComparedToControl: "exaggerated_vs_control",
    coldValidity: "valid",
    coldRepeated: "present",
    ept: "positive",
    eptComparedToControl: "similar_to_control",
    eptValidity: "valid",
    eptRepeated: "present",
    percussion: "moderate",
    palpation: "none",
    biting: "mild",
    toothSloothBiting: "not_performed",
    transillumination: "not_performed",
    fluorescentLight: "not_performed",
  },
  periodontal: {
    isolatedDeepPocket: "unknown",
    mobility: "unknown",
    occlusionTrauma: "unknown",
  },
  imaging: {
    periapicalRadiolucency: "absent",
    jShapedPeriapicalRadiolucency: "absent",
    widenedPeriodontalLigament: "unknown",
    laminaDuraLoss: "unknown",
    multiplePaViews: "unknown",
    internalResorption: "not_assessed",
    externalResorption: "not_assessed",
  },
  confounders: {
    recentAnesthesia: "absent",
    calcificationSuspected: "unknown",
    poorIsolation: "unknown",
    generalizedLowResponsiveness: "unknown",
    recentTrauma: "absent",
  },
  additionalNotes: "",
};
