import { z } from "zod";

/** Exact strings used in UI, API, and LLM output contract */
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

export type PulpalDiagnosis = (typeof PULPAL_DIAGNOSES)[number];
export type ApicalDiagnosis = (typeof APICAL_DIAGNOSES)[number];

export const pulpalDiagnosisSchema = z.enum(PULPAL_DIAGNOSES);
export const apicalDiagnosisSchema = z.enum(APICAL_DIAGNOSES);

export const visualFindingSchema = z.enum(["absent", "present", "unknown"]);

export const optionalClinicalTestSchema = z.enum([
  "not_performed",
  "positive",
  "negative",
]);

export const imagingFindingSchema = z.enum(["present", "absent"]);

export const visualExamSchema = z.object({
  sinusTract: visualFindingSchema,
  swelling: visualFindingSchema,
  decay: visualFindingSchema,
  traumaSigns: visualFindingSchema,
});

export const clinicalTestsSchema = z.object({
  cold: z.enum(["normal", "exaggerated_non_lingering", "lingering", "negative"]),
  coldLingerSeconds: z.number().positive().optional(),
  percussion: z.enum(["positive", "negative"]),
  palpation: z.enum(["positive", "negative"]),
  ept: optionalClinicalTestSchema,
  fluorescentLight: optionalClinicalTestSchema,
  toothSloothBiting: optionalClinicalTestSchema,
});

export const imagingFindingsSchema = z.object({
  periapicalRadiolucency: imagingFindingSchema,
  jShapedPeriapicalRadiolucency: imagingFindingSchema,
  widenedPeriodontalLigament: imagingFindingSchema,
  internalResorption: imagingFindingSchema,
  externalResorption: imagingFindingSchema,
});

export const clinicalCaseSchema = z.object({
  visual: visualExamSchema,
  clinical: clinicalTestsSchema,
  imaging: imagingFindingsSchema,
  additionalNotes: z.string().max(4000).optional(),
});

export type ClinicalCase = z.infer<typeof clinicalCaseSchema>;

export const finalDiagnosisSchema = z.object({
  pulpalDiagnosis: pulpalDiagnosisSchema,
  apicalDiagnosis: apicalDiagnosisSchema,
  finalDiagnosisLine: z.string(),
  biologicalJustification: z.string(),
  warnings: z.array(z.string()).optional(),
});

export type FinalDiagnosis = z.infer<typeof finalDiagnosisSchema>;

export const correctionIngestSchema = z.object({
  case: clinicalCaseSchema,
  agentPulpal: pulpalDiagnosisSchema,
  agentApical: apicalDiagnosisSchema,
  correctedPulpal: pulpalDiagnosisSchema,
  correctedApical: apicalDiagnosisSchema,
  correctionReasoning: z.string().min(10).max(8000),
  misunderstoodSummary: z.string().max(4000).optional(),
});

export type CorrectionIngest = z.infer<typeof correctionIngestSchema>;

export const defaultClinicalCase: ClinicalCase = {
  visual: {
    sinusTract: "absent",
    swelling: "absent",
    decay: "unknown",
    traumaSigns: "absent",
  },
  clinical: {
    cold: "lingering",
    coldLingerSeconds: 30,
    percussion: "positive",
    palpation: "negative",
    ept: "positive",
    fluorescentLight: "not_performed",
    toothSloothBiting: "not_performed",
  },
  imaging: {
    periapicalRadiolucency: "absent",
    jShapedPeriapicalRadiolucency: "absent",
    widenedPeriodontalLigament: "absent",
    internalResorption: "absent",
    externalResorption: "absent",
  },
  additionalNotes: "",
};
