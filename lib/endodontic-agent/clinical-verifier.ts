import {
  TAXONOMY_VERSION,
  type ClinicalCase,
  type FinalDiagnosis,
  type DiagnosticStatus,
} from "@/lib/schemas/clinical-case";

export type GateResult = {
  status: DiagnosticStatus;
  missingRequiredData: string[];
  conflicts: string[];
  recommendedNextTests: string[];
  warnings: string[];
  evidenceFor: FinalDiagnosis["evidenceFor"];
  evidenceAgainst: FinalDiagnosis["evidenceAgainst"];
};

function isSevere(value: ClinicalCase["clinical"]["percussion"]): boolean {
  return value === "moderate" || value === "severe";
}

function sensibilityReliable(c: ClinicalCase): boolean {
  const coldOk =
    c.clinical.coldValidity === "valid" &&
    c.clinical.cold !== "not_performed" &&
    c.clinical.cold !== "unable_to_test";
  const eptOk =
    c.clinical.eptValidity === "valid" &&
    (c.clinical.ept === "positive" || c.clinical.ept === "negative");
  return coldOk || eptOk;
}

/**
 * Stage 0 / deterministic prerequisite gate.
 * Decides whether the case is in scope and sufficiently evidenced before any
 * AAE enum is permitted. The LLM may propose; this gate decides permissibility.
 */
export function runClinicalGate(c: ClinicalCase): GateResult {
  const missing: string[] = [];
  const conflicts: string[] = [];
  const nextTests: string[] = [];
  const warnings: string[] = [];
  const evidenceFor: FinalDiagnosis["evidenceFor"] = [];
  const evidenceAgainst: FinalDiagnosis["evidenceAgainst"] = [];

  if (c.taxonomyVersion !== TAXONOMY_VERSION) {
    return {
      status: "out_of_scope",
      missingRequiredData: [],
      conflicts: [`Unsupported taxonomyVersion ${c.taxonomyVersion}`],
      recommendedNextTests: [],
      warnings: [],
      evidenceFor,
      evidenceAgainst,
    };
  }

  if (c.treatmentHistory !== "untreated") {
    return {
      status: "out_of_scope",
      missingRequiredData: [],
      conflicts: [],
      recommendedNextTests: [
        "Refer for evaluation of previously initiated/obturated/regenerative teeth under the full AAE 2009 categories Previously Initiated Therapy / Previously Treated.",
      ],
      warnings: [
        "This MVP diagnoses only untreated teeth. Previously treated or previously initiated cases are out of scope.",
      ],
      evidenceFor,
      evidenceAgainst: [
        {
          claim: `treatmentHistory=${c.treatmentHistory}`,
          source: "verifier",
        },
      ],
    };
  }

  if (c.tooth.dentition === "primary") {
    return {
      status: "out_of_scope",
      missingRequiredData: [],
      conflicts: [],
      recommendedNextTests: [],
      warnings: ["Primary dentition is out of scope for this MVP."],
      evidenceFor,
      evidenceAgainst: [{ claim: "Primary dentition", source: "verifier" }],
    };
  }

  // Minimum clinical sufficiency for a probable AAE diagnosis.
  if (c.clinical.cold === "not_performed" && c.clinical.ept === "not_performed") {
    missing.push("At least one pulp sensibility test (Cold or EPT)");
  }
  if (
    c.clinical.coldComparedToControl === "not_compared" &&
    c.clinical.eptComparedToControl === "not_compared"
  ) {
    missing.push("Comparison of sensibility tests to adjacent/contralateral control teeth");
    nextTests.push("Repeat Cold and/or EPT on control teeth and record comparative response");
  }
  if (c.clinical.percussion === "not_assessed" || c.clinical.percussion === "unknown") {
    missing.push("Percussion assessment");
  }
  if (c.clinical.palpation === "not_assessed" || c.clinical.palpation === "unknown") {
    missing.push("Palpation assessment");
  }
  if (
    c.imaging.periapicalRadiolucency === "not_assessed" ||
    c.imaging.periapicalRadiolucency === "unknown"
  ) {
    missing.push("Clinician radiographic PARL assessment (present/absent)");
    nextTests.push("Obtain/review a diagnostic periapical radiograph");
  }

  // Confounders that block confident necrosis / vital labels.
  const confoundersActive =
    c.confounders.recentTrauma === "present" ||
    c.confounders.recentAnesthesia === "present" ||
    c.confounders.calcificationSuspected === "present" ||
    c.confounders.poorIsolation === "present" ||
    c.confounders.generalizedLowResponsiveness === "present" ||
    c.tooth.apexMaturity === "open" ||
    c.visual.crownPresent === "present";

  if (
    c.clinical.cold === "negative" &&
    (c.clinical.ept === "negative" || c.clinical.ept === "not_performed")
  ) {
    if (c.clinical.coldComparedToControl === "not_compared") {
      conflicts.push(
        "Negative Cold without control-tooth comparison cannot alone establish pulp necrosis.",
      );
      nextTests.push("Compare Cold (and EPT if available) to a known vital control tooth");
    }
    if (c.clinical.coldValidity !== "valid" || c.clinical.coldRepeated !== "present") {
      missing.push("Repeated, valid Cold testing relative to controls before considering necrosis");
      nextTests.push("Repeat Cold testing with confirmed technique and isolation");
    }
    if (confoundersActive) {
      conflicts.push(
        "Negative sensibility tests coincide with confounders (trauma, open apex, crown, calcification, anesthesia, or poor isolation) that commonly cause false negatives.",
      );
      nextTests.push("Reassess sensibility after confounders are addressed or document supporting evidence of necrosis");
    }
  }

  // Abscess prerequisites (AAE terminology).
  if (c.visual.sinusTract === "present" && c.visual.sinusTractTraced !== "present") {
    warnings.push(
      "A sinus tract is reported but not traced; Chronic Apical Abscess requires drainage through a sinus tract (tracing recommended).",
    );
    nextTests.push("Trace the sinus tract radiographically with gutta-percha");
  }

  // Imaging alone never establishes endodontic origin.
  if (
    c.imaging.periapicalRadiolucency === "present" &&
    c.clinical.cold !== "negative" &&
    c.clinical.ept !== "negative"
  ) {
    conflicts.push(
      "Radiographic PARL with responsive pulp tests — differential includes non-endodontic radiolucency; do not assume pulpal origin from imaging alone.",
    );
  }

  if (c.ctSupport?.candidateLowAttenuationRegion && !c.ctSupport.clinicianReviewed) {
    warnings.push(
      "CT support shows a candidate low-attenuation region that has not been clinician-reviewed; it cannot establish endodontic origin.",
    );
  }
  if (c.ctSupport && !c.ctSupport.qualityGatePassed) {
    warnings.push("CT quality gate failed; ignore quantitative CT support features.");
  }

  // Evidence snippets for the UI table (deterministic).
  if (c.clinical.cold !== "not_performed") {
    evidenceFor.push({
      claim: `Cold=${c.clinical.cold}; validity=${c.clinical.coldValidity}; vs control=${c.clinical.coldComparedToControl}`,
      source: "sensibility",
    });
  }
  if (c.clinical.ept !== "not_performed") {
    evidenceFor.push({
      claim: `EPT=${c.clinical.ept}; validity=${c.clinical.eptValidity}; vs control=${c.clinical.eptComparedToControl}`,
      source: "sensibility",
    });
  }
  if (isSevere(c.clinical.percussion)) {
    evidenceFor.push({
      claim: `Percussion severity=${c.clinical.percussion}`,
      source: "mechanical",
    });
  }
  if (c.visual.swelling === "present") {
    evidenceFor.push({
      claim: `Swelling present (severity=${c.visual.swellingSeverity}, rapid onset=${c.visual.rapidOnsetSwelling}, pus=${c.visual.pus})`,
      source: "visual",
    });
  }
  if (c.visual.sinusTract === "present") {
    evidenceFor.push({
      claim: `Sinus tract present; traced=${c.visual.sinusTractTraced}`,
      source: "visual",
    });
  }
  if (c.imaging.periapicalRadiolucency === "present") {
    evidenceFor.push({
      claim: "Clinician-assessed periapical radiolucency present",
      source: "imaging",
    });
  }

  if (!sensibilityReliable(c)) {
    missing.push("At least one valid Cold or EPT result");
  }

  // Status precedence: out_of_scope already returned; then conflicting; then insufficient; else diagnosable.
  if (conflicts.length > 0 && missing.length > 0) {
    return {
      status: "conflicting_data",
      missingRequiredData: missing,
      conflicts,
      recommendedNextTests: nextTests,
      warnings,
      evidenceFor,
      evidenceAgainst,
    };
  }
  if (conflicts.length > 0) {
    return {
      status: "conflicting_data",
      missingRequiredData: missing,
      conflicts,
      recommendedNextTests: nextTests,
      warnings,
      evidenceFor,
      evidenceAgainst,
    };
  }
  if (missing.length > 0) {
    return {
      status: "insufficient_data",
      missingRequiredData: missing,
      conflicts,
      recommendedNextTests: nextTests,
      warnings,
      evidenceFor,
      evidenceAgainst,
    };
  }

  return {
    status: "diagnosable",
    missingRequiredData: [],
    conflicts: [],
    recommendedNextTests: nextTests,
    warnings,
    evidenceFor,
    evidenceAgainst,
  };
}

/**
 * Verifies an LLM proposal against hard clinical prerequisites.
 * Never lets CT force an apical disease label. Never forces necrosis from Cold−/EPT− alone.
 */
export function verifyDiagnosisProposal(
  c: ClinicalCase,
  gate: GateResult,
  proposal: {
    status?: DiagnosticStatus;
    pulpalDiagnosis: FinalDiagnosis["pulpalDiagnosis"];
    apicalDiagnosis: FinalDiagnosis["apicalDiagnosis"];
  },
): { ok: true } | { ok: false; reason: string } {
  if (gate.status !== "diagnosable") {
    if (proposal.pulpalDiagnosis !== null || proposal.apicalDiagnosis !== null) {
      return {
        ok: false,
        reason: `Gate status is ${gate.status}; pulpalDiagnosis and apicalDiagnosis must be null.`,
      };
    }
    return { ok: true };
  }

  if (!proposal.pulpalDiagnosis || !proposal.apicalDiagnosis) {
    return {
      ok: false,
      reason: "Diagnosable cases require both pulpal and apical AAE enums.",
    };
  }

  // Acute abscess prerequisites.
  if (proposal.apicalDiagnosis === "Acute Apical Abscess") {
    const hasRapid =
      c.visual.rapidOnsetSwelling === "present" ||
      c.visual.swelling === "present";
    const hasPusOrFluctuance =
      c.visual.pus === "present" || c.visual.fluctuance === "present";
    if (!hasRapid || (!hasPusOrFluctuance && c.visual.swellingSeverity === "none")) {
      return {
        ok: false,
        reason:
          "Acute Apical Abscess requires rapid-onset swelling with signs of purulence/fluctuance (AAE terminology). Insufficient visual findings.",
      };
    }
  }

  // Chronic abscess prerequisites.
  if (proposal.apicalDiagnosis === "Chronic Apical Abscess") {
    if (c.visual.sinusTract !== "present") {
      return {
        ok: false,
        reason:
          "Chronic Apical Abscess requires drainage through a sinus tract (AAE terminology).",
      };
    }
  }

  // Necrosis: cannot be forced by tests alone when confounders / no controls.
  if (proposal.pulpalDiagnosis === "Pulp Necrosis") {
    if (
      c.clinical.coldComparedToControl === "not_compared" &&
      c.clinical.eptComparedToControl === "not_compared"
    ) {
      return {
        ok: false,
        reason:
          "Pulp Necrosis is not permitted without control-tooth comparison of sensibility tests.",
      };
    }
    if (
      c.confounders.recentTrauma === "present" ||
      c.tooth.apexMaturity === "open" ||
      c.confounders.recentAnesthesia === "present"
    ) {
      if (c.clinical.coldRepeated !== "present") {
        return {
          ok: false,
          reason:
            "Pulp Necrosis with active false-negative confounders requires repeated valid testing and supporting findings; abstain or gather more data.",
        };
      }
    }
  }

  // CT must never be the sole basis for apical disease.
  if (
    proposal.apicalDiagnosis !== "Normal Apical Tissues" &&
    c.imaging.periapicalRadiolucency !== "present" &&
    c.clinical.percussion === "none" &&
    c.clinical.palpation === "none" &&
    c.visual.swelling !== "present" &&
    c.visual.sinusTract !== "present"
  ) {
    if (c.ctSupport?.candidateLowAttenuationRegion) {
      return {
        ok: false,
        reason:
          "CT candidate low-attenuation alone cannot establish apical disease or endodontic origin. Require clinical apical findings or clinician-confirmed radiographic disease.",
      };
    }
  }

  return { ok: true };
}

export function abstentionResult(gate: GateResult): FinalDiagnosis {
  return {
    taxonomyVersion: TAXONOMY_VERSION,
    status: gate.status,
    pulpalDiagnosis: null,
    apicalDiagnosis: null,
    finalDiagnosisLine: null,
    evidenceFor: gate.evidenceFor,
    evidenceAgainst: gate.evidenceAgainst,
    conflicts: gate.conflicts,
    missingRequiredData: gate.missingRequiredData,
    recommendedNextTests: gate.recommendedNextTests,
    biologicalJustification: null,
    warnings: [
      ...gate.warnings,
      "No AAE enum was assigned. Reassess, complete missing tests, or refer as appropriate.",
    ],
    clinicianConfirmationRequired: true,
  };
}
