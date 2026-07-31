import {
  provisionalPlanSchema,
  treatmentCandidateSchema,
  type EcrEvidencePackage,
  type PatelNetworkResult,
  type ProvisionalPlan,
  type TreatmentCandidate,
} from "@/lib/ecr/agents/schemas";
import { rankManagementOptions } from "@/lib/ecr/rules/treatment-options";
import { CANNOT_DETERMINE_FROM_CBCT } from "@/lib/ecr/rules/treatment-options";
import type { FinalDiagnosis } from "@/lib/schemas/clinical-case";
import type { EcrMeasurementInput, PatelResult, PlanningFeatures } from "@/lib/ecr/schemas";

function toPatelResult(network: PatelNetworkResult): PatelResult {
  return {
    status:
      network.classificationStatus === "complete"
        ? "complete"
        : network.classificationStatus === "abstained" ||
            network.classificationStatus === "rejected"
          ? "abstained"
          : "incomplete",
    height: {
      status: network.height.value != null ? "complete" : "indeterminate",
      value: network.height.value,
      mostApicalExtentMmFromCEJ: null,
      relativeToLocalCrest: null,
      rootThird: null,
      borderline: network.height.borderline,
      plausibleValues: network.height.value
        ? [network.height.value, ...network.height.alternatives.filter((n): n is 1 | 2 | 3 | 4 => [1, 2, 3, 4].includes(n as 1 | 2 | 3 | 4))].slice(0, 2) as Array<1 | 2 | 3 | 4>
        : [],
      notes: [],
    },
    circumference: {
      status: network.circumference.value != null ? "complete" : "indeterminate",
      value: network.circumference.value,
      maximumAngleDegrees: network.circumference.maximumAngleDegrees,
      borderline: network.circumference.borderline,
      plausibleValues: network.circumference.value
        ? ([network.circumference.value, ...network.circumference.alternatives] as Array<
            "A" | "B" | "C" | "D"
          >).slice(0, 2)
        : [],
      notes: [],
    },
    canalProximity: {
      status: network.canalProximity.value != null ? "complete" : "indeterminate",
      value: network.canalProximity.value,
      minimumSeparationMm: network.canalProximity.minimumSeparationMm,
      lowerUncertaintyBoundMm: null,
      canalWallStatus: null,
      borderline: network.canalProximity.borderline,
      plausibleValues: network.canalProximity.value
        ? ([network.canalProximity.value, ...network.canalProximity.alternatives] as Array<
            "d" | "p"
          >).slice(0, 2)
        : [],
      notes: [],
    },
    code: network.code,
    rootsAffected: [],
    manualReviewRequired: network.requiresSpecialistReview,
  };
}

function planningFromEvidence(pkg: EcrEvidencePackage): PlanningFeatures {
  return {
    portalSurface: pkg.planningFeatures.portalSurface,
    portalAreaMm2: pkg.planningFeatures.portalAreaMm2,
    portalSupracrestal: null,
    lesionVolumeMm3: null,
    lesionMaxDepthMm: null,
    lesionMaxWidthMm: null,
    externalAccessProxy: pkg.planningFeatures.externalAccessProxy,
    internalAccessProxy: pkg.planningFeatures.internalAccessProxy,
    structuralContinuity: pkg.planningFeatures.structuralContinuity,
    furcationInvolvement: "not_applicable",
    boneCrestLossProxy: null,
    adjacentAnatomyWarnings: pkg.planningFeatures.adjacentAnatomyWarnings,
    existingTreatmentFindings: [],
    fractureRiskProxy: pkg.planningFeatures.fractureRiskProxy as PlanningFeatures["fractureRiskProxy"],
    rootFormForReplantation:
      pkg.planningFeatures.rootFormForReplantation as PlanningFeatures["rootFormForReplantation"],
  };
}

function optionToCandidate(
  opt: ReturnType<typeof rankManagementOptions>["options"][number],
  evidenceIds: string[],
): TreatmentCandidate {
  return treatmentCandidateSchema.parse({
    strategy: opt.option,
    status: opt.status,
    treatmentObjective: `Conditional imaging-supported consideration of ${opt.option.replace(/_/g, " ")}`,
    proceduralSequence: [
      "Confirm clinical diagnosis and restorability",
      "Confirm isolation and access",
      `If activated: proceed with ${opt.option.replace(/_/g, " ")} pathway under specialist judgment`,
    ],
    supportingEvidenceIds: evidenceIds.slice(0, 10),
    limitingEvidenceIds: [],
    requiredClinicalConfirmations: opt.requiresClinicalConfirmation,
    activateIf: opt.supportingCbctFeatures,
    rejectIf: opt.limitingCbctFeatures,
    stopConditions: [
      "Unresolved clinical vitality / apical diagnosis when RCT pathway contemplated",
      "Inability to isolate or restore",
      "Patient declines or medical contraindications",
    ],
    alternatives: [],
    followUp: [
      "Serial clinical review",
      "Repeat imaging if activity suspected",
    ],
    sourceRuleIds: opt.supportingRuleIds,
    evidenceVersion: opt.evidenceVersion,
  });
}

/** External Repair Agent — proposes external pathways only. */
export function runExternalRepairAgent(
  candidates: TreatmentCandidate[],
): TreatmentCandidate[] {
  return candidates.filter((c) =>
    c.strategy.startsWith("external_repair"),
  );
}

/** Internal Repair / RCT Agent. */
export function runInternalRepairAgent(
  candidates: TreatmentCandidate[],
): TreatmentCandidate[] {
  return candidates.filter((c) => c.strategy === "internal_repair_with_RCT");
}

/** Preservation / Monitoring Agent. */
export function runPreservationAgent(
  candidates: TreatmentCandidate[],
): TreatmentCandidate[] {
  return candidates.filter((c) => c.strategy === "periodic_review");
}

/** Surgical Alternatives Agent. */
export function runSurgicalAlternativesAgent(
  candidates: TreatmentCandidate[],
): TreatmentCandidate[] {
  return candidates.filter((c) =>
    [
      "intentional_replantation_discussion",
      "decoronation_or_removal_replacement_discussion",
      "extraction_discussion",
    ].includes(c.strategy),
  );
}

export function runTreatmentCritic(input: {
  candidates: TreatmentCandidate[];
  network: PatelNetworkResult;
  clinical?: FinalDiagnosis | null;
}): { status: "passed" | "revision_requested" | "rejected"; warnings: string[] } {
  const warnings: string[] = [];

  // Never unconditional RCT / extraction / monitoring from CBCT alone
  for (const c of input.candidates) {
    if (
      c.requiredClinicalConfirmations.length === 0 ||
      !c.requiredClinicalConfirmations.some((x) => /pulp|clinical|restor/i.test(x))
    ) {
      warnings.push(`${c.strategy} missing required clinical confirmations`);
    }
    if (/definitive|must extract|mandatory RCT/i.test(c.treatmentObjective)) {
      return {
        status: "rejected",
        warnings: [`${c.strategy} used definitive language from imaging`],
      };
    }
  }

  // Patel p must not imply pulp necrosis / force RCT without clinical
  if (input.network.canalProximity.value === "p" && !input.clinical) {
    const rctForced = input.candidates.filter(
      (c) =>
        c.strategy.includes("RCT") &&
        c.status === "favored_on_imaging" &&
        c.activateIf.every((a) => !/clinical/i.test(a)),
    );
    if (rctForced.length && rctForced.every((c) => c.rejectIf.length === 0)) {
      warnings.push(
        "Patel p present without clinical evidence — RCT pathways remain conditional only",
      );
    }
  }

  // Preserve overlapping alternatives (e.g. 2Bp external+RCT and internal+RCT)
  const strategies = new Set(input.candidates.map((c) => c.strategy));
  if (
    input.network.code?.endsWith("p") &&
    strategies.has("external_repair_with_RCT_consideration") &&
    !strategies.has("internal_repair_with_RCT")
  ) {
    // Not always required — only warn when ESE table typically overlaps
    const code = input.network.code;
    if (code === "2Bp" || code === "2Cp" || code === "3Cp") {
      warnings.push(
        `${code}: consider whether internal_repair_with_RCT should remain as overlapping alternative`,
      );
    }
  }

  if (warnings.some((w) => /definitive/i.test(w))) {
    return { status: "rejected", warnings };
  }
  return {
    status: warnings.length ? "revision_requested" : "passed",
    warnings,
  };
}

export function runPlanSynthesizer(input: {
  allCandidates: TreatmentCandidate[];
  external: TreatmentCandidate[];
  internal: TreatmentCandidate[];
  preservation: TreatmentCandidate[];
  surgical: TreatmentCandidate[];
  evidenceHash: string;
  planStatus: ProvisionalPlan["planStatus"];
  criticStatus: ProvisionalPlan["criticStatus"];
  warnings: string[];
}): ProvisionalPlan {
  // Preserve overlapping alternatives — do not collapse to one code→one treatment
  const merged = new Map<string, TreatmentCandidate>();
  for (const c of [
    ...input.external,
    ...input.internal,
    ...input.preservation,
    ...input.surgical,
    ...input.allCandidates,
  ]) {
    const prev = merged.get(c.strategy);
    if (!prev || (c.status === "favored_on_imaging" && prev.status !== "favored_on_imaging")) {
      merged.set(c.strategy, c);
    }
  }
  const candidates = [...merged.values()];
  const favored = candidates.filter((c) => c.status === "favored_on_imaging");
  const primary = favored[0] ?? candidates[0] ?? null;
  const alternatives = candidates.filter((c) => c.strategy !== primary?.strategy);

  return provisionalPlanSchema.parse({
    planStatus: input.planStatus,
    primaryStrategy: primary?.strategy ?? null,
    candidates,
    alternativeStrategies: alternatives,
    proceduralSequence: primary?.proceduralSequence ?? [],
    requiredClinicalConfirmations: [
      ...new Set(candidates.flatMap((c) => c.requiredClinicalConfirmations)),
    ],
    decisionCheckpoints: [
      "Confirm pulpal/apical diagnosis clinically",
      "Confirm access and restorability",
      "Re-evaluate if clinical findings conflict with imaging priors",
    ],
    stopConditions: [
      ...new Set(candidates.flatMap((c) => c.stopConditions)),
    ],
    followUp: [...new Set(candidates.flatMap((c) => c.followUp))],
    sourceRuleIds: [...new Set(candidates.flatMap((c) => c.sourceRuleIds))],
    unsupportedVariables: [...CANNOT_DETERMINE_FROM_CBCT],
    definitiveTreatmentPlanAvailable: false,
    caseEvidenceHash: input.evidenceHash,
    criticStatus: input.criticStatus,
    safetyStatus: "not_run",
    warnings: input.warnings,
  });
}

export function runFinalSafetyAgent(plan: ProvisionalPlan): ProvisionalPlan {
  const warnings = [...plan.warnings];
  let safetyStatus: ProvisionalPlan["safetyStatus"] = "passed";

  if (plan.definitiveTreatmentPlanAvailable !== false) {
    safetyStatus = "rejected";
    warnings.push("Definitive treatment plan from CBCT alone is forbidden");
  }

  const unconditional =
    plan.requiredClinicalConfirmations.length === 0 &&
    plan.candidates.some((c) =>
      ["extraction_discussion", "internal_repair_with_RCT", "periodic_review"].includes(
        c.strategy,
      ),
    );
  if (unconditional && plan.planStatus === "provisional_cbct_based") {
    // Still allow candidates but force clinical confirmations list
    warnings.push("CBCT-only plan remains provisional — no unconditional RCT/extraction/monitoring");
  }

  if (/necrosis/i.test(JSON.stringify(plan.candidates)) && plan.planStatus === "provisional_cbct_based") {
    warnings.push("Pulpal necrosis language must not be derived from CBCT/Patel p alone");
  }

  const planStatus =
    safetyStatus === "rejected" ? "needs_specialist_review" : plan.planStatus;

  return provisionalPlanSchema.parse({
    ...plan,
    planStatus,
    safetyStatus,
    warnings,
    definitiveTreatmentPlanAvailable: false,
  });
}

/**
 * Full treatment-agent network after Patel verification.
 * Uses ESE multilabel priors as evidence-backed candidates — not a 1:1 lookup.
 */
export function runTreatmentAgentNetwork(input: {
  evidence: EcrEvidencePackage;
  network: PatelNetworkResult;
  measurements: EcrMeasurementInput;
  clinical?: FinalDiagnosis | null;
}): ProvisionalPlan {
  if (
    input.network.requiresSpecialistReview ||
    input.network.classificationStatus === "needs_specialist_review" ||
    !input.network.code
  ) {
    return provisionalPlanSchema.parse({
      planStatus: "needs_specialist_review",
      primaryStrategy: null,
      candidates: [],
      alternativeStrategies: [],
      proceduralSequence: [],
      requiredClinicalConfirmations: [
        "specialist review of entire CBCT volume and clinical findings",
      ],
      decisionCheckpoints: [],
      stopConditions: [],
      followUp: [],
      sourceRuleIds: [],
      unsupportedVariables: [...CANNOT_DETERMINE_FROM_CBCT],
      definitiveTreatmentPlanAvailable: false,
      caseEvidenceHash: input.evidence.caseEvidenceHash,
      criticStatus: "not_run",
      safetyStatus: "passed",
      warnings: input.network.warnings,
    });
  }

  const patel = toPatelResult(input.network);
  const planning = planningFromEvidence(input.evidence);
  const ranked = rankManagementOptions({ patel, planning });
  const allCandidates = ranked.options.map((o) =>
    optionToCandidate(o, input.evidence.evidenceIds),
  );

  // Clinical diagnosis may influence RCT branch activation notes — never Patel code.
  if (input.clinical?.pulpalDiagnosis && input.clinical.status === "diagnosable") {
    for (const c of allCandidates) {
      if (c.strategy.includes("RCT")) {
        c.activateIf = [
          ...c.activateIf,
          `clinical pulpal diagnosis: ${input.clinical.pulpalDiagnosis}`,
        ];
        if (input.clinical.pulpalDiagnosis === "Normal Pulp" || input.clinical.pulpalDiagnosis === "Reversible Pulpitis") {
          c.rejectIf = [
            ...c.rejectIf,
            "vital pulp clinical diagnosis may favor external repair without automatic RCT when imaging d/p and access allow",
          ];
        }
      }
    }
  }

  const external = runExternalRepairAgent(allCandidates);
  const internal = runInternalRepairAgent(allCandidates);
  const preservation = runPreservationAgent(allCandidates);
  const surgical = runSurgicalAlternativesAgent(allCandidates);

  const critic = runTreatmentCritic({
    candidates: allCandidates,
    network: input.network,
    clinical: input.clinical,
  });

  const planStatus: ProvisionalPlan["planStatus"] =
    input.clinical?.status === "diagnosable"
      ? "provisional_integrated"
      : "provisional_cbct_based";

  const synthesized = runPlanSynthesizer({
    allCandidates,
    external,
    internal,
    preservation,
    surgical,
    evidenceHash: input.evidence.caseEvidenceHash,
    planStatus: critic.status === "rejected" ? "needs_specialist_review" : planStatus,
    criticStatus:
      critic.status === "passed"
        ? "passed"
        : critic.status === "rejected"
          ? "rejected"
          : "revision_requested",
    warnings: critic.warnings,
  });

  return runFinalSafetyAgent(synthesized);
}
