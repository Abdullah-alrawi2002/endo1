/**
 * Evidence-versioned ECR management option ruleset (ESE root resorption 2023 corrected priors).
 * Stored as code — not prompt text. Multilabel conditional options only.
 */

export const RULESET_VERSION = "ESE_RR_2023_CORRECTED_v1" as const;

export const CANNOT_DETERMINE_FROM_CBCT = [
  "pulp sensibility / pulpal diagnosis",
  "apical diagnosis",
  "symptoms and lesion activity",
  "bleeding on probing / periodontal probing depths",
  "clinical accessibility and isolatability",
  "definitive restorability and esthetic acceptability",
  "periodontal prognosis",
  "patient age, growth status, medical risk, preferences, and tolerance",
  "whether RCT, monitoring, intentional replantation, decoronation, or extraction is definitively indicated",
] as const;

import type {
  ManagementOption,
  ManagementSupport,
  PatelResult,
  PlanningFeatures,
} from "@/lib/ecr/schemas";

type RankContext = {
  patel: PatelResult;
  planning: PlanningFeatures;
};

function baseConfirmation(extra: string[] = []): string[] {
  return [
    "pulpal and apical diagnosis from clinical testing",
    "periodontal probing and clinical access",
    "restorability and isolation",
    "patient factors and preferences",
    "specialist confirmation of entire CBCT volume",
    ...extra,
  ];
}

function option(
  partial: Omit<ManagementOption, "evidenceVersion" | "requiresClinicalConfirmation"> & {
    requiresClinicalConfirmation?: string[];
  },
): ManagementOption {
  return {
    ...partial,
    evidenceVersion: RULESET_VERSION,
    requiresClinicalConfirmation:
      partial.requiresClinicalConfirmation ?? baseConfirmation(),
  };
}

/**
 * Rank conditional imaging options. Never returns a single forced treatment.
 * Does not attach patient-level survival probabilities.
 */
export function rankManagementOptions(ctx: RankContext): ManagementSupport {
  const { patel, planning } = ctx;
  const options: ManagementOption[] = [];
  const h = patel.height.value;
  const c = patel.circumference.value;
  const p = patel.canalProximity.value;
  const ext = planning.externalAccessProxy;
  const intAcc = planning.internalAccessProxy;

  if (!h || !c || !p || !patel.code) {
    return {
      rulesetVersion: RULESET_VERSION,
      outputType: "conditional_imaging_option_set",
      options: [],
      definitiveTreatmentPlanAvailable: false,
      cannotDetermineFromCbct: [...CANNOT_DETERMINE_FROM_CBCT],
    };
  }

  // External repair without automatic RCT — localized d with favorable external access.
  if (
    [1, 2].includes(h) &&
    ["A", "B"].includes(c) &&
    p === "d" &&
    ext === "favorable"
  ) {
    options.push(
      option({
        option: "external_repair_without_automatic_RCT",
        status: "favored_on_imaging",
        supportingRuleIds: ["ESE-ECR-EXT-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "resolvable dentine barrier (d)",
          "favorable external access proxy",
        ],
        limitingCbctFeatures: [],
      }),
    );
  }

  // External repair with RCT consideration when p and external access feasible.
  if (p === "p" && (ext === "favorable" || ext === "possible") && [1, 2].includes(h)) {
    options.push(
      option({
        option: "external_repair_with_RCT_consideration",
        status: ext === "favorable" ? "favored_on_imaging" : "possible",
        supportingRuleIds: ["ESE-ECR-EXT-RCT-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "probable canal involvement (p)",
          `external access proxy=${ext}`,
        ],
        limitingCbctFeatures:
          ext === "possible" ? ["external access only possible, not clearly favorable"] : [],
        requiresClinicalConfirmation: baseConfirmation([
          "need for root canal treatment based on clinical pulp tests",
        ]),
      }),
    );
  }

  // Internal repair with RCT — canal-related, poor external, feasible internal.
  if (
    p === "p" &&
    (ext === "unfavorable" || ext === "not_assessable") &&
    (intAcc === "favorable" || intAcc === "possible")
  ) {
    options.push(
      option({
        option: "internal_repair_with_RCT",
        status: "possible",
        supportingRuleIds: ["ESE-ECR-INT-RCT-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "probable canal involvement (p)",
          `internal access proxy=${intAcc}`,
        ],
        limitingCbctFeatures: [`external access proxy=${ext}`],
        requiresClinicalConfirmation: baseConfirmation([
          "internal access feasibility after clinical entry",
        ]),
      }),
    );
  }

  // Also allow internal option for mid/extensive classes often listed in ESE examples.
  if (
    p === "p" &&
    ((h === 2 && ["B", "C", "D"].includes(c)) ||
      (h === 3 && ["C", "D"].includes(c))) &&
    intAcc !== "unfavorable"
  ) {
    if (!options.some((o) => o.option === "internal_repair_with_RCT")) {
      options.push(
        option({
          option: "internal_repair_with_RCT",
          status: "possible",
          supportingRuleIds: ["ESE-ECR-INT-RCT-02"],
          supportingCbctFeatures: [`Patel ${patel.code}`, "ESE overlapping example set"],
          limitingCbctFeatures: [],
        }),
      );
    }
  }

  // Intentional replantation discussion — height 3, limited circumference, poor conventional access.
  if (
    h === 3 &&
    ["A", "B"].includes(c) &&
    ext === "unfavorable" &&
    planning.rootFormForReplantation !== "imaging_prohibitive"
  ) {
    options.push(
      option({
        option: "intentional_replantation_discussion",
        status: "possible",
        supportingRuleIds: ["ESE-ECR-REPLANT-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "limited circumferential spread",
          "unfavorable conventional external access",
        ],
        limitingCbctFeatures:
          planning.rootFormForReplantation === "not_assessable"
            ? ["root form for replantation not fully assessable on imaging"]
            : [],
        requiresClinicalConfirmation: baseConfirmation([
          "surgical candidacy and root morphology clinical assessment",
        ]),
      }),
    );
  }

  // Extensive lesions — monitoring and removal/replacement as alternatives; do not choose between them.
  if ([3, 4].includes(h) && ["C", "D"].includes(c)) {
    options.push(
      option({
        option: "periodic_review",
        status: "possible",
        supportingRuleIds: ["ESE-ECR-REVIEW-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "extensive circumferential/vertical imaging involvement",
        ],
        limitingCbctFeatures: ["intervention may appear disproportionately destructive on imaging"],
        requiresClinicalConfirmation: baseConfirmation([
          "activity monitoring and serial clinical/imaging review",
        ]),
      }),
    );
    options.push(
      option({
        option: "decoronation_or_removal_replacement_discussion",
        status: "possible",
        supportingRuleIds: ["ESE-ECR-DECOR-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          `structural continuity=${planning.structuralContinuity}`,
        ],
        limitingCbctFeatures: [],
        requiresClinicalConfirmation: baseConfirmation([
          "prosthetic replacement planning",
          "growth/age considerations where relevant",
        ]),
      }),
    );
  }

  // Extraction discussion only as imaging-guarded conversation — never definitive.
  if (
    planning.structuralContinuity === "severely_compromised" ||
    (h === 4 && c === "D" && planning.externalAccessProxy === "unfavorable")
  ) {
    options.push(
      option({
        option: "extraction_discussion",
        status: "possible",
        supportingRuleIds: ["ESE-ECR-EXTX-01"],
        supportingCbctFeatures: [
          `Patel ${patel.code}`,
          "imaging suggests severe hard-tissue compromise or no feasible repair corridor",
        ],
        limitingCbctFeatures: [
          "No single Patel code defines extraction; clinical restorability required",
        ],
        requiresClinicalConfirmation: baseConfirmation([
          "clinical restorability and shared decision-making",
        ]),
      }),
    );
  }

  // If nothing matched but code is complete, still offer periodic review as not_assessable fallback discussion.
  if (!options.length) {
    options.push(
      option({
        option: "periodic_review",
        status: "not_assessable",
        supportingRuleIds: ["ESE-ECR-FALLBACK-01"],
        supportingCbctFeatures: [`Patel ${patel.code}`],
        limitingCbctFeatures: [
          "No strongly matched imaging option prior for this combination — specialist judgment required",
        ],
      }),
    );
  }

  // Safety: mark any accidental definitive language is prevented by schema + this flag.
  return {
    rulesetVersion: RULESET_VERSION,
    outputType: "conditional_imaging_option_set",
    options,
    definitiveTreatmentPlanAvailable: false,
    cannotDetermineFromCbct: [...CANNOT_DETERMINE_FROM_CBCT],
  };
}
