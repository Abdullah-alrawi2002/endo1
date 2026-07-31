import assert from "node:assert/strict";
import {
  buildPatelCode,
  circumferenceFromAngle,
  computeCanalProximity,
  computePatelHeight,
  unionAngularSectors,
} from "../lib/ecr/geometry/patel";
import { calculatePatelFromMeasurements } from "../lib/ecr/geometry";
import { rankManagementOptions } from "../lib/ecr/rules/treatment-options";
import { verifyEcrAnalysis } from "../lib/ecr/rules/verifier";
import { runEcrPipeline } from "../lib/ecr/pipeline";
import {
  ecrMeasurementInputSchema,
  type EcrMeasurementInput,
  type PlanningFeatures,
} from "../lib/ecr/schemas";

function baseInput(
  overrides: Partial<EcrMeasurementInput> = {},
): EcrMeasurementInput {
  return ecrMeasurementInputSchema.parse({
    toothLabel: "11",
    rootsAffected: ["single_root"],
    mostApicalExtentMmFromCEJ: 5,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: 2,
    maximumCircumferenceDegrees: 120,
    minimumLesionCanalSeparationMm: 0.5,
    separationUncertaintyLowerBoundMm: 0.4,
    lesionCanalContactOrIntersection: false,
    continuousDentineBarrierVisible: true,
    ecrDifferential: "appearance_consistent_with_ecr",
    maskReviewStatus: "clinician_reviewed",
    scan: {
      qualityStatus: "pass",
      oodStatus: "in_domain",
      nativeVoxelMm: [0.2, 0.2, 0.2],
      fovCompleteForTarget: true,
      targetStructuresVisible: {
        crownRootComplex: true,
        cejRegion: true,
        localAlveolarCrest: true,
        apex: true,
        canalBoundaryNearLesion: true,
        lesionMargins: true,
      },
      artifactWarnings: [],
      qualityGateFailures: [],
    },
    planning: {
      externalAccessProxy: "favorable",
      internalAccessProxy: "possible",
      structuralContinuity: "reduced",
      furcationInvolvement: "not_applicable",
      adjacentAnatomyWarnings: [],
      existingTreatmentFindings: [],
      fractureRiskProxy: "moderate",
      rootFormForReplantation: "compatible",
    },
    ...overrides,
  });
}

function planning(p: Partial<PlanningFeatures> = {}): PlanningFeatures {
  return {
    portalSurface: null,
    portalAreaMm2: null,
    portalSupracrestal: null,
    lesionVolumeMm3: null,
    lesionMaxDepthMm: null,
    lesionMaxWidthMm: null,
    externalAccessProxy: "favorable",
    internalAccessProxy: "possible",
    structuralContinuity: "reduced",
    furcationInvolvement: "not_applicable",
    boneCrestLossProxy: null,
    adjacentAnatomyWarnings: [],
    existingTreatmentFindings: [],
    fractureRiskProxy: "moderate",
    rootFormForReplantation: "compatible",
    ...p,
  };
}

// --- Circumference exact boundaries ---
assert.equal(circumferenceFromAngle(90).value, "A");
assert.equal(circumferenceFromAngle(90.5).borderline, true);
assert.equal(circumferenceFromAngle(91).value, "B");
assert.equal(circumferenceFromAngle(180).value, "B");
assert.equal(circumferenceFromAngle(181).value, "C");
assert.equal(circumferenceFromAngle(270).value, "C");
assert.equal(circumferenceFromAngle(271).value, "D");
assert.equal(circumferenceFromAngle(null).status, "indeterminate");

// Angular union must not silently sum disjoint levels as if one cross-section.
assert.ok(unionAngularSectors([{ startDeg: 0, endDeg: 90 }]) >= 89);
assert.ok(unionAngularSectors([{ startDeg: 0, endDeg: 90 }]) <= 91);
const wrap = unionAngularSectors([{ startDeg: 350, endDeg: 20 }]);
assert.ok(wrap >= 29 && wrap <= 31);

// --- Height crest / thirds ---
assert.equal(
  computePatelHeight({
    mostApicalExtentMmFromCEJ: 1,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: 2,
  }).value,
  1,
);
assert.equal(
  computePatelHeight({
    mostApicalExtentMmFromCEJ: 4,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: 2,
  }).value,
  2,
);
assert.equal(
  computePatelHeight({
    mostApicalExtentMmFromCEJ: 8,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: 2,
  }).value,
  3,
);
assert.equal(
  computePatelHeight({
    mostApicalExtentMmFromCEJ: 12,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: 2,
  }).value,
  4,
);
assert.equal(
  computePatelHeight({
    mostApicalExtentMmFromCEJ: 5,
    rootLengthCejToApexMm: 15,
    localCrestDistanceMmFromCEJ: null,
  }).status,
  "indeterminate",
);

// --- d/p topology ---
assert.equal(
  computeCanalProximity({
    minimumSeparationMm: 0,
    lesionCanalContactOrIntersection: true,
    continuousDentineBarrierVisible: false,
  }).value,
  "p",
);
assert.equal(
  computeCanalProximity({
    minimumSeparationMm: 0.5,
    lowerUncertaintyBoundMm: 0.4,
    lesionCanalContactOrIntersection: false,
    continuousDentineBarrierVisible: true,
  }).value,
  "d",
);
assert.equal(
  computeCanalProximity({
    minimumSeparationMm: null,
    lesionCanalContactOrIntersection: null,
    continuousDentineBarrierVisible: null,
  }).status,
  "indeterminate",
);

assert.equal(buildPatelCode(2, "B", "p"), "2Bp");
assert.equal(buildPatelCode(2, "B", null), null);

// Full measurement → code
const patel = calculatePatelFromMeasurements(
  baseInput({
    mostApicalExtentMmFromCEJ: 4.8,
    maximumCircumferenceDegrees: 142,
    lesionCanalContactOrIntersection: true,
    continuousDentineBarrierVisible: false,
    minimumLesionCanalSeparationMm: 0,
  }),
);
assert.equal(patel.code, "2Bp");

// Differential out-of-scope abstains options
const caries = runEcrPipeline(
  baseInput({ ecrDifferential: "caries_more_likely" }),
);
assert.equal(caries.patel.code, null);
assert.equal(caries.managementSupport.options.length, 0);
assert.equal(caries.managementSupport.definitiveTreatmentPlanAvailable, false);

// Quality fail abstains
const failQ = runEcrPipeline(
  baseInput({
    scan: {
      ...baseInput().scan,
      qualityStatus: "fail",
    },
  }),
);
assert.equal(failQ.patel.code, null);

// Verifier rejects code mismatch
const bad = verifyEcrAnalysis({
  scan: baseInput().scan,
  ecrStatus: "appearance_consistent_with_ecr",
  maskReviewStatus: "clinician_reviewed",
  patel: { ...patel, code: "1Ad" },
});
assert.equal(bad.ok, false);
assert.ok(bad.issues.some((i) => i.code === "code_mismatch"));

// Option engine: 2Bd favorable external → external without automatic RCT favored
const opts2Bd = rankManagementOptions({
  patel: calculatePatelFromMeasurements(
    baseInput({
      lesionCanalContactOrIntersection: false,
      continuousDentineBarrierVisible: true,
      minimumLesionCanalSeparationMm: 0.6,
      separationUncertaintyLowerBoundMm: 0.5,
      maximumCircumferenceDegrees: 120,
    }),
  ),
  planning: planning({ externalAccessProxy: "favorable" }),
});
assert.ok(
  opts2Bd.options.some(
    (o: { option: string; status: string }) =>
      o.option === "external_repair_without_automatic_RCT" &&
      o.status === "favored_on_imaging",
  ),
);
assert.equal(opts2Bd.definitiveTreatmentPlanAvailable, false);

// Overlapping options remain available for 2Bp
const opts2Bp = rankManagementOptions({
  patel,
  planning: planning({
    externalAccessProxy: "favorable",
    internalAccessProxy: "possible",
  }),
});
assert.ok(opts2Bp.options.length >= 1);
assert.ok(
  opts2Bp.options.every(
    (o: { requiresClinicalConfirmation: string[] }) =>
      o.requiresClinicalConfirmation.length > 0,
  ),
);

// Safety: pipeline never marks definitive plan
const full = runEcrPipeline(
  baseInput({
    lesionCanalContactOrIntersection: true,
    continuousDentineBarrierVisible: false,
  }),
);
assert.equal(full.managementSupport.definitiveTreatmentPlanAvailable, false);
assert.match(full.outputLabel, /not a definitive treatment plan/i);
assert.equal(full.evidenceScope, "CBCT_ONLY");
assert.equal(full.review.specialistConfirmationRequired, true);

console.log("ecr-geometry-and-rules: all assertions passed");

// --- Mask review must abstain (not warn) ---
{
  const unreviewed = verifyEcrAnalysis({
    scan: baseInput().scan,
    ecrStatus: "appearance_consistent_with_ecr",
    maskReviewStatus: "proposed",
    patel,
  });
  assert.equal(unreviewed.mayRankOptions, false);
  assert.ok(unreviewed.issues.some((i) => i.code === "masks_unreviewed" && i.severity === "abstain"));
}

// --- Evidence package + Patel agent network (offline) ---
process.env.ENDO_ECR_SKIP_LLM = "1";

const { buildEvidencePackage, heightEvidenceView, recomputeEvidenceHash } =
  await import("../lib/ecr/evidence/build-evidence-package");
const { runPatelNetwork } = await import("../lib/ecr/agents/run-patel-network");
const { runTreatmentAgentNetwork } = await import(
  "../lib/ecr/agents/run-treatment-network"
);
const { buildIntegratedPlan } = await import(
  "../lib/integration/build-integrated-plan"
);
const { isEligibleForPromptRetrieval } = await import(
  "../lib/correction-rag/store"
);
const { sha256EvidenceHash } = await import("../lib/ecr/evidence/evidence-hash");

const m2Bp = baseInput({
  mostApicalExtentMmFromCEJ: 4.8,
  maximumCircumferenceDegrees: 142,
  lesionCanalContactOrIntersection: true,
  continuousDentineBarrierVisible: false,
  minimumLesionCanalSeparationMm: 0,
  separationUncertaintyLowerBoundMm: 0,
});
const pkg = buildEvidencePackage({ measurements: m2Bp });
assert.match(pkg.caseEvidenceHash, /^sha256:[a-f0-9]{64}$/);
assert.equal(recomputeEvidenceHash(pkg), pkg.caseEvidenceHash);
assert.ok(!("code" in pkg.measurements));
assert.ok(JSON.stringify(pkg).indexOf("2Bp") === -1);

const heightView = heightEvidenceView(pkg);
assert.ok(!("patelCode" in heightView));
assert.ok(!JSON.stringify(heightView).match(/"code"\s*:/));
assert.ok(heightView.prohibited.includes("peer_component_conclusion"));
assert.ok(heightView.prohibited.includes("deterministic_patel_code"));

// Exact boundaries via network
for (const [angle, letter] of [
  [90, "A"],
  [180, "B"],
  [270, "C"],
] as const) {
  const net = await runPatelNetwork({
    evidence: buildEvidencePackage({
      measurements: baseInput({
        maximumCircumferenceDegrees: angle,
        lesionCanalContactOrIntersection: false,
        continuousDentineBarrierVisible: true,
        minimumLesionCanalSeparationMm: 0.5,
        separationUncertaintyLowerBoundMm: 0.4,
      }),
    }),
    measurements: baseInput({
      maximumCircumferenceDegrees: angle,
      lesionCanalContactOrIntersection: false,
      continuousDentineBarrierVisible: true,
      minimumLesionCanalSeparationMm: 0.5,
      separationUncertaintyLowerBoundMm: 0.4,
    }),
  });
  assert.equal(net.circumference.value, letter, `angle ${angle}`);
  assert.equal(net.deterministicCompatibility, "passed");
}

// Uncertainty interval crossing 180°
{
  const measurements = baseInput({
    maximumCircumferenceDegrees: 179,
    lesionCanalContactOrIntersection: false,
    continuousDentineBarrierVisible: true,
    minimumLesionCanalSeparationMm: 0.5,
    separationUncertaintyLowerBoundMm: 0.4,
  });
  const evidence = buildEvidencePackage({
    measurements,
    circumferenceUncertaintyDegrees: 3,
  });
  const net = await runPatelNetwork({ evidence, measurements });
  assert.ok(
    net.circumference.borderline ||
      net.componentAgents.some(
        (a) =>
          a.agentRole === "patel_circumference" && a.uncertainties.length > 0,
      ),
  );
}

// Height above/below crest
{
  const below = await runPatelNetwork({
    evidence: buildEvidencePackage({
      measurements: baseInput({
        mostApicalExtentMmFromCEJ: 1.5,
        localCrestDistanceMmFromCEJ: 2,
      }),
    }),
    measurements: baseInput({
      mostApicalExtentMmFromCEJ: 1.5,
      localCrestDistanceMmFromCEJ: 2,
    }),
  });
  assert.equal(below.height.value, 1);

  const above = await runPatelNetwork({
    evidence: buildEvidencePackage({
      measurements: baseInput({
        mostApicalExtentMmFromCEJ: 3,
        localCrestDistanceMmFromCEJ: 2,
        rootLengthCejToApexMm: 15,
      }),
    }),
    measurements: baseInput({
      mostApicalExtentMmFromCEJ: 3,
      localCrestDistanceMmFromCEJ: 2,
      rootLengthCejToApexMm: 15,
    }),
  });
  assert.equal(above.height.value, 2);
}

// d/p contact and barrier
{
  const contact = await runPatelNetwork({
    evidence: buildEvidencePackage({ measurements: m2Bp }),
    measurements: m2Bp,
  });
  assert.equal(contact.canalProximity.value, "p");
  assert.equal(contact.code, "2Bp");
  assert.ok(!/necrosis/i.test(JSON.stringify(contact.componentAgents)));

  const barrierM = baseInput({
    lesionCanalContactOrIntersection: false,
    continuousDentineBarrierVisible: true,
    minimumLesionCanalSeparationMm: 0.6,
    separationUncertaintyLowerBoundMm: 0.5,
    maximumCircumferenceDegrees: 120,
  });
  const barrier = await runPatelNetwork({
    evidence: buildEvidencePackage({ measurements: barrierM }),
    measurements: barrierM,
  });
  assert.equal(barrier.canalProximity.value, "d");
}

// Missing quality / non-ECR
{
  const failM = baseInput({
    scan: { ...baseInput().scan, qualityStatus: "fail" },
  });
  const failNet = await runPatelNetwork({
    evidence: buildEvidencePackage({ measurements: failM }),
    measurements: failM,
  });
  assert.ok(
    failNet.requiresSpecialistReview ||
      failNet.classificationStatus === "abstained" ||
      failNet.classificationStatus === "needs_specialist_review",
  );

  const cariesM = baseInput({ ecrDifferential: "caries_more_likely" });
  const cariesNet = await runPatelNetwork({
    evidence: buildEvidencePackage({ measurements: cariesM }),
    measurements: cariesM,
  });
  assert.ok(
    cariesNet.classificationStatus === "abstained" ||
      cariesNet.requiresSpecialistReview ||
      cariesNet.classificationStatus === "needs_specialist_review",
  );
}

// Agent citation of nonexistent evidence
{
  const { verifyPatelNetwork } = await import(
    "../lib/ecr/agents/verify-patel-network"
  );
  const evidence = buildEvidencePackage({ measurements: m2Bp });
  const forged = {
    agentRole: "patel_height" as const,
    agentContractVersion: "3.0" as const,
    promptVersion: "test",
    modelId: "test",
    caseEvidenceHash: evidence.caseEvidenceHash,
    status: "complete" as const,
    conclusion: 2 as const,
    alternatives: [],
    evidenceIds: ["eh_FORGED_NOT_REAL"],
    conciseRationale: "forged",
    uncertainties: [],
    abstain: false,
  };
  const v = verifyPatelNetwork({
    evidence,
    measurements: m2Bp,
    components: [forged],
    adjudicator: null,
    critic: null,
    reevaluationUsed: false,
  });
  assert.ok(v.issues.some((i) => /nonexistent evidence/i.test(i.message)));
}

// Stale evidence after measurement edits
{
  const evidence = buildEvidencePackage({ measurements: m2Bp });
  const tampered = {
    ...evidence,
    measurements: {
      ...evidence.measurements,
      maximumCircumferenceDegrees: 300,
    },
  };
  assert.notEqual(recomputeEvidenceHash(tampered), evidence.caseEvidenceHash);
}

// Treatment network: overlapping alternatives for 2Bp; never definitive
{
  const evidence = buildEvidencePackage({ measurements: m2Bp });
  const network = await runPatelNetwork({ evidence, measurements: m2Bp });
  const plan = runTreatmentAgentNetwork({
    evidence,
    network,
    measurements: m2Bp,
  });
  assert.equal(plan.definitiveTreatmentPlanAvailable, false);
  assert.equal(plan.planStatus, "provisional_cbct_based");
  assert.ok(plan.requiredClinicalConfirmations.length > 0);
  assert.ok(
    plan.candidates.length + plan.alternativeStrategies.length >= 1,
  );
  // No unconditional extraction/RCT from CBCT alone
  assert.ok(
    plan.candidates.every((c) => c.requiredClinicalConfirmations.length > 0),
  );
}

// Integrated plan walls
{
  const integrated = await buildIntegratedPlan({ measurements: m2Bp });
  assert.equal(integrated.unified.walls?.cbctCannotEstablishVitality, true);
  assert.equal(integrated.unified.walls?.clinicalCannotChangePatelCode, true);
  assert.equal(
    integrated.unified.treatmentFeasibility?.vitalityEstablishedByCbct,
    false,
  );
  assert.equal(
    integrated.unified.treatmentFeasibility?.patelPUsedAsPulpNecrosis,
    false,
  );
  assert.equal(
    integrated.unified.treatmentFeasibility?.clinicalMayAlterPatelCode,
    false,
  );
  assert.equal(integrated.provisionalPlan.planStatus, "provisional_cbct_based");
}

// Correction RAG governance
assert.equal(
  isEligibleForPromptRetrieval({
    id: "x",
    createdAt: 0,
    caseCanonical: "",
    agentPulpal: "",
    agentApical: "",
    correctedPulpal: "",
    correctedApical: "",
    reasoning: "",
    misunderstood: null,
    embedDocument: "",
    embeddingDim: 0,
    approvalStatus: "evaluation_only",
    taxonomyVersion: "AAE_2009",
    reviewerCount: 2,
    containsPHI: false,
  }),
  false,
);
assert.equal(
  isEligibleForPromptRetrieval({
    id: "x",
    createdAt: 0,
    caseCanonical: "",
    agentPulpal: "",
    agentApical: "",
    correctedPulpal: "",
    correctedApical: "",
    reasoning: "",
    misunderstood: null,
    embedDocument: "",
    embeddingDim: 0,
    approvalStatus: "approved",
    taxonomyVersion: "AAE_2009",
    reviewerCount: 2,
    containsPHI: false,
  }),
  true,
);
assert.equal(
  isEligibleForPromptRetrieval({
    id: "x",
    createdAt: 0,
    caseCanonical: "",
    agentPulpal: "",
    agentApical: "",
    correctedPulpal: "",
    correctedApical: "",
    reasoning: "",
    misunderstood: null,
    embedDocument: "",
    embeddingDim: 0,
    approvalStatus: "approved",
    taxonomyVersion: "AAE_2009",
    reviewerCount: 1,
    containsPHI: false,
  }),
  false,
);

// Hash stability
{
  const a = sha256EvidenceHash({ x: 1, y: [2, 3] });
  const b = sha256EvidenceHash({ y: [2, 3], x: 1 });
  assert.equal(a, b);
}

console.log("ecr-agent-network-and-governance: all assertions passed");
