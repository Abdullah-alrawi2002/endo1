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
