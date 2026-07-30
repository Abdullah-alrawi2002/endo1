import {
  buildPatelCode,
  circumferenceFromAngle,
  computeCanalProximity,
  computePatelHeight,
} from "@/lib/ecr/geometry/patel";
import type {
  EcrMeasurementInput,
  PatelResult,
} from "@/lib/ecr/schemas";

/** Run deterministic Patel three-axis calculation from measurement inputs. */
export function calculatePatelFromMeasurements(
  input: EcrMeasurementInput,
): PatelResult {
  const height = computePatelHeight({
    mostApicalExtentMmFromCEJ: input.mostApicalExtentMmFromCEJ,
    rootLengthCejToApexMm: input.rootLengthCejToApexMm,
    localCrestDistanceMmFromCEJ: input.localCrestDistanceMmFromCEJ,
  });

  const circumference = circumferenceFromAngle(input.maximumCircumferenceDegrees);

  const canalProximity = computeCanalProximity({
    minimumSeparationMm: input.minimumLesionCanalSeparationMm,
    lowerUncertaintyBoundMm: input.separationUncertaintyLowerBoundMm ?? null,
    lesionCanalContactOrIntersection: input.lesionCanalContactOrIntersection,
    continuousDentineBarrierVisible: input.continuousDentineBarrierVisible,
  });

  const code = buildPatelCode(
    height.value,
    circumference.value,
    canalProximity.value,
  );

  const componentsComplete =
    height.status === "complete" &&
    circumference.status === "complete" &&
    canalProximity.status === "complete" &&
    code !== null;

  const manualReviewRequired =
    height.borderline ||
    circumference.borderline ||
    canalProximity.borderline ||
    !componentsComplete;

  return {
    status: componentsComplete ? "complete" : "incomplete",
    height: {
      status: height.status,
      value: height.value,
      mostApicalExtentMmFromCEJ: height.mostApicalExtentMmFromCEJ,
      relativeToLocalCrest: height.relativeToLocalCrest,
      rootThird: height.rootThird,
      borderline: height.borderline,
      plausibleValues: height.plausibleValues,
      notes: height.notes,
    },
    circumference: {
      status: circumference.status,
      value: circumference.value,
      maximumAngleDegrees: circumference.maximumAngleDegrees,
      borderline: circumference.borderline,
      plausibleValues: circumference.plausibleValues,
      notes: circumference.notes,
    },
    canalProximity: {
      status: canalProximity.status,
      value: canalProximity.value,
      minimumSeparationMm: canalProximity.minimumSeparationMm,
      lowerUncertaintyBoundMm: canalProximity.lowerUncertaintyBoundMm ?? null,
      canalWallStatus: canalProximity.canalWallStatus,
      borderline: canalProximity.borderline,
      plausibleValues: canalProximity.plausibleValues,
      notes: canalProximity.notes,
    },
    code,
    rootsAffected: input.rootsAffected,
    manualReviewRequired,
  };
}
