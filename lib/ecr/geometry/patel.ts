/**
 * Deterministic Patel ECR geometry helpers.
 * All angles in degrees; lengths in mm along a tooth-aligned root centerline.
 */

export const CIRCUMFERENCE_BOUNDARIES = [90, 180, 270] as const;
/** Boundary tolerance (degrees) for borderline circumference codes. */
export const CIRCUMFERENCE_BOUNDARY_TOLERANCE_DEG = 2;
/** Spatial uncertainty floor (mm) for claiming a dentine barrier (d). */
export const CANAL_SEPARATION_UNCERTAINTY_FLOOR_MM = 0.15;

export type HeightValue = 1 | 2 | 3 | 4;
export type CircumferenceValue = "A" | "B" | "C" | "D";
export type CanalProximityValue = "d" | "p";

export type HeightComputationInput = {
  mostApicalExtentMmFromCEJ: number | null;
  rootLengthCejToApexMm: number | null;
  localCrestDistanceMmFromCEJ: number | null;
};

export type HeightComputation = {
  status: "complete" | "indeterminate" | "not_assessable";
  value: HeightValue | null;
  mostApicalExtentMmFromCEJ: number | null;
  relativeToLocalCrest:
    | "coronal_to_crest"
    | "at_crest"
    | "apical_to_crest"
    | "indeterminate"
    | null;
  rootThird:
    | "cej_or_coronal_to_crest"
    | "coronal"
    | "middle"
    | "apical"
    | "indeterminate"
    | null;
  borderline: boolean;
  plausibleValues: HeightValue[];
  notes: string[];
};

export function computePatelHeight(input: HeightComputationInput): HeightComputation {
  const notes: string[] = [];
  const {
    mostApicalExtentMmFromCEJ: apical,
    rootLengthCejToApexMm: rootLen,
    localCrestDistanceMmFromCEJ: crest,
  } = input;

  if (apical === null || rootLen === null || rootLen <= 0) {
    return {
      status: "indeterminate",
      value: null,
      mostApicalExtentMmFromCEJ: apical,
      relativeToLocalCrest: null,
      rootThird: null,
      borderline: false,
      plausibleValues: [],
      notes: ["CEJ-to-apex root length or most apical lesion extent unresolved."],
    };
  }

  let relativeToLocalCrest: HeightComputation["relativeToLocalCrest"] = "indeterminate";
  if (crest === null) {
    notes.push("Local alveolar crest unresolved — height may be incomplete.");
    return {
      status: "indeterminate",
      value: null,
      mostApicalExtentMmFromCEJ: apical,
      relativeToLocalCrest: "indeterminate",
      rootThird: "indeterminate",
      borderline: false,
      plausibleValues: [],
      notes,
    };
  }

  const crestEps = 0.2;
  if (apical < crest - crestEps) {
    relativeToLocalCrest = "coronal_to_crest";
  } else if (Math.abs(apical - crest) <= crestEps) {
    relativeToLocalCrest = "at_crest";
  } else {
    relativeToLocalCrest = "apical_to_crest";
  }

  // Height 1: at CEJ or coronal to local crest.
  if (relativeToLocalCrest === "coronal_to_crest" || relativeToLocalCrest === "at_crest") {
    const borderline = relativeToLocalCrest === "at_crest";
    return {
      status: "complete",
      value: 1,
      mostApicalExtentMmFromCEJ: apical,
      relativeToLocalCrest,
      rootThird: "cej_or_coronal_to_crest",
      borderline,
      plausibleValues: borderline ? [1, 2] : [1],
      notes: borderline
        ? ["Lesion at crest boundary — manual review of height 1 vs 2 recommended."]
        : notes,
    };
  }

  // Apical to crest: map into root thirds of the CEJ–apex length.
  const third = rootLen / 3;
  const coronalEnd = third;
  const middleEnd = 2 * third;
  const boundaryTol = Math.max(0.25, rootLen * 0.02);

  let value: HeightValue;
  let rootThird: HeightComputation["rootThird"];
  let borderline = false;
  const plausible: HeightValue[] = [];

  if (apical <= coronalEnd + boundaryTol) {
    value = 2;
    rootThird = "coronal";
    if (Math.abs(apical - coronalEnd) <= boundaryTol) {
      borderline = true;
      plausible.push(2, 3);
      notes.push("Near coronal/middle third boundary.");
    } else {
      plausible.push(2);
    }
  } else if (apical <= middleEnd + boundaryTol) {
    value = 3;
    rootThird = "middle";
    if (Math.abs(apical - middleEnd) <= boundaryTol) {
      borderline = true;
      plausible.push(3, 4);
      notes.push("Near middle/apical third boundary.");
    } else if (Math.abs(apical - coronalEnd) <= boundaryTol) {
      borderline = true;
      plausible.push(2, 3);
      notes.push("Near coronal/middle third boundary.");
    } else {
      plausible.push(3);
    }
  } else {
    value = 4;
    rootThird = "apical";
    if (Math.abs(apical - middleEnd) <= boundaryTol) {
      borderline = true;
      plausible.push(3, 4);
      notes.push("Near middle/apical third boundary.");
    } else {
      plausible.push(4);
    }
  }

  return {
    status: "complete",
    value,
    mostApicalExtentMmFromCEJ: apical,
    relativeToLocalCrest,
    rootThird,
    borderline,
    plausibleValues: plausible,
    notes,
  };
}

export function circumferenceFromAngle(
  maximumAngleDegrees: number | null,
): {
  status: "complete" | "indeterminate" | "not_assessable";
  value: CircumferenceValue | null;
  maximumAngleDegrees: number | null;
  borderline: boolean;
  plausibleValues: CircumferenceValue[];
  notes: string[];
} {
  if (maximumAngleDegrees === null || Number.isNaN(maximumAngleDegrees)) {
    return {
      status: "indeterminate",
      value: null,
      maximumAngleDegrees: null,
      borderline: false,
      plausibleValues: [],
      notes: ["Angular circumferential extent unresolved."],
    };
  }
  if (maximumAngleDegrees < 0 || maximumAngleDegrees > 360) {
    return {
      status: "not_assessable",
      value: null,
      maximumAngleDegrees,
      borderline: false,
      plausibleValues: [],
      notes: ["Circumference angle outside [0, 360]."],
    };
  }

  const angle = maximumAngleDegrees;
  const tol = CIRCUMFERENCE_BOUNDARY_TOLERANCE_DEG;

  const near = (boundary: number) => Math.abs(angle - boundary) <= tol;

  let value: CircumferenceValue;
  let borderline = false;
  let plausible: CircumferenceValue[] = [];
  const notes: string[] = [];

  if (angle <= 90) {
    value = "A";
    plausible = near(90) ? ["A", "B"] : ["A"];
    borderline = near(90);
  } else if (angle <= 180) {
    value = "B";
    if (near(90)) {
      borderline = true;
      plausible = ["A", "B"];
    } else if (near(180)) {
      borderline = true;
      plausible = ["B", "C"];
    } else {
      plausible = ["B"];
    }
  } else if (angle <= 270) {
    value = "C";
    if (near(180)) {
      borderline = true;
      plausible = ["B", "C"];
    } else if (near(270)) {
      borderline = true;
      plausible = ["C", "D"];
    } else {
      plausible = ["C"];
    }
  } else {
    value = "D";
    plausible = near(270) ? ["C", "D"] : ["D"];
    borderline = near(270);
  }

  if (borderline) {
    notes.push(
      `Angle ${angle.toFixed(1)}° is within ±${tol}° of a Patel circumference boundary.`,
    );
  }

  // Exact boundaries: ≤90 A, >90–≤180 B, >180–≤270 C, >270 D
  // Re-affirm exact boundary assignment for non-tolerance path already handled.
  if (angle === 90) value = "A";
  if (angle === 180) value = "B";
  if (angle === 270) value = "C";

  return {
    status: "complete",
    value,
    maximumAngleDegrees: angle,
    borderline,
    plausibleValues: plausible.length ? plausible : [value],
    notes,
  };
}

/**
 * Union of angular sectors around the canal/root center.
 * Each sector is [startDeg, endDeg) in [0, 360), allowing wrap-around.
 */
export function unionAngularSectors(
  sectors: Array<{ startDeg: number; endDeg: number }>,
): number {
  if (!sectors.length) return 0;
  const coverage = new Array<boolean>(3600).fill(false); // 0.1° bins
  for (const sector of sectors) {
    const start = ((sector.startDeg % 360) + 360) % 360;
    const end = ((sector.endDeg % 360) + 360) % 360;
    if (start === end) {
      // Full circle if explicitly identical and intended as full — treat as 0 unless 360 span encoded differently.
      continue;
    }
    if (end < start) {
      // wrap
      for (let i = Math.floor(start * 10); i < 3600; i++) coverage[i] = true;
      for (let i = 0; i < Math.floor(end * 10); i++) coverage[i] = true;
    } else {
      for (let i = Math.floor(start * 10); i < Math.floor(end * 10); i++) {
        coverage[i] = true;
      }
    }
  }
  const bins = coverage.filter(Boolean).length;
  return bins / 10;
}

export type CanalProximityInput = {
  minimumSeparationMm: number | null;
  lowerUncertaintyBoundMm?: number | null;
  lesionCanalContactOrIntersection: boolean | null;
  continuousDentineBarrierVisible: boolean | null;
};

export function computeCanalProximity(input: CanalProximityInput): {
  status: "complete" | "indeterminate" | "not_assessable";
  value: CanalProximityValue | null;
  minimumSeparationMm: number | null;
  lowerUncertaintyBoundMm: number | null;
  canalWallStatus:
    | "continuous_dentine_barrier"
    | "probable_contact"
    | "breach_or_intersection"
    | "unresolved"
    | null;
  borderline: boolean;
  plausibleValues: CanalProximityValue[];
  notes: string[];
} {
  const notes: string[] = [];
  const {
    minimumSeparationMm: sep,
    lesionCanalContactOrIntersection: contact,
    continuousDentineBarrierVisible: barrier,
  } = input;
  const lower =
    input.lowerUncertaintyBoundMm ??
    (sep === null ? null : sep - CANAL_SEPARATION_UNCERTAINTY_FLOOR_MM);

  if (contact === true) {
    return {
      status: "complete",
      value: "p",
      minimumSeparationMm: sep ?? 0,
      lowerUncertaintyBoundMm: lower,
      canalWallStatus: "breach_or_intersection",
      borderline: false,
      plausibleValues: ["p"],
      notes: ["Lesion contacts or intersects canal mask — probable pulpal involvement (p)."],
    };
  }

  if (barrier === null || sep === null || contact === null) {
    return {
      status: "indeterminate",
      value: null,
      minimumSeparationMm: sep,
      lowerUncertaintyBoundMm: lower,
      canalWallStatus: "unresolved",
      borderline: false,
      plausibleValues: [],
      notes: ["Canal or lesion boundary unresolved — d/p not assessable."],
    };
  }

  if (barrier === true && lower !== null && lower > 0) {
    const borderline = lower <= CANAL_SEPARATION_UNCERTAINTY_FLOOR_MM * 2;
    return {
      status: "complete",
      value: "d",
      minimumSeparationMm: sep,
      lowerUncertaintyBoundMm: lower,
      canalWallStatus: "continuous_dentine_barrier",
      borderline,
      plausibleValues: borderline ? ["d", "p"] : ["d"],
      notes: borderline
        ? ["Dentine bridge near spatial uncertainty floor — manual d/p review required."]
        : notes,
    };
  }

  if (barrier === false || (lower !== null && lower <= 0)) {
    return {
      status: "complete",
      value: "p",
      minimumSeparationMm: sep,
      lowerUncertaintyBoundMm: lower,
      canalWallStatus: "probable_contact",
      borderline: lower !== null && Math.abs(lower) <= CANAL_SEPARATION_UNCERTAINTY_FLOOR_MM,
      plausibleValues:
        lower !== null && Math.abs(lower) <= CANAL_SEPARATION_UNCERTAINTY_FLOOR_MM
          ? ["d", "p"]
          : ["p"],
      notes: [
        "No resolvable continuous dentine barrier within validated spatial uncertainty — p.",
        "p means probable pulpal involvement on imaging, not pulp necrosis.",
      ],
    };
  }

  return {
    status: "indeterminate",
    value: null,
    minimumSeparationMm: sep,
    lowerUncertaintyBoundMm: lower,
    canalWallStatus: "unresolved",
    borderline: false,
    plausibleValues: [],
    notes: ["Insufficient evidence to assign d or p."],
  };
}

export function buildPatelCode(
  height: HeightValue | null,
  circumference: CircumferenceValue | null,
  canal: CanalProximityValue | null,
): string | null {
  if (height === null || circumference === null || canal === null) return null;
  return `${height}${circumference}${canal}`;
}
