import { randomUUID } from "crypto";
import type { EcrMeasurementInput } from "@/lib/ecr/schemas";
import {
  ECR_EVIDENCE_PACKAGE_VERSION,
  type CanalEvidenceView,
  type CircumferenceEvidenceView,
  type EcrEvidencePackage,
  type HeightEvidenceView,
  ecrEvidencePackageSchema,
} from "@/lib/ecr/agents/schemas";
import { sha256EvidenceHash } from "@/lib/ecr/evidence/evidence-hash";

function eid(prefix: string): string {
  return `${prefix}_${randomUUID().slice(0, 8)}`;
}

/**
 * Build an immutable ECR evidence package from reviewed measurements.
 * Does NOT include deterministic Patel codes or component categories.
 */
export function buildEvidencePackage(input: {
  measurements: EcrMeasurementInput;
  circumferenceUncertaintyDegrees?: number | null;
}): EcrEvidencePackage {
  const m = input.measurements;
  const evidenceIds = [
    eid("eq"),
    eid("eh"),
    eid("ec"),
    eid("en"),
    eid("er"),
    eid("ep"),
    eid("es"),
  ];

  const planning = m.planning ?? {};

  const hashable = {
    packageVersion: ECR_EVIDENCE_PACKAGE_VERSION,
    scope: "CBCT_ONLY" as const,
    quality: {
      qualityStatus: m.scan.qualityStatus,
      oodStatus: m.scan.oodStatus,
      fovCompleteForTarget: m.scan.fovCompleteForTarget,
      nativeVoxelMm: m.scan.nativeVoxelMm,
      artifactWarnings: m.scan.artifactWarnings ?? [],
      qualityGateFailures: m.scan.qualityGateFailures ?? [],
      targetStructuresVisible: m.scan.targetStructuresVisible,
    },
    ecrDifferential: m.ecrDifferential,
    maskReviewStatus: m.maskReviewStatus,
    targetTooth: {
      label: m.toothLabel,
      rootsAffected: m.rootsAffected,
    },
    measurements: {
      mostApicalExtentMmFromCEJ: m.mostApicalExtentMmFromCEJ,
      rootLengthCejToApexMm: m.rootLengthCejToApexMm,
      localCrestDistanceMmFromCEJ: m.localCrestDistanceMmFromCEJ,
      maximumCircumferenceDegrees: m.maximumCircumferenceDegrees,
      circumferenceUncertaintyDegrees:
        input.circumferenceUncertaintyDegrees ?? null,
      minimumLesionCanalSeparationMm: m.minimumLesionCanalSeparationMm,
      separationUncertaintyLowerBoundMm:
        m.separationUncertaintyLowerBoundMm ?? null,
      lesionCanalContactOrIntersection: m.lesionCanalContactOrIntersection,
      continuousDentineBarrierVisible: m.continuousDentineBarrierVisible,
    },
    planningFeatures: {
      portalSurface: m.portalSurface ?? planning.portalSurface ?? null,
      portalAreaMm2: m.portalAreaMm2 ?? planning.portalAreaMm2 ?? null,
      externalAccessProxy: planning.externalAccessProxy ?? "not_assessable",
      internalAccessProxy: planning.internalAccessProxy ?? "not_assessable",
      structuralContinuity: planning.structuralContinuity ?? "not_assessable",
      fractureRiskProxy: planning.fractureRiskProxy ?? "not_assessable",
      rootFormForReplantation: planning.rootFormForReplantation ?? "not_assessable",
      adjacentAnatomyWarnings: planning.adjacentAnatomyWarnings ?? [],
    },
    evidenceIds,
    prohibitedInferences: [
      "pulp_vitality",
      "pulpal_diagnosis",
      "symptoms",
      "periodontal_probing",
      "clinical_access",
      "patient_preference",
      "deterministic_patel_code",
      "peer_component_conclusion",
    ],
  };

  const caseEvidenceHash = sha256EvidenceHash(hashable);

  return ecrEvidencePackageSchema.parse({
    ...hashable,
    caseEvidenceHash,
  });
}

const PROHIBITED = [
  "deterministic_patel_code",
  "deterministic_component_categories",
  "peer_component_conclusion",
  "pulp_vitality",
  "pulpal_diagnosis",
] as const;

/** Role-scoped views — never expose codes or peer conclusions. */
export function heightEvidenceView(pkg: EcrEvidencePackage): HeightEvidenceView {
  return {
    caseEvidenceHash: pkg.caseEvidenceHash,
    evidenceIds: pkg.evidenceIds.filter((id) => id.startsWith("eh_") || id.startsWith("eq_") || id.startsWith("er_")),
    mostApicalExtentMmFromCEJ: pkg.measurements.mostApicalExtentMmFromCEJ,
    rootLengthCejToApexMm: pkg.measurements.rootLengthCejToApexMm,
    localCrestDistanceMmFromCEJ: pkg.measurements.localCrestDistanceMmFromCEJ,
    structuresVisible: {
      cejRegion: pkg.quality.targetStructuresVisible.cejRegion,
      localAlveolarCrest: pkg.quality.targetStructuresVisible.localAlveolarCrest,
      apex: pkg.quality.targetStructuresVisible.apex,
      crownRootComplex: pkg.quality.targetStructuresVisible.crownRootComplex,
    },
    prohibited: [...PROHIBITED],
  };
}

export function circumferenceEvidenceView(
  pkg: EcrEvidencePackage,
): CircumferenceEvidenceView {
  return {
    caseEvidenceHash: pkg.caseEvidenceHash,
    evidenceIds: pkg.evidenceIds.filter((id) => id.startsWith("ec_") || id.startsWith("eq_")),
    maximumCircumferenceDegrees: pkg.measurements.maximumCircumferenceDegrees,
    circumferenceUncertaintyDegrees:
      pkg.measurements.circumferenceUncertaintyDegrees,
    structuresVisible: {
      lesionMargins: pkg.quality.targetStructuresVisible.lesionMargins,
    },
    prohibited: [...PROHIBITED],
  };
}

export function canalEvidenceView(pkg: EcrEvidencePackage): CanalEvidenceView {
  return {
    caseEvidenceHash: pkg.caseEvidenceHash,
    evidenceIds: pkg.evidenceIds.filter((id) => id.startsWith("en_") || id.startsWith("eq_")),
    minimumLesionCanalSeparationMm:
      pkg.measurements.minimumLesionCanalSeparationMm,
    separationUncertaintyLowerBoundMm:
      pkg.measurements.separationUncertaintyLowerBoundMm,
    lesionCanalContactOrIntersection:
      pkg.measurements.lesionCanalContactOrIntersection,
    continuousDentineBarrierVisible:
      pkg.measurements.continuousDentineBarrierVisible,
    structuresVisible: {
      canalBoundaryNearLesion:
        pkg.quality.targetStructuresVisible.canalBoundaryNearLesion,
      lesionMargins: pkg.quality.targetStructuresVisible.lesionMargins,
    },
    prohibited: [...PROHIBITED],
  };
}

/** Recompute hash from package body excluding the hash field itself. */
export function recomputeEvidenceHash(pkg: EcrEvidencePackage): string {
  const { caseEvidenceHash: _, ...rest } = pkg;
  return sha256EvidenceHash(rest);
}
