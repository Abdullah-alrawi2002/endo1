"""Descriptive, measurement-based extent / severity summary of a suspected ECR candidate (rev 2.9).

This replaces the Patel 3D classifier in the agent conference.  It reports **continuous physical quantities**
(mm, degrees, fractions, booleans) taken from the verified half of the feature bundle, and three coarse
*descriptors* that the treatment planner keys its safe set on:

  extent_descriptor    limited | moderate | extensive        (volume, long-axis, circumferential, dentine fraction)
  pulp_relationship    dentine_barrier_preserved | thin_dentine_barrier | canal_communication | not_assessable
  access_descriptor    supracrestal_portal | shallow_subcrestal_portal | deep_subcrestal_portal |
                       no_surface_portal | crest_unknown

The descriptors are engineering bands on measurements (thresholds below), *not* a staging system: no height
class, circumferential letter or canal-proximity code is produced, and nothing here maps to or from such codes.
Band edges are reported together with the distance to the nearest edge so a reviewer sees when a descriptor is
fragile (`near_boundary`), which is also what triggers the planner's re-inspection request.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .perception.feature_schema import FeatureBundle

EXTENT_LEVELS = ["limited", "moderate", "extensive"]
PULP_LEVELS = ["dentine_barrier_preserved", "thin_dentine_barrier", "canal_communication", "not_assessable"]
ACCESS_LEVELS = ["supracrestal_portal", "shallow_subcrestal_portal", "deep_subcrestal_portal", "no_surface_portal", "crest_unknown"]

# --- band thresholds (engineering choices; documented in docs/VLM_FEATURE_MEASUREMENT.md) ---------------------
EXTENT_RULES = {                      # each contributes 0 / 1 / 2 points
    "long_axis_extent_mm": (3.0, 6.0),
    "circumferential_fraction": (1 / 3, 2 / 3),
    "dentine_volume_fraction_involved": (0.05, 0.15),
    "lesion_volume_mm3": (20.0, 60.0),
}
EXTENT_CUTS = (2, 5)                  # score <= 1 limited, 2..4 moderate, >= 5 extensive
THIN_BARRIER_MM = 0.5                 # dentine thinner than this between defect and canal -> thin barrier
SHALLOW_SUBCRESTAL_MM = 3.0           # portal level up to this far below the crest still counts as externally reachable
RESTORABILITY_CONCERN_FRACTION = 0.30
REPAIR_PHASE_INCLUSION_FRACTION = 0.5


@dataclass
class ExtentSummary:
    component_id: int
    lesion_volume_mm3: Optional[float]
    long_axis_extent_mm: Optional[float]
    coronal_limit_from_cej_mm: Optional[float]
    apical_limit_from_cej_mm: Optional[float]
    portal_level_from_crest_mm: Optional[float]      # coronal limit relative to the bone crest (+ = below crest)
    apical_limit_from_crest_mm: Optional[float]      # + = extends apical to the crest
    mesio_distal_extent_mm: Optional[float]
    bucco_lingual_extent_mm: Optional[float]
    max_depth_from_surface_mm: Optional[float]
    circumferential_extent_deg: Optional[float]
    circumferential_fraction: Optional[float]
    min_canal_distance_mm: Optional[float]
    canal_communication: Optional[bool]
    dentine_volume_fraction_involved: Optional[float]
    repair_tissue_fraction: float
    undermining_ratio: Optional[float]
    portal_present: bool
    # descriptors
    extent_descriptor: str
    extent_score: int
    pulp_relationship: str
    access_descriptor: str
    restorability_concern: bool
    repair_phase_suspected: bool
    near_boundary: List[str] = field(default_factory=list)
    measurement_confidence: Dict[str, float] = field(default_factory=dict)
    unavailable: List[str] = field(default_factory=list)
    narrative: str = ""
    scheme: str = "measurement_v1"

    @property
    def cell_key(self) -> str:
        return f"extent={self.extent_descriptor}|pulp={self.pulp_relationship}|access={self.access_descriptor}"

    @property
    def borderline(self) -> bool:
        return bool(self.near_boundary)

    @property
    def deep(self) -> bool:
        """External access unlikely or lesion large: mid/apical reach by measurement, not by stage."""
        return self.extent_descriptor == "extensive" or self.access_descriptor == "deep_subcrestal_portal"

    @property
    def pulp_involved(self) -> bool:
        return self.pulp_relationship == "canal_communication"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["cell_key"] = self.cell_key
        d["borderline"] = self.borderline
        d["is_staging_code"] = False
        return d


def _band_points(value: Optional[float], cuts) -> int:
    if value is None:
        return 0
    lo, hi = cuts
    return 2 if value > hi else 1 if value > lo else 0


def summarise_extent(bundle: FeatureBundle, descriptors=None) -> ExtentSummary:
    """Build the summary from *verified* (voxel-geometry) measurements only.

    Model-only values never feed a descriptor: a quantity that geometry could not measure is listed in
    `unavailable` and the corresponding descriptor falls back to its "not assessable / unknown" level.
    """
    g = lambda name: bundle.measurement_value(name, verified_only=True)  # noqa: E731
    conf = {k: round(m.confidence, 3) for k, m in bundle.measurements.items() if m.verified}
    unavailable = [k for k, m in bundle.measurements.items() if not m.verified]
    vol, long_ax = g("lesion_volume_mm3"), g("long_axis_extent_mm")
    circ = g("circumferential_extent_deg")
    circ_frac = None if circ is None else float(circ) / 360.0
    dent = g("dentine_volume_fraction_involved")
    dmin, comm = g("min_distance_to_canal_mm"), g("root_canal_communication")
    portal_crest, apical_crest = g("coronal_limit_from_crest_mm"), g("apical_limit_from_crest_mm")
    portal_present = bool(getattr(descriptors, "external_surface_connection", True))
    repair_frac = float(getattr(descriptors, "radiopaque_inclusion_fraction", 0.0) or 0.0)
    undermining = getattr(descriptors, "undermining_ratio", None)
    near: List[str] = []
    # --- extent -------------------------------------------------------------------------------------------------
    values = {"long_axis_extent_mm": long_ax, "circumferential_fraction": circ_frac,
              "dentine_volume_fraction_involved": dent, "lesion_volume_mm3": vol}
    score = sum(_band_points(values[k], EXTENT_RULES[k]) for k in EXTENT_RULES)
    extent = "limited" if score < EXTENT_CUTS[0] else "extensive" if score >= EXTENT_CUTS[1] else "moderate"
    if score in (EXTENT_CUTS[0] - 1, EXTENT_CUTS[0], EXTENT_CUTS[1] - 1, EXTENT_CUTS[1]):
        near.append(f"extent_score_{score}_within_1_of_band_edge")
    # --- pulp relationship ---------------------------------------------------------------------------------------
    pitch = float(min(bundle.spacing_zyx_mm))
    if comm is None or dmin is None:
        pulp = "not_assessable"
    elif comm:
        pulp = "canal_communication"
    elif dmin < THIN_BARRIER_MM:
        pulp = "thin_dentine_barrier"
        if abs(dmin - THIN_BARRIER_MM) <= pitch:
            near.append("canal_distance_within_one_voxel_of_thin_barrier_edge")
    else:
        pulp = "dentine_barrier_preserved"
        if abs(dmin - THIN_BARRIER_MM) <= pitch:
            near.append("canal_distance_within_one_voxel_of_thin_barrier_edge")
    # --- access -------------------------------------------------------------------------------------------------
    if not portal_present:
        access = "no_surface_portal"
    elif portal_crest is None:
        access = "crest_unknown"
    elif portal_crest <= 0:
        access = "supracrestal_portal"
        if abs(portal_crest) <= pitch:
            near.append("portal_level_within_one_voxel_of_bone_crest")
    elif portal_crest <= SHALLOW_SUBCRESTAL_MM:
        access = "shallow_subcrestal_portal"
        if abs(portal_crest - SHALLOW_SUBCRESTAL_MM) <= pitch or portal_crest <= pitch:
            near.append("portal_level_within_one_voxel_of_access_band_edge")
    else:
        access = "deep_subcrestal_portal"
        if abs(portal_crest - SHALLOW_SUBCRESTAL_MM) <= pitch:
            near.append("portal_level_within_one_voxel_of_access_band_edge")
    restor = bool(dent is not None and dent >= RESTORABILITY_CONCERN_FRACTION)
    repair = bool(repair_frac >= REPAIR_PHASE_INCLUSION_FRACTION)
    s = ExtentSummary(component_id=bundle.candidate_id, lesion_volume_mm3=vol, long_axis_extent_mm=long_ax,
                      coronal_limit_from_cej_mm=g("coronal_limit_from_cej_mm"), apical_limit_from_cej_mm=g("apical_limit_from_cej_mm"),
                      portal_level_from_crest_mm=portal_crest, apical_limit_from_crest_mm=apical_crest,
                      mesio_distal_extent_mm=g("mesio_distal_extent_mm"), bucco_lingual_extent_mm=g("bucco_lingual_extent_mm"),
                      max_depth_from_surface_mm=g("max_depth_from_surface_mm"), circumferential_extent_deg=circ,
                      circumferential_fraction=None if circ_frac is None else round(circ_frac, 3),
                      min_canal_distance_mm=dmin, canal_communication=comm, dentine_volume_fraction_involved=dent,
                      repair_tissue_fraction=round(repair_frac, 3), undermining_ratio=undermining, portal_present=portal_present,
                      extent_descriptor=extent, extent_score=int(score), pulp_relationship=pulp, access_descriptor=access,
                      restorability_concern=restor, repair_phase_suspected=repair, near_boundary=near,
                      measurement_confidence=conf, unavailable=unavailable)
    s.narrative = narrative(s)
    return s


def narrative(s: ExtentSummary) -> str:
    def f(v, fmt="{:.1f}", unit=""):
        return "not measured" if v is None else fmt.format(v) + unit
    bits = [f"{s.extent_descriptor} extent (score {s.extent_score}/8): long-axis {f(s.long_axis_extent_mm, unit=' mm')}, "
            f"circumferential {f(s.circumferential_extent_deg, '{:.0f}', ' deg')}, "
            f"{f(s.dentine_volume_fraction_involved, '{:.0%}')} of root dentine, volume {f(s.lesion_volume_mm3, unit=' mm3')}"]
    if s.coronal_limit_from_cej_mm is not None:
        bits.append(f"coronal limit {s.coronal_limit_from_cej_mm:+.1f} mm from CEJ, apical limit {s.apical_limit_from_cej_mm:+.1f} mm")
    bits.append({"supracrestal_portal": "portal level at or above the bone crest",
                 "shallow_subcrestal_portal": f"portal level {s.portal_level_from_crest_mm:.1f} mm below the crest" if s.portal_level_from_crest_mm is not None else "portal shallow subcrestal",
                 "deep_subcrestal_portal": f"portal level {s.portal_level_from_crest_mm:.1f} mm below the crest (external access unlikely)" if s.portal_level_from_crest_mm is not None else "portal deep subcrestal",
                 "no_surface_portal": "no external surface portal in the candidate mask",
                 "crest_unknown": "bone crest level not identified"}[s.access_descriptor])
    bits.append({"dentine_barrier_preserved": f"dentine barrier to canal {f(s.min_canal_distance_mm, '{:.2f}', ' mm')}",
                 "thin_dentine_barrier": f"thin dentine barrier {f(s.min_canal_distance_mm, '{:.2f}', ' mm')} (< {THIN_BARRIER_MM} mm)",
                 "canal_communication": "candidate communicates with the canal space",
                 "not_assessable": "canal relationship not assessable"}[s.pulp_relationship])
    if s.repair_phase_suspected:
        bits.append(f"radiopaque inclusions {s.repair_tissue_fraction:.0%} (repair-phase appearance possible)")
    if s.restorability_concern:
        bits.append("dentine loss raises a restorability concern")
    if s.near_boundary:
        bits.append("descriptor(s) near a band edge: " + ", ".join(s.near_boundary))
    return "; ".join(bits) + ". Descriptive measurements, not a staging code."
