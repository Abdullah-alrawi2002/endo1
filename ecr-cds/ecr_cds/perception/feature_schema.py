"""Typed schema for the VLM feature-measurement bundle (rev 2.9).

The bundle has two halves that are kept apart on purpose:

* **visual features** - categorical descriptions a vision-language model gives of the evidence frame
  (texture, borders, pericanal line, portal, restoration relation, artefact confounders).  These are
  *opinions about an image* and are stored as such.
* **measurements** - physical quantities (mm, mm3, fractions, booleans).  Every measurement states its
  `source`: computed from voxel geometry (segmentation + spacing), or reported by the model and *not*
  verifiable.  A model number never overrides a geometric one; when both exist the geometric value is kept and
  the agreement is recorded.

Nothing here stages, classifies or recommends.  No Patel (height / circumferential / canal-proximity)
codes are produced or consumed anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

SCHEMA_VERSION = "vlm_features_v1"

YNU = ["yes", "no", "uncertain"]

# Categorical visual features the model is asked for, with their allowed values.  The last value of each list is
# the fallback used when the model answers outside the enumeration (recorded as a validation error).
VISUAL_ENUMS: Dict[str, List[str]] = {
    "radiolucency_pattern": ["homogeneous", "heterogeneous", "mottled", "not_visible", "uncertain"],
    "border_definition": ["well_defined", "irregular", "smooth", "ill_defined", "uncertain"],
    "fibro_osseous_appearance": YNU,                      # mottled radiopaque islands inside the defect (repair tissue)
    "pericanal_radiopaque_line": ["intact", "partially_interrupted", "interrupted", "not_assessable"],
    "portal_of_entry_visible": YNU,
    "portal_surface": ["buccal", "lingual_palatal", "mesial", "distal", "uncertain"],
    "relation_to_restoration_margin": ["no_restoration_visible", "separate_from_margin", "adjacent_to_margin",
                                       "continuous_with_margin", "uncertain"],
    "burnout_artifact_suspected": YNU,
    "metal_beam_hardening": ["none", "mild", "marked", "uncertain"],
}

# Numeric / boolean estimates the model may give.  Each maps onto a geometric measurement of the same name so it can
# be cross-checked; `kind` drives the tolerance used for the comparison.
ESTIMATE_KINDS: Dict[str, str] = {
    "long_axis_extent_mm": "mm",
    "mesio_distal_extent_mm": "mm",
    "bucco_lingual_extent_mm": "mm",
    "coronal_limit_from_cej_mm": "mm_signed",
    "apical_limit_from_crest_mm": "mm_signed",
    "dentine_volume_fraction_involved": "fraction",
    "min_distance_to_canal_mm": "mm",
    "root_canal_communication": "bool",
}

MEASUREMENT_UNITS: Dict[str, str] = {
    "long_axis_extent_mm": "mm", "mesio_distal_extent_mm": "mm", "bucco_lingual_extent_mm": "mm",
    "coronal_limit_from_cej_mm": "mm", "apical_limit_from_cej_mm": "mm",
    "coronal_limit_from_crest_mm": "mm", "apical_limit_from_crest_mm": "mm",
    "max_depth_from_surface_mm": "mm", "circumferential_extent_deg": "deg",
    "dentine_volume_fraction_involved": "fraction", "min_distance_to_canal_mm": "mm",
    "root_canal_communication": "bool", "lesion_volume_mm3": "mm3",
}

SOURCE_VOXEL = "voxel_geometry"
SOURCE_VLM_UNVERIFIED = "vlm_unverified"
SOURCE_UNAVAILABLE = "unavailable"

# JSON schema sent to the model (OpenAI-compatible backends append it to the system prompt).
VLM_FEATURE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_features": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": v} for k, v in VISUAL_ENUMS.items()},
            "required": list(VISUAL_ENUMS),
        },
        "estimates": {
            "type": "object",
            "description": "Rough physical estimates read from the panels using the stated scale; null when not estimable.",
            "properties": {
                **{k: {"type": ["number", "null"]} for k, kind in ESTIMATE_KINDS.items() if kind != "bool"},
                "root_canal_communication": {"type": "string", "enum": YNU},
            },
        },
        "panels_used": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string", "description": "<= 60 words; only what is visible"},
        "lesion_visible_in_image": {"type": "string", "enum": ["yes", "no", "uncertain", "no_image"]},
    },
    "required": ["visual_features", "estimates", "lesion_visible_in_image"],
}


class FeatureSchemaError(ValueError):
    """Raised when a model answer is structurally unusable (not an object, no visual_features block)."""


def _fallback(name: str) -> str:
    return VISUAL_ENUMS[name][-1]


def validate_vlm_response(obj: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce a raw model answer into the typed shape.  Returns (clean, errors).

    * enum fields outside their allowed values fall back to the enumeration's "uncertain"-type value;
    * numeric estimates must be finite; negative lengths/fractions are dropped; fractions are clipped to [0, 1];
    * `root_canal_communication` becomes True / False / None;
    * unknown keys are dropped.
    Every coercion is recorded in `errors` so the agent can show how literal the model was.
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        raise FeatureSchemaError(f"model answer is not an object ({type(obj).__name__})")
    vf = obj.get("visual_features")
    if not isinstance(vf, dict):
        raise FeatureSchemaError("model answer has no visual_features object")
    clean_vf: Dict[str, str] = {}
    for k, allowed in VISUAL_ENUMS.items():
        v = vf.get(k)
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_") if v is not None else None
        if s in allowed:
            clean_vf[k] = s
        else:
            clean_vf[k] = _fallback(k)
            errors.append(f"visual_features.{k}: {v!r} not in {allowed} -> {clean_vf[k]}")
    est_in = obj.get("estimates")
    if est_in is None:
        est_in = {}
    if not isinstance(est_in, dict):
        errors.append("estimates is not an object; ignored")
        est_in = {}
    clean_est: Dict[str, Optional[Union[float, bool]]] = {}
    for k, kind in ESTIMATE_KINDS.items():
        if k not in est_in:
            continue
        v = est_in[k]
        if kind == "bool":
            s = str(v).strip().lower() if v is not None else "uncertain"
            if isinstance(v, bool):
                clean_est[k] = v
            elif s in ("yes", "true", "present"):
                clean_est[k] = True
            elif s in ("no", "false", "absent"):
                clean_est[k] = False
            else:
                clean_est[k] = None
                if s not in ("uncertain", "none", "null"):
                    errors.append(f"estimates.{k}: {v!r} not yes/no/uncertain -> uncertain")
            continue
        if v is None:
            clean_est[k] = None
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            clean_est[k] = None
            errors.append(f"estimates.{k}: {v!r} is not a number -> null")
            continue
        if not math.isfinite(f):
            clean_est[k] = None
            errors.append(f"estimates.{k}: non-finite -> null")
            continue
        if kind == "fraction":
            if f < 0 or f > 1.0:
                errors.append(f"estimates.{k}: {f} outside [0,1] -> clipped")
            clean_est[k] = float(min(max(f, 0.0), 1.0))
        elif kind == "mm":
            if f < 0:
                clean_est[k] = None
                errors.append(f"estimates.{k}: negative length {f} -> null")
            else:
                clean_est[k] = f
        else:  # signed mm
            clean_est[k] = f
    for k in est_in:
        if k not in ESTIMATE_KINDS:
            errors.append(f"estimates.{k}: unknown field dropped")
    vis = str(obj.get("lesion_visible_in_image", "uncertain")).strip().lower()
    if vis not in ("yes", "no", "uncertain", "no_image"):
        errors.append(f"lesion_visible_in_image: {obj.get('lesion_visible_in_image')!r} -> uncertain")
        vis = "uncertain"
    panels = obj.get("panels_used") or []
    panels = [str(p)[:40] for p in panels][:16] if isinstance(panels, list) else []
    desc = obj.get("description")
    desc = str(desc)[:600] if desc is not None else ""
    return {"visual_features": clean_vf, "estimates": clean_est, "panels_used": panels, "description": desc,
            "lesion_visible_in_image": vis}, errors


# --------------------------------------------------------------------------- #
@dataclass
class VisualFeatures:
    """Categorical image description from the model.  `assessed` is False for a geometry-only bundle."""
    assessed: bool = False
    radiolucency_pattern: str = "uncertain"
    border_definition: str = "uncertain"
    fibro_osseous_appearance: str = "uncertain"
    pericanal_radiopaque_line: str = "not_assessable"
    portal_of_entry_visible: str = "uncertain"
    portal_surface: str = "uncertain"
    relation_to_restoration_margin: str = "uncertain"
    burnout_artifact_suspected: str = "uncertain"
    metal_beam_hardening: str = "uncertain"
    lesion_visible_in_image: str = "no_image"
    description: str = ""
    panels_used: List[str] = field(default_factory=list)

    @classmethod
    def from_clean(cls, clean: Dict[str, Any]) -> "VisualFeatures":
        return cls(assessed=True, lesion_visible_in_image=clean["lesion_visible_in_image"],
                   description=clean.get("description", ""), panels_used=list(clean.get("panels_used", [])),
                   **clean["visual_features"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Measurement:
    """One physical quantity with its grounding.

    value        geometric value when available, else the model's value (then `verified` is False)
    source       voxel_geometry | vlm_unverified | unavailable
    verified     True only when the value was computed from voxels
    confidence   0..1, uncalibrated; reflects mask / landmark reliability, never model certainty
    provenance   frame panels, slice ranges and the axis the value was measured along
    vlm_value    what the model reported, if anything
    vlm_agreement  agrees | disagrees | not_reported | not_comparable
    """
    name: str
    value: Optional[Union[float, bool]]
    unit: str
    source: str
    verified: bool
    confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    vlm_value: Optional[Union[float, bool]] = None
    vlm_agreement: str = "not_reported"
    discrepancy: Optional[float] = None
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.value, float):
            d["value"] = round(self.value, 3)
        if isinstance(self.vlm_value, float):
            d["vlm_value"] = round(self.vlm_value, 3)
        if self.discrepancy is not None:
            d["discrepancy"] = round(self.discrepancy, 3)
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class FeatureBundle:
    """Everything the feature-measurement agent publishes for one candidate."""
    candidate_id: int
    source_id: str
    spacing_zyx_mm: List[float]
    visual: VisualFeatures
    measurements: Dict[str, Measurement]
    vlm: Dict[str, Any]                     # status, backend, dse, images_sent, validation_errors, raw_estimates
    unverified_vlm_values: List[str] = field(default_factory=list)
    vlm_geometry_disagreements: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    disclaimer: str = ("Feature bundle for clinician review. Measurements derive from an automatic candidate mask and "
                       "automatic landmarks unless stated; visual features are a vision-model reading of one evidence "
                       "frame. No staging code, diagnosis or treatment is implied. A negative reading does not exclude ECR.")

    def measurement_value(self, name: str, verified_only: bool = True):
        m = self.measurements.get(name)
        if m is None or m.value is None:
            return None
        if verified_only and not m.verified:
            return None
        return m.value

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "candidate_id": self.candidate_id, "source_id": self.source_id,
                "spacing_zyx_mm": self.spacing_zyx_mm, "visual": self.visual.to_dict(),
                "measurements": {k: m.to_dict() for k, m in self.measurements.items()},
                "vlm": self.vlm, "unverified_vlm_values": list(self.unverified_vlm_values),
                "vlm_geometry_disagreements": list(self.vlm_geometry_disagreements),
                "caveats": list(self.caveats), "disclaimer": self.disclaimer}


def validate_bundle_dict(d: Dict[str, Any]) -> List[str]:
    """Structural check of a serialised bundle (used by tests and by consumers that load reports)."""
    problems: List[str] = []
    if d.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {d.get('schema_version')!r} != {SCHEMA_VERSION}")
    vis = d.get("visual") or {}
    for k, allowed in VISUAL_ENUMS.items():
        if vis.get(k) not in allowed:
            problems.append(f"visual.{k}={vis.get(k)!r} not in enum")
    ms = d.get("measurements") or {}
    for name, unit in MEASUREMENT_UNITS.items():
        m = ms.get(name)
        if m is None:
            problems.append(f"measurement {name} missing")
            continue
        if m.get("unit") != unit:
            problems.append(f"measurement {name}: unit {m.get('unit')!r} != {unit!r}")
        if m.get("source") not in (SOURCE_VOXEL, SOURCE_VLM_UNVERIFIED, SOURCE_UNAVAILABLE):
            problems.append(f"measurement {name}: bad source {m.get('source')!r}")
        if m.get("verified") and m.get("source") != SOURCE_VOXEL:
            problems.append(f"measurement {name}: verified but source is {m.get('source')!r}")
        if not (0.0 <= float(m.get("confidence", -1)) <= 1.0):
            problems.append(f"measurement {name}: confidence out of range")
        v = m.get("value")
        if unit == "bool" and v is not None and not isinstance(v, bool):
            problems.append(f"measurement {name}: value is not boolean")
        if unit != "bool" and v is not None and not isinstance(v, (int, float)):
            problems.append(f"measurement {name}: value is not numeric")
    for k in ("vlm", "unverified_vlm_values", "vlm_geometry_disagreements", "caveats", "disclaimer"):
        if k not in d:
            problems.append(f"bundle key {k} missing")
    return problems
