"""Interchange schema for Diagnocat output.

Diagnocat's Partner API payloads are not public (the Postman collection is partner-gated), so this is
*our* normalised schema.  `load_diagnocat_json` accepts it directly and also maps the common field
spellings seen in Diagnocat reports (ISO/FDI tooth numbers, condition names from the 4.0 manual,
probabilities 0-0.99).  Anything unrecognised is kept in `raw` so nothing is silently lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Diagnocat condition names (AI 4.0 user manual) -> internal codes.  Lower-cased, punctuation-free keys.
CONDITION_ALIASES: Dict[str, str] = {
    # resorption / pulp
    "internal resorption": "internal_resorption",
    "root resorption": "root_resorption_unspecified",
    "external resorption": "external_resorption_flag",        # only if added manually by the clinician
    "external cervical resorption": "external_resorption_flag",
    "cervical resorption": "external_resorption_flag",
    "pulp stone": "pulp_stone", "pulp calcification": "pulp_stone",
    "endodontically treated tooth": "endo_treated", "endodontic treatment": "endo_treated", "root canal filling": "endo_treated",
    "post": "post", "post and core": "post",
    "periapical radiolucency": "periapical_radiolucency", "periapical lesion": "periapical_radiolucency",
    "pdl space widening along the root": "pdl_widening_root", "pdl space widening in periapical region": "pdl_widening_apical",
    "periodontal ligament space widening": "pdl_widening_root",
    # hard tissue loss
    "caries signs": "caries_signs", "caries": "caries_signs", "secondary caries": "secondary_caries",
    "root caries": "root_caries", "caries signs root": "root_caries",
    "abfraction": "abfraction", "abrasion": "abfraction", "erosion": "abfraction", "wedge shaped defect": "abfraction",
    "root fracture": "root_fracture", "horizontal root fracture": "root_fracture", "vertical root fracture": "root_fracture",
    "crown fracture": "crown_fracture",
    # restorations
    "filling": "filling", "restoration": "filling", "artificial crown": "crown", "crown": "crown",
    "overhang": "overhang", "open margin": "open_margin", "implant": "implant", "veneer": "crown", "inlay": "filling", "onlay": "filling",
    # periodontium / bone
    "periodontal bone loss": "bone_loss", "bone loss": "bone_loss", "furcation lesion": "furcation", "furcation involvement": "furcation",
    "vertical bone loss": "bone_loss_vertical", "horizontal bone loss": "bone_loss",
    # eruption / ortho
    "impaction": "impaction", "impacted tooth": "impaction", "orthodontic appliance": "ortho_appliance", "braces": "ortho_appliance",
    "missing": "missing", "missing tooth": "missing", "dental fragment": "fragment",
}

_SEVERITY_WORDS = {"mild": 0.33, "moderate": 0.66, "severe": 1.0}


def normalise_condition_name(name: str) -> str:
    key = "".join(ch if ch.isalnum() or ch == " " else " " for ch in name.lower()).split()
    key = " ".join(key)
    if key in CONDITION_ALIASES:
        return CONDITION_ALIASES[key]
    # strip severity / side qualifiers and retry ("periodontal bone loss severe" -> bone_loss)
    words = [w for w in key.split() if w not in _SEVERITY_WORDS and w not in ("buccal", "lingual", "palatal", "mesial", "distal")]
    key2 = " ".join(words)
    return CONDITION_ALIASES.get(key2, "unknown:" + key.replace(" ", "_"))


@dataclass
class DiagnocatCondition:
    code: str                           # normalised internal code (see CONDITION_ALIASES)
    name: str                           # Diagnocat's display name as given
    probability: Optional[float] = None  # 0-0.99 in Diagnocat; None if not shown
    source: str = "ai"                  # "ai" | "clinician" (manually added / approved in Diagnocat)
    severity: Optional[str] = None      # e.g. mild / moderate / severe for bone loss
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class DiagnocatTooth:
    iso_number: int                     # FDI / ISO 3950 two-digit number (Diagnocat's default numbering)
    conditions: List[DiagnocatCondition] = field(default_factory=list)
    seed_zyx: Optional[List[int]] = None       # voxel inside the tooth in *our* scan grid, if known
    stl_path: Optional[str] = None             # Diagnocat tooth STL export
    canal_stl_path: Optional[str] = None       # Diagnocat pulp/canal STL export
    slices: List[str] = field(default_factory=list)   # paths to Diagnocat's per-tooth slice images
    comment: Optional[str] = None              # clinician comment entered in Diagnocat
    raw: Dict[str, Any] = field(default_factory=dict)

    def has(self, code: str, min_p: float = 0.0) -> Optional[DiagnocatCondition]:
        for c in self.conditions:
            if c.code == code and (c.probability is None or c.probability >= min_p or c.source == "clinician"):
                return c
        return None

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class DiagnocatStudy:
    study_id: str
    teeth: List[DiagnocatTooth]
    source: str                         # "partner_api" | "json_export" | "report_pdf" | "mock"
    diagnocat_version: Optional[str] = None
    report_pdf: Optional[str] = None
    label_map_path: Optional[str] = None       # if a voxel label map was exported / converted
    notes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def tooth(self, iso: int) -> Optional[DiagnocatTooth]:
        return next((t for t in self.teeth if t.iso_number == int(iso)), None)

    def to_dict(self):
        return {"study_id": self.study_id, "source": self.source, "diagnocat_version": self.diagnocat_version,
                "report_pdf": self.report_pdf, "label_map_path": self.label_map_path, "notes": self.notes,
                "teeth": [t.to_dict() for t in self.teeth]}
