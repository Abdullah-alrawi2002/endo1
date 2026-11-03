"""Patient chart retrieval ("RAG with the chart") and contextual filtering.

The chart is structured, clinician-entered context: age, chief complaint (and where it is), dental history
events per tooth, symptoms.  `retrieve(tooth)` returns the entries relevant to one tooth so the conference
can (a) set clinical modifiers it would otherwise have to ask for, (b) shift hypothesis priors for
*predisposing factors* and (c) prioritise which teeth get the full conference first.  Chart context never
creates an imaging finding; it only changes how established findings are interpreted and ordered.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# FDI quadrant -> teeth; Universal numbering is accepted via `universal_to_iso`.
QUADRANT_TEETH = {q: [q * 10 + i for i in range(1, 9)] for q in (1, 2, 3, 4)}
_UNIVERSAL_TO_ISO = {**{i: 18 - (i - 1) for i in range(1, 9)}, **{i: 21 + (i - 9) for i in range(9, 17)},
                     **{i: 38 - (i - 17) for i in range(17, 25)}, **{i: 41 + (i - 25) for i in range(25, 33)}}


def universal_to_iso(n: int) -> int:
    return _UNIVERSAL_TO_ISO[int(n)]


# history event -> (prior shift tag, modifier hint); shifts are applied in the Diagnocat/context layer.
EVENT_EFFECTS: Dict[str, Dict[str, Any]] = {
    "orthodontics": {"predisposing": "orthodontic treatment (ESE 2023 / Patel: most frequent reported predisposing factor)", "ECR": +0.5},
    "trauma": {"predisposing": "dental trauma", "ECR": +0.5},
    "bleaching": {"predisposing": "internal bleaching", "ECR": +0.4},
    "rct": {"modifier": {"pulp_vital": False}, "canal_hint": "hyperdense_filling", "ECR": 0.0},
    "crown": {"modifier": {"restorable": True}, "restoration_margin_or_burnout": +0.6},
    "filling": {"restoration_margin_or_burnout": +0.4},
    "perio_treatment": {"modifier": {"perio_support_adequate": False}, "anatomical_concavity_or_furcation": +0.2},
    "bruxism": {"noncarious_cervical_lesion": +0.5},
    "high_caries_risk": {"caries": +0.5},
}


@dataclass
class HistoryEvent:
    event: str                       # key in EVENT_EFFECTS or free text
    tooth_iso: Optional[int] = None  # None = whole mouth / unknown
    years_ago: Optional[float] = None
    note: str = ""


@dataclass
class PatientChart:
    age_years: Optional[float] = None
    sex: Optional[str] = None
    chief_complaint: str = ""
    complaint_teeth_iso: List[int] = field(default_factory=list)
    complaint_quadrants: List[int] = field(default_factory=list)     # FDI quadrants 1-4
    symptoms: List[str] = field(default_factory=list)                # e.g. "cold sensitivity", "bleeding gum", "pink spot"
    history: List[HistoryEvent] = field(default_factory=list)
    medications_or_conditions: List[str] = field(default_factory=list)
    free_text: str = ""

    # ------------------------------------------------------------------ #
    def focus_teeth(self) -> List[int]:
        teeth = list(self.complaint_teeth_iso)
        for q in self.complaint_quadrants:
            teeth += QUADRANT_TEETH.get(int(q), [])
        return sorted(set(int(t) for t in teeth))

    def retrieve(self, tooth_iso: Optional[int]) -> Dict[str, Any]:
        """Chart entries relevant to one tooth: events on that tooth or mouth-wide, symptom focus, priority weight."""
        ev = [h for h in self.history if h.tooth_iso is None or (tooth_iso is not None and int(h.tooth_iso) == int(tooth_iso))]
        in_focus = tooth_iso is not None and int(tooth_iso) in self.focus_teeth()
        shifts: Dict[str, float] = {}
        modifiers: Dict[str, Any] = {}
        predisposing: List[str] = []
        canal_hint = None
        for h in ev:
            eff = EVENT_EFFECTS.get(h.event.lower().replace(" ", "_"), {})
            for k, v in eff.items():
                if k == "modifier":
                    modifiers.update(v)
                elif k == "predisposing":
                    predisposing.append(v + (f" ({h.years_ago:g} y ago)" if h.years_ago is not None else ""))
                elif k == "canal_hint":
                    canal_hint = v
                elif isinstance(v, (int, float)) and v:
                    shifts[k] = shifts.get(k, 0.0) + float(v)
        sym = [s.lower() for s in self.symptoms]
        if in_focus and any(("sensitiv" in s or "pain" in s or "ache" in s) for s in sym):
            modifiers["symptomatic"] = True
        if self.age_years is not None and self.age_years < 18:
            modifiers["growing_patient"] = True
        priority = 1.0 + (2.0 if in_focus else 0.0) + 0.5 * len(predisposing)
        return {"tooth_iso": tooth_iso, "in_complaint_focus": in_focus, "events": [asdict(h) for h in ev],
                "prior_shifts": shifts, "modifier_hints": modifiers, "predisposing_factors": predisposing,
                "canal_hint": canal_hint, "priority": priority,
                "clinical_findings_hints": {"symptomatic": modifiers.get("symptomatic"),
                                            "history_trauma_ortho_bleaching": bool(predisposing) or None,
                                            "pink_spot": True if any("pink" in s for s in sym) and in_focus else None,
                                            "bleeding_on_probing_defect": True if any("bleed" in s for s in sym) and in_focus else None}}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientChart":
        d = dict(d or {})
        d["history"] = [HistoryEvent(**h) if isinstance(h, dict) else h for h in d.get("history", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
