"""Per-tooth context from Diagnocat findings + patient chart -> prior shifts, modifiers, triage priority.

Rules (all shifts are additive log-odds on the hypothesis prior; p = Diagnocat probability, 1.0 for
clinician-added findings).  None of this adds an imaging finding: if our screen finds no candidate on a
tooth, the tooth is reported as "no candidate" regardless of the Diagnocat context.

  internal resorption (p)          -> internal_resorption +2.0p; canal hint "ai_internal_resorption"
  root caries / caries signs (p)   -> caries +1.2p
  abfraction (p)                   -> noncarious_cervical_lesion +1.2p
  crown / filling / overhang /
  open margin (p)                  -> restoration_margin_or_burnout +1.0p; imaging_artifact +0.5p (metal)
  endo treated (p>=0.5)            -> pulp_vital=False; canal hint "hyperdense_filling"
  periapical radiolucency /
  PDL widening apical (p>=0.5)     -> pulp_vital=False hint, symptomatic unknown
  bone loss severe / furcation     -> perio_support_adequate=False
  impaction / ortho appliance      -> predisposing factor (ECR +0.4)
  external resorption flag         -> clinician-added only: ECR +1.0 (never from AI; Diagnocat has no such class)
  molar (FDI x6-x8)                -> multirooted=True
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..agents.chart import PatientChart
from ..agents.criteria import HYPOTHESES, PRIOR
from ..knowledge import tooth_type_from_iso
from .schema import DiagnocatStudy, DiagnocatTooth

_RESTORATION = ("crown", "filling", "overhang", "open_margin", "secondary_caries", "post")


@dataclass
class ToothContext:
    iso_number: int
    tooth_type: str
    prior_shifts: Dict[str, float] = field(default_factory=dict)
    modifier_hints: Dict[str, Any] = field(default_factory=dict)
    clinical_findings_hints: Dict[str, Any] = field(default_factory=dict)
    canal_hint: Optional[str] = None
    predisposing_factors: List[str] = field(default_factory=list)
    metal_expected: bool = False
    priority: float = 1.0
    diagnocat_conditions: List[Dict[str, Any]] = field(default_factory=list)
    chart: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _p(c) -> float:
    return 1.0 if c.source == "clinician" else (c.probability if c.probability is not None else 0.5)


def context_from_tooth(t: DiagnocatTooth, chart: Optional[PatientChart] = None) -> ToothContext:
    ctx = ToothContext(iso_number=t.iso_number, tooth_type=tooth_type_from_iso(t.iso_number),
                       diagnocat_conditions=[c.to_dict() for c in t.conditions])
    sh = ctx.prior_shifts
    mods = ctx.modifier_hints
    if ctx.tooth_type == "molar":
        mods["multirooted"] = True
    pri = 1.0
    skip = False
    for c in t.conditions:
        p = _p(c)
        if c.code == "internal_resorption":
            sh["internal_resorption"] = sh.get("internal_resorption", 0) + 2.0 * p
            ctx.canal_hint = ctx.canal_hint or "ai_internal_resorption"
            pri += 2.0 * p
        elif c.code in ("root_caries", "caries_signs"):
            sh["caries"] = sh.get("caries", 0) + 1.2 * p
            pri += 0.5 * p
        elif c.code == "abfraction":
            sh["noncarious_cervical_lesion"] = sh.get("noncarious_cervical_lesion", 0) + 1.2 * p
            pri += 0.5 * p
        elif c.code in _RESTORATION:
            sh["restoration_margin_or_burnout"] = sh.get("restoration_margin_or_burnout", 0) + 1.0 * p
            sh["imaging_artifact"] = sh.get("imaging_artifact", 0) + 0.5 * p
            ctx.metal_expected = ctx.metal_expected or c.code in ("crown", "post") or p >= 0.7
            mods.setdefault("restorable", True)
            ctx.clinical_findings_hints["restoration_at_defect"] = True
        elif c.code == "endo_treated" and p >= 0.5:
            mods["pulp_vital"] = False
            ctx.canal_hint = "hyperdense_filling"
        elif c.code in ("periapical_radiolucency", "pdl_widening_apical") and p >= 0.5:
            mods.setdefault("pulp_vital", False)
            ctx.notes.append(f"{c.name} (p={p:.2f}): pulp status likely non-vital; confirm with sensibility tests")
            pri += 0.5
        elif c.code in ("bone_loss", "bone_loss_vertical", "furcation"):
            if (c.severity or "").startswith("sev") or c.code == "furcation" or p >= 0.8:
                mods["perio_support_adequate"] = False
            pri += 0.3
        elif c.code in ("impaction", "ortho_appliance"):
            ctx.predisposing_factors.append(c.name + " (Diagnocat)")
            sh["ECR"] = sh.get("ECR", 0) + 0.4 * p
            pri += 0.5
        elif c.code == "external_resorption_flag":
            if c.source == "clinician":
                sh["ECR"] = sh.get("ECR", 0) + 1.0
                pri += 3.0
                ctx.notes.append("external resorption flagged manually in Diagnocat by a clinician")
            else:
                ctx.notes.append("'external resorption' labelled as AI output - Diagnocat has no such class; treated as unverified, no shift")
        elif c.code == "root_fracture":
            mods["root_fracture_risk_high"] = True
            pri += 1.0
        elif c.code == "missing":
            skip = True
            ctx.notes.append("tooth missing - skipped")
        elif c.code == "implant":
            skip = True
            ctx.notes.append("implant - skipped")
    if t.comment:
        ctx.notes.append("Diagnocat clinician comment: " + t.comment)
        pri += 1.0
    if chart is not None:
        r = chart.retrieve(t.iso_number)
        ctx.chart = r
        for k, v in r["prior_shifts"].items():
            sh[k] = sh.get(k, 0) + v
        for k, v in r["modifier_hints"].items():
            mods.setdefault(k, v)
        ctx.predisposing_factors += r["predisposing_factors"]
        ctx.canal_hint = ctx.canal_hint or r["canal_hint"]
        for k, v in r["clinical_findings_hints"].items():
            if v is not None:
                ctx.clinical_findings_hints.setdefault(k, v)
        pri += r["priority"] - 1.0
        if chart.age_years is not None:
            mods.setdefault("growing_patient", chart.age_years < 18)
    ctx.priority = 0.0 if skip else round(pri, 2)
    return ctx


def build_contexts(study: Optional[DiagnocatStudy], chart: Optional[PatientChart] = None,
                   teeth_iso: Optional[List[int]] = None) -> List[ToothContext]:
    """Contexts for all teeth (Diagnocat study and/or explicit list), sorted by triage priority (desc)."""
    teeth: Dict[int, DiagnocatTooth] = {}
    if study is not None:
        for t in study.teeth:
            teeth[t.iso_number] = t
    for iso in teeth_iso or []:
        teeth.setdefault(int(iso), DiagnocatTooth(iso_number=int(iso)))
    out = [context_from_tooth(t, chart) for t in teeth.values()]
    out = [c for c in out if c.priority > 0]
    return sorted(out, key=lambda c: -c.priority)


def prior_from_shifts(shifts: Dict[str, float], base: Optional[Dict[str, float]] = None, cap: float = 2.5) -> Dict[str, float]:
    """Apply additive log-odds shifts (clipped to +-cap) to the hypothesis prior and renormalise."""
    base = base or PRIOR
    logp = {h: math.log(base[h]) + max(-cap, min(cap, float(shifts.get(h, 0.0)))) for h in HYPOTHESES}
    m = max(logp.values())
    z = sum(math.exp(v - m) for v in logp.values())
    return {h: round(math.exp(v - m) / z, 4) for h, v in logp.items()}
