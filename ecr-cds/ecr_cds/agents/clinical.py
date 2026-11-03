"""Question 8: does the clinical picture fit the 3-D lesion?  (ESE 2023 position statement)

Clinical findings are entered by the clinician; nothing here is inferred from the image.
Each finding shifts the ECR log-odds by an expert-set amount and is reported as consistent /
inconsistent / not assessed.  Unknown findings never change the posterior.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .criteria import HYPOTHESES


@dataclass
class ClinicalFindings:
    pink_spot: Optional[bool] = None                     # cervical pink discolouration
    cervical_defect_hard_irregular: Optional[bool] = None  # probing: hard, irregular, sharp-edged defect (soft/sticky -> caries)
    bleeding_on_probing_defect: Optional[bool] = None    # profuse bleeding from the defect on probing
    pulp_sensibility_normal: Optional[bool] = None       # positive, normal response (early ECR usually vital)
    symptomatic: Optional[bool] = None                   # pulpitis / apical periodontitis symptoms
    history_trauma_ortho_bleaching: Optional[bool] = None  # recognised predisposing factors
    restoration_at_defect: Optional[bool] = None         # restoration margin at/over the defect

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# (finding, value) -> log-odds shifts per hypothesis; expert-set, uncalibrated
_SHIFTS: Dict[Tuple[str, bool], Dict[str, float]] = {
    ("pink_spot", True): {"ECR": 1.2, "internal_resorption": 0.6, "caries": -0.5},
    ("cervical_defect_hard_irregular", True): {"ECR": 0.8, "caries": -1.0, "noncarious_cervical_lesion": 0.2},
    ("cervical_defect_hard_irregular", False): {"caries": 1.0, "ECR": -0.6},
    ("bleeding_on_probing_defect", True): {"ECR": 0.9, "caries": -0.4, "noncarious_cervical_lesion": -0.6,
                                           "restoration_margin_or_burnout": -0.4, "imaging_artifact": -0.8,
                                           "anatomical_concavity_or_furcation": -0.8},
    ("bleeding_on_probing_defect", False): {"ECR": -0.3},
    ("pulp_sensibility_normal", True): {"ECR": 0.3, "internal_resorption": 0.1},
    ("pulp_sensibility_normal", False): {"ECR": 0.0, "internal_resorption": 0.3},   # late ECR / internal both possible
    ("history_trauma_ortho_bleaching", True): {"ECR": 0.5, "internal_resorption": 0.3},
    ("restoration_at_defect", True): {"restoration_margin_or_burnout": 1.0, "caries": 0.4, "ECR": -0.2},
    ("restoration_at_defect", False): {"restoration_margin_or_burnout": -1.5},
}

_CONSISTENT_WITH_ECR = {("pink_spot", True), ("cervical_defect_hard_irregular", True),
                        ("bleeding_on_probing_defect", True), ("pulp_sensibility_normal", True),
                        ("history_trauma_ortho_bleaching", True)}
_INCONSISTENT_WITH_ECR = {("cervical_defect_hard_irregular", False), ("bleeding_on_probing_defect", False),
                          ("restoration_at_defect", True)}


def correlate(posterior: Dict[str, float], findings: ClinicalFindings) -> Tuple[Dict[str, float], Dict]:
    """Return (adjusted posterior, report)."""
    logp = np.log(np.array([max(posterior[h], 1e-6) for h in HYPOTHESES]))
    consistent, inconsistent, applied = [], [], []
    for k, v in findings.to_dict().items():
        shift = _SHIFTS.get((k, bool(v)))
        if shift is None:
            continue
        for h, w in shift.items():
            logp[HYPOTHESES.index(h)] += w
        applied.append({"finding": k, "value": bool(v), "shift": shift})
        if (k, bool(v)) in _CONSISTENT_WITH_ECR:
            consistent.append(k)
        if (k, bool(v)) in _INCONSISTENT_WITH_ECR:
            inconsistent.append(k)
    p = np.exp(logp - logp.max()); p /= p.sum()
    adjusted = {h: round(float(x), 4) for h, x in zip(HYPOTHESES, p)}
    n_assessed = len(findings.to_dict())
    if n_assessed == 0:
        verdict = "not_assessed"
    elif consistent and not inconsistent:
        verdict = "consistent_with_ecr"
    elif inconsistent and not consistent:
        verdict = "inconsistent_with_ecr"
    elif consistent and inconsistent:
        verdict = "mixed"
    else:
        verdict = "non_discriminating"
    return adjusted, {"verdict": verdict, "n_findings": n_assessed, "consistent_with_ecr": consistent,
                      "inconsistent_with_ecr": inconsistent, "applied": applied,
                      "note": "Clinical findings are clinician-entered; shifts are expert-set and uncalibrated (ESE 2023 clinical signs)."}
