"""Rule-based decision matrix: Patel 3D class -> treatment recommendation.

Evidence anchors
----------------
* ESE Position Statement on Root Resorption (Patel et al., Int Endod J 2023;
  doi:10.1111/iej.13916), Table for ECR:
    External repair only ............ 1-2Ad, 2Bd
    External repair + RCT ........... 1-2Ap, 2Bp, 2Cp
    Internal repair + RCT ........... 2Bp, 2Cp, 2Dp, 3Cp, 3Dp
    Intentional replantation ........ 3Ad/p, 3Bd/p
    Decoronation .................... 3-4Cp, 3-4Cd, 3-4Dp, 3-4Dd
    Periodic review ................. 2-4Cp, 2-4Dp
    Extraction ...................... untreatable ECR (no class restriction)
* Patel et al. 2018 part 2 - management (doi:10.1111/iej.12946): internal
  approach preferred when external access is impossible or would remove
  excessive sound tooth structure (entry points small, intra-osseous);
  intentional replantation for interproximal / mid- and apical-third lesions;
  extraction when untreatable with aesthetic/functional/symptomatic issues.
* Heithersay 2007 outcomes (quoted in Patel 2018 part 2): class 1-2 = 100 %
  success, class 3 = 77.8 %, class 4 = 12.5 % -> prognosis prior.
* Talpoş-Niculescu et al. 2025 (RJME 66(2):391): 2Bp and 3Bp lesions that are
  subcrestal, perforating and inaccessible with persistent haemorrhage ended
  in extraction / root amputation -> intra-operative modifiers matter and
  hemisection is a valid option in multirooted teeth.

The matrix returns a SAFE SET (allowed actions) plus a default ranking; the
RL layer re-ranks only within the safe set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Treatment(str, Enum):
    MONITOR = "periodic_review"                 # no active treatment, annual CBCT/clinical review
    EXTERNAL_REPAIR = "external_repair"         # periodontal flap surgery, curettage, TCA, bioceramic/GIC repair, no RCT
    EXTERNAL_REPAIR_VPT = "external_repair_vpt"  # external repair + vital pulp therapy (bioceramic capping)
    EXTERNAL_REPAIR_RCT = "external_repair_rct"  # external repair + root canal treatment
    INTERNAL_REPAIR_RCT = "internal_repair_rct"  # RCT + repair of defect from within the canal
    INTENTIONAL_REPLANT = "intentional_replantation"
    HEMISECTION = "root_resection_hemisection"  # multirooted teeth only
    DECORONATION = "decoronation"               # ridge preservation, growing patients / implant site
    EXTRACTION = "extraction"

    @property
    def irreversible(self) -> bool:
        return self in {Treatment.INTENTIONAL_REPLANT, Treatment.HEMISECTION,
                        Treatment.DECORONATION, Treatment.EXTRACTION}


@dataclass
class Recommendation:
    patel_code: str
    primary: Treatment
    alternatives: List[Treatment]
    safe_set: List[Treatment]
    prognosis_prior: float          # expected success probability (0-1), Heithersay-anchored
    evidence: str                   # 'ESE-2023' | 'Patel-2018' | 'expert-extrapolation'
    rationale: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    missing_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "missing_inputs": self.missing_inputs,
            "patel_code": self.patel_code,
            "primary": self.primary.value,
            "alternatives": [a.value for a in self.alternatives],
            "safe_set": [a.value for a in self.safe_set],
            "prognosis_prior": round(self.prognosis_prior, 3),
            "evidence": self.evidence,
            "rationale": self.rationale,
            "flags": self.flags,
            "requires_confirmation": self.requires_confirmation,
        }


T = Treatment

# --------------------------------------------------------------------------- #
# 32-cell base matrix: (height, spread, proximity) -> (primary, alternatives, prior, evidence)
# Prognosis priors: Heithersay-anchored (1-2 ≈ 0.95, 3 ≈ 0.78, 4 ≈ 0.15) and
# reduced for p-lesions / wider spread, consistent with Irinakis 2022 failure
# gradient (class II 17 %, III 33 %, IV 50 %).
# --------------------------------------------------------------------------- #
DECISION_MATRIX: Dict[str, Tuple[T, List[T], float, str]] = {
    # ---------------- Height 1: supracrestal ----------------
    "1Ad": (T.EXTERNAL_REPAIR, [T.MONITOR], 0.95, "ESE-2023"),
    "1Bd": (T.EXTERNAL_REPAIR, [T.MONITOR], 0.92, "ESE-2023"),
    "1Cd": (T.EXTERNAL_REPAIR, [T.EXTERNAL_REPAIR_RCT, T.MONITOR], 0.80, "Patel-2018"),
    "1Dd": (T.EXTERNAL_REPAIR_RCT, [T.EXTERNAL_REPAIR, T.MONITOR, T.EXTRACTION], 0.60, "expert-extrapolation"),
    "1Ap": (T.EXTERNAL_REPAIR_VPT, [T.EXTERNAL_REPAIR_RCT], 0.90, "ESE-2023"),
    "1Bp": (T.EXTERNAL_REPAIR_RCT, [T.EXTERNAL_REPAIR_VPT], 0.85, "ESE-2023"),
    "1Cp": (T.EXTERNAL_REPAIR_RCT, [T.INTERNAL_REPAIR_RCT, T.MONITOR], 0.70, "Patel-2018"),
    "1Dp": (T.EXTERNAL_REPAIR_RCT, [T.INTERNAL_REPAIR_RCT, T.MONITOR, T.EXTRACTION], 0.50, "expert-extrapolation"),
    # ---------------- Height 2: coronal third, subcrestal ----------------
    "2Ad": (T.EXTERNAL_REPAIR, [T.MONITOR], 0.92, "ESE-2023"),
    "2Bd": (T.EXTERNAL_REPAIR, [T.EXTERNAL_REPAIR_RCT, T.MONITOR], 0.88, "ESE-2023"),
    "2Cd": (T.EXTERNAL_REPAIR_RCT, [T.EXTERNAL_REPAIR, T.INTERNAL_REPAIR_RCT, T.MONITOR], 0.70, "Patel-2018"),
    "2Dd": (T.MONITOR, [T.INTERNAL_REPAIR_RCT, T.DECORONATION, T.EXTRACTION], 0.45, "expert-extrapolation"),
    "2Ap": (T.EXTERNAL_REPAIR_RCT, [T.EXTERNAL_REPAIR_VPT, T.INTERNAL_REPAIR_RCT], 0.85, "ESE-2023"),
    "2Bp": (T.EXTERNAL_REPAIR_RCT, [T.INTERNAL_REPAIR_RCT, T.HEMISECTION, T.EXTRACTION], 0.78, "ESE-2023"),
    "2Cp": (T.INTERNAL_REPAIR_RCT, [T.EXTERNAL_REPAIR_RCT, T.MONITOR, T.EXTRACTION], 0.65, "ESE-2023"),
    "2Dp": (T.INTERNAL_REPAIR_RCT, [T.MONITOR, T.DECORONATION, T.EXTRACTION], 0.45, "ESE-2023"),
    # ---------------- Height 3: middle third ----------------
    "3Ad": (T.INTENTIONAL_REPLANT, [T.EXTERNAL_REPAIR, T.MONITOR], 0.78, "ESE-2023"),
    "3Bd": (T.INTENTIONAL_REPLANT, [T.EXTERNAL_REPAIR_RCT, T.INTERNAL_REPAIR_RCT, T.MONITOR], 0.72, "ESE-2023"),
    "3Cd": (T.MONITOR, [T.INTERNAL_REPAIR_RCT, T.DECORONATION, T.EXTRACTION], 0.45, "ESE-2023"),
    "3Dd": (T.MONITOR, [T.DECORONATION, T.EXTRACTION], 0.30, "ESE-2023"),
    "3Ap": (T.INTENTIONAL_REPLANT, [T.INTERNAL_REPAIR_RCT, T.EXTERNAL_REPAIR_RCT], 0.72, "ESE-2023"),
    "3Bp": (T.INTENTIONAL_REPLANT, [T.INTERNAL_REPAIR_RCT, T.HEMISECTION, T.EXTRACTION], 0.65, "ESE-2023"),
    "3Cp": (T.INTERNAL_REPAIR_RCT, [T.MONITOR, T.DECORONATION, T.EXTRACTION], 0.45, "ESE-2023"),
    "3Dp": (T.INTERNAL_REPAIR_RCT, [T.MONITOR, T.DECORONATION, T.EXTRACTION], 0.30, "ESE-2023"),
    # ---------------- Height 4: apical third ----------------
    "4Ad": (T.INTENTIONAL_REPLANT, [T.MONITOR, T.EXTRACTION], 0.40, "Patel-2018"),
    "4Bd": (T.MONITOR, [T.INTENTIONAL_REPLANT, T.EXTRACTION], 0.30, "Patel-2018"),
    "4Cd": (T.MONITOR, [T.DECORONATION, T.EXTRACTION], 0.15, "ESE-2023"),
    "4Dd": (T.EXTRACTION, [T.DECORONATION, T.MONITOR], 0.10, "ESE-2023"),
    "4Ap": (T.INTENTIONAL_REPLANT, [T.INTERNAL_REPAIR_RCT, T.EXTRACTION], 0.35, "Patel-2018"),
    "4Bp": (T.EXTRACTION, [T.INTENTIONAL_REPLANT, T.HEMISECTION, T.MONITOR], 0.25, "Patel-2018"),
    "4Cp": (T.EXTRACTION, [T.DECORONATION, T.MONITOR], 0.12, "ESE-2023"),
    "4Dp": (T.EXTRACTION, [T.DECORONATION, T.MONITOR], 0.08, "ESE-2023"),
}

END_STAGE_CODES = {"2Dd", "2Dp", "3Cd", "3Cp", "3Dd", "3Dp",
                   "4Ad", "4Ap", "4Bd", "4Bp", "4Cd", "4Cp", "4Dd", "4Dp"}


@dataclass
class ClinicalModifiers:
    """Non-imaging inputs that modulate the matrix (all optional)."""
    symptomatic: Optional[bool] = None           # pain / pulpitis / apical periodontitis
    pulp_vital: Optional[bool] = None
    restorable: Optional[bool] = None            # ferrule / remaining coronal dentine adequate
    perio_support_adequate: Optional[bool] = None
    multirooted: Optional[bool] = None
    lesion_on_root: Optional[str] = None         # e.g. 'DB' for multirooted teeth
    surgically_accessible: Optional[bool] = None  # buccal/lingual & not interproximal/deep subcrestal
    persistent_bleeding: Optional[bool] = None   # intra-operative, after CaOH2 x2
    growing_patient: Optional[bool] = None       # < ~18 y, implant contraindicated
    aesthetic_zone: Optional[bool] = None
    strategic_value_high: Optional[bool] = None  # abutment, key occlusal unit
    patient_prefers_conservative: Optional[bool] = None
    repair_phase: Optional[bool] = None          # calcified_fraction high / histology-like stasis
    root_fracture_risk_high: Optional[bool] = None  # long curved roots (IR contraindication)


# --------------------------------------------------------------------------- #
def recommend_from_matrix(code: str, mods: ClinicalModifiers | None = None,
                          portal_angle_deg: Optional[float] = None,
                          calcified_fraction: Optional[float] = None) -> Recommendation:
    """Look up the base cell then apply clinical modifiers as ordered rules."""
    if code not in DECISION_MATRIX:
        raise KeyError(f"Unknown Patel code {code}")
    mods = mods or ClinicalModifiers()
    primary, alts, prior, evidence = DECISION_MATRIX[code]
    ranking: List[T] = [primary] + [a for a in alts if a != primary]
    rationale = [f"Base cell {code}: {primary.value} ({evidence})."]
    flags: List[str] = []
    height = int(code[0]); spread = code[1]; prox = code[2]

    # Derive accessibility from portal angle if not given:
    # within ±45° of buccal (0°) or lingual (180°) counts as accessible.
    if mods.surgically_accessible is None and portal_angle_deg is not None:
        a = portal_angle_deg % 360
        mods.surgically_accessible = (a <= 45 or a >= 315 or 135 <= a <= 225) and height <= 2
        why = "accessible" if mods.surgically_accessible else (
            "interproximal portal" if not (a <= 45 or a >= 315 or 135 <= a <= 225) else "deep (mid/apical third)")
        rationale.append(f"Accessibility inferred from portal angle {a:.0f}° and height {height} -> {why}.")
    if mods.repair_phase is None and calcified_fraction is not None:
        mods.repair_phase = calcified_fraction >= 0.5

    def promote(t: T, why: str):
        nonlocal ranking
        ranking = [t] + [r for r in ranking if r != t]
        rationale.append(why)

    def demote(t: T, why: str):
        nonlocal ranking
        if t in ranking:
            ranking = [r for r in ranking if r != t] + [t]
            rationale.append(why)

    def remove(t: T, why: str):
        nonlocal ranking
        if t in ranking:
            ranking = [r for r in ranking if r != t]
            rationale.append(why)

    # R1. External approach not possible -> internal approach (Patel 2018 pt 2)
    if mods.surgically_accessible is False and ranking[0] in {T.EXTERNAL_REPAIR, T.EXTERNAL_REPAIR_RCT, T.EXTERNAL_REPAIR_VPT}:
        if height <= 2:
            promote(T.INTERNAL_REPAIR_RCT, "R1: external access not feasible (interproximal / subcrestal) -> internal repair with RCT.")
        else:
            promote(T.INTENTIONAL_REPLANT, "R1: external access not feasible in mid/apical third -> intentional replantation.")
    # R2. Perforation / pulpal signs -> RCT mandatory (no repair-only, no VPT if necrotic)
    if prox == "p" or mods.symptomatic or mods.pulp_vital is False:
        remove(T.EXTERNAL_REPAIR, "R2: pulpal involvement or symptoms -> repair without RCT not permitted.")
        if mods.pulp_vital is False:
            remove(T.EXTERNAL_REPAIR_VPT, "R2b: non-vital pulp -> vital pulp therapy not permitted.")
    # R3. Persistent haemorrhage after 2x CaOH2 contraindicates internal repair (Talpoş-Niculescu 2025)
    if mods.persistent_bleeding:
        demote(T.INTERNAL_REPAIR_RCT, "R3: persistent intracanal haemorrhage -> internal repair unlikely to seal; escalate.")
        if mods.multirooted:
            promote(T.HEMISECTION, "R3b: multirooted tooth with localised root lesion -> root resection/hemisection preserves function.")
        else:
            promote(T.EXTRACTION if not mods.growing_patient else T.DECORONATION,
                    "R3c: single-rooted, uncontrollable bleeding + perforation -> extraction (decoronation if growing).")
    # R4. Multirooted + lesion confined to one root -> hemisection before extraction
    if mods.multirooted and mods.lesion_on_root and T.EXTRACTION in ranking:
        idx = ranking.index(T.EXTRACTION)
        if T.HEMISECTION not in ranking[:idx]:
            ranking = [r for r in ranking if r != T.HEMISECTION]
            ranking.insert(idx, T.HEMISECTION)
            rationale.append("R4: multirooted tooth, lesion on a single root -> hemisection ranked above extraction.")
    # R5. Non-restorable or inadequate periodontal support -> no repair; monitor vs remove
    if mods.restorable is False or mods.perio_support_adequate is False:
        for t in (T.EXTERNAL_REPAIR, T.EXTERNAL_REPAIR_VPT, T.EXTERNAL_REPAIR_RCT, T.INTERNAL_REPAIR_RCT, T.INTENTIONAL_REPLANT):
            remove(t, f"R5: {'non-restorable' if mods.restorable is False else 'inadequate periodontal support'} -> repair/replantation not viable.")
        if not ranking:
            ranking = [T.MONITOR, T.EXTRACTION]
        flags.append("untreatable")
    # R6. Untreatable + asymptomatic -> patient may elect monitoring (Patel 2018 pt 2)
    if "untreatable" in flags and mods.symptomatic is False and T.MONITOR in ranking:
        promote(T.MONITOR, "R6: untreatable but asymptomatic -> periodic (annual) review is a legitimate first option.")
    # R7. Untreatable + symptomatic -> extraction / decoronation first
    if "untreatable" in flags and mods.symptomatic:
        promote(T.DECORONATION if mods.growing_patient else T.EXTRACTION,
                "R7: untreatable and symptomatic -> remove tooth (decoronation preserves ridge in growing patients).")
    # R8. Growing patient: decoronation preferred over extraction whenever extraction is ranked
    if mods.growing_patient and T.EXTRACTION in ranking:
        idx = ranking.index(T.EXTRACTION)
        ranking = [r for r in ranking if r != T.DECORONATION]
        ranking.insert(idx, T.DECORONATION)
        rationale.append("R8: growing patient -> decoronation ranked above extraction (ridge preservation).")
    # R9. Repair-phase lesion (bone-like calcification, asymptomatic) -> monitoring is reasonable
    if mods.repair_phase and mods.symptomatic is not True and T.MONITOR in ranking and height >= 3:
        promote(T.MONITOR, "R9: lesion shows reparative calcification and no symptoms -> monitor before irreversible treatment.")
    # R10. Intentional replantation contraindications
    if mods.root_fracture_risk_high or mods.perio_support_adequate is False:
        remove(T.INTENTIONAL_REPLANT, "R10: fracture risk / periodontal disease -> intentional replantation contraindicated.")
    # R11. End-stage escalation hint
    if code in END_STAGE_CODES:
        flags.append("end_stage")
        prior_adj = prior
    else:
        prior_adj = prior
    if mods.symptomatic:
        prior_adj *= 0.9
    if mods.repair_phase:
        prior_adj = min(1.0, prior_adj * 1.1)

    if not ranking:
        ranking = [T.MONITOR]
    primary_final = ranking[0]
    # Clinical inputs that materially change the ranking but were not supplied
    required = ["symptomatic", "pulp_vital", "restorable", "perio_support_adequate", "surgically_accessible"]
    if height >= 3 or spread in "CD":
        required += ["growing_patient", "multirooted"]
    missing = [k for k in required if getattr(mods, k) is None]
    if missing:
        flags.append("incomplete_clinical_inputs")
        rationale.append("Unknown clinical inputs: " + ", ".join(missing) +
                         " -> ranking is provisional; enter them before acting.")
    return Recommendation(
        patel_code=code, primary=primary_final, alternatives=ranking[1:],
        safe_set=list(ranking), prognosis_prior=float(prior_adj), evidence=evidence,
        rationale=rationale, flags=flags,
        requires_confirmation=primary_final.irreversible or bool(missing),
        missing_inputs=missing,
    )


def manual_patel_code(height: Optional[int], spread: Optional[str], proximity: Optional[str]) -> Optional[str]:
    """Build a Patel code from clinician-reviewed components; return None when
    any component is unknown so that no code (and no recommendation) is issued
    from incomplete observations."""
    if height is None or spread is None or proximity is None:
        return None
    if int(height) not in (1, 2, 3, 4):
        raise ValueError("height must be 1-4")
    spread = str(spread).upper(); proximity = str(proximity).lower()
    if spread not in "ABCD" or len(spread) != 1:
        raise ValueError("spread must be A, B, C or D")
    if proximity not in ("d", "p"):
        raise ValueError("proximity must be 'd' or 'p'")
    return f"{int(height)}{spread}{proximity}"


# --------------------------------------------------------------------------- #
# End-stage lesion management matrix
# --------------------------------------------------------------------------- #
def end_stage_matrix() -> List[Dict[str, str]]:
    """Decision matrix for end-stage lesions (height 3-4 with C/D spread, 2D, 4A/B).

    Rows are (symptom status x restorability/perio x patient factor) and the
    cell lists the ordered options. This is the table a clinician sees when
    the Patel code falls in END_STAGE_CODES.
    """
    rows = []
    def add(sym, struct, patient, first, second, third, note):
        rows.append({"symptoms": sym, "structure_perio": struct, "patient": patient,
                     "first_line": first, "second_line": second, "third_line": third, "note": note})

    add("asymptomatic", "restorable, perio adequate", "adult",
        "internal_repair_rct (3Cp/3Dp/2Dp) or periodic_review (d-lesions)",
        "intentional_replantation (4A/B, accessible root form)", "extraction",
        "ESE 2023 lists internal repair for 2Dp/3Cp/3Dp and periodic review for 2-4C/Dp.")
    add("asymptomatic", "restorable, perio adequate", "growing (<18 y)",
        "periodic_review (annual CBCT-free clinical review, CBCT on change)",
        "internal_repair_rct", "decoronation",
        "Delay irreversible treatment; decoronation preserves alveolar ridge for later implant.")
    add("asymptomatic", "non-restorable or perio compromised", "adult",
        "periodic_review (patient-elected, informed of fracture/discoloration/pain risk)",
        "extraction + replacement (implant / bridge / RBB)", "decoronation (if implant site preservation desired)",
        "Patel 2018 pt 2: untreatable + asymptomatic may be monitored until symptoms.")
    add("asymptomatic", "non-restorable or perio compromised", "growing (<18 y)",
        "periodic_review", "decoronation", "extraction + space management",
        "Multidisciplinary (ortho/prostho) planning per ESE 2023.")
    add("symptomatic (pulpitis / AP / pain)", "restorable, perio adequate", "adult",
        "internal_repair_rct if bleeding controllable", "hemisection (multirooted, single-root lesion)",
        "extraction + replacement",
        "Talpoş-Niculescu 2025: persistent bleeding after 2x CaOH2 -> abandon internal repair.")
    add("symptomatic (pulpitis / AP / pain)", "restorable, perio adequate", "growing (<18 y)",
        "internal_repair_rct", "decoronation", "extraction",
        "Avoid implants in growing patients; keep ridge.")
    add("symptomatic (pulpitis / AP / pain)", "non-restorable or perio compromised", "adult",
        "extraction (atraumatic / surgical, fracture risk high)", "decoronation (if ridge preservation wanted)",
        "-", "Patel 2018 pt 2: extensive ECR weakens cervical region -> plan surgical extraction.")
    add("symptomatic (pulpitis / AP / pain)", "non-restorable or perio compromised", "growing (<18 y)",
        "decoronation", "extraction + space management", "-",
        "Decoronation retains root for ridge development until growth completes.")
    add("any", "multirooted, lesion confined to one root", "any",
        "root_resection_hemisection + RCT of remaining roots", "extraction", "-",
        "Talpoş-Niculescu 2025 case 2 (3Bp on DB root of #17) managed by root amputation.")
    add("any", "repair-phase lesion (bone-like calcification, no symptoms)", "any",
        "periodic_review", "-", "-",
        "Mavridou: bone ingrowth through portal of entry + no symptoms -> monitor.")
    return rows
