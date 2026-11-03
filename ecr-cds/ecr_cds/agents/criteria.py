"""Radiological criteria that map visual descriptors to a differential diagnosis.

Each criterion is a predicate on `CandidateDescriptors` plus a vector of
log-likelihood-ratio *weights* per hypothesis.  The weights are expert-set
engineering priors derived from the descriptive literature (Patel et al. 2018
IEJ 51:1205, ESE 2023 position statement, Mavridou et al. 2016, DMFR 2023
adolescent mimics study) and are NOT calibrated on labelled clinical data.
`RadiologistAgent.update_weights_from_feedback` can refine them once enough
clinician-confirmed cases exist; the report always states which version is in
use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .visual_features import CandidateDescriptors

HYPOTHESES = ["ECR", "internal_resorption", "caries", "noncarious_cervical_lesion", "restoration_margin_or_burnout",
              "anatomical_concavity_or_furcation", "imaging_artifact"]

# prior over hypotheses for a hypodense root defect that reached the screen
PRIOR = {"ECR": 0.20, "internal_resorption": 0.08, "caries": 0.20, "noncarious_cervical_lesion": 0.16,
         "restoration_margin_or_burnout": 0.12, "anatomical_concavity_or_furcation": 0.12,
         "imaging_artifact": 0.12}


@dataclass
class Criterion:
    name: str
    description: str
    source: str
    predicate: Callable[[CandidateDescriptors], Optional[bool]]   # None = not assessable
    weights: Dict[str, float]          # log-LR contribution per hypothesis when predicate is True
    weights_if_false: Dict[str, float] = field(default_factory=dict)


def _w(**kw) -> Dict[str, float]:
    return {h: float(kw.get(h, 0.0)) for h in HYPOTHESES}


CRITERIA: List[Criterion] = [
    Criterion("portal_in_cervical_band",
              "External surface connection within the cervical band (ECR portal of entry is at/just apical to the CEJ)",
              "Patel 2018 IEJ 51:1205 (ECR part 1); ESE 2023 position statement",
              lambda d: d.cervical_surface_connection,
              _w(ECR=1.4, caries=-0.4, noncarious_cervical_lesion=1.0, restoration_margin_or_burnout=0.6,
                 anatomical_concavity_or_furcation=0.3),
              _w(ECR=-1.5, caries=0.4, noncarious_cervical_lesion=-0.6, anatomical_concavity_or_furcation=0.4)),
    Criterion("extends_subcrestal",
              "Most of the defect lies apical to the alveolar crest (ECR spreads within root dentine under bone; caries and NCCL are exposed-surface lesions)",
              "Patel 2018 (height levels 2-4 are subcrestal); DMFR 2023 mimics study",
              lambda d: None if (d.apical_extent_from_cej_mm is None or d.bone_crest_from_cej_mm is None)
              else d.apical_extent_from_cej_mm > d.bone_crest_from_cej_mm + 0.5,
              _w(ECR=1.0, caries=-1.0, noncarious_cervical_lesion=-0.8, anatomical_concavity_or_furcation=0.3),
              _w(ECR=-0.3, caries=0.5, noncarious_cervical_lesion=0.5)),
    Criterion("entirely_coronal_to_cej",
              "Defect lies entirely coronal to the CEJ (enamel/crown lesion rather than a cervical root lesion)",
              "Patel 2018; DMFR 2023 (caries mimics ECR)",
              lambda d: None if d.apical_extent_from_cej_mm is None else d.apical_extent_from_cej_mm < -0.5,
              _w(ECR=-1.2, caries=1.4, restoration_margin_or_burnout=0.5, noncarious_cervical_lesion=-0.3)),
    Criterion("small_portal_wide_interior",
              "Narrow portal of entry with wider internal spread (undermining ratio > 1.5): hallmark of ECR; open wedge/saucer defects are the opposite",
              "Patel 2018 part 1; Mavridou 2016 JOE 42:1737",
              lambda d: d.undermining_ratio > 1.5 if d.portal_area_mm2 > 0 else None,
              _w(ECR=1.5, caries=0.2, noncarious_cervical_lesion=-1.5, anatomical_concavity_or_furcation=-1.5,
                 restoration_margin_or_burnout=-0.5, imaging_artifact=-0.5),
              _w(ECR=-0.6, noncarious_cervical_lesion=0.6, anatomical_concavity_or_furcation=0.5)),
    Criterion("canal_outline_intact",
              "Dentine separates the defect from the canal (pericanalar resorption-resistant sheet); the canal outline is traceable through the lesion",
              "Patel 2018 part 1; Mavridou 2016",
              lambda d: d.canal_wall_intact,
              _w(ECR=0.6),
              _w(ECR=0.2, caries=0.2)),   # perforation is late-stage ECR or deep caries; weakly informative
    Criterion("irregular_margins",
              "Irregular, ill-defined margins (ECR) versus smooth, well-defined contour (NCCL, developmental concavity)",
              "Patel 2018 part 1; DMFR 2023",
              lambda d: None if (d.max_depth_mm < 0.6 or d.margin_irregularity > 0.8) else d.margin_irregularity > 0.25,
              _w(ECR=0.7, noncarious_cervical_lesion=-0.7, anatomical_concavity_or_furcation=-0.7, caries=0.2),
              _w(ECR=-0.3, noncarious_cervical_lesion=0.5, anatomical_concavity_or_furcation=0.5)),
    Criterion("mottled_radiopaque_inclusions",
              "Mottled appearance / radiopaque inclusions inside the defect (fibro-osseous repair tissue of the reparative phase)",
              "Patel 2018 part 1 (radiographic presentation); Mavridou 2016",
              lambda d: d.radiopaque_inclusion_fraction > 0.08,
              _w(ECR=1.0, caries=-0.6, noncarious_cervical_lesion=-0.6, anatomical_concavity_or_furcation=-0.4)),
    Criterion("circumferential_spread_over_90deg",
              "Defect wraps more than 90 degrees around the root (ECR spreads circumferentially; NCCL is usually a single buccal facet)",
              "Patel 2018 (spread A-D)",
              lambda d: d.angular_spread_deg > 90,
              _w(ECR=0.5, noncarious_cervical_lesion=-0.4, anatomical_concavity_or_furcation=-0.2)),
    Criterion("restoration_or_enamel_contact",
              "Defect is in contact with hyperdense material (restoration margin, enamel) - cervical burnout and marginal gaps mimic ECR",
              "DMFR 2023 mimics study",
              lambda d: d.dense_material_contact_fraction > 0.10,
              _w(ECR=-0.3, caries=0.5, restoration_margin_or_burnout=1.5)),
    Criterion("near_metal",
              "Defect within 3 mm of metal-level voxels (beam-hardening / streak artefact)",
              "ESE 2023 (artefact caveat for CBCT interpretation)",
              lambda d: d.near_metal_fraction > 0.3,
              _w(imaging_artifact=2.0, ECR=-0.5, caries=-0.5, noncarious_cervical_lesion=-0.5,
                 restoration_margin_or_burnout=0.3, anatomical_concavity_or_furcation=-0.5)),
    Criterion("threshold_unstable",
              "Candidate shrinks markedly or merges when the intensity threshold is perturbed by 10 % (noise/partial-volume effect)",
              "engineering criterion (threshold-sensitivity test in ecr_detection)",
              lambda d: d.stable_core_fraction < 0.5 or d.threshold_expansion_ratio > 2.0,
              _w(imaging_artifact=0.8, ECR=-0.3)),
    Criterion("shallow_elongated_groove",
              "Shallow (< 0.5 mm), axially elongated surface depression: developmental root groove / furcation concavity",
              "DMFR 2023 (root concavity mimics ECR)",
              lambda d: d.max_depth_mm < 0.5 and d.elongation_axial > 3.0,
              _w(anatomical_concavity_or_furcation=1.5, ECR=-0.8, noncarious_cervical_lesion=0.2)),
    Criterion("deep_root_location",
              "Defect begins more than 4 mm apical to the CEJ (coronal extent) - root caries and NCCL are confined to the exposed cervical surface; deep root defects favour anatomy, periodontal-related resorption or artefact",
              "Patel 2018 (ECR originates at the cervical portal); DMFR 2023",
              lambda d: None if d.coronal_extent_from_cej_mm is None else d.coronal_extent_from_cej_mm > 4.0,
              _w(caries=-1.5, noncarious_cervical_lesion=-1.2, restoration_margin_or_burnout=-1.0,
                 anatomical_concavity_or_furcation=0.6, ECR=-0.4, imaging_artifact=0.3)),
    Criterion("volume_over_2mm3",
              "Defect volume above 2 mm3 (sub-millimetre findings are below the reliable detection size on CBCT)",
              "Al Fartousi 2026 (low sensitivity for sub-millimetre lesions)",
              lambda d: d.volume_mm3 > 2.0,
              _w(ECR=0.3, caries=0.3, noncarious_cervical_lesion=0.2),
              _w(imaging_artifact=0.6, ECR=-0.3)),
    # ---- cue-level criteria for the diagnostic questions (rev 2.5) ----
    Criterion("external_contour_interrupted",
              "The external root contour is interrupted on most axial levels spanned by the lesion (defect traceable from the root surface inward)",
              "ESE 2023 position statement; Patel 2018 part 1",
              lambda d: None if d.n_axial_levels == 0 else (d.external_surface_connection and (d.portal_slice_persistence >= 0.1 or d.portal_area_mm2 >= 0.5)),
              _w(ECR=1.0, caries=1.0, noncarious_cervical_lesion=1.0, restoration_margin_or_burnout=0.6, internal_resorption=-2.0),
              _w(internal_resorption=1.2, ECR=-0.8, caries=-0.8, noncarious_cervical_lesion=-0.8)),
    Criterion("eccentric_to_canal",
              "Lesion is eccentric to the root canal (centroid offset > 25 % of the root radius) rather than centred on it",
              "ESE 2023 (external vs internal resorption); AAE CDT guide 2024",
              lambda d: None if d.eccentricity_ratio is None else d.eccentricity_ratio > 0.25,
              _w(ECR=0.8, caries=0.8, noncarious_cervical_lesion=0.8, restoration_margin_or_burnout=0.5,
                 anatomical_concavity_or_furcation=0.5, internal_resorption=-1.5),
              _w(internal_resorption=1.5, ECR=-0.8, caries=-0.8, noncarious_cervical_lesion=-0.8)),
    Criterion("canal_traceable_through_lesion",
              "The canal remains identifiable on most levels the lesion spans (lesion passes beside or around it)",
              "ESE 2023; Patel 2018 part 1 (pericanalar resorption-resistant sheet)",
              lambda d: None if d.canal_traceable_fraction is None else d.canal_traceable_fraction >= 0.7,
              _w(ECR=0.5, caries=0.5, noncarious_cervical_lesion=0.5, restoration_margin_or_burnout=0.5,
                 anatomical_concavity_or_furcation=0.5, internal_resorption=-1.2),
              _w(internal_resorption=1.0, ECR=-0.4, caries=-0.4)),
    Criterion("canal_balloons_into_lesion",
              "Canal space expands into the lesion (central canal-plus-lesion cross-section > 1.5 x the canal just outside the span)",
              "ESE 2023 (internal inflammatory resorption appearance)",
              lambda d: None if d.canal_expansion_ratio is None else d.canal_expansion_ratio > 1.5,
              _w(internal_resorption=1.8, ECR=-0.8, caries=-0.3),
              _w(internal_resorption=-0.8)),
    Criterion("enters_one_side_and_wraps",
              "Lesion enters from a narrow angular sector and wraps around the canal (angular spread > 2 x portal width and > 90 degrees)",
              "Patel 2018 (circumferential spread); ESE 2023",
              lambda d: None if d.portal_angular_width_deg == 0 else (d.wrap_ratio > 2.0 and d.angular_spread_deg > 90),
              _w(ECR=1.2, noncarious_cervical_lesion=-0.8, caries=-0.4, anatomical_concavity_or_furcation=-0.8, internal_resorption=-0.3)),
    Criterion("asymmetric_infiltrative",
              "Asymmetric, infiltrative loss of dentine (angular mass on one side with rough margins) rather than a smooth symmetric cavity",
              "ESE 2023; Patel 2018 part 1",
              lambda d: None if d.max_depth_mm < 0.6 else (d.asymmetry_index > 0.5 and d.margin_irregularity > 0.25),
              _w(ECR=0.4, caries=0.3, internal_resorption=-0.6, noncarious_cervical_lesion=-0.4)),
    Criterion("thin_surface_shell",
              "Thin surface shell (< 0.5 mm deep, interior no wider than the opening): partial-volume / noise skin on the root surface, not a lesion",
              "multi-planar verification (ESE 2023 CBCT interpretation); DMFR 2023",
              lambda d: None if not d.external_surface_connection else (d.max_depth_mm < 0.5 and d.undermining_ratio < 0.2),
              _w(imaging_artifact=1.5, restoration_margin_or_burnout=0.4, ECR=-1.2, caries=-0.6, noncarious_cervical_lesion=-0.3,
                 anatomical_concavity_or_furcation=0.3)),
    Criterion("margin_noise_dominated",
              "Margin roughness above 0.8 (the whole boundary is irregular at 0.3 mm): noise or partial-volume dominated contour, not an infiltrative lesion margin",
              "multi-planar verification (ESE 2023 CBCT interpretation)",
              lambda d: d.margin_irregularity > 0.8,
              _w(imaging_artifact=1.0, ECR=-0.5, caries=-0.3)),
    Criterion("shallow_open_saucer",
              "Shallow (< 1 mm), open, non-undermining, supracrestal surface defect: the classic non-carious cervical lesion / abrasion shape",
              "DMFR 2023; ESE 2023 (differential diagnosis)",
              lambda d: None if not d.external_surface_connection else
              (d.max_depth_mm < 1.0 and d.undermining_ratio < 0.7 and (d.fraction_subcrestal or 0) < 0.2),
              _w(noncarious_cervical_lesion=1.2, caries=0.3, ECR=-0.9, internal_resorption=-0.5),
              _w(noncarious_cervical_lesion=-0.5, ECR=0.2)),
    Criterion("persists_in_three_planes",
              "Defect persists across >= 3 consecutive axial levels and >= 0.3 mm in both in-plane directions (true 3-D defect, not a single-slice artefact)",
              "multi-planar verification (ESE 2023 CBCT interpretation)",
              lambda d: d.persistent_3d,
              _w(ECR=0.3, caries=0.3, noncarious_cervical_lesion=0.3, internal_resorption=0.3, imaging_artifact=-1.0),
              _w(imaging_artifact=2.0, ECR=-1.0, caries=-0.8, noncarious_cervical_lesion=-0.8, internal_resorption=-0.8)),
]


# --------------------------------------------------------------------------- #
# The eight diagnostic questions a clinician works through, answered from the descriptors.
# Each returns (answer, evidence-string); answer in {"yes", "no", "uncertain", "not_assessable"}.
def _q1_external_origin(d):
    if not d.external_surface_connection:
        return "no", "no connection to the external root surface; lesion is internal to the dentine/canal"
    bits = [f"external contour interrupted on {d.portal_slice_persistence:.0%} of lesion levels"]
    if d.cervical_surface_connection is True:
        bits.append(f"surface breach inside the cervical band (CEJ {d.landmark_source})")
    elif d.cervical_surface_connection is False:
        bits.append("surface breach outside the estimated cervical band")
    if d.eccentricity_ratio is not None:
        bits.append(f"eccentricity {d.eccentricity_ratio:.2f} of root radius")
    eccentric = d.eccentricity_ratio is None or d.eccentricity_ratio > 0.25
    traceable = d.portal_slice_persistence >= 0.1 or d.portal_area_mm2 >= 0.5
    cerv_ok = d.cervical_surface_connection is not False
    if eccentric and traceable and cerv_ok:
        return "yes", "; ".join(bits) + " -> defect traceable from the root surface inward, eccentric to the canal"
    if eccentric and traceable:
        return "uncertain", "; ".join(bits) + " -> external and eccentric, but outside the estimated cervical band (check CEJ)"
    return "no", "; ".join(bits) + " -> not convincingly arising from the external root surface"


def _q2_independent_of_canal(d):
    if d.canal_traceable_fraction is None:
        return "not_assessable", "no canal mask / proxy"
    bits = [f"canal identifiable on {d.canal_traceable_fraction:.0%} of lesion levels"]
    if d.canal_expansion_ratio is not None:
        bits.append(f"canal-space expansion ratio {d.canal_expansion_ratio:.2f}")
    if d.canal_traceable_fraction >= 0.7 and (d.canal_expansion_ratio is None or d.canal_expansion_ratio <= 1.5):
        return "yes", "; ".join(bits) + " -> lesion passes beside/around an identifiable canal"
    if (d.canal_expansion_ratio is not None and d.canal_expansion_ratio > 1.5) or \
            (d.canal_traceable_fraction < 0.3 and not d.external_surface_connection):
        return "no", "; ".join(bits) + " -> canal balloons into / is replaced by the lesion (internal pattern)"
    return "uncertain", "; ".join(bits)


def _q3_growth_pattern(d):
    bits = [f"angular spread {d.angular_spread_deg:.0f} deg", f"portal width {d.portal_angular_width_deg:.0f} deg",
            f"undermining ratio {d.undermining_ratio:.2f}"]
    if d.wrap_ratio > 2.0 and d.angular_spread_deg > 90:
        return "yes", "; ".join(bits) + " -> enters from one side and wraps around the canal"
    if d.undermining_ratio > 1.5 or d.angular_spread_deg > 90:
        return "uncertain", "; ".join(bits) + " -> lateral spread present but entry not clearly narrow"
    return "no", "; ".join(bits) + " -> open defect without lateral/circumferential spread"


def _q4_preserved_dentine(d):
    if d.min_canal_distance_mm is None:
        return "not_assessable", "no canal mask / proxy"
    bits = [f"minimum lesion-canal distance {d.min_canal_distance_mm:.2f} mm"]
    if d.canal_wall_intact and d.angular_spread_deg > 180:
        return "yes", "; ".join(bits) + f" with {d.angular_spread_deg:.0f} deg spread -> lesion around a preserved canal wall"
    if d.canal_wall_intact:
        return "uncertain", "; ".join(bits) + " -> dentine preserved, lesion does not yet wrap the canal"
    return "no", "; ".join(bits) + " -> lesion reaches the canal (probable pulpal involvement / perforation)"


def _q5_resorption_not_caries(d):
    bits = [f"margin roughness {d.margin_irregularity:.2f}", f"asymmetry {d.asymmetry_index:.2f}",
            f"radiopaque inclusions {d.radiopaque_inclusion_fraction:.0%}",
            f"dense-material contact {d.dense_material_contact_fraction:.0%}"]
    if d.apical_extent_from_cej_mm is not None:
        bits.append(f"apical extent {d.apical_extent_from_cej_mm:.1f} mm from CEJ")
    score = 0
    score += d.undermining_ratio > 1.5
    score += d.margin_irregularity > 0.15 and d.max_depth_mm >= 0.6
    score += d.radiopaque_inclusion_fraction > 0.08
    score += (d.apical_extent_from_cej_mm or 0) > 0.5
    score += d.angular_spread_deg > 90
    score -= d.dense_material_contact_fraction > 0.10
    score -= (d.apical_extent_from_cej_mm is not None and d.apical_extent_from_cej_mm < -0.5)
    return ("yes" if score >= 3 else "uncertain" if score >= 1 else "no"), "; ".join(bits) + f" -> resorption-pattern score {score}/5"


def _q6_multiplanar(d):
    bits = [f"{d.n_axial_levels} axial levels", f"{d.extent_y_mm:.1f} x {d.extent_x_mm:.1f} mm in-plane",
            f"stable core {d.stable_core_fraction:.2f}", f"near-metal fraction {d.near_metal_fraction:.0%}"]
    if d.persistent_3d and d.stable_core_fraction >= 0.5 and d.near_metal_fraction <= 0.3:
        return "yes", "; ".join(bits) + " -> consistent 3-D defect"
    if d.persistent_3d:
        return "uncertain", "; ".join(bits) + " -> 3-D defect but threshold-sensitive or near metal"
    return "no", "; ".join(bits) + " -> not a persistent 3-D defect"


def _q7_extent(d):
    bits = []
    if d.apical_extent_from_cej_mm is not None:
        bits.append(f"apical extent {d.apical_extent_from_cej_mm:.1f} mm from CEJ")
    if d.bone_crest_from_cej_mm is not None:
        bits.append(f"bone crest {d.bone_crest_from_cej_mm:.1f} mm from CEJ")
    bits.append(f"spread {d.angular_spread_deg:.0f} deg")
    if d.min_canal_distance_mm is not None:
        bits.append("canal reached" if not d.canal_wall_intact else f"{d.min_canal_distance_mm:.2f} mm dentine to canal")
    return "measured", "; ".join(bits) + " -> full extent summary measured by the feature-measurement / extent-summary agents"


DIAGNOSTIC_QUESTIONS = [
    ("Q1", "Is the abnormality arising from the external cervical root?", _q1_external_origin),
    ("Q2", "Does the lesion behave independently of the root canal (external, not internal, resorption)?", _q2_independent_of_canal),
    ("Q3", "Does it show the ECR growth pattern (enters from one side, spreads laterally/circumferentially)?", _q3_growth_pattern),
    ("Q4", "Is there preserved dentine around the canal while the lesion spreads around it?", _q4_preserved_dentine),
    ("Q5", "Does the morphology look like resorption rather than caries / a smooth cavity?", _q5_resorption_not_caries),
    ("Q6", "Does the lesion persist in all three planes as a true 3-D defect?", _q6_multiplanar),
    ("Q7", "How extensive is it (apical reach / circumferential extent / pulp relationship, measured)?", _q7_extent),
]


def answer_questions(d) -> List[Dict[str, str]]:
    out = []
    for qid, text, fn in DIAGNOSTIC_QUESTIONS:
        try:
            ans, ev = fn(d)
        except Exception as e:  # never let a descriptor edge case abort the conference
            ans, ev = "not_assessable", f"error: {e}"
        out.append({"id": qid, "question": text, "answer": ans, "evidence": ev})
    return out


def default_weight_matrix() -> np.ndarray:
    """(n_criteria, 2, n_hypotheses) weights for predicate True / False."""
    W = np.zeros((len(CRITERIA), 2, len(HYPOTHESES)))
    for i, c in enumerate(CRITERIA):
        W[i, 0] = [c.weights.get(h, 0.0) for h in HYPOTHESES]
        W[i, 1] = [c.weights_if_false.get(h, 0.0) for h in HYPOTHESES]
    return W


def evaluate_criteria(d: CandidateDescriptors) -> List[Optional[bool]]:
    out = []
    for c in CRITERIA:
        try:
            v = c.predicate(d)
        except Exception:
            v = None
        out.append(None if v is None else bool(v))
    return out


# Criteria are correlated (several describe the same morphology); their summed log-LRs are tempered so a
# single mechanism cannot drive the posterior to 1.0.  Engineering constant, not calibrated.
TEMPERATURE = 1.6


def posterior_from_criteria(truth: List[Optional[bool]], W: Optional[np.ndarray] = None,
                            prior: Optional[Dict[str, float]] = None, temperature: float = TEMPERATURE) -> Dict[str, float]:
    W = default_weight_matrix() if W is None else W
    prior = prior or PRIOR
    logp = np.log(np.array([prior[h] for h in HYPOTHESES]))
    acc = np.zeros(len(HYPOTHESES))
    for i, v in enumerate(truth):
        if v is None:
            continue
        acc += W[i, 0 if v else 1]
    logp += acc / max(temperature, 1e-6)
    p = np.exp(logp - logp.max())
    p /= p.sum()
    return {h: round(float(x), 4) for h, x in zip(HYPOTHESES, p)}


def confidence_band(p: float, capped: bool) -> str:
    """Qualitative band for an *uncalibrated* posterior; `capped` keeps it below 'high'."""
    if p < 0.35:
        return "low"
    if p < 0.6 or capped:
        return "moderate"
    return "high"
