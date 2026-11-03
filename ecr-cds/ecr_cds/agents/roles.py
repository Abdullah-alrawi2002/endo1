"""Agent roles of the ECR case conference.

ImagingQAAgent      -> is the scan fit for cervical-resorption assessment?
AnatomyAgent        -> tooth / canal / landmarks and their reliability
LesionScreeningAgent-> candidate defects + quantitative visual descriptors + tiles
RadiologistAgent    -> per-candidate differential diagnosis (criteria + optional LLM/VLM)
SkepticAgent        -> adversarial review: artefacts, landmark errors, over-reading
FeatureMeasurementAgent (feature_measurement.py, rev 2.9)
                    -> per-candidate VLM visual features + voxel-geometry measurements (feature bundle)
ExtentSummaryAgent  (feature_measurement.py, rev 2.9)
                    -> descriptive, measurement-based extent / severity summary for the accepted candidate
ClassifierAgent     -> LEGACY (staging_scheme="patel_3d" only): Patel 3D code; not used by the default path
TreatmentPlannerAgent -> safe set + modifiers + constrained RL policy keyed off the extent summary (default)
                       or the legacy Patel matrix, with contingencies
ModeratorAgent      -> runs the debate rounds and writes the consensus

Every agent appends `Evidence` to the shared blackboard; none can finalise a
diagnosis.  The consensus is a *suggestion* for the treating clinician.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from .blackboard import Blackboard, Evidence
from .criteria import (CRITERIA, HYPOTHESES, PRIOR, answer_questions, confidence_band, default_weight_matrix,
                       evaluate_criteria, posterior_from_criteria)
from .clinical import ClinicalFindings, correlate
from .llm_backend import LLMBackend, NullBackend
from .visual_features import CandidateDescriptors, describe_candidates, render_candidate_tiles
from .feature_measurement import bundle_feature_notes
from ..decision_matrix import (ClinicalModifiers, END_STAGE_CODES, SCHEME_MEASUREMENT, Treatment, end_stage_matrix,
                               recommend_from_matrix, recommend_from_measurements)
from ..ecr_detection import CervicalResorptionDetector
from ..patel_classifier import classify_patel
from ..preprocess import preprocess_volume, crop_to_bbox
from ..rl_policy import ClinicalContext, ConstrainedBanditPolicy
from ..segmentation import ToothSegmentation


class Agent:
    name = "agent"

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.backend = backend or NullBackend()

    def _llm(self, bb: Blackboard, system: str, user: str, schema: Dict, images=None) -> Optional[Dict]:
        if isinstance(self.backend, NullBackend):
            return None
        out = self.backend.complete_json(system, user, schema, images_png=images)
        if out is None:
            return None
        if "_error" in out:
            bb.arrays["backend_failures"] = bb.arrays.get("backend_failures", 0) + 1
            bb.say(self.name, f"model backend failed ({out['_error'][:120]}); using rule result only")
            return None
        bb.arrays["backend_calls_ok"] = bb.arrays.get("backend_calls_ok", 0) + 1
        return out


# --------------------------------------------------------------------------- #
class ImagingQAAgent(Agent):
    name = "imaging_qa"

    def run(self, bb: Blackboard, volume, qa: Dict) -> Evidence:
        sp = np.asarray(volume.spacing, float)
        fov = np.asarray(volume.shape) * sp
        issues = list(qa.get("issues", []))
        caveats = []
        if sp.max() > 0.2:
            caveats.append("voxel_size_over_0.2mm_limits_early_ecr_detection")
        if fov.max() > 80:
            caveats.append("large_fov_scan_not_optimised_for_endodontic_detail")
        metal_frac = float((volume.array >= np.percentile(volume.array, 99.9)) .mean())
        hi = float(np.percentile(volume.array, 99.99))
        if hi > 8000:
            caveats.append("hyperdense_material_in_fov_streak_artefacts_possible")
        fit = "adequate" if not issues and sp.max() <= 0.2 else "limited"
        ev = Evidence(self.name, "qa", {"fit_for_purpose": fit, "voxel_mm": sp.round(3).tolist(),
                                        "fov_mm": fov.round(1).tolist(), "qa_issues": issues,
                                        "p99_99_intensity": hi, "dense_fraction": round(metal_frac, 5),
                                        "reference": "ESE 2023: small-FOV high-resolution CBCT recommended for ECR"},
                      confidence=None, basis="measurement", caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"Scan {fit} for ECR assessment: voxel {sp.max():.2f} mm, FOV {fov.max():.0f} mm"
                          + (f"; caveats: {', '.join(caveats)}" if caveats else ""))
        return ev


# --------------------------------------------------------------------------- #
class AnatomyAgent(Agent):
    name = "anatomy"

    def run(self, bb: Blackboard, volume, tooth_segmenter, seed_zyx, seed_source: str,
            auto=None, canal_source: str = "automatic", roi_margin_mm: float = 3.0) -> Evidence:
        seg_full: ToothSegmentation = tooth_segmenter(volume.array, volume.spacing)
        sl = crop_to_bbox(seg_full.tooth, margin=int(round(roi_margin_mm / min(volume.spacing))))
        raw = volume.array[sl].astype(np.float32)
        seg = ToothSegmentation(seg_full.tooth[sl], seg_full.canal[sl], seg_full.bone[sl], intensity=raw)
        bb.arrays.update(seg=seg, crop=sl, raw=raw, norm=preprocess_volume(raw, roi=seg.tooth))
        vox = float(np.prod(volume.spacing))
        dentine_mode = float(np.median(raw[seg.tooth & ~seg.canal])) if (seg.tooth & ~seg.canal).any() else float(np.median(raw[seg.tooth]))
        # without Otsu thresholds from auto_segment, use a conservative fraction of the dentine level
        t_tooth = auto.t_tooth if auto is not None else 0.85 * dentine_mode
        t_dense = auto.t_dense if auto is not None else float(np.percentile(raw[seg.tooth], 97))
        caveats = ["segmentation_automatic_unreviewed"] if "auto" in str(seed_source) or auto is not None else []
        canal_mode = getattr(tooth_segmenter, "canal_mode", canal_source)
        if "hyperdense" in str(canal_mode):
            caveats.append("root_filling_present_canal_is_filling_not_pulp")
        if "expert" not in str(canal_mode):
            caveats.append("canal_is_automatic_proxy")
        content = {"tooth_volume_mm3": round(float(seg.tooth.sum() * vox), 1),
                   "canal_volume_mm3": round(float(seg.canal.sum() * vox), 2),
                   "crop_bbox_zyx": [[int(s.start), int(s.stop)] for s in sl],
                   "seed_zyx": list(map(int, seed_zyx)), "seed_source": seed_source,
                   "canal_mode": str(canal_mode), "t_tooth": t_tooth, "t_dense": t_dense,
                   "dentine_mode": dentine_mode}
        ev = Evidence(self.name, "anatomy", content, basis="measurement", caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"Target tooth {content['tooth_volume_mm3']} mm3 at seed {content['seed_zyx']} "
                          f"({seed_source}); canal mode {canal_mode}.")
        return ev


# --------------------------------------------------------------------------- #
class LesionScreeningAgent(Agent):
    name = "lesion_screening"

    def run(self, bb: Blackboard, spacing, cej_zyx=None, apex_zyx=None, canal_source="automatic",
            render: bool = True) -> List[Evidence]:
        seg, raw, norm = bb.arrays["seg"], bb.arrays["raw"], bb.arrays["norm"]
        anat = bb.latest("anatomy").content
        det = CervicalResorptionDetector(anat["t_tooth"], anat["t_dense"], spacing,
                                         cej_zyx=cej_zyx, apex_zyx=apex_zyx, canal_source=canal_source)
        det(norm, seg)
        rep = det.last_report
        geometry = dict(det.last_geometry)
        geometry.update(source=rep["landmarks"]["source"], flags=rep["landmarks"]["flags"])
        # The screen only keeps externally connected cervical components. For the differential the conference
        # also examines large components rejected *only* for lacking a surface connection (possible internal
        # resorption), and large deep external components outside the estimated cervical band (CEJ may be wrong).
        labels = det.last_labels.copy()
        internal_only = {"no_external_surface_connection", "no_connection_in_estimated_cervical_band"}
        for c in rep["candidates"]:
            if c["disposition"] != "rejected_by_screen" or c["volume_mm3"] < 2.0:
                continue
            reasons = set(c["rejection_reasons"])
            if reasons and reasons <= internal_only:
                labels[det.last_all_labels == c["component_id"]] = c["component_id"]
                c["disposition"] = ("internal_component_for_differential" if "no_external_surface_connection" in reasons
                                    else "external_outside_estimated_cervical_band")
        bb.arrays.update(candidate_labels=labels, geometry=geometry, screen_report=rep)
        descs = describe_candidates(raw, seg, labels, spacing, geometry, rep["candidates"],
                                    anat["t_tooth"], anat["t_dense"], anat["dentine_mode"])
        bb.arrays["descriptors"] = {d.component_id: d for d in descs}
        tiles = {}
        if render:
            for d in descs:
                try:
                    tiles[d.component_id] = render_candidate_tiles(raw, seg, labels, d.component_id, spacing)
                except Exception:
                    pass
        bb.arrays["tiles"] = tiles
        evs = []
        bb.post(Evidence(self.name, "screen_summary",
                         {"n_components_screened": len(rep["candidates"]),
                          "n_review_candidates": rep["n_review_candidates"],
                          "landmarks": rep["landmarks"], "thresholds": rep["intensity_thresholds"],
                          "rejected": [{"component_id": c["component_id"], "volume_mm3": c["volume_mm3"],
                                        "reasons": c["rejection_reasons"]} for c in rep["candidates"]
                                       if c["disposition"] == "rejected_by_screen"]},
                         basis="measurement", caveats=list(rep["landmarks"]["flags"])))
        for d in descs:
            row = next(c for c in rep["candidates"] if c["component_id"] == d.component_id)
            ev = Evidence(self.name, "candidate", {**d.to_dict(), "screen_disposition": row["disposition"],
                                                   "screen_flags": row["flags"]},
                          basis="measurement", refers_to=d.component_id, caveats=list(row["flags"]))
            bb.post(ev)
            evs.append(ev)
        bb.say(self.name, f"{len(rep['candidates'])} hypodense components screened, {len(descs)} retained for the "
                          f"conference" + ("" if descs else "; a negative screen does not exclude ECR"))
        return evs

    def reconcile(self, bb: Blackboard) -> Optional[Evidence]:
        """Rev 2.9: compare each candidate's screen disposition with the feature bundle (geometry + VLM reading).

        The screen is never overruled: disagreements are posted as `screen_reconciliation` evidence for the skeptic
        and the moderator, and a candidate is never removed here.
        """
        bundles = bb.arrays.get("feature_bundles", {})
        if not bundles:
            return None
        rows = []
        for cid, b in bundles.items():
            bd = b.to_dict() if hasattr(b, "to_dict") else b
            cand = bb.latest("candidate", refers_to=cid)
            disp = (cand.content.get("screen_disposition") if cand else None) or "unknown"
            vis = bd.get("visual", {})
            ms = bd.get("measurements", {})
            notes = []
            reader = "local image reading" if (bd.get("vlm") or {}).get("status") == "local_fallback" else "model"
            comm = (ms.get("root_canal_communication") or {}).get("value")
            if disp == "internal_component_for_differential" and vis.get("assessed") and vis.get("portal_of_entry_visible") == "yes":
                notes.append(f"screen found no surface connection but the {reader} reports a visible portal")
            if disp not in ("internal_component_for_differential", "unknown") and vis.get("assessed") and vis.get("portal_of_entry_visible") == "no":
                notes.append(f"screen found a surface connection but the {reader} saw no portal of entry")
            if vis.get("assessed") and vis.get("metal_beam_hardening") == "marked":
                notes.append(f"{reader} flags marked metal / beam-hardening streaks around the candidate")
            if vis.get("assessed") and vis.get("burnout_artifact_suspected") == "yes":
                notes.append(f"{reader} suspects cervical burnout")
            if comm is True and vis.get("assessed") and vis.get("pericanal_radiopaque_line") == "intact":
                notes.append(f"geometry shows canal communication but the {reader} reads the pericanal line as intact")
            rows.append({"component_id": cid, "screen_disposition": disp, "vlm_status": bd.get("vlm", {}).get("status"),
                         "unverified_vlm_values": bd.get("unverified_vlm_values", []),
                         "vlm_geometry_disagreements": bd.get("vlm_geometry_disagreements", []), "notes": notes})
        ev = Evidence(self.name, "screen_reconciliation", {"candidates": rows,
                                                           "rule": "screen dispositions are never changed here; disagreements become challenges"},
                      basis="measurement", caveats=["reconciliation_is_informational"])
        bb.post(ev)
        n = sum(len(r["notes"]) for r in rows)
        bb.say(self.name, f"screen vs feature bundle: {n} disagreement note(s) across {len(rows)} candidate(s); dispositions unchanged")
        return ev


# --------------------------------------------------------------------------- #
RADIOLOGIST_SYSTEM = (
    "You are an endodontic radiologist reviewing quantitative descriptors (and, if provided, image tiles) of a "
    "hypodense root defect on CBCT, together with the answers to the seven diagnostic questions (external origin, "
    "independence from the canal, growth pattern, preserved pericanalar dentine, resorption-vs-caries morphology, "
    "multiplanar persistence, extent). Give a differential between external cervical resorption (ECR), internal "
    "resorption, caries, non-carious cervical lesion, restoration margin / cervical burnout, anatomical concavity or "
    "furcation, and imaging artefact. Internal resorption is canal-centred, balloons the canal and has no external "
    "defect initially. Use the Patel 2018 / ESE 2023 descriptions: ECR has a portal of entry at or just apical to "
    "the CEJ, spreads within dentine with a narrow portal and wider interior, irregular margins, may show mottled "
    "radiopaque repair tissue, and the canal outline is traceable through it until late perforation. Be explicit "
    "about uncertainty; probabilities must sum to 1. If an image is attached it is an 8-panel evidence frame: "
    "row 1 = three neighbouring axial slices (-0.8 mm, centre, +0.8 mm) and a coronal view; row 2 = sagittal, "
    "tooth long-axis reformat, perpendicular-to-axis reformat, and the unmodified full slice with the crop box. "
    "Tooth outline cyan, canal proxy yellow, candidate region red. Describe only what is visible in the panels; "
    "do not infer findings that are not in the image. Answer the diagnostic questions in question_answers with "
    "exactly yes / no / uncertain.")

RADIOLOGIST_SCHEMA = {
    "type": "object",
    "properties": {
        "differential": {"type": "array", "items": {"type": "object", "properties": {
            "diagnosis": {"type": "string", "enum": HYPOTHESES}, "probability": {"type": "number"}},
            "required": ["diagnosis", "probability"]}},
        "key_findings_for": {"type": "array", "items": {"type": "string"}},
        "key_findings_against": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "question_answers": {"type": "object", "description": "Q1..Q6 -> yes|no|uncertain (visual reading of the frame)",
                             "additionalProperties": {"type": "string", "enum": ["yes", "no", "uncertain"]}},
        "lesion_visible_in_image": {"type": "string", "enum": ["yes", "no", "uncertain", "no_image"]},
    },
    "required": ["differential"]}

LLM_WEIGHT_MAX = 0.5        # second opinion never outweighs the measured rule posterior
LLM_WEIGHT_WEAK_GROUNDING = 0.2

# rev 2.9: log-odds nudges from the model's *visual* feature reading (things geometry cannot see).  Each is small
# and the total is capped so the feature bundle can colour, but never drive, the rule posterior.
FEATURE_LOGODDS = {
    ("portal_of_entry_visible", "yes"): {"ECR": +0.3},
    ("pericanal_radiopaque_line", "intact"): {"ECR": +0.3, "internal_resorption": -0.3},
    ("pericanal_radiopaque_line", "interrupted"): {"internal_resorption": +0.2},
    ("fibro_osseous_appearance", "yes"): {"ECR": +0.3, "caries": -0.3},
    ("border_definition", "smooth"): {"ECR": -0.3, "noncarious_cervical_lesion": +0.3},
    ("relation_to_restoration_margin", "continuous_with_margin"): {"restoration_margin_or_burnout": +0.4, "ECR": -0.2},
    ("burnout_artifact_suspected", "yes"): {"restoration_margin_or_burnout": +0.4, "ECR": -0.2},
    ("metal_beam_hardening", "marked"): {"imaging_artifact": +0.4},
    ("lesion_visible_in_image", "no"): {"ECR": -0.3, "imaging_artifact": +0.2},
}
FEATURE_LOGODDS_CAP = 0.6


def feature_logodds(bundle_dict: Optional[Dict]) -> Dict[str, float]:
    """Capped per-hypothesis log-odds shift from an *assessed, non-abstained* visual feature reading."""
    if not bundle_dict:
        return {}
    vis = bundle_dict.get("visual") or {}
    if not vis.get("assessed") or bundle_dict.get("vlm", {}).get("status") != "ok":
        return {}
    acc: Dict[str, float] = {}
    for (k, v), shifts in FEATURE_LOGODDS.items():
        if vis.get(k) == v:
            for h, w in shifts.items():
                acc[h] = acc.get(h, 0.0) + w
    return {h: float(np.clip(w, -FEATURE_LOGODDS_CAP, FEATURE_LOGODDS_CAP)) for h, w in acc.items()}


def _llm_posterior(llm_out: Optional[Dict]) -> Optional[Dict[str, float]]:
    if not llm_out or not isinstance(llm_out.get("differential"), list):
        return None
    tmp = {h: 0.0 for h in HYPOTHESES}
    for it in llm_out["differential"]:
        if isinstance(it, dict) and it.get("diagnosis") in tmp:
            try:
                tmp[it["diagnosis"]] = max(0.0, float(it.get("probability", 0) or 0))
            except (TypeError, ValueError):
                pass
    s = sum(tmp.values())
    return {h: round(v / s, 4) for h, v in tmp.items()} if s > 0 else None


class RadiologistAgent(Agent):
    name = "radiologist"

    def __init__(self, backend=None, weights: Optional[np.ndarray] = None, weights_version: str = "expert_prior_v1"):
        super().__init__(backend)
        self.W = default_weight_matrix() if weights is None else np.asarray(weights, float)
        self.weights_version = weights_version

    def assess(self, bb: Blackboard, d: CandidateDescriptors, rnd: int = 0,
               challenges: Optional[List[Dict]] = None, prior: Optional[Dict[str, float]] = None) -> Evidence:
        truth = evaluate_criteria(d)
        rule_post = posterior_from_criteria(truth, self.W, prior=prior)
        questions = answer_questions(d)
        fired_for = [c.name for c, v in zip(CRITERIA, truth) if v is True and c.weights.get("ECR", 0) > 0]
        fired_against = [c.name for c, v in zip(CRITERIA, truth) if (v is True and c.weights.get("ECR", 0) < 0)
                         or (v is False and c.weights_if_false.get("ECR", 0) < 0)]
        not_assessable = [c.name for c, v in zip(CRITERIA, truth) if v is None]
        # adversarial penalties from the skeptic (previous round)
        penalty = 0.0
        for ch in challenges or []:
            if ch.get("target") in (None, d.component_id):
                penalty += {"major": 0.8, "moderate": 0.4, "minor": 0.15}.get(ch.get("severity"), 0.0)
        # rev 2.9: feature bundle (geometry + model visual reading) for this candidate, if the agent ran
        fb_ev = bb.latest("feature_bundle", refers_to=d.component_id)
        bundle = fb_ev.content if fb_ev else None
        f_shift = feature_logodds(bundle)
        f_notes = bundle_feature_notes(bundle) if bundle else {"for": [], "against": [], "confounders": []}
        feature_nudge_dropped = None

        def shifted(shift):
            if not penalty and not shift:
                return dict(rule_post)
            logp = np.log(np.array([max(rule_post[h], 1e-9) for h in HYPOTHESES]))
            logp[HYPOTHESES.index("ECR")] -= penalty
            for h, w in shift.items():
                logp[HYPOTHESES.index(h)] += w
            p = np.exp(logp - logp.max()); p /= p.sum()
            return {h: round(float(x), 4) for h, x in zip(HYPOTHESES, p)}
        post = shifted(f_shift)
        # optional model second opinion
        llm_post, llm_out = None, None
        user = json.dumps({"descriptors": d.to_dict(),
                           "diagnostic_questions": questions,
                           "criteria_fired": {c.name: v for c, v in zip(CRITERIA, truth)},
                           "skeptic_challenges": challenges or [],
                           "feature_bundle": None if not bundle else {
                               "visual": bundle.get("visual"),
                               "measurements": {k: {kk: m.get(kk) for kk in ("value", "unit", "source", "verified", "confidence")}
                                                for k, m in (bundle.get("measurements") or {}).items()},
                               "unverified_vlm_values": bundle.get("unverified_vlm_values"),
                               "vlm_geometry_disagreements": bundle.get("vlm_geometry_disagreements")}}, default=str)
        llm_out = self._llm(bb, RADIOLOGIST_SYSTEM, user, RADIOLOGIST_SCHEMA,
                            images=[bb.arrays.get("frames_png", {}).get(d.component_id) or bb.arrays.get("tiles", {}).get(d.component_id)]
                            if self.backend.supports_images and (bb.arrays.get("frames_png", {}).get(d.component_id)
                                                                 or bb.arrays.get("tiles", {}).get(d.component_id)) else None)
        llm_post = _llm_posterior(llm_out) if llm_out else None
        # -- weight of the model opinion: entropy-aware abstention + counterfactual grounding check ----------
        llm_weight, llm_meta = 0.0, None
        if llm_post:
            dse = (llm_out or {}).get("_dse")
            abstain = bool((llm_out or {}).get("_abstain"))
            llm_weight = LLM_WEIGHT_MAX
            if dse:
                # linear down-weighting with disagreement among samples (0 -> full weight, >= threshold -> abstain)
                thr = float((llm_out or {}).get("_dse_threshold") or 0.3)
                llm_weight = 0.0 if abstain else LLM_WEIGHT_MAX * max(0.0, 1.0 - dse["dse"] / max(thr, 1e-6))
            grounding = None
            cf_png = bb.arrays.get("cf_frames_png", {}).get(d.component_id)
            if llm_weight > 0 and cf_png is not None and self.backend.supports_images:
                # CLEAR-style perturbation test: the candidate voxels are filled with dentine-level values and the
                # candidate overlay removed.  A model that still reports the same lesion is not reading the image.
                cf_out = self._llm(bb, RADIOLOGIST_SYSTEM, user, RADIOLOGIST_SCHEMA, images=[cf_png])
                cf_post = _llm_posterior(cf_out) if cf_out else None
                if cf_post is not None:
                    delta = abs(cf_post["ECR"] - llm_post["ECR"])
                    still_sees = str(cf_out.get("lesion_visible_in_image", "")).lower() == "yes"
                    weak = delta < 0.10 and (still_sees or llm_post["ECR"] >= 0.3)
                    grounding = {"counterfactual_p_ecr": cf_post["ECR"], "original_p_ecr": llm_post["ECR"], "delta": round(delta, 3),
                                 "lesion_visible_after_fill": cf_out.get("lesion_visible_in_image"),
                                 "verdict": "weak" if weak else "responsive"}
                    if weak:
                        llm_weight = min(llm_weight, LLM_WEIGHT_WEAK_GROUNDING)
                else:
                    grounding = {"verdict": "not_tested", "reason": "counterfactual call failed"}
            llm_meta = {"weight": round(llm_weight, 3), "abstained": abstain, "dse": dse, "grounding": grounding,
                        "backend": (llm_out or {}).get("_backend") or getattr(self.backend, "name", None),
                        "lesion_visible_in_image": (llm_out or {}).get("lesion_visible_in_image"),
                        "question_answers": (llm_out or {}).get("question_answers")}
            if abstain:
                bb.say(self.name, f"candidate {d.component_id}: model opinion abstained (DSE {dse['dse']:.2f} > {llm_out.get('_dse_threshold')}, "
                                  f"{dse['n_clusters']} clusters / {dse['n_samples']} samples); rule result only")
                if f_shift:
                    # the same model read the same frame inconsistently: its categorical feature reading gets no weight either
                    feature_nudge_dropped = "model_abstained_on_differential_for_same_frame"
                    f_shift = {}
                    post = shifted(f_shift)
            elif grounding and grounding.get("verdict") == "weak":
                bb.say(self.name, f"candidate {d.component_id}: model opinion weakly grounded (P(ECR) unchanged after filling the "
                                  f"candidate: {grounding['original_p_ecr']:.2f} -> {grounding['counterfactual_p_ecr']:.2f}); weight {llm_weight:.2f}")
        combined = post
        if llm_post and llm_weight > 0:
            combined = {h: round((1 - llm_weight) * post[h] + llm_weight * llm_post[h], 4) for h in HYPOTHESES}
            s = sum(combined.values()); combined = {h: round(v / s, 4) for h, v in combined.items()}
        leading = max(combined, key=combined.get)
        capped = bool(d.landmark_flags) or d.landmark_source != "reviewed_points" or "canal_is_automatic_proxy" in (
            bb.latest("anatomy").caveats if bb.latest("anatomy") else [])
        ev = Evidence(self.name, "differential",
                      {"leading": leading, "posterior": combined, "rule_posterior": rule_post,
                       "prior_used": prior or dict(PRIOR), "prior_source": "context_shifted" if prior else "default",
                       "diagnostic_questions": questions,
                       "llm_posterior": llm_post, "llm_weight": round(llm_weight, 3), "llm_meta": llm_meta,
                       "llm_notes": None if not llm_out else
                       {k: llm_out.get(k) for k in ("key_findings_for", "key_findings_against", "uncertainties")},
                       "criteria_for_ecr": fired_for, "criteria_against_ecr": fired_against,
                       "criteria_not_assessable": not_assessable, "skeptic_penalty_logodds": round(penalty, 3),
                       "feature_logodds": {h: round(w, 3) for h, w in f_shift.items()}, "feature_notes": f_notes,
                       "feature_bundle_used": bool(bundle), "feature_nudge_dropped": feature_nudge_dropped,
                       "weights_version": self.weights_version, "calibrated": False,
                       "band": confidence_band(combined["ECR"], capped), "band_capped_by_unverified_anatomy": capped},
                      confidence=combined["ECR"], basis="llm+rule" if (llm_post and llm_weight > 0) else "rule",
                      refers_to=d.component_id, round=rnd,
                      caveats=["posterior_is_uncalibrated_expert_prior"] + (["landmarks_unverified"] if capped else []))
        bb.post(ev)
        bb.say(self.name, f"candidate {d.component_id}: leading {leading} (P(ECR)={combined['ECR']:.2f}, "
                          f"{ev.content['band']}); for: {fired_for}; against: {fired_against}"
                          + (f"; visual features for: {f_notes['for']}; against: {f_notes['against']}" if bundle and (f_notes["for"] or f_notes["against"]) else ""), rnd)
        return ev

    # -- learning from clinician-confirmed diagnoses --------------------------------
    def update_weights_from_feedback(self, labelled: List[Dict], min_cases: int = 20, lr: float = 0.05,
                                     l2_to_prior: float = 1.0, epochs: int = 200) -> Dict:
        """Refine the ECR column of the criterion weights with confirmed labels.

        labelled: [{"criteria": [True/False/None,...], "ecr": bool}, ...] from independent cases.
        A ridge penalty toward the expert prior keeps the update conservative; nothing changes below `min_cases`.
        """
        if len(labelled) < min_cases:
            return {"updated": False, "reason": f"need >= {min_cases} confirmed cases, have {len(labelled)}"}
        W0 = default_weight_matrix()
        j = HYPOTHESES.index("ECR")
        w = self.W[:, :, j].copy()
        X = np.array([[0 if v is None else (1 if v else -1) for v in c["criteria"]] for c in labelled], float)
        y = np.array([1.0 if c["ecr"] else 0.0 for c in labelled])
        b = np.log(PRIOR["ECR"] / (1 - PRIOR["ECR"]))
        for _ in range(epochs):
            contrib = np.where(X > 0, w[:, 0], np.where(X < 0, w[:, 1], 0.0))
            z = b + contrib.sum(1)
            p = 1 / (1 + np.exp(-z))
            g = (p - y)[:, None]
            g_true = (g * (X > 0)).mean(0); g_false = (g * (X < 0)).mean(0)
            w[:, 0] -= lr * (g_true + l2_to_prior * (w[:, 0] - W0[:, 0, j]) / len(labelled))
            w[:, 1] -= lr * (g_false + l2_to_prior * (w[:, 1] - W0[:, 1, j]) / len(labelled))
        self.W[:, :, j] = w
        self.weights_version = f"feedback_tuned_n{len(labelled)}"
        return {"updated": True, "n": len(labelled), "weights_version": self.weights_version}


# --------------------------------------------------------------------------- #
SKEPTIC_SYSTEM = (
    "You are a sceptical second reader whose job is to find reasons the leading diagnosis could be wrong: "
    "artefacts, segmentation and landmark errors, mimics (caries, cervical burnout, developmental grooves, "
    "restoration margins), over-reading of sub-millimetre findings. List concrete challenges with severity.")
SKEPTIC_SCHEMA = {"type": "object", "properties": {"challenges": {"type": "array", "items": {"type": "object",
                  "properties": {"issue": {"type": "string"}, "severity": {"type": "string", "enum": ["minor", "moderate", "major"]}},
                  "required": ["issue", "severity"]}}}, "required": ["challenges"]}


class SkepticAgent(Agent):
    name = "skeptic"

    def challenge(self, bb: Blackboard, d: CandidateDescriptors, diff: Evidence, rnd: int = 0) -> Evidence:
        ch: List[Dict] = []
        def add(issue, sev):
            ch.append({"issue": issue, "severity": sev, "target": d.component_id})
        if "cej_uses_unverified_bone_crest_offset" in d.landmark_flags:
            add("CEJ was not measured; it is the bone crest minus 1.5 mm. The cervical-band test depends on it.", "major")
        elif "automatic_cej_not_reviewed" in d.landmark_flags:
            add("Automatic CEJ estimate has not been reviewed.", "moderate")
        if "oblique_tooth_axis_hull_uncertainty" in d.landmark_flags:
            add("Tooth axis oblique to the scan axis; the convex hull may create false surface defects.", "moderate")
        if d.near_metal_fraction > 0.1:
            add(f"{d.near_metal_fraction:.0%} of the candidate is within 3 mm of metal-level voxels (streak artefact).", "major" if d.near_metal_fraction > 0.3 else "moderate")
        if d.dense_material_contact_fraction > 0.1:
            add("Candidate touches hyperdense material: restoration margin or enamel partial-volume effect possible.", "moderate")
        if d.volume_mm3 < 2.0:
            add(f"Volume {d.volume_mm3:.2f} mm3 is near the resolution limit; sub-millimetre findings are unreliable.", "moderate")
        if d.stable_core_fraction < 0.5:
            add("Candidate is threshold-sensitive (core fraction < 0.5).", "moderate")
        if d.touches_crop_boundary:
            add("Candidate touches the crop boundary; the defect may be truncated.", "moderate")
        anat = bb.latest("anatomy")
        if anat and "root_filling_present_canal_is_filling_not_pulp" in anat.caveats:
            add("Root filling present: the 'canal' is the filling; canal-proximity measures and internal-vs-external "
                "distinction are less reliable, and filling streaks can mimic defects.", "moderate")
        if d.canal_wall_intact is False:
            add("Candidate reaches the canal proxy: deep caries, perforating ECR or internal resorption all possible.", "minor")
        # rev 2.9: challenges from the feature bundle (unverified model numbers, model/geometry disagreement, confounders)
        fb_ev = bb.latest("feature_bundle", refers_to=d.component_id)
        bundle = fb_ev.content if fb_ev else None
        if bundle:
            local_read = (bundle.get("vlm") or {}).get("status") == "local_fallback"
            who = "Local image reading" if local_read else "Model"
            if bundle.get("unverified_vlm_values"):
                add(f"{who}-reported values without geometric verification: " + ", ".join(bundle["unverified_vlm_values"])
                    + "; treat as unmeasured.", "minor")
            if bundle.get("vlm_geometry_disagreements"):
                add(f"{who} estimates disagree with voxel geometry for " + ", ".join(bundle["vlm_geometry_disagreements"])
                    + "; geometry is kept.", "moderate")
            notes = bundle_feature_notes(bundle)
            for c in notes["confounders"]:
                add(f"{who} flags a confounder on the frame: {c}.", "moderate" if ("marked" in c or c == "burnout") else "minor")
            if bundle.get("vlm", {}).get("status") == "abstained":
                add("Vision model samples disagreed on the visual features (abstained); the visual reading was not used.", "minor")
            recon = bb.latest("screen_reconciliation")
            if recon:
                for row in recon.content.get("candidates", []):
                    if row.get("component_id") == d.component_id:
                        for n in row.get("notes", []):
                            if "portal" in n:
                                add(f"Screen / model disagreement: {n}.", "moderate")
        llm = self._llm(bb, SKEPTIC_SYSTEM, json.dumps({"descriptors": d.to_dict(), "radiologist": diff.content,
                                                        "feature_bundle_visual": (bundle or {}).get("visual")},
                                                       default=str), SKEPTIC_SCHEMA)
        if llm and isinstance(llm.get("challenges"), list):
            for it in llm["challenges"][:5]:
                if isinstance(it, dict) and it.get("issue"):
                    # model challenges are logged at reduced weight (never 'major')
                    sev = "moderate" if it.get("severity") == "major" else "minor"
                    ch.append({"issue": f"[model] {it['issue']}", "severity": sev, "target": d.component_id})
        ev = Evidence(self.name, "challenge", {"challenges": ch, "n_major": sum(c["severity"] == "major" for c in ch)},
                      basis="llm+rule" if llm else "rule", refers_to=d.component_id, round=rnd)
        bb.post(ev)
        bb.say(self.name, f"candidate {d.component_id}: {len(ch)} challenges ({ev.content['n_major']} major)", rnd)
        return ev


# --------------------------------------------------------------------------- #
class ClassifierAgent(Agent):
    """LEGACY (rev <= 2.8).  Only runs when `CaseConference(staging_scheme="patel_3d")` is requested explicitly.
    The default path uses `ExtentSummaryAgent` (measurement-based, no staging code)."""
    name = "patel_classifier"

    def run(self, bb: Blackboard, cid: int, spacing) -> Evidence:
        seg, norm, labels = bb.arrays["seg"], bb.arrays["norm"], bb.arrays["candidate_labels"]
        lesion = labels == cid
        geometry = bb.arrays["geometry"]
        cej_level = None
        if geometry.get("source") == "reviewed_points":
            # reviewed CEJ plane: express along the classifier's own axis convention via the detector geometry
            cej_level = None  # classifier re-estimates from enamel; reviewed points are reported alongside
        patel = classify_patel(lesion, seg, spacing, volume_norm=norm, cej_level_mm=cej_level)
        caveats = ["patel_from_automatic_candidate_mask", "components_require_clinician_review"]
        if patel.borderline:
            caveats.append("borderline_component_margins")
        if geometry.get("flags"):
            caveats += list(geometry["flags"])
        ev = Evidence(self.name, "patel", {**patel.to_dict(), "component_id": cid, "suggested": True},
                      basis="measurement", refers_to=cid, caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"candidate {cid}: suggested Patel {patel.code} (height margin {patel.height_margin_mm:.2f} mm, "
                          f"spread {patel.spread_deg:.0f} deg, canal distance {patel.min_canal_distance_mm:.2f} mm, "
                          f"borderline={patel.borderline})")
        bb.arrays["patel"] = patel
        return ev


# --------------------------------------------------------------------------- #
class TreatmentPlannerAgent(Agent):
    name = "treatment_planner"
    _BOOL_FIELDS = ["symptomatic", "pulp_vital", "restorable", "perio_support_adequate", "surgically_accessible",
                    "growing_patient", "aesthetic_zone", "root_fracture_risk_high"]

    def run(self, bb: Blackboard, mods: ClinicalModifiers, policy: ConstrainedBanditPolicy,
            tooth_type: str = "unknown", age_years: Optional[float] = None) -> Evidence:
        """Safe set + ranking.  Default (rev 2.9): keyed off the measurement-based extent summary; the legacy
        Patel matrix is used only when the conference ran the legacy classifier."""
        summary = bb.arrays.get("extent_summary")
        if summary is not None:
            def recommend(m):
                return recommend_from_measurements(summary, m)
            ctx = ClinicalContext(extent=summary, mods=mods, tooth_type=tooth_type, age_years=age_years)
            refers_to = bb.latest("extent_summary").refers_to
            key_label = f"measurement cell {summary.cell_key}"
            end_stage = False
            caveats = ["plan_conditional_on_unconfirmed_ecr_and_unreviewed_measurements",
                       "safe_set_keyed_by_measurement_descriptors_not_a_staging_code"]
            escalation = None
        else:   # legacy
            patel = bb.arrays["patel"]

            def recommend(m):
                return recommend_from_matrix(patel.code, m, portal_angle_deg=patel.portal_angle_deg,
                                             calcified_fraction=patel.calcified_fraction)
            ctx = ClinicalContext(patel=patel, mods=mods, tooth_type=tooth_type, age_years=age_years)
            refers_to = bb.latest("patel").refers_to
            key_label = f"Patel {patel.code}"
            end_stage = patel.code in END_STAGE_CODES
            caveats = ["plan_conditional_on_unconfirmed_ecr_and_unreviewed_patel_components"]
            escalation = end_stage_matrix() if end_stage else None
        rec = recommend(mods)
        decision = policy.decide(ctx, rec)
        # contingency branches for the clinical inputs that are still unknown
        contingencies = []
        for f in rec.missing_inputs:
            if f not in self._BOOL_FIELDS:
                continue
            branch = {}
            for val in (True, False):
                r2 = recommend(ClinicalModifiers(**{**asdict(mods), f: val}))
                branch[str(val)] = r2.primary.value
            if branch["True"] != branch["False"] or branch["True"] != rec.primary.value:
                contingencies.append({"input": f, "if_true": branch["True"], "if_false": branch["False"]})
        follow_up = ("Clinical and radiographic review after treatment; repeat CBCT only when it would change "
                     "management (ESE 2023 position statement). Periodic review is itself a matrix option for "
                     "non-progressive lesions in asymptomatic teeth.")
        content = {"cell_key": rec.cell_key, "cell_scheme": rec.cell_scheme,
                   "rule_recommendation": rec.to_dict(), "policy_decision": decision.to_dict(),
                   "end_stage": end_stage, "end_stage_matrix": escalation,
                   "contingencies": contingencies, "follow_up": follow_up,
                   "irreversible_options_in_safe_set": [t.value for t in rec.safe_set if t.irreversible],
                   "policy_feedback_cases": policy.n_cases}
        if summary is not None:
            content["extent_summary_ref"] = {"extent_descriptor": summary.extent_descriptor, "pulp_relationship": summary.pulp_relationship,
                                             "access_descriptor": summary.access_descriptor, "near_boundary": list(summary.near_boundary)}
            content["escalation_considerations"] = _escalation_considerations(summary)
        else:
            content["patel_code"] = rec.cell_key
        if rec.missing_inputs:
            caveats.append("missing_clinical_inputs:" + ",".join(rec.missing_inputs))
        ev = Evidence(self.name, "plan", content, basis="rule+policy", refers_to=refers_to, caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"{key_label}: primary {rec.primary.value}; policy ranking "
                          f"{' > '.join(decision.to_dict()['ranking'])}; missing inputs {rec.missing_inputs}")
        return ev


def _escalation_considerations(summary) -> List[str]:
    """Measurement-indexed replacement for the Patel end-stage table: plain-language considerations."""
    out = []
    if summary.extent_descriptor == "extensive":
        out.append("Extensive measured dentine loss: restorability, fracture risk and the patient's strategic value of the tooth decide "
                   "between repair, replantation, decoronation (growing patients) and extraction.")
    if summary.pulp_relationship == "canal_communication":
        out.append("Canal communication: root canal treatment is part of any repair; persistent intracanal bleeding after two "
                   "calcium-hydroxide dressings argues against internal repair (Talpos-Niculescu 2025).")
    if summary.access_descriptor == "deep_subcrestal_portal":
        out.append("Portal more than 3 mm below the bone crest: external access would sacrifice supporting bone; internal approach or "
                   "intentional replantation are the usual alternatives (Patel 2018 part 2 management principles).")
    if summary.access_descriptor == "no_surface_portal":
        out.append("No external surface portal measured: reconsider internal resorption or a closed cavity before lesion-directed treatment.")
    if summary.repair_phase_suspected:
        out.append("Radiopaque inclusions suggest a reparative phase; in an asymptomatic tooth periodic review is reasonable before "
                   "irreversible treatment.")
    if summary.restorability_concern:
        out.append("Dentine loss above 30% of the root dentine in the crop: confirm restorability (ferrule, remaining wall thickness) clinically.")
    return out


# --------------------------------------------------------------------------- #
class ClinicalCorrelationAgent(Agent):
    """Question 8: compare the imaging differential with the clinician-entered findings."""
    name = "clinical_correlation"

    def run(self, bb: Blackboard, cid: int, posterior: Dict[str, float], findings: Optional[ClinicalFindings]) -> Evidence:
        findings = findings or ClinicalFindings()
        adjusted, rep = correlate(posterior, findings)
        # rev 2.9: cross-check entered findings against the measured feature bundle (text only; no posterior change)
        fb_ev = bb.latest("feature_bundle", refers_to=cid)
        cross = _clinical_feature_crosscheck(fb_ev.content if fb_ev else None, findings)
        ev = Evidence(self.name, "clinical_correlation",
                      {"findings": findings.to_dict(), "imaging_posterior": posterior, "adjusted_posterior": adjusted,
                       **rep, "feature_crosscheck": cross}, basis="clinician" if findings.to_dict() else "rule", refers_to=cid,
                      caveats=[] if findings.to_dict() else ["no_clinical_findings_entered"])
        bb.post(ev)
        bb.say(self.name, f"candidate {cid}: clinical picture {rep['verdict']} ({rep['n_findings']} findings); "
                          f"P(ECR) {posterior['ECR']:.2f} -> {adjusted['ECR']:.2f}"
                          + (f"; feature cross-check: {'; '.join(cross)}" if cross else ""))
        return ev


def _clinical_feature_crosscheck(bundle: Optional[Dict], findings: ClinicalFindings) -> List[str]:
    """Plain-language consistency notes between measured features and clinician-entered findings."""
    if not bundle:
        return []
    notes: List[str] = []
    ms = bundle.get("measurements", {}) or {}
    vis = bundle.get("visual", {}) or {}
    comm = (ms.get("root_canal_communication") or {})
    if comm.get("verified") and comm.get("value") is True and findings.pulp_sensibility_normal is True:
        notes.append("measured canal communication with a normal pulp response: check the canal mask (root filling / proxy) or expect late-stage vitality loss")
    if comm.get("verified") and comm.get("value") is False and findings.symptomatic is True:
        notes.append("symptoms with a measured dentine barrier to the canal: consider another source of symptoms")
    portal = (ms.get("coronal_limit_from_crest_mm") or {})
    if portal.get("verified") and portal.get("value") is not None and portal["value"] > 3.0 and findings.cervical_defect_hard_irregular is True:
        notes.append("a hard irregular defect was probed but the measured portal lies > 3 mm below the crest: probing site and candidate may differ")
    if vis.get("assessed") and vis.get("relation_to_restoration_margin") == "continuous_with_margin" and findings.restoration_at_defect is False:
        notes.append("model sees the defect continuous with a restoration margin but no restoration was recorded at the defect")
    if vis.get("assessed") and vis.get("portal_of_entry_visible") == "yes" and findings.cervical_defect_hard_irregular is True:
        notes.append("visible portal of entry and a hard irregular cervical defect on probing are mutually consistent")
    return notes


# --------------------------------------------------------------------------- #
# rev 2.7: agent-layer roles that select perception tools, verify, track coverage and retrieve knowledge
class ContextAgent(Agent):
    """Posts upstream context (Diagnocat per-tooth findings, patient chart) as evidence.  Context shifts priors
    and fills clinical inputs; it never creates an imaging finding."""
    name = "context"

    def run(self, bb: Blackboard, context) -> Optional[Evidence]:
        if context is None:
            bb.say(self.name, "no upstream context (Diagnocat / chart) supplied; default priors used")
            return None
        c = context.to_dict() if hasattr(context, "to_dict") else dict(context)
        caveats = ["context_shifts_priors_only_never_creates_findings"]
        if c.get("diagnocat_conditions"):
            caveats.append("diagnocat_has_no_external_cervical_resorption_class")
        ev = Evidence(self.name, "context", c, basis="clinician" if c.get("chart") else "rule", caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"tooth {c.get('iso_number')} ({c.get('tooth_type')}): prior shifts {c.get('prior_shifts')}, "
                          f"modifier hints {c.get('modifier_hints')}, priority {c.get('priority')}")
        return ev


class VerifierAgent(Agent):
    """Critic re-inspection ("detect-verify-reason"): asks the perception toolkit to re-delineate each candidate
    under alternative processing and to build the linked multiplanar evidence frame.  Low reproducibility
    becomes a skeptic-style challenge; it never silently deletes a candidate."""
    name = "verifier"

    def run(self, bb: Blackboard, toolkit, d: CandidateDescriptors, spacing, source_id: str, full_slice=None,
            clahe: bool = False, mode: str = "smooth", sharpen: bool = False, counterfactual: bool = False) -> Evidence:
        seg, raw, labels = bb.arrays["seg"], bb.arrays["raw"], bb.arrays["candidate_labels"]
        anat = bb.latest("anatomy").content
        geom = bb.arrays.get("geometry", {})
        axis = geom.get("axis")
        levels = bb.arrays.get("levels")
        if levels is None:
            levels = toolkit.call("reference_levels", raw=raw, seg=seg, t_dense=anat["t_dense"])["levels"]
            bb.arrays["levels"] = levels
        rep = toolkit.call("reproduce_candidate", raw=raw, seg=seg, labels=labels, cid=d.component_id, spacing=spacing,
                           t_tooth=anat["t_tooth"], soft_level=levels["soft_tissue"], tooth_axis=axis, mode=mode)
        frame_rec = None
        try:
            fr = toolkit.call("evidence_frame", raw=raw, seg=seg, labels=labels, cid=d.component_id, spacing=spacing,
                              crop_offset_zyx=[s.start for s in bb.arrays["crop"]], source_id=source_id, levels=levels,
                              tooth_axis=axis, full_slice=full_slice, clahe=clahe, sharpen=sharpen)["frame"]
            bb.arrays.setdefault("frames_png", {})[d.component_id] = fr.png
            frame_rec = fr.to_dict()
            if counterfactual:
                cfr = toolkit.call("evidence_frame", raw=raw, seg=seg, labels=labels, cid=d.component_id, spacing=spacing,
                                   crop_offset_zyx=[s.start for s in bb.arrays["crop"]], source_id=source_id, levels=levels,
                                   tooth_axis=axis, full_slice=full_slice, clahe=clahe, sharpen=sharpen, counterfactual=True)["frame"]
                bb.arrays.setdefault("cf_frames_png", {})[d.component_id] = cfr.png
                frame_rec["counterfactual_frame_rendered"] = True
        except Exception as e:  # rendering must never break the conference
            frame_rec = {"error": str(e)}
        r = rep["reproducibility"]
        verdict = "reproduced" if r >= 0.7 else "partially_reproduced" if r >= 0.4 else "not_reproduced"
        challenges = []
        if r < 0.7:
            challenges.append({"issue": f"Candidate only {verdict.replace('_', ' ')} under alternative processing "
                                        f"({'test-time augmentation' if mode == 'tta' else 'smoothing'}; mean Dice {r:.2f}"
                                        f"{'' if rep.get('dice_min') is None else f', min {rep['dice_min']:.2f}'}); "
                                        f"finding depends on the processing path.", "severity": "moderate" if r < 0.4 else "minor",
                               "target": d.component_id})
        ap = rep.get("axis_persistence") or {}
        if ap and ap.get("long_axis_extent_mm", 0) < 0.5:
            challenges.append({"issue": "Extent along the tooth axis < 0.5 mm: single-plane finding.", "severity": "moderate",
                               "target": d.component_id})
        content = {"reproducibility": r, "verdict": verdict, "mode": rep.get("mode", mode), "dice_by_sigma": rep.get("dice_by_sigma"),
                   "dice_by_path": rep.get("dice_by_path"), "dice_min": rep.get("dice_min"), "axis_persistence": ap,
                   "evidence_frame": frame_rec, "challenges": challenges, "reference_levels": {k: v for k, v in levels.items() if k != "method"}}
        ev = Evidence(self.name, "verification", content, basis="measurement", refers_to=d.component_id,
                      caveats=["verification_tests_processing_dependence_not_pathology"])
        bb.post(ev)
        bb.say(self.name, f"candidate {d.component_id}: {verdict} (reproducibility {r:.2f}); {len(challenges)} challenge(s); "
                          f"{len((frame_rec or {}).get('panels', []))}-panel evidence frame")
        return ev


class CoverageAgent(Agent):
    """Tracks what has and has not been examined and which prerequisites for a diagnosis are missing."""
    name = "coverage"

    def run(self, bb: Blackboard, context, clinical_findings, examined_teeth: Optional[List[int]] = None,
            focus_teeth: Optional[List[int]] = None, tool_log: Optional[List[Dict]] = None) -> Evidence:
        anat = bb.latest("anatomy")
        screen = bb.latest("screen_summary")
        missing: List[Dict[str, str]] = []
        def miss(item, why, how):
            missing.append({"item": item, "why": why, "how_to_resolve": how})
        flags = set((screen.content.get("landmarks", {}) or {}).get("flags", [])) if screen else set()
        if not screen or (screen.content.get("landmarks", {}) or {}).get("source") != "reviewed_points":
            miss("CEJ and apex landmarks", "cervical band and the CEJ-referenced extent measurements are estimated, not measured",
                 "mark CEJ_ZYX / APEX_ZYX and re-run")
        if anat and "canal_is_automatic_proxy" in anat.caveats:
            miss("canal delineation", "canal distance / communication measurements use an automatic proxy", "import an expert canal mask")
        if anat and "root_filling_present_canal_is_filling_not_pulp" in anat.caveats:
            miss("pre-treatment pulp anatomy", "root filling present: internal-vs-external distinction less reliable",
                 "compare with pre-treatment imaging if available")
        cf = (clinical_findings.to_dict() if clinical_findings is not None and hasattr(clinical_findings, "to_dict") else (clinical_findings or {}))
        if not cf:
            miss("clinical examination", "ESE 2023: diagnosis requires clinical correlation (probing, bleeding, pulp tests, history)",
                 "enter ClinicalFindings")
        if context is None:
            miss("upstream context", "no Diagnocat findings or patient chart; default priors used", "supply a DiagnocatStudy and/or PatientChart")
        if not bb.query(kind="verification"):
            miss("re-verification", "candidates were not re-inspected under alternative processing", "enable the verifier")
        examined = sorted(set(examined_teeth or []))
        unexamined_focus = [t for t in (focus_teeth or []) if t not in examined]
        if unexamined_focus:
            miss("chief-complaint teeth", f"teeth {unexamined_focus} are in the complaint focus but have not been conferenced",
                 "run the per-tooth conference on them")
        content = {"examined_teeth_iso": examined, "focus_teeth_iso": sorted(set(focus_teeth or [])),
                   "unexamined_focus_teeth": unexamined_focus, "missing_prerequisites": missing,
                   "tools_used": sorted({c["tool"] for c in (tool_log or [])}), "n_tool_calls": len(tool_log or []),
                   "completeness": round(1.0 - min(len(missing), 6) / 6.0, 2)}
        ev = Evidence(self.name, "coverage", content, basis="rule", caveats=["completeness_is_a_checklist_not_a_probability"])
        bb.post(ev)
        bb.say(self.name, f"{len(missing)} missing prerequisite(s); examined {examined or 'target tooth only'}; focus {content['focus_teeth_iso']}")
        return ev


class KnowledgeAgent(Agent):
    """Retrieves guideline statements to *interpret* established findings and to name missing prerequisites.
    It cannot add findings: every citation is attached to an existing question answer, plan step or gap."""
    name = "knowledge"

    def __init__(self, backend=None, kb=None):
        super().__init__(backend)
        from ..knowledge import KnowledgeBase
        self.kb = kb or KnowledgeBase()

    def _cite(self, tags):
        return [{"id": s["id"], "source": s["source"], "url": s.get("url"), "text": s["text"]} for s in self.kb.for_tags(tags)]

    def run(self, bb: Blackboard, consensus: Dict, tooth_type: str = "unknown") -> Evidence:
        from ..knowledge import cdt_codes_for
        dx = consensus.get("suggested_diagnosis") or {}
        q_cites = {}
        for q in dx.get("diagnostic_questions", []) or []:
            qid = q.get("id") or q.get("question_id") or q.get("q")
            if qid:
                q_cites[qid] = [c["id"] for c in self._cite([qid])]
        plan = consensus.get("suggested_plan")
        plan_cites, cdt = [], {}
        if plan:
            plan_cites = self._cite(["plan"])
            rec = plan.get("rule_recommendation", {})
            options = [rec.get("primary")] + list(rec.get("safe_set", []) or [])
            for opt in [o for o in options if o]:
                cdt[opt] = cdt_codes_for(opt, tooth_type)
            plan["cdt_codes"] = cdt
            plan["knowledge_citations"] = [c["id"] for c in plan_cites]
        gaps = []
        cov = bb.latest("coverage")
        if cov:
            for m in cov.content.get("missing_prerequisites", []):
                tag = "clinical" if "clinical" in m["item"] else "staging" if "CEJ" in m["item"] else "imaging"
                gaps.append({**m, "citations": [c["id"] for c in self._cite([tag])]})
        ctx = bb.latest("context")
        diag_cites = self._cite(["diagnocat"]) if ctx and ctx.content.get("diagnocat_conditions") else []
        # rev 2.8: whenever a vision-language model opinion entered the fusion, cite its documented limits
        vlm_used = any((r.get("llm_meta") or {}) for r in consensus.get("candidates", []) or [])
        vlm_cites = self._cite(["vlm_limits"]) if vlm_used else []
        content = {"kb_version": self.kb.version, "question_citations": q_cites,
                   "plan_citations": [c["id"] for c in plan_cites], "cdt_codes": cdt,
                   "gap_citations": gaps, "diagnocat_scope_citations": [c["id"] for c in diag_cites],
                   "vlm_limit_citations": [c["id"] for c in vlm_cites],
                   "statements": {s["id"]: {"source": s["source"], "url": s.get("url"), "text": s["text"]} for s in self.kb.statements
                                  if s["id"] in set(sum(q_cites.values(), [])) | {c["id"] for c in plan_cites + diag_cites + vlm_cites}
                                  | {cid for g in gaps for cid in g["citations"]}}}
        ev = Evidence(self.name, "knowledge", content, basis="rule",
                      caveats=["knowledge_interprets_established_findings_only", "statements_are_paraphrases_read_the_sources"])
        bb.post(ev)
        bb.say(self.name, f"attached {len(content['statements'])} guideline statements; CDT codes for {list(cdt)}; {len(gaps)} gap(s) cited")
        return ev
