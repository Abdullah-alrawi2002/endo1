"""Rev 2.9 agents: VLM feature measurement and the measurement-based extent summary.

FeatureMeasurementAgent  -> per candidate: geometric measurements + vision-model visual features, reconciled,
                            published as a `feature_bundle` Evidence (basis "measurement" or "vlm+measurement").
ExtentSummaryAgent       -> for the candidate the moderator accepts: descriptive severity / extent summary built
                            from the *verified* half of the bundle (replaces the Patel classifier in this path;
                            no staging code is produced).

Both agents only add evidence.  Neither can change a status, a posterior threshold or a plan directly; the
treatment planner reads the extent summary, and the radiologist / skeptic read the bundle, through the blackboard.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .blackboard import Blackboard, Evidence
from .llm_backend import LLMBackend, NullBackend
from .visual_features import CandidateDescriptors
from ..extent_summary import ExtentSummary, summarise_extent
from ..perception.feature_schema import FeatureBundle
from ..perception.vlm_features import bundle_summary_line, measure_candidate_features


class FeatureMeasurementAgent:
    name = "feature_measurement"

    def __init__(self, backend: Optional[LLMBackend] = None, buccal_direction=None):
        self.backend = backend or NullBackend()
        self.buccal_direction = buccal_direction

    def run(self, bb: Blackboard, d: CandidateDescriptors, spacing, source_id: str) -> Evidence:
        seg, raw, labels = bb.arrays["seg"], bb.arrays["raw"], bb.arrays["candidate_labels"]
        anat = bb.latest("anatomy")
        geometry = bb.arrays.get("geometry", {})
        cid = d.component_id
        frame_png = bb.arrays.get("frames_png", {}).get(cid)
        tiles_png = bb.arrays.get("tiles", {}).get(cid)
        ver = bb.latest("verification", refers_to=cid)
        frame_meta = (ver.content.get("evidence_frame") if ver else None) or None
        if frame_meta and "panels" not in frame_meta:
            frame_meta = None
        bundle = measure_candidate_features(raw, seg, labels, cid, spacing, geometry, backend=self.backend,
                                            frame_png=frame_png, tiles_png=tiles_png, frame_meta=frame_meta,
                                            crop_offset_zyx=[s.start for s in bb.arrays["crop"]], source_id=source_id,
                                            descriptors=d, anatomy_caveats=list(anat.caveats) if anat else [],
                                            buccal_direction=self.buccal_direction)
        bb.arrays.setdefault("feature_bundles", {})[cid] = bundle
        vlm_status = bundle.vlm.get("status")
        if vlm_status == "failed":
            bb.arrays["backend_failures"] = bb.arrays.get("backend_failures", 0) + 1
        elif vlm_status in ("ok", "abstained"):
            bb.arrays["backend_calls_ok"] = bb.arrays.get("backend_calls_ok", 0) + 1
        basis = "vlm+measurement" if vlm_status == "ok" else "measurement"
        caveats = list(bundle.caveats) + ["no_staging_code_no_diagnosis_no_treatment_implied"]
        ev = Evidence(self.name, "feature_bundle", bundle.to_dict(), basis=basis, refers_to=cid, caveats=caveats)
        bb.post(ev)
        bb.say(self.name, f"candidate {cid}: " + bundle_summary_line(bundle))
        if bundle.unverified_vlm_values:
            bb.say(self.name, f"candidate {cid}: model values kept WITHOUT geometric verification: "
                              f"{', '.join(bundle.unverified_vlm_values)}")
        if bundle.vlm_geometry_disagreements:
            bb.say(self.name, f"candidate {cid}: model estimates disagree with voxel geometry for "
                              f"{', '.join(bundle.vlm_geometry_disagreements)}; geometry kept")
        return ev


class ExtentSummaryAgent:
    name = "extent_summary"

    def run(self, bb: Blackboard, cid: int) -> Evidence:
        bundle: FeatureBundle = bb.arrays.get("feature_bundles", {})[cid]
        d: CandidateDescriptors = bb.arrays["descriptors"][cid]
        summary: ExtentSummary = summarise_extent(bundle, d)
        geometry = bb.arrays.get("geometry", {})
        caveats = ["extent_from_automatic_candidate_mask", "descriptors_are_measurement_bands_not_a_staging_system",
                   "components_require_clinician_review"]
        if summary.near_boundary:
            caveats.append("descriptor_near_band_edge")
        if summary.unavailable:
            caveats.append("unavailable_measurements:" + ",".join(summary.unavailable))
        caveats += list(geometry.get("flags", []) or [])
        ev = Evidence(self.name, "extent_summary", {**summary.to_dict(), "suggested": True}, basis="measurement",
                      refers_to=cid, caveats=caveats)
        bb.post(ev)
        bb.arrays["extent_summary"] = summary
        bb.say(self.name, f"candidate {cid}: {summary.narrative}")
        return ev


def bundle_feature_notes(bundle_dict: Dict) -> Dict[str, List[str]]:
    """Visual features that argue for / against ECR and the confounders the model flagged (text, for the agents)."""
    vis = bundle_dict.get("visual", {}) or {}
    if not vis.get("assessed"):
        return {"for": [], "against": [], "confounders": []}
    pro, con, conf = [], [], []
    if vis.get("portal_of_entry_visible") == "yes":
        pro.append("portal of entry visible on the frame")
    if vis.get("pericanal_radiopaque_line") == "intact":
        pro.append("pericanal radiopaque line intact where the defect approaches the canal")
    if vis.get("fibro_osseous_appearance") == "yes" or vis.get("radiolucency_pattern") == "mottled":
        pro.append("mottled / fibro-osseous interior (repair tissue appearance)")
    if vis.get("border_definition") == "irregular":
        pro.append("irregular borders")
    if vis.get("border_definition") == "smooth":
        con.append("smooth borders (non-carious cervical lesion / anatomical concavity pattern)")
    if vis.get("relation_to_restoration_margin") == "continuous_with_margin":
        con.append("defect continuous with a restoration margin")
    if vis.get("burnout_artifact_suspected") == "yes":
        con.append("cervical burnout appearance suspected")
        conf.append("burnout")
    if vis.get("metal_beam_hardening") == "marked":
        conf.append("marked metal / beam-hardening streaks")
    elif vis.get("metal_beam_hardening") == "mild":
        conf.append("mild metal / beam-hardening streaks")
    if vis.get("lesion_visible_in_image") == "no":
        con.append("model did not see a lesion in the frame")
    return {"for": pro, "against": con, "confounders": conf}
