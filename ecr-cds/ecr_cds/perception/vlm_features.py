"""VLM-driven feature measurement of a suspected-tooth candidate (rev 2.9).

Flow for one candidate
----------------------
1. `measure_candidate_geometry`  - physical measurements from the candidate mask, the tooth / canal / bone masks
                                   and the voxel spacing, along the tooth long axis; each with confidence and
                                   provenance (which panels / slices of the evidence frame contain it).
2. `query_vlm_features`          - one structured request to an OpenAI-compatible vision model (the existing
                                   `LLMBackend` abstraction: `NemotronLocalBackend`, `OpenAICompatibleBackend`,
                                   or any local server).  The model sees the provenance-stamped evidence frame and
                                   the *scale* (voxel size, crop width); it is **not** given the geometric numbers,
                                   so its estimates can be cross-checked rather than echoed.
3. `reconcile`                   - every model estimate is compared with the geometric value of the same name.
                                   Geometry always wins; agreement / disagreement is recorded.  Estimates for
                                   quantities geometry could not compute (no canal, no CEJ, ...) are kept as
                                   `vlm_unverified` and listed in `unverified_vlm_values`.
4. `measure_candidate_features`  - assembles the `FeatureBundle`.

The module never stages, classifies or recommends, and it emits no Patel codes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..auto_segment import tooth_hull_convex
from ..segmentation import ToothSegmentation
from .feature_schema import (ESTIMATE_KINDS, MEASUREMENT_UNITS, SOURCE_UNAVAILABLE, SOURCE_VLM_UNVERIFIED, SOURCE_VOXEL,
                             VLM_FEATURE_RESPONSE_SCHEMA, FeatureBundle, FeatureSchemaError, Measurement, VisualFeatures,
                             validate_vlm_response)

# --------------------------------------------------------------------------- prompts
FEATURE_SYSTEM_PROMPT = (
    "You are a dental radiology assistant describing ONE suspected cervical root defect on a CBCT evidence frame. "
    "The image is an 8-panel frame: row 1 = three neighbouring axial slices (-0.8 mm, centre, +0.8 mm) and a coronal "
    "view; row 2 = sagittal, tooth long-axis reformat, perpendicular-to-axis reformat, and the unmodified full slice "
    "with the crop box. Overlays: tooth outline cyan, canal proxy yellow, candidate region red. Each cropped panel "
    "spans the physical width given in the user message (crop_width_mm); use it and the voxel size as your only scale. "
    "Describe only what is visible. Give categorical visual features for the red region: radiolucency texture, border "
    "definition, mottled / fibro-osseous (radiopaque islands inside the defect), integrity of the thin radiopaque line "
    "around the canal where the defect approaches it, whether a surface portal of entry is visible and on which surface, "
    "relation to any restoration margin, cervical burnout (projection-dependent band of lucency at the CEJ), and metal / "
    "beam-hardening streaks. Then give ROUGH physical estimates in millimetres (null if you cannot estimate): extent along "
    "the tooth long axis (cervical to apical), mesio-distal and bucco-lingual extents, coronal limit relative to the "
    "CEJ (negative = coronal to CEJ), apical limit relative to the bone crest (positive = below the crest), fraction of "
    "the root dentine involved (0-1), minimum dentine thickness between defect and canal, and whether the defect "
    "communicates with the canal. Your numbers are cross-checked against voxel geometry, so prefer null to a guess. "
    "Do NOT stage, grade or classify the lesion, do not name any classification system, do not diagnose and do not "
    "suggest treatment. Answer with JSON only.")


def _feature_user_payload(cid: int, spacing, frame_meta: Optional[Dict[str, Any]], crop_width_mm: float,
                          tiles_only: bool) -> str:
    panels = []
    for p in (frame_meta or {}).get("panels", []) or []:
        panels.append({"name": p.get("name"), "plane": p.get("plane"), "index_in_volume": p.get("index")})
    return json.dumps({"candidate_id": int(cid), "voxel_size_mm_zyx": [round(float(s), 3) for s in spacing],
                       "crop_width_mm": crop_width_mm,
                       "image": "three orthogonal 12 mm tiles" if tiles_only else "8-panel evidence frame",
                       "panels": panels,
                       "instructions": "Report categorical visual features and rough mm estimates read from the panels; "
                                       "null when not estimable. No staging, no diagnosis, no treatment."}, default=str)


def feature_cluster_key(obj: Dict[str, Any]) -> str:
    """Discrete semantic key for entropy-based abstention on feature answers (categorical fields only)."""
    vf = obj.get("visual_features") if isinstance(obj, dict) else None
    if not isinstance(vf, dict):
        return "invalid"
    parts = [f"{k}={str(vf.get(k)).lower()[:24]}" for k in sorted(vf)]
    est = obj.get("estimates") or {}
    if isinstance(est, dict) and "root_canal_communication" in est:
        parts.append(f"canal={str(est['root_canal_communication']).lower()[:10]}")
    parts.append(f"visible={str(obj.get('lesion_visible_in_image')).lower()[:10]}")
    return "|".join(parts)


# --------------------------------------------------------------------------- geometry
def _bone_shell_points(seg, spacing, shell_mm: float) -> np.ndarray:
    """Physical coordinates (mm, zyx) of bone voxels within `shell_mm` of the tooth surface.  Computed on the tooth
    bounding box via a distance transform (a full-volume ball dilation at 0.1 mm voxels costs tens of seconds)."""
    idx = np.argwhere(seg.tooth)
    if idx.size == 0:
        return np.zeros((0, 3))
    sp = np.asarray(spacing, float)
    pad = np.ceil(shell_mm / sp).astype(int) + 1
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.asarray(seg.tooth.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    tooth_c = seg.tooth[sl]
    dist = ndi.distance_transform_edt(~tooth_c, sampling=spacing)
    shell = (dist > 0) & (dist <= shell_mm) & seg.bone[sl]
    return (np.argwhere(shell) + lo) * sp


def _angular_spread_deg(angles_deg: np.ndarray, bin_deg: float = 6.0, close_gap_deg: float = 6.0) -> float:
    if angles_deg.size == 0:
        return 0.0
    nb = int(round(360 / bin_deg))
    occ = np.zeros(nb, bool)
    occ[(np.floor((angles_deg % 360) / bin_deg).astype(int)) % nb] = True
    k = int(round(close_gap_deg / bin_deg))
    if k > 0:
        occ = ndi.binary_closing(np.r_[occ, occ, occ], structure=np.ones(2 * k + 1))[nb:2 * nb]
    return float(occ.sum() * bin_deg)


def _in_plane_frame(axis: np.ndarray, buccal_direction: Optional[np.ndarray], radial_hint: Optional[np.ndarray]):
    """Return (bl_unit, md_unit, definition) perpendicular to the tooth axis.

    With a patient buccal direction the bucco-lingual axis is its projection onto the plane and the mesio-distal axis
    is perpendicular to it.  Without one, the radial direction through the candidate centroid is used as a proxy for
    the bucco-lingual axis (the surface the defect enters from), which is only right for buccal / lingual defects; the
    definition string records which was used so the consumer can discount the labels.
    """
    a = axis / (np.linalg.norm(axis) + 1e-9)
    if buccal_direction is not None and np.linalg.norm(buccal_direction) > 0:
        b = np.asarray(buccal_direction, float)
        b = b - (b @ a) * a
        if np.linalg.norm(b) > 1e-6:
            b /= np.linalg.norm(b)
            return b, np.cross(a, b), "patient_buccal_direction"
    if radial_hint is not None and np.linalg.norm(radial_hint) > 1e-6:
        b = radial_hint / np.linalg.norm(radial_hint)
        return b, np.cross(a, b), "radial_direction_through_candidate_centroid_proxy"
    ref = np.array([0, 1.0, 0]) if abs(a[1]) < 0.9 else np.array([0, 0, 1.0])
    b = np.cross(a, ref); b /= np.linalg.norm(b) + 1e-9
    return b, np.cross(a, b), "arbitrary_in_plane_axes"


def measure_candidate_geometry(raw: np.ndarray, seg: ToothSegmentation, labels: np.ndarray, cid: int, spacing,
                               geometry: Dict[str, Any], crop_offset_zyx=(0, 0, 0), descriptors=None,
                               anatomy_caveats: Optional[List[str]] = None, buccal_direction=None,
                               frame_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Measurement]:
    """Physical measurements of candidate `cid` from voxel geometry, each with confidence and provenance.

    geometry: the screen geometry (`centre_mm`, `axis`, `cej_t_mm`, `source`, `flags`) - the same frame the
              descriptors and the Patel-free extent summary use, so all numbers share one axis convention
              (apical positive along `axis`, origin at `centre_mm`).
    """
    spacing = np.asarray(spacing, float)
    pitch = float(spacing.min())
    vox_vol = float(np.prod(spacing))
    off = np.asarray(crop_offset_zyx, int)
    centre = np.asarray(geometry["centre_mm"], float)
    axis = np.asarray(geometry["axis"], float); axis /= np.linalg.norm(axis) + 1e-9
    cej = geometry.get("cej_t_mm")
    flags_landmark = list(geometry.get("flags", []) or [])
    landmark_source = str(geometry.get("source", "automatic_unreviewed"))
    anatomy_caveats = list(anatomy_caveats or [])
    m = labels == cid
    if not m.any():
        raise ValueError(f"candidate {cid} has no voxels")
    q = np.argwhere(m)
    pts = q * spacing
    t = (pts - centre) @ axis
    radial_vec = (pts - centre) - t[:, None] * axis
    # local tooth-axis point at the candidate's levels (the screen origin need not lie on the tooth axis)
    tpts = np.argwhere(seg.tooth) * spacing
    tt = (tpts - centre) @ axis
    sel = (tt >= t.min()) & (tt <= t.max())
    local_centre = tpts[sel].mean(0) if int(sel.sum()) > 50 else centre
    radial_local = (pts - local_centre) - ((pts - local_centre) @ axis)[:, None] * axis
    # --- reliability factors shared by all measurements -----------------------------------------------------
    base_conf = 0.9
    stable = float(getattr(descriptors, "stable_core_fraction", 1.0) or 1.0)
    near_metal = float(getattr(descriptors, "near_metal_fraction", 0.0) or 0.0)
    touches_edge = bool(getattr(descriptors, "touches_crop_boundary", False))
    mask_conf = base_conf * max(0.3, min(1.0, stable))
    if near_metal > 0.3:
        mask_conf *= 0.7
    common_flags: List[str] = ["candidate_mask_is_automatic_screen_output"]
    if stable < 0.5:
        common_flags.append("candidate_threshold_sensitive")
    if touches_edge:
        common_flags.append("candidate_touches_crop_boundary_extent_may_be_truncated")
    if near_metal > 0.3:
        common_flags.append("near_metal_streaks_may_distort_mask")
    cej_conf = mask_conf
    cej_flags: List[str] = []
    if landmark_source != "reviewed_points":
        cej_conf = min(cej_conf, 0.5)
        cej_flags.append("cej_automatic_unreviewed")
    if "cej_uses_unverified_bone_crest_offset" in flags_landmark:
        cej_conf = min(cej_conf, 0.3)
        cej_flags.append("cej_is_bone_crest_minus_1.5mm_assumption")
    canal_conf = mask_conf
    canal_flags: List[str] = []
    if "canal_is_automatic_proxy" in anatomy_caveats:
        canal_conf = min(canal_conf, 0.6)
        canal_flags.append("canal_is_automatic_proxy")
    if "root_filling_present_canal_is_filling_not_pulp" in anatomy_caveats:
        canal_conf = min(canal_conf, 0.4)
        canal_flags.append("canal_is_root_filling_not_pulp")
    # --- provenance ------------------------------------------------------------------------------------------
    zr = [int(q[:, 0].min() + off[0]), int(q[:, 0].max() + off[0])]
    yr = [int(q[:, 1].min() + off[1]), int(q[:, 1].max() + off[1])]
    xr = [int(q[:, 2].min() + off[2]), int(q[:, 2].max() + off[2])]
    panels = [{"name": p.get("name"), "plane": p.get("plane"), "index": p.get("index")}
              for p in ((frame_meta or {}).get("panels") or [])]
    prov_base = {"candidate_id": int(cid), "mask": "candidate_labels (automatic cervical screen component)",
                 "volume_slice_range_z": zr, "volume_row_range_y": yr, "volume_col_range_x": xr,
                 "centroid_volume_zyx": (q.mean(0) + off).round(1).tolist(),
                 "tooth_axis_zyx": axis.round(3).tolist(), "axis_origin_crop_mm": centre.round(2).tolist(),
                 "spacing_zyx_mm": spacing.round(4).tolist(), "evidence_frame_panels": panels}

    def prov(**extra):
        return {**prov_base, **extra}

    def meas(name, value, conf, flags, **extra) -> Measurement:
        unit = MEASUREMENT_UNITS[name]
        if value is None:
            return Measurement(name, None, unit, SOURCE_UNAVAILABLE, False, 0.0, prov(**extra), flags=list(flags))
        if unit == "bool":
            value = bool(value)
        else:
            value = float(value)
        return Measurement(name, value, unit, SOURCE_VOXEL, True, float(min(max(conf, 0.0), 1.0)), prov(**extra),
                           flags=list(flags))

    out: Dict[str, Measurement] = {}
    # --- long axis ---------------------------------------------------------------------------------------------
    t_min, t_max = float(t.min()), float(t.max())
    out["long_axis_extent_mm"] = meas("long_axis_extent_mm", t_max - t_min + pitch, mask_conf, common_flags,
                                      measured_along="tooth_long_axis", t_range_mm_from_origin=[round(t_min, 3), round(t_max, 3)],
                                      panels_showing="long_axis, coronal, sagittal")
    # --- CEJ-referenced limits ---------------------------------------------------------------------------------
    if cej is None:
        out["coronal_limit_from_cej_mm"] = meas("coronal_limit_from_cej_mm", None, 0, common_flags + ["cej_unavailable"])
        out["apical_limit_from_cej_mm"] = meas("apical_limit_from_cej_mm", None, 0, common_flags + ["cej_unavailable"])
    else:
        out["coronal_limit_from_cej_mm"] = meas("coronal_limit_from_cej_mm", t_min - float(cej), cej_conf, common_flags + cej_flags,
                                                sign="negative = coronal to CEJ", cej_t_mm=round(float(cej), 3), landmark_source=landmark_source)
        out["apical_limit_from_cej_mm"] = meas("apical_limit_from_cej_mm", t_max - float(cej), cej_conf, common_flags + cej_flags,
                                               sign="positive = apical to CEJ", cej_t_mm=round(float(cej), 3), landmark_source=landmark_source)
    # --- bone crest along the axis (2nd percentile of bone voxels in a 1 mm shell around the tooth) --------------
    bpts = _bone_shell_points(seg, spacing, 1.0)
    crest_t = float(np.percentile((bpts - centre) @ axis, 2)) if len(bpts) >= 100 else None
    if crest_t is None:
        for k in ("coronal_limit_from_crest_mm", "apical_limit_from_crest_mm"):
            out[k] = meas(k, None, 0, common_flags + ["bone_crest_not_identified"])
    else:
        cflags = common_flags + ["bone_crest_is_automatic_estimate"]
        out["coronal_limit_from_crest_mm"] = meas("coronal_limit_from_crest_mm", t_min - crest_t, min(mask_conf, 0.6), cflags,
                                                  sign="positive = portal level below the bone crest", crest_t_mm=round(crest_t, 3))
        out["apical_limit_from_crest_mm"] = meas("apical_limit_from_crest_mm", t_max - crest_t, min(mask_conf, 0.6), cflags,
                                                 sign="positive = extends apical to the bone crest", crest_t_mm=round(crest_t, 3))
    # --- in-plane extents ---------------------------------------------------------------------------------------
    bl, md, axes_def = _in_plane_frame(axis, None if buccal_direction is None else np.asarray(buccal_direction, float),
                                       radial_local.mean(0))
    ext_bl = float(np.ptp(radial_vec @ bl) + pitch)
    ext_md = float(np.ptp(radial_vec @ md) + pitch)
    ip_conf = mask_conf if axes_def == "patient_buccal_direction" else min(mask_conf, 0.5)
    ip_flags = common_flags + ([] if axes_def == "patient_buccal_direction" else ["bl_md_axes_not_patient_oriented:" + axes_def])
    out["bucco_lingual_extent_mm"] = meas("bucco_lingual_extent_mm", ext_bl, ip_conf, ip_flags, axis_definition=axes_def,
                                          direction_zyx=bl.round(3).tolist(), panels_showing="axial_0..2, short_axis")
    out["mesio_distal_extent_mm"] = meas("mesio_distal_extent_mm", ext_md, ip_conf, ip_flags, axis_definition=axes_def,
                                         direction_zyx=md.round(3).tolist(), panels_showing="axial_0..2, short_axis")
    # --- depth from the root surface and circumferential extent --------------------------------------------------
    hull = tooth_hull_convex(seg.tooth)
    depth = ndi.distance_transform_edt(hull, sampling=spacing)
    out["max_depth_from_surface_mm"] = meas("max_depth_from_surface_mm", float(depth[tuple(q.T)].max()), mask_conf, common_flags,
                                            reference="convex hull of the tooth mask")
    ref = np.cross(axis, [1., 0., 0.])
    if np.linalg.norm(ref) < 1e-6:
        ref = np.cross(axis, [0., 1., 0.])
    ref /= np.linalg.norm(ref)
    ref2 = np.cross(axis, ref)
    ang = np.degrees(np.arctan2(radial_local @ ref2, radial_local @ ref))
    out["circumferential_extent_deg"] = meas("circumferential_extent_deg", _angular_spread_deg(ang), mask_conf, common_flags,
                                             measured_as="occupied 6-degree bins around the local tooth-axis centroid, 1-bin gaps closed",
                                             axis_point_mm=local_centre.round(2).tolist(),
                                             panels_showing="axial_1, short_axis")
    # --- dentine fraction ---------------------------------------------------------------------------------------
    dentine = seg.tooth & ~seg.canal
    n_dent = int(dentine.sum())
    if n_dent > 0:
        frac = float((m & dentine).sum() / n_dent)
        dflags = common_flags + ["denominator_is_tooth_mask_minus_canal_in_crop"]
        if touches_edge:
            dflags.append("tooth_mask_may_be_truncated_by_crop")
        out["dentine_volume_fraction_involved"] = meas("dentine_volume_fraction_involved", min(frac, 1.0), min(mask_conf, 0.7), dflags,
                                                       tooth_minus_canal_mm3=round(n_dent * vox_vol, 1))
    else:
        out["dentine_volume_fraction_involved"] = meas("dentine_volume_fraction_involved", None, 0, common_flags + ["no_dentine_mask"])
    # --- canal relation -----------------------------------------------------------------------------------------
    if seg.canal.any():
        canal_dist = ndi.distance_transform_edt(~seg.canal, sampling=spacing)
        dmin = float(canal_dist[tuple(q.T)].min())
        contact = bool((m & seg.canal).any()) or dmin <= pitch
        out["min_distance_to_canal_mm"] = meas("min_distance_to_canal_mm", 0.0 if contact else dmin, canal_conf, common_flags + canal_flags,
                                               canal_voxels=int(seg.canal.sum()), panels_showing="axial_1, long_axis, short_axis")
        out["root_canal_communication"] = meas("root_canal_communication", contact, canal_conf, common_flags + canal_flags,
                                               rule=f"candidate overlaps the canal mask or lies within one voxel ({pitch:.2f} mm) of it")
    else:
        out["min_distance_to_canal_mm"] = meas("min_distance_to_canal_mm", None, 0, common_flags + ["no_canal_mask"])
        out["root_canal_communication"] = meas("root_canal_communication", None, 0, common_flags + ["no_canal_mask"])
    out["lesion_volume_mm3"] = meas("lesion_volume_mm3", float(m.sum() * vox_vol), mask_conf, common_flags, voxels=int(m.sum()))
    return out


# --------------------------------------------------------------------------- VLM
def query_vlm_features(backend, cid: int, spacing, image_png: Optional[bytes], frame_meta: Optional[Dict[str, Any]] = None,
                       crop_width_mm: float = 12.0, tiles_only: bool = False) -> Dict[str, Any]:
    """One structured feature request.  Returns {"status": ok|abstained|failed|not_configured|no_image, ...}."""
    from ..agents.llm_backend import NullBackend
    if backend is None or isinstance(backend, NullBackend):
        return {"status": "not_configured", "backend": "none"}
    name = getattr(backend, "name", type(backend).__name__)
    if image_png is None or not getattr(backend, "supports_images", False):
        return {"status": "no_image", "backend": name,
                "reason": "no evidence frame available" if image_png is None else "backend does not accept images"}
    user = _feature_user_payload(cid, spacing, frame_meta, crop_width_mm, tiles_only)
    # entropy clustering on categorical feature fields (Nemotron backend samples several answers)
    old_key = getattr(backend, "cluster_key", None)
    if old_key is not None:
        backend.cluster_key = feature_cluster_key
    try:
        out = backend.complete_json(FEATURE_SYSTEM_PROMPT, user, VLM_FEATURE_RESPONSE_SCHEMA, images_png=[image_png])
    finally:
        if old_key is not None:
            backend.cluster_key = old_key
    if out is None:
        return {"status": "failed", "backend": name, "error": "backend returned nothing"}
    if "_error" in out:
        return {"status": "failed", "backend": name, "error": str(out["_error"])[:200]}
    try:
        clean, errors = validate_vlm_response(out)
    except FeatureSchemaError as e:
        return {"status": "failed", "backend": name, "error": f"schema: {e}", "raw_keys": sorted(out)[:12]}
    meta = {"status": "abstained" if out.get("_abstain") else "ok", "backend": out.get("_backend") or name,
            "dse": out.get("_dse"), "dse_threshold": out.get("_dse_threshold"), "images_sent": 1,
            "validation_errors": errors, "clean": clean}
    return meta


# --------------------------------------------------------------------------- reconcile
def _tolerance(kind: str, geo: float, pitch: float) -> float:
    if kind == "fraction":
        return 0.15
    return max(2.0 * pitch, 0.2 * abs(geo), 0.5)


def reconcile(geom: Dict[str, Measurement], estimates: Dict[str, Any], pitch: float) -> Tuple[List[str], List[str]]:
    """Attach model estimates to the geometric measurements in place.  Returns (unverified, disagreements)."""
    unverified: List[str] = []
    disagreements: List[str] = []
    for name, kind in ESTIMATE_KINDS.items():
        if name not in estimates:
            continue
        v = estimates[name]
        mm = geom.get(name)
        if mm is None:
            continue
        if v is None:
            mm.vlm_agreement = "not_reported"
            continue
        mm.vlm_value = v
        if mm.source == SOURCE_VOXEL and mm.value is not None:
            if kind == "bool":
                agree = bool(v) == bool(mm.value)
                mm.discrepancy = None
            else:
                mm.discrepancy = float(v) - float(mm.value)
                agree = abs(mm.discrepancy) <= _tolerance(kind, float(mm.value), pitch)
            mm.vlm_agreement = "agrees" if agree else "disagrees"
            if not agree:
                mm.flags.append("vlm_value_disagrees_with_geometry_geometry_kept")
                disagreements.append(name)
        else:
            # geometry could not measure this quantity: keep the model's value but mark it as unverifiable
            mm.value = v
            mm.source = SOURCE_VLM_UNVERIFIED
            mm.verified = False
            mm.confidence = min(mm.confidence if mm.confidence > 0 else 0.3, 0.3)
            mm.vlm_agreement = "not_comparable"
            mm.flags.append("vlm_value_not_geometrically_verifiable")
            unverified.append(name)
    return unverified, disagreements


# --------------------------------------------------------------------------- bundle
def measure_candidate_features(raw: np.ndarray, seg: ToothSegmentation, labels: np.ndarray, cid: int, spacing,
                               geometry: Dict[str, Any], *, backend=None, frame_png: Optional[bytes] = None,
                               tiles_png: Optional[bytes] = None, frame_meta: Optional[Dict[str, Any]] = None,
                               crop_offset_zyx=(0, 0, 0), source_id: str = "case", descriptors=None,
                               anatomy_caveats: Optional[List[str]] = None, buccal_direction=None) -> FeatureBundle:
    """Geometry first, model second, reconciliation last.  Works without a model (geometry-only bundle)."""
    spacing_arr = np.asarray(spacing, float)
    geom = measure_candidate_geometry(raw, seg, labels, cid, spacing_arr, geometry, crop_offset_zyx, descriptors,
                                      anatomy_caveats, buccal_direction, frame_meta)
    image = frame_png if frame_png is not None else tiles_png
    vlm = query_vlm_features(backend, cid, spacing_arr, image, frame_meta, tiles_only=frame_png is None and tiles_png is not None)
    caveats = ["measurements_from_automatic_candidate_mask", "visual_features_are_model_readings_of_one_frame_not_findings"]
    visual = VisualFeatures()
    unverified: List[str] = []
    disagreements: List[str] = []
    if vlm["status"] in ("ok", "abstained"):
        clean = vlm.pop("clean")
        vlm["raw_estimates"] = clean["estimates"]
        if vlm["status"] == "ok":
            visual = VisualFeatures.from_clean(clean)
            unverified, disagreements = reconcile(geom, clean["estimates"], float(spacing_arr.min()))
        else:
            caveats.append("vlm_abstained_samples_disagreed_visual_features_not_used")
            vlm["abstained_visual_features"] = clean["visual_features"]
    elif vlm["status"] == "not_configured":
        caveats.append("no_vision_model_configured_geometry_only")
    elif vlm["status"] == "no_image":
        caveats.append("vision_model_received_no_image_geometry_only")
    else:
        caveats.append("vision_model_call_failed_geometry_only")
    if unverified:
        caveats.append("some_values_are_model_estimates_without_geometric_verification")
    if disagreements:
        caveats.append("model_estimates_disagree_with_geometry_for_some_values")
    if any(not m.verified and m.source == SOURCE_UNAVAILABLE for m in geom.values()):
        caveats.append("some_measurements_unavailable_missing_landmark_or_mask")
    return FeatureBundle(candidate_id=int(cid), source_id=str(source_id), spacing_zyx_mm=spacing_arr.round(4).tolist(),
                         visual=visual, measurements=geom, vlm=vlm, unverified_vlm_values=unverified,
                         vlm_geometry_disagreements=disagreements, caveats=caveats)


def bundle_summary_line(b: FeatureBundle) -> str:
    """One transcript line for the conference."""
    def val(name, fmt="{:.1f}"):
        m = b.measurements.get(name)
        if m is None or m.value is None:
            return "n/a"
        s = fmt.format(m.value) if not isinstance(m.value, bool) else ("yes" if m.value else "no")
        return s + ("" if m.verified else "*")
    parts = [f"long axis {val('long_axis_extent_mm')} mm", f"MD {val('mesio_distal_extent_mm')} / BL {val('bucco_lingual_extent_mm')} mm",
             f"CEJ->coronal limit {val('coronal_limit_from_cej_mm')} mm", f"below crest {val('apical_limit_from_crest_mm')} mm",
             f"dentine fraction {val('dentine_volume_fraction_involved', '{:.2f}')}",
             f"canal distance {val('min_distance_to_canal_mm', '{:.2f}')} mm", f"canal communication {val('root_canal_communication')}"]
    vlm = b.vlm.get("status")
    tail = f"; VLM {vlm}" + (f", {len(b.unverified_vlm_values)} unverified, {len(b.vlm_geometry_disagreements)} disagreeing" if vlm == "ok" else "")
    return "; ".join(parts) + tail + (" (* = model value, not geometrically verified)" if any(
        not m.verified and m.value is not None for m in b.measurements.values()) else "")
