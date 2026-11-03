"""End-to-end pipeline: DICOM folder -> structured recommendation report."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Callable, Dict, Optional

import numpy as np

from .dicom_loader import CBCTVolume, load_dicom_series
from .preprocess import preprocess_volume, crop_to_bbox
from .segmentation import ToothSegmentation, ToothSegmenter, LesionSegmenter, lesion_mask_from_probability
from .patel_classifier import classify_patel
from .decision_matrix import ClinicalModifiers, recommend_from_matrix, END_STAGE_CODES, end_stage_matrix
from .rl_policy import ClinicalContext, ConstrainedBanditPolicy


def run_pipeline(dicom_dir: Optional[str],
                 tooth_segmenter: ToothSegmenter,
                 lesion_segmenter: LesionSegmenter,
                 modifiers: Optional[ClinicalModifiers] = None,
                 policy: Optional[ConstrainedBanditPolicy] = None,
                 volume: Optional[CBCTVolume] = None,
                 lesion_threshold: float = 0.5,
                 tooth_type: str = "unknown",
                 age_years: Optional[float] = None,
                 buccal_direction: Optional[np.ndarray] = None,
                 expected_slices: Optional[int] = None,
                 case_id: Optional[str] = None,
                 model_card: Optional[Dict] = None,
                 lesion_source: str = "model",
                 roi_margin_mm: float = 3.0,
                 ecr_confirmed: bool = False) -> Dict:
    """Run the full CDS chain and return a JSON-serialisable report.

    Either `dicom_dir` or a pre-loaded `volume` must be supplied.
    """
    t0 = time.time()
    if volume is None:
        if dicom_dir is None:
            raise ValueError("Provide dicom_dir or volume")
        volume = load_dicom_series(dicom_dir, expected_slices=expected_slices)
    qa = _quality_checks(volume)

    # 1. tooth / canal / bone
    seg: ToothSegmentation = tooth_segmenter(volume.array, volume.spacing)
    # 1b. crop to the target tooth + margin (bone crest shell, lesion halo); keeps a
    #     500-slice volume within a few hundred MB instead of several GB of float maps
    margin_vox = int(round(roi_margin_mm / min(volume.spacing)))
    sl = crop_to_bbox(seg.tooth, margin=margin_vox)
    array = volume.array[sl]
    seg = ToothSegmentation(tooth=seg.tooth[sl], canal=seg.canal[sl], bone=seg.bone[sl],
                            cej_z=seg.cej_z, apex_z=seg.apex_z, intensity=array.astype(np.float32))
    roi_bbox = [[int(s_.start), int(s_.stop)] for s_ in sl]
    # 2. intensity normalisation inside the tooth ROI
    norm = preprocess_volume(array, roi=seg.tooth)
    # 3. lesion probability -> mask
    prob = lesion_segmenter(norm, seg)
    if prob.shape != array.shape and prob.shape == volume.array.shape:
        prob = prob[sl]  # segmenter returned a full-volume map (e.g. expert mask)
    if lesion_source == "ecr_candidate":
        if not hasattr(lesion_segmenter, "last_report"):
            raise ValueError("Pass a CervicalResorptionDetector instance for ecr_candidate")
        report = dict(lesion_segmenter.last_report)
        offset = np.array([s.start for s in sl])
        for c in report["candidates"]:
            c["centroid_volume_zyx"] = (np.asarray(c["centroid_crop_zyx"])+offset).tolist()
            c["bbox_volume_zyx"] = [[a+int(o), b+int(o)]
                                    for (a, b), o in zip(c["bbox_crop_zyx"], offset)]
        report.update(case_id=case_id or volume.metadata.get("SeriesInstanceUID_hash"),
                      qa=qa, acquisition=volume.metadata, roi_bbox_zyx=roi_bbox,
                      runtime_s=round(time.time()-t0, 2))
        return report
    lesion = lesion_mask_from_probability(prob, seg, threshold=lesion_threshold)
    evidence_level, evidence_note = _evidence_level(model_card, lesion_source)
    if not ecr_confirmed:
        return {"status": "ecr_not_confirmed", "case_id": case_id or volume.metadata.get("SeriesInstanceUID_hash"),
                "qa": qa, "acquisition": volume.metadata, "roi_bbox_zyx": roi_bbox,
                "evidence_level": evidence_level, "evidence_note": evidence_note,
                "candidate_volume_mm3": float(lesion.sum()*np.prod(volume.spacing)),
                "ecr_diagnosis": None, "patel": None, "decision": None,
                "requires_confirmation": True,
                "note": "A segmentation is not an ECR diagnosis. Confirm ECR and review anatomy before "
                        "classification or treatment planning. A negative segmentation does not exclude ECR."}
    if lesion.sum() == 0:
        return {"status": "no_lesion_detected", "qa": qa, "acquisition": volume.metadata, "roi_bbox_zyx": roi_bbox,
                "case_id": case_id or volume.metadata.get("SeriesInstanceUID_hash"),
                "evidence_level": evidence_level, "evidence_note": evidence_note,
                "note": "No ECR lesion above threshold. Sub-millimetre lesions have low "
                        "sensitivity on CBCT AI (Al Fartousi 2026: 33.6 %); clinical correlation advised."}
    # 4. Patel 3D parameters
    patel = classify_patel(lesion, seg, volume.spacing, volume_norm=norm,
                           cej_level_mm=None, buccal_direction=buccal_direction)
    # 5. rule matrix -> safe set
    mods = modifiers or ClinicalModifiers()
    rec = recommend_from_matrix(patel.code, mods, portal_angle_deg=patel.portal_angle_deg,
                                calcified_fraction=patel.calcified_fraction)
    # 6. RL re-ranking inside the safe set
    policy = policy or ConstrainedBanditPolicy()
    ctx = ClinicalContext(patel=patel, mods=mods, tooth_type=tooth_type, age_years=age_years)
    decision = policy.decide(ctx, rec)

    report = {
        "status": "provisional" if (evidence_level != "validated" or rec.missing_inputs) else "ok",
        "case_id": case_id or volume.metadata.get("SeriesInstanceUID_hash"),
        "evidence_level": evidence_level,
        "evidence_note": evidence_note,
        "runtime_s": round(time.time() - t0, 2),
        "acquisition": volume.metadata,
        "qa": qa,
        "roi_bbox_zyx": roi_bbox,
        "patel": patel.to_dict(),
        "modifiers": {k: v for k, v in asdict(mods).items() if v is not None},
        "tooth_type": tooth_type,
        "age_years": age_years,
        "lesion_threshold": lesion_threshold,
        "rule_recommendation": rec.to_dict(),
        "decision": decision.to_dict(),
        "end_stage": patel.code in END_STAGE_CODES,
        "end_stage_matrix": end_stage_matrix() if patel.code in END_STAGE_CODES else None,
        "disclaimer": "Advisory output. Irreversible options require clinician confirmation. "
                      "Final decision rests with the treating clinician.",
    }
    return report


def _evidence_level(model_card: Optional[Dict], lesion_source: str):
    if lesion_source == "radiolucency_rule":
        return "rule_based_radiolucency", ("Lesion = rule-based hypodense defect inside the tooth hull reaching the external "
                                           "surface (no learned model). ECR, caries, cervical abrasion and artefacts are not "
                                           "distinguished; sensitivity unknown. Research output, not for clinical use.")
    if lesion_source == "expert_mask":
        return "expert_mask", "Lesion mask drawn/reviewed by a clinician; measurements depend on that mask."
    if model_card is None:
        return "unvalidated_model", "Lesion model has no training card; performance unknown."
    data = str(model_card.get("data", "")).lower()
    if "synthetic" in data or "phantom" in data:
        return "research_phantom_model", ("Lesion model trained on SYNTHETIC phantoms only; no clinical "
                                          "performance is established. Research output, not for clinical use.")
    if model_card.get("external_validation"):
        return "validated", "Lesion model externally validated (see card)."
    return "internal_model", "Lesion model trained on clinical data without external validation."


def _quality_checks(v: CBCTVolume) -> Dict:
    vox = max(v.spacing)
    issues = []
    if vox > 0.2:
        issues.append(f"Voxel size {vox:.3f} mm > 0.2 mm: ECR proximity (d/p) calls are unreliable; "
                      "recommend small-FOV high-resolution rescan (≤0.1 mm).")
    if not v.metadata.get("Isotropic", True):
        issues.append("Anisotropic voxels: all measurements use per-axis spacing, but partial-volume "
                      "effects differ by axis; interpret d/p margins conservatively.")
    if v.metadata.get("PaddingVoxels", 0):
        issues.append(f"{v.metadata['PaddingVoxels']} padding voxels set to the minimum intensity (excluded from tissue).")
    if v.array.ndim != 3 or min(v.array.shape) < 50:
        issues.append("Volume too small or not 3D.")
    return {"voxel_mm": vox, "spacing_mm": list(v.spacing), "slices": int(v.array.shape[0]),
            "shape_zyx": list(v.array.shape), "fov_mm": list(v.metadata.get("FieldOfView_mm", [])),
            "intensity_range": list(v.metadata.get("IntensityRange", [])),
            "max_slice_position_residual_mm": v.metadata.get("MaxSlicePositionResidual_mm"),
            "housekeeping": v.metadata.get("Housekeeping", {}), "issues": issues,
            "note": "CBCT grey values are not Hounsfield-calibrated; all thresholds are relative."}


def to_json(report: Dict, path: str):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
