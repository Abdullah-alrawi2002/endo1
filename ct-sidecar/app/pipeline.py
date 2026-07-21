from __future__ import annotations

import os
from pathlib import Path

import pydicom
import SimpleITK as sitk

from .dicom_io import load_dicom_volume
from .models import CTSupport
from .physics import generate_mpr_images, run_physics
from .segmentation import isolate_target_tooth, run_dental_segmentator
from .vision import analyze_mpr


def _dicom_modality(files: list[str]) -> str:
    try:
        dataset = pydicom.dcmread(files[0], stop_before_pixels=True)
        return str(getattr(dataset, "Modality", "")).strip().upper() or "CBCT"
    except Exception:
        return "CBCT"


def _quality_gates(image: sitk.Image, modality: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    size = image.GetSize()
    spacing = image.GetSpacing()
    if min(size) < 16:
        failures.append("Volume is too small for reliable apical assessment")
    if max(spacing) > 0.4:
        failures.append(
            f"Largest voxel spacing is {max(spacing):.3f} mm; CBCT resolution may be insufficient"
        )
    if modality not in {"CT", "CBCT", "DX", "OT", ""}:
        failures.append(f"Unexpected DICOM modality {modality}")
    return len(failures) == 0, failures


def analyze_ct_series(
    input_dir: Path,
    work_dir: Path,
    target_tooth: str,
    clinician_seed_provided: bool,
) -> CTSupport:
    if not clinician_seed_provided:
        return CTSupport(
            status="failed",
            targetToothUniversal=int(target_tooth),
            clinicianSeedProvided=False,
            qualityGatePassed=False,
            qualityGateFailures=[
                "Clinician tooth seed/identification is required; automated tooth numbering is disabled."
            ],
            artifactWarnings=[],
            clinicianReviewed=False,
        )

    image, dicom_files = load_dicom_volume(input_dir)
    image = sitk.Cast(image, sitk.sitkFloat32)
    modality = _dicom_modality(dicom_files)
    is_hu = modality == "CT" and os.getenv(
        "CT_ASSUME_CALIBRATED_HU", "false"
    ).lower() in {"1", "true", "yes"}
    quality_ok, quality_failures = _quality_gates(image, modality)
    warnings: list[str] = []
    if modality == "CT" and not is_hu:
        warnings.append(
            "DICOM reports Modality=CT, but values are treated as relative GV unless CT_ASSUME_CALIBRATED_HU=true."
        )

    segmentation_image, segmentation_info, seg_warnings = run_dental_segmentator(
        image, work_dir
    )
    warnings.extend(seg_warnings)

    if segmentation_image is None:
        return CTSupport(
            status="partial",
            targetToothUniversal=int(target_tooth),
            clinicianSeedProvided=True,
            qualityGatePassed=quality_ok,
            qualityGateFailures=quality_failures,
            artifactWarnings=warnings,
            morphologyNotes=(
                f"Segmentation {segmentation_info.status}: {segmentation_info.method}"
            ),
            clinicianReviewed=False,
        )

    if (
        segmentation_image.GetSize() != image.GetSize()
        or segmentation_image.GetSpacing() != image.GetSpacing()
    ):
        segmentation_image = sitk.Resample(
            segmentation_image,
            image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

    # Isolation remains heuristic and experimental; results are non-diagnostic.
    tooth_mask, isolation_method, isolation_warnings = isolate_target_tooth(
        segmentation_image, target_tooth
    )
    warnings.extend(isolation_warnings)
    warnings.append(
        f"Tooth isolation method (experimental): {isolation_method}. Clinician must verify."
    )

    if tooth_mask is None or not quality_ok:
        return CTSupport(
            status="partial",
            targetToothUniversal=int(target_tooth),
            clinicianSeedProvided=True,
            qualityGatePassed=quality_ok,
            qualityGateFailures=quality_failures,
            artifactWarnings=warnings,
            morphologyNotes="Target tooth could not be isolated safely or quality gate failed.",
            clinicianReviewed=False,
        )

    physics = run_physics(
        image, segmentation_image, tooth_mask, target_tooth, is_hu
    )
    warnings.extend(physics.warnings)

    mpr_dir = work_dir / "mpr"
    mpr_dir.mkdir()
    mpr_images = generate_mpr_images(image, physics.apex_zyx, mpr_dir)
    vertucci, vision, vision_warnings = analyze_mpr(mpr_images, target_tooth)
    warnings.extend(vision_warnings)

    # Relative drop is reported as a quantitative observation, never a diagnostic flag.
    # candidateLowAttenuationRegion means apex mean < same-scan reference — clinician review required.
    candidate = physics.candidate_low_attenuation
    if physics.density_drop_percent is not None:
        warnings.append(
            "relativeAttenuationDropPercent is scanner/protocol-dependent on CBCT "
            "and must not be interpreted with a universal percentage threshold."
        )

    return CTSupport(
        status="completed" if physics.density.valid else "partial",
        targetToothUniversal=int(target_tooth),
        clinicianSeedProvided=True,
        qualityGatePassed=quality_ok,
        qualityGateFailures=quality_failures,
        candidateLowAttenuationRegion=candidate,
        candidateLocation=f"Apical region near estimated apex index {physics.apex_zyx}",
        candidateVolumeMm3=None,
        relativeAttenuationDropPercent=physics.density_drop_percent,
        artifactWarnings=warnings,
        morphologyNotes=vision.morphologyNotes,
        vertucciScreen=vertucci,
        canalLengthEstimateMm=physics.working_length_mm,
        canalLengthEstimateNote=(
            "Anatomical low-density path estimate only — not a clinical working length; "
            "confirm with apex locator and working-length radiograph."
        ),
        clinicianReviewed=False,
    )
