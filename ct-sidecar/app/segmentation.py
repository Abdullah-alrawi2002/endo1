from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .models import SegmentationInfo


MODEL_NAME = "DentalSegmentator nnU-Net v2 (Dataset112)"


def _universal_to_arch_position(target_tooth: str) -> tuple[str, int]:
    cleaned = target_tooth.strip().lstrip("#")
    try:
        number = int(cleaned)
    except ValueError as exc:
        raise ValueError("Target tooth must use Universal numbering (1–32)") from exc
    if not 1 <= number <= 32:
        raise ValueError("Target tooth must use Universal numbering (1–32)")
    if number <= 16:
        return "upper", number - 1  # right -> left
    return "lower", 32 - number  # right -> left


def run_dental_segmentator(
    image: sitk.Image, work_dir: Path
) -> tuple[sitk.Image | None, SegmentationInfo, list[str]]:
    """
    Execute the published DentalSegmentator weights through nnUNet v2.

    The Docker image includes nnUNet, but model weights are intentionally
    mounted separately via nnUNet_results. This avoids silently downloading a
    230 MB clinical model at runtime and makes the exact weights auditable.
    """
    warnings: list[str] = []
    executable = shutil.which("nnUNetv2_predict")
    results_dir = os.getenv("nnUNet_results")
    if not executable or not results_dir:
        return (
            None,
            SegmentationInfo(
                status="unavailable",
                model=MODEL_NAME,
                targetToothLocalized=False,
                method="nnUNet v2 unavailable or nnUNet_results is not mounted",
            ),
            [
                "DentalSegmentator was not run. Install/mount Dataset112 weights and set nnUNet_results."
            ],
        )

    input_dir = work_dir / "nnunet-input"
    output_dir = work_dir / "nnunet-output"
    input_dir.mkdir()
    output_dir.mkdir()
    sitk.WriteImage(image, str(input_dir / "case_0000.nii.gz"))

    command = [
        executable,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        os.getenv("NNUNET_DATASET_ID", "112"),
        "-c",
        os.getenv("NNUNET_CONFIGURATION", "3d_fullres"),
        "-f",
        os.getenv("NNUNET_FOLDS", "all"),
        "--disable_tta",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("CT_SEGMENTATION_TIMEOUT_SECONDS", "1200")),
        )
        result = sitk.ReadImage(str(output_dir / "case.nii.gz"))
        return (
            result,
            SegmentationInfo(
                status="completed",
                model=MODEL_NAME,
                targetToothLocalized=False,
                method="DentalSegmentator five-class anatomical segmentation",
            ),
            warnings,
        )
    except (subprocess.SubprocessError, RuntimeError, OSError) as exc:
        return (
            None,
            SegmentationInfo(
                status="failed",
                model=MODEL_NAME,
                targetToothLocalized=False,
                method="DentalSegmentator execution failed",
            ),
            [f"DentalSegmentator failed: {str(exc)[:300]}"],
        )


def isolate_target_tooth(
    segmentation: sitk.Image, target_tooth: str
) -> tuple[np.ndarray | None, str, list[str]]:
    """
    Isolate a tooth candidate from DentalSegmentator's bulk upper/lower-teeth
    class using connected components and Universal-number ordering.

    DentalSegmentator does not provide per-tooth instance labels. This heuristic
    is therefore explicitly low-confidence and must be reviewed. If contacts
    merge components or teeth are missing, we refuse rather than fabricate.
    """
    arch, arch_position = _universal_to_arch_position(target_tooth)
    array = sitk.GetArrayFromImage(segmentation)
    label_value = 3 if arch == "upper" else 4
    binary = array == label_value
    labels, count = ndimage.label(binary)
    objects = ndimage.find_objects(labels)
    components: list[tuple[float, int, np.ndarray]] = []

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        coords = np.argwhere(labels == component_id)
        if coords.shape[0] < 100:
            continue
        centroid_zyx = coords.mean(axis=0)
        index_xyz = tuple(float(v) for v in centroid_zyx[::-1])
        physical_x = segmentation.TransformContinuousIndexToPhysicalPoint(index_xyz)[0]
        components.append((physical_x, component_id, labels == component_id))

    components.sort(key=lambda item: item[0])  # patient right -> left in LPS
    warnings: list[str] = []
    if len(components) < 8:
        return (
            None,
            "Target isolation unavailable: bulk tooth mask did not separate into enough components",
            [
                f"Only {len(components)} separate {arch} tooth components were found; contacts or restorations may have merged teeth."
            ],
        )

    # Map the requested ordinal across the components that are actually present.
    # This supports partial-FOV scans but is unsafe with missing teeth, so it is
    # always marked for clinician review.
    index = round((arch_position / 15) * (len(components) - 1))
    index = min(max(index, 0), len(components) - 1)
    warnings.append(
        "Target tooth was isolated by spatial ordering of connected components; verify it because DentalSegmentator supplies arch-level, not per-tooth, labels."
    )
    return (
        components[index][2],
        "Connected-component isolation ordered by Universal tooth position",
        warnings,
    )
