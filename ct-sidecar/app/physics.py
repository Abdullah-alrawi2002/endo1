from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage.graph import route_through_array

from .models import DensityInfo


@dataclass
class PhysicsResult:
    apex_zyx: tuple[int, int, int]
    working_length_mm: float | None
    working_length_method: str | None
    density_drop_percent: float | None
    """True when apex mean is lower than same-scan reference (observation only)."""
    candidate_low_attenuation: bool | None
    density: DensityInfo
    warnings: list[str]


def _arch(target_tooth: str) -> str:
    number = int(target_tooth.strip().lstrip("#"))
    return "upper" if number <= 16 else "lower"


def _endpoint_centroid(
    mask: np.ndarray, image: sitk.Image, target_tooth: str, apical: bool
) -> tuple[int, int, int]:
    coords_zyx = np.argwhere(mask)
    coords_xyz = coords_zyx[:, ::-1].astype(float)
    spacing = np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    origin = np.asarray(image.GetOrigin())
    physical = origin + (direction @ (coords_xyz * spacing).T).T
    superior = physical[:, 2]

    upper = _arch(target_tooth) == "upper"
    choose_high = upper if apical else not upper
    cutoff = np.percentile(superior, 95 if choose_high else 5)
    selected = coords_zyx[superior >= cutoff] if choose_high else coords_zyx[superior <= cutoff]
    center = np.rint(selected.mean(axis=0)).astype(int)
    return tuple(int(v) for v in center)


def _nearest_mask_voxel(mask: np.ndarray, point: tuple[int, int, int]) -> tuple[int, int, int]:
    if mask[point]:
        return point
    _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
    return tuple(int(indices[axis][point]) for axis in range(3))


def estimate_low_density_path_length(
    volume: np.ndarray,
    tooth_mask: np.ndarray,
    image: sitk.Image,
    target_tooth: str,
) -> tuple[float | None, str | None, tuple[int, int, int], list[str]]:
    """
    Estimate canal length through a low-density path constrained to the tooth.
    This is an operational estimate, not an electronic/radiographic working
    length and cannot resolve individual canals in multi-rooted teeth.
    """
    warnings: list[str] = []
    apex = _nearest_mask_voxel(
        tooth_mask, _endpoint_centroid(tooth_mask, image, target_tooth, apical=True)
    )
    coronal = _nearest_mask_voxel(
        tooth_mask, _endpoint_centroid(tooth_mask, image, target_tooth, apical=False)
    )

    coords = np.argwhere(tooth_mask)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = np.minimum(coords.max(axis=0) + 3, np.asarray(tooth_mask.shape))
    crop = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    local_mask = tooth_mask[crop]
    local_volume = volume[crop]
    start = tuple(int(coronal[i] - lo[i]) for i in range(3))
    end = tuple(int(apex[i] - lo[i]) for i in range(3))

    tooth_values = local_volume[local_mask]
    if tooth_values.size < 100:
        return None, None, apex, ["Target tooth mask is too small for working-length estimation."]
    p5, p95 = np.percentile(tooth_values, [5, 95])
    if p95 <= p5:
        return None, None, apex, ["Target tooth has insufficient intensity contrast."]

    normalized = np.clip((local_volume - p5) / (p95 - p5), 0, 1)
    cost = 1.0 + normalized * 4.0
    cost[~local_mask] = 1_000_000.0
    try:
        path, _ = route_through_array(
            cost, start, end, fully_connected=True, geometric=True
        )
    except (ValueError, IndexError):
        return None, None, apex, ["A continuous low-density tooth path could not be computed."]

    path_arr = np.asarray(path, dtype=float)
    if path_arr.shape[0] < 2:
        return None, None, apex, ["Working-length path was degenerate."]
    # ndarray axes are z,y,x; SimpleITK spacing is x,y,z.
    spacing_zyx = np.asarray(image.GetSpacing())[::-1]
    length = np.linalg.norm(np.diff(path_arr, axis=0) * spacing_zyx, axis=1).sum()
    warnings.append(
        "Working length is a low-density centerline estimate from CBCT, not a clinical working length; verify with an apex locator and working-length radiograph."
    )
    return (
        round(float(length), 2),
        "3D minimum-cost path through low-radiodensity voxels constrained to the isolated tooth",
        apex,
        warnings,
    )


def _sphere(
    shape: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    inner_mm: float,
    outer_mm: float,
) -> np.ndarray:
    spacing_zyx = np.asarray(spacing_xyz)[::-1]
    radii = np.ceil(outer_mm / spacing_zyx).astype(int)
    z, y, x = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    distance = np.sqrt(
        (z * spacing_zyx[0]) ** 2
        + (y * spacing_zyx[1]) ** 2
        + (x * spacing_zyx[2]) ** 2
    )
    kernel = (distance <= outer_mm) & (distance >= inner_mm)
    result = np.zeros(shape, dtype=bool)
    center = np.asarray(center_zyx)
    starts = np.maximum(center - radii, 0)
    ends = np.minimum(center + radii + 1, np.asarray(shape))
    kernel_starts = starts - (center - radii)
    kernel_ends = kernel_starts + (ends - starts)
    result[tuple(slice(starts[i], ends[i]) for i in range(3))] = kernel[
        tuple(slice(kernel_starts[i], kernel_ends[i]) for i in range(3))
    ]
    return result


def calculate_apical_density(
    volume: np.ndarray,
    segmentation: np.ndarray,
    tooth_mask: np.ndarray,
    apex_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    target_tooth: str,
    is_hu: bool,
) -> tuple[float | None, bool | None, DensityInfo, list[str]]:
    warnings: list[str] = []
    upper = _arch(target_tooth) == "upper"
    jaw_mask = segmentation == (1 if upper else 2)
    all_teeth = (segmentation == 3) | (segmentation == 4)

    apex_sphere = _sphere(volume.shape, apex_zyx, spacing_xyz, 0.0, 3.0)
    reference_shell = _sphere(volume.shape, apex_zyx, spacing_xyz, 4.0, 8.0)
    apex_values = volume[apex_sphere & jaw_mask & ~tooth_mask]
    reference_values = volume[reference_shell & jaw_mask & ~all_teeth]

    unit = "HU" if is_hu else "GV"
    method = (
        "3 mm apical sphere versus robust 4–8 mm adjacent cancellous-bone shell; "
        "teeth excluded; same-scan relative attenuation"
    )
    if apex_values.size < 20 or reference_values.size < 50:
        density = DensityInfo(
            apexMean=None,
            referenceMean=None,
            unit=unit,
            method=method,
            valid=False,
        )
        return (
            None,
            None,
            density,
            [
                "Insufficient jaw-bone voxels for a valid apex/reference density comparison."
            ],
        )

    # Robustly reject air, cortical outliers, metal, and streaks. The remaining
    # central distribution approximates cancellous bone in the local shell.
    ref_low, ref_high = np.percentile(reference_values, [20, 80])
    robust_ref = reference_values[
        (reference_values >= ref_low) & (reference_values <= ref_high)
    ]
    apex_low, apex_high = np.percentile(apex_values, [5, 95])
    robust_apex = apex_values[(apex_values >= apex_low) & (apex_values <= apex_high)]
    reference_mean = float(np.mean(robust_ref))
    apex_mean = float(np.mean(robust_apex))

    if reference_mean <= 0:
        density = DensityInfo(
            apexMean=round(apex_mean, 2),
            referenceMean=round(reference_mean, 2),
            unit=unit,
            method=method,
            valid=False,
        )
        return (
            None,
            None,
            density,
            [
                "Reference mean is non-positive, so the requested ratio is mathematically unstable. Review scan calibration and ROI location."
            ],
        )

    drop = ((reference_mean - apex_mean) / reference_mean) * 100.0
    # Observational only: apex lower than same-scan reference. No universal
    # percentage threshold (CBCT gray values are scanner/protocol-dependent).
    candidate_low = apex_mean < reference_mean
    if not is_hu:
        warnings.append(
            "CBCT values are scanner-dependent gray values, not absolute HU; "
            "relativeAttenuationDropPercent is an observation for clinician review, not a diagnostic criterion."
        )
    density = DensityInfo(
        apexMean=round(apex_mean, 2),
        referenceMean=round(reference_mean, 2),
        unit=unit,
        method=method,
        valid=True,
    )
    return round(drop, 2), candidate_low, density, warnings


def run_physics(
    image: sitk.Image,
    segmentation_image: sitk.Image,
    tooth_mask: np.ndarray,
    target_tooth: str,
    is_hu: bool,
) -> PhysicsResult:
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    segmentation = sitk.GetArrayFromImage(segmentation_image)
    length, length_method, apex, warnings = estimate_low_density_path_length(
        volume, tooth_mask, image, target_tooth
    )
    drop, candidate_low, density, density_warnings = calculate_apical_density(
        volume,
        segmentation,
        tooth_mask,
        apex,
        image.GetSpacing(),
        target_tooth,
        is_hu,
    )
    warnings.extend(density_warnings)
    return PhysicsResult(
        apex_zyx=apex,
        working_length_mm=length,
        working_length_method=length_method,
        density_drop_percent=drop,
        candidate_low_attenuation=candidate_low,
        density=density,
        warnings=warnings,
    )


def generate_mpr_images(
    image: sitk.Image, center_zyx: tuple[int, int, int], output_dir: Path
) -> dict[str, Path]:
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    low, high = np.percentile(volume[np.isfinite(volume)], [1, 99])
    width = max(high - low, 1.0)
    spacing_x, spacing_y, spacing_z = image.GetSpacing()

    def to_png(
        array: np.ndarray, name: str, row_spacing: float, column_spacing: float
    ) -> Path:
        normalized = np.clip((array - low) / width, 0, 1)
        pixels = (normalized * 255).astype(np.uint8)
        rendered = Image.fromarray(pixels)
        physical_width = array.shape[1] * column_spacing
        physical_height = array.shape[0] * row_spacing
        scale = min(768 / max(physical_width, physical_height), 4.0)
        rendered = rendered.resize(
            (
                max(1, round(physical_width * scale)),
                max(1, round(physical_height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )
        path = output_dir / f"{name}.png"
        rendered.save(path)
        return path

    z, y, x = center_zyx
    return {
        "axial": to_png(volume[z, :, :], "axial", spacing_y, spacing_x),
        "coronal": to_png(volume[:, y, :], "coronal", spacing_z, spacing_x),
        "sagittal": to_png(volume[:, :, x], "sagittal", spacing_z, spacing_y),
    }
