"""Documented, reproducible display and intensity-reference transforms.

CBCT grey values are scanner-specific and are *not* Hounsfield units, so no absolute cut-off is used anywhere
in this package: every reference level is derived from the volume itself (Otsu splits, tissue percentiles,
modes).  Display transforms (windowing, CLAHE) are for human / VLM viewing only and are recorded alongside
the image; measurements are always taken from the unmodified voxel data.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


def tissue_reference_levels(raw: np.ndarray, tooth: np.ndarray, canal: Optional[np.ndarray], t_dense: float) -> Dict[str, float]:
    """Soft-tissue, dentine, enamel/dense and metal reference levels, all relative to this volume."""
    tooth = tooth.astype(bool)
    dent = tooth & ~(canal.astype(bool) if canal is not None else False)
    dent_v = raw[dent] if dent.any() else raw[tooth]
    dentine_mode = float(np.median(dent_v)) if dent_v.size else float(np.median(raw))
    soft = float(np.percentile(raw[~tooth], 50)) if (~tooth).any() else float(raw.min())
    hard = raw[raw >= t_dense]
    if hard.size >= 50:
        hist, edges = np.histogram(hard, bins=64)
        enamel_mode = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    else:
        enamel_mode = float(t_dense)
    # metal: well above the enamel mode by a multiple of the dentine->enamel contrast, bounded by the tail of the hard voxels
    contrast = max(enamel_mode - dentine_mode, 1e-6)
    metal_level = max(enamel_mode + 2.0 * contrast, float(np.percentile(hard, 99.5)) if hard.size >= 50 else enamel_mode + 2.0 * contrast)
    return {"soft_tissue": soft, "dentine_mode": dentine_mode, "t_dense": float(t_dense), "enamel_mode": enamel_mode,
            "metal_level": float(metal_level), "method": "volume-relative (median soft tissue, dentine median, hard-voxel mode, "
                                                        "metal = enamel_mode + 2x dentine-enamel contrast or 99.5th pct of hard voxels)"}


def metal_mask(raw: np.ndarray, metal_level: float, spacing, exclude: Optional[np.ndarray] = None,
               streak_mm: float = 3.0) -> Dict[str, np.ndarray]:
    """Metal core and a 'streak zone' (distance band) where intensities are unreliable.

    `exclude` (e.g. the root-filling canal proxy) is not counted as metal core but still gets a streak zone.
    """
    core = raw >= metal_level
    if exclude is not None:
        core_ex = core & ~ndi.binary_dilation(exclude.astype(bool), iterations=2)
    else:
        core_ex = core
    if core.any():
        dist = ndi.distance_transform_edt(~core, sampling=spacing)
        zone = dist <= streak_mm
    else:
        zone = np.zeros_like(core)
    return {"metal_core": core_ex, "metal_any": core, "streak_zone": zone,
            "metal_fraction": float(core_ex.mean())}


def window_from_levels(levels: Dict[str, float], mode: str = "tooth") -> Tuple[float, float]:
    """Display window.  'tooth': soft tissue .. just above enamel; 'bone': soft .. dentine; 'wide': soft .. metal."""
    s, d, e, m = levels["soft_tissue"], levels["dentine_mode"], levels["enamel_mode"], levels["metal_level"]
    if mode == "bone":
        return (s, d * 1.05)
    if mode == "wide":
        return (s, m)
    return (s, e * 1.05)


def apply_window(img: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    lo, hi = window
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def clahe2d(img01: np.ndarray, tiles: int = 8, clip_limit: float = 0.01, bins: int = 256) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (numpy; display only).

    Tile histograms are clipped at `clip_limit` x (pixels per tile / bins) with the excess redistributed, mapped
    through their CDFs and bilinearly blended across tile centres.  Deterministic given the parameters.
    """
    img = np.clip(img01, 0, 1)
    h, w = img.shape
    q = np.minimum((img * (bins - 1)).astype(int), bins - 1)
    th, tw = int(np.ceil(h / tiles)), int(np.ceil(w / tiles))
    cdfs = np.zeros((tiles, tiles, bins), np.float32)
    for i in range(tiles):
        for j in range(tiles):
            blk = q[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if blk.size == 0:
                cdfs[i, j] = np.linspace(0, 1, bins)
                continue
            hist = np.bincount(blk.ravel(), minlength=bins).astype(np.float32)
            limit = max(clip_limit * blk.size, 1.0)
            excess = np.clip(hist - limit, 0, None).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            cdfs[i, j] = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-6)
    # bilinear interpolation between tile centres
    yc = (np.arange(tiles) + 0.5) * th
    xc = (np.arange(tiles) + 0.5) * tw
    yy, xx = np.mgrid[0:h, 0:w]
    fy = np.clip((yy - yc[0]) / th, 0, tiles - 1); fx = np.clip((xx - xc[0]) / tw, 0, tiles - 1)
    i0 = np.floor(fy).astype(int); j0 = np.floor(fx).astype(int)
    i1 = np.minimum(i0 + 1, tiles - 1); j1 = np.minimum(j0 + 1, tiles - 1)
    wy = fy - i0; wx = fx - j0
    out = ((1 - wy) * (1 - wx) * cdfs[i0, j0, q] + (1 - wy) * wx * cdfs[i0, j1, q]
           + wy * (1 - wx) * cdfs[i1, j0, q] + wy * wx * cdfs[i1, j1, q])
    return out.astype(np.float32)


def unsharp2d(img01: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Gaussian unsharp masking (display only).  Xu et al. (Sci Rep 2025) reported that Gaussian sharpening of
    sagittal CBCT slices raised a root-resorption CNN from 0.75 to 0.97 accuracy; here it is an *ablatable* display
    transform for the VLM/human evidence path, never applied to the voxels that measurements use."""
    from scipy import ndimage as ndi
    img = np.clip(img01, 0, 1).astype(np.float32)
    blur = ndi.gaussian_filter(img, sigma)
    return np.clip(img + amount * (img - blur), 0, 1).astype(np.float32)


def display_transform_record(window: Tuple[float, float], clahe: Optional[Dict] = None, sharpen: Optional[Dict] = None) -> Dict:
    return {"window_lo_hi": [float(window[0]), float(window[1])], "clahe": clahe or None, "unsharp_mask": sharpen or None,
            "note": "display only; measurements use unmodified voxel values; CBCT grey values are not HU"}
