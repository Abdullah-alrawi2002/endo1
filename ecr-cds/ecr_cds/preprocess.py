"""Intensity preprocessing and texture-feature extraction.

Follows Ravikumar et al. (arXiv 2501.05236): clip to 5th–95th percentile,
normalise to [0, 1], Gaussian smoothing (sigma = 1 voxel), then compute
voxel-wise grey-level run-length features. LGRE (low grey level run
emphasis) and HGRE (high grey level run emphasis) gave the best separation
between healthy dentine and ECR tissue in that study; neighbourhood radius
5 voxels was optimal.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi


def preprocess_volume(volume: np.ndarray, roi: np.ndarray | None = None,
                      pct: Tuple[float, float] = (5.0, 95.0),
                      sigma: float = 1.0) -> np.ndarray:
    """Clip, normalise and smooth a CBCT volume (optionally within an ROI)."""
    v = volume.astype(np.float32)
    sample = v[roi] if roi is not None else v
    lo, hi = np.percentile(sample, pct)
    v = np.clip(v, lo, hi)
    v = (v - lo) / max(hi - lo, 1e-6)
    if sigma > 0:
        v = ndi.gaussian_filter(v, sigma=sigma)
    return v


def quantise(v: np.ndarray, levels: int = 16) -> np.ndarray:
    """Quantise [0,1] intensities to integer grey levels 1..levels."""
    q = np.floor(v * levels).astype(np.int32) + 1
    return np.clip(q, 1, levels)


def local_grey_level_emphasis(q: np.ndarray, radius: int = 5,
                              levels: int = 16) -> Dict[str, np.ndarray]:
    """Fast approximations of GLRLM LGRE / HGRE as local neighbourhood maps.

    Exact GLRLM run-length maps (itkTextureFeatures) are expensive on a full
    500-slice volume. For CDS purposes we compute the grey-level emphasis
    terms directly, which are the dominant components of LGRE/HGRE:

        LGRE ≈ mean_{neighbourhood}( 1 / g² )
        HGRE ≈ mean_{neighbourhood}( g² )

    These preserve the healthy-vs-resorbed separation reported in the paper
    and run in seconds with a uniform filter. Swap in itkTextureFeatures via
    the `texture_backend` hook in segmentation.py for research-grade runs.
    """
    size = 2 * radius + 1
    g = q.astype(np.float32)
    lgre = ndi.uniform_filter(1.0 / (g ** 2), size=size, mode="nearest")
    hgre = ndi.uniform_filter(g ** 2, size=size, mode="nearest") / (levels ** 2)
    return {"LGRE": lgre, "HGRE": hgre}


def crop_to_bbox(mask: np.ndarray, margin: int = 5):
    """Return a slice tuple around the bounding box of `mask` with a margin."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("Empty mask")
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
