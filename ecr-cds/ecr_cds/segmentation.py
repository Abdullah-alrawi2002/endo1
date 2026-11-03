"""Segmentation stage: tooth, root canal, bone crest and ECR lesion.

Architecture (pluggable backends)
---------------------------------
1. Tooth + canal segmentation  : DentalSegmentator (nnU-Net, 3D Slicer) or
                                 any model returning per-tooth labels and a
                                 pulp/canal label. Hook: `ToothSegmenter`.
2. Bone crest surface          : alveolar bone label from the same model or a
                                 simple intensity+morphology estimate.
3. ECR lesion segmentation     : texture-feature SVM (Ravikumar et al. 2025)
                                 with erosion -> largest component -> dilation
                                 post-processing, or an nnU-Net trained on
                                 ECR masks. Hook: `LesionSegmenter`.

Every backend returns probability maps so the decision threshold can be
re-tuned (Youden / calibration) per site, as Al Fartousi et al. (JCM 2026)
showed a fixed 0.50 threshold is rarely optimal on CBCT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
from scipy import ndimage as ndi

from .preprocess import local_grey_level_emphasis, quantise


@dataclass
class ToothSegmentation:
    tooth: np.ndarray        # bool mask of the target tooth (enamel+dentine+pulp)
    canal: np.ndarray        # bool mask of pulp chamber + root canal(s)
    bone: np.ndarray         # bool mask of alveolar bone
    cej_z: Optional[float] = None       # CEJ level along tooth axis (voxels)
    apex_z: Optional[float] = None      # root apex level along tooth axis
    intensity: Optional[np.ndarray] = None  # raw (rescaled) intensities of the same crop, for rule-based detectors


class ToothSegmenter(Protocol):
    def __call__(self, volume: np.ndarray, spacing) -> ToothSegmentation: ...


class LesionSegmenter(Protocol):
    def __call__(self, volume: np.ndarray, seg: ToothSegmentation) -> np.ndarray:
        """Return lesion probability map in [0,1], same shape as volume."""
        ...


# --------------------------------------------------------------------------- #
# Texture-feature SVM lesion segmenter (reference implementation)
# --------------------------------------------------------------------------- #
class TextureSVMLesionSegmenter:
    """Linear SVM on standardised (LGRE, HGRE) voxel features.

    Train with `fit(volumes, tooth_masks, lesion_masks)` on expert-labelled
    cases; predict returns a calibrated probability map (Platt scaling via
    sklearn `probability=True` or sigmoid on the decision function).
    """

    def __init__(self, radius: int = 5, levels: int = 16, threshold: float = 0.5,
                 morph_kernel: int = 6):
        self.radius = radius
        self.levels = levels
        self.threshold = threshold
        self.morph_kernel = morph_kernel
        self._scaler = None
        self._clf = None

    # ---- features ----
    def _features(self, vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
        q = quantise(vol, self.levels)
        maps = local_grey_level_emphasis(q, self.radius, self.levels)
        return np.stack([maps["LGRE"][mask], maps["HGRE"][mask]], axis=1)

    # ---- training ----
    def fit(self, volumes, tooth_masks, lesion_masks):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        X, y = [], []
        for v, t, l in zip(volumes, tooth_masks, lesion_masks):
            # crop to lesion bbox + margin to limit healthy-class imbalance
            from .preprocess import crop_to_bbox
            sl = crop_to_bbox(l, margin=5)
            m = t[sl]
            X.append(self._features(v[sl], m))
            y.append(l[sl][m].astype(int))
        X = np.concatenate(X)
        y = np.concatenate(y)
        self._scaler = StandardScaler().fit(X)
        base = LinearSVC(C=1.0, class_weight="balanced")
        self._clf = CalibratedClassifierCV(base, cv=3).fit(self._scaler.transform(X), y)
        return self

    # ---- inference ----
    def __call__(self, volume: np.ndarray, seg: ToothSegmentation) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Segmenter not trained. Call fit() or load weights.")
        prob = np.zeros(volume.shape, dtype=np.float32)
        mask = seg.tooth
        X = self._scaler.transform(self._features(volume, mask))
        prob[mask] = self._clf.predict_proba(X)[:, 1]
        return prob

    def postprocess(self, prob: np.ndarray, seg: ToothSegmentation) -> np.ndarray:
        """Threshold -> exclude canal -> erode -> largest CC -> dilate."""
        binary = (prob >= self.threshold) & seg.tooth & ~seg.canal
        k = self.morph_kernel
        struct = ndi.generate_binary_structure(3, 1)
        eroded = ndi.binary_erosion(binary, struct, iterations=max(k // 2, 1))
        lab, n = ndi.label(eroded)
        if n == 0:
            return np.zeros_like(binary)
        sizes = ndi.sum(eroded, lab, index=np.arange(1, n + 1))
        largest = lab == (int(np.argmax(sizes)) + 1)
        dilated = ndi.binary_dilation(largest, struct, iterations=max(k // 2, 1))
        return dilated & seg.tooth & ~seg.canal


def lesion_mask_from_probability(prob: np.ndarray, seg: ToothSegmentation,
                                 threshold: float = 0.5, kernel: int = 6) -> np.ndarray:
    """Standalone post-processing identical to TextureSVMLesionSegmenter.postprocess."""
    s = TextureSVMLesionSegmenter(threshold=threshold, morph_kernel=kernel)
    return s.postprocess(prob, seg)
