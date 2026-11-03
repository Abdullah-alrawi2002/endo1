"""Adapters that connect the pipeline to real segmentation tools.

DentalSegmentator (3D Slicer extension, Dot et al. 2024) produces a 5-class
label map: upper skull, mandible, upper teeth, lower teeth, mandibular canal.
It does NOT isolate single teeth or the pulp canal, so this module adds:

* `SlicerLabelmapToothSegmenter` - loads the exported label map (NIfTI/NRRD
  via nibabel, or .npy), picks the target tooth as the connected component of
  the teeth label that contains a seed point, builds the bone mask from the
  jaw labels, and extracts the root canal by intensity inside the tooth.
* `extract_canal` - low-intensity connected component inside the tooth.
* `PretrainedLesionSegmenter` - loads a trained TextureSVMLesionSegmenter
  (joblib) or a user-provided lesion mask file for review-mode runs.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from .segmentation import ToothSegmentation, TextureSVMLesionSegmenter

# Default DentalSegmentator label values (verify against your export's
# segment table; Slicer may renumber when exporting a label map).
DEFAULT_TEETH_LABELS = (3, 4)      # upper teeth, lower teeth
DEFAULT_BONE_LABELS = (1, 2)       # upper skull (maxilla), mandible


class MaskGeometryError(ValueError):
    """Mask/label map does not match the reference volume geometry."""


def load_labelmap(path: str, volume=None, spacing_tol: float = 0.02,
                  allow_resample: bool = False) -> np.ndarray:
    """Load a label map as (Z, Y, X) int array aligned with the DICOM volume.

    NIfTI is stored (X, Y, Z) with an affine in RAS. We reorder/flip axes so the
    array is (Z, Y, X) in LPS order matching `CBCTVolume`. If `volume` is given
    the shape and voxel spacing are verified and a translation mismatch larger
    than one voxel is rejected (shifted masks must be resampled explicitly).
    """
    if path.endswith(".npy"):
        arr = np.load(path)
        if volume is not None and arr.shape != volume.shape:
            raise MaskGeometryError(f"Mask shape {arr.shape} != volume {volume.shape}")
        return arr.astype(np.int32)
    import nibabel as nib
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise MaskGeometryError(f"Expected a 3-D label map, got shape {arr.shape}")
    aff = img.affine
    # axis codes in RAS; convert to LPS by flipping R->L and A->P
    codes = nib.aff2axcodes(aff)
    # nibabel data axes 0,1,2 -> anatomical codes; we want (Z, Y, X) = (S/I, P/A, L/R)
    want = {"L": 2, "R": 2, "P": 1, "A": 1, "S": 0, "I": 0}
    order = [None, None, None]
    flips = []
    for data_axis, code in enumerate(codes):
        out_axis = want[code]
        if order[out_axis] is not None:
            raise MaskGeometryError(f"Degenerate affine orientation {codes}")
        order[out_axis] = data_axis
        # LPS positive directions: L, P, S. Flip if the data axis points R, A or I.
        if code in ("R", "A", "I"):
            flips.append(out_axis)
    arr = np.transpose(arr, order)
    for ax in flips:
        arr = np.flip(arr, axis=ax)
    arr = np.ascontiguousarray(arr).astype(np.int32)
    if volume is not None:
        vox = np.sqrt((aff[:3, :3] ** 2).sum(0))  # spacing per data axis
        vox_zyx = np.array([vox[order[0]], vox[order[1]], vox[order[2]]])
        if not np.allclose(vox_zyx, volume.spacing, rtol=spacing_tol):
            if not allow_resample:
                raise MaskGeometryError(f"Mask spacing {vox_zyx} != volume spacing {volume.spacing}; "
                                        "set allow_resample=True for nearest-neighbour resampling")
            from scipy.ndimage import zoom
            arr = zoom(arr, np.array(vox_zyx) / np.array(volume.spacing), order=0)
        if arr.shape != volume.shape:
            raise MaskGeometryError(f"Mask shape {arr.shape} != volume shape {volume.shape}; "
                                    "export the mask on the same reference volume")
        # translation check: compare mask origin (LPS) with volume origin
        origin_ras = aff[:3, 3]
        origin_lps = origin_ras * np.array([-1, -1, 1])
        vol_origin = np.asarray(volume.origin)
        # mask origin refers to the voxel that ends up at index 0 after flips; tolerate
        # one voxel because flipping moves the corner voxel to the opposite edge.
        extent = np.array(volume.spacing)[::-1] * (np.array(volume.shape)[::-1] - 1)  # x,y,z extents
        corners = [origin_lps, origin_lps - extent * np.array([1, 1, 0]), origin_lps - extent, origin_lps - extent * np.array([0, 0, 1]),
                   origin_lps - extent * np.array([1, 0, 0]), origin_lps - extent * np.array([0, 1, 0]),
                   origin_lps - extent * np.array([1, 0, 1]), origin_lps - extent * np.array([0, 1, 1])]
        if not any(np.allclose(c, vol_origin, atol=1.5 * max(volume.spacing)) for c in corners) \
                and not np.allclose(origin_ras, 0) and not np.allclose(vol_origin, 0):
            raise MaskGeometryError("Mask origin does not coincide with the volume origin (shifted mask); "
                                    "resample the mask onto the DICOM grid before use")
    return arr


def _otsu(vals: np.ndarray, nbins: int = 256) -> float:
    hist, edges = np.histogram(vals, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    var_b = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(var_b))])


def extract_canal(volume: np.ndarray, tooth: np.ndarray, pct: float = 20.0,
                  min_voxels: int = 50) -> np.ndarray:
    """Pulp chamber + root canal = dark connected region inside the tooth.

    Erode the tooth first so that the periodontal ligament space / lesion at
    the surface is not confused with the canal, then keep the components whose
    centroid lies near the tooth axis.
    """
    core = ndi.binary_erosion(tooth, iterations=3)
    # Restrict candidates to the central 45 % of the local tooth radius along the
    # PCA axis so that surface lesions (also dark) are not mistaken for the canal
    # - the pulp/ECR confusion reported by Ravikumar et al.
    idx = np.argwhere(tooth).astype(np.float64)
    c = idx.mean(0)
    axis = np.linalg.svd(idx - c, full_matrices=False)[2][0]
    t = (idx - c) @ axis
    radial = np.linalg.norm((idx - c) - np.outer(t, axis), axis=1)
    nb = 40
    bins = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * (nb - 1)).astype(int), 0, nb - 1)
    local_r = np.zeros(nb)
    for b_ in range(nb):
        m = bins == b_
        local_r[b_] = np.percentile(radial[m], 98) if m.any() else 0
    central = np.zeros_like(tooth)
    keep = radial <= 0.45 * local_r[bins]
    central[tuple(idx[keep].astype(int).T)] = True
    core &= central
    vals = volume[core]
    # Otsu split between the dark pulp mode and the dentine mode; `pct` is only
    # used as an upper bound so a lesion-free, canal-free core cannot yield a
    # threshold inside dentine.
    thr = min(_otsu(vals), np.percentile(vals, max(pct, 50)))
    dark = core & (volume <= thr)
    dark = ndi.binary_opening(dark, iterations=1)
    lab, n = ndi.label(dark)
    if n == 0:
        return np.zeros_like(tooth)
    sizes = ndi.sum(dark, lab, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    if not keep:
        return np.zeros_like(tooth)
    # largest component is the pulp system; add other large ones aligned with it (multi-canal)
    order = sorted(keep, key=lambda i: -sizes[i - 1])
    canal = lab == order[0]
    c0 = np.array(ndi.center_of_mass(canal))
    for i in order[1:4]:
        ci = np.array(ndi.center_of_mass(lab == i))
        if np.linalg.norm((ci - c0)[1:]) < 0.35 * np.sqrt(tooth.sum() / tooth.shape[0]):
            canal |= lab == i
    return canal


class SlicerLabelmapToothSegmenter:
    """ToothSegmenter backed by a DentalSegmentator label map.

    seed : (z, y, x) voxel index inside the target tooth, e.g. taken from the
           slice/row/column the clinician clicks in the viewer.
    """

    def __init__(self, labelmap_path: str, seed: Tuple[int, int, int],
                 teeth_labels: Sequence[int] = DEFAULT_TEETH_LABELS,
                 bone_labels: Sequence[int] = DEFAULT_BONE_LABELS,
                 canal_pct: float = 20.0):
        self.labelmap_path = labelmap_path
        self.labels = load_labelmap(labelmap_path)
        self.seed = tuple(int(v) for v in seed)
        self.canal_mask_path: Optional[str] = None  # optional expert canal segment
        self.teeth_labels = tuple(teeth_labels)
        self.bone_labels = tuple(bone_labels)
        self.canal_pct = canal_pct

    def __call__(self, volume: np.ndarray, spacing) -> ToothSegmentation:
        if self.labels.shape != volume.shape:
            raise ValueError(f"Label map {self.labels.shape} does not match volume {volume.shape}; "
                             "export the segmentation on the same reference volume.")
        teeth = np.isin(self.labels, self.teeth_labels)
        # separate touching teeth: erode, label, pick seed component, grow back within teeth mask
        eroded = ndi.binary_erosion(teeth, iterations=2)
        lab, n = ndi.label(eroded)
        seed_lab = lab[self.seed]
        if seed_lab == 0:
            if not teeth[self.seed]:
                raise ValueError(f"Seed {self.seed} is not inside any teeth label {self.teeth_labels}")
            # seed fell in the eroded gap: take nearest labelled voxel
            dist, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
            seed_lab = lab[tuple(idx[:, self.seed[0], self.seed[1], self.seed[2]])]
        core = lab == seed_lab
        tooth = ndi.binary_dilation(core, iterations=2) & teeth
        # keep only voxels closer to this tooth's core than to any other tooth core
        other = eroded & ~core
        if other.any():
            d_self = ndi.distance_transform_edt(~core)
            d_other = ndi.distance_transform_edt(~other)
            tooth &= d_self <= d_other
        tooth = ndi.binary_fill_holes(tooth)
        bone = np.isin(self.labels, self.bone_labels)
        if self.canal_mask_path:
            canal = load_labelmap(self.canal_mask_path) > 0
        else:
            canal = extract_canal(volume, tooth, pct=self.canal_pct)
        if tooth.sum() == 0:
            raise ValueError("Seed does not fall inside a tooth label; check seed (z,y,x) and label values")
        return ToothSegmentation(tooth=tooth, canal=canal, bone=bone)


class PretrainedLesionSegmenter:
    """LesionSegmenter that either runs a trained texture SVM or returns a
    clinician-drawn lesion mask (review mode) as a probability map."""

    def __init__(self, model_path: Optional[str] = None, mask_path: Optional[str] = None):
        if not model_path and not mask_path:
            raise ValueError("Provide model_path (trained SVM) or mask_path (manual lesion mask)")
        self.model: Optional[TextureSVMLesionSegmenter] = None
        self.mask: Optional[np.ndarray] = None
        if model_path:
            import joblib
            self.model = joblib.load(model_path)
        if mask_path:
            self.mask = load_labelmap(mask_path) > 0

    def __call__(self, volume_norm: np.ndarray, seg: ToothSegmentation) -> np.ndarray:
        if self.mask is not None:
            return self.mask.astype(np.float32)
        assert self.model is not None
        return self.model(volume_norm, seg)
