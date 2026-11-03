"""Classical automatic tooth / bone / canal segmentation with seed proposal.

Fallback for the case where no DentalSegmentator label map and no expert
masks are available. It is deliberately simple and transparent:

1. Downsample to ~0.16 mm, smooth, estimate two data-driven thresholds
   (Otsu): `t_hard` separates soft tissue from mineralised tissue and
   `t_dense` separates dentine/bone from enamel, root filling and metal.
2. Teeth candidates = morphological opening (ball ~0.8 mm) of
   `smoothed > t_tooth`, where `t_tooth` lies between `t_hard` and the dentine
   mode; the opening removes thin cortical plates and trabeculae while roots
   survive. Connected components are kept as teeth when they are at least
   `min_tooth_mm3` large, span >= `min_length_mm` along z and contain dense
   material (enamel, filling or restoration).
3. Bone = remaining mineralised tissue, closed to include trabecular spaces.
4. Seed proposal: candidates ranked by in-plane distance to the field-of-view
   centre (small-FOV CBCT is acquired centred on the tooth of interest).
   The proposal must be confirmed by the clinician - it is reported as
   `seed_source = auto_fov_centre`.
5. Canal: hyperdense central material (root filling) when present, else the
   dark central region (pulp) - see `extract_canal_auto`.

Everything produced here is labelled `auto_classical` in the report. It is
not a validated segmentation and it is expected to fail around large metal
restorations (streak artefacts) and for teeth cut by the FOV border.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .adapters import extract_canal, _otsu
from .segmentation import ToothSegmentation

LABEL_BONE = 1
LABEL_TEETH = 3


@dataclass
class ToothCandidate:
    rank: int
    label: int
    seed_zyx: Tuple[int, int, int]
    centroid_zyx: Tuple[float, float, float]
    volume_mm3: float
    length_mm: float
    dist_to_fov_centre_mm: float
    max_intensity: float
    hyperdense_fraction: float
    note: str = ""


@dataclass
class AutoSegmentation:
    labels: np.ndarray                 # (Z,Y,X) uint8: 0 bg, 1 bone, 3 teeth
    tooth_ids: np.ndarray              # (Z,Y,X) int: per-tooth component id (0 = none)
    candidates: List[ToothCandidate]
    seed_zyx: Optional[Tuple[int, int, int]]
    t_hard: float
    t_tooth: float
    t_dense: float
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"method": "auto_classical", "thresholds": {"t_hard": self.t_hard, "t_tooth": self.t_tooth, "t_dense": self.t_dense},
                "n_teeth": len(self.candidates), "seed_zyx": self.seed_zyx,
                "seed_source": "auto_fov_centre" if self.seed_zyx is not None else None,
                "candidates": [asdict(c) for c in self.candidates], "notes": self.notes}


def _ball(r: int) -> np.ndarray:
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz ** 2 + yy ** 2 + xx ** 2) <= r * r


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a.astype(np.float32)
    Z, Y, X = [s // f * f for s in a.shape]
    return a[:Z, :Y, :X].reshape(Z // f, f, Y // f, f, X // f, f).mean((1, 3, 5)).astype(np.float32)


def _upsample(m: np.ndarray, f: int, shape) -> np.ndarray:
    if f == 1:
        return m
    up = np.repeat(np.repeat(np.repeat(m, f, 0), f, 1), f, 2)
    out = np.zeros(shape, dtype=m.dtype)
    z, y, x = [min(s, u) for s, u in zip(shape, up.shape)]
    out[:z, :y, :x] = up[:z, :y, :x]
    return out


def auto_segment(array: np.ndarray, spacing, target_voxel_mm: float = 0.16,
                 opening_mm: float = 0.8, min_tooth_mm3: float = 50.0, min_length_mm: float = 6.0,
                 tooth_closing_mm: float = 1.2, fov_centre_zyx: Optional[Tuple[int, int, int]] = None) -> AutoSegmentation:
    sp = np.asarray(spacing, dtype=float)
    f = max(1, int(round(target_voxel_mm / sp.min())))
    a2 = _block_mean(array, f)
    sp2 = sp * f
    sm = ndi.gaussian_filter(a2, 0.7)
    notes: List[str] = []

    # field of view: CBCT pads outside the reconstruction cylinder with the minimum value
    vmin = float(array.min())
    fov = sm > vmin + 0.05 * (float(np.percentile(a2, 99.9)) - vmin)
    v = sm[fov]
    v = v[(v > np.percentile(v, 1)) & (v < np.percentile(v, 99.5))]
    t_hard = float(_otsu(v[v > np.percentile(v, 40)]))      # soft vs mineralised
    vh = v[v > t_hard]
    t_dense = float(_otsu(vh))                               # dentine/bone vs enamel/filling/metal
    # dentine mode: the dominant mode of mineralised, non-dense voxels
    hist, edges = np.histogram(vh[vh < t_dense], bins=64)
    dentine_mode = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    t_tooth = float(t_hard + 0.6 * (dentine_mode - t_hard))
    notes.append(f"thresholds (scanner units): hard {t_hard:.0f}, tooth {t_tooth:.0f}, dense {t_dense:.0f}, dentine mode {dentine_mode:.0f}")

    hard = (sm > t_hard) & fov
    r_open = max(1, int(round(opening_mm / sp2.min())))
    core = ndi.binary_opening((sm > t_tooth) & fov, structure=_ball(r_open))
    lab, n = ndi.label(core)
    if n == 0:
        notes.append("no tooth-like components found")
        return AutoSegmentation(np.zeros(array.shape, np.uint8), np.zeros(array.shape, np.int16), [], None, t_hard, t_tooth, t_dense, notes)

    vox_mm3 = float(np.prod(sp2))
    sizes = ndi.sum(core, lab, index=np.arange(1, n + 1)) * vox_mm3
    if fov_centre_zyx is None:
        fov_centre_zyx = tuple(s // 2 for s in array.shape)
    cz, cy, cx = fov_centre_zyx
    cands = []
    teeth2 = np.zeros_like(core)
    tooth_ids2 = np.zeros(core.shape, np.int16)
    for i in range(n):
        if sizes[i] < min_tooth_mm3:
            continue
        m = lab == i + 1
        idx = np.argwhere(m)
        length = float((idx[:, 0].max() - idx[:, 0].min() + 1) * sp2[0])
        mx = float(a2[m].max())
        hd = float((a2[m] > t_dense).mean())
        if length < min_length_mm or mx <= t_dense:
            continue  # cortical fragments / zygoma: no enamel, filling or restoration
        cen2 = idx.mean(0)
        cen = tuple(float(c) for c in cen2 * f)
        dist = float(np.hypot((cen[1] - cy) * sp[1], (cen[2] - cx) * sp[2]))
        # seed: tooth voxel nearest the centroid with dentine-range intensity
        dent = m & (a2 <= t_dense)
        pts = np.argwhere(dent if dent.any() else m)
        j = np.argmin(((pts - cen2) ** 2).sum(1))
        seed = tuple(int(round(p * f + f / 2)) for p in pts[j])
        note = ""
        if hd > 0.3:
            note = "large dense/metal fraction: restoration or artefact - inspect"
        elif mx > t_dense and hd > 0.02:
            note = "contains hyperdense central material (root filling / post?)"
        cands.append(ToothCandidate(0, i + 1, seed, cen, float(sizes[i]), length, dist, mx, hd, note))
        teeth2 |= m
        tooth_ids2[m] = i + 1
    cands.sort(key=lambda c: c.dist_to_fov_centre_mm)
    for k, c in enumerate(cands):
        c.rank = k + 1

    # coarse closing (fills surface cavities / empty canal up to ~tooth_closing_mm) and bone
    k_cl = max(1, int(round(tooth_closing_mm / sp2.min())))
    teeth2_closed = np.zeros_like(teeth2)
    for c_ in cands:
        m = tooth_ids2 == c_.label
        mc = ndi.binary_closing(m, iterations=k_cl) | m
        mc = ndi.binary_fill_holes(mc) & fov
        teeth2_closed |= mc
        tooth_ids2[mc & (tooth_ids2 == 0)] = c_.label
    bone2 = hard & ~ndi.binary_dilation(teeth2_closed, iterations=max(1, int(round(0.3 / sp2.min()))))
    bone2 = ndi.binary_closing(bone2, iterations=max(1, int(round(0.4 / sp2.min())))) & ~teeth2_closed & fov

    # full-resolution refinement inside the teeth bounding box only
    labels = np.zeros(array.shape, np.uint8)
    ids_full = _upsample(tooth_ids2, f, array.shape)
    labels[_upsample(bone2, f, array.shape)] = LABEL_BONE
    if cands:
        bb = ndi.find_objects((ids_full > 0).astype(np.int8))[0]
        pad = f + 2
        sl = tuple(slice(max(0, b.start - pad), min(n_, b.stop + pad)) for b, n_ in zip(bb, array.shape))
        crop = array[sl].astype(np.float32)
        sm_c = ndi.gaussian_filter(crop, 0.7) if f > 1 else crop
        ids_c = ids_full[sl]
        grown = ndi.grey_dilation(ids_c, size=(2 * f + 1,) * 3)       # nearest-id growth (bounded)
        teeth_c = (grown > 0) & ((sm_c > t_tooth) | (ids_c > 0))
        teeth_c = ndi.binary_closing(teeth_c, iterations=max(1, int(round(0.3 / sp.min())))) | teeth_c
        teeth_c = ndi.binary_fill_holes(teeth_c)
        ids_c = np.where(teeth_c, grown, 0).astype(np.int16)
        ids_full = np.zeros(array.shape, np.int16)
        ids_full[sl] = ids_c
        labels[ids_full > 0] = LABEL_TEETH
        labels[(labels == LABEL_BONE) & (ids_full > 0)] = LABEL_TEETH
    teeth_full = ids_full > 0
    seed = cands[0].seed_zyx if cands else None
    if seed is not None and not teeth_full[seed]:
        pts = np.argwhere(ids_full == cands[0].label)
        j = np.argmin(((pts - np.asarray(seed)) ** 2).sum(1))
        seed = tuple(int(p) for p in pts[j]); cands[0].seed_zyx = seed
    notes.append(f"{len(cands)} tooth candidates; seed proposed from FOV-centre ranking - clinician must confirm")
    return AutoSegmentation(labels, ids_full.astype(np.int16), cands, seed, t_hard, t_tooth, t_dense, notes)


def extract_canal_auto(volume: np.ndarray, tooth: np.ndarray, t_dense: float, min_voxels: int = 50) -> Tuple[np.ndarray, str]:
    """Root canal = hyperdense central filling if present, else dark pulp (Otsu)."""
    core = ndi.binary_erosion(tooth, iterations=3)
    idx = np.argwhere(tooth).astype(np.float64)
    c = idx.mean(0)
    axis = np.linalg.svd(idx - c, full_matrices=False)[2][0]
    t = (idx - c) @ axis
    radial = np.linalg.norm((idx - c) - np.outer(t, axis), axis=1)
    nb = 40
    bins = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * (nb - 1)).astype(int), 0, nb - 1)
    local_r = np.array([np.percentile(radial[bins == b], 98) if (bins == b).any() else 0 for b in range(nb)])
    central = np.zeros_like(tooth)
    keep = radial <= 0.45 * local_r[bins]
    central[tuple(idx[keep].astype(int).T)] = True
    dense = core & central & (volume > t_dense)
    dense = ndi.binary_opening(dense, iterations=1)
    lab, n = ndi.label(dense)
    if n:
        sizes = ndi.sum(dense, lab, index=np.arange(1, n + 1))
        big = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
        if big:
            filled = np.isin(lab, big)
            # a root filling is elongated along the tooth axis: require >= 3 mm-equivalent extent
            ext = np.ptp((np.argwhere(filled) - c) @ axis)
            if ext >= 0.25 * np.ptp(t):
                return ndi.binary_dilation(filled, iterations=1) & tooth, "hyperdense_filling"
    return extract_canal(volume, tooth), "dark_pulp_otsu"


class AutoToothSegmenter:
    """ToothSegmenter backed by `auto_segment` output (no external label map)."""

    def __init__(self, auto: AutoSegmentation, seed: Optional[Tuple[int, int, int]] = None):
        self.auto = auto
        self.seed = tuple(int(v) for v in (seed if seed is not None else auto.seed_zyx))
        self.canal_mode: Optional[str] = None
        self.canal_mask: Optional[np.ndarray] = None   # optional expert canal mask (bool array)

    def __call__(self, volume: np.ndarray, spacing) -> ToothSegmentation:
        ids = self.auto.tooth_ids
        tid = int(ids[self.seed])
        if tid == 0:
            raise ValueError(f"Seed {self.seed} is not inside an automatically segmented tooth")
        tooth = ids == tid
        bone = self.auto.labels == LABEL_BONE
        if self.canal_mask is not None:
            canal, self.canal_mode = self.canal_mask & tooth, "expert_mask"
        else:
            canal, self.canal_mode = extract_canal_auto(volume, tooth, self.auto.t_dense)
        return ToothSegmentation(tooth=tooth, canal=canal, bone=bone)


def save_labelmap_nifti(labels: np.ndarray, volume, path: str) -> None:
    """Write (Z,Y,X) LPS labels as a RAS NIfTI aligned with the DICOM volume."""
    import nibabel as nib
    aff = np.diag([-1., -1., 1., 1.]) @ volume.affine_lps()
    nib.save(nib.Nifti1Image(np.transpose(labels.astype(np.uint16), (2, 1, 0)), aff), path)


# --------------------------------------------------------------------------- #
# Rule-based radiolucent-defect detector (model-free fallback)
# --------------------------------------------------------------------------- #
def tooth_hull(tooth: np.ndarray, spacing, closing_mm: float = 2.5) -> np.ndarray:
    """Convex-ish hull of the tooth: morphological closing that fills surface
    cavities (resorption, caries) up to ~closing_mm across. Computed on a 2x grid."""
    sp = float(np.min(spacing))
    f = 2
    Z, Y, X = [s // f * f for s in tooth.shape]
    t2 = tooth[:Z, :Y, :X].reshape(Z // f, f, Y // f, f, X // f, f).max((1, 3, 5))
    k = max(1, int(round(closing_mm / (sp * f))))
    h2 = ndi.binary_fill_holes(ndi.binary_closing(t2, iterations=k) | t2)
    hull = np.zeros(tooth.shape, bool)
    up = np.repeat(np.repeat(np.repeat(h2, f, 0), f, 1), f, 2)
    hull[:Z, :Y, :X] = up
    return ndi.binary_closing(hull | tooth, iterations=2) | tooth


def tooth_hull_convex(tooth: np.ndarray, min_px: int = 10) -> np.ndarray:
    """Per-slice, per-component 2D convex hull of the tooth cross-section.

    Each connected cross-section (one per root in the furcation region) is
    replaced by its own convex hull, so external cavities (resorption, caries)
    are filled while the space between separate roots is not."""
    from scipy.spatial import ConvexHull, QhullError
    hull = tooth.copy()
    for z in range(tooth.shape[0]):
        sl = tooth[z]
        if not sl.any():
            continue
        lab, n = ndi.label(sl)
        for i in range(1, n + 1):
            pts = np.argwhere(lab == i)
            if len(pts) < min_px:
                continue
            try:
                ch = ConvexHull(pts)
            except QhullError:
                continue
            y0, x0 = pts.min(0); y1, x1 = pts.max(0)
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            P = np.stack([yy.ravel(), xx.ravel(), np.ones(yy.size)], 1)
            inside = np.all(P @ ch.equations.T <= 1e-9, axis=1).reshape(yy.shape)
            hull[z, y0:y1 + 1, x0:x1 + 1] |= inside
    return hull


def _central_region(mask: np.ndarray, frac: float) -> np.ndarray:
    """Voxels of `mask` within `frac` of the local radius around the PCA axis."""
    idx = np.argwhere(mask).astype(np.float64)
    c = idx.mean(0)
    axis = np.linalg.svd(idx - c, full_matrices=False)[2][0]
    t = (idx - c) @ axis
    radial = np.linalg.norm((idx - c) - np.outer(t, axis), axis=1)
    nb = 40
    bins = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * (nb - 1)).astype(int), 0, nb - 1)
    local_r = np.array([np.percentile(radial[bins == b], 98) if (bins == b).any() else 0 for b in range(nb)])
    out = np.zeros_like(mask)
    keep = radial <= frac * local_r[bins]
    out[tuple(idx[keep].astype(int).T)] = True
    return out


class RadiolucencyLesionDetector:
    """LesionSegmenter: hypodense defect inside the tooth hull, outside the canal,
    reaching the external surface. Uses raw intensities (`seg.intensity`) and the
    data-driven soft/hard threshold `t_hard` from `auto_segment`.

    Probability = depth of hypodensity below `t_defect`, scaled to [0,1] by the
    soft-tissue level, so `lesion_mask_from_probability` can threshold it
    (prob 0.3 ~ 30 % of the way from tooth tissue down to soft tissue).
    Not a learned model - see `_evidence_level('radiolucency_rule')`.
    """

    def __init__(self, t_defect: float, soft_level: float, min_defect_mm3: float = 1.0,
                 surface_contact_min: float = 0.05, canal_margin_vox: int = 4, open_iters: int = 3):
        """t_defect: intensity below which hull voxels count as defect - use `auto.t_tooth`
        (tooth-tissue threshold) so intermediate-density fibro-osseous repair tissue is included."""
        self.t_hard, self.soft_level = float(t_defect), float(soft_level)
        self.min_defect_mm3, self.surface_contact_min = min_defect_mm3, surface_contact_min
        self.canal_margin_vox, self.open_iters = canal_margin_vox, open_iters
        self.last_components: List[dict] = []

    def __call__(self, volume_norm: np.ndarray, seg: ToothSegmentation, spacing=(0.08, 0.08, 0.08)) -> np.ndarray:
        raw = seg.intensity
        if raw is None:
            raise ValueError("RadiolucencyLesionDetector needs seg.intensity (raw crop); run through run_pipeline")
        sm = ndi.gaussian_filter(raw.astype(np.float32), 0.7)
        hull = tooth_hull_convex(seg.tooth) | tooth_hull(seg.tooth, spacing)
        # central dark core (pulp chamber / canal, whether or not `seg.canal` captured it fully):
        # ECR is external, so defects are measured outside the central 45 % of the local radius
        central = _central_region(hull, 0.45)
        core_dark = ndi.binary_dilation(central & (sm < self.t_hard), iterations=2)
        inner = hull & ~ndi.binary_dilation(seg.canal, iterations=self.canal_margin_vox) & ~core_dark
        defect = ndi.binary_opening(inner & (sm < self.t_hard), iterations=self.open_iters)
        near_surface = ndi.binary_dilation(~hull, iterations=4)
        vox_mm3 = float(np.prod(spacing))
        lab, n = ndi.label(defect)
        keep = np.zeros_like(defect)
        self.last_components = []
        for i in range(1, n + 1):
            m = lab == i
            mm3 = m.sum() * vox_mm3
            if mm3 < self.min_defect_mm3:
                continue
            contact = float((m & near_surface).sum() / m.sum())
            in_tooth = float((m & seg.tooth).sum() / m.sum())
            self.last_components.append({"mm3": round(mm3, 2), "surface_contact": round(contact, 2),
                                         "fraction_inside_tooth_mask": round(in_tooth, 2),
                                         "centroid_crop_zyx": [int(v) for v in np.argwhere(m).mean(0)]})
            if contact >= self.surface_contact_min:
                keep |= m
        prob = np.zeros(raw.shape, np.float32)
        depth = np.clip((self.t_hard - sm) / max(self.t_hard - self.soft_level, 1.0), 0, 1)
        prob[keep] = 0.5 + 0.5 * depth[keep]          # >= 0.5 for every kept defect voxel
        # The defect lies outside the thresholded tooth mask by construction (it is the
        # missing tooth substance). Restore it to the tooth so the downstream
        # post-processing (restricted to the tooth) and Patel geometry see it.
        seg.tooth |= keep
        return prob


def soft_tissue_level(array: np.ndarray, t_hard: float) -> float:
    v = array[(array < t_hard) & (array > t_hard - 3000)]
    return float(np.median(v)) if v.size else 0.0


def score_candidates_radiolucency(auto: AutoSegmentation, array: np.ndarray, spacing, margin_mm: float = 3.0) -> List[dict]:
    """Run the rule-based detector on every candidate tooth; returns per-tooth defect volumes.
    Used to let the system propose the seed from lesion evidence rather than FOV centre only."""
    soft = soft_tissue_level(array, auto.t_hard)
    det = RadiolucencyLesionDetector(auto.t_tooth, soft)
    margin = int(round(margin_mm / min(spacing)))
    out = []
    for c in auto.candidates:
        tooth = auto.tooth_ids == c.label
        idx = np.argwhere(tooth)
        lo = np.maximum(idx.min(0) - margin, 0); hi = np.minimum(idx.max(0) + margin + 1, array.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        crop = array[sl].astype(np.float32)
        t = tooth[sl]
        canal, mode = extract_canal_auto(crop, t, auto.t_dense)
        seg = ToothSegmentation(tooth=t, canal=canal, bone=(auto.labels[sl] == LABEL_BONE), intensity=crop)
        prob = det(None, seg, spacing)
        mm3 = float((prob >= 0.5).sum() * np.prod(spacing))
        out.append({"rank_fov": c.rank, "label": int(c.label), "seed_zyx": list(c.seed_zyx), "radiolucent_defect_mm3": round(mm3, 1),
                    "canal_mode": mode, "components": det.last_components, "dist_to_fov_centre_mm": round(c.dist_to_fov_centre_mm, 1),
                    "note": c.note})
    return out


def choose_seed(scores: List[dict], min_defect_mm3: float = 5.0) -> Tuple[Tuple[int, int, int], str]:
    """Seed = tooth with the largest radiolucent defect if any exceeds `min_defect_mm3`
    (excluding teeth flagged as metal/artefact), else the FOV-centre tooth."""
    usable = [s for s in scores if "metal" not in s["note"]]
    best = max(usable or scores, key=lambda s: s["radiolucent_defect_mm3"])
    if best["radiolucent_defect_mm3"] >= min_defect_mm3:
        return tuple(best["seed_zyx"]), "auto_radiolucency"
    centre = min(scores, key=lambda s: s["rank_fov"])
    return tuple(centre["seed_zyx"]), "auto_fov_centre"
