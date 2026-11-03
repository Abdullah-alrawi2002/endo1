"""Patel 3D classification from segmentation masks.

Patel, Foschi, Mannocci & Patel (Int Endod J 2018; doi:10.1111/iej.12824):

  Height               1  at CEJ or coronal to bone crest (supracrestal)
                       2  into coronal third of root AND apical to bone crest
                       3  into middle third of root
                       4  into apical third of root
  Circumferential      A  ≤ 90°      B  > 90°–180°
  spread               C  > 180°–270°  D  > 270°
  Proximity to canal   d  confined to dentine
                       p  probable pulpal involvement

Geometry conventions
--------------------
All measurements are made along the tooth long axis (PCA of the tooth mask),
so the result is independent of the acquisition plane - this matters because
plane definition changes CBCT measurements (Spin-Neto et al. 2011).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .segmentation import ToothSegmentation


@dataclass
class PatelParameters:
    height: int                     # 1..4
    spread: str                     # 'A'..'D'
    proximity: str                  # 'd' | 'p'
    # continuous evidence behind each categorical call
    lesion_volume_mm3: float
    apical_extent_mm: float         # lesion apical limit measured from CEJ (+ = apical)
    coronal_extent_mm: float        # lesion coronal limit from CEJ (- = coronal to CEJ)
    bone_crest_mm: float            # bone crest level from CEJ
    root_length_mm: float
    spread_deg: float
    min_canal_distance_mm: float    # 0 if perforated
    canal_contact_voxels: int
    calcified_fraction: float       # repair-phase proxy (HGRE-high voxels in lesion)
    portal_angle_deg: Optional[float]   # angular position of most coronal lesion surface
    # margins to the nearest category boundary (used for confidence / reinspection)
    height_margin_mm: float
    spread_margin_deg: float
    proximity_margin_mm: float
    borderline: bool

    @property
    def code(self) -> str:
        return f"{self.height}{self.spread}{self.proximity}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["code"] = self.code
        return d


# --------------------------------------------------------------------------- #
def _tooth_axis(tooth: np.ndarray, spacing) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroid_mm, unit axis) of the tooth via PCA, mm coordinates."""
    pts = np.argwhere(tooth).astype(np.float64) * np.asarray(spacing)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    return c, axis / np.linalg.norm(axis)


def _project(pts_mm: np.ndarray, c: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (pts_mm - c) @ axis


def _mm(idx: np.ndarray, spacing) -> np.ndarray:
    return idx.astype(np.float64) * np.asarray(spacing)


def classify_patel(lesion: np.ndarray, seg: ToothSegmentation, spacing,
                   volume_norm: Optional[np.ndarray] = None,
                   cej_level_mm: Optional[float] = None,
                   apex_level_mm: Optional[float] = None,
                   buccal_direction: Optional[np.ndarray] = None,
                   proximity_threshold_mm: Optional[float] = None,
                   angular_gap_close_deg: float = 6.0) -> PatelParameters:
    """Compute Patel 3D parameters from masks.

    cej_level_mm / apex_level_mm: positions along the tooth axis (mm, apical
    positive, measured from the tooth centroid). If None they are estimated:
    apex = most apical tooth voxel; CEJ = level where enamel (top-intensity
    shell) ends if `volume_norm` is given, else the most coronal bone crest
    level minus 1.5 mm (typical biologic width) as a fallback.
    """
    spacing = np.asarray(spacing, dtype=float)
    vox_mm = float(np.max(spacing))
    if proximity_threshold_mm is None:
        # 2 voxels or 0.3 mm (≈ predentine + partial-volume margin), whichever larger
        proximity_threshold_mm = max(2 * vox_mm, 0.3)

    if lesion.sum() == 0:
        raise ValueError("Empty lesion mask - nothing to classify")

    c, axis = _tooth_axis(seg.tooth, spacing)
    tooth_pts = _mm(np.argwhere(seg.tooth), spacing)
    t_tooth = _project(tooth_pts, c, axis)
    canal_pts = _mm(np.argwhere(seg.canal), spacing)
    t_canal = _project(canal_pts, c, axis)
    # Orient axis so that apical is +. The apex is the NARROW end of the tooth:
    # compare mean perpendicular distance from the axis in the two end-deciles.
    radial = np.linalg.norm((tooth_pts - c) - np.outer(t_tooth, axis), axis=1)
    lo, hi = np.percentile(t_tooth, [10, 90])
    width_low = radial[t_tooth <= lo].mean()
    width_high = radial[t_tooth >= hi].mean()
    shell = ndi.binary_dilation(seg.tooth, iterations=max(int(round(1.0 / vox_mm)), 1)) & ~seg.tooth
    crest_pts = _mm(np.argwhere(seg.bone & shell), spacing)
    flip = width_low < width_high  # narrow end at low t -> apex at low t
    if len(crest_pts) > 100:
        # Bone is the stronger cue: the root (apical end) is surrounded by alveolar bone, the
        # crown is not. Crown-truncated or heavily restored teeth can look narrower than the
        # root, so the width heuristic alone can invert the axis.
        t_sh = _project(crest_pts, c, axis)
        bone_low = float((t_sh <= np.percentile(t_tooth, 30)).sum())
        bone_high = float((t_sh >= np.percentile(t_tooth, 70)).sum())
        if max(bone_low, bone_high) >= 2 * max(min(bone_low, bone_high), 1.0):
            flip = bone_low > bone_high
    if flip:
        axis, t_tooth, t_canal = -axis, -t_tooth, -t_canal
    t_apex = float(apex_level_mm) if apex_level_mm is not None else float(t_tooth.max())

    # Bone crest: most coronal bone voxel within ~1 mm of the tooth surface
    if len(crest_pts):
        t_crest = float(np.percentile(_project(crest_pts, c, axis), 2))
    else:
        t_crest = float(t_tooth.min() + 0.4 * (t_apex - t_tooth.min()))

    if cej_level_mm is not None:
        t_cej = float(cej_level_mm)
    elif volume_norm is not None:
        canal_dil = ndi.binary_dilation(seg.canal, iterations=max(int(round(0.4 / vox_mm)), 1))
        t_cej = _estimate_cej_from_enamel(volume_norm, seg.tooth, spacing, c, axis, exclude=canal_dil)
        # anatomical sanity: the CEJ lies coronal to (or within ~3 mm of) the bone crest
        if t_cej > t_crest + 3.0:
            t_cej = t_crest - 1.5
    else:
        t_cej = t_crest - 1.5

    root_len = max(t_apex - t_cej, 1e-3)

    # ---- height ----
    les_idx = np.argwhere(lesion)
    les_pts = _mm(les_idx, spacing)
    t_les = _project(les_pts, c, axis)
    apical_extent = float(t_les.max() - t_cej)
    coronal_extent = float(t_les.min() - t_cej)
    crest_rel = t_crest - t_cej
    b1, b2 = root_len / 3.0, 2.0 * root_len / 3.0
    if apical_extent > b2:
        height = 4
        hm = apical_extent - b2
    elif apical_extent > b1:
        height = 3
        hm = min(apical_extent - b1, b2 - apical_extent)
    elif apical_extent > crest_rel:
        height = 2
        hm = min(apical_extent - crest_rel, b1 - apical_extent)
    else:
        height = 1
        hm = crest_rel - apical_extent

    # ---- circumferential spread ----
    # Use angle around the canal/tooth axis in the plane perpendicular to it.
    e1 = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    rel = les_pts - c
    ang = np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360.0
    bins = np.zeros(360, dtype=bool)
    bins[np.floor(ang).astype(int) % 360] = True
    # close small angular gaps caused by segmentation noise (circular closing)
    k = int(angular_gap_close_deg)
    if k > 0:
        padded = np.concatenate([bins[-k:], bins, bins[:k]])
        padded = ndi.binary_closing(padded, structure=np.ones(k))
        bins = padded[k:-k]
    spread_deg = float(bins.sum())
    if spread_deg <= 90:
        spread, sm = "A", 90 - spread_deg
    elif spread_deg <= 180:
        spread, sm = "B", min(spread_deg - 90, 180 - spread_deg)
    elif spread_deg <= 270:
        spread, sm = "C", min(spread_deg - 180, 270 - spread_deg)
    else:
        spread, sm = "D", spread_deg - 270

    # ---- proximity ----
    contact = int((lesion & seg.canal).sum())
    if seg.canal.any():
        dist_to_canal = ndi.distance_transform_edt(~seg.canal, sampling=spacing)
        dmin = float(dist_to_canal[lesion].min())
    else:
        dmin = float("inf")
    if contact > 0:
        dmin = 0.0
    proximity = "p" if dmin <= proximity_threshold_mm else "d"
    pm = abs(dmin - proximity_threshold_mm)

    # ---- extras ----
    vol_mm3 = float(lesion.sum() * np.prod(spacing))
    calc_frac = 0.0
    if volume_norm is not None:
        from .preprocess import quantise, local_grey_level_emphasis
        q = quantise(volume_norm)
        hgre = local_grey_level_emphasis(q, radius=5)["HGRE"][lesion]
        # bimodal split (k=2 proxy): voxels above the median of healthy dentine HGRE
        dent = local_grey_level_emphasis(q, radius=5)["HGRE"][seg.tooth & ~lesion & ~seg.canal]
        thr = float(np.percentile(dent, 50)) if dent.size else 0.5
        calc_frac = float((hgre >= thr).mean())

    portal = None
    if len(t_les):
        top = np.argsort(t_les)[: max(len(t_les) // 20, 1)]  # most coronal 5 %
        a = ang[top]
        portal = float(np.degrees(np.arctan2(np.sin(np.radians(a)).mean(),
                                             np.cos(np.radians(a)).mean())) % 360)
        if buccal_direction is not None:
            bd = np.asarray(buccal_direction, float)
            bd = bd - (bd @ axis) * axis
            ref = np.degrees(np.arctan2(bd @ e2, bd @ e1)) % 360
            portal = float((portal - ref) % 360)  # 0° = buccal, 180° = lingual/palatal

    borderline = (hm < 0.5) or (sm < 10) or (pm < vox_mm)

    return PatelParameters(
        height=height, spread=spread, proximity=proximity,
        lesion_volume_mm3=vol_mm3,
        apical_extent_mm=apical_extent, coronal_extent_mm=coronal_extent,
        bone_crest_mm=float(crest_rel), root_length_mm=float(root_len),
        spread_deg=spread_deg, min_canal_distance_mm=dmin,
        canal_contact_voxels=contact, calcified_fraction=calc_frac,
        portal_angle_deg=portal,
        height_margin_mm=float(hm), spread_margin_deg=float(sm),
        proximity_margin_mm=float(pm), borderline=bool(borderline),
    )


def _estimate_cej_from_enamel(vol: np.ndarray, tooth: np.ndarray, spacing,
                              c: np.ndarray, axis: np.ndarray, exclude: np.ndarray = None) -> float:
    """CEJ ≈ most apical level at which a high-intensity (enamel) shell exists.

    `exclude` (e.g. the dilated canal) removes root fillings / posts, which are
    brighter than enamel and would otherwise drag the estimate towards the apex.
    Enamel is a surface shell: only the outer 35 % of the local radius is used."""
    region = tooth & ~exclude if exclude is not None else tooth
    idx = np.argwhere(region)
    pts = _mm(idx, spacing)
    t = _project(pts, c, axis)
    radial = np.linalg.norm((pts - c) - np.outer(t, axis), axis=1)
    nb = 40
    bins = np.clip(((t - t.min()) / (np.ptp(t) + 1e-9) * (nb - 1)).astype(int), 0, nb - 1)
    local_r = np.array([np.percentile(radial[bins == b], 98) if (bins == b).any() else 0 for b in range(nb)])
    outer = radial >= 0.65 * local_r[bins]
    inten = vol[tuple(idx.T)]
    ref = inten[outer] if outer.sum() > 200 else inten
    enamel = outer & (inten >= np.percentile(ref, 90))
    if enamel.sum() < 50:
        return float(np.percentile(t, 35))
    return float(np.percentile(t[enamel], 98))
