"""Quantitative visual descriptors of a candidate defect in the CBCT crop.

These are the "visual information" the reasoning agents work from.  Each
descriptor is a measurement on the voxel grid; the radiological *meaning*
(e.g. "small portal with wide internal spread suggests ECR") lives in
`criteria.py`, not here, so the two can be audited separately.

Everything here depends on the automatic tooth / canal masks and the
threshold-based candidate map.  If those are wrong the descriptors are wrong.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..segmentation import ToothSegmentation
from ..auto_segment import tooth_hull_convex


def _ball_mm(radius, spacing):
    r = np.ceil(radius / np.asarray(spacing)).astype(int)
    zz, yy, xx = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    return sum((g * s) ** 2 for g, s in zip((zz, yy, xx), spacing)) <= radius ** 2


def _angular_spread_deg(angles_deg: np.ndarray, bin_deg: float = 6.0, close_gap_deg: float = 6.0) -> float:
    if angles_deg.size == 0:
        return 0.0
    nb = int(round(360 / bin_deg))
    occ = np.zeros(nb, bool)
    occ[(np.floor((angles_deg % 360) / bin_deg).astype(int)) % nb] = True
    # close single-bin gaps (partial volume)
    k = int(round(close_gap_deg / bin_deg))
    if k > 0:
        occ = ndi.binary_closing(np.r_[occ, occ, occ], structure=np.ones(2 * k + 1))[nb:2 * nb]
    return float(occ.sum() * bin_deg)


@dataclass
class CandidateDescriptors:
    component_id: int
    volume_mm3: float
    coronal_extent_from_cej_mm: Optional[float]   # negative = coronal to CEJ
    apical_extent_from_cej_mm: Optional[float]    # positive = apical to CEJ
    fraction_subcrestal: Optional[float]
    bone_crest_from_cej_mm: Optional[float]
    cervical_surface_connection: Optional[bool]
    external_surface_connection: bool
    portal_area_mm2: float
    max_cross_section_mm2: float
    undermining_ratio: float          # max internal cross-section / portal area (>1: wider inside)
    max_depth_mm: float
    angular_spread_deg: float
    min_canal_distance_mm: Optional[float]
    canal_wall_intact: Optional[bool]
    margin_irregularity: float        # margin roughness at 0.3 mm (fraction of voxels changed by opening+closing)
    elongation_axial: float           # extent along axis / mean radial extent
    mean_intensity_rel_dentine: float
    intensity_heterogeneity: float    # std / dentine mode
    radiopaque_inclusion_fraction: float   # mottled interior (repair tissue proxy)
    dense_material_contact_fraction: float  # enamel / restoration within 0.4 mm
    near_metal_fraction: float        # voxels within 3 mm of metal-level intensity
    stable_core_fraction: float
    threshold_expansion_ratio: float
    touches_crop_boundary: bool
    landmark_source: str
    landmark_flags: List[str]
    # --- cue-level descriptors for the diagnostic questions (rev 2.5) ---
    eccentricity_ratio: Optional[float] = None     # lesion centroid offset from canal axis / local root radius (0 = centred on canal)
    canal_traceable_fraction: Optional[float] = None  # fraction of lesion-spanned axial levels where the canal is still identifiable
    canal_expansion_ratio: Optional[float] = None  # canal cross-section inside the lesion span / just outside it (>1.5 = canal balloons)
    portal_angular_width_deg: float = 0.0          # angular width of the surface opening
    wrap_ratio: float = 0.0                        # angular spread / portal angular width (>2: enters from one side and wraps)
    portal_slice_persistence: float = 0.0          # fraction of lesion axial levels with a surface breach (external contour interruption)
    n_axial_levels: int = 0                        # consecutive axial levels spanned
    extent_y_mm: float = 0.0
    extent_x_mm: float = 0.0
    persistent_3d: bool = False                    # >= 3 levels and >= 0.3 mm in every in-plane direction
    asymmetry_index: float = 0.0                   # 1 - (angular mass balance); 1 = all on one side

    def to_dict(self) -> Dict:
        return asdict(self)


def describe_candidates(raw: np.ndarray, seg: ToothSegmentation, labels: np.ndarray,
                        spacing, geometry: Dict, screen_rows: List[Dict],
                        t_tooth: float, t_dense: float, dentine_mode: float,
                        metal_level: Optional[float] = None) -> List[CandidateDescriptors]:
    """Measure every non-rejected candidate in `labels` (0 = background)."""
    spacing = np.asarray(spacing, float)
    vox_vol = float(np.prod(spacing))
    vox_area = float(spacing[1] * spacing[2])
    centre, axis, cej = geometry["centre_mm"], np.asarray(geometry["axis"], float), geometry["cej_t_mm"]
    hull = tooth_hull_convex(seg.tooth)
    depth = ndi.distance_transform_edt(hull, sampling=spacing)
    canal_dist = ndi.distance_transform_edt(~seg.canal, sampling=spacing) if seg.canal.any() else None
    dense = hull & (raw >= t_dense)
    dense_dist = ndi.distance_transform_edt(~dense, sampling=spacing) if dense.any() else None
    if metal_level is None:   # volume-relative level (CBCT grey values are not HU): see perception.display
        from ..perception.display import tissue_reference_levels
        metal_level = tissue_reference_levels(raw, seg.tooth, seg.canal, t_dense)["metal_level"]
    # a root filling inside the canal proxy is not counted as metal (it is flagged separately by the anatomy agent)
    metal = (raw >= metal_level) & ~ndi.binary_dilation(seg.canal, structure=_ball_mm(0.3, spacing))
    metal_dist = ndi.distance_transform_edt(~metal, sampling=spacing) if metal.any() else None
    # bone crest along the axis (2nd percentile of bone voxels in a 1 mm shell)
    shell = ndi.binary_dilation(seg.tooth, structure=_ball_mm(1.0, spacing)) & ~seg.tooth
    bpts = np.argwhere(shell & seg.bone) * spacing
    crest_t = float(np.percentile((bpts - centre) @ axis, 2)) if len(bpts) >= 100 else None
    rows = {r["component_id"]: r for r in screen_rows}
    out: List[CandidateDescriptors] = []
    for cid in np.unique(labels):
        if cid == 0:
            continue
        m = labels == cid
        q = np.argwhere(m)
        ix = tuple(q.T)
        pts = q * spacing
        t = (pts - centre) @ axis
        radial_vec = (pts - centre) - t[:, None] * axis
        n = len(q)
        # reference direction for angles: any vector orthogonal to axis
        ref = np.cross(axis, [1., 0., 0.])
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(axis, [0., 1., 0.])
        ref /= np.linalg.norm(ref)
        ref2 = np.cross(axis, ref)
        ang = np.degrees(np.arctan2(radial_vec @ ref2, radial_vec @ ref))
        on_surface = depth[ix] <= 0.25
        portal_area = float(on_surface.sum() * vox_area)
        # cross-sections perpendicular to the axis: bin t at voxel pitch
        pitch = float(spacing.min())
        bins = np.floor((t - t.min()) / pitch).astype(int)
        cs = np.bincount(bins) * vox_area
        max_cs = float(cs.max()) if cs.size else 0.0
        undermining = float(max_cs / portal_area) if portal_area > 0 else float("inf")
        # morphology on a padded bounding box only (full-crop 3-D openings are too slow for 500-slice scans)
        vol = n * vox_vol
        pad = np.ceil(0.5 / spacing).astype(int) + 1
        lo_b = np.maximum(q.min(0) - pad, 0); hi_b = np.minimum(q.max(0) + pad + 1, np.asarray(m.shape))
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo_b, hi_b))
        mb, rawb, hullb, canalb = m[box], raw[box], hull[box], seg.canal[box]
        # margin roughness at the 0.3 mm scale: protrusions removed by opening + indentations filled by closing
        b3 = _ball_mm(0.3, spacing)
        opened = ndi.binary_opening(mb, structure=b3)
        closed3 = ndi.binary_closing(mb, structure=b3)
        irregularity = float(((mb & ~opened).sum() + (closed3 & ~mb).sum()) / n)
        radial_extent = float(np.linalg.norm(radial_vec, axis=1).std() * 2 + pitch)
        elongation = float((np.ptp(t) + pitch) / radial_extent) if radial_extent > 0 else 0.0
        vals = raw[ix].astype(float)
        # mottled interior: radiopaque islands enclosed by the defect envelope (closing then erosion by
        # the same radius removes the dentine rim that a plain closing would add)
        b4 = _ball_mm(0.4, spacing)
        envelope = ndi.binary_erosion(ndi.binary_closing(mb, structure=b4), structure=b4) & hullb & ~canalb
        inclusion = float(((envelope & ~mb) & (rawb >= t_tooth)).sum() / envelope.sum()) if envelope.sum() > 50 else 0.0
        # ---- cue-level descriptors --------------------------------------------------
        levels = np.unique(bins)
        n_levels = int(levels.size)
        surf_levels = np.unique(bins[on_surface]) if on_surface.any() else np.array([], int)
        portal_persist = float(surf_levels.size / max(1, n_levels))
        portal_width = _angular_spread_deg(ang[on_surface]) if on_surface.any() else 0.0
        total_spread = _angular_spread_deg(ang)
        wrap = float(total_spread / portal_width) if portal_width > 0 else 0.0
        ext_y = float(np.ptp(q[:, 1]) * spacing[1] + spacing[1]); ext_x = float(np.ptp(q[:, 2]) * spacing[2] + spacing[2])
        persistent = bool(n_levels >= 3 and ext_y >= 0.3 and ext_x >= 0.3)
        # angular mass balance: resultant length of unit vectors (1 = all on one side, 0 = symmetric)
        asym = float(np.hypot(np.cos(np.radians(ang)).mean(), np.sin(np.radians(ang)).mean()))
        ecc = trace_frac = expansion = None
        tooth_pts = np.argwhere(seg.tooth) * spacing
        tt = (tooth_pts - centre) @ axis
        tn = np.abs(tt - t.mean()) <= 0.5
        root_r = float(np.percentile(np.linalg.norm((tooth_pts[tn] - centre) - tt[tn, None] * axis, axis=1), 95)) if tn.sum() > 10 else 3.0
        if seg.canal.any():
            cpts = np.argwhere(seg.canal) * spacing
            ct = (cpts - centre) @ axis
            near = np.abs(ct - t.mean()) <= 1.0
            if near.sum() >= 5:
                c_centre = ((cpts[near] - centre) - ct[near, None] * axis).mean(0)
                ecc = float(np.linalg.norm(radial_vec.mean(0) - c_centre) / max(root_r, 1e-3))
            # canal identifiable at each lesion level?
            cbins = np.floor((ct - t.min()) / pitch).astype(int)
            inside = (cbins >= 0) & (cbins <= int(levels.max()))
            trace_frac = float(np.isin(levels, np.unique(cbins[inside])).mean())
            # canal space per level inside the span (canal voxels + lesion voxels in the central zone)
            # versus canal voxels per level in the 1 mm bands just above and below
            out_band = ((ct < t.min()) & (ct >= t.min() - 1.0)) | ((ct > t.max()) & (ct <= t.max() + 1.0))
            n_out_levels = max(1.0, 2.0 / pitch)
            canal_out_per_level = float(out_band.sum()) / n_out_levels
            if canal_out_per_level > 0:
                r_canal_out = float(np.sqrt(canal_out_per_level * vox_area / np.pi))
                central = np.linalg.norm(radial_vec, axis=1) <= max(0.5 * root_r, r_canal_out + 0.5)
                expansion = float((inside.sum() + central.sum()) / n_levels / canal_out_per_level)
        r = rows.get(int(cid), {})
        out.append(CandidateDescriptors(
            component_id=int(cid), volume_mm3=round(vol, 4),
            coronal_extent_from_cej_mm=None if cej is None else round(float(t.min() - cej), 3),
            apical_extent_from_cej_mm=None if cej is None else round(float(t.max() - cej), 3),
            fraction_subcrestal=None if crest_t is None else round(float((t >= crest_t).mean()), 3),
            bone_crest_from_cej_mm=None if (crest_t is None or cej is None) else round(crest_t - cej, 3),
            cervical_surface_connection=r.get("cervical_surface_connection"),
            external_surface_connection=bool(on_surface.any()),
            portal_area_mm2=round(portal_area, 4), max_cross_section_mm2=round(max_cs, 4),
            undermining_ratio=round(undermining, 3) if np.isfinite(undermining) else 99.0,
            max_depth_mm=round(float(depth[ix].max()), 3),
            angular_spread_deg=round(_angular_spread_deg(ang), 1),   # all voxels, as in Patel's axial spread
            min_canal_distance_mm=None if canal_dist is None else round(float(canal_dist[ix].min()), 3),
            canal_wall_intact=None if canal_dist is None else bool(canal_dist[ix].min() >= max(2 * pitch, 0.3)),
            margin_irregularity=round(irregularity, 3), elongation_axial=round(elongation, 3),
            mean_intensity_rel_dentine=round(float(vals.mean() / dentine_mode), 3) if dentine_mode else 0.0,
            intensity_heterogeneity=round(float(vals.std() / dentine_mode), 3) if dentine_mode else 0.0,
            radiopaque_inclusion_fraction=round(inclusion, 3),
            dense_material_contact_fraction=round(float((dense_dist[ix] <= 0.4).mean()), 3) if dense_dist is not None else 0.0,
            near_metal_fraction=round(float((metal_dist[ix] <= 3.0).mean()), 3) if metal_dist is not None else 0.0,
            stable_core_fraction=float(r.get("stable_core_fraction", 1.0)),
            threshold_expansion_ratio=float(r.get("upper_threshold_expansion_ratio", 1.0)),
            touches_crop_boundary="candidate_touches_crop_boundary" in r.get("flags", []),
            landmark_source=str(geometry.get("source", "automatic_unreviewed")),
            landmark_flags=list(geometry.get("flags", [])),
            eccentricity_ratio=None if ecc is None else round(ecc, 3),
            canal_traceable_fraction=None if trace_frac is None else round(trace_frac, 3),
            canal_expansion_ratio=None if expansion is None else round(expansion, 3),
            portal_angular_width_deg=round(float(portal_width), 1), wrap_ratio=round(wrap, 2),
            portal_slice_persistence=round(portal_persist, 3), n_axial_levels=n_levels,
            extent_y_mm=round(ext_y, 3), extent_x_mm=round(ext_x, 3), persistent_3d=persistent,
            asymmetry_index=round(asym, 3),
        ))
    return out


def render_candidate_tiles(raw: np.ndarray, seg: ToothSegmentation, labels: np.ndarray, cid: int,
                           spacing, window: Optional[tuple] = None, half_mm: float = 6.0) -> bytes:
    """Three orthogonal crops through the candidate centroid as one PNG (for humans and VLMs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = labels == cid
    c = np.argwhere(m).mean(0).astype(int)
    h = (half_mm / np.asarray(spacing)).astype(int)
    lo = np.maximum(c - h, 0)
    hi = np.minimum(c + h, np.asarray(raw.shape) - 1)
    if window is None:
        ref = raw[seg.tooth & ~seg.canal] if (seg.tooth & ~seg.canal).any() else raw[seg.tooth]
        window = (float(np.percentile(raw[~seg.tooth], 5)), float(np.percentile(ref, 99)) * 1.05)   # soft tissue .. dentine/enamel
    vmin, vmax = window
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    views = [("axial", raw[c[0], lo[1]:hi[1], lo[2]:hi[2]], m[c[0], lo[1]:hi[1], lo[2]:hi[2]],
              seg.canal[c[0], lo[1]:hi[1], lo[2]:hi[2]], seg.tooth[c[0], lo[1]:hi[1], lo[2]:hi[2]]),
             ("coronal", raw[lo[0]:hi[0], c[1], lo[2]:hi[2]], m[lo[0]:hi[0], c[1], lo[2]:hi[2]],
              seg.canal[lo[0]:hi[0], c[1], lo[2]:hi[2]], seg.tooth[lo[0]:hi[0], c[1], lo[2]:hi[2]]),
             ("sagittal", raw[lo[0]:hi[0], lo[1]:hi[1], c[2]], m[lo[0]:hi[0], lo[1]:hi[1], c[2]],
              seg.canal[lo[0]:hi[0], lo[1]:hi[1], c[2]], seg.tooth[lo[0]:hi[0], lo[1]:hi[1], c[2]])]
    for ax, (name, img, mm, cc, tt) in zip(axs, views):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        if tt.any():
            ax.contour(tt, levels=[0.5], colors="cyan", linewidths=0.6)
        if cc.any():
            ax.contour(cc, levels=[0.5], colors="yellow", linewidths=0.6)
        if mm.any():
            ax.contour(mm, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(f"{name} (candidate {cid})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Research view: tooth cyan, canal proxy yellow, candidate red", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
