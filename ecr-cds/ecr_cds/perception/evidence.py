"""Provenance-preserving multiplanar evidence frames.

For one candidate: linked axial / coronal / sagittal crops at the centroid, neighbouring axial slices
(so a single slice never stands for the volume), a tooth-axis oblique reformat (long-axis and
perpendicular), plus the full-slice context view with the crop box drawn.  Every panel records the
source volume id, voxel spacing, crop offset, slice indices and the display transform so a reader can
go back to the exact voxels.  Generative enhancement is never applied.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .display import apply_window, clahe2d, display_transform_record, unsharp2d


@dataclass
class Panel:
    name: str
    plane: str                         # axial | coronal | sagittal | long_axis | short_axis | context
    index: Optional[int]               # slice index in the *volume* frame, when applicable
    extent_volume_zyx: List[List[int]]  # [[z0,z1],[y0,y1],[x0,x1]] in volume voxels
    display: Dict
    note: str = ""


@dataclass
class EvidenceFrame:
    candidate_id: int
    source_id: str
    spacing_zyx_mm: List[float]
    crop_offset_zyx: List[int]
    centroid_volume_zyx: List[float]
    tooth_axis_zyx: Optional[List[float]]
    panels: List[Panel] = field(default_factory=list)
    png: Optional[bytes] = None

    def to_dict(self, with_png: bool = False) -> Dict:
        d = {k: v for k, v in asdict(self).items() if k != "png"}
        if with_png and self.png is not None:
            import base64
            d["png_base64"] = base64.b64encode(self.png).decode()
        return d


def _oblique_slab(raw: np.ndarray, centre: np.ndarray, axis: np.ndarray, spacing, half_mm: float, step_mm: float = None):
    """Resample a plane containing the tooth axis (long axis view) and the plane perpendicular to it."""
    sp = np.asarray(spacing, float)
    step = step_mm or float(sp.min())
    a = axis / (np.linalg.norm(axis) + 1e-9)
    # build an orthonormal frame (a, u, v) in mm space
    ref = np.array([0, 1.0, 0]) if abs(a[1]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(a, ref); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(a, u)
    n = int(2 * half_mm / step) + 1
    t = (np.arange(n) - n // 2) * step
    c_mm = centre * sp
    def sample(e1, e2):
        g1, g2 = np.meshgrid(t, t, indexing="ij")
        pts = c_mm[None, None, :] + g1[..., None] * e1 + g2[..., None] * e2
        vox = (pts / sp).reshape(-1, 3).T
        return ndi.map_coordinates(raw, vox, order=1, mode="nearest").reshape(n, n)
    return sample(a, u), sample(u, v), {"u": u.round(3).tolist(), "v": v.round(3).tolist(), "step_mm": step, "n": n}


def build_evidence_frame(raw: np.ndarray, seg, labels: np.ndarray, cid: int, spacing, crop_offset_zyx, source_id: str,
                         window: Tuple[float, float], tooth_axis: Optional[np.ndarray] = None, half_mm: float = 6.0,
                         neighbour_mm: float = 0.8, full_slice: Optional[np.ndarray] = None, clahe: bool = False,
                         sharpen: bool = False, counterfactual_fill: Optional[float] = None) -> EvidenceFrame:
    """Render the 8-panel evidence frame for candidate `cid`.

    sharpen:            optional unsharp-mask display transform (recorded in each panel's display record).
    counterfactual_fill: when given, the candidate voxels are replaced by this value (e.g. the dentine reference
                        level) and the candidate overlay is omitted.  Used only for the perturbation-based grounding
                        test of a vision-language model (CLEAR-style); never part of the diagnostic evidence path.
                        The frame title states that it is a counterfactual.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = labels == cid
    c = np.argwhere(m).mean(0)
    ci = c.astype(int)
    cf = counterfactual_fill is not None
    if cf:
        raw = raw.astype(np.float32).copy()
        raw[ndi.binary_dilation(m, iterations=1)] = float(counterfactual_fill)
        m_draw = np.zeros_like(m)
    else:
        m_draw = m
    sp = np.asarray(spacing, float)
    h = (half_mm / sp).astype(int)
    lo = np.maximum(ci - h, 0); hi = np.minimum(ci + h, np.asarray(raw.shape) - 1)
    off = np.asarray(crop_offset_zyx, int)
    dz = max(1, int(round(neighbour_mm / sp[0])))
    disp_rec = display_transform_record(window, {"tiles": 8, "clip_limit": 0.01} if clahe else None,
                                        {"sigma": 1.0, "amount": 1.0} if sharpen else None)
    if cf:
        disp_rec = {**disp_rec, "counterfactual_fill": float(counterfactual_fill)}

    def show(ax, img, overlays, title):
        im = apply_window(img, window)
        if clahe:
            im = clahe2d(im)
        if sharpen:
            im = unsharp2d(im)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        for arr, col, lw in overlays:
            if arr is not None and arr.any():
                ax.contour(arr, levels=[0.5], colors=col, linewidths=lw)
        ax.set_title(title, fontsize=8); ax.axis("off")

    panels: List[Panel] = []
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
    sl = (slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    # row 1: axial neighbours (z-dz, z, z+dz) + coronal
    for k, z in enumerate([ci[0] - dz, ci[0], ci[0] + dz]):
        z = int(np.clip(z, 0, raw.shape[0] - 1))
        show(axs[0, k], raw[z][sl], [(seg.tooth[z][sl], "cyan", 0.6), (seg.canal[z][sl], "yellow", 0.6), (m_draw[z][sl], "red", 0.8)],
             f"axial z={z + off[0]} ({'centre' if k == 1 else f'{(z - ci[0]) * sp[0]:+.1f} mm'})")
        panels.append(Panel(f"axial_{k}", "axial", int(z + off[0]), [[int(z + off[0])] * 2, [int(lo[1] + off[1]), int(hi[1] + off[1])],
                                                                   [int(lo[2] + off[2]), int(hi[2] + off[2])]], disp_rec))
    show(axs[0, 3], raw[lo[0]:hi[0], ci[1], lo[2]:hi[2]], [(seg.tooth[lo[0]:hi[0], ci[1], lo[2]:hi[2]], "cyan", 0.6),
                                                           (seg.canal[lo[0]:hi[0], ci[1], lo[2]:hi[2]], "yellow", 0.6),
                                                           (m_draw[lo[0]:hi[0], ci[1], lo[2]:hi[2]], "red", 0.8)], f"coronal y={ci[1] + off[1]}")
    panels.append(Panel("coronal", "coronal", int(ci[1] + off[1]), [[int(lo[0] + off[0]), int(hi[0] + off[0])], [int(ci[1] + off[1])] * 2,
                                                                    [int(lo[2] + off[2]), int(hi[2] + off[2])]], disp_rec))
    # row 2: sagittal, long-axis reformat, short-axis reformat, context
    show(axs[1, 0], raw[lo[0]:hi[0], lo[1]:hi[1], ci[2]], [(seg.tooth[lo[0]:hi[0], lo[1]:hi[1], ci[2]], "cyan", 0.6),
                                                           (seg.canal[lo[0]:hi[0], lo[1]:hi[1], ci[2]], "yellow", 0.6),
                                                           (m_draw[lo[0]:hi[0], lo[1]:hi[1], ci[2]], "red", 0.8)], f"sagittal x={ci[2] + off[2]}")
    panels.append(Panel("sagittal", "sagittal", int(ci[2] + off[2]), [[int(lo[0] + off[0]), int(hi[0] + off[0])],
                                                                      [int(lo[1] + off[1]), int(hi[1] + off[1])], [int(ci[2] + off[2])] * 2], disp_rec))
    axis_rec = None
    if tooth_axis is not None and np.linalg.norm(tooth_axis) > 0:
        long_img, short_img, frame = _oblique_slab(raw, c, np.asarray(tooth_axis, float), sp, half_mm)
        long_m, short_m, _ = _oblique_slab(m_draw.astype(np.float32), c, np.asarray(tooth_axis, float), sp, half_mm)
        long_c, short_c, _ = _oblique_slab(seg.canal.astype(np.float32), c, np.asarray(tooth_axis, float), sp, half_mm)
        show(axs[1, 1], long_img, [(long_c > 0.5, "yellow", 0.6), (long_m > 0.5, "red", 0.8)], "tooth long-axis reformat")
        show(axs[1, 2], short_img, [(short_c > 0.5, "yellow", 0.6), (short_m > 0.5, "red", 0.8)], "perpendicular to tooth axis")
        axis_rec = frame
        ext = [[int(lo[i] + off[i]), int(hi[i] + off[i])] for i in range(3)]
        panels.append(Panel("long_axis", "long_axis", None, ext, {**disp_rec, "oblique_frame": frame}, "plane spanned by tooth axis and u"))
        panels.append(Panel("short_axis", "short_axis", None, ext, {**disp_rec, "oblique_frame": frame}, "plane spanned by u and v"))
    else:
        for k in (1, 2):
            axs[1, k].axis("off"); axs[1, k].set_title("tooth axis unavailable", fontsize=8)
    if full_slice is not None:
        show(axs[1, 3], full_slice, [], f"full slice z={ci[0] + off[0]} (crop box)")
        axs[1, 3].add_patch(plt.Rectangle((lo[2] + off[2], lo[1] + off[1]), hi[2] - lo[2], hi[1] - lo[1], fill=False, ec="red", lw=0.8))
        panels.append(Panel("context", "context", int(ci[0] + off[0]), [[int(ci[0] + off[0])] * 2, [0, full_slice.shape[0]], [0, full_slice.shape[1]]], disp_rec,
                            "unmodified full axial slice; red box = crop"))
    else:
        axs[1, 3].axis("off")
    fig.suptitle((f"COUNTERFACTUAL (candidate filled, not evidence) - " if cf else "")
                 + f"Candidate {cid} - source {source_id} - spacing {sp.round(3).tolist()} mm - tooth cyan, canal proxy yellow"
                 + ("" if cf else ", candidate red") + f"{' - CLAHE display' if clahe else ''}{' - unsharp display' if sharpen else ''}", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return EvidenceFrame(candidate_id=cid, source_id=source_id, spacing_zyx_mm=sp.round(4).tolist(), crop_offset_zyx=off.tolist(),
                         centroid_volume_zyx=(c + off).round(1).tolist(),
                         tooth_axis_zyx=None if tooth_axis is None else np.asarray(tooth_axis, float).round(3).tolist(),
                         panels=panels, png=buf.getvalue())
