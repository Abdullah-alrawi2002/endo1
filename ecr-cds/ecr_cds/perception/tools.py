"""Perception toolkit: the only route from the agent layer to the voxels.

Perception layer (this module): identify teeth, propose suspicious regions, delineate a clinician-selected
structure, measure, reformat, and *re-verify* a region under a different processing path.  Each call is
logged with its arguments so the conference transcript shows which tools were used and why.  The agent
layer (roles / orchestrator) selects tools, asks for extra views, tracks coverage and checks completeness;
it never edits voxel-level results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..auto_segment import auto_segment, score_candidates_radiolucency, choose_seed, AutoToothSegmenter
from ..preprocess import preprocess_volume
from ..segmentation import ToothSegmentation
from .display import tissue_reference_levels, metal_mask, window_from_levels
from .evidence import build_evidence_frame


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    summary: Dict[str, Any]
    seconds: float
    t: float = field(default_factory=time.time)


class PerceptionToolkit:
    """Registry of perception tools with a call log (`self.log`)."""

    def __init__(self):
        self.log: List[ToolCall] = []
        self._tools: Dict[str, Callable] = {
            "segment_teeth": self.segment_teeth,
            "propose_regions": self.propose_regions,
            "delineate_tooth": self.delineate_tooth,
            "reference_levels": self.reference_levels,
            "metal_mask": self.metal_mask,
            "evidence_frame": self.evidence_frame,
            "reproduce_candidate": self.reproduce_candidate,
        }

    def names(self) -> List[str]:
        return list(self._tools)

    def call(self, name: str, **kwargs):
        t0 = time.time()
        out = self._tools[name](**kwargs)
        summ = out.get("summary", {}) if isinstance(out, dict) else {}
        self.log.append(ToolCall(name, {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else type(v).__name__)
                                        for k, v in kwargs.items()}, summ, round(time.time() - t0, 3)))
        return out

    def log_dicts(self) -> List[Dict]:
        return [{"tool": c.tool, "args": c.args, "summary": c.summary, "seconds": c.seconds} for c in self.log]

    # ------------------------------------------------------------------ tools
    def segment_teeth(self, array: np.ndarray, spacing) -> Dict:
        auto = auto_segment(array, spacing)
        return {"auto": auto, "summary": {**auto.summary(), "n_teeth": len(auto.candidates)}}

    def propose_regions(self, auto, array: np.ndarray, spacing) -> Dict:
        scores = score_candidates_radiolucency(auto, array, spacing)
        seed, why = choose_seed(scores)
        return {"scores": scores, "seed_zyx": seed, "reason": why,
                "summary": {"n_scored_teeth": len(scores), "seed_zyx": list(map(int, seed)), "reason": why}}

    def delineate_tooth(self, auto, seed_zyx, array: np.ndarray, spacing, canal_mask: Optional[np.ndarray] = None) -> Dict:
        segm = AutoToothSegmenter(auto, seed=tuple(seed_zyx))
        if canal_mask is not None:
            segm.canal_mask = canal_mask
        seg = segm(array, spacing)
        return {"segmenter": segm, "seg": seg, "summary": {"tooth_voxels": int(seg.tooth.sum()), "canal_voxels": int(seg.canal.sum()),
                                                            "canal_mode": str(getattr(segm, "canal_mode", "automatic"))}}

    def reference_levels(self, raw, seg: ToothSegmentation, t_dense: float) -> Dict:
        lv = tissue_reference_levels(raw, seg.tooth, seg.canal, t_dense)
        return {"levels": lv, "summary": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in lv.items() if k != "method"}}

    def metal_mask(self, raw, metal_level: float, spacing, exclude=None) -> Dict:
        mm = metal_mask(raw, metal_level, spacing, exclude=exclude)
        return {**mm, "summary": {"metal_fraction": round(mm["metal_fraction"], 5), "streak_zone_fraction": round(float(mm["streak_zone"].mean()), 4)}}

    def evidence_frame(self, raw, seg, labels, cid: int, spacing, crop_offset_zyx, source_id: str, levels: Dict,
                       tooth_axis=None, full_slice=None, clahe: bool = False, sharpen: bool = False, counterfactual: bool = False) -> Dict:
        fill = float(levels.get("dentine", levels.get("enamel", 0))) if counterfactual else None
        fr = build_evidence_frame(raw, seg, labels, cid, spacing, crop_offset_zyx, source_id, window_from_levels(levels, "tooth"),
                                  tooth_axis=tooth_axis, full_slice=full_slice, clahe=clahe, sharpen=sharpen, counterfactual_fill=fill)
        return {"frame": fr, "summary": {"candidate_id": cid, "n_panels": len(fr.panels), "centroid_volume_zyx": fr.centroid_volume_zyx,
                                         "counterfactual": bool(counterfactual), "sharpen": bool(sharpen)}}

    TTA_PATHS = (("gain", 0.95), ("gain", 1.05), ("noise", 0.02), ("thresh", 0.97), ("thresh", 1.03), ("smooth", 1.0))

    def reproduce_candidate(self, raw, seg: ToothSegmentation, labels: np.ndarray, cid: int, spacing, t_tooth: float, soft_level: float,
                            tooth_axis=None, sigmas=(0.0, 2.0), hull_mask: Optional[np.ndarray] = None,
                            mode: str = "smooth", tta_paths=None, seed: int = 0) -> Dict:
        """Critic re-inspection: re-delineate the hypodense region under alternative processing paths, at the
        candidate's own location, and measure overlap (Dice) with the original component; also count long-axis
        planes in which it persists.

        mode="smooth": Gaussian smoothing at `sigmas` (rev 2.7 behaviour; frozen baseline).
        mode="tta":    test-time-augmentation paths - multiplicative gain, additive noise (fraction of the
                       dentine-soft contrast), threshold jitter and smoothing.  TTA gave the strongest
                       error-detection signal under matched budgets in the decision-aware uncertainty protocol
                       (2026); whether that transfers to CBCT candidates is what the ablation tests.

        Returns reproducibility in [0,1] (mean Dice across paths) and persistence along the tooth axis.
        A low value means the finding depends on one processing path - not that it is absent.
        """
        comp = labels == cid
        if not comp.any():
            return {"reproducibility": 0.0, "summary": {"candidate_id": cid, "reproducibility": 0.0, "note": "empty component"}}
        hull = hull_mask if hull_mask is not None else ndi.binary_closing(seg.tooth, iterations=6)
        dil = ndi.binary_dilation(comp, iterations=3)
        rng = np.random.default_rng(seed)
        contrast = max(float(t_tooth - soft_level), 1.0)
        paths = [("smooth", float(s)) for s in sigmas] if mode == "smooth" else list(tta_paths or self.TTA_PATHS)
        dices, keys = [], []
        base = raw.astype(np.float32)
        for kind, par in paths:
            v, thr = base, t_tooth
            if kind == "smooth" and par > 0:
                v = ndi.gaussian_filter(base, par)
            elif kind == "gain":
                v = base * par
            elif kind == "noise":
                v = base + rng.normal(0.0, par * contrast, base.shape).astype(np.float32)
            elif kind == "thresh":
                thr = t_tooth * par
            hypo = (v < thr) & (v > soft_level * 0.5) & hull & ~seg.canal & dil
            keys.append(f"{kind}:{par:g}")
            lab, n = ndi.label(hypo)
            if n == 0:
                dices.append(0.0); continue
            # take the component(s) overlapping the original
            ids = np.unique(lab[comp]); ids = ids[ids > 0]
            alt = np.isin(lab, ids)
            inter = (alt & comp).sum()
            dices.append(float(2 * inter / max(alt.sum() + comp.sum(), 1)))
        persistence = None
        if tooth_axis is not None and np.linalg.norm(tooth_axis) > 0:
            pts = np.argwhere(comp) * np.asarray(spacing)
            a = np.asarray(tooth_axis, float); a /= np.linalg.norm(a)
            t = pts @ a
            n_planes = int(np.ptp(t) / max(float(min(spacing)), 1e-6)) + 1 if len(t) else 0
            persistence = {"long_axis_extent_mm": round(float(np.ptp(t)), 3) if len(t) else 0.0, "n_planes_along_axis": n_planes}
        rep = float(np.mean(dices)) if dices else 0.0
        by_path = dict(zip(keys, [round(d, 3) for d in dices]))
        return {"reproducibility": round(rep, 3), "mode": mode, "dice_by_path": by_path,
                "dice_by_sigma": {k.split(":")[1]: v for k, v in by_path.items() if k.startswith("smooth:")} if mode == "smooth" else None,
                "dice_min": round(float(min(dices)), 3) if dices else 0.0,
                "axis_persistence": persistence,
                "summary": {"candidate_id": cid, "reproducibility": round(rep, 3), "mode": mode, "axis_persistence": persistence}}
