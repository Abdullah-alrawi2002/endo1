"""Experimental cervical-resorption candidate screening, NOT an ECR classifier.

All numerical cutoffs below are engineering defaults, not clinical thresholds.
Keeps separate external components, looks for a cervical surface connection,
measures physical depth and threshold stability, and abstains on ambiguous
landmarks/restorations. No probability of ECR, Patel code or treatment is issued.
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .auto_segment import tooth_hull_convex
from .patel_classifier import _estimate_cej_from_enamel


@dataclass(frozen=True)
class ECRDetectionConfig:
    cervical_half_width_mm: float = 2.0
    min_volume_mm3: float = 0.25
    min_depth_mm: float = 0.20
    surface_distance_mm: float = 0.25
    min_axial_extent_mm: float = 0.30
    opening_mm: float = 0.10
    stable_core_fraction: float = 0.50
    threshold_factors: Tuple[float, float, float] = (0.9, 1.0, 1.1)

    def __post_init__(self):
        numeric = (self.cervical_half_width_mm, self.min_volume_mm3,
                   self.min_depth_mm, self.surface_distance_mm,
                   self.min_axial_extent_mm, self.opening_mm)
        if any(not np.isfinite(x) or x <= 0 for x in numeric):
            raise ValueError("Physical thresholds must be positive and finite")
        if not 0 < self.stable_core_fraction <= 1:
            raise ValueError("stable_core_fraction must be in (0, 1]")
        if (len(self.threshold_factors) != 3
                or not all(np.isfinite(x) and x > 0 for x in self.threshold_factors)
                or self.threshold_factors != tuple(sorted(self.threshold_factors))
                or self.threshold_factors[1] != 1.):
            raise ValueError("Use three sorted positive factors with 1.0 in the middle")


def _ball_mm(radius, spacing):
    r = np.ceil(radius / np.asarray(spacing)).astype(int)
    zz, yy, xx = np.mgrid[-r[0]:r[0]+1, -r[1]:r[1]+1, -r[2]:r[2]+1]
    return sum((g * s)**2 for g, s in zip((zz, yy, xx), spacing)) <= radius**2


class CervicalResorptionDetector:
    """Callable returning a binary candidate map plus a structured review report.

    CEJ/apex overrides are crop-local z,y,x voxel points. Both must be supplied
    together. They define a reviewed axial CEJ plane, not a full anatomical CEJ
    surface. Automatic landmarks are expressly unreviewed and can be wrong.
    """
    def __init__(self, threshold, dense_threshold, spacing, config=None,
                 cej_zyx: Optional[Tuple[float, float, float]] = None,
                 apex_zyx: Optional[Tuple[float, float, float]] = None,
                 canal_source="automatic"):
        self.threshold = float(threshold)
        self.dense_threshold = float(dense_threshold)
        self.spacing = np.asarray(spacing, dtype=float)
        if self.spacing.shape != (3,) or not np.all(np.isfinite(self.spacing)) or np.any(self.spacing <= 0):
            raise ValueError("spacing must contain three positive finite values")
        if not np.isfinite(self.threshold) or not np.isfinite(self.dense_threshold):
            raise ValueError("Invalid intensity threshold")
        if self.dense_threshold <= self.threshold:
            raise ValueError("dense_threshold must exceed threshold")
        if (cej_zyx is None) != (apex_zyx is None):
            raise ValueError("Supply both CEJ and apex points, or neither")
        self.cej_zyx, self.apex_zyx = cej_zyx, apex_zyx
        self.canal_source = str(canal_source or "automatic")
        self.config = config or ECRDetectionConfig()
        self.last_report = {}
        self.last_mask = None
        self.last_labels = None
        self.last_geometry = None
        self.last_all_labels = None

    def _landmarks(self, raw, seg):
        pts = np.argwhere(seg.tooth) * self.spacing
        centre = pts.mean(0)
        _, _, vh = np.linalg.svd((pts-centre)[::max(1, len(pts)//50000)], full_matrices=False)
        axis = vh[0]
        t = (pts-centre) @ axis
        notes = []
        if self.cej_zyx is not None:
            cej = np.asarray(self.cej_zyx, float)
            apex = np.asarray(self.apex_zyx, float)
            for p in (cej, apex):
                if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p >= raw.shape):
                    raise ValueError("Reviewed points must lie inside the crop")
                if not seg.tooth[tuple(np.clip(np.rint(p).astype(int), 0, np.asarray(raw.shape)-1))]:
                    raise ValueError("Reviewed points must lie inside the tooth mask")
            v = (apex-cej) * self.spacing
            if np.linalg.norm(v) < 2:
                raise ValueError("CEJ/apex separation must be at least 2 mm")
            axis = v / np.linalg.norm(v)
            centre = cej * self.spacing
            return centre, axis, 0., {"source": "reviewed_points",
                "cej_crop_zyx": cej.tolist(), "apex_crop_zyx": apex.tolist(), "flags": []}
        shell = ndi.binary_dilation(seg.tooth, structure=_ball_mm(1., self.spacing)) & ~seg.tooth
        bpts = np.argwhere(shell & seg.bone) * self.spacing
        low, high = np.percentile(t, [30, 70])
        if len(bpts) >= 100:
            bt = (bpts-centre) @ axis
            counts = [int(np.sum(bt <= low)), int(np.sum(bt >= high))]
        else:
            counts = [0, 0]
        if max(counts) >= 2 * max(1, min(counts)) and max(counts) >= 100:
            if counts[0] > counts[1]:
                axis = -axis
        else:
            radial = np.linalg.norm((pts-centre)-t[:, None]*axis, axis=1)
            if radial[t <= np.percentile(t, 10)].mean() < radial[t >= np.percentile(t, 90)].mean():
                axis = -axis
            notes.append("axis_orientation_uncertain")
        exclusion = ndi.binary_dilation(seg.canal, structure=_ball_mm(.4, self.spacing))
        remaining = seg.tooth & ~exclusion
        if remaining.sum() < 100:
            return centre, axis, None, {"source": "unavailable",
                                        "flags": notes + ["insufficient_tooth_for_cej"]}
        cej = _estimate_cej_from_enamel(raw, seg.tooth, self.spacing, centre, axis, exclude=exclusion)
        if len(bpts):
            crest = float(np.percentile((bpts-centre) @ axis, 2))
            if cej > crest+3:
                # Do not silently make an anatomical assumption into a measured CEJ.
                cej = crest-1.5
                notes.append("cej_uses_unverified_bone_crest_offset")
        notes.append("automatic_cej_not_reviewed")
        return centre, axis, cej, {"source": "automatic_unreviewed",
            "estimated_cej_axis_mm": float(cej), "axis_zyx_physical": axis.tolist(),
            "centre_crop_mm": centre.tolist(), "flags": notes}

    def __call__(self, norm, seg):
        cfg = self.config
        if seg.intensity is None:
            raise ValueError("Raw intensities required")
        raw = seg.intensity
        if any(m.shape != raw.shape for m in (seg.tooth, seg.canal, seg.bone)):
            raise ValueError("Masks must match raw crop")
        if seg.tooth.sum() < 100 or not np.all(np.isfinite(raw)):
            raise ValueError("Insufficient tooth mask or nonfinite intensity")
        original = seg.tooth.copy()
        # Morphology assumes near-isotropic input in the legacy hull implementation.
        if self.spacing.max()/self.spacing.min() > 1.1:
            raise ValueError("Cervical screening requires isotropic resampling first")
        centre, axis, cej, landmarks = self._landmarks(raw, seg)
        self.last_geometry = {"centre_mm": centre, "axis": axis, "cej_t_mm": cej}
        # Avoid the old downsample/upsample closing: it creates a one-voxel
        # low-density halo even around an intact digital cylinder.
        hull = tooth_hull_convex(original)
        sm = ndi.gaussian_filter(raw.astype(np.float32), sigma=.08/self.spacing)
        depth = ndi.distance_transform_edt(hull, sampling=self.spacing)
        surface = hull & (depth <= cfg.surface_distance_mm)
        canal_dist = ndi.distance_transform_edt(~seg.canal, sampling=self.spacing) if seg.canal.any() else None
        # No blanket removal of the central 45%: external lesions may approach the canal.
        admissible = hull & ~seg.canal
        opening = _ball_mm(cfg.opening_mm, self.spacing)
        proposals = []
        # Perturb relative to local background, not an absolute HU assumption.
        soft = float(np.percentile(sm[~hull], 50)) if (~hull).any() else float(sm.min())
        soft = min(soft, self.threshold-1.)
        thresholds = [soft+f*(self.threshold-soft) for f in cfg.threshold_factors]
        for thr in thresholds:
            mask = ndi.binary_opening(admissible & (sm < thr), structure=opening)
            proposals.append(mask)
        lab, n = ndi.label(proposals[1])  # 6-connected in 3D
        self.last_all_labels = lab.astype(np.uint16)
        upper_lab, _ = ndi.label(proposals[2])
        upper_sizes = np.bincount(upper_lab.ravel())
        self.last_labels = np.zeros(raw.shape, np.uint16)
        review_mask = np.zeros(raw.shape, bool)
        rows = []
        voxel_vol = float(np.prod(self.spacing))
        dense = hull & (sm >= self.dense_threshold)
        dense_dist = ndi.distance_transform_edt(~dense, sampling=self.spacing) if dense.any() else None
        for i, box in enumerate(ndi.find_objects(lab), 1):
            if box is None:
                continue
            local = lab[box] == i
            count = int(local.sum())
            if count*voxel_vol < cfg.min_volume_mm3:
                continue
            q = np.argwhere(local) + np.array([b.start for b in box])
            ix = tuple(q.T)
            t = (q*self.spacing-centre) @ axis
            on_surface = surface[ix]
            external = bool(on_surface.any())
            near_cej = (np.abs(t-cej) <= cfg.cervical_half_width_mm) if cej is not None else np.zeros(len(t), bool)
            cervical = bool((on_surface & near_cej).any()) if cej is not None else None
            core_fraction = float(proposals[0][ix].mean())
            upper_ids = np.unique(upper_lab[ix])
            upper_ids = upper_ids[upper_ids != 0]
            expansion = float(upper_sizes[upper_ids].sum()/count)
            maxdepth = float(depth[ix].max())
            extent = float(np.ptp(t)+min(self.spacing))
            dense_near = float((dense_dist[ix] <= .4).mean()) if dense_dist is not None else 0.
            flags = list(landmarks["flags"])
            reasons = []
            if not external:
                reasons.append("no_external_surface_connection")
            if cervical is False:
                reasons.append("no_connection_in_estimated_cervical_band")
            if maxdepth < cfg.min_depth_mm:
                reasons.append("shallow_surface_signal")
            if extent < cfg.min_axial_extent_mm:
                reasons.append("insufficient_3d_extent")
            if core_fraction < cfg.stable_core_fraction:
                flags.append("threshold_sensitive")
            if expansion > 2.0:
                flags.append("threshold_expansion_or_component_merging")
            if abs(axis[0]) < np.cos(np.deg2rad(20)):
                flags.append("oblique_tooth_axis_hull_uncertainty")
            if dense_near > .1:
                flags.append("adjacent_dense_material_enamel_or_restoration")
            if not seg.canal.any():
                flags.append("canal_missing")
            if "expert" not in self.canal_source:
                flags.append("canal_is_automatic_proxy")
            if np.any(q.min(0) == 0) or np.any(q.max(0) == np.asarray(raw.shape)-1):
                flags.append("candidate_touches_crop_boundary")
            decision = "rejected_by_screen" if reasons else "possible_ecr_review_required"
            if not reasons and (cej is None or flags):
                decision = "indeterminate_review_required"
            row = {"component_id": i, "disposition": decision,
                "volume_mm3": round(count*voxel_vol, 4),
                "centroid_crop_zyx": q.mean(0).round(2).tolist(),
                "bbox_crop_zyx": [[int(b.start), int(b.stop)] for b in box],
                "external_surface_connection": external,
                "cervical_surface_connection": cervical,
                "extent_along_axis_mm": round(extent, 3),
                "depth_in_reconstructed_hull_mm": round(maxdepth, 3),
                "stable_core_fraction": round(core_fraction, 3),
                "upper_threshold_expansion_ratio": round(expansion, 3),
                "min_distance_to_canal_proxy_mm": round(float(canal_dist[ix].min()), 3) if canal_dist is not None else None,
                "flags": flags, "rejection_reasons": reasons,
                "differential_not_resolved": ["ECR", "caries", "restoration_margin",
                    "noncarious_cervical_defect", "root_concavity", "imaging_artifact"]}
            rows.append(row)
            if not reasons:
                review_mask[ix] = True
                self.last_labels[ix] = i
        selected = [r for r in rows if r["disposition"] != "rejected_by_screen"]
        self.last_mask = review_mask
        self.last_report = {
            "status": "candidates_for_review" if selected else "no_candidate_not_exclusion",
            "evidence_level": "experimental_ecr_candidate_screen",
            "ecr_diagnosis": None, "clinical_validation": False,
            "patel": None, "decision": None, "requires_confirmation": True,
            "landmarks": landmarks, "candidates": rows,
            "n_review_candidates": len(selected), "config": asdict(cfg),
            "intensity_thresholds": thresholds, "thresholds_are_clinically_validated": False,
            "note": "Experimental candidate screening only. No ECR diagnosis, calibrated probability, "
                    "Patel code or treatment recommendation. A negative screen does not exclude ECR. "
                    "Automatic anatomy, large defects, repair tissue and restorations can defeat this screen."}
        assert np.array_equal(seg.tooth, original), "Detector must not rewrite the tooth mask"
        return review_mask.astype(np.float32)
