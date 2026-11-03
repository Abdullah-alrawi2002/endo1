"""Training utilities shared by the CLI and the Colab notebook.

* `train_from_cases`      fit the texture-SVM lesion segmenter on (volume, tooth, lesion) triples
* `lopo_dice`             leave-one-case-out Dice / precision / recall (Ravikumar et al. protocol)
* `youden_threshold`      threshold maximising sensitivity + specificity - 1 on held-out voxels
* `save_model/load_model` joblib persistence with a JSON training card
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .segmentation import TextureSVMLesionSegmenter, ToothSegmentation, lesion_mask_from_probability


def train_from_cases(volumes: Sequence[np.ndarray], segs: Sequence[ToothSegmentation],
                     lesions: Sequence[np.ndarray], radius: int = 5, threshold: float = 0.5,
                     ) -> TextureSVMLesionSegmenter:
    tooth_masks = [s.tooth & ~s.canal for s in segs]
    return TextureSVMLesionSegmenter(radius=radius, threshold=threshold).fit(volumes, tooth_masks, lesions)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / max(a.sum() + b.sum(), 1))


def lopo_dice(volumes, segs, lesions, radius: int = 5, threshold: float = 0.5) -> List[Dict[str, float]]:
    """Leave-one-case-out cross-validation."""
    out = []
    n = len(volumes)
    for k in range(n):
        tr = [i for i in range(n) if i != k]
        model = train_from_cases([volumes[i] for i in tr], [segs[i] for i in tr], [lesions[i] for i in tr],
                                 radius=radius, threshold=threshold)
        prob = model(volumes[k], segs[k])
        pred = lesion_mask_from_probability(prob, segs[k], threshold=threshold)
        gt = lesions[k]
        tp = np.logical_and(pred, gt).sum()
        out.append({"case": k, "dice": dice(pred, gt),
                    "precision": float(tp / max(pred.sum(), 1)), "recall": float(tp / max(gt.sum(), 1))})
    return out


def youden_threshold(model: TextureSVMLesionSegmenter, volumes, segs, lesions,
                     grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17))) -> Tuple[float, List[Dict]]:
    """Pick the probability cut-off maximising Youden's J on held-out voxels
    (Al Fartousi et al. 2026 showed fixed 0.50 is rarely optimal on CBCT)."""
    probs, labels = [], []
    for v, s, l in zip(volumes, segs, lesions):
        p = model(v, s)
        m = s.tooth & ~s.canal
        probs.append(p[m]); labels.append(l[m])
    p = np.concatenate(probs); y = np.concatenate(labels).astype(bool)
    rows = []
    for t in grid:
        pred = p >= t
        sens = np.logical_and(pred, y).sum() / max(y.sum(), 1)
        spec = np.logical_and(~pred, ~y).sum() / max((~y).sum(), 1)
        rows.append({"threshold": float(t), "sensitivity": float(sens), "specificity": float(spec),
                     "youden_j": float(sens + spec - 1)})
    best = max(rows, key=lambda r: r["youden_j"])
    return best["threshold"], rows


def save_model(model: TextureSVMLesionSegmenter, path: str, card: Dict) -> None:
    import joblib, sklearn
    joblib.dump(model, path)
    card = dict(card)
    card.update({"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sklearn_version": sklearn.__version__,
                 "numpy_version": np.__version__, "radius": model.radius, "levels": model.levels,
                 "threshold": model.threshold, "morph_kernel": model.morph_kernel})
    with open(os.path.splitext(path)[0] + ".card.json", "w") as f:
        json.dump(card, f, indent=2)


def load_model(path: str) -> TextureSVMLesionSegmenter:
    import joblib
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Nested, patient-grouped evaluation (review revision 2.0.0)
# --------------------------------------------------------------------------- #
def nested_patient_cv(volumes, segs, lesions, patient_ids: Sequence[str], radius: int = 5,
                      min_patients: int = 4) -> Dict:
    """Outer leave-one-PATIENT-out; inner threshold selection on the remaining
    patients only (never on the outer test patient). Repeated scans from the
    same patient always stay on the same side of every split.

    Returns per-outer-fold Dice/precision/recall with the inner-selected
    threshold, plus the pooled summary. Requires >= `min_patients` distinct
    patients; this minimum is a software requirement, not a validation sample size.
    """
    pids = [str(p) for p in patient_ids]
    uniq = sorted(set(pids))
    if len(uniq) < min_patients:
        raise ValueError(f"Nested CV needs >= {min_patients} distinct patients, got {len(uniq)}")
    # duplicate-data guard: identical lesion masks under different patient ids
    sigs = {}
    for i, l in enumerate(lesions):
        sig = (int(l.sum()), tuple(np.argwhere(l).mean(0).round(2)) if l.any() else ())
        if sig in sigs and pids[sigs[sig]] != pids[i]:
            raise ValueError(f"Cases {sigs[sig]} and {i} look identical but carry different patient ids")
        sigs[sig] = i
    folds = []
    for test_pid in uniq:
        test_idx = [i for i, p in enumerate(pids) if p == test_pid]
        train_idx = [i for i, p in enumerate(pids) if p != test_pid]
        train_pids = sorted({pids[i] for i in train_idx})
        # inner: leave-one-patient-out inside the training patients to pick the threshold
        inner_probs, inner_labels = [], []
        for val_pid in train_pids:
            inner_tr = [i for i in train_idx if pids[i] != val_pid]
            inner_va = [i for i in train_idx if pids[i] == val_pid]
            if not inner_tr:
                continue
            m = train_from_cases([volumes[i] for i in inner_tr], [segs[i] for i in inner_tr], [lesions[i] for i in inner_tr], radius=radius)
            for i in inner_va:
                mask = segs[i].tooth & ~segs[i].canal
                inner_probs.append(m(volumes[i], segs[i])[mask]); inner_labels.append(lesions[i][mask])
        p = np.concatenate(inner_probs); y = np.concatenate(inner_labels).astype(bool)
        best_t, best_j = 0.5, -1
        for t in np.linspace(0.1, 0.9, 17):
            pred = p >= t
            j = np.logical_and(pred, y).sum() / max(y.sum(), 1) + np.logical_and(~pred, ~y).sum() / max((~y).sum(), 1) - 1
            if j > best_j:
                best_j, best_t = j, float(t)
        # outer: train on all training patients, test on the untouched patient
        model = train_from_cases([volumes[i] for i in train_idx], [segs[i] for i in train_idx], [lesions[i] for i in train_idx], radius=radius)
        for i in test_idx:
            prob = model(volumes[i], segs[i])
            pred = lesion_mask_from_probability(prob, segs[i], threshold=best_t)
            gt = lesions[i]; tp = np.logical_and(pred, gt).sum()
            folds.append({"test_patient": test_pid, "case": i, "inner_threshold": best_t, "inner_youden_j": float(best_j),
                          "dice": dice(pred, gt), "precision": float(tp / max(pred.sum(), 1)),
                          "recall": float(tp / max(gt.sum(), 1)), "gt_voxels": int(gt.sum())})
    d = np.array([f["dice"] for f in folds])
    return {"folds": folds, "n_patients": len(uniq), "n_cases": len(volumes),
            "dice_mean": float(d.mean()), "dice_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "threshold_median": float(np.median([f["inner_threshold"] for f in folds])),
            "note": "Thresholds selected inside training folds; outer patients untouched. Not a clinical validation."}
