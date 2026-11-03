"""Evaluate perception and report quality separately, against a frozen baseline.

Perception metrics (does the system *see* the lesion?): best-candidate Dice with the true lesion, centroid hit
(within 1 mm), number of false candidates, verifier reproducibility on true vs false candidates.
Report metrics (does it *reason* correctly given what it saw?): top-1 diagnosis vs label, P(ECR) Brier score,
ECR recall / false-positive rate, plan primary when ECR.

`run_ablation` runs named variants of CaseConference.run kwargs on the same synthetic cases and reports paired
differences from the frozen baseline.  Synthetic phantoms with injected confounders are software tests, not
clinical validation; a variant that helps here is a hypothesis to test on real, labelled scans.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..agents import CaseConference
from ..dicom_loader import CBCTVolume
from ..finetune.dataset import MORPH_TO_LABEL, TEST_SEED, augment, sample_case
from ..phantom import SHAPE, VOX, make_phantom
from ..segmentation import ToothSegmentation

FROZEN_BASELINE: Dict[str, Any] = {"verify": False, "context": None, "max_rounds": 1}   # rev 2.6 behaviour


def make_case(spec: Dict[str, Any], rng: random.Random, case_id: str):
    if spec["morphology"] == "intact":
        vol, seg, lesion, truth = make_phantom(0.0, 10, False, morphology="wedge")
        lesion = np.zeros(SHAPE, bool)
        vol = vol.copy(); vol[seg.tooth & (vol < 1000)] = 1600
    else:
        vol, seg, lesion, truth = make_phantom(spec["height_mm"], spec["spread_deg"], spec["perforate"],
                                               portal_angle_deg=spec["portal_angle_deg"], morphology=spec["morphology"])
    v, conf = augment(vol, seg, lesion, rng, level=spec["augment"])
    cb = CBCTVolume(v.astype(np.float32), (VOX, VOX, VOX), (0, 0, 0), np.eye(3), {"SeriesInstanceUID_hash": case_id})
    return cb, seg, lesion, conf, MORPH_TO_LABEL[spec["morphology"]]


def perception_metrics(bb, lesion: np.ndarray, spacing=(VOX, VOX, VOX), consensus: Optional[Dict] = None) -> Dict[str, Any]:
    labels, crop = bb.arrays.get("candidate_labels"), bb.arrays.get("crop")
    if labels is None:
        return {"n_candidates": 0, "best_dice": 0.0, "hit": False, "n_false": 0}
    les = lesion[tuple(crop)]
    has_lesion = bool(les.any())
    out = {"n_candidates": int(len(np.unique(labels[labels > 0]))), "best_dice": 0.0, "hit": False, "n_false": 0,
           "has_lesion": has_lesion, "reproducibility_true": [], "reproducibility_false": [], "candidates": []}
    ver = {e.refers_to: e.content["reproducibility"] for e in bb.query(kind="verification")}
    p_by_cid = {r["component_id"]: r.get("p_ecr", 0.0) for r in (consensus or {}).get("candidates", [])}
    les_d = ndi.binary_dilation(les, iterations=int(round(1.0 / spacing[0]))) if has_lesion else les
    for cid in np.unique(labels[labels > 0]):
        m = labels == cid
        dice = float(2 * (m & les).sum() / max(m.sum() + les.sum(), 1)) if has_lesion else 0.0
        cen = tuple(np.argwhere(m).mean(0).astype(int))
        hit = bool(has_lesion and les_d[cen])
        is_true = (m & les_d).sum() / max(m.sum(), 1) > 0.5 if has_lesion else False
        out["best_dice"] = max(out["best_dice"], dice)
        out["hit"] = out["hit"] or hit
        if not is_true:
            out["n_false"] += 1
        if cid in ver:
            (out["reproducibility_true"] if is_true else out["reproducibility_false"]).append(ver[int(cid)])
        out["candidates"].append({"cid": int(cid), "is_true": bool(is_true), "dice": round(dice, 3),
                                  "reproducibility": ver.get(int(cid)), "p_ecr": p_by_cid.get(int(cid))})
    # decision-aware review proxy: with a review budget of ONE candidate per case, which ranking key puts a true
    # candidate first?  (protocol of the 2026 decision-aware uncertainty evaluation; here at candidate level)
    if has_lesion and out["candidates"]:
        for key in ("p_ecr", "reproducibility"):
            ranked = [c for c in out["candidates"] if c.get(key) is not None]
            out[f"budget1_true_first_by_{key}"] = bool(ranked and max(ranked, key=lambda c: c[key])["is_true"]) if ranked else None
    return out


def auroc(pos: List[float], neg: List[float]) -> Optional[float]:
    """Rank-based AUROC (probability a true candidate scores above a false one); None if a class is empty."""
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def report_metrics(consensus: Dict, label: Optional[str]) -> Dict[str, Any]:
    dx = consensus.get("suggested_diagnosis")
    p_ecr = dx["p_ecr_uncalibrated"] if dx else 0.0
    top1 = dx["leading"] if dx else None
    y = 1.0 if label == "ECR" else 0.0
    return {"label": label, "top1": top1, "top1_correct": (top1 == label) if label else None,
            "p_ecr": p_ecr, "brier": (p_ecr - y) ** 2, "status": consensus["status"],
            "ecr_suggested": consensus["status"] == "suggested_ecr_with_plan",
            "plan_primary": (consensus.get("suggested_plan") or {}).get("rule_recommendation", {}).get("primary")}


_BACKEND_CACHE: Dict[str, Any] = {}


def _resolve_backend(spec):
    """variant["backend"]: None | "nemotron" | "auto" | an LLMBackend instance.  Unreachable servers raise."""
    if spec is None or spec == "none":
        return None
    if not isinstance(spec, str):
        return spec
    if spec not in _BACKEND_CACHE:
        from ..agents.llm_backend import auto_backend
        _BACKEND_CACHE[spec] = auto_backend(spec)
    return _BACKEND_CACHE[spec]


def run_case(cb, seg, lesion, label, variant: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    kw = dict(variant)
    max_rounds = kw.pop("max_rounds", 1)
    backend = _resolve_backend(kw.pop("backend", None))
    cc = CaseConference(max_rounds=max_rounds, backend=backend)
    segmenter = lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone)  # noqa: E731
    t0 = time.time()
    cons = cc.run(cb, segmenter, TEST_SEED, "phantom_truth", render_tiles=False, case_id=case_id, **kw)
    llm = [r.get("llm_meta") for r in cons.get("candidates", []) if r.get("llm_meta")]
    return {"perception": perception_metrics(cc.last_blackboard, lesion, consensus=cons), "report": report_metrics(cons, label),
            "llm": {"n_opinions": len(llm), "n_abstained": sum(1 for m in llm if m.get("abstained")),
                    "n_weak_grounding": sum(1 for m in llm if (m.get("grounding") or {}).get("verdict") == "weak"),
                    "mean_weight": round(float(np.mean([m.get("weight", 0) for m in llm])), 3) if llm else None,
                    "backend_failures": cc.last_blackboard.arrays.get("backend_failures", 0)},
            "seconds": round(time.time() - t0, 1)}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 3) if xs else None
    rep = [r["report"] for r in rows]; per = [r["perception"] for r in rows]
    ecr = [r for r in rep if r["label"] == "ECR"]; non = [r for r in rep if r["label"] != "ECR"]
    return {"n": len(rows),
            "perception": {"best_dice_when_lesion": mean([p["best_dice"] for p in per if p.get("has_lesion")]),
                           "hit_rate_when_lesion": mean([p["hit"] for p in per if p.get("has_lesion")]),
                           "false_candidates_per_case": mean([p["n_false"] for p in per]),
                           "reproducibility_true": mean(sum([p["reproducibility_true"] for p in per], [])),
                           "reproducibility_false": mean(sum([p["reproducibility_false"] for p in per], [])),
                           "reproducibility_auroc_true_vs_false": auroc(sum([p["reproducibility_true"] for p in per], []),
                                                                        sum([p["reproducibility_false"] for p in per], [])),
                           "budget1_true_first_by_p_ecr": mean([p.get("budget1_true_first_by_p_ecr") for p in per if p.get("has_lesion")]),
                           "budget1_true_first_by_reproducibility": mean([p.get("budget1_true_first_by_reproducibility") for p in per if p.get("has_lesion")])},
            "report": {"top1_accuracy": mean([r["top1_correct"] for r in rep]),
                       "ecr_recall": mean([r["ecr_suggested"] for r in ecr]),
                       "ecr_fpr": mean([r["ecr_suggested"] for r in non]),
                       "brier": mean([r["brier"] for r in rep])},
            "llm": {"opinions": int(sum(r.get("llm", {}).get("n_opinions", 0) for r in rows)),
                    "abstained": int(sum(r.get("llm", {}).get("n_abstained", 0) for r in rows)),
                    "weak_grounding": int(sum(r.get("llm", {}).get("n_weak_grounding", 0) for r in rows)),
                    "mean_weight": mean([r.get("llm", {}).get("mean_weight") for r in rows]),
                    "backend_failures": int(sum(r.get("llm", {}).get("backend_failures", 0) for r in rows))},
            "seconds_per_case": mean([r["seconds"] for r in rows])}


DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "verify_on": {"verify": True, "context": None, "max_rounds": 1},
    "verify_2rounds": {"verify": True, "context": None, "max_rounds": 2},
    "tta_verify": {"verify": True, "verify_mode": "tta", "context": None, "max_rounds": 1},
}

NEMOTRON_VARIANTS: Dict[str, Dict[str, Any]] = {
    "nemotron_frame": {"verify": True, "context": None, "max_rounds": 1, "backend": "nemotron"},
    "nemotron_frame_cf": {"verify": True, "context": None, "max_rounds": 1, "backend": "nemotron", "counterfactual_check": True},
    "nemotron_sharpen_cf": {"verify": True, "context": None, "max_rounds": 1, "backend": "nemotron", "counterfactual_check": True,
                            "sharpen_display": True},
}


def run_ablation(variants: Dict[str, Dict[str, Any]], n_cases: int = 8, seed: int = 0, out_path: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """Run every variant (plus the frozen baseline) on the same synthetic cases; report paired differences."""
    rng = random.Random(seed)
    specs = [sample_case(rng) for _ in range(n_cases)]
    all_variants = {"frozen_baseline": FROZEN_BASELINE, **variants}
    results: Dict[str, List[Dict]] = {k: [] for k in all_variants}
    for i, spec in enumerate(specs):
        cb, seg, lesion, conf, label = make_case(spec, random.Random(seed * 1000 + i), f"abl_{i}")
        for name, var in all_variants.items():
            r = run_case(cb, seg, lesion, label, var, f"abl_{i}")
            r["spec"] = spec; r["confounders"] = conf.__dict__
            results[name].append(r)
            if verbose:
                print(f"case {i} {spec['morphology']:8s} {name:18s} top1={r['report']['top1']} p_ecr={r['report']['p_ecr']:.2f} "
                      f"dice={r['perception']['best_dice']:.2f} false={r['perception']['n_false']} {r['seconds']}s", flush=True)
    summary = {name: summarise(rows) for name, rows in results.items()}
    base = summary["frozen_baseline"]
    deltas = {}
    for name in variants:
        d = {}
        for sec in ("perception", "report"):
            for k, v in summary[name][sec].items():
                b = base[sec].get(k)
                d[f"{sec}.{k}"] = None if (v is None or b is None) else round(v - b, 3)
        deltas[name] = d
    out = {"n_cases": n_cases, "seed": seed, "variants": all_variants, "summary": summary, "delta_vs_frozen_baseline": deltas,
           "note": "synthetic phantoms with injected confounders; differences are hypotheses to test on labelled clinical scans",
           "per_case": results}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="perception-vs-report ablation against the frozen baseline")
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_results.json")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="comma-separated subset of the named variants")
    ap.add_argument("--llm", default="none", help="'nemotron' adds the local-VLM variants (requires a reachable NEMOTRON_BASE_URL)")
    a = ap.parse_args(argv)
    pool = {**DEFAULT_VARIANTS, **(NEMOTRON_VARIANTS if a.llm == "nemotron" else {})}
    names = [n for n in a.variants.split(",") if n] + (list(NEMOTRON_VARIANTS) if a.llm == "nemotron" else [])
    variants = {n: pool[n] for n in names if n in pool}
    if a.llm == "nemotron":
        from ..agents.llm_backend import NemotronLocalBackend
        NemotronLocalBackend()   # fail early with a clear message when the server is not up
    res = run_ablation(variants, n_cases=a.n_cases, seed=a.seed, out_path=a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
