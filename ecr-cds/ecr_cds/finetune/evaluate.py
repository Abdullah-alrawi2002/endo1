"""Score any backend (rules, hosted model, fine-tuned model) on the held-out conference prompts.

    python -m ecr_cds.finetune.evaluate --eval finetune_data/split/eval.jsonl --backend rules
    python -m ecr_cds.finetune.evaluate --eval ... --backend perplexity --limit 40
    LLM_API_KEY=... LLM_MODEL=ft:gpt-4o-mini:...  python -m ecr_cds.finetune.evaluate --eval ... --backend openai
    LLM_LOCAL_MODEL=ft/qwen7b-ecr python -m ecr_cds.finetune.evaluate --eval ... --backend local

Radiologist metrics (against the case truth, not against the smoothed target):
  json_valid, top1_agreement, brier_ecr, logloss_ecr, ecr_recall, ecr_false_positive_rate,
  internal_vs_ecr_confusions, overconfident_on_ambiguous (P>0.9 on open-wedge cases)
Skeptic metrics: confounder_recall (injected confounders mentioned), mean_challenges, json_valid.

Synthetic truth is phantom truth; a good score shows the model follows the Q1-Q8 reasoning and notices
confounders - it is not evidence of clinical accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from typing import Dict, List, Optional

import numpy as np

from ..agents.criteria import HYPOTHESES, evaluate_criteria, posterior_from_criteria
from ..agents.llm_backend import LLMBackend, NullBackend
from ..agents.visual_features import CandidateDescriptors

_CONFOUNDER_PATTERNS = {
    "metal": r"metal|streak|beam.?harden|hyperdense object",
    "noise_sigma": r"nois",
    "partial_volume": r"partial.?volume|resolution|blur",
    "intensity_drift": r"grey|gray|intensity|drift|threshold",
    "restoration_band": r"restorat|burnout|filling material",
    "oblique_deg": r"obliqu|tilt|axis",
}


class RulesBackend(LLMBackend):
    """Baseline: the deterministic criteria posterior, formatted like a model answer."""
    name = "rules"

    def complete_json(self, system, user, schema, images_png=None):
        payload = json.loads(user)
        d = payload.get("descriptors", {})
        fields = {k: v for k, v in d.items() if k in CandidateDescriptors.__dataclass_fields__}
        desc = CandidateDescriptors(**fields)
        if "diagnostic_questions" in payload:
            post = posterior_from_criteria(evaluate_criteria(desc))
            return {"differential": [{"diagnosis": h, "probability": p} for h, p in sorted(post.items(), key=lambda kv: -kv[1])],
                    "key_findings_for": [], "key_findings_against": [], "uncertainties": []}
        return {"challenges": []}


def make_backend(kind: str) -> LLMBackend:
    if kind == "rules":
        return RulesBackend()
    if kind == "perplexity":
        from ..agents.llm_backend import PerplexitySDKBackend
        return PerplexitySDKBackend()
    if kind == "openai":
        from ..agents.llm_backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(supports_images=False)
    if kind == "local":
        from ..agents.llm_backend import LocalHFBackend
        return LocalHFBackend()
    return NullBackend()


def _parse_differential(out) -> Optional[Dict[str, float]]:
    if not isinstance(out, dict) or not isinstance(out.get("differential"), list):
        return None
    post = {h: 0.0 for h in HYPOTHESES}
    for it in out["differential"]:
        if isinstance(it, dict) and it.get("diagnosis") in post:
            try:
                post[it["diagnosis"]] = max(0.0, float(it["probability"]))
            except (TypeError, ValueError):
                pass
    s = sum(post.values())
    if s <= 0:
        return None
    return {h: v / s for h, v in post.items()}


def _strip_schema(system: str) -> str:
    return system.split("\nRespond with JSON only")[0]


def evaluate(rows: List[Dict], backend: LLMBackend, limit: Optional[int] = None, verbose: bool = False) -> Dict:
    rad, sk = [], []
    t0 = time.time()
    for r in rows[: limit or len(rows)]:
        system, user = r["messages"][0]["content"], r["messages"][1]["content"]
        target = json.loads(r["messages"][2]["content"])
        schema = {"type": "object"}
        try:
            out = backend.complete_json(_strip_schema(system), user, schema)
        except Exception as e:
            out = {"_error": str(e)}
        meta = r.get("meta", {})
        if r.get("task") == "radiologist":
            truth = meta.get("label")  # None = incidental / intact -> artefact-type target
            truth_h = truth or "imaging_artifact"
            post = _parse_differential(out)
            rec = {"valid": post is not None, "truth": truth_h, "ambiguous": bool(meta.get("ambiguous"))}
            if post:
                lead = max(post, key=post.get)
                p_ecr = min(max(post["ECR"], 1e-6), 1 - 1e-6)
                y = 1.0 if truth == "ECR" else 0.0
                rec.update(lead=lead, top1=lead == truth_h, p_ecr=p_ecr, brier=(p_ecr - y) ** 2,
                           logloss=-(y * math.log(p_ecr) + (1 - y) * math.log(1 - p_ecr)),
                           internal_confusion=(truth == "internal_resorption" and lead == "ECR") or (truth == "ECR" and lead == "internal_resorption"),
                           overconf_ambiguous=bool(meta.get("ambiguous")) and max(post.values()) > 0.9)
            rad.append(rec)
            if verbose:
                print(meta.get("case_id"), truth_h, "->", rec.get("lead"), f"P(ECR)={rec.get('p_ecr', float('nan')):.2f}")
        else:
            conf = meta.get("confounders", {}) or {}
            injected = [k for k, v in conf.items() if (v if isinstance(v, bool) else abs(float(v)) >= (80 if k == "noise_sigma" else 150 if k == "intensity_drift" else 6 if k == "oblique_deg" else 1e-9)) and k != "metal_distance_mm"]
            issues = []
            if isinstance(out, dict) and isinstance(out.get("challenges"), list):
                issues = [str(c.get("issue", "")) for c in out["challenges"] if isinstance(c, dict)]
            text = " ".join(issues).lower()
            hits = [k for k in injected if re.search(_CONFOUNDER_PATTERNS[k], text)]
            sk.append({"valid": bool(issues) or (isinstance(out, dict) and "challenges" in out), "n": len(issues),
                       "n_injected": len(injected), "n_hit": len(hits)})
    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return round(float(np.mean(xs)), 3) if xs else None
    valid = [r for r in rad if r["valid"]]
    ecr_rows = [r for r in valid if r["truth"] == "ECR"]
    non_ecr = [r for r in valid if r["truth"] != "ECR"]
    amb = [r for r in valid if r["ambiguous"]]
    summary = {
        "backend": backend.name, "n_radiologist": len(rad), "n_skeptic": len(sk), "seconds": round(time.time() - t0, 1),
        "radiologist": {
            "json_valid": mean([r["valid"] for r in rad]),
            "top1_agreement": mean([r["top1"] for r in valid]),
            "brier_ecr": mean([r["brier"] for r in valid]),
            "logloss_ecr": mean([r["logloss"] for r in valid]),
            "ecr_recall": mean([r["lead"] == "ECR" for r in ecr_rows]),
            "ecr_false_positive_rate": mean([r["lead"] == "ECR" for r in non_ecr]),
            "mean_p_ecr_when_ecr": mean([r["p_ecr"] for r in ecr_rows]),
            "mean_p_ecr_when_not_ecr": mean([r["p_ecr"] for r in non_ecr]),
            "internal_vs_ecr_confusions": int(sum(r["internal_confusion"] for r in valid)),
            "overconfident_on_ambiguous": mean([r["overconf_ambiguous"] for r in amb]),
        },
        "skeptic": {
            "json_valid": mean([r["valid"] for r in sk]),
            "mean_challenges": mean([r["n"] for r in sk]),
            "confounder_recall": (round(sum(r["n_hit"] for r in sk) / max(1, sum(r["n_injected"] for r in sk)), 3) if sk else None),
        },
        "note": "Synthetic phantom truth; measures reasoning/format fidelity and confounder awareness, not clinical accuracy.",
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--backend", default="rules", choices=["rules", "perplexity", "openai", "local"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default="radiologist,skeptic")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",")
    rows = [json.loads(l) for l in open(a.eval) if l.strip()]
    rows = [r for r in rows if r.get("task") in tasks]
    s = evaluate(rows, make_backend(a.backend), a.limit, a.verbose)
    print(json.dumps(s, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
