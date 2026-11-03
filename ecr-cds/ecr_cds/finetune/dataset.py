"""Build supervised fine-tuning data for the conference's language-model roles.

Two tasks are recorded exactly as the agents prompt the model at run time:

* ``radiologist`` - descriptors + answers to Q1-Q7 + fired criteria  ->  differential (JSON)
* ``skeptic``     - descriptors + radiologist evidence                ->  concrete challenges (JSON)

Sources of examples
-------------------
1. **Synthetic cases** from ``ecr_cds.phantom`` with *known* morphology (ECR, open wedge, NCCL,
   internal resorption, intact) and *known injected confounders* (noise, metal streaks, partial
   volume blur, intensity drift, restoration band, oblique axis).  The target differential is the
   phantom truth (label-smoothed), and the skeptic target lists the confounders that were really
   injected.  This teaches the model the Q1-Q8 reasoning, strict JSON output and confounder
   awareness - it does **not** teach clinical truth, because no real pathology is involved.
2. **Clinician-confirmed cases**: a saved ``conference_report.json`` plus the confirmed diagnosis
   (``from_confirmed_case``).  These are the only examples that carry clinical ground truth; keep
   them in a separate split and weight them up.

Everything is written as OpenAI chat-format JSONL (``{"messages": [...], "meta": {...}}``), which
both the LoRA trainer and the hosted fine-tuning exporter consume.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..dicom_loader import CBCTVolume
from ..phantom import SHAPE, VOX, make_phantom
from ..segmentation import ToothSegmentation
from ..agents.criteria import HYPOTHESES
from ..agents.llm_backend import LLMBackend
from ..agents.orchestrator import CaseConference

MORPH_TO_LABEL = {"ecr": "ECR", "wedge": "ECR", "nccl": "noncarious_cervical_lesion",
                  "internal": "internal_resorption", "intact": None}
TEST_SEED = (125, 80, 105)


# --------------------------------------------------------------------------- #
class RecordingBackend(LLMBackend):
    """Captures every prompt the agents would send and answers nothing (rule-only run)."""
    name = "recording"

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def complete_json(self, system, user, schema, images_png=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "n_images": len(images_png or [])})
        return None


# --------------------------------------------------------------------------- #
@dataclass
class Confounders:
    noise_sigma: float = 0.0
    metal: bool = False
    metal_distance_mm: float = 0.0
    partial_volume: bool = False
    intensity_drift: float = 0.0
    restoration_band: bool = False
    oblique_deg: float = 0.0

    def challenges(self) -> List[Dict[str, str]]:
        out = []
        if self.metal:
            out.append({"issue": f"Hyperdense (metal-level) object about {self.metal_distance_mm:.0f} mm from the candidate; "
                                 "streak and dark-band artefacts can create or distort hypodense defects.",
                        "severity": "major" if self.metal_distance_mm < 4 else "moderate"})
        if self.noise_sigma >= 80:
            out.append({"issue": "High image noise; thresholded boundaries and margin roughness are unreliable.", "severity": "moderate"})
        if self.partial_volume:
            out.append({"issue": "Reduced effective resolution (partial volume): sub-millimetre margins and portal width are blurred.",
                        "severity": "moderate"})
        if abs(self.intensity_drift) >= 150:
            out.append({"issue": "Grey-level drift relative to the expected dentine value; fixed thresholds may mis-segment the tooth.",
                        "severity": "minor"})
        if self.restoration_band:
            out.append({"issue": "Dense restorative material at the cervical margin: restoration margin, cervical burnout or "
                                 "partial-volume lucency can mimic a defect.", "severity": "moderate"})
        if self.oblique_deg >= 6:
            out.append({"issue": "Tooth axis oblique to the scan axis; convex-hull surface and level-wise measurements are less reliable.",
                        "severity": "moderate"})
        return out


def _add_metal(vol: np.ndarray, rng: random.Random, seed_zyx, distance_mm: float) -> None:
    """Spherical metal-level blob plus bright/dark streaks in the axial plane (crude beam-hardening look)."""
    z0, y0, x0 = seed_zyx
    ang = rng.uniform(0, 2 * np.pi)
    d = int(distance_mm / VOX)
    cy, cx = int(np.clip(y0 + d * np.sin(ang), 8, SHAPE[1] - 9)), int(np.clip(x0 + d * np.cos(ang), 8, SHAPE[2] - 9))
    cz = int(np.clip(z0 + rng.randint(-40, 40), 8, SHAPE[0] - 9))
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    ball = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= 7 ** 2
    vol[ball] = 6000   # metal-level (well above enamel); rev 2.7: was 3500, which sat below the metal reference level
    # streaks: alternating bright/dark rays through the blob in slices near it
    for k in range(6):
        a = rng.uniform(0, np.pi)
        for s in np.linspace(-80, 80, 161):
            y = int(round(cy + s * np.sin(a))); x = int(round(cx + s * np.cos(a)))
            if 0 <= y < SHAPE[1] and 0 <= x < SHAPE[2] and abs(s) > 7:
                sign = 1 if k % 2 == 0 else -1
                vol[cz - 10:cz + 11, y, x] += sign * 250 * np.exp(-abs(s) / 60)


def _add_restoration_band(vol: np.ndarray, seg: ToothSegmentation, lesion: np.ndarray) -> None:
    """Dense band (restoration) on the crown surface just coronal to the CEJ."""
    z_cej = int(12.0 / VOX)
    shell = seg.tooth & ~ndi.binary_erosion(seg.tooth, iterations=6)
    band = np.zeros(SHAPE, bool)
    band[z_cej - 20:z_cej - 2] = True
    sel = shell & band & ~lesion
    vol[sel] = 3000


def augment(vol: np.ndarray, seg: ToothSegmentation, lesion: np.ndarray, rng: random.Random,
            seed_zyx=TEST_SEED, level: str = "random") -> Tuple[np.ndarray, Confounders]:
    v = vol.astype(np.float32).copy()
    c = Confounders()
    if level == "clean":
        return v, c
    if rng.random() < 0.6:
        c.noise_sigma = rng.choice([30.0, 60.0, 100.0, 140.0])
    if rng.random() < 0.35:
        c.metal = True
        c.metal_distance_mm = rng.choice([2.5, 4.0, 6.0, 9.0])
        _add_metal(v, rng, seed_zyx, c.metal_distance_mm)
    if rng.random() < 0.3:
        c.partial_volume = True
        v = ndi.gaussian_filter(v, sigma=(1.0, 1.4, 1.4))
    if rng.random() < 0.4:
        c.intensity_drift = rng.choice([-250.0, -120.0, 120.0, 250.0])
        v = v * (1 + c.intensity_drift / 1600.0)
    if rng.random() < 0.3:
        c.restoration_band = True
        _add_restoration_band(v, seg, lesion)
    if c.noise_sigma:
        v = v + np.random.default_rng(rng.randint(0, 2 ** 31 - 1)).normal(0, c.noise_sigma, size=v.shape).astype(np.float32)
    return v, c


# --------------------------------------------------------------------------- #
def _target_differential(label: Optional[str], ambiguous: bool, questions: List[Dict]) -> List[Dict]:
    """Label-smoothed target over HYPOTHESES.  Synthetic truth gets 0.80 (0.60 if the geometry is a
    legacy open wedge, which is genuinely ambiguous with a wide NCCL / caries); the remainder goes to
    the plausible mimics in a fixed order so the model learns a *ranked* differential."""
    if label is None:   # intact tooth: anything that reached the screen is noise
        base = {"imaging_artifact": 0.55, "anatomical_concavity_or_furcation": 0.2, "noncarious_cervical_lesion": 0.1,
                "ECR": 0.05, "caries": 0.05, "restoration_margin_or_burnout": 0.05, "internal_resorption": 0.0}
    else:
        p_true = 0.6 if ambiguous else 0.8
        mimics = {"ECR": ["caries", "noncarious_cervical_lesion", "restoration_margin_or_burnout", "internal_resorption"],
                  "noncarious_cervical_lesion": ["caries", "ECR", "restoration_margin_or_burnout", "anatomical_concavity_or_furcation"],
                  "internal_resorption": ["ECR", "caries", "imaging_artifact", "anatomical_concavity_or_furcation"]}[label]
        rest = 1 - p_true
        base = {h: 0.0 for h in HYPOTHESES}
        base[label] = p_true
        for h, w in zip(mimics, [0.45, 0.3, 0.15, 0.1]):
            base[h] = round(rest * w, 3)
    s = sum(base.values())
    return [{"diagnosis": h, "probability": round(base[h] / s, 3)} for h in sorted(base, key=lambda h: -base[h])]


def _findings_from_questions(questions: List[Dict], label: Optional[str]) -> Tuple[List[str], List[str], List[str]]:
    for_, against, unc = [], [], []
    for q in questions:
        txt = f"{q['id']}: {q['evidence']}"
        if q["answer"] == "yes":
            (for_ if label == "ECR" else against).append(txt)
        elif q["answer"] == "no":
            (against if label == "ECR" else for_).append(txt)
        elif q["answer"] in ("uncertain", "not_assessable"):
            unc.append(txt)
    return for_[:4], against[:4], unc[:4]


@dataclass
class Example:
    task: str
    messages: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"messages": self.messages, "task": self.task, "meta": self.meta}, default=str)


def _schema_system(system: str, schema: Dict) -> str:
    return system + "\nRespond with JSON only, matching this schema: " + json.dumps(schema)


def candidate_lesion_overlap(bb, lesion: np.ndarray) -> Dict[int, float]:
    """Fraction of each candidate's voxels that fall inside the (slightly dilated) true lesion."""
    labels = bb.arrays.get("candidate_labels")
    crop = bb.arrays.get("crop")
    if labels is None:
        return {}
    les = lesion[tuple(crop)] if crop is not None else lesion
    if les.shape != labels.shape:
        return {}
    les = ndi.binary_dilation(les, iterations=2)
    out = {}
    for cid in np.unique(labels[labels > 0]):
        m = labels == cid
        out[int(cid)] = float((m & les).sum() / max(1, m.sum()))
    return out


def build_examples_from_conference(consensus: Dict, calls: List[Dict], label: Optional[str], ambiguous: bool,
                                   confounders: Optional[Confounders], meta: Dict,
                                   overlap: Optional[Dict[int, float]] = None) -> List[Example]:
    """Pair every recorded prompt with its supervised target.  ``overlap`` (candidate -> fraction inside
    the true lesion) decides which component carries the case label; other components are incidental."""
    out: List[Example] = []
    overlap = overlap or {}
    cands = {r["component_id"]: r for r in consensus.get("candidates", [])}
    rule_ch = {}
    for e in consensus["conference"]["evidence"]:
        if e["kind"] == "challenge":
            rule_ch[e["refers_to"]] = e["content"]["challenges"]
    for call in calls:
        try:
            payload = json.loads(call["user"])
        except Exception:
            continue
        d = payload.get("descriptors", {})
        cid = d.get("component_id")
        cand = cands.get(cid)
        is_rad = "diagnostic_questions" in payload
        if is_rad:
            qs = payload["diagnostic_questions"]
            # only the lesion component of a synthetic case carries the truth label; incidental
            # fragments (volume < 1 mm3 on an intact-like region) are labelled as artefact/noise
            if overlap:
                frac = overlap.get(cid, 0.0)
                if frac >= 0.5 and (cand is None or cand["volume_mm3"] < 1.0):
                    continue   # sub-mm3 fragment of the true lesion: no clean label, skip rather than mislabel
                lab = label if frac >= 0.5 else None
            else:
                lab = label if (cand is None or cand["volume_mm3"] >= 1.0 or label is None) else None
            for_, against, unc = _findings_from_questions(qs, lab)
            if confounders:
                unc += [c["issue"] for c in confounders.challenges()][:2]
            target = {"differential": _target_differential(lab, ambiguous, qs),
                      "key_findings_for": for_, "key_findings_against": against, "uncertainties": unc}
            out.append(Example("radiologist",
                               [{"role": "system", "content": _schema_system(call["system"], call["schema"])},
                                {"role": "user", "content": call["user"]},
                                {"role": "assistant", "content": json.dumps(target)}],
                               {**meta, "component_id": cid, "label": lab, "ambiguous": ambiguous}))
        else:
            ch = list(confounders.challenges()) if confounders else []
            # add the geometry-based rule challenges the skeptic itself raised (ground truth for this case)
            for c in rule_ch.get(cid, [])[:4]:
                if not c["issue"].startswith("[model]"):
                    ch.append({"issue": c["issue"], "severity": c["severity"]})
            if not ch:
                ch = [{"issue": "No specific confounder identified; the differential rests on the measured geometry and an "
                                "unreviewed CEJ estimate.", "severity": "minor"}]
            seen, dedup = set(), []
            for c in ch:
                if c["issue"] not in seen:
                    seen.add(c["issue"]); dedup.append(c)
            out.append(Example("skeptic",
                               [{"role": "system", "content": _schema_system(call["system"], call["schema"])},
                                {"role": "user", "content": call["user"]},
                                {"role": "assistant", "content": json.dumps({"challenges": dedup[:5]})}],
                               {**meta, "component_id": cid, "label": label}))
    return out


# --------------------------------------------------------------------------- #
def sample_case(rng: random.Random) -> Dict[str, Any]:
    morph = rng.choices(["ecr", "wedge", "nccl", "internal", "intact"], weights=[0.4, 0.15, 0.2, 0.15, 0.1])[0]
    h = rng.choice([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5]) if morph in ("ecr", "wedge") else \
        rng.choice([0.5, 1.0, 1.5]) if morph == "nccl" else rng.choice([2.0, 3.0, 4.0, 5.0])
    spread = rng.choice([40, 70, 100, 150, 200, 260, 300])
    perforate = (rng.random() < 0.35) if morph in ("ecr", "wedge") else False
    return {"morphology": morph, "height_mm": h, "spread_deg": spread, "perforate": perforate,
            "portal_angle_deg": rng.choice([0, 45, 90, 180, 270]),
            "augment": rng.choice(["clean", "random", "random"])}


def generate_case(spec: Dict[str, Any], rng: random.Random, case_id: str, max_rounds: int = 1) -> Tuple[List[Example], Dict]:
    if spec["morphology"] == "intact":
        vol, seg, lesion, truth = make_phantom(0.0, 10, False, morphology="wedge")
        lesion = np.zeros(SHAPE, bool)
        vol = vol.copy(); vol[seg.tooth & (vol < 1000)] = 1600  # fill the tiny wedge back in
    else:
        vol, seg, lesion, truth = make_phantom(spec["height_mm"], spec["spread_deg"], spec["perforate"],
                                               portal_angle_deg=spec["portal_angle_deg"], morphology=spec["morphology"])
    v, conf = augment(vol, seg, lesion, rng, level=spec["augment"])
    cb = CBCTVolume(v.astype(np.float32), (VOX, VOX, VOX), (0, 0, 0), np.eye(3), {"SeriesInstanceUID_hash": case_id})
    segmenter = lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone)  # noqa: E731
    rec = RecordingBackend()
    t0 = time.time()
    conf_runner = CaseConference(backend=rec, max_rounds=max_rounds)
    cons = conf_runner.run(cb, segmenter, TEST_SEED, "phantom_truth", render_tiles=False, case_id=case_id)
    overlap = candidate_lesion_overlap(conf_runner.last_blackboard, lesion)
    label = MORPH_TO_LABEL[spec["morphology"]]
    meta = {"case_id": case_id, "source": "synthetic_phantom", "spec": spec, "truth": truth,
            "confounders": conf.__dict__, "status": cons["status"], "seconds": round(time.time() - t0, 1),
            "lesion_overlap": overlap}
    exs = build_examples_from_conference(cons, rec.calls, label, spec["morphology"] == "wedge", conf, meta, overlap)
    return exs, meta


def from_confirmed_case(conference_report_path: str, confirmed_diagnosis: str, weight: int = 3) -> List[Example]:
    """Turn a saved real-case conference report + the clinician's confirmed diagnosis into examples.
    The prompts are rebuilt from the report's evidence; the confirmed diagnosis becomes the target
    (0.85 on the confirmed hypothesis).  Duplicated ``weight`` times so real cases count more."""
    from ..agents.roles import RADIOLOGIST_SCHEMA, RADIOLOGIST_SYSTEM, SKEPTIC_SCHEMA, SKEPTIC_SYSTEM
    from ..agents.criteria import CRITERIA, evaluate_criteria
    from ..agents.visual_features import CandidateDescriptors
    if confirmed_diagnosis not in HYPOTHESES:
        raise ValueError(f"confirmed_diagnosis must be one of {HYPOTHESES}")
    rep = json.load(open(conference_report_path))
    dx = rep.get("suggested_diagnosis") or {}
    cid = dx.get("candidate_id")
    cand_ev = next(e for e in rep["conference"]["evidence"] if e["kind"] == "candidate" and e["refers_to"] == cid)
    fields = {k: v for k, v in cand_ev["content"].items() if k in CandidateDescriptors.__dataclass_fields__}
    d = CandidateDescriptors(**fields)
    truth = evaluate_criteria(d)
    qs = dx.get("diagnostic_questions", [])
    challenges = next((e["content"]["challenges"] for e in rep["conference"]["evidence"]
                       if e["kind"] == "challenge" and e["refers_to"] == cid), [])
    user = json.dumps({"descriptors": d.to_dict(), "diagnostic_questions": qs,
                       "criteria_fired": {c.name: v for c, v in zip(CRITERIA, truth)},
                       "skeptic_challenges": challenges}, default=str)
    base = {h: 0.15 / (len(HYPOTHESES) - 1) for h in HYPOTHESES}
    base[confirmed_diagnosis] = 0.85
    for_, against, unc = _findings_from_questions(qs, confirmed_diagnosis if confirmed_diagnosis == "ECR" else None)
    target = {"differential": [{"diagnosis": h, "probability": round(p, 3)} for h, p in sorted(base.items(), key=lambda kv: -kv[1])],
              "key_findings_for": for_, "key_findings_against": against, "uncertainties": unc}
    meta = {"case_id": rep.get("case_id"), "source": "clinician_confirmed", "label": confirmed_diagnosis, "component_id": cid}
    rad = Example("radiologist", [{"role": "system", "content": _schema_system(RADIOLOGIST_SYSTEM, RADIOLOGIST_SCHEMA)},
                                  {"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(target)}], meta)
    sk_user = json.dumps({"descriptors": d.to_dict(), "radiologist": {"leading": confirmed_diagnosis, "diagnostic_questions": qs}}, default=str)
    sk = Example("skeptic", [{"role": "system", "content": _schema_system(SKEPTIC_SYSTEM, SKEPTIC_SCHEMA)},
                             {"role": "user", "content": sk_user},
                             {"role": "assistant", "content": json.dumps({"challenges": [
                                 {"issue": c["issue"].replace("[model] ", ""), "severity": c["severity"]} for c in challenges[:5]]})}], meta)
    return [rad, sk] * weight


# --------------------------------------------------------------------------- #
def write_jsonl(examples: List[Example], path: str, append: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for e in examples:
            f.write(e.to_json() + "\n")


def split(path: str, out_dir: str, eval_fraction: float = 0.2, seed: int = 0) -> Dict[str, int]:
    """Split by *case* (never by example) so the same phantom never appears in both splits."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cases = sorted({r["meta"]["case_id"] for r in rows})
    rng = random.Random(seed); rng.shuffle(cases)
    n_eval = max(1, int(len(cases) * eval_fraction))
    ev = set(cases[:n_eval])
    os.makedirs(out_dir, exist_ok=True)
    counts = {"train": 0, "eval": 0}
    with open(os.path.join(out_dir, "train.jsonl"), "w") as ft, open(os.path.join(out_dir, "eval.jsonl"), "w") as fe:
        for r in rows:
            f = fe if r["meta"]["case_id"] in ev else ft
            f.write(json.dumps(r) + "\n"); counts["eval" if f is fe else "train"] += 1
    counts.update(n_cases=len(cases), n_eval_cases=n_eval)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic SFT examples for the conference LLM roles")
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--out", default="finetune_data/raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0, help="worker index (cases are strided over workers)")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    specs = [sample_case(rng) for _ in range(a.n_cases)]
    done = set()
    if os.path.exists(a.out):   # resume
        for l in open(a.out):
            try:
                done.add(json.loads(l)["meta"]["case_id"])
            except Exception:
                pass
    for i, spec in enumerate(specs):
        if i % a.n_workers != a.worker:
            continue
        cid = f"syn{a.seed:02d}_{i:04d}"
        if cid in done:
            continue
        crng = random.Random(hash((a.seed, i)) & 0xFFFFFFFF)
        try:
            exs, meta = generate_case(spec, crng, cid, max_rounds=a.rounds)
        except Exception as e:  # keep going; log the failure
            print(f"[{cid}] FAILED {spec}: {e}", flush=True)
            continue
        write_jsonl(exs, a.out)
        print(f"[{cid}] {spec['morphology']:8s} h={spec['height_mm']} s={spec['spread_deg']} aug={spec['augment']:6s} "
              f"-> {meta['status']} {len(exs)} examples ({meta['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
