"""Command-line entry points.

  python -m ecr_cds.cli run       ...   run a case end-to-end and write a report
  python -m ecr_cds.cli feedback  ...   record clinician decision / outcome, update policy
  python -m ecr_cds.cli train     ...   train the texture-SVM lesion segmenter
  python -m ecr_cds.cli evaluate  ...   off-policy evaluate a candidate policy vs. the log
  python -m ecr_cds.cli matrix    ...   print the 32-cell matrix / end-stage matrix
  python -m ecr_cds.cli conference ... multi-agent case conference: suggested diagnosis + plan (research)
  python -m ecr_cds.cli vlm-measure ... CBCT folder/ZIP -> QA -> features -> conference JSON (no Patel codes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

from .decision_matrix import (ClinicalModifiers, DECISION_MATRIX, END_STAGE_CODES, Treatment,
                              end_stage_matrix, recommend_from_matrix)
from .pipeline import run_pipeline, to_json
from .rl_policy import (ClinicalContext, ConstrainedBanditPolicy, Decision, Feedback,
                        build_reward, offline_evaluate, promotion_gate)
from .patel_classifier import PatelParameters


def _bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    return v.lower() in {"1", "true", "yes", "y"}


def _modifiers(ns) -> ClinicalModifiers:
    return ClinicalModifiers(
        symptomatic=_bool(ns.symptomatic), pulp_vital=_bool(ns.pulp_vital),
        restorable=_bool(ns.restorable), perio_support_adequate=_bool(ns.perio),
        multirooted=_bool(ns.multirooted), lesion_on_root=ns.lesion_on_root,
        surgically_accessible=_bool(ns.accessible), persistent_bleeding=_bool(ns.bleeding),
        growing_patient=_bool(ns.growing), aesthetic_zone=_bool(ns.aesthetic),
        root_fracture_risk_high=_bool(ns.fracture_risk),
    )


def _load_policy(path: Optional[str]) -> ConstrainedBanditPolicy:
    if path and os.path.exists(path):
        return ConstrainedBanditPolicy.load(path)
    return ConstrainedBanditPolicy()


# --------------------------------------------------------------------------- #
def cmd_run(ns):
    from .adapters import SlicerLabelmapToothSegmenter, PretrainedLesionSegmenter
    seed = tuple(int(v) for v in ns.seed.split(","))
    tooth_seg = SlicerLabelmapToothSegmenter(ns.labelmap, seed,
                                             teeth_labels=[int(v) for v in ns.teeth_labels.split(",")],
                                             bone_labels=[int(v) for v in ns.bone_labels.split(",")])
    lesion_seg = PretrainedLesionSegmenter(model_path=ns.lesion_model, mask_path=ns.lesion_mask)
    card = None
    if ns.lesion_model and os.path.exists(ns.lesion_model.replace(".joblib", ".card.json")):
        card = json.load(open(ns.lesion_model.replace(".joblib", ".card.json")))
    policy = _load_policy(ns.policy)
    buccal = np.array([float(v) for v in ns.buccal.split(",")]) if ns.buccal else None
    report = run_pipeline(ns.dicom, tooth_seg, lesion_seg, modifiers=_modifiers(ns), policy=policy,
                          lesion_threshold=ns.threshold, tooth_type=ns.tooth_type, age_years=ns.age,
                          buccal_direction=buccal, expected_slices=ns.expected_slices,
                          case_id=ns.case_id, model_card=card,
                          lesion_source="expert_mask" if ns.lesion_mask else "model",
                          ecr_confirmed=ns.ecr_confirmed)
    os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
    to_json(report, ns.out)
    if ns.policy:
        policy.save(ns.policy)  # persists priors created for this cell
    _print_summary(report)


def _print_summary(rep):
    if rep["status"] not in ("ok", "provisional"):
        print(rep["status"], "-", rep.get("note", ""))
        return
    print(f"STATUS: {rep['status'].upper()} | evidence: {rep['evidence_level']} - {rep['evidence_note']}")
    p, d = rep["patel"], rep["decision"]
    print(f"Patel {p['code']}  | apical {p['apical_extent_mm']:.1f} mm, spread {p['spread_deg']:.0f}°, "
          f"canal dist {p['min_canal_distance_mm']:.2f} mm, volume {p['lesion_volume_mm3']:.0f} mm³, "
          f"calcified {p['calcified_fraction']:.2f}, borderline={p['borderline']}")
    print("Recommendation:", d["chosen"], "(confirm)" if d["requires_confirmation"] else "")
    if rep["rule_recommendation"].get("missing_inputs"):
        print("MISSING clinical inputs:", ", ".join(rep["rule_recommendation"]["missing_inputs"]))
    print("Ranking:", " > ".join(d["ranking"]))
    print("Propensities:", {k: round(v, 2) for k, v in d["propensities"].items()})
    for r in d["rationale"]:
        print("  -", r)
    if rep["end_stage"]:
        print("END-STAGE lesion: see end_stage_matrix in the report")
    if rep["qa"]["issues"]:
        print("QA:", *rep["qa"]["issues"], sep="\n  ")


# --------------------------------------------------------------------------- #
def cmd_feedback(ns):
    """Record the clinician's decision (now) and/or outcome (later) for a report."""
    with open(ns.report) as f:
        rep = json.load(f)
    if not rep.get("patel") or not rep.get("decision"):
        raise ValueError("Feedback learning is blocked: report contains no reviewed ECR treatment decision.")
    policy = _load_policy(ns.policy)
    patel = PatelParameters(**{k: v for k, v in rep["patel"].items() if k != "code"})
    # rebuild context & decision objects from the report
    mods = ClinicalModifiers(**rep.get("modifiers", {})) if rep.get("modifiers") else ClinicalModifiers()
    ctx = ClinicalContext(patel=patel, mods=mods, tooth_type=rep.get("tooth_type", "unknown"),
                          age_years=rep.get("age_years"))
    d = rep["decision"]
    decision = Decision(cell_key=d.get("cell_key") or d["patel_code"], cell_scheme=d.get("cell_scheme", "patel_3d"),
                        chosen=Treatment(d["chosen"]),
                        ranking=[Treatment(x) for x in d["ranking"]], propensities=d["propensities"],
                        safe_set=[Treatment(x) for x in d["safe_set"]],
                        requires_confirmation=d["requires_confirmation"],
                        reinspect_requested=d["reinspect_requested"], rationale=d["rationale"])
    fb = Feedback(clinician_choice=Treatment(ns.clinician_choice),
                  outcome_success=_bool(ns.success), outcome_survival=_bool(ns.survival),
                  follow_up_months=ns.months, reinspection_changed_code=_bool(ns.reinspection_changed),
                  case_id=str(rep.get("case_id", "unknown")))
    reward, comps = build_reward(decision, fb, borderline=patel.borderline,
                                 rubric_conf=max(d["propensities"].values()),
                                 reinspections_so_far=rep.get("reinspections", 0))
    applied = policy.update(ctx, decision, fb, reward, comps)
    if not applied:
        print(f"event {fb.event_id} already recorded - nothing changed"); return
    policy.save(ns.policy)
    rep.setdefault("feedback", []).append({"clinician_choice": ns.clinician_choice, "success": ns.success,
                                           "survival": ns.survival, "months": ns.months,
                                           "reward": reward, "components": comps})
    with open(ns.report, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"event {fb.event_id}: reward={reward:.3f} {comps} | policy saved to {ns.policy} "
          f"({len(policy.log)} events, {policy.n_cases} independent cases)")


# --------------------------------------------------------------------------- #
def cmd_train(ns):
    """Train the texture-SVM lesion segmenter from a manifest.

    manifest.json: [{"dicom": "...", "labelmap": "...", "seed": [z,y,x], "lesion_mask": "..."}, ...]
    """
    import joblib
    from .dicom_loader import load_dicom_series
    from .preprocess import preprocess_volume
    from .adapters import SlicerLabelmapToothSegmenter, load_labelmap
    from .segmentation import TextureSVMLesionSegmenter

    with open(ns.manifest) as f:
        cases = json.load(f)
    vols, tooth_masks, lesion_masks = [], [], []
    for c in cases:
        v = load_dicom_series(c["dicom"])
        seg = SlicerLabelmapToothSegmenter(c["labelmap"], tuple(c["seed"]))(v.array, v.spacing)
        norm = preprocess_volume(v.array, roi=seg.tooth)
        vols.append(norm); tooth_masks.append(seg.tooth & ~seg.canal)
        lesion_masks.append(load_labelmap(c["lesion_mask"]) > 0)
        print("loaded", c["dicom"], "lesion voxels:", int(lesion_masks[-1].sum()))
    model = TextureSVMLesionSegmenter(radius=ns.radius, threshold=ns.threshold).fit(vols, tooth_masks, lesion_masks)
    joblib.dump(model, ns.out)
    print("saved", ns.out)


# --------------------------------------------------------------------------- #
def cmd_evaluate(ns):
    """Hold out the most recent cases; fit a candidate on the rest (or load one);
    evaluate on the held-out cases only."""
    from .rl_policy import split_log_by_case
    deployed = ConstrainedBanditPolicy.load(ns.policy)
    if not deployed.log:
        print("no logged cases yet"); return
    train_log, hold_log = split_log_by_case(deployed.log, ns.holdout)
    if ns.candidate:
        candidate = ConstrainedBanditPolicy.load(ns.candidate)
    else:
        # candidate refit from the training part of the log only (tabular part)
        candidate = ConstrainedBanditPolicy()
        for e in train_log:
            key = (e["code"], e["chosen"])
            r01 = float(np.clip((e["reward"] + 1.0) / 2.4, 0, 1))
            candidate.alpha[key] = candidate.alpha.get(key, 1.0) + r01
            candidate.beta[key] = candidate.beta.get(key, 1.0) + (1 - r01)
    target = []
    for e in hold_log:
        cell = e["code"]
        acts = [a for (c, a) in candidate.alpha if c == cell] or [e["chosen"]]
        means = np.array([candidate.alpha.get((cell, a), 1) / (candidate.alpha.get((cell, a), 1) + candidate.beta.get((cell, a), 1)) for a in acts])
        probs = np.exp(8 * means); probs /= probs.sum()
        target.append(dict(zip(acts, probs.tolist())))
    ev = offline_evaluate(hold_log, target)
    ev["n_train_events"] = len(train_log)
    print(json.dumps(ev, indent=2))
    print("PROMOTE (held-out gate passed)" if promotion_gate(ev) else "DO NOT PROMOTE")


def cmd_matrix(ns):
    if ns.end_stage:
        for r in end_stage_matrix():
            print(json.dumps(r))
    else:
        for code, (prim, alts, prior, ev) in DECISION_MATRIX.items():
            rec = recommend_from_matrix(code)
            print(f"{code}: {prim.value:28s} alts={[a.value for a in alts]} prior={prior} {ev}"
                  f"{'  [END-STAGE]' if code in END_STAGE_CODES else ''}")


# --------------------------------------------------------------------------- #
def cmd_conference(ns):
    """Run the multi-agent case conference on a DICOM folder/ZIP (automatic segmentation unless a label map is given)."""
    from .dicom_loader import load_dicom_series
    from .agents import CaseConference, auto_backend, consensus_markdown
    vol = load_dicom_series(ns.dicom, expected_slices=ns.expected_slices)
    auto, seed_source = None, "clinician"
    if ns.labelmap:
        from .adapters import SlicerLabelmapToothSegmenter
        seed = tuple(int(v) for v in ns.seed.split(","))
        tooth_seg = SlicerLabelmapToothSegmenter(ns.labelmap, seed)
        if ns.canal_mask:
            tooth_seg.canal_mask_path = ns.canal_mask
    else:
        from .auto_segment import auto_segment, AutoToothSegmenter, score_candidates_radiolucency, choose_seed
        auto = auto_segment(vol.array, vol.spacing)
        if ns.seed:
            seed = tuple(int(v) for v in ns.seed.split(","))
        else:
            seed, seed_source = choose_seed(score_candidates_radiolucency(auto, vol.array, vol.spacing))
        tooth_seg = AutoToothSegmenter(auto, seed)
    conf = CaseConference(backend=auto_backend(ns.llm), max_rounds=ns.rounds, staging_scheme=ns.staging_scheme,
                          vlm_features=not ns.no_vlm_features)
    cej = tuple(int(v) for v in ns.cej.split(",")) if ns.cej else None
    apex = tuple(int(v) for v in ns.apex.split(",")) if ns.apex else None
    rep = conf.run(vol, tooth_seg, seed, seed_source=seed_source, auto=auto,
                   canal_source="expert_mask" if ns.canal_mask else None, cej_zyx=cej, apex_zyx=apex,
                   modifiers=_modifiers(ns), policy=_load_policy(ns.policy), tooth_type=ns.tooth_type,
                   age_years=ns.age, case_id=ns.case_id, verify_mode=ns.verify_mode,
                   sharpen_display=ns.sharpen_display, counterfactual_check=ns.counterfactual,
                   buccal_direction=_buccal(ns))
    paths = conf.save_outputs(rep, ns.out, volume=vol)
    print(consensus_markdown(rep))
    print("written:", *paths, sep="\n  ")


def cmd_diagnocat_run(ns):
    """Diagnocat study (JSON export | report PDF | Partner API) + chart -> prioritised per-tooth conferences."""
    from .dicom_loader import load_dicom_series
    from .agents import auto_backend
    from .agents.chart import PatientChart
    from .diagnocat import load_diagnocat_json, load_diagnocat_pdf, DiagnocatClient
    from .diagnocat.pipeline import run_diagnocat_pipeline, patient_markdown
    vol = load_dicom_series(ns.dicom, expected_slices=ns.expected_slices)
    study = None
    if ns.diagnocat_json:
        study = load_diagnocat_json(ns.diagnocat_json)
    elif ns.diagnocat_pdf:
        study = load_diagnocat_pdf(ns.diagnocat_pdf)
    elif ns.diagnocat_api:
        study = DiagnocatClient().process(ns.dicom)
    chart = PatientChart.from_dict(json.load(open(ns.chart))) if ns.chart else None
    teeth = [int(t) for t in ns.teeth.split(",")] if ns.teeth else None
    rep = run_diagnocat_pipeline(vol, study, chart=chart, teeth_iso=teeth, max_teeth=ns.max_teeth,
                                 backend=auto_backend(ns.llm), max_rounds=ns.rounds, modifiers=_modifiers(ns),
                                 out_dir=ns.out, verify=not ns.no_verify, verify_mode=ns.verify_mode,
                                 sharpen_display=ns.sharpen_display, counterfactual_check=ns.counterfactual,
                                 staging_scheme=ns.staging_scheme, vlm_features=not ns.no_vlm_features,
                                 buccal_direction=_buccal(ns))
    print(patient_markdown(rep)[:4000])
    print("written:", os.path.join(ns.out, "patient_report.json"), os.path.join(ns.out, "patient_summary.md"))


def _rev28_flags(p):
    """Rev 2.8 options (all off by default so the rev 2.7 frozen baseline is reproduced unless asked)."""
    p.add_argument("--verify-mode", dest="verify_mode", default="smooth", choices=["smooth", "tta"],
                   help="critic re-inspection: 'smooth' (frozen baseline) or 'tta' (gain/noise/threshold/smoothing paths)")
    p.add_argument("--counterfactual", action="store_true",
                   help="also show the vision model a candidate-filled frame; weak image grounding caps its weight at 0.2")
    p.add_argument("--sharpen-display", dest="sharpen_display", action="store_true",
                   help="unsharp-mask the evidence frames (display only; recorded in the frame provenance)")


def _rev29_flags(p):
    """Rev 2.9 feature-measurement options (docs/VLM_FEATURE_MEASUREMENT.md)."""
    p.add_argument("--staging-scheme", dest="staging_scheme", default="measurement", choices=["measurement", "patel_3d"],
                   help="'measurement' (default): descriptive extent summary from VLM features + voxel geometry, no staging "
                        "codes; 'patel_3d': legacy classifier, reproduces rev <= 2.8 outputs")
    p.add_argument("--no-vlm-features", dest="no_vlm_features", action="store_true",
                   help="skip the VLM visual-feature query; measurements are still computed from the voxel geometry")
    p.add_argument("--buccal", help="buccal direction vector z,y,x in voxel axes (gives the BL / MD extents a patient frame)")


def _buccal(ns):
    b = getattr(ns, "buccal", None)
    return tuple(float(v) for v in b.split(",")) if b else None


def cmd_phantom_dicom(ns):
    """Write a tiny synthetic CBCT series (no patient data) for the feature-measurement smoke test."""
    from .phantom import TINY_VOX, make_tiny_cbct, write_dicom_series
    vol = make_tiny_cbct()
    os.makedirs(ns.out, exist_ok=True)
    write_dicom_series(vol, ns.out, voxel_mm=TINY_VOX)
    print(f"wrote {vol.shape[0]} synthetic CBCT slices ({vol.shape[1]}x{vol.shape[2]}, {TINY_VOX} mm) to {ns.out}")


def cmd_vlm_measure(ns):
    """Load a CBCT folder or ZIP and run the measurement conference.  No Patel codes.

    Feature reader:
      VLM_BASE_URL unset -> deterministic local image statistics (no GPU).
      VLM_BASE_URL set   -> that OpenAI-compatible vision server.
    Conference reasoning stays on the rule agents.  The command prints which mode ran.
    """
    from .dicom_loader import load_dicom_series
    from .pipeline import _quality_checks
    from .agents import CaseConference, NullBackend, consensus_markdown
    from .agents.llm_backend import resolve_feature_backend

    try:
        feature_backend, mode_id, detail = resolve_feature_backend()
    except RuntimeError as e:
        print(f"VLM mode: openai_compatible — not started\n{e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(f"VLM mode: {mode_id} ({detail})")

    vol = load_dicom_series(ns.dicom, expected_slices=ns.expected_slices)
    qa = _quality_checks(vol)
    print(f"Loaded {qa['slices']} slices, shape {qa['shape_zyx']}, spacing {qa['spacing_mm']} mm")
    print("Geometry QA:", "no issues" if not qa["issues"] else "")
    for issue in qa["issues"]:
        print("  QA:", issue)

    auto, seed_source = None, "clinician"
    if ns.labelmap:
        from .adapters import SlicerLabelmapToothSegmenter
        seed = tuple(int(v) for v in ns.seed.split(","))
        tooth_seg = SlicerLabelmapToothSegmenter(ns.labelmap, seed)
    else:
        from .auto_segment import auto_segment, AutoToothSegmenter, score_candidates_radiolucency, choose_seed
        auto = auto_segment(vol.array, vol.spacing)
        if ns.seed:
            seed = tuple(int(v) for v in ns.seed.split(","))
            seed_source = "clinician"
        elif not auto.candidates:
            seed = None
        else:
            seed, seed_source = choose_seed(score_candidates_radiolucency(auto, vol.array, vol.spacing))
        if seed is None:
            print("No tooth seed from automatic segmentation; cannot measure.", file=sys.stderr)
            raise SystemExit(3)
        tooth_seg = AutoToothSegmenter(auto, seed)
        print(f"Perception seed {seed} ({seed_source}); {len(auto.candidates)} tooth candidate(s)")

    conf = CaseConference(backend=NullBackend(), feature_backend=feature_backend, max_rounds=ns.rounds,
                          staging_scheme="measurement", vlm_features=True)
    cej = tuple(int(v) for v in ns.cej.split(",")) if ns.cej else None
    apex = tuple(int(v) for v in ns.apex.split(",")) if ns.apex else None
    rep = conf.run(vol, tooth_seg, seed, seed_source=seed_source, auto=auto,
                   cej_zyx=cej, apex_zyx=apex, modifiers=_modifiers(ns), tooth_type=ns.tooth_type,
                   age_years=ns.age, case_id=ns.case_id, verify_mode=ns.verify_mode,
                   sharpen_display=ns.sharpen_display, buccal_direction=_buccal(ns))
    statuses = {cid: (b.get("vlm") or {}).get("status") for cid, b in (rep.get("feature_bundles") or {}).items()}
    rep["vlm_mode"] = {"id": mode_id, "detail": detail, "feature_backend": getattr(feature_backend, "name", None),
                       "per_candidate_status": {str(k): v for k, v in statuses.items()},
                       "conference_reasoning": "rules",
                       "staging_scheme": "measurement"}
    rep.setdefault("architecture", {}).setdefault("settings", {})["vlm_mode"] = mode_id
    if "suggested_patel" in rep or (rep.get("suggested_plan") or {}).get("cell_scheme") == "patel_3d":
        print("Internal error: measurement path emitted a Patel staging output.", file=sys.stderr)
        raise SystemExit(4)
    paths = conf.save_outputs(rep, ns.out, volume=vol)
    print(consensus_markdown(rep))
    print(f"VLM mode ran: {mode_id} ({detail}); candidate feature status: {statuses or 'no candidates'}")
    print("written:", *paths, sep="\n  ")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ecr_cds")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--dicom", required=True); r.add_argument("--labelmap", required=True)
    r.add_argument("--seed", required=True, help="z,y,x voxel inside the target tooth")
    r.add_argument("--lesion-model"); r.add_argument("--lesion-mask")
    r.add_argument("--ecr-confirmed", action="store_true",
                   help="Explicitly attest a clinician has established ECR; does not validate the model.")
    r.add_argument("--teeth-labels", default="3,4"); r.add_argument("--bone-labels", default="1,2")
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--policy", default="policy.json"); r.add_argument("--out", default="report.json")
    r.add_argument("--case-id", default="case"); r.add_argument("--expected-slices", type=int)
    r.add_argument("--tooth-type", default="unknown"); r.add_argument("--age", type=float)
    r.add_argument("--buccal", help="buccal direction vector z,y,x in voxel axes")
    for m in ("symptomatic", "pulp-vital", "restorable", "perio", "multirooted", "accessible",
              "bleeding", "growing", "aesthetic", "fracture-risk"):
        r.add_argument(f"--{m}", dest=m.replace("-", "_"))
    r.add_argument("--lesion-on-root")
    r.set_defaults(func=cmd_run)

    fbp = sub.add_parser("feedback")
    fbp.add_argument("--report", required=True); fbp.add_argument("--policy", default="policy.json")
    fbp.add_argument("--clinician-choice", required=True, choices=[t.value for t in Treatment])
    fbp.add_argument("--success"); fbp.add_argument("--survival"); fbp.add_argument("--months", type=float)
    fbp.add_argument("--reinspection-changed")
    fbp.set_defaults(func=cmd_feedback)

    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True); t.add_argument("--out", default="lesion_svm.joblib")
    t.add_argument("--radius", type=int, default=5); t.add_argument("--threshold", type=float, default=0.5)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--policy", default="policy.json"); e.add_argument("--candidate")
    e.add_argument("--holdout", type=float, default=0.3, help="fraction of most recent cases held out")
    e.set_defaults(func=cmd_evaluate)

    m = sub.add_parser("matrix"); m.add_argument("--end-stage", action="store_true")
    m.set_defaults(func=cmd_matrix)

    c = sub.add_parser("conference")
    c.add_argument("--dicom", required=True); c.add_argument("--out", default="conference_out")
    c.add_argument("--labelmap"); c.add_argument("--canal-mask"); c.add_argument("--seed", help="z,y,x (required with --labelmap)")
    c.add_argument("--cej", help="reviewed CEJ point z,y,x"); c.add_argument("--apex", help="reviewed apex point z,y,x")
    c.add_argument("--llm", default="auto", choices=["auto", "none", "openai", "perplexity", "nemotron"],
                   help="'nemotron' = local Nemotron 3.5 Bolt via an OpenAI-compatible server (NEMOTRON_BASE_URL)")
    c.add_argument("--rounds", type=int, default=2); c.add_argument("--policy", default="policy.json")
    _rev28_flags(c); _rev29_flags(c)
    c.add_argument("--case-id"); c.add_argument("--expected-slices", type=int)
    c.add_argument("--tooth-type", default="unknown"); c.add_argument("--age", type=float)
    for m_ in ("symptomatic", "pulp-vital", "restorable", "perio", "multirooted", "accessible",
               "bleeding", "growing", "aesthetic", "fracture-risk"):
        c.add_argument(f"--{m_}", dest=m_.replace("-", "_"))
    c.add_argument("--lesion-on-root")
    c.set_defaults(func=cmd_conference)

    d = sub.add_parser("diagnocat-run", help="Diagnocat findings + chart -> per-tooth conferences -> patient report")
    d.add_argument("--dicom", required=True); d.add_argument("--out", default="diagnocat_out")
    d.add_argument("--diagnocat-json"); d.add_argument("--diagnocat-pdf")
    d.add_argument("--diagnocat-api", action="store_true", help="use the Partner API (DIAGNOCAT_BASE_URL / DIAGNOCAT_API_KEY)")
    d.add_argument("--chart", help="PatientChart JSON"); d.add_argument("--teeth", help="FDI numbers to force, e.g. 26,11")
    d.add_argument("--max-teeth", type=int, default=3); d.add_argument("--rounds", type=int, default=2)
    d.add_argument("--llm", default="auto", choices=["auto", "none", "openai", "perplexity", "nemotron"],
                   help="'nemotron' = local Nemotron 3.5 Bolt via an OpenAI-compatible server (NEMOTRON_BASE_URL)")
    d.add_argument("--no-verify", action="store_true"); d.add_argument("--expected-slices", type=int)
    _rev28_flags(d); _rev29_flags(d)
    for m_ in ("symptomatic", "pulp-vital", "restorable", "perio", "multirooted", "accessible",
               "bleeding", "growing", "aesthetic", "fracture-risk"):
        d.add_argument(f"--{m_}", dest=m_.replace("-", "_"))
    d.add_argument("--lesion-on-root")
    d.set_defaults(func=cmd_diagnocat_run)

    vm = sub.add_parser("vlm-measure",
                        help="CBCT folder or ZIP -> geometry QA -> perception -> VLM features -> conference JSON (no Patel codes)")
    vm.add_argument("--dicom", required=True, help="CBCT DICOM folder or ZIP")
    vm.add_argument("--out", default="vlm_measure_out")
    vm.add_argument("--labelmap"); vm.add_argument("--seed", help="z,y,x (required with --labelmap)")
    vm.add_argument("--cej", help="reviewed CEJ point z,y,x"); vm.add_argument("--apex", help="reviewed apex point z,y,x")
    vm.add_argument("--rounds", type=int, default=1)
    vm.add_argument("--case-id"); vm.add_argument("--expected-slices", type=int)
    vm.add_argument("--tooth-type", default="unknown"); vm.add_argument("--age", type=float)
    _rev28_flags(vm)
    vm.add_argument("--buccal", help="buccal direction vector z,y,x in voxel axes")
    for m_ in ("symptomatic", "pulp-vital", "restorable", "perio", "multirooted", "accessible",
               "bleeding", "growing", "aesthetic", "fracture-risk"):
        vm.add_argument(f"--{m_}", dest=m_.replace("-", "_"))
    vm.add_argument("--lesion-on-root")
    vm.set_defaults(func=cmd_vlm_measure)

    ph = sub.add_parser("phantom-dicom", help="write a tiny synthetic CBCT DICOM folder (no patient data)")
    ph.add_argument("--out", default="tiny_cbct")
    ph.set_defaults(func=cmd_phantom_dicom)

    ns = ap.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
