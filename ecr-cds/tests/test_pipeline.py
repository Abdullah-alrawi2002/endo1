"""Synthetic-phantom tests for the ECR CDS pipeline.

Builds a 500-slice CBCT-like DICOM series (0.1 mm isotropic) containing a
cylindrical tooth with a tapered root canal, alveolar bone, enamel cap and a
lesion of controllable height / circumferential spread / canal proximity.
Then checks that the Patel classifier recovers the planted parameters and
that the decision matrix + RL layer behave as designed.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecr_cds.dicom_loader import load_dicom_series, CBCTVolume
from ecr_cds.segmentation import ToothSegmentation
from ecr_cds.patel_classifier import classify_patel
from ecr_cds.decision_matrix import (ClinicalModifiers, recommend_from_matrix, Treatment,
                                     DECISION_MATRIX, END_STAGE_CODES)
from ecr_cds.rl_policy import (ConstrainedBanditPolicy, ClinicalContext, Feedback, build_reward,
                               offline_evaluate, promotion_gate)
from ecr_cds.pipeline import run_pipeline

from ecr_cds.phantom import make_phantom, write_dicom_series, VOX, SHAPE


# --------------------------------------------------------------------------- #
def test_dicom_roundtrip_500_slices():
    vol, seg, lesion, truth = make_phantom(3.0, 60, False)
    with tempfile.TemporaryDirectory() as d:
        write_dicom_series(vol, d)
        v = load_dicom_series(d, expected_slices=500)
    assert v.shape == SHAPE and v.metadata["NumberOfSlices"] == 500
    assert np.allclose(v.spacing, (VOX, VOX, VOX))
    assert np.allclose(v.array, np.clip(vol, 0, 4095).astype(np.uint16), atol=1)
    print("DICOM round-trip OK:", v.shape, v.spacing, v.metadata["Isotropic"])


def test_patel_recovery():
    cases = [
        # (apical height mm, spread deg, perforate) -> expected code
        (1.0, 60, False, "1Ad"),
        (3.0, 60, False, "2Ad"),
        (3.0, 150, True, "2Bp"),
        (5.0, 240, True, "2Cp"),
        (9.0, 150, True, "3Bp"),
        (9.0, 300, False, "3Dd"),
        (16.0, 60, True, "4Ap"),
        (16.0, 320, True, "4Dp"),
    ]
    ok = 0
    for h, s, p, expected in cases:
        vol, seg, lesion, truth = make_phantom(h, s, p)
        norm = (np.clip(vol, 0, 3000) / 3000).astype(np.float32)
        pp = classify_patel(lesion, seg, (VOX, VOX, VOX), volume_norm=norm)
        status = "OK " if pp.code == expected else "BAD"
        ok += pp.code == expected
        print(f"{status} planted h={h:>4}mm s={s:>3}° perf={p!s:5} -> {pp.code} (exp {expected}) "
              f"| meas apical={pp.apical_extent_mm:.1f}mm crest={pp.bone_crest_mm:.1f} "
              f"spread={pp.spread_deg:.0f}° dmin={pp.min_canal_distance_mm:.2f}mm borderline={pp.borderline}")
    assert ok == len(cases), f"{ok}/{len(cases)} Patel codes recovered"


def test_matrix_complete_and_rules():
    codes = [f"{h}{s}{p}" for h in "1234" for s in "ABCD" for p in "dp"]
    assert set(codes) == set(DECISION_MATRIX), "matrix must cover all 32 cells"
    # ESE anchors
    assert recommend_from_matrix("1Ad").primary == Treatment.EXTERNAL_REPAIR
    assert recommend_from_matrix("2Bd").primary == Treatment.EXTERNAL_REPAIR
    assert recommend_from_matrix("2Bp").primary == Treatment.EXTERNAL_REPAIR_RCT
    assert recommend_from_matrix("3Cp").primary == Treatment.INTERNAL_REPAIR_RCT
    assert recommend_from_matrix("3Bd").primary == Treatment.INTENTIONAL_REPLANT
    assert recommend_from_matrix("4Dp").primary == Treatment.EXTRACTION
    # Modifier rules
    r = recommend_from_matrix("2Bp", ClinicalModifiers(surgically_accessible=False))
    assert r.primary == Treatment.INTERNAL_REPAIR_RCT, r.primary
    r = recommend_from_matrix("3Bp", ClinicalModifiers(persistent_bleeding=True, multirooted=True, lesion_on_root="DB"))
    assert r.primary == Treatment.HEMISECTION, r.primary       # RJME case 2 (#17 DB root)
    r = recommend_from_matrix("2Bp", ClinicalModifiers(surgically_accessible=False, persistent_bleeding=True, multirooted=False))
    assert r.primary == Treatment.EXTRACTION, r.primary        # RJME case 3 (#35)
    r = recommend_from_matrix("4Dd", ClinicalModifiers(growing_patient=True))
    assert r.primary == Treatment.DECORONATION, r.primary
    r = recommend_from_matrix("3Dp", ClinicalModifiers(restorable=False, symptomatic=False))
    assert r.primary == Treatment.MONITOR and "untreatable" in r.flags
    full = ClinicalModifiers(symptomatic=False, pulp_vital=True, restorable=True, perio_support_adequate=True,
                             surgically_accessible=True, growing_patient=False, multirooted=False)
    assert all(recommend_from_matrix(c, full).requires_confirmation == recommend_from_matrix(c, full).primary.irreversible for c in codes)
    assert all(recommend_from_matrix(c).requires_confirmation for c in codes)  # unknown inputs always need confirmation
    print("Decision matrix: 32/32 cells, ESE anchors and modifier rules OK")


def test_rl_loop_learns_within_safe_set():
    """Simulate a clinic where, for 2Cp, clinicians consistently prefer the
    external approach and it succeeds; the bandit should shift from the
    guideline default (internal repair) to external repair + RCT while never
    leaving the safe set."""
    vol, seg, lesion, truth = make_phantom(5.0, 240, True)
    pp = classify_patel(lesion, seg, (VOX, VOX, VOX))
    mods = ClinicalModifiers(symptomatic=False, pulp_vital=True, restorable=True, surgically_accessible=True)
    rec = recommend_from_matrix(pp.code, mods)
    assert rec.primary == Treatment.INTERNAL_REPAIR_RCT
    pol = ConstrainedBanditPolicy(seed=1)
    ctx = ClinicalContext(pp, mods, tooth_type="premolar", age_years=40)
    first = pol.decide(ctx, rec).chosen
    rng = np.random.default_rng(3)
    for i in range(80):
        d = pol.decide(ctx, rec)
        assert d.chosen in rec.safe_set
        fb = Feedback(clinician_choice=Treatment.EXTERNAL_REPAIR_RCT, case_id=f"sim{i}",
                      outcome_success=bool(rng.random() < (0.9 if d.chosen == Treatment.EXTERNAL_REPAIR_RCT else 0.6)),
                      outcome_survival=True, follow_up_months=12,
                      reinspection_changed_code=False)
        r, comp = build_reward(d, fb, borderline=pp.borderline, rubric_conf=0.8, reinspections_so_far=0)
        pol.update(ctx, d, fb, r, comp)
    final = pol.decide(ctx, rec)
    print(f"RL: initial={first.value} -> after 80 cases={final.chosen.value} "
          f"props={ {k: round(v,2) for k,v in final.propensities.items()} }")
    assert final.chosen == Treatment.EXTERNAL_REPAIR_RCT
    # off-policy evaluation + promotion gate on the logged data
    target = [final.propensities for _ in pol.log]
    ev = offline_evaluate(pol.log, target)
    print("OPE:", {k: round(v, 3) for k, v in ev.items()}, "promote:", promotion_gate(ev))
    # guardrail: a violating decision gets the hard penalty
    bad = pol.decide(ctx, rec); bad.safe_set = [Treatment.MONITOR]
    r, comp = build_reward(bad, Feedback(Treatment.MONITOR, case_id="bad"), False, 0.9, 0)
    assert comp["format"] == -1.0


def test_end_to_end_pipeline_on_dicom():
    vol, seg, lesion, truth = make_phantom(9.0, 150, True)   # 3Bp like RJME cases 1-2
    with tempfile.TemporaryDirectory() as d:
        write_dicom_series(vol, d)
        tooth_seg = lambda arr, sp: seg                       # stand-in for DentalSegmentator
        lesion_seg = lambda norm, s: lesion.astype(np.float32)  # stand-in for texture-SVM / nnU-Net
        rep = run_pipeline(d, tooth_seg, lesion_seg,
                           modifiers=ClinicalModifiers(symptomatic=True, pulp_vital=True, multirooted=True,
                                                       lesion_on_root="DB", persistent_bleeding=True),
                           tooth_type="molar", age_years=52, expected_slices=500,
                           ecr_confirmed=True)  # known planted lesion; NOT patient diagnosis
    assert rep["status"] in ("ok", "provisional") and rep["evidence_level"] == "unvalidated_model"
    print("E2E:", rep["status"], rep["patel"]["code"], "->", rep["decision"]["chosen"],
          "| ranking:", rep["decision"]["ranking"][:3], "| end_stage:", rep["end_stage"],
          f"| {rep['runtime_s']} s")
    import json
    os.makedirs("/home/user/workspace/ecr_cds/examples", exist_ok=True)
    with open("/home/user/workspace/ecr_cds/examples/example_report_3Bp.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)


if __name__ == "__main__":
    test_dicom_roundtrip_500_slices()
    test_patel_recovery()
    test_matrix_complete_and_rules()
    test_rl_loop_learns_within_safe_set()
    test_end_to_end_pipeline_on_dicom()
    print("\nALL TESTS PASSED")
