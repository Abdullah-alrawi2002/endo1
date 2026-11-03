"""Software tests for the multi-agent case conference (phantom geometry; not clinical validation)."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecr_cds.phantom import make_phantom, VOX  # noqa: E402
from ecr_cds.dicom_loader import CBCTVolume  # noqa: E402
from ecr_cds.segmentation import ToothSegmentation  # noqa: E402
from ecr_cds.agents import (CaseConference, HYPOTHESES, CRITERIA, NullBackend, LLMBackend,  # noqa: E402
                            record_diagnosis_feedback, consensus_markdown, ClinicalFindings)
from ecr_cds.agents.criteria import posterior_from_criteria, evaluate_criteria  # noqa: E402
from ecr_cds.agents.visual_features import CandidateDescriptors  # noqa: E402

SEED = (125, 80, 105)


def _case(h, s, perf, morphology):
    arr, seg, lesion, truth = make_phantom(h, s, perf, morphology=morphology)
    vol = CBCTVolume(arr.astype(np.float32), (VOX, VOX, VOX), (0., 0., 0.), np.eye(3),
                     {"SeriesInstanceUID_hash": f"ph_{morphology}_{h}_{s}"})
    return vol, (lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone)), truth


class ScriptedBackend(LLMBackend):
    """Deterministic stand-in for an LLM: always votes for caries, and raises one challenge."""
    name = "scripted"

    def complete_json(self, system, user, schema, images_png=None):
        if "challenges" in schema["properties"]:
            return {"challenges": [{"issue": "scripted doubt", "severity": "major"}]}
        return {"differential": [{"diagnosis": "caries", "probability": 0.7}, {"diagnosis": "ECR", "probability": 0.3}],
                "key_findings_for": ["x"], "key_findings_against": ["y"], "uncertainties": ["z"]}


def test_ecr_phantom_suggests_ecr_with_extent_and_plan():
    vol, segm, truth = _case(3.0, 150, False, "ecr")
    c = CaseConference(max_rounds=2).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    assert c["status"] == "suggested_ecr_with_plan", c["status"]
    dx = c["suggested_diagnosis"]
    assert dx["leading"] == "ECR" and dx["p_ecr_uncalibrated"] >= 0.5
    assert dx["band"] == "moderate", "band must be capped while landmarks are automatic"
    se = c["suggested_extent"]   # rev 2.9: descriptive measurement summary replaces the Patel code on the default path
    assert (se["extent_descriptor"], se["pulp_relationship"], se["access_descriptor"]) == \
        ("moderate", "dentine_barrier_preserved", "supracrestal_portal"), se
    assert se["is_staging_code"] is False and "suggested_patel" not in c
    qa = {q["id"]: q["answer"] for q in dx["diagnostic_questions"]}
    assert qa["Q1"] == "yes" and qa["Q2"] == "yes" and qa["Q3"] == "yes" and qa["Q6"] == "yes", qa
    assert dx["clinical_correlation"] == "not_assessed"
    plan = c["suggested_plan"]
    assert plan["rule_recommendation"]["primary"] == "external_repair"
    assert plan["policy_decision"]["ranking"][0] in plan["rule_recommendation"]["safe_set"]
    assert any(k["input"] == "pulp_vital" for k in plan["contingencies"])
    assert c["requires_confirmation"] is True and c["clinical_validation"] is False
    # evidence trail is complete and serialisable
    kinds = {e["kind"] for e in c["conference"]["evidence"]}
    assert {"qa", "anatomy", "screen_summary", "candidate", "differential", "challenge", "feature_bundle",
            "extent_summary", "plan"} <= kinds and "patel" not in kinds
    js = json.dumps(c, default=str)
    assert "patel_code" not in js and plan["rule_recommendation"]["cell_scheme"] == "measurement_v1"
    md = consensus_markdown(c)
    assert "Suggested treatment plan" in md and "Conference transcript" in md and "Measured extent summary" in md
    print("PASS ecr phantom ->", dx["statement"][:90], "| extent", se["extent_descriptor"], se["pulp_relationship"])


def test_deep_perforating_ecr_gets_extensive_communication_and_escalation_logic():
    vol, segm, truth = _case(9.0, 150, True, "ecr")
    c = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    assert c["status"] == "suggested_ecr_with_plan"
    se = c["suggested_extent"]
    assert se["extent_descriptor"] == "extensive" and se["pulp_relationship"] == "canal_communication", se
    assert c["suggested_plan"]["end_stage"] is False
    assert "intentional_replantation" in c["suggested_plan"]["irreversible_options_in_safe_set"]
    assert any("Canal communication" in x for x in c["suggested_plan"]["escalation_considerations"])
    assert any("Irreversible" in a for a in c["clinician_actions_required"])
    print("PASS deep perforating ecr -> extensive / canal_communication, irreversible options flagged")


def test_legacy_patel_scheme_opt_in_reproduces_rev28():
    vol, segm, truth = _case(9.0, 150, True, "ecr")
    c = CaseConference(max_rounds=1, staging_scheme="patel_3d").run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    assert c["status"] == "suggested_ecr_with_plan" and c["suggested_patel"]["code"] == "3Bp"
    assert c["suggested_plan"]["cell_scheme"] == "patel_3d" and "suggested_extent" not in c
    print("PASS legacy patel_3d opt-in -> 3Bp reproduced")


def test_nccl_phantom_not_called_ecr():
    vol, segm, truth = _case(1.0, 80, False, "nccl")
    c = CaseConference(max_rounds=2).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    assert c["status"] == "no_ecr_suggested", c["status"]
    assert c["suggested_extent"] is None and c["suggested_plan"] is None
    assert "does not exclude ECR" in c["suggested_diagnosis"]["statement"]
    print("PASS nccl phantom ->", c["suggested_diagnosis"]["leading"])


def test_internal_resorption_phantom_is_not_called_ecr():
    vol, segm, truth = _case(4.0, 150, False, "internal")
    c = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    dx = c["suggested_diagnosis"]
    assert c["status"] == "no_ecr_suggested" and dx["leading"] == "internal_resorption", (c["status"], dx["leading"])
    qa = {q["id"]: q["answer"] for q in dx["diagnostic_questions"]}
    assert qa["Q1"] == "no" and qa["Q2"] == "no", qa
    assert any("root canal treatment" in x for x in dx["suggested_next_steps"])
    print("PASS internal resorption phantom -> internal_resorption; Q1/Q2 no; RCT work-up suggested")


def test_clinical_findings_shift_posterior_and_are_logged():
    vol, segm, truth = _case(3.0, 150, True, "wedge")   # ambiguous open wedge
    base = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    pos = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False,
                                           clinical_findings=ClinicalFindings(pink_spot=True, bleeding_on_probing_defect=True,
                                                                              cervical_defect_hard_irregular=True))
    neg = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False,
                                           clinical_findings=ClinicalFindings(cervical_defect_hard_irregular=False,
                                                                              restoration_at_defect=True))
    p0, pp, pn = (x["suggested_diagnosis"]["p_ecr_uncalibrated"] for x in (base, pos, neg))
    assert pp > p0 > pn, (pp, p0, pn)
    assert pos["suggested_diagnosis"]["clinical_correlation"] == "consistent_with_ecr"
    assert neg["suggested_diagnosis"]["clinical_correlation"] == "inconsistent_with_ecr"
    assert pos["clinical_findings"] == {"pink_spot": True, "bleeding_on_probing_defect": True, "cervical_defect_hard_irregular": True}
    ev = [e for e in pos["conference"]["evidence"] if e["kind"] == "clinical_correlation"]
    assert ev and ev[-1]["basis"] == "clinician"
    print(f"PASS clinical correlation: P(ECR) {pn:.2f} < {p0:.2f} < {pp:.2f}")


def test_intact_tooth_has_no_candidates():
    arr, seg, lesion, truth = make_phantom(1.0, 60, False, morphology="ecr")
    arr = arr.copy(); arr[lesion] = 1600 + np.random.default_rng(0).normal(0, 60, int(lesion.sum()))  # heal it
    vol = CBCTVolume(arr.astype(np.float32), (VOX, VOX, VOX), (0., 0., 0.), np.eye(3), {"SeriesInstanceUID_hash": "intact"})
    c = CaseConference(max_rounds=1).run(vol, lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone),
                                         SEED, "phantom_truth", render_tiles=False)
    assert c["status"] in ("no_candidates", "no_ecr_suggested"), c["status"]
    assert c["suggested_plan"] is None
    print("PASS intact tooth ->", c["status"])


def test_llm_backend_is_second_opinion_not_override():
    vol, segm, truth = _case(3.0, 150, False, "ecr")
    c = CaseConference(backend=ScriptedBackend(), max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    dx = c["suggested_diagnosis"]
    ev = [e for e in c["conference"]["evidence"] if e["kind"] == "differential" and e["refers_to"] == dx["candidate_id"]][-1]
    assert ev["content"]["llm_posterior"] is not None and ev["content"]["rule_posterior"] is not None
    assert ev["basis"] == "llm+rule"
    # the model's caries vote is averaged with the rules, so P(ECR) drops but the trail shows both
    assert ev["content"]["rule_posterior"]["ECR"] > ev["content"]["posterior"]["ECR"]
    # model challenges are never recorded as 'major'
    chs = [ch for e in c["conference"]["evidence"] if e["kind"] == "challenge" for ch in e["content"]["challenges"]]
    assert all(ch["severity"] != "major" for ch in chs if ch["issue"].startswith("[model]"))
    assert dx["band"] != "high"
    print("PASS scripted backend logged as second opinion; P(ECR) rule", ev["content"]["rule_posterior"]["ECR"],
          "-> combined", ev["content"]["posterior"]["ECR"])


def test_criteria_math_and_feedback_learning():
    truth = [None] * len(CRITERIA)
    p = posterior_from_criteria(truth)
    assert abs(sum(p.values()) - 1) < 1e-6 and set(p) == set(HYPOTHESES)
    # feedback: fewer than min_cases -> no change; enough cases -> ECR column moves toward labels
    from ecr_cds.agents.roles import RadiologistAgent
    ra = RadiologistAgent()
    W0 = ra.W.copy()
    fake = [{"criteria": [True] + [None] * (len(CRITERIA) - 1), "ecr": True} for _ in range(5)]
    r = ra.update_weights_from_feedback(fake, min_cases=20)
    assert r["updated"] is False and np.array_equal(ra.W, W0)
    fake = ([{"criteria": [True] + [None] * (len(CRITERIA) - 1), "ecr": True} for _ in range(15)] +
            [{"criteria": [False] + [None] * (len(CRITERIA) - 1), "ecr": False} for _ in range(15)])
    r = ra.update_weights_from_feedback(fake, min_cases=20)
    j = HYPOTHESES.index("ECR")
    assert r["updated"] and ra.W[0, 0, j] > W0[0, 0, j] and ra.W[0, 1, j] < W0[0, 1, j]
    assert np.array_equal(ra.W[1:], W0[1:]) or np.allclose(ra.W[1:], W0[1:])
    print("PASS criteria posterior + gated feedback learning")


def test_record_diagnosis_feedback_idempotent():
    vol, segm, truth = _case(3.0, 150, False, "ecr")
    c = CaseConference(max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "dx_feedback.json")
        r1 = record_diagnosis_feedback(log, c, "ECR")
        r2 = record_diagnosis_feedback(log, c, "caries")   # correction overwrites the same case/candidate
        assert r1["n_labelled"] == 1 and r2["n_labelled"] == 1
        assert json.load(open(log))[0]["confirmed_diagnosis"] == "caries"
        try:
            record_diagnosis_feedback(log, c, "unicorn"); raise AssertionError("bad label accepted")
        except ValueError:
            pass
    print("PASS diagnosis feedback log idempotent and validated")


if __name__ == "__main__":
    for t in [test_criteria_math_and_feedback_learning, test_ecr_phantom_suggests_ecr_with_extent_and_plan,
              test_deep_perforating_ecr_gets_extensive_communication_and_escalation_logic,
              test_legacy_patel_scheme_opt_in_reproduces_rev28, test_nccl_phantom_not_called_ecr,
              test_intact_tooth_has_no_candidates, test_llm_backend_is_second_opinion_not_override,
              test_internal_resorption_phantom_is_not_called_ecr, test_clinical_findings_shift_posterior_and_are_logged,
              test_record_diagnosis_feedback_idempotent]:
        t()
    print("\nAll multi-agent tests passed (software behaviour on phantoms; not clinical validation).")
