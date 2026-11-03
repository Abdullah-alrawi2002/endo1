"""Rev 2.8 tests: local Nemotron backend (semantic-entropy abstention, thinking-tag stripping, single-image payload),
counterfactual grounding, TTA verifier, unsharp display, evaluation metrics.  Software behaviour only.

Run:  python tests/test_nemotron_backend.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fake_vlm_server as fake  # noqa: E402
from ecr_cds.agents import CaseConference  # noqa: E402
from ecr_cds.agents.llm_backend import (NemotronLocalBackend, auto_backend, default_cluster_key,  # noqa: E402
                                        discrete_semantic_entropy, _strip_thinking, _extract_json)
from ecr_cds.dicom_loader import CBCTVolume  # noqa: E402
from ecr_cds.evaluation import auroc, perception_metrics  # noqa: E402
from ecr_cds.perception.display import unsharp2d  # noqa: E402
from ecr_cds.phantom import VOX, make_phantom  # noqa: E402
from ecr_cds.segmentation import ToothSegmentation  # noqa: E402

SRV, URL = fake.start()
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        raise AssertionError(name)


def test_entropy_and_parsing():
    e = discrete_semantic_entropy(["a"] * 5)
    check("dse zero when all agree", e["dse"] == 0.0 and e["n_clusters"] == 1)
    e = discrete_semantic_entropy(["a", "b", "c", "d", "e"])
    check("dse maximal when all differ", abs(e["dse"] - e["dse_max"]) < 1e-6 and e["n_clusters"] == 5)
    k = default_cluster_key({"differential": [{"diagnosis": "ECR", "probability": 0.7}, {"diagnosis": "caries", "probability": 0.3}],
                             "question_answers": {"Q2": "yes", "Q1": "no"}})
    check("cluster key uses leading dx + sorted answers", k == "ECR|Q1=no|Q2=yes", k)
    check("thinking tags stripped", _strip_thinking("<think>hmm\nmore</think> {\"a\":1}") == '{"a":1}')
    check("json extracted from fenced / noisy output", _extract_json("```json\n{\"a\": 2}\n```") == {"a": 2}
          and _extract_json("Sure: {\"a\": 3} done") == {"a": 3})


def test_backend_probe_and_sampling():
    fake.STATE.update(mode="agree", requests=[], think=True)
    be = NemotronLocalBackend(base_url=URL, n_samples=5, dse_threshold=0.3, enable_thinking=True)
    check("model id discovered from /v1/models", be.model == "nemotron-3.5-bolt-74b-a9b-24gb", be.model)
    out = be.complete_json("sys", "user", {"type": "object"}, images_png=[b"\x89PNG fake"])
    check("5 samples -> low entropy, not abstained", out["_dse"]["n_samples"] == 5 and out["_dse"]["dse"] == 0.0 and not out["_abstain"])
    req = fake.STATE["requests"][-1]
    check("chat_template_kwargs.enable_thinking sent", req["chat_template_kwargs"]["enable_thinking"] is True)
    check("temperature 1.0 / top_p 0.95 for sampling", req["temperature"] == 1.0 and req["top_p"] == 0.95)
    parts = req["messages"][1]["content"]
    check("exactly one image attached per request", sum(1 for p in parts if p["type"] == "image_url") == 1)
    check("thinking text did not break JSON", out["differential"][0]["diagnosis"] == "ECR")
    fake.STATE.update(mode="disagree", requests=[], think=False)
    out = be.complete_json("sys", "user", {"type": "object"})
    check("disagreeing samples -> abstain", out["_abstain"] and out["_dse"]["n_clusters"] >= 3, out["_dse"])
    be1 = NemotronLocalBackend(base_url=URL, n_samples=1)
    be1.complete_json("sys", "user", {"type": "object"})
    check("single-sample mode uses low temperature", fake.STATE["requests"][-1]["temperature"] <= 0.1)
    try:
        NemotronLocalBackend(base_url="http://127.0.0.1:9/v1")
        check("unreachable server raises", False)
    except RuntimeError as e:
        check("unreachable server raises", "not reachable" in str(e))
    os.environ["NEMOTRON_BASE_URL"] = URL
    try:
        check("auto_backend('nemotron') resolves", auto_backend("nemotron").name.startswith("nemotron_local"))
    finally:
        del os.environ["NEMOTRON_BASE_URL"]


def _phantom_case():
    vol, seg, lesion, truth = make_phantom(3.0, 150, False, morphology="ecr")
    cb = CBCTVolume(vol.astype(np.float32), (VOX, VOX, VOX), (0, 0, 0), np.eye(3), {"SeriesInstanceUID_hash": "t28"})
    segm = lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone)  # noqa: E731
    return cb, seg, lesion, segm


def test_conference_fusion_and_grounding():
    from ecr_cds.finetune.dataset import TEST_SEED
    cb, seg, lesion, segm = _phantom_case()
    # (a) rules-only reference
    cc0 = CaseConference(max_rounds=1)
    c0 = cc0.run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, verify_mode="tta", case_id="t28")
    ver = c0["verification"]
    check("tta verifier reports per-path dice", all(v.get("mode") == "tta" and v.get("dice_by_path") for v in ver.values()), list(ver.values())[0].get("dice_by_path"))
    pm = perception_metrics(cc0.last_blackboard, lesion, consensus=c0)
    check("perception metrics include budget-1 review proxies", "budget1_true_first_by_reproducibility" in pm and pm["candidates"])
    check("auroc helper", auroc([0.9, 0.8], [0.1, 0.5]) == 1.0 and auroc([], [0.1]) is None)
    p0 = c0["suggested_diagnosis"]["p_ecr_uncalibrated"]
    # (b) agreeing VLM: fused posterior moves toward the model, weight 0.5
    fake.STATE.update(mode="agree", requests=[])
    be = NemotronLocalBackend(base_url=URL, n_samples=3)
    cc = CaseConference(backend=be, max_rounds=1)
    c1 = cc.run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, case_id="t28")
    rows = [r for r in c1["candidates"] if r.get("llm_meta")]
    check("model opinion recorded with weight 0.5", rows and all(abs(r["llm_weight"] - 0.5) < 1e-6 for r in rows), [r.get("llm_weight") for r in rows])
    check("evidence frame (not tiles) sent to the model", any(int(p["image_url"]["url"].strip("<>").split()[0]) > 20000 for q in fake.STATE["requests"]
                                                               for p in q["messages"][1]["content"] if p["type"] == "image_url"))
    check("settings recorded in architecture", c1["architecture"]["settings"]["backend"].startswith("nemotron_local"))
    # (c) disagreeing VLM: abstains, posterior equals rules-only
    fake.STATE.update(mode="disagree", requests=[])
    c2 = CaseConference(backend=be, max_rounds=1).run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, case_id="t28")
    rows = [r for r in c2["candidates"] if r.get("llm_meta")]
    check("abstention -> weight 0 and rule posterior kept", rows and all(r["llm_weight"] == 0.0 and r["llm_meta"]["abstained"] for r in rows)
          and abs(c2["suggested_diagnosis"]["p_ecr_uncalibrated"] - p0) < 1e-6)
    # (d) blind VLM fails the counterfactual test -> weight capped at 0.2
    fake.STATE.update(mode="blind", requests=[])
    cc3 = CaseConference(backend=be, max_rounds=1)
    c3 = cc3.run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, counterfactual_check=True, case_id="t28")
    rows = [r for r in c3["candidates"] if r.get("llm_meta")]
    g = [r["llm_meta"]["grounding"] for r in rows]
    check("counterfactual frame rendered and sent", all(x and x.get("verdict") == "weak" for x in g), g)
    check("weak grounding caps weight at 0.2", all(abs(r["llm_weight"] - 0.2) < 1e-6 for r in rows))
    arrs = cc3.last_blackboard.arrays
    check("counterfactual frames stored separately from evidence frames",
          set(arrs.get("cf_frames_png", {})) == set(arrs.get("frames_png", {})) and
          all(arrs["cf_frames_png"][k] != arrs["frames_png"][k] for k in arrs["frames_png"]))
    check("no counterfactual frame in rev-2.7 path", "cf_frames_png" not in cc.last_blackboard.arrays)
    # (e) responsive VLM passes the test (keys on counterfactual image lengths)
    fake.STATE.update(mode="responsive", requests=[])
    cc5 = CaseConference(backend=be, max_rounds=1)
    # first pass to learn the counterfactual image lengths
    cc5.run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, counterfactual_check=True, case_id="t28")
    cf_pngs = cc5.last_blackboard.arrays.get("cf_frames_png", {})
    import base64
    fake.STATE["cf_lengths"] = {len("data:image/png;base64," + base64.b64encode(p).decode()) for p in cf_pngs.values()}
    fake.STATE["requests"] = []
    c5 = cc5.run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, counterfactual_check=True, case_id="t28")
    rows = [r for r in c5["candidates"] if r.get("llm_meta")]
    check("responsive model keeps full weight", rows and all(r["llm_meta"]["grounding"]["verdict"] == "responsive" and abs(r["llm_weight"] - 0.5) < 1e-6
                                                             for r in rows), [r["llm_meta"]["grounding"] for r in rows])
    md = __import__("ecr_cds.agents.orchestrator", fromlist=["consensus_markdown"]).consensus_markdown(c5)
    check("summary has VLM second-opinion section", "Vision-language second opinion" in md and "grounding test" in md)
    # (f) sharpen display is recorded in the frame provenance and is display-only
    c6 = CaseConference(max_rounds=1).run(cb, segm, TEST_SEED, "phantom_truth", render_tiles=False, verify=True, sharpen_display=True, case_id="t28")
    check("unsharp display recorded in panel provenance", c6["architecture"]["settings"]["sharpen_display"] is True and
          all(abs(a - b) < 1e-6 for a, b in zip([v["reproducibility"] for v in c6["verification"].values()],
                                                [v["reproducibility"] for v in c1["verification"].values()])))
    img = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    out = unsharp2d(img)
    check("unsharp2d keeps [0,1] range and shape", out.shape == img.shape and out.min() >= 0 and out.max() <= 1)


if __name__ == "__main__":
    test_entropy_and_parsing()
    test_backend_probe_and_sampling()
    test_conference_fusion_and_grounding()
    SRV.shutdown()
    print(f"\nAll {len(RESULTS)} rev-2.8 tests passed (software behaviour with a scripted stand-in server; not clinical validation).")
