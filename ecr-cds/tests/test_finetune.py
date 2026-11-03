"""Fine-tuning data pipeline tests (no GPU, no model download). ~1.5 min: one synthetic case is generated."""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecr_cds.agents.criteria import HYPOTHESES  # noqa: E402
from ecr_cds.finetune import dataset as ds  # noqa: E402
from ecr_cds.finetune.evaluate import RulesBackend, evaluate  # noqa: E402


def test_target_differential_is_normalised_and_ranked():
    for lab in ["ECR", "noncarious_cervical_lesion", "internal_resorption", None]:
        for amb in (False, True):
            diff = ds._target_differential(lab, amb, [])
            assert abs(sum(x["probability"] for x in diff) - 1) < 1e-6
            assert {x["diagnosis"] for x in diff} == set(HYPOTHESES)
            top = diff[0]["diagnosis"]
            assert top == (lab or "imaging_artifact"), (lab, top)
            if lab and amb:
                assert diff[0]["probability"] <= 0.61
    print("PASS target differentials normalised, truth ranked first, ambiguous cases softened")


def test_synthetic_case_yields_paired_prompts_with_truth_labels():
    rng = random.Random(1)
    spec = {"morphology": "ecr", "height_mm": 3.0, "spread_deg": 150, "perforate": False, "portal_angle_deg": 0, "augment": "random"}
    exs, meta = ds.generate_case(spec, rng, "t_case", max_rounds=1)
    tasks = [e.task for e in exs]
    assert "radiologist" in tasks and "skeptic" in tasks, tasks
    rad = [e for e in exs if e.task == "radiologist"]
    labelled = [e for e in rad if e.meta["label"] == "ECR"]
    assert labelled, "the true lesion component must carry the ECR label"
    assert all(e.meta["lesion_overlap"].get(str(e.meta["component_id"]), e.meta["lesion_overlap"].get(e.meta["component_id"], 0)) >= 0.5
               for e in labelled)
    for e in exs:   # prompts are exactly what the agents send: system + schema, user JSON, assistant JSON
        assert [m["role"] for m in e.messages] == ["system", "user", "assistant"]
        json.loads(e.messages[1]["content"]); json.loads(e.messages[2]["content"])
        assert "Respond with JSON only" in e.messages[0]["content"]
    sk = [e for e in exs if e.task == "skeptic"][0]
    target = json.loads(sk.messages[2]["content"])["challenges"]
    conf = ds.Confounders(**meta["confounders"])
    for c in conf.challenges():   # every injected confounder appears in the skeptic target
        assert any(c["issue"] == t["issue"] for t in target)
    # split by case and the rules baseline runs end to end
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.jsonl")
        ds.write_jsonl(exs, raw)
        counts = ds.split(raw, os.path.join(td, "split"), eval_fraction=0.5)
        assert counts["train"] + counts["eval"] == len(exs)
        rows = [json.loads(l) for l in open(raw)]
        summ = evaluate(rows, RulesBackend())
        assert summ["radiologist"]["json_valid"] == 1.0
        assert summ["radiologist"]["ecr_recall"] == 1.0, summ
    print(f"PASS synthetic case -> {len(exs)} paired prompts; confounders {meta['confounders']}; rules baseline evaluated")


def test_confirmed_case_converter_on_saved_report():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "ecr_conference_v25", "conference", "conference_report.json")
    if not os.path.exists(path):
        print("SKIP no saved real-case report")
        return
    exs = ds.from_confirmed_case(path, "restoration_margin_or_burnout", weight=2)
    assert len(exs) == 4 and exs[0].meta["source"] == "clinician_confirmed"
    diff = json.loads(exs[0].messages[2]["content"])["differential"]
    assert diff[0]["diagnosis"] == "restoration_margin_or_burnout" and diff[0]["probability"] == 0.85
    try:
        ds.from_confirmed_case(path, "not_a_dx")
        raise AssertionError("invalid label accepted")
    except ValueError:
        pass
    print("PASS confirmed-case converter (label validated, prompts rebuilt from the report)")


if __name__ == "__main__":
    for t in (test_target_differential_is_normalised_and_ranked, test_synthetic_case_yields_paired_prompts_with_truth_labels,
              test_confirmed_case_converter_on_saved_report):
        t()
    print("All fine-tuning pipeline tests passed; no model was trained or validated clinically.")
