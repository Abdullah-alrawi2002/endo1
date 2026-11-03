"""Tests for rev 2.7: Diagnocat ingestion, context -> priors, perception tools, knowledge/CDT, layered evaluation."""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ecr_cds.diagnocat import (DiagnocatStudy, DiagnocatTooth, DiagnocatCondition, load_diagnocat_json, stl_to_mask,  # noqa: E402
                               build_contexts, prior_from_shifts)
from ecr_cds.diagnocat.adapter import parse_report_text, DiagnocatClient  # noqa: E402
from ecr_cds.diagnocat.schema import normalise_condition_name  # noqa: E402
from ecr_cds.agents.chart import PatientChart, HistoryEvent, universal_to_iso  # noqa: E402
from ecr_cds.agents.criteria import PRIOR, HYPOTHESES  # noqa: E402
from ecr_cds.knowledge import KnowledgeBase, cdt_codes_for, tooth_type_from_iso  # noqa: E402
from ecr_cds.perception import clahe2d, metal_mask, tissue_reference_levels, PerceptionToolkit  # noqa: E402


def test_condition_aliases_and_no_ecr_class():
    assert normalise_condition_name("Internal resorption") == "internal_resorption"
    assert normalise_condition_name("Periodontal bone loss severe") == "bone_loss"
    assert normalise_condition_name("Artificial crown") == "crown"
    assert normalise_condition_name("something new").startswith("unknown:")
    # an "external resorption" label can only come from a clinician in Diagnocat
    assert normalise_condition_name("External cervical resorption") == "external_resorption_flag"


def test_json_load_with_field_mapping_and_percent_probabilities():
    data = {"studyId": "S1", "tooth_list": [{"toothNumber": 26, "findings": [{"type": "Internal resorption", "confidence": 72},
                                                                              {"type": "Endodontically treated tooth", "confidence": 0.9, "origin": "doctor"}]}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
    st = load_diagnocat_json(f.name)
    assert st.study_id == "S1" and st.source == "json_export"
    t = st.tooth(26)
    assert t.has("internal_resorption").probability == 0.72
    assert t.has("endo_treated").source == "clinician"
    json.dumps(st.to_dict())


def test_pdf_text_parser_blocks():
    st = parse_report_text("Diagnocat AI 4.0\nTooth 26\nInternal resorption 72%\nArtificial crown 95%\nTooth 11\nCaries signs 0.61\n"
                           "Periodontal bone loss severe")
    assert st.diagnocat_version == "4.0"
    assert {t.iso_number for t in st.teeth} == {11, 26}
    assert st.tooth(26).has("internal_resorption").probability == 0.72
    assert st.tooth(11).has("bone_loss").severity == "severe"


def test_client_unconfigured_is_explicit():
    c = DiagnocatClient(base_url=None, api_key=None)
    assert not c.available
    try:
        c.process("x.zip")
        assert False
    except RuntimeError as e:
        assert "not configured" in str(e)


def test_context_shifts_modifiers_and_triage():
    st = DiagnocatStudy("m", [
        DiagnocatTooth(26, [DiagnocatCondition("internal_resorption", "Internal resorption", 0.7),
                            DiagnocatCondition("crown", "Artificial crown", 0.95),
                            DiagnocatCondition("endo_treated", "Endodontically treated tooth", 0.9)]),
        DiagnocatTooth(11, [DiagnocatCondition("caries_signs", "Caries signs", 0.6)]),
        DiagnocatTooth(36, [DiagnocatCondition("missing", "Missing tooth", 0.99)]),
    ], "mock")
    chart = PatientChart(age_years=34, complaint_quadrants=[2], symptoms=["pink spot"], history=[HistoryEvent("orthodontics", None, 10)])
    ctxs = build_contexts(st, chart)
    assert [c.iso_number for c in ctxs] == [26, 11]          # missing tooth dropped; complaint quadrant first
    c26 = ctxs[0]
    assert c26.tooth_type == "molar" and c26.modifier_hints["multirooted"] is True
    assert c26.modifier_hints["pulp_vital"] is False and c26.canal_hint == "hyperdense_filling"
    assert c26.prior_shifts["internal_resorption"] > 1.0 and c26.prior_shifts["ECR"] == 0.5   # ortho history only
    assert c26.clinical_findings_hints["pink_spot"] is True
    pr = prior_from_shifts(c26.prior_shifts)
    assert abs(sum(pr.values()) - 1) < 1e-3 and pr["internal_resorption"] > PRIOR["internal_resorption"]
    # AI-labelled "external resorption" never shifts ECR
    st2 = DiagnocatStudy("m2", [DiagnocatTooth(21, [DiagnocatCondition("external_resorption_flag", "External resorption", 0.8, "ai")])], "mock")
    assert "ECR" not in build_contexts(st2)[0].prior_shifts
    st3 = DiagnocatStudy("m3", [DiagnocatTooth(21, [DiagnocatCondition("external_resorption_flag", "External resorption", None, "clinician")])], "mock")
    assert build_contexts(st3)[0].prior_shifts["ECR"] == 1.0


def test_chart_retrieval_and_numbering():
    assert universal_to_iso(3) == 16 and universal_to_iso(19) == 36 and universal_to_iso(30) == 46
    ch = PatientChart(age_years=15, complaint_teeth_iso=[36], symptoms=["cold sensitivity lower left"],
                      history=[HistoryEvent("rct", 36, 2), HistoryEvent("trauma", None, 5)])
    r = ch.retrieve(36)
    assert r["in_complaint_focus"] and r["modifier_hints"]["symptomatic"] is True
    assert r["modifier_hints"]["pulp_vital"] is False and r["modifier_hints"]["growing_patient"] is True
    assert r["canal_hint"] == "hyperdense_filling" and r["prior_shifts"]["ECR"] == 0.5
    assert ch.retrieve(11)["events"] == [{"event": "trauma", "tooth_iso": None, "years_ago": 5, "note": ""}]
    assert PatientChart.from_dict(ch.to_dict()).retrieve(36)["priority"] == r["priority"]


def test_cdt_mapping_and_knowledge_tags():
    assert tooth_type_from_iso(26) == "molar" and tooth_type_from_iso(14) == "premolar" and tooth_type_from_iso(11) == "anterior"
    codes = cdt_codes_for("external_repair_rct", "molar")
    assert [c["code"] for c in codes] == ["D3473", "D3330"]
    assert cdt_codes_for("extraction")[0]["code"] == "D7140"
    kb = KnowledgeBase()
    assert kb.for_tags(["Q7"]) and kb.for_tags(["diagnocat"]) and all(s.get("url") for s in kb.statements)


def test_stl_voxelise_cube():
    # binary STL of a 2 mm cube at (1..3) mm
    v = np.array([[1, 1, 1], [3, 1, 1], [3, 3, 1], [1, 3, 1], [1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3]], float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    with tempfile.NamedTemporaryFile("wb", suffix=".stl", delete=False) as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
        for a, b, c in faces:
            f.write(struct.pack("<3f", 0, 0, 0) + v[a].astype("<f4").tobytes() + v[b].astype("<f4").tobytes() + v[c].astype("<f4").tobytes() + b"\0\0")
    m = stl_to_mask(f.name, (20, 20, 20), (0.25, 0.25, 0.25))
    vol_mm3 = m.sum() * 0.25 ** 3
    assert 7.0 < vol_mm3 < 12.5, vol_mm3           # 8 mm3 cube; surface sampling adds up to half a voxel per side
    assert m[8, 8, 8] and not m[2, 2, 2]


def test_display_transforms_relative_and_clahe():
    rng = np.random.default_rng(0)
    raw = rng.normal(300, 20, (10, 40, 40)).astype(np.float32)
    tooth = np.zeros_like(raw, bool); tooth[:, 10:30, 10:30] = True
    raw[tooth] = 1600; raw[:, 12:14, 12:14] = 5000     # dentine; metal-level
    canal = np.zeros_like(tooth); canal[:, 18:22, 18:22] = True; raw[canal] = 300
    lv = tissue_reference_levels(raw, tooth, canal, t_dense=2000)
    assert lv["soft_tissue"] < lv["dentine_mode"] < lv["metal_level"] and "not HU" not in lv["method"]
    mm = metal_mask(raw, lv["metal_level"], (0.25, 0.25, 0.25))
    assert mm["metal_core"].sum() > 0 and mm["streak_zone"].sum() > mm["metal_core"].sum()
    out = clahe2d(np.clip((raw[0] - 200) / 2500, 0, 1))
    assert out.shape == raw[0].shape and 0 <= out.min() and out.max() <= 1


def test_toolkit_logs_calls():
    tk = PerceptionToolkit()
    assert {"segment_teeth", "propose_regions", "reproduce_candidate", "evidence_frame"} <= set(tk.names())
    raw = np.full((6, 12, 12), 1500, np.float32); tooth = np.ones_like(raw, bool); canal = np.zeros_like(tooth)
    from ecr_cds.segmentation import ToothSegmentation
    tk.call("reference_levels", raw=raw, seg=ToothSegmentation(tooth, canal, ~tooth), t_dense=2000)
    assert tk.log[0].tool == "reference_levels" and "dentine_mode" in tk.log[0].summary


def test_conference_with_context_verifier_and_knowledge():
    from ecr_cds.phantom import make_phantom, VOX
    from ecr_cds.dicom_loader import CBCTVolume
    from ecr_cds.segmentation import ToothSegmentation
    from ecr_cds.agents import CaseConference, consensus_markdown
    arr, seg, lesion, truth = make_phantom(3.0, 150, False, morphology="ecr")
    vol = CBCTVolume(arr.astype(np.float32), (VOX, VOX, VOX), (0., 0., 0.), np.eye(3), {"SeriesInstanceUID_hash": "ctx_test"})
    segm = lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone)  # noqa: E731
    st = DiagnocatStudy("m", [DiagnocatTooth(11, [DiagnocatCondition("filling", "Filling", 0.6)])], "mock")
    ctx = build_contexts(st, PatientChart(age_years=30, complaint_teeth_iso=[11], history=[HistoryEvent("orthodontics", None, 8)]))[0]
    cc = CaseConference(max_rounds=1)
    c = cc.run(vol, segm, (125, 80, 105), "phantom_truth", render_tiles=False, context=ctx, examined_teeth=[11], focus_teeth=[11])
    assert c["status"] == "suggested_ecr_with_plan"
    assert c["prior_used"] != "default" and c["suggested_diagnosis"]["leading"] == "ECR"
    assert all(v["verdict"] in ("reproduced", "partially_reproduced", "not_reproduced") for v in c["verification"].values())
    assert any(v["verdict"] == "reproduced" for v in c["verification"].values())
    assert c["coverage"]["unexamined_focus_teeth"] == [] and {"evidence_frame", "reproduce_candidate"} <= set(c["coverage"]["tools_used"])
    assert c["suggested_plan"]["cdt_codes"]["external_repair"][0]["code"] == "D3471"
    assert c["knowledge"]["question_citations"]["Q7"]
    assert c["modifiers_used"]["restorable"] is True and c["clinical_findings"]["history_trauma_ortho_bleaching"] is True
    kinds = {e["kind"] for e in c["conference"]["evidence"]}
    assert {"context", "verification", "coverage", "knowledge"} <= kinds
    md = consensus_markdown(c)
    assert "Critic re-inspection" in md and "CDT 2024 codes" in md and "Guideline statements cited" in md
    json.dumps(c, default=str)
    with tempfile.TemporaryDirectory() as d:
        cc.save_outputs(c, d, volume=vol)
        assert any(f.endswith("_evidence_frame.png") for f in os.listdir(d))


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\nAll {len(fns)} rev-2.7 tests passed (software behaviour; not clinical validation).")
