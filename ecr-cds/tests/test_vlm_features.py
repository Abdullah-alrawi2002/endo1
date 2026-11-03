"""Rev 2.9 tests: VLM feature-measurement layer (schema validation, geometric grounding, flagging of unverifiable model
values, measurement-based extent summary + safe-set mapping, conference end-to-end with a fake local VLM server).

Software behaviour on synthetic phantoms only; not clinical validation.  No Patel codes are produced on this path and
the tests assert that.

Run:  python tests/test_vlm_features.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fake_vlm_server as fake  # noqa: E402
from ecr_cds.agents import CaseConference, NullBackend, consensus_markdown  # noqa: E402
from ecr_cds.agents.feature_measurement import bundle_feature_notes  # noqa: E402
from ecr_cds.agents.llm_backend import NemotronLocalBackend  # noqa: E402
from ecr_cds.agents.roles import feature_logodds  # noqa: E402
from ecr_cds.decision_matrix import (SCHEME_MEASUREMENT, Treatment, ClinicalModifiers, measurement_matrix_rows,  # noqa: E402
                                     recommend_from_measurements)
from ecr_cds.dicom_loader import CBCTVolume  # noqa: E402
from ecr_cds.extent_summary import ACCESS_LEVELS, EXTENT_LEVELS, PULP_LEVELS, summarise_extent  # noqa: E402
from ecr_cds.perception.feature_schema import (MEASUREMENT_UNITS, SOURCE_VLM_UNVERIFIED, SOURCE_VOXEL, VISUAL_ENUMS,  # noqa: E402
                                               FeatureSchemaError, validate_bundle_dict, validate_vlm_response)
from ecr_cds.perception.vlm_features import (FEATURE_SYSTEM_PROMPT, measure_candidate_features,  # noqa: E402
                                             measure_candidate_geometry, query_vlm_features)
from ecr_cds.phantom import VOX, make_phantom  # noqa: E402
from ecr_cds.rl_policy import ClinicalContext, ConstrainedBanditPolicy  # noqa: E402
from ecr_cds.segmentation import ToothSegmentation  # noqa: E402

SRV, URL = fake.start()
RESULTS = []
PATEL_CODE = re.compile(r'"[1-4][A-D][dp]"')
SEED = (125, 80, 105)


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        raise AssertionError(f"{name}: {detail}")


# --------------------------------------------------------------------------- helpers
def phantom_case(h, s, perf, morphology="ecr"):
    """Phantom with ground-truth lesion mask as the candidate labels and a geometry frame built from the truth
    (axis = +z, CEJ at truth['cej_mm'])."""
    arr, seg, lesion, truth = make_phantom(h, s, perf, morphology=morphology)
    labels = lesion.astype(np.int32)
    geometry = {"centre_mm": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "cej_t_mm": truth["cej_mm"],
                "source": "reviewed_points", "flags": []}
    return arr, seg, labels, truth, geometry


def _backend(n_samples=3):
    return NemotronLocalBackend(base_url=URL, n_samples=n_samples, dse_threshold=0.3)


# --------------------------------------------------------------------------- schema
def test_schema_validation_and_coercion():
    good = {"visual_features": {k: v[0] for k, v in VISUAL_ENUMS.items()},
            "estimates": {"long_axis_extent_mm": 3.2, "root_canal_communication": "no", "dentine_volume_fraction_involved": 0.1},
            "lesion_visible_in_image": "yes", "panels_used": ["axial_1"], "description": "x"}
    clean, errors = validate_vlm_response(good)
    check("well-formed answer validates without errors", errors == [] and clean["estimates"]["root_canal_communication"] is False, errors)
    bad = {"visual_features": {"radiolucency_pattern": "Swiss cheese", "metal_beam_hardening": "LOTS"},
           "estimates": {"long_axis_extent_mm": "about three", "dentine_volume_fraction_involved": 7.0, "mesio_distal_extent_mm": -2.0,
                         "root_canal_communication": "maybe", "patel_height": 3, "min_distance_to_canal_mm": float("nan")},
           "lesion_visible_in_image": "sort of"}
    clean, errors = validate_vlm_response(bad)
    vf = clean["visual_features"]
    check("enum values outside the schema fall back to the uncertain-type value",
          vf["radiolucency_pattern"] == "uncertain" and vf["metal_beam_hardening"] == "uncertain"
          and all(vf[k] in VISUAL_ENUMS[k] for k in VISUAL_ENUMS))
    est = clean["estimates"]
    check("non-numeric -> null, fraction clipped, negative length -> null, 'maybe' -> None, NaN -> null",
          est["long_axis_extent_mm"] is None and est["dentine_volume_fraction_involved"] == 1.0
          and est["mesio_distal_extent_mm"] is None and est["root_canal_communication"] is None
          and est["min_distance_to_canal_mm"] is None, est)
    check("unknown estimate keys (e.g. a staging field) are dropped and reported",
          "patel_height" not in est and any("patel_height" in e for e in errors), errors)
    check("every coercion is recorded", len(errors) >= 7, len(errors))
    check("lesion_visible coerced", clean["lesion_visible_in_image"] == "uncertain")
    try:
        validate_vlm_response({"estimates": {}})
        check("answer without visual_features raises FeatureSchemaError", False)
    except FeatureSchemaError:
        check("answer without visual_features raises FeatureSchemaError", True)
    try:
        validate_vlm_response(["not", "an", "object"])
        check("non-object answer raises FeatureSchemaError", False)
    except FeatureSchemaError:
        check("non-object answer raises FeatureSchemaError", True)
    low = FEATURE_SYSTEM_PROMPT.lower()
    check("feature prompt forbids staging / diagnosis / treatment and names no classification system",
          "do not stage" in low and "do not diagnose" in low and "patel" not in low and "heithersay" not in low)


# --------------------------------------------------------------------------- geometry
def test_geometric_measurements_match_phantom_truth():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    t0 = time.time()
    geom = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), geometry)
    dt = time.time() - t0
    check("all schema measurements present", set(geom) == set(MEASUREMENT_UNITS), set(MEASUREMENT_UNITS) - set(geom))
    check("geometry per candidate is fast enough for the conference (< 10 s)", dt < 10, f"{dt:.1f}s")
    la = geom["long_axis_extent_mm"].value
    check("long-axis extent ~ planted height + 0.5 mm coronal lip (3.5 mm)", abs(la - 3.5) < 0.4, la)
    check("apical limit from CEJ ~ planted 3.0 mm", abs(geom["apical_limit_from_cej_mm"].value - 3.0) < 0.3,
          geom["apical_limit_from_cej_mm"].value)
    check("coronal limit from CEJ is coronal (negative, ~ -0.5 mm)", -0.8 < geom["coronal_limit_from_cej_mm"].value < 0.0,
          geom["coronal_limit_from_cej_mm"].value)
    crest = geom["apical_limit_from_crest_mm"]
    check("apical limit from crest ~ height - 1.5 mm (crest 1.5 mm below CEJ)", crest.value is not None and abs(crest.value - 1.5) < 0.5,
          crest.value)
    circ = geom["circumferential_extent_deg"].value
    check("circumferential extent ~ planted 150 deg", abs(circ - 150) <= 15, circ)
    check("non-perforating lesion: canal not communicating, distance > 0",
          geom["root_canal_communication"].value is False and geom["min_distance_to_canal_mm"].value > 0.2,
          geom["min_distance_to_canal_mm"].value)
    check("all values are voxel-geometry sourced and verified",
          all(m.source == SOURCE_VOXEL and m.verified for m in geom.values()))
    check("reviewed CEJ -> CEJ-referenced confidence not capped at 0.5", geom["apical_limit_from_cej_mm"].confidence > 0.5)
    check("provenance carries slice ranges and axis", "volume_slice_range_z" in crest.provenance and "tooth_axis_zyx" in crest.provenance)
    # perforating, deep lesion
    arr, seg, labels, truth, geometry = phantom_case(9.0, 200, True)
    geom = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), geometry)
    check("perforating phantom: canal communication True and distance 0",
          geom["root_canal_communication"].value is True and geom["min_distance_to_canal_mm"].value == 0.0)
    check("deep lesion long axis ~ 9.5 mm", abs(geom["long_axis_extent_mm"].value - 9.5) < 0.6, geom["long_axis_extent_mm"].value)
    frac = geom["dentine_volume_fraction_involved"].value
    check("dentine fraction is a sane fraction", 0.0 < frac < 0.5, frac)
    # automatic (unreviewed) landmarks lower confidence; crest-offset assumption lowers it further
    geometry2 = {**geometry, "source": "automatic_unreviewed", "flags": ["cej_uses_unverified_bone_crest_offset"]}
    geom2 = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), geometry2)
    check("unreviewed CEJ from crest offset -> confidence <= 0.3 and flagged",
          geom2["apical_limit_from_cej_mm"].confidence <= 0.3 and "cej_is_bone_crest_minus_1.5mm_assumption" in geom2["apical_limit_from_cej_mm"].flags)
    # no CEJ at all -> unavailable, never guessed
    geom3 = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), {**geometry, "cej_t_mm": None})
    check("missing CEJ -> CEJ-referenced measurements unavailable (not guessed)",
          geom3["apical_limit_from_cej_mm"].value is None and geom3["apical_limit_from_cej_mm"].source == "unavailable")
    # in-plane frame: with a buccal direction the BL / MD axes are patient-defined
    geom4 = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, buccal_direction=(0, 1, 0))
    check("buccal direction given -> patient frame recorded and in-plane confidence not capped",
          geom4["bucco_lingual_extent_mm"].provenance["axis_definition"] == "patient_buccal_direction"
          and geom4["bucco_lingual_extent_mm"].confidence > 0.5)
    check("no buccal direction -> in-plane extents flagged as proxy axes",
          geom["bucco_lingual_extent_mm"].provenance["axis_definition"] != "patient_buccal_direction"
          and geom["bucco_lingual_extent_mm"].confidence <= 0.5)


# --------------------------------------------------------------------------- VLM + reconciliation
def test_geometry_only_bundle_without_model():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=NullBackend(), source_id="ph")
    check("NullBackend -> geometry-only bundle, visual not assessed",
          b.vlm["status"] == "not_configured" and b.visual.assessed is False
          and "no_vision_model_configured_geometry_only" in b.caveats)
    d = b.to_dict()
    check("serialised bundle passes structural validation", validate_bundle_dict(d) == [], validate_bundle_dict(d))
    js = json.dumps(d)
    check("bundle JSON contains no staging codes", not PATEL_CODE.search(js) and "patel" not in js.lower())


def test_vlm_agreeing_values_are_verified_and_disagreeing_values_flagged():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    geom = measure_candidate_geometry(arr, seg, labels, 1, (VOX, VOX, VOX), geometry)
    # model echoes numbers close to the geometry (as a careful reader would)
    fake.STATE.update(feature_mode="agree", feature_requests=[], requests=[],
                      feature_estimates={"long_axis_extent_mm": round(geom["long_axis_extent_mm"].value + 0.3, 1),
                                         "mesio_distal_extent_mm": round(geom["mesio_distal_extent_mm"].value, 1),
                                         "root_canal_communication": "no",
                                         "min_distance_to_canal_mm": round(geom["min_distance_to_canal_mm"].value, 1),
                                         "dentine_volume_fraction_involved": None})
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=_backend(), frame_png=b"\x89PNG fake",
                                   frame_meta={"panels": [{"name": "axial_1", "plane": "axial", "index": 120}]}, source_id="ph")
    check("VLM answered and visual features were parsed", b.vlm["status"] == "ok" and b.visual.assessed and b.visual.portal_surface == "buccal",
          b.vlm)
    req = fake.STATE["feature_requests"][-1]
    sys_txt = req["messages"][0]["content"]
    user_txt = req["messages"][1]["content"][0]["text"]
    check("feature request sent with one image and the feature schema", sum(1 for p in req["messages"][1]["content"] if p["type"] == "image_url") == 1
          and "visual_features" in sys_txt)
    check("model is given scale only, never the geometric numbers",
          "voxel_size_mm_zyx" in user_txt and f"{geom['long_axis_extent_mm'].value:.2f}" not in user_txt
          and "long_axis_extent_mm" not in user_txt)
    la = b.measurements["long_axis_extent_mm"]
    check("agreeing estimate -> geometry kept, agreement recorded", la.source == SOURCE_VOXEL and la.verified and la.vlm_agreement == "agrees"
          and la.vlm_value is not None and abs(la.value - geom["long_axis_extent_mm"].value) < 1e-9)
    check("null estimate -> not_reported", b.measurements["dentine_volume_fraction_involved"].vlm_agreement == "not_reported")
    check("boolean agreement", b.measurements["root_canal_communication"].vlm_agreement == "agrees")
    check("no unverified / disagreeing values", b.unverified_vlm_values == [] and b.vlm_geometry_disagreements == [])
    # now the model is badly off
    fake.STATE.update(feature_estimates={"long_axis_extent_mm": 12.0, "root_canal_communication": "yes", "min_distance_to_canal_mm": 0.0})
    b2 = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=_backend(), frame_png=b"\x89PNG fake", source_id="ph")
    la2 = b2.measurements["long_axis_extent_mm"]
    check("disagreeing estimate -> geometry kept, flagged, listed",
          la2.value == la.value and la2.verified and la2.vlm_agreement == "disagrees"
          and "vlm_value_disagrees_with_geometry_geometry_kept" in la2.flags
          and set(b2.vlm_geometry_disagreements) >= {"long_axis_extent_mm", "root_canal_communication"}, b2.vlm_geometry_disagreements)
    check("bundle caveat records the disagreement", "model_estimates_disagree_with_geometry_for_some_values" in b2.caveats)
    check("serialised bundle still validates", validate_bundle_dict(b2.to_dict()) == [])


def test_unverifiable_vlm_values_are_flagged_and_excluded():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    seg_nocanal = ToothSegmentation(tooth=seg.tooth, canal=np.zeros_like(seg.canal), bone=seg.bone)
    fake.STATE.update(feature_mode="agree", feature_estimates={"min_distance_to_canal_mm": 0.9, "root_canal_communication": "no",
                                                               "coronal_limit_from_cej_mm": -0.4})
    geometry_nocej = {**geometry, "cej_t_mm": None, "source": "automatic_unreviewed"}
    b = measure_candidate_features(arr, seg_nocanal, labels, 1, (VOX, VOX, VOX), geometry_nocej, backend=_backend(), frame_png=b"img", source_id="ph")
    dmin = b.measurements["min_distance_to_canal_mm"]
    check("no canal mask: model distance kept as vlm_unverified with low confidence",
          dmin.source == SOURCE_VLM_UNVERIFIED and dmin.verified is False and dmin.value == 0.9 and dmin.confidence <= 0.3
          and "vlm_value_not_geometrically_verifiable" in dmin.flags)
    check("unverified names listed on the bundle",
          set(b.unverified_vlm_values) == {"min_distance_to_canal_mm", "root_canal_communication", "coronal_limit_from_cej_mm"},
          b.unverified_vlm_values)
    check("caveat says some values are model estimates", "some_values_are_model_estimates_without_geometric_verification" in b.caveats)
    check("measurement_value() hides unverified values by default",
          b.measurement_value("min_distance_to_canal_mm") is None and b.measurement_value("min_distance_to_canal_mm", verified_only=False) == 0.9)
    s = summarise_extent(b)
    check("extent summary ignores unverified canal values -> pulp relationship not_assessable",
          s.pulp_relationship == "not_assessable" and "min_distance_to_canal_mm" in s.unavailable, (s.pulp_relationship, s.unavailable))
    rec = recommend_from_measurements(s, ClinicalModifiers())
    check("planner asks for canal delineation instead of trusting the model", "canal_delineation" in rec.missing_inputs, rec.missing_inputs)
    check("structural validation accepts the mixed-source bundle", validate_bundle_dict(b.to_dict()) == [], validate_bundle_dict(b.to_dict()))


def test_vlm_failure_modes():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    fake.STATE.update(feature_mode="disagree_samples", feature_estimates={})
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=_backend(n_samples=5), frame_png=b"img", source_id="ph")
    check("samples disagree -> abstained; visual features not used; geometry intact",
          b.vlm["status"] == "abstained" and b.visual.assessed is False
          and "vlm_abstained_samples_disagreed_visual_features_not_used" in b.caveats
          and b.measurements["long_axis_extent_mm"].verified)
    fake.STATE.update(feature_mode="invalid")
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=_backend(), frame_png=b"img", source_id="ph")
    check("invalid enums / estimates coerced, validation errors recorded, bundle still valid",
          b.vlm["status"] == "ok" and len(b.vlm["validation_errors"]) >= 4 and b.visual.radiolucency_pattern == "uncertain"
          and validate_bundle_dict(b.to_dict()) == [], b.vlm.get("validation_errors"))
    d = b.to_dict()
    check("staging-like field from the model is dropped (only the validation log mentions it)",
          "patel_height" not in json.dumps(d["vlm"]["raw_estimates"]) and "patel_height" not in json.dumps(d["measurements"])
          and "patel_height" not in json.dumps(d["visual"]))
    fake.STATE.update(feature_mode="no_features")
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=_backend(), frame_png=b"img", source_id="ph")
    check("structurally unusable answer -> failed, geometry-only", b.vlm["status"] == "failed" and "schema" in b.vlm["error"]
          and "vision_model_call_failed_geometry_only" in b.caveats)
    fake.STATE.update(feature_mode="agree")
    out = query_vlm_features(_backend(), 1, (VOX,) * 3, None)
    check("no image -> no model call", out["status"] == "no_image")
    be = _backend()
    be.supports_images = False
    out = query_vlm_features(be, 1, (VOX,) * 3, b"img")
    check("text-only backend -> no model call", out["status"] == "no_image")


# --------------------------------------------------------------------------- extent summary + mapping
def test_extent_summary_and_measurement_mapping():
    arr, seg, labels, truth, geometry = phantom_case(3.0, 150, False)
    b = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=NullBackend(), source_id="ph")
    s = summarise_extent(b)
    check("descriptors are from the documented vocabularies",
          s.extent_descriptor in EXTENT_LEVELS and s.pulp_relationship in PULP_LEVELS and s.access_descriptor in ACCESS_LEVELS)
    check("3 mm / 150 deg non-perforating -> moderate / barrier preserved / supracrestal",
          (s.extent_descriptor, s.pulp_relationship, s.access_descriptor) == ("moderate", "dentine_barrier_preserved", "supracrestal_portal"),
          (s.extent_descriptor, s.pulp_relationship, s.access_descriptor))
    d = s.to_dict()
    check("summary declares it is not a staging code", d["is_staging_code"] is False and d["scheme"] == "measurement_v1")
    check("cell key is descriptive", s.cell_key.startswith("extent=") and "|pulp=" in s.cell_key and not PATEL_CODE.search(json.dumps(d)))
    rec = recommend_from_measurements(s, ClinicalModifiers())
    check("moderate / preserved / supracrestal -> external repair primary, scheme measurement",
          rec.primary == Treatment.EXTERNAL_REPAIR and rec.cell_scheme == SCHEME_MEASUREMENT and rec.cell_key == s.cell_key, rec.primary)
    rd = rec.to_dict()
    check("recommendation dict carries cell_key/cell_scheme and no patel_code", "patel_code" not in rd and rd["cell_scheme"] == SCHEME_MEASUREMENT)
    check("requires confirmation always true", rec.requires_confirmation is True)
    # deep perforating lesion
    arr, seg, labels, truth, geometry = phantom_case(9.0, 200, True)
    b2 = measure_candidate_features(arr, seg, labels, 1, (VOX, VOX, VOX), geometry, backend=NullBackend(), source_id="ph")
    s2 = summarise_extent(b2)
    check("9 mm / 200 deg perforating -> extensive / canal communication",
          s2.extent_descriptor == "extensive" and s2.pulp_relationship == "canal_communication", (s2.extent_descriptor, s2.pulp_relationship))
    rec2 = recommend_from_measurements(s2, ClinicalModifiers())
    check("extensive + communication -> RCT-containing primary and irreversible alternatives offered, never chosen",
          rec2.primary in (Treatment.INTERNAL_REPAIR_RCT, Treatment.EXTERNAL_REPAIR_RCT)
          and any(t.irreversible for t in rec2.safe_set) and rec2.requires_confirmation, rec2.to_dict())
    # clinical modifiers still apply on the measurement path
    rec3 = recommend_from_measurements(s2, ClinicalModifiers(restorable=False))
    check("unrestorable -> extraction-type primary on the measurement path", rec3.primary in (Treatment.EXTRACTION, Treatment.DECORONATION), rec3.primary)
    rec4 = recommend_from_measurements(s, ClinicalModifiers(surgically_accessible=False))
    check("inaccessible portal -> internal approach / replantation preferred", rec4.primary != Treatment.EXTERNAL_REPAIR, rec4.primary)
    # policy context works off the extent summary
    ctx = ClinicalContext(extent=s, mods=ClinicalModifiers(), tooth_type="incisor")
    v = ctx.vector()
    check("ClinicalContext(extent=...) vectorises and reports measurement scheme", len(v) == len(ClinicalContext(extent=s2).vector())
          and ctx.scheme == SCHEME_MEASUREMENT)
    pol = ConstrainedBanditPolicy()
    dec = pol.decide(ctx, rec)
    check("policy ranking stays inside the safe set", set(dec.ranking) <= set(rec.safe_set) and dec.cell_scheme == SCHEME_MEASUREMENT)
    check("policy decision dict has no patel_code on the measurement path", "patel_code" not in dec.to_dict())
    # the full mapping is enumerable and documented
    rows = measurement_matrix_rows()
    check("measurement matrix covers every descriptor combination",
          len(rows) == len(EXTENT_LEVELS) * len(PULP_LEVELS) * len(ACCESS_LEVELS), len(rows))
    check("no staging codes in the mapping table", not any(PATEL_CODE.search(json.dumps(r)) for r in rows))


def test_feature_notes_and_logodds_are_bounded():
    bundle = {"visual": {"assessed": True, "portal_of_entry_visible": "yes", "fibro_osseous_appearance": "yes",
                         "pericanal_radiopaque_line": "intact", "radiolucency_pattern": "mottled", "border_definition": "irregular",
                         "relation_to_restoration_margin": "no_restoration_visible", "burnout_artifact_suspected": "no",
                         "metal_beam_hardening": "none", "portal_surface": "buccal"},
              "vlm": {"status": "ok"}, "measurements": {}}
    notes = bundle_feature_notes(bundle)
    check("feature notes list supporting features", len(notes["for"]) >= 2 and isinstance(notes["confounders"], list), notes)
    lo = feature_logodds(bundle)
    check("feature log-odds nudge is per hypothesis and capped at +/-0.6",
          lo and lo.get("ECR", 0) > 0 and all(abs(v) <= 0.6 + 1e-9 for v in lo.values()), lo)
    bundle["vlm"]["status"] = "abstained"
    check("abstained -> no log-odds nudge", feature_logodds(bundle) == {})
    bundle["vlm"]["status"] = "ok"; bundle["visual"]["assessed"] = False
    check("geometry-only -> no log-odds nudge", feature_logodds(bundle) == {})


# --------------------------------------------------------------------------- conference end-to-end
def _conference_case(h, s, perf, morphology="ecr"):
    arr, seg, lesion, truth = make_phantom(h, s, perf, morphology=morphology)
    vol = CBCTVolume(arr.astype(np.float32), (VOX, VOX, VOX), (0., 0., 0.), np.eye(3), {"SeriesInstanceUID_hash": f"ph_{morphology}_{h}_{s}"})
    return vol, (lambda a, sp: ToothSegmentation(seg.tooth, seg.canal, seg.bone))


def test_conference_end_to_end_with_fake_vlm():
    fake.STATE.update(mode="agree", feature_mode="agree", feature_estimates={"root_canal_communication": "no"}, requests=[],
                      feature_requests=[], cf_lengths=set())
    vol, segm = _conference_case(3.0, 150, False)
    be = _backend(n_samples=3)
    t0 = time.time()
    c = CaseConference(backend=be, max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False, case_id="e2e")
    dt = time.time() - t0
    print(f"  conference {dt:.1f}s status={c['status']}")
    check("conference reaches a plan", c["status"] == "suggested_ecr_with_plan", c["status"])
    kinds = {e["kind"] for e in c["conference"]["evidence"]}
    check("feature_bundle and extent_summary evidence posted; no patel evidence",
          {"feature_bundle", "extent_summary", "screen_reconciliation", "plan"} <= kinds and "patel" not in kinds, kinds)
    check("feature requests reached the fake server", len(fake.STATE["feature_requests"]) >= 1)
    fb = c["feature_bundles"]
    cid = c["suggested_diagnosis"]["candidate_id"]
    check("bundle for the leading candidate exists and validates", str(cid) in {str(k) for k in fb}
          and all(validate_bundle_dict(b) == [] for b in fb.values()))
    lead = fb[cid] if cid in fb else fb[str(cid)]
    check("leading bundle used the VLM", lead["vlm"]["status"] == "ok" and lead["visual"]["assessed"])
    se = c["suggested_extent"]
    check("measured extent summary present and descriptive",
          se["extent_descriptor"] == "moderate" and se["pulp_relationship"] == "dentine_barrier_preserved"
          and se["access_descriptor"] == "supracrestal_portal" and se["is_staging_code"] is False, se)
    check("no Patel output on the default path", "suggested_patel" not in c)
    js = json.dumps(c, default=str)
    check("report JSON carries no staging codes or patel keys", "patel_code" not in js and not PATEL_CODE.search(js))
    pl = c["suggested_plan"]
    rr = pl["rule_recommendation"]
    check("plan keyed by the measurement cell", rr["cell_scheme"] == SCHEME_MEASUREMENT and rr["cell_key"] == se["cell_key"]
          and pl["extent_summary_ref"]["extent_descriptor"] == "moderate")
    check("external repair primary for this phantom", rr["primary"] == "external_repair", rr["primary"])
    check("safety: requires confirmation, not clinically validated", c["requires_confirmation"] is True and c["clinical_validation"] is False)
    check("plan is conditional on clinical confirmation", "clinical confirmation" in pl["conditional_on"])
    dx = c["suggested_diagnosis"]
    check("diagnosis is a suggestion with an uncalibrated posterior and never 'confirmed'",
          "p_ecr_uncalibrated" in dx and "ECR_CONFIRMED" not in js and dx["band"] == "moderate")
    diff = next(e for e in c["conference"]["evidence"] if e["kind"] == "differential" and e["refers_to"] == cid)
    flo = diff["content"].get("feature_logodds") or {}
    check("radiologist consumed the feature bundle with a capped nudge", diff["content"].get("feature_bundle_used") is True
          and all(abs(v) <= 0.6 + 1e-9 for v in flo.values()), flo)
    md = consensus_markdown(c)
    check("markdown renders extent + feature sections, no Patel section",
          "Measured extent summary" in md and "Feature measurements" in md and "Patel 3D classification" not in md)
    check("clinician actions mention reviewing the measured extent", any("measured extent" in a for a in c["clinician_actions_required"]))
    # negative screen is never an exclusion
    check("disclaimer: negative screen does not exclude ECR", "does not exclude" in c["disclaimer"].lower())


def test_conference_flags_model_disagreement_and_perforation():
    fake.STATE.update(mode="agree", feature_mode="agree", requests=[], feature_requests=[],
                      feature_estimates={"long_axis_extent_mm": 1.0, "root_canal_communication": "no"})
    vol, segm = _conference_case(9.0, 200, True)
    c = CaseConference(backend=_backend(n_samples=3), max_rounds=1).run(vol, segm, SEED, "phantom_truth", render_tiles=False, case_id="e2e2")
    check("deep perforating phantom reaches a plan", c["status"] == "suggested_ecr_with_plan", c["status"])
    se = c["suggested_extent"]
    check("extensive / canal communication measured", se["extent_descriptor"] == "extensive" and se["pulp_relationship"] == "canal_communication", se)
    cid = c["suggested_diagnosis"]["candidate_id"]
    lead = c["feature_bundles"].get(cid) or c["feature_bundles"].get(str(cid))
    check("model's wrong extent / communication flagged, geometry kept",
          set(lead["vlm_geometry_disagreements"]) >= {"long_axis_extent_mm", "root_canal_communication"}
          and lead["measurements"]["root_canal_communication"]["value"] is True, lead["vlm_geometry_disagreements"])
    chal = [e for e in c["conference"]["evidence"] if e["kind"] == "challenge"]
    issues = " ".join(ch["issue"] for e in chal for ch in e["content"].get("challenges", []))
    check("skeptic raises the VLM / geometry disagreement", "disagree" in issues.lower(), issues[:300])
    pl = c["suggested_plan"]
    check("irreversible options present require confirmation and are flagged",
          pl["irreversible_options_in_safe_set"] and pl["rule_recommendation"]["requires_confirmation"]
          and any("Irreversible" in a for a in c["clinician_actions_required"]))
    check("escalation considerations are measurement-indexed text", any("Canal communication" in x for x in pl["escalation_considerations"]))
    check("no staging codes anywhere", not PATEL_CODE.search(json.dumps(c, default=str)))


def test_legacy_scheme_still_reproduces_patel_outputs_when_opted_in():
    vol, segm = _conference_case(3.0, 150, False)
    c = CaseConference(max_rounds=1, staging_scheme="patel_3d").run(vol, segm, SEED, "phantom_truth", render_tiles=False)
    check("legacy opt-in reproduces rev 2.8 code 2Bd and plan", c.get("suggested_patel", {}).get("code") == "2Bd"
          and c["suggested_plan"]["rule_recommendation"]["primary"] == "external_repair" and "suggested_extent" not in c)
    check("legacy plan labelled with its scheme", c["suggested_plan"]["cell_scheme"] == "patel_3d")


if __name__ == "__main__":
    tests = [test_schema_validation_and_coercion, test_geometric_measurements_match_phantom_truth, test_geometry_only_bundle_without_model,
             test_vlm_agreeing_values_are_verified_and_disagreeing_values_flagged, test_unverifiable_vlm_values_are_flagged_and_excluded,
             test_vlm_failure_modes, test_extent_summary_and_measurement_mapping, test_feature_notes_and_logodds_are_bounded,
             test_conference_end_to_end_with_fake_vlm, test_conference_flags_model_disagreement_and_perforation,
             test_legacy_scheme_still_reproduces_patel_outputs_when_opted_in]
    only = sys.argv[1:]
    for t in tests:
        if only and t.__name__ not in only:
            continue
        print(f"=== {t.__name__}")
        t()
    SRV.shutdown()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed (software behaviour on phantoms; not clinical validation)")
    sys.exit(1 if n_fail else 0)
