"""Engineered unit fixtures, not patient validation or accuracy estimates."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

from ecr_cds.ecr_detection import CervicalResorptionDetector
from ecr_cds.segmentation import ToothSegmentation
from ecr_cds.pipeline import run_pipeline
from ecr_cds.dicom_loader import CBCTVolume


def fixture(location="cervical", restoration=False):
    z, y, x = np.mgrid[:80, :80, :80]
    r = np.hypot(y-40, x-40)
    tooth = (r <= 20) & (z >= 5) & (z <= 74)
    canal = (r <= 3) & tooth
    bone = (r > 22) & (r < 30) & (z > 38)
    raw = np.zeros(tooth.shape, np.float32)
    raw[bone] = 1500
    raw[tooth] = 2000
    raw[canal] = 200
    if location != "negative":
        if location == "internal":
            raw[28:36, 36:44, 47:51] = 300
        else:
            zz = 28 if location == "cervical" else 59
            raw[zz:zz+8, 36:44, 56:61] = 300
    if restoration:
        raw[28:36, 36:44, 53:56] = 8000
    return ToothSegmentation(tooth, canal, bone, intensity=raw)


def detector():
    return CervicalResorptionDetector(1100, 5000, (.2,)*3,
        cej_zyx=(30,40,40), apex_zyx=(70,40,40), canal_source="expert_mask")


def test_cervical_vs_mimics():
    seg = fixture()
    before = seg.tooth.copy()
    d = detector()
    m = d(None, seg)
    assert m.any()
    assert np.array_equal(before, seg.tooth)
    assert d.last_report["n_review_candidates"] >= 1
    assert any(c["disposition"] == "possible_ecr_review_required" for c in d.last_report["candidates"])
    assert d.last_report["ecr_diagnosis"] is None and d.last_report["decision"] is None
    for kind in ("negative", "internal", "apical"):
        d = detector()
        d(None, fixture(kind))
        assert d.last_report["n_review_candidates"] == 0, (kind, d.last_report)


def test_restoration_and_landmark_flags():
    d = detector()
    d(None, fixture(restoration=True))
    assert any("adjacent_dense_material_enamel_or_restoration" in c["flags"] for c in d.last_report["candidates"])
    a = CervicalResorptionDetector(1100, 5000, (.2,)*3)
    a(None, fixture())
    assert a.last_report["landmarks"]["source"] == "automatic_unreviewed"
    assert a.last_report["ecr_diagnosis"] is None


def test_invalid_inputs_and_empty():
    try:
        CervicalResorptionDetector(1000, 5000, (.2,)*3, cej_zyx=(30,40,40))
        raise AssertionError("Missing apex accepted")
    except ValueError:
        pass
    seg = fixture()
    seg.tooth[:] = False
    try:
        detector()(None, seg)
        raise AssertionError("Empty tooth accepted")
    except ValueError:
        pass


def test_no_treatment_from_unconfirmed_segmentation():
    seg = fixture()
    vol = CBCTVolume(seg.intensity, (.2,)*3, (0,0,0), np.eye(3))
    report = run_pipeline(None, lambda a,s: seg, lambda n,s: s.tooth.astype(float), volume=vol)
    assert report["status"] == "ecr_not_confirmed"
    assert report["patel"] is None and report["decision"] is None


def test_multiple_components_and_repeated_calls():
    seg = fixture()
    seg.intensity[28:36, 36:44, 20:25] = 300  # distinct opposite-side defect
    d = detector()
    first = d(None, seg)
    labels = d.last_labels.copy()
    second = d(None, seg)
    assert np.array_equal(first, second) and np.array_equal(labels, d.last_labels)
    assert d.last_report["n_review_candidates"] == 2


def test_pipeline_cannot_promote_screen_to_diagnosis():
    seg = fixture()
    vol = CBCTVolume(seg.intensity, (.2,)*3, (0,0,0), np.eye(3))
    d = detector()
    report = run_pipeline(None, lambda a,s: seg, d, volume=vol,
                          lesion_source="ecr_candidate", ecr_confirmed=True)
    assert report["ecr_diagnosis"] is None and report["decision"] is None
    assert report["n_review_candidates"] >= 1
    assert "centroid_volume_zyx" in report["candidates"][0]


if __name__ == "__main__":
    tests = [test_cervical_vs_mimics, test_restoration_and_landmark_flags,
             test_invalid_inputs_and_empty, test_no_treatment_from_unconfirmed_segmentation,
             test_multiple_components_and_repeated_calls,
             test_pipeline_cannot_promote_screen_to_diagnosis]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("All ECR-screen logic tests passed; not clinical validation.")
