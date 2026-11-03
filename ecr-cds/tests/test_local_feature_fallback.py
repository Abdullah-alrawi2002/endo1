"""Local deterministic feature reading: no network, no Patel codes."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ecr_cds.agents.llm_backend import LocalDeterministicBackend, resolve_feature_backend  # noqa: E402
from ecr_cds.perception.vlm_features import measure_candidate_features  # noqa: E402
from ecr_cds.segmentation import ToothSegmentation  # noqa: E402


def test_unset_url_selects_local_fallback():
    os.environ.pop("VLM_BASE_URL", None)
    backend, mode, detail = resolve_feature_backend()
    assert isinstance(backend, LocalDeterministicBackend)
    assert mode == "local_deterministic_fallback"
    assert "unset" in detail
    print("PASS resolve_feature_backend -> local_deterministic_fallback")


def test_local_reading_is_visual_plus_verified_geometry():
    z, y, x = 24, 24, 24
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    r = np.hypot(yy - 12, xx - 12)
    tooth = (r <= 8) & (zz >= 4) & (zz <= 20)
    canal = (r <= 2) & (zz >= 8) & (zz <= 18)
    bone = (r > 9) & (r <= 11) & (zz >= 10)
    lesion = tooth & (zz >= 8) & (zz <= 12) & (xx >= 16) & (r >= 5)
    raw = np.zeros((z, y, x), np.float32)
    raw[tooth] = 1600
    raw[canal] = 300
    raw[bone] = 900
    raw[lesion] = 600
    labels = np.zeros((z, y, x), np.uint16)
    labels[lesion] = 1
    seg = ToothSegmentation(tooth=tooth, canal=canal, bone=bone)
    geometry = {"centre_mm": [2.4, 2.4, 2.4], "axis": [1.0, 0.0, 0.0], "cej_t_mm": 0.6,
                "source": "automatic_unreviewed", "flags": []}
    b = measure_candidate_features(raw, seg, labels, 1, (0.2, 0.2, 0.2), geometry,
                                   backend=LocalDeterministicBackend(), source_id="unit")
    assert b.vlm["status"] == "local_fallback"
    assert b.vlm["backend"] == "local_deterministic_fallback"
    assert b.visual.assessed is True
    assert b.visual.lesion_visible_in_image == "yes"
    assert b.visual.portal_of_entry_visible == "yes"
    vol = b.measurements["lesion_volume_mm3"]
    assert vol.verified and vol.source == "voxel_geometry" and vol.value > 0
    assert b.measurements["long_axis_extent_mm"].verified
    js = json.dumps(b.to_dict())
    assert "patel" not in js.lower()
    # same inputs, same reading
    b2 = measure_candidate_features(raw, seg, labels, 1, (0.2, 0.2, 0.2), geometry,
                                    backend=LocalDeterministicBackend(), source_id="unit")
    assert b2.visual.to_dict() == b.visual.to_dict()
    print("PASS local fallback visual+geometry, deterministic, no patel strings")


if __name__ == "__main__":
    test_unset_url_selects_local_fallback()
    test_local_reading_is_visual_plus_verified_geometry()
    print("ALL PASS")
