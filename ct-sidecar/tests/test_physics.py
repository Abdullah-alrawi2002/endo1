import numpy as np

from app.physics import calculate_apical_density


def test_relative_density_reports_drop_without_percent_threshold() -> None:
    shape = (31, 31, 31)
    center = (15, 15, 15)
    volume = np.full(shape, 100.0, dtype=np.float32)
    segmentation = np.ones(shape, dtype=np.uint8)  # upper jaw
    tooth_mask = np.zeros(shape, dtype=bool)

    zz, yy, xx = np.indices(shape)
    lesion = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) <= 3**2
    volume[lesion] = 60.0

    drop, candidate_low, density, warnings = calculate_apical_density(
        volume,
        segmentation,
        tooth_mask,
        center,
        (1.0, 1.0, 1.0),
        "8",
        False,
    )

    assert drop == 40.0
    assert candidate_low is True  # apex mean < reference (observation only)
    assert density.valid is True
    assert density.unit == "GV"
    assert any("not a diagnostic criterion" in warning for warning in warnings)


def test_small_relative_drop_still_candidate_without_25_percent_rule() -> None:
    """A modest drop must not be discarded by a universal ≥25% threshold."""
    shape = (31, 31, 31)
    center = (15, 15, 15)
    volume = np.full(shape, 100.0, dtype=np.float32)
    segmentation = np.ones(shape, dtype=np.uint8)
    tooth_mask = np.zeros(shape, dtype=bool)

    zz, yy, xx = np.indices(shape)
    soft = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) <= 3**2
    volume[soft] = 90.0  # 10% relative drop

    drop, candidate_low, density, _warnings = calculate_apical_density(
        volume,
        segmentation,
        tooth_mask,
        center,
        (1.0, 1.0, 1.0),
        "8",
        False,
    )

    assert drop is not None and 8.0 <= drop <= 12.0
    assert candidate_low is True
    assert density.valid is True
