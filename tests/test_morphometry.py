import numpy as np
import pytest

import plate_rod_thinning.morphometry as morphometry
from plate_rod_thinning.morphometry import compute_its_morphometry


def test_compute_its_morphometry_reports_all_standard_its_summary_metrics():
    skeleton_labels = np.zeros((7, 7, 7), dtype=np.uint8)
    skeleton_labels[2, 2, 2:5] = 2
    skeleton_labels[4, 2:5, 2] = 1
    full_labels = skeleton_labels.copy()
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    summary = result.summary
    assert summary["plate_count"] == 1
    assert summary["rod_count"] == 1
    assert summary["pTb.N"] == pytest.approx((1 / 343) ** (1 / 3))
    assert summary["rTb.N"] == pytest.approx((1 / 343) ** (1 / 3))
    assert summary["PR.N"] == pytest.approx(1.0)
    assert summary["pTb.Th_mm"] > 0
    assert summary["rTb.Th_mm"] > 0
    assert summary["pTb.S_mm2"] > 0
    assert summary["rTb.l_mm"] == pytest.approx(2.0)
    assert "aBV/TV" in summary


def test_compute_its_morphometry_counts_typed_junction_densities():
    full_labels = np.zeros((5, 7, 5), dtype=np.uint8)
    skeleton_labels = np.zeros_like(full_labels)
    skeleton_labels[2, 1:3, 2] = 1
    skeleton_labels[2, 3, 2] = 3
    skeleton_labels[2, 4:6, 2] = 2
    full_labels[skeleton_labels > 0] = skeleton_labels[skeleton_labels > 0]
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.summary["P-R Junc.D"] == pytest.approx(1 / 175)
    assert result.summary["P-P Junc.D"] == 0.0
    assert result.summary["R-R Junc.D"] == 0.0
    assert len(result.junctions) == 1
    assert result.junctions[0].junction_type == "P-R"


def test_compute_its_morphometry_measures_junctions_without_per_junction_full_volume_shift(monkeypatch):
    full_labels = np.zeros((5, 7, 5), dtype=np.uint8)
    skeleton_labels = np.zeros_like(full_labels)
    skeleton_labels[2, 1:3, 2] = 1
    skeleton_labels[2, 3, 2] = 3
    skeleton_labels[2, 4:6, 2] = 2
    full_labels[skeleton_labels > 0] = skeleton_labels[skeleton_labels > 0]
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    monkeypatch.setattr(
        morphometry,
        "_measure_junction",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("junctions should be measured in one pass")),
    )

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert len(result.junctions) == 1
    assert result.junctions[0].junction_type == "P-R"


def test_compute_its_morphometry_measures_components_without_full_volume_masks_per_component(monkeypatch):
    skeleton_labels = np.zeros((7, 7, 7), dtype=np.uint8)
    skeleton_labels[2, 2, 2:5] = 2
    skeleton_labels[4, 2:5, 2] = 1
    full_labels = skeleton_labels.copy()
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    monkeypatch.setattr(
        morphometry,
        "_measure_component",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("components should be accumulated by label")),
    )

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.summary["plate_count"] == 1
    assert result.summary["rod_count"] == 1
    assert result.summary["rTb.l_mm"] == pytest.approx(2.0)


def test_voxel_surface_area_uses_exposed_faces():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    assert morphometry._voxel_surface_area(mask, (1.0, 2.0, 3.0)) == pytest.approx(22.0)
