import numpy as np
import pytest

import plate_rod_thinning.morphometry as morphometry
from plate_rod_thinning.morphometry import compute_its_morphometry
from plate_rod_thinning.classification import (
    ARC_ARC_JUNCTION,
    SURFACE_CURVE_JUNCTION,
    SURFACE_SURFACE_JUNCTION,
)


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


def test_compute_its_morphometry_reports_plate_surface_from_skeleton_mid_surface():
    skeleton_labels = np.zeros((3, 4, 5), dtype=np.uint8)
    skeleton_labels[1, 1:3, 1:4] = 1
    full_labels = skeleton_labels.copy()
    full_labels[0:3, 1:3, 1:4] = 1
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.summary["plate_count"] == 1
    assert result.summary["pTb.S_mm2"] == pytest.approx(6.0)


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


def test_compute_its_morphometry_preserves_original_topology_junction_types():
    full_labels = np.zeros((5, 9, 5), dtype=np.uint8)
    skeleton_labels = np.zeros_like(full_labels)
    topology_classes = np.zeros_like(full_labels)
    skeleton_labels[2, 1:3, 2] = 1
    skeleton_labels[2, 3, 2] = 3
    skeleton_labels[2, 4:6, 2] = 1
    topology_classes[2, 3, 2] = SURFACE_SURFACE_JUNCTION
    skeleton_labels[2, 6, 2] = 3
    skeleton_labels[2, 7:9, 2] = 2
    topology_classes[2, 6, 2] = SURFACE_CURVE_JUNCTION
    skeleton_labels[1, 7, 2] = 3
    skeleton_labels[0, 7, 2] = 2
    topology_classes[1, 7, 2] = ARC_ARC_JUNCTION
    full_labels[skeleton_labels > 0] = skeleton_labels[skeleton_labels > 0]
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        topology_classes=topology_classes,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.summary["P-P Junc.D"] == pytest.approx(1 / full_labels.size)
    assert result.summary["P-R Junc.D"] == pytest.approx(1 / full_labels.size)
    assert result.summary["R-R Junc.D"] == pytest.approx(1 / full_labels.size)


def test_compute_its_morphometry_splits_components_by_dilated_junction_zones():
    skeleton_labels = np.zeros((5, 7, 5), dtype=np.uint8)
    topology_classes = np.zeros_like(skeleton_labels)
    skeleton_labels[2, 1:6, 2] = 1
    skeleton_labels[2, 3, 2] = 3
    topology_classes[2, 1:3, 2] = 2
    topology_classes[2, 3, 2] = SURFACE_SURFACE_JUNCTION
    topology_classes[2, 4:6, 2] = 2

    full_labels = np.zeros_like(skeleton_labels)
    full_labels[2, 1:6, 2] = 1
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        topology_classes=topology_classes,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        junction_dilation_voxels=1,
    )

    assert result.summary["plate_count"] == 2
    assert result.summary["P-P Junc.D"] == pytest.approx(1 / full_labels.size)
    assert sorted(np.unique(result.component_labels[full_labels > 0]).tolist()) == [1, 2]


def test_skeleton_graph_elements_uses_matlab_style_minimum_component_sizes_after_junction_removal():
    labels = np.zeros((5, 9, 5), dtype=np.uint8)
    topology_classes = np.zeros_like(labels)
    labels[2, 1:4, 2] = 2
    topology_classes[2, 1:4, 2] = 6
    labels[2, 5:9, 2] = 1
    topology_classes[2, 5:9, 2] = 2

    element_labels, element_types, _, _ = morphometry._skeleton_graph_elements(
        labels,
        topology_classes,
        min_plate_voxels=4,
        min_rod_voxels=4,
    )

    assert np.count_nonzero(element_types == 1) == 1
    assert np.count_nonzero(element_types == 2) == 0
    assert np.count_nonzero(element_labels) == 4


def test_component_neighbor_pair_junctions_require_matching_topology_support_when_provided():
    component_labels = np.zeros((3, 7, 3), dtype=np.int32)
    component_labels[1, 1:3, 1] = 1
    component_labels[1, 3:5, 1] = 2
    component_labels[1, 5, 1] = 1
    component_labels[1, 6, 1] = 2
    element_types = np.asarray([0, 1, 2], dtype=np.uint8)
    topology_classes = np.zeros_like(component_labels, dtype=np.uint8)
    topology_classes[1, 3, 1] = SURFACE_CURVE_JUNCTION

    junctions = morphometry._component_neighbor_pair_junctions(
        component_labels,
        element_types,
        voxel_volume=1.0,
        topology_classes=topology_classes,
        support_radius_voxels=1,
    )

    assert len(junctions) == 1
    assert junctions[0].junction_type == "P-R"


def test_compute_its_morphometry_counts_pairwise_junction_edges_within_cluster():
    full_labels = np.zeros((7, 7, 7), dtype=np.uint8)
    skeleton_labels = np.zeros_like(full_labels)
    skeleton_labels[1:3, 3, 3] = 1
    skeleton_labels[4:6, 3, 3] = 1
    skeleton_labels[3, 1:3, 3] = 2
    skeleton_labels[3, 4:6, 3] = 2
    skeleton_labels[3, 3, 3] = 3
    full_labels[skeleton_labels > 0] = skeleton_labels[skeleton_labels > 0]
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.summary["P-P Junc.D"] == pytest.approx(1 / 343)
    assert result.summary["P-R Junc.D"] == pytest.approx(4 / 343)
    assert result.summary["R-R Junc.D"] == pytest.approx(1 / 343)
    assert [junction.junction_type for junction in result.junctions].count("P-P") == 1
    assert [junction.junction_type for junction in result.junctions].count("P-R") == 4
    assert [junction.junction_type for junction in result.junctions].count("R-R") == 1


def test_skeleton_graph_elements_type_plate_rod_and_plate_plate_junctions():
    labels = np.zeros((5, 7, 7), dtype=np.uint8)
    labels[2, 1:3, 2] = 1
    labels[2, 3, 2] = 3
    labels[2, 4:6, 2] = 2
    labels[2, 1:3, 5] = 1
    labels[2, 3, 5] = 3
    labels[2, 4:6, 5] = 1

    element_labels, element_types, junction_labels, junctions = morphometry._skeleton_graph_elements(labels)

    assert element_labels.max() == 4
    assert sorted(element_types[1:].tolist()) == [1, 1, 1, 2]
    assert junction_labels.max() == 2
    assert [junction.junction_type for junction in junctions] == ["P-R", "P-P"]


def test_compute_its_morphometry_keeps_rods_split_by_graph_junction_when_full_labels_touch():
    skeleton_labels = np.zeros((3, 7, 3), dtype=np.uint8)
    skeleton_labels[1, 1:3, 1] = 2
    skeleton_labels[1, 3, 1] = 3
    skeleton_labels[1, 4:6, 1] = 2
    full_labels = skeleton_labels.copy()
    full_labels[1, 3, 1] = 2
    analysis_mask = np.ones_like(full_labels, dtype=bool)

    result = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=analysis_mask,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    rod_ids = sorted({int(value) for value in np.unique(result.component_labels[full_labels > 0]) if value})
    assert len(rod_ids) == 2
    assert result.summary["rod_count"] == 2
    assert result.summary["R-R Junc.D"] == pytest.approx(1 / full_labels.size)
    assert result.summary["rTb.l_mm"] == pytest.approx(1.0)


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
