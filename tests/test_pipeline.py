import numpy as np
import pytest

from plate_rod_thinning import backend
import plate_rod_thinning.pipeline as pipeline
from plate_rod_thinning.pipeline import (
    PlateRodParameters,
    PlateRodResult,
    classify_skeleton_preview,
    label_full_thickness,
    plate_rod_analysis,
)


def test_classify_skeleton_preview_uses_degree_and_slenderness() -> None:
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 1:4] = True
    skeleton[2, 1:4, 2] = True

    no_slenderness = classify_skeleton_preview(skeleton, slenderness=0)
    one_pass = classify_skeleton_preview(skeleton, slenderness=1)

    assert np.count_nonzero(no_slenderness == 1) == 5
    assert np.count_nonzero(no_slenderness == 2) == 0
    assert one_pass[2, 2, 2] == 1
    assert np.count_nonzero(one_pass == 2) == 4


def test_label_full_thickness_assigns_bone_voxels_to_nearest_skeleton_label() -> None:
    bone = np.zeros((3, 7, 3), dtype=bool)
    bone[1, 1:6, 1] = True
    skeleton_labels = np.zeros_like(bone, dtype=np.uint8)
    skeleton_labels[1, 1, 1] = 1
    skeleton_labels[1, 5, 1] = 2

    full = label_full_thickness(bone, skeleton_labels)

    assert np.all(full[bone] > 0)
    assert full[1, 1, 1] == 1
    assert full[1, 5, 1] == 2
    assert full[1, 2, 1] == 1
    assert full[1, 4, 1] == 2


def test_label_full_thickness_uses_backend_propagation(monkeypatch) -> None:
    bone = np.zeros((3, 3, 3), dtype=bool)
    seed_labels = np.zeros_like(bone, dtype=np.uint8)
    expected = np.ones_like(seed_labels, dtype=np.uint8)

    def fake_propagate(received_bone, received_seed_labels):
        assert received_bone is bone
        assert received_seed_labels is seed_labels
        return expected

    monkeypatch.setattr(pipeline, "propagate_labels_6_connected", fake_propagate)

    assert label_full_thickness(bone, seed_labels) is expected


def test_default_parameters_remove_tiny_rod_fragments() -> None:
    params = PlateRodParameters()
    assert params.min_plate_voxels == 0
    assert params.min_rod_voxels == 5
    assert params.junction_dilation_voxels == 2
    assert params.junction_support_radius_voxels is None


def test_plate_rod_analysis_returns_named_outputs_and_summary_statistics() -> None:
    bone = np.zeros((7, 7, 7), dtype=bool)
    bone[3, 3, 1:6] = True
    bone[3, 1:6, 3] = True
    params = PlateRodParameters(slenderness=1, skeletonize=False)

    result = plate_rod_analysis(bone, parameters=params)

    assert isinstance(result, PlateRodResult)
    assert result.skeleton.shape == bone.shape
    assert result.topology_labels.shape == bone.shape
    assert result.skeleton_labels.shape == bone.shape
    assert result.full_thickness_labels.shape == bone.shape
    assert result.component_labels.shape == bone.shape
    assert result.summary["bone_voxels"] == int(bone.sum())
    assert result.summary["skeleton_voxels"] == int(result.skeleton.sum())
    assert "plate_skeleton_voxels" in result.summary
    assert "rod_skeleton_voxels" in result.summary


def test_plate_rod_analysis_uses_topological_classifier_not_degree_preview() -> None:
    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[2, 2, 1:4] = True
    params = PlateRodParameters(skeletonize=False)

    result = plate_rod_analysis(bone, parameters=params)

    assert result.summary["classifier"] == "saha_topology"
    assert result.skeleton_labels[2, 2, 2] == 2
    assert result.summary["rBV_voxels"] == 3


def test_plate_rod_analysis_reports_plate_rod_volume_fractions_and_ratio() -> None:
    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[2, 2, 1:4] = True
    bone[2, 1:4, 2] = True
    analysis_mask = bone.copy()
    analysis_mask[0, 0, 0] = True
    params = PlateRodParameters(slenderness=1, skeletonize=False, voxel_spacing_mm=(0.1, 0.2, 0.3))

    result = plate_rod_analysis(bone, analysis_mask=analysis_mask, parameters=params)

    assert result.summary["tissue_voxels"] == 6
    assert result.summary["BV/TV"] == 5 / 6
    assert result.summary["pBV_voxels"] == 1
    assert result.summary["rBV_voxels"] == 4
    assert result.summary["pBV/TV"] == 1 / 6
    assert result.summary["rBV/TV"] == 4 / 6
    assert result.summary["pBV/BV"] == 1 / 5
    assert result.summary["rBV/BV"] == 4 / 5
    assert result.summary["PR_volume_ratio"] == 1 / 4
    assert result.summary["pBV/rBV"] == 1 / 4
    assert result.summary["voxel_volume_mm3"] == pytest.approx(0.006)
    assert result.summary["pBV_mm3"] == pytest.approx(0.006)
    assert result.summary["rBV_mm3"] == pytest.approx(0.024)
    assert "pTb.N" in result.summary
    assert "rTb.N" in result.summary
    assert "PR.N" in result.summary
    assert "pTb.Th_mm" in result.summary
    assert "rTb.Th_mm" in result.summary
    assert "pTb.S_mm2" in result.summary
    assert "rTb.l_mm" in result.summary
    assert "P-P Junc.D" in result.summary
    assert "P-R Junc.D" in result.summary
    assert "R-R Junc.D" in result.summary
    assert "aBV/TV" in result.summary


def test_plate_rod_analysis_skeletonizes_only_mask_bounding_box_by_default(monkeypatch) -> None:
    bone = np.zeros((9, 9, 9), dtype=bool)
    bone[3:6, 4:7, 2:5] = True
    seen_shapes = []

    def fake_skeletonize(image, *, max_iterations):
        assert max_iterations == 200
        seen_shapes.append(image.shape)
        return image.copy()

    monkeypatch.setattr("plate_rod_thinning.pipeline.skeletonize_surface", fake_skeletonize)

    result = plate_rod_analysis(bone)

    assert seen_shapes == [(3, 3, 3)]
    assert result.skeleton.shape == bone.shape
    assert np.array_equal(result.skeleton, bone)


def test_plate_rod_analysis_can_disable_bounding_box_crop(monkeypatch) -> None:
    bone = np.zeros((9, 9, 9), dtype=bool)
    bone[3:6, 4:7, 2:5] = True
    seen_shapes = []

    def fake_skeletonize(image, *, max_iterations):
        assert max_iterations == 200
        seen_shapes.append(image.shape)
        return image.copy()

    monkeypatch.setattr("plate_rod_thinning.pipeline.skeletonize_surface", fake_skeletonize)

    plate_rod_analysis(bone, parameters=PlateRodParameters(crop_to_mask=False))

    assert seen_shapes == [bone.shape]


def test_plate_rod_analysis_passes_max_iterations_to_skeletonizer(monkeypatch) -> None:
    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[1:4, 1:4, 1:4] = True
    seen_max_iterations = []

    def fake_skeletonize(image, *, max_iterations):
        seen_max_iterations.append(max_iterations)
        return image.copy()

    monkeypatch.setattr("plate_rod_thinning.pipeline.skeletonize_surface", fake_skeletonize)

    plate_rod_analysis(bone, parameters=PlateRodParameters(max_iterations=7))

    assert seen_max_iterations == [7]


def test_plate_rod_analysis_uses_backend_skeletonizer_dispatcher() -> None:
    assert pipeline.skeletonize_surface is backend.skeletonize_surface
