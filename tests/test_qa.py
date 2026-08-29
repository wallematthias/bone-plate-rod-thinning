import numpy as np

import plate_rod_thinning.qa as qa
from plate_rod_thinning.qa import (
    component_summary,
    degree_class_image,
    dilated_for_visualization,
    skeleton_degree,
)


def test_skeleton_degree_counts_26_connected_neighbors() -> None:
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 1:4] = True
    skeleton[2, 3, 2] = True

    degree = skeleton_degree(skeleton)

    assert degree[2, 2, 2] == 3
    assert degree[2, 2, 1] == 2
    assert degree[0, 0, 0] == 0


def test_degree_class_image_marks_endpoints_curves_and_junctions() -> None:
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 1:4] = True

    classes = degree_class_image(skeleton)

    assert classes[2, 2, 1] == 2
    assert classes[2, 2, 2] == 3
    assert classes[2, 2, 3] == 2
    assert classes[0, 0, 0] == 0

    skeleton[2, 3, 2] = True
    classes = degree_class_image(skeleton)

    assert classes[2, 2, 2] == 4


def test_component_summary_reports_26_connected_components() -> None:
    skeleton = np.zeros((6, 6, 6), dtype=bool)
    skeleton[1, 1, 1] = True
    skeleton[1, 1, 2] = True
    skeleton[4, 4, 4] = True

    labels, summary = component_summary(skeleton)

    assert summary.component_count == 2
    assert summary.voxel_count == 3
    assert summary.largest_components[:2] == (2, 1)
    assert labels[1, 1, 1] == labels[1, 1, 2]
    assert labels[4, 4, 4] != labels[1, 1, 1]


def test_component_summary_uses_compiled_connected_component_labeling(monkeypatch) -> None:
    skeleton = np.zeros((6, 6, 6), dtype=bool)
    skeleton[1, 1, 1:3] = True
    skeleton[4, 4, 4] = True

    def fail_flood_fill(*_args, **_kwargs):
        raise AssertionError("component_summary should use compiled connected-component labeling")

    monkeypatch.setattr(qa, "_flood_fill_component", fail_flood_fill)

    _, summary = component_summary(skeleton)

    assert summary.component_count == 2
    assert summary.largest_components[:2] == (2, 1)


def test_dilated_for_visualization_preserves_original_mask_boundary() -> None:
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 2] = True
    mask = np.zeros_like(skeleton)
    mask[2, 2, 1:4] = True

    dilated = dilated_for_visualization(skeleton, mask=mask, radius=1)

    assert dilated.sum() == 3
    assert np.all(dilated <= mask)
