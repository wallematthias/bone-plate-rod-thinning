import numpy as np

import plate_rod_thinning.classification as classification
from plate_rod_thinning.classification import (
    ARC_ENDPOINT,
    ARC_INNER,
    ISOLATED,
    PLATE,
    ROD,
    SURFACE_ENDPOINT,
    classify_skeleton_topology,
    plate_rod_labels_from_topology,
)
from plate_rod_thinning.topology import initial_classes_from_keys as real_initial_classes_from_keys


def test_classify_skeleton_topology_identifies_rod_line_points():
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 1:4] = True

    classes = classify_skeleton_topology(skeleton)

    assert classes[2, 2, 1] == ARC_ENDPOINT
    assert classes[2, 2, 2] == ARC_INNER
    assert classes[2, 2, 3] == ARC_ENDPOINT


def test_classify_skeleton_topology_identifies_isolated_point():
    skeleton = np.zeros((3, 3, 3), dtype=bool)
    skeleton[1, 1, 1] = True

    classes = classify_skeleton_topology(skeleton)

    assert classes[1, 1, 1] == ISOLATED


def test_classify_skeleton_topology_uses_keyed_initial_classes(monkeypatch):
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 1:4] = True
    calls = []

    def fail_array_classifier(_config):
        raise AssertionError("topology classification should use keyed initial classes")

    def recording_initial_classes(keys):
        calls.append(np.asarray(keys).copy())
        return real_initial_classes_from_keys(keys)

    monkeypatch.setattr(classification, "classify_neighborhood", fail_array_classifier)
    monkeypatch.setattr(classification, "initial_classes_from_keys", recording_initial_classes)

    classes = classify_skeleton_topology(skeleton)

    assert calls
    assert classes[2, 2, 1] == ARC_ENDPOINT
    assert classes[2, 2, 2] == ARC_INNER
    assert classes[2, 2, 3] == ARC_ENDPOINT


def test_plate_rod_labels_use_topological_surface_and_curve_classes():
    topology = np.asarray([SURFACE_ENDPOINT, ARC_ENDPOINT, ARC_INNER, ISOLATED], dtype=np.uint8).reshape((2, 2, 1))

    labels = plate_rod_labels_from_topology(topology)

    assert labels[0, 0, 0] == PLATE
    assert labels[0, 1, 0] == ROD
    assert labels[1, 0, 0] == ROD
    assert labels[1, 1, 0] == ROD
