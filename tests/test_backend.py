import numpy as np

from plate_rod_thinning import backend
from plate_rod_thinning.skeletonize import skeletonize_surface
from plate_rod_thinning.topology import initial_classes_from_keys, linear_to_subscript, neighborhood_keys_3x3_at


def test_backend_neighborhood_keys_match_reference_python_encoder():
    image = np.zeros((7, 7, 7), dtype=bool)
    image[1:6, 2:5, 2:6] = True
    coords = np.asarray([(2, 3, 3), (4, 4, 4), (5, 2, 2)], dtype=np.int64)

    assert np.array_equal(
        backend.neighborhood_keys_3x3_at(image, coords),
        neighborhood_keys_3x3_at(image, coords),
    )


def test_backend_reports_active_implementation_name():
    assert backend.backend_name() in {"python", "compiled", "metal+python", "metal+compiled"}


def test_backend_initial_classes_match_reference_python_classifier():
    keys = np.asarray([
        1 << 13,
        (1 << 13) | (1 << 4),
        (1 << 13) | (1 << 4) | (1 << 10),
    ], dtype=np.uint32)

    assert np.array_equal(
        backend.initial_classes_from_keys(keys),
        initial_classes_from_keys(keys),
    )


def test_backend_initial_classes_match_reference_for_random_neighborhoods():
    rng = np.random.default_rng(20260828)
    keys = rng.integers(0, 1 << 27, size=512, dtype=np.uint32)

    assert np.array_equal(
        backend.initial_classes_from_keys(keys),
        initial_classes_from_keys(keys),
    )


def test_backend_full_thickness_labels_match_reference_wavefront():
    bone = np.zeros((4, 7, 4), dtype=bool)
    bone[1:3, 1:6, 1:3] = True
    seed_labels = np.zeros_like(bone, dtype=np.uint8)
    seed_labels[1, 1, 1] = 1
    seed_labels[2, 5, 2] = 2

    expected = np.zeros_like(seed_labels)
    queue = [(1, 1, 1), (2, 5, 2)]
    expected[1, 1, 1] = 1
    expected[2, 5, 2] = 2
    for x, y, z in queue:
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            neighbor = (x + dx, y + dy, z + dz)
            if bone[neighbor] and expected[neighbor] == 0:
                expected[neighbor] = expected[x, y, z]
                queue.append(neighbor)

    assert np.array_equal(
        backend.propagate_labels_6_connected(bone, seed_labels),
        expected,
    )


def test_compiled_backend_exposes_full_thickness_label_propagation_when_available():
    if backend._c_backend is not None:
        assert hasattr(backend._c_backend, "propagate_labels_6_connected")


def test_compiled_backend_exposes_initial_class_batcher_when_available():
    if backend._c_backend is not None:
        assert hasattr(backend._c_backend, "initial_classes_from_keys")


def test_backend_uses_metal_key_dispatcher_when_requested(monkeypatch):
    image = np.zeros((5, 5, 5), dtype=bool)
    coords = np.asarray([(2, 2, 2)], dtype=np.int64)
    expected = np.asarray([123], dtype=np.uint32)

    class Status:
        available = True

    class FakeMetalBackend:
        @staticmethod
        def status():
            return Status()

        @staticmethod
        def neighborhood_keys_3x3_at(received_image, received_coords):
            assert received_image.dtype == np.bool_
            assert received_coords.dtype == np.int64
            return expected

    monkeypatch.setenv("PLATE_ROD_USE_METAL", "1")
    monkeypatch.setattr(backend, "metal_backend", FakeMetalBackend)

    assert backend.backend_name().startswith("metal+")
    assert np.array_equal(backend.neighborhood_keys_3x3_at(image, coords), expected)


def test_backend_uses_full_metal_skeletonizer_when_requested(monkeypatch):
    image = np.zeros((5, 5, 5), dtype=bool)
    expected = np.ones_like(image, dtype=bool)

    class Status:
        available = True

    class FakeMetalBackend:
        @staticmethod
        def status():
            return Status()

        @staticmethod
        def skeletonize_surface(received_image, *, max_iterations):
            assert received_image.dtype == np.bool_
            assert max_iterations == 7
            return expected

    monkeypatch.setenv("PLATE_ROD_USE_METAL_FULL", "1")
    monkeypatch.setattr(backend, "metal_backend", FakeMetalBackend)

    assert backend.skeletonize_surface(image, max_iterations=7) is expected


def test_compiled_backend_exposes_full_skeletonizer_when_available():
    if backend.backend_name() == "compiled":
        assert hasattr(backend._c_backend, "skeletonize_surface")


def test_backend_skeletonize_surface_matches_reference():
    image = np.zeros((9, 9, 9), dtype=bool)
    image[2:7, 2:7, 4:6] = True

    assert np.array_equal(
        backend.skeletonize_surface(image, max_iterations=3),
        skeletonize_surface(image, max_iterations=3),
    )


def test_backend_skeletonize_surface_includes_final_erosion_pass():
    image = np.zeros((5, 5, 5), dtype=bool)
    image[2, 2, 2] = True
    for index in (38, 58, 64):
        image[linear_to_subscript(index, (5, 5, 5))] = True

    result = backend.skeletonize_surface(image, max_iterations=0)

    assert not result[2, 2, 2]
