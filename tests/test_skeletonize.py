import numpy as np

import plate_rod_thinning.skeletonize as skeletonize
from plate_rod_thinning.skeletonize import (
    _outer_layer_candidates,
    _outer_layer_candidates_vectorized,
    skeletonize_surface,
)
from plate_rod_thinning.topology import EOPEN


def test_one_voxel_thick_rod_is_unchanged():
    image = np.zeros((9, 9, 9), dtype=bool)
    image[4, 4, 2:7] = True

    skeleton = skeletonize_surface(image)

    assert np.array_equal(skeleton, image)


def test_two_voxel_thick_slab_reduces_but_keeps_surface():
    image = np.zeros((9, 9, 9), dtype=bool)
    image[2:7, 2:7, 4:6] = True

    skeleton = skeletonize_surface(image, max_iterations=20)

    assert 0 < skeleton.sum() < image.sum()
    assert skeleton.any(axis=2).sum() >= 9


def test_vectorized_outer_layer_candidates_match_reference_classifier():
    image = np.zeros((11, 11, 11), dtype=bool)
    image[3:8, 3:8, 3:8] = True
    image[5, 5, 2] = True
    protected = np.zeros_like(image, dtype=bool)
    protected[3, 3, 3] = True

    reference_coords, reference_types = _outer_layer_candidates(image, protected)
    vectorized_coords, vectorized_types = _outer_layer_candidates_vectorized(image, protected)

    reference = sorted(zip(map(tuple, reference_coords), reference_types.tolist(), strict=False))
    vectorized = sorted(zip(map(tuple, vectorized_coords), vectorized_types.tolist(), strict=False))
    assert vectorized == reference


def test_skeletonizer_uses_keyed_simple_point_path(monkeypatch):
    image = np.zeros((9, 9, 9), dtype=bool)
    image[2:7, 2:7, 4:6] = True
    seen_keys = []
    original = skeletonize.simple_point_from_key

    def recording_simple_point_from_key(key):
        seen_keys.append(key)
        return original(key)

    monkeypatch.setattr(skeletonize, "simple_point_from_key", recording_simple_point_from_key)

    result = skeletonize_surface(image, max_iterations=1)

    assert result.shape == image.shape
    assert seen_keys
    assert all(isinstance(key, int) for key in seen_keys)


def test_eopen_candidates_use_condition_3_instead_of_simple_point_lookup(monkeypatch):
    image = np.zeros((5, 5, 5), dtype=bool)
    image[2, 2, 2] = True
    coord = np.asarray([[4, 4, 4]], dtype=np.int64)

    monkeypatch.setattr(
        skeletonize,
        "_outer_layer_candidates_vectorized",
        lambda start, protected: (coord, np.asarray([EOPEN], dtype=np.uint8)),
    )
    monkeypatch.setattr(skeletonize, "_is_curve_endpoint", lambda *args: False)
    monkeypatch.setattr(skeletonize, "shape_preserving_point", lambda config: False)
    monkeypatch.setattr(skeletonize, "tunnel_preserving_e_point", lambda config: True)
    monkeypatch.setattr(
        skeletonize,
        "simple_point_from_key",
        lambda key: (_ for _ in ()).throw(AssertionError("e-open branch should not use simple lookup")),
    )

    result = skeletonize_surface(image, max_iterations=1, final_erode=False)

    assert not result[2, 2, 2]


def test_final_erosion_removes_simple_condition_456_points(monkeypatch):
    image = np.zeros((5, 5, 5), dtype=bool)
    image[2, 2, 2] = True

    monkeypatch.setattr(
        skeletonize,
        "_outer_layer_candidates_vectorized",
        lambda start, protected: (np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.uint8)),
    )
    monkeypatch.setattr(skeletonize, "final_erosion_point", lambda config: True)
    monkeypatch.setattr(skeletonize, "simple_point_from_key", lambda key: True)

    result = skeletonize_surface(image, max_iterations=1)

    assert not result[2, 2, 2]
