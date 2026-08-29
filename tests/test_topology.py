import numpy as np
import pytest

import plate_rod_thinning.topology as topology
from plate_rod_thinning.lookup_audit import MATLAB_ROOT, load_lookup_tables
from plate_rod_thinning.topology import (
    BASE_SPOINTS_BY_CLASS,
    MATLAB_SPOINTS,
    SOPEN,
    EOPEN,
    VOPEN,
    border_point_type,
    classify_neighborhood,
    effective_point_indices,
    final_erosion_point,
    generate_classification_lookup,
    generate_simple_point_lookup,
    legacy_border_point_type,
    linear_to_subscript,
    neighborhood_from_key_3x3,
    neighborhood_key_3x3,
    neighborhood_keys_3x3_at,
    neighborhood_keys_3x3,
    initial_class_from_key,
    rotation_to_base_spoints,
    simple_point_from_key,
    simple_point,
    shape_preserving_point,
    subscript_to_linear,
    tunnel_preserving_e_point,
)

requires_matlab_lookup_tables = pytest.mark.skipif(
    not MATLAB_ROOT.exists(),
    reason="MATLAB source lookup tables are local audit data and are not available in CI",
)


def test_matlab_linear_index_mapping_matches_3x3x3_neighborhood():
    assert linear_to_subscript(14, shape=(3, 3, 3)) == (1, 1, 1)
    assert subscript_to_linear((1, 1, 1), shape=(3, 3, 3)) == 14

    spoint_coords = [linear_to_subscript(i, shape=(3, 3, 3)) for i in MATLAB_SPOINTS]
    assert spoint_coords == [
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2),
    ]


def test_matlab_linear_index_mapping_is_cached():
    linear_to_subscript.cache_clear()

    assert linear_to_subscript(14, shape=(3, 3, 3)) == (1, 1, 1)
    assert linear_to_subscript(14, shape=(3, 3, 3)) == (1, 1, 1)

    assert linear_to_subscript.cache_info().hits == 1


def test_matlab_linear_index_mapping_matches_5x5x5_neighborhood():
    assert linear_to_subscript(63, shape=(5, 5, 5)) == (2, 2, 2)
    assert subscript_to_linear((2, 2, 2), shape=(5, 5, 5)) == 63

    sopen_coords = [linear_to_subscript(i, shape=(5, 5, 5)) for i in (38, 58, 62, 64, 68, 88)]
    assert sopen_coords == [
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (3, 2, 2),
        (2, 3, 2),
        (2, 2, 3),
    ]


def test_effective_point_counts_match_saha_class_table():
    expected = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 4, 6: 4, 7: 7, 8: 12, 9: 20}

    for class_id, count in expected.items():
        assert len(effective_point_indices(BASE_SPOINTS_BY_CLASS[class_id])) == count


def test_effective_points_for_class_9_exclude_center_and_s_points():
    effective = set(effective_point_indices(BASE_SPOINTS_BY_CLASS[9]))

    assert 14 not in effective
    assert not effective.intersection(MATLAB_SPOINTS)
    assert len(effective) == 20


def test_rotation_to_base_spoints_is_deterministic():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(11, shape=(3, 3, 3))] = True
    config[linear_to_subscript(17, shape=(3, 3, 3))] = True

    rotated, rotation_index = rotation_to_base_spoints(config, class_id=6)

    assert rotation_index == rotation_to_base_spoints(config, class_id=6)[1]
    assert [bool(rotated[linear_to_subscript(i, shape=(3, 3, 3))]) for i in MATLAB_SPOINTS] == [
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_legacy_border_point_type_matches_outer_layer_examples():
    filled = np.ones((5, 5, 5), dtype=bool)
    assert legacy_border_point_type(filled) == SOPEN

    missing_all_spoints = filled.copy()
    for index in (38, 58, 62, 64, 68, 88):
        missing_all_spoints[linear_to_subscript(index, shape=(5, 5, 5))] = False

    missing_epoint = missing_all_spoints.copy()
    missing_epoint[linear_to_subscript(33, shape=(5, 5, 5))] = False
    assert legacy_border_point_type(missing_epoint) == EOPEN

    missing_vpoint = missing_all_spoints.copy()
    missing_vpoint[linear_to_subscript(32, shape=(5, 5, 5))] = False
    assert legacy_border_point_type(missing_vpoint) == VOPEN


def test_correct_border_point_type_uses_background_s_neighbor_for_sopen():
    filled = np.ones((5, 5, 5), dtype=bool)
    assert border_point_type(filled) == 0

    missing_spoint = filled.copy()
    missing_spoint[linear_to_subscript(38, shape=(5, 5, 5))] = False
    assert border_point_type(missing_spoint) == SOPEN


def test_correct_border_point_type_keeps_eopen_and_vopen_exclusive():
    filled = np.ones((5, 5, 5), dtype=bool)

    missing_epoint = filled.copy()
    missing_epoint[linear_to_subscript(33, shape=(5, 5, 5))] = False
    assert border_point_type(missing_epoint) == EOPEN

    missing_vpoint = filled.copy()
    missing_vpoint[linear_to_subscript(32, shape=(5, 5, 5))] = False
    assert border_point_type(missing_vpoint) == VOPEN

    missing_spoint_and_epoint = missing_epoint.copy()
    missing_spoint_and_epoint[linear_to_subscript(38, shape=(5, 5, 5))] = False
    assert border_point_type(missing_spoint_and_epoint) == SOPEN


def test_shape_preserving_point_identifies_two_voxel_thick_surface():
    config = np.zeros((5, 5, 5), dtype=bool)
    config[2:4, 1:4, 1:4] = True

    assert shape_preserving_point(config)


def test_tunnel_preserving_e_point_ports_saha_condition_3_reference_case():
    config = np.ones((5, 5, 5), dtype=bool)

    assert not tunnel_preserving_e_point(config)


def test_final_erosion_point_ports_saha_conditions_4_to_6_reference_case():
    config = np.zeros((5, 5, 5), dtype=bool)
    for index in (38, 58, 64):
        config[linear_to_subscript(index, shape=(5, 5, 5))] = True

    assert final_erosion_point(config)


@requires_matlab_lookup_tables
def test_generated_classification_lookup_matches_matlab_for_classes_0_to_8():
    matlab = load_lookup_tables("classification")

    for class_id in range(9):
        generated = generate_classification_lookup(class_id)
        assert generated.dtype == np.int64
        assert np.array_equal(generated, matlab[class_id].astype(np.int64))


@requires_matlab_lookup_tables
def test_generated_simple_point_lookup_matches_matlab_for_classes_0_to_8():
    matlab = load_lookup_tables("thinning")

    for class_id in range(9):
        generated = generate_simple_point_lookup(class_id)
        assert generated.dtype == np.int64
        assert np.array_equal(generated, matlab[class_id].astype(np.int64))


def test_simple_point_classification_uses_deterministic_rotation():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(11, shape=(3, 3, 3))] = True
    config[linear_to_subscript(17, shape=(3, 3, 3))] = True

    assert simple_point(config) == simple_point(np.rot90(config, axes=(0, 1)))


def test_simple_point_uses_lookup_for_audited_saha_classes(monkeypatch):
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(14, shape=(3, 3, 3))] = True
    config[linear_to_subscript(5, shape=(3, 3, 3))] = True

    def fail_topology_numbers(*args, **kwargs):
        raise AssertionError("audited lookup should answer this simple-point query")

    monkeypatch.setattr(topology, "topological_numbers", fail_topology_numbers)

    assert simple_point(config)


def test_neighborhood_key_3x3_uses_matlab_linear_order_bits():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(14, shape=(3, 3, 3))] = True
    config[linear_to_subscript(5, shape=(3, 3, 3))] = True

    key = neighborhood_key_3x3(config)

    assert key == (1 << 13) | (1 << 4)
    assert np.array_equal(neighborhood_from_key_3x3(key), config)


def test_simple_point_from_key_matches_array_simple_point():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(14, shape=(3, 3, 3))] = True
    config[linear_to_subscript(5, shape=(3, 3, 3))] = True

    assert simple_point_from_key(neighborhood_key_3x3(config)) == simple_point(config)


def test_initial_class_from_key_matches_array_classification():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(14, shape=(3, 3, 3))] = True
    config[linear_to_subscript(5, shape=(3, 3, 3))] = True
    config[linear_to_subscript(11, shape=(3, 3, 3))] = True

    assert initial_class_from_key(neighborhood_key_3x3(config)) == classify_neighborhood(config).initial_class


def test_neighborhood_keys_3x3_vectorizes_scalar_key_encoding():
    image = np.zeros((5, 5, 5), dtype=bool)
    image[1:4, 1:4, 1:4] = True
    image[0, 2, 2] = True

    keys = neighborhood_keys_3x3(image)

    for coord in ((2, 2, 2), (1, 2, 2), (3, 3, 3)):
        x, y, z = coord
        expected = neighborhood_key_3x3(image[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2])
        assert int(keys[coord]) == expected


def test_neighborhood_keys_3x3_at_encodes_only_requested_coordinates():
    image = np.zeros((6, 6, 6), dtype=bool)
    image[1:5, 1:5, 1:5] = True
    coords = np.asarray([(2, 2, 2), (3, 3, 3)], dtype=np.int64)

    keys = neighborhood_keys_3x3_at(image, coords)

    expected = [
        neighborhood_key_3x3(image[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2])
        for x, y, z in coords
    ]
    assert keys.dtype == np.uint32
    assert keys.tolist() == expected


def test_classify_neighborhood_returns_eps_mu_delta_for_arbitrary_config():
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(5, shape=(3, 3, 3))] = True

    result = classify_neighborhood(config)

    assert result.class_id == 8
    assert result.epsilon == 1
    assert result.mu == 0
    assert result.delta == 0
    assert result.initial_class == 2


def test_classify_neighborhood_avoids_giant_lookup_for_class_9(monkeypatch):
    config = np.zeros((3, 3, 3), dtype=bool)
    config[linear_to_subscript(14, shape=(3, 3, 3))] = True

    def fail_lookup(class_id):
        if class_id == 9:
            raise AssertionError("class 9 should use direct topology, not a million-row lookup")
        return topology.generate_classification_lookup(class_id)

    monkeypatch.setattr(topology, "generate_classification_lookup", fail_lookup)

    result = classify_neighborhood(config)

    assert result.class_id == 9
    assert result.initial_class == 1
