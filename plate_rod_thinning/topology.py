from __future__ import annotations

from functools import cache
from itertools import product
from typing import NamedTuple

import numpy as np
from scipy import ndimage as ndi


MATLAB_SPOINTS = (5, 11, 13, 17, 15, 23)
MATLAB_CENTER = 14
MATLAB_CORNERS = (1, 7, 3, 9, 19, 21, 25, 27)
SOPEN = 1
EOPEN = 2
VOPEN = 3
_SOPEN_5X5_INDICES = (38, 58, 62, 64, 68, 88)

BASE_SPOINTS_BY_CLASS = {
    0: (5, 11, 13, 17, 15, 23),
    1: (5, 11, 13, 17, 15),
    2: (5, 23, 13, 15),
    3: (5, 23, 13, 11),
    4: (5, 23, 13),
    5: (5, 11, 13),
    6: (5, 23),
    7: (5, 13),
    8: (5,),
    9: (),
}


class NeighborhoodClassification(NamedTuple):
    class_id: int
    lookup_key: int
    epsilon: int
    mu: int
    delta: int
    initial_class: int


@cache
def linear_to_subscript(index: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a 1-based MATLAB column-major index to a 0-based subscript."""
    if index < 1 or index > int(np.prod(shape)):
        raise ValueError(f"index {index} is outside shape {shape}")
    return np.unravel_index(index - 1, shape, order="F")


@cache
def subscript_to_linear(subscript: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    """Convert a 0-based subscript to a 1-based MATLAB column-major index."""
    return int(np.ravel_multi_index(subscript, shape, order="F") + 1)


def make_configuration(indices: tuple[int, ...] | list[int]) -> np.ndarray:
    """Create a 3x3x3 boolean neighborhood from 1-based MATLAB linear indices."""
    config = np.zeros((3, 3, 3), dtype=bool)
    for index in indices:
        config[linear_to_subscript(index, config.shape)] = True
    return config


def effective_point_indices(spoints: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return effective 3x3x3 points for a base s-point configuration."""
    config = make_configuration(spoints)
    effective = np.ones((3, 3, 3), dtype=bool)

    for index in (MATLAB_CENTER, *MATLAB_SPOINTS):
        effective[linear_to_subscript(index, effective.shape)] = False

    dead_surfaces = {
        5: range(1, 10),
        11: (1, 2, 3, 10, 11, 12, 19, 20, 21),
        13: (1, 4, 7, 10, 13, 16, 19, 22, 25),
        17: (7, 8, 9, 16, 17, 18, 25, 26, 27),
        15: (3, 6, 9, 12, 15, 18, 21, 24, 27),
        23: range(19, 28),
    }
    for spoint, surface in dead_surfaces.items():
        if config[linear_to_subscript(spoint, config.shape)]:
            for index in surface:
                effective[linear_to_subscript(index, effective.shape)] = False

    return tuple(index for index in range(1, 28) if effective[linear_to_subscript(index, effective.shape)])


@cache
def cube_rotations() -> tuple[np.ndarray, ...]:
    """Return deterministic proper rotations of 3D cube coordinates."""
    rotations = []
    basis = np.eye(3, dtype=int)
    for columns in product((-1, 0, 1), repeat=9):
        matrix = np.array(columns, dtype=int).reshape(3, 3)
        if not np.all(np.sum(np.abs(matrix), axis=0) == 1):
            continue
        if not np.all(np.sum(np.abs(matrix), axis=1) == 1):
            continue
        if round(np.linalg.det(matrix)) != 1:
            continue
        if not any(np.array_equal(matrix, old) for old in rotations):
            rotations.append(matrix)

    rotations.sort(key=lambda item: tuple(item.ravel()))
    assert len(rotations) == 24
    assert any(np.array_equal(rotation, basis) for rotation in rotations)
    return tuple(rotations)


def rotate_configuration(config: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate a 3x3x3 configuration around its center using a rotation matrix."""
    if config.shape != (3, 3, 3):
        raise ValueError("only 3x3x3 configurations are supported")

    out = np.zeros_like(config, dtype=bool)
    for subscript in product(range(3), repeat=3):
        vector = np.array(subscript) - 1
        rotated = tuple((rotation @ vector + 1).tolist())
        out[rotated] = config[subscript]
    return out


def rotation_to_base_spoints(config: np.ndarray, class_id: int) -> tuple[np.ndarray, int]:
    """Rotate a class configuration to its canonical base s-point pattern."""
    base = make_configuration(BASE_SPOINTS_BY_CLASS[class_id])
    base_pattern = _spoint_pattern(base)

    for index, rotation in enumerate(cube_rotations()):
        rotated = rotate_configuration(config, rotation)
        if _spoint_pattern(rotated) == base_pattern:
            return rotated, index
    raise ValueError(f"configuration cannot be rotated to base class {class_id}")


@cache
def generate_classification_lookup(class_id: int) -> np.ndarray:
    """Generate one Saha topo_para classification lookup table."""
    spoints = BASE_SPOINTS_BY_CLASS[class_id]
    base = make_configuration(spoints)
    effective = effective_point_indices(spoints)
    rows = []

    for value in range(2 ** len(effective) - 1, -1, -1):
        bits = _bits(value, len(effective))
        config = base.copy()
        for index, bit in zip(effective, bits, strict=True):
            config[linear_to_subscript(index, config.shape)] = bool(bit)

        eps, mu, delta = topological_numbers(config, force_class0_delta=(class_id == 0))
        rows.append((value, eps, mu, delta))

    return np.asarray(rows, dtype=np.int64)


def classify_neighborhood(config: np.ndarray) -> NeighborhoodClassification:
    """Classify a 3x3x3 neighborhood around a deleted/ignored center point."""
    if config.shape != (3, 3, 3):
        raise ValueError("only 3x3x3 configurations are supported")

    class_id = classify_spoint_configuration(_spoint_pattern(config))
    if class_id == 9:
        epsilon, mu, delta = topological_numbers(config)
        return NeighborhoodClassification(
            class_id=class_id,
            lookup_key=effective_point_key(config, class_id),
            epsilon=epsilon,
            mu=mu,
            delta=delta,
            initial_class=preclassify(epsilon, mu, delta),
        )
    rotated, _ = rotation_to_base_spoints(config, class_id)
    lookup_key = effective_point_key(rotated, class_id)
    row = _lookup_row(generate_classification_lookup(class_id), lookup_key)
    epsilon, mu, delta = (int(row[1]), int(row[2]), int(row[3]))
    return NeighborhoodClassification(
        class_id=class_id,
        lookup_key=lookup_key,
        epsilon=epsilon,
        mu=mu,
        delta=delta,
        initial_class=preclassify(epsilon, mu, delta),
    )


def simple_point(config: np.ndarray) -> bool:
    """Return whether the center point is simple under 26/6 local topology."""
    if config.shape != (3, 3, 3):
        raise ValueError("only 3x3x3 configurations are supported")

    return simple_point_from_key(neighborhood_key_3x3(config))


def neighborhood_key_3x3(config: np.ndarray) -> int:
    """Pack a 3x3x3 boolean neighborhood into MATLAB-linear-order bits."""
    if config.shape != (3, 3, 3):
        raise ValueError("only 3x3x3 configurations are supported")
    flat = np.asarray(config, dtype=bool).ravel(order="F")
    key = 0
    for bit_index, occupied in enumerate(flat):
        if occupied:
            key |= 1 << bit_index
    return key


def neighborhood_from_key_3x3(key: int) -> np.ndarray:
    """Unpack a MATLAB-linear-order 3x3x3 neighborhood key."""
    if key < 0 or key >= (1 << 27):
        raise ValueError("3x3x3 neighborhood key must be in [0, 2**27)")
    flat = np.zeros(27, dtype=bool)
    for bit_index in range(27):
        flat[bit_index] = bool(key & (1 << bit_index))
    return flat.reshape((3, 3, 3), order="F")


def neighborhood_keys_3x3(image: np.ndarray) -> np.ndarray:
    """Pack every 3x3x3 neighborhood in an image into integer keys."""
    binary = np.asarray(image, dtype=bool)
    keys = np.zeros(binary.shape, dtype=np.uint32)
    for bit_index in range(27):
        coord = np.asarray(np.unravel_index(bit_index, (3, 3, 3), order="F"))
        offset = tuple((coord - 1).tolist())
        keys |= _shifted_binary(binary, offset).astype(np.uint32) << bit_index
    return keys


def neighborhood_keys_3x3_at(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Pack 3x3x3 neighborhoods at selected coordinates into integer keys."""
    binary = np.asarray(image, dtype=bool)
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3)")
    keys = np.zeros(len(coords), dtype=np.uint32)
    if len(coords) == 0:
        return keys
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    for bit_index in range(27):
        coord = np.asarray(np.unravel_index(bit_index, (3, 3, 3), order="F"))
        dx, dy, dz = (coord - 1).tolist()
        keys |= binary[x + dx, y + dy, z + dz].astype(np.uint32) << bit_index
    return keys


def simple_point_from_key(key: int) -> bool:
    """Return simple-point status from a packed 3x3x3 neighborhood key."""
    return _simple_point_cached(int(key))


def initial_class_from_key(key: int) -> int:
    """Return Saha initial topology class from a packed 3x3x3 neighborhood key."""
    return _initial_class_cached(int(key))


def initial_classes_from_keys(keys: np.ndarray) -> np.ndarray:
    """Return Saha initial topology classes for packed 3x3x3 neighborhood keys."""
    key_array = np.asarray(keys, dtype=np.uint32)
    if key_array.size == 0:
        return np.zeros(key_array.shape, dtype=np.uint8)
    unique_keys, inverse = np.unique(key_array, return_inverse=True)
    classes = np.asarray([initial_class_from_key(int(key)) for key in unique_keys], dtype=np.uint8)
    return classes[inverse].reshape(key_array.shape)


@cache
def _simple_point_cached(key: int) -> bool:
    config = neighborhood_from_key_3x3(key)
    class_id = classify_spoint_configuration(_spoint_pattern(config))
    if class_id == 0:
        return False
    if class_id < 9:
        rotated, _ = rotation_to_base_spoints(config, class_id)
        lookup_key = effective_point_key(rotated, class_id)
        return _simple_point_lookup(class_id)[lookup_key]

    epsilon, mu, _ = topological_numbers(config)
    return class_id != 0 and bool(config.sum()) and epsilon == 1 and mu == 0


@cache
def _initial_class_cached(key: int) -> int:
    return classify_neighborhood(neighborhood_from_key_3x3(key)).initial_class


def classify_spoint_configuration(spoints: list[int] | tuple[int, ...] | np.ndarray) -> int:
    """Classify a six-s-neighbor occupancy pattern using the Saha classes."""
    c = np.asarray(spoints, dtype=bool)
    if c.shape != (6,):
        raise ValueError("spoints must contain six 6-neighbor occupancy values")

    num_spoints = int(c.sum())
    opposite_pairs = ((0, 5), (2, 4), (1, 3))
    num_opposite = 2 * sum(bool(c[i] and c[j]) for i, j in opposite_pairs)
    num_adjacent = num_spoints - num_opposite

    if num_spoints == 6:
        return 0
    if num_spoints == 5:
        return 1
    if num_opposite == 4:
        return 2
    if num_opposite == 2 and num_adjacent == 2:
        return 3
    if num_opposite == 2 and num_adjacent == 1:
        return 4
    if num_adjacent == 3:
        return 5
    if num_opposite == 2:
        return 6
    if num_adjacent == 2:
        return 7
    if num_adjacent == 1:
        return 8
    if num_spoints == 0:
        return 9
    raise ValueError(f"unclassified s-point configuration: {c.astype(int).tolist()}")


def effective_point_key(config: np.ndarray, class_id: int) -> int:
    """Pack effective-point occupancy for a canonical class configuration."""
    key = 0
    for index in effective_point_indices(BASE_SPOINTS_BY_CLASS[class_id]):
        key = (key << 1) | int(bool(config[linear_to_subscript(index, config.shape)]))
    return key


def preclassify(epsilon: int, mu: int, delta: int) -> int:
    """Map epsilon, mu, delta to the initial topological class N1-N8."""
    if epsilon == 0 and mu == 0 and delta == 0:
        return 1
    if epsilon == 1 and mu == 0 and delta == 0:
        return 2
    if epsilon == 2 and mu == 0 and delta == 0:
        return 3
    if epsilon > 2 and mu == 0 and delta == 0:
        return 4
    if epsilon == 1 and mu == 1 and delta == 0:
        return 5
    if epsilon > 1 and mu >= 1 and delta == 0:
        return 6
    if epsilon == 1 and mu > 1 and delta == 0:
        return 7
    if epsilon == 1 and mu == 0 and delta == 1:
        return 8
    raise ValueError(f"unclassified topological numbers: eps={epsilon}, mu={mu}, delta={delta}")


def legacy_border_point_type(config: np.ndarray) -> int:
    """Port of sk_definitions.m using the original 5x5x5 MATLAB indices."""
    if config.shape != (5, 5, 5):
        raise ValueError("border point definitions require a 5x5x5 configuration")

    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    if any(c(index) for index in _SOPEN_5X5_INDICES):
        return SOPEN

    return _border_point_type_after_sopen(config)


def border_point_type(config: np.ndarray) -> int:
    """Classify an outer-layer point using corrected Saha-style open-point signs.

    The input convention is explicit: ``True`` is object/bone and ``False`` is
    background/marrow. Under that convention, an s-open point has at least one
    background 6-neighbor.
    """
    if config.shape != (5, 5, 5):
        raise ValueError("border point definitions require a 5x5x5 configuration")

    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    if any(not c(index) for index in _SOPEN_5X5_INDICES):
        return SOPEN

    return _border_point_type_after_sopen(config)


def shape_preserving_point(config: np.ndarray) -> bool:
    """Return whether a 5x5x5 point should be preserved during primary thinning."""
    if config.shape != (5, 5, 5):
        raise ValueError("shape preservation requires a 5x5x5 configuration")

    return _shape_preserving_point_cached(_config_cache_key(config))


def tunnel_preserving_e_point(config: np.ndarray) -> bool:
    """Return Saha condition 3 for e-open candidate deletion.

    The legacy MATLAB thinning branch uses this predicate for e-open points
    instead of the simple-point lookup. It is kept separate from
    ``shape_preserving_point`` because it has a different role in the paper
    algorithm: avoiding drilling while allowing e-point erosion.
    """
    if config.shape != (5, 5, 5):
        raise ValueError("condition 3 requires a 5x5x5 configuration")

    rotations = (
        config,
        rot3d90(config, 2),
        rot3d90(config, 3),
    )
    return all(_c3_reference(rotated) for rotated in rotations)


def final_erosion_point(config: np.ndarray) -> bool:
    """Return Saha conditions 4/5/6 for final one-voxel-thinning erosion."""
    if config.shape != (5, 5, 5):
        raise ValueError("final erosion requires a 5x5x5 configuration")

    rotations = (
        config,
        rot3d90(config, 2),
        rot3d90(config, 3),
    )
    checks = [_c456_single_orientation(rotated) for rotated in rotations]
    cond_i = np.asarray([check[0] for check in checks], dtype=bool)
    mbep = np.asarray([check[1] for check in checks], dtype=bool)
    mcfp = np.asarray([check[2] for check in checks], dtype=bool)
    n_cond = int(cond_i.sum())
    if n_cond == 1:
        return bool(mcfp[cond_i].sum() > 0 and mbep[cond_i].sum() > 0)
    if n_cond == 2:
        return bool(mcfp[cond_i].sum() > 1 or mbep[cond_i].sum() > 1)
    if n_cond == 3:
        return True
    return False


@cache
def _shape_preserving_point_cached(config_key: bytes) -> bool:
    config = np.frombuffer(config_key, dtype=np.bool_).reshape((5, 5, 5))
    rotations = (
        config,
        rot3d90(config, 2),
        rot3d90(config, 3),
    )
    return any(_c12_reference(rotated) for rotated in rotations)


def rot3d90(config: np.ndarray, axis: int, turns: int = 1) -> np.ndarray:
    """Rotate a 3D array like the MATLAB rot3d90 helper."""
    turns = turns % 4
    out = config
    for _ in range(turns):
        if axis == 1:
            out = np.flip(np.transpose(out, (0, 2, 1)), axis=1)
        elif axis == 2:
            out = np.transpose(np.flip(np.transpose(out, (1, 2, 0)), axis=1), (1, 0, 2))
        elif axis == 3:
            out = np.transpose(np.flip(out, axis=1), (1, 0, 2))
        else:
            raise ValueError("axis must be 1, 2, or 3")
    return out


def _c12_reference(config: np.ndarray) -> bool:
    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    n = config[1:4, 1:4, 1:4]
    n_sum = n.sum(axis=0) > 0

    en_alt = config[2, 0:4, 0:4]
    cond1 = _encircles(en_alt) and bool(n[0, :, :].sum() > 0) and bool(n[2, :, :].sum() > 0)
    cond2 = (not c(62)) and ((not c(64)) or (not c(65))) and int(n_sum.sum()) == 9
    return bool(cond1 or cond2)


def _c3_reference(config: np.ndarray) -> bool:
    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    all_epoints = c(33) and c(43) and c(83) and c(93)
    middle_plane = config[2, 1:4, 1:4].copy()
    middle_plane[1, 1] = False
    single_component = _count_2d_components(middle_plane) == 1
    no_tunnel = not (c(38) and c(58) and c(68) and c(88))
    return bool((not all_epoints) or (single_component and no_tunnel))


def _c456_single_orientation(config: np.ndarray) -> tuple[bool, bool, bool]:
    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    mbep_plane = config[1:4, 2, 1:4].copy()
    mcfp_plane = config[1:4, 1:4, 2].copy()
    mbep_plane[1, 1] = False
    mcfp_plane[1, 1] = False

    mbep = _count_2d_components(mbep_plane) == 1 and sum(c(index) for index in (38, 62, 64, 88)) != 4
    mcfp = _count_2d_components(mcfp_plane) == 1 and sum(c(index) for index in (58, 62, 64, 68)) != 4
    cond_i = (not c(62)) and (not c(65)) and c(58) and c(38) and c(64)
    return bool(cond_i), bool(mbep), bool(mcfp)


def _count_2d_components(mask: np.ndarray) -> int:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    return int(count)


def _encircles(config_4x4: np.ndarray) -> bool:
    if config_4x4.shape != (4, 4):
        raise ValueError("encircle check requires a 4x4 configuration")
    flat = config_4x4.ravel(order="F")

    def c(index: int) -> bool:
        return bool(flat[index - 1])

    index_sets = (
        (1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 16),
        (1, 2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16),
        (1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 15, 16),
        (5, 6, 7, 8, 9, 12, 13, 14, 15, 16),
        (1, 2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16),
        (2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16),
        (6, 7, 8, 10, 12, 14, 15, 16),
        (2, 3, 4, 6, 8, 10, 12, 14, 15, 16),
    )
    return any(not any(c(index) for index in indices) for indices in index_sets)


def _border_point_type_after_sopen(config: np.ndarray) -> int:
    def c(index: int) -> bool:
        return bool(config[linear_to_subscript(index, config.shape)])

    if (
        (not c(33) and c(13) and c(53))
        or (not c(37) and c(13) and c(61))
        or (not c(39) and c(13) and c(65))
        or (not c(43) and c(13) and c(73))
        or (not c(57) and c(53) and c(61))
        or (not c(59) and c(53) and c(65))
        or (not c(67) and c(61) and c(73))
        or (not c(69) and c(73) and c(65))
        or (not c(83) and c(113) and c(53))
        or (not c(87) and c(113) and c(61))
        or (not c(89) and c(113) and c(65))
        or (not c(93) and c(113) and c(73))
    ):
        return EOPEN

    if (
        (not c(32) and c(13) and c(53) and c(61))
        or (not c(34) and c(13) and c(53) and c(65))
        or (not c(42) and c(13) and c(73) and c(61))
        or (not c(44) and c(13) and c(73) and c(65))
        or (not c(82) and c(113) and c(53) and c(61))
        or (not c(84) and c(113) and c(53) and c(65))
        or (not c(92) and c(113) and c(73) and c(61))
        or (not c(94) and c(113) and c(73) and c(65))
    ):
        return VOPEN

    return 0


@cache
def generate_simple_point_lookup(class_id: int) -> np.ndarray:
    """Generate one simple-point lookup table for thinning."""
    spoints = BASE_SPOINTS_BY_CLASS[class_id]
    base = make_configuration(spoints)
    effective = effective_point_indices(spoints)
    rows = []

    for value in range(2 ** len(effective) - 1, -1, -1):
        bits = _bits(value, len(effective))
        config = base.copy()
        for index, bit in zip(effective, bits, strict=True):
            config[linear_to_subscript(index, config.shape)] = bool(bit)

        eps, mu, _ = topological_numbers(config)
        has_neighbor = bool(config.sum())
        simple = len(spoints) < 6 and has_neighbor and eps == 1 and mu == 0
        rows.append((value, int(simple)))

    return np.asarray(rows, dtype=np.int64)


def compare_generated_lookup_tables(max_class_id: int = 8) -> dict[str, list[str]]:
    """Compare generated Python tables to imported MATLAB lookup tables."""
    from plate_rod_thinning.lookup_audit import load_lookup_tables

    errors: list[str] = []
    checked: list[str] = []
    matlab_classification = load_lookup_tables("classification")
    matlab_thinning = load_lookup_tables("thinning")

    for class_id in range(max_class_id + 1):
        generated_classification = generate_classification_lookup(class_id)
        generated_simple = generate_simple_point_lookup(class_id)
        if not np.array_equal(generated_classification, matlab_classification[class_id].astype(np.int64)):
            errors.append(f"classification lookup mismatch for class {class_id}")
        if not np.array_equal(generated_simple, matlab_thinning[class_id].astype(np.int64)):
            errors.append(f"simple-point lookup mismatch for class {class_id}")
        checked.append(f"class {class_id}")

    return {"checked": checked, "errors": errors}


def topological_numbers(config: np.ndarray, force_class0_delta: bool = False) -> tuple[int, int, int]:
    """Compute local epsilon, mu, delta for a 3x3x3 neighborhood."""
    black = config.copy()
    black[linear_to_subscript(MATLAB_CENTER, black.shape)] = False
    _, epsilon = ndi.label(black, structure=np.ones((3, 3, 3), dtype=bool))

    white_domain = config.copy()
    white_domain[linear_to_subscript(MATLAB_CENTER, white_domain.shape)] = True
    for corner in MATLAB_CORNERS:
        white_domain[linear_to_subscript(corner, white_domain.shape)] = True

    labels, num_labels = ndi.label(~white_domain, structure=_six_connected_structure())
    intersecting = set()
    for spoint in MATLAB_SPOINTS:
        label = labels[linear_to_subscript(spoint, labels.shape)]
        if label:
            intersecting.add(int(label))
    mu = len(intersecting) - 1

    if force_class0_delta:
        mu = 0
    delta = 1 if force_class0_delta else 0
    return int(epsilon), int(mu), delta


def _spoint_pattern(config: np.ndarray) -> tuple[bool, ...]:
    return tuple(bool(config[linear_to_subscript(index, config.shape)]) for index in MATLAB_SPOINTS)


def _bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _lookup_row(table: np.ndarray, key: int) -> np.ndarray:
    matches = table[table[:, 0] == key]
    if len(matches) != 1:
        raise ValueError(f"lookup key {key} has {len(matches)} matches")
    return matches[0]


@cache
def _simple_point_lookup(class_id: int) -> dict[int, bool]:
    return {int(key): bool(value) for key, value in generate_simple_point_lookup(class_id)}


def _config_cache_key(config: np.ndarray) -> bytes:
    return np.ascontiguousarray(config, dtype=np.bool_).tobytes()


def _shifted_binary(image: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(image.shape, dtype=bool)
    source_slices = []
    dest_slices = []
    for axis, delta in enumerate(offset):
        size = image.shape[axis]
        source_start = max(delta, 0)
        source_stop = size + min(delta, 0)
        dest_start = max(-delta, 0)
        dest_stop = size - max(delta, 0)
        source_slices.append(slice(source_start, source_stop))
        dest_slices.append(slice(dest_start, dest_stop))
    out[tuple(dest_slices)] = image[tuple(source_slices)]
    return out


@cache
def _six_connected_structure() -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=bool)
    center = np.array((1, 1, 1))
    for subscript in product(range(3), repeat=3):
        if np.abs(np.array(subscript) - center).sum() <= 1:
            structure[subscript] = True
    return structure
