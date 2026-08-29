from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage as ndi

from plate_rod_thinning.backend import neighborhood_keys_3x3_at
from plate_rod_thinning.topology import (
    EOPEN,
    SOPEN,
    VOPEN,
    border_point_type,
    final_erosion_point,
    linear_to_subscript,
    shape_preserving_point,
    simple_point_from_key,
    tunnel_preserving_e_point,
)


@dataclass(frozen=True)
class SkeletonizationResult:
    skeleton: np.ndarray
    iterations: int
    removed_voxels: int


def skeletonize_surface(
    image: np.ndarray,
    *,
    max_iterations: int = 200,
    final_erode: bool = True,
    return_result: bool = False,
) -> np.ndarray | SkeletonizationResult:
    """Topology-preserving, surface-retaining thinning for 3D binary images.

    This is the first corrected Python driver. It follows the Saha-style
    two-version thinning structure, but it deliberately uses the corrected
    open-point convention from ``border_point_type`` rather than the legacy
    MATLAB sign convention.
    """
    if image.ndim != 3:
        raise ValueError("skeletonize_surface expects a 3D array")

    current = np.pad(np.asarray(image, dtype=bool), 2)
    protected = np.zeros_like(current, dtype=bool)
    total_removed = 0
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        start = current.copy()
        removed_this_iteration = 0
        candidate_coords, candidate_types = _outer_layer_candidates_vectorized(start, protected)

        if len(candidate_coords) == 0:
            break

        for point_type in (1, 2, 3):
            typed = candidate_coords[candidate_types == point_type]
            for parity in product((0, 1), repeat=3):
                subfield = typed[np.all(typed % 2 == parity, axis=1)]
                if len(subfield) == 0:
                    continue
                simple_keys = (
                    neighborhood_keys_3x3_at(current, subfield) if point_type != EOPEN else np.zeros(len(subfield), dtype=np.uint32)
                )
                for (x, y, z), simple_key in zip(subfield, simple_keys, strict=True):
                    if not current[x, y, z] or protected[x, y, z]:
                        continue
                    if _is_curve_endpoint(current, x, y, z):
                        protected[x, y, z] = True
                        continue
                    if shape_preserving_point(start[x - 2 : x + 3, y - 2 : y + 3, z - 2 : z + 3]):
                        protected[x, y, z] = True
                        continue
                    if point_type == EOPEN and tunnel_preserving_e_point(start[x - 2 : x + 3, y - 2 : y + 3, z - 2 : z + 3]):
                        current[x, y, z] = False
                        removed_this_iteration += 1
                    elif point_type != EOPEN and simple_point_from_key(int(simple_key)):
                        current[x, y, z] = False
                        removed_this_iteration += 1

        total_removed += removed_this_iteration
        if removed_this_iteration == 0:
            break

    if final_erode:
        final_removed = _final_erode(current)
        total_removed += final_removed

    skeleton = current[2:-2, 2:-2, 2:-2]
    if return_result:
        return SkeletonizationResult(skeleton=skeleton, iterations=iterations, removed_voxels=total_removed)
    return skeleton


def _outer_layer_candidates(image: np.ndarray, protected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighborhood_count = ndi.convolve(
        image.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    candidate_mask = image & ~protected & (neighborhood_count < 27)
    coords = np.argwhere(candidate_mask)
    point_types = np.zeros(len(coords), dtype=np.uint8)

    for i, (x, y, z) in enumerate(coords):
        point_types[i] = border_point_type(image[x - 2 : x + 3, y - 2 : y + 3, z - 2 : z + 3])

    keep = point_types > 0
    return coords[keep], point_types[keep]


def _outer_layer_candidates_vectorized(image: np.ndarray, protected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighborhood_count = ndi.convolve(
        image.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    candidate_mask = image & ~protected & (neighborhood_count < 27)
    point_types = _border_point_types_vectorized(image)
    keep = candidate_mask & (point_types > 0)
    return np.argwhere(keep), point_types[keep]


def _border_point_types_vectorized(image: np.ndarray) -> np.ndarray:
    def c(index: int) -> np.ndarray:
        coord = np.asarray(linear_to_subscript(index, (5, 5, 5)))
        offset = tuple((coord - 2).tolist())
        return _shifted(image, offset)

    point_types = np.zeros(image.shape, dtype=np.uint8)
    sopen = np.zeros(image.shape, dtype=bool)
    for index in (38, 58, 62, 64, 68, 88):
        sopen |= ~c(index)
    point_types[sopen] = SOPEN

    not_sopen = ~sopen
    eopen = (
        (~c(33) & c(13) & c(53))
        | (~c(37) & c(13) & c(61))
        | (~c(39) & c(13) & c(65))
        | (~c(43) & c(13) & c(73))
        | (~c(57) & c(53) & c(61))
        | (~c(59) & c(53) & c(65))
        | (~c(67) & c(61) & c(73))
        | (~c(69) & c(73) & c(65))
        | (~c(83) & c(113) & c(53))
        | (~c(87) & c(113) & c(61))
        | (~c(89) & c(113) & c(65))
        | (~c(93) & c(113) & c(73))
    )
    point_types[not_sopen & eopen] = EOPEN

    unopened = not_sopen & ~eopen
    vopen = (
        (~c(32) & c(13) & c(53) & c(61))
        | (~c(34) & c(13) & c(53) & c(65))
        | (~c(42) & c(13) & c(73) & c(61))
        | (~c(44) & c(13) & c(73) & c(65))
        | (~c(82) & c(113) & c(53) & c(61))
        | (~c(84) & c(113) & c(53) & c(65))
        | (~c(92) & c(113) & c(73) & c(61))
        | (~c(94) & c(113) & c(73) & c(65))
    )
    point_types[unopened & vopen] = VOPEN
    return point_types


def _shifted(image: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
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


def _is_curve_endpoint(image: np.ndarray, x: int, y: int, z: int) -> bool:
    neighborhood = image[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
    return int(neighborhood.sum()) <= 2


def _final_erode(image: np.ndarray) -> int:
    coords = np.argwhere(image)
    if len(coords) == 0:
        return 0
    interior = np.all((coords >= 2) & (coords < (np.asarray(image.shape) - 2)), axis=1)
    coords = coords[interior]
    keys = neighborhood_keys_3x3_at(image, coords) if len(coords) else np.zeros(0, dtype=np.uint32)
    delete = []
    for (x, y, z), key in zip(coords, keys, strict=True):
        if final_erosion_point(image[x - 2 : x + 3, y - 2 : y + 3, z - 2 : z + 3]) and simple_point_from_key(int(key)):
            delete.append((x, y, z))
    if not delete:
        return 0
    delete_coords = np.asarray(delete, dtype=np.int64)
    image[delete_coords[:, 0], delete_coords[:, 1], delete_coords[:, 2]] = False
    return len(delete)
