from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from plate_rod_thinning.backend import initial_classes_from_keys, neighborhood_keys_3x3_at
from plate_rod_thinning.topology import classify_neighborhood


BACKGROUND = 0
SURFACE_ENDPOINT = 1
SURFACE_INNER = 2
SURFACE_SURFACE_JUNCTION = 3
SURFACE_CURVE_JUNCTION = 4
ARC_ENDPOINT = 5
ARC_INNER = 6
ARC_ARC_JUNCTION = 7
ISOLATED = 8

PLATE = 1
ROD = 2
JUNCTION = 3


def classify_skeleton_topology(skeleton: np.ndarray) -> np.ndarray:
    """Classify skeleton voxels using Saha preclasses and MATLAB final rules."""
    binary = np.asarray(skeleton, dtype=bool)
    if binary.ndim != 3:
        raise ValueError("classify_skeleton_topology expects a 3D array")

    padded = np.pad(binary, 1)
    pre = np.zeros(padded.shape, dtype=np.uint8)
    coords = np.argwhere(padded)
    if len(coords) != 0:
        keys = neighborhood_keys_3x3_at(padded, coords)
        pre[tuple(coords.T)] = initial_classes_from_keys(keys)

    nhood = np.ones((3, 3, 3), dtype=np.uint8)
    nhood[1, 1, 1] = 0
    occupied = pre > 0
    n3n4_voxel = (pre == 3) | (pre == 4)
    one_asp = ndi.convolve(occupied.astype(np.uint8), nhood, mode="constant", cval=0) == 1
    some_asp = ndi.convolve(occupied.astype(np.uint8), nhood, mode="constant", cval=0) > 1
    bone_neighbors = ndi.convolve(occupied.astype(np.uint8), nhood, mode="constant", cval=0)
    n3n4_neighbors = ndi.convolve(n3n4_voxel.astype(np.uint8), nhood, mode="constant", cval=0)
    all_n3n4 = (n3n4_neighbors == bone_neighbors) & occupied
    some_n3n4 = n3n4_neighbors != 0
    no_n3n4 = n3n4_neighbors == 0

    final = np.zeros_like(pre)
    n2 = pre == 2
    n5 = pre == 5
    n6 = pre == 6
    n7 = pre == 7
    n8 = pre == 8

    final[n2 & some_asp] = SURFACE_ENDPOINT
    final[n5 & ~all_n3n4] = SURFACE_INNER
    final[(n6 | n7 | n8) & no_n3n4] = SURFACE_SURFACE_JUNCTION
    final[(n6 | n7 | n8) & some_n3n4] = SURFACE_CURVE_JUNCTION
    final[n2 & one_asp] = ARC_ENDPOINT
    final[pre == 3] = ARC_INNER
    final[pre == 4] = ARC_ARC_JUNCTION
    final[n5 & all_n3n4] = ARC_ARC_JUNCTION
    final[(n6 | n7 | n8) & all_n3n4] = ARC_ARC_JUNCTION
    final[pre == 1] = ISOLATED

    return final[1:-1, 1:-1, 1:-1]


def plate_rod_labels_from_topology(topology_classes: np.ndarray) -> np.ndarray:
    """Collapse final topology classes into plate, rod, and junction labels."""
    topology = np.asarray(topology_classes, dtype=np.uint8)
    labels = np.zeros(topology.shape, dtype=np.uint8)
    labels[(topology == SURFACE_ENDPOINT) | (topology == SURFACE_INNER)] = PLATE
    labels[
        (topology == ARC_ENDPOINT)
        | (topology == ARC_INNER)
        | (topology == ISOLATED)
        | (topology == SURFACE_SURFACE_JUNCTION)
        | (topology == SURFACE_CURVE_JUNCTION)
        | (topology == ARC_ARC_JUNCTION)
    ] = ROD
    labels[(topology == SURFACE_SURFACE_JUNCTION) | (topology == SURFACE_CURVE_JUNCTION) | (topology == ARC_ARC_JUNCTION)] = JUNCTION
    return labels
