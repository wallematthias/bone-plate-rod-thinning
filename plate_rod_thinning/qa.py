"""Quality-assurance helpers for inspecting 3-D skeleton outputs."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ComponentSummary:
    """Compact 26-connected component summary for a binary skeleton."""

    voxel_count: int
    component_count: int
    largest_components: tuple[int, ...]


def skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    """Count 26-connected skeleton neighbors for every skeleton voxel."""

    binary = np.asarray(skeleton, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    degree = np.zeros(binary.shape, dtype=np.uint8)
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        degree += padded[
            1 + dx : 1 + dx + binary.shape[0],
            1 + dy : 1 + dy + binary.shape[1],
            1 + dz : 1 + dz + binary.shape[2],
        ]
    degree[~binary] = 0
    return degree


def degree_class_image(skeleton: np.ndarray) -> np.ndarray:
    """Create a simple local-degree QA label image.

    Labels are visualization diagnostics, not the official plate/rod classes:
    0 background, 1 isolated voxel, 2 endpoint, 3 curve-like, 4 branch/sheet-like.
    """

    degree = skeleton_degree(skeleton)
    labels = np.zeros(degree.shape, dtype=np.uint8)
    skel = np.asarray(skeleton, dtype=bool)
    labels[skel & (degree == 0)] = 1
    labels[skel & (degree == 1)] = 2
    labels[skel & (degree == 2)] = 3
    labels[skel & (degree >= 3)] = 4
    return labels


def component_summary(skeleton: np.ndarray) -> tuple[np.ndarray, ComponentSummary]:
    """Label 26-connected skeleton components and summarize their sizes."""

    binary = np.asarray(skeleton, dtype=bool)
    labels, component_id = ndi.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    labels = labels.astype(np.int32, copy=False)
    counts = np.bincount(labels.ravel())
    component_sizes = counts[1:].astype(np.int64, copy=False)

    sizes = tuple(int(value) for value in np.sort(component_sizes)[::-1])
    summary = ComponentSummary(
        voxel_count=int(binary.sum()),
        component_count=int(component_id),
        largest_components=sizes,
    )
    return labels, summary


def dilated_for_visualization(
    skeleton: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    radius: int = 1,
) -> np.ndarray:
    """Thicken a skeleton for display without changing the analysis skeleton."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    binary = np.asarray(skeleton, dtype=bool)
    if radius == 0:
        dilated = binary.copy()
    else:
        dilated = binary.copy()
        for _ in range(radius):
            dilated = _dilate_once(dilated)
    if mask is not None:
        dilated &= np.asarray(mask, dtype=bool)
    return dilated


_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _flood_fill_component(
    binary: np.ndarray,
    labels: np.ndarray,
    start: tuple[int, int, int],
    component_id: int,
) -> int:
    queue: deque[tuple[int, int, int]] = deque([start])
    labels[start] = component_id
    size = 0
    max_x, max_y, max_z = binary.shape

    while queue:
        x, y, z = queue.popleft()
        size += 1
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            nx, ny, nz = neighbor
            if (
                0 <= nx < max_x
                and 0 <= ny < max_y
                and 0 <= nz < max_z
                and binary[neighbor]
                and labels[neighbor] == 0
            ):
                labels[neighbor] = component_id
                queue.append(neighbor)
    return size


def _dilate_once(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    dilated = padded[
        1 : 1 + binary.shape[0],
        1 : 1 + binary.shape[1],
        1 : 1 + binary.shape[2],
    ].copy()
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        dilated |= padded[
            1 + dx : 1 + dx + binary.shape[0],
            1 + dy : 1 + dy + binary.shape[1],
            1 + dz : 1 + dz + binary.shape[2],
        ]
    return dilated
