from __future__ import annotations

import os

import numpy as np

from plate_rod_thinning import metal_backend
from plate_rod_thinning.topology import initial_classes_from_keys as _python_initial_classes_from_keys
from plate_rod_thinning.topology import neighborhood_keys_3x3_at as _python_neighborhood_keys_3x3_at

try:
    from plate_rod_thinning import _c_backend
except Exception:
    _c_backend = None


def backend_name() -> str:
    """Return the active low-level implementation name."""
    base = "compiled" if _c_backend is not None else "python"
    if _use_metal_full() or _use_metal_keys():
        return f"metal+{base}"
    return base


def neighborhood_keys_3x3_at(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Pack selected 3x3x3 neighborhoods using the best available backend."""
    if _use_metal_keys():
        return metal_backend.neighborhood_keys_3x3_at(np.asarray(image, dtype=np.bool_), np.asarray(coords, dtype=np.int64))
    if _c_backend is not None:
        return _c_backend.neighborhood_keys_3x3_at(np.asarray(image, dtype=np.bool_), np.asarray(coords, dtype=np.int64))
    return _python_neighborhood_keys_3x3_at(image, coords)


def initial_classes_from_keys(keys: np.ndarray) -> np.ndarray:
    """Classify packed 3x3x3 neighborhoods using the best available backend."""
    if _c_backend is not None and hasattr(_c_backend, "initial_classes_from_keys"):
        return _c_backend.initial_classes_from_keys(np.asarray(keys, dtype=np.uint32))
    return _python_initial_classes_from_keys(keys)


def propagate_labels_6_connected(bone: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    """Propagate seed labels through a 6-connected binary bone mask."""
    bone_mask = np.asarray(bone, dtype=np.bool_)
    labels = np.asarray(seed_labels, dtype=np.uint8)
    if bone_mask.shape != labels.shape:
        raise ValueError("bone and seed_labels must have the same shape")
    if _c_backend is not None and hasattr(_c_backend, "propagate_labels_6_connected"):
        return _c_backend.propagate_labels_6_connected(bone_mask, labels)
    return _propagate_labels_6_connected_python(bone_mask, labels)


def skeletonize_surface(image: np.ndarray, *, max_iterations: int = 200) -> np.ndarray:
    """Run full topology-preserving thinning using the best available backend."""
    if _use_metal_full():
        return metal_backend.skeletonize_surface(np.asarray(image, dtype=np.bool_), max_iterations=int(max_iterations))
    if _c_backend is not None and hasattr(_c_backend, "skeletonize_surface"):
        return _c_backend.skeletonize_surface(np.asarray(image, dtype=np.bool_), int(max_iterations))
    from plate_rod_thinning.skeletonize import skeletonize_surface as _python_skeletonize_surface

    return _python_skeletonize_surface(image, max_iterations=max_iterations)


def _use_metal_keys() -> bool:
    return os.environ.get("PLATE_ROD_USE_METAL") == "1" and metal_backend.status().available


def _use_metal_full() -> bool:
    return os.environ.get("PLATE_ROD_USE_METAL_FULL") == "1" and metal_backend.status().available


def _propagate_labels_6_connected_python(bone_mask: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    from collections import deque

    output = np.zeros(bone_mask.shape, dtype=np.uint8)
    queue: deque[tuple[int, int, int]] = deque()
    for coord in zip(*np.nonzero(seed_labels), strict=False):
        if not bone_mask[coord]:
            continue
        output[coord] = seed_labels[coord]
        queue.append(coord)

    max_x, max_y, max_z = bone_mask.shape
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            neighbor = (x + dx, y + dy, z + dz)
            nx, ny, nz = neighbor
            if (
                0 <= nx < max_x
                and 0 <= ny < max_y
                and 0 <= nz < max_z
                and bone_mask[neighbor]
                and output[neighbor] == 0
            ):
                output[neighbor] = output[x, y, z]
                queue.append(neighbor)
    return output
