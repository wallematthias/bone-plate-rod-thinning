"""High-level plate/rod analysis pipeline.

This module is the boundary Slicer should call. It keeps the thinning,
topological classification, full-thickness recovery, and ITS-style summary
metrics in the reusable core package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plate_rod_thinning.classification import classify_skeleton_topology, plate_rod_labels_from_topology
from plate_rod_thinning.morphometry import compute_its_morphometry
from plate_rod_thinning.qa import component_summary, skeleton_degree
from plate_rod_thinning.backend import propagate_labels_6_connected, skeletonize_surface


BACKGROUND = 0
PLATE = 1
ROD = 2
JUNCTION = 3


@dataclass(frozen=True)
class PlateRodParameters:
    """User-facing knobs for plate/rod analysis."""

    slenderness: int = 0
    min_plate_voxels: int = 0
    min_rod_voxels: int = 0
    skeletonize: bool = True
    crop_to_mask: bool = True
    max_iterations: int = 200
    voxel_spacing_mm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PlateRodResult:
    """Core outputs consumed by Slicer and batch workflows."""

    skeleton: np.ndarray
    topology_labels: np.ndarray
    skeleton_labels: np.ndarray
    full_thickness_labels: np.ndarray
    component_labels: np.ndarray
    summary: dict[str, int | float | str]
    parameters: PlateRodParameters = field(default_factory=PlateRodParameters)


def classify_skeleton_preview(skeleton: np.ndarray, *, slenderness: int = 0) -> np.ndarray:
    """Classify skeleton voxels as plate-like or rod-like using local degree.

    This is a deterministic preview classifier. Final Saha/Stauber topology
    classes should replace it behind the same public API.
    """

    if slenderness < 0:
        raise ValueError("slenderness must be non-negative")

    binary = np.asarray(skeleton, dtype=bool)
    labels = np.zeros(binary.shape, dtype=np.uint8)
    degree = skeleton_degree(binary)
    labels[binary & (degree >= 3)] = PLATE
    labels[binary & (degree >= 1) & (degree <= 2)] = ROD
    labels[binary & (degree == 0)] = ROD

    for _ in range(slenderness):
        labels = _reduce_slender_plate_edges(labels)
    return labels


def label_full_thickness(bone: np.ndarray, skeleton_labels: np.ndarray) -> np.ndarray:
    """Propagate skeleton labels back into the original trabecular thickness."""

    bone_mask = np.asarray(bone, dtype=bool)
    seed_labels = np.asarray(skeleton_labels, dtype=np.uint8)
    if bone_mask.shape != seed_labels.shape:
        raise ValueError("bone and skeleton_labels must have the same shape")

    return propagate_labels_6_connected(bone_mask, seed_labels)


def plate_rod_analysis(
    bone: np.ndarray,
    *,
    analysis_mask: np.ndarray | None = None,
    parameters: PlateRodParameters | None = None,
) -> PlateRodResult:
    """Run the current plate/rod pipeline on a binary trabecular bone mask."""

    params = parameters or PlateRodParameters()
    bone_mask = np.asarray(bone, dtype=bool)
    tissue_mask = np.asarray(analysis_mask, dtype=bool) if analysis_mask is not None else np.ones(bone_mask.shape, dtype=bool)
    if tissue_mask.shape != bone_mask.shape:
        raise ValueError("analysis_mask and bone must have the same shape")
    skeleton = (
        _skeletonize_bounded(
            bone_mask,
            crop_to_mask=params.crop_to_mask,
            max_iterations=params.max_iterations,
        )
        if params.skeletonize
        else bone_mask.copy()
    )
    topology_classes = classify_skeleton_topology(skeleton)
    skeleton_labels = plate_rod_labels_from_topology(topology_classes)
    for _ in range(params.slenderness):
        skeleton_labels = _reduce_slender_plate_edges(skeleton_labels)
    skeleton_labels = _remove_small_classes(
        skeleton_labels,
        min_plate_voxels=params.min_plate_voxels,
        min_rod_voxels=params.min_rod_voxels,
    )
    full_labels = label_full_thickness(bone_mask, skeleton_labels)
    its = compute_its_morphometry(
        full_labels=full_labels,
        skeleton_labels=skeleton_labels,
        analysis_mask=tissue_mask,
        voxel_spacing_mm=params.voxel_spacing_mm,
    )
    _, components = component_summary(skeleton_labels > 0)
    bone_voxels = int(bone_mask.sum())
    tissue_voxels = int(tissue_mask.sum())
    plate_full_voxels = int(np.count_nonzero(full_labels == PLATE))
    rod_full_voxels = int(np.count_nonzero(full_labels == ROD))
    summary = _volume_summary(
        bone_voxels=bone_voxels,
        tissue_voxels=tissue_voxels,
        plate_voxels=plate_full_voxels,
        rod_voxels=rod_full_voxels,
        voxel_spacing_mm=params.voxel_spacing_mm,
    )

    summary.update({
        "classifier": "saha_topology",
        "bone_voxels": bone_voxels,
        "tissue_voxels": tissue_voxels,
        "skeleton_voxels": int(skeleton.sum()),
        "plate_skeleton_voxels": int(np.count_nonzero(skeleton_labels == PLATE)),
        "rod_skeleton_voxels": int(np.count_nonzero(skeleton_labels == ROD)),
        "plate_full_thickness_voxels": plate_full_voxels,
        "rod_full_thickness_voxels": rod_full_voxels,
        "component_count_26_connected": int(components.component_count),
        "largest_component_voxels": int(components.largest_components[0]) if components.largest_components else 0,
        "slenderness": int(params.slenderness),
        "max_iterations": int(params.max_iterations),
    })
    summary.update(its.summary)
    return PlateRodResult(
        skeleton=skeleton,
        topology_labels=topology_classes,
        skeleton_labels=skeleton_labels,
        full_thickness_labels=full_labels,
        component_labels=its.component_labels,
        summary=summary,
        parameters=params,
    )


def _skeletonize_bounded(bone_mask: np.ndarray, *, crop_to_mask: bool, max_iterations: int) -> np.ndarray:
    if not crop_to_mask:
        return skeletonize_surface(bone_mask, max_iterations=max_iterations)
    bounds = _mask_bounds(bone_mask)
    if bounds is None:
        return np.zeros_like(bone_mask, dtype=bool)
    cropped = bone_mask[bounds]
    cropped_skeleton = skeletonize_surface(cropped, max_iterations=max_iterations)
    skeleton = np.zeros_like(bone_mask, dtype=bool)
    skeleton[bounds] = cropped_skeleton
    return skeleton


def _volume_summary(
    *,
    bone_voxels: int,
    tissue_voxels: int,
    plate_voxels: int,
    rod_voxels: int,
    voxel_spacing_mm: tuple[float, float, float] | None,
) -> dict[str, int | float]:
    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    summary: dict[str, int | float] = {
        "BV/TV": fraction(bone_voxels, tissue_voxels),
        "pBV_voxels": int(plate_voxels),
        "rBV_voxels": int(rod_voxels),
        "pBV/TV": fraction(plate_voxels, tissue_voxels),
        "rBV/TV": fraction(rod_voxels, tissue_voxels),
        "pBV/BV": fraction(plate_voxels, bone_voxels),
        "rBV/BV": fraction(rod_voxels, bone_voxels),
        "PR_volume_ratio": fraction(plate_voxels, rod_voxels),
        "pBV/rBV": fraction(plate_voxels, rod_voxels),
    }
    if voxel_spacing_mm is not None:
        voxel_volume = float(np.prod(tuple(float(value) for value in voxel_spacing_mm)))
        summary.update(
            {
                "voxel_volume_mm3": voxel_volume,
                "TV_mm3": tissue_voxels * voxel_volume,
                "BV_mm3": bone_voxels * voxel_volume,
                "pBV_mm3": plate_voxels * voxel_volume,
                "rBV_mm3": rod_voxels * voxel_volume,
            }
        )
    return summary


def _mask_bounds(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return None
    return tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)


def _reduce_slender_plate_edges(labels: np.ndarray) -> np.ndarray:
    plate = labels == PLATE
    plate_degree = _degree_6(plate)
    out = labels.copy()
    slender_plate_edge = plate & (plate_degree <= 1)
    out[slender_plate_edge] = ROD
    return out


def _remove_small_classes(
    labels: np.ndarray,
    *,
    min_plate_voxels: int,
    min_rod_voxels: int,
) -> np.ndarray:
    if min_plate_voxels < 0 or min_rod_voxels < 0:
        raise ValueError("minimum class sizes must be non-negative")
    out = labels.copy()
    if min_plate_voxels:
        out = _remove_small_label_components(out, PLATE, min_plate_voxels)
    if min_rod_voxels:
        out = _remove_small_label_components(out, ROD, min_rod_voxels)
    return out


def _remove_small_label_components(labels: np.ndarray, label: int, minimum_size: int) -> np.ndarray:
    component_labels, components = component_summary(labels == label)
    if components.component_count == 0:
        return labels
    counts = np.bincount(component_labels.ravel())
    remove_component = counts < minimum_size
    remove_component[0] = False
    out = labels.copy()
    out[remove_component[component_labels]] = BACKGROUND
    return out


_NEIGHBOR_OFFSETS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def _degree_6(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    degree = np.zeros(binary.shape, dtype=np.uint8)
    for dx, dy, dz in _NEIGHBOR_OFFSETS_6:
        degree += padded[
            1 + dx : 1 + dx + binary.shape[0],
            1 + dy : 1 + dy + binary.shape[1],
            1 + dz : 1 + dz + binary.shape[2],
        ]
    degree[~binary] = 0
    return degree
