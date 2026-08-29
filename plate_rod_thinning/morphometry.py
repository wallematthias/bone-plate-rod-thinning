from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy import ndimage as ndi

from plate_rod_thinning.classification import ARC_ARC_JUNCTION, SURFACE_CURVE_JUNCTION, SURFACE_SURFACE_JUNCTION
from plate_rod_thinning.backend import propagate_labels_6_connected


PLATE = 1
ROD = 2
JUNCTION = 3


@dataclass(frozen=True)
class TrabeculaComponent:
    component_id: int
    component_type: str
    voxel_count: int
    volume_mm3: float
    thickness_mm: float
    surface_area_mm2: float
    length_mm: float
    principal_axis: tuple[float, float, float]
    axial_alignment: float


@dataclass(frozen=True)
class TrabeculaJunction:
    junction_id: int
    junction_type: str
    plate_components: tuple[int, ...]
    rod_components: tuple[int, ...]
    voxel_count: int
    volume_mm3: float


@dataclass(frozen=True)
class GraphJunction:
    junction_id: int
    junction_type: str
    plate_elements: tuple[int, ...]
    rod_elements: tuple[int, ...]
    voxel_count: int


@dataclass(frozen=True)
class ITSMorphometry:
    component_labels: np.ndarray
    components: tuple[TrabeculaComponent, ...]
    junction_labels: np.ndarray
    junctions: tuple[TrabeculaJunction, ...]
    summary: dict[str, int | float | str]


def compute_its_morphometry(
    *,
    full_labels: np.ndarray,
    skeleton_labels: np.ndarray,
    topology_classes: np.ndarray | None = None,
    analysis_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float] | None = None,
) -> ITSMorphometry:
    """Measure ITS-style plate/rod components and typed junction densities."""
    full = np.asarray(full_labels, dtype=np.uint8)
    skeleton = np.asarray(skeleton_labels, dtype=np.uint8)
    tissue = np.asarray(analysis_mask, dtype=bool)
    if full.shape != skeleton.shape or full.shape != tissue.shape:
        raise ValueError("full_labels, skeleton_labels, and analysis_mask must have the same shape")
    topology = None if topology_classes is None else np.asarray(topology_classes, dtype=np.uint8)
    if topology is not None and topology.shape != skeleton.shape:
        raise ValueError("topology_classes must have the same shape as skeleton_labels")

    spacing = tuple(float(value) for value in (voxel_spacing_mm or (1.0, 1.0, 1.0)))
    voxel_volume = float(np.prod(spacing))
    tv_voxels = int(tissue.sum())
    tv_mm3 = tv_voxels * voxel_volume

    skeleton_element_labels, element_types, skeleton_junction_labels, graph_junctions = _skeleton_graph_elements(skeleton, topology)
    component_labels = _full_thickness_element_labels(full, skeleton_element_labels)
    element_count = int(len(element_types) - 1)

    thickness = 2.0 * ndi.distance_transform_edt(full > 0, sampling=spacing)
    components = _measure_graph_components(
        component_labels=component_labels,
        skeleton_element_labels=skeleton_element_labels,
        element_types=element_types,
        skeleton=skeleton,
        full=full,
        thickness=thickness,
        spacing=spacing,
        voxel_volume=voxel_volume,
    )

    junction_labels = skeleton_junction_labels.astype(np.int32, copy=False)
    junctions = _graph_junctions_to_trabecula_junctions(graph_junctions, voxel_volume)

    plate_components = tuple(component for component in components if component.component_type == "plate")
    rod_components = tuple(component for component in components if component.component_type == "rod")
    plate_volume = sum(component.volume_mm3 for component in plate_components)
    rod_volume = sum(component.volume_mm3 for component in rod_components)
    bone_volume = plate_volume + rod_volume + sum(junction.volume_mm3 for junction in junctions)

    summary = {
        "plate_count": int(len(plate_components)),
        "rod_count": int(len(rod_components)),
        "junction_count": int(len(junctions)),
        "skeleton_graph_element_count": element_count,
        "pTb.N": _number_density(len(plate_components), tv_mm3),
        "rTb.N": _number_density(len(rod_components), tv_mm3),
        "PR.N": _safe_divide(_number_density(len(plate_components), tv_mm3), _number_density(len(rod_components), tv_mm3)),
        "pTb.Th_mm": _mean(component.thickness_mm for component in plate_components),
        "rTb.Th_mm": _mean(component.thickness_mm for component in rod_components),
        "pTb.S_mm2": _mean(component.surface_area_mm2 for component in plate_components),
        "rTb.l_mm": _mean(component.length_mm for component in rod_components),
        "P-P Junc.D": _junction_density(junctions, "P-P", tv_mm3),
        "P-R Junc.D": _junction_density(junctions, "P-R", tv_mm3),
        "R-R Junc.D": _junction_density(junctions, "R-R", tv_mm3),
        "aBV/TV": _safe_divide(
            sum(component.volume_mm3 * component.axial_alignment for component in components),
            tv_mm3,
        ),
        "mean_plate_axial_alignment": _mean(component.axial_alignment for component in plate_components),
        "mean_rod_axial_alignment": _mean(component.axial_alignment for component in rod_components),
        "measured_BV_mm3": bone_volume,
        "measured_pBV_mm3": plate_volume,
        "measured_rBV_mm3": rod_volume,
    }
    return ITSMorphometry(
        component_labels=component_labels,
        components=tuple(components),
        junction_labels=junction_labels.astype(np.int32),
        junctions=junctions,
        summary=summary,
    )


def _measure_component(
    component_id: int,
    component_type: str,
    component: np.ndarray,
    skeleton_class: np.ndarray,
    thickness: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume: float,
) -> TrabeculaComponent:
    coords = np.argwhere(component)
    voxel_count = int(len(coords))
    principal_axis, axial_alignment = _principal_axis(coords, spacing)
    skeleton_component = component & skeleton_class
    return TrabeculaComponent(
        component_id=int(component_id),
        component_type=component_type,
        voxel_count=voxel_count,
        volume_mm3=voxel_count * voxel_volume,
        thickness_mm=float(thickness[component].mean()) if voxel_count else 0.0,
        surface_area_mm2=_voxel_surface_area(component, spacing) if component_type == "plate" else 0.0,
        length_mm=_skeleton_length(skeleton_component, spacing) if component_type == "rod" else 0.0,
        principal_axis=principal_axis,
        axial_alignment=axial_alignment,
    )


def _measure_components(
    *,
    plate_labels: np.ndarray,
    rod_labels: np.ndarray,
    plate_count: int,
    rod_count: int,
    skeleton: np.ndarray,
    thickness: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume: float,
) -> tuple[TrabeculaComponent, ...]:
    plate_voxels = np.bincount(plate_labels.ravel(), minlength=plate_count + 1)
    rod_voxels = np.bincount(rod_labels.ravel(), minlength=rod_count + 1)
    plate_thickness = np.bincount(plate_labels.ravel(), weights=thickness.ravel(), minlength=plate_count + 1)
    rod_thickness = np.bincount(rod_labels.ravel(), weights=thickness.ravel(), minlength=rod_count + 1)
    plate_surface = _surface_areas_by_label(plate_labels, plate_count, spacing)
    rod_lengths = _skeleton_lengths_by_label(rod_labels, skeleton == ROD, rod_count, spacing)
    plate_axes = _principal_axes_by_label(plate_labels, plate_count, spacing)
    rod_axes = _principal_axes_by_label(rod_labels, rod_count, spacing)

    components: list[TrabeculaComponent] = []
    for label_id in range(1, plate_count + 1):
        voxel_count = int(plate_voxels[label_id])
        principal_axis, axial_alignment = plate_axes[label_id]
        components.append(
            TrabeculaComponent(
                component_id=int(label_id),
                component_type="plate",
                voxel_count=voxel_count,
                volume_mm3=voxel_count * voxel_volume,
                thickness_mm=_safe_divide(float(plate_thickness[label_id]), voxel_count),
                surface_area_mm2=float(plate_surface[label_id]),
                length_mm=0.0,
                principal_axis=principal_axis,
                axial_alignment=axial_alignment,
            )
        )
    for label_id in range(1, rod_count + 1):
        voxel_count = int(rod_voxels[label_id])
        principal_axis, axial_alignment = rod_axes[label_id]
        components.append(
            TrabeculaComponent(
                component_id=int(plate_count + label_id),
                component_type="rod",
                voxel_count=voxel_count,
                volume_mm3=voxel_count * voxel_volume,
                thickness_mm=_safe_divide(float(rod_thickness[label_id]), voxel_count),
                surface_area_mm2=0.0,
                length_mm=float(rod_lengths[label_id]),
                principal_axis=principal_axis,
                axial_alignment=axial_alignment,
            )
        )
    return tuple(components)


def _skeleton_graph_elements(
    skeleton_labels: np.ndarray,
    topology_classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[GraphJunction, ...]]:
    skeleton = np.asarray(skeleton_labels, dtype=np.uint8)
    structure = np.ones((3, 3, 3), dtype=bool)
    element_labels = np.zeros(skeleton.shape, dtype=np.int32)
    element_type_chunks = [np.asarray([0], dtype=np.uint8)]
    next_element_id = 1

    for element_type in (PLATE, ROD):
        labels, count = ndi.label(skeleton == element_type, structure=structure)
        if count:
            mask = labels > 0
            element_labels[mask] = labels[mask] + next_element_id - 1
            element_type_chunks.append(np.full(count, element_type, dtype=np.uint8))
            next_element_id += count

    element_types = np.concatenate(element_type_chunks)
    junction_labels, junctions = _typed_junction_clusters(skeleton, topology_classes, element_labels, element_types, structure)
    return element_labels, element_types, junction_labels.astype(np.int32), junctions


def _typed_junction_clusters(
    skeleton: np.ndarray,
    topology_classes: np.ndarray | None,
    element_labels: np.ndarray,
    element_types: np.ndarray,
    structure: np.ndarray,
) -> tuple[np.ndarray, tuple[GraphJunction, ...]]:
    if topology_classes is None:
        junction_labels, junction_count = ndi.label(skeleton == JUNCTION, structure=structure)
        return (
            junction_labels.astype(np.int32),
            _typed_graph_junctions(junction_labels, junction_count, element_labels, element_types),
        )

    topology = np.asarray(topology_classes, dtype=np.uint8)
    typed_masks = (
        ("P-P", topology == SURFACE_SURFACE_JUNCTION),
        ("P-R", topology == SURFACE_CURVE_JUNCTION),
        ("R-R", topology == ARC_ARC_JUNCTION),
    )
    combined_labels = np.zeros(skeleton.shape, dtype=np.int32)
    all_junctions: list[GraphJunction] = []
    next_junction_id = 1
    for forced_type, mask in typed_masks:
        local_labels, local_count = ndi.label(mask & (skeleton == JUNCTION), structure=structure)
        if local_count == 0:
            continue
        active = local_labels > 0
        combined_labels[active] = local_labels[active] + next_junction_id - 1
        local_junctions = _typed_graph_junctions(
            local_labels,
            local_count,
            element_labels,
            element_types,
            forced_type=forced_type,
            first_junction_id=next_junction_id,
        )
        all_junctions.extend(local_junctions)
        next_junction_id += local_count
    return combined_labels, tuple(all_junctions)


def _typed_graph_junctions(
    junction_labels: np.ndarray,
    junction_count: int,
    element_labels: np.ndarray,
    element_types: np.ndarray,
    *,
    forced_type: str | None = None,
    first_junction_id: int = 1,
) -> tuple[GraphJunction, ...]:
    if junction_count == 0:
        return ()

    element_neighbors: list[set[int]] = [set() for _ in range(junction_count + 1)]
    for offset in _OFFSETS_26:
        center_slices, neighbor_slices = _neighbor_pair_slices(junction_labels.shape, offset)
        junction_view = junction_labels[center_slices]
        element_view = element_labels[neighbor_slices]
        neighbor_mask = (junction_view > 0) & (element_view > 0)
        if np.any(neighbor_mask):
            pairs = np.unique(np.column_stack((junction_view[neighbor_mask], element_view[neighbor_mask])), axis=0)
            for junction_id, element_id in pairs:
                element_neighbors[int(junction_id)].add(int(element_id))

    voxel_counts = np.bincount(junction_labels.ravel(), minlength=junction_count + 1)
    junctions: list[GraphJunction] = []
    for junction_id in range(1, junction_count + 1):
        neighbors = element_neighbors[junction_id]
        plate_elements = tuple(sorted(element_id for element_id in neighbors if element_types[element_id] == PLATE))
        rod_elements = tuple(sorted(element_id for element_id in neighbors if element_types[element_id] == ROD))
        global_junction_id = first_junction_id + junction_id - 1
        junctions.append(
            GraphJunction(
                junction_id=int(global_junction_id),
                junction_type=forced_type or _junction_type(plate_elements, rod_elements),
                plate_elements=plate_elements,
                rod_elements=rod_elements,
                voxel_count=int(voxel_counts[junction_id]),
            )
        )
    return tuple(junctions)


def _junction_type(plate_elements: tuple[int, ...], rod_elements: tuple[int, ...]) -> str:
    if plate_elements and rod_elements:
        return "P-R"
    if len(plate_elements) >= 2:
        return "P-P"
    if len(rod_elements) >= 2:
        return "R-R"
    return "untyped"


def _full_thickness_element_labels(full: np.ndarray, skeleton_element_labels: np.ndarray) -> np.ndarray:
    tissue = np.asarray(full) > 0
    seeds = np.asarray(skeleton_element_labels, dtype=np.int32)
    if not tissue.any() or not np.any(seeds):
        return np.zeros(tissue.shape, dtype=np.int32)
    return propagate_labels_6_connected(tissue, seeds).astype(np.int32, copy=False)


def _measure_graph_components(
    *,
    component_labels: np.ndarray,
    skeleton_element_labels: np.ndarray,
    element_types: np.ndarray,
    skeleton: np.ndarray,
    full: np.ndarray,
    thickness: np.ndarray,
    spacing: tuple[float, float, float],
    voxel_volume: float,
) -> tuple[TrabeculaComponent, ...]:
    element_count = int(len(element_types) - 1)
    if element_count <= 0:
        return ()

    element_voxels = np.bincount(component_labels.ravel(), minlength=element_count + 1)
    element_thickness = np.bincount(component_labels.ravel(), weights=thickness.ravel(), minlength=element_count + 1)
    plate_component_labels = component_labels * (full == PLATE)
    rod_component_labels = component_labels * (full == ROD)
    plate_voxels = np.bincount(plate_component_labels.ravel(), minlength=element_count + 1)
    rod_voxels = np.bincount(rod_component_labels.ravel(), minlength=element_count + 1)
    plate_thickness = np.bincount(plate_component_labels.ravel(), weights=thickness.ravel(), minlength=element_count + 1)
    rod_thickness = np.bincount(rod_component_labels.ravel(), weights=thickness.ravel(), minlength=element_count + 1)
    skeleton_lengths = _skeleton_lengths_by_label(skeleton_element_labels, skeleton == ROD, element_count, spacing)
    surface_areas = _skeleton_plate_areas_by_label(skeleton_element_labels, skeleton == PLATE, element_count, spacing)
    axes = _principal_axes_by_label(component_labels, element_count, spacing)
    components: list[TrabeculaComponent] = []

    for element_id in range(1, element_count + 1):
        element_type = int(element_types[element_id])
        if element_type not in (PLATE, ROD):
            continue
        if element_type == PLATE:
            voxel_count = int(plate_voxels[element_id])
            thickness_sum = float(plate_thickness[element_id])
        else:
            voxel_count = int(rod_voxels[element_id])
            thickness_sum = float(rod_thickness[element_id])
        if voxel_count == 0:
            voxel_count = int(element_voxels[element_id])
            thickness_sum = float(element_thickness[element_id])
        principal_axis, axial_alignment = axes[element_id]
        components.append(
            TrabeculaComponent(
                component_id=int(element_id),
                component_type="plate" if element_type == PLATE else "rod",
                voxel_count=voxel_count,
                volume_mm3=voxel_count * voxel_volume,
                thickness_mm=_safe_divide(thickness_sum, voxel_count),
                surface_area_mm2=float(surface_areas[element_id]) if element_type == PLATE else 0.0,
                length_mm=float(skeleton_lengths[element_id]) if element_type == ROD else 0.0,
                principal_axis=principal_axis,
                axial_alignment=axial_alignment,
            )
        )
    return tuple(components)


def _graph_junctions_to_trabecula_junctions(
    graph_junctions: tuple[GraphJunction, ...],
    voxel_volume: float,
) -> tuple[TrabeculaJunction, ...]:
    trabecula_junctions: list[TrabeculaJunction] = []
    next_id = 1
    for junction in graph_junctions:
        for junction_type, plate_components, rod_components in _junction_component_edges(junction):
            trabecula_junctions.append(
                TrabeculaJunction(
                    junction_id=next_id,
                    junction_type=junction_type,
                    plate_components=plate_components,
                    rod_components=rod_components,
                    voxel_count=junction.voxel_count,
                    volume_mm3=junction.voxel_count * voxel_volume,
                )
            )
            next_id += 1
    return tuple(trabecula_junctions)


def _junction_component_edges(
    junction: GraphJunction,
) -> Iterable[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    for left, right in combinations(junction.plate_elements, 2):
        yield "P-P", (left, right), ()
    for plate in junction.plate_elements:
        for rod in junction.rod_elements:
            yield "P-R", (plate,), (rod,)
    for left, right in combinations(junction.rod_elements, 2):
        yield "R-R", (), (left, right)
    if not junction.plate_elements and not junction.rod_elements:
        yield "untyped", (), ()
    elif len(junction.plate_elements) + len(junction.rod_elements) == 1:
        yield (
            junction.junction_type,
            junction.plate_elements,
            junction.rod_elements,
        )


def _skeleton_plate_areas_by_label(
    labels: np.ndarray,
    plate_skeleton: np.ndarray,
    label_count: int,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    areas = np.zeros(label_count + 1, dtype=float)
    if label_count == 0:
        return areas
    voxel_area = float(np.prod(spacing) ** (2.0 / 3.0))
    active_labels = np.asarray(labels, dtype=np.int32)[plate_skeleton]
    if len(active_labels) == 0:
        return areas
    return np.bincount(
        active_labels.ravel(),
        weights=np.full(len(active_labels), voxel_area, dtype=float),
        minlength=label_count + 1,
    )


def _surface_areas_by_label(labels: np.ndarray, label_count: int, spacing: tuple[float, float, float]) -> np.ndarray:
    areas = np.zeros(label_count + 1, dtype=float)
    if label_count == 0:
        return areas
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1, 1:-1]
    sx, sy, sz = spacing
    for neighbor, area in (
        (padded[:-2, 1:-1, 1:-1], sy * sz),
        (padded[2:, 1:-1, 1:-1], sy * sz),
        (padded[1:-1, :-2, 1:-1], sx * sz),
        (padded[1:-1, 2:, 1:-1], sx * sz),
        (padded[1:-1, 1:-1, :-2], sx * sy),
        (padded[1:-1, 1:-1, 2:], sx * sy),
    ):
        exposed = (center > 0) & (neighbor != center)
        if np.any(exposed):
            areas += np.bincount(
                center[exposed].ravel(),
                weights=np.full(np.count_nonzero(exposed), area),
                minlength=label_count + 1,
            )
    return areas


def _skeleton_lengths_by_label(
    labels: np.ndarray,
    skeleton: np.ndarray,
    label_count: int,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    lengths = np.zeros(label_count + 1, dtype=float)
    if label_count == 0:
        return lengths
    spacing_array = np.asarray(spacing, dtype=float)
    for offset in _POSITIVE_OFFSETS_26:
        center_slices, neighbor_slices = _neighbor_pair_slices(labels.shape, offset)
        center_labels = labels[center_slices]
        neighbor_labels = labels[neighbor_slices]
        connected = (
            skeleton[center_slices]
            & skeleton[neighbor_slices]
            & (center_labels > 0)
            & (center_labels == neighbor_labels)
        )
        if np.any(connected):
            length = float(np.linalg.norm(np.asarray(offset, dtype=float) * spacing_array))
            lengths += np.bincount(
                center_labels[connected].ravel(),
                weights=np.full(np.count_nonzero(connected), length),
                minlength=label_count + 1,
            )
    return lengths


def _principal_axes_by_label(
    labels: np.ndarray,
    label_count: int,
    spacing: tuple[float, float, float],
) -> list[tuple[tuple[float, float, float], float]]:
    axes = [((0.0, 0.0, 1.0), 1.0) for _ in range(label_count + 1)]
    if label_count == 0:
        return axes
    coords = np.argwhere(labels > 0)
    if len(coords) == 0:
        return axes
    coord_labels = labels[tuple(coords.T)]
    order = np.argsort(coord_labels, kind="stable")
    coords = coords[order]
    coord_labels = coord_labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(coord_labels)) + 1]
    stops = np.r_[starts[1:], len(coord_labels)]
    for start, stop in zip(starts, stops, strict=True):
        label_id = int(coord_labels[start])
        axes[label_id] = _principal_axis(coords[start:stop], spacing)
    return axes


def _measure_junction(
    junction_id: int,
    junction: np.ndarray,
    plate_labels: np.ndarray,
    rod_labels: np.ndarray,
    plate_count: int,
    voxel_volume: float,
) -> TrabeculaJunction:
    neighbor_plate_ids: set[int] = set()
    neighbor_rod_ids: set[int] = set()
    for dx, dy, dz in _OFFSETS_26:
        shifted_junction = _shifted(junction, (dx, dy, dz))
        neighbor_plate_ids.update(int(value) for value in np.unique(plate_labels[shifted_junction]) if value)
        neighbor_rod_ids.update(int(value) + plate_count for value in np.unique(rod_labels[shifted_junction]) if value)

    if neighbor_plate_ids and neighbor_rod_ids:
        junction_type = "P-R"
    elif len(neighbor_plate_ids) >= 2:
        junction_type = "P-P"
    elif len(neighbor_rod_ids) >= 2:
        junction_type = "R-R"
    elif neighbor_plate_ids:
        junction_type = "P-P"
    elif neighbor_rod_ids:
        junction_type = "R-R"
    else:
        junction_type = "untyped"

    voxel_count = int(junction.sum())
    return TrabeculaJunction(
        junction_id=int(junction_id),
        junction_type=junction_type,
        plate_components=tuple(sorted(neighbor_plate_ids)),
        rod_components=tuple(sorted(neighbor_rod_ids)),
        voxel_count=voxel_count,
        volume_mm3=voxel_count * voxel_volume,
    )


def _measure_junctions(
    junction_labels: np.ndarray,
    junction_count: int,
    plate_labels: np.ndarray,
    rod_labels: np.ndarray,
    plate_count: int,
    voxel_volume: float,
) -> tuple[TrabeculaJunction, ...]:
    if junction_count == 0:
        return ()

    plate_neighbors: list[set[int]] = [set() for _ in range(junction_count + 1)]
    rod_neighbors: list[set[int]] = [set() for _ in range(junction_count + 1)]
    for offset in _OFFSETS_26:
        center_slices, neighbor_slices = _neighbor_pair_slices(junction_labels.shape, offset)
        junction_view = junction_labels[center_slices]

        plate_view = plate_labels[neighbor_slices]
        plate_mask = (junction_view > 0) & (plate_view > 0)
        if np.any(plate_mask):
            pairs = np.unique(np.column_stack((junction_view[plate_mask], plate_view[plate_mask])), axis=0)
            for junction_id, plate_id in pairs:
                plate_neighbors[int(junction_id)].add(int(plate_id))

        rod_view = rod_labels[neighbor_slices]
        rod_mask = (junction_view > 0) & (rod_view > 0)
        if np.any(rod_mask):
            pairs = np.unique(np.column_stack((junction_view[rod_mask], rod_view[rod_mask])), axis=0)
            for junction_id, rod_id in pairs:
                rod_neighbors[int(junction_id)].add(int(rod_id) + plate_count)

    voxel_counts = np.bincount(junction_labels.ravel(), minlength=junction_count + 1)
    junctions = []
    for junction_id in range(1, junction_count + 1):
        neighbor_plate_ids = plate_neighbors[junction_id]
        neighbor_rod_ids = rod_neighbors[junction_id]
        if neighbor_plate_ids and neighbor_rod_ids:
            junction_type = "P-R"
        elif len(neighbor_plate_ids) >= 2:
            junction_type = "P-P"
        elif len(neighbor_rod_ids) >= 2:
            junction_type = "R-R"
        elif neighbor_plate_ids:
            junction_type = "P-P"
        elif neighbor_rod_ids:
            junction_type = "R-R"
        else:
            junction_type = "untyped"
        voxel_count = int(voxel_counts[junction_id])
        junctions.append(
            TrabeculaJunction(
                junction_id=int(junction_id),
                junction_type=junction_type,
                plate_components=tuple(sorted(neighbor_plate_ids)),
                rod_components=tuple(sorted(neighbor_rod_ids)),
                voxel_count=voxel_count,
                volume_mm3=voxel_count * voxel_volume,
            )
        )
    return tuple(junctions)


def _neighbor_pair_slices(shape: tuple[int, int, int], offset: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    center_slices = []
    neighbor_slices = []
    for size, delta in zip(shape, offset, strict=True):
        if delta >= 0:
            center_slices.append(slice(0, size - delta))
            neighbor_slices.append(slice(delta, size))
        else:
            center_slices.append(slice(-delta, size))
            neighbor_slices.append(slice(0, size + delta))
    return tuple(center_slices), tuple(neighbor_slices)


def _number_density(count: int, tv_mm3: float) -> float:
    return float((count / tv_mm3) ** (1.0 / 3.0)) if tv_mm3 > 0 and count > 0 else 0.0


def _junction_density(junctions: tuple[TrabeculaJunction, ...], junction_type: str, tv_mm3: float) -> float:
    return _safe_divide(sum(1 for junction in junctions if junction.junction_type == junction_type), tv_mm3)


def _mean(values) -> float:
    materialized = [float(value) for value in values]
    return float(np.mean(materialized)) if materialized else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _principal_axis(coords: np.ndarray, spacing: tuple[float, float, float]) -> tuple[tuple[float, float, float], float]:
    if len(coords) < 2:
        return (0.0, 0.0, 1.0), 1.0
    physical = coords.astype(float) * np.asarray(spacing, dtype=float)
    centered = physical - physical.mean(axis=0)
    covariance = centered.T @ centered / len(coords)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[2] < 0:
        axis = -axis
    axis_tuple = tuple(float(value) for value in axis)
    return axis_tuple, float(abs(axis[2]))


def _voxel_surface_area(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return 0.0

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    sx, sy, sz = spacing
    return float(
        np.count_nonzero(center & ~padded[:-2, 1:-1, 1:-1]) * sy * sz
        + np.count_nonzero(center & ~padded[2:, 1:-1, 1:-1]) * sy * sz
        + np.count_nonzero(center & ~padded[1:-1, :-2, 1:-1]) * sx * sz
        + np.count_nonzero(center & ~padded[1:-1, 2:, 1:-1]) * sx * sz
        + np.count_nonzero(center & ~padded[1:-1, 1:-1, :-2]) * sx * sy
        + np.count_nonzero(center & ~padded[1:-1, 1:-1, 2:]) * sx * sy
    )


def _skeleton_length(skeleton: np.ndarray, spacing: tuple[float, float, float]) -> float:
    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        return 0.0
    skel = np.asarray(skeleton, dtype=bool)
    length = 0.0
    spacing_array = np.asarray(spacing, dtype=float)
    for dx, dy, dz in _POSITIVE_OFFSETS_26:
        connected = skel & _shifted(skel, (dx, dy, dz))
        length += float(np.count_nonzero(connected) * np.linalg.norm(np.asarray((dx, dy, dz), dtype=float) * spacing_array))
    return length


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


_OFFSETS_26 = tuple(offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0))
_POSITIVE_OFFSETS_26 = tuple(offset for offset in _OFFSETS_26 if offset > (0, 0, 0))
