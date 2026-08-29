"""Derivative-writing batch workflow for plate/rod morphometry."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bone_imaging_derivatives import (
    DerivativeManifest,
    DerivativeProgressEvent,
    DerivativeRecord,
    discover_manifests,
    find_records,
    write_manifest,
)
from bone_imaging_derivatives.layout import record_output_path

from .pipeline import PlateRodParameters, plate_rod_analysis


_FAMILY = "PlateRodMorphometry"
_MASK_ROLES = ("trabecular_mask", "bone_segmentation")
_FILENAME = re.compile(
    r"(?:sub-(?P<subject>[^_]+)_)?(?:ses-(?P<session>[^_]+)_)?(?:site-(?P<site>[^_]+)_)?"
    r".*(?:trab(?:ecular)?|bone).*\.npy$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BatchWorkflowResult:
    """Artifacts created (or planned) by a plate/rod batch run."""

    manifest: DerivativeManifest
    records: tuple[DerivativeRecord, ...]
    output_root: Path


@dataclass(frozen=True)
class _BatchInput:
    bone: DerivativeRecord
    common_region: DerivativeRecord | None


def run_plate_rod_batch(
    dataset_root: Path,
    *,
    subject_id: str | None = None,
    site: str | None = None,
    output_root: Path | None = None,
    manifests: Sequence[DerivativeManifest] | None = None,
    generate_missing: bool = False,
    force: bool = False,
    dry_run: bool = False,
    progress: Callable[[DerivativeProgressEvent], None] | None = None,
    use_common_region: bool = True,
    parameters: PlateRodParameters | None = None,
) -> BatchWorkflowResult:
    """Measure manifest-discovered trabecular masks and write derivatives.

    ``generate_missing`` is accepted for the shared batch-workflow shape but
    cannot create segmentation or common-region prerequisites in this package.
    Unmanifested ``.npy`` trabecular/bone masks are supported as a small,
    portable fallback for command-line fixtures and simple datasets.
    """
    del generate_missing, force
    root = Path(dataset_root).resolve()
    destination = Path(output_root).resolve() if output_root is not None else root / "derivatives" / _FAMILY
    known_manifests = list(discover_manifests(root) if manifests is None else manifests)
    inputs = _discover_inputs(root, known_manifests, subject_id=subject_id, site=site, use_common_region=use_common_region)
    records: list[DerivativeRecord] = []
    for item in inputs:
        _emit(progress, item.bone, "measure", "started", "Running plate/rod morphometry")
        bone = _load_mask(item.bone.path)
        common = _load_mask(item.common_region.path) if item.common_region is not None else None
        if common is not None and common.shape != bone.shape:
            raise ValueError(f"Common-region mask shape differs from bone mask: {item.common_region.path}")
        # The common scan region limits the actual skeletonization/classification
        # input, not merely downstream summary denominators.
        effective_bone = bone if common is None else bone & common
        result = plate_rod_analysis(effective_bone, analysis_mask=common, parameters=parameters)
        case_records = _output_records(root, item.bone, item.common_region)
        if not dry_run:
            _write_outputs(case_records, result)
        records.extend(case_records)
        _emit(progress, item.bone, "measure", "completed", "Wrote plate/rod derivative outputs")

    manifest = DerivativeManifest.create(
        _FAMILY,
        root,
        {"name": "plate-rod-thinning", "version": "0.1.3"},
        tuple(records),
    )
    if not dry_run:
        write_manifest(manifest, destination / "manifest.json")
    return BatchWorkflowResult(manifest=manifest, records=tuple(records), output_root=destination)


def _discover_inputs(
    root: Path,
    manifests: Sequence[DerivativeManifest],
    *,
    subject_id: str | None,
    site: str | None,
    use_common_region: bool,
) -> list[_BatchInput]:
    masks = [
        record
        for role in _MASK_ROLES
        for record in find_records(manifests, role=role, subject_id=subject_id, site=site, space="native")
    ]
    if not masks:
        masks = _filename_fallback(root, subject_id=subject_id, site=site)
    common_regions = find_records(
        manifests,
        derivative="CommonRegion",
        role="scan_region_native_common",
        subject_id=subject_id,
        site=site,
        space="native",
    )
    inputs: list[_BatchInput] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for mask in masks:
        key = (mask.subject_id, mask.site, mask.session_id, str(mask.path))
        if key in seen:
            continue
        seen.add(key)
        common = next(
            (record for record in common_regions if _same_case(record, mask)), None
        ) if use_common_region else None
        inputs.append(_BatchInput(mask, common))
    return inputs


def _filename_fallback(root: Path, *, subject_id: str | None, site: str | None) -> list[DerivativeRecord]:
    records: list[DerivativeRecord] = []
    for path in sorted(root.rglob("*.npy")):
        if "derivatives" in path.parts:
            continue
        match = _FILENAME.match(path.name)
        if match is None:
            continue
        parsed_subject = match.group("subject") or "unknown"
        parsed_site = match.group("site") or "unknown"
        if (subject_id is not None and parsed_subject != subject_id) or (site is not None and parsed_site != site):
            continue
        records.append(DerivativeRecord(
            derivative="Segmentation", role="trabecular_mask", subject_id=parsed_subject,
            site=parsed_site, session_id=match.group("session"), stack_index=None,
            space="native", path=path, source="provided", content_type="mask",
        ))
    return records


def _same_case(left: DerivativeRecord, right: DerivativeRecord) -> bool:
    return (left.subject_id, left.site, left.session_id, left.stack_index) == (
        right.subject_id, right.site, right.session_id, right.stack_index
    )


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"Only .npy masks are supported by this batch workflow: {path}")
    return np.asarray(np.load(path), dtype=bool)


def _output_records(
    root: Path,
    input_record: DerivativeRecord,
    common_record: DerivativeRecord | None,
) -> tuple[DerivativeRecord, ...]:
    subject, case_site, session = input_record.subject_id, input_record.site, input_record.session_id
    session_part = f"ses-{session}" if session is not None else "ses-none"
    base_parts = ("native_space", session_part)
    inputs = (input_record.record_id,) + ((common_record.record_id,) if common_record is not None else ())
    label_path = record_output_path(root, _FAMILY, subject, case_site, *base_parts, "maps", f"sub-{subject}_{session_part}_site-{case_site}_desc-plate-rod-label.npy")
    skeleton_path = record_output_path(root, _FAMILY, subject, case_site, *base_parts, "maps", f"sub-{subject}_{session_part}_site-{case_site}_desc-skeleton.npy")
    table_path = record_output_path(root, _FAMILY, subject, case_site, *base_parts, "tables", f"sub-{subject}_{session_part}_site-{case_site}_desc-plate-rod-measurements.csv")
    common = dict(subject_id=subject, site=case_site, session_id=session, stack_index=input_record.stack_index,
                  space="native", source="generated", inputs=inputs)
    return (
        DerivativeRecord(derivative=_FAMILY, role="plate_rod_label_map", path=label_path, content_type="image", **common),
        DerivativeRecord(derivative=_FAMILY, role="skeleton_map", path=skeleton_path, content_type="image", **common),
        DerivativeRecord(derivative=_FAMILY, role="plate_rod_measurements_table", path=table_path, space="table", content_type="table", **{key: value for key, value in common.items() if key != "space"}),
    )


def _write_outputs(records: Sequence[DerivativeRecord], result: object) -> None:
    by_role = {record.role: record.path for record in records}
    for path in by_role.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(by_role["plate_rod_label_map"], result.full_thickness_labels)
    np.save(by_role["skeleton_map"], result.skeleton.astype(np.uint8))
    fieldnames = ["subject_id", "site", "session_id", *sorted(result.summary)]
    row = {"subject_id": records[0].subject_id, "site": records[0].site, "session_id": records[0].session_id or "", **result.summary}
    with by_role["plate_rod_measurements_table"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _emit(
    progress: Callable[[DerivativeProgressEvent], None] | None,
    record: DerivativeRecord,
    step: str,
    status: str,
    message: str,
) -> None:
    if progress is not None:
        progress(DerivativeProgressEvent(_FAMILY, record.subject_id, record.site, record.session_id, step, status, message))
