"""Derivative-writing batch workflow for plate/rod morphometry."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from bone_imaging_derivatives import (
    DerivativeManifest,
    DerivativeProgressEvent,
    DerivativeRecord,
    discover_manifests,
    find_records,
    read_manifest,
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
    del generate_missing
    root = Path(dataset_root).resolve()
    destination = Path(output_root).resolve() if output_root is not None else root / "derivatives" / _FAMILY
    known_manifests = list(discover_manifests(root) if manifests is None else manifests)
    existing_manifest = _read_existing_manifest(destination)
    inputs = _discover_inputs(root, known_manifests, subject_id=subject_id, site=site, use_common_region=use_common_region)
    if not inputs:
        raise ValueError("No trabecular or bone masks were found for plate/rod analysis")
    settings_hash = _settings_hash(parameters, use_common_region=use_common_region)
    records: list[DerivativeRecord] = []
    active_cases = {_case_key(item.bone) for item in inputs}
    for item in inputs:
        expected_records = _output_records(root, destination, item.bone, item.common_region, settings_hash)
        reusable = _compatible_records(existing_manifest, expected_records)
        if reusable is not None and not force:
            records.extend(reusable)
            _emit(progress, item.bone, "measure", "reused", "Reused compatible plate/rod derivative outputs")
            continue
        if dry_run:
            records.extend(expected_records)
            _emit(progress, item.bone, "measure", "planned", "Planned plate/rod derivative outputs")
            continue
        _emit(progress, item.bone, "measure", "started", "Running plate/rod morphometry")
        bone = _load_mask(item.bone.path)
        common = _load_mask(item.common_region.path) if item.common_region is not None else None
        if common is not None and common.shape != bone.shape:
            raise ValueError(f"Common-region mask shape differs from bone mask: {item.common_region.path}")
        # The common scan region limits the actual skeletonization/classification
        # input, not merely downstream summary denominators.
        effective_bone = bone if common is None else bone & common
        result = plate_rod_analysis(effective_bone, analysis_mask=common, parameters=parameters)
        if not dry_run:
            _write_outputs(expected_records, result)
        records.extend(expected_records)
        _emit(progress, item.bone, "measure", "completed", "Wrote plate/rod derivative outputs")

    preserved = tuple(
        record for record in (existing_manifest.records if existing_manifest is not None else ())
        if not (record.derivative == _FAMILY and _case_key(record) in active_cases and record.role in _OUTPUT_ROLES)
    )
    all_records = (*preserved, *records)

    manifest = DerivativeManifest.create(
        _FAMILY,
        root,
        {"name": "plate-rod-thinning", "version": "0.1.5"},
        all_records,
    )
    if not dry_run:
        write_manifest(manifest, destination / "manifest.json")
    return BatchWorkflowResult(manifest=manifest, records=all_records, output_root=destination)


def _discover_inputs(
    root: Path,
    manifests: Sequence[DerivativeManifest],
    *,
    subject_id: str | None,
    site: str | None,
    use_common_region: bool,
) -> list[_BatchInput]:
    masks_by_case: dict[tuple[str, str, str | None, int | None], DerivativeRecord] = {}
    for role in _MASK_ROLES:
        for record in find_records(manifests, role=role, subject_id=subject_id, site=site, space="native"):
            masks_by_case.setdefault(_case_key(record), record)
    masks = list(masks_by_case.values())
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
    for mask in masks:
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
    return _case_key(left) == _case_key(right)


def _case_key(record: DerivativeRecord) -> tuple[str, str, str | None, int | None]:
    return record.subject_id, record.site, record.session_id, record.stack_index


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=bool)
    if path.name.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    raise ValueError(f"Unsupported mask format for plate/rod batch workflow: {path}")


def _output_records(
    root: Path,
    destination: Path,
    input_record: DerivativeRecord,
    common_record: DerivativeRecord | None,
    settings_hash: str,
) -> tuple[DerivativeRecord, ...]:
    subject, case_site, session = input_record.subject_id, input_record.site, input_record.session_id
    session_part = f"ses-{session}" if session is not None else "ses-none"
    base_parts = ("native_space", session_part)
    inputs = (input_record.record_id,) + ((common_record.record_id,) if common_record is not None else ())
    label_path = _artifact_path(root, destination, subject, case_site, *base_parts, "maps", f"sub-{subject}_{session_part}_site-{case_site}_desc-plate-rod-label.npy")
    skeleton_path = _artifact_path(root, destination, subject, case_site, *base_parts, "maps", f"sub-{subject}_{session_part}_site-{case_site}_desc-skeleton.npy")
    table_path = _artifact_path(root, destination, subject, case_site, *base_parts, "tables", f"sub-{subject}_{session_part}_site-{case_site}_desc-plate-rod-measurements.csv")
    common = dict(subject_id=subject, site=case_site, session_id=session, stack_index=input_record.stack_index,
                  space="native", source="generated", inputs=inputs, settings_hash=settings_hash)
    return (
        DerivativeRecord(derivative=_FAMILY, role="plate_rod_label_map", path=label_path, content_type="image", **common),
        DerivativeRecord(derivative=_FAMILY, role="skeleton_map", path=skeleton_path, content_type="image", **common),
        DerivativeRecord(derivative=_FAMILY, role="plate_rod_measurements_table", path=table_path, space="table", content_type="table", **{key: value for key, value in common.items() if key != "space"}),
    )


_OUTPUT_ROLES = frozenset({"plate_rod_label_map", "skeleton_map", "plate_rod_measurements_table"})


def _artifact_path(root: Path, destination: Path, subject: str, site: str, *parts: str) -> Path:
    default_destination = root / "derivatives" / _FAMILY
    if destination == default_destination:
        return record_output_path(root, _FAMILY, subject, site, *parts)
    return destination / f"sub-{subject}" / f"site-{site}" / Path(*parts)


def _read_existing_manifest(destination: Path) -> DerivativeManifest | None:
    path = destination / "manifest.json"
    return read_manifest(path) if path.exists() else None


def _settings_hash(parameters: PlateRodParameters | None, *, use_common_region: bool) -> str:
    payload = {"parameters": asdict(parameters or PlateRodParameters()), "use_common_region": use_common_region}
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _compatible_records(
    existing_manifest: DerivativeManifest | None,
    expected_records: Sequence[DerivativeRecord],
) -> tuple[DerivativeRecord, ...] | None:
    if existing_manifest is None:
        return None
    records: list[DerivativeRecord] = []
    for expected in expected_records:
        record = next(
            (
                candidate for candidate in existing_manifest.records
                if candidate.derivative == _FAMILY
                and candidate.role == expected.role
                and _same_case(candidate, expected)
                and candidate.inputs == expected.inputs
                and candidate.settings_hash == expected.settings_hash
                and candidate.path == expected.path
                and candidate.path.exists()
            ),
            None,
        )
        if record is None:
            return None
        records.append(record)
    return tuple(records)


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
