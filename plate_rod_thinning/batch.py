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
    BatchArtifact,
    CaseKey,
    DerivativeManifest,
    DerivativeProgressEvent,
    DerivativeRecord,
    case_keys_match,
    discover_derivative_artifacts,
    discover_artifacts,
    discover_manifests,
    discover_raw_xct_images,
    find_records,
    normalize_session_id,
    normalize_site,
    normalize_subject_id,
    preferred_contours,
    prerequisite_status,
    read_manifest,
    write_manifest,
)
from bone_imaging_derivatives.layout import record_output_path, voi_token

from .pipeline import PLATE, ROD, PlateRodParameters, plate_rod_analysis, _volume_summary
from .backend import backend_name


_FAMILY = "PlateRodMorphometry"
_MASK_ROLES = ("trabecular_mask", "bone_segmentation")
_MASK_FAMILY_PRIORITY = {"IPLContours": 0, "ImportedContours": 1, "BoneContours": 2, "Segmentation": 3}
_FILENAME = re.compile(
    r"(?:sub-(?P<subject>[^_]+)_)?(?:ses-(?P<session>[^_]+)_)?(?:site-(?P<site>[^_]+)_)?"
    r".*(?:trab(?:ecular)?|bone|seg).*\.npy$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BatchWorkflowResult:
    """Artifacts created (or planned) by a plate/rod batch run."""

    manifest: DerivativeManifest
    records: tuple[DerivativeRecord, ...]
    output_root: Path


@dataclass(frozen=True)
class PlateRodBatchRow:
    """One normalized source image and its plate/rod prerequisite status."""

    image: BatchArtifact
    status: str
    missing_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BatchInput:
    bone_segmentation: DerivativeRecord
    trabecular_mask: DerivativeRecord
    common_region: DerivativeRecord | None


def discover_plate_rod_batch(dataset_root) -> tuple[PlateRodBatchRow, ...]:
    """Return normalized plate/rod batch rows with shared prerequisite states."""
    root = Path(dataset_root).resolve()
    if root.name == "derivatives":
        root = root.parent
    images = discover_raw_xct_images(root)
    contours = (
        *discover_derivative_artifacts(root, "IPLContours"),
        *discover_derivative_artifacts(root, "ImportedContours"),
        *discover_derivative_artifacts(root, "BoneContours"),
    )
    existing = discover_derivative_artifacts(root, _FAMILY)
    rows: list[PlateRodBatchRow] = []
    for image in images:
        result = prerequisite_status(
            image,
            preferred_contours(contours, image.key),
            required_roles=("segmentation", "trab"),
            existing_outputs=existing,
        )
        rows.append(PlateRodBatchRow(image=image, status=result.status, missing_roles=result.missing_roles))
    return tuple(rows)


def run_plate_rod_batch(
    dataset_root: Path,
    *,
    subject_id: str | None = None,
    site: str | None = None,
    session_id: str | None = None,
    output_root: Path | None = None,
    manifests: Sequence[DerivativeManifest] | None = None,
    generate_missing: bool = False,
    force: bool = False,
    dry_run: bool = False,
    progress: Callable[[DerivativeProgressEvent], None] | None = None,
    use_common_region: bool = True,
    require_common_region: bool = False,
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
    inputs = _filter_inputs(
        _discover_inputs(root, known_manifests, subject_id=subject_id, site=site, use_common_region=use_common_region),
        session_id=session_id,
    )
    if not inputs:
        raise ValueError("No trabecular or bone masks were found for plate/rod analysis")
    missing_common = [item for item in inputs if require_common_region and item.common_region is None]
    if missing_common:
        raise ValueError(_missing_common_region_message(missing_common))
    measurement_hash = _settings_hash(parameters, use_common_region=use_common_region)
    map_hash = _settings_hash(parameters, use_common_region=False)
    records: list[DerivativeRecord] = []
    for item in inputs:
        map_records = _map_output_records(root, destination, item.bone_segmentation, item.trabecular_mask, map_hash)
        measurement_record = _measurement_output_record(
            root,
            destination,
            item.bone_segmentation,
            item.trabecular_mask,
            item.common_region if use_common_region else None,
            measurement_hash,
        )
        reusable_measurement = None if force else _compatible_records(existing_manifest, (measurement_record,))
        if reusable_measurement is not None:
            records.extend(reusable_measurement)
            _emit(progress, item.trabecular_mask, "measure", "reused", "Reused compatible plate/rod measurements")
            continue
        if dry_run:
            records.extend((*map_records, measurement_record))
            _emit(progress, item.trabecular_mask, "measure", "planned", "Planned plate/rod derivative outputs")
            continue
        _emit(progress, item.trabecular_mask, "measure", "started", f"Running plate/rod morphometry (backend={backend_name()})")
        bone_segmentation = _load_mask(item.bone_segmentation.path)
        trabecular_mask = _load_mask(item.trabecular_mask.path)
        common = _load_mask(item.common_region.path) if item.common_region is not None else None
        if bone_segmentation.shape != trabecular_mask.shape:
            raise ValueError(f"Bone segmentation and trabecular mask shapes differ: {item.bone_segmentation.path}, {item.trabecular_mask.path}")
        if common is not None and common.shape != bone_segmentation.shape:
            raise ValueError(f"Common-region mask shape differs from bone mask: {item.common_region.path}")
        native_analysis_mask = trabecular_mask
        native_bone = bone_segmentation & native_analysis_mask
        reusable_maps = None if force else _compatible_records(existing_manifest, map_records)
        if reusable_maps is None:
            result = plate_rod_analysis(native_bone, analysis_mask=native_analysis_mask, parameters=parameters)
            _write_map_outputs(map_records, result)
            records.extend(map_records)
        else:
            result = None
            records.extend(reusable_maps)
            _emit(progress, item.trabecular_mask, "maps", "reused", "Reused native plate/rod maps")

        analysis_mask = native_analysis_mask if common is None or not use_common_region else native_analysis_mask & common
        summary = (
            result.summary
            if result is not None and common is None
            else _summarize_existing_maps(map_records, analysis_mask, parameters)
        )
        _write_measurement_output(measurement_record, summary)
        records.append(measurement_record)
        _emit(progress, item.trabecular_mask, "measure", "completed", "Wrote plate/rod measurements")

    superseded = {_manifest_record_identity(record) for record in records}
    preserved = tuple(
        record for record in (existing_manifest.records if existing_manifest is not None else ())
        if _manifest_record_identity(record) not in superseded
    )
    all_records = (*_deduplicate_manifest_records(preserved), *records)

    manifest = DerivativeManifest.create(
        _FAMILY,
        root,
        {"name": "plate-rod-thinning", "version": "0.1.7"},
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
    bone_by_case: dict[tuple[str, str, str | None, int | None], DerivativeRecord] = {}
    trab_by_case: dict[tuple[str, str, str | None, int | None], DerivativeRecord] = {}
    bone_records = find_records(manifests, role="bone_segmentation", subject_id=subject_id, site=site, space="native")
    trab_records = find_records(manifests, role="trabecular_mask", subject_id=subject_id, site=site, space="native")
    for record in sorted(bone_records, key=_record_priority):
        bone_by_case.setdefault(_grouping_case_key(record), record)
    for record in sorted(trab_records, key=_record_priority):
        trab_by_case.setdefault(_grouping_case_key(record), record)
    has_manifest_masks = bool(bone_by_case or trab_by_case)
    for record in _shared_contour_records(root, subject_id=subject_id, site=site):
        if record.role == "bone_segmentation":
            bone_by_case.setdefault(_grouping_case_key(record), record)
        elif record.role == "trabecular_mask":
            trab_by_case.setdefault(_grouping_case_key(record), record)
    has_manifest_masks = bool(bone_by_case or trab_by_case)
    fallback_records = _filename_fallback(root, subject_id=subject_id, site=site)
    for record in fallback_records:
        if has_manifest_masks and (record.subject_id == "unknown" or record.site == "unknown"):
            continue
        if record.role == "bone_segmentation":
            bone_by_case.setdefault(_grouping_case_key(record), record)
        elif record.role == "trabecular_mask":
            trab_by_case.setdefault(_grouping_case_key(record), record)
    common_regions = find_records(
        manifests,
        derivative="CommonRegion",
        role="scan_region_native_common",
        subject_id=subject_id,
        site=site,
        space="native",
    )
    inputs: list[_BatchInput] = []
    for key in sorted(set(bone_by_case) | set(trab_by_case), key=_sortable_case_key):
        bone = bone_by_case.get(key)
        trab = trab_by_case.get(key)
        if bone is None and trab is None:
            continue
        if trab is None:
            trab = bone
        if bone is None:
            bone = trab
        common = next(
            (record for record in common_regions if _same_case(record, trab)), None
        ) if use_common_region else None
        inputs.append(_BatchInput(bone, trab, common))
    return inputs


def _shared_contour_records(root: Path, *, subject_id: str | None, site: str | None) -> list[DerivativeRecord]:
    records: list[DerivativeRecord] = []
    requested_subject = normalize_subject_id(subject_id) if subject_id is not None else None
    requested_site = normalize_site(site) if site is not None else None
    for family in ("IPLContours", "ImportedContours", "BoneContours"):
        for artifact in discover_derivative_artifacts(root, family):
            if artifact.role not in {"segmentation", "trab"}:
                continue
            if requested_subject is not None and artifact.key.subject_id != requested_subject:
                continue
            if requested_site is not None and artifact.key.voi != requested_site:
                continue
            role = "bone_segmentation" if artifact.role == "segmentation" else "trabecular_mask"
            source = artifact.source if artifact.source in {"generated", "provided", "derived", "legacy", "virtual"} else "provided"
            records.append(
                DerivativeRecord(
                    derivative=family,
                    role=role,
                    subject_id=artifact.key.subject_id,
                    site=artifact.key.voi,
                    session_id=artifact.key.session_id,
                    stack_index=artifact.key.stack_index,
                    space="native",
                    path=artifact.path,
                    source=source,
                    content_type="mask",
                    metadata=dict(artifact.metadata),
                    record_id=f"{family}:{role}:{artifact.key.subject_id}:{artifact.key.voi}:{artifact.key.session_id}:{artifact.key.stack_index}:native",
                )
            )
    return records


def _filter_inputs(inputs: list[_BatchInput], *, session_id: str | None) -> list[_BatchInput]:
    requested = _session_key(session_id)
    if not requested:
        return list(inputs)
    return [item for item in inputs if _session_key(item.trabecular_mask.session_id) == requested]


def _missing_common_region_message(inputs: Sequence[_BatchInput]) -> str:
    parts = []
    for item in inputs:
        subject, site, session, stack = _case_key(item.trabecular_mask)
        text = f"sub-{subject}, ses-{session or 'unknown'}, voi-{site}"
        if stack is not None:
            text = f"{text}, stack-{int(stack):02d}"
        parts.append(text)
    return "Plate/rod registered batch prerequisites are incomplete: missing common region for " + "; ".join(parts)


def _filename_fallback(root: Path, *, subject_id: str | None, site: str | None) -> list[DerivativeRecord]:
    records: list[DerivativeRecord] = []
    for artifact in discover_artifacts(root, include_derivatives=True).records:
        if artifact.kind != "mask" or artifact.role not in {"segmentation", "trab"}:
            continue
        parsed_subject = artifact.subject_id or "unknown"
        parsed_site = artifact.site or "unknown"
        if (subject_id is not None and parsed_subject != subject_id) or (site is not None and parsed_site != site):
            continue
        role = "bone_segmentation" if artifact.role == "segmentation" else "trabecular_mask"
        records.append(DerivativeRecord(
            derivative="Segmentation",
            role=role,
            subject_id=parsed_subject,
            site=parsed_site,
            session_id=artifact.session_id,
            stack_index=artifact.stack_index,
            space="native",
            path=artifact.path,
            source="provided",
            content_type="mask",
            record_id=f"Segmentation:{role}:{parsed_subject}:{parsed_site}:{artifact.session_id}:{artifact.stack_index}:native",
        ))
    for path in sorted(root.rglob("*.npy")):
        match = _FILENAME.match(path.name)
        if match is None:
            continue
        parsed_subject = match.group("subject")
        parsed_site = match.group("site")
        parsed_session = match.group("session")
        for part in path.relative_to(root).parts[:-1]:
            lower = part.lower()
            if parsed_subject is None and lower.startswith(("sub-", "sub_")):
                parsed_subject = part[4:]
            elif parsed_site is None and lower.startswith(("site-", "site_")):
                parsed_site = part[5:]
            elif parsed_site is None and lower.startswith(("voi-", "voi_")):
                parsed_site = part[4:]
            elif parsed_session is None and lower.startswith(("ses-", "ses_")):
                parsed_session = part[4:]
        parsed_subject = normalize_subject_id(parsed_subject) or "unknown"
        parsed_site = normalize_site(parsed_site) or "unknown"
        parsed_session = normalize_session_id(parsed_session)
        if (subject_id is not None and parsed_subject != subject_id) or (site is not None and parsed_site != site):
            continue
        role = "trabecular_mask" if re.search(r"trab(?:ecular)?", path.name, re.IGNORECASE) else "bone_segmentation"
        records.append(DerivativeRecord(
            derivative="Segmentation", role=role, subject_id=parsed_subject,
            site=parsed_site, session_id=parsed_session, stack_index=None,
            space="native", path=path, source="provided", content_type="mask",
        ))
    return records


def _same_case(left: DerivativeRecord, right: DerivativeRecord) -> bool:
    return case_keys_match(_record_case_key(left), _record_case_key(right))


def _case_key(record: DerivativeRecord) -> tuple[str, str, str | None, int | None]:
    return record.subject_id, record.site, record.session_id, record.stack_index


def _grouping_case_key(record: DerivativeRecord) -> tuple[str, str, str | None, int | None]:
    return _case_key(record)


def _record_priority(record: DerivativeRecord) -> tuple[int, str]:
    return _MASK_FAMILY_PRIORITY.get(record.derivative, 99), str(record.path)


def _record_case_key(record: DerivativeRecord) -> CaseKey:
    session = record.session_id or ""
    return CaseKey(record.subject_id, session, record.site, record.stack_index)


def _sortable_case_key(key: tuple[str, str, str | None, int | None]) -> tuple[str, str, str, int]:
    subject, site, session, stack = key
    return str(subject), str(site), "" if session is None else str(session), 0 if stack is None else int(stack)


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=bool)
    if path.name.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
    if re.search(r"\.aim(?:;\d+)?$", path.name, re.IGNORECASE):
        try:
            import py_aimio
        except ImportError as exc:
            raise RuntimeError("AIM masks require aimio-py.") from exc

        array, _metadata = py_aimio.read_aim(str(path), density=False, hu=False)
        return np.asarray(array) > 0
    raise ValueError(f"Unsupported mask format for plate/rod batch workflow: {path}")


def _map_output_records(
    root: Path,
    destination: Path,
    bone_record: DerivativeRecord,
    trab_record: DerivativeRecord,
    settings_hash: str,
) -> tuple[DerivativeRecord, ...]:
    subject, case_site, session = trab_record.subject_id, trab_record.site, trab_record.session_id
    session_part = f"ses-{session}" if session is not None else "ses-none"
    inputs = tuple(dict.fromkeys((bone_record.record_id, trab_record.record_id)))
    prefix = f"sub-{subject}_{session_part}_voi-{voi_token(case_site)}"
    label_path = _artifact_path(root, destination, subject, case_site, session_part, "maps", f"{prefix}_desc-plate-rod-label.npy")
    skeleton_path = _artifact_path(root, destination, subject, case_site, session_part, "maps", f"{prefix}_desc-skeleton.npy")
    common = dict(subject_id=subject, site=case_site, session_id=session, stack_index=trab_record.stack_index,
                  space="native", source="generated", inputs=inputs, settings_hash=settings_hash)
    return (
        DerivativeRecord(derivative=_FAMILY, role="plate_rod_label_map", path=label_path, content_type="image", **common),
        DerivativeRecord(derivative=_FAMILY, role="skeleton_map", path=skeleton_path, content_type="image", **common),
    )


def _measurement_output_record(
    root: Path,
    destination: Path,
    bone_record: DerivativeRecord,
    trab_record: DerivativeRecord,
    common_record: DerivativeRecord | None,
    settings_hash: str,
) -> DerivativeRecord:
    subject, case_site, session = trab_record.subject_id, trab_record.site, trab_record.session_id
    session_part = f"ses-{session}" if session is not None else "ses-none"
    inputs = tuple(dict.fromkeys((bone_record.record_id, trab_record.record_id))) + ((common_record.record_id,) if common_record is not None else ())
    prefix = f"sub-{subject}_{session_part}_voi-{voi_token(case_site)}"
    measurement_dir = "registered_measurements" if common_record is not None else "measurements"
    table_path = _artifact_path(
        root,
        destination,
        subject,
        case_site,
        session_part,
        measurement_dir,
        f"{prefix}_desc-plate-rod-measurements.csv",
    )
    return DerivativeRecord(
        derivative=_FAMILY,
        role="plate_rod_measurements_table",
        subject_id=subject,
        site=case_site,
        session_id=session,
        stack_index=trab_record.stack_index,
        space="table",
        path=table_path,
        source="generated",
        inputs=inputs,
        settings_hash=settings_hash,
        metadata={"use_common_region": common_record is not None},
        content_type="table",
    )


_OUTPUT_ROLES = frozenset({"plate_rod_label_map", "skeleton_map", "plate_rod_measurements_table"})


def _record_replacement_signature(record: DerivativeRecord) -> tuple[str, bool]:
    if record.role == "plate_rod_measurements_table":
        return record.role, bool(record.metadata.get("use_common_region"))
    return record.role, False


def _manifest_record_identity(record: DerivativeRecord) -> tuple[tuple[str, str, str | None, int | None], tuple[str, bool]]:
    return _grouping_case_key(record), _record_replacement_signature(record)


def _deduplicate_manifest_records(records: Sequence[DerivativeRecord]) -> tuple[DerivativeRecord, ...]:
    by_identity: dict[tuple[tuple[str, str, str | None, int | None], tuple[str, bool]], DerivativeRecord] = {}
    for record in records:
        by_identity[_manifest_record_identity(record)] = record
    return tuple(by_identity.values())


def _artifact_path(root: Path, destination: Path, subject: str, site: str, *parts: str) -> Path:
    default_destination = root / "derivatives" / _FAMILY
    if destination == default_destination:
        return record_output_path(root, _FAMILY, subject, site, *parts)
    cleaned_parts = [str(part) for part in parts if str(part)]
    session = None
    for index, part in enumerate(list(cleaned_parts)):
        if part.startswith("ses-"):
            session = part
            cleaned_parts.pop(index)
            break
    base = destination / f"sub-{subject}"
    if session:
        base = base / session / "xct"
    else:
        base = base / "xct"
    return base / Path(*cleaned_parts)


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


def _write_map_outputs(records: Sequence[DerivativeRecord], result: object) -> None:
    by_role = {record.role: record.path for record in records}
    for path in by_role.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(by_role["plate_rod_label_map"], result.full_thickness_labels)
    np.save(by_role["skeleton_map"], result.skeleton.astype(np.uint8))


def _write_measurement_output(record: DerivativeRecord, summary: dict[str, int | float | str]) -> None:
    record.path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "site", "session_id", *sorted(summary)]
    row = {"subject_id": record.subject_id, "site": record.site, "session_id": record.session_id or "", **summary}
    with record.path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _summarize_existing_maps(
    map_records: Sequence[DerivativeRecord],
    analysis_mask: np.ndarray,
    parameters: PlateRodParameters | None,
) -> dict[str, int | float | str]:
    by_role = {record.role: record.path for record in map_records}
    labels = np.asarray(np.load(by_role["plate_rod_label_map"], allow_pickle=False))
    skeleton = np.asarray(np.load(by_role["skeleton_map"], allow_pickle=False)) > 0
    mask = np.asarray(analysis_mask, dtype=bool)
    if labels.shape != mask.shape or skeleton.shape != mask.shape:
        raise ValueError("Plate/rod maps and analysis mask must have matching shapes")
    params = parameters or PlateRodParameters()
    masked_labels = labels.copy()
    masked_labels[~mask] = 0
    bone = masked_labels > 0
    summary = _volume_summary(
        bone_voxels=int(np.count_nonzero(bone)),
        tissue_voxels=int(np.count_nonzero(mask)),
        plate_voxels=int(np.count_nonzero(masked_labels == PLATE)),
        rod_voxels=int(np.count_nonzero(masked_labels == ROD)),
        voxel_spacing_mm=params.voxel_spacing_mm,
    )
    summary.update(
        {
            "classifier": "saha_topology" if params.skeletonize else "degree_preview",
            "bone_voxels": int(np.count_nonzero(bone)),
            "tissue_voxels": int(np.count_nonzero(mask)),
            "skeleton_voxels": int(np.count_nonzero(skeleton & mask)),
            "plate_skeleton_voxels": 0,
            "rod_skeleton_voxels": 0,
            "plate_full_thickness_voxels": int(np.count_nonzero(masked_labels == PLATE)),
            "rod_full_thickness_voxels": int(np.count_nonzero(masked_labels == ROD)),
            "slenderness": int(params.slenderness),
            "junction_dilation_voxels": int(params.junction_dilation_voxels),
            "junction_support_radius_voxels": -1
            if params.junction_support_radius_voxels is None
            else int(params.junction_support_radius_voxels),
            "min_plate_voxels": int(params.min_plate_voxels),
            "min_rod_voxels": int(params.min_rod_voxels),
            "max_iterations": int(params.max_iterations),
        }
    )
    return summary


def _session_key(session_id) -> str:
    value = str(session_id or "").strip()
    upper = value.upper()
    if upper.startswith("SES-"):
        upper = upper[4:]
    if upper.startswith("Y") and upper[1:].isdigit():
        upper = upper[1:]
    return upper.lstrip("0") or ("0" if value else "")


def _emit(
    progress: Callable[[DerivativeProgressEvent], None] | None,
    record: DerivativeRecord,
    step: str,
    status: str,
    message: str,
) -> None:
    if progress is not None:
        progress(DerivativeProgressEvent(_FAMILY, record.subject_id, record.site, record.session_id, step, status, message))
