from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest


def _write_segmentation_manifest(dataset_root: Path, *, bone_path: Path, common_path: Path | None = None) -> None:
    records = [
        DerivativeRecord(
            derivative="Segmentation",
            role="trabecular_mask",
            subject_id="SAMPLE001",
            site="tibia",
            session_id="1",
            stack_index=None,
            space="native",
            path=bone_path,
            source="provided",
            content_type="mask",
        )
    ]
    if common_path is not None:
        records.append(
            DerivativeRecord(
                derivative="CommonRegion",
                role="scan_region_native_common",
                subject_id="SAMPLE001",
                site="tibia",
                session_id="1",
                stack_index=None,
                space="native",
                path=common_path,
                source="derived",
                content_type="mask",
            )
        )
    write_manifest(
        DerivativeManifest.create(
            "Segmentation", dataset_root, {"name": "fixture", "version": "1"}, tuple(records)
        ),
        dataset_root / "derivatives" / "Segmentation" / "manifest.json",
    )


def _tiny_bone() -> np.ndarray:
    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[2, 2, 1:4] = True
    return bone


def test_run_plate_rod_batch_discovers_manifest_masks_and_writes_derivative_outputs(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[2, 2, 1:4] = True
    bone_path = tmp_path / "inputs" / "sub-SAMPLE001_ses-1_site-tibia_mask-trabecular.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, bone)
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)

    result = run_plate_rod_batch(
        tmp_path,
        parameters=PlateRodParameters(skeletonize=False),
    )

    manifest_path = tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json"
    manifest = read_manifest(manifest_path)
    assert result.manifest == manifest
    assert {record.role for record in manifest.records} == {
        "plate_rod_label_map",
        "plate_rod_measurements_table",
        "skeleton_map",
    }
    assert all(record.path.exists() for record in manifest.records)
    table = next(record.path for record in manifest.records if record.role == "plate_rod_measurements_table")
    assert "SAMPLE001" in table.read_text(encoding="utf-8")


def test_run_plate_rod_batch_accepts_nifti_manifest_mask(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    sitk = pytest.importorskip("SimpleITK")
    bone = _tiny_bone()
    bone_path = tmp_path / "inputs" / "sub-SAMPLE001_ses-1_site-tibia_mask-trabecular.nii.gz"
    bone_path.parent.mkdir(parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(bone.astype(np.uint8)), str(bone_path))
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)

    result = run_plate_rod_batch(
        tmp_path,
        parameters=PlateRodParameters(skeletonize=False),
    )

    assert len(result.records) == 3
    assert all(record.path.exists() for record in result.records)


def test_run_plate_rod_batch_clips_measurement_not_native_maps(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone = np.zeros((5, 5, 5), dtype=bool)
    bone[2, 2, 1:4] = True
    common = np.zeros_like(bone)
    common[2, 2, 1:3] = True
    bone_path = tmp_path / "inputs" / "trab.npy"
    common_path = tmp_path / "inputs" / "common.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, bone)
    np.save(common_path, common)
    _write_segmentation_manifest(tmp_path, bone_path=bone_path, common_path=common_path)

    result = run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
    )

    label_record = next(record for record in result.records if record.role == "plate_rod_label_map")
    labels = np.load(label_record.path)
    assert labels[~common].any()
    table_record = next(record for record in result.records if record.role == "plate_rod_measurements_table")
    assert ",2," in table_record.path.read_text(encoding="utf-8")


def test_run_plate_rod_batch_uses_filename_fallback_for_unmanifested_mask(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    mask_path = tmp_path / "sub-SAMPLE002_ses-2_site-radius_mask-trabecular.npy"
    np.save(mask_path, np.pad(np.ones((1, 1, 3), dtype=bool), ((2, 2), (2, 2), (1, 1))))

    result = run_plate_rod_batch(tmp_path, parameters=PlateRodParameters(skeletonize=False))

    assert len(result.records) == 3
    assert {record.subject_id for record in result.records} == {"SAMPLE002"}
    assert {record.site for record in result.records} == {"radius"}


def test_cli_run_batch_writes_plate_rod_manifest(tmp_path: Path) -> None:
    mask_path = tmp_path / "sub-SAMPLE003_ses-1_site-tibia_mask-trabecular.npy"
    np.save(mask_path, np.pad(np.ones((1, 1, 3), dtype=bool), ((2, 2), (2, 2), (1, 1))))

    completed = subprocess.run(
        [sys.executable, "-m", "plate_rod_thinning.cli", "run-batch", str(tmp_path), "--no-skeletonize"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BONE_DERIVATIVES_PROGRESS" in completed.stdout
    assert (tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json").exists()


def test_run_plate_rod_batch_intersects_bone_segmentation_with_trabecular_mask_for_one_case(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab_path = tmp_path / "inputs" / "trab.npy"
    bone_path = tmp_path / "inputs" / "bone.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, _tiny_bone())
    np.save(bone_path, _tiny_bone())
    trab = DerivativeRecord("Segmentation", "trabecular_mask", "SAMPLE001", "tibia", "1", None, "native", trab_path, "provided")
    bone = DerivativeRecord("Segmentation", "bone_segmentation", "SAMPLE001", "tibia", "1", None, "native", bone_path, "provided")
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "fixture", "version": "1"}, (trab, bone)),
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
    )

    result = run_plate_rod_batch(tmp_path, parameters=PlateRodParameters(skeletonize=False))

    assert len(result.records) == 3
    assert len({record.record_id for record in result.records}) == 3
    assert len({record.path for record in result.records}) == 3
    assert {record.inputs for record in result.records} == {(bone.record_id, trab.record_id)}


def test_run_plate_rod_batch_pairs_contour_segmentation_and_trab_roi_from_derivatives(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-SAMPLE001" / "site-radius" / "ses-1" / "masks"
    mask_dir.mkdir(parents=True)
    seg_path = mask_dir / "sub-SAMPLE001_ses-1_site-radius_mask-seg.npy"
    trab_path = mask_dir / "sub-SAMPLE001_ses-1_site-radius_mask-trab.npy"
    seg = np.zeros((5, 5, 5), dtype=bool)
    seg[2, 2, 1:4] = True
    trab = np.zeros_like(seg)
    trab[2, 2, 1:3] = True
    np.save(seg_path, seg)
    np.save(trab_path, trab)

    result = run_plate_rod_batch(tmp_path, parameters=PlateRodParameters(skeletonize=False))

    assert len(result.records) == 3
    assert {len(record.inputs) for record in result.records} == {2}
    table = next(record.path for record in result.records if record.role == "plate_rod_measurements_table")
    assert ",2," in table.read_text(encoding="utf-8")


def test_run_plate_rod_batch_dry_run_discovers_contour_aim_masks_from_derivatives(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-STRAMBO_0001" / "site-radius" / "ses-Y00" / "masks"
    mask_dir.mkdir(parents=True)
    (mask_dir / "STRAMBO_0001_RL_Y00_mask-seg.AIM").write_bytes(b"aim")
    (mask_dir / "STRAMBO_0001_RL_Y00_mask-trab.AIM").write_bytes(b"aim")

    result = run_plate_rod_batch(tmp_path, parameters=PlateRodParameters(skeletonize=False), dry_run=True)

    assert len(result.records) == 3
    assert {record.subject_id for record in result.records} == {"STRAMBO_0001"}
    assert {record.site for record in result.records} == {"radiusleft"}
    assert {record.session_id for record in result.records} == {"00"}


def test_run_plate_rod_batch_prefers_normalized_ipl_contours_over_bone_contours(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import _discover_inputs, run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    root = tmp_path
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    bone_dir = root / "derivatives" / "BoneContours" / "sub-001" / "ses-001" / "xct"
    ipl_dir = root / "derivatives" / "IPLContours" / "sub-001" / "ses-001" / "xct"
    image.parent.mkdir(parents=True)
    image.write_text("image", encoding="utf-8")
    for directory in (bone_dir, ipl_dir):
        directory.mkdir(parents=True)
        (directory / "sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM").write_bytes(b"aim")
        (directory / "sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM").write_bytes(b"aim")

    result = run_plate_rod_batch(root, parameters=PlateRodParameters(skeletonize=False), dry_run=True)

    inputs = _discover_inputs(root, (), subject_id=None, site=None, use_common_region=False)

    assert len(inputs) == 1
    assert "IPLContours" in str(inputs[0].bone_segmentation.path)
    assert "IPLContours" in str(inputs[0].trabecular_mask.path)
    assert len(result.records) == 3


def test_run_plate_rod_batch_writes_all_artifacts_under_external_output_root(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone_path = tmp_path / "inputs" / "trab.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, _tiny_bone())
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)
    output_root = tmp_path.parent / "external-plate-rod-output"

    result = run_plate_rod_batch(tmp_path, output_root=output_root, parameters=PlateRodParameters(skeletonize=False))

    assert (output_root / "manifest.json").exists()
    assert all(record.path.is_relative_to(output_root) and record.path.exists() for record in result.records)


def test_run_plate_rod_batch_reuses_compatible_outputs_and_preserves_unrelated_records(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone_path = tmp_path / "inputs" / "trab.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, _tiny_bone())
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)
    parameters = PlateRodParameters(skeletonize=False)
    first = run_plate_rod_batch(tmp_path, parameters=parameters)
    label = next(record for record in first.records if record.role == "plate_rod_label_map")
    label.path.touch()
    old_time_ns = 1_000_000_000
    import os
    os.utime(label.path, ns=(old_time_ns, old_time_ns))
    manifest_path = tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json"
    first_manifest = read_manifest(manifest_path)
    unrelated_path = tmp_path / "derivatives" / "PlateRodMorphometry" / "sub-OTHER" / "site-radius" / "other.csv"
    unrelated_path.parent.mkdir(parents=True)
    unrelated_path.write_text("metric,value\n", encoding="utf-8")
    unrelated = DerivativeRecord("PlateRodMorphometry", "plate_rod_measurements_table", "OTHER", "radius", "1", None, "table", unrelated_path, "generated")
    write_manifest(replace(first_manifest, records=(*first_manifest.records, unrelated)), manifest_path)

    second = run_plate_rod_batch(tmp_path, parameters=parameters)

    assert label.path.stat().st_mtime_ns == old_time_ns
    assert unrelated.record_id in {record.record_id for record in second.records}


def test_run_plate_rod_batch_force_recomputes_existing_outputs(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone_path = tmp_path / "inputs" / "trab.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, _tiny_bone())
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)
    parameters = PlateRodParameters(skeletonize=False)
    first = run_plate_rod_batch(tmp_path, parameters=parameters)
    label = next(record for record in first.records if record.role == "plate_rod_label_map")
    old_time_ns = 1_000_000_000
    import os
    os.utime(label.path, ns=(old_time_ns, old_time_ns))

    run_plate_rod_batch(tmp_path, parameters=parameters, force=True)

    assert label.path.stat().st_mtime_ns != old_time_ns


def test_run_plate_rod_batch_dry_run_does_not_run_analysis(tmp_path: Path, monkeypatch) -> None:
    import plate_rod_thinning.batch as batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    bone_path = tmp_path / "inputs" / "trab.npy"
    bone_path.parent.mkdir(parents=True)
    np.save(bone_path, _tiny_bone())
    _write_segmentation_manifest(tmp_path, bone_path=bone_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry-run must not execute plate_rod_analysis")

    monkeypatch.setattr(batch, "plate_rod_analysis", fail_if_called)

    result = batch.run_plate_rod_batch(tmp_path, parameters=PlateRodParameters(skeletonize=False), dry_run=True)

    assert len(result.records) == 3
    assert not (tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json").exists()
    assert not any(record.path.exists() for record in result.records)


def test_run_plate_rod_batch_rejects_empty_dataset_without_writing_manifest(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch

    with pytest.raises(ValueError, match="No trabecular or bone masks"):
        run_plate_rod_batch(tmp_path)

    assert not (tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json").exists()


def test_discover_plate_rod_batch_reports_missing_and_loadable_from_shared_contract(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import discover_plate_rod_batch

    root = tmp_path
    image = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.AIM"
    contours = root / "derivatives" / "ImportedContours" / "sub-001" / "ses-001" / "xct"
    outputs = root / "derivatives" / "PlateRodMorphometry" / "sub-001" / "ses-001" / "xct"
    image.parent.mkdir(parents=True)
    contours.mkdir(parents=True)
    outputs.mkdir(parents=True)
    image.touch()
    (contours / "sub-001_ses-001_voi-radiusleft_desc-seg_mask.AIM").touch()

    missing = discover_plate_rod_batch(root)
    assert missing[0].status == "missing"
    assert missing[0].missing_roles == ("trab",)

    (contours / "sub-001_ses-001_voi-radiusleft_desc-trab_mask.AIM").touch()
    output_path = outputs / "sub-001_ses-001_voi-radiusleft_desc-plate-rod-label.npy"
    output_path.touch()
    write_manifest(
        DerivativeManifest.create(
            "PlateRodMorphometry",
            root,
            {"name": "test", "version": "1"},
            (
                DerivativeRecord(
                    "PlateRodMorphometry",
                    "plate_rod_label_map",
                    "001",
                    "radiusleft",
                    "001",
                    None,
                    "native",
                    output_path,
                    "generated",
                ),
            ),
        ),
        root / "derivatives" / "PlateRodMorphometry" / "manifest.json",
    )
    loadable = discover_plate_rod_batch(root)
    assert loadable[0].status == "loadable"


def test_cli_run_batch_accepts_session_filter(tmp_path: Path) -> None:
    for session in ("001", "002"):
        mask_path = tmp_path / f"sub-SAMPLE003_ses-{session}_site-tibia_mask-trabecular.npy"
        np.save(mask_path, np.pad(np.ones((1, 1, 3), dtype=bool), ((2, 2), (2, 2), (1, 1))))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "plate_rod_thinning.cli",
            "run-batch",
            str(tmp_path),
            "--session",
            "001",
            "--no-common-region",
            "--no-skeletonize",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_manifest(tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json")
    assert completed.returncode == 0
    assert {record.session_id for record in manifest.records} == {"001"}


def test_registered_plate_rod_measurement_reuses_native_maps(tmp_path: Path, monkeypatch) -> None:
    import plate_rod_thinning.batch as batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab = np.zeros((5, 5, 5), dtype=bool)
    trab[2, 2, 1:4] = True
    common = np.zeros_like(trab)
    common[2, 2, 1:3] = True
    trab_path = tmp_path / "inputs" / "trab.npy"
    common_path = tmp_path / "inputs" / "common.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, trab)
    np.save(common_path, common)
    _write_segmentation_manifest(tmp_path, bone_path=trab_path, common_path=common_path)

    calls = {"count": 0}
    real_analysis = batch.plate_rod_analysis

    def counted_analysis(*args, **kwargs):
        calls["count"] += 1
        return real_analysis(*args, **kwargs)

    monkeypatch.setattr(batch, "plate_rod_analysis", counted_analysis)

    native = batch.run_plate_rod_batch(
        tmp_path,
        use_common_region=False,
        parameters=PlateRodParameters(skeletonize=False),
    )
    registered = batch.run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        require_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
    )

    assert calls["count"] == 1
    assert any("/maps/" in str(record.path).replace("\\", "/") for record in native.records)
    assert any("/maps/" in str(record.path).replace("\\", "/") for record in registered.records)
    table = next(
        record.path
        for record in registered.records
        if record.role == "plate_rod_measurements_table"
        and "/registered_measurements/" in str(record.path).replace("\\", "/")
    )
    assert "/registered_measurements/" in str(table).replace("\\", "/")
    assert ",2," in table.read_text(encoding="utf-8")


def test_registered_plate_rod_accepts_common_region_with_matching_stack_identity(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab = _tiny_bone()
    common = np.ones_like(trab)
    trab_path = tmp_path / "inputs" / "trab.npy"
    common_path = tmp_path / "inputs" / "common.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, trab)
    np.save(common_path, common)
    records = (
        DerivativeRecord("Segmentation", "trabecular_mask", "SAMPLE001", "tibia", "1", None, "native", trab_path, "provided"),
        DerivativeRecord("Segmentation", "bone_segmentation", "SAMPLE001", "tibia", "1", None, "native", trab_path, "provided"),
        DerivativeRecord("CommonRegion", "scan_region_native_common", "SAMPLE001", "tibia", "1", None, "native", common_path, "generated"),
    )
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "fixture", "version": "1"}, records),
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
    )

    result = run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        require_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
    )

    assert any(record.role == "plate_rod_measurements_table" for record in result.records)


def test_plate_rod_batch_keeps_unstacked_bone_contours_and_stack_one_segmentation_distinct(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import _discover_inputs

    bone_path = tmp_path / "derivatives" / "BoneContours" / "sub-SAMPLE001" / "ses-1" / "xct" / "seg.npy"
    trab_path = tmp_path / "derivatives" / "BoneContours" / "sub-SAMPLE001" / "ses-1" / "xct" / "trab.npy"
    duplicate_bone_path = tmp_path / "derivatives" / "Segmentation" / "sub-SAMPLE001" / "ses-1" / "xct" / "stack-seg.npy"
    duplicate_trab_path = tmp_path / "derivatives" / "Segmentation" / "sub-SAMPLE001" / "ses-1" / "xct" / "stack-trab.npy"
    for path in (bone_path, trab_path, duplicate_bone_path, duplicate_trab_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, _tiny_bone())
    records = (
        DerivativeRecord("BoneContours", "bone_segmentation", "SAMPLE001", "tibia", "1", None, "native", bone_path, "generated"),
        DerivativeRecord("BoneContours", "trabecular_mask", "SAMPLE001", "tibia", "1", None, "native", trab_path, "generated"),
        DerivativeRecord("Segmentation", "bone_segmentation", "SAMPLE001", "tibia", "1", 1, "native", duplicate_bone_path, "generated"),
        DerivativeRecord("Segmentation", "trabecular_mask", "SAMPLE001", "tibia", "1", 1, "native", duplicate_trab_path, "generated"),
    )
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "fixture", "version": "1"}, records),
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
    )

    inputs = _discover_inputs(tmp_path, (read_manifest(tmp_path / "derivatives" / "Segmentation" / "manifest.json"),), subject_id=None, site=None, use_common_region=False)

    assert len(inputs) == 2
    by_stack = {item.bone_segmentation.stack_index: item for item in inputs}
    assert by_stack[None].bone_segmentation.path == bone_path
    assert by_stack[None].trabecular_mask.path == trab_path
    assert by_stack[1].bone_segmentation.path == duplicate_bone_path
    assert by_stack[1].trabecular_mask.path == duplicate_trab_path


def test_plate_rod_manifest_replaces_matching_registered_and_native_outputs(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab = _tiny_bone()
    trab_path = tmp_path / "inputs" / "trab.npy"
    common_path = tmp_path / "inputs" / "common.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, trab)
    np.save(common_path, np.ones_like(trab))
    records = (
        DerivativeRecord("Segmentation", "trabecular_mask", "SAMPLE001", "tibia", "1", None, "native", trab_path, "provided"),
        DerivativeRecord("Segmentation", "bone_segmentation", "SAMPLE001", "tibia", "1", None, "native", trab_path, "provided"),
        DerivativeRecord("CommonRegion", "scan_region_native_common", "SAMPLE001", "tibia", "1", None, "native", common_path, "generated"),
    )
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "fixture", "version": "1"}, records),
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
    )

    run_plate_rod_batch(tmp_path, use_common_region=False, parameters=PlateRodParameters(skeletonize=False))
    run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        require_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
    )
    run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        require_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
        force=True,
    )

    tables = [
        record
        for record in read_manifest(tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json").records
        if record.role == "plate_rod_measurements_table"
    ]
    assert len(tables) == 2
    assert {bool(record.metadata.get("use_common_region")) for record in tables} == {False, True}


def test_native_and_registered_plate_rod_measurements_are_independent_outputs(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab = _tiny_bone()
    common = np.zeros_like(trab)
    common[2, 2, 1:3] = True
    trab_path = tmp_path / "inputs" / "trab.npy"
    common_path = tmp_path / "inputs" / "common.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, trab)
    np.save(common_path, common)
    _write_segmentation_manifest(tmp_path, bone_path=trab_path, common_path=common_path)

    native = run_plate_rod_batch(tmp_path, use_common_region=False, parameters=PlateRodParameters(skeletonize=False))
    registered = run_plate_rod_batch(
        tmp_path,
        use_common_region=True,
        require_common_region=True,
        parameters=PlateRodParameters(skeletonize=False),
    )

    native_table = next(record.path for record in native.records if record.role == "plate_rod_measurements_table")
    registered_table = next(
        record.path
        for record in registered.records
        if record.role == "plate_rod_measurements_table"
        and "/registered_measurements/" in str(record.path).replace("\\", "/")
    )
    manifest = read_manifest(tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json")
    manifest_tables = [
        str(record.path).replace("\\", "/")
        for record in manifest.records
        if record.role == "plate_rod_measurements_table"
    ]
    assert "/measurements/" in str(native_table).replace("\\", "/")
    assert "/registered_measurements/" in str(registered_table).replace("\\", "/")
    assert native_table.exists()
    assert registered_table.exists()
    assert any("/measurements/" in path for path in manifest_tables)
    assert any("/registered_measurements/" in path for path in manifest_tables)


def test_run_plate_rod_batch_keeps_unstacked_and_stack_one_outputs_distinct(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    records = []
    for stack in (None, 1, 2):
        suffix = "" if stack is None else f"_stack-{stack:02d}"
        trab_path = tmp_path / "inputs" / f"sub-SAMPLE001_ses-1_site-tibia{suffix}_mask-trabecular.npy"
        bone_path = tmp_path / "inputs" / f"sub-SAMPLE001_ses-1_site-tibia{suffix}_mask-seg.npy"
        trab_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(trab_path, _tiny_bone())
        np.save(bone_path, _tiny_bone())
        records.extend(
            [
                DerivativeRecord("Segmentation", "trabecular_mask", "SAMPLE001", "tibia", "1", stack, "native", trab_path, "provided"),
                DerivativeRecord("Segmentation", "bone_segmentation", "SAMPLE001", "tibia", "1", stack, "native", bone_path, "provided"),
            ]
        )
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "fixture", "version": "1"}, tuple(records)),
        tmp_path / "derivatives" / "Segmentation" / "manifest.json",
    )

    result = run_plate_rod_batch(
        tmp_path,
        subject_id="SAMPLE001",
        site="tibia",
        session_id="1",
        use_common_region=False,
        parameters=PlateRodParameters(skeletonize=False),
        dry_run=True,
    )

    assert len(result.records) == 9
    assert {record.stack_index for record in result.records} == {None, 1, 2}


def test_plate_rod_batch_progress_reports_backend(tmp_path: Path) -> None:
    from plate_rod_thinning.batch import run_plate_rod_batch
    from plate_rod_thinning.pipeline import PlateRodParameters

    trab = _tiny_bone()
    trab_path = tmp_path / "inputs" / "trab.npy"
    trab_path.parent.mkdir(parents=True)
    np.save(trab_path, trab)
    _write_segmentation_manifest(tmp_path, bone_path=trab_path)
    events = []

    run_plate_rod_batch(
        tmp_path,
        use_common_region=False,
        parameters=PlateRodParameters(skeletonize=False),
        progress=events.append,
    )

    started = [event for event in events if event.status == "started"]
    assert started
    assert "backend=" in started[0].message
