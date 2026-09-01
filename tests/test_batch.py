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


def test_run_plate_rod_batch_clips_bone_before_plate_rod_analysis(tmp_path: Path) -> None:
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
    assert not labels[~common].any()
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


def test_run_plate_rod_batch_prefers_trabecular_mask_over_bone_segmentation_for_one_case(tmp_path: Path) -> None:
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
    assert {record.inputs for record in result.records} == {(trab.record_id,)}


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
