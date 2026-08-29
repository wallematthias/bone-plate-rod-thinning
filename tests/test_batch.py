from pathlib import Path
import subprocess
import sys

import numpy as np

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
        [sys.executable, "-m", "plate_rod_thinning.cli", "run-batch", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BONE_DERIVATIVES_PROGRESS" in completed.stdout
    assert (tmp_path / "derivatives" / "PlateRodMorphometry" / "manifest.json").exists()
