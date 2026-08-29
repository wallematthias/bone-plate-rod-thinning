from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from plate_rod_thinning import metal_backend
from plate_rod_thinning.pipeline import PlateRodParameters, plate_rod_analysis
from plate_rod_thinning.skeletonize import skeletonize_surface


MINI_SAMPLE_ROOT = Path(
    "/Users/matthias.walle/Documents/10_Data/HR-pQCT/SyntheticMiniBone/"
    "SAMPLE526_tibia_T1_T4_registered_shrink045"
)
SEG_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_mask-seg.nii.gz"
TRAB_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_mask-trab.nii.gz"
SKELETON_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_seg-trab_plate_rod_python_skeleton.nii.gz"
SKELETON_LABELS_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_seg-trab_plate_rod_python_skeleton-labels.nii.gz"
FULL_LABELS_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_seg-trab_plate_rod_python_full-labels.nii.gz"
SUMMARY_PATH = MINI_SAMPLE_ROOT / "sub-SAMPLE526MINI_site-tibia_seg-trab_plate_rod_python_summary.json"
UCSF_FULL_ROOT = Path(
    "/Users/matthias.walle/Documents/10_Data/HR-pQCT/UCSF_single/RegisteredMicroarchitecture/"
    "sub-SAMPLE345/site-tibia/native_space/ses-3/masks"
)
UCSF_FULL_SEG_PATH = UCSF_FULL_ROOT / "sub-SAMPLE345_ses-3_site-tibia_mask-seg.nii.gz"
UCSF_FULL_TRAB_PATH = UCSF_FULL_ROOT / "sub-SAMPLE345_ses-3_site-tibia_mask-trab.nii.gz"

LOCKED_SUMMARY = {
    "bone_voxels": 443522,
    "tissue_voxels": 3383952,
    "skeleton_voxels": 77760,
    "plate_skeleton_voxels": 35263,
    "rod_skeleton_voxels": 38883,
    "plate_full_thickness_voxels": 176342,
    "rod_full_thickness_voxels": 242446,
    "junction_count": 2677,
    "component_count_26_connected": 1989,
    "largest_component_voxels": 73451,
}

LOCKED_FLOAT_SUMMARY = {
    "BV/TV": 0.1310662798999513,
    "pBV/BV": 0.39759470781607226,
    "rBV/BV": 0.5466380472671029,
    "pBV/rBV": 0.7273454707440007,
    "pTb.N": 1.2694153101262546,
    "rTb.N": 1.5816025086338295,
    "PR.N": 0.8026133641016803,
    "P-P Junc.D": 0.007928508385283686,
    "P-R Junc.D": 2.1922325685309394,
    "R-R Junc.D": 1.337275080984515,
}


def test_tibia_mini_generated_outputs_match_locked_summary():
    nib = pytest.importorskip("nibabel")
    for path in (SKELETON_PATH, SKELETON_LABELS_PATH, FULL_LABELS_PATH, SUMMARY_PATH):
        if not path.exists():
            pytest.skip(f"mini-sample output is not present: {path}")

    skeleton = np.asanyarray(nib.load(str(SKELETON_PATH)).dataobj) > 0
    skeleton_labels = np.asanyarray(nib.load(str(SKELETON_LABELS_PATH)).dataobj).astype(np.uint8)
    full_labels = np.asanyarray(nib.load(str(FULL_LABELS_PATH)).dataobj).astype(np.uint8)
    payload = json.loads(SUMMARY_PATH.read_text())
    summary = payload["summary"]

    assert int(skeleton.sum()) == LOCKED_SUMMARY["skeleton_voxels"]
    assert int(np.count_nonzero(skeleton_labels == 1)) == LOCKED_SUMMARY["plate_skeleton_voxels"]
    assert int(np.count_nonzero(skeleton_labels == 2)) == LOCKED_SUMMARY["rod_skeleton_voxels"]
    assert int(np.count_nonzero(full_labels == 1)) == LOCKED_SUMMARY["plate_full_thickness_voxels"]
    assert int(np.count_nonzero(full_labels == 2)) == LOCKED_SUMMARY["rod_full_thickness_voxels"]
    assert np.all(skeleton_labels[skeleton] > 0)

    for key, expected in LOCKED_SUMMARY.items():
        assert summary[key] == expected
    for key, expected in LOCKED_FLOAT_SUMMARY.items():
        assert summary[key] == pytest.approx(expected)


def test_tibia_mini_crop_metal_skeleton_matches_reference():
    nib = pytest.importorskip("nibabel")
    if not metal_backend.status().available:
        pytest.skip("Metal helper is not available")
    if not SEG_PATH.exists() or not TRAB_PATH.exists():
        pytest.skip("local tibia mini-sample inputs are not present")

    seg = np.asanyarray(nib.load(str(SEG_PATH)).dataobj) > 0
    trab = np.asanyarray(nib.load(str(TRAB_PATH)).dataobj) > 0
    crop = (seg & trab)[123:171, 106:154, 24:72]

    metal = metal_backend.skeletonize_surface(crop, max_iterations=8)
    reference = skeletonize_surface(crop, max_iterations=8)

    assert crop.shape == (48, 48, 48)
    assert int(crop.sum()) == 10216
    assert int(metal.sum()) == 1936
    assert np.array_equal(metal, reference)


@pytest.mark.skipif(
    os.environ.get("PLATE_ROD_RUN_LOCAL_DATA_TESTS") != "1",
    reason="set PLATE_ROD_RUN_LOCAL_DATA_TESTS=1 to run the local UCSF full-sample Metal benchmark",
)
def test_ucsf_full_sample_metal_skeleton_matches_backend_reference():
    nib = pytest.importorskip("nibabel")
    if not metal_backend.status().available:
        pytest.skip("Metal helper is not available")
    if not UCSF_FULL_SEG_PATH.exists() or not UCSF_FULL_TRAB_PATH.exists():
        pytest.skip("local UCSF full-sample inputs are not present")

    from plate_rod_thinning import backend

    seg = np.asanyarray(nib.load(str(UCSF_FULL_SEG_PATH)).dataobj) > 0
    trab = np.asanyarray(nib.load(str(UCSF_FULL_TRAB_PATH)).dataobj) > 0
    bone = seg & trab
    coords = np.argwhere(bone)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    crop = bone[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))]

    reference = backend.skeletonize_surface(crop, max_iterations=200)
    metal = metal_backend.skeletonize_surface(crop, max_iterations=200)

    assert crop.shape == (553, 491, 168)
    assert int(bone.sum()) == 6828536
    assert int(reference.sum()) == 991330
    assert np.array_equal(metal, reference)


@pytest.mark.skipif(
    os.environ.get("PLATE_ROD_RUN_LOCAL_DATA_TESTS") != "1",
    reason="set PLATE_ROD_RUN_LOCAL_DATA_TESTS=1 to recompute the local tibia mini-sample regression",
)
def test_tibia_mini_full_pipeline_recomputes_locked_summary():
    nib = pytest.importorskip("nibabel")
    if not SEG_PATH.exists() or not TRAB_PATH.exists():
        pytest.skip("local tibia mini-sample inputs are not present")

    seg_img = nib.load(str(SEG_PATH))
    trab_img = nib.load(str(TRAB_PATH))
    bone = (np.asanyarray(seg_img.dataobj) > 0) & (np.asanyarray(trab_img.dataobj) > 0)
    trab = np.asanyarray(trab_img.dataobj) > 0
    spacing = tuple(float(value) for value in seg_img.header.get_zooms()[:3])

    result = plate_rod_analysis(
        bone,
        analysis_mask=trab,
        parameters=PlateRodParameters(max_iterations=200, voxel_spacing_mm=spacing),
    )

    for key, expected in LOCKED_SUMMARY.items():
        assert result.summary[key] == expected
    for key, expected in LOCKED_FLOAT_SUMMARY.items():
        assert result.summary[key] == pytest.approx(expected)
