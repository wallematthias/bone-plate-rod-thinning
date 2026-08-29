from __future__ import annotations

import json
import numpy as np
import pytest
import shutil

from plate_rod_thinning import metal_backend
from plate_rod_thinning.topology import neighborhood_keys_3x3_at


def test_metal_backend_reports_capability_without_required_gpu():
    status = metal_backend.status()

    assert status.backend == "metal"
    assert isinstance(status.available, bool)
    assert status.reason


def test_metal_helper_source_is_packaged_with_runtime_shader():
    source = metal_backend.helper_source_path()

    assert source.exists()
    text = source.read_text()
    assert "MTLCreateSystemDefaultDevice" in text
    assert "makeLibrary(source:" in text


def test_metal_probe_runs_when_swift_compiler_is_available():
    if shutil.which("swiftc") is None:
        return

    result = metal_backend.probe()
    payload = json.loads(result.stdout)

    assert "available" in payload
    if payload["available"]:
        assert payload["device"]


def test_metal_neighborhood_keys_match_reference_when_available():
    if not metal_backend.status().available:
        pytest.skip("Metal helper is not available")
    image = np.zeros((7, 7, 7), dtype=bool)
    image[1:6, 2:5, 2:6] = True
    coords = np.asarray([(2, 3, 3), (4, 4, 4), (5, 2, 2)], dtype=np.int64)

    assert np.array_equal(
        metal_backend.neighborhood_keys_3x3_at(image, coords),
        neighborhood_keys_3x3_at(image, coords),
    )


def test_metal_skeletonize_surface_matches_reference_when_available():
    if not metal_backend.status().available:
        pytest.skip("Metal helper is not available")
    image = np.zeros((9, 9, 9), dtype=bool)
    image[2:7, 2:7, 4:6] = True

    assert np.array_equal(
        metal_backend.skeletonize_surface(image, max_iterations=3),
        metal_backend.reference_skeletonize_surface(image, max_iterations=3),
    )
