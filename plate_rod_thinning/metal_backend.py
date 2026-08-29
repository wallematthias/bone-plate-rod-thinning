from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


@dataclass(frozen=True)
class MetalBackendStatus:
    backend: str
    available: bool
    reason: str
    device: str | None = None


def helper_source_path() -> Path:
    """Return the bundled Swift/Metal helper source path."""
    return Path(__file__).with_name("metal").joinpath("PlateRodMetal.swift")


@cache
def status() -> MetalBackendStatus:
    """Report whether the experimental Metal helper can run on this machine."""
    if shutil.which("swiftc") is None:
        return MetalBackendStatus(
            backend="metal",
            available=False,
            reason="swiftc is not available; install Xcode command line tools to build the helper.",
        )
    result = probe()
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return MetalBackendStatus(
            backend="metal",
            available=False,
            reason=(result.stderr or "Metal helper returned non-JSON output.").strip(),
        )
    return MetalBackendStatus(
        backend="metal",
        available=bool(payload.get("available")),
        reason=str(payload.get("reason") or "Metal helper is available."),
        device=payload.get("device"),
    )


def probe() -> subprocess.CompletedProcess[str]:
    """Compile and run the Swift/Metal helper probe.

    The helper uses Metal's runtime shader compiler, so this does not require
    the separate ``xcrun metal`` command. That keeps the backend usable from a
    normal Slicer/Python install on Apple Silicon once a production kernel is
    added.
    """
    source = helper_source_path()
    if not source.exists():
        payload = {"available": False, "device": None, "reason": f"missing helper source: {source}"}
        return subprocess.CompletedProcess(args=["swiftc"], returncode=1, stdout=json.dumps(payload), stderr="")

    try:
        binary = _helper_binary_path()
    except RuntimeError as error:
        payload = {"available": False, "device": None, "reason": str(error)}
        return subprocess.CompletedProcess(args=["swiftc"], returncode=1, stdout=json.dumps(payload), stderr=str(error))
    return subprocess.run([str(binary), "--probe"], text=True, capture_output=True, check=False)


def neighborhood_keys_3x3_at(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Pack selected 3x3x3 neighborhoods on the Metal backend."""
    binary = np.ascontiguousarray(image, dtype=np.uint8)
    coord_array = np.asarray(coords, dtype=np.int64)
    if binary.ndim != 3 or coord_array.ndim != 2 or coord_array.shape[1] != 3:
        raise ValueError("image must be 3D and coords must have shape (n, 3)")
    if len(coord_array) == 0:
        return np.zeros(0, dtype=np.uint32)
    if np.any(coord_array <= 0) or np.any(coord_array[:, 0] >= binary.shape[0] - 1) or np.any(coord_array[:, 1] >= binary.shape[1] - 1) or np.any(coord_array[:, 2] >= binary.shape[2] - 1):
        raise ValueError("coords must be at least one voxel inside image bounds")

    binary_path: Path
    coords_path: Path
    keys_path: Path
    with tempfile.TemporaryDirectory(prefix="plate-rod-metal-keys-") as tmp:
        tmpdir = Path(tmp)
        binary_path = tmpdir / "image.u8"
        coords_path = tmpdir / "coords.u32"
        keys_path = tmpdir / "keys.u32"
        binary.tofile(binary_path)
        np.ascontiguousarray(coord_array, dtype=np.uint32).tofile(coords_path)
        helper = _helper_binary_path()
        result = subprocess.run(
            [
                str(helper),
                "--keys",
                str(binary_path),
                str(coords_path),
                str(keys_path),
                str(binary.shape[0]),
                str(binary.shape[1]),
                str(binary.shape[2]),
                str(len(coord_array)),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Metal helper failed").strip())
        return np.fromfile(keys_path, dtype=np.uint32, count=len(coord_array))


def skeletonize_surface(image: np.ndarray, *, max_iterations: int = 200) -> np.ndarray:
    """Run full-volume topology-preserving thinning through the Metal helper."""
    binary = np.ascontiguousarray(image, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError("skeletonize_surface expects a 3D array")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    with tempfile.TemporaryDirectory(prefix="plate-rod-metal-skeleton-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "image.u8"
        output_path = tmpdir / "skeleton.u8"
        binary.tofile(input_path)
        helper = _helper_binary_path()
        result = subprocess.run(
            [
                str(helper),
                "--skeletonize",
                str(input_path),
                str(output_path),
                str(binary.shape[0]),
                str(binary.shape[1]),
                str(binary.shape[2]),
                str(int(max_iterations)),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Metal helper failed").strip())
        output = np.fromfile(output_path, dtype=np.uint8, count=binary.size)
        if output.size != binary.size:
            raise RuntimeError("Metal helper returned an incomplete skeleton")
        return output.reshape(binary.shape).astype(np.bool_)


def reference_skeletonize_surface(image: np.ndarray, *, max_iterations: int = 200) -> np.ndarray:
    """Return the Python reference skeletonizer for Metal parity tests."""
    from plate_rod_thinning.skeletonize import skeletonize_surface as _reference

    return _reference(image, max_iterations=max_iterations)


def _compile_helper(binary: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["swiftc", str(helper_source_path()), "-O", "-o", str(binary)],
        text=True,
        capture_output=True,
        check=False,
    )


def _helper_binary_path() -> Path:
    if shutil.which("swiftc") is None:
        raise RuntimeError("swiftc is not available; cannot build Metal helper")
    source = helper_source_path()
    if not source.exists():
        raise RuntimeError(f"missing Metal helper source: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_dir = Path(os.environ.get("PLATE_ROD_METAL_CACHE", Path.home() / ".cache" / "plate-rod-thinning"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / f"PlateRodMetal-{digest}"
    if binary.exists():
        return binary
    tmp_binary = binary.with_suffix(".tmp")
    compile_result = _compile_helper(tmp_binary)
    if compile_result.returncode != 0:
        raise RuntimeError(compile_result.stderr.strip())
    tmp_binary.replace(binary)
    return binary
