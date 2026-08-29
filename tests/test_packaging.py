from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_matches_slicer_import_expectations() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "plate-rod-thinning"
    assert pyproject["project"]["version"] == "0.1.0"
    assert pyproject["project"]["readme"] == "README.md"
    assert (ROOT / "README.md").exists()
    assert "numpy" in pyproject["project"]["dependencies"]
    assert "scipy" in pyproject["project"]["dependencies"]
    assert "scikit-image" not in pyproject["project"]["dependencies"]


def test_compiled_backend_is_opt_in_for_slicer_safe_installs() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert (ROOT / "plate_rod_thinning" / "_c_backend.c").exists()
    assert 'PLATE_ROD_BUILD_EXT") != "1"' in setup_source
    assert '"plate_rod_thinning._c_backend"' in setup_source
    assert "numpy" in pyproject["build-system"]["requires"]
    assert "include plate_rod_thinning/_c_backend.c" in manifest


def test_github_actions_build_macos_binary_wheels_for_slicer() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build-wheels.yml").read_text(encoding="utf-8")

    assert "cibuildwheel" in workflow
    assert "macos-15" in workflow
    assert "macos-15-intel" in workflow
    assert "macos-arm64" in workflow
    assert "macos-x86_64" in workflow
    assert "CIBW_ARCHS: ${{ matrix.cibw_arch }}" in workflow
    assert 'PLATE_ROD_BUILD_EXT: "1"' in workflow
    assert 'CIBW_BUILD: "cp312-* cp313-*"' in workflow
    assert "plate_rod_thinning._c_backend" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "https://pypi.org/p/plate-rod-thinning" in workflow


def test_github_actions_can_publish_artifacts_from_existing_run() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish-from-run.yml").read_text(encoding="utf-8")

    assert "source_run_ids" in workflow
    assert "actions/runs/${run_id}/artifacts" in workflow
    assert "twine check dist/*" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
