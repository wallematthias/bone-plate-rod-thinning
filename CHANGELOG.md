# Changelog

## 0.1.8 - 2026-09-03

### Changed

- Build the compiled C backend by default for wheel and source installs. Set `PLATE_ROD_BUILD_EXT=0` only when a pure-Python fallback install is explicitly needed.

### Added

- Added a standard trusted-publishing workflow for PyPI releases.

## 0.1.3 - 2026-08-29

### Added

- Added ITS-like graph decomposition controls for typed junction dilation, minimum plate size, minimum rod size, and optional topology-supported junction filtering.
- Added regression tests for junction-zone element splitting and typed-support junction filtering.
- Documented current default parameters and the scope of validation.

### Changed

- Updated the public pipeline defaults to `junction_dilation_voxels=2`, `min_plate_voxels=0`, and `min_rod_voxels=5`.
- Full-thickness element morphometry now measures reconstructed graph elements after junction-zone decomposition instead of relying only on collapsed plate/rod skeleton components.
- Junction densities in the ITS decomposition path are counted from unique neighboring reconstructed element pairs.

### Notes

- The current defaults were sanity-checked on an HR-pQCT II tibia test case for plausible plate/rod graph morphometry output.
- This package is an open, auditable plate/rod implementation, not a parameter-matched clone of proprietary ITS software.
