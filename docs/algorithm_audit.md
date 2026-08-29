# Plate/Rod Thinning Algorithm Audit

Date: 2026-08-28

## Scope

This note starts the audit for porting the legacy MATLAB plate/rod skeletonization code. Local lookup-table comparisons can be run by setting `PLATE_ROD_MATLAB_ROOT` to the legacy `matdevelopment` directory.

The immediate goal is not to reproduce MATLAB `bwskel`. The goal is to port and improve the custom topology-preserving thinning path:

1. `SK_Skeleton/sk_thinning3D.m`
2. `SK_Skeleton/sk_definitions.m`
3. `SK_Skeleton/sk_preserve.m`
4. `SK_Skeleton/sk_c3_ref.m`
5. `SK_Skeleton/sk_c456_ref.m`
6. `SK_Skeleton/LKTSK_LookupSkeleton/*.m`
7. `CI_Classification/ci_classify_image.m`
8. `CI_Classification/ci_platerodimage.m`
9. `CI_Classification/ci_deskeltonize.m`

The larger graph/statistics layer in `XH_Helpers/trabecular_skeleton.m` should be ported after the local topology, thinning, classification, and label expansion are verified.

## Source Trail

Primary intended sources named by the MATLAB code and thesis:

- P. K. Saha, B. B. Chaudhuri, D. Dutta Majumder. "A new shape preserving parallel thinning algorithm for 3D digital images." Pattern Recognition 30(12), 1939-1955, 1997. DOI: `10.1016/S0031-3203(97)00016-2`.
- P. K. Saha, B. B. Chaudhuri. "3D Digital Topology under Binary Transformation with Applications." Computer Vision and Image Understanding 63(3), 418-429, 1996. DOI: `10.1006/cviu.1996.0032`.
- M. Stauber, R. Muller. "Volumetric spatial decomposition of trabecular bone into rods and plates--A new method for local bone morphometry." Bone 38, 475-484, 2006. DOI: `10.1016/j.bone.2005.09.019`.
- T. M. Bernard, A. Manzanera. "Improved Low Complexity Fully Parallel Thinning Algorithm." ICIAP 1999.
- M. J. E. Golay. "Hexagonal Parallel Pattern Transformations." IEEE Transactions on Computers C-18(8), 733-740, 1969.

Access note: direct download of the Saha 1996 and 1997 publisher PDFs returned HTTP 403 in this environment. The local thesis source and bibliography were available, and the Bernard/Manzanera PDF was accessible. Exact template-by-template verification against the Saha figures remains open until the PDFs or scans are available.

## What The MATLAB Code Implements

### Thinning

`sk_thinning3D.m` implements an iterative two-version thinning process:

- Pad by two voxels.
- Encode marrow/background as `-intmax` and unmarked bone as `0`.
- At each iteration, identify outer-layer candidates with `sk_border_config`.
- Split candidates into 8 subfields with `sk_disjoint_image`.
- Process s-open, e-open, and v-open candidates by subfield.
- Preserve shape points using Saha conditions 1/2.
- Erode s-open and v-open simple points.
- Erode e-open points using Saha condition 3.
- Stop when the number of candidate voxels is unchanged between iterations.
- Run `sk_final_erode` using Saha conditions 4/5/6 to reduce remaining
  two-voxel-thick surfaces.

This is a genuine topology-driven thinning implementation. It is not equivalent to calling MATLAB `bwskel`.

### Surface/Outer-Layer Definitions

The thesis defines the surface as the union of s-open, e-open, and v-open points. The MATLAB implementation maps these definitions in `sk_definitions.m` using 5x5x5 linear indices.

Initial audit status:

- s-open: the legacy code checks the six 6-neighbor positions, but its boolean convention appears inconsistent with the thesis wording.
- e-open: code checks several e-point/f1 combinations after excluding s-open.
- v-open: code checks several v-point/f1/f2-style combinations after excluding s-open and e-open.

Correction from the first Python porting pass: the legacy code does **not** currently check whether one of the six 6-neighbor positions is white if `true` means bone. In `sk_border_config.m`, the local image passed to `sk_definitions.m` is `image >= thr`, so bone voxels are `true` and background voxels are `false`. Under that convention, `sk_definitions.m` labels a fully filled 5x5x5 neighborhood as `sopen`, because its first branch is `c(38) || c(58) || c(62) || c(64) || c(68) || c(88)`.

This conflicts with the thesis definition, where an s-open point requires at least one s-point of `N_p` to be white/background before the iteration. Treat this as a candidate sign or convention mismatch. The Python code keeps the direct behavior as `legacy_border_point_type` for parity, but the corrected thinning implementation should not adopt it blindly.

Open audit item: map every hard-coded index in `sk_definitions.m` back to named points `a,b,c,d,e,f`, `e(a,b,p)`, `v(a,b,c,p)`, `f1`, `f2`, and `f3`. This should be done before using this file in the corrected thinning loop, because direct translation of numeric indices would preserve any old mistakes.

### Simple Points And Lookup Tables

The MATLAB code uses two lookup-table families:

- `CI_Classification/lkt_data.mat`: `[comb, eps, mu, delta]`
- `SK_Skeleton/lktsk_data.mat`: `[comb, simplepoint]`

The generated table sizes match the expected number of effective point configurations:

| Class | Effective points | Rows |
| --- | ---: | ---: |
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 2 | 0 | 1 |
| 3 | 1 | 2 |
| 4 | 2 | 4 |
| 5 | 4 | 16 |
| 6 | 4 | 16 |
| 7 | 7 | 128 |
| 8 | 12 | 4096 |
| 9 | 20 | 1048576 |

Initial audit status:

- The MATLAB class selection in `ci_identify_class.m` and `sk_identify_class.m` is clear and portable.
- The lookup generation uses object 26-connectivity and background 6-connectivity, consistent with the thesis text.
- The lookup key generation is fragile: it converts bit arrays through string formatting. Python should use direct bit packing.
- The canonical rotation step is fragile: it rotates randomly until the s-point pattern matches the base class pattern. Python should enumerate the 24 cube rotations deterministically and choose a canonical transform.

### Delta Definition Mismatch

The thesis text says: "`delta(p)` is always equal to 1 except when all s-points are black points."

The MATLAB classification lookup generator does the opposite:

- `delta = 0` for ordinary classes.
- `delta = 1` only when all six s-points are black.

This matches the code's simple-point condition, where simple points require no cavity. The thesis sentence should be treated as a likely wording error until checked directly against Saha 1996.

### Plate/Rod Classification

`ci_classify_image.m` performs two-stage skeleton classification:

1. Preclassification from `(eps, mu, delta)` into initial classes `N1` through `N8`.
2. Final classification from the 26-neighborhood of preclassified voxels into surface, arc, junction, and isolated types.

`ci_platerodimage.m` then collapses final classes:

- final classes `< 3` become plate/surface labels
- final classes `> 2` become rod/arc/junction labels
- labels are expanded back into the original binary mask using `ci_deskeltonize`

This is the part that should become a clean Python API:

- `skeletonize_surface(binary)`
- `classify_skeleton(skeleton)`
- `plate_rod_image(binary)`
- `plate_rod_components(binary)`
- `plate_rod_statistics(binary, spacing=None)`

## Porting Rules

The Python implementation should deliberately improve these pieces:

1. Use named neighbor coordinates instead of MATLAB linear indices.
2. Generate lookup tables from explicit definitions, then cache them.
3. Compare generated Python lookup tables against the old `.mat` tables.
4. Replace randomized rotations with deterministic cube rotations.
5. Add rotation-invariance tests for 90-degree rotations.
6. Add topology tests for synthetic components: solid block, slab, rod, tunnel, cavity, plate-rod junction, and disconnected objects.
7. Optimize only after correctness: start with clear NumPy/SciPy, then move hot local kernels to Numba or Cython.

## First Audit Harness

The first scaffold lives in:

- `plate_rod_thinning/lookup_audit.py`
- `plate_rod_thinning/topology.py`
- `tests/test_lookup_audit.py`
- `tests/test_topology.py`

It currently verifies:

- the MATLAB source tree and `.mat` lookup files are available through `PLATE_ROD_MATLAB_ROOT`;
- lookup table row counts match effective point counts;
- s-point class examples match the MATLAB/Saha class scheme;
- `delta=1` appears only in the class-0 all-six-s-points cavity case;
- the likely thesis delta wording mismatch is explicitly flagged.

## Port Progress

The first Python porting slice is in `plate_rod_thinning/topology.py`.

Implemented:

- MATLAB-compatible 1-based column-major index conversion for 3x3x3 neighborhoods.
- MATLAB-compatible 1-based column-major index conversion for 5x5x5 neighborhoods.
- Named constants for the center voxel, six s-points, and eight corner voxels.
- Base s-point configurations for Saha classes 0 through 9.
- Effective-point detection using the dead-surface rules from `sk_conf_to_eff.m` / `lkt_conf_to_eff.m`.
- Deterministic enumeration of all 24 proper cube rotations.
- Deterministic rotation of a 3x3x3 configuration into its class base s-point pattern.
- Python generation of the classification lookup table `[comb, eps, mu, delta]`.
- Python generation of the thinning lookup table `[comb, simplepoint]`.
- Direct legacy-index port of `sk_definitions.m` as `legacy_border_point_type`.
- Corrected border classification as `border_point_type`, where `True` means bone and s-open means at least one 6-neighbor is background.
- Arbitrary 3x3x3 neighborhood classification as `classify_neighborhood`.
- Arbitrary 3x3x3 simple-point testing as `simple_point`.
- ITS-style summary metrics: BV/TV, pBV/TV, rBV/TV, pBV/BV, rBV/BV,
  pBV/rBV, pTb.N, rTb.N, PR.N, pTb.Th, rTb.Th, pTb.S, rTb.l,
  P-P/P-R/R-R junction density, aBV/TV, component counts, and physical
  TV/BV/pBV/rBV volumes when voxel spacing is provided.

Verified so far:

- Generated classification lookup tables match MATLAB for classes 0-8.
- Generated simple-point lookup tables match MATLAB for classes 0-8.
- The legacy-index `sk_definitions.m` behavior is reproducible in Python for representative s-open, e-open, and v-open examples.
- Corrected border classification no longer labels a fully filled 5x5x5 neighborhood as s-open.
- Corrected border classification preserves the intended exclusivity order: s-open before e-open before v-open.
- Saha condition 3 (`sk_c3_ref`) is ported as `tunnel_preserving_e_point`.
- Saha conditions 4/5/6 (`sk_c456_ref`) are ported as `final_erosion_point`
  and used in the final erosion pass.
- The optional C backend now contains the same primary thinning, e-open, and
  final erosion branches as the Python reference.
- The opt-in Metal backend (`PLATE_ROD_USE_METAL_FULL=1`) contains the same
  primary thinning, e-open, and final erosion branches in Swift-hosted Metal
  kernels, and is byte-for-byte checked against the Python reference on
  synthetic cases and a real tibia mini-sample crop.
- The tibia mini sample full pipeline is locked by generated artifacts and an
  opt-in recompute test. The full recompute currently produces 77,760 skeleton
  voxels from 443,522 trabecular bone voxels.
- Arbitrary-neighborhood simple-point testing is stable under a 90-degree rotation example.
- Class 9 has 1,048,576 rows and should be checked by an explicit audit command rather than in the default fast unit tests.

## Next Work

1. Add an explicit slow audit command for class 9 lookup comparison.
2. Implement deterministic 5x5x5 coordinate maps.
3. Re-express `sk_definitions.m` using named coordinates and produce a table mapping each MATLAB index to a geometric point name.
4. Validate per-component and junction outputs against known MATLAB outputs
   on archived scans once matching `.mat` result tables are identified.
