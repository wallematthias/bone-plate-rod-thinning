# Plate/Rod Pipeline Implementation Plan

## Boundary

Core algorithm logic lives in this repository. SlicerBoneImagingToolbox should only provide input selection, parameter controls, output loading, display setup, and table/model visualization.

## Current State

- Topology-preserving thinning scaffold is ported from the MATLAB code.
- Lookup table generation/audit is implemented.
- A preview pipeline API exists in `plate_rod_thinning.pipeline`.
- The current classifier is explicitly `preview_degree`, not the final Saha/Stauber classifier.

## Production Pipeline

1. Run validated topology-preserving thinning on the trabecular bone mask.
2. Classify skeleton voxels into plate, rod, and junction classes using the original Saha/Stauber definitions.
3. Apply `slenderness` cleanup by iteratively removing surface-edge/slender plate voxels and reclassifying.
4. Remove small plate and rod components using user-controlled minimum sizes.
5. Propagate skeleton labels back into the full trabecular mask.
6. Build per-object component labels and junction graph tables.
7. Export summary statistics and optional visualization meshes.

## Slicer Outputs

- Skeleton labelmap: background, plate, rod, junction.
- Full-thickness plate/rod labelmap.
- Component labelmap.
- Measurement table.
- Optional plate surface mesh and rod line/tube model.

## Statistics

- Bone voxels and physical bone volume.
- Skeleton voxels by class.
- Full-thickness voxels and volume by class.
- Plate/rod fraction of trabecular bone volume.
- Plate and rod object counts.
- Component size distribution.
- Junction counts by type: plate-plate, plate-rod, rod-rod.
- Principal orientation vectors and anisotropy per component.

## Validation

- Synthetic topology cases: rods, plates, plate-rod junctions, tunnels, cavities, disconnected components.
- MATLAB fixture comparisons on legacy examples where the old code is believed correct.
- Mini HR-pQCT sample smoke test using `seg & trab`.
- Slicer wrapper tests should assert import/call boundaries and UI controls, not duplicate algorithm behavior.
