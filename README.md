# plate-rod-thinning

`plate-rod-thinning` provides topology-preserving trabecular bone plate/rod thinning and morphometry for HR-pQCT-style binary masks.

The package ports the original MATLAB plate/rod workflow into Python while keeping the core algorithm outside the Slicer Bone Imaging Toolbox. It produces 1-voxel skeleton topology labels, full-thickness plate/rod/junction labels, individual full-thickness component labels, and summary morphometry metrics.

## Method

The default pipeline uses topology-preserving thinning followed by Saha/Stauber-style local skeleton classification. Final topology classes are decomposed into individual trabecular elements using an ITS-like graph workflow:

1. classify skeleton voxels into surface, curve, and typed junction classes;
2. dilate typed junction zones before element decomposition;
3. label plate and rod components after junction-zone removal;
4. reconstruct individual element labels back into the full-thickness trabecular mask;
5. count P-P, P-R, and R-R junctions from unique neighboring reconstructed element pairs.

The current defaults provide a conservative decomposition for HR-pQCT-style trabecular masks:

```python
PlateRodParameters(
    junction_dilation_voxels=2,
    min_plate_voxels=0,
    min_rod_voxels=5,
    junction_support_radius_voxels=None,
)
```

`junction_support_radius_voxels` is available for experiments that require reconstructed junctions to be spatially supported by matching typed skeleton junction classes. It is disabled by default because this package treats the reconstructed element graph as the primary output.

## Validation Notes

The defaults were checked on an HR-pQCT II tibia test case to confirm that the outputs are numerically plausible and internally consistent for trabecular bone analysis. This package implements an open, auditable plate/rod thinning and graph morphometry workflow; it is not intended to be a bitwise or parameter-matched clone of any proprietary ITS implementation.

## Install

```bash
python -m pip install plate-rod-thinning
```

Prebuilt macOS wheels include the compiled C backend when available. The package also has a Python fallback, and on Apple platforms the Metal skeletonizer can be enabled with:

```bash
export PLATE_ROD_USE_METAL_FULL=1
```

## Citation

If you use the plate/rod network connectivity or morphometry workflow, please cite:

Walle M, Yeritsyan D, Abbasian M, Oftadeh R, Müller R, Nazarian A. A graph model to describe the network connectivity of trabecular plates and rods. Front Bioeng Biotechnol. 2024 May 6;12:1384280. doi: 10.3389/fbioe.2024.1384280. PMID: 38770275; PMCID: PMC11103010.

## Development

The compiled extension is opt-in for local source builds:

```bash
PLATE_ROD_BUILD_EXT=1 python -m pip install .
python -m pytest -q
```
