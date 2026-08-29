# plate-rod-thinning

`plate-rod-thinning` provides topology-preserving trabecular bone plate/rod thinning and morphometry for HR-pQCT-style binary masks.

The package ports the original MATLAB plate/rod workflow into Python while keeping the core algorithm outside the Slicer Bone Imaging Toolbox. It produces 1-voxel skeleton topology labels, full-thickness plate/rod/junction labels, individual full-thickness component labels, and summary morphometry metrics.

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
