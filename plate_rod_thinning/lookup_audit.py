from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat


_MATLAB_ROOT_ENV = os.environ.get("PLATE_ROD_MATLAB_ROOT")
MATLAB_ROOT = Path(_MATLAB_ROOT_ENV).expanduser() if _MATLAB_ROOT_ENV else None

CLASS_EFFECTIVE_POINT_COUNTS = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 4,
    6: 4,
    7: 7,
    8: 12,
    9: 20,
}

def _table_paths() -> dict[str, Path]:
    if MATLAB_ROOT is None:
        raise FileNotFoundError(
            "MATLAB lookup-table audits require PLATE_ROD_MATLAB_ROOT to point "
            "to the legacy matdevelopment directory."
        )
    return {
        "classification": MATLAB_ROOT / "CI_Classification" / "lkt_data.mat",
        "thinning": MATLAB_ROOT / "SK_Skeleton" / "lktsk_data.mat",
    }


def load_lookup_tables(kind: str) -> dict[int, np.ndarray]:
    """Load MATLAB lookup tables as class-id keyed NumPy arrays."""
    table_paths = _table_paths()
    if kind not in table_paths:
        expected = ", ".join(sorted(table_paths))
        raise ValueError(f"unknown lookup table kind {kind!r}; expected {expected}")

    raw = loadmat(table_paths[kind])
    return {i: np.asarray(raw[f"class{i}"]) for i in range(10)}


def classify_spoint_configuration(spoints: list[int] | tuple[int, ...] | np.ndarray) -> int:
    """Classify a six-s-neighbor occupancy pattern using the MATLAB/Saha classes."""
    c = np.asarray(spoints, dtype=bool)
    if c.shape != (6,):
        raise ValueError("spoints must contain six 6-neighbor occupancy values")

    num_spoints = int(c.sum())
    opposite_pairs = ((0, 5), (2, 4), (1, 3))
    num_opposite = 2 * sum(bool(c[i] and c[j]) for i, j in opposite_pairs)
    num_adjacent = num_spoints - num_opposite

    if num_spoints == 6:
        return 0
    if num_spoints == 5:
        return 1
    if num_opposite == 4:
        return 2
    if num_opposite == 2 and num_adjacent == 2:
        return 3
    if num_opposite == 2 and num_adjacent == 1:
        return 4
    if num_adjacent == 3:
        return 5
    if num_opposite == 2:
        return 6
    if num_adjacent == 2:
        return 7
    if num_adjacent == 1:
        return 8
    if num_spoints == 0:
        return 9
    raise ValueError(f"unclassified s-point configuration: {c.astype(int).tolist()}")


def audit_lookup_tables() -> dict[str, list[str]]:
    """Run first-pass consistency checks on the old MATLAB lookup tables."""
    errors: list[str] = []
    warnings: list[str] = []

    classification = load_lookup_tables("classification")
    thinning = load_lookup_tables("thinning")

    for class_id, effective_count in CLASS_EFFECTIVE_POINT_COUNTS.items():
        expected_rows = 2**effective_count
        if classification[class_id].shape != (expected_rows, 4):
            errors.append(
                f"classification class {class_id} has shape "
                f"{classification[class_id].shape}, expected {(expected_rows, 4)}"
            )
        if thinning[class_id].shape != (expected_rows, 2):
            errors.append(
                f"thinning class {class_id} has shape "
                f"{thinning[class_id].shape}, expected {(expected_rows, 2)}"
            )

    if np.array_equal(classification[0][:, 1:], np.array([[1, 0, 1]])):
        warnings.append(
            "The MATLAB lookup table sets delta=1 only for class 0, where all six "
            "s-points are black. This disagrees with the thesis sentence saying "
            "delta is always 1 except when all s-points are black; that sentence "
            "should be treated as a likely typo until checked against Saha 1996."
        )
    else:
        errors.append("classification class 0 does not match expected cavity case [eps=1, mu=0, delta=1]")

    for class_id in range(1, 10):
        if not np.all(classification[class_id][:, 3] == 0):
            errors.append(f"classification class {class_id} contains nonzero delta values")

    warnings.append(
        "The legacy sk_definitions.m s-open branch returns true when any 5x5x5 "
        "6-neighbor position is black/bone. The thesis definition says s-open "
        "requires at least one s-point to be white/background. Treat this as a "
        "candidate sign or convention mismatch before porting the thinning loop."
    )

    return {"errors": errors, "warnings": warnings}
