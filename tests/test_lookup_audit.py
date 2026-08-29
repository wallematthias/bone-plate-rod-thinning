from pathlib import Path

import numpy as np
import pytest

from plate_rod_thinning.lookup_audit import (
    CLASS_EFFECTIVE_POINT_COUNTS,
    MATLAB_ROOT,
    audit_lookup_tables,
    classify_spoint_configuration,
    load_lookup_tables,
)

pytestmark = pytest.mark.skipif(
    MATLAB_ROOT is None or not MATLAB_ROOT.exists(),
    reason="set PLATE_ROD_MATLAB_ROOT to run local MATLAB lookup-table audits",
)


def test_matlab_source_tree_is_available():
    assert MATLAB_ROOT.exists()
    assert (MATLAB_ROOT / "SK_Skeleton" / "lktsk_data.mat").exists()
    assert (MATLAB_ROOT / "CI_Classification" / "lkt_data.mat").exists()


def test_loaded_lookup_tables_have_expected_class_sizes():
    classification = load_lookup_tables("classification")
    thinning = load_lookup_tables("thinning")

    for class_id, effective_count in CLASS_EFFECTIVE_POINT_COUNTS.items():
        expected_rows = 2**effective_count
        assert classification[class_id].shape == (expected_rows, 4)
        assert thinning[class_id].shape == (expected_rows, 2)


def test_spoint_configuration_classification_matches_saha_classes():
    assert classify_spoint_configuration([1, 1, 1, 1, 1, 1]) == 0
    assert classify_spoint_configuration([1, 1, 1, 1, 1, 0]) == 1
    assert classify_spoint_configuration([1, 1, 1, 0, 0, 0]) == 5
    assert classify_spoint_configuration([1, 0, 0, 0, 0, 1]) == 6
    assert classify_spoint_configuration([0, 0, 0, 0, 0, 0]) == 9


def test_classification_lookup_delta_is_only_cavity_case():
    classification = load_lookup_tables("classification")

    assert np.array_equal(classification[0][:, 1:], np.array([[1, 0, 1]]))
    for class_id in range(1, 10):
        assert np.all(classification[class_id][:, 3] == 0)


def test_audit_flags_known_thesis_delta_text_mismatch():
    findings = audit_lookup_tables()

    assert findings["errors"] == []
    assert any("delta" in warning.lower() for warning in findings["warnings"])
    assert any("s-open" in warning.lower() for warning in findings["warnings"])
