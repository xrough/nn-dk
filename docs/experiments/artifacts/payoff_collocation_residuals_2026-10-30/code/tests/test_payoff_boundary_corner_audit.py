"""Controlled initial-time boundary-strip incompatibility, without data labels."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_boundary_corner as audit  # noqa: E402


def test_strict_interior_corner_defects_persist_under_arbitrary_corrections():
    result = audit.run_control()
    assert result["strictly_interior"] and result["minimum_mass"] > result["stopping_mass"]
    assert abs(result["barrier"]-.5) < 1e-12
    first, second = result["records"]
    np.testing.assert_allclose(first["observed_integral_mean_residual_at_zero"],
                               second["observed_integral_mean_residual_at_zero"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(first["observed_terminal_covariance_residual_at_zero"],
                               second["observed_terminal_covariance_residual_at_zero"], atol=2e-12, rtol=0)
    assert np.linalg.norm(first["exact_integral_mean_corner_defect"]) > .1
    assert np.linalg.norm(first["exact_terminal_covariance_corner_defect"]) > .1
    assert max(row["maximum_mean_identity_error"] for row in result["records"]) < 2e-12
    assert max(row["maximum_covariance_identity_error"] for row in result["records"]) < 2e-12
    assert result["no_checkpoint_reference_learner_or_final_test_data_opened"]
