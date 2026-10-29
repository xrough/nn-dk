"""Independent native coefficients and dense Hessians check the new ansatz."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_payoff_boundary_formulation import (  # noqa: E402
    absorption_and_finiteness_control,
    dense_derivative_control,
    gate_boundary_order_control,
    initial_corner_control,
    loss_scaling_control,
    native_interaction_anchor_control,
    small_time_covariance_control,
)


def test_native_initial_slopes_hold_with_arbitrary_heads_near_boundary():
    torch.set_num_threads(1)
    result = initial_corner_control()
    assert len(result["records"]) == 6
    assert result["arbitrary_heads"]


def test_all_covariance_leading_orders_including_nonzero_cross_block():
    torch.set_num_threads(1)
    result = small_time_covariance_control()
    assert result["shared_mode_cross_is_nonzero"]
    assert result["records"][-1]["leading_coefficient_relative_error"] < 2e-6


def test_moving_gate_and_native_quadratic_anchor_jets_match_dense_hessians():
    torch.set_num_threads(1)
    result = dense_derivative_control()
    assert len(result["records"]) == 2


def test_absorbed_values_psd_and_near_corner_parameter_gradients():
    torch.set_num_threads(1)
    result = absorption_and_finiteness_control()
    assert len(result["records"]) == 8
    assert all(row["all_jets_residuals_parameter_gradients_finite"]
               for row in result["records"])


def test_erf_gate_has_linear_boundary_order_and_flat_initial_time_partials():
    torch.set_num_threads(1)
    result = gate_boundary_order_control()
    assert result["zero_time_cell_factors_one_all_partials_zero"]


def test_fixed_unit_robust_loss_has_bounded_time_independent_row_influence():
    torch.set_num_threads(1)
    result = loss_scaling_control()
    assert result["inverse_time_weighting"] is False
    assert len(result["records"]) == 2


def test_native_quadratic_anchor_matches_independent_mass_conserving_stencils():
    torch.set_num_threads(1)
    result = native_interaction_anchor_control()
    assert result["small_time"][-1]["mean_slope_relative_error"] < 5e-6
