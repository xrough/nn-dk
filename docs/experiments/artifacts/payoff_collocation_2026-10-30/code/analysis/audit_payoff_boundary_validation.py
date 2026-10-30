"""Replay fixed-unit robust validation losses and every physical stratum."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import (  # noqa: E402
    bind_analysis_dependencies,
    bind_study,
    isolated_module,
)
from nn_dk.payoff_boundary import BoundaryPayoffModel  # noqa: E402

_implementation = isolated_module(
    "audit_payoff_regimes_validation.py", "analysis._boundary_regimes_validation_audit_impl")
_original_summary = _implementation.residual_summary


def normalized_residual(value, horizons, scaling, horizon, stage):
    if stage not in ("mean", "covariance"):
        raise ValueError("Use mean or covariance residuals")
    if (scaling.get("inverse_time_weighting") is not False
            or scaling.get("normalization_time_powers") != [0, 0, 0, 0]):
        raise ValueError("Fixed-unit scaling must explicitly disable inverse-time weights")
    return horizon*np.asarray(value, dtype=float)/np.asarray(scaling[stage+"_scale"])


def robust_values(value, delta):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Positive finite fixed pseudo-Huber parameter required")
    return 2*value**2/(np.hypot(1., value/delta)+1.)


def residual_summary(value, horizons, scaling, model, stage):
    result = _original_summary(value, horizons, scaling, model, stage)
    if result.get("empty"):
        return result
    normalized = normalized_residual(value, horizons, scaling, model.horizon, stage)
    result["normalized_squared_objective"] = result["normalized_objective"]
    result["normalized_objective"] = float(np.mean(robust_values(
        normalized, scaling["pseudo_huber_delta"])))
    result["objective_definition"] = "mean(2*delta^2*(sqrt(1+(normalized/delta)^2)-1))"
    result["inverse_time_weighting"] = False
    result["pseudo_huber_delta"] = scaling["pseudo_huber_delta"]
    return result


bind_study(_implementation, __file__,
           KIND="postfit-pre-reference-boundary-repair-robust-stratified-validation-v1",
           RepairPayoffModel=BoundaryPayoffModel,
           normalized_residual=normalized_residual, residual_summary=residual_summary)
bind_analysis_dependencies(_implementation, run_payoff_regimes=runner)


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
