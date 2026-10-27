"""Audit fresh neural-initial pools without drawing any final histograms."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import audit_payoff_boundary_regimes as pretest  # noqa: E402
from analysis import evaluate_payoff_boundary_deployment as deployment  # noqa: E402
from analysis import evaluate_payoff_boundary_regimes as original  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import (  # noqa: E402
    bind_analysis_dependencies,
    bind_study,
    isolated_module,
)

_implementation = isolated_module(
    "audit_payoff_regimes_deployment.py", "analysis._boundary_regimes_deployment_audit_impl")
bind_study(_implementation, __file__,
           KIND="independent-boundary-repair-initial-pool-pretest-address-audit-v1",
           historical_identifier_inventory=pretest.historical_identifier_inventory)
bind_analysis_dependencies(_implementation, evaluate_payoff_regimes=original,
                           evaluate_payoff_regimes_deployment=deployment,
                           run_payoff_regimes=runner)


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
