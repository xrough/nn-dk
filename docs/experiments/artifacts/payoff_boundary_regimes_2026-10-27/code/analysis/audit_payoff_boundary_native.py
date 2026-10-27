"""Replay native fresh-reference, candidate and initial-pool evidence."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import audit_payoff_boundary_regimes as pretest  # noqa: E402
from analysis import evaluate_payoff_boundary_deployment as deployment  # noqa: E402
from analysis import evaluate_payoff_boundary_regimes as evaluator  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402

_implementation = isolated_module(
    "audit_payoff_regimes_native.py", "analysis._boundary_regimes_native_audit_impl")
bind_study(_implementation, __file__,
           KIND="independent-completed-boundary-repair-native-evidence-audit-v1",
           pretest=pretest, evaluator=evaluator, deployment=deployment, runner=runner)


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
