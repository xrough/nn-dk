"""Independent pre-outcome audit for the fresh boundary-repair study."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import evaluate_payoff_boundary_regimes as evaluator  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import (  # noqa: E402
    bind_analysis_dependencies,
    bind_study,
    isolated_module,
)

_implementation = isolated_module(
    "audit_payoff_regimes.py", "analysis._boundary_regimes_pretest_audit_impl")
_historical_h5_inventory = _implementation.historical_identifier_inventory


def historical_identifier_inventory(runs, excluded_root, current):
    """Read HDF5/NPZ family IDs only; never historical states or outcomes."""
    runs, excluded_root, current = Path(runs).resolve(), Path(excluded_root).resolve(), set(current)
    identifiers, records = _historical_h5_inventory(runs, excluded_root, current)
    for path in sorted(runs.rglob("*.npz")):
        if path.resolve().is_relative_to(excluded_root):
            continue
        with np.load(path, allow_pickle=False) as store:
            keys = (store.files if path.name == "planned_initial_families.npz" else
                    [key for key in store.files if key in ("family_ids", "trajectory_ids")])
            seen, inventory = set(), {}
            for key in keys:
                values = store[key]
                if values.dtype.kind not in ("U", "S"):
                    raise ValueError("Historical family inventory must contain only string identifiers")
                decoded = values.astype(str).reshape(-1)
                unique = {value for value in decoded if value}
                seen.update(unique)
                inventory[key] = {"rows": len(decoded), "unique_identifiers": len(unique),
                                  "identifiers_hash": _implementation.stable_hash(sorted(unique))}
        if seen & current:
            raise ValueError(f"Historical NPZ trajectory-family overlap: {path}")
        identifiers.update(seen)
        if inventory:
            records.append({"path": str(path.resolve()), "unique_identifiers": len(seen),
                "identifiers_hash": _implementation.stable_hash(sorted(seen)),
                "datasets_read": inventory, "container": "NPZ-family-identifiers-only",
                "overlap_with_current": 0, "historical_states_or_outcomes_read": False})
    return identifiers, records


bind_study(_implementation, __file__,
           KIND="independent-boundary-repair-pretest-and-analytic-audit-v1",
           historical_identifier_inventory=historical_identifier_inventory)
bind_analysis_dependencies(_implementation,
                           evaluate_payoff_regimes=evaluator, run_payoff_regimes=runner)


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
