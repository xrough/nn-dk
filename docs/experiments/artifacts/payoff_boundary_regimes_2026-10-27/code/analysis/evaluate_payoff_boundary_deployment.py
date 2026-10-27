"""Fresh independent initial-law integration for boundary-repaired moments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import evaluate_payoff_boundary_regimes as original  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402

KIND = "independent-boundary-repair-neural-initial-integration-v1"
SEED = 2026102927
_implementation = isolated_module(
    "evaluate_payoff_regimes_deployment.py", "analysis._boundary_regimes_deployment_impl")
_original_evaluate = _implementation.evaluate


def evaluate(root, cases=None):
    records = _original_evaluate(root, cases)
    root = Path(root).resolve()
    study = runner.load_plan(root)
    for regime in runner.case_names(study) if cases is None else cases:
        for profile in runner.reference_configuration(regime, study).cases:
            group = profile+"__ensemble__0"
            original.add_legacy_contrasts(root, regime, group,
                root/"deployment/accuracy"/regime/group)
    return records


bind_study(_implementation, __file__, KIND=KIND, SEED=SEED,
           original=original, runner=runner, evaluate=evaluate)


def __getattr__(name):
    return getattr(_implementation, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--root", type=Path, default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--case", action="append")
    args = parser.parse_args(argv)
    result = (_implementation.prepare(args.root) if args.command == "prepare" else
              _implementation.evaluate(args.root, args.case))
    print(json.dumps({"complete": True, "command": args.command,
                      "records": len(result) if isinstance(result, list) else None}))


if __name__ == "__main__":
    main()
