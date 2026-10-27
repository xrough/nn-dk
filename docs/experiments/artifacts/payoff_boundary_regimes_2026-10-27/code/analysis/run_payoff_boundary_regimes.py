"""Fresh, locked harder-regime study of the boundary-consistent payoff solver.

Only the model/configuration hooks and independent stream namespace differ
from the frozen earlier runner. Native Euler reduction, Brownian refinement,
failure handling, strict fit locking and resumable chunk validation are reused
in an isolated implementation; historical module globals remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.payoff_boundary import (  # noqa: E402
    BoundaryPayoffModel,
    BoundaryRepairConfig,
    fit_boundary_payoff,
    load_boundary_payoff_model,
)

_implementation = isolated_module(
    "run_payoff_regimes.py", "analysis._boundary_regimes_runner_impl")
_historical_configuration = _implementation.configuration
_historical_design = _implementation._design

WORKFLOW = "boundary-consistent-native-dk-payoff-comparison-v1"
STREAM = "fresh-boundary-consistent-payoff-families-2026102901-v1"


@dataclass(frozen=True)
class StudyPlan(_implementation.StudyPlan):
    """Keep the scientific allocation fixed while drawing wholly fresh data."""

    train_seed: int = 2026102901
    reference_seed: int = 2026102902


def configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return replace(_historical_configuration(case, plan),
                   name=f"payoff-boundary-regimes-{case}")


def repaired_configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return BoundaryRepairConfig(configuration(case, plan),
                                quadrature_nodes=plan.quadrature_nodes)


def slots(plan=None):
    plan = StudyPlan() if plan is None else plan
    return [{"case": case, "seed": seed,
             "bank": _implementation.payoff_bank(case, plan).name,
             "name": f"{case}_boundary_seed{seed}", "variant": "boundary-repaired"}
            for case in _implementation.case_names(plan) for seed in plan.optimizer_seeds]


def load_declared_model(item):
    path = Path(item["checkpoint_path"])
    if item.get("variant", "boundary-repaired") != "boundary-repaired":
        raise ValueError("Only declared boundary-consistent models belong to this study")
    if item.get("checkpoint_sha256", _implementation._sha(path)) != _implementation._sha(path):
        raise ValueError("Declared boundary-consistent checkpoint changed")
    return load_boundary_payoff_model(path)


def untrained_model(case, plan=None):
    """Exactly the unfitted architecture used by this separately named study."""
    plan = StudyPlan() if plan is None else plan
    cfg = _implementation.reference_configuration(case, plan)
    repair = repaired_configuration(case, plan)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BoundaryPayoffModel(
            cfg.spec, cfg.context("uniform"), cfg.horizon,
            _implementation.payoff_bank(case, plan), width=repair.study.width,
            quadrature_nodes=repair.quadrature_nodes)
    return model.eval()


def _design(plan):
    return {**_historical_design(plan),
            "predecessor": "harder-native-dk-repaired-payoff-comparison-v1",
            "fresh_training_validation_reference_and_deployment_streams": True,
            "historical_analysis_modules_mutated": False,
            "boundary_model_configurations": {
                case: repaired_configuration(case, plan).to_dict()
                for case in _implementation.case_names(plan)}}


bind_study(
    _implementation, __file__,
    WORKFLOW=WORKFLOW, STREAM=STREAM, SCRIPT=Path(__file__).resolve(),
    HELPERS=("run_payoff_regimes.py", "payoff_study_adapter.py",
             "evaluate_payoff_boundary_regimes.py", "evaluate_payoff_boundary_deployment.py",
             "evaluate_payoff_regimes.py", "evaluate_payoff_regimes_deployment.py",
             "payoff_legacy_control.py", "audit_payoff_boundary_regimes.py",
             "audit_payoff_boundary_deployment.py", "audit_payoff_boundary_native.py",
             "audit_payoff_boundary_metrics.py", "audit_payoff_boundary_validation.py",
             "audit_payoff_boundary_formulation.py", "audit_payoff_regimes_deployment.py",
             "audit_payoff_regimes_native.py", "audit_payoff_regimes_metrics.py",
             "audit_payoff_regimes_validation.py",
             "benchmark_path_generalization.py", "benchmark_payoff_generalization.py",
             "evaluate_path_generalization.py", "probe_path_workloads.py",
             "path_generalization_reference.py", "run_payoff_repair.py",
             "compiled_native.py", "native_euler_control.cpp",
             "native_gaussian_baseline.py", "audit_payoff_regimes.py"),
    StudyPlan=StudyPlan, configuration=configuration,
    repaired_configuration=repaired_configuration, slots=slots,
    load_declared_model=load_declared_model, _design=_design,
    fit_repaired_payoff=fit_boundary_payoff,
    load_repaired_payoff_model=load_boundary_payoff_model,
)


def __getattr__(name):
    return getattr(_implementation, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibrate", "fit", "lock", "reference"))
    parser.add_argument("--root", type=Path,
                        default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args()
    if (not 0 < args.budget_seconds <= 8*60*60
            or (args.seed is not None) != (args.stage == "fit")
            or (args.plan is not None and args.stage != "prepare")
            or (args.case is not None and args.stage != "reference")
            or (args.max_chunks is not None
                and (args.stage != "reference" or args.max_chunks < 0))):
        parser.error("Use a positive <=8h budget; seed only for fit; plan only for prepare; "
                     "case/chunk limit only for reference")
    deadline = time.monotonic()+args.budget_seconds
    try:
        if args.stage == "prepare":
            plan = StudyPlan.from_dict(_implementation._read(args.plan)) if args.plan else None
            print(_implementation.prepare(args.root, plan, deadline), flush=True)
        elif args.stage == "calibrate":
            print(json.dumps(_implementation.calibration(args.root)), flush=True)
        elif args.stage == "fit":
            result = _implementation.fit_seed(args.root, args.seed, deadline)
            print(f"Completed seed {args.seed}: {len(result['models'])} fits", flush=True)
        elif args.stage == "lock":
            print(f"Locked {_implementation.lock(args.root)['expected_models']} fits", flush=True)
        else:
            result = _implementation.generate_reference(
                args.root, args.case, deadline, args.max_chunks)
            print(json.dumps({key: {name: value[name] for name in ("status", "committed_chunks")}
                              for key, value in result.items()}), flush=True)
    except _implementation.BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
