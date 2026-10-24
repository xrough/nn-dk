"""Executable experiments; the same configuration controls every stage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ExperimentConfig, preset
from .convergence_config import ConvergenceConfig, convergence_preset
from .data import generate_dataset, json_ready, prepare_run, write_json
from .experiments import BudgetExhausted, calibrate, evaluate_run, fit, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Neural Dean–Kawasaki coarse dynamics benchmark")
    parser.add_argument("command", choices=("calibrate", "generate", "fit", "evaluate", "report", "run", "convergence"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--preset", choices=("smoke", "pilot", "n2048", "resolution64", "long-horizon",
                                           "a-smoke", "a-first-batch", "a-full", "a-final"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--budget-seconds", type=float, help="Per-invocation wall budget; checkpoints preserve progress")
    args = parser.parse_args(argv)
    if args.command == "convergence":
        return _convergence_main(args, parser)
    if args.preset and args.preset.startswith("a-"):
        parser.error("Project A presets require the convergence command")
    # Standalone stage calls can resolve the immutable saved configuration.
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.preset:
        config = preset(args.preset)
    elif args.output and (args.output / "config.json").exists():
        config = ExperimentConfig.load(args.output / "config.json")
    else:
        config = preset("smoke")
    directory = (args.output or Path("runs") / config.name / config.config_hash[:12]).resolve()
    manifest = prepare_run(directory, config)
    started = time.monotonic()
    budget = args.budget_seconds if args.budget_seconds is not None else config.budget_seconds
    if budget <= 0:
        parser.error("Budget must be positive")
    deadline = started + budget
    exit_code = 0
    try:
        commands = ("calibrate", "generate", "fit", "evaluate", "report") if args.command == "run" else (args.command,)
        for stage in commands:
            print(f"[{config.name}] {stage}: {directory}", flush=True)
            if stage == "calibrate":
                result = calibrate(directory, config, deadline)
            elif stage == "generate":
                result = generate_dataset(directory, config, deadline)
                if not result["complete"]:
                    raise BudgetExhausted("Generation checkpointed; resume the same run")
            elif stage == "fit":
                result = fit(directory, config, deadline)
            elif stage == "evaluate":
                result = evaluate_run(directory, config, deadline)
            else:
                result = report(directory, config)
            manifest["last_completed_stage"] = stage
            manifest["status"] = "complete" if stage == "report" else stage + "-complete"
            manifest.pop("error", None)
            write_json(directory / "manifest.json", manifest)
            print(json.dumps(json_ready({"stage": stage, "output": str(directory)})), flush=True)
    except BudgetExhausted as error:
        manifest["status"] = "budget-exhausted"
        print(str(error), file=sys.stderr)
        exit_code = 75
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
    return exit_code


def _convergence_main(args, parser):
    from .convergence import run_convergence

    if args.config:
        config = ConvergenceConfig.load(args.config)
    elif args.preset:
        if not args.preset.startswith("a-"):
            parser.error("The convergence command requires an a-* preset")
        config = convergence_preset(args.preset)
    elif args.output and (args.output / "config.json").exists():
        config = ConvergenceConfig.load(args.output / "config.json")
    else:
        config = convergence_preset("a-smoke")
    budget = config.budget_seconds if args.budget_seconds is None else args.budget_seconds
    if budget <= 0:
        parser.error("Budget must be positive")
    directory = (args.output or Path("runs") / config.name / config.config_hash[:12]).resolve()
    print(f"[{config.name}] A-only stopped DK convergence: {directory}", flush=True)
    try:
        result = run_convergence(directory, config, time.monotonic() + budget)
        print(json.dumps(json_ready(result)), flush=True)
        return 0
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
