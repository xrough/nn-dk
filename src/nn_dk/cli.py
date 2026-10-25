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
    parser.add_argument("command", choices=("calibrate", "generate", "fit", "evaluate", "report", "run", "convergence", "bsde", "bsde-study", "generator", "path-solver"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--preset", choices=("smoke", "pilot", "n2048", "resolution64", "long-horizon",
                                           "a-smoke", "a-first-batch", "a-full", "a-final",
                                           "bsde-smoke", "bsde-first-batch", "bsde-refinement",
                                           "bsde-study-smoke", "bsde-anchor-study",
                                           "generator-smoke", "generator-accuracy",
                                           "path-smoke", "path-controls", "path-first-batch",
                                           "path-pilot"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--budget-seconds", type=float, help="Per-invocation wall budget; checkpoints preserve progress")
    parser.add_argument("--stage", choices=("all", "calibrate", "generate", "fit", "evaluate", "report"),
                        default="all", help="Independent stage of a statistics or density-path workflow")
    args = parser.parse_args(argv)
    if args.command == "path-solver":
        return _path_main(args, parser)
    if args.preset and args.preset.startswith("path-"):
        parser.error("Density-path presets require the path-solver command")
    if args.command == "generator":
        return _generator_main(args, parser)
    if args.preset and args.preset.startswith("generator-"):
        parser.error("Generator presets require the generator command")
    if args.command == "bsde-study":
        return _bsde_study_main(args, parser)
    if args.preset in ("bsde-study-smoke", "bsde-anchor-study"):
        parser.error("Study presets require the bsde-study command")
    if args.command == "bsde":
        return _bsde_main(args, parser)
    if args.stage != "all":
        parser.error("--stage is supported by the bsde command and bsde-study command")
    if args.preset and args.preset.startswith("bsde-"):
        parser.error("BSDE presets require the bsde command")
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


def _bsde_main(args, parser):
    from .bsde import run_bsde
    from .bsde_config import BSDEConfig, bsde_preset

    if args.config:
        config = BSDEConfig.load(args.config)
    elif args.preset:
        if not args.preset.startswith("bsde-"):
            parser.error("The bsde command requires a bsde-* preset")
        config = bsde_preset(args.preset)
    elif args.output and (args.output / "config.json").exists():
        config = BSDEConfig.load(args.output / "config.json")
    else:
        config = bsde_preset("bsde-smoke")
    budget = config.budget_seconds if args.budget_seconds is None else args.budget_seconds
    if budget <= 0:
        parser.error("Budget must be positive")
    directory = (args.output or Path("runs") / config.name / config.config_hash[:12]).resolve()
    print(f"[{config.name}] conditional BSDE statistics ({args.stage}): {directory}", flush=True)
    try:
        result = run_bsde(directory, config, time.monotonic() + budget, stage=args.stage)
        print(json.dumps(json_ready(result)), flush=True)
        return 0
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75


def _bsde_study_main(args, parser):
    from .bsde_study import run_bsde_study
    from .bsde_study_config import BSDEStudyConfig, bsde_study_preset

    if args.config:
        config = BSDEStudyConfig.load(args.config)
    elif args.preset:
        if args.preset not in ("bsde-study-smoke", "bsde-anchor-study"):
            parser.error("The bsde-study command requires a study preset")
        config = bsde_study_preset(args.preset)
    elif args.output and (args.output / "config.json").exists():
        config = BSDEStudyConfig.load(args.output / "config.json")
    else:
        config = bsde_study_preset("bsde-study-smoke")
    budget = config.base_config.budget_seconds if args.budget_seconds is None else args.budget_seconds
    if not 0 < budget < float("inf"):
        parser.error("Budget must be finite and positive")
    directory = (args.output or Path("runs") / config.name / config.config_hash[:12]).resolve()
    print(f"[{config.name}] coverage/structure study ({args.stage}): {directory}", flush=True)
    try:
        result = run_bsde_study(directory, config, time.monotonic()+budget, stage=args.stage)
        print(json.dumps(json_ready(result)), flush=True)
        return 0
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75


def _generator_main(args, parser):
    from .generator_config import GeneratorStudyConfig, generator_preset
    from .generator_study import run_generator

    if args.config:
        config = GeneratorStudyConfig.load(args.config)
    elif args.preset:
        if not args.preset.startswith("generator-"):
            parser.error("The generator command requires a generator-* preset")
        config = generator_preset(args.preset)
    elif args.output and (args.output / "config.json").exists():
        config = GeneratorStudyConfig.load(args.output / "config.json")
    else:
        config = generator_preset("generator-smoke")
    budget = config.budget_seconds if args.budget_seconds is None else args.budget_seconds
    if not 0 < budget < float("inf"):
        parser.error("Budget must be finite and positive")
    directory = (args.output or Path("runs") / config.name / config.config_hash[:12]).resolve()
    print(f"[{config.name}] continuous-generator moment accuracy ({args.stage}): {directory}", flush=True)
    try:
        result = run_generator(directory, config, time.monotonic()+budget, stage=args.stage)
        print(json.dumps(json_ready(result)), flush=True)
        return 0
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75


def _path_main(args, parser):
    from .path_config import PathConfig, path_preset
    from .path_study import run_path_study

    if args.config:
        config = PathConfig.load(args.config)
    elif args.preset:
        if not args.preset.startswith("path-"):
            parser.error("The path-solver command requires a path-* preset")
        config = path_preset(args.preset)
    elif args.output and (args.output/"config.json").exists():
        config = PathConfig.load(args.output/"config.json")
    else:
        config = path_preset("path-smoke")
    budget = config.budget_seconds if args.budget_seconds is None else args.budget_seconds
    if not 0 < budget < float("inf"):
        parser.error("Budget must be finite and positive")
    directory = (args.output or Path("runs")/config.name/config.config_hash[:12]).resolve()
    print(f"[{config.name}] full density-path Ito solver ({args.stage}): {directory}", flush=True)
    try:
        result = run_path_study(directory, config, time.monotonic()+budget, args.stage)
        print(json.dumps(json_ready(result)), flush=True)
        return 0
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
