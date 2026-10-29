"""Resumable Deep BSDE calibration, conditional moments and DK experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .bsde_data import generate_bsde_pool, load_bsde_pool
from .bsde_training import fit_bsde, load_bsde_model
from .data import source_hash, write_json
from .experiments import BudgetExhausted

WORKFLOW = "conditional-bsde-moments-v1"


def prepare_bsde_run(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("workflow") != WORKFLOW
                or manifest.get("source_hash") != source_hash()
                or manifest.get("config_hash") != config.config_hash):
            raise ValueError("BSDE run source/configuration mismatch; use a new output directory")
        if json.loads((directory / "config.json").read_text()) != config.to_dict():
            raise ValueError("Saved BSDE configuration differs from its manifest")
        return manifest
    manifest = {"workflow": WORKFLOW, "status": "prepared", "config": config.to_dict(),
                "config_hash": config.config_hash, "source_hash": source_hash(),
                "targets": {p: config.target_id(p) for p in ("ou", "dk")},
                "native_target": asdict(config.spec), "query_bank": asdict(config.queries),
                "covariance_scope": "Four-by-four same-time blocks only; no joint temporal covariance",
                "python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "created_unix": time.time(), "elapsed_seconds": 0.,
                "dependencies": {name: importlib.metadata.version(name) for name in (
                    "numpy", "scipy", "torch", "h5py", "matplotlib")},
                "scope": "Conditional moments of the stopped Euler chain at the recorded forward dt; "
                         "continuous-time DK accuracy requires a separate reference-bias argument"}
    write_json(directory / "config.json", config.to_dict())
    write_json(path, manifest)
    return manifest


def _check_budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("BSDE checkpointed; resume the same source, configuration and directory")


def _expected_keys(config):
    return {(problem, seed) for problem in ("ou", "dk") for seed in config.optimizer_seeds}


def _evaluation_keys(evaluations):
    keys = [(r["problem"], r["seed"]) for r in evaluations]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate BSDE evaluation records")
    return set(keys)


def _checkpoint(directory, problem, seed):
    return Path(directory) / "checkpoints" / problem / f"seed_{seed}.pt"


def report_bsde(directory, config):
    from .bsde_reporting import write_bsde_report

    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        raise ValueError("Incomplete BSDE study; prepare and execute its stages first")
    prepare_bsde_run(directory, config)
    calibration = json.loads((directory / "calibration.json").read_text())
    evaluations = json.loads((directory / "evaluations.json").read_text())
    if (not calibration.get("complete") or not calibration.get("passed")
            or _evaluation_keys(evaluations) != _expected_keys(config)
            or any(not record.get("complete", False) for record in evaluations)):
        raise ValueError("Incomplete BSDE study; resume before publishing a complete report")
    for problem, seed in _expected_keys(config):
        path = _checkpoint(directory, problem, seed)
        if not path.exists():
            raise ValueError("Incomplete BSDE study: missing coefficient-independent value checkpoint")
        model = load_bsde_model(path)
        if model.config.config_hash != config.config_hash or model.problem != problem:
            raise ValueError("BSDE report/checkpoint target mismatch")
        record = next(r for r in evaluations if (r["problem"], r["seed"]) == (problem, seed))
        if (record.get("config_hash") != config.config_hash
                or record.get("source_hash") != source_hash()
                or record.get("model", {}).get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("BSDE report evaluation/checkpoint provenance mismatch")
    return write_bsde_report(directory, config, calibration, evaluations)


def run_bsde(directory, config, deadline=None, stage="all"):
    from .bsde_calibration import calibrate_bsde
    from .bsde_evaluation import evaluate_bsde_model

    if stage not in ("all", "calibrate", "generate", "fit", "evaluate", "report"):
        raise ValueError("Unknown BSDE stage")
    directory = Path(directory)
    manifest = prepare_bsde_run(directory, config)
    torch.set_num_threads(config.training.get("threads", 1))
    started = time.monotonic()
    records_path = directory / "evaluations.json"
    evaluations = json.loads(records_path.read_text()) if records_path.exists() else []
    keys = _evaluation_keys(evaluations)
    if not keys <= _expected_keys(config):
        raise ValueError("Cached BSDE evaluations do not match the configured allocation")

    def mark(name):
        manifest["last_completed_stage"] = name
        manifest["status"] = name + "-complete"
        manifest.pop("error", None)
        manifest.pop("reason", None)
        write_json(directory / "manifest.json", manifest)

    try:
        _check_budget(deadline)
        if manifest.get("status") == "complete" and stage == "all":
            return report_bsde(directory, config)
        if stage == "report":
            result = report_bsde(directory, config)
            manifest["status"] = "complete"
            return result
        if stage in ("all", "calibrate"):
            print("[BSDE] numerical calibration", flush=True)
            calibration = calibrate_bsde(directory, config, deadline)
            if not calibration.get("complete"):
                raise BudgetExhausted("BSDE numerical calibration checkpointed")
            if not calibration.get("passed"):
                raise ValueError("BSDE numerical calibration failed; generation and fitting are blocked")
            mark("calibrate")
            if stage == "calibrate":
                return {"calibration": str(directory / "calibration.json")}
        else:
            calibration = json.loads((directory / "calibration.json").read_text())
            if not calibration.get("complete") or not calibration.get("passed"):
                raise ValueError("BSDE numerical calibration must pass before subsequent stages")
        for problem in ("ou", "dk"):
            _check_budget(deadline)
            if stage in ("all", "generate"):
                print(f"[BSDE] generate {problem} forward families", flush=True)
                generated = generate_bsde_pool(directory, config, problem, deadline)
                generation = manifest.setdefault("forward_pool_generation_seconds", {})
                generation[problem] = generation.get(problem, 0.) + generated.get("seconds_this_call", 0.)
                if not generated.get("complete"):
                    raise BudgetExhausted("BSDE forward generation checkpointed")
                mark(f"generate-{problem}")
            if stage == "generate":
                continue
            training = validation = None
            if stage in ("all", "fit"):
                training = load_bsde_pool(directory, config, problem, "train")
                validation = load_bsde_pool(directory, config, problem, "validation")
                if set(training.anchor_ids) & set(validation.anchor_ids):
                    raise ValueError("Training and validation BSDE initial anchors overlap")
            for seed in config.optimizer_seeds:
                _check_budget(deadline)
                path = _checkpoint(directory, problem, seed)
                if stage in ("all", "fit"):
                    print(f"[BSDE] fit {problem}, optimizer seed {seed}", flush=True)
                    metadata = fit_bsde(training, validation, config, problem, seed,
                                        path, deadline=deadline)
                    metadata["shared_forward_pool_generation_seconds"] = manifest.get(
                        "forward_pool_generation_seconds", {}).get(problem)
                    metadata["generation_cost_scope"] = "Shared train/validation pool including initialization and HDF5 writes; do not sum across optimizer seeds"
                    write_json(path.with_suffix(".json"), metadata)
                    if not metadata.get("complete"):
                        raise BudgetExhausted("BSDE training checkpointed at an epoch boundary")
                    mark(f"fit-{problem}-{seed}")
                if stage in ("all", "evaluate") and (problem, seed) not in keys:
                    print(f"[BSDE] evaluate {problem}, optimizer seed {seed}", flush=True)
                    load_started = time.perf_counter()
                    model = load_bsde_model(path)
                    model.checkpoint_loading_seconds = time.perf_counter() - load_started
                    evaluation = evaluate_bsde_model(directory, config, model, seed, deadline)
                    if not evaluation.get("complete", False):
                        raise BudgetExhausted("BSDE evaluation checkpointed")
                    if (evaluation.get("problem"), evaluation.get("seed")) != (problem, seed):
                        raise ValueError("BSDE evaluator returned the wrong model identity")
                    evaluations.append(evaluation)
                    keys.add((problem, seed))
                    write_json(records_path, evaluations)
                    mark(f"evaluate-{problem}-{seed}")
            del training, validation
        if stage == "all":
            _check_budget(deadline)
            result = report_bsde(directory, config)
            manifest["status"] = "complete"
            manifest["last_completed_stage"] = "report"
            manifest.pop("error", None)
            manifest.pop("reason", None)
            return result
        mark(stage)
        return {"output": str(directory), "stage": stage}
    except BudgetExhausted as error:
        manifest["status"] = "budget-exhausted"
        manifest["reason"] = str(error)
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "complete":
            exported = directory / "report" / "bsde_results.json"
            if exported.exists():
                payload = json.loads(exported.read_text())
                payload["manifest"] = manifest
                write_json(exported, payload)
