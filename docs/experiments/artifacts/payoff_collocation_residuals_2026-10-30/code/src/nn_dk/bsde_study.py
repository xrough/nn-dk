"""Preregistered, resumable coverage-by-value-architecture BSDE comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch

from .bsde_study_data import generate_study_pool, load_study_pool, study_pool_identity
from .bsde_study_evaluation import StudyModelDescriptor, evaluate_study_model, test_bank_identity
from .bsde_training import fit_bsde, load_bsde_model
from .data import source_hash, write_json
from .experiments import BudgetExhausted
from .types import stable_hash

WORKFLOW = "conditional-bsde-anchor-study-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _design(config):
    base = config.base_config
    return {"workflow": WORKFLOW, "study_config_hash": config.config_hash,
            "source_hash": source_hash(), "config": config.to_dict(),
            "training_path_budget": config.training_paths,
            "validation": list(config.validation), "structured_residual_scale": 0.1,
            "comparison": "initial-state coverage versus continuations at a fixed path budget; value architecture factorial",
            "model_selection": "validation early stopping only; every fit is locked before final test evaluation",
            "test_bank_ids": {p: test_bank_identity(base, p) for p in ("ou", "dk")},
            "test_seed": base.seed, "previous_observed_test_seed": 20261024,
            "independent_test": base.seed != 20261024,
            "integrand_control": "Same architecture and initial parameter draws for matched optimizer seeds",
            "primary_metrics": "Same-step conditional means and variances; ensemble and fine-reference errors separate",
            "contrasts": "Structured-minus-MLP at each allocation; largest-minus-smallest anchor allocation per architecture; seeds separate"}


def prepare_study_run(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    design = _design(config)
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("workflow") != WORKFLOW
                or manifest.get("source_hash") != source_hash()
                or manifest.get("config_hash") != config.config_hash):
            raise ValueError("Study source/configuration changed; use a new output directory")
        if json.loads((directory / "config.json").read_text()) != config.to_dict():
            raise ValueError("Study saved configuration mismatch")
        preregistration = json.loads((directory / "preregistration.json").read_text())
        if preregistration != design or stable_hash(preregistration) != manifest["preregistration_hash"]:
            raise ValueError("Study preregistration changed after preparation")
        return manifest
    manifest = {"workflow": WORKFLOW, "status": "prepared", "config_hash": config.config_hash,
                "source_hash": source_hash(), "config": config.to_dict(),
                "preregistration_hash": stable_hash(design), "created_unix": time.time(),
                "elapsed_seconds": 0., "expected_models": config.expected_models,
                "device": "cpu", "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "scipy", "torch", "h5py", "matplotlib")}}
    write_json(directory / "config.json", config.to_dict())
    write_json(directory / "base_config.json", config.base_config.to_dict())
    write_json(directory / "preregistration.json", design)
    write_json(path, manifest)
    return manifest


def _models(config):
    for problem in ("ou", "dk"):
        for anchors, children in config.allocations:
            for architecture in config.architectures:
                for seed in config.base_config.optimizer_seeds:
                    yield problem, architecture, anchors, children, seed


def _checkpoint(directory, key):
    problem, architecture, anchors, children, seed = key
    return Path(directory) / "checkpoints" / problem / architecture / f"a{anchors}_c{children}" / f"seed_{seed}.pt"


def _key(record):
    factors = record["factors"]
    return (record["evaluation"]["problem"], factors["architecture"], factors["train_anchors"],
            factors["train_continuations"], factors["optimizer_seed"])


def _inventory(directory, config):
    """Verify all fits before revealing or reporting any held-out outcomes."""
    inventory = []
    validation_fingerprints, training_fingerprints = {}, {}
    for key in _models(config):
        path = _checkpoint(directory, key)
        if not path.exists() or not path.with_suffix(".json").exists():
            raise ValueError("Every configured fit must complete before final test evaluation")
        metadata = json.loads(path.with_suffix(".json").read_text())
        model = load_bsde_model(path)
        problem, architecture, anchors, children, seed = key
        if (not metadata.get("complete") or model.problem != problem
                or getattr(model, "architecture", "mlp") != architecture
                or model.config.config_hash != config.base_config.config_hash
                or model.provenance["seed"] != seed
                or metadata.get("training_anchors") != anchors
                or metadata.get("training_paths") != anchors*children
                or metadata.get("validation_anchors") != config.validation[0]
                or metadata.get("validation_paths") != config.validation[0]*config.validation[1]):
            raise ValueError("Study checkpoint metadata does not match its declared factors")
        training_fp = model.provenance["training_fingerprint"]
        validation_fp = model.provenance["validation_fingerprint"]
        view_key = (problem, anchors, children)
        if (view_key in training_fingerprints and training_fingerprints[view_key] != training_fp
                or problem in validation_fingerprints and validation_fingerprints[problem] != validation_fp):
            raise ValueError("Study architectures do not share identical training/validation traces")
        training_fingerprints[view_key] = training_fp
        validation_fingerprints[problem] = validation_fp
        inventory.append({"key": list(key), "checkpoint_sha256": _sha(path),
                          "training_fingerprint": training_fp,
                          "validation_fingerprint": validation_fp})
    return inventory


def _verify_records(directory, config, records):
    expected = set(_models(config))
    keys = [_key(record) for record in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Duplicate or undeclared study model records")
    for record in records:
        key = _key(record)
        if (record.get("study_config_hash") != config.config_hash
                or record.get("checkpoint_sha256") != _sha(_checkpoint(directory, key))
                or not record["evaluation"].get("complete")):
            raise ValueError("Study evaluation/checkpoint provenance mismatch")
    return set(keys)


def report_study(directory, config):
    from .bsde_study_reporting import write_study_report

    directory = Path(directory)
    manifest = prepare_study_run(directory, config)
    if not manifest.get("fit_lock") or _inventory(directory, config) != manifest["fit_lock"]:
        raise ValueError("Incomplete or changed study fits; resume before reporting")
    path = directory / "evaluations.json"
    records = json.loads(path.read_text()) if path.exists() else []
    if _verify_records(directory, config, records) != set(_models(config)):
        raise ValueError("Incomplete study evaluations; resume before reporting")
    contrast_path = directory / "contrasts.json"
    if not contrast_path.exists():
        raise ValueError("Incomplete study contrasts; resume final evaluation")
    calibration = json.loads((directory / "common" / "calibration.json").read_text())
    return write_study_report(directory, config, calibration, records,
                              json.loads(contrast_path.read_text()))


def run_bsde_study(directory, config, deadline=None, stage="all"):
    from .bsde_calibration import calibrate_bsde
    from .bsde_structured import StructuredBSDEMomentModel
    from .bsde_study_evaluation import paired_study_contrasts

    if stage not in ("all", "calibrate", "generate", "fit", "evaluate", "report"):
        raise ValueError("Unknown study stage")
    directory = Path(directory)
    manifest = prepare_study_run(directory, config)
    base = config.base_config
    torch.set_num_threads(base.training["threads"])
    started = time.monotonic()
    common, pools = directory / "common", directory / "pools"
    common.mkdir(exist_ok=True)
    records_path = directory / "evaluations.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []

    def budget():
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("Study checkpointed; resume with the same source and configuration")

    def mark(value):
        manifest["status"] = value
        manifest.pop("error", None)
        manifest.pop("reason", None)
        write_json(directory / "manifest.json", manifest)

    options = {"allocations": config.allocations, "validation": config.validation,
               "seed": base.seed}
    try:
        budget()
        if stage == "report" or manifest.get("status") == "complete" and stage == "all":
            result = report_study(directory, config)
            mark("complete")
            return result
        if stage in ("all", "calibrate"):
            print("[study] numerical calibration", flush=True)
            calibration = calibrate_bsde(common, base, deadline)
            if not calibration.get("complete"):
                raise BudgetExhausted("Study calibration checkpointed")
            if not calibration.get("passed"):
                raise ValueError("Study numerical calibration failed")
            mark("calibrate-complete")
            if stage == "calibrate":
                return {"calibration": str(common / "calibration.json")}
        else:
            calibration_path = common / "calibration.json"
            if not calibration_path.exists():
                raise ValueError("Numerical calibration must pass before later study stages")
            calibration = json.loads(calibration_path.read_text())
            if not calibration.get("complete") or not calibration.get("passed"):
                raise ValueError("Numerical calibration must pass before later study stages")
        if stage in ("all", "generate") and not manifest.get("fit_lock"):
            for problem in ("ou", "dk"):
                budget()
                print(f"[study] generate nested {problem} families", flush=True)
                generated = generate_study_pool(pools, base, problem, deadline=deadline, **options)
                costs = manifest.setdefault("shared_pool_generation_seconds", {})
                costs[problem] = costs.get(problem, 0.) + generated.get("seconds_this_call", 0.)
                manifest.setdefault("data_pools", {})[problem] = study_pool_identity(base, problem, **options)
                if not generated.get("complete"):
                    raise BudgetExhausted("Study data generation checkpointed")
                mark(f"generate-{problem}-complete")
            if stage == "generate":
                return {"output": str(directory), "stage": stage}
        if stage in ("all", "fit") and not manifest.get("fit_lock"):
            for problem in ("ou", "dk"):
                validation = load_study_pool(pools, base, problem, "validation", **options)
                for anchors, children in config.allocations:
                    training = load_study_pool(pools, base, problem, "train",
                                               allocation=(anchors, children), **options)
                    for architecture in config.architectures:
                        for seed in base.optimizer_seeds:
                            budget()
                            key = (problem, architecture, anchors, children, seed)
                            path = _checkpoint(directory, key)
                            print(f"[study] fit {problem} {architecture} {anchors}x{children} seed {seed}", flush=True)
                            factory = StructuredBSDEMomentModel if architecture == "structured" else None
                            metadata = fit_bsde(training, validation, base, problem, seed, path,
                                                deadline=deadline, model_factory=factory)
                            metadata.update(architecture=architecture, allocation=[anchors, children],
                                            study_config_hash=config.config_hash,
                                            training_view_fingerprint=training.fingerprint,
                                            validation_fingerprint=validation.fingerprint,
                                            shared_pool_generation_seconds=manifest.get(
                                                "shared_pool_generation_seconds", {}).get(problem))
                            write_json(path.with_suffix(".json"), metadata)
                            if not metadata.get("complete"):
                                raise BudgetExhausted("Study training checkpointed at an epoch boundary")
                            manifest["last_fit"] = list(key)
                            mark("fitting")
                    del training
                del validation
            mark("fit-complete")
            if stage == "fit":
                return {"output": str(directory), "stage": stage}
        if stage in ("all", "evaluate"):
            inventory = _inventory(directory, config)
            if manifest.get("fit_lock") and manifest["fit_lock"] != inventory:
                raise ValueError("Study fits changed after the final test bank was unlocked")
            if not manifest.get("fit_lock"):
                manifest["fit_lock"] = inventory
                manifest["test_unlocked_unix"] = time.time()
                mark("fits-locked")
            keys = _verify_records(directory, config, records)
            for key in _models(config):
                if key in keys:
                    continue
                budget()
                problem, architecture, anchors, children, seed = key
                path = _checkpoint(directory, key)
                tick = time.perf_counter()
                model = load_bsde_model(path)
                model.checkpoint_loading_seconds = time.perf_counter() - tick
                model.study_allocation, model.optimizer_seed = (anchors, children), seed
                print(f"[study] evaluate {problem} {architecture} {anchors}x{children} seed {seed}", flush=True)
                descriptor = StudyModelDescriptor(architecture, anchors, children, seed)
                record = evaluate_study_model(common, base, model, descriptor, deadline=deadline)
                record.update(checkpoint_path=str(path.resolve()), study_config_hash=config.config_hash)
                records.append(record)
                keys.add(key)
                write_json(records_path, records)
                manifest["completed_evaluations"] = len(records)
                mark("evaluating")
            budget()
            print("[study] paired error contrasts", flush=True)
            contrasts = paired_study_contrasts(records, common, base, deadline=deadline)
            write_json(directory / "contrasts.json", contrasts)
            if stage == "evaluate":
                mark("evaluate-complete")
                return {"output": str(directory), "stage": stage}
        budget()
        result = report_study(directory, config)
        mark("complete")
        return result
    except BudgetExhausted as error:
        manifest["status"], manifest["reason"] = "budget-exhausted", str(error)
        raise
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic()-started
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "complete":
            exported = directory / "report" / "study_results.json"
            if exported.exists():
                payload = json.loads(exported.read_text())
                payload["manifest"] = manifest
                write_json(exported, payload)
