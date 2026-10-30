"""Locked, reproducible accuracy experiments for neural generator moment maps.

Known physical coefficients enter the training residual.  The independent
reference integrates the full native moment operator and bounds first-exit
effects; neither a noisy empirical reference nor an unstopped-law substitution
can make this workflow's accuracy gate pass.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .data import json_ready, source_hash, write_json
from .experiments import BudgetExhausted
from .generator_config import GeneratorStudyConfig
from .native import initialize_masses
from .types import seeded_rng, stable_hash

WORKFLOW = "continuous-generator-neural-accuracy-v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _body_hash(value):
    return stable_hash(json_ready(value))


def _array_hashes(bank):
    hashes = {}
    for name, value in bank.items():
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(stable_hash([value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value))
        hashes[name] = digest.hexdigest()
    return hashes


def _artifact_inventory(directory):
    directory = Path(directory)
    required = ("config.json", "geometry_adapter.json", "preregistration.json", "calibration.json",
                "development_results.json", "evaluations.json", "inputs/development.npz",
                "inputs/locked-final.npz")
    paths = {directory / name for name in required}
    paths |= set((directory / "checkpoints").glob("*.pt"))
    paths |= set((directory / "checkpoints").glob("*.json"))
    paths |= set((directory / "reference").rglob("*.json"))
    paths |= set((directory / "evaluation").rglob("*.json"))
    if any(not path.is_file() for path in paths):
        raise ValueError("Completed generator artifact inventory is missing a required file")
    return {str(path.relative_to(directory)): _sha(path) for path in sorted(paths)}


def _validate_artifact_lock(directory, manifest):
    if manifest.get("artifact_lock") != _artifact_inventory(directory):
        raise ValueError("A completed generator artifact changed after its final content lock")


def _budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Generator accuracy work checkpointed; resume the identical configuration")


def _design(config):
    return {"workflow": WORKFLOW, "config_hash": config.config_hash,
            "target_ids": {p: config.target_id(p) for p in ("ou", "dk")},
            "law": "continuous stopped finite-volume DK; continuous four-mode OU control",
            "profiles": list(config.cases), "optimizer_seeds": list(config.optimizer_seeds),
            "conditional_anchors_per_profile": config.conditional_anchors,
            "ou_initial_states": config.ou_test_anchors,
            "horizons": list(config.query_times), "covariance": "full same-time 4x4",
            "initial_ensemble": "exact known IID multinomial law, with profile-integration enclosure",
            "training": "deterministic known-generator residual; learned time coefficients",
            "reference": "independent native moment operator plus validated continuous first-exit bound",
            "accuracy": {"maximum_pointwise_error": config.tolerance,
                         "each_uncertainty_budget": config.uncertainty_budget,
                         "seeds_pooled": False, "rmse_is_not_acceptance": True},
            "final_input_namespace": [config.seed, config.config_hash, "locked-final"],
            "previous_inspected_banks_are_development_only": [20261024, 20261025],
            "all_models_locked_before_final_inputs": True}


def prepare_generator_run(directory, config):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    design = _design(config)
    if path.exists():
        manifest = _read(path)
        if (manifest.get("workflow") != WORKFLOW
                or manifest.get("source_hash") != source_hash()
                or manifest.get("config_hash") != config.config_hash
                or _read(directory / "config.json") != config.to_dict()
                or _read(directory / "preregistration.json") != design
                or manifest.get("preregistration_hash") != stable_hash(design)):
            raise ValueError("Generator source/configuration/design changed; preserve this run and use a new directory")
        if manifest.get("artifact_lock") is not None:
            _validate_artifact_lock(directory, manifest)
        return manifest
    manifest = {"workflow": WORKFLOW, "source_hash": source_hash(),
                "config_hash": config.config_hash, "config": config.to_dict(),
                "preregistration_hash": stable_hash(design), "created_unix": time.time(),
                "python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "dtype": "float64", "status": "prepared",
                "expected_models": config.expected_models, "elapsed_seconds": 0.}
    write_json(directory / "config.json", config.to_dict())
    write_json(directory / "geometry_adapter.json", config.base_config.to_dict())
    write_json(directory / "preregistration.json", design)
    write_json(path, manifest)
    return manifest


def _input_bank(directory, config, role, *, unlocked=False):
    if role not in ("development", "locked-final") or role == "locked-final" and not unlocked:
        raise ValueError("Final initial states require the complete immutable fit lock")
    path = Path(directory) / "inputs" / f"{role}.npz"
    identity = {"config_hash": config.config_hash, "source_hash": source_hash(),
                "role": role, "seed": config.seed}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("Generator initial-state bank identity changed")
            bank = {key: saved[key] for key in ("ou", *config.cases)}
            if ("array_hashes" not in saved
                    or json.loads(str(saved["array_hashes"])) != _array_hashes(bank)):
                raise ValueError("Generator initial-state bank array content changed")
            return bank
    count = config.validation_anchors if role == "development" else config.conditional_anchors
    ou_count = config.validation_anchors if role == "development" else config.ou_test_anchors
    bank = {"ou": seeded_rng(config.seed, config.config_hash, role, "ou").normal(size=(ou_count, 4))}
    for case in config.cases:
        seed = int(seeded_rng(config.seed, config.config_hash, role, case).integers(0, 2**63-1))
        bank[case] = initialize_masses(config.context(case), config.spec, count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **bank, identity=np.asarray(json.dumps(identity)),
                            array_hashes=np.asarray(json.dumps(_array_hashes(bank), sort_keys=True)))
    temporary.replace(path)
    return bank


def _ou_reference(base, initial, horizons):
    rates = np.asarray(base.mode_rates)
    times = np.asarray(horizons)
    mean = np.asarray(initial)[:, None, :] * np.exp(-times[:, None] * rates)
    diagonal = -np.expm1(-2 * times[:, None] * rates)
    covariance = np.broadcast_to(np.asarray([np.diag(v) for v in diagonal]),
                                 (*mean.shape, 4)).copy()
    return {"mean": mean, "scaled_covariance": covariance,
            "mean_error_bound": 1e-12 * (1 + np.abs(mean)),
            "scaled_covariance_error_bound": np.full_like(covariance, 1e-12),
            "certificate": {"valid": True, "method": "analytic continuous OU moments",
                            "numerical_allowance": 1e-12, "stopping": "not applicable"}}


def _reference(directory, config, role, case, scope, initial=None, deadline=None):
    from .generator_reference import reference_conditional, reference_ensemble

    path = Path(directory) / "reference" / role / f"{case}_{scope}.json"
    identity = {"source_hash": source_hash(), "config_hash": config.config_hash,
                "target_id": config.target_id("dk"), "role": role, "case": case, "scope": scope,
                "initial_hash": stable_hash(np.asarray(initial).tolist()) if initial is not None else None,
                "input_bank_sha256": _sha(Path(directory) / "inputs" / f"{role}.npz")
                if role in ("development", "locked-final") else None}
    if path.exists():
        saved = _read(path)
        if saved["identity"] != identity:
            raise ValueError("Generator reference identity differs from its saved input")
        if saved.get("reference_sha256") != _body_hash(saved["reference"]):
            raise ValueError("Generator reference cached body content changed")
        return saved["reference"]
    _budget(deadline)
    tick = time.perf_counter()
    if scope == "conditional":
        result = reference_conditional(config.base_config, initial, config.query_times,
                                       settings=config.reference_settings)
    else:
        result = reference_ensemble(config.base_config, config.context(case), config.query_times,
                                    settings=config.reference_settings)
    result["reference_seconds"] = time.perf_counter() - tick
    write_json(path, {"identity": identity, "reference": result,
                      "reference_sha256": _body_hash(result)})
    return result


def calibrate_generator(directory, config, deadline=None):
    """Independent uniform-density Gaussian fluctuation identity for the reference."""
    path = Path(directory) / "calibration.json"
    if path.exists():
        saved = _read(path)
        if (saved.get("source_hash") != source_hash() or saved.get("config_hash") != config.config_hash
                or saved.get("passed") is not True):
            raise ValueError("Generator calibration is failed or belongs to a different source/configuration")
        body = {key: value for key, value in saved.items() if key != "content_sha256"}
        if saved.get("content_sha256") != _body_hash(body):
            raise ValueError("Generator calibration cached body content changed")
        return saved
    initial = np.full((1, config.n_cells), 1 / config.n_cells)
    reference = _reference(directory, config, "calibration", "uniform", "conditional", initial, deadline)
    scale = config.base_config.observable_scale
    exact = _ou_reference(config.base_config, np.zeros((1, 4)), config.query_times)
    expected = exact["scaled_covariance"] * np.outer(scale, scale)
    mean_error = float(np.max(np.abs(reference["mean"])))
    covariance_error = float(np.max(np.abs(reference["scaled_covariance"] - expected)))
    passed = bool(reference["certificate"].get("valid")
                  and mean_error < 1e-8 and covariance_error < 1e-8)
    result = {"passed": passed, "source_hash": source_hash(), "config_hash": config.config_hash,
              "mean_error": mean_error, "covariance_error": covariance_error,
              "interpretation": "Constant expected density implies exact modewise OU covariance before the separately bounded first exit"}
    result["content_sha256"] = _body_hash(result)
    write_json(path, result)
    if not passed:
        raise RuntimeError("Independent generator reference calibration failed")
    return result


def _inventory(directory, config):
    result = []
    for problem in ("ou", "dk"):
        for seed in config.optimizer_seeds:
            path = Path(directory) / "checkpoints" / f"{problem}_seed{seed}.pt"
            metadata_path = path.with_suffix(".json")
            if not path.exists() or not metadata_path.exists():
                raise ValueError("Every generator fit must finish before final evaluation")
            metadata = _read(metadata_path)
            if not metadata.get("complete"):
                raise ValueError("Cannot unlock final evaluation with an incomplete generator fit")
            result.append({"problem": problem, "seed": seed, "checkpoint_sha256": _sha(path),
                           "metadata_sha256": _sha(metadata_path)})
    return result


def _evaluate(directory, config, role, inventory, deadline=None):
    from .generator_evaluation import compare_certified_moments
    from .generator_reference import profile_probabilities, reference_ou
    from .generator_training import load_generator_model

    if role == "locked-final":
        manifest = _read(Path(directory) / "manifest.json")
        if manifest.get("fit_lock") != inventory or not manifest.get("test_unlocked_unix"):
            raise ValueError("Final generator inputs require an authoritative fit lock")
        if _inventory(directory, config) != inventory:
            raise ValueError("A generator fit changed after the final test bank unlocked")
    bank = _input_bank(directory, config, role, unlocked=role == "locked-final")
    input_bank_hash = _sha(Path(directory) / "inputs" / f"{role}.npz")
    output = []
    for item in inventory:
        problem, seed = item["problem"], item["seed"]
        checkpoint = Path(directory) / "checkpoints" / f"{problem}_seed{seed}.pt"
        _budget(deadline)
        model = load_generator_model(checkpoint)
        if model.target_id != config.target_id(problem):
            raise ValueError("Loaded generator model has a different continuous target")
        cases = ("analytic",) if problem == "ou" else config.cases
        for case in cases:
            scopes = ("conditional",) if problem == "ou" else ("conditional", "ensemble")
            for scope in scopes:
                path = Path(directory) / "evaluation" / role / f"{problem}_{case}_{scope}_seed{seed}.json"
                identity = {**item, "config_hash": config.config_hash, "source_hash": source_hash(),
                            "role": role, "problem": problem, "case": case, "scope": scope,
                            "input_bank_sha256": input_bank_hash}
                if path.exists():
                    saved = _read(path)
                    if saved["identity"] != identity:
                        raise ValueError("Saved generator evaluation identity changed")
                    record = saved["record"]
                    if saved.get("record_sha256") != _body_hash(record):
                        raise ValueError("Saved generator evaluation body content changed")
                    expected_record = {"problem": problem, "seed": seed, "case": case, "scope": scope,
                                       "role": role, "target_id": config.target_id(problem),
                                       "checkpoint_sha256": item["checkpoint_sha256"],
                                       "checkpoint_path": str(checkpoint)}
                    if any(record.get(key) != value for key, value in expected_record.items()):
                        raise ValueError("Saved generator evaluation body differs from its cache identity")
                    output.append(record)
                    continue
                _budget(deadline)
                tick = time.perf_counter()
                if problem == "ou":
                    prediction = model.predict_conditional(bank["ou"], config.query_times)
                    reference = reference_ou(config.base_config, bank["ou"], config.query_times)
                elif scope == "conditional":
                    prediction = model.predict_conditional(bank[case], config.query_times)
                    reference = _reference(directory, config, role, case, scope, bank[case], deadline)
                else:
                    probabilities, _ = profile_probabilities(config.base_config, config.context(case))
                    prediction = model.predict_ensemble(probabilities, config.query_times)
                    reference = _reference(directory, config, role, case, scope, deadline=deadline)
                comparison = compare_certified_moments(
                    prediction, reference, n_particles=1 if problem == "ou" else config.n_particles,
                    std=np.ones(4) if problem == "ou" else config.base_config.observable_scale,
                    tolerance=config.tolerance, uncertainty_budget=config.uncertainty_budget)
                record = {"problem": problem, "seed": seed, "case": case, "scope": scope,
                          "role": role, "target_id": config.target_id(problem),
                          "checkpoint_path": str(checkpoint), "checkpoint_sha256": item["checkpoint_sha256"],
                          "comparison": comparison, "prediction_diagnostics": prediction.diagnostics,
                          "reference_certificate": reference["certificate"],
                          "evaluation_seconds": time.perf_counter() - tick,
                          "training_seconds": _read(checkpoint.with_suffix(".json")).get("training_seconds")}
                write_json(path, {"identity": identity, "record": record,
                                  "record_sha256": _body_hash(record)})
                output.append(record)
    return output


def run_generator(directory, config, deadline=None, stage="all"):
    from .generator_reporting import write_generator_report
    from .generator_training import fit_generator

    if stage not in ("all", "calibrate", "generate", "fit", "evaluate", "report"):
        raise ValueError("Unsupported generator stage")
    directory = Path(directory).resolve()
    manifest = prepare_generator_run(directory, config)
    started = time.monotonic()
    try:
        if stage in ("all", "calibrate", "fit", "evaluate"):
            manifest["status"] = "calibrating"
            write_json(directory / "manifest.json", manifest)
            calibrate_generator(directory, config, deadline)
        if stage in ("all", "generate", "fit"):
            _input_bank(directory, config, "development")
        if stage in ("all", "fit"):
            manifest["status"] = "fitting"
            write_json(directory / "manifest.json", manifest)
            for problem in ("ou", "dk"):
                for seed in config.optimizer_seeds:
                    _budget(deadline)
                    print(f"[generator] fit {problem} seed {seed}", flush=True)
                    path = directory / "checkpoints" / f"{problem}_seed{seed}.pt"
                    metadata = fit_generator(config.base_config, problem, seed, path,
                                             config.training_settings, deadline)
                    metadata_path = path.with_suffix(".json")
                    if not metadata_path.exists() or _read(metadata_path) != metadata:
                        write_json(metadata_path, metadata)
                    if not metadata["complete"]:
                        raise BudgetExhausted("Generator fit checkpointed")
        if stage in ("all", "fit", "evaluate"):
            inventory = _inventory(directory, config)
            development = _evaluate(directory, config, "development", inventory, deadline)
            write_json(directory / "development_results.json", development)
            passed = all(row["comparison"]["accuracy_gate"]["status"] == "resolved_within_tolerance"
                         for row in development)
            manifest["development_accuracy_passed"] = passed
            if not passed:
                manifest["status"] = "development-not-accurate"
                return {"status": manifest["status"], "final_bank_unlocked": False,
                        "development_results": str(directory / "development_results.json")}
            if stage in ("all", "evaluate"):
                if manifest.get("fit_lock") and manifest["fit_lock"] != inventory:
                    raise ValueError("A generator checkpoint changed after the final fit lock")
                manifest["fit_lock"] = inventory
                manifest.setdefault("test_unlocked_unix", time.time())
                manifest["status"] = "evaluating"
                write_json(directory / "manifest.json", manifest)
                records = _evaluate(directory, config, "locked-final", inventory, deadline)
                write_json(directory / "evaluations.json", records)
                manifest["accuracy_threshold_met"] = all(
                    row["comparison"]["accuracy_gate"]["status"] == "resolved_within_tolerance" for row in records)
                manifest["completed_evaluations"] = len(records)
                manifest["status"] = "complete"
                manifest["artifact_lock"] = _artifact_inventory(directory)
                write_generator_report(directory, config, records, manifest)
        if stage == "report":
            records = _read(directory / "evaluations.json")
            if manifest.get("fit_lock") != _inventory(directory, config):
                raise ValueError("Generator report requires unchanged locked models")
            write_generator_report(directory, config, records, manifest)
        if stage in ("calibrate", "generate", "fit"):
            manifest["status"] = stage + "-complete"
        return {"status": manifest["status"], "accuracy_threshold_met": manifest.get("accuracy_threshold_met", False),
                "output": str(directory)}
    except BudgetExhausted:
        manifest["status"] = "budget-exhausted"
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
        if manifest["status"] == "complete" and (directory / "evaluations.json").exists():
            write_generator_report(directory, config, _read(directory / "evaluations.json"), manifest)


def report_generator(directory):
    directory = Path(directory).resolve()
    return run_generator(directory, GeneratorStudyConfig.load(directory / "config.json"), stage="report")
