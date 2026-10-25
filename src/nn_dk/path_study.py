"""A-only, reproducible density-path experiments with explicit physical drivers."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.linalg import expm

from .data import json_ready, source_hash, write_json
from .experiments import BudgetExhausted
from .path_config import PathConfig
from .types import stable_hash

WORKFLOW = "causal-ito-full-density-paths-v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Density-path work checkpointed; resume the identical configuration")


def _checkpoint(directory, case, dt, seed):
    return Path(directory)/"checkpoints"/f"{case}_{stable_hash(dt)[:10]}_seed{seed}.pt"


def _design(config):
    return {"workflow": WORKFLOW, "config_hash": config.config_hash,
            "targets": {f"{case}:{dt}": config.target_id(case, dt)
                        for case, dt in product(config.cases, config.step_sizes)},
            "output": "Every finite-volume mass, every integration time, every path",
            "training": "Known integrated left-point Ito residual; no future trajectory labels",
            "stochastic_driver": "All independent physical face Brownian increments; same driver on both residual sides",
            "architecture": "Fixed learned spectral Volterra kernel with causal finite-depth Picard blocks; known native mobility and interaction",
            "final_inputs": "Generated only after all configured models are complete and byte locked",
            "roles": {"train": config.train_paths, "validation": config.validation_paths,
                      "locked-final": config.test_paths},
            "optimizer_seeds_separate": list(config.optimizer_seeds),
            "step_specific_fit": True, "all_grid_modes_retained": True,
            "control_validation_gate": {"metric": "normalized full-state path supremum RMS",
                                        "tolerance": config.tolerance},
            "comparison_laws": {"same_step": "Specified Euler polygon target, no discretization reference bias",
                                "fine_reference": "Finite reference difference; continuous stopping bias uncertified"}}


def _artifacts(directory):
    directory = Path(directory)
    names = ("config.json", "preregistration.json", "calibration.json", "evaluations.json",
             "path_train.h5", "path_validation.h5", "path_test.h5")
    paths = {directory/name for name in names}
    for pattern in ("checkpoints/*", "evaluation/*.json", "predictions/*.h5", "development/*.json"):
        paths.update(directory.glob(pattern))
    if any(not p.is_file() for p in paths):
        raise ValueError("A completed density-path artifact is missing")
    return {str(p.relative_to(directory)): _sha(p) for p in sorted(paths)}


def prepare_path_run(directory, config: PathConfig):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"manifest.json"
    design = _design(config)
    if path.exists():
        manifest = _read(path)
        if (manifest.get("workflow") != WORKFLOW or manifest.get("source_hash") != source_hash()
                or manifest.get("config_hash") != config.config_hash
                or _read(directory/"config.json") != config.to_dict()
                or _read(directory/"preregistration.json") != design
                or manifest.get("preregistration_hash") != stable_hash(design)):
            raise ValueError("Density-path source/configuration/design changed; use a new run directory")
        if manifest.get("artifact_lock") is not None and manifest["artifact_lock"] != _artifacts(directory):
            raise ValueError("A completed density-path artifact changed after its final lock")
        return manifest
    manifest = {"workflow": WORKFLOW, "source_hash": source_hash(),
                "config_hash": config.config_hash, "preregistration_hash": stable_hash(design),
                "created_unix": time.time(), "status": "prepared", "elapsed_seconds": 0.,
                "python": platform.python_version(), "platform": platform.platform(),
                "device": "cpu", "dtype": "float64", "threads": 1,
                "expected_models": config.expected_models}
    write_json(directory/"config.json", config.to_dict())
    write_json(directory/"preregistration.json", design)
    write_json(path, manifest)
    return manifest


def calibrate_paths(directory, config):
    """Independent full-grid Gaussian matrix-exponential and driver checks."""
    from .observations import incidence_matrix
    from .path_numerics import linear_moments, path_coefficients, torch_euler

    path = Path(directory)/"calibration.json"
    identity = {"source_hash": source_hash(), "config_hash": config.config_hash}
    if path.exists():
        saved = _read(path)
        if saved.get("identity") != identity or saved.get("body_hash") != stable_hash(saved["result"]):
            raise ValueError("Density-path calibration provenance/content changed")
        if not saved["result"]["passed"]:
            raise ValueError("Density-path numerical calibration previously failed")
        return saved["result"]
    g, context = config.n_cells, config.context("additive")
    spec = config.spec
    initial = np.full(g, 1/g)
    result = linear_moments(initial, np.zeros((g, g)), config.query_times,
                            context, spec, "additive")
    a, q = result["drift_matrix"], result["noise_covariance"]
    error = 0.
    for index, t in enumerate(config.query_times):
        van_loan = expm(t*np.block([[a, q], [np.zeros_like(a), -a.T]]))
        covariance = van_loan[:g, g:]@expm(t*a.T)
        error = max(error, float(np.max(np.abs(covariance-result["covariance"][index]))))
    _, amplitudes = path_coefficients(torch.tensor(initial), context, spec, "additive")
    b = incidence_matrix(g)
    q_error = float(np.max(np.abs(b@np.diag(amplitudes.numpy()**2)@b.T/context.n_particles-q)))
    dw = np.zeros((2, 2, g))
    heat = torch_euler(np.tile(initial, (2, 1)), dw, context, spec,
                       min(config.step_sizes), "heat")["states"].numpy()
    mass_error = float(np.max(np.abs(heat.sum(-1)-1)))
    passed = bool(error < 1e-10 and q_error < 1e-12 and mass_error < 1e-12)
    packet = {"passed": passed, "full_covariance_matrix_exponential_error": error,
              "physical_face_covariance_error": q_error, "mass_error": mass_error,
              "noise": "Independent shared-face Brownian drivers; Ito left endpoint",
              "stopping": "Native first Euler-segment crossing tests cover uncensored proposals and freezing",
              "continuous_reference_bias_certified": False}
    write_json(path, {"identity": identity, "result": packet, "body_hash": stable_hash(packet)})
    if not passed:
        raise RuntimeError("Full-grid path calibration failed")
    return packet


def _inventory(directory, config):
    from .path_training import load_path_model

    models = []
    for case, dt, seed in product(config.cases, config.step_sizes, config.optimizer_seeds):
        path = _checkpoint(directory, case, dt, seed)
        metadata = path.with_suffix(".json")
        if not path.exists() or not metadata.exists() or not _read(metadata).get("complete"):
            raise ValueError("Every declared path fit must complete before final inputs unlock")
        model = load_path_model(path)
        if (model.target_id != config.target_id(case, dt) or model.case != case or model.dt != dt
                or model.provenance["seed"] != seed
                or model.provenance["config_hash"] != config.config_hash):
            raise ValueError("A path checkpoint has incompatible provenance or target")
        models.append({"case": case, "dt": dt, "seed": seed,
                       "checkpoint_path": str(path), "checkpoint_sha256": _sha(path),
                       "metadata_path": str(metadata), "metadata_sha256": _sha(metadata)})
    return {"expected_models": config.expected_models, "models": models}


def _development(directory, config, case, dt, seed, bank, model):
    from .path_numerics import torch_euler, torch_stop_paths

    path = Path(directory)/"development"/f"{case}_{stable_hash(dt)[:10]}_seed{seed}.json"
    identity = {"case": case, "dt": dt, "seed": seed, "source_hash": source_hash(),
                "config_hash": config.config_hash, "checkpoint_sha256": _sha(model.checkpoint_path),
                "validation_bank_sha256": _sha(Path(directory)/"path_validation.h5")}
    if path.exists():
        saved = _read(path)
        if saved.get("identity") != identity or saved.get("body_hash") != stable_hash(saved["record"]):
            raise ValueError("Path control development content or provenance changed")
        return saved["record"]
    with torch.no_grad():
        raw = model(bank["initial"], bank["brownian"], dt, config.context(case))
        prediction = torch_stop_paths(bank["initial"], raw[:, 1:], config.spec, dt,
                                      stopping=config.problem(case) == "dk")["states"].numpy()
        reference = torch_euler(bank["initial"], bank["brownian"], config.context(case),
                                config.spec, dt, config.problem(case))["states"].numpy()
    energy = config.n_particles*np.sum((prediction-reference)**2, axis=-1)/(1-1/config.n_cells)
    value = float(np.sqrt(energy.max(1).mean()))
    record = {"case": case, "dt": dt, "seed": seed, "role": "validation",
              "pathwise_supremum_rms": value, "within_tolerance": value <= config.tolerance,
              "interpretation": "Development control gate; no final test inputs used"}
    write_json(path, {"identity": identity, "record": record, "body_hash": stable_hash(record)})
    return record


def _evaluate(directory, config, lock, deadline):
    from .path_data import load_path_bank
    from .path_evaluation import (
        empirical_proposal_quadratic_variation,
        evaluate_paths,
        verify_causality,
        verify_same_noise_replay,
    )
    from .path_numerics import ito_residual, torch_euler
    from .path_training import load_path_model

    output = []
    for item in lock["models"]:
        _budget(deadline)
        case, dt, seed = item["case"], item["dt"], item["seed"]
        cache = Path(directory)/"evaluation"/f"{case}_{stable_hash(dt)[:10]}_seed{seed}.json"
        identity = {**item, "source_hash": source_hash(), "config_hash": config.config_hash,
                    "test_bank_sha256": _sha(Path(directory)/"path_test.h5")}
        if cache.exists():
            saved = _read(cache)
            if saved["identity"] != identity or saved["body_hash"] != stable_hash(saved["record"]):
                raise ValueError("Density-path evaluation content/identity changed")
            prediction_path = Path(directory)/"predictions"/f"{cache.stem}.h5"
            if (saved["record"].get("prediction_path") != str(prediction_path)
                    or not prediction_path.is_file()
                    or saved["record"].get("prediction_sha256") != _sha(prediction_path)):
                raise ValueError("A cached full density trajectory changed")
            output.append(saved["record"])
            continue
        tick = time.perf_counter()
        bank = load_path_bank(directory, config, "test", case, dt)
        model = load_path_model(item["checkpoint_path"])
        context, problem = config.context(case), config.problem(case)
        with torch.no_grad():
            raw = model(bank["initial"], bank["brownian"], dt, context)
            prediction = ito_residual(bank["initial"], raw[:, 1:], bank["brownian"],
                                      context, config.spec, dt, problem)
            exact = torch_euler(bank["initial"], bank["brownian"], context, config.spec, dt, problem)
        states = prediction["states"].numpy()
        times = np.arange(states.shape[1])*dt
        initial, increments = bank["initial"][:min(3, config.test_paths)], bank["brownian"][:min(3, config.test_paths)]

        def call(m0, dw):
            with torch.no_grad():
                return model(m0, dw, dt, context).numpy()

        causal = verify_causality(call, initial, increments, increments.shape[1]//2, seed=config.seed)
        replay = verify_same_noise_replay(call, initial, increments)
        common = {"family_ids": bank["family_ids"].tolist(),
                  "reference_family_ids": bank["family_ids"].tolist(),
                  "expected_paths": config.test_paths, "coupling_verified": True,
                  "observation_times": config.query_times, "n_bootstrap": config.n_bootstrap,
                  "seed": config.seed, "tolerance": config.tolerance,
                  "uncertainty_budget": config.uncertainty_budget,
                  "require_nonnegative": problem != "additive"}
        stops = prediction["stopped_at"].numpy() if problem == "dk" else None
        exact_stops = exact["stopped_at"].numpy() if problem == "dk" else None
        fine_stops = bank["reference_stopped_at"] if problem == "dk" else None
        same = evaluate_paths(states, exact["states"].numpy(), times, context, config.spec,
                              stopped_at=stops, reference_stopped_at=exact_stops,
                              reference_bias={"status": "resolved", "strong_normalized_bound": 0.,
                                              "weak_normalized_bound": 0.},
                              comparison_kind="neural-versus-same-step-euler", **common)
        difference = evaluate_paths(exact["states"].numpy(), bank["reference_states"], times,
                                    context, config.spec, stopped_at=exact_stops,
                                    reference_stopped_at=fine_stops,
                                    comparison_kind="euler-versus-fine-euler", **common)
        total = evaluate_paths(states, bank["reference_states"], times, context, config.spec,
                               stopped_at=stops, reference_stopped_at=fine_stops,
                               comparison_kind="neural-versus-fine-euler", **common)
        if problem == "heat":
            qv = {"status": "not-applicable-deterministic"}
        else:
            qv = empirical_proposal_quadratic_variation(
                states[:, :-1], raw[:, 1:].numpy(), times, context.n_particles,
                active_at_start=prediction["active"].numpy(),
                crossing_proposals=((raw[:, 1:].numpy() <= config.spec.mass_stop).any(-1)
                                    & prediction["active"].numpy()) if problem == "dk" else None,
                predictable_drift=prediction["predictable_drift"].numpy(),
                predictable_face_amplitudes=prediction["predictable_amplitudes"].numpy())
        key = cache.stem
        prediction_path = Path(directory)/"predictions"/f"{key}.h5"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(prediction_path.with_suffix(".h5.tmp"), "w") as saved:
            saved.attrs["identity"] = json.dumps(identity, sort_keys=True)
            saved.create_dataset("states", data=states, chunks=True, compression="gzip")
            saved.create_dataset("raw_proposals", data=raw[:, 1:].numpy(), chunks=True, compression="gzip")
            saved.create_dataset("stopped_at", data=prediction["stopped_at"].numpy())
            saved.create_dataset("times", data=times)
            saved.create_dataset("family_ids", data=np.asarray(bank["family_ids"], dtype="S64"))
        prediction_path.with_suffix(".h5.tmp").replace(prediction_path)
        record = {"case": case, "problem": problem, "dt": dt, "seed": seed,
                  "role": "locked-final", "target_id": config.target_id(case, dt),
                  "checkpoint_path": item["checkpoint_path"], "checkpoint_sha256": item["checkpoint_sha256"],
                  "training_seconds": _read(item["metadata_path"])["training_seconds"],
                  "evaluation_seconds": time.perf_counter()-tick,
                  "prediction_path": str(prediction_path), "prediction_sha256": _sha(prediction_path),
                  "prediction_diagnostics": {"mass_error": float(np.max(np.abs(states.sum(-1)-1))),
                                             "min_mass": float(np.min(states)),
                                             "causality": causal["passed"], "replay": replay["passed"],
                                             "causality_check": causal, "replay_check": replay},
                  "same_step": same, "reference_difference": difference, "total": total, "qv": qv}
        record = json_ready(record)
        write_json(cache, {"identity": identity, "record": record, "body_hash": stable_hash(record)})
        output.append(record)
    return output


def run_path_study(directory, config, deadline=None, stage="all"):
    from .path_data import ensure_path_bank, load_path_bank
    from .path_reporting import write_path_report
    from .path_training import fit_path, load_path_model

    if stage not in ("all", "calibrate", "generate", "fit", "evaluate", "report"):
        raise ValueError("Unsupported density-path stage")
    directory = Path(directory).resolve()
    manifest = prepare_path_run(directory, config)
    started = time.monotonic()
    torch.set_num_threads(1)
    try:
        if stage in ("all", "calibrate", "generate", "fit", "evaluate"):
            _budget(deadline)
            calibrate_paths(directory, config)
        if stage in ("all", "generate", "fit"):
            for role in ("train", "validation"):
                if not ensure_path_bank(directory, config, role, deadline)["complete"]:
                    raise BudgetExhausted("Density-path input generation checkpointed")
        if stage in ("all", "fit"):
            manifest["status"] = "fitting"
            write_json(directory/"manifest.json", manifest)
            ordered = [c for c in ("heat", "additive") if c in config.cases]
            ordered += [c for c in config.cases if c not in ordered]
            for case in ordered:
                checks = []
                for dt, seed in product(config.step_sizes, config.optimizer_seeds):
                    _budget(deadline)
                    print(f"[path] fit {case} dt={dt:g} seed={seed}", flush=True)
                    train = load_path_bank(directory, config, "train", case, dt)
                    validation = load_path_bank(directory, config, "validation", case, dt)
                    path = _checkpoint(directory, case, dt, seed)
                    metadata = fit_path(config, case, dt, seed, path, train, validation, deadline)
                    if not path.with_suffix(".json").exists() or _read(path.with_suffix(".json")) != metadata:
                        write_json(path.with_suffix(".json"), metadata)
                    if not metadata["complete"]:
                        raise BudgetExhausted("Density-path fit checkpointed")
                    if case in ("heat", "additive"):
                        checks.append(_development(directory, config, case, dt, seed,
                                                   validation, load_path_model(path)))
                if checks and not all(check["within_tolerance"] for check in checks):
                    manifest["status"] = "control-validation-unresolved"
                    manifest["unresolved_control"] = case
                    return {"status": manifest["status"], "final_inputs_unlocked": False}
        if stage in ("all", "evaluate"):
            lock = _inventory(directory, config)
            for case, dt, seed in product((c for c in ("heat", "additive") if c in config.cases),
                                          config.step_sizes, config.optimizer_seeds):
                _budget(deadline)
                validation = load_path_bank(directory, config, "validation", case, dt)
                model = load_path_model(_checkpoint(directory, case, dt, seed))
                if not _development(directory, config, case, dt, seed, validation, model)["within_tolerance"]:
                    manifest["status"] = "control-validation-unresolved"
                    manifest["unresolved_control"] = case
                    return {"status": manifest["status"], "final_inputs_unlocked": False}
            if manifest.get("fit_lock") is not None and manifest["fit_lock"] != lock:
                raise ValueError("A path model changed after the fit lock")
            manifest["fit_lock"] = lock
            manifest.setdefault("test_unlocked_unix", time.time())
            manifest["status"] = "evaluating"
            write_json(directory/"manifest.json", manifest)
            if not ensure_path_bank(directory, config, "test", deadline)["complete"]:
                raise BudgetExhausted("Density-path final reference generation checkpointed")
            records = _evaluate(directory, config, lock, deadline)
            write_json(directory/"evaluations.json", records)
            manifest["completed_evaluations"] = len(records)
            manifest["same_step_accuracy_passed"] = all(
                row["same_step"]["accuracy_gate"]["status"] == "resolved_within_tolerance"
                and row["prediction_diagnostics"]["causality"]
                and row["prediction_diagnostics"]["replay"] for row in records)
            manifest["continuous_time_accuracy_certified"] = False
            manifest["status"] = "complete"
            manifest["artifact_lock"] = _artifacts(directory)
        if stage in ("all", "evaluate", "report"):
            if manifest.get("fit_lock") != _inventory(directory, config):
                raise ValueError("Path report requires unchanged locked models")
            write_path_report(directory, config, _read(directory/"evaluations.json"), manifest)
        if stage in ("calibrate", "generate", "fit"):
            manifest["status"] = stage+"-complete"
        return {"status": manifest["status"], "output": str(directory),
                "same_step_accuracy_passed": manifest.get("same_step_accuracy_passed"),
                "continuous_time_accuracy_certified": False}
    except BudgetExhausted:
        manifest["status"] = "budget-exhausted"
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic()-started
        write_json(directory/"manifest.json", manifest)
        if manifest["status"] == "complete" and (directory/"evaluations.json").exists():
            write_path_report(directory, config, _read(directory/"evaluations.json"), manifest)
