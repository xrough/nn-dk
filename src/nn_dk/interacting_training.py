"""Resumable full-state Itô-generator training, without future moment labels.

The default objective retains the unscaled residual sum.  Optional initial-RMS
balancing freezes separate dimensionless residual scales using training states
only, before optimization. Neither objective nor a small collocation loss
certifies statistical error.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .data import source_hash
from .interacting_config import InteractingStudyConfig
from .interacting_model import INTERACTING_MOMENT_MODEL_KIND, InteractingMomentModel
from .types import stable_hash

INTERACTING_TRAINING_KIND = "full-state-continuous-generator-mean-covariance-residual-v1"


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_hash(state):
    return stable_hash({key: _array_hash(value.detach().cpu().numpy())
                        for key, value in sorted(state.items())})


def _bank(bank, cfg, role):
    required = {"states", "horizons", "family_ids", "content_hash", "source_hash", "config_hash"}
    if not isinstance(bank, dict) or not required <= set(bank):
        raise ValueError("An interacting collocation bank requires states, horizons, families and provenance")
    if (bank["source_hash"] != source_hash() or bank["config_hash"] != cfg.config_hash
            or bank.get("target_id", cfg.target_id) != cfg.target_id
            or bank.get("role", role) != role or not bank.get("complete", True)
            or bank.get("future_payoff_labels", False)):
        raise ValueError("Interacting collocation bank source, configuration, role or target mismatch")
    if not isinstance(bank["content_hash"], str) or not bank["content_hash"]:
        raise ValueError("A nonempty committed bank content hash is required")
    states = np.asarray(bank["states"], dtype=np.float64)
    horizons = np.asarray(bank["horizons"], dtype=np.float64)
    families = np.asarray(bank["family_ids"], dtype=str)
    paths = cfg.train_paths if role == "train" else cfg.validation_paths
    n_families = paths * len(cfg.cases)
    repeats = len(cfg.state_times) * cfg.horizon_samples
    n_rows = n_families * repeats
    if (states.shape != (n_rows, cfg.n_cells) or horizons.shape != (n_rows,)
            or families.shape != (n_rows,) or not np.all(np.isfinite(states))
            or np.any(states < 0) or not np.allclose(states.sum(-1), 1., atol=1e-10, rtol=0)
            or not np.all(np.isfinite(horizons)) or np.any(horizons < 0)
            or np.any(horizons > cfg.horizon) or any(not family for family in families.tolist())):
        raise ValueError("Finite configured unit-mass states, compatible horizons and family IDs are required")
    unique, counts = np.unique(families, return_counts=True)
    if len(unique) != n_families or np.any(counts != repeats):
        raise ValueError("Every configured trajectory family must retain all derived collocation samples")
    fingerprint = {"states": _array_hash(states), "horizons": _array_hash(horizons),
                   "family_ids": stable_hash(families.tolist()), "role": role,
                   "content_hash": bank["content_hash"], "n_rows": n_rows,
                   "independent_families": n_families, "samples_per_family": repeats}
    return {"states": torch.tensor(states, dtype=torch.float64),
            "horizons": torch.tensor(horizons, dtype=torch.float64),
            "family_ids": families}, fingerprint


def interacting_objective(model, states, horizons, loss_scaling=None):
    mean, covariance = model.residual(states, horizons)
    mean_scale = 1. if loss_scaling is None else loss_scaling["mean_scale"]
    covariance_scale = 1. if loss_scaling is None else loss_scaling["covariance_scale"]
    loss = model.horizon**2 * (mean.square().mean()/mean_scale**2
                              + covariance.square().mean()/covariance_scale**2)
    return loss, (mean, covariance)


def _initial_loss_scaling(model, training, cfg, train_fingerprint):
    """Freeze scales on initial TRAIN residuals; no validation/reference input."""
    squared = np.zeros(2, dtype=np.float64)
    size = len(training["states"])
    model.eval()
    with torch.no_grad():
        for start in range(0, size, cfg.batch_size):
            end = min(start+cfg.batch_size, size)
            mean, covariance = model.residual(training["states"][start:end],
                                               training["horizons"][start:end])
            values = np.asarray((float(mean.square().mean()),
                                 float(covariance.square().mean())))
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("Nonfinite initial training residual scale")
            squared += (end-start)/size*values
    raw_rms = np.sqrt(squared)
    dimensionless_rms = cfg.horizon*raw_rms
    scales = (np.maximum(dimensionless_rms, cfg.loss_scale_floor)
              if cfg.loss_balancing == "training-initial-rms" else np.ones(2))
    return {"mode": cfg.loss_balancing,
            "units": "fixed-observable-scaled-residual-times-physical-horizon",
            "mean_scale": float(scales[0]), "covariance_scale": float(scales[1]),
            "mean_initial_train_raw_rms": float(raw_rms[0]),
            "covariance_initial_train_raw_rms": float(raw_rms[1]),
            "mean_initial_train_dimensionless_rms": float(dimensionless_rms[0]),
            "covariance_initial_train_dimensionless_rms": float(dimensionless_rms[1]),
            "floor": cfg.loss_scale_floor,
            "mean_floor_applied": bool(cfg.loss_balancing == "training-initial-rms"
                                       and dimensionless_rms[0] < cfg.loss_scale_floor),
            "covariance_floor_applied": bool(cfg.loss_balancing == "training-initial-rms"
                                             and dimensionless_rms[1] < cfg.loss_scale_floor),
            "floor_interpretation": "numerical-loss-scale-safeguard-only-no-physical-covariance-floor",
            "frozen_before_optimizer": True, "source_role": "train",
            "training_fingerprint_hash": stable_hash(train_fingerprint),
            "initial_model_state_hash": _state_hash(model.state_dict())}


def _validation(model, bank, cfg, loss_scaling):
    result = {"objective": 0., "mean_raw_rms": 0., "covariance_raw_rms": 0.,
              "mean_raw_maximum": 0., "covariance_raw_maximum": 0.,
              "n_states": len(bank["states"]),
              "independent_families": len(set(bank["family_ids"].tolist()))}
    model.eval()
    with torch.no_grad():
        size = len(bank["states"])
        for start in range(0, size, cfg.batch_size):
            end = min(start + cfg.batch_size, size)
            weight = (end - start) / size
            loss, (mean, covariance) = interacting_objective(
                model, bank["states"][start:end], bank["horizons"][start:end], loss_scaling)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite interacting generator validation residual")
            result["objective"] += weight * float(loss)
            result["mean_raw_rms"] += weight * float(mean.square().mean())
            result["covariance_raw_rms"] += weight * float(covariance.square().mean())
            result["mean_raw_maximum"] = max(result["mean_raw_maximum"], float(mean.abs().max()))
            result["covariance_raw_maximum"] = max(result["covariance_raw_maximum"],
                                                    float(covariance.abs().max()))
    result["mean_raw_rms"] **= .5
    result["covariance_raw_rms"] **= .5
    for component in ("mean", "covariance"):
        normalized = cfg.horizon*result[f"{component}_raw_rms"]/loss_scaling[f"{component}_scale"]
        result[f"{component}_normalized_rms"] = normalized
        result[f"{component}_normalized_mse"] = normalized**2
    result["raw_objective"] = cfg.horizon**2 * (
        result["mean_raw_rms"]**2+result["covariance_raw_rms"]**2)
    result["loss_balancing"] = loss_scaling["mode"]
    result["raw_maximum"] = max(result["mean_raw_maximum"], result["covariance_raw_maximum"])
    return result


def _atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _read(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def _metadata(saved, path):
    provenance = saved["provenance"]
    return {"path": str(path.resolve()), "kind": saved["kind"],
            "seed": provenance["seed"], "target_id": provenance["target_id"],
            "config_hash": provenance["config_hash"], "source_hash": provenance["source_hash"],
            "identity_hash": provenance["identity_hash"], "data_hash": provenance["data_hash"],
            "complete": saved["complete"], "adam_steps": saved["adam_steps_done"],
            "lbfgs_steps": saved["lbfgs_steps_done"], "best_step": saved["best_step"],
            "current_step": saved["adam_steps_done"] + saved["lbfgs_steps_done"],
            "initial_validation": saved["initial_validation"],
            "best_validation": saved["best_validation"], "history": saved["history"],
            "training_seconds": saved["training_seconds"],
            "parameter_count": saved["parameter_count"],
            "architecture": provenance["architecture"],
            "loss_scaling": provenance["loss_scaling"],
            "interpretation": "Allocated fitting completed is distinct from convergence or statistical accuracy"}


def _verify(saved):
    if saved.get("schema_version") != 1 or saved.get("kind") != INTERACTING_MOMENT_MODEL_KIND:
        raise ValueError("Expected a full-state interacting generator moment checkpoint")
    provenance = saved.get("provenance", {})
    identity = {key: value for key, value in provenance.items() if key != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("Interacting checkpoint provenance identity mismatch")
    if provenance.get("source_hash") != source_hash():
        raise ValueError("Interacting checkpoint source hash differs from the current source")
    cfg = InteractingStudyConfig(**provenance["config"])
    if (cfg.config_hash != provenance.get("config_hash") or cfg.target_id != provenance.get("target_id")
            or provenance.get("seed") not in cfg.optimizer_seeds
            or provenance.get("training") != INTERACTING_TRAINING_KIND
            or stable_hash(provenance.get("data_fingerprints")) != provenance.get("data_hash")):
        raise ValueError("Interacting checkpoint configuration, target or data fingerprint mismatch")
    scaling = provenance.get("loss_scaling", {})
    if (scaling.get("mode") != cfg.loss_balancing
            or scaling.get("floor") != cfg.loss_scale_floor
            or scaling.get("source_role") != "train"
            or scaling.get("frozen_before_optimizer") is not True
            or scaling.get("training_fingerprint_hash") != stable_hash(
                provenance["data_fingerprints"]["training"])):
        raise ValueError("Interacting checkpoint training-only loss scaling mismatch")
    for component in ("mean", "covariance"):
        raw = scaling.get(f"{component}_initial_train_raw_rms")
        dimensionless = scaling.get(f"{component}_initial_train_dimensionless_rms")
        scale = scaling.get(f"{component}_scale")
        if (any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
                for value in (raw, dimensionless, scale))
                or dimensionless != cfg.horizon*raw
                or scale != (max(dimensionless, cfg.loss_scale_floor)
                             if cfg.loss_balancing == "training-initial-rms" else 1.)
                or scaling.get(f"{component}_floor_applied") != bool(
                    cfg.loss_balancing == "training-initial-rms"
                    and dimensionless < cfg.loss_scale_floor)):
            raise ValueError("Interacting checkpoint frozen residual scales are invalid")
    if (saved.get("best_state_hash") != _state_hash(saved["best_state"])
            or saved.get("model_state_hash") != _state_hash(saved["model_state"])):
        raise ValueError("Interacting checkpoint model state checksum mismatch")
    adam_done, lbfgs_done = saved["adam_steps_done"], saved["lbfgs_steps_done"]
    if (type(adam_done) is not int or type(lbfgs_done) is not int
            or not 0 <= adam_done <= cfg.adam_steps or not 0 <= lbfgs_done <= cfg.lbfgs_steps
            or (lbfgs_done > 0 and adam_done != cfg.adam_steps)
            or bool(saved.get("complete")) != (adam_done == cfg.adam_steps
                                               and lbfgs_done == cfg.lbfgs_steps)
            or not 0 <= saved["best_step"] <= adam_done + lbfgs_done):
        raise ValueError("Interacting checkpoint optimization allocation mismatch")
    return cfg


def fit_interacting(cfg, seed, path, train_bank, validation_bank, deadline=None):
    """Fit residuals with independent validation and exact atomic continuation.

    AdamW takes one independently shuffled minibatch per configured step. LBFGS
    then contracts the complete fixed training pool in bounded chunks. Physical
    reference paths and future moment labels are never read by this function.
    """
    if not isinstance(cfg, InteractingStudyConfig):
        raise TypeError("An interacting study configuration is required")
    if type(seed) is not int or seed not in cfg.optimizer_seeds:
        raise ValueError("A declared nonnegative interacting optimizer seed is required")
    if path is None:
        raise ValueError("An interacting checkpoint destination is required")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("A finite absolute monotonic deadline is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    training, train_fingerprint = _bank(train_bank, cfg, "train")
    validation, validation_fingerprint = _bank(validation_bank, cfg, "validation")
    if set(training["family_ids"]) & set(validation["family_ids"]):
        raise ValueError("Training and validation trajectory families must be disjoint")
    torch.manual_seed(seed)
    model = InteractingMomentModel(cfg.spec, cfg.physics, cfg.horizon, cfg.width)
    fingerprints = {"training": train_fingerprint, "validation": validation_fingerprint}
    loss_scaling = _initial_loss_scaling(model, training, cfg, train_fingerprint)
    provenance = {"kind": model.kind, "training": INTERACTING_TRAINING_KIND,
                  "seed": seed, "target_id": cfg.target_id, "config": cfg.to_dict(),
                  "config_hash": cfg.config_hash, "source_hash": source_hash(),
                  "data_fingerprints": fingerprints, "data_hash": stable_hash(fingerprints),
                  "weight_decay": 0., "threads": 1,
                  "objective": "horizon^2*(mean-residual-entry-MSE/mean_scale^2"
                               "+full-covariance-residual-entry-MSE/covariance_scale^2)",
                  "loss_scaling": loss_scaling,
                  "selection": "minimum-independent-validation-objective-at-declared-checks",
                  "architecture": model.architecture_metadata,
                  "learner_inputs": "current-full-native-state-and-remaining-horizon",
                  "future_moment_labels": False}
    provenance["identity_hash"] = stable_hash(provenance)
    adam = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=cfg.lbfgs_max_iter,
                             history_size=30, tolerance_grad=1e-12, tolerance_change=1e-14,
                             line_search_fn="strong_wolfe")
    shuffle = torch.Generator().manual_seed(0x494D4F4D + seed)
    adam_done, lbfgs_done, elapsed_prior, history, best_step = 0, 0, 0., [], 0
    if path.exists():
        saved = _read(path)
        _verify(saved)
        if saved["provenance"]["identity_hash"] != provenance["identity_hash"]:
            raise ValueError("Interacting checkpoint source, configuration, target or input data mismatch")
        if saved["complete"]:
            return _metadata(saved, path)
        model.load_state_dict(saved["model_state"])
        adam.load_state_dict(saved["adam_state"])
        lbfgs.load_state_dict(saved["lbfgs_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        shuffle.set_state(saved["shuffle_rng_state"])
        adam_done, lbfgs_done = saved["adam_steps_done"], saved["lbfgs_steps_done"]
        elapsed_prior, history = saved["training_seconds"], saved["history"]
        initial_validation, best_validation = saved["initial_validation"], saved["best_validation"]
        best_state, best_step = saved["best_state"], saved["best_step"]
    else:
        best_validation = _validation(model, validation, cfg, loss_scaling)
        initial_validation = best_validation.copy()
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def complete():
        return adam_done == cfg.adam_steps and lbfgs_done == cfg.lbfgs_steps

    def checkpoint():
        model_state = model.state_dict()
        _atomic_save({"schema_version": 1, "kind": model.kind, "provenance": provenance,
                      "model_state": model_state, "best_state": best_state,
                      "model_state_hash": _state_hash(model_state), "best_state_hash": _state_hash(best_state),
                      "adam_state": adam.state_dict(), "lbfgs_state": lbfgs.state_dict(),
                      "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": shuffle.get_state(),
                      "adam_steps_done": adam_done, "lbfgs_steps_done": lbfgs_done,
                      "initial_validation": initial_validation, "best_validation": best_validation,
                      "best_step": best_step, "complete": complete(), "history": history,
                      "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                      "training_seconds": elapsed_prior + time.monotonic() - started}, path)

    checkpoint()
    size = len(training["states"])
    while not complete() and (deadline is None or time.monotonic() < deadline):
        tick = time.perf_counter()
        model.train()
        if adam_done < cfg.adam_steps:
            phase = "adamw"
            selected = torch.randperm(size, generator=shuffle)[:cfg.batch_size]
            adam.zero_grad(set_to_none=True)
            loss, _ = interacting_objective(model, training["states"][selected],
                                             training["horizons"][selected], loss_scaling)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite interacting generator training residual")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters()):
                raise FloatingPointError("Nonfinite interacting generator training gradient")
            adam.step()
            training_objective = float(loss.detach())
            adam_done += 1
        else:
            phase = "lbfgs"

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for start in range(0, size, cfg.batch_size):
                    end = min(start + cfg.batch_size, size)
                    weight = (end - start) / size
                    chunk_loss, _ = interacting_objective(model, training["states"][start:end],
                                                           training["horizons"][start:end], loss_scaling)
                    if not bool(torch.isfinite(chunk_loss)):
                        raise FloatingPointError("Nonfinite interacting LBFGS residual")
                    (weight * chunk_loss).backward()
                    total += weight * chunk_loss.detach()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite interacting LBFGS gradient")
                return total

            training_objective = float(lbfgs.step(closure))
            lbfgs_done += 1
        step = adam_done + lbfgs_done
        diagnostics = None
        if (step % cfg.validation_every == 0 or complete()
                or (phase == "adamw" and adam_done == cfg.adam_steps)):
            diagnostics = _validation(model, validation, cfg, loss_scaling)
            if diagnostics["objective"] < best_validation["objective"]:
                best_validation, best_step = diagnostics, step
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"step": step, "phase": phase, "adam_steps": adam_done,
                        "lbfgs_steps": lbfgs_done, "training_objective": training_objective,
                        "validation": diagnostics, "step_seconds": time.perf_counter() - tick})
        checkpoint()
    return _metadata(_read(path), path)


def load_interacting_model(path):
    """Load only a complete, integrity-checked selected checkpoint for evaluation."""
    path = Path(path)
    saved = _read(path)
    cfg = _verify(saved)
    if not saved["complete"]:
        raise ValueError("Cannot evaluate an incomplete interacting moment checkpoint")
    with torch.random.fork_rng():
        torch.manual_seed(saved["provenance"]["seed"])
        model = InteractingMomentModel(cfg.spec, cfg.physics, cfg.horizon, cfg.width)
    if saved["provenance"].get("architecture") != model.architecture_metadata:
        raise ValueError("Interacting checkpoint architecture metadata mismatch")
    scaling = saved["provenance"]["loss_scaling"]
    if scaling["initial_model_state_hash"] != _state_hash(model.state_dict()):
        raise ValueError("Interacting checkpoint initial loss-scaling model mismatch")
    model.load_state_dict(saved["best_state"])
    model.provenance, model.checkpoint_path = saved["provenance"], str(path.resolve())
    model.config, model.complete = cfg, True
    model.loss_scaling = scaling
    model.eval()
    return model
