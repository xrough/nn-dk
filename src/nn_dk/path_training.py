"""Resumable pathwise Itô-residual training with separate physical/shuffle RNGs."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .data import source_hash
from .path_config import PathConfig
from .path_model import PATH_MODEL_KIND, CausalPathOperator
from .path_numerics import ito_residual
from .types import stable_hash

PATH_TRAINING_KIND = "uncensored-integrated-left-ito-residual-v1"


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


def _bank(bank, config, dt, expected_paths):
    if not isinstance(bank, dict) or not {"initial", "brownian", "family_ids"} <= set(bank):
        raise ValueError("A path bank requires initial states, Brownian increments and family IDs")
    initial = np.asarray(bank["initial"], dtype=np.float64)
    brownian = np.asarray(bank["brownian"], dtype=np.float64)
    families = np.asarray(bank["family_ids"], dtype=str)
    if (initial.shape != (expected_paths, config.n_cells)
            or brownian.shape != (expected_paths, round(config.horizon/dt), config.n_cells)
            or families.shape != (expected_paths,)
            or len(set(families.tolist())) != expected_paths
            or any(not family for family in families.tolist())
            or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(brownian))
            or np.any(initial < 0)
            or not np.allclose(initial.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Finite unit-mass initial histograms, matching noise and unique families required")
    fingerprint = {"initial": _array_hash(initial), "brownian": _array_hash(brownian),
                   "family_ids": stable_hash(families.tolist()), "n_paths": expected_paths}
    return {"initial": torch.tensor(initial, dtype=torch.float64),
            "brownian": torch.tensor(brownian, dtype=torch.float64),
            "family_ids": families}, fingerprint


def path_objective(model, initial, brownian, context, spec, dt, problem):
    """All-path integrated and local residual loss, without survivor conditioning.

    A crossing proposal contributes its complete pre-absorption residual. The
    fixed path/time denominator includes absorbed paths. Physical noise in the
    ansatz and the left-point stochastic integral is exactly the same input.
    """
    raw = model(initial, brownian, dt, context)
    packet = ito_residual(initial, raw[:, 1:], brownian, context, spec, dt, problem)
    cumulative_energy = packet["cumulative"].square().sum(-1).mean()
    local_energy = packet["local"].square().sum(-1).mean()
    loss = context.n_particles/(1-1/spec.n_cells)*(cumulative_energy+local_energy/dt)
    return loss, packet


def _validation(model, bank, config, case, dt):
    context, problem = config.context(case), config.problem(case)
    result = {"objective": 0., "local_rms": 0., "cumulative_rms": 0.,
              "raw_residual_maximum": 0., "mass_error": 0., "stopping_probability": 0.}
    size = len(bank["initial"])
    model.eval()
    with torch.no_grad():
        for start in range(0, size, config.batch_size):
            end = min(start+config.batch_size, size)
            weight = (end-start)/size
            loss, packet = path_objective(model, bank["initial"][start:end],
                                          bank["brownian"][start:end], context, config.spec,
                                          dt, problem)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite path validation residual")
            result["objective"] += weight*float(loss)
            result["local_rms"] += weight*float(packet["local"].square().mean())
            result["cumulative_rms"] += weight*float(packet["cumulative"].square().mean())
            result["raw_residual_maximum"] = max(result["raw_residual_maximum"],
                                                  float(packet["local"].abs().max()),
                                                  float(packet["cumulative"].abs().max()))
            result["mass_error"] = max(result["mass_error"],
                                        float((packet["states"].sum(-1)-1).abs().max()))
            result["stopping_probability"] += weight*float(
                torch.isfinite(packet["stopped_at"]).double().mean())
    result["local_rms"] **= .5
    result["cumulative_rms"] **= .5
    return result


def _atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _read(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def _metadata(saved, path):
    provenance = saved["provenance"]
    return {"path": str(path), "kind": saved["kind"], "case": provenance["case"],
            "problem": provenance["problem"], "dt": provenance["dt"],
            "seed": provenance["seed"], "target_id": provenance["target_id"],
            "config_hash": provenance["config_hash"], "source_hash": provenance["source_hash"],
            "identity_hash": provenance["identity_hash"], "data_hash": provenance["data_hash"],
            "complete": saved["complete"], "adam_steps": saved["adam_steps_done"],
            "lbfgs_steps": saved["lbfgs_steps_done"], "best_step": saved["best_step"],
            "current_step": saved["adam_steps_done"]+saved["lbfgs_steps_done"],
            "initial_validation": saved["initial_validation"],
            "best_validation": saved["best_validation"], "history": saved["history"],
            "training_seconds": saved["training_seconds"],
            "parameter_count": saved["parameter_count"],
            "interpretation": "Known-mechanism physical-noise path residual; completed fitting is not an accuracy certificate"}


def _verify(saved):
    if saved.get("schema_version") != 1 or saved.get("kind") != PATH_MODEL_KIND:
        raise ValueError("Expected a causal density-path checkpoint")
    provenance = saved.get("provenance", {})
    identity = {key: value for key, value in provenance.items() if key != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("Path checkpoint provenance identity mismatch")
    if provenance.get("source_hash") != source_hash():
        raise ValueError("Path checkpoint source hash differs from the current source")
    config = PathConfig(**provenance["config"])
    if (config.config_hash != provenance.get("config_hash")
            or provenance.get("dt") not in config.step_sizes
            or provenance.get("seed") not in config.optimizer_seeds
            or config.target_id(provenance["case"], provenance["dt"]) != provenance.get("target_id")
            or config.problem(provenance["case"]) != provenance.get("problem")
            or provenance.get("training") != PATH_TRAINING_KIND
            or stable_hash(provenance.get("data_fingerprints")) != provenance.get("data_hash")):
        raise ValueError("Path checkpoint configuration, target or data fingerprint mismatch")
    if (saved.get("best_state_hash") != _state_hash(saved["best_state"])
            or saved.get("model_state_hash") != _state_hash(saved["model_state"])):
        raise ValueError("Path checkpoint model state checksum mismatch")
    expected_complete = (saved["adam_steps_done"] == config.adam_steps
                         and saved["lbfgs_steps_done"] == config.lbfgs_steps)
    if (not 0 <= saved["adam_steps_done"] <= config.adam_steps
            or not 0 <= saved["lbfgs_steps_done"] <= config.lbfgs_steps
            or bool(saved["complete"]) != expected_complete):
        raise ValueError("Path checkpoint optimization allocation mismatch")
    return config


def fit_path(config, case, dt, seed, path, train_bank, validation_bank, deadline=None):
    """Fit fixed operator weights, saving every complete optimizer outer step.

    Adam uses a dedicated shuffle RNG, independent of the physical Brownian
    bank. LBFGS evaluates its full training objective in memory-bounded chunks.
    Validation chooses a complete saved parameter state. Exact continuation
    preserves both optimizer histories, global Torch RNG and shuffle RNG.
    """
    if not isinstance(config, PathConfig) or case not in config.cases or dt not in config.step_sizes:
        raise ValueError("A declared PathConfig case and numerical timestep are required")
    if type(seed) is not int or seed not in config.optimizer_seeds:
        raise ValueError("A declared nonnegative path optimizer seed is required")
    if path is None:
        raise ValueError("A path checkpoint destination is required")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("A finite absolute monotonic deadline is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    training, train_fingerprint = _bank(train_bank, config, dt, config.train_paths)
    validation, validation_fingerprint = _bank(validation_bank, config, dt, config.validation_paths)
    if set(training["family_ids"]) & set(validation["family_ids"]):
        raise ValueError("Training and validation trajectory families must be disjoint")
    torch.manual_seed(seed)
    model = CausalPathOperator(config.spec, config.problem(case), config.width, config.depth)
    fingerprints = {"training": train_fingerprint, "validation": validation_fingerprint}
    provenance = {"kind": model.kind, "training": PATH_TRAINING_KIND, "case": case,
                  "problem": config.problem(case), "seed": seed, "dt": dt,
                  "target_id": config.target_id(case, dt), "config": config.to_dict(),
                  "config_hash": config.config_hash, "source_hash": source_hash(),
                  "data_fingerprints": fingerprints, "data_hash": stable_hash(fingerprints),
                  "weight_decay": 0., "threads": 1,
                  "selection": "minimum independent-validation integrated-plus-local residual"}
    provenance["identity_hash"] = stable_hash(provenance)
    adam = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=config.lbfgs_max_iter,
                             history_size=30, tolerance_grad=1e-12, tolerance_change=1e-14,
                             line_search_fn="strong_wolfe")
    shuffle = torch.Generator().manual_seed(0x50415448+seed)
    adam_done, lbfgs_done, elapsed_prior, history = 0, 0, 0., []
    best_validation = _validation(model, validation, config, case, dt)
    initial_validation = best_validation.copy()
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_step = 0
    if path.exists():
        saved = _read(path)
        _verify(saved)
        if saved["provenance"]["identity_hash"] != provenance["identity_hash"]:
            raise ValueError("Path checkpoint source, configuration, target or input data mismatch")
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

    def complete():
        return adam_done == config.adam_steps and lbfgs_done == config.lbfgs_steps

    def checkpoint():
        model_state = model.state_dict()
        _atomic_save({"schema_version": 1, "kind": model.kind, "provenance": provenance,
                      "model_state": model_state, "best_state": best_state,
                      "model_state_hash": _state_hash(model_state),
                      "best_state_hash": _state_hash(best_state),
                      "adam_state": adam.state_dict(), "lbfgs_state": lbfgs.state_dict(),
                      "torch_rng_state": torch.get_rng_state(),
                      "shuffle_rng_state": shuffle.get_state(),
                      "adam_steps_done": adam_done, "lbfgs_steps_done": lbfgs_done,
                      "initial_validation": initial_validation,
                      "best_validation": best_validation, "best_step": best_step,
                      "complete": complete(), "history": history,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "training_seconds": elapsed_prior+time.monotonic()-started}, path)

    checkpoint()
    context, problem = config.context(case), config.problem(case)
    while not complete() and (deadline is None or time.monotonic() < deadline):
        tick = time.perf_counter()
        model.train()
        if adam_done < config.adam_steps:
            phase = "adamw"
            selected = torch.randperm(config.train_paths, generator=shuffle)[:config.batch_size]
            adam.zero_grad(set_to_none=True)
            loss, _ = path_objective(model, training["initial"][selected],
                                     training["brownian"][selected], context, config.spec, dt, problem)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite path training residual")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise FloatingPointError("Nonfinite path training gradient")
            adam.step()
            training_objective = float(loss.detach())
            adam_done += 1
        else:
            phase = "lbfgs"

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for start in range(0, config.train_paths, config.batch_size):
                    end = min(start+config.batch_size, config.train_paths)
                    weight = (end-start)/config.train_paths
                    chunk_loss, _ = path_objective(model, training["initial"][start:end],
                                                   training["brownian"][start:end], context,
                                                   config.spec, dt, problem)
                    if not bool(torch.isfinite(chunk_loss)):
                        raise FloatingPointError("Nonfinite LBFGS path residual")
                    (weight*chunk_loss).backward()
                    total += weight*chunk_loss.detach()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                    raise FloatingPointError("Nonfinite LBFGS path gradient")
                return total

            training_objective = float(lbfgs.step(closure))
            lbfgs_done += 1
        diagnostics = _validation(model, validation, config, case, dt)
        step = adam_done+lbfgs_done
        if diagnostics["objective"] < best_validation["objective"]:
            best_validation, best_step = diagnostics, step
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"step": step, "phase": phase, "adam_steps": adam_done,
                        "lbfgs_steps": lbfgs_done, "training_objective": training_objective,
                        "validation": diagnostics, "step_seconds": time.perf_counter()-tick})
        checkpoint()
    return _metadata(_read(path), path)


def load_path_model(path):
    """Restore selected fixed weights only after strict provenance/integrity checks."""
    path = Path(path)
    saved = _read(path)
    config = _verify(saved)
    if not saved["complete"]:
        raise ValueError("Cannot evaluate an incomplete density-path checkpoint")
    provenance = saved["provenance"]
    model = CausalPathOperator(config.spec, provenance["problem"], config.width, config.depth)
    model.load_state_dict(saved["best_state"])
    model.provenance, model.checkpoint_path = provenance, str(path)
    model.path_config, model.case, model.dt = config, provenance["case"], provenance["dt"]
    model.target_id = provenance["target_id"]
    model.eval()
    return model
