"""Resumable deterministic generator training for time-only neural moments."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .bsde_config import BSDEConfig
from .data import source_hash
from .generator_model import (
    DK_GENERATOR_MODEL_KIND,
    OU_GENERATOR_MODEL_KIND,
    GeneratorMomentModel,
)
from .types import stable_hash

DEFAULT_SETTINGS = {"width": 32, "collocation_nodes": 65, "validation_nodes": 257,
                    "adam_steps": 1000, "adam_learning_rate": 1e-3,
                    "lbfgs_steps": 100, "lbfgs_max_iter": 10,
                    "lbfgs_learning_rate": 1., "residual_tolerance": 1e-6,
                    "threads": 1}


def _settings(settings):
    result = {**DEFAULT_SETTINGS, **settings}
    unknown = set(result)-set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown generator training settings: {sorted(unknown)}")
    for key in ("width", "collocation_nodes", "validation_nodes", "lbfgs_max_iter", "threads"):
        minimum = 2 if key in ("width", "collocation_nodes", "validation_nodes") else 1
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f"Invalid integer generator setting {key}")
    for key in ("adam_steps", "lbfgs_steps"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"Nonnegative integer {key} required")
    if result["adam_steps"]+result["lbfgs_steps"] < 1:
        raise ValueError("At least one generator optimizer step required")
    for key in ("adam_learning_rate", "lbfgs_learning_rate", "residual_tolerance"):
        value = result[key]
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"Positive finite generator setting {key} required")
    return result


def _residuals(model, times):
    mean, covariance = model.pde_residual(times)
    # Normalize derivative scales for optimization only. Gates and validation
    # retain RAW residuals too; a small normalized objective is not a certificate.
    mean_scale = 1+model.mode_rates
    covariance_scale = 1+model.mode_rates[:, None]+model.mode_rates[None, :]
    normalized_mean, normalized_covariance = mean/mean_scale, covariance/covariance_scale
    objective = normalized_mean.square().mean()+normalized_covariance.square().mean()
    return objective, mean, covariance, normalized_mean, normalized_covariance


def generator_objective(model, times):
    return _residuals(model, times)[0]


def validation_residuals(model, times):
    with torch.no_grad():
        objective, mean, covariance, normalized_mean, normalized_covariance = _residuals(model, times)
        return {"objective": float(objective),
                "mean_raw_maximum": float(mean.abs().max()),
                "covariance_raw_maximum": float(covariance.abs().max()),
                "raw_maximum": max(float(mean.abs().max()), float(covariance.abs().max())),
                "normalized_maximum": max(float(normalized_mean.abs().max()),
                                           float(normalized_covariance.abs().max())),
                "mean_raw_rms": float(mean.square().mean().sqrt()),
                "covariance_raw_rms": float(covariance.square().mean().sqrt()),
                "n_time_nodes": len(times)}


def _atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def fit_generator(base, problem, seed, path, settings, deadline=None):
    """Fit using known continuous-generator residuals, not moment forecasts.

    Collocation is a fixed uniform time grid; validation uses an independent
    offset grid. Checkpoints preserve complete Adam steps and bounded LBFGS
    outer calls. ``complete`` means the declared optimization budget finished;
    ``optimized`` additionally requires the validation raw residual tolerance.
    Neither flag certifies statistical accuracy or the stopping correction.
    """
    if hasattr(base, "base_config"):
        base = base.base_config
    settings = _settings(settings)
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer generator optimizer seed required")
    if path is None:
        raise ValueError("A generator checkpoint path is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(seed)
    model = GeneratorMomentModel(base, problem, settings["width"])
    train_times = torch.linspace(0., base.horizon, settings["collocation_nodes"], dtype=torch.float64)
    validation_times = ((torch.arange(settings["validation_nodes"], dtype=torch.float64)
                         + .3819660112501051)
                        * base.horizon/settings["validation_nodes"])
    provenance = {"kind": model.kind, "problem": problem, "seed": seed,
                  "target_id": model.target_id, "config": base.to_dict(),
                  "config_hash": base.config_hash, "source_hash": source_hash(),
                  "settings": settings, "settings_hash": stable_hash(settings),
                  "training": "known-continuous-generator-residual-no-oracle-moment-labels-v1",
                  "state_dependence": "learned-time-propagators-and-equivariant-affine-covariance",
                  "covariance_ansatz": ("t*(F0+t*A)*(F0+t*A)^T+t^2*B*B^T; known instantaneous F0; learned A,B"
                                        if problem == "dk" else "t*learned-amplitude-squared; instantaneous-noise initialization"),
                  "stopping": "initial-freeze-exact; future-correction-certified-independently",
                  "collocation_hash": stable_hash(train_times.tolist()),
                  "validation_collocation_hash": stable_hash(validation_times.tolist())}
    provenance["identity_hash"] = stable_hash(provenance)
    adam = torch.optim.Adam(model.parameters(), lr=settings["adam_learning_rate"])
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=settings["lbfgs_learning_rate"],
                             max_iter=settings["lbfgs_max_iter"], history_size=50,
                             tolerance_grad=1e-12, tolerance_change=1e-14,
                             line_search_fn="strong_wolfe")
    adam_done, lbfgs_done, elapsed_prior, history = 0, 0, 0., []
    best_validation = validation_residuals(model, validation_times)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_step = 0
    existing = path.exists()
    if existing:
        saved = _load(path)
        if (saved.get("kind") != model.kind
                or saved.get("provenance", {}).get("identity_hash") != provenance["identity_hash"]):
            raise ValueError("Generator checkpoint source, configuration, settings or target mismatch")
        model.load_state_dict(saved["model_state"])
        adam.load_state_dict(saved["adam_state"])
        lbfgs.load_state_dict(saved["lbfgs_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        adam_done, lbfgs_done = saved["adam_steps_done"], saved["lbfgs_steps_done"]
        elapsed_prior, history = saved["training_seconds"], saved["history"]
        best_validation, best_state = saved["best_validation"], saved["best_state"]
        best_step = saved["best_step"]

    def optimized():
        return best_validation["raw_maximum"] <= settings["residual_tolerance"]

    def complete():
        return optimized() or (adam_done >= settings["adam_steps"]
                               and lbfgs_done >= settings["lbfgs_steps"])

    def checkpoint():
        _atomic_save({"schema_version": 1, "kind": model.kind, "provenance": provenance,
                      "model_state": model.state_dict(), "best_state": best_state,
                      "adam_state": adam.state_dict(), "lbfgs_state": lbfgs.state_dict(),
                      "torch_rng_state": torch.get_rng_state(),
                      "adam_steps_done": adam_done, "lbfgs_steps_done": lbfgs_done,
                      "best_validation": best_validation, "best_step": best_step,
                      "complete": complete(), "optimized": optimized(), "history": history,
                      "training_seconds": elapsed_prior+time.monotonic()-started}, path)

    if not (existing and complete()):
        checkpoint()
    while not complete() and (deadline is None or time.monotonic() < deadline):
        tick = time.perf_counter()
        model.train()
        if adam_done < settings["adam_steps"]:
            phase = "adam"
            adam.zero_grad(set_to_none=True)
            objective = generator_objective(model, train_times)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite generator physics objective")
            objective.backward()
            adam.step()
            adam_done += 1
        else:
            phase = "lbfgs"

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss = generator_objective(model, train_times)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite LBFGS generator objective")
                loss.backward()
                return loss

            lbfgs.step(closure)
            lbfgs_done += 1
        model.eval()
        validation = validation_residuals(model, validation_times)
        with torch.no_grad():
            training_objective = float(generator_objective(model, train_times))
        step = adam_done+lbfgs_done
        if validation["raw_maximum"] < best_validation["raw_maximum"]:
            best_validation, best_step = validation, step
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"step": step, "phase": phase, "adam_steps": adam_done,
                        "lbfgs_steps": lbfgs_done, "training_objective": training_objective,
                        "validation": validation, "step_seconds": time.perf_counter()-tick})
        checkpoint()
    saved = _load(path)
    return {"path": str(path), "kind": model.kind, "problem": problem, "seed": seed,
            "target_id": model.target_id, "complete": saved["complete"],
            "optimized": saved["optimized"], "best_step": saved["best_step"],
            "adam_steps": adam_done, "lbfgs_steps": lbfgs_done,
            "best_validation": saved["best_validation"], "history": saved["history"],
            "training_seconds": saved["training_seconds"],
            "identity_hash": provenance["identity_hash"],
            "source_hash": provenance["source_hash"], "config_hash": base.config_hash,
            "settings_hash": provenance["settings_hash"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "collocation_nodes": settings["collocation_nodes"],
            "validation_nodes": settings["validation_nodes"],
            "interpretation": "Physics-residual fitting; stopping and statistical accuracy require independent evaluation"}


def load_generator_model(path):
    saved = _load(Path(path))
    if saved.get("kind") not in (DK_GENERATOR_MODEL_KIND, OU_GENERATOR_MODEL_KIND):
        raise ValueError("Expected a continuous generator moment checkpoint")
    if not saved.get("complete"):
        raise ValueError("Cannot evaluate an incomplete generator checkpoint")
    provenance = saved["provenance"]
    if provenance.get("source_hash") != source_hash():
        raise ValueError("Generator checkpoint source hash differs from the current source")
    identity = {key: value for key, value in provenance.items() if key != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("Generator checkpoint provenance identity mismatch")
    base = BSDEConfig(**provenance["config"])
    settings = _settings(provenance["settings"])
    if (base.config_hash != provenance["config_hash"]
            or stable_hash(settings) != provenance["settings_hash"]):
        raise ValueError("Generator checkpoint configuration/settings hash mismatch")
    model = GeneratorMomentModel(base, provenance["problem"], settings["width"])
    if model.kind != saved["kind"] or model.target_id != provenance["target_id"]:
        raise ValueError("Generator checkpoint continuous target/kind mismatch")
    model.load_state_dict(saved["best_state"])
    model.provenance, model.checkpoint_path = provenance, str(Path(path))
    model.eval()
    return model
