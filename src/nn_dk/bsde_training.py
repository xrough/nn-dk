"""Resumable terminal-matching fits for conditional BSDE moment predictions."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .bsde_model import (
    DK_BSDE_MODEL_KIND,
    OU_BSDE_MODEL_KIND,
    BSDEMomentModel,
    moment_payoff,
)
from .data import source_hash
from .types import stable_hash


def _prepare(trace, config, problem):
    if trace.problem != problem:
        raise ValueError("Training trace problem does not match the requested BSDE target")
    if not np.isclose(trace.dt, config.dt, atol=1e-12, rtol=0):
        raise ValueError("Forward discretization differs from the declared training target")
    expected_dimension = config.spec.n_cells if problem == "dk" else 4
    if trace.states.shape[-1] != expected_dimension or trace.increments.shape[-1] != expected_dimension:
        raise ValueError("BSDE trace state or Brownian driver dimension mismatch")
    if trace.times[0] != 0 or trace.times[-1] < max(config.query_times) - 1e-12:
        raise ValueError("Forward traces must cover every declared payoff horizon from time zero")
    steps = {float(horizon): int(round(horizon / trace.dt)) for horizon in config.query_times}
    for horizon, index in steps.items():
        if not np.isclose(trace.times[index], horizon, atol=1e-10, rtol=0):
            raise ValueError("Payoff horizons must be present on the forward trace grid")
    return {"states": torch.as_tensor(trace.states, dtype=torch.float64),
            "increments": torch.as_tensor(trace.increments, dtype=torch.float64),
            "active": torch.as_tensor(trace.active, dtype=torch.bool),
            "times": torch.as_tensor(trace.times, dtype=torch.float64), "steps": steps,
            "anchor_ids": tuple(trace.anchor_ids), "family_ids": tuple(trace.family_ids),
            "fingerprint": trace.fingerprint, "n_paths": trace.n_paths}


def terminal_residual(model, data, ids, horizon):
    """Whole-path residual, using only predictable martingale integrand inputs.

    Full increments on crossing steps are essential: conditioning an increment on
    its boundary crossing fraction would destroy the zero-mean control identity.
    """
    end = data["steps"][float(horizon)]
    states = data["states"][ids, :end]
    initial = data["states"][ids, 0]
    times = data["times"][:end]
    horizons = torch.full((len(ids),), float(horizon), dtype=torch.float64)
    center = model.deterministic_center(initial, horizons)
    target = moment_payoff(model.scaled_observables(data["states"][ids, end]), center)
    value = model.initial_payoff(initial, horizons, center)
    z = model.integrands(states, (horizon - times)[None].expand(len(ids), -1),
                         horizons[:, None].expand(-1, end), center[:, None].expand(-1, end, -1))
    increments = data["increments"][ids, :end]
    active = data["active"][ids, :end]
    martingale = torch.einsum("bspd,bsd,bs->bp", z, increments, active.to(torch.float64))
    return target - value - martingale


def terminal_objective(model, data, ids):
    """Equal block weight for four means and ten symmetric second moments."""
    losses = []
    for horizon in model.config.query_times:
        residual = terminal_residual(model, data, ids, horizon)
        losses.append(residual[:, :4].square().mean() + residual[:, 4:].square().mean())
    return torch.stack(losses).mean()


def _residual_diagnostics(model, data, batch_size):
    diagnostics = {}
    with torch.no_grad():
        for horizon in model.config.query_times:
            rows = [terminal_residual(model, data, ids, horizon)
                    for ids in torch.arange(data["n_paths"]).split(batch_size)]
            residual = torch.cat(rows)
            diagnostics[str(horizon)] = {
                "mean_residual": residual.mean(0).tolist(),
                "mean_block_mse": float(residual[:, :4].square().mean()),
                "second_moment_block_mse": float(residual[:, 4:].square().mean()),
                "n_paths": data["n_paths"],
                "interpretation": "Terminal mismatch includes finite-step martingale representation error",
            }
    return diagnostics


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load(path):
    # This workflow only loads locally generated, trusted experiment checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_bsde(training, validation, config, problem, seed, output_path, deadline=None):
    """Train independent value/integrand nets, saving complete epochs atomically.

    The source, numerical configuration, data, and optimizer seed are immutable
    on resume. Wall-clock expiry preserves the current optimizer and shuffle RNG.
    """
    started = time.monotonic()
    if output_path is None:
        raise ValueError("A BSDE checkpoint output path is required")
    if set(training.anchor_ids) & set(validation.anchor_ids):
        raise ValueError("Training and validation initial-anchor families must be disjoint")
    if set(training.family_ids) & set(validation.family_ids):
        raise ValueError("Training and validation continuation families must be disjoint")
    settings = dict(config.training)
    torch.set_num_threads(settings.get("threads", 1))
    torch.manual_seed(seed)
    model = BSDEMomentModel(config, problem)
    train, valid = _prepare(training, config, problem), _prepare(validation, config, problem)
    model.fit_normalization(train["states"][:, 0], train["anchor_ids"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 713)
    provenance = {
        "kind": model.kind, "problem": problem, "seed": int(seed),
        "target_id": model.target_id, "config": config.to_dict(),
        "config_hash": config.config_hash, "source_hash": source_hash(),
        "training_fingerprint": train["fingerprint"],
        "validation_fingerprint": valid["fingerprint"],
        "normalization": "unique-training-initial-anchors-only-v1",
        "martingale": "shared-forward-full-brownian-increments-active-at-start-v1",
        "value_inputs": "initial-mass-zero-coordinates-and-horizon-only",
        "integrand_inputs": "prestep-state-remaining-time-horizon-deterministic-center",
        "second_moments": "centered-svec-sqrt2-offdiagonals-v1",
        "block_weights": {"mean": 1., "second_moment": 1.},
    }
    provenance["identity_hash"] = stable_hash(provenance)
    epoch, best_loss, patience_count = 0, float("inf"), 0
    elapsed_prior, history = 0., []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = path.exists()
    if existing_checkpoint:
        saved = _load(path)
        if saved.get("kind") != model.kind or saved.get("provenance", {}).get("identity_hash") != provenance["identity_hash"]:
            raise ValueError("BSDE checkpoint source, configuration, data, seed, or target mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        epoch, best_loss = saved["epoch"], saved["best_loss"]
        patience_count, elapsed_prior = saved["patience_count"], saved["training_seconds"]
        history, best_state = saved["history"], saved["best_state"]
    complete = epoch >= settings["epochs"] or patience_count >= settings["patience"]

    def checkpoint():
        _atomic_save({"schema_version": 1, "kind": model.kind, "provenance": provenance,
                      "model_state": model.state_dict(), "best_state": best_state,
                      "optimizer_state": optimizer.state_dict(),
                      "generator_state": generator.get_state(),
                      "torch_rng_state": torch.get_rng_state(), "epoch": epoch,
                      "best_loss": best_loss, "patience_count": patience_count,
                      "complete": complete, "history": history,
                      "training_seconds": elapsed_prior + time.monotonic() - started}, path)

    # A completed fit is immutable: replay must preserve its SHA and cost metadata.
    if not (existing_checkpoint and complete):
        checkpoint()
    while not complete and (deadline is None or time.monotonic() < deadline):
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0., 0
        for ids in torch.randperm(train["n_paths"], generator=generator).split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = terminal_objective(model, train, ids)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BSDE terminal training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 100.))
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        model.eval()
        with torch.no_grad():
            valid_loss = sum(float(terminal_objective(model, valid, ids)) * len(ids)
                             for ids in torch.arange(valid["n_paths"]).split(settings["batch_size"])) / valid["n_paths"]
        if not np.isfinite(valid_loss):
            raise FloatingPointError("Nonfinite BSDE terminal validation objective")
        epoch += 1
        if valid_loss < best_loss:
            best_loss, patience_count = valid_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience_count += 1
        history.append({"epoch": epoch, "training_loss": total / count,
                        "validation_loss": valid_loss,
                        "epoch_seconds": time.perf_counter() - epoch_started})
        complete = epoch >= settings["epochs"] or patience_count >= settings["patience"]
        checkpoint()
    saved = _load(path)
    model.load_state_dict(best_state)
    model.eval()
    return {"path": str(path), "kind": model.kind, "problem": problem, "seed": int(seed),
            "complete": bool(saved["complete"]), "epochs": epoch,
            "best_validation_loss": best_loss if np.isfinite(best_loss) else None,
            "training_seconds": saved["training_seconds"],
            "first_epoch_seconds": history[0]["epoch_seconds"] if history else None,
            "training_paths": train["n_paths"], "validation_paths": valid["n_paths"],
            "training_anchors": len(set(train["anchor_ids"])),
            "validation_anchors": len(set(valid["anchor_ids"])),
            "identity_hash": provenance["identity_hash"], "history": history,
            "validation_residuals": _residual_diagnostics(model, valid, settings["batch_size"])
            if saved["complete"] else {}}


def load_bsde_model(path):
    from .bsde_config import BSDEConfig

    saved = _load(Path(path))
    if saved.get("kind") not in (DK_BSDE_MODEL_KIND, OU_BSDE_MODEL_KIND):
        raise ValueError("Expected a conditional BSDE moment checkpoint")
    if not saved.get("complete", False):
        raise ValueError("Cannot evaluate an incomplete BSDE checkpoint")
    provenance = saved["provenance"]
    if provenance.get("source_hash") != source_hash():
        raise ValueError("BSDE checkpoint source hash differs from the current numerical source")
    identity = {key: value for key, value in provenance.items() if key != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("BSDE checkpoint provenance or data identity hash mismatch")
    config = BSDEConfig(**provenance["config"])
    if config.config_hash != provenance["config_hash"]:
        raise ValueError("BSDE checkpoint configuration hash mismatch")
    model = BSDEMomentModel(config, provenance["problem"])
    if model.kind != saved["kind"] or model.target_id != provenance["target_id"]:
        raise ValueError("BSDE checkpoint target identity mismatch")
    model.load_state_dict(saved["best_state"])
    model.provenance = provenance
    model.checkpoint_path = str(Path(path))
    model.eval()
    return model
