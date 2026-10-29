"""Grouped training, independent replicate labels, resumable checkpoints and baselines."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .models import (
    ConservativeTransition,
    DirectMomentRegressor,
    GaussianAutoregression,
    RidgeMomentRegressor,
)
from .types import Context, QueryBank, TargetSpec, TrajectoryBatch, stable_hash


def _validate_groups(train, validation, target):
    if not train or not validation:
        raise ValueError("Separate nonempty training and validation contexts required")
    train_ids = {c.context_id for c, _ in train}
    validation_ids = {c.context_id for c, _ in validation}
    if train_ids & validation_ids:
        raise ValueError("Training and validation context groups overlap")
    for collection in (train, validation):
        if len({c.context_id for c, _ in collection}) != len(collection):
            raise ValueError("Each context must occur once with all of its replicate paths")
        for context, batch in collection:
            target.require_compatible(batch.target)
            if not batch.valid.all():
                raise ValueError("Training cannot silently discard failed trajectories")
            identifiers = batch.diagnostics.get("trajectory_ids")
            if identifiers is not None and (len(identifiers) != len(batch.states) or len(set(identifiers)) != len(identifiers)):
                raise ValueError("Independent trajectories require unique identifiers")
            recorded_context = batch.diagnostics.get("context_id")
            if recorded_context is not None and recorded_context != context.context_id:
                raise ValueError("Training trajectory context identifier mismatch")
            if len(context.features()) != 10:
                raise ValueError("Three-mode initial-profile descriptors required")


def transition_examples(collection, lag):
    states, next_states, contexts, elapsed = [], [], [], []
    for context, batch in collection:
        lookup = {round(float(t), 10): i for i, t in enumerate(batch.times)}
        for i, t in enumerate(batch.times):
            j = lookup.get(round(float(t + lag), 10))
            if j is not None:
                count = len(batch.states)
                states.append(batch.states[:, i])
                next_states.append(batch.states[:, j])
                contexts.append(np.tile(context.features(), (count, 1)))
                elapsed.append(np.full(count, t))
    if not states:
        raise ValueError("No observed transitions at the requested learned lag")
    return tuple(torch.as_tensor(np.concatenate(a), dtype=torch.float32)
                 for a in (states, next_states, contexts, elapsed))


def independent_pair_labels(collection, queries):
    """Disjoint trajectory pairs per context; never differences sharing a replicate.

    The mean head sees both outcomes. The covariance head sees their single
    unbiased N/2 difference outer product. An unpaired final path is excluded
    from this pair-based objective; fit metadata reports the excluded count.
    """
    features, outcomes, covariance = [], [], []
    for context, batch in collection:
        if not batch.valid.all():
            raise ValueError("Invalid ensembles cannot supply training labels")
        identifiers = batch.diagnostics.get("trajectory_ids")
        if identifiers is not None and (len(identifiers) != len(batch.states) or len(set(identifiers)) != len(identifiers)):
            raise ValueError("Covariance pairs require distinct trajectory identifiers")
        y = queries.extract(batch)
        n = len(y) // 2
        if not n:
            raise ValueError("At least two independent replicates per context required")
        pairs = y[:2*n].reshape(n, 2, -1)
        difference = pairs[:, 0] - pairs[:, 1]
        features.append(np.tile(context.features(), (n, 1)))
        outcomes.append(pairs)
        covariance.append(context.n_particles / 2 * np.einsum("ni,nj->nij", difference, difference))
    return tuple(torch.as_tensor(np.concatenate(a), dtype=torch.float32)
                 for a in (features, outcomes, covariance))


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _fingerprint(collection):
    digest = hashlib.sha256()
    for c, batch in collection:
        digest.update(c.context_id.encode())
        digest.update(batch.times.tobytes())
        digest.update(batch.states.tobytes())
    return digest.hexdigest()


def _checkpoint(model, optimizer, epoch, best_state, best_loss, patience, generator,
                provenance, complete, elapsed_seconds, history):
    return dict(schema_version=1, model_state=model.state_dict(), best_state=best_state,
                optimizer_state=optimizer.state_dict(), epoch=epoch, best_loss=best_loss,
                patience_count=patience, generator_state=generator.get_state(),
                torch_rng_state=torch.get_rng_state(), provenance=provenance,
                complete=complete, elapsed_seconds=elapsed_seconds, history=history)


def _load(path):
    # Checkpoints are local trusted experiment outputs; loading arbitrary files is unsafe.
    return torch.load(path, map_location="cpu", weights_only=False)


def _fit_one(kind, train, validation, target, queries, lag, config, output_dir, seed, deadline):
    started = time.monotonic()
    torch.manual_seed(seed + (0 if kind == "a" else 1_000_003))
    torch.set_num_threads(int(config.get("threads", 1)))
    epochs = int(config.get("epochs", 100))
    patience_limit = int(config.get("patience", 10))
    batch_size = int(config.get("batch_size", 128))
    if epochs < 1 or patience_limit < 1 or batch_size < 1:
        raise ValueError("Positive epochs, patience and batch size required")
    model = ConservativeTransition(target, lag) if kind == "a" else DirectMomentRegressor(target, queries)
    model.supported_particles = tuple(sorted({c.n_particles for c, _ in train}))
    if any(c.n_particles not in model.supported_particles for c, _ in validation):
        raise ValueError("Validation particle counts outside training support")
    model.normalization.fit([c.features() for c, _ in train])
    training_data = transition_examples(train, lag) if kind == "a" else independent_pair_labels(train, queries)
    validation_data = transition_examples(validation, lag) if kind == "a" else independent_pair_labels(validation, queries)
    if kind == "b":
        with torch.no_grad():
            outcomes = training_data[1].flatten(0, 1)
            model.mean_center.copy_(outcomes.mean(0))
            model.mean_scale.copy_(outcomes.std(0, unbiased=False).clamp_min(1e-3))
            diagonal_mean = torch.diagonal(training_data[2], dim1=-2, dim2=-1).mean().clamp_min(1e-6)
            model.covariance_scale.copy_(diagonal_mean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    generator = torch.Generator().manual_seed(seed + 37)
    path = output_dir / f"{kind}_seed{seed}.pt"
    provenance = dict(kind=kind, target=asdict(target), query_bank=asdict(queries), lag=lag,
                      width=model.width, seed=seed, supported_particles=list(model.supported_particles),
                      config=dict(config), train_hash=_fingerprint(train), validation_hash=_fingerprint(validation))
    # Epoch and wall budget may be increased for a continued fit.
    identity = {**provenance, "config": {k: v for k, v in config.items() if k != "epochs"}}
    provenance["identity_hash"] = stable_hash(identity)
    epoch, best_loss, patience_count, elapsed_prior, history = 0, float("inf"), 0, 0., []
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if path.exists():
        checkpoint = _load(path)
        if checkpoint["provenance"]["identity_hash"] != provenance["identity_hash"]:
            raise ValueError(f"Checkpoint {path} belongs to different data/configuration")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
        best_state, patience_count = checkpoint["best_state"], checkpoint["patience_count"]
        generator.set_state(checkpoint["generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        elapsed_prior, history = checkpoint["elapsed_seconds"], checkpoint["history"]
    def objective(data, indices):
        if kind == "a":
            return model.nll(*(x[indices] for x in data))
        x, outcomes, covariance = (x[indices] for x in data)
        mean, prediction = model(x)
        mean_error = ((mean[:, None] - outcomes) / model.mean_scale).square().mean()
        covariance_error = ((prediction - covariance) / model.covariance_scale).square().mean()
        return mean_error + covariance_error
    complete = epoch >= epochs or patience_count >= patience_limit
    # Persist an initialization checkpoint even if the global budget has just expired.
    _atomic_save(_checkpoint(model, optimizer, epoch, best_state, best_loss, patience_count,
                             generator, provenance, complete, elapsed_prior, history), path)
    while not complete and (deadline is None or time.monotonic() < deadline):
        model.train()
        permutation = torch.randperm(len(training_data[0]), generator=generator)
        train_loss, count = 0., 0
        # Finish an epoch atomically for exact resume without duplicate partial updates.
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(training_data, indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {kind} training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
            count += len(indices)
        model.eval()
        with torch.no_grad():
            losses = [(float(objective(validation_data, ids)) * len(ids), len(ids))
                      for ids in torch.arange(len(validation_data[0])).split(batch_size)]
        validation_loss = sum(x[0] for x in losses) / sum(x[1] for x in losses)
        epoch += 1
        if np.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss, patience_count = validation_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        history.append(dict(epoch=epoch, training_loss=train_loss/count, validation_loss=validation_loss))
        complete = epoch >= epochs or patience_count >= patience_limit
        _atomic_save(_checkpoint(model, optimizer, epoch, best_state, best_loss, patience_count,
                                 generator, provenance, complete,
                                 elapsed_prior + time.monotonic() - started, history), path)
    final = _load(path)
    return dict(path=str(path), complete=final["complete"], epochs=epoch,
                best_validation_loss=best_loss if np.isfinite(best_loss) else None,
                training_seconds=final["elapsed_seconds"], training_examples=len(training_data[0]),
                validation_examples=len(validation_data[0]), seed=seed)


def fit_models(train: list[tuple[Context, TrajectoryBatch]], validation, target: TargetSpec,
               queries: QueryBank, lag: float, config: dict, output_dir: Path, seed: int,
               deadline: float | None = None) -> dict:
    _validate_groups(train, validation, target)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    a = _fit_one("a", train, validation, target, queries, lag, config, output_dir, seed, deadline)
    b = _fit_one("b", train, validation, target, queries, lag, config, output_dir, seed, deadline)
    metadata = dict(a=a, b=b, complete=a["complete"] and b["complete"], seed=seed,
                    independent_covariance_pairs=True, normalization="training-contexts-only",
                    excluded_unpaired_replicates={"training": sum(len(b.states) % 2 for _, b in train),
                                                  "validation": sum(len(b.states) % 2 for _, b in validation)})
    (output_dir / f"training_seed{seed}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_a(path) -> ConservativeTransition:
    checkpoint = _load(Path(path))
    p = checkpoint["provenance"]
    if p["kind"] != "a":
        raise ValueError("Expected neural transition checkpoint")
    model = ConservativeTransition(TargetSpec(**p["target"]), p["lag"], p["width"])
    model.load_state_dict(checkpoint["best_state"])
    model.supported_particles = tuple(p["supported_particles"])
    model.eval()
    return model


def load_b(path) -> DirectMomentRegressor:
    checkpoint = _load(Path(path))
    p = checkpoint["provenance"]
    if p["kind"] != "b":
        raise ValueError("Expected direct moment checkpoint")
    model = DirectMomentRegressor(TargetSpec(**p["target"]), QueryBank(**p["query_bank"]), p["width"])
    model.load_state_dict(checkpoint["best_state"])
    model.supported_particles = tuple(p["supported_particles"])
    model.eval()
    return model


def _regression_targets(collection, queries):
    x, outcome, covariance = independent_pair_labels(collection, queries)
    return x.numpy(), np.column_stack((outcome.mean(1).numpy(), covariance.flatten(1).numpy()))


def fit_baselines(train, validation, target, queries, lag, output_dir):
    _validate_groups(train, validation, target)
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = np.stack([c.features() for c, _ in train])
    center = features.mean(0)
    deviation = features.std(0)
    scale = np.where(deviation > 1e-6, deviation, 1.)
    support = sorted({c.n_particles for c, _ in train})
    states, next_states, contexts, elapsed = (x.numpy().astype(float) for x in transition_examples(train, lag))
    from scipy.linalg import helmert
    q = helmert(target.n_cells).T
    design = np.column_stack((np.ones(len(states)), states @ q, (contexts-center)/scale, elapsed))
    response = (next_states-states) @ q
    penalty = 1e-4 * np.eye(design.shape[1])
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T@design+penalty, design.T@response)
    residual = response-design@coefficients
    # Fit innovation quadratic variation, restoring N / lag per trajectory.
    scaled_residual = residual * np.sqrt(np.exp(contexts[:, 9])/lag)[:, None]
    covariance = scaled_residual.T@scaled_residual/len(residual)
    np.savez_compressed(output_dir/"autoregression.npz", coefficients=coefficients, covariance=covariance,
                        center=center, scale=scale)
    x, y = _regression_targets(train, queries)
    xv, yv = _regression_targets(validation, queries)
    x, xv = (x-center)/scale, (xv-center)/scale
    loss_scale = np.maximum(y.std(0), 1e-3)
    models = {}
    for method in ("polynomial_ridge", "kernel_ridge"):
        best = None
        for gamma in ((0.03, 0.1, 0.3) if method == "kernel_ridge" else (0.1,)):
            a = RidgeMomentRegressor.polynomial(x) if method == "polynomial_ridge" else RidgeMomentRegressor.kernel(x,x,gamma)
            av = RidgeMomentRegressor.polynomial(xv) if method == "polynomial_ridge" else RidgeMomentRegressor.kernel(xv,x,gamma)
            for alpha in (1e-4, 0.01, 1.0):
                if method == "polynomial_ridge":
                    coef = np.linalg.solve(a.T@a + alpha*np.eye(a.shape[1]), a.T@y)
                else:
                    coef = np.linalg.solve(a+alpha*np.eye(len(a)), y)
                error = float(np.mean(((av@coef-yv)/loss_scale)**2))
                if best is None or error < best[0]:
                    best = (error, gamma, alpha, coef)
        error, gamma, alpha, coef = best
        np.savez_compressed(output_dir/f"{method}.npz", center=center, scale=scale, x_train=x,
                            coefficients=coef, gamma=gamma, alpha=alpha)
        models[method] = dict(validation_loss=error, gamma=gamma, alpha=alpha)
    metadata = dict(target=asdict(target), query_bank=asdict(queries), lag=lag,
                    supported_particles=support, training_seconds=time.perf_counter()-started,
                    excluded_unpaired_replicates={"training": sum(len(b.states) % 2 for _, b in train),
                                                  "validation": sum(len(b.states) % 2 for _, b in validation)},
                    training_hash=_fingerprint(train), validation_hash=_fingerprint(validation),
                    statistical_models=models, covariance_output_policy="unconstrained symmetric output; indefinite results reported")
    (output_dir/"baselines.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def load_baselines(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"baselines.json").read_text())
    target, queries = TargetSpec(**metadata["target"]), QueryBank(**metadata["query_bank"])
    with np.load(directory/"autoregression.npz") as a:
        autoregression = GaussianAutoregression(target, metadata["lag"], a["coefficients"], a["covariance"],
                                                a["center"], a["scale"], metadata["supported_particles"])
    models = {"autoregression": autoregression}
    for method in ("polynomial_ridge", "kernel_ridge"):
        with np.load(directory/f"{method}.npz") as a:
            models[method] = RidgeMomentRegressor(target, queries, method, a["center"], a["scale"],
                a["x_train"], a["coefficients"], metadata["supported_particles"], float(a["gamma"]), float(a["alpha"]))
    return models
