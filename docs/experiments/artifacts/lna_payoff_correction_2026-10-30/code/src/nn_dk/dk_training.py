"""A-only coefficient fitting with family splits and exact epoch-boundary resume."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .dk_model import COEFFICIENT_MODEL_KIND, DKCoefficientModel
from .native_types import DKTrainingBatch, NativeDKSpec, physics_features
from .types import stable_hash


def _validate_groups(train, validation, spec):
    if not train or not validation:
        raise ValueError("Separate nonempty training and validation families required")
    families = []
    sample_ids = set()
    for collection in (train, validation):
        fold_families = set()
        for batch in collection:
            if not isinstance(batch, DKTrainingBatch):
                raise ValueError("Native DKTrainingBatch data required")
            spec.require_compatible(batch.spec)
            for identifier in batch.sample_ids:
                if identifier in sample_ids:
                    raise ValueError("Duplicate training/validation sample IDs")
                sample_ids.add(identifier)
            fold_families.update(batch.family_ids)
            if not np.allclose(batch.states.sum(-1), 1, atol=1e-10, rtol=0):
                raise ValueError("Training origins must be finite-volume probability masses")
        families.append(fold_families)
    if families[0] & families[1]:
        raise ValueError("Training and validation trajectory families overlap")


def _fingerprint(collection):
    digest = hashlib.sha256()
    for batch in collection:
        digest.update(batch.spec.target_id.encode())
        digest.update(batch.context.context_id.encode())
        digest.update(repr((batch.family_ids, batch.sample_ids, batch.proposal_dt)).encode())
        for value in (batch.states, batch.active, batch.proposals):
            digest.update(b"absent" if value is None else np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _prepare(collection, spec, require_proposals=False, oracle_labels=True):
    keys = ["states", "physics", "particles", "proposals", "dt", "has_proposal"]
    if oracle_labels:
        keys += ["drift", "sigma"]
    arrays = {key: [] for key in keys}
    families, samples = set(), []
    negative_proposals, inactive = 0, 0
    for batch in collection:
        spec.require_compatible(batch.spec)
        active = np.asarray(batch.active, dtype=bool)
        inactive += int((~active).sum())
        if not active.any():
            continue
        states = batch.states[active]
        if np.any(states <= spec.mass_stop):
            raise ValueError("Active-at-start origins must be strictly above the stopping boundary")
        if require_proposals and batch.proposals is None:
            raise ValueError("Increment supervision requires uncensored raw Euler proposals")
        n = len(states)
        values = dict(states=states, physics=np.tile(physics_features(batch.context), (n, 1)),
                      particles=np.full(n, batch.context.n_particles),
                      proposals=np.zeros_like(states) if batch.proposals is None else batch.proposals[active],
                      dt=np.zeros(n) if batch.proposal_dt is None else np.full(n, batch.proposal_dt),
                      has_proposal=np.full(n, batch.proposals is not None, dtype=bool))
        if oracle_labels:
            from .native import native_coefficients
            # Bound temporary covariance allocation; retain drift/face factors
            # instead of an O(samples * G**2) label tensor.
            pieces = []
            for x in np.array_split(states, max(1, (len(states) + 1023) // 1024)):
                coefficients = native_coefficients(x, batch.context, spec)
                pieces.append((coefficients.drift, coefficients.face_amplitudes))
            values.update(drift=np.concatenate([x[0] for x in pieces]),
                          sigma=np.concatenate([x[1] for x in pieces]))
        if batch.proposals is not None:
            negative_proposals += int((batch.proposals[active].min(-1) < 0).sum())
            if not np.allclose(batch.proposals[active].sum(-1), 1, atol=1e-10, rtol=0):
                raise ValueError("Uncensored proposals must preserve total mass")
        for key, value in values.items():
            arrays[key].append(np.asarray(value))
        families.update(np.asarray(batch.family_ids)[active].tolist())
        samples.extend(np.asarray(batch.sample_ids)[active].tolist())
    if not arrays["states"]:
        raise ValueError("At least one active-at-start proposal origin required per fold")
    result = {key: torch.as_tensor(np.concatenate(value), dtype=(torch.bool if key == "has_proposal" else torch.float64))
              for key, value in arrays.items()}
    result["metadata"] = dict(samples=len(samples), families=len(families), inactive_starts=inactive,
                              negative_exit_proposals=negative_proposals,
                              selection="active-at-start; exit proposals retained")
    return result


def _truth_covariance(sigma, incidence):
    factor = incidence[None] * sigma[:, None, :]
    return factor @ factor.transpose(-1, -2)


def _training_scales(data, model):
    drift_scale = data["drift"].square().mean().sqrt()
    total, count = 0., 0
    for sigma in data["sigma"].split(1024):
        covariance = _truth_covariance(sigma, model.incidence)
        total += float(covariance.square().sum())
        count += covariance.numel()
    # These are objective units, not regularization of any covariance.
    return dict(drift=float(drift_scale) if drift_scale > 1e-14 else 1.,
                covariance=np.sqrt(total / count) if total > 1e-28 else 1.)


def _objective(model, data, ids, mode, scales):
    if mode == "increments":
        return model.nll(data["states"][ids], data["proposals"][ids], data["physics"][ids],
                         data["dt"][ids], data["particles"][ids])
    fitted = model.coefficients(data["states"][ids], data["physics"][ids])
    covariance = _truth_covariance(data["sigma"][ids], model.incidence)
    return ((fitted.drift - data["drift"][ids]) / scales["drift"]).square().mean() + (
        (fitted.covariance - covariance) / scales["covariance"]).square().mean()


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load(path):
    # Only load checkpoints produced by trusted local experiments.
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_a(train: list[DKTrainingBatch], validation: list[DKTrainingBatch], spec: NativeDKSpec,
          mode="oracle", width=32, seed=0, config=None, output_path=None, deadline=None):
    """Fit native coefficients; a deadline stops only after a complete saved epoch.

    Epoch limits may be extended. Other configuration/data changes are rejected on
    resume; both the current optimizer state and the best validation model survive.
    """
    started = time.monotonic()
    if mode not in ("oracle", "increments"):
        raise ValueError("mode must be oracle or increments")
    if output_path is None:
        raise ValueError("A checkpoint output_path is required")
    config = dict(config or {})
    config.setdefault("epochs", 100)
    config.setdefault("batch_size", 128)
    config.setdefault("patience", 10)
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("threads", 1)
    for key in ("epochs", "batch_size", "patience", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Positive integer {key} required")
    for key in ("learning_rate", "weight_decay"):
        if not np.isfinite(config[key]) or config[key] < 0 or (key == "learning_rate" and config[key] == 0):
            raise ValueError(f"Invalid {key}")
    _validate_groups(train, validation, spec)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    model = DKCoefficientModel(spec, width)
    training = _prepare(train, spec, require_proposals=mode == "increments", oracle_labels=mode == "oracle")
    validating = _prepare(validation, spec, require_proposals=mode == "increments", oracle_labels=mode == "oracle")
    model.normalization.fit(training["physics"])
    scales = _training_scales(training, model) if mode == "oracle" else {"drift": 1., "covariance": 1.}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 37)
    provenance = dict(kind=COEFFICIENT_MODEL_KIND, spec=asdict(spec), mode=mode, width=width,
                      seed=seed, config=config, train_hash=_fingerprint(train),
                      validation_hash=_fingerprint(validation), objective_scales=scales,
                      normalization="active-training-physics-only",
                      state_features="mass-and-exact-fv-fourier-convolution-fields-v1",
                      coefficient_training_labels=mode == "oracle")
    identity = {**provenance, "config": {k: v for k, v in config.items() if k != "epochs"}}
    provenance["identity_hash"] = stable_hash(identity)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch, best_loss, patience_count, elapsed_prior, history = 0, float("inf"), 0, 0., []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if path.exists():
        saved = _load(path)
        if saved.get("kind") != COEFFICIENT_MODEL_KIND or saved.get("provenance", {}).get("identity_hash") != provenance["identity_hash"]:
            raise ValueError("Checkpoint is a different model, target, data, or configuration")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        epoch, best_loss = saved["epoch"], saved["best_loss"]
        patience_count, elapsed_prior = saved["patience_count"], saved["training_seconds"]
        history, best_state = saved["history"], saved["best_state"]
    complete = epoch >= config["epochs"] or patience_count >= config["patience"]

    def checkpoint():
        _atomic_save(dict(schema_version=1, kind=COEFFICIENT_MODEL_KIND, provenance=provenance,
                         model_state=model.state_dict(), best_state=best_state,
                         optimizer_state=optimizer.state_dict(), generator_state=generator.get_state(),
                         torch_rng_state=torch.get_rng_state(), epoch=epoch, best_loss=best_loss,
                         patience_count=patience_count, complete=complete, history=history,
                         training_seconds=elapsed_prior + time.monotonic() - started), path)

    checkpoint()
    while not complete and (deadline is None or time.monotonic() < deadline):
        model.train()
        permutation = torch.randperm(len(training["states"]), generator=generator)
        total, count = 0., 0
        for ids in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = _objective(model, training, ids, mode, scales)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native DK training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        model.eval()
        with torch.no_grad():
            losses = [float(_objective(model, validating, ids, mode, scales)) * len(ids)
                      for ids in torch.arange(len(validating["states"])).split(config["batch_size"])]
        validation_loss = sum(losses) / len(validating["states"])
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite native DK validation objective")
        epoch += 1
        if validation_loss < best_loss:
            best_loss, patience_count = validation_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        history.append(dict(epoch=epoch, training_loss=total / count, validation_loss=validation_loss))
        complete = epoch >= config["epochs"] or patience_count >= config["patience"]
        checkpoint()
    saved = _load(path)
    return dict(path=str(path), kind=COEFFICIENT_MODEL_KIND, mode=mode, width=width, seed=seed,
                complete=bool(saved["complete"]), epochs=epoch,
                best_validation_loss=best_loss if np.isfinite(best_loss) else None,
                training_seconds=saved["training_seconds"], training=training["metadata"],
                validation=validating["metadata"], training_examples=len(training["states"]),
                validation_examples=len(validating["states"]), identity_hash=provenance["identity_hash"],
                history=history)


def load_coefficient_model(path) -> DKCoefficientModel:
    checkpoint = _load(Path(path))
    if checkpoint.get("kind") != COEFFICIENT_MODEL_KIND:
        raise ValueError("Expected native autonomous DK coefficient checkpoint, not legacy transition")
    provenance = checkpoint["provenance"]
    if provenance.get("kind") != COEFFICIENT_MODEL_KIND:
        raise ValueError("Native coefficient checkpoint provenance kind mismatch")
    model = DKCoefficientModel(NativeDKSpec(**provenance["spec"]), provenance["width"])
    model.load_state_dict(checkpoint["best_state"])
    model.eval()
    return model


def coefficient_diagnostics(model, batches):
    """Held-out coefficient error and uncensored-proposal Gaussian calibration."""
    data = _prepare(batches, model.spec)
    drift_error, drift_norm, covariance_error, covariance_norm = 0., 0., 0., 0.
    white_count, white_sum, white_outer, nll_sum = 0, None, None, 0.
    model.eval()
    with torch.no_grad():
        for ids in torch.arange(len(data["states"])).split(256):
            fitted = model.coefficients(data["states"][ids], data["physics"][ids])
            truth = _truth_covariance(data["sigma"][ids], model.incidence)
            drift_error += float((fitted.drift - data["drift"][ids]).square().sum())
            drift_norm += float(data["drift"][ids].square().sum())
            covariance_error += float((fitted.covariance - truth).square().sum())
            covariance_norm += float(truth.square().sum())
            ids = ids[data["has_proposal"][ids]]
            if not len(ids):
                continue
            white, nll = model.proposal_residuals(data["states"][ids], data["proposals"][ids],
                                                 data["physics"][ids], data["dt"][ids],
                                                 data["particles"][ids])
            total, outer = white.sum(0), white.T @ white
            white_sum = total if white_sum is None else white_sum + total
            white_outer = outer if white_outer is None else white_outer + outer
            white_count += len(white)
            nll_sum += float(nll.sum())
    count, g = len(data["states"]), model.spec.n_cells
    metadata = dict(data["metadata"])
    metadata.update(drift_rmse=np.sqrt(drift_error / (count * g)),
                    covariance_rmse=np.sqrt(covariance_error / (count * g * g)),
                    normalized_drift_rmse=np.sqrt(drift_error / drift_norm) if drift_norm > 1e-28 else None,
                    normalized_covariance_rmse=np.sqrt(covariance_error / covariance_norm) if covariance_norm > 1e-28 else None,
                    normalization="held-out target RMS; null if zero target",
                    proposal_samples=white_count, validation_nll=nll_sum / white_count if white_count else None)
    if white_count:
        mean = white_sum / white_count
        covariance = white_outer / white_count - mean[:, None] * mean[None, :]
        metadata.update(whitened_mean=mean.tolist(), whitened_covariance=covariance.tolist(),
                        whitened_covariance_identity_rmse=float((covariance - torch.eye(g - 1)).square().mean().sqrt()))
    return metadata
