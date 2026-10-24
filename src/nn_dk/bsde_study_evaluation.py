"""Common held-out banks for the initial-state/architecture BSDE experiment.

Training allocations and architectures change checkpoint identities, not the
physical target or reference families.  A narrow adapter lets the established
evaluator reuse one fixed reference bank while preserving each fit's provenance.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bsde_evaluation import (
    _cache_chunk,
    _check_deadline,
    _load_chunk,
    _provenance,
    evaluate_bsde_model,
)
from .data import json_ready
from .native import initialize_masses
from .types import seeded_rng, stable_hash


@dataclass(frozen=True)
class StudyModelDescriptor:
    """Factors that identify one independently fitted model in the comparison."""

    architecture: str
    train_anchors: int
    train_continuations: int
    optimizer_seed: int

    def __post_init__(self):
        if not self.architecture or not isinstance(self.architecture, str):
            raise ValueError("A nonempty architecture identifier is required")
        if any(not isinstance(value, int) or value < 1
               for value in (self.train_anchors, self.train_continuations)):
            raise ValueError("Positive independent-anchor and continuation counts required")
        if not isinstance(self.optimizer_seed, int) or self.optimizer_seed < 0:
            raise ValueError("A nonnegative optimizer seed is required")

    @property
    def train_paths(self):
        return self.train_anchors * self.train_continuations

    def to_dict(self):
        return {"architecture": self.architecture, "train_anchors": self.train_anchors,
                "train_continuations": self.train_continuations,
                "train_paths": self.train_paths, "optimizer_seed": self.optimizer_seed}


class _CommonBankModel:
    """Expose test-bank configuration without changing the fitted value map."""

    def __init__(self, model, config):
        self._model, self.config = model, config

    def __getattr__(self, name):
        return getattr(self._model, name)

    @property
    def target_id(self):
        return self.config.target_id(self.problem)

    def predict_conditional(self, initial, horizons):
        return self._model.predict_conditional(initial, horizons)


def test_bank_identity(config, problem):
    """Stable identity including every configured held-out/reference allocation."""
    if problem not in ("ou", "dk"):
        raise ValueError("The follow-up study supports only OU and native DK")
    return stable_hash({"kind": "bsde-followup-common-test-bank-v1", "problem": problem,
                        "target_id": config.target_id(problem), "seed": config.seed,
                        "cases": config.cases, "ou_anchors": config.ou_test_anchors,
                        "conditional_anchors": config.conditional_anchors,
                        "conditional_children": config.conditional_paths,
                        "ensemble_initial_states": config.outer_paths,
                        "reference_dt": config.reference_dt,
                        "reference_batch_size": config.reference_batch_size})


# This is a production function, rather than a pytest test imported by clients.
test_bank_identity.__test__ = False


def evaluate_study_model(directory, config, model, descriptor, deadline=None):
    """Evaluate one fit on the shared fresh bank, retaining training provenance.

    ``config`` describes the common evaluation bank.  The model may have a
    different allocation or architecture configuration, but its complete
    physical target, forward step and query bank must match.  References are
    cached once under ``directory`` and reused by every fit, irrespective of
    optimizer seed or allocation.  The immutable legacy evaluation record is
    returned separately from the factor labels so it cannot be overwritten by
    a different fit's metadata.
    """
    if not isinstance(descriptor, StudyModelDescriptor):
        raise TypeError("A StudyModelDescriptor is required")
    if model.problem not in ("ou", "dk"):
        raise ValueError("The follow-up study supports only OU and native DK")
    target = config.target_id(model.problem)
    if model.target_id != target or model.config.target_id(model.problem) != target:
        raise ValueError("Study model and common bank have incompatible physical targets")
    allocation = getattr(model, "study_allocation", None)
    if allocation is not None and tuple(allocation) != (descriptor.train_anchors,
                                                       descriptor.train_continuations):
        raise ValueError("Study allocation does not match the fitted model configuration")
    architecture = getattr(model, "architecture", "mlp")
    if architecture != descriptor.architecture:
        raise ValueError("Study architecture does not match the fitted model configuration")
    checkpoint = getattr(model, "checkpoint_path", None)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise ValueError("A saved study checkpoint is required for evaluation")
    recorded_seed = getattr(model, "optimizer_seed", descriptor.optimizer_seed)
    if recorded_seed != descriptor.optimizer_seed:
        raise ValueError("Study optimizer seed does not match the fitted model")
    checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    bank = test_bank_identity(config, model.problem)
    factors = descriptor.to_dict()
    identity = stable_hash({"bank": bank, "factors": factors,
                            "checkpoint_sha256": checkpoint_sha256})
    evaluation = evaluate_bsde_model(directory, config, _CommonBankModel(model, config),
                                    descriptor.optimizer_seed, deadline=deadline)
    record = {"study_model_id": identity, "test_bank_id": bank,
              "factors": factors, "fit_config_hash": model.config.config_hash,
              "evaluation_config_hash": config.config_hash,
              "checkpoint_sha256": checkpoint_sha256,
              "training_view_fingerprint": getattr(model, "provenance", {}).get("training_fingerprint"),
              "validation_fingerprint": getattr(model, "provenance", {}).get("validation_fingerprint"),
              "evaluation": evaluation}
    if model.problem == "dk":
        record["prediction_caches"] = _store_ensemble_predictions(
            directory, config, model, identity)
    return json_ready(record)


def _read_reference_samples(directory, config, case, kind):
    """Read a complete common bank; this function never simulates missing paths."""
    conditional = kind == "conditional"
    anchors = config.conditional_anchors if conditional else config.outer_paths
    children = config.conditional_paths if conditional else 1
    total = anchors * children
    initial_seed = int(seeded_rng(config.seed, "bsde-test-initial", case, kind).integers(0, 2**63-1))
    root = Path(directory) / "evaluation_cache" / case / kind
    chunks = []
    for start in range(0, total, config.reference_batch_size):
        end = min(total, start + config.reference_batch_size)
        path = root / f"chunk_{start:08d}.npz"
        if not path.is_file():
            raise ValueError("Paired study contrasts require the complete saved common reference bank")
        expected = {"case": case, "kind": kind, "start": start, "end": end,
                    "target_id": config.spec.target_id, "dt": config.dt,
                    "reference_dt": config.reference_dt, "initial_seed": initial_seed}
        arrays, metadata = _load_chunk(path, config, expected)
        if not metadata.get("all_stopped_paths_included"):
            raise ValueError("Paired contrasts must include every stopped reference path")
        chunks.append(arrays)
    values = {key: np.concatenate([chunk[key] for chunk in chunks])
              for key in ("coarse", "fine", "family_ids", "coarse_stopped_at", "fine_stopped_at")}
    if len(set(values["family_ids"].tolist())) != total:
        raise ValueError("Duplicate reference continuation families in the study bank")
    if conditional:
        values["coarse"] = values["coarse"].reshape(anchors, children, len(config.query_times), 4)
        values["fine"] = values["fine"].reshape(anchors, children, len(config.query_times), 4)
    values["initial_seed"] = initial_seed
    return values


def _store_ensemble_predictions(directory, config, model, identity):
    """Persist value predictions needed by a paired initial-family bootstrap."""
    root, records = Path(directory), {}
    for case in config.cases:
        relative = Path("study_prediction_cache") / identity / f"{case}_ensemble.npz"
        path = root / relative
        expected = {"study_model_id": identity, "case": case, "kind": "ensemble",
                    "target_id": model.target_id}
        if path.is_file():
            _, metadata = _load_chunk(path, config, expected)
        else:
            initial_path = root / "study_initial_cache" / f"{case}_ensemble.npz"
            initial_expected = {"case": case, "kind": "ensemble", "n_initial": config.outer_paths,
                                "target_id": model.target_id}
            if initial_path.is_file():
                initial_arrays, initial_metadata = _load_chunk(initial_path, config, initial_expected)
                initial = initial_arrays["initial"]
            else:
                # Missing references must never be silently regenerated here.
                reference = _read_reference_samples(root, config, case, "ensemble")
                tick = time.perf_counter()
                initial = initialize_masses(config.context(case), config.spec, config.outer_paths,
                                            reference["initial_seed"])
                initial_metadata = {**_provenance(config), **initial_expected,
                                    "initial_seed": reference["initial_seed"],
                                    "seconds": time.perf_counter()-tick}
                _cache_chunk(initial_path, {"initial": initial}, initial_metadata)
            tick = time.perf_counter()
            prediction = model.predict_conditional(initial, config.query_times)
            mean = np.asarray(prediction["mean"] if isinstance(prediction, dict) else prediction.mean)
            covariance = np.asarray(prediction["scaled_covariance"] if isinstance(prediction, dict)
                                    else prediction.scaled_covariance)
            if (mean.shape != (config.outer_paths, len(config.query_times), 4)
                    or covariance.shape != (*mean.shape, 4)
                    or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))):
                raise ValueError("Invalid value predictions for paired ensemble-family bootstrap")
            metadata = {**_provenance(config), **expected,
                        "seconds": time.perf_counter()-tick,
                        "shared_initialization_seconds": initial_metadata["seconds"],
                        "cost_scope": "one additional value-only call; initial bank shared across all fits; no forward paths"}
            _cache_chunk(path, {"mean": mean, "scaled_covariance": covariance}, metadata)
        records[case] = {"path": str(relative), "seconds": metadata["seconds"],
                         "shared_initialization_seconds": metadata["shared_initialization_seconds"],
                         "cost_scope": metadata["cost_scope"]}
    return records


def study_metric_rows(records):
    """Flat per-fit metrics, without averaging independent optimizer seeds."""
    rows, seen = [], set()
    for record in records:
        identity, evaluation = record["study_model_id"], record["evaluation"]
        if identity in seen:
            raise ValueError("Duplicate study model evaluation")
        seen.add(identity)
        if not evaluation.get("complete"):
            raise ValueError("Incomplete model evaluation cannot enter the study table")
        common = {**record["factors"], "problem": evaluation["problem"],
                  "study_model_id": identity, "test_bank_id": record["test_bank_id"]}
        if evaluation["problem"] == "ou":
            comparisons = (("analytic", "conditional", evaluation["neural_error"]),)
        elif evaluation["problem"] == "dk":
            comparisons = [(case, scope, result[f"{scope}_neural_error"])
                           for case, result in evaluation["cases"].items()
                           for scope in ("conditional", "ensemble")]
        else:
            raise ValueError("Unknown study evaluation problem")
        for case, scope, comparison in comparisons:
            error = comparison["errors"]
            rows.append({**common, "case": case, "scope": scope,
                         "mean_rmse": error["mean_normalized_rmse"],
                         "covariance_rmse": error["scaled_covariance_normalized_rmse"],
                         "variance_rmse": error["scaled_variance_normalized_rmse"],
                         "maximum_mean_error": error["maximum_mean_normalized_error"],
                         "maximum_covariance_error": error["maximum_scaled_covariance_normalized_error"],
                         "accuracy_gate": comparison["accuracy_gate"]["status"]})
    return rows


def paired_metric_contrasts(records):
    """Descriptive paired contrasts for data and architecture, one seed at a time.

    A negative ``candidate_minus_baseline`` means a smaller error on the same
    test bank.  These are paired point estimates; no significance or resolved
    accuracy claim is inferred from a sign.  Continuation noise is common to
    both models but is not independent across comparisons.
    """
    rows = study_metric_rows(records)
    contrasts = []
    endpoints = {}
    for row in rows:
        key = (row["problem"], row["case"], row["scope"], row["optimizer_seed"],
               row["test_bank_id"], row["architecture"], row["train_paths"])
        endpoints.setdefault(key, []).append(row["train_anchors"])
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            baseline, candidate = first, second
            match = ("problem", "case", "scope", "optimizer_seed", "test_bank_id")
            if any(baseline[key] != candidate[key] for key in match):
                continue
            allocation_a = (baseline["train_anchors"], baseline["train_continuations"])
            allocation_b = (candidate["train_anchors"], candidate["train_continuations"])
            if baseline["train_paths"] != candidate["train_paths"]:
                continue
            if baseline["architecture"] == candidate["architecture"] and allocation_a != allocation_b:
                factor = "independent_initial_states"
                if baseline["train_anchors"] > candidate["train_anchors"]:
                    baseline, candidate = candidate, baseline
                key = (baseline["problem"], baseline["case"], baseline["scope"],
                       baseline["optimizer_seed"], baseline["test_bank_id"],
                       baseline["architecture"], baseline["train_paths"])
                if ((baseline["train_anchors"], candidate["train_anchors"])
                        != (min(endpoints[key]), max(endpoints[key]))):
                    continue
            elif allocation_a == allocation_b and baseline["architecture"] != candidate["architecture"]:
                factor = "architecture"
                if candidate["architecture"] == "mlp":
                    baseline, candidate = candidate, baseline
            else:
                continue
            metrics = ("mean_rmse", "covariance_rmse", "variance_rmse")
            contrasts.append({**{key: baseline[key] for key in match}, "factor": factor,
                              "baseline_model_id": baseline["study_model_id"],
                              "candidate_model_id": candidate["study_model_id"],
                              "baseline_architecture": baseline["architecture"],
                              "candidate_architecture": candidate["architecture"],
                              "baseline_train_anchors": baseline["train_anchors"],
                              "candidate_train_anchors": candidate["train_anchors"],
                              "fixed_train_paths": baseline["train_paths"],
                              "candidate_minus_baseline": {
                                  key: float(candidate[key] - baseline[key]) for key in metrics},
                              "uncertainty": "paired point estimate only; optimizer seeds remain separate"})
    if not all(np.isfinite(value) for row in contrasts
               for value in row["candidate_minus_baseline"].values()):
        raise ValueError("Nonfinite study error contrast")
    return contrasts


def _normalized_sample_moments(samples):
    """Moments over the third-last (whole-trajectory) axis in fluctuation units."""
    count = samples.shape[-3]
    mean = samples.mean(axis=-3)
    centered = samples - np.expand_dims(mean, axis=-3)
    covariance = np.einsum("...phi,...phj->...hij", centered, centered)/(count-1)
    return mean, covariance


def _score_draws(mean_error, covariance_error):
    # First two axes are model and bootstrap draw.  All remaining indices belong
    # to a fixed bank's anchors/horizons/modes, and enter the declared RMSE.
    mean_axes = tuple(range(2, mean_error.ndim))
    covariance_axes = tuple(range(2, covariance_error.ndim))
    diagonal = np.diagonal(covariance_error, axis1=-2, axis2=-1)
    return {"mean_rmse": np.sqrt(np.mean(mean_error**2, axis=mean_axes)),
            "covariance_rmse": np.sqrt(np.mean(covariance_error**2, axis=covariance_axes)),
            "variance_rmse": np.sqrt(np.mean(diagonal**2, axis=tuple(range(2, diagonal.ndim))))}


def _conditional_score_bootstrap(records, reference, config, case, draws, deadline):
    std = np.asarray(config.observable_scale)
    samples = np.sqrt(config.n_particles)*reference["coarse"]/std
    means, covariances = [], []
    for record in records:
        comparison = record["evaluation"]["cases"][case]["conditional_neural_error"]
        error = comparison["errors"]
        means.append(np.asarray(error["mean_normalized_signed"])
                     + np.sqrt(config.n_particles)*np.asarray(comparison["reference_mean"])/std)
        covariances.append(np.asarray(error["scaled_covariance_normalized_signed"])
                           + np.asarray(comparison["reference_scaled_covariance"])/np.outer(std, std))
    means, covariances = np.asarray(means), np.asarray(covariances)
    rng = seeded_rng(config.seed, "bsde-study-paired-bootstrap", case, "conditional")
    anchors, children = samples.shape[:2]
    reference_means, reference_covariances = [], []
    for _ in range(draws):
        _check_deadline(deadline)
        # Every anchor remains fixed; its independent children are resampled as
        # whole trajectories, preserving all horizons and mode dependence.
        indices = rng.integers(0, children, size=(anchors, children))
        sample = samples[np.arange(anchors)[:, None], indices]
        mean, covariance = _normalized_sample_moments(sample)
        reference_means.append(mean)
        reference_covariances.append(covariance)
    scores = _score_draws(means[:, None]-np.asarray(reference_means)[None],
                          covariances[:, None]-np.asarray(reference_covariances)[None])
    return scores


def _weighted_moments(weights, samples):
    """Bootstrap moments from multinomial family weights, without copying paths."""
    count, horizons, dimension = samples.shape
    mean = (weights @ samples.reshape(count, -1)).reshape(len(weights), horizons, dimension)/count
    outer = samples[..., :, None]*samples[..., None, :]
    second = (weights @ outer.reshape(count, -1)).reshape(len(weights), horizons, dimension, dimension)
    covariance = (second-count*mean[..., :, None]*mean[..., None, :])/(count-1)
    return mean, covariance


def _ensemble_score_bootstrap(directory, records, reference, config, case, draws, deadline):
    std = np.asarray(config.observable_scale)
    samples = np.sqrt(config.n_particles)*reference["coarse"]/std
    count = len(samples)
    rng = seeded_rng(config.seed, "bsde-study-paired-bootstrap", case, "ensemble")
    # One matrix of matched whole-family bootstrap weights is reused by every
    # fit and by the physical reference; covariance includes initial randomness.
    weights = np.asarray([np.bincount(rng.integers(0, count, count), minlength=count)
                          for _ in range(draws)], dtype=float)
    reference_mean, reference_covariance = _weighted_moments(weights, samples)
    means, covariances = [], []
    for record in records:
        _check_deadline(deadline)
        metadata = record["prediction_caches"][case]
        expected = {"study_model_id": record["study_model_id"], "case": case, "kind": "ensemble",
                    "target_id": config.target_id("dk")}
        arrays, _ = _load_chunk(Path(directory)/metadata["path"], config, expected)
        mean = np.sqrt(config.n_particles)*arrays["mean"]/std
        covariance = arrays["scaled_covariance"]/np.outer(std, std)
        if mean.shape != samples.shape or covariance.shape != (*samples.shape, 4):
            raise ValueError("Study prediction bank does not match paired reference families")
        boot_mean, between = _weighted_moments(weights, mean)
        within = (weights @ covariance.reshape(count, -1)).reshape(draws, *covariance.shape[1:])/count
        means.append(boot_mean)
        covariances.append(within+between)
    return _score_draws(np.asarray(means)-reference_mean[None],
                        np.asarray(covariances)-reference_covariance[None])


def paired_study_contrasts(records, directory, config, n_bootstrap=None, deadline=None):
    """Paired 95% RMSE-improvement intervals using only saved common references.

    Compares the smallest/largest independent-anchor allocations at a fixed
    architecture and structured versus MLP at each fixed allocation, separately
    for each optimizer seed.  All models use identical bootstrap reference draws.
    Conditional intervals concern the FIXED diagnostic anchor bank: children are
    resampled independently within each anchor.  Ensemble intervals additionally
    resample initial-histogram/forcing families.  Neither interval certifies a
    population-wide conditional generalization error or residual reference bias.
    """
    draws = config.bootstrap_samples if n_bootstrap is None else n_bootstrap
    if type(draws) is not int or draws < 2:
        raise ValueError("At least two paired bootstrap draws are required")
    contrasts = paired_metric_contrasts(records)
    records_by_id = {record["study_model_id"]: record for record in records}
    score_cache = {}
    for problem in dict.fromkeys(row["problem"] for row in contrasts):
        relevant = [record for record in records if record["evaluation"]["problem"] == problem]
        expected_bank = test_bank_identity(config, problem)
        if any(record["test_bank_id"] != expected_bank
               or record["evaluation_config_hash"] != config.config_hash for record in relevant):
            raise ValueError("Paired study contrast does not match the declared common test bank")
        if problem == "ou":
            continue
        for case in config.cases:
            for scope in ("conditional", "ensemble"):
                _check_deadline(deadline)
                reference = _read_reference_samples(directory, config, case, scope)
                if scope == "conditional":
                    scores = _conditional_score_bootstrap(relevant, reference, config, case, draws, deadline)
                else:
                    scores = _ensemble_score_bootstrap(directory, relevant, reference, config, case,
                                                       draws, deadline)
                for index, record in enumerate(relevant):
                    key = (record["study_model_id"], case, scope)
                    score_cache[key] = {metric: values[index] for metric, values in scores.items()}
    for contrast in contrasts:
        metrics = contrast["candidate_minus_baseline"]
        if contrast["problem"] == "ou":
            contrast["paired_uncertainty"] = {
                "method": "analytic reference on fixed OU initial-state bank",
                "reference_sampling_error": 0.,
                "candidate_minus_baseline_ci95": {metric: [value, value] for metric, value in metrics.items()},
                "anchor_generalization_uncertainty_included": False}
        else:
            case, scope = contrast["case"], contrast["scope"]
            candidate = score_cache[(contrast["candidate_model_id"], case, scope)]
            baseline = score_cache[(contrast["baseline_model_id"], case, scope)]
            intervals = {metric: np.quantile(candidate[metric]-baseline[metric], (.025, .975))
                         for metric in metrics}
            contrast["paired_uncertainty"] = {
                "method": "paired whole continuation trajectories within each fixed anchor"
                if scope == "conditional" else "paired whole initial-histogram and forcing families",
                "confidence_level": .95, "n_bootstrap": draws,
                "candidate_minus_baseline_ci95": intervals,
                "anchor_generalization_uncertainty_included": scope == "ensemble",
                "optimizer_uncertainty_included": False, "reference_bias_certified": False}
            # A shared reference has an exact bank identity even when a previous
            # model fit used a different architecture/allocation configuration.
            assert records_by_id[contrast["baseline_model_id"]]["test_bank_id"] == contrast["test_bank_id"]
        contrast["reference_level"] = "training-step Euler moments"
        contrast["uncertainty"] = "paired intervals on the declared bank; optimizer seeds remain separate"
    return json_ready(contrasts)
