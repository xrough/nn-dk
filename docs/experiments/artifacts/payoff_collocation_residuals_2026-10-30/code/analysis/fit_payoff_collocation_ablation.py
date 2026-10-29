"""Controlled proposal/loss ablations of the fixed v3 generator formulation.

This engine does not modify historical trainers or global Torch bindings.
Adam proposals carry p/q outside the per-row loss; full-pool LBFGS uses natural
row weights. Every variant selects on the same natural validation MSE. A
completed allocation is software completion, not a prediction certificate.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analysis.payoff_collocation_sampling import (
    BoundaryStrataConfig,
    CollocationSampler,
)
from nn_dk.data import source_hash
from nn_dk.payoff_boundary import BoundaryPayoffModel, BoundaryRepairConfig, _scaling
from nn_dk.payoff_learning import PayoffBank
from nn_dk.types import stable_hash

KIND = "matched-pool-importance-corrected-staged-payoff-ablation-v1"
BANK_KIND = "flat-parent-grouped-generator-collocation-v1"
STAGES = ("mean", "covariance")
OPTIMIZERS = ("adam", "lbfgs")


def _digest(value):
    """Hash tensors and optimizer trees without executable pickle comparison."""
    def encode(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            return ["tensor", str(item.dtype), list(item.shape), hashlib.sha256(array.tobytes()).hexdigest()]
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return ["array", str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
        if isinstance(item, dict):
            return ["dict", [[encode(key), encode(item[key])] for key in sorted(
                item, key=lambda key: (type(key).__name__, repr(key)))]]
        if isinstance(item, (tuple, list)):
            return [type(item).__name__, [encode(value) for value in item]]
        if item is None or type(item) in (str, bool, int, float):
            return [type(item).__name__, item]
        raise TypeError(f"Unsupported checkpoint value: {type(item).__name__}")
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources():
    return {"package_source_hash": source_hash(), "helper_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("payoff_collocation_sampling.py"))}}


@dataclass(frozen=True)
class CollocationAblationConfig:
    repair: BoundaryRepairConfig
    proposal: str = "uniform"
    loss: str = "pseudo_huber"
    strata: BoundaryStrataConfig | None = None

    def __post_init__(self):
        if not isinstance(self.repair, BoundaryRepairConfig):
            raise TypeError("A fixed BoundaryRepairConfig is required")
        if self.proposal not in ("uniform", "stratified") or self.loss not in ("pseudo_huber", "squared"):
            raise ValueError("Declare a uniform/stratified proposal and pseudo_huber/squared loss")
        study = self.repair.study
        if self.strata is None:
            object.__setattr__(self, "strata", BoundaryStrataConfig(
                study.horizon, study.spec.mass_stop, .05/study.n_cells))
        if (not isinstance(self.strata, BoundaryStrataConfig)
                or self.strata.horizon != study.horizon
                or self.strata.mass_stop != study.spec.mass_stop
                or self.strata.boundary_width != .05/study.n_cells):
            raise ValueError("Strata must use the declared model's horizon, stop and boundary width")

    def to_dict(self):
        return {"repair": self.repair.to_dict(), "proposal": self.proposal,
                "loss": self.loss, "strata": self.strata.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete collocation ablation configuration required")
        return cls(**(value | {"repair": BoundaryRepairConfig.from_dict(value["repair"]),
                              "strata": BoundaryStrataConfig.from_dict(value["strata"])}))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())


def make_flat_bank(states, horizons, family_ids, *, role, config, parent_family_ids=None,
                   row_kinds=None, provenance):
    """Bind an explicitly declared native/stress pool without future labels.

    Provenance requires stream_namespace, parent_pool_sha256 and the explicit
    stress_augmentation flag. Family IDs may repeat; independent units are the
    parent_family_ids. Native-only validation remains the selection population.
    """
    if not isinstance(config, CollocationAblationConfig) or role not in ("train", "validation"):
        raise ValueError("Declared ablation configuration and train/validation role required")
    states = np.array(states, dtype=np.float64, copy=True)
    horizons = np.array(horizons, dtype=np.float64, copy=True)
    if states.ndim != 2 or states.shape[1] != config.repair.study.n_cells:
        raise ValueError("Flat bank must use the declared native grid")
    families = np.asarray(family_ids)
    parents = families.copy() if parent_family_ids is None else np.asarray(parent_family_ids)
    kinds = (np.full(len(states), "native") if row_kinds is None else np.asarray(row_kinds, dtype=str))
    # The proposal constructor independently validates finite simplex rows and
    # remaining horizons, and freezes exact ordered labels and row content.
    sampler = CollocationSampler(states, horizons, families, config.strata, mode="uniform", seed=0)
    if (parents.shape != families.shape or any(not isinstance(name, (str, bytes)) or not name
                                              for name in parents.tolist())
            or kinds.shape != families.shape or not set(kinds.tolist()) <= {"native", "stress"}
            or (role == "validation" and np.any(kinds != "native"))):
        raise ValueError("Valid parent groups and native-only natural validation rows required")
    families = np.asarray([name.decode() if isinstance(name, bytes) else name for name in families.tolist()])
    parents = np.asarray([name.decode() if isinstance(name, bytes) else name for name in parents.tolist()])
    if not isinstance(provenance, dict) or set(provenance) != {
            "stream_namespace", "parent_pool_sha256", "stress_augmentation"}:
        raise ValueError("Explicit stream and parent-pool provenance required")
    if (not isinstance(provenance["stream_namespace"], str) or not provenance["stream_namespace"]
            or not isinstance(provenance["parent_pool_sha256"], str)
            or len(provenance["parent_pool_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in provenance["parent_pool_sha256"])
            or type(provenance["stress_augmentation"]) is not bool
            or provenance["stress_augmentation"] != bool(np.any(kinds == "stress"))):
        raise ValueError("Stream/hash or explicit stress augmentation declaration is invalid")
    result = {"kind": BANK_KIND, "role": role, "states": states, "horizons": horizons,
        "family_ids": families.tolist(), "parent_family_ids": parents.tolist(),
        "row_kinds": kinds.tolist(), "provenance": copy.deepcopy(provenance),
        "future_payoff_labels": False, "source_hash": source_hash(),
        "study_config_hash": config.repair.study.config_hash,
        "row_identity": {key: sampler.identity[key] for key in ("states", "horizons", "families_sha256")}}
    result["content_sha256"] = _digest(result)
    return result


def _bank(value, config, role):
    keys = {"kind", "role", "states", "horizons", "family_ids", "parent_family_ids", "row_kinds",
            "provenance", "future_payoff_labels", "source_hash", "study_config_hash",
            "row_identity", "content_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or value["kind"] != BANK_KIND
            or value["role"] != role or value["future_payoff_labels"] is not False
            or value["source_hash"] != source_hash()
            or value["study_config_hash"] != config.repair.study.config_hash):
        raise ValueError("Exact current-source flat-bank schema required; future labels are forbidden")
    rebuilt = make_flat_bank(value["states"], value["horizons"], value["family_ids"],
        role=role, config=config, parent_family_ids=value["parent_family_ids"],
        row_kinds=value["row_kinds"], provenance=value["provenance"])
    if rebuilt["content_sha256"] != value["content_sha256"] or rebuilt["row_identity"] != value["row_identity"]:
        raise ValueError("Flat collocation pool content checksum changed")
    fingerprint = {"kind": BANK_KIND, "role": role, "content_sha256": value["content_sha256"],
        "rows": len(value["states"]), "parent_families": len(set(value["parent_family_ids"])),
        "parent_family_sha256": _digest(value["parent_family_ids"]),
        "row_counts": {kind: value["row_kinds"].count(kind) for kind in ("native", "stress")},
        "provenance": copy.deepcopy(value["provenance"])}
    return {"states": torch.tensor(rebuilt["states"], dtype=torch.float64),
            "horizons": torch.tensor(rebuilt["horizons"], dtype=torch.float64)}, fingerprint


def row_objective(residual, scale, horizon, *, loss, delta):
    """Fixed-unit per-row losses; importance weights must multiply the result."""
    scale = torch.as_tensor(scale, dtype=residual.dtype, device=residual.device)
    if (residual.ndim not in (2, 3) or not len(residual) or scale.shape != residual.shape[1:]
            or not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any()) or not math.isfinite(horizon) or horizon <= 0
            or loss not in ("pseudo_huber", "squared") or not math.isfinite(delta) or delta <= 0):
        raise ValueError("Compatible finite residuals, fixed positive scales and a declared loss required")
    normalized = horizon*residual/scale
    values = (normalized.square() if loss == "squared" else
              2*normalized.square()/(torch.sqrt(1+(normalized/delta).square())+1))
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite normalized per-row residual loss")
    return values.flatten(1).mean(-1)


def _loss(model, states, horizons, scaling, stage, config, weights=None):
    residual = getattr(model, f"{stage}_residual")(states, horizons)
    rows = row_objective(residual, scaling[stage+"_scale"], model.horizon,
                         loss=config.loss, delta=config.repair.pseudo_huber_delta)
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=rows.dtype, device=rows.device)
        if weights.shape != rows.shape or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Positive literal p/q for every proposed row required")
        rows = weights*rows
    result = rows.mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Nonfinite collocation ablation loss")
    return result


def natural_validation(model, bank, scaling, stage, config):
    """All variants select by literal natural-occupancy fixed-unit MSE."""
    result = {"natural_mse": 0., "natural_huber": 0., "rows": len(bank["states"])}
    model.eval()
    with torch.no_grad():
        for start in range(0, result["rows"], config.repair.study.batch_size):
            end = min(result["rows"], start+config.repair.study.batch_size)
            residual = getattr(model, f"{stage}_residual")(
                bank["states"][start:end], bank["horizons"][start:end])
            for name, loss in (("natural_mse", "squared"), ("natural_huber", "pseudo_huber")):
                rows = row_objective(residual, scaling[stage+"_scale"], model.horizon,
                                     loss=loss, delta=config.repair.pseudo_huber_delta)
                result[name] += float(rows.sum())/result["rows"]
    return result


def _model(config, bank, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        study = config.repair.study
        return BoundaryPayoffModel(study.spec, study.physics, study.horizon, bank,
                                   study.width, config.repair.quadrature_nodes)


def _weights(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _selected_mean(state):
    return {key: value for key, value in state.items() if key.startswith("mean_network.")}


def _done(steps, config, stage):
    return all(steps[stage+"_"+opt] == getattr(config.repair, stage+"_"+opt+"_steps") for opt in OPTIMIZERS)


def _read(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "payload_sha256" not in saved:
        raise ValueError("Checksummed collocation checkpoint required")
    if saved["payload_sha256"] != _digest({k: v for k, v in saved.items() if k != "payload_sha256"}):
        raise ValueError("Collocation checkpoint payload checksum changed")
    _verify(saved)
    return saved


def _finite_optimizer_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_optimizer_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_optimizer_tree(item) for item in value)
    return type(value) is not float or math.isfinite(value)


def _verify(saved):
    if saved.get("schema_version") != 1 or saved.get("kind") != KIND:
        raise ValueError("Distinct collocation ablation checkpoint schema required")
    identity = saved["identity"]
    config = CollocationAblationConfig.from_dict(identity["config"])
    if identity["sources"] != _sources() or identity["selection"] != "natural-validation-fixed-unit-MSE":
        raise ValueError("Checkpoint source, helper or validation-selection identity changed")
    bank = PayoffBank.from_dict(identity["bank"])
    if (identity["config_hash"] != config.config_hash or identity["bank_hash"] != bank.bank_hash
            or type(identity["seed"]) is not int or identity["seed"] not in config.repair.study.optimizer_seeds
            or identity["future_payoff_labels"] is not False):
        raise ValueError("Checkpoint configuration hash changed")
    template = _model(config, bank, identity["seed"])
    if (identity["target_id"] != template.target_id
            or identity["initial_state_sha256"] != _digest(saved["initial_state"])
            or _digest(saved["initial_state"]) != _digest(template.state_dict())):
        raise ValueError("Checkpoint initial weights or physical buffers changed")
    keys = {stage+"_"+opt for stage in STAGES for opt in OPTIMIZERS}
    steps = saved["steps_done"]
    if set(steps) != keys or any(type(steps[key]) is not int or not 0 <= steps[key] <=
            getattr(config.repair, key+"_steps") for key in keys):
        raise ValueError("Checkpoint optimizer allocation changed")
    complete = all(_done(steps, config, stage) for stage in STAGES)
    if (type(saved["complete"]) is not bool or saved["complete"] != complete
            or saved["stage"] not in STAGES
            or (saved["stage"] == "mean" and any(steps["covariance_"+opt] for opt in OPTIMIZERS))
            or (saved["stage"] == "covariance" and not _done(steps, config, "mean"))
            or (complete and saved["stage"] != "covariance")):
        raise ValueError("Checkpoint completion/stage contradicts configured optimizer steps")
    for name in ("model_state", "best_mean_state", "best_state"):
        weights = saved[name]
        if set(weights) != set(saved["initial_state"]):
            raise ValueError("Checkpoint weight inventory changed")
        for key, value in weights.items():
            initial = saved["initial_state"][key]
            if value.shape != initial.shape or value.dtype != initial.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError("Checkpoint weights are incompatible/nonfinite")
            if not key.startswith(("mean_network.", "covariance_network.")) and not torch.equal(value, initial):
                raise ValueError("Checkpoint physical buffers changed")
    if saved["stage"] == "covariance" and any(_digest(_selected_mean(saved[name])) !=
            _digest(_selected_mean(saved["best_mean_state"])) for name in ("model_state", "best_state")):
        raise ValueError("Covariance stage changed the frozen selected mean")
    expected_draws = config.repair.study.batch_size*sum(steps[stage+"_adam"] for stage in STAGES)
    if saved["sampler_state"]["draws"] != expected_draws:
        raise ValueError("Sampler draw count contradicts completed Adam updates")
    if set(saved["optimizer_states"]) != keys or len(saved["history"]) != sum(steps.values()):
        raise ValueError("Checkpoint optimizer/history inventory changed")
    if not _finite_optimizer_tree(saved["optimizer_states"]):
        raise ValueError("Nonfinite checkpoint optimizer state")
    rng = saved["torch_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.get_rng_state().shape:
        raise ValueError("Checkpoint Torch RNG state is incompatible")
    scaling = saved["loss_scaling"]
    if (scaling.get("source_role") != "train" or scaling.get("inverse_time_weighting") is not False
            or scaling.get("normalization_time_powers") != [0]*4
            or scaling.get("pseudo_huber_delta") != config.repair.pseudo_huber_delta
            or scaling.get("training_fingerprint_hash") != stable_hash(identity["data"]["train"])):
        raise ValueError("Training-only fixed-unit loss scaling changed")
    for stage, shape in (("mean", (4,)), ("covariance", (4, 4))):
        scale = np.asarray(scaling[stage+"_scale"])
        if scale.shape != shape or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Finite positive loss scales required")
    return config, bank


@contextmanager
def _exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this ablation checkpoint") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _save(path, payload):
    payload = copy.deepcopy(payload)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _digest(payload)
    _verify(payload)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _metadata(saved, path):
    return {"path": str(path.resolve()), "kind": KIND, "complete": saved["complete"],
        "method_id": _digest(saved["identity"]), "identity": saved["identity"],
        "stage": saved["stage"], "steps_done": saved["steps_done"],
        "best_validation": saved["best_validation"], "best_step": saved["best_step"],
        "training_seconds": saved["training_seconds"], "stage_seconds": saved["stage_seconds"],
        "timing_scope": {
            "training_seconds": "monotonic loop elapsed before checkpoint save; includes prior saves",
            "stage_seconds": "optimizer and periodic validation; excludes checkpoint writes",
            "excluded": "pre-loop integrity/model/scales/initial-validation and final save; measure full invocation separately"},
        "interpretation": "Optimization allocation completion is not statistical certification"}


def fit_collocation_ablation(config, bank, seed, path, train, validation, *, deadline=None, max_updates=None):
    """Fit a declared proposal/loss pair, then strictly resume or return it.

    max_updates is an invocation checkpoint budget, not a changed scientific
    allocation. All covariance updates use the selected immutable mean. Caller
    Torch RNG state is preserved; row proposals use a private checkpointed RNG.
    """
    if (not isinstance(config, CollocationAblationConfig) or not isinstance(bank, PayoffBank)
            or type(seed) is not int or seed not in config.repair.study.optimizer_seeds
            or (max_updates is not None and (type(max_updates) is not int or max_updates < 0))
            or (deadline is not None and not math.isfinite(deadline))):
        raise ValueError("Declared configuration, bank, optimizer seed and valid invocation budget required")
    training, tf = _bank(train, config, "train")
    valid, vf = _bank(validation, config, "validation")
    if (set(train["parent_family_ids"]) & set(validation["parent_family_ids"])
            or set(train["family_ids"]) & set(validation["family_ids"])):
        raise ValueError("Training and natural-validation parent families must be disjoint")
    sampler = CollocationSampler(train["states"], train["horizons"], train["parent_family_ids"],
                                  config.strata, mode=config.proposal, seed=0x434F4C4C+seed)
    path = Path(path)
    with _exclusive(path), torch.random.fork_rng():
        torch.manual_seed(seed)
        return _fit(config, bank, seed, path, training, valid, tf, vf, sampler, deadline, max_updates)


def _fit(config, bank, seed, path, training, valid, tf, vf, sampler, deadline, max_updates):
    model, study = _model(config, bank, seed), config.repair.study
    initial = _weights(model)
    identity = {"kind": KIND, "config": config.to_dict(), "config_hash": config.config_hash,
        "bank": bank.to_dict(), "bank_hash": bank.bank_hash, "target_id": model.target_id,
        "seed": seed, "sources": _sources(), "data": {"train": tf, "validation": vf},
        "sampler_identity": sampler.identity, "selection": "natural-validation-fixed-unit-MSE",
        "future_payoff_labels": False, "initial_state_sha256": _digest(initial),
        "environment": {"numpy": np.__version__, "torch": str(torch.__version__),
                        "torch_num_threads": torch.get_num_threads()},
        "adam": {"lr": study.learning_rate, "weight_decay": 0.},
        "lbfgs": {"lr": 1., "max_iter": study.lbfgs_max_iter, "history_size": 30,
                  "tolerance_grad": 1e-12, "tolerance_change": 1e-14,
                  "line_search_fn": "strong_wolfe"}}
    optimizers = {}
    for stage in STAGES:
        parameters = list(getattr(model, stage+"_network").parameters())
        optimizers[stage+"_adam"] = torch.optim.AdamW(parameters, **identity["adam"])
        optimizers[stage+"_lbfgs"] = torch.optim.LBFGS(parameters, **identity["lbfgs"])
    scaling = _scaling(model, training, config.repair, tf)
    initial_validation = {stage: natural_validation(model, valid, scaling, stage, config) for stage in STAGES}
    payload = {"schema_version": 1, "kind": KIND, "identity": identity,
        "initial_state": initial, "model_state": initial, "best_mean_state": initial,
        "best_state": initial, "loss_scaling": scaling, "stage": "mean", "complete": False,
        "steps_done": {stage+"_"+opt: 0 for stage in STAGES for opt in OPTIMIZERS},
        "initial_validation": initial_validation, "best_validation": copy.deepcopy(initial_validation),
        "best_step": {stage: 0 for stage in STAGES}, "history": [], "training_seconds": 0.,
        "stage_seconds": {stage: 0. for stage in STAGES}}
    if path.exists():
        payload = _read(path)
        if payload["identity"] != identity:
            raise ValueError("Ablation resume source, helpers, pool, RNG, configuration or environment changed")
        if payload["complete"]:
            return _metadata(payload, path)
        model.load_state_dict(payload["model_state"])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizer_states"][key])
        sampler.load_state_dict(payload["sampler_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["loss_scaling"] != scaling:
            raise ValueError("Resume training-only scaling does not match the literal input pool")
    prior = payload["training_seconds"]
    started, updates = time.monotonic(), 0

    def save():
        payload.update(model_state=_weights(model), sampler_state=sampler.state_dict(),
            optimizer_states={key: value.state_dict() for key, value in optimizers.items()},
            torch_rng_state=torch.get_rng_state(), training_seconds=prior+time.monotonic()-started,
            complete=all(_done(payload["steps_done"], config, stage) for stage in STAGES))
        _save(path, payload)

    save()
    while not payload["complete"] and (deadline is None or time.monotonic() < deadline) and (
            max_updates is None or updates < max_updates):
        stage, steps = payload["stage"], payload["steps_done"]
        if stage == "mean" and _done(steps, config, stage):
            model.load_state_dict(payload["best_mean_state"])
            payload["stage"] = stage = "covariance"
            payload["best_validation"][stage] = natural_validation(model, valid, scaling, stage, config)
            payload["covariance_initial_selected_mean_validation"] = copy.deepcopy(payload["best_validation"][stage])
            payload["best_state"] = _weights(model)
        for name in STAGES:
            getattr(model, name+"_network").requires_grad_(name == stage)
        opt = "adam" if steps[stage+"_adam"] < getattr(config.repair, stage+"_adam_steps") else "lbfgs"
        optimizer = optimizers[stage+"_"+opt]
        tick = time.perf_counter()
        model.train()
        if opt == "adam":
            draw = sampler.draw(study.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, training["states"][draw.indices], training["horizons"][draw.indices],
                         scaling, stage, config, draw.importance_weights)
            loss.backward()
        else:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for start in range(0, len(training["states"]), study.batch_size):
                    end = min(len(training["states"]), start+study.batch_size)
                    loss = _loss(model, training["states"][start:end], training["horizons"][start:end],
                                 scaling, stage, config)
                    weight = (end-start)/len(training["states"])
                    (weight*loss).backward()
                    total += weight*loss.detach()
                gradients()
                return total

        def gradients():
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                   for parameter in getattr(model, stage+"_network").parameters()):
                raise FloatingPointError("Nonfinite ablation parameter gradient")

        if opt == "adam":
            gradients()
            optimizer.step()
            value = float(loss.detach())
        else:
            value = float(optimizer.step(closure))
        steps[stage+"_"+opt] += 1
        updates += 1
        step = sum(steps[stage+"_"+name] for name in OPTIMIZERS)
        validation_result = None
        if step % study.validation_every == 0 or _done(steps, config, stage) or (
                opt == "adam" and steps[stage+"_adam"] == getattr(config.repair, stage+"_adam_steps")):
            validation_result = natural_validation(model, valid, scaling, stage, config)
            if validation_result["natural_mse"] < payload["best_validation"][stage]["natural_mse"]:
                payload["best_validation"][stage], payload["best_step"][stage] = validation_result, step
                payload["best_mean_state" if stage == "mean" else "best_state"] = _weights(model)
        seconds = time.perf_counter()-tick
        payload["stage_seconds"][stage] += seconds
        payload["history"].append({"stage": stage, "optimizer": opt, "step": step,
            "training_objective": value, "validation": validation_result, "step_seconds": seconds})
        save()
    return _metadata(_read(path), path)


def load_collocation_ablation_model(path):
    saved = _read(path)
    config, bank = _verify(saved)
    if not saved["complete"]:
        raise ValueError("Cannot evaluate an incomplete collocation ablation checkpoint")
    model = _model(config, bank, saved["identity"]["seed"])
    model.load_state_dict(saved["best_state"])
    model.config, model.complete, model.loss_scaling = config, True, saved["loss_scaling"]
    model.provenance, model.checkpoint_path = saved["identity"], str(Path(path).resolve())
    model.method_kind, model.method_id = KIND, _digest(saved["identity"])
    model.mean_network.requires_grad_(False)
    return model.eval()
