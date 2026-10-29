"""Supervised corrections to supplied nonlinear-LNA conditional moments.

Inputs and outputs use physical payoff means and N-scaled covariances. LNA
ODEs and native MC labels are prepared and charged by the caller; this module
neither simulates trajectories nor differentiates a detached numerical ODE.
The target is the explicitly declared finite-Euler label law, not a generator
residual or a continuous stopping-boundary certificate.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nn_dk.data import source_hash
from nn_dk.native_types import NativeDKSpec
from nn_dk.payoff_learning import PayoffBank
from nn_dk.types import Context, stable_hash

KIND = "supervised-boundary-adapted-nonlinear-lna-payoff-correction-v1"
BANK_KIND = "parent-grouped-native-mc-lna-supervised-payoff-bank-v1"
STAGES = ("mean", "covariance")
OPTIMIZERS = ("adam", "lbfgs")
PROVENANCE_KEYS = {"stream_namespace", "role", "parent_pool_sha256", "label_manifest_sha256",
                   "baseline_manifest_sha256", "normalizer_policy", "training_mixture",
                   "all_stopped_paths_retained"}


def _digest(value):
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
    return hashlib.sha256(json.dumps(encode(value), separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources():
    paths = (Path(__file__), Path(__file__).with_name("native_gaussian_baseline.py"))
    return {"package_source_hash": source_hash(), "helper_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def _environment():
    return {"numpy": np.__version__, "torch": str(torch.__version__), "python": platform.python_version(),
            "platform": platform.platform(), "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@dataclass(frozen=True)
class LNAPayoffCorrectionConfig:
    horizon: float
    kappa: float
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    a2: float = .1
    rho_stop: float = .01/(2*math.pi)
    width: int = 32
    depth: int = 2
    mean_adam_steps: int = 1000
    covariance_adam_steps: int = 1000
    mean_lbfgs_steps: int = 30
    covariance_lbfgs_steps: int = 30
    lbfgs_max_iter: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64
    validation_every: int = 20
    label_step: float = 2**-12
    psd_tolerance: float = 1e-10

    def __post_init__(self):
        for name in ("n_cells", "n_particles", "width", "depth", "lbfgs_max_iter", "batch_size", "validation_every"):
            minimum = 8 if name == "n_cells" else 2 if name in ("n_particles", "width") else 1
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError("Positive declared integer grid/network/optimizer allocations required")
        for stage in STAGES:
            values = [getattr(self, stage+"_"+optimizer+"_steps") for optimizer in OPTIMIZERS]
            if any(type(value) is not int or value < 0 for value in values) or sum(values) < 1:
                raise ValueError("Each stage requires a nonempty nonnegative optimizer allocation")
        for name in ("horizon", "diffusion", "learning_rate", "label_step", "psd_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Finite positive horizon, physical scales and numerical tolerances required")
        if not math.isfinite(self.kappa) or self.kappa < 0 or not math.isfinite(self.a2):
            raise ValueError("Finite declared interaction coefficients required")
        spec = self.spec
        velocity = abs(self.kappa)+2*abs(self.a2)
        if velocity*spec.h > 2*self.diffusion+1e-12 or self.label_step*(
                2*self.diffusion/spec.h**2+velocity/spec.h) > 1+1e-12:
            raise ValueError("Declared finite-Euler centered-flux/CFL condition is invalid")
        if any(not math.isclose(t/self.label_step, round(t/self.label_step), abs_tol=1e-9, rel_tol=0)
               for t in self.query_times):
            raise ValueError("All three label queries must lie on the declared finite-Euler grid")

    @property
    def spec(self):
        return NativeDKSpec(self.n_cells, self.rho_stop)

    @property
    def context(self):
        return Context(kappa=self.kappa, a2=self.a2, diffusion=self.diffusion, n_particles=self.n_particles)

    @property
    def query_times(self):
        return tuple(self.horizon*fraction for fraction in (.25, .5, 1.))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete LNA correction configuration required")
        return cls(**value)

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())


def _network(inputs, config, outputs):
    layers = []
    for _ in range(config.depth):
        layers.extend((nn.Linear(inputs, config.width), nn.SiLU()))
        inputs = config.width
    layers.append(nn.Linear(inputs, outputs))
    result = nn.Sequential(*layers).double()
    with torch.no_grad():
        result[-1].weight.zero_()
        result[-1].bias.zero_()
    return result


def _moments(mean, covariance, shape, tolerance, name):
    if (mean.shape != (*shape, 4) or covariance.shape != (*shape, 4, 4)
            or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all())):
        raise ValueError(f"Finite physical {name} means and N-scaled covariances required")
    scale = covariance.abs().amax(dim=(-2, -1)).clamp_min(1.)
    if (bool(((covariance-covariance.transpose(-1, -2)).abs().amax(dim=(-2, -1)) > tolerance*scale).any())
            or bool((torch.linalg.eigvalsh(covariance).amin(-1) < -tolerance*scale).any())):
        raise ValueError(f"Symmetric PSD {name} covariance required; no clipping, floor or repair")


class LNAPayoffCorrection(nn.Module):
    """A supervised head over supplied, independently accounted LNA moments.

    Congruence is evaluated as F K F^T in physical coordinates, where
    F_ij=delta_ij+(s_i/s_j)x^(1+I_i-I_j)A_ij. This is exactly the requested
    D_x(I+xA)D_x^-1 geometry in bank-normalized coordinates, but avoids
    numerical division by small powers of time. Signed roundoff is retained.
    """
    kind = KIND

    def __init__(self, config, bank):
        super().__init__()
        if (not isinstance(config, LNAPayoffCorrectionConfig) or not isinstance(bank, PayoffBank)
                or bank.n_cells != config.n_cells or bank.horizon != config.horizon
                or bank.length != config.spec.length):
            raise ValueError("A matching fixed physical configuration and payoff bank are required")
        self.config, self.bank = config, bank
        self.mean_network = _network(config.n_cells+1, config, 4)
        self.covariance_network = _network(config.n_cells+1, config, 16)
        self.register_buffer("terminal_weights", torch.tensor(bank.terminal_weights.copy()))
        self.register_buffer("scales", torch.tensor(bank.scales.copy()))
        powers = 1+bank.integral_rows[:, None].astype(int)-bank.integral_rows[None, :].astype(int)
        self.register_buffer("congruence_powers", torch.tensor(powers))
        self.complete = False
        self.provenance = {}

    @property
    def target_id(self):
        physical = {key: self.config.to_dict()[key] for key in (
            "horizon", "kappa", "n_cells", "n_particles", "diffusion", "a2", "rho_stop")}
        return stable_hash({"kind": KIND, "physics": physical, "bank": self.bank.to_dict(),
            "label_law": "stopped-finite-Euler-polygon-terminal-frozen-running-killed-v1",
            "label_step": self.config.label_step, "covariance": "N-scaled-same-time-full-4x4"})

    @property
    def architecture_metadata(self):
        return {"kind": KIND, "width": self.config.width, "depth": self.config.depth,
            "features": "sqrt(NG)*(m-1/G),t/T; no MC labels or profile descriptor",
            "mean": "phi+B*(mu_LNA-phi+bank.scales/sqrt(N)*(t/T)^2*f)",
            "covariance": "bank-normalized B*D*(I+x*A)*D^-1*K*D^-1*(I+x*A)^T*D",
            "boundary_gate": "product-erf((m-mass_stop)/(.05/G*sqrt(t/T)))",
            "covariance_floor": None, "ODE": "supplied uncached nonlinear LNA; caller accounts cost",
            "prediction_scope": "fixed bank, conditional moments only; no law, residual or sensitivity certificate"}

    def _inputs(self, states, times, baseline_mean, baseline_scaled_covariance):
        states = torch.as_tensor(states, dtype=torch.float64, device=self.scales.device)
        times = torch.as_tensor(times, dtype=torch.float64, device=self.scales.device)
        mean = torch.as_tensor(baseline_mean, dtype=torch.float64, device=self.scales.device)
        covariance = torch.as_tensor(baseline_scaled_covariance, dtype=torch.float64, device=self.scales.device)
        if (states.ndim != 2 or not len(states) or states.shape[1] != self.config.n_cells
                or not bool(torch.isfinite(states).all()) or bool((states < 0).any())
                or not bool(torch.allclose(states.sum(-1), torch.ones_like(states[:, 0]), atol=1e-10, rtol=0))):
            raise ValueError("Finite nonnegative unit-mass anchor rows [N,G] required")
        if times.ndim == 1:
            times = times[None].expand(len(states), -1)
        if (times.ndim != 2 or times.shape[0] != len(states) or times.shape[1] < 1
                or not bool(torch.isfinite(times).all()) or bool((times < 0).any())
                or bool((times > self.config.horizon).any())):
            raise ValueError("Finite nonnegative remaining times [Q] or [N,Q] within the horizon required")
        _moments(mean, covariance, times.shape, self.config.psd_tolerance, "baseline")
        return states, times, mean, covariance

    def _prediction(self, states, times, baseline_mean, baseline_covariance, kind):
        n, q = times.shape
        points = states[:, None].expand(n, q, -1).reshape(-1, self.config.n_cells)
        x = (times/self.config.horizon).reshape(-1)
        start = (states @ self.terminal_weights.T)[:, None].expand(n, q, 4).reshape(-1, 4)
        mean, covariance = start.clone(), torch.zeros_like(baseline_covariance).reshape(-1, 4, 4)
        selected = ((points > self.config.spec.mass_stop).all(-1) & (x > 0)).nonzero().flatten()
        if len(selected):
            m, t = points[selected], x[selected]
            gate = torch.erf((m-self.config.spec.mass_stop)/(.05/self.config.n_cells*torch.sqrt(t[:, None]))).prod(-1)
            bm, bc = baseline_mean.reshape(-1, 4)[selected], baseline_covariance.reshape(-1, 4, 4)[selected]
            values = bm-start[selected]
            if kind == "trained":
                features = torch.cat((math.sqrt(self.config.n_particles*self.config.n_cells)*(
                    m-1/self.config.n_cells), t[:, None]), -1)
                values = values+self.scales/math.sqrt(self.config.n_particles)*t[:, None].square()*self.mean_network(features)
                raw = self.covariance_network(features).reshape(-1, 4, 4)
                transform = torch.eye(4, dtype=torch.float64, device=m.device)+raw*(
                    self.scales[:, None]/self.scales[None, :])*t[:, None, None]**self.congruence_powers
                bc = transform @ bc @ transform.transpose(-1, -2)
            mean = mean.index_copy(0, selected, start[selected]+gate[:, None]*values)
            covariance = covariance.index_copy(0, selected, gate[:, None, None]*bc)
        return {"mean": mean.reshape(n, q, 4), "scaled_covariance": covariance.reshape(n, q, 4, 4)}

    def predict_conditional(self, states, times, baseline_mean, baseline_scaled_covariance, *, kind="trained"):
        if kind not in ("trained", "gated_zero"):
            raise ValueError("Use the trained correction or declared gated_zero control")
        values = self._inputs(states, times, baseline_mean, baseline_scaled_covariance)
        result = self._prediction(*values, kind)
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("Nonfinite corrected moment; no sample may be dropped")
        return result


def _names(values, size):
    values = np.asarray(values)
    if (values.shape != (size,) or any(not isinstance(value, (str, bytes)) or not value for value in values.tolist())):
        raise ValueError("Per-anchor nonempty family and parent identifiers required")
    return [value.decode() if isinstance(value, bytes) else value for value in values.tolist()]


def make_supervised_bank(states, times, baseline_mean, baseline_scaled_covariance, label_mean,
                         label_scaled_covariance, family_ids, *, role, config, bank,
                         parent_family_ids=None, row_kinds=None, provenance):
    """Freeze literal anchor/query arrays and declared supervised information.

    Labels are physical sample means and unbiased N-scaled sample covariances,
    made from all stopped paths. Their simulation/refinement identities and costs
    belong to the caller's manifests. Natural validation excludes stress rows.
    """
    model = _model(config, bank, 0)
    states, times, bm, bc = model._inputs(states, times, baseline_mean, baseline_scaled_covariance)
    n = len(states)
    if times.shape[1] != 3 or not torch.equal(times, times.new_tensor(config.query_times).expand(n, -1)):
        raise ValueError("Every anchor requires all three fixed quarter/half/final query times")
    lm, lc = torch.as_tensor(label_mean, dtype=torch.float64), torch.as_tensor(label_scaled_covariance, dtype=torch.float64)
    _moments(lm, lc, times.shape, config.psd_tolerance, "unbiased MC label")
    families = _names(family_ids, n)
    parents = families.copy() if parent_family_ids is None else _names(parent_family_ids, n)
    kinds = ["native"]*n if row_kinds is None else _names(row_kinds, n)
    if (role not in ("train", "validation") or not set(kinds) <= {"native", "stress"}
            or (role == "validation" and "native" not in kinds)
            or any(len({parents[i] for i, value in enumerate(families) if value == family}) != 1
                   for family in set(families))):
        raise ValueError("Declared native/stress parent grouping and natural validation are required")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        raise ValueError("Complete supervised pool, label and baseline provenance required")
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    if (provenance["role"] != role or provenance["all_stopped_paths_retained"] is not True
            or any(not isinstance(provenance[key], str) or not provenance[key]
                   for key in ("stream_namespace", "normalizer_policy"))
            or any(not isinstance(provenance[key], str) or len(provenance[key]) != 64
                   or any(char not in "0123456789abcdef" for char in provenance[key])
                   for key in ("parent_pool_sha256", "label_manifest_sha256", "baseline_manifest_sha256"))):
        raise ValueError("Valid source hashes, information role and retention declarations required")
    absorbed = (states <= config.spec.mass_stop).any(-1)
    if bool(absorbed.any()):
        known = (states[absorbed] @ model.terminal_weights.T)[:, None].expand(-1, 3, -1)
        if (not torch.allclose(lm[absorbed], known, atol=config.psd_tolerance, rtol=0)
                or bool((lc[absorbed].abs() > config.psd_tolerance).any())):
            raise ValueError("Initially absorbed native labels must have exact frozen/killed moments")
    result = {"kind": BANK_KIND, "role": role, "config_hash": config.config_hash, "bank_hash": bank.bank_hash,
        "source_hash": source_hash(), "family_ids": families, "parent_family_ids": parents,
        "row_kinds": kinds, "provenance": provenance, "supervised_native_MC_labels": True,
        **{name: value.detach().cpu().numpy().copy() for name, value in (
            ("states", states), ("times", times), ("baseline_mean", bm), ("baseline_scaled_covariance", bc),
            ("label_mean", lm), ("label_scaled_covariance", lc))}}
    result["content_sha256"] = _digest(result)
    return result


def _bank(value, config, bank, role):
    keys = {"kind", "role", "config_hash", "bank_hash", "source_hash", "family_ids", "parent_family_ids",
            "row_kinds", "provenance", "supervised_native_MC_labels", "states", "times", "baseline_mean",
            "baseline_scaled_covariance", "label_mean", "label_scaled_covariance", "content_sha256"}
    if (not isinstance(value, dict) or set(value) != keys or value["kind"] != BANK_KIND
            or value["role"] != role or value["config_hash"] != config.config_hash
            or value["bank_hash"] != bank.bank_hash or value["source_hash"] != source_hash()
            or value["supervised_native_MC_labels"] is not True):
        raise ValueError("Exact current-source supervised bank schema required")
    rebuilt = make_supervised_bank(*(value[key] for key in ("states", "times", "baseline_mean",
        "baseline_scaled_covariance", "label_mean", "label_scaled_covariance", "family_ids")),
        role=role, config=config, bank=bank, parent_family_ids=value["parent_family_ids"],
        row_kinds=value["row_kinds"], provenance=value["provenance"])
    if rebuilt["content_sha256"] != value["content_sha256"]:
        raise ValueError("Supervised bank or baseline content checksum changed")
    arrays = {key: torch.tensor(rebuilt[key], dtype=torch.float64) for key in (
        "states", "times", "baseline_mean", "baseline_scaled_covariance", "label_mean", "label_scaled_covariance")}
    arrays["natural"] = torch.tensor([kind == "native" for kind in rebuilt["row_kinds"]])
    return arrays, {"kind": BANK_KIND, "role": role, "content_sha256": rebuilt["content_sha256"],
        "anchors": len(rebuilt["states"]), "queries_per_anchor": 3,
        "parent_families": len(set(rebuilt["parent_family_ids"])),
        "parent_family_sha256": _digest(rebuilt["parent_family_ids"]),
        "row_counts": {kind: rebuilt["row_kinds"].count(kind) for kind in ("native", "stress")},
        "baseline_arrays_sha256": _digest([rebuilt["baseline_mean"], rebuilt["baseline_scaled_covariance"]]),
        "label_arrays_sha256": _digest([rebuilt["label_mean"], rebuilt["label_scaled_covariance"]]),
        "provenance": rebuilt["provenance"]}


def _model(config, bank, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return LNAPayoffCorrection(config, bank)


def _loss(model, arrays, stage, indices=None):
    data = arrays if indices is None else {key: value[indices] for key, value in arrays.items()}
    prediction = model._prediction(*(data[key] for key in (
        "states", "times", "baseline_mean", "baseline_scaled_covariance")), "trained")
    scale = model.scales/math.sqrt(model.config.n_particles) if stage == "mean" else model.scales[:, None]*model.scales[None, :]
    key = "mean" if stage == "mean" else "scaled_covariance"
    result = ((prediction[key]-data["label_"+key])/scale).square().mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Nonfinite supervised moment loss")
    return result


def natural_validation(model, arrays, stage):
    indices = arrays["natural"].nonzero().flatten()
    model.eval()
    with torch.no_grad():
        total = sum(float(_loss(model, arrays, stage, part))*len(part) for part in
                    indices.split(model.config.batch_size))/len(indices)
    return {"natural_mse": total, "anchors": len(indices), "queries_per_anchor": 3}


def _weights(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _mean_weights(state):
    return {key: value for key, value in state.items() if key.startswith("mean_network.")}


def _done(steps, config, stage):
    return all(steps[stage+"_"+optimizer] == getattr(config, stage+"_"+optimizer+"_steps") for optimizer in OPTIMIZERS)


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return type(value) is not float or math.isfinite(value)


def _loss_scaling(config, bank):
    return {"mean_scale": (bank.scales/math.sqrt(config.n_particles)).tolist(),
        "covariance_scale": np.outer(bank.scales, bank.scales).tolist(),
        "normalization": "sqrt(N)*physical-mean/bank.scales; Nscaled-covariance/bank.scales-outer",
        "inverse_time_weighting": False, "training_label_law": "declared-finite-Euler-stopped-polygons"}


def _optimizer_configs(config):
    return {"adam": {"lr": config.learning_rate, "weight_decay": 0.},
        "lbfgs": {"lr": 1., "max_iter": config.lbfgs_max_iter, "history_size": 30,
                  "tolerance_grad": 1e-12, "tolerance_change": 1e-14, "line_search_fn": "strong_wolfe"}}


def _verify(payload):
    if payload.get("kind") != KIND or payload.get("schema_version") != 1:
        raise ValueError("Distinct supervised LNA checkpoint schema required")
    identity = payload["identity"]
    config, bank = LNAPayoffCorrectionConfig.from_dict(identity["config"]), PayoffBank.from_dict(identity["bank"])
    if (identity["sources"] != _sources() or identity["environment"] != _environment()
            or identity["config_hash"] != config.config_hash or identity["bank_hash"] != bank.bank_hash
            or identity["selection"] != "natural-validation-bank-fixed-unit-MSE"
            or identity.get("supervised_native_MC_labels") is not True
            or type(identity["seed"]) is not int or identity["seed"] not in (0, 1, 2)):
        raise ValueError("Checkpoint source, environment, configuration or selection changed")
    for role in ("train", "validation"):
        fingerprint = identity["data"][role]
        if (fingerprint.get("role") != role or fingerprint.get("kind") != BANK_KIND
                or fingerprint.get("queries_per_anchor") != 3 or type(fingerprint.get("anchors")) is not int
                or fingerprint["anchors"] < 1 or sum(fingerprint["row_counts"].values()) != fingerprint["anchors"]
                or set(fingerprint["row_counts"]) != {"native", "stress"}
                or (role == "validation" and fingerprint["row_counts"]["native"] < 1)
                or any(not isinstance(fingerprint.get(key), str) or len(fingerprint[key]) != 64
                       or any(character not in "0123456789abcdef" for character in fingerprint[key])
                       for key in ("content_sha256", "parent_family_sha256", "baseline_arrays_sha256", "label_arrays_sha256"))):
            raise ValueError("Checkpoint literal supervised data fingerprint changed")
    if (identity["loss_scaling"] != _loss_scaling(config, bank)
            or any(identity[name] != value for name, value in _optimizer_configs(config).items())
            or identity["sampler"] != {"seed": 0x4C4E41+identity["seed"],
                "law": "uniform-anchor-IID-replacement; all three query rows together"}):
        raise ValueError("Checkpoint normalization, optimizer or private sampler identity changed")
    template = _model(config, bank, identity["seed"])
    initial = payload["initial_state"]
    if (identity["target_id"] != template.target_id or identity["initial_state_sha256"] != _digest(initial)
            or _digest(initial) != _digest(template.state_dict())):
        raise ValueError("Checkpoint initial weights or physical buffers changed")
    keys = {stage+"_"+opt for stage in STAGES for opt in OPTIMIZERS}
    steps = payload["steps_done"]
    if set(steps) != keys or any(type(value) is not int or not 0 <= value <= getattr(config, key+"_steps")
                               for key, value in steps.items()):
        raise ValueError("Checkpoint optimizer allocation changed")
    complete = all(_done(steps, config, stage) for stage in STAGES)
    if (type(payload["complete"]) is not bool or payload["complete"] != complete
            or payload["stage"] not in STAGES or (complete and payload["stage"] != "covariance")
            or (payload["stage"] == "mean" and any(steps["covariance_"+opt] for opt in OPTIMIZERS))
            or (payload["stage"] == "covariance" and not _done(steps, config, "mean"))
            or set(payload["optimizer_states"]) != keys or len(payload["history"]) != sum(steps.values())
            or not _finite(payload["optimizer_states"])):
        raise ValueError("Checkpoint completion, history, stage or optimizer states changed")
    counters = {key: 0 for key in keys}
    for row in payload["history"]:
        stage = "covariance" if _done(counters, config, "mean") else "mean"
        optimizer = "adam" if counters[stage+"_adam"] < getattr(config, stage+"_adam_steps") else "lbfgs"
        counters[stage+"_"+optimizer] += 1
        if (row["stage"] != stage or row["optimizer"] != optimizer or row["step"] != sum(
                counters[stage+"_"+name] for name in OPTIMIZERS) or not _finite(row)
                or row["step_seconds"] < 0):
            raise ValueError("Checkpoint ordered optimizer/update history changed")
    if counters != steps:
        raise ValueError("Checkpoint history/update counters disagree")
    for stage in STAGES:
        parameters = list(getattr(template, stage+"_network").parameters())
        for optimizer in OPTIMIZERS:
            state = payload["optimizer_states"][stage+"_"+optimizer]
            groups = state.get("param_groups", [])
            if (len(groups) != 1 or groups[0].get("params") != list(range(len(parameters)))
                    or not set(state.get("state", {})) <= set(range(len(parameters)))
                    or (steps[stage+"_"+optimizer] > 0 and not state["state"])):
                raise ValueError("Checkpoint optimizer parameter/state inventory changed")
            if optimizer == "adam" and (len(state["state"]) != (len(parameters) if steps[stage+"_adam"] else 0)
                    or any(float(value["step"]) != steps[stage+"_adam"] for value in state["state"].values())):
                raise ValueError("Checkpoint Adam state step count changed")
        maximum = sum(steps[stage+"_"+name] for name in OPTIMIZERS)
        if (type(payload["best_step"][stage]) is not int or not 0 <= payload["best_step"][stage] <= maximum
                or not _finite(payload["best_validation"][stage])
                or payload["best_validation"][stage]["natural_mse"] < 0):
            raise ValueError("Checkpoint selected validation state changed")
    for name in ("model_state", "best_mean_state", "best_state"):
        state = payload[name]
        if set(state) != set(initial):
            raise ValueError("Checkpoint weight inventory changed")
        for key, value in state.items():
            if value.shape != initial[key].shape or value.dtype != initial[key].dtype or not _finite(value):
                raise ValueError("Checkpoint weights are incompatible or nonfinite")
            if not key.startswith(("mean_network.", "covariance_network.")) and not torch.equal(value, initial[key]):
                raise ValueError("Checkpoint physical buffers changed")
    if payload["stage"] == "covariance" and any(_digest(_mean_weights(payload[name])) !=
            _digest(_mean_weights(payload["best_mean_state"])) for name in ("model_state", "best_state")):
        raise ValueError("Covariance stage changed the selected frozen mean")
    rng = payload["sampler_rng_state"]
    if (not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.Generator().get_state().shape
            or payload["sampler_draws"] != config.batch_size*sum(steps[stage+"_adam"] for stage in STAGES)
            or payload["loss_scaling"] != identity["loss_scaling"]):
        raise ValueError("Checkpoint RNG, sample count or fixed loss units changed")
    return config, bank


def _read(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("payload_sha256") != _digest(
            {key: value for key, value in payload.items() if key != "payload_sha256"})):
        raise ValueError("Supervised checkpoint payload checksum changed")
    _verify(payload)
    return payload


@contextmanager
def _exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this supervised checkpoint") from error
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


def _metadata(payload, path):
    return {"path": str(path.resolve()), "kind": KIND, "complete": payload["complete"],
        "method_id": _digest(payload["identity"]), "identity": payload["identity"],
        **{name: payload[name] for name in ("stage", "steps_done", "initial_validation", "best_validation",
            "best_step", "loss_scaling", "training_seconds", "stage_seconds")},
        "timing_scope": "loop wall before final save; excludes bank/ODE/MC preparation, initialization and first validation; caller measures full invocation",
        "interpretation": "Supervised allocation completion is not a label-free or continuous-law accuracy certificate"}


def fit_lna_correction(config, bank, seed, checkpoint, training, validation, *, resume=True,
                       deadline=None, max_updates=None):
    if (not isinstance(config, LNAPayoffCorrectionConfig) or not isinstance(bank, PayoffBank)
            or type(seed) is not int or seed not in (0, 1, 2) or type(resume) is not bool
            or (deadline is not None and not math.isfinite(deadline))
            or (max_updates is not None and (type(max_updates) is not int or max_updates < 0))):
        raise ValueError("Declared config, bank, seed, resume mode and valid invocation budget required")
    train, tf = _bank(training, config, bank, "train")
    valid, vf = _bank(validation, config, bank, "validation")
    if (set(training["family_ids"]+training["parent_family_ids"]) &
            set(validation["family_ids"]+validation["parent_family_ids"])):
        raise ValueError("Training and validation parent/anchor families must be disjoint")
    checkpoint = Path(checkpoint)
    with _exclusive(checkpoint), torch.random.fork_rng():
        if checkpoint.exists() and not resume:
            raise FileExistsError("An existing checkpoint cannot be overwritten with resume=False")
        torch.manual_seed(seed)
        return _fit(config, bank, seed, checkpoint, train, valid, tf, vf, deadline, max_updates)


def _fit(config, bank, seed, path, training, validation, tf, vf, deadline, max_updates):
    model = _model(config, bank, seed)
    initial = _weights(model)
    scaling = _loss_scaling(config, bank)
    identity = {"kind": KIND, "config": config.to_dict(), "config_hash": config.config_hash,
        "bank": bank.to_dict(), "bank_hash": bank.bank_hash, "target_id": model.target_id,
        "seed": seed, "sources": _sources(), "environment": _environment(), "data": {"train": tf, "validation": vf},
        "initial_state_sha256": _digest(initial), "loss_scaling": scaling,
        "selection": "natural-validation-bank-fixed-unit-MSE", "supervised_native_MC_labels": True,
        "sampler": {"seed": 0x4C4E41+seed, "law": "uniform-anchor-IID-replacement; all three query rows together"},
        **_optimizer_configs(config)}
    optimizers = {}
    for stage in STAGES:
        parameters = list(getattr(model, stage+"_network").parameters())
        optimizers[stage+"_adam"] = torch.optim.AdamW(parameters, **identity["adam"])
        optimizers[stage+"_lbfgs"] = torch.optim.LBFGS(parameters, **identity["lbfgs"])
    rng = torch.Generator().manual_seed(identity["sampler"]["seed"])
    initial_validation = {stage: natural_validation(model, validation, stage) for stage in STAGES}
    payload = {"schema_version": 1, "kind": KIND, "identity": identity, "initial_state": initial,
        "model_state": initial, "best_mean_state": initial, "best_state": initial, "stage": "mean", "complete": False,
        "steps_done": {stage+"_"+opt: 0 for stage in STAGES for opt in OPTIMIZERS},
        "initial_validation": initial_validation, "best_validation": copy.deepcopy(initial_validation),
        "best_step": {stage: 0 for stage in STAGES}, "history": [], "loss_scaling": scaling,
        "training_seconds": 0., "stage_seconds": {stage: 0. for stage in STAGES}, "sampler_draws": 0}
    if path.exists():
        payload = _read(path)
        if payload["identity"] != identity:
            raise ValueError("Supervised resume pool, baseline, label, config, source or environment changed")
        if payload["complete"]:
            return _metadata(payload, path)
        model.load_state_dict(payload["model_state"])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizer_states"][key])
        rng.set_state(payload["sampler_rng_state"])
    prior, started, updates = payload["training_seconds"], time.monotonic(), 0

    def save():
        payload.update(model_state=_weights(model), sampler_rng_state=rng.get_state(),
            optimizer_states={key: value.state_dict() for key, value in optimizers.items()},
            training_seconds=prior+time.monotonic()-started,
            complete=all(_done(payload["steps_done"], config, stage) for stage in STAGES))
        _save(path, payload)

    save()
    while not payload["complete"] and (deadline is None or time.monotonic() < deadline) and (
            max_updates is None or updates < max_updates):
        stage, steps = payload["stage"], payload["steps_done"]
        if stage == "mean" and _done(steps, config, stage):
            model.load_state_dict(payload["best_mean_state"])
            payload["stage"] = stage = "covariance"
            payload["best_state"] = _weights(model)
            payload["best_validation"][stage] = natural_validation(model, validation, stage)
        for name in STAGES:
            getattr(model, name+"_network").requires_grad_(name == stage)
        optimizer_name = "adam" if steps[stage+"_adam"] < getattr(config, stage+"_adam_steps") else "lbfgs"
        optimizer = optimizers[stage+"_"+optimizer_name]
        tick = time.perf_counter()
        model.train()

        def gradients():
            if any(parameter.grad is not None and not _finite(parameter.grad)
                   for parameter in getattr(model, stage+"_network").parameters()):
                raise FloatingPointError("Nonfinite supervised correction gradient")

        if optimizer_name == "adam":
            indices = torch.randint(len(training["states"]), (config.batch_size,), generator=rng)
            payload["sampler_draws"] += config.batch_size
            optimizer.zero_grad(set_to_none=True)
            objective = _loss(model, training, stage, indices)
            objective.backward()
            gradients()
            optimizer.step()
            value = float(objective.detach())
        else:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for indices in torch.arange(len(training["states"])).split(config.batch_size):
                    weight = len(indices)/len(training["states"])
                    objective = _loss(model, training, stage, indices)
                    (weight*objective).backward()
                    total += weight*objective.detach()
                gradients()
                return total
            value = float(optimizer.step(closure))
        steps[stage+"_"+optimizer_name] += 1
        updates += 1
        step = sum(steps[stage+"_"+name] for name in OPTIMIZERS)
        measured = None
        if (step % config.validation_every == 0 or _done(steps, config, stage)
                or (optimizer_name == "adam" and steps[stage+"_adam"] == getattr(config, stage+"_adam_steps"))):
            measured = natural_validation(model, validation, stage)
            if measured["natural_mse"] < payload["best_validation"][stage]["natural_mse"]:
                payload["best_validation"][stage], payload["best_step"][stage] = measured, step
                payload["best_mean_state" if stage == "mean" else "best_state"] = _weights(model)
        seconds = time.perf_counter()-tick
        payload["stage_seconds"][stage] += seconds
        payload["history"].append({"stage": stage, "optimizer": optimizer_name, "step": step,
            "training_objective": value, "validation": measured, "step_seconds": seconds})
        save()
    return _metadata(_read(path), path)


def load_lna_correction(checkpoint):
    payload = _read(checkpoint)
    config, bank = _verify(payload)
    if not payload["complete"]:
        raise ValueError("Cannot evaluate an incomplete supervised correction")
    model = _model(config, bank, payload["identity"]["seed"])
    model.load_state_dict(payload["best_state"])
    model.complete, model.provenance, model.loss_scaling = True, payload["identity"], payload["loss_scaling"]
    model.method_kind, model.method_id = KIND, _digest(payload["identity"])
    model.checkpoint_path = str(Path(checkpoint).resolve())
    model.mean_network.requires_grad_(False)
    return model.eval()
