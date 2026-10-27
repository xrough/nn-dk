"""Fresh generator heads for declared terminal and killed integral payoffs.

These heads make a new-payoff comparison an actual learning experiment.  They
are not interchangeable with historical four-mode checkpoints.  Training uses
only current states, remaining horizons, and the known autonomous generator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import source_hash
from .generator_jets import (
    GeneratorJet,
    jet_constant,
    jet_multiply,
    jet_scale,
    jet_stack,
    jet_unsqueeze,
)
from .interacting_config import InteractingStudyConfig
from .interacting_model import InteractingMomentModel
from .interacting_training import (
    _atomic_save,
    _bank,
    _initial_loss_scaling,
    _metadata,
    _read,
    _state_hash,
    _validation,
    interacting_objective,
)
from .native import native_coefficients
from .native_types import NativeDKSpec
from .types import stable_hash

PAYOFF_MODEL_KIND = "continuous-native-dk-terminal-killed-integral-moments-v1"


@dataclass(frozen=True)
class PayoffBank:
    """Four fixed physical functionals phi(m_tau)+integral_0^(tau^stop) q(m_s)ds.

    Each output is either a terminal linear observable or a pure running
    integral.  Terminal values are evaluated on the frozen stopped state;
    running integrals cease accumulating at stopping.  Coefficients are exact
    integrals of the physical sine/cosine functions against each constant cell.
    """

    name: str
    terminal_weights: np.ndarray
    running_weights: np.ndarray
    horizon: float
    length: float = 2 * math.pi
    labels: tuple[str, ...] = ("payoff0", "payoff1", "payoff2", "payoff3")

    def __post_init__(self):
        phi = np.array(self.terminal_weights, dtype=np.float64, copy=True)
        q = np.array(self.running_weights, dtype=np.float64, copy=True)
        if (not self.name or phi.ndim != 2 or phi.shape[0] != 4 or phi.shape[1] < 8
                or q.shape != phi.shape or not np.isfinite(phi).all() or not np.isfinite(q).all()
                or not np.isfinite(self.horizon) or self.horizon <= 0
                or not np.isfinite(self.length) or self.length <= 0
                or len(self.labels) != 4 or len(set(self.labels)) != 4):
            raise ValueError("A named finite four-output physical payoff bank is required")
        terminal = np.any(phi != 0, axis=1)
        running = np.any(q != 0, axis=1)
        if np.any(terminal == running):
            raise ValueError("Every output must be either terminal or pure integral")
        scales = np.sqrt(np.mean(phi**2 + self.horizon**2 * q**2, axis=1))
        if np.any(scales < 1e-12):
            raise ValueError("Payoffs must have nonzero fixed physical normalization")
        phi.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "terminal_weights", phi)
        object.__setattr__(self, "running_weights", q)
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def n_cells(self):
        return self.terminal_weights.shape[1]

    @property
    def integral_rows(self):
        return ~np.any(self.terminal_weights != 0, axis=1)

    @property
    def scales(self):
        return np.sqrt(np.mean(self.terminal_weights**2
                               + self.horizon**2 * self.running_weights**2, axis=1))

    def to_dict(self):
        return {"name": self.name, "terminal_weights": self.terminal_weights.tolist(),
                "running_weights": self.running_weights.tolist(), "horizon": self.horizon,
                "length": self.length, "labels": list(self.labels),
                "stopping": "terminal-frozen-integral-killed-at-first-hit-v1",
                "normalization": "sqrt-cell-average-phi2-plus-T2q2-v1"}

    @classmethod
    def from_dict(cls, value):
        expected = {"name", "terminal_weights", "running_weights", "horizon", "length",
                    "labels", "stopping", "normalization"}
        if set(value) != expected:
            raise ValueError("Payoff bank metadata inventory mismatch")
        result = cls(**{key: value[key] for key in expected - {"stopping", "normalization"}})
        if result.to_dict() != value:
            raise ValueError("Payoff bank definition or stopping convention changed")
        return result

    @property
    def bank_hash(self):
        return stable_hash(self.to_dict())

    def evaluate_paths(self, states, times, stopped_at):
        """Reduce complete stopped Euler polygons at every supplied physical node.

        Running integrals use the exact integral of a linear observable on
        each segment, ending at its interpolated hit time. Supply the complete
        integration grid; subsampling changes the polygon and quadrature target.
        """
        states, times = np.asarray(states, dtype=float), np.asarray(times, dtype=float)
        stops = np.asarray(stopped_at, dtype=float)
        if (states.ndim != 3 or states.shape[0] < 1 or states.shape[2] != self.n_cells
                or times.shape != (states.shape[1],) or len(times) < 1 or times[0] != 0
                or not np.isfinite(states).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or times[-1] > self.horizon+1e-12
                or stops.shape != (states.shape[0],) or np.isnan(stops).any()
                or np.any(stops < 0) or np.any(np.isfinite(stops) & (stops > times[-1]))):
            raise ValueError("Complete finite stopped polygons and declared per-path hit times required")
        result = states @ self.terminal_weights.T
        if len(times) > 1:
            exposure = np.minimum(np.diff(times)[None, :],
                                  np.maximum(stops[:, None]-times[None, :-1], 0.))
            running = states @ self.running_weights.T
            segments = .5*exposure[..., None]*(running[:, :-1]+running[:, 1:])
            result[:, 1:] += np.cumsum(segments, axis=1)
        return result

    @staticmethod
    def _weights(spec, modes):
        centers = (np.arange(spec.n_cells) + .5) * spec.h
        return np.asarray([getattr(np, kind)(2 * np.pi * mode * centers / spec.length)
                           * np.sinc(mode / spec.n_cells)
                           for mode, kind in modes])

    @classmethod
    def terminal_high_modes(cls, spec: NativeDKSpec, horizon: float):
        phi = cls._weights(spec, ((3, "sin"), (3, "cos"), (4, "sin"), (4, "cos")))
        return cls("terminal-modes34", phi, np.zeros_like(phi), horizon, spec.length,
                   ("terminal-sin3", "terminal-cos3", "terminal-sin4", "terminal-cos4"))

    @classmethod
    def mixed_terminal_integral(cls, spec: NativeDKSpec, horizon: float):
        modes = cls._weights(spec, ((3, "sin"), (3, "cos"), (1, "sin"), (1, "cos")))
        phi, q = modes.copy(), modes.copy() / horizon
        phi[2:] = 0
        q[:2] = 0
        return cls("terminal3-integral1", phi, q, horizon, spec.length,
                   ("terminal-sin3", "terminal-cos3", "killed-integral-sin1-over-T",
                    "killed-integral-cos1-over-T"))


class PayoffMomentModel(InteractingMomentModel):
    """A four-payoff full-state backward generator mean/covariance head."""

    kind = PAYOFF_MODEL_KIND
    architecture = "full-state-terminal-killed-integral-generator"

    def __init__(self, spec, context, horizon, bank: PayoffBank, width=32,
                 transfer_state=None):
        if (not isinstance(bank, PayoffBank) or bank.n_cells != spec.n_cells
                or bank.length != spec.length or bank.horizon != horizon):
            raise ValueError("Payoff bank must match the physical grid and trained horizon")
        super().__init__(spec, context, horizon, width)
        self.bank = bank
        self.observation_weights = torch.tensor(bank.terminal_weights.copy())
        self.observation_std = torch.tensor(bank.scales)
        self.normalized_weights = self.observation_weights / self.observation_std[:, None]
        self.register_buffer("normalized_running_weights",
                             torch.tensor(bank.running_weights.copy()) / self.observation_std[:, None])
        self.register_buffer("integral_rows", torch.tensor(bank.integral_rows))
        # Terminal factors use their instantaneous QV. Integral factors encode
        # the exact leading tau^3/3 integrated-noise variance. This is a known
        # local generator quantity, not a simulated future-moment label.
        uniform = np.full(spec.n_cells, 1 / spec.n_cells)
        qv = native_coefficients(uniform, context, spec).covariance
        phi, q = bank.terminal_weights, bank.running_weights
        leading = (phi@qv@phi.T + .5*horizon*(phi@qv@q.T + q@qv@phi.T)
                   + horizon**2/3*q@qv@q.T)
        factor = np.linalg.cholesky(leading / bank.scales[:, None] / bank.scales[None, :])
        entries = torch.tensor(factor)[self.factor_rows, self.factor_columns]
        diagonal = self.factor_rows == self.factor_columns
        entries[diagonal] += torch.log(-torch.expm1(-entries[diagonal]))
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()
            self.network[-1].bias[4:] = entries
        self.transfer_hash = None
        if transfer_state is not None:
            selected = {}
            current = self.state_dict()
            for key in current:
                if key.startswith(("network.0.", "network.2.", "network.4.")):
                    value = transfer_state.get(key)
                    if (not isinstance(value, torch.Tensor) or value.shape != current[key].shape
                            or not bool(torch.isfinite(value).all())):
                        raise ValueError("Transferred trunk tensor inventory/shape is incompatible")
                    selected[key] = value.detach().clone()
            self.load_state_dict(selected, strict=False)
            self.transfer_hash = _state_hash(selected)

    @property
    def target_id(self):
        from .interacting_config import interacting_target_id
        return stable_hash({"native_law": interacting_target_id(
            self.spec, self.context, self.horizon), "kind": self.kind, "bank": self.bank.to_dict()})

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "payoff_bank_hash": self.bank.bank_hash,
                "covariance": "tau-gate-D-L-Lt-D; D=tau/T-for-pure-integrals-else-1",
                "initial_factor": "known-uniform-leading-QV; terminal-integral-T/2-integral-integral-T2/3",
                "integrals": "killed-at-native-first-hit",
                "transfer": "hidden-trunk-only-final-head-reset"}

    def _values_unchecked(self, initial, horizons):
        mean, covariance = super()._values_unchecked(initial, horizons)
        factors = torch.where(self.integral_rows, horizons[..., None] / self.horizon, 1.)
        return mean, covariance * factors[..., :, None] * factors[..., None, :]

    def _moment_jets(self, initial, horizons):
        mean, covariance = super()._moment_jets(initial, horizons)
        time_jet = GeneratorJet(horizons, torch.ones_like(horizons),
                                torch.zeros_like(horizons),
                                horizons.new_zeros((*horizons.shape, self.spec.n_cells)))
        factors = jet_stack([jet_scale(time_jet, 1 / self.horizon) if integral
                             else jet_constant(torch.ones_like(horizons), self.spec.n_cells)
                             for integral in self.bank.integral_rows])
        covariance = jet_multiply(covariance, jet_multiply(
            jet_unsqueeze(factors, -1), jet_unsqueeze(factors, -2)))
        return mean, covariance

    def residual(self, initial, horizons):
        initial, horizons = self._inputs(initial, horizons)
        if initial.ndim != 2:
            raise ValueError("Payoff residual requires a [batch,cells] collocation bank")
        mean, covariance = self._moment_jets(initial, horizons)
        source = math.sqrt(self.context.n_particles) * (initial @ self.normalized_running_weights.T)
        source = torch.where(self._active(initial)[..., None], source, 0.)
        quadratic = mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2)
        return (mean.time_derivative - mean.generator - source,
                covariance.time_derivative - covariance.generator - quadratic)

    def predict_conditional(self, initial, horizons):
        result = super().predict_conditional(initial, horizons)
        result.diagnostics.update(payoff_bank=self.bank.to_dict(), payoff_bank_hash=self.bank.bank_hash,
                                  killed_integral=True, covariance_small_time_orders="1,2,3")
        return result


def _verify(saved):
    if saved.get("schema_version") != 1 or saved.get("kind") != PAYOFF_MODEL_KIND:
        raise ValueError("Expected a declared terminal/integral payoff checkpoint")
    provenance = saved.get("provenance", {})
    identity = {key: value for key, value in provenance.items() if key != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("Payoff checkpoint provenance identity mismatch")
    if provenance.get("source_hash") != source_hash():
        raise ValueError("Payoff checkpoint source hash differs from current source")
    cfg = InteractingStudyConfig(**provenance["config"])
    bank = PayoffBank.from_dict(provenance["payoff_bank"])
    if (cfg.config_hash != provenance["config_hash"] or provenance["seed"] not in cfg.optimizer_seeds
            or bank.bank_hash != provenance.get("payoff_bank_hash")
            or stable_hash(provenance["data_fingerprints"]) != provenance["data_hash"]):
        raise ValueError("Payoff checkpoint configuration or input fingerprint mismatch")
    with torch.random.fork_rng():
        torch.manual_seed(provenance["seed"])
        model = PayoffMomentModel(cfg.spec, cfg.physics, cfg.horizon, bank, cfg.width)
    if (model.target_id != provenance["target_id"]
            or model.architecture_metadata != provenance["architecture"]):
        raise ValueError("Payoff checkpoint target or architecture mismatch")
    expected_buffers = {key: value for key, value in model.state_dict().items()
                        if not key.startswith("network.")}
    for name in ("model_state", "best_state", "initial_state"):
        expected_state = model.state_dict()
        if (set(saved[name]) != set(expected_state)
                or any(not isinstance(saved[name][key], torch.Tensor)
                       or saved[name][key].shape != value.shape
                       or not bool(torch.isfinite(saved[name][key]).all())
                       for key, value in expected_state.items())):
            raise ValueError("Payoff checkpoint state tensor inventory or finiteness mismatch")
        if _state_hash(saved[name]) != saved[f"{name}_hash"]:
            raise ValueError("Payoff checkpoint state checksum mismatch")
        if any(not isinstance(saved[name].get(key), torch.Tensor)
               or not torch.equal(saved[name][key], value) for key, value in expected_buffers.items()):
            raise ValueError("Payoff checkpoint observable or geometry buffers changed")
    initial_expected = model.state_dict()
    transferred = provenance.get("transfer_trunk_hash")
    initial_trunk = {key: value for key, value in saved["initial_state"].items()
                     if key.startswith(("network.0.", "network.2.", "network.4."))}
    if transferred is not None and _state_hash(initial_trunk) != transferred:
        raise ValueError("Payoff checkpoint transferred trunk hash mismatch")
    for key, value in initial_expected.items():
        if (transferred is None or not key.startswith(("network.0.", "network.2.", "network.4."))) \
                and not torch.equal(saved["initial_state"][key], value):
            raise ValueError("Payoff checkpoint initialization or head reset changed")
    if provenance["loss_scaling"]["initial_model_state_hash"] != saved["initial_state_hash"]:
        raise ValueError("Payoff checkpoint loss normalization initial state mismatch")
    scaling = provenance["loss_scaling"]
    if (scaling.get("mode") != cfg.loss_balancing or scaling.get("source_role") != "train"
            or not scaling.get("frozen_before_optimizer")
            or scaling.get("training_fingerprint_hash") != stable_hash(provenance["data_fingerprints"]["training"])):
        raise ValueError("Payoff checkpoint training-only residual scaling mismatch")
    for component in ("mean", "covariance"):
        raw = scaling[f"{component}_initial_train_raw_rms"]
        scale = scaling[f"{component}_scale"]
        expected = max(cfg.horizon * raw, cfg.loss_scale_floor) if cfg.loss_balancing != "none" else 1.
        if (not np.isfinite(raw) or raw < 0 or scale != expected
                or scaling[f"{component}_initial_train_dimensionless_rms"] != cfg.horizon * raw):
            raise ValueError("Payoff checkpoint frozen scales invalid")
    adam_done, lbfgs_done = saved["adam_steps_done"], saved["lbfgs_steps_done"]
    if (type(adam_done) is not int or type(lbfgs_done) is not int
            or not 0 <= adam_done <= cfg.adam_steps or not 0 <= lbfgs_done <= cfg.lbfgs_steps
            or (lbfgs_done and adam_done != cfg.adam_steps)
            or saved["complete"] != (adam_done == cfg.adam_steps and lbfgs_done == cfg.lbfgs_steps)
            or not 0 <= saved["best_step"] <= adam_done + lbfgs_done):
        raise ValueError("Payoff checkpoint optimization allocation mismatch")
    return cfg, bank


def fit_payoff(config, bank, seed, path, train, val, transfer_state=None, deadline=None):
    """Fit a cold or trunk-transferred fresh payoff head with exact continuation."""
    cfg = config
    if (not isinstance(cfg, InteractingStudyConfig) or type(seed) is not int
            or seed not in cfg.optimizer_seeds or path is None):
        raise ValueError("A declared interacting configuration/optimizer seed is required")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("Use an absolute finite monotonic deadline")
    started = time.monotonic()
    torch.set_num_threads(1)
    training, tf = _bank(train, cfg, "train")
    validation, vf = _bank(val, cfg, "validation")
    if set(training["family_ids"]) & set(validation["family_ids"]):
        raise ValueError("Training/validation trajectory families must be disjoint")
    torch.manual_seed(seed)
    model = PayoffMomentModel(cfg.spec, cfg.physics, cfg.horizon, bank, cfg.width, transfer_state)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    scaling = _initial_loss_scaling(model, training, cfg, tf)
    provenance = {"kind": model.kind, "seed": seed, "target_id": model.target_id,
                  "config": cfg.to_dict(), "config_hash": cfg.config_hash, "source_hash": source_hash(),
                  "payoff_bank": bank.to_dict(), "payoff_bank_hash": bank.bank_hash,
                  "data_fingerprints": {"training": tf, "validation": vf},
                  "data_hash": stable_hash({"training": tf, "validation": vf}),
                  "architecture": model.architecture_metadata, "transfer_trunk_hash": model.transfer_hash,
                  "future_moment_labels": False, "loss_scaling": scaling,
                  "training": "known-generator-mean-covariance-residual-killed-running-source-v1",
                  "objective": "T^2*(mean-entry-MSE/mean_scale^2+cov-entry-MSE/cov_scale^2)",
                  "selection": "minimum-independent-validation-at-declared-checks", "weight_decay": 0.,
                  "learner_inputs": "full-current-masses-and-remaining-horizon"}
    provenance["identity_hash"] = stable_hash(provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adam = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=cfg.lbfgs_max_iter,
                             history_size=30, tolerance_grad=1e-12, tolerance_change=1e-14,
                             line_search_fn="strong_wolfe")
    shuffle = torch.Generator().manual_seed(0x5041594F + seed)
    adam_done, lbfgs_done, prior, history, best_step = 0, 0, 0., [], 0
    if path.exists():
        saved = _read(path)
        _verify(saved)
        if saved["provenance"]["identity_hash"] != provenance["identity_hash"]:
            raise ValueError("Payoff resume source, data, bank or initialization changed")
        if saved["complete"]:
            return _metadata(saved, path)
        model.load_state_dict(saved["model_state"])
        adam.load_state_dict(saved["adam_state"])
        lbfgs.load_state_dict(saved["lbfgs_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        shuffle.set_state(saved["shuffle_rng_state"])
        adam_done, lbfgs_done = saved["adam_steps_done"], saved["lbfgs_steps_done"]
        prior, history, best_step = saved["training_seconds"], saved["history"], saved["best_step"]
        best_state, best_validation = saved["best_state"], saved["best_validation"]
        initial_validation = saved["initial_validation"]
    else:
        initial_validation = _validation(model, validation, cfg, scaling)
        best_validation = initial_validation.copy()
        best_state = initial_state.copy()

    def complete():
        return adam_done == cfg.adam_steps and lbfgs_done == cfg.lbfgs_steps

    def save():
        current = model.state_dict()
        payload = {"schema_version": 1, "kind": model.kind, "provenance": provenance,
                   "model_state": current, "best_state": best_state, "initial_state": initial_state,
                   "model_state_hash": _state_hash(current), "best_state_hash": _state_hash(best_state),
                   "initial_state_hash": _state_hash(initial_state), "adam_state": adam.state_dict(),
                   "lbfgs_state": lbfgs.state_dict(), "torch_rng_state": torch.get_rng_state(),
                   "shuffle_rng_state": shuffle.get_state(), "adam_steps_done": adam_done,
                   "lbfgs_steps_done": lbfgs_done, "best_step": best_step,
                   "initial_validation": initial_validation, "best_validation": best_validation,
                   "complete": complete(), "history": history,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "training_seconds": prior + time.monotonic() - started}
        _atomic_save(payload, path)

    def objective(states, horizons):
        loss, _ = interacting_objective(model, states, horizons, scaling)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite fresh-payoff residual")
        return loss

    def check_gradients():
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite fresh-payoff parameter gradient")

    save()
    size = len(training["states"])
    while not complete() and (deadline is None or time.monotonic() < deadline):
        tick = time.perf_counter()
        model.train()
        if adam_done < cfg.adam_steps:
            phase = "adamw"
            selected = torch.randperm(size, generator=shuffle)[:cfg.batch_size]
            adam.zero_grad(set_to_none=True)
            loss = objective(training["states"][selected], training["horizons"][selected])
            loss.backward()
            check_gradients()
            adam.step()
            value = float(loss.detach())
            adam_done += 1
        else:
            phase = "lbfgs"

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for start in range(0, size, cfg.batch_size):
                    end = min(start + cfg.batch_size, size)
                    weight = (end - start) / size
                    loss = objective(training["states"][start:end], training["horizons"][start:end])
                    (weight * loss).backward()
                    total += weight * loss.detach()
                check_gradients()
                return total

            value = float(lbfgs.step(closure))
            lbfgs_done += 1
        step = adam_done + lbfgs_done
        diagnostics = None
        if step % cfg.validation_every == 0 or complete() or (phase == "adamw" and adam_done == cfg.adam_steps):
            diagnostics = _validation(model, validation, cfg, scaling)
            if diagnostics["objective"] < best_validation["objective"]:
                best_validation, best_step = diagnostics, step
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"step": step, "phase": phase, "training_objective": value,
                        "validation": diagnostics, "step_seconds": time.perf_counter() - tick})
        save()
    return _metadata(_read(path), path)


def load_payoff_model(path):
    """Restore only a complete declared-bank checkpoint with strict integrity checks."""
    path = Path(path)
    saved = _read(path)
    cfg, bank = _verify(saved)
    if not saved["complete"]:
        raise ValueError("Cannot evaluate an incomplete payoff checkpoint")
    with torch.random.fork_rng():
        torch.manual_seed(saved["provenance"]["seed"])
        model = PayoffMomentModel(cfg.spec, cfg.physics, cfg.horizon, bank, cfg.width)
    model.load_state_dict(saved["best_state"])
    model.provenance, model.checkpoint_path = saved["provenance"], str(path.resolve())
    model.config, model.complete, model.loss_scaling = cfg, True, saved["provenance"]["loss_scaling"]
    model.eval()
    return model
