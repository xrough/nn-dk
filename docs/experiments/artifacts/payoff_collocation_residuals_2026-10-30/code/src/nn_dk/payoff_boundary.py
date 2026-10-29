"""Moving-boundary, native-slope moment solver with fixed-unit robust loss.

Historical v2 source and checkpoints are preserved.  The moving heat-style
gate is flat at zero remaining time at every strict interior state, while
positive-time absorbed values remain frozen.  No future outcomes or estimated
moment labels enter the exact-generator objective.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .generator_jets import (
    GeneratorJet,
    jet_add,
    jet_cat,
    jet_constant,
    jet_linear,
    jet_multiply,
    jet_scale,
    jet_sequential,
    jet_slice,
    jet_unsqueeze,
)
from .interacting_config import InteractingStudyConfig
from .path_numerics import path_coefficients
from .payoff_repair import RepairPayoffModel
from .types import stable_hash

BOUNDARY_PAYOFF_KIND = "continuous-native-dk-moving-boundary-native-slope-moments-v3"


@dataclass(frozen=True)
class BoundaryRepairConfig:
    study: InteractingStudyConfig
    mean_adam_steps: int | None = None
    covariance_adam_steps: int | None = None
    mean_lbfgs_steps: int | None = None
    covariance_lbfgs_steps: int | None = None
    quadrature_nodes: int = 16
    pseudo_huber_delta: float = 3.

    def __post_init__(self):
        if not isinstance(self.study, InteractingStudyConfig):
            raise TypeError("A fixed interacting study is required")
        for stage in ("mean", "covariance"):
            for optimizer in ("adam", "lbfgs"):
                key = f"{stage}_{optimizer}_steps"
                value = getattr(self, key)
                if value is None:
                    value = getattr(self.study, f"{optimizer}_steps")
                    object.__setattr__(self, key, value)
                if type(value) is not int or value < 0:
                    raise ValueError("Nonnegative optimizer allocations required")
            if sum(getattr(self, f"{stage}_{o}_steps") for o in ("adam", "lbfgs")) < 1:
                raise ValueError("Both stages require a nonempty optimizer allocation")
        if type(self.quadrature_nodes) is not int or not 2 <= self.quadrature_nodes <= 32:
            raise ValueError("Declare 2 through 32 deterministic quadrature nodes")
        if not math.isfinite(self.pseudo_huber_delta) or self.pseudo_huber_delta <= 0:
            raise ValueError("A finite positive fixed pseudo-Huber delta is required")

    @property
    def scale_time_floor(self):
        # Checkpoint-schema bridge only.  A value of one disables every
        # inherited time-power safeguard; the v3 objective never uses it.
        return 1.

    def to_dict(self):
        return {**asdict(self), "study": self.study.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(**(value | {"study": InteractingStudyConfig(**value["study"])}))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())


def _roll_jet(jet, shift):
    return GeneratorJet(*(torch.roll(getattr(jet, key), shift, dims=-2 if key == "face_derivatives" else -1)
                          for key in ("value", "time_derivative", "generator", "face_derivatives")))


class BoundaryPayoffModel(RepairPayoffModel):
    """Stopped conditional means/covariances with native short-time anchors."""

    kind = BOUNDARY_PAYOFF_KIND
    architecture = "moving-heat-boundary-native-slope-disjoint-staged-generator"

    def __init__(self, spec, context, horizon, bank, width=32, quadrature_nodes=16):
        super().__init__(spec, context, horizon, bank, width, quadrature_nodes)
        cells = torch.arange(spec.n_cells, dtype=torch.float64)
        difference = ((cells+1)[:, None]-(cells+.5)[None, :])*spec.h
        force = (-context.kappa*np.sinc(spec.h/(2*np.pi))*torch.sin(difference)
                 + 2*context.a2*np.sinc(spec.h/np.pi)*torch.sin(2*difference))
        self.register_buffer("interaction_force", force)

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata, "architecture": self.architecture,
            "boundary_gate": "product[erf(distance*sqrt(T/(width_squared*remaining_time)))]",
            "boundary_zero_time": "strict-interior gate1; explicit exact native jets; absorbed override",
            "gate_exponent_evaluation": "log-domain; logx capped230 only in machine-zero exponential tail",
            "mean_skip": "free-heat-plus-t*native-interaction-terminal-drift",
            "mean_correction": "moving-gate*t^2/T*NN-for-every-output",
            "covariance": "moving-gate*H*C_heat*Ht; same time-ordered congruence",
            "small_time_mean": "exact native terminal slope and killed-integral source, independent of heads",
            "training_loss": "fixed-train-full-horizon-scales; no inverse-time powers; pseudo-Huber"}

    def _gate_terms(self, safe, horizons):
        """Cell factors and exact partials, evaluated without masked 0/0.

        Combining exponential and polynomial factors in log space avoids
        0*infinity in small-time tails. The capped x is at least exp(230)
        whenever capping is active, where all represented exp(-x) terms vanish.
        """
        positive = horizons > 0
        time = torch.where(positive, horizons, torch.full_like(horizons, self.horizon))
        distance = safe-self.spec.mass_stop
        logd, logt = torch.log(distance), torch.log(time)[..., None]
        logx = 2*logd-math.log(self.boundary_width**2/self.horizon)-logt
        x = torch.exp(logx.clamp_max(230.))
        value = torch.erf(torch.sqrt(x))
        first = 2/math.sqrt(math.pi)*torch.exp(-x+.5*logx-logd)
        second = -4/math.sqrt(math.pi)*torch.exp(-x+1.5*logx-2*logd)
        dt = -1/math.sqrt(math.pi)*torch.exp(-x+.5*logx-logt)
        mask = positive[..., None]
        return (torch.where(mask, value, 1.), torch.where(mask, dt, 0.),
                torch.where(mask, first, 0.), torch.where(mask, second, 0.))

    def moving_barrier(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial, torch.full_like(initial, 1/self.spec.n_cells))
        gate = self._gate_terms(safe, horizons)[0].prod(-1)
        return torch.where(active, gate, 0.)

    def _origins(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial, torch.full_like(initial, 1/self.spec.n_cells))
        drift, amplitude = path_coefficients(safe, self.context, self.spec)
        face = self.incidence*amplitude.unsqueeze(-2)/math.sqrt(self.context.n_particles)
        state = GeneratorJet(safe, torch.zeros_like(safe), drift, face)
        projection = math.sqrt(self.context.n_particles*self.spec.n_cells)*self.basis.T
        features = jet_linear(state, projection,
            -safe.new_full((self.spec.n_cells,), 1/self.spec.n_cells) @ projection.T)
        time_jet = GeneratorJet(horizons, torch.ones_like(horizons), torch.zeros_like(horizons),
                               horizons.new_zeros((*horizons.shape, self.spec.n_cells)))
        inputs = jet_cat([features, jet_unsqueeze(jet_scale(time_jet, 1/self.horizon), -1)])
        value, dt, first, second = self._gate_terms(safe, horizons)
        factors = GeneratorJet(value, dt, first*drift+.5*second*face.square().sum(-1),
                               first[..., None]*face)
        barrier = jet_constant(torch.ones_like(horizons), self.spec.n_cells)
        for cell in range(self.spec.n_cells):
            barrier = jet_multiply(barrier, jet_slice(factors, (..., cell)))
        return active, state, time_jet, inputs, barrier

    def _interaction_jet(self, state):
        density = jet_scale(jet_add(state, _roll_jet(state, -1)), 1/(2*self.spec.h))
        force = jet_linear(state, self.interaction_force)
        return jet_linear(jet_multiply(density, force), self.incidence)

    def _interaction_value(self, state):
        flux = (state+torch.roll(state, -1, -1))/(2*self.spec.h)*(state @ self.interaction_force.T)
        return torch.roll(flux, 1, -1)-flux

    def _mean_jet(self, initial, horizons, origins=None, coefficients=None):
        active, state, time_jet, inputs, barrier = origins or self._origins(initial, horizons)
        mc, md, *_ = coefficients or self._time_coefficients(horizons)
        baseline = self._linear_jet(state, mc, md)
        weights = math.sqrt(self.context.n_particles)*self.normalized_weights
        start = jet_linear(state, weights)
        interaction = jet_linear(self._interaction_jet(state), weights)
        anchored = jet_add(jet_add(baseline, jet_scale(start, -1.)),
                           jet_multiply(jet_unsqueeze(time_jet, -1), interaction))
        raw = jet_sequential(inputs, self.mean_network)
        factor = jet_scale(jet_multiply(time_jet, time_jet), 1/self.horizon)
        evolution = jet_add(anchored, jet_multiply(jet_unsqueeze(factor, -1), raw))
        mean = jet_add(start, jet_multiply(jet_unsqueeze(barrier, -1), evolution))
        zero = horizons == 0
        source = math.sqrt(self.context.n_particles)*(state.value @ self.normalized_running_weights.T)
        exact = GeneratorJet(start.value, start.generator+source, start.generator, start.face_derivatives)
        return GeneratorJet(*(
            torch.where(active[..., None, None] if key == "face_derivatives" else active[..., None],
                torch.where(zero[..., None, None] if key == "face_derivatives" else zero[..., None],
                            getattr(exact, key), getattr(mean, key)),
                self.scaled_observables(initial) if key == "value" else 0.)
            for key in ("value", "time_derivative", "generator", "face_derivatives")))

    def _covariance_jet(self, initial, horizons, origins=None, coefficients=None):
        origins = origins or self._origins(initial, horizons)
        covariance = super()._covariance_jet(initial, horizons, origins, coefficients)
        active, state, *_ = origins
        gradient = torch.einsum("ig,...gf->...if",
            math.sqrt(self.context.n_particles)*self.normalized_weights, state.face_derivatives)
        derivative = gradient @ gradient.transpose(-1, -2)
        derivative = torch.where(active[..., None, None], derivative, 0.)
        zero = horizons == 0
        return GeneratorJet(*(
            torch.where(zero[..., None, None, None] if key == "face_derivatives" else zero[..., None, None],
                        derivative if key == "time_derivative" else 0., getattr(covariance, key))
            for key in ("value", "time_derivative", "generator", "face_derivatives")))

    def _values_unchecked(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial, torch.full_like(initial, 1/self.spec.n_cells))
        features = math.sqrt(self.context.n_particles*self.spec.n_cells)*((safe-1/self.spec.n_cells) @ self.basis)
        inputs = torch.cat((features, (horizons/self.horizon).unsqueeze(-1)), -1)
        mc, _, cc, _ = self._time_coefficients(horizons)
        start = self.scaled_observables(initial)
        baseline = torch.einsum("...ig,...g->...i", mc, safe)
        interaction = math.sqrt(self.context.n_particles)*(self._interaction_value(safe) @ self.normalized_weights.T)
        gate = self._gate_terms(safe, horizons)[0].prod(-1)
        mean = start+gate[..., None]*(baseline-start+horizons[..., None]*interaction
                     + (horizons.square()/self.horizon)[..., None]*self.mean_network(inputs))
        baseline_covariance = torch.einsum("...ijg,...g->...ij", cc, safe)
        raw = self.covariance_network(inputs).reshape(*horizons.shape, 4, 4)
        transform = torch.eye(4, dtype=torch.float64, device=initial.device)+(horizons/self.horizon)[
            ..., None, None]**self.congruence_powers*raw
        covariance = gate[..., None, None]*(transform @ baseline_covariance @ transform.transpose(-1, -2))
        zero = horizons == 0
        if bool(zero.any()):
            # The value at zero is exact, and autodifferentiating this branch
            # also returns the correct right derivative rather than zero.
            drift, amplitude = path_coefficients(safe, self.context, self.spec)
            slope = math.sqrt(self.context.n_particles)*(
                drift @ self.normalized_weights.T+safe @ self.normalized_running_weights.T)
            gradient = self.normalized_weights @ (self.incidence*amplitude.unsqueeze(-2))
            gamma = gradient @ gradient.transpose(-1, -2)
            mean = torch.where(zero[..., None], start+horizons[..., None]*slope, mean)
            covariance = torch.where(zero[..., None, None], horizons[..., None, None]*gamma, covariance)
        return (torch.where(active[..., None], mean, start), torch.where(active[..., None, None], covariance, 0.))


def _scaling(model, training, cfg, fingerprint):
    # Reuse only the train-only known-heat scale computation, without any
    # residual or held-out outcomes. Physical small-time orders are metadata.
    from .payoff_repair import _scaling as heat_scaling

    return {**heat_scaling(model, training, cfg, fingerprint),
        "mode": "fixed-train-full-horizon-heat-units-pseudo-Huber-v3",
        "normalization_time_powers": [0, 0, 0, 0], "inverse_time_weighting": False,
        "pseudo_huber_delta": cfg.pseudo_huber_delta,
        "floor_interpretation": "schema-value-one; no time safeguard or covariance floor used"}


def _normalized_residual(model, residual, scaling, stage):
    scale = torch.as_tensor(scaling[f"{stage}_scale"], dtype=residual.dtype, device=residual.device)
    return model.horizon*residual/scale


def _objective(model, states, horizons, scaling, stage):
    residual = getattr(model, f"{stage}_residual")(states, horizons)
    normalized = _normalized_residual(model, residual, scaling, stage)
    delta = scaling["pseudo_huber_delta"]
    # Algebraically 2*d²*(sqrt(1+(z/d)²)-1), with no small-z cancellation.
    radius = torch.sqrt(1+(normalized/delta).square())
    loss = (2*normalized.square()/(radius+1)).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite moving-boundary residual objective")
    return loss, residual


def _validation(model, bank, cfg, scaling, stage):
    result = {"objective": 0., "raw_rms": 0., "raw_maximum": 0., "normalized_rms": 0.,
              "normalized_maximum": 0., "stage": stage, "n_states": len(bank["states"])}
    model.eval()
    size = len(bank["states"])
    with torch.no_grad():
        for start in range(0, size, cfg.study.batch_size):
            end = min(size, start+cfg.study.batch_size)
            loss, residual = _objective(model, bank["states"][start:end], bank["horizons"][start:end], scaling, stage)
            normalized = _normalized_residual(model, residual, scaling, stage)
            weight = (end-start)/size
            result["objective"] += weight*float(loss)
            result["raw_rms"] += weight*float(residual.square().mean())
            result["normalized_rms"] += weight*float(normalized.square().mean())
            result["raw_maximum"] = max(result["raw_maximum"], float(residual.abs().max()))
            result["normalized_maximum"] = max(result["normalized_maximum"], float(normalized.abs().max()))
    result["raw_rms"] **= .5
    result["normalized_rms"] **= .5
    return result


@lru_cache(maxsize=1)
def _training_core():
    """Private v2 optimizer engine; canonical module/globals stay untouched.

    The optimizer order, strict state checks, checkpoint resumption, frozen
    selected mean, and RNG protocol are reused. All model/config/loss bindings
    are explicit in this separate namespace, with a distinct v3 identity.
    """
    name = "nn_dk._moving_boundary_training_core_v3"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("payoff_repair.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.RepairConfig = BoundaryRepairConfig
    module.RepairPayoffModel = BoundaryPayoffModel
    module.REPAIRED_PAYOFF_KIND = BOUNDARY_PAYOFF_KIND
    module._scaling, module._objective, module._validation = _scaling, _objective, _validation
    original_verify = module._verify

    def verify(saved):
        cfg, bank = original_verify(saved)
        scaling = saved["provenance"]["loss_scaling"]
        if (scaling.get("normalization_time_powers") != [0]*4
                or scaling.get("inverse_time_weighting") is not False
                or scaling.get("pseudo_huber_delta") != cfg.pseudo_huber_delta
                or scaling.get("mode") != "fixed-train-full-horizon-heat-units-pseudo-Huber-v3"):
            raise ValueError("Moving-boundary robust loss identity changed")
        return cfg, bank

    module._verify = verify
    return module


def fit_boundary_payoff(config, bank, seed, path, train, val, deadline=None):
    return _training_core().fit_repaired_payoff(config, bank, seed, path, train, val, deadline)


def load_boundary_payoff_model(path):
    return _training_core().load_repaired_payoff_model(path)
