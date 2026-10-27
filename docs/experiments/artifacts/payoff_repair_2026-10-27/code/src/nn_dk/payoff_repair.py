"""Staged, heat-anchored generator moments for terminal and killed integrals.

This is a distinct experiment/model identity; historical checkpoints retain
their original implementation.  The baseline is the *unstopped free-heat*
moment solution (eight-point deterministic covariance quadrature).  The native
interacting stopped generator, not that approximation, defines the residual.
No future moment labels or stochastic trace samples enter fitting.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import source_hash
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
    jet_stack,
    jet_tanh,
    jet_unsqueeze,
)
from .interacting_config import InteractingStudyConfig
from .interacting_model import InteractingMomentModel
from .interacting_training import _atomic_save, _bank, _read, _state_hash
from .observations import incidence_matrix
from .path_numerics import path_coefficients
from .payoff_learning import PayoffBank
from .types import stable_hash

REPAIRED_PAYOFF_KIND = "continuous-native-dk-heat-anchored-staged-payoff-moments-v2"


@dataclass(frozen=True)
class RepairConfig:
    study: InteractingStudyConfig
    mean_adam_steps: int | None = None
    covariance_adam_steps: int | None = None
    mean_lbfgs_steps: int | None = None
    covariance_lbfgs_steps: int | None = None
    quadrature_nodes: int = 8
    scale_time_floor: float = 1e-3

    def __post_init__(self):
        if not isinstance(self.study, InteractingStudyConfig):
            raise TypeError("Repair configuration requires an interacting study")
        for stage in ("mean", "covariance"):
            for optimizer in ("adam", "lbfgs"):
                key = f"{stage}_{optimizer}_steps"
                value = getattr(self, key)
                if value is None:
                    value = getattr(self.study, f"{optimizer}_steps")
                    object.__setattr__(self, key, value)
                if type(value) is not int or value < 0:
                    raise ValueError("Nonnegative staged optimizer allocations required")
            if getattr(self, f"{stage}_adam_steps") + getattr(self, f"{stage}_lbfgs_steps") < 1:
                raise ValueError("Both fitting stages need a nonempty optimizer allocation")
        if type(self.quadrature_nodes) is not int or not 2 <= self.quadrature_nodes <= 32:
            raise ValueError("Use 2 through 32 fixed Gauss-Legendre covariance quadrature nodes")
        if not np.isfinite(self.scale_time_floor) or not 0 < self.scale_time_floor < 1:
            raise ValueError("A finite loss-only small-time safeguard in (0,1) is required")

    def to_dict(self):
        return {**asdict(self), "study": self.study.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(**(value | {"study": InteractingStudyConfig(**value["study"])}))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())


def _network(inputs, width, outputs):
    layers = []
    for _ in range(3):
        layers.extend((nn.Linear(inputs, width), nn.SiLU()))
        inputs = width
    layers.append(nn.Linear(width, outputs))
    result = nn.Sequential(*layers).double()
    with torch.no_grad():
        result[-1].weight.zero_()
        result[-1].bias.zero_()
    return result


def _jet_transpose(jet):
    return GeneratorJet(jet.value.transpose(-1, -2),
                        jet.time_derivative.transpose(-1, -2),
                        jet.generator.transpose(-1, -2),
                        jet.face_derivatives.transpose(-2, -3))


def _jet_matmul(left, right):
    # Four-dimensional matrices are small; explicit index contractions retain
    # every physical face derivative and the exact Ito product cross term.
    return GeneratorJet(
        left.value @ right.value,
        left.time_derivative @ right.value + left.value @ right.time_derivative,
        left.generator @ right.value + left.value @ right.generator
        + torch.einsum("...ikf,...kjf->...ij", left.face_derivatives,
                       right.face_derivatives),
        torch.einsum("...ikf,...kj->...ijf", left.face_derivatives, right.value)
        + torch.einsum("...ik,...kjf->...ijf", left.value, right.face_derivatives))


class RepairPayoffModel(InteractingMomentModel):
    """Independent mean/covariance networks with physical small-time anchors."""

    kind = REPAIRED_PAYOFF_KIND
    architecture = "heat-anchored-disjoint-mean-covariance-staged-generator"

    def __init__(self, spec, context, horizon, bank, width=32, quadrature_nodes=8):
        if (not isinstance(bank, PayoffBank) or bank.n_cells != spec.n_cells
                or bank.length != spec.length or bank.horizon != horizon):
            raise ValueError("Payoff bank must match the native grid and physical horizon")
        if type(quadrature_nodes) is not int or not 2 <= quadrature_nodes <= 32:
            raise ValueError("A declared deterministic Gauss quadrature order is required")
        super().__init__(spec, context, horizon, width)
        del self.network
        self.bank, self.quadrature_nodes = bank, quadrature_nodes
        self.observation_weights = torch.tensor(bank.terminal_weights.copy())
        self.observation_std = torch.tensor(bank.scales)
        self.normalized_weights = self.observation_weights / self.observation_std[:, None]
        self.register_buffer("normalized_running_weights",
                             torch.tensor(bank.running_weights.copy()) / self.observation_std[:, None])
        self.register_buffer("integral_rows", torch.tensor(bank.integral_rows))
        # Congruence is formed in coordinates D=diag(1,t/T for integrals).
        # Its entries t^(1+I_i-I_j) preserve the leading t,t²,t³ coefficients.
        powers = 1 + bank.integral_rows[:, None].astype(int) - bank.integral_rows[None, :]
        self.register_buffer("congruence_powers", torch.tensor(powers))
        self.mean_network = _network(spec.n_cells, width, 4)
        self.covariance_network = _network(spec.n_cells, width, 16)
        heat = context.diffusion / spec.h**2 * (
            np.roll(np.eye(spec.n_cells), 1, axis=0)
            + np.roll(np.eye(spec.n_cells), -1, axis=0) - 2*np.eye(spec.n_cells))
        eigenvalues, eigenvectors = np.linalg.eigh(heat)
        eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.
        self.register_buffer("heat_eigenvalues", torch.tensor(eigenvalues))
        self.register_buffer("heat_eigenvectors", torch.tensor(eigenvectors))
        self._coefficient_cache = {}
        self.baseline_preparation_seconds = 0.

    @property
    def target_id(self):
        from .interacting_config import interacting_target_id

        return stable_hash({"native_law": interacting_target_id(
            self.spec, self.context, self.horizon), "kind": self.kind,
            "bank": self.bank.to_dict()})

    @property
    def architecture_metadata(self):
        return {"architecture": self.architecture, "width": self.width,
                "hidden_layers_per_network": 3, "activation": "SiLU",
                "state_features": "all-sqrt(NG)-scaled-mass-zero-coordinates",
                "learner_inputs": "current-masses-and-remaining-horizon-only",
                "mean_skip": "exact-native-free-heat-terminal-and-running-linear-mean",
                "mean_correction": "gate*t*NN; additional-t/T-for-integrals",
                "covariance_baseline": "native-free-heat-linear-state-positive-Gauss-quadrature",
                "quadrature_nodes": self.quadrature_nodes,
                "covariance": "gate*H*C_heat*Ht; H_ij=delta_ij+(t/T)^(1+I_i-I_j)*NN_ij",
                "covariance_floor": None, "boundary_width": self.boundary_width,
                "boundary_gate": "product-tanh-positive-boundary-strip",
                "trace": "exact-all-native-face-directions",
                "payoff_bank_hash": self.bank.bank_hash,
                "training": "mean-first-independent-network-frozen-before-covariance"}

    def _time_coefficients(self, horizons):
        """Known time-only linear coefficients, cached by exact physical time.

        Each covariance coefficient is linear in m.  Its generator therefore
        has zero state Hessian; only the trained congruence requires nonlinear
        generator calculus.  This cache contains no fitted/future labels.
        """
        started = time.perf_counter()
        flat = horizons.detach().cpu().numpy().reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        missing = [float(x) for x in unique if float(x) not in self._coefficient_cache]
        u = self.heat_eigenvectors.detach().cpu().numpy()
        lam = self.heat_eigenvalues.detach().cpu().numpy()
        phi = self.normalized_weights.detach().cpu().numpy()
        q = self.normalized_running_weights.detach().cpu().numpy()
        b = incidence_matrix(self.spec.n_cells)
        face = self.context.diffusion / self.spec.h**2 * (
            np.eye(self.spec.n_cells) + np.roll(np.eye(self.spec.n_cells), -1, axis=0))
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_nodes)
        nodes, weights = (nodes+1)/2, weights/2

        def transition(t):
            return np.einsum("ik,...k,jk->...ij", u, np.exp(t[..., None]*lam), u)

        def integral(t):
            z = t[..., None]*lam
            quotient = np.divide(np.expm1(z), lam, out=np.broadcast_to(t[..., None], z.shape).copy(),
                                 where=lam != 0)
            return np.einsum("ik,...k,jk->...ij", u, quotient, u)

        for offset in range(0, len(missing), 128):
            times = np.asarray(missing[offset:offset+128])
            p = transition(times)
            a = integral(times)
            mean = math.sqrt(self.context.n_particles) * (phi @ p + q @ a)
            mean_dt = math.sqrt(self.context.n_particles) * ((phi @ (u*lam) @ u.T + q) @ p)
            s, r = times[:, None]*nodes, times[:, None]*(1-nodes)
            ps, pr = transition(s), transition(r)
            ar = integral(r)
            response = phi @ pr + q @ ar
            response_dt = (1-nodes)[None, :, None, None] * (
                (phi @ (u*lam) @ u.T + q) @ pr)
            gradient, gradient_dt = response @ b, response_dt @ b
            face_p = face @ ps
            face_p_dt = nodes[None, :, None, None] * (face @ (u*lam) @ u.T @ ps)
            gram = np.einsum("bqif,bqjf->bqijf", gradient, gradient)
            gram_dt = (np.einsum("bqif,bqjf->bqijf", gradient_dt, gradient)
                       + np.einsum("bqif,bqjf->bqijf", gradient, gradient_dt))
            core = np.einsum("q,bqijf,bqfk->bijk", weights, gram, face_p)
            core_dt = (np.einsum("q,bqijf,bqfk->bijk", weights, gram_dt, face_p)
                       + np.einsum("q,bqijf,bqfk->bijk", weights, gram, face_p_dt))
            covariance = times[:, None, None, None]*core
            covariance_dt = core + times[:, None, None, None]*core_dt
            for i, t in enumerate(times):
                self._coefficient_cache[float(t)] = tuple(torch.tensor(v[i]) for v in
                    (mean, mean_dt, covariance, covariance_dt))
        self.baseline_preparation_seconds += time.perf_counter()-started
        gather = torch.tensor(inverse, dtype=torch.int64)
        values = [torch.stack([self._coefficient_cache[float(t)][i] for t in unique]).index_select(0, gather)
                  .reshape(*horizons.shape, *((4, self.spec.n_cells) if i < 2 else
                                             (4, 4, self.spec.n_cells)))
                  .to(horizons.device) for i in range(4)]
        return values

    def _origins(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial, torch.full_like(initial, 1/self.spec.n_cells))
        drift, amplitude = path_coefficients(safe, self.context, self.spec)
        face = self.incidence * amplitude.unsqueeze(-2) / math.sqrt(self.context.n_particles)
        state = GeneratorJet(safe, torch.zeros_like(safe), drift, face)
        projection = math.sqrt(self.context.n_particles*self.spec.n_cells)*self.basis.T
        features = jet_linear(state, projection,
                              -safe.new_full((self.spec.n_cells,), 1/self.spec.n_cells) @ projection.T)
        time_jet = GeneratorJet(horizons, torch.ones_like(horizons), torch.zeros_like(horizons),
                                horizons.new_zeros((*horizons.shape, self.spec.n_cells)))
        inputs = jet_cat([features, jet_unsqueeze(jet_scale(time_jet, 1/self.horizon), -1)])
        ratios = jet_tanh(jet_add(jet_scale(state, 1/self.boundary_width),
                                  jet_constant(torch.full_like(safe, -self.spec.mass_stop
                                               / self.boundary_width), self.spec.n_cells)))
        barrier = jet_constant(torch.ones_like(horizons), self.spec.n_cells)
        for cell in range(self.spec.n_cells):
            barrier = jet_multiply(barrier, jet_slice(ratios, (..., cell)))
        return active, state, time_jet, inputs, barrier

    @staticmethod
    def _linear_jet(state, coefficients, derivative):
        return GeneratorJet(torch.einsum("...ig,...g->...i", coefficients, state.value),
                            torch.einsum("...ig,...g->...i", derivative, state.value),
                            torch.einsum("...ig,...g->...i", coefficients, state.generator),
                            torch.einsum("...ig,...gf->...if", coefficients, state.face_derivatives))

    def _mean_jet(self, initial, horizons, origins=None, coefficients=None):
        active, state, time_jet, inputs, barrier = origins or self._origins(initial, horizons)
        mc, md, *_ = coefficients or self._time_coefficients(horizons)
        baseline = self._linear_jet(state, mc, md)
        start = jet_linear(state, math.sqrt(self.context.n_particles)*self.normalized_weights)
        evolution = jet_add(baseline, jet_scale(start, -1.))
        raw = jet_sequential(inputs, self.mean_network)
        factors = jet_stack([jet_scale(time_jet, 1/self.horizon) if integral
                             else jet_constant(torch.ones_like(horizons), self.spec.n_cells)
                             for integral in self.bank.integral_rows])
        correction = jet_multiply(jet_multiply(jet_unsqueeze(time_jet, -1), factors), raw)
        mean = jet_add(start, jet_multiply(jet_unsqueeze(barrier, -1),
                                           jet_add(evolution, correction)))
        return GeneratorJet(torch.where(active[..., None], mean.value, self.scaled_observables(initial)),
                            torch.where(active[..., None], mean.time_derivative, 0.),
                            torch.where(active[..., None], mean.generator, 0.),
                            torch.where(active[..., None, None], mean.face_derivatives, 0.))

    def _covariance_jet(self, initial, horizons, origins=None, coefficients=None):
        active, state, time_jet, inputs, barrier = origins or self._origins(initial, horizons)
        _, _, cc, cd = coefficients or self._time_coefficients(horizons)
        covariance = GeneratorJet(
            torch.einsum("...ijg,...g->...ij", cc, state.value),
            torch.einsum("...ijg,...g->...ij", cd, state.value),
            torch.einsum("...ijg,...g->...ij", cc, state.generator),
            torch.einsum("...ijg,...gf->...ijf", cc, state.face_derivatives))
        raw_flat = jet_sequential(inputs, self.covariance_network)
        raw = GeneratorJet(*(getattr(raw_flat, key).reshape(
            *horizons.shape, 4, 4, *(() if key != "face_derivatives" else (self.spec.n_cells,)))
                             for key in ("value", "time_derivative", "generator", "face_derivatives")))
        one = jet_constant(torch.ones_like(horizons), self.spec.n_cells)
        time_scaled = jet_scale(time_jet, 1/self.horizon)
        powers = [one, time_scaled, jet_multiply(time_scaled, time_scaled)]
        factor = jet_stack([jet_stack([powers[int(p)] for p in row], dim=-1)
                            for row in self.congruence_powers], dim=-2)
        transform = jet_add(jet_constant(torch.eye(4, dtype=torch.float64,
                                                   device=initial.device), self.spec.n_cells),
                            jet_multiply(factor, raw))
        covariance = jet_multiply(jet_unsqueeze(jet_unsqueeze(barrier, -1), -1),
                                  _jet_matmul(_jet_matmul(transform, covariance), _jet_transpose(transform)))
        return GeneratorJet(torch.where(active[..., None, None], covariance.value, 0.),
                            torch.where(active[..., None, None], covariance.time_derivative, 0.),
                            torch.where(active[..., None, None], covariance.generator, 0.),
                            torch.where(active[..., None, None, None], covariance.face_derivatives, 0.))

    def _moment_jets(self, initial, horizons):
        origins, coefficients = self._origins(initial, horizons), self._time_coefficients(horizons)
        return (self._mean_jet(initial, horizons, origins, coefficients),
                self._covariance_jet(initial, horizons, origins, coefficients))

    def _values_unchecked(self, initial, horizons):
        # Values use the same algebra as the jets without materializing
        # state/face derivatives during deployment.
        active = self._active(initial)
        safe = torch.where(active[..., None], initial, torch.full_like(initial, 1/self.spec.n_cells))
        features = math.sqrt(self.context.n_particles*self.spec.n_cells)*(
            (safe-1/self.spec.n_cells) @ self.basis)
        inputs = torch.cat((features, (horizons/self.horizon).unsqueeze(-1)), -1)
        mc, _, cc, _ = self._time_coefficients(horizons)
        baseline_mean = torch.einsum("...ig,...g->...i", mc, safe)
        start = self.scaled_observables(initial)
        gate = self.barrier(safe)
        factors = torch.where(self.integral_rows, horizons[..., None]/self.horizon, 1.)
        mean = start+gate[..., None]*(baseline_mean-start
                    + horizons[..., None]*factors*self.mean_network(inputs))
        baseline_covariance = torch.einsum("...ijg,...g->...ij", cc, safe)
        raw = self.covariance_network(inputs).reshape(*horizons.shape, 4, 4)
        transform = torch.eye(4, dtype=torch.float64, device=initial.device)+(horizons/self.horizon)[
            ..., None, None]**self.congruence_powers*raw
        covariance = gate[..., None, None]*(transform @ baseline_covariance @ transform.transpose(-1, -2))
        return (torch.where(active[..., None], mean, start),
                torch.where(active[..., None, None], covariance, 0.))

    def mean_residual(self, initial, horizons):
        initial, horizons = self._inputs(initial, horizons)
        mean = self._mean_jet(initial, horizons)
        source = math.sqrt(self.context.n_particles)*(initial @ self.normalized_running_weights.T)
        source = torch.where(self._active(initial)[..., None], source, 0.)
        return mean.time_derivative-mean.generator-source

    def covariance_residual(self, initial, horizons):
        initial, horizons = self._inputs(initial, horizons)
        origins, coefficients = self._origins(initial, horizons), self._time_coefficients(horizons)
        # Staging makes this target independent of covariance parameters.  This
        # explicit detach also prevents accidental joint-objective regression.
        with torch.no_grad():
            mean = self._mean_jet(initial, horizons, origins, coefficients)
            source = mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2)
        covariance = self._covariance_jet(initial, horizons, origins, coefficients)
        return covariance.time_derivative-covariance.generator-source

    def residual(self, initial, horizons):
        return self.mean_residual(initial, horizons), self.covariance_residual(initial, horizons)

    def predict_conditional(self, initial, horizons):
        result = super().predict_conditional(initial, horizons)
        result.diagnostics.update(payoff_bank=self.bank.to_dict(), payoff_bank_hash=self.bank.bank_hash,
                                  killed_integral=True, covariance_small_time_orders="1,2,3",
                                  covariance_floor=None, heat_baseline="unstopped-analytic-control-only",
                                  baseline_preparation_seconds=self.baseline_preparation_seconds,
                                  staged_mean_frozen=True)
        return result


def _scaling(model, training, cfg, fingerprint):
    with torch.no_grad():
        states = training["states"]
        horizons = torch.full((len(states),), model.horizon, dtype=torch.float64)
        _, _, coefficient, _ = model._time_coefficients(horizons)
        covariance = torch.einsum("bijg,bg->bij", coefficient, states)
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).mean(0).clamp_min(1e-24)
        mean_scale = diagonal.sqrt()
        covariance_scale = (diagonal[:, None]*diagonal[None, :]).sqrt()
    return {"mode": "fixed-train-heat-conditional-std-and-pair-products-v2",
            "mean_scale": mean_scale.tolist(), "covariance_scale": covariance_scale.tolist(),
            "small_time_powers": model.bank.integral_rows.astype(int).tolist(),
            "scale_time_floor": cfg.scale_time_floor, "source_role": "train",
            "training_fingerprint_hash": stable_hash(fingerprint),
            "frozen_before_optimizer": True,
            "floor_interpretation": "loss-normalization-only-no-model-covariance-floor"}


def _objective(model, states, horizons, scaling, stage):
    residual = getattr(model, f"{stage}_residual")(states, horizons)
    power = torch.tensor(scaling["small_time_powers"], dtype=torch.float64)
    time = (horizons/model.horizon).clamp_min(scaling["scale_time_floor"])
    scale = torch.tensor(scaling[f"{stage}_scale"], dtype=torch.float64)
    if stage == "covariance":
        power = power[:, None]+power[None, :]
        time = time[:, None, None]
    else:
        time = time[:, None]
    normalized = model.horizon*residual/scale/time**power
    loss = normalized.square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite repaired-payoff residual")
    return loss, residual


def _validation(model, bank, cfg, scaling, stage):
    result = {"objective": 0., "raw_rms": 0., "raw_maximum": 0., "stage": stage,
              "n_states": len(bank["states"])}
    model.eval()
    with torch.no_grad():
        size = len(bank["states"])
        for start in range(0, size, cfg.study.batch_size):
            end = min(size, start+cfg.study.batch_size)
            loss, residual = _objective(model, bank["states"][start:end], bank["horizons"][start:end],
                                        scaling, stage)
            result["objective"] += (end-start)/size*float(loss)
            result["raw_rms"] += (end-start)/size*float(residual.square().mean())
            result["raw_maximum"] = max(result["raw_maximum"], float(residual.abs().max()))
    result["raw_rms"] **= .5
    return result


def _configuration(config):
    return RepairConfig(config) if isinstance(config, InteractingStudyConfig) else config


def _verify(saved):
    if saved.get("schema_version") != 2 or saved.get("kind") != REPAIRED_PAYOFF_KIND:
        raise ValueError("Expected a repaired v2 staged payoff checkpoint")
    provenance = saved.get("provenance", {})
    identity = {k: v for k, v in provenance.items() if k != "identity_hash"}
    if stable_hash(identity) != provenance.get("identity_hash"):
        raise ValueError("Repaired checkpoint provenance identity mismatch")
    if provenance.get("source_hash") != source_hash():
        raise ValueError("Repaired checkpoint source hash differs from current source")
    cfg, bank = RepairConfig.from_dict(provenance["config"]), PayoffBank.from_dict(provenance["payoff_bank"])
    if (cfg.config_hash != provenance["config_hash"] or bank.bank_hash != provenance["payoff_bank_hash"]
            or provenance["seed"] not in cfg.study.optimizer_seeds
            or stable_hash(provenance["data_fingerprints"]) != provenance["data_hash"]
            or provenance.get("future_moment_labels") is not False
            or provenance.get("training") != "exact-generator-staged-mean-frozen-covariance-v2"):
        raise ValueError("Repaired checkpoint config, bank, seed or data identity mismatch")
    with torch.random.fork_rng():
        torch.manual_seed(provenance["seed"])
        model = RepairPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon,
                                  bank, cfg.study.width, cfg.quadrature_nodes)
    if model.target_id != provenance["target_id"] or model.architecture_metadata != provenance["architecture"]:
        raise ValueError("Repaired checkpoint target or architecture mismatch")
    expected = model.state_dict()
    for name in ("initial_state", "model_state", "best_mean_state", "best_state"):
        state = saved[name]
        if (set(state) != set(expected) or any(not isinstance(state[k], torch.Tensor)
                or state[k].shape != v.shape or not bool(torch.isfinite(state[k]).all())
                for k, v in expected.items()) or _state_hash(state) != saved[f"{name}_hash"]):
            raise ValueError("Repaired checkpoint tensor inventory/checksum mismatch")
        if any(not torch.equal(state[k], v) for k, v in expected.items()
               if not k.startswith(("mean_network.", "covariance_network."))):
            raise ValueError("Repaired checkpoint physical geometry buffers changed")
    if any(not torch.equal(saved["initial_state"][k], v) for k, v in expected.items()):
        raise ValueError("Repaired checkpoint zero-correction initialization changed")
    scaling = provenance["loss_scaling"]
    if (scaling["source_role"] != "train" or not scaling["frozen_before_optimizer"]
            or scaling["training_fingerprint_hash"] != stable_hash(provenance["data_fingerprints"]["training"])
            or scaling["small_time_powers"] != bank.integral_rows.astype(int).tolist()
            or scaling["scale_time_floor"] != cfg.scale_time_floor
            or np.shape(scaling["mean_scale"]) != (4,)
            or np.shape(scaling["covariance_scale"]) != (4, 4)
            or not np.isfinite(scaling["mean_scale"]).all()
            or not np.isfinite(scaling["covariance_scale"]).all()
            or np.any(np.asarray(scaling["mean_scale"]) <= 0)
            or not np.allclose(np.asarray(scaling["covariance_scale"]),
                               np.outer(scaling["mean_scale"], scaling["mean_scale"]), rtol=1e-14)):
        raise ValueError("Repaired checkpoint training-only scales changed")
    for stage in ("mean", "covariance"):
        for opt in ("adam", "lbfgs"):
            n, limit = saved["steps_done"][f"{stage}_{opt}"], getattr(cfg, f"{stage}_{opt}_steps")
            if type(n) is not int or not 0 <= n <= limit:
                raise ValueError("Repaired checkpoint optimizer allocation mismatch")
        if saved["steps_done"][f"{stage}_lbfgs"] and (
                saved["steps_done"][f"{stage}_adam"] != getattr(cfg, f"{stage}_adam_steps")):
            raise ValueError("Repaired checkpoint LBFGS preceded Adam completion")
    finished = all(saved["steps_done"][f"{stage}_{opt}"] == getattr(cfg, f"{stage}_{opt}_steps")
                   for stage in ("mean", "covariance") for opt in ("adam", "lbfgs"))
    if saved["complete"] != finished or saved["stage"] not in ("mean", "covariance"):
        raise ValueError("Repaired checkpoint phase/completion mismatch")
    if ((saved["stage"] == "covariance" and any(
            saved["steps_done"][f"mean_{opt}"] != getattr(cfg, f"mean_{opt}_steps")
            for opt in ("adam", "lbfgs"))) or (saved["stage"] == "mean" and any(
            saved["steps_done"][f"covariance_{opt}"] for opt in ("adam", "lbfgs")))):
        raise ValueError("Repaired checkpoint stages are out of order")
    if saved["stage"] == "covariance":
        for name in ("model_state", "best_state"):
            if any(not torch.equal(saved[name][k], saved["best_mean_state"][k])
                   for k in expected if k.startswith("mean_network.")):
                raise ValueError("Covariance stage changed the frozen selected mean")
    return cfg, bank


def _metadata(saved, path):
    return {"path": str(path.resolve()), "kind": saved["kind"],
            **{k: saved["provenance"][k] for k in ("seed", "target_id", "config_hash", "source_hash",
                                                   "identity_hash", "data_hash", "architecture", "loss_scaling")},
            **{k: saved[k] for k in ("complete", "stage", "steps_done", "initial_validation",
                                     "best_validation", "history", "training_seconds", "stage_seconds",
                                     "parameter_count")},
            "best_step": saved["best_step"],
            "adam_steps": sum(saved["steps_done"][f"{s}_adam"] for s in ("mean", "covariance")),
            "lbfgs_steps": sum(saved["steps_done"][f"{s}_lbfgs"] for s in ("mean", "covariance")),
            "interpretation": "Allocated optimization completion is not statistical certification"}


def fit_repaired_payoff(config, bank, seed, path, train, val, deadline=None):
    """Fit the mean, select it on validation, freeze it, then fit covariance.

    Checkpoints include both optimizer histories/RNGs and the literal pool
    fingerprints. Resume neither restarts a stage nor changes its loss scales.
    """
    cfg = _configuration(config)
    if (not isinstance(cfg, RepairConfig) or type(seed) is not int
            or seed not in cfg.study.optimizer_seeds or path is None):
        raise ValueError("A declared repair configuration, seed and destination are required")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("A finite absolute monotonic deadline is required")
    started = time.monotonic()
    torch.set_num_threads(1)
    training, tf = _bank(train, cfg.study, "train")
    validation, vf = _bank(val, cfg.study, "validation")
    if set(training["family_ids"]) & set(validation["family_ids"]):
        raise ValueError("Training and validation families must be disjoint")
    torch.manual_seed(seed)
    model = RepairPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon,
                              bank, cfg.study.width, cfg.quadrature_nodes)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scaling = _scaling(model, training, cfg, tf)
    provenance = {"kind": model.kind, "seed": seed, "target_id": model.target_id,
                  "config": cfg.to_dict(), "config_hash": cfg.config_hash, "source_hash": source_hash(),
                  "payoff_bank": bank.to_dict(), "payoff_bank_hash": bank.bank_hash,
                  "data_fingerprints": {"training": tf, "validation": vf},
                  "data_hash": stable_hash({"training": tf, "validation": vf}),
                  "architecture": model.architecture_metadata, "future_moment_labels": False,
                  "loss_scaling": scaling, "weight_decay": 0.,
                  "selection": "separate-independent-validation-mean-then-covariance",
                  "training": "exact-generator-staged-mean-frozen-covariance-v2"}
    provenance["identity_hash"] = stable_hash(provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizers = {}
    for stage in ("mean", "covariance"):
        parameters = getattr(model, f"{stage}_network").parameters()
        optimizers[f"{stage}_adam"] = torch.optim.AdamW(parameters, lr=cfg.study.learning_rate,
                                                         weight_decay=0.)
        optimizers[f"{stage}_lbfgs"] = torch.optim.LBFGS(
            getattr(model, f"{stage}_network").parameters(), lr=1., max_iter=cfg.study.lbfgs_max_iter,
            history_size=30, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    shuffle = torch.Generator().manual_seed(0x52455041+seed)
    steps = {f"{s}_{o}": 0 for s in ("mean", "covariance") for o in ("adam", "lbfgs")}
    stage, prior, history = "mean", 0., []
    stage_seconds = {"mean": 0., "covariance": 0.}
    best_mean, best_state = initial.copy(), initial.copy()
    initial_validation = {s: _validation(model, validation, cfg, scaling, s)
                          for s in ("mean", "covariance")}
    best_validation = {s: v.copy() for s, v in initial_validation.items()}
    best_step = {"mean": 0, "covariance": 0}
    if path.exists():
        saved = _read(path)
        _verify(saved)
        if provenance["identity_hash"] != saved["provenance"]["identity_hash"]:
            raise ValueError("Repaired resume source, data, initialization or allocation changed")
        if saved["complete"]:
            return _metadata(saved, path)
        model.load_state_dict(saved["model_state"])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizer_states"][key])
        torch.set_rng_state(saved["torch_rng_state"])
        shuffle.set_state(saved["shuffle_rng_state"])
        steps, stage, prior, history = (saved[k] for k in ("steps_done", "stage", "training_seconds", "history"))
        stage_seconds, best_mean, best_state = (saved[k] for k in ("stage_seconds", "best_mean_state", "best_state"))
        initial_validation, best_validation, best_step = (saved[k] for k in
            ("initial_validation", "best_validation", "best_step"))

    def done(which):
        return all(steps[f"{which}_{o}"] == getattr(cfg, f"{which}_{o}_steps") for o in ("adam", "lbfgs"))

    def save():
        states = {"initial_state": initial, "model_state": model.state_dict(),
                  "best_mean_state": best_mean, "best_state": best_state}
        payload = {"schema_version": 2, "kind": model.kind, "provenance": provenance,
                   **states, **{f"{k}_hash": _state_hash(v) for k, v in states.items()},
                   "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
                   "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": shuffle.get_state(),
                   "steps_done": steps, "stage": stage, "complete": done("mean") and done("covariance"),
                   "initial_validation": initial_validation, "best_validation": best_validation,
                   "best_step": best_step, "history": history, "stage_seconds": stage_seconds,
                   "training_seconds": prior+time.monotonic()-started,
                   "parameter_count": sum(p.numel() for p in model.parameters())}
        _atomic_save(payload, path)

    save()
    size = len(training["states"])
    while not (done("mean") and done("covariance")) and (deadline is None or time.monotonic() < deadline):
        if stage == "mean" and done("mean"):
            model.load_state_dict(best_mean)
            stage = "covariance"
            # Covariance validation is recomputed against the selected mean's
            # Gamma, rather than compared to the obsolete initial mean target.
            best_validation[stage] = _validation(model, validation, cfg, scaling, stage)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for name, network in (("mean", model.mean_network), ("covariance", model.covariance_network)):
            network.requires_grad_(name == stage)
        tick = time.perf_counter()
        opt = "adam" if steps[f"{stage}_adam"] < getattr(cfg, f"{stage}_adam_steps") else "lbfgs"
        optimizer = optimizers[f"{stage}_{opt}"]
        model.train()

        def gradients():
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in getattr(model, f"{stage}_network").parameters()):
                raise FloatingPointError("Nonfinite repaired-payoff parameter gradient")

        if opt == "adam":
            selected = torch.randperm(size, generator=shuffle)[:cfg.study.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _objective(model, training["states"][selected], training["horizons"][selected], scaling, stage)
            loss.backward()
            gradients()
            optimizer.step()
            value = float(loss.detach())
        else:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                total = torch.zeros((), dtype=torch.float64)
                for start in range(0, size, cfg.study.batch_size):
                    end = min(size, start+cfg.study.batch_size)
                    loss, _ = _objective(model, training["states"][start:end], training["horizons"][start:end],
                                         scaling, stage)
                    weight = (end-start)/size
                    (weight*loss).backward()
                    total += weight*loss.detach()
                gradients()
                return total
            value = float(optimizer.step(closure))
        steps[f"{stage}_{opt}"] += 1
        step = steps[f"{stage}_adam"]+steps[f"{stage}_lbfgs"]
        validation_value = None
        if step % cfg.study.validation_every == 0 or done(stage) or (
                opt == "adam" and steps[f"{stage}_adam"] == getattr(cfg, f"{stage}_adam_steps")):
            validation_value = _validation(model, validation, cfg, scaling, stage)
            if validation_value["objective"] < best_validation[stage]["objective"]:
                best_validation[stage], best_step[stage] = validation_value, step
                selected_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if stage == "mean":
                    best_mean = selected_state
                else:
                    best_state = selected_state
        seconds = time.perf_counter()-tick
        stage_seconds[stage] += seconds
        history.append({"stage": stage, "optimizer": opt, "step": step,
                        "training_objective": value, "validation": validation_value,
                        "step_seconds": seconds})
        save()
    return _metadata(_read(path), path)


def load_repaired_payoff_model(path):
    path = Path(path)
    saved = _read(path)
    cfg, bank = _verify(saved)
    if not saved["complete"]:
        raise ValueError("Cannot evaluate an incomplete repaired checkpoint")
    with torch.random.fork_rng():
        torch.manual_seed(saved["provenance"]["seed"])
        model = RepairPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon,
                                  bank, cfg.study.width, cfg.quadrature_nodes)
    model.load_state_dict(saved["best_state"])
    model.provenance, model.checkpoint_path = saved["provenance"], str(path.resolve())
    model.config, model.complete, model.loss_scaling = cfg, True, saved["provenance"]["loss_scaling"]
    model.mean_network.requires_grad_(False)
    model.eval()
    return model
