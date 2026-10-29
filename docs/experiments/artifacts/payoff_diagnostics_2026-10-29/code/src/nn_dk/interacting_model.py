"""Full-state conditional moment solver for interacting stopped native DK.

Unlike the noninteracting time-only family, this ansatz retains all mass-zero
coordinates.  The known Itô generator supplies residuals, not future moment
labels.  Hard initial and absorbing-boundary values and a Gram head preserve
basic consistency; finite collocation accuracy is not a convergence proof.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from .bsde_types import MomentPrediction
from .generator_jets import (
    GeneratorJet,
    jet_add,
    jet_cat,
    jet_constant,
    jet_gram,
    jet_linear,
    jet_multiply,
    jet_scale,
    jet_sequential,
    jet_slice,
    jet_softplus,
    jet_tanh,
    jet_unsqueeze,
)
from .native import native_coefficients, physical_observable_weights
from .native_types import NativeDKSpec
from .observations import incidence_matrix, tangent_basis
from .path_numerics import path_coefficients
from .types import Context, QueryBank, seeded_rng

INTERACTING_MOMENT_MODEL_KIND = "continuous-interacting-native-dk-generator-moments-v1"


class InteractingMomentModel(nn.Module):
    """Autonomous-physics conditional means and PSD same-time covariances.

    Internal means are sqrt(N) times normalized observations; internal
    covariances are their ordinary covariances.  Public predictions return
    physical means and N-scaled physical covariance, matching MomentPrediction.
    """

    kind = INTERACTING_MOMENT_MODEL_KIND
    architecture = "full-state-continuous-generator"

    def __init__(self, spec: NativeDKSpec, context: Context, horizon: float, width: int = 32):
        super().__init__()
        if not isinstance(spec, NativeDKSpec) or not isinstance(context, Context):
            raise TypeError("Native grid and fixed physical context are required")
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("A finite positive trained horizon is required")
        if type(width) is not int or width < 2:
            raise ValueError("Integer network width >=2 required")
        self.spec, self.context = spec, context
        self.horizon, self.width = float(horizon), width
        self.boundary_width = .05 / spec.n_cells
        self.state_dimension, self.feature_dimension = spec.n_cells, spec.n_cells - 1
        self.problem = "dk"
        weights = physical_observable_weights(spec, QueryBank())
        scales = np.sqrt(np.mean(weights**2, axis=1))
        self.register_buffer("observation_weights", torch.tensor(weights))
        self.register_buffer("observation_std", torch.tensor(scales))
        self.register_buffer("normalized_weights", torch.tensor(weights / scales[:, None]))
        self.register_buffer("basis", torch.tensor(tangent_basis(spec.n_cells)))
        self.register_buffer("incidence", torch.tensor(incidence_matrix(spec.n_cells)))
        indices = torch.tril_indices(4, 4)
        self.register_buffer("factor_rows", indices[0])
        self.register_buffer("factor_columns", indices[1])
        layers, inputs = [], spec.n_cells
        for _ in range(3):
            layers.extend((nn.Linear(inputs, width), nn.SiLU()))
            inputs = width
        layers.append(nn.Linear(width, 14))
        self.network = nn.Sequential(*layers).double()
        uniform = np.full(spec.n_cells, 1 / spec.n_cells)
        instantaneous = native_coefficients(uniform, context, spec).covariance
        normalized = weights / scales[:, None]
        factor = np.linalg.cholesky(normalized @ instantaneous @ normalized.T)
        entries = torch.tensor(factor)[indices[0], indices[1]]
        diagonal = indices[0] == indices[1]
        entries[diagonal] = entries[diagonal] + torch.log(-torch.expm1(-entries[diagonal]))
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()
            self.network[-1].bias[4:] = entries
        self.provenance = {}

    @property
    def target_id(self):
        from .interacting_config import interacting_target_id

        return interacting_target_id(self.spec, self.context, self.horizon)

    def require_physics(self, context: Context):
        if not isinstance(context, Context) or any(
                getattr(context, name) != getattr(self.context, name)
                for name in ("n_particles", "diffusion", "kappa", "a2")):
            raise ValueError("Context physics differs from the fixed interacting target")

    @property
    def architecture_metadata(self):
        return {"architecture": self.architecture, "width": self.width,
                "hidden_layers": 3, "activation": "SiLU",
                "state_features": "all-sqrt(NG)-scaled-mass-zero-coordinates",
                "boundary_gate": "product-tanh-positive-boundary-strip",
                "boundary_width": self.boundary_width,
                "covariance": "time-times-gate-times-Cholesky-Gram-no-floor",
                "trace": "exact-all-native-face-directions"}

    def _inputs(self, initial, horizons):
        initial = torch.as_tensor(initial, dtype=torch.float64, device=self.basis.device)
        horizons = torch.as_tensor(horizons, dtype=torch.float64, device=self.basis.device)
        if initial.ndim < 1 or initial.shape[-1] != self.spec.n_cells or not initial.numel():
            raise ValueError("Native initial states must have a final cell axis")
        if (not bool(torch.isfinite(initial).all()) or bool((initial < 0).any())
                or not bool(torch.allclose(initial.sum(-1), torch.ones_like(initial[..., 0]),
                                           atol=1e-10, rtol=0))):
            raise ValueError("Initial masses must be finite, nonnegative and sum to one")
        horizons = torch.broadcast_to(horizons, initial.shape[:-1])
        if (not bool(torch.isfinite(horizons).all()) or bool((horizons < 0).any())
                or bool((horizons > self.horizon).any())):
            raise ValueError("Horizons must lie in the finite trained interval")
        return initial, horizons

    def _active(self, initial):
        return (initial > self.spec.mass_stop).all(-1)

    def scaled_observables(self, initial):
        return math.sqrt(self.context.n_particles) * (initial @ self.normalized_weights.T)

    def barrier(self, initial):
        """Smooth absorbing strip, almost one away from its fixed boundary width.

        The width is an ansatz conditioning choice, not a change to the target
        SDE or its stopping threshold. Boundary/corner approximation errors still
        require independent evaluation.
        """
        return torch.tanh((initial - self.spec.mass_stop) / self.boundary_width).prod(-1)

    def _factor(self, raw):
        diagonal = self.factor_rows == self.factor_columns
        entries = torch.where(diagonal, torch.logaddexp(torch.zeros_like(raw), raw), raw)
        factor = raw.new_zeros((*raw.shape[:-1], 4, 4))
        factor[..., self.factor_rows, self.factor_columns] = entries
        return factor

    def _values_unchecked(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial,
                           torch.full_like(initial, 1 / self.spec.n_cells))
        features = math.sqrt(self.context.n_particles * self.spec.n_cells) * (
            (safe - 1 / self.spec.n_cells) @ self.basis)
        raw = self.network(torch.cat((features, (horizons / self.horizon).unsqueeze(-1)), -1))
        weight = horizons * self.barrier(safe)
        initial_mean = self.scaled_observables(initial)
        mean = initial_mean + weight[..., None] * raw[..., :4]
        factor = self._factor(raw[..., 4:])
        covariance = weight[..., None, None] * (factor @ factor.transpose(-1, -2))
        return (torch.where(active[..., None], mean, initial_mean),
                torch.where(active[..., None, None], covariance, 0.))

    def moment_values(self, initial, horizons):
        initial, horizons = self._inputs(initial, horizons)
        return self._values_unchecked(initial, horizons)

    def forward(self, initial, horizons):
        return self.moment_values(initial, horizons)

    def _factor_jet(self, raw):
        diagonal = self.factor_rows == self.factor_columns
        entries = jet_add(jet_scale(raw, ~diagonal), jet_scale(jet_softplus(raw), diagonal))
        values = []
        for name in ("value", "time_derivative", "generator", "face_derivatives"):
            source = getattr(entries, name)
            shape = (*source.shape[:-1], 4, 4) if name != "face_derivatives" else (
                *source.shape[:-2], 4, 4, source.shape[-1])
            target = source.new_zeros(shape)
            if name == "face_derivatives":
                target[..., self.factor_rows, self.factor_columns, :] = source
            else:
                target[..., self.factor_rows, self.factor_columns] = source
            values.append(target)
        return GeneratorJet(*values)

    def _moment_jets(self, initial, horizons):
        active = self._active(initial)
        safe = torch.where(active[..., None], initial,
                           torch.full_like(initial, 1 / self.spec.n_cells))
        drift, amplitudes = path_coefficients(safe, self.context, self.spec)
        face = self.incidence * amplitudes.unsqueeze(-2) / math.sqrt(self.context.n_particles)
        state = GeneratorJet(safe, torch.zeros_like(safe), drift, face)
        scale = math.sqrt(self.context.n_particles * self.spec.n_cells)
        projection = scale * self.basis.T
        bias = -torch.full_like(safe[0], 1 / self.spec.n_cells) @ projection.T
        features = jet_linear(state, projection, bias)
        time_jet = jet_constant(horizons, self.spec.n_cells)
        time_jet = GeneratorJet(time_jet.value, torch.ones_like(horizons),
                                time_jet.generator, time_jet.face_derivatives)
        inputs = jet_cat([features, jet_unsqueeze(jet_scale(time_jet, 1 / self.horizon), -1)])
        raw = jet_sequential(inputs, self.network)
        ratios = jet_tanh(jet_add(jet_scale(state, 1 / self.boundary_width),
                                  jet_constant(torch.full_like(safe, -self.spec.mass_stop
                                               / self.boundary_width), self.spec.n_cells)))
        barrier = jet_constant(torch.ones_like(horizons), self.spec.n_cells)
        for cell in range(self.spec.n_cells):
            barrier = jet_multiply(barrier, jet_slice(ratios, (..., cell)))
        weight = jet_multiply(time_jet, barrier)
        initial_mean = jet_linear(state, math.sqrt(self.context.n_particles)
                                  * self.normalized_weights)
        mean = jet_add(initial_mean, jet_multiply(jet_unsqueeze(weight, -1),
                                                  jet_slice(raw, (..., slice(0, 4)))))
        covariance = jet_multiply(jet_unsqueeze(jet_unsqueeze(weight, -1), -1),
                                  jet_gram(self._factor_jet(jet_slice(raw, (..., slice(4, None))))))
        # On absorbed states the target is constant in time and its generator
        # is zero. Safe coefficient origins above are only unused calculations.
        actual_initial_mean = self.scaled_observables(initial)
        mean = GeneratorJet(torch.where(active[..., None], mean.value, actual_initial_mean),
                            torch.where(active[..., None], mean.time_derivative, 0.),
                            torch.where(active[..., None], mean.generator, 0.),
                            torch.where(active[..., None, None], mean.face_derivatives, 0.))
        covariance = GeneratorJet(
            torch.where(active[..., None, None], covariance.value, 0.),
            torch.where(active[..., None, None], covariance.time_derivative, 0.),
            torch.where(active[..., None, None], covariance.generator, 0.),
            torch.where(active[..., None, None, None], covariance.face_derivatives, 0.))
        return mean, covariance

    def residual(self, initial, horizons):
        initial, horizons = self._inputs(initial, horizons)
        if initial.ndim != 2:
            raise ValueError("Residual collocation requires [batch,cells] states")
        mean, covariance = self._moment_jets(initial, horizons)
        source = mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2)
        return (mean.time_derivative - mean.generator,
                covariance.time_derivative - covariance.generator - source)

    def predict_conditional(self, initial, horizons):
        started = time.perf_counter()
        initial = np.asarray(initial, dtype=np.float64)
        if initial.ndim == 1:
            initial = initial[None]
        horizons = np.atleast_1d(np.asarray(horizons, dtype=np.float64))
        if (initial.ndim != 2 or horizons.ndim != 1 or not len(horizons)
                or np.any(np.diff(horizons) <= 0)):
            raise ValueError("A batch of states and unique increasing horizons are required")
        states = torch.as_tensor(initial, dtype=torch.float64, device=self.basis.device)
        times = torch.as_tensor(horizons, dtype=torch.float64, device=self.basis.device)
        self.eval()
        with torch.no_grad():
            mean, covariance = self.moment_values(
                states[:, None, :].expand(-1, len(horizons), -1),
                times[None, :].expand(len(states), -1))
            mean = mean * self.observation_std / math.sqrt(self.context.n_particles)
            covariance = covariance * self.observation_std[:, None] * self.observation_std[None, :]
        return MomentPrediction(mean.cpu().numpy(), covariance.cpu().numpy(), horizons, "dk",
                                self.target_id, {"kind": self.kind, "conditional": True,
                                    "continuous_target": True,
                                    "cross_time_covariance_supported": False,
                                    "boundary_values": "hard-absorbing-tanh-product-strip",
                                    "boundary_width": self.boundary_width,
                                    "generator_trace": "all-native-face-directions-exact",
                                    "inference_seconds": time.perf_counter() - started,
                                    "accuracy_certified": False})

    def predict_ensemble(self, probabilities, horizons, n_samples=4096, seed=0):
        """Integrate nonlinear conditional predictions by independent IID histograms.

        This is Monte Carlo over the known initialization law, with no future
        trajectory labels. Its sampling uncertainty is separate from model and
        reference errors and must be audited by the experiment evaluator.
        """
        started = time.perf_counter()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (probabilities.shape != (self.spec.n_cells,)
                or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("A finite nonnegative normalized cell-probability vector is required")
        if (isinstance(n_samples, (bool, np.bool_))
                or not isinstance(n_samples, (int, np.integer)) or n_samples < 2):
            raise ValueError("At least two independent initial histogram samples are required")
        if (isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("A nonnegative integer initialization sampling seed is required")
        rng = seeded_rng(int(seed), "interacting-initial-mc", self.target_id)
        initial = rng.multinomial(self.context.n_particles, probabilities,
                                  size=int(n_samples)) / self.context.n_particles
        initialization_seconds = time.perf_counter() - started
        conditional = self.predict_conditional(initial, horizons)
        mean = conditional.mean.mean(0)
        centered = conditional.mean - mean
        between = np.einsum("bhi,bhj->hij", centered, centered) / (n_samples - 1)
        covariance = conditional.scaled_covariance.mean(0) + self.context.n_particles * between
        mean_standard_error = conditional.mean.std(0, ddof=1) / math.sqrt(n_samples)
        return MomentPrediction(mean, covariance, conditional.horizons, "dk", self.target_id,
                                {"kind": self.kind, "conditional": False,
                                 "initial_law_integration": "independent-multinomial-monte-carlo",
                                 "initial_samples": int(n_samples), "initial_seed": int(seed),
                                 "initialization_seconds": initialization_seconds,
                                 "inference_seconds": conditional.diagnostics["inference_seconds"],
                                 "total_prediction_seconds": time.perf_counter() - started,
                                 "initial_mean_standard_error": mean_standard_error.tolist(),
                                 "initial_sampling_covariance_included": True,
                                 "initial_sampling_uncertainty_certified": False,
                                 "cross_time_covariance_supported": False,
                                 "accuracy_certified": False})
