"""Conditional moments through a value network and predictable BSDE integrands.

Only the value network is evaluated at inference. The second network is a
training-time martingale control, driven by the SAME increments as the forward
SDE. Neither oracle coefficient labels nor future path information are inputs.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native import initialize_masses, physical_observable_weights
from .observations import tangent_basis
from .types import Context, QueryBank

DK_BSDE_MODEL_KIND = "native-dk-bsde-moments-v1"
OU_BSDE_MODEL_KIND = "linear-ou-bsde-moments-v1"
MEAN_DIMENSION = 4
SECOND_DIMENSION = 10
PAYOFF_DIMENSION = MEAN_DIMENSION + SECOND_DIMENSION


def svec(matrix: torch.Tensor) -> torch.Tensor:
    """Isometric symmetric vectorization with sqrt(2)-weighted off diagonals."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError("svec requires symmetric four-by-four matrices")
    indices = torch.tril_indices(4, 4, device=matrix.device)
    weights = torch.ones(10, dtype=matrix.dtype, device=matrix.device)
    weights[indices[0] != indices[1]] = np.sqrt(2.)
    return matrix[..., indices[0], indices[1]] * weights


def moment_payoff(observables: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    difference = observables - center
    return torch.cat((observables, svec(difference.unsqueeze(-1) * difference.unsqueeze(-2))), -1)


def _mlp(n_inputs, n_outputs, width):
    layers = []
    for _ in range(3):
        layers.extend((nn.Linear(n_inputs, width), nn.SiLU()))
        n_inputs = width
    layers.append(nn.Linear(width, n_outputs))
    return nn.Sequential(*layers).double()


class BSDEMomentModel(nn.Module):
    """Horizon-conditional mean and positive joint same-time covariance.

    Covariance matrices used internally refer to fluctuation-scaled observables.
    Public DK outputs undo that scaling: means are physical, covariance is N Cov.
    The OU calibration uses identity observables and N=1.
    """

    def __init__(self, config, problem="dk"):
        super().__init__()
        if problem not in ("dk", "ou"):
            raise ValueError("BSDE problem must be dk or ou")
        self.config, self.problem = config, problem
        self.kind = DK_BSDE_MODEL_KIND if problem == "dk" else OU_BSDE_MODEL_KIND
        self.driver_dimension = config.spec.n_cells if problem == "dk" else 4
        self.state_dimension = self.driver_dimension
        self.feature_dimension = self.state_dimension - 1 if problem == "dk" else 4
        if problem == "dk":
            weights = physical_observable_weights(config.spec, QueryBank())
            std = np.sqrt(np.mean(weights**2, axis=1) - np.mean(weights, axis=1)**2)
            self.register_buffer("observation_weights", torch.tensor(weights))
            self.register_buffer("observation_std", torch.tensor(std))
            self.register_buffer("basis", torch.tensor(tangent_basis(config.spec.n_cells)))
        else:
            self.register_buffer("observation_weights", torch.eye(4, dtype=torch.float64))
            self.register_buffer("observation_std", torch.ones(4, dtype=torch.float64))
            self.register_buffer("basis", torch.eye(4, dtype=torch.float64))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dimension, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(self.feature_dimension, dtype=torch.float64))
        self.value_network = _mlp(self.feature_dimension + 1, PAYOFF_DIMENSION, config.width)
        # State, remaining time, requested horizon, and deterministic initial query center.
        self.integrand_network = _mlp(self.feature_dimension + 6,
                                      PAYOFF_DIMENSION * self.driver_dimension, config.width)
        with torch.no_grad():
            self.value_network[-1].weight.zero_()
            self.value_network[-1].bias.zero_()
            indices = torch.tril_indices(4, 4)
            diagonal = (indices[0] == indices[1]).nonzero().flatten() + 4
            self.value_network[-1].bias[diagonal] = np.log(np.expm1(1.))
            self.integrand_network[-1].weight.zero_()
            self.integrand_network[-1].bias.zero_()
        self.provenance = {}

    @property
    def target_id(self):
        return self.config.target_id(self.problem)

    def raw_features(self, state):
        if self.problem == "dk":
            scale = np.sqrt(self.config.n_particles * self.config.spec.n_cells)
            return scale * (state - 1 / self.config.spec.n_cells) @ self.basis
        return state

    def normalized_features(self, state):
        return (self.raw_features(state) - self.feature_mean) / self.feature_scale

    def fit_normalization(self, initial, anchor_ids):
        """Fit on one initial state per TRAINING anchor, never its continuations."""
        seen, unique = {}, []
        for i, anchor in enumerate(anchor_ids):
            if anchor in seen:
                if not torch.equal(initial[i], initial[seen[anchor]]):
                    raise ValueError("Continuations of an anchor have different initial states")
            else:
                seen[anchor] = i
                unique.append(i)
        with torch.no_grad():
            features = self.raw_features(initial[unique])
            self.feature_mean.copy_(features.mean(0))
            scale = features.std(0, unbiased=False)
            self.feature_scale.copy_(torch.where(scale > 1e-12, scale, torch.ones_like(scale)))

    def scaled_observables(self, state):
        values = state @ self.observation_weights.T
        if self.problem == "dk":
            values = np.sqrt(self.config.n_particles) * values / self.observation_std
        return values

    def initially_stopped(self, initial):
        if self.problem == "dk":
            return torch.any(initial <= self.config.spec.mass_stop, dim=-1)
        return torch.zeros(initial.shape[:-1], dtype=torch.bool, device=initial.device)

    def deterministic_center(self, initial, horizons):
        """Euler-heat center for second moments, excluded from value inputs."""
        from .bsde_data import ou_rates

        rates = torch.as_tensor(ou_rates(self.config), dtype=torch.float64,
                                device=initial.device)
        steps = horizons / self.config.dt
        decay = torch.pow(1 - self.config.dt * rates, steps.unsqueeze(-1))
        y0 = self.scaled_observables(initial)
        center = decay * y0
        return torch.where(self.initially_stopped(initial).unsqueeze(-1), y0, center)

    def moment_values(self, initial, horizons):
        """Return scaled means/covariances for one horizon per initial state."""
        if initial.shape[-1] != self.state_dimension:
            raise ValueError("Initial state dimension does not match BSDE target")
        horizons = torch.broadcast_to(horizons, initial.shape[:-1])
        features = torch.cat((self.normalized_features(initial), horizons.unsqueeze(-1)), -1)
        output = self.value_network(features)
        y0 = self.scaled_observables(initial)
        mean = y0 + horizons.unsqueeze(-1) * output[..., :4]
        indices = torch.tril_indices(4, 4, device=initial.device)
        factor = output.new_zeros((*output.shape[:-1], 4, 4))
        values = output[..., 4:]
        diagonal = indices[0] == indices[1]
        values = torch.where(diagonal, F.softplus(values), values)
        factor[..., indices[0], indices[1]] = values
        covariance = horizons[..., None, None] * factor @ factor.transpose(-1, -2)
        frozen = self.initially_stopped(initial) | (horizons == 0)
        mean = torch.where(frozen.unsqueeze(-1), y0, mean)
        covariance = torch.where(frozen[..., None, None], 0., covariance)
        return mean, covariance

    def initial_payoff(self, initial, horizons, center):
        mean, covariance = self.moment_values(initial, horizons)
        difference = mean - center
        second = covariance + difference.unsqueeze(-1) * difference.unsqueeze(-2)
        return torch.cat((mean, svec(second)), -1)

    def integrands(self, states, remaining, horizons, center):
        """Predictable inputs only; vectorizes all leading path/time dimensions."""
        features = torch.cat((self.normalized_features(states), remaining.unsqueeze(-1),
                              horizons.unsqueeze(-1), center), -1)
        return self.integrand_network(features).reshape(
            *states.shape[:-1], PAYOFF_DIMENSION, self.driver_dimension)

    def _validate_initial(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None]
        if states.ndim != 2 or states.shape[-1] != self.state_dimension or not np.all(np.isfinite(states)):
            raise ValueError("Finite initial states with shape (batch,state dimension) required")
        if self.problem == "dk" and (np.any(states < 0) or not np.allclose(states.sum(-1), 1, atol=1e-10, rtol=0)):
            raise ValueError("DK initial masses must be nonnegative and conserve unit mass")
        return states

    def _validate_horizons(self, horizons):
        horizons = np.atleast_1d(np.asarray(horizons, dtype=np.float64))
        if horizons.ndim != 1 or not len(horizons) or not np.all(np.isfinite(horizons)):
            raise ValueError("A nonempty one-dimensional finite horizon bank is required")
        if np.any(np.diff(horizons) <= 0):
            raise ValueError("Prediction horizons must be unique and increasing")
        declared = (0., *self.config.query_times)
        if any(not any(np.isclose(t, allowed, atol=1e-12, rtol=0) for allowed in declared) for t in horizons):
            raise ValueError("Only time zero and the declared trained horizons are supported")
        return horizons

    def predict_conditional(self, masses, horizons):
        from .bsde_types import MomentPrediction

        started = time.perf_counter()
        states, horizons = self._validate_initial(masses), self._validate_horizons(horizons)
        self.eval()
        with torch.no_grad():
            initial = torch.as_tensor(states, dtype=torch.float64)[:, None, :].expand(-1, len(horizons), -1)
            times = torch.as_tensor(horizons, dtype=torch.float64)[None, :].expand(len(states), -1)
            mean, covariance = self.moment_values(initial, times)
            if self.problem == "dk":
                mean = mean * self.observation_std / np.sqrt(self.config.n_particles)
                covariance = covariance * self.observation_std[:, None] * self.observation_std[None, :]
        prediction = MomentPrediction(mean=mean.cpu().numpy(), scaled_covariance=covariance.cpu().numpy(),
                                      horizons=tuple(horizons), problem=self.problem, target_id=self.target_id,
                                      diagnostics={"kind": self.kind, "n_initial": len(states),
                                                   "conditional": True,
                                                   "cross_time_covariance_supported": False})
        prediction.diagnostics["inference_seconds"] = time.perf_counter() - started
        return prediction

    def predict_ensemble(self, context, query_bank, n_initial, seed):
        from .bsde_types import MomentPrediction

        if self.problem != "dk":
            raise ValueError("OU calibration supports explicit conditional initial states only")
        if not isinstance(context, Context) or (context.kappa, context.a2) != (0., 0.):
            raise ValueError("This BSDE target supports only the noninteracting native DK context")
        if context.n_particles != self.config.n_particles or context.diffusion != self.config.diffusion:
            raise ValueError("Context physics differs from the fixed BSDE target")
        if query_bank.modes != (1, 2):
            raise ValueError("The first BSDE model supports only Fourier modes one and two")
        if not isinstance(n_initial, (int, np.integer)) or n_initial < 2:
            raise ValueError("At least two independent initial histograms required")
        in_training_family = any(
            len(context.cos_coeffs) == len(training_context.cos_coeffs)
            and np.allclose(context.cos_coeffs, training_context.cos_coeffs, atol=1e-12, rtol=0)
            and np.allclose(context.sin_coeffs, training_context.sin_coeffs, atol=1e-12, rtol=0)
            for training_context in (self.config.context(case) for case in self.config.cases))
        started = time.perf_counter()
        initial = initialize_masses(context, self.config.spec, n_initial, seed)
        initialization_seconds = time.perf_counter() - started
        conditional = self.predict_conditional(initial, query_bank.times)
        mean = conditional.mean.mean(0)
        centered = conditional.mean - mean
        between = np.einsum("bhi,bhj->hij", centered, centered) / (n_initial - 1)
        covariance = conditional.scaled_covariance.mean(0) + self.config.n_particles * between
        prediction = MomentPrediction(mean=mean, scaled_covariance=covariance,
                                      horizons=conditional.horizons, problem=self.problem,
                                      target_id=self.target_id,
                                      diagnostics={"kind": self.kind, "conditional": False,
                                                   "n_initial": n_initial, "seed": int(seed),
                                                   "initialization_seconds": initialization_seconds,
                                                   "inference_seconds": conditional.diagnostics["inference_seconds"],
                                                   "initial_sampling_covariance_included": True,
                                                   "initial_profile_ood": not in_training_family,
                                                   "ood_generalization_certified": False,
                                                   "cross_time_covariance_supported": False})
        prediction.diagnostics["total_prediction_seconds"] = time.perf_counter() - started
        return prediction
