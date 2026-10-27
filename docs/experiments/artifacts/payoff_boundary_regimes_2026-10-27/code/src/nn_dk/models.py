"""Conservative fixed-lag neural dynamics and fixed-query statistical models.

The Gaussian likelihood lives on the mass-zero subspace. No full-space jitter,
state clipping, or survivor-conditioned statistical estimates are used here.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.linalg import helmert
from torch import nn
from torch.nn import functional as F

from .types import (
    Context,
    QueryBank,
    StatisticalPrediction,
    TargetSpec,
    TrajectoryBatch,
    seeded_rng,
)


def _geometry(target: TargetSpec):
    from .observations import incidence_matrix, tangent_basis, uniform_quadratic_variation
    return (incidence_matrix(target.n_cells), tangent_basis(target.n_cells),
            uniform_quadratic_variation(target, 1.0))


def _initial(context: Context, target: TargetSpec, n_paths: int, seed: int) -> np.ndarray:
    from .observations import observe_particles
    from .particles import sample_initial
    particles = sample_initial(context, n_paths, seeded_rng(seed, "learned-initial"))
    return observe_particles(particles, target)


def _check_times(times, lag):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Finite increasing nonnegative sample times required")
    steps = np.rint(times / lag).astype(int)
    if not np.all(np.isfinite(times)) or not np.allclose(times, steps * lag, atol=1e-10, rtol=0):
        raise ValueError("Sample times must be multiples of the trained finite lag")
    return times, steps


class ContextNormalization(nn.Module):
    def __init__(self, dimension=10):
        super().__init__()
        self.register_buffer("center", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))

    def fit(self, features):
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != len(self.center):
            raise ValueError("v0.1 requires the three-mode initial-profile family")
        self.center.copy_(x.mean(0))
        deviation = x.std(0, unbiased=False)
        self.scale.copy_(torch.where(deviation > 1e-6, deviation, torch.ones_like(deviation)))

    def forward(self, x):
        return (x - self.center) / self.scale


class ConservativeTransition(nn.Module):
    """A finite-lag Markov surrogate; changing ``lag`` requires a new fit."""
    def __init__(self, target: TargetSpec, lag: float = 1 / 64, width: int = 32):
        super().__init__()
        if not np.isfinite(lag) or lag <= 0:
            raise ValueError("lag must be positive and finite")
        self.target, self.lag, self.width = target, float(lag), width
        self.supported_particles: tuple[int, ...] = ()
        b, q, c = _geometry(target)
        self.register_buffer("incidence", torch.as_tensor(b, dtype=torch.float64))
        self.register_buffer("tangent", torch.as_tensor(q, dtype=torch.float64))
        self.register_buffer("uniform_covariance", torch.as_tensor(c, dtype=torch.float64))
        # The uniform observation quadratic variation is positive on the tangent.
        reduced = q.T @ c @ q
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        if eigenvalues.min() < -1e-10:
            raise ValueError("Uniform quadratic variation must be positive semidefinite")
        self.register_buffer("uniform_factor", torch.as_tensor(
            q @ (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))), dtype=torch.float64))
        self.register_buffer("mode_weights", torch.as_tensor(QueryBank().weights(target.n_cells), dtype=torch.float32))
        self.normalization = ContextNormalization()
        self.hidden = nn.Sequential(
            nn.Conv1d(16, width, 5, padding=2, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), nn.SiLU())
        self.output = nn.Conv1d(width, 6, 1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def inputs(self, states, contexts, elapsed):
        """Inputs depend on the present state and declared metadata only."""
        states = states.to(dtype=torch.float32)
        contexts = contexts.to(dtype=torch.float32)
        if contexts.shape[-1] != 10:
            raise ValueError("v0.1 requires ten declared context features")
        count, m = states.shape
        low = states @ self.mode_weights.T
        elapsed = torch.as_tensor(elapsed, dtype=torch.float32, device=states.device)
        if elapsed.ndim == 0:
            elapsed = elapsed.expand(count)
        globals_ = torch.cat((low, self.normalization(contexts), elapsed[:, None]), dim=1)
        return torch.cat(((m * states - 1)[:, None], globals_[:, :, None].expand(-1, -1, m)), dim=1)

    def components(self, states, contexts, elapsed):
        raw = self.output(self.hidden(self.inputs(states, contexts, elapsed)))
        m = self.target.n_cells
        current = raw[:, 0].double() / m
        diagonals = raw[:, 1:].double()
        diagonal = F.softplus(diagonals[:, 2])
        diagonals = torch.cat((diagonals[:, :2], diagonal[:, None], diagonals[:, 3:]), dim=1)
        eye = torch.eye(m, dtype=torch.float64, device=raw.device)
        factor = sum(diagonals[:, k, :, None] * torch.roll(eye, k - 2, dims=1)
                     for k in range(5))
        drift = current @ self.incidence.T
        noise_factor = self.incidence @ factor
        diffusion = contexts[:, 8].double()
        covariance = noise_factor @ noise_factor.transpose(-1, -2)
        covariance = covariance + 1e-6 * diffusion[:, None, None] * self.uniform_covariance
        return drift, covariance, noise_factor

    def forward(self, states, contexts, elapsed):
        drift, covariance, _ = self.components(states, contexts, elapsed)
        return drift, covariance

    def nll(self, states, next_states, contexts, elapsed):
        """Mean Gaussian negative log likelihood in M-1 orthonormal coordinates."""
        drift, covariance = self(states, contexts, elapsed)
        n_particles = contexts[:, 9].double().exp()
        residual = (next_states.double() - states.double() - self.lag * drift) @ self.tangent
        reduced = self.tangent.T @ covariance @ self.tangent
        reduced = self.lag / n_particles[:, None, None] * reduced
        chol = torch.linalg.cholesky(reduced)
        white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return (0.5 * (white.square().sum(-1) + logdet + (self.target.n_cells - 1) * np.log(2 * np.pi))).mean()

    def sample(self, context: Context, times, n_paths: int, seed: int) -> TrajectoryBatch:
        if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        if self.supported_particles and context.n_particles not in self.supported_particles:
            raise ValueError("Particle count is outside the fitted model support; retrain for transfer")
        times, steps = _check_times(times, self.lag)
        started = time.perf_counter()
        current = _initial(context, self.target, n_paths, seed)
        initialization_seconds = time.perf_counter() - started
        minimum_mass = float(current.min())
        maximum_mass_error = float(np.abs(current.sum(-1) - 1).max())
        result = np.full((n_paths, len(times), self.target.n_cells), np.nan)
        valid = np.ones(n_paths, dtype=bool)
        failures = np.full(n_paths, np.nan)
        rng = seeded_rng(seed, "learned-innovations")
        features = torch.as_tensor(np.tile(context.features(), (n_paths, 1)), dtype=torch.float32)
        requested = {s: i for i, s in enumerate(steps)}
        if 0 in requested:
            result[:, requested[0]] = current
        self.eval()
        with torch.no_grad():
            for step in range(1, int(steps[-1]) + 1):
                active = np.flatnonzero(valid)
                if len(active):
                    states = torch.as_tensor(current[active], dtype=torch.float64)
                    drift, _, factor = self.components(states, features[active], (step - 1) * self.lag)
                    z = torch.as_tensor(rng.normal(size=(len(active), self.target.n_cells)), dtype=torch.float64)
                    z0 = torch.as_tensor(rng.normal(size=(len(active), self.target.n_cells - 1)), dtype=torch.float64)
                    innovation = (factor @ z[..., None]).squeeze(-1)
                    innovation += np.sqrt(1e-6 * context.diffusion) * z0 @ self.uniform_factor.T
                    proposal = states + self.lag * drift + np.sqrt(self.lag / context.n_particles) * innovation
                    current[active] = proposal.numpy()
                    minimum_mass = min(minimum_mass, float(np.nanmin(current[active])))
                    maximum_mass_error = max(maximum_mass_error, float(np.nanmax(np.abs(current[active].sum(-1) - 1))))
                    failed = (~np.isfinite(current[active]).all(1)) | (current[active].min(1) < -1e-12)
                    valid[active[failed]] = False
                    failures[active[failed]] = step * self.lag
                if step in requested:
                    result[valid, requested[step]] = current[valid]
                if not valid.any():
                    break
        return TrajectoryBatch(times, result, self.target, valid, failures, {
            "method": "neural_transition", "lag": self.lag, "context_id": context.context_id,
            "initialization_seconds": initialization_seconds,
            "inference_seconds": time.perf_counter() - started - initialization_seconds,
            "invalid_paths": int((~valid).sum()), "survivor_conditioning": False,
            "minimum_mass": minimum_mass, "maximum_mass_error": maximum_mass_error})


class DirectMomentRegressor(nn.Module):
    def __init__(self, target: TargetSpec, query_bank: QueryBank, width: int = 64):
        super().__init__()
        self.target, self.query_bank, self.width = target, query_bank, width
        self.supported_particles: tuple[int, ...] = ()
        d = query_bank.dimension
        self.normalization = ContextNormalization()
        self.hidden = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.mean_head = nn.Linear(width, d)
        self.covariance_head = nn.Linear(width, d * (d + 1) // 2)
        self.register_buffer("mean_center", torch.zeros(d))
        self.register_buffer("mean_scale", torch.ones(d))
        self.register_buffer("covariance_scale", torch.ones(()))
        rows, columns = torch.tril_indices(d, d)
        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_columns", columns)

    def forward(self, contexts):
        h = self.hidden(self.normalization(contexts.float()))
        mean = self.mean_center + self.mean_scale * self.mean_head(h)
        raw = self.covariance_head(h)
        d = self.query_bank.dimension
        lower = raw.new_zeros((len(raw), d, d))
        lower[:, self.tril_rows, self.tril_columns] = raw
        diag = torch.arange(d, device=raw.device)
        lower[:, diag, diag] = F.softplus(lower[:, diag, diag]) + 1e-6
        lower = lower.double()
        covariance = self.covariance_scale.double() * lower @ lower.transpose(-1, -2)
        return mean, covariance

    def predict(self, context: Context, query_bank: QueryBank) -> StatisticalPrediction:
        return self.predict_batch([context], query_bank)[0]

    def predict_batch(self, contexts: list[Context], query_bank: QueryBank) -> list[StatisticalPrediction]:
        """One neural forward pass for independent prediction-time contexts."""
        if query_bank != self.query_bank:
            raise ValueError("Direct model supports only its fitted fixed query bank")
        contexts = list(contexts)
        if not contexts:
            return []
        if self.supported_particles and any(c.n_particles not in self.supported_particles for c in contexts):
            raise ValueError("Particle count is outside the fitted model support")
        started = time.perf_counter()
        self.eval()
        with torch.no_grad():
            mean, covariance = self(torch.as_tensor(np.stack([c.features() for c in contexts]), dtype=torch.float32))
        elapsed = time.perf_counter() - started
        return [StatisticalPrediction(mean[i].numpy(), covariance[i].numpy(), self.target, query_bank,
                    context.context_id, {"method": "direct_neural", "inference_seconds": elapsed / len(contexts),
                                         "batch_inference_seconds": elapsed, "batch_size": len(contexts)})
                for i, context in enumerate(contexts)]


class GaussianAutoregression:
    """Linear Gaussian dynamics in orthonormal mass-zero coordinates."""
    def __init__(self, target, lag, coefficients, covariance, center, scale, supported_particles):
        self.target, self.lag = target, lag
        self.coefficients, self.covariance = coefficients, covariance
        self.center, self.scale = center, scale
        self.supported_particles = tuple(supported_particles)
        self.tangent = helmert(target.n_cells).T

    def sample(self, context, times, n_paths, seed):
        if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        times, steps = _check_times(times, self.lag)
        if context.n_particles not in self.supported_particles:
            raise ValueError("Particle count outside fitted autoregression support")
        started = time.perf_counter()
        r = _initial(context, self.target, n_paths, seed)
        init_seconds = time.perf_counter() - started
        minimum_mass = float(r.min())
        maximum_mass_error = float(np.abs(r.sum(-1) - 1).max())
        states = np.full((n_paths, len(times), self.target.n_cells), np.nan)
        valid, failures = np.ones(n_paths, bool), np.full(n_paths, np.nan)
        requested = dict(zip(steps, range(len(steps))))
        if 0 in requested:
            states[:, requested[0]] = r
        eig, vectors = np.linalg.eigh(self.covariance)
        factor = vectors * np.sqrt(np.maximum(eig, 0))
        rng = seeded_rng(seed, "autoregression")
        c = (context.features() - self.center) / self.scale
        for s in range(1, int(steps[-1]) + 1):
            active = np.flatnonzero(valid)
            if len(active):
                z = r[active] @ self.tangent
                design = np.column_stack((np.ones(len(active)), z, np.tile(c, (len(active), 1)), np.full(len(active), (s - 1) * self.lag)))
                dz = design @ self.coefficients
                dz += np.sqrt(self.lag / context.n_particles) * rng.normal(size=z.shape) @ factor.T
                r[active] += dz @ self.tangent.T
                minimum_mass = min(minimum_mass, float(np.nanmin(r[active])))
                maximum_mass_error = max(maximum_mass_error, float(np.nanmax(np.abs(r[active].sum(-1) - 1))))
                failed = (~np.isfinite(r[active]).all(1)) | (r[active].min(1) < -1e-12)
                failures[active[failed]], valid[active[failed]] = s * self.lag, False
            if s in requested:
                states[valid, requested[s]] = r[valid]
            if not valid.any():
                break
        return TrajectoryBatch(times, states, self.target, valid, failures, {
            "method": "autoregression", "context_id": context.context_id, "initialization_seconds": init_seconds,
            "inference_seconds": time.perf_counter() - started - init_seconds,
            "invalid_paths": int((~valid).sum()), "survivor_conditioning": False,
            "minimum_mass": minimum_mass, "maximum_mass_error": maximum_mass_error})


class RidgeMomentRegressor:
    def __init__(self, target, query_bank, method, center, scale, x_train, coefficients,
                 supported_particles, gamma=0.1, alpha=0.01):
        self.target, self.query_bank, self.method = target, query_bank, method
        self.center, self.scale, self.x_train = center, scale, x_train
        self.coefficients, self.gamma, self.alpha = coefficients, gamma, alpha
        self.supported_particles = tuple(supported_particles)

    @staticmethod
    def polynomial(x):
        i, j = np.triu_indices(x.shape[1])
        return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))

    @staticmethod
    def kernel(x, y, gamma):
        distance = np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None] - 2*x@y.T, 0)
        return np.exp(-gamma * distance)

    def predict(self, context, query_bank):
        if query_bank != self.query_bank or context.n_particles not in self.supported_particles:
            raise ValueError("Context or query bank outside fitted baseline support")
        started = time.perf_counter()
        x = (context.features()[None] - self.center) / self.scale
        design = self.polynomial(x) if self.method == "polynomial_ridge" else self.kernel(x, self.x_train, self.gamma)
        output = (design @ self.coefficients)[0]
        d = query_bank.dimension
        covariance = output[d:].reshape(d, d)
        covariance = (covariance + covariance.T) / 2
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        return StatisticalPrediction(output[:d], covariance, self.target, query_bank, context.context_id,
            {"method": self.method, "covariance_minimum_eigenvalue": minimum_eigenvalue,
             "covariance_indefinite": minimum_eigenvalue < -1e-10,
             "inference_seconds": time.perf_counter()-started})
