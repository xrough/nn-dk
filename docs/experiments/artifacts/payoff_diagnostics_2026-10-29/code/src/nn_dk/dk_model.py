"""Autonomous conservative coefficient networks for native stopped discrete DK.

This is a generator approximation, not the legacy finite-lag particle surrogate.
Covariances returned here are quadratic variations *before* division by N.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native_types import DKCoefficients, NativeDKSpec, physics_features
from .observations import incidence_matrix, tangent_basis
from .types import Context

COEFFICIENT_MODEL_KIND = "native-dk-autonomous-diagonal-face-v1"


class PhysicsNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(3, dtype=torch.float64))

    def fit(self, physics):
        x = torch.as_tensor(physics, dtype=torch.float64, device=self.center.device)
        if x.ndim != 2 or x.shape[1] != 3 or not torch.isfinite(x).all():
            raise ValueError("Finite (kappa, a2, diffusion) training features required")
        with torch.no_grad():
            self.center.copy_(x.mean(0))
            scale = x.std(0, unbiased=False)
            self.scale.copy_(torch.where(scale > 1e-12, scale, torch.ones_like(scale)))

    def forward(self, x):
        return (x - self.center) / self.scale


def finite_volume_mode_weights(spec: NativeDKSpec) -> np.ndarray:
    """Cell averages of sqrt(2) cos(kx), sin(kx), k=1,2."""
    left = np.arange(spec.n_cells) * spec.h
    right = left + spec.h
    return np.sqrt(2) * np.asarray([
        value for k in (1, 2) for value in (
            (np.sin(k * right) - np.sin(k * left)) / (k * spec.h),
            (np.cos(k * left) - np.cos(k * right)) / (k * spec.h))])


class DKCoefficientModel(nn.Module):
    """Autonomous native DK coefficients with equivariant nonlocal features.

    Four exact finite-volume Fourier convolution fields at faces provide the
    global interaction information while preserving periodic shift equivariance.
    """
    kind = COEFFICIENT_MODEL_KIND

    def __init__(self, spec: NativeDKSpec, width: int = 32):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("Positive integer CNN width required")
        self.spec, self.width = spec, width
        self.register_buffer("incidence", torch.as_tensor(incidence_matrix(spec.n_cells)))
        self.register_buffer("tangent", torch.as_tensor(tangent_basis(spec.n_cells)))
        centers = (np.arange(spec.n_cells) + 0.5) * spec.h
        faces = (np.arange(spec.n_cells) + 1) * spec.h
        difference = faces[:, None] - centers[None, :]
        kernels = np.asarray([np.sinc(k * spec.h / (2 * np.pi)) * function(k * difference)
                              for k in (1, 2) for function in (np.cos, np.sin)])
        self.register_buffer("convolution_weights", torch.as_tensor(kernels))
        self.normalization = PhysicsNormalization()
        self.hidden = nn.Sequential(
            nn.Conv1d(8, width, 5, padding=2, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2, padding_mode="circular"), nn.SiLU())
        self.output = nn.Conv1d(width, 2, 1)
        self.double()
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        # Natural FV units; no additive noise/covariance floor is present.
        self.flux_unit = 1 / (spec.n_cells * spec.h**2)
        self.noise_unit = np.sqrt(2 / (spec.n_cells * spec.h**2))

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float64, device=self.incidence.device)

    def _inputs(self, masses, physics):
        masses = self._tensor(masses)
        if masses.ndim < 1 or masses.shape[-1] != self.spec.n_cells:
            raise ValueError("Native masses must end in the fitted finite-volume grid")
        leading = masses.shape[:-1]
        flat = masses.reshape(-1, self.spec.n_cells)
        physics = physics_features(physics) if isinstance(physics, Context) else physics
        features = self._tensor(physics)
        if features.ndim < 1 or features.shape[-1] != 3:
            raise ValueError("Autonomous physics must be (kappa, a2, diffusion)")
        try:
            features = torch.broadcast_to(features, (*leading, 3)).reshape(-1, 3)
        except RuntimeError as exc:
            raise ValueError("Physics batch shape must match native masses") from exc
        if not torch.isfinite(flat).all() or not torch.isfinite(features).all() or (features[:, 2] <= 0).any():
            raise ValueError("Finite state/physics and positive diffusion required")
        # Integrate cos/sin(k(x_face-y)) against the reconstructed current
        # density. Constant absolute Fourier moments would hide the phase at
        # each face and could not represent the nonlocal DK interaction drift.
        fields = torch.einsum("bj,cij->bci", flat, self.convolution_weights)
        physics_channels = self.normalization(features)[:, :, None].expand(-1, -1, self.spec.n_cells)
        channels = torch.cat(((self.spec.n_cells * flat)[:, None], fields, physics_channels), 1)
        return channels, leading

    def _parts(self, masses, physics):
        channels, leading = self._inputs(masses, physics)
        outputs = self.output(self.hidden(channels))
        flux = outputs[:, 0] * self.flux_unit
        amplitudes = F.softplus(outputs[:, 1]) * self.noise_unit
        drift = flux @ self.incidence.T
        g = self.spec.n_cells
        return (drift.reshape(*leading, g), amplitudes.reshape(*leading, g),
                flux.reshape(*leading, g))

    def coefficients(self, masses, physics) -> DKCoefficients:
        drift, amplitudes, flux = self._parts(masses, physics)
        factor = self.incidence * amplitudes[..., None, :]
        covariance = factor @ factor.transpose(-1, -2)
        return DKCoefficients(drift, covariance, amplitudes, flux)

    def forward(self, masses, physics):
        return self.coefficients(masses, physics)

    def proposal_residuals(self, states, proposals, physics, dt, n_particles):
        """Whiten uncensored Euler proposals on the fixed mass-zero subspace."""
        states, proposals = self._tensor(states), self._tensor(proposals)
        if states.ndim != 2 or states.shape != proposals.shape:
            raise ValueError("Matching (samples, cells) proposal origins/responses required")
        count = len(states)
        dt = torch.broadcast_to(self._tensor(dt), (count,))
        n_particles = torch.broadcast_to(self._tensor(n_particles), (count,))
        if (dt <= 0).any() or (n_particles <= 0).any() or not torch.isfinite(dt).all() or not torch.isfinite(n_particles).all():
            raise ValueError("Positive finite timestep and particle count required")
        coefficients = self.coefficients(states, physics)
        residual = (proposals - states - dt[:, None] * coefficients.drift) @ self.tangent
        covariance = (self.tangent.T @ coefficients.covariance @ self.tangent
                      * (dt / n_particles)[:, None, None])
        # An invalid covariance is a model failure; no hidden jitter or spectral clipping.
        chol = torch.linalg.cholesky(covariance)
        white = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
        logdet = 2 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
        nll = 0.5 * (white.square().sum(-1) + logdet
                     + (self.spec.n_cells - 1) * np.log(2 * np.pi))
        return white, nll

    def nll(self, states, proposals, physics, dt, n_particles):
        return self.proposal_residuals(states, proposals, physics, dt, n_particles)[1].mean()

    def numpy_coefficients(self, masses, context):
        self.eval()
        with torch.no_grad():
            c = self.coefficients(masses, context)
        return DKCoefficients(*(x.detach().cpu().numpy() if x is not None else None
                                for x in (c.drift, c.covariance, c.face_amplitudes, c.flux)))

    def sampling_coefficients(self, masses, context):
        """Backend adapter avoids materializing covariance during every Euler step."""
        self.eval()
        with torch.no_grad():
            drift, amplitudes, flux = self._parts(masses, context)
        return DKCoefficients(drift.cpu().numpy(), None, amplitudes.cpu().numpy(), flux.cpu().numpy())

    def sample(self, context, times, n_paths, seed, step_size, ablation="both",
               initial_masses=None, brownian_increments=None, family_ids=None):
        from .native import simulate_native
        return simulate_native(context, self.spec, times, step_size, seed, n_paths=n_paths,
                               coefficients=self.sampling_coefficients, ablation=ablation,
                               initial_masses=initial_masses, brownian_increments=brownian_increments,
                               family_ids=family_ids)
