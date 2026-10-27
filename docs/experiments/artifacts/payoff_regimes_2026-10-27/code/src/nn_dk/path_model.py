"""Causal, fixed-weight neural solution operators for density trajectories.

The neural network parameterizes a full-resolution spectral Volterra kernel.
Known finite-volume drift and physical face-noise mobility remain in the
equation.  A fixed number of parallel-in-time Picard blocks introduces the
nonlinearity of DK; no parameters are fitted during a realization.

This is a discrete-time ansatz.  Its pole constraint and features depend on
``dt``; a checkpoint at one resolution does not establish convergence at another.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from .native_types import NativeDKSpec
from .path_numerics import PATH_PROBLEMS, divergence, path_coefficients, torch_stop_paths
from .types import Context, seeded_rng, stable_hash

PATH_MODEL_KIND = "causal-full-grid-volterra-picard-v1"


class CausalPathOperator(nn.Module):
    """Map an initial mass field and physical Brownian increments to raw paths.

    ``forward(initial, increments, dt, context)`` returns ``[paths, steps+1,
    cells]``, including the exact initial state. Every real Fourier mode is
    retained, including Nyquist. The mass mode is fixed. A lower-triangular
    Volterra contraction enforces causality in the computational graph.

    For heat and additive noise the kernel contains their exact stable Euler
    response. Its initial poles correspond to HALF the known instantaneous heat
    decay rates, so training changes full trajectory predictions. The initial
    kernel is not an oracle future solution. A learned bounded lag correction
    permits departure from a semigroup kernel.

    For DK, each Picard block evaluates the KNOWN interaction and mobility on
    the preceding block's stopped path. Intermediates are frozen at their first
    crossing, never clipped or renormalized. The returned path remains raw;
    the caller applies the declared stopping rule and retains exit proposals.
    """

    kind = PATH_MODEL_KIND

    def __init__(self, spec: NativeDKSpec, problem: str = "heat", width: int = 32,
                 depth: int = 3):
        super().__init__()
        if not isinstance(spec, NativeDKSpec):
            raise TypeError("A NativeDKSpec is required")
        if problem not in PATH_PROBLEMS:
            raise ValueError(f"Use one of {PATH_PROBLEMS}")
        for value, name in ((width, "width"), (depth, "depth")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"A positive integer {name} is required")
        self.spec, self.problem, self.width, self.depth = spec, problem, width, depth
        frequencies = torch.arange(spec.n_cells // 2 + 1, dtype=torch.float64)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("laplacian_rates", 4 / spec.h**2 * torch.sin(
            math.pi * frequencies / spec.n_cells).square())
        self.kernel_network = nn.Sequential(
            nn.Linear(7, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2))
        self.double()
        # Pole correction and lag correction both start at zero. Hidden layers
        # retain random features and become trainable as the last layer moves.
        nn.init.zeros_(self.kernel_network[-1].weight)
        nn.init.zeros_(self.kernel_network[-1].bias)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.frequencies.device)

    def _check_context(self, context):
        if not isinstance(context, Context):
            raise TypeError("Declared Context physics, not oracle path metadata, is required")
        if self.problem != "dk" and (context.kappa != 0 or context.a2 != 0):
            raise ValueError("Heat/additive controls require zero interaction")

    def kernel_values(self, n_steps: int, dt: float, context: Context) -> torch.Tensor:
        """Return the learned causal lag kernel, including lag zero, in rFFT space.

        Positive Euler poles avoid an unstable hidden linear response. The
        supported resolution has ``dt*lambda_max < 1``; this includes the
        declared 32-cell step ladder starting at ``2**-7``. All lower Fourier
        modes, not just the four reporting observables, enter this kernel.
        """
        self._check_context(context)
        if (not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1
                or not math.isfinite(dt) or dt <= 0):
            raise ValueError("A positive step count and finite positive timestep are required")
        physical_rates = context.diffusion * self.laplacian_rates
        pole_margin = 1e-8
        scaled_rates = dt * physical_rates
        if bool((scaled_rates >= 1 - pole_margin).any()):
            raise ValueError("The positive-pole neural kernel requires dt*lambda_max < 1")
        times = torch.arange(n_steps + 1, dtype=torch.float64,
                             device=self.frequencies.device) * dt
        n_lags, n_modes = len(times), len(self.frequencies)
        frequency = self.frequencies / (self.spec.n_cells / 2)
        sine_squared = torch.sin(math.pi * self.frequencies / self.spec.n_cells).square()
        features = torch.stack(torch.broadcast_tensors(
            frequency[None], sine_squared[None], times[:, None],
            self._tensor(context.kappa).expand(n_lags, n_modes),
            self._tensor(context.a2).expand(n_lags, n_modes),
            self._tensor(context.diffusion).expand(n_lags, n_modes),
            self._tensor(dt).expand(n_lags, n_modes)), dim=-1)
        output = self.kernel_network(features)
        # The cap and baseline use only instantaneous, declared diffusion.
        # At initialization lambda_theta = .5*lambda_physical. At nonzero
        # frequencies the exact Euler pole is inside this parameter family.
        baseline = .5 * scaled_rates / (1 - pole_margin)
        baseline = torch.where(self.frequencies == 0, torch.full_like(baseline, .5), baseline)
        logits = torch.logit(baseline) + output[0, :, 0]
        poles = 1 - (1 - pole_margin) * torch.sigmoid(logits)
        powers = torch.arange(n_steps + 1, dtype=torch.float64,
                              device=self.frequencies.device)[:, None]
        correction = 1 + .1 * (times / (1 + times))[:, None] * torch.tanh(output[..., 1])
        kernel = poles[None].pow(powers) * correction
        # No learned correction or noise enters the conserved zero mode.
        return torch.where(self.frequencies[None] == 0, torch.ones_like(kernel), kernel)

    def _response(self, initial, forcing, kernel):
        """Apply a lower-triangular causal convolution, retaining every grid mode."""
        n_steps = forcing.shape[1]
        initial_hat = torch.fft.rfft(initial, dim=-1)
        forcing_hat = torch.fft.rfft(forcing, dim=-1)
        nonzero = (self.frequencies != 0).to(torch.float64)
        forcing_hat = forcing_hat * nonzero
        positions = torch.arange(n_steps, device=initial.device)
        lags = positions[:, None] - positions[None, :]
        # An actual lower triangular matrix, rather than a temporal FFT with
        # canceling future terms, gives exact causal dependency and gradients.
        convolution = kernel[lags.clamp_min(0)] * (lags >= 0)[..., None]
        stochastic_hat = torch.einsum("njf,pjf->pnf", convolution.to(forcing_hat.dtype),
                                     forcing_hat)
        update_hat = ((kernel[1:][None] - 1) * initial_hat[:, None]
                      + stochastic_hat) * nonzero
        subsequent = initial[:, None] + torch.fft.irfft(update_hat, n=self.spec.n_cells, dim=-1)
        return torch.cat((initial[:, None], subsequent), dim=1)

    def forward(self, initial, increments, dt: float, context: Context) -> torch.Tensor:
        self._check_context(context)
        initial, increments = self._tensor(initial), self._tensor(increments)
        if (initial.ndim != 2 or increments.ndim != 3 or initial.shape[0] < 1
                or initial.shape[-1] != self.spec.n_cells
                or increments.shape[0] != initial.shape[0]
                or increments.shape[-1] != self.spec.n_cells or increments.shape[1] < 1):
            raise ValueError("Initial [paths,cells] and physical increments [paths,steps,faces] required")
        if not bool(torch.isfinite(initial).all() and torch.isfinite(increments).all()):
            raise ValueError("Initial states and physical Brownian increments must be finite")
        if not bool(torch.allclose(initial.sum(-1), torch.ones_like(initial[:, 0]),
                                   atol=1e-10, rtol=0)):
            raise ValueError("Initial cell masses must sum to one; no renormalization is used")
        if self.problem == "dk" and bool((initial < 0).any()):
            raise ValueError("DK initial states must be nonnegative")
        n_steps = increments.shape[1]
        kernel = self.kernel_values(n_steps, dt, context)
        zeros = torch.zeros_like(increments)
        if self.problem == "heat":
            return self._response(initial, zeros, kernel)
        if self.problem == "additive":
            _, amplitudes = path_coefficients(initial[:, None], context, self.spec, "additive")
            forcing = divergence(amplitudes * increments / math.sqrt(context.n_particles))
            return self._response(initial, forcing, kernel)
        previous = self._response(initial, zeros, kernel)
        for _ in range(self.depth):
            # Previous Picard outputs can cross the physical boundary. Evaluate
            # mobility on their stopped paths, with NO sqrt extension or clipping.
            stopped = torch_stop_paths(initial, previous[:, 1:], self.spec, dt)
            origins = stopped["states"][:, :-1]
            # An initially absorbed histogram may contain zero adjacent cells.
            # Its coefficients never enter the pre-stop equation. Evaluate a
            # safe arbitrary origin there to avoid masked sqrt'(0) in backward.
            active = stopped["active"][..., None]
            evaluation_origins = torch.where(active, origins,
                                              torch.full_like(origins, 1 / self.spec.n_cells))
            full_drift, amplitudes = path_coefficients(evaluation_origins, context, self.spec, "dk")
            heat_drift = context.diffusion / self.spec.h**2 * (
                torch.roll(evaluation_origins, 1, dims=-1)
                + torch.roll(evaluation_origins, -1, dims=-1) - 2 * evaluation_origins)
            amplitudes = torch.where(active, amplitudes, torch.zeros_like(amplitudes))
            interaction_drift = torch.where(active, full_drift - heat_drift,
                                             torch.zeros_like(full_drift))
            forcing = dt * interaction_drift + divergence(
                amplitudes * increments / math.sqrt(context.n_particles))
            previous = self._response(initial, forcing, kernel)
        return previous

    def sample(self, context, times, n_paths=1, seed=0, *, step_size=None,
               initial_masses=None, brownian_increments=None, family_ids=None):
        """Generate density realizations with fixed weights and explicit face noise.

        Returned states have shape ``(paths, requested_times, cells)``. The full
        uncensored proposals and integration grid are retained separately.
        Loaded checkpoints reject a different step, physics or trained horizon;
        changing the numerical resolution requires a separately validated fit.
        """
        from .native import initialize_masses

        started = time.perf_counter()
        self._check_context(context)
        if type(n_paths) is not int or n_paths < 1 or type(seed) is not int or seed < 0:
            raise ValueError("Positive path count and nonnegative integer seed required")
        dt = getattr(self, "dt", None) if step_size is None else step_size
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Supply the fixed integration step for an unfitted operator")
        if hasattr(self, "dt") and dt != self.dt:
            raise ValueError("Checkpoint timestep differs; refit for a different noise resolution")
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ValueError("Finite increasing nonnegative sample times required")
        indices = np.rint(times/dt).astype(int)
        if not np.allclose(indices*dt, times, atol=1e-12, rtol=0):
            raise ValueError("Sample times must lie on the checkpoint integration grid")
        if hasattr(self, "path_config"):
            if context != self.path_config.context(self.case):
                raise ValueError("Checkpoint physical/initial-law context is incompatible")
            if times[-1] > self.path_config.horizon+1e-12:
                raise ValueError("Sample time exceeds the trained horizon")
        steps = int(indices[-1])
        if initial_masses is None:
            initial_seed = int(seeded_rng(seed, self.kind, "initial").integers(0, 2**63-1))
            initial_masses = initialize_masses(context, self.spec, n_paths, initial_seed)
        initial_masses = np.asarray(initial_masses, dtype=np.float64)
        if (initial_masses.shape != (n_paths, self.spec.n_cells)
                or not np.isfinite(initial_masses).all() or np.any(initial_masses < 0)
                or not np.allclose(initial_masses.sum(-1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Nonnegative unit-mass initial histograms must match path count")
        supplied_noise = brownian_increments is not None
        if not supplied_noise:
            brownian_increments = seeded_rng(seed, self.kind, context.context_id,
                                              "physical-face-noise").normal(
                0, math.sqrt(dt), (n_paths, steps, self.spec.n_cells))
        brownian_increments = np.asarray(brownian_increments, dtype=np.float64)
        if (brownian_increments.shape != (n_paths, steps, self.spec.n_cells)
                or not np.isfinite(brownian_increments).all()):
            raise ValueError("Physical Brownian increments must match paths, steps and faces")
        families = (tuple(stable_hash([seed, self.kind, context.context_id, i])[:24]
                          for i in range(n_paths)) if family_ids is None else tuple(family_ids))
        if len(families) != n_paths or len(set(families)) != n_paths:
            raise ValueError("Unique trajectory-family identifiers must match the path count")
        self.eval()
        with torch.no_grad():
            if steps:
                raw = self(initial_masses, brownian_increments, dt, context)
                packet = torch_stop_paths(initial_masses, raw[:, 1:], self.spec, dt,
                                          stopping=self.problem == "dk")
                states, proposals = packet["states"].numpy(), raw[:, 1:].numpy()
                stops = packet["stopped_at"].numpy()
            else:
                states = initial_masses[:, None].copy()
                proposals = np.empty((n_paths, 0, self.spec.n_cells))
                stops = np.where(np.any(initial_masses <= self.spec.mass_stop, axis=-1)
                                 & (self.problem == "dk"), 0., np.inf)
        return {"states": states[:, indices], "times": times,
                "raw_proposals": proposals, "integration_times": np.arange(steps+1)*dt,
                "stopped_at": stops, "family_ids": families,
                "target_id": getattr(self, "target_id", None),
                "diagnostics": {"kind": self.kind, "problem": self.problem,
                    "full_density_paths": True, "fixed_weights_during_simulation": True,
                    "supplied_physical_noise": supplied_noise,
                    "maximum_mass_error": float(np.max(np.abs(states.sum(-1)-1))),
                    "minimum_mass": float(np.min(states)),
                    "stopped_paths": int(np.sum(stops <= times[-1])),
                    "inference_seconds": time.perf_counter()-started}}
