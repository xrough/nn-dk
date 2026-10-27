"""Cached, causal fixed-weight inference and matched nonlearned Picard controls."""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import torch

from .frozen_models import current_context, load_frozen_model
from .native_types import NativeDKSpec
from .path_acceleration import prepare_response, vectorized_stop_paths
from .path_model import CausalPathOperator
from .types import stable_hash


class PreparedPathRuntime:
    """Same learned response with faster algebra; exact-kernel control is separate.

    Cache preparation is explicit and must be charged on a new (step,horizon,
    physics) query. Gradients in initial states/noise reuse the fixed plan.
    Physics derivatives construct a fresh differentiable plan.
    """

    def __init__(self, model, context, dt, horizon, *, mode="dense-cached", learned=True,
                 depth=None, provenance=None):
        self.model, self.context = model, context
        self.dt, self.horizon = float(dt), float(horizon)
        if (not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.horizon)
                or self.horizon <= 0):
            raise ValueError("Finite positive integration step and horizon required")
        self.spec, self.problem = model.spec, model.problem
        model._check_context(context)
        if 4*context.diffusion*self.dt/self.spec.h**2 >= 1-1e-8:
            raise ValueError("The positive-pole response requires a finer step")
        self.mode, self.learned = mode, bool(learned)
        self.depth = model.depth if depth is None else depth
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("Positive Picard depth required")
        self.provenance = dict(provenance or {})
        self.kind = ("prepared-learned-causal-picard-v1" if learned else "exact-kernel-classical-picard-v1")
        self.target_id = stable_hash({"native_spec": asdict(self.spec), "physics": asdict(context),
                                     "dt": self.dt, "horizon": self.horizon})
        self._plans = {}
        cells = torch.arange(self.spec.n_cells, dtype=torch.float64)
        centers, faces = (cells+.5)*self.spec.h, (cells+1)*self.spec.h
        self.moment_matrix = torch.stack((centers.cos(), centers.sin(), (2*centers).cos(),
                                          (2*centers).sin()), -1)
        self.reconstruction = torch.stack((faces.sin(), -faces.cos(), (2*faces).sin(),
                                           -(2*faces).cos()), 0)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path, commit, *, mode="dense-cached", learned=True, depth=None):
        frozen, provenance = load_frozen_model(path, commit, kind="path")
        spec = NativeDKSpec(**asdict(frozen.spec))
        model = CausalPathOperator(spec, frozen.problem, frozen.width, frozen.depth)
        model.load_state_dict(frozen.state_dict())
        context = current_context(frozen.path_config.context(frozen.case))
        return cls(model, context, frozen.dt, frozen.path_config.horizon, mode=mode,
                   learned=learned, depth=depth, provenance=provenance)

    def kernel(self, steps, physics=None):
        if type(steps) is not int or steps < 1:
            raise ValueError("Positive integer step count required")
        if physics is not None:
            physics = self._physics(physics)
        if physics is None:
            if self.learned:
                return self.model.kernel_values(steps, self.dt, self.context)
            rates = self.context.diffusion*self.model.laplacian_rates
            return (1-self.dt*rates)[None].pow(torch.arange(steps+1, dtype=torch.float64)[:, None])
        diffusion, kappa, a2 = physics.unbind()
        rates = diffusion*self.model.laplacian_rates
        if bool((self.dt*rates >= 1-1e-8).any()):
            raise ValueError("Positive-pole response requires a finer step")
        powers = torch.arange(steps+1, dtype=torch.float64)[:, None]
        if not self.learned:
            return (1-self.dt*rates)[None].pow(powers)
        frequencies = self.model.frequencies
        times = torch.arange(steps+1, dtype=torch.float64)*self.dt
        shape = (steps+1, len(frequencies))
        features = torch.stack(torch.broadcast_tensors(
            (frequencies/(self.spec.n_cells/2))[None],
            torch.sin(math.pi*frequencies/self.spec.n_cells).square()[None],
            times[:, None], kappa.expand(shape), a2.expand(shape), diffusion.expand(shape),
            torch.full(shape, self.dt, dtype=torch.float64)), -1)
        output = self.model.kernel_network(features)
        baseline = .5*self.dt*rates/(1-1e-8)
        baseline = torch.where(frequencies == 0, .5, baseline)
        poles = 1-(1-1e-8)*torch.sigmoid(torch.logit(baseline)+output[0, :, 0])
        correction = 1+.1*(times/(1+times))[:, None]*torch.tanh(output[..., 1])
        result = poles[None].pow(powers)*correction
        return torch.where(frequencies[None] == 0, 1., result)

    def prepare(self, steps):
        if steps not in self._plans:
            with torch.no_grad():
                self._plans[steps] = prepare_response(self.kernel(steps), self.spec.n_cells, mode=self.mode)
        return self._plans[steps]

    @staticmethod
    def _physics(physics):
        value = torch.as_tensor(physics, dtype=torch.float64)
        if (value.shape != (3,) or not bool(torch.isfinite(value).all()) or value[0] <= 0):
            raise ValueError("Finite [diffusion,kappa,a2] with positive diffusion required")
        return value

    def coefficients(self, states, physics=None):
        if physics is None:
            diffusion, kappa, a2 = (self.context.diffusion, self.context.kappa, self.context.a2)
        else:
            physics = self._physics(physics)
            diffusion, kappa, a2 = physics.unbind()
        following = torch.roll(states, -1, -1)
        face_density = (states+following)/(2*self.spec.h)
        scale = torch.stack((-torch.as_tensor(kappa, dtype=torch.float64)
                             *np.sinc(self.spec.h/(2*np.pi)),)*2
                            +(2*torch.as_tensor(a2, dtype=torch.float64)
                              *np.sinc(self.spec.h/np.pi),)*2)
        velocity = ((states@self.moment_matrix)*scale)@self.reconstruction
        flux = face_density*velocity-diffusion/self.spec.h**2*(following-states)
        drift = torch.roll(flux, 1, -1)-flux
        amplitude = torch.sqrt(2*diffusion/self.spec.h*face_density)
        return drift, amplitude

    def __call__(self, initial, increments, *, physics=None, return_raw=False):
        if physics is not None:
            physics = self._physics(physics)
            if self.problem != "dk" and bool((physics[1:] != 0).any()):
                raise ValueError("Heat/additive controls require zero interaction")
        initial = torch.as_tensor(initial, dtype=torch.float64)
        increments = torch.as_tensor(increments, dtype=torch.float64)
        if (initial.ndim != 2 or increments.ndim != 3 or initial.shape[0] < 1
                or initial.shape[0] != increments.shape[0]
                or initial.shape[-1] != self.spec.n_cells or increments.shape[-1] != self.spec.n_cells
                or increments.shape[1] < 1 or increments.shape[1]*self.dt > self.horizon+1e-12
                or not bool(torch.isfinite(initial).all() and torch.isfinite(increments).all())
                or bool((initial < 0).any())
                or not bool(torch.allclose(initial.sum(-1), torch.ones_like(initial[:, 0]), atol=1e-10, rtol=0))):
            raise ValueError("Compatible nonnegative unit-mass states, finite forcing and trained horizon required")
        steps = increments.shape[1]
        plan = (self.prepare(steps) if physics is None else
                prepare_response(self.kernel(steps, physics), self.spec.n_cells, mode=self.mode))
        zeros = torch.zeros_like(increments)
        if self.problem == "heat":
            raw = plan.apply(initial, zeros)
        elif self.problem == "additive":
            diffusion = self.context.diffusion if physics is None else physics[0]
            amplitude = torch.sqrt(torch.as_tensor(2*diffusion/(self.spec.n_cells*self.spec.h**2),
                                                   dtype=torch.float64))
            face = amplitude*increments/math.sqrt(self.context.n_particles)
            raw = plan.apply(initial, torch.roll(face, 1, -1)-face)
        else:
            raw = plan.apply(initial, zeros)
            for _ in range(self.depth):
                packet = vectorized_stop_paths(initial, raw[:, 1:], self.spec, self.dt)
                active = packet["active"][..., None]
                origins = torch.where(active, packet["states"][:, :-1], 1/self.spec.n_cells)
                drift, amplitude = self.coefficients(origins, physics)
                diffusion = self.context.diffusion if physics is None else physics[0]
                heat = diffusion/self.spec.h**2*(torch.roll(origins, 1, -1)
                                                +torch.roll(origins, -1, -1)-2*origins)
                face = torch.where(active, amplitude, 0.)*increments/math.sqrt(self.context.n_particles)
                forcing = self.dt*torch.where(active, drift-heat, 0.)+torch.roll(face, 1, -1)-face
                raw = plan.apply(initial, forcing)
        packet = vectorized_stop_paths(initial, raw[:, 1:], self.spec, self.dt, stopping=self.problem == "dk")
        return (packet, raw) if return_raw else packet["states"]
