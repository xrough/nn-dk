"""Time-only neural moments trained against the known continuous generator.

The network supplies every time propagator and covariance kernel. Geometry and
the multinomial initialization law determine state dependence. The polynomial
interior moment family does not close for the stopped law globally: stopping
corrections must be bounded independently by the experiment's reference audit.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bsde_types import MomentPrediction
from .native import native_coefficients, physical_observable_weights
from .types import QueryBank

DK_GENERATOR_MODEL_KIND = "continuous-native-dk-generator-moments-v1"
OU_GENERATOR_MODEL_KIND = "continuous-linear-ou-generator-moments-v1"


def _rotation(angle):
    result = np.zeros((4, 4))
    for index, mode in enumerate((1, 2)):
        cosine, sine = np.cos(mode*angle), np.sin(mode*angle)
        result[2*index:2*index+2, 2*index:2*index+2] = ((cosine, -sine), (sine, cosine))
    return result


class GeneratorMomentModel(nn.Module):
    """Learned continuous-time propagators with structural state dependence.

    Internal means use sqrt(N) times the normalized physical observations;
    internal covariance is N Cov in those same normalized units. Public methods
    return physical means and N-scaled physical covariance, as the BSDE models.
    """

    architecture = "continuous-generator"

    def __init__(self, config, problem="dk", width=32):
        super().__init__()
        if hasattr(config, "base_config"):
            config = config.base_config
        if problem not in ("ou", "dk") or not isinstance(width, int) or width < 2:
            raise ValueError("Generator model requires OU/DK and integer width >=2")
        self.config, self.problem, self.width = config, problem, width
        self.kind = DK_GENERATOR_MODEL_KIND if problem == "dk" else OU_GENERATOR_MODEL_KIND
        self.state_dimension = config.n_cells if problem == "dk" else 4
        if problem == "dk":
            context = config.context(config.cases[0])
            if context.kappa != 0 or context.a2 != 0:
                raise ValueError("The generator moment family requires noninteracting physics")
            weights = physical_observable_weights(config.spec, QueryBank())
            scales = np.sqrt(np.mean(weights**2, axis=1))
            normalized = weights/scales[:, None]
            initial = np.zeros(config.n_cells)
            initial[0] = 1.
            covariance = native_coefficients(initial, context, config.spec).covariance
            forcing = normalized @ covariance @ normalized.T
            rotations = np.asarray([_rotation(j*config.spec.h) for j in range(config.n_cells)])
            identity = np.eye(config.n_cells)
            faces = np.column_stack((identity[1]-identity[0], identity[0]-identity[-1]))
            initial_factor = normalized @ faces*np.sqrt(config.diffusion)/config.spec.h
        else:
            weights, scales = np.eye(4), np.ones(4)
            forcing = np.diag(2*config.mode_rates)
            rotations = np.eye(4)[None]
        self.register_buffer("observation_weights", torch.tensor(weights))
        self.register_buffer("observation_std", torch.tensor(scales))
        self.register_buffer("normalized_weights", torch.tensor(weights/scales[:, None]))
        self.register_buffer("mode_rates", torch.tensor(config.mode_rates))
        self.register_buffer("noise_kernel", torch.tensor(forcing))
        if problem == "dk":
            self.register_buffer("initial_noise_factor", torch.tensor(initial_factor))
        self.register_buffer("rotations", torch.tensor(rotations))
        self.register_buffer("rotation_plus", torch.tensor(_rotation(config.spec.h)))
        self.register_buffer("rotation_minus", torch.tensor(_rotation(-config.spec.h)))
        layers, inputs = [], 1
        for _ in range(3):
            layers.extend((nn.Linear(inputs, width), nn.SiLU()))
            inputs = width
        layers.append(nn.Linear(width, 18 if problem == "dk" else 4))
        self.time_network = nn.Sequential(*layers).double()
        with torch.no_grad():
            final = self.time_network[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[:2] = np.log(np.expm1(1.))
            if problem == "dk":
                # Instantaneous forcing initializes the derivative, never a
                # future moment. Additional rank directions start nonzero so
                # their Gram-factor gradients can grow under optimization.
                complement = np.linalg.qr(initial_factor, mode="complete")[0][:, 2:]
                final.bias[10:] = torch.tensor(.1*complement).reshape(-1)
            else:
                final.bias[2:] = torch.sqrt(2*self.mode_rates[::2])
        self.provenance = {}

    @property
    def target_id(self):
        from .generator_config import generator_target_id

        return generator_target_id(self.config, self.problem)

    def time_coefficients(self, times):
        """Return learned mode propagators and normalized covariance kernels.

        No future decay or exact moment formulas are evaluated here. The known
        instantaneous noise factor fixes K'(0), like the exact K(0)=0 value;
        the network learns all future corrections and propagators.
        """
        outputs = self.time_network(times.unsqueeze(-1)/self.config.horizon)
        rates = F.softplus(outputs[..., :2]).repeat_interleave(2, dim=-1)
        decays = torch.exp(-times.unsqueeze(-1)*rates)
        if self.problem == "dk":
            correction = outputs[..., 2:10].reshape(*times.shape, 4, 2)
            additional = outputs[..., 10:].reshape(*times.shape, 4, 2)
            factor = self.initial_noise_factor+times[..., None, None]*correction
            kernel = (times[..., None, None]*factor @ factor.transpose(-1, -2)
                      + times[..., None, None].square()*additional @ additional.transpose(-1, -2))
        else:
            variances = (times.unsqueeze(-1)*outputs[..., 2:].square()).repeat_interleave(2, -1)
            kernel = torch.diag_embed(variances)
        return decays, kernel

    def pde_residual(self, times):
        """Differentiable continuous-generator residuals in normalized units."""
        (decays, kernel), (derivative, kernel_derivative) = torch.func.jvp(
            self.time_coefficients, (times,), (torch.ones_like(times),))
        mean_residual = derivative + self.mode_rates*decays
        forcing = decays[..., :, None]*self.noise_kernel*decays[..., None, :]
        if self.problem == "dk":
            diffusion = self.config.diffusion/self.config.spec.h**2
            transport = diffusion*(self.rotation_plus @ kernel @ self.rotation_plus.T
                                   + self.rotation_minus @ kernel @ self.rotation_minus.T
                                   - 2*kernel)
        else:
            transport = 0.
        return mean_residual, kernel_derivative-transport-forcing

    def initially_stopped(self, initial):
        if self.problem == "dk":
            return torch.any(initial <= self.config.spec.mass_stop, dim=-1)
        return torch.zeros(initial.shape[:-1], dtype=torch.bool, device=initial.device)

    def scaled_observables(self, initial):
        scale = np.sqrt(self.config.n_particles) if self.problem == "dk" else 1.
        return scale*(initial @ self.normalized_weights.T)

    def _rotated_kernels(self, kernel):
        return torch.einsum("gai,...ij,gbj->...gab", self.rotations, kernel, self.rotations)

    def moment_values(self, initial, horizons):
        """Normalized moments for one horizon per initial state."""
        if initial.shape[-1] != self.state_dimension:
            raise ValueError("Initial state dimension differs from the generator target")
        horizons = torch.broadcast_to(horizons, initial.shape[:-1])
        decays, kernel = self.time_coefficients(horizons)
        y0 = self.scaled_observables(initial)
        mean = y0*decays
        covariance = (torch.einsum("...g,...gab->...ab", initial, self._rotated_kernels(kernel))
                      if self.problem == "dk" else kernel)
        frozen = self.initially_stopped(initial) | (horizons == 0)
        return (torch.where(frozen.unsqueeze(-1), y0, mean),
                torch.where(frozen[..., None, None], 0., covariance))

    def _validate_initial(self, initial):
        initial = np.asarray(initial, dtype=np.float64)
        if initial.ndim == 1:
            initial = initial[None]
        if (initial.ndim != 2 or initial.shape[1] != self.state_dimension
                or not np.all(np.isfinite(initial))):
            raise ValueError("A finite batch of compatible initial states is required")
        if self.problem == "dk" and (np.any(initial < 0)
                or not np.allclose(initial.sum(-1), 1., atol=1e-10, rtol=0)):
            raise ValueError("DK initial masses must be nonnegative and sum to one")
        return initial

    def _validate_times(self, horizons):
        horizons = np.atleast_1d(np.asarray(horizons, dtype=np.float64))
        if (horizons.ndim != 1 or not len(horizons) or not np.all(np.isfinite(horizons))
                or np.any(np.diff(horizons) <= 0) or horizons[0] < 0
                or horizons[-1] > self.config.horizon):
            raise ValueError("Unique increasing horizons within the trained interval are required")
        return horizons

    def _physical_units(self, mean, covariance):
        if self.problem == "dk":
            mean = mean*self.observation_std/np.sqrt(self.config.n_particles)
            covariance = covariance*self.observation_std[:, None]*self.observation_std[None, :]
        return mean, covariance

    def predict_conditional(self, initial, horizons):
        started = time.perf_counter()
        states, horizons = self._validate_initial(initial), self._validate_times(horizons)
        self.eval()
        with torch.no_grad():
            masses = torch.tensor(states)
            times = torch.tensor(horizons)
            decays, kernel = self.time_coefficients(times)
            y0 = self.scaled_observables(masses)
            mean = y0[:, None, :]*decays[None]
            covariance = (torch.einsum("bg,hgij->bhij", masses, self._rotated_kernels(kernel))
                          if self.problem == "dk" else kernel[None].expand(len(masses), -1, -1, -1))
            frozen = self.initially_stopped(masses)[:, None] | (times[None] == 0)
            mean = torch.where(frozen.unsqueeze(-1), y0[:, None], mean)
            covariance = torch.where(frozen[..., None, None], 0., covariance)
            mean, covariance = self._physical_units(mean, covariance)
        return MomentPrediction(mean.numpy(), covariance.numpy(), horizons, self.problem,
                                self.target_id, {"kind": self.kind, "conditional": True,
                                    "continuous_target": True, "cross_time_covariance_supported": False,
                                    "stopping_correction": "Requires independent certificate outside initially stopped states",
                                    "inference_seconds": time.perf_counter()-started})

    def predict_ensemble(self, probabilities, horizons):
        """Integrate the known IID histogram law, retaining neural dynamics.

        This is exact integration of the model's linear/affine interior moment
        map. Empty initial cells and future stopping need independent bounds.
        No particle samples or analytic dynamic moment forecasts are used.
        """
        if self.problem != "dk":
            raise ValueError("Analytic histogram initialization is supported for DK only")
        started = time.perf_counter()
        probabilities = self._validate_initial(probabilities)
        if len(probabilities) != 1:
            raise ValueError("A single initial cell-probability vector is required")
        horizons = self._validate_times(horizons)
        self.eval()
        with torch.no_grad():
            probability = torch.tensor(probabilities[0])
            decays, kernel = self.time_coefficients(torch.tensor(horizons))
            normalized_mean = self.normalized_weights @ probability
            mean = np.sqrt(self.config.n_particles)*decays*normalized_mean
            within = torch.einsum("g,hgij->hij", probability, self._rotated_kernels(kernel))
            initial_covariance = ((self.normalized_weights*probability) @ self.normalized_weights.T
                                  - normalized_mean[:, None]*normalized_mean[None, :])
            covariance = within+decays[:, :, None]*initial_covariance*decays[:, None, :]
            mean, covariance = self._physical_units(mean, covariance)
        return MomentPrediction(mean.numpy(), covariance.numpy(), horizons, "dk", self.target_id,
                                {"kind": self.kind, "conditional": False,
                                 "initial_law_integration": "exact-multinomial-moments-of-neural-interior-map",
                                 "stopping_correction": "Initial-empty-cell and future-exit effects require independent certificate",
                                 "cross_time_covariance_supported": False,
                                 "inference_seconds": time.perf_counter()-started})
