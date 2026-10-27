"""Learned modewise value heads with full-state residuals for stopped DK.

The OU control has four fitted positive scalars and contains the stable Euler
OU moment family.  No physical decay rate or analytic forecast is supplied to
its predictions.  Stopped DK uses the same learned backbone and an unrestricted
initial-state residual: stopping does not admit an exact low-mode closure.
The predictable martingale network and noise convention are inherited unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bsde_model import PAYOFF_DIMENSION, BSDEMomentModel, _mlp

STRUCTURED_DK_BSDE_MODEL_KIND = "native-dk-bsde-structured-moments-v1"
STRUCTURED_OU_BSDE_MODEL_KIND = "linear-ou-bsde-structured-moments-v1"
STRUCTURED_RESIDUAL_SCALE = 0.1


def _inverse_softplus(value):
    """Stable inverse on positive inputs, without imposing a covariance floor."""
    return value + torch.log(-torch.expm1(-value))


class _StructuredValueHead(nn.Module):
    def __init__(self, feature_dimension, width, problem):
        super().__init__()
        # Both rates and asymptotic variances start at one, independently of
        # physics.  Analytic OU coefficients appear only in calibration tests.
        initial = float(np.log(np.expm1(1.)))
        self.raw_rates = nn.Parameter(torch.full((2,), initial, dtype=torch.float64))
        self.raw_scales = nn.Parameter(torch.full((2,), initial, dtype=torch.float64))
        self.residual_network = (_mlp(feature_dimension + 1, PAYOFF_DIMENSION, width)
                                 if problem == "dk" else None)
        if self.residual_network is not None:
            with torch.no_grad():
                self.residual_network[-1].weight.zero_()
                self.residual_network[-1].bias.zero_()

    @property
    def rates(self):
        return F.softplus(self.raw_rates)

    @property
    def scales(self):
        """Positive asymptotic variance per cosine/sine mode pair."""
        return F.softplus(self.raw_scales)

    def forward(self, features):
        if self.residual_network is None:
            return features.new_zeros((*features.shape[:-1], PAYOFF_DIMENSION))
        return self.residual_network(features)


class StructuredBSDEMomentModel(BSDEMomentModel):
    """Conditional moments with learned mode propagation and optional residual.

    The four OU parameters are shared across horizons and initial states.
    DK retains all mass-zero coordinates in its residual, including its
    conditional covariance head.  The residual scale is fixed at 0.1 for the
    architecture comparison, and never fitted against the evaluation bank.
    """

    architecture = "structured"

    def __init__(self, config, problem="dk"):
        # Preserve the exact random draws used by the baseline integrand.
        # Replacing the value head before parent construction would alter them.
        super().__init__(config, problem)
        self.kind = (STRUCTURED_DK_BSDE_MODEL_KIND if problem == "dk"
                     else STRUCTURED_OU_BSDE_MODEL_KIND)
        self.value_network = _StructuredValueHead(self.feature_dimension, config.width,
                                                  problem)

    def moment_values(self, initial, horizons):
        if initial.shape[-1] != self.state_dimension:
            raise ValueError("Initial state dimension does not match BSDE target")
        horizons = torch.broadcast_to(horizons, initial.shape[:-1])
        y0 = self.scaled_observables(initial)
        rates = self.value_network.rates.repeat_interleave(2)
        scales = self.value_network.scales.repeat_interleave(2)
        duration = horizons.unsqueeze(-1)
        decay = torch.exp(-duration * rates)
        mean = decay * y0

        if self.problem == "ou":
            variance = scales * -torch.expm1(-2 * duration * rates)
            covariance = torch.diag_embed(variance)
        else:
            features = torch.cat((self.normalized_features(initial), duration), -1)
            residual = STRUCTURED_RESIDUAL_SCALE * self.value_network(features)
            mean = mean + duration * residual[..., :4]

            # lim_{T->0} v*(1-exp(-2*r*T))/T = 2*r*v.  Safe denominators
            # keep both branches finite, including backward passes at T=0.
            safe_duration = torch.where(duration == 0, 1., duration)
            ratio = torch.where(duration == 0, 2 * rates,
                                -torch.expm1(-2 * duration * rates) / safe_duration)
            base = torch.sqrt(scales * ratio)
            indices = torch.tril_indices(4, 4, device=initial.device)
            diagonal = indices[0] == indices[1]
            values = residual[..., 4:].clone()
            values[..., diagonal] = F.softplus(
                _inverse_softplus(base) + values[..., diagonal])
            factor = values.new_zeros((*values.shape[:-1], 4, 4))
            factor[..., indices[0], indices[1]] = values
            covariance = (horizons[..., None, None]
                          * factor @ factor.transpose(-1, -2))

        frozen = self.initially_stopped(initial) | (horizons == 0)
        mean = torch.where(frozen.unsqueeze(-1), y0, mean)
        covariance = torch.where(frozen[..., None, None], 0., covariance)
        return mean, covariance

