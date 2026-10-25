"""Exact local generator calculus through neural networks.

The face derivatives are ``grad(f) S / sqrt(N)`` at the current state.  The
coefficient matrix is frozen during each local generator contraction: its
derivatives do not enter the Itô generator.  Every physical face is retained;
there is no randomized trace estimator or auxiliary Brownian sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GeneratorJet:
    """A value, its time derivative, generator action and noise derivatives."""

    value: torch.Tensor
    time_derivative: torch.Tensor
    generator: torch.Tensor
    face_derivatives: torch.Tensor


def jet_constant(value: torch.Tensor, n_faces: int) -> GeneratorJet:
    zero = torch.zeros_like(value)
    return GeneratorJet(value, zero, zero, value.new_zeros((*value.shape, n_faces)))


def jet_add(left: GeneratorJet, right: GeneratorJet) -> GeneratorJet:
    return GeneratorJet(left.value + right.value,
                        left.time_derivative + right.time_derivative,
                        left.generator + right.generator,
                        left.face_derivatives + right.face_derivatives)


def jet_scale(jet: GeneratorJet, scale) -> GeneratorJet:
    scale = torch.as_tensor(scale, dtype=jet.value.dtype, device=jet.value.device)
    return GeneratorJet(jet.value * scale, jet.time_derivative * scale,
                        jet.generator * scale, jet.face_derivatives * scale.unsqueeze(-1))


def jet_multiply(left: GeneratorJet, right: GeneratorJet) -> GeneratorJet:
    """Itô product rule, including the full physical cross variation."""
    return GeneratorJet(
        left.value * right.value,
        left.time_derivative * right.value + left.value * right.time_derivative,
        left.generator * right.value + left.value * right.generator
        + (left.face_derivatives * right.face_derivatives).sum(-1),
        left.face_derivatives * right.value.unsqueeze(-1)
        + left.value.unsqueeze(-1) * right.face_derivatives)


def jet_activation(jet: GeneratorJet, value, first, second) -> GeneratorJet:
    return GeneratorJet(
        value, first * jet.time_derivative,
        first * jet.generator + .5 * second * jet.face_derivatives.square().sum(-1),
        first.unsqueeze(-1) * jet.face_derivatives)


def jet_silu(jet: GeneratorJet) -> GeneratorJet:
    x = jet.value
    sigmoid = torch.sigmoid(x)
    slope = sigmoid * (1 - sigmoid)
    return jet_activation(jet, F.silu(x), sigmoid + x * slope,
                          slope * (2 + x * (1 - 2 * sigmoid)))


def jet_softplus(jet: GeneratorJet) -> GeneratorJet:
    sigmoid = torch.sigmoid(jet.value)
    # logaddexp avoids the threshold-dependent linear branch in F.softplus.
    value = torch.logaddexp(torch.zeros_like(jet.value), jet.value)
    return jet_activation(jet, value, sigmoid, sigmoid * (1 - sigmoid))


def jet_tanh(jet: GeneratorJet) -> GeneratorJet:
    value = torch.tanh(jet.value)
    slope = 1 - value.square()
    return jet_activation(jet, value, slope, -2 * value * slope)


def jet_linear(jet: GeneratorJet, weight: torch.Tensor,
               bias: torch.Tensor | None = None) -> GeneratorJet:
    return GeneratorJet(F.linear(jet.value, weight, bias),
                        F.linear(jet.time_derivative, weight),
                        F.linear(jet.generator, weight),
                        torch.einsum("...jf,ij->...if", jet.face_derivatives, weight))


def jet_sequential(jet: GeneratorJet, network: nn.Sequential) -> GeneratorJet:
    for layer in network:
        if isinstance(layer, nn.Linear):
            jet = jet_linear(jet, layer.weight, layer.bias)
        elif isinstance(layer, nn.SiLU):
            jet = jet_silu(jet)
        else:
            raise TypeError(f"Unsupported exact generator-jet layer {type(layer).__name__}")
    return jet


def _dimension(dim: int, ndim: int, *, inserting: bool = False) -> int:
    limit = ndim + int(inserting)
    normalized = dim + limit if dim < 0 else dim
    if not 0 <= normalized < limit:
        raise IndexError("Generator-jet dimension is outside the value axes")
    return normalized


def jet_unsqueeze(jet: GeneratorJet, dim: int) -> GeneratorJet:
    dim = _dimension(dim, jet.value.ndim, inserting=True)
    return GeneratorJet(*(value.unsqueeze(dim) for value in
                          (jet.value, jet.time_derivative, jet.generator,
                           jet.face_derivatives)))


def jet_sum(jet: GeneratorJet, dim: int) -> GeneratorJet:
    dim = _dimension(dim, jet.value.ndim)
    return GeneratorJet(*(value.sum(dim) for value in
                          (jet.value, jet.time_derivative, jet.generator,
                           jet.face_derivatives)))


def jet_slice(jet: GeneratorJet, index) -> GeneratorJet:
    indices = index if isinstance(index, tuple) else (index,)
    return GeneratorJet(jet.value[indices], jet.time_derivative[indices],
                        jet.generator[indices], jet.face_derivatives[(*indices, slice(None))])


def jet_cat(jets: list[GeneratorJet], dim: int = -1) -> GeneratorJet:
    dim = _dimension(dim, jets[0].value.ndim)
    return GeneratorJet(*(torch.cat([getattr(jet, name) for jet in jets], dim=dim)
                          for name in ("value", "time_derivative", "generator",
                                       "face_derivatives")))


def jet_stack(jets: list[GeneratorJet], dim: int = -1) -> GeneratorJet:
    dim = _dimension(dim, jets[0].value.ndim, inserting=True)
    return GeneratorJet(*(torch.stack([getattr(jet, name) for jet in jets], dim=dim)
                          for name in ("value", "time_derivative", "generator",
                                       "face_derivatives")))


def jet_gram(factor: GeneratorJet) -> GeneratorJet:
    """Return the exact jet of ``factor @ factor.T`` on the final two axes."""
    return jet_sum(jet_multiply(jet_unsqueeze(factor, -2),
                               jet_unsqueeze(factor, -3)), -1)
