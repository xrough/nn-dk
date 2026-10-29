"""One smooth partition-of-unity observation for both particles and fields.

The cardinal cubic splines are centered at j L/M. Fields are piecewise constant
on cells [j L/G, (j+1) L/G), represented by probability masses, not densities.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from .types import TargetSpec


def spline_weights(x: np.ndarray, target: TargetSpec, derivative: int = 0) -> np.ndarray:
    """Evaluate all periodic cubic B-splines, or their first/second derivatives."""
    if derivative not in (0, 1, 2):
        raise ValueError("Only spline derivative orders 0, 1, and 2 are supported")
    m = target.n_cells
    h = target.length / m
    r = (np.asarray(x, dtype=float)[..., None] / h - np.arange(m) + m / 2) % m - m / 2
    a = np.abs(r)
    if derivative == 0:
        values = np.where(a < 1, 2 / 3 - a * a + a ** 3 / 2,
                          np.where(a < 2, (2 - a) ** 3 / 6, 0.0))
    elif derivative == 1:
        values = np.where(a < 1, -2 * r + 1.5 * r * a,
                          np.where(a < 2, -0.5 * (2 - a) ** 2 * np.sign(r), 0.0)) / h
    else:
        values = np.where(a < 1, -2 + 3 * a,
                          np.where(a < 2, 2 - a, 0.0)) / h ** 2
    return values


def observe_particles(x: np.ndarray, target: TargetSpec) -> np.ndarray:
    """Observe (..., N) periodic particle positions as (..., M) probability masses."""
    x = np.asarray(x, dtype=float)
    if x.ndim < 1 or x.shape[-1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Finite particle positions with a nonempty particle axis required")
    # Accumulate four nonzero spline values rather than allocate (..., N, M).
    scaled = (x % target.length) * (target.n_cells / target.length)
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    out = np.zeros((int(np.prod(x.shape[:-1])), target.n_cells))
    rows = np.arange(len(out))[:, None]
    for offset in (-1, 0, 1, 2):
        r = fraction - offset
        a = np.abs(r)
        value = np.where(a < 1, 2 / 3 - a * a + a ** 3 / 2, (2 - a) ** 3 / 6)
        np.add.at(out, (rows, ((base + offset) % target.n_cells).reshape(len(out), -1)),
                  value.reshape(len(out), -1) / x.shape[-1])
    return out.reshape((*x.shape[:-1], target.n_cells))


@lru_cache(maxsize=32)
def _observation_matrix(target: TargetSpec, n_grid: int) -> np.ndarray:
    if n_grid < 1:
        raise ValueError("A positive field grid size is required")
    # Split at every field and spline knot; four-point Gauss integrates each cubic exactly.
    breaks = np.unique(np.concatenate((np.arange(n_grid + 1) * target.length / n_grid,
                                       np.arange(target.n_cells + 1) * target.length / target.n_cells)))
    nodes, weights = leggauss(4)
    result = np.zeros((target.n_cells, n_grid))
    h = target.length / n_grid
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        if right - left < 1e-14:
            continue
        mid = (left + right) / 2
        xx = mid + (right - left) / 2 * nodes
        cell = min(int(mid / h), n_grid - 1)
        result[:, cell] += spline_weights(xx, target).T @ weights * (right - left) / (2 * h)
    result.setflags(write=False)
    return result


def observation_matrix(target: TargetSpec, n_grid: int) -> np.ndarray:
    """M by G exact spline averages over G finite-volume cells."""
    return _observation_matrix(target, int(n_grid)).copy()


def observe_field(masses: np.ndarray, target: TargetSpec) -> np.ndarray:
    """Integrate a piecewise constant field against the same particle splines."""
    masses = np.asarray(masses, dtype=float)
    if masses.ndim < 1 or not np.all(np.isfinite(masses)):
        raise ValueError("Finite field masses with a cell axis required")
    return masses @ _observation_matrix(target, masses.shape[-1]).T


def uniform_quadratic_variation(target: TargetSpec, diffusion: float = 1.0) -> np.ndarray:
    """Exact uniform-particle scaled covariance rate 2D/L integral w'_i w'_j."""
    if diffusion <= 0 or not np.isfinite(diffusion):
        raise ValueError("Positive finite diffusion required")
    nodes, weights = leggauss(3)  # product of quadratic derivatives: exact on knot intervals
    h = target.length / target.n_cells
    x = (np.arange(target.n_cells)[:, None] + (nodes[None, :] + 1) / 2) * h
    d = spline_weights(x.ravel(), target, derivative=1)
    quadrature = np.tile(weights * h / 2, target.n_cells)
    return (2 * diffusion / target.length) * (d.T @ (quadrature[:, None] * d))


def incidence_matrix(n_cells: int) -> np.ndarray:
    """Conservative divergence: face i removes mass i and adds it to i+1."""
    if n_cells < 2:
        raise ValueError("At least two cells required")
    b = -np.eye(n_cells)
    b[(np.arange(n_cells) + 1) % n_cells, np.arange(n_cells)] += 1
    return b


def tangent_basis(n_cells: int) -> np.ndarray:
    """Fixed Helmert orthonormal columns spanning the mass-zero hyperplane."""
    if n_cells < 2:
        raise ValueError("At least two cells required")
    basis = np.zeros((n_cells, n_cells - 1))
    for j in range(n_cells - 1):
        basis[:j + 1, j] = 1 / np.sqrt((j + 1) * (j + 2))
        basis[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return basis
