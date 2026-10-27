"""Whole-trajectory uncertainty and task reductions for the broader comparison.

The reference is a declared stopped Euler polygon. Statistical intervals do
not certify its difference from a continuous or continuum DK target.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402

from analysis.probe_path_workloads import (  # noqa: E402
    many_observables,
    workload_values,
)
from nn_dk.payoff_learning import PayoffBank  # noqa: E402


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def bootstrap_counts(n_paths, seed, n_bootstrap=128):
    """Multinomial whole-family resampling; no derived node is split off."""
    return np.random.default_rng(seed).multinomial(
        n_paths, np.full(n_paths, 1/n_paths), size=n_bootstrap).astype(float)


def empirical(values, counts=None):
    """Means and unbiased covariances at each horizon, keeping family axis 0."""
    y = np.asarray(values, dtype=float)
    if y.ndim == 2:
        y = y[:, None]
    if y.ndim != 3 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("At least two finite whole trajectories [P,T,O] required")
    n = len(y)
    if counts is None:
        mean = y.mean(0)
        centered = y-mean
        cov = np.einsum("pti,ptj->tij", centered, centered)/(n-1)
        return mean, cov
    counts = np.asarray(counts, dtype=float)
    if (counts.ndim != 2 or counts.shape[1] != n or np.any(counts < 0)
            or not np.all(counts.sum(-1) == n)):
        raise ValueError("Whole-trajectory multinomial bootstrap counts required")
    # Center first for stability when a mean dwarfs its fluctuations.
    center = y.mean(0)
    z = y-center
    mean_delta = (counts @ z.reshape(n, -1)).reshape(len(counts), *y.shape[1:])/n
    products = np.einsum("pti,ptj->ptij", z, z)
    second = (counts @ products.reshape(n, -1)).reshape(len(counts), *products.shape[1:])
    covariance = (second-n*np.einsum("bti,btj->btij", mean_delta, mean_delta))/(n-1)
    return center+mean_delta, covariance


def moment_comparison(reference, candidate=None, *, predicted_mean=None,
                      predicted_scaled_covariance=None, n_particles=2048,
                      bootstrap_predicted_mean=None, bootstrap_predicted_scaled_covariance=None,
                      counts=None, paired=True, seed=2026102614):
    """Simultaneous-in-bank centered-bootstrap error bounds, separately by time.

    A bound is a sampling diagnostic for this finite reference, not a
    reference-bias or population-coverage guarantee. All optimizer seeds are
    evaluated separately and none is selected using this result.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 2:
        reference = reference[:, None]
    counts = bootstrap_counts(len(reference), seed) if counts is None else counts
    rm, rc = empirical(reference)
    brm, brc = empirical(reference, counts)
    if candidate is None:
        cm = np.asarray(predicted_mean, dtype=float)
        cc = np.asarray(predicted_scaled_covariance, dtype=float)/n_particles
        if (cm.shape != rm.shape or cc.shape != rc.shape
                or not np.isfinite(cm).all() or not np.isfinite(cc).all()):
            raise ValueError("Predicted means/covariances must match the declared bank")
        bcm, bcc = cm[None], cc[None]
        if (bootstrap_predicted_mean is None) != (bootstrap_predicted_scaled_covariance is None):
            raise ValueError("Supply both mean and covariance integration bootstrap replicates")
        if bootstrap_predicted_mean is not None:
            bcm = np.asarray(bootstrap_predicted_mean, dtype=float)
            bcc = np.asarray(bootstrap_predicted_scaled_covariance, dtype=float)/n_particles
            if (bcm.shape != brm.shape or bcc.shape != brc.shape
                    or not np.isfinite(bcm).all() or not np.isfinite(bcc).all()):
                raise ValueError("Prediction integration bootstrap must match reference resampling")
    else:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.ndim == 2:
            candidate = candidate[:, None]
        if candidate.shape[1:] != reference.shape[1:]:
            raise ValueError("Candidate and reference bank dimensions differ")
        cm, cc = empirical(candidate)
        if paired and candidate.shape != reference.shape:
            raise ValueError("Paired comparisons require identical whole-family pools")
        candidate_counts = counts if paired else bootstrap_counts(len(candidate), seed+1, len(counts))
        bcm, bcc = empirical(candidate, candidate_counts)
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    if np.any(std <= 1e-14):
        raise ValueError("Degenerate observable needs an explicit separate normalization")
    scales = std[..., :, None]*std[..., None, :]
    dm, dc = (cm-rm)/std, (cc-rc)/scales
    bdm, bdc = (bcm-brm)/std, (bcc-brc)/scales
    records = []
    for t in range(len(rm)):
        mean_point, cov_point = np.max(abs(dm[t])), np.max(abs(dc[t]))
        mean_sampling = np.quantile(np.max(abs(bdm[:, t]-dm[t]), axis=-1), .95)
        covariance_sampling = np.quantile(np.max(abs(bdc[:, t]-dc[t]), axis=(-2, -1)), .95)
        records.append({"time_index": t,
                        "max_mean_error_in_reference_std": float(mean_point),
                        "max_normalized_covariance_error": float(cov_point),
                        "mean_sampling95_simultaneous_radius": float(mean_sampling),
                        "covariance_sampling95_simultaneous_radius": float(covariance_sampling),
                        "mean_error_plus_sampling95": float(mean_point+mean_sampling),
                        "covariance_error_plus_sampling95": float(cov_point+covariance_sampling),
                        "reference_mean": rm[t].tolist(),
                        "candidate_mean": cm[t].tolist(),
                        "reference_scaled_covariance": (n_particles*rc[t]).tolist(),
                        "candidate_scaled_covariance": (n_particles*cc[t]).tolist(),
                        "normalized_mean_difference": dm[t].tolist(),
                        "normalized_covariance_difference": dc[t].tolist()})
    return {"records": records, "n_reference": len(reference),
            "n_candidate": None if candidate is None else len(candidate),
            "paired": candidate is not None and paired,
            "bootstrap": {"unit": "whole-trajectory-family", "count": len(counts),
                          "counts_sha256": array_sha(counts)},
            "scope": "sampling-only; finite stopped reference; no reference-bias certificate"}


def mixture_moments(conditional_mean, conditional_scaled_covariance, *,
                    n_particles=2048, counts=None):
    """Total expectation/covariance over IID initial families, including bootstrap.

    Conditional covariance is N-scaled. Covariance of conditional means uses
    ddof=1; bootstrap repeats complete initial states and all their horizons.
    Use the same family counts as the paired reference when initials are shared.
    """
    mu = np.asarray(conditional_mean, dtype=float)
    cov = np.asarray(conditional_scaled_covariance, dtype=float)
    if (mu.ndim != 3 or cov.shape != (*mu.shape, mu.shape[-1])
            or not np.isfinite(mu).all() or not np.isfinite(cov).all()):
        raise ValueError("Finite conditional means [P,T,O] and scaled covariances [P,T,O,O] required")
    mean, between = empirical(mu, counts)
    if counts is None:
        within = cov.mean(0)
    else:
        within = (counts @ cov.reshape(len(cov), -1)).reshape(len(counts), *cov.shape[1:])/len(cov)
    return mean, within+n_particles*between


def diagonal_comparison(reference, candidate=None, *, predicted_mean=None,
                        predicted_scaled_variance=None, n_particles=2048, counts=None):
    """Many-query means/variances without claiming unsupported full-cov bounds."""
    r = np.asarray(reference)
    c = None if candidate is None else np.asarray(candidate)
    if r.ndim != 2 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("Finite many-query outcomes [P,O] required")
    n = len(r)
    counts = bootstrap_counts(n, 2026102615) if counts is None else counts
    rm = r.mean(0)
    centered = r-rm
    rv = (centered**2).sum(0)/(n-1)
    if np.any(rv <= 1e-28):
        raise ValueError("Degenerate observables need a separate declared normalization")
    brdelta = counts @ centered/n
    brm = rm+brdelta
    brv = (counts @ centered**2-n*brdelta**2)/(n-1)
    if c is None:
        cm, cv = np.asarray(predicted_mean), np.asarray(predicted_scaled_variance)/n_particles
        if (cm.shape != rm.shape or cv.shape != rv.shape or not np.isfinite(cm).all()
                or not np.isfinite(cv).all() or np.any(cv < 0)):
            raise ValueError("Finite compatible predicted means and nonnegative scaled variances required")
        bcm, bcv = cm[None], cv[None]
    else:
        if c.shape != r.shape or not np.isfinite(c).all():
            raise ValueError("Many-query comparison requires paired trajectory families")
        cm = c.mean(0)
        z = c-cm
        cv = (z*z).sum(0)/(n-1)
        bdelta = counts @ z/n
        bcm, bcv = cm+bdelta, (counts @ (z*z)-n*bdelta**2)/(n-1)
    dm, dv = (cm-rm)/np.sqrt(rv), (cv-rv)/rv
    bdm, bdv = (bcm-brm)/np.sqrt(rv), (bcv-brv)/rv
    return {"n_observables": r.shape[1], "mean_max_std": float(abs(dm).max()),
            "variance_max_relative": float(abs(dv).max()),
            "mean_rms_std": float(np.sqrt(np.mean(dm**2))),
            "variance_rms_relative": float(np.sqrt(np.mean(dv**2))),
            "mean_sampling95_simultaneous_radius": float(np.quantile(abs(bdm-dm).max(-1), .95)),
            "variance_sampling95_simultaneous_radius": float(np.quantile(abs(bdv-dv).max(-1), .95)),
            "mean_errors": dm.tolist(), "variance_errors": dv.tolist(),
            "joint_covariance_uncertainty": "not estimated for this 480-output bank"}


def distribution_comparison(reference, candidate):
    r, c = np.asarray(reference), np.asarray(candidate)
    r, c = r.reshape(len(r), -1), c.reshape(len(c), -1)
    return {"wasserstein1_in_reference_std": [float(wasserstein_distance(x, y)/x.std(ddof=1))
                                              for x, y in zip(r.T, c.T, strict=True)],
            "ks_distance": [float(ks_2samp(x, y).statistic)
                            for x, y in zip(r.T, c.T, strict=True)],
            "scope": "empirical scalar distances; no p-values or distribution certificate"}


def _write_json(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _banks(spec, horizon):
    return (PayoffBank.terminal_high_modes(spec, horizon),
            PayoffBank.mixed_terminal_integral(spec, horizon))


def reduce_paths(packet, spec, dt, horizon, common_step=2**-8):
    """Native paths and density-path operator use identical evaluation maps."""
    states, stops = packet["states"], packet["stopped_at"]
    if isinstance(states, torch.Tensor):
        states, stops = states.detach().numpy(), stops.detach().numpy()
    n = states.shape[1]-1
    physical = np.asarray((.25*horizon, .5*horizon, horizon))
    common_times = np.arange(round(horizon/common_step)+1)*common_step
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(horizon) or horizon <= 0
            or not np.isfinite(common_step) or common_step <= 0
            or not np.isclose(n*dt, horizon, atol=1e-12, rtol=0)
            or not np.allclose(physical/dt, np.rint(physical/dt), atol=1e-10, rtol=0)
            or not np.isclose(common_times[-1], horizon, atol=1e-12, rtol=0)
            or not np.allclose(common_times/dt, np.rint(common_times/dt), atol=1e-10, rtol=0)):
        raise ValueError("Complete paths with exactly nested physical observation grids required")
    times = np.arange(n+1)*dt
    qi = np.rint(physical/dt).astype(int)
    common = np.rint(common_times/dt).astype(int)
    common_states = states[:, common]
    bank_values = {bank.name: bank.evaluate_paths(states, times, stops)[:, qi]
                   for bank in _banks(spec, horizon)}
    # many_observables uses evenly spaced nodes: 17 gives the exact 17-node
    # common grid here, then remove its deterministic t=0 block.
    many = many_observables(common_states, spec, n_times=len(common))[:, 2*(spec.n_cells//2-1):]
    return {"banks": bank_values, "common_states": common_states,
            "many": many, "nonlinear": workload_values(common_states, spec, common_step),
            "stopped_at": stops}


