"""Reference checks performed before interpreting learned-model comparisons."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .evaluation import estimate_queries, json_safe, query_labels
from .observations import (
    incidence_matrix,
    observation_matrix,
    observe_field,
    observe_particles,
    spline_weights,
    uniform_quadratic_variation,
)
from .particles import (
    coupled_refinement,
    free_particle_statistics,
    pairwise_force,
    particle_force,
    sample_initial,
    simulate_particles,
)
from .types import Context, QueryBank, TargetSpec, seeded_rng


def _paired_refinement(coarse, fine, queries: QueryBank, context: Context,
                       rng: np.random.Generator, n_bootstrap: int = 100) -> dict[str, Any]:
    yc, yf = queries.extract(coarse), queries.extract(fine)
    n = len(yc)
    def cov(y):
        return context.n_particles * np.cov(y, rowvar=False, ddof=1)
    delta_mean = (yf - yc).mean(axis=0)
    delta_covariance = cov(yf) - cov(yc)
    bootstrap_mean, bootstrap_covariance = [], []
    for _ in range(n_bootstrap):
        ids = rng.integers(0, n, n)
        bootstrap_mean.append((yf[ids] - yc[ids]).mean(axis=0))
        bootstrap_covariance.append(cov(yf[ids]) - cov(yc[ids]))
    return {
        "n_paired_independent_paths": n,
        "mean_difference_fine_minus_coarse": delta_mean,
        "scaled_covariance_difference_fine_minus_coarse": delta_covariance,
        "mean_difference_ci95": np.quantile(bootstrap_mean, [0.025, 0.975], axis=0).T,
        "scaled_covariance_difference_ci95": np.moveaxis(np.quantile(bootstrap_covariance, [0.025, 0.975], axis=0), 0, -1),
        "coarse_dt": coarse.diagnostics["dt"], "fine_dt": fine.diagnostics["dt"],
        "max_paired_coarse_mass_difference": float(np.max(np.abs(fine.states - coarse.states))),
        "interpretation": "Paired finite-ensemble timestep sensitivity; finite refinement cannot prove absence of numerical bias",
    }


def _certify_reference(implementation_passed: bool, covariance_ci: np.ndarray,
                       analytic_covariance: np.ndarray, covariance_scale: np.ndarray,
                       refinement_ci: np.ndarray, tolerance: float) -> dict[str, Any]:
    """A narrow but biased free-particle estimate must not pass the reference gate."""
    halfwidth = float(np.max((covariance_ci[..., 1] - covariance_ci[..., 0]) / (2 * covariance_scale)))
    oracle_bound = float(np.max(np.max(np.abs(covariance_ci - analytic_covariance[..., None]), axis=-1) / covariance_scale))
    refinement_bound = float(np.max(np.max(np.abs(refinement_ci), axis=-1) / covariance_scale))
    precision_resolved = halfwidth <= tolerance
    oracle_resolved = oracle_bound <= tolerance
    refinement_resolved = refinement_bound <= tolerance
    return {
        "status": "resolved" if implementation_passed and precision_resolved and oracle_resolved and refinement_resolved else "unresolved",
        "target_relative_fluctuation_tolerance": tolerance,
        "free_reference_uncertainty_resolved": precision_resolved,
        "free_particle_oracle_consistency_resolved": oracle_resolved,
        "paired_refinement_difference_resolved": refinement_resolved,
        "maximum_normalized_free_oracle_discrepancy_ci95_bound": oracle_bound,
        "maximum_normalized_paired_refinement_ci95_bound": refinement_bound,
        "interpretation": "Calibration covers these configured cases only. Resolution requires small reference intervals, agreement of the free-particle estimate with the exact filtered heat oracle including interval uncertainty, and small paired timestep differences. Intervals are pointwise bootstrap intervals, not a simultaneous certification. Small ensembles can validate implementation while leaving a 5% interacting-particle comparison unresolved; every benchmark still requires its own reference intervals.",
    }


def run_calibration(target: TargetSpec, dt: float, save_interval: float, horizon: float,
                    seed: int, n_paths: int = 256, n_particles: int = 512,
                    refinement_paths: int = 8, n_bootstrap: int = 100,
                    tolerance: float = 0.05, diffusion: float = 1.0) -> dict[str, Any]:
    """Run cheap deterministic identities, free-particle statistics, and coupled refinement.

    The certification gate intentionally distinguishes validated implementation
    checks from a resolved 5% scientific comparison. Small smoke ensembles will
    normally leave the latter unresolved.
    """
    started = time.perf_counter()
    if n_paths < 2 or refinement_paths < 2 or n_bootstrap < 2 or not 0 < tolerance < 1:
        raise ValueError("Calibration needs at least two paths/bootstrap draws and a tolerance between zero and one")
    count = int(round(horizon / save_interval))
    if count < 1 or not np.isclose(count * save_interval, horizon, rtol=0, atol=1e-10):
        raise ValueError("Calibration horizon must be an integer multiple of its save interval")
    times = np.arange(count + 1) * save_interval
    selected = sorted(set(max(1, int(round(count * fraction))) for fraction in (0.25, 0.5, 1.0)))
    queries = QueryBank(tuple(float(times[i]) for i in selected))
    rng = seeded_rng(seed, "calibration-checks")
    force_context = Context(kappa=0.7, a2=0.15, n_particles=n_particles, diffusion=diffusion)
    positions = rng.uniform(-target.length, 2 * target.length, size=(3, 17))
    force_difference = float(np.max(np.abs(particle_force(positions, force_context) - pairwise_force(positions, force_context))))
    sign_context = Context(kappa=0.7, a2=0, n_particles=2)
    sign_force = particle_force(np.array([[0, np.pi / 2]]), sign_context)[0]
    grid = np.linspace(-target.length, 2 * target.length, 1025)
    spline_partition_error = float(np.max(np.abs(spline_weights(grid, target).sum(axis=-1) - 1)))
    derivative_sum_error = float(np.max(np.abs(spline_weights(grid, target, 1).sum(axis=-1))))
    q = uniform_quadratic_variation(target, diffusion)
    b = incidence_matrix(target.n_cells)
    h = target.length / target.n_cells
    # The declared mass-flux amplitude at uniform density is sqrt(2D/(N h L)).
    flux_noise = rng.normal(size=(4096, target.n_cells)) * np.sqrt(2 * diffusion / (n_particles * h * target.length))
    mass_noise = flux_noise @ b.T
    empirical_noise_covariance = n_particles * np.cov(mass_noise, rowvar=False)
    expected_noise_covariance = (2 * diffusion / (h * target.length)) * (b @ b.T)
    noise_error = float(np.linalg.norm(empirical_noise_covariance - expected_noise_covariance) /
                        np.linalg.norm(expected_noise_covariance))
    checks = {
        "fast_direct_force_maximum_error": force_difference,
        "two_particle_attractive_force": sign_force,
        "two_particle_force_expected": [0.35, -0.35],
        "spline_partition_of_unity_error": spline_partition_error,
        "spline_derivative_sum_error": derivative_sum_error,
        "uniform_observation_quadratic_variation_row_sum_error": float(np.max(np.abs(q.sum(axis=-1)))),
        "uniform_observation_quadratic_variation_minimum_eigenvalue": float(np.linalg.eigvalsh(q).min()),
        "finite_volume_noise_mass_sum_error": float(np.max(np.abs(mass_noise.sum(axis=-1)))),
        "finite_volume_noise_relative_covariance_error": noise_error,
        "finite_volume_noise_n_independent_draws": len(mass_noise),
    }
    deterministic_pass = (
        force_difference < 1e-12 and np.max(np.abs(sign_force - [0.35, -0.35])) < 1e-12
        and spline_partition_error < 1e-12 and derivative_sum_error < 1e-12
        and checks["uniform_observation_quadratic_variation_row_sum_error"] < 1e-10
        and checks["uniform_observation_quadratic_variation_minimum_eigenvalue"] > -1e-10
        and checks["finite_volume_noise_mass_sum_error"] < 1e-12
    )
    free_context = Context(cos_coeffs=(0.35, -0.15, 0.05), sin_coeffs=(0.2, 0.05, -0.1),
                           n_particles=n_particles, diffusion=diffusion)
    free = simulate_particles(free_context, target, times, dt, seed + 1001, n_paths)
    estimated = estimate_queries(free, queries, free_context, n_bootstrap=n_bootstrap, seed=seed + 1002)
    truth = free_particle_statistics(free_context, target, queries)
    uniform_truth = free_particle_statistics(Context(n_particles=n_particles, diffusion=diffusion), target, queries)
    std = np.sqrt(np.maximum(np.diag(uniform_truth.scaled_covariance), 1e-15))
    covariance_scale = np.outer(std, std)
    empirical_covariance = np.asarray(estimated["scaled_covariance"])
    covariance_ci = np.asarray(estimated["uncertainty"]["scaled_covariance_ci95"])
    mean_ci = np.asarray(estimated["uncertainty"]["mean_ci95"])
    mean_covered = (mean_ci[:, 0] <= truth.mean) & (truth.mean <= mean_ci[:, 1])
    covariance_covered = (covariance_ci[..., 0] <= truth.scaled_covariance) & (truth.scaled_covariance <= covariance_ci[..., 1])
    maximum_halfwidth = float(np.max((covariance_ci[..., 1] - covariance_ci[..., 0]) / (2 * covariance_scale)))
    initial = sample_initial(free_context, min(n_paths, 32), seeded_rng(seed, "projection-initial"))
    particle_initial = observe_particles(initial, target)
    projection = []
    spatial_noise = []
    for size in (target.n_cells, 2 * target.n_cells, 4 * target.n_cells):
        histograms = np.asarray([np.histogram(x, bins=size, range=(0, target.length))[0] / n_particles for x in initial])
        field_initial = observe_field(histograms, target)
        error = field_initial - particle_initial
        projection.append({"n_grid": size,
                           "rms_initial_observation_difference": float(np.sqrt(np.mean(error ** 2))),
                           "maximum_initial_observation_difference": float(np.max(np.abs(error))),
                           "paired_initial_particles": True})
        bg = incidence_matrix(size)
        g = observation_matrix(target, size)
        field_q = (2 * diffusion * size / target.length**2) * (g @ bg @ bg.T @ g.T)
        spatial_noise.append({"n_grid": size,
                              "relative_uniform_observation_covariance_rate_error": float(np.linalg.norm(field_q - q) / np.linalg.norm(q))})
    interacting_context = Context(cos_coeffs=(0.2, -0.1, 0.05), sin_coeffs=(0.1, 0.0, 0.03),
                                  kappa=0.7, a2=0.15, n_particles=n_particles, diffusion=diffusion)
    refined = coupled_refinement(interacting_context, target, times, dt, seed + 2001, refinement_paths)
    refinements = {
        f"{coarse}_to_{fine}": _paired_refinement(refined[coarse], refined[fine], queries, interacting_context,
                                                seeded_rng(seed, "paired-bootstrap", coarse, fine), n_bootstrap)
        for coarse, fine in (("coarse", "half"), ("half", "quarter"))
    }
    refinement_ci = np.asarray(refinements["half_to_quarter"]["scaled_covariance_difference_ci95"])
    certification = _certify_reference(deterministic_pass, covariance_ci, truth.scaled_covariance,
                                       covariance_scale, refinement_ci, tolerance)
    return json_safe({
        "status": "passed" if deterministic_pass else "failed",
        "target": asdict(target), "target_id": target.target_id,
        "settings": {"dt": dt, "save_interval": save_interval, "horizon": horizon, "seed": seed,
                     "n_paths": n_paths, "n_particles": n_particles, "diffusion": diffusion, "refinement_paths": refinement_paths,
                     "queries": asdict(queries)},
        "implementation_checks": checks,
        "free_particles": {
            "context": asdict(free_context), "query_labels": query_labels(queries),
            "analytic_method": "Spectral heat-semigroup integration of the actual periodic spline-filtered observable, including spline aliases",
            "analytic_mean": truth.mean, "analytic_scaled_covariance": truth.scaled_covariance,
            "estimate": estimated,
            "mean_ci95_covered_fraction": float(mean_covered.mean()),
            "covariance_ci95_covered_fraction": float(covariance_covered.mean()),
            "scaled_covariance_normalized_rmse": float(np.sqrt(np.mean(((empirical_covariance - truth.scaled_covariance) / covariance_scale) ** 2))),
            "maximum_normalized_covariance_ci95_halfwidth": maximum_halfwidth,
        },
        "initialization_projection": projection,
        "spatial_noise_refinement": spatial_noise,
        "interacting_timestep_refinement": {"context": asdict(interacting_context), "comparisons": refinements},
        "reference_certification": certification,
        "elapsed_seconds": time.perf_counter() - started,
    })
