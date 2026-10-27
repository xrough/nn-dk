"""Posthoc local generator diagnostic; no training or trajectory simulation.

Run with the project Python, for example::

    .venv/bin/python analysis/probe_uniform_generator.py runs/a-first-batch

The Fourier spectrum describes the Jacobian at the uniform mass state only.
It is not a nonlinear rollout-error bound or a weak-convergence certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nn_dk.convergence_config import ConvergenceConfig
from nn_dk.data import source_hash, write_json
from nn_dk.dk_training import load_coefficient_model
from nn_dk.native import native_coefficients


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_checkpoint(path: Path, config: ConvergenceConfig) -> dict:
    """Differentiate the autonomous drift and inspect all nonconstant modes."""
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("complete", False):
        raise ValueError(f"Checkpoint is incomplete: {path}")
    model = load_coefficient_model(path)
    config.spec.require_compatible(model.spec)
    context, spec = config.context("uniform"), model.spec
    if context.kappa != 0 or context.a2 != 0 or any(context.cos_coeffs + context.sin_coeffs):
        raise ValueError("Only the noninteracting uniform initial-profile case is supported")
    g, h, diffusion = spec.n_cells, spec.h, context.diffusion
    masses = torch.full((g,), 1 / g, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda state: model.coefficients(state, context).drift, masses
    ).detach().numpy()
    coefficients = model.numpy_coefficients(masses.detach().numpy(), context)
    oracle = native_coefficients(masses.detach().numpy(), context, spec)
    circulant = sum(np.roll(jacobian, (shift, shift), axis=(0, 1)) for shift in range(g)) / g
    denominator = max(float(np.linalg.norm(jacobian)), np.finfo(float).tiny)
    exact_jacobian = diffusion / h**2 * (
        np.roll(np.eye(g), 1, axis=0) + np.roll(np.eye(g), -1, axis=0) - 2 * np.eye(g)
    )
    rows = []
    exact_eigenvalue_error, maximum_fourier_residual = 0., 0.
    for k in range(1, g):
        vector = np.exp(2j * np.pi * k * np.arange(g) / g) / np.sqrt(g)
        learned = np.vdot(vector, jacobian @ vector)
        exact = -4 * diffusion / h**2 * np.sin(np.pi * k / g)**2
        exact_eigenvalue_error = max(exact_eigenvalue_error,
                                    float(abs(np.vdot(vector, exact_jacobian @ vector) - exact)))
        residual = float(np.linalg.norm(jacobian @ vector - learned * vector)
                         / max(np.linalg.norm(jacobian @ vector), np.finfo(float).tiny))
        maximum_fourier_residual = max(maximum_fourier_residual, residual)
        learned_noise = float(np.vdot(vector, coefficients.covariance @ vector).real)
        exact_noise = float(np.vdot(vector, oracle.covariance @ vector).real)
        rows.append({
            "mode": k, "signed_mode": k if k <= g // 2 else k - g,
            "exact_decay_eigenvalue": float(exact),
            "learned_eigenvalue_real": float(learned.real),
            "learned_eigenvalue_imaginary": float(learned.imag),
            "relative_complex_eigenvalue_error": float(abs(learned - exact) / abs(exact)),
            "inferred_diffusion_ratio": float(learned.real / exact),
            "imaginary_phase_over_exact_decay": float(learned.imag / abs(exact)),
            "fourier_eigenvector_relative_residual": residual,
            "learned_noise_quadratic_variation_before_division_N": learned_noise,
            "exact_noise_quadratic_variation_before_division_N": exact_noise,
            "noise_quadratic_variation_ratio": learned_noise / exact_noise,
            "local_stationary_variance_ratio": (
                float(learned_noise / (-2 * learned.real) * g) if learned.real < 0 else None),
        })
    # A conserved physical perturbation checks autograd without changing total mass.
    direction = np.cos(2 * np.pi * np.arange(g) / g)
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    epsilon = 1e-7
    center = masses.detach().numpy()
    plus = model.numpy_coefficients(center + epsilon * direction, context).drift
    minus = model.numpy_coefficients(center - epsilon * direction, context).drift
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic_direction = jacobian @ direction
    derivative_error = float(np.linalg.norm(finite_difference - analytic_direction)
                             / max(np.linalg.norm(analytic_direction), np.finfo(float).tiny))
    oracle_finite_difference = (
        native_coefficients(center + epsilon * direction, context, spec).drift
        - native_coefficients(center - epsilon * direction, context, spec).drift
    ) / (2 * epsilon)
    oracle_derivative_error = float(np.linalg.norm(oracle_finite_difference - exact_jacobian @ direction)
                                    / np.linalg.norm(exact_jacobian @ direction))
    if derivative_error > 1e-6 or oracle_derivative_error > 1e-6 or exact_eigenvalue_error > 1e-10:
        raise AssertionError("Jacobian/Fourier finite-difference verification failed")
    provenance = checkpoint["provenance"]
    return {
        "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
        "checkpoint_identity_hash": provenance["identity_hash"],
        "training_mode": provenance["mode"], "optimizer_seed": provenance["seed"],
        "width": provenance["width"], "epochs": checkpoint["epoch"],
        "best_validation_loss": checkpoint["best_loss"],
        "uniform_drift_maximum_absolute": float(np.max(np.abs(coefficients.drift))),
        "jacobian_conservation_maximum_column_sum": float(np.max(np.abs(jacobian.sum(0)))),
        "jacobian_relative_circulant_residual": float(np.linalg.norm(jacobian - circulant) / denominator),
        "maximum_fourier_eigenvector_relative_residual": maximum_fourier_residual,
        "maximum_real_nonconstant_fourier_eigenvalue": max(row["learned_eigenvalue_real"] for row in rows),
        "all_nonconstant_fourier_modes_dissipative": all(row["learned_eigenvalue_real"] < 0 for row in rows),
        "uniform_face_amplitude_mean": float(np.mean(coefficients.face_amplitudes)),
        "uniform_face_amplitude_spread": float(np.ptp(coefficients.face_amplitudes)),
        "exact_uniform_face_amplitude": float(oracle.face_amplitudes[0]),
        "uniform_face_amplitude_relative_error": float(np.linalg.norm(coefficients.face_amplitudes - oracle.face_amplitudes)
                                                     / np.linalg.norm(oracle.face_amplitudes)),
        "uniform_covariance_relative_frobenius_error": float(np.linalg.norm(coefficients.covariance - oracle.covariance)
                                                          / np.linalg.norm(oracle.covariance)),
        "verification": {
            "autograd_vs_mode1_centered_difference_relative_error": derivative_error,
            "oracle_matrix_vs_mode1_centered_difference_relative_error": oracle_derivative_error,
            "oracle_matrix_fourier_eigenvalue_maximum_absolute_error": exact_eigenvalue_error,
            "finite_difference_epsilon": epsilon,
        },
        "modes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", help="Default: all complete uniform checkpoints")
    parser.add_argument("--output", type=Path, help="Default: RUN/report")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    config_path, manifest_path = args.run / "config.json", args.run / "manifest.json"
    config = ConvergenceConfig(**json.loads(config_path.read_text()))
    manifest = json.loads(manifest_path.read_text())
    if config.config_hash != manifest["config_hash"]:
        raise ValueError("Saved configuration does not match the run manifest")
    current_hash = source_hash()
    if current_hash != manifest["source_hash"]:
        raise ValueError("Current source differs from the run; probe its recorded source revision")
    if "uniform" not in config.cases:
        raise ValueError("Run must declare the uniform case")
    paths = args.checkpoint or sorted((args.run / "checkpoints" / "uniform").glob("*.pt"))
    if not paths:
        raise ValueError("No complete uniform coefficient checkpoints found")
    uniform_directory = (args.run / "checkpoints" / "uniform").resolve()
    if any(path.resolve().parent != uniform_directory for path in paths):
        raise ValueError("Checkpoint must belong to the uniform-case directory of the selected run")
    skipped = []
    if args.checkpoint is None:
        complete = []
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("complete", False):
                complete.append(path)
            else:
                skipped.append(str(path.resolve()))
        paths = complete
        if not paths:
            raise ValueError("No complete uniform coefficient checkpoints found")
    output = args.output or args.run / "report"
    output.mkdir(parents=True, exist_ok=True)
    results = [probe_checkpoint(path, config) for path in paths]
    report = {
        "analysis": "uniform-local-generator-fourier-probe-v1",
        "scope": "Autonomous generator linearized at uniform masses; no resimulation or fitting",
        "limitation": "Local Jacobian spectra are not nonlinear rollout bounds or weak-convergence certificates",
        "fourier_convention": "v_j=exp(+2*pi*i*k*j/G)/sqrt(G); eigenvalue=v.conj().T J v",
        "covariance_convention": "Quadratic variation before division by N",
        "local_stationary_variance_convention": (
            "Hypothetical unstopped linearization with noise frozen at uniform masses: "
            "variance of the sqrt(N)-scaled unitary mass mode is A_k/(-2 Re(lambda_k)), "
            "normalized by the native value 1/G. This is not a stationary-law assertion "
            "for either stopped or nonlinear dynamics."),
        "run_directory": str(args.run.resolve()),
        "run_manifest_status": manifest.get("status"),
        "run_config_hash": manifest["config_hash"], "run_source_hash": manifest["source_hash"],
        "current_source_hash": current_hash,
        "source_matches_run": current_hash == manifest["source_hash"],
        "config_file_sha256": sha256(config_path), "manifest_file_sha256": sha256(manifest_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "n_cells": config.n_cells, "n_particles": config.n_particles, "diffusion": config.diffusion,
        "skipped_incomplete_checkpoints": skipped,
        "results": results,
    }
    write_json(output / "generator_probe.json", report)
    flat = [dict(checkpoint=Path(result["checkpoint"]).name, optimizer_seed=result["optimizer_seed"],
                 training_mode=result["training_mode"], width=result["width"], **row)
            for result in results for row in result["modes"]]
    with (output / "generator_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for result in results:
            rows = [row for row in result["modes"] if row["mode"] <= config.n_cells // 2]
            label = Path(result["checkpoint"]).stem
            axes[0].plot([row["mode"] for row in rows],
                         [row["inferred_diffusion_ratio"] for row in rows], "o-", label=label)
            axes[1].plot([row["mode"] for row in rows],
                         [row["imaginary_phase_over_exact_decay"] for row in rows], "o-", label=label)
        axes[0].axhline(1, color="black", linestyle="--", label="Exact DK")
        axes[1].axhline(0, color="black", linestyle="--")
        for axis in axes:
            axis.set_xlabel("Fourier mode")
            axis.grid(alpha=.25)
        axes[0].set_ylabel("Local effective diffusion / exact diffusion")
        axes[1].set_ylabel("Imaginary phase / exact decay magnitude")
        axes[0].legend(fontsize=7)
        fig.suptitle("Uniform-state local generator diagnostic")
        fig.savefig(output / "generator_probe.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"report": str(output / "generator_probe.json"), "checkpoints": len(results),
                      "source_matches_run": report["source_matches_run"]}, indent=2))


if __name__ == "__main__":
    main()
