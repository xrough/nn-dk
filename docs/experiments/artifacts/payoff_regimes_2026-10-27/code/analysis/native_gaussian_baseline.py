"""Deterministic mean-field plus native linear-noise statistical baseline.

The coefficients are known finite-volume DK coefficients. Linearization gives
an *unstopped Gaussian approximation*, not a sampler of the stopped DK law:
    mu' = b(mu), C' = Jb(mu) C + C Jb(mu)^T + S(mu) S(mu)^T.
C denotes N-scaled covariance. No training or particle Monte Carlo is needed.
Linear time integrals are included by augmenting the state with q'=W_running m.
The method retains joint-time covariance and flags deterministic boundary
encounters; Gaussian stopping/tail bias remains unresolved even in the interior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402

from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.types import Context  # noqa: E402

LAW = "unstopped-native-linear-noise-approximation-v1"
EULER_LAW = "unstopped-native-linearized-euler-gaussian-v1"


@lru_cache(maxsize=64)
def _geometry(g: int, length: float, kappa: float, a2: float):
    h = length/g
    cells, faces = (np.arange(g)+.5)*h, (np.arange(g)+1)*h
    moments = np.stack((np.cos(cells), np.sin(cells), np.cos(2*cells),
                        np.sin(2*cells)), axis=-1)
    reconstruction = np.stack((np.sin(faces), -np.cos(faces), np.sin(2*faces),
                               -np.cos(2*faces)), axis=0)
    scales = np.array((-kappa*np.sinc(h/(2*np.pi)),)*2
                      +(2*a2*np.sinc(h/np.pi),)*2)
    force = (reconstruction.T*scales) @ moments.T
    identity = np.eye(g)
    neighbor = np.roll(identity, 1, axis=1)
    return moments, reconstruction, scales, force, identity, neighbor


def drift_jacobian_covariance(mean, context: Context, spec: NativeDKSpec):
    """Return b(mu), its exact Jacobian, and SS^T (before division by N).

    The force uses the four exact native Fourier moments. The Jacobian is
    materialized only for this small G-dimensional deterministic covariance ODE.
    """
    m = np.asarray(mean, dtype=float)
    if (m.shape != (spec.n_cells,) or not np.isfinite(m).all() or np.any(m < 0)):
        raise ValueError("Finite nonnegative native mean masses [G] required; no clipping")
    moments, reconstruction, scales, force, identity, neighbor = _geometry(
        spec.n_cells, spec.length, context.kappa, context.a2)
    following = np.roll(m, -1)
    face = (m+following)/(2*spec.h)
    velocity = ((m @ moments)*scales) @ reconstruction
    flux = face*velocity-context.diffusion/spec.h**2*(following-m)
    jac_flux = (velocity[:, None]/(2*spec.h)*(identity+neighbor)
                +face[:, None]*force+context.diffusion/spec.h**2*(identity-neighbor))
    jacobian = np.roll(jac_flux, 1, axis=0)-jac_flux
    drift = np.roll(flux, 1)-flux
    squared_amplitude = context.diffusion/spec.h**2*(m+following)
    covariance = np.diag(squared_amplitude+np.roll(squared_amplitude, 1))
    cells = np.arange(spec.n_cells)
    covariance[cells, (cells+1) % spec.n_cells] = -squared_amplitude
    covariance[(cells+1) % spec.n_cells, cells] = -squared_amplitude
    return drift, jacobian, covariance


def iid_scaled_covariance(probabilities):
    """N Cov(histogram) for N IID particles with declared cell probabilities."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Nonnegative cell probabilities summing to one required")
    return np.diag(p)-np.outer(p, p)


def _weights(value, g, *, identity_default=False):
    if value is None:
        return np.eye(g) if identity_default else np.empty((0, g))
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[-1] != g or not np.isfinite(array).all():
        raise ValueError("Finite linear observable weights [observables,G] required")
    return array


def linear_gaussian(initial, context: Context, spec: NativeDKSpec, times, step,
                    initial_scaled_covariance=None, terminal_weights=None, running_weights=None,
                    *, return_joint=True, scheme="continuous-rk4"):
    """Predict native linear-Gaussian moments using deterministic fixed-step RK4.

    ``initial`` is a fixed conditional histogram or the ensemble mean histogram.
    The default initial covariance is zero (conditional prediction). For IID
    ensemble initialization, explicitly pass ``iid_scaled_covariance(p)``.
    At every requested time, outputs concatenate W_terminal*m(t) and
    integral_0^t W_running*m(s) ds. ``mean`` has shape [T,O],
    ``scaled_covariance`` [T,O,O], and ``joint_scaled_covariance`` [T,O,T,O].
    Full native/augmented means and covariances are retained for audit. All
    covariance outputs are N-scaled. No stopping, clipping or PSD repair occurs.

    ``scheme='linearized-euler'`` instead gives the Gaussian approximation to
    native finite-Euler dynamics: F=I+step*J, C_next=F*C*F.T+step*Q. This is a
    distinct numerical target. Query times must lie on its fixed step grid, and
    running rewards use the matching left-point Euler accumulation convention.

    Interval fundamental matrices are restarted at identity; cross covariances
    are propagated directly. This avoids inverse heat propagators, which become
    ill-conditioned at long times. ``step`` is a maximum RK4 step, each interval
    being divided into equal substeps, so query times are represented exactly.
    """
    g = spec.n_cells
    m = np.asarray(initial, dtype=float)
    times = np.asarray(times, dtype=float)
    if (m.shape != (g,) or not np.isfinite(m).all() or np.any(m < 0)
            or not np.isclose(m.sum(), 1, atol=1e-10, rtol=0)
            or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.isfinite(step) or step <= 0):
        raise ValueError("Unit-mass initial [G], increasing nonnegative times and positive step required")
    if scheme not in ("continuous-rk4", "linearized-euler"):
        raise ValueError("Use continuous-rk4 or linearized-euler scheme")
    if scheme == "linearized-euler" and not np.allclose(
            times/step, np.rint(times/step), atol=1e-10, rtol=0):
        raise ValueError("Linearized Euler queries must lie on the fixed step grid")
    covariance = (np.zeros((g, g)) if initial_scaled_covariance is None else
                  np.asarray(initial_scaled_covariance, dtype=float))
    if (covariance.shape != (g, g) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
            or np.linalg.eigvalsh(covariance).min() < -1e-12
            or not np.allclose(covariance.sum(-1), 0, atol=1e-10, rtol=0)):
        raise ValueError("Finite symmetric PSD mass-zero N-scaled initial covariance required")
    terminal, running = _weights(terminal_weights, g, identity_default=True), _weights(
        running_weights, g)
    r = len(running)
    d = g+r
    mean = np.concatenate((m, np.zeros(r)))
    full_covariance = np.zeros((d, d))
    full_covariance[:g, :g] = covariance
    output = np.zeros((len(terminal)+r, d))
    output[:len(terminal), :g] = terminal
    output[len(terminal):, g:] = np.eye(r)
    means, covariances, transitions = [], [], []
    minimum_mean = float(m.min())
    symmetry_error = 0.
    integration_steps = 0

    def parts(mu):
        nonlocal minimum_mean
        minimum_mean = min(minimum_mean, float(mu[:g].min()))
        drift, jacobian, forcing = drift_jacobian_covariance(mu[:g], context, spec)
        if r:
            a = np.zeros((d, d))
            a[:g, :g] = jacobian
            a[g:, :g] = running
            q = np.zeros((d, d))
            q[:g, :g] = forcing
            drift = np.concatenate((drift, running @ mu[:g]))
        else:
            a, q = jacobian, forcing
        return drift, a, q

    def rhs(mu, cov, transition):
        drift, a, q = parts(mu)
        return drift, a @ cov+cov @ a.T+q, a @ transition

    previous_time = 0.
    started = time.perf_counter()
    for target_time in times:
        length = float(target_time-previous_time)
        count = (int(np.rint(length/step)) if scheme == "linearized-euler" else
                 int(np.ceil(length/step))) if length else 0
        dt = length/count if count else 0.
        velocity = abs(context.kappa)+2*abs(context.a2)
        if (scheme == "continuous-rk4"
                and dt*(4*context.diffusion/spec.h**2+2*velocity/spec.h) > 2.5):
            raise ValueError("RK4 covariance stability bound violated; refine step")
        if (scheme == "linearized-euler"
                and dt*(2*context.diffusion/spec.h**2+velocity/spec.h) > 1):
            raise ValueError("Native linearized Euler stability condition violated")
        transition = np.eye(d)
        for _ in range(count):
            if scheme == "linearized-euler":
                drift, a, q = parts(mean)
                f = np.eye(d)+dt*a
                mean = mean+dt*drift
                full_covariance = f @ full_covariance @ f.T+dt*q
                transition = f @ transition
                continue
            first = rhs(mean, full_covariance, transition)
            second = rhs(mean+dt/2*first[0], full_covariance+dt/2*first[1],
                         transition+dt/2*first[2])
            third = rhs(mean+dt/2*second[0], full_covariance+dt/2*second[1],
                        transition+dt/2*second[2])
            fourth = rhs(mean+dt*third[0], full_covariance+dt*third[1], transition+dt*third[2])
            mean = mean+dt/6*(first[0]+2*second[0]+2*third[0]+fourth[0])
            full_covariance = full_covariance+dt/6*(first[1]+2*second[1]+2*third[1]+fourth[1])
            transition = transition+dt/6*(first[2]+2*second[2]+2*third[2]+fourth[2])
        integration_steps += count
        minimum_mean = min(minimum_mean, float(mean[:g].min()))
        symmetry_error = max(symmetry_error, float(np.abs(full_covariance-full_covariance.T).max()))
        means.append(mean.copy())
        covariances.append(full_covariance.copy())
        transitions.append(transition)
        previous_time = target_time
    means, covariances = np.asarray(means), np.asarray(covariances)
    projected_mean = means @ output.T
    projected_covariance = np.einsum("ai,tij,bj->tab", output, covariances, output)
    joint = None
    joint_projected = None
    if return_joint:
        joint = np.zeros((len(times), d, len(times), d))
        for i in range(len(times)):
            joint[i, :, i, :] = covariances[i]
            cross = covariances[i]
            for j in range(i+1, len(times)):
                cross = cross @ transitions[j].T
                joint[i, :, j, :] = cross
                joint[j, :, i, :] = cross.T
        joint_projected = np.einsum("ai,tisj,bj->tasb", output, joint, output)
    elapsed = time.perf_counter()-started
    min_eigenvalue = min(float(np.linalg.eigvalsh(cov).min()) for cov in covariances)
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Nonfinite Gaussian approximation; no results may be averaged")
    return {"law": LAW if scheme == "continuous-rk4" else EULER_LAW,
            "times": times, "mean": projected_mean,
            "scaled_covariance": projected_covariance,
            "joint_scaled_covariance": joint_projected,
            "mean_states": means[:, :g], "mean_augmented": means,
            "scaled_covariance_states": covariances[:, :g, :g],
            "scaled_covariance_augmented": covariances,
            "joint_scaled_covariance_augmented": joint,
            "terminal_weights": terminal, "running_weights": running,
            "diagnostics": {"step": float(step), "integration_steps": integration_steps,
                            "integration_seconds": elapsed, "integrator": (
                                "fixed-step-RK4" if scheme == "continuous-rk4"
                                else "linearized-Euler-covariance-recursion"),
                            "scheme": scheme,
                            "time_target": "continuous-linear-noise-ODE" if scheme == "continuous-rk4"
                            else "fixed-step-native-Euler-linearization",
                            "running_reward": "unstopped-integral" if scheme == "continuous-rk4"
                            else "unstopped-left-point-sum",
                            "killed_or_stopped_reward_equivalence": False,
                            "minimum_deterministic_mass": minimum_mean,
                            "deterministic_boundary_encountered": minimum_mean <= spec.mass_stop,
                            "stopped_dk_equivalence": False, "stopping_effects": "unresolved",
                            "law_approximation_error": "unresolved",
                            "covariance_scale": "N-times-physical-covariance",
                            "initialization": "conditional" if initial_scaled_covariance is None
                            else "supplied-N-scaled-initial-covariance",
                            "minimum_covariance_eigenvalue": min_eigenvalue,
                            "covariance_symmetry_error": symmetry_error,
                            "mean_mass_error": float(np.abs(means[:, :g].sum(-1)-1).max()),
                            "covariance_mass_error": float(np.abs(
                                covariances[:, :g, :g].sum(-1)).max()),
                            "no_covariance_repair": True}}


def sample_gaussian_process(result, n_particles, n_paths, seed=0):
    """Sample the declared joint Gaussian approximation, without stopping/repair.

    Tiny roundoff-negative eigenvalues within 1e-10 of the covariance scale are
    excluded from a singular Gaussian factor; material negative eigenvalues
    raise. This is factorization of a singular mass-zero law, never density
    clipping. Returned Gaussian samples can have negative native masses.
    """
    if (result.get("law") not in (LAW, EULER_LAW)
            or result["joint_scaled_covariance_augmented"] is None
            or n_particles < 1 or n_paths < 1):
        raise ValueError("Joint native Gaussian approximation and positive counts required")
    means = np.asarray(result["mean_augmented"])
    joint = np.asarray(result["joint_scaled_covariance_augmented"])
    covariance = joint.reshape(means.size, means.size)
    asymmetry = np.abs(covariance-covariance.T).max()
    if asymmetry > 1e-10:
        raise ValueError("Joint covariance is not symmetric")
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 1e-10*max(1., float(np.abs(values).max()))
    if values.min() < -tolerance:
        raise ValueError("Joint Gaussian covariance has a material negative eigenvalue; refine integration")
    selected = values > 0
    factor = vectors[:, selected]*np.sqrt(values[selected]/n_particles)
    noise = np.random.default_rng(seed).normal(size=(n_paths, selected.sum()))
    samples = (means.flatten()+noise @ factor.T).reshape(n_paths, *means.shape)
    return {"samples_augmented": samples,
            "diagnostics": {"law": result["law"], "seed": seed, "n_paths": n_paths,
                            "minimum_factor_eigenvalue": float(values.min()),
                            "zero_or_roundoff_negative_modes": int((values <= 0).sum()),
                            "stopping_applied": False, "density_repair_applied": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--scheme", choices=("continuous-rk4", "linearized-euler"),
                        default="continuous-rk4")
    args = parser.parse_args()
    spec, context = NativeDKSpec(n_cells=32), Context(kappa=.75, a2=.1, n_particles=2048)
    centers = (np.arange(32)+.5)*spec.h
    initial = np.exp(.2*np.cos(centers))
    initial /= initial.sum()
    weights = np.stack((np.cos(centers), np.sin(centers), np.cos(2*centers),
                        np.sin(2*centers)))*np.sqrt(2)
    weights *= np.repeat(np.sinc(np.arange(1, 3)*spec.h/(2*np.pi)), 2)[:, None]
    records = []
    for step in (2**-7, 2**-8, 2**-9, 2**-10):
        for initial_covariance in (None, iid_scaled_covariance(initial)):
            arguments = (initial, context, spec, (.015625, .03125, .0625), step)
            kwargs = {"initial_scaled_covariance": initial_covariance, "terminal_weights": weights,
                      "scheme": args.scheme}
            result = linear_gaussian(*arguments, **kwargs)
            samples = []
            for _ in range(args.repeats):
                tick = time.perf_counter()
                result = linear_gaussian(*arguments, **kwargs)
                samples.append(time.perf_counter()-tick)
            records.append({"step": step, "initialization": result["diagnostics"]["initialization"],
                            "median_seconds": float(np.median(samples)), "samples_seconds": samples,
                            "mean": result["mean"].tolist(),
                            "scaled_covariance": result["scaled_covariance"].tolist(),
                            "joint_scaled_covariance": result["joint_scaled_covariance"].tolist(),
                            "mean_states": result["mean_states"].tolist(),
                            "scaled_covariance_states": result["scaled_covariance_states"].tolist(),
                            "diagnostics": result["diagnostics"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "law": result["law"], "records": records,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scheme": args.scheme, "numpy_version": np.__version__,
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "context_id": context.context_id, "native_spec_id": spec.target_id,
        "physics": {"G": spec.n_cells, "length": spec.length, "rho_stop": spec.rho_stop,
                    "N": context.n_particles, "D": context.diffusion,
                    "kappa": context.kappa, "a2": context.a2},
        "rng": "none; deterministic declared profile probabilities",
        "initial": initial.tolist(), "terminal_weights": weights.tolist(),
        "timing_scope": "Warm deterministic full-state covariance ODE+joint projection; no imports/I/O/training",
    }, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records)}))


if __name__ == "__main__":
    main()
