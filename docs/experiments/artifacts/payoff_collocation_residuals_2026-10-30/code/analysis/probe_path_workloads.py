"""Fair classical full-path and pathwise-sensitivity workload probes.

All solvers consume the same physical face Brownian increments. The NumPy
forward/JVP solvers and Torch reverse-mode solver use four Fourier moments for
the native interaction, rather than materializing G by G force matrices at
every step. Timings include full-path construction and interpolated stopping;
they exclude imports, initialization, noise generation, and serialization.
These are finite-Euler workloads, not continuous-DK accuracy certificates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.types import Context  # noqa: E402


@lru_cache(maxsize=32)
def _geometry(g: int, length: float):
    h = length / g
    cells, faces = (np.arange(g) + .5) * h, (np.arange(g) + 1) * h
    # Columns are cos(x), sin(x), cos(2x), sin(2x).
    moments = np.stack((np.cos(cells), np.sin(cells), np.cos(2*cells),
                        np.sin(2*cells)), axis=-1)
    reconstruction = np.stack((np.sin(faces), -np.cos(faces), np.sin(2*faces),
                               -np.cos(2*faces)), axis=0)
    return moments, reconstruction


def _validate(initial, increments, context, spec, dt):
    initial, increments = np.asarray(initial, dtype=float), np.asarray(increments, dtype=float)
    if (initial.ndim != 2 or initial.shape[-1] != spec.n_cells or initial.shape[0] < 1
            or increments.ndim != 3 or increments.shape[0] != initial.shape[0]
            or increments.shape[-1] != spec.n_cells or increments.shape[1] < 1
            or not np.isfinite(dt) or dt <= 0
            or not np.isfinite(initial).all() or not np.isfinite(increments).all()
            or np.any(initial < 0) or not np.allclose(initial.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Finite nonnegative unit-mass initial [B,G] and noise [B,K,G] required")
    velocity = abs(context.kappa) + 2*abs(context.a2)
    if (dt * (2*context.diffusion/spec.h**2 + velocity/spec.h) > 1 + 1e-12
            or velocity*spec.h > 2*context.diffusion + 1e-12):
        raise ValueError("Native centered Euler stability condition violated")
    return initial, increments


def _scales(context, spec):
    return np.array((-context.kappa*np.sinc(spec.h/(2*np.pi)),)*2
                    + (2*context.a2*np.sinc(spec.h/np.pi),)*2)


def fourier_coefficients(initial, context: Context, spec: NativeDKSpec):
    """Native drift and face amplitudes, before division by sqrt(N)."""
    masses = np.asarray(initial, dtype=float)
    moments, reconstruction = _geometry(spec.n_cells, spec.length)
    velocity = ((masses @ moments) * _scales(context, spec)) @ reconstruction
    following = np.roll(masses, -1, axis=-1)
    flux = ((masses + following)/(2*spec.h)*velocity
            - context.diffusion/spec.h**2*(following-masses))
    return np.roll(flux, 1, axis=-1)-flux, np.sqrt(
        context.diffusion/spec.h**2*(masses+following))


def numpy_euler(initial, increments, context: Context, spec: NativeDKSpec, dt: float,
                *, initial_tangent=None, physics_tangent=None):
    """Fast full-path native Euler; optionally propagate one exact pathwise JVP.

    ``physics_tangent`` orders (diffusion, kappa, a2). ``initial_tangent`` is
    a conservative direction in native masses, not a derivative through IID
    histogram sampling. The first-crossing interpolation is differentiated
    piecewise; derivatives at switching/tied boundary events need not exist.
    """
    initial, increments = _validate(initial, increments, context, spec, dt)
    m = initial.copy()
    states = np.empty((len(m), increments.shape[1]+1, spec.n_cells))
    states[:, 0] = m
    stop = np.where((m <= spec.mass_stop).any(-1), 0., np.inf)
    active = np.isinf(stop)
    tangent = initial_tangent is not None or physics_tangent is not None
    if tangent:
        v = (np.zeros_like(m) if initial_tangent is None else
             np.broadcast_to(np.asarray(initial_tangent, dtype=float), m.shape).copy())
        dp = np.zeros(3) if physics_tangent is None else np.asarray(physics_tangent, dtype=float)
        if (dp.shape != (3,) or not np.isfinite(dp).all() or not np.isfinite(v).all()
                or not np.allclose(v.sum(-1), 0, atol=1e-12, rtol=0)):
            raise ValueError("Finite conservative initial tangent and physics tangent [D,kappa,a2] required")
        derivatives = np.empty_like(states)
        derivatives[:, 0] = v
        stop_derivative = np.zeros(len(m))
    moments, reconstruction = _geometry(spec.n_cells, spec.length)
    scale = _scales(context, spec)
    if tangent:
        dscale = np.array((-dp[1]*np.sinc(spec.h/(2*np.pi)),)*2
                         + (2*dp[2]*np.sinc(spec.h/np.pi),)*2)
    for index in range(increments.shape[1]):
        # Every frozen realization remains in both outputs and all averages.
        origin = np.where(active[:, None], m, 1/spec.n_cells)
        following = np.roll(origin, -1, axis=-1)
        face = (origin+following)/(2*spec.h)
        summary = origin @ moments
        velocity = (summary*scale) @ reconstruction
        flux = face*velocity - context.diffusion/spec.h**2*(following-origin)
        amplitude = np.sqrt(context.diffusion/spec.h**2*(origin+following))
        noise = amplitude*increments[:, index]/np.sqrt(context.n_particles)
        proposal = m + dt*(np.roll(flux, 1, axis=-1)-flux) + np.roll(noise, 1, axis=-1)-noise
        proposal = np.where(active[:, None], proposal, m)
        if tangent:
            dv = np.where(active[:, None], v, 0)
            vf = np.roll(dv, -1, axis=-1)
            dvelocity = ((dv @ moments)*scale+summary*dscale) @ reconstruction
            dflux = ((dv+vf)/(2*spec.h)*velocity + face*dvelocity
                     - context.diffusion/spec.h**2*(vf-dv)
                     - dp[0]/spec.h**2*(following-origin))
            da = (context.diffusion*(dv+vf)+dp[0]*(origin+following))/(2*spec.h**2*amplitude)
            dn = da*increments[:, index]/np.sqrt(context.n_particles)
            proposed_v = v + dt*(np.roll(dflux, 1, axis=-1)-dflux)+np.roll(dn, 1, axis=-1)-dn
            proposed_v = np.where(active[:, None], proposed_v, v)
        descending = (proposal <= spec.mass_stop) & (proposal < m)
        denominator = np.where(descending, m-proposal, 1.)
        ratios = np.where(descending, (m-spec.mass_stop)/denominator, 1.)
        cells = ratios.argmin(-1)
        rows = np.arange(len(m))
        fraction = ratios[rows, cells]
        crossed = active & (proposal <= spec.mass_stop).any(-1)
        if tangent:
            numerator = m[rows, cells]-spec.mass_stop
            den = denominator[rows, cells]
            dnum, dden = v[rows, cells], v[rows, cells]-proposed_v[rows, cells]
            dfraction = np.where(crossed, (dnum*den-numerator*dden)/den**2, 0.)
            new_v = (v+fraction[:, None]*(proposed_v-v)
                     + dfraction[:, None]*(proposal-m))
            v = np.where(active[:, None], new_v, v)
            stop_derivative = np.where(crossed, dt*dfraction, stop_derivative)
            derivatives[:, index+1] = v
        m = np.where(active[:, None], m+fraction[:, None]*(proposal-m), m)
        stop = np.where(crossed, (index+fraction)*dt, stop)
        active &= ~crossed
        states[:, index+1] = m
    result = {"states": states, "stopped_at": stop}
    if tangent:
        result.update(tangent=derivatives, stopped_at_tangent=stop_derivative)
    return result


def torch_fourier_euler(initial, increments, context: Context, spec: NativeDKSpec, dt: float,
                        *, physics=None):
    """Autograd native Euler with O(G K_int) Fourier interaction per step.

    Caller tensors stay attached to their graphs. Geometry is prepared once;
    there are no finite-check synchronizations or dense forces inside the loop.
    Physics is an optional differentiable tensor (diffusion, kappa, a2).
    """
    initial = torch.as_tensor(initial, dtype=torch.float64)
    increments = torch.as_tensor(increments, dtype=torch.float64, device=initial.device)
    _validate(initial.detach().cpu().numpy(), increments.detach().cpu().numpy(), context, spec, dt)
    p = (initial.new_tensor((context.diffusion, context.kappa, context.a2)) if physics is None
         else torch.as_tensor(physics, dtype=torch.float64, device=initial.device))
    if p.shape != (3,) or not bool(torch.isfinite(p).all()) or bool(p[0] <= 0):
        raise ValueError("Finite physics [D,kappa,a2] with positive D required")
    # Validate the supplied physics, rather than only a possibly different Context.
    pcheck = p.detach()
    if (float(dt*(2*pcheck[0]/spec.h**2+(pcheck[1].abs()+2*pcheck[2].abs())/spec.h)) > 1+1e-12
            or float((pcheck[1].abs()+2*pcheck[2].abs())*spec.h) > float(2*pcheck[0])+1e-12):
        raise ValueError("Native centered Euler stability condition violated")
    moments, reconstruction = (initial.new_tensor(array) for array in
                               _geometry(spec.n_cells, spec.length))
    scale = torch.stack((-p[1]*np.sinc(spec.h/(2*np.pi)),)*2
                        + (2*p[2]*np.sinc(spec.h/np.pi),)*2)
    current = initial
    stopped_at = torch.where((initial <= spec.mass_stop).any(-1),
                             torch.zeros_like(initial[:, 0]), torch.full_like(initial[:, 0], torch.inf))
    active = torch.isinf(stopped_at)
    states = [initial]
    for index in range(increments.shape[1]):
        origin = torch.where(active[:, None], current, torch.full_like(current, 1/spec.n_cells))
        following = torch.roll(origin, -1, dims=-1)
        velocity = ((origin @ moments)*scale) @ reconstruction
        flux = ((origin+following)/(2*spec.h)*velocity-p[0]/spec.h**2*(following-origin))
        noise = torch.sqrt(p[0]/spec.h**2*(origin+following))*increments[:, index]/np.sqrt(
            context.n_particles)
        proposal = (current+dt*(torch.roll(flux, 1, dims=-1)-flux)
                    + torch.roll(noise, 1, dims=-1)-noise)
        proposal = torch.where(active[:, None], proposal, current)
        descending = (proposal <= spec.mass_stop) & (proposal < current)
        denominator = torch.where(descending, current-proposal, torch.ones_like(current))
        ratios = torch.where(descending, (current-spec.mass_stop)/denominator,
                             torch.ones_like(current))
        fraction = ratios.min(-1).values
        crossed = active & (proposal <= spec.mass_stop).any(-1)
        current = torch.where(active[:, None], current+fraction[:, None]*(proposal-current), current)
        stopped_at = torch.where(crossed, (index+fraction)*dt, stopped_at)
        active = active & ~crossed
        states.append(current)
    return {"states": torch.stack(states, dim=1), "stopped_at": stopped_at}


def workload_values(states, spec: NativeDKSpec, dt: float, *, temperature=.05):
    """Three path tasks: terminal mode 3, density-energy integral, smooth maximum.

    Energy uses trapezoidal time integration of sum(m_i^2)/h. The smooth maximum
    is temperature*log(mean(exp(mode3/temperature))) over the full path. This is
    a specified differentiable path functional, not a first-passage surrogate.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive smooth-maximum temperature required")
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    weights = np.sqrt(2)*np.sinc(3*spec.h/(2*np.pi))*np.cos(3*centers)
    if isinstance(states, torch.Tensor):
        mode = states @ states.new_tensor(weights)
        energy = (states.square().sum(-1)/spec.h)
        energy = dt*(energy[:, 1:-1].sum(-1)+.5*(energy[:, 0]+energy[:, -1]))
        smoothmax = temperature*(torch.logsumexp(mode/temperature, dim=1)
                                  -np.log(mode.shape[1]))
        return torch.stack((mode[:, -1], energy, smoothmax), dim=-1)
    mode = np.asarray(states) @ weights
    energy = np.square(states).sum(-1)/spec.h
    energy = dt*(energy[:, 1:-1].sum(-1)+.5*(energy[:, 0]+energy[:, -1]))
    peak = mode.max(-1)
    smoothmax = peak+temperature*np.log(np.exp((mode-peak[:, None])/temperature).mean(-1))
    return np.stack((mode[:, -1], energy, smoothmax), axis=-1)


def workload_jvp(states, tangent, spec: NativeDKSpec, dt: float, *, temperature=.05):
    """Analytic derivative of the three workloads for a supplied state-path JVP."""
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    weights = np.sqrt(2)*np.sinc(3*spec.h/(2*np.pi))*np.cos(3*centers)
    mode, dmode = states @ weights, tangent @ weights
    denergy = 2*(states*tangent).sum(-1)/spec.h
    denergy = dt*(denergy[:, 1:-1].sum(-1)+.5*(denergy[:, 0]+denergy[:, -1]))
    unnormalized = np.exp((mode-mode.max(-1, keepdims=True))/temperature)
    softweights = unnormalized/unnormalized.sum(-1, keepdims=True)
    return np.stack((dmode[:, -1], denergy, (softweights*dmode).sum(-1)), axis=-1)


def workload_state_gradient(states, spec: NativeDKSpec, dt: float, *,
                            coefficients=(1., 1., 1.), temperature=.05):
    """Path-state gradient of a fixed scalarization of the three workloads."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError("Three finite workload scalarization coefficients required")
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    weights = np.sqrt(2)*np.sinc(3*spec.h/(2*np.pi))*np.cos(3*centers)
    mode = states @ weights
    softweights = np.exp((mode-mode.max(-1, keepdims=True))/temperature)
    softweights /= softweights.sum(-1, keepdims=True)
    gradient = coefficients[2]*softweights[..., None]*weights
    quadrature = np.full(states.shape[1], dt)
    quadrature[[0, -1]] *= .5
    gradient += coefficients[1]*2*states/spec.h*quadrature[None, :, None]
    gradient[:, -1] += coefficients[0]*weights
    return gradient


def numpy_adjoint(initial, increments, context: Context, spec: NativeDKSpec, dt: float,
                  *, path_gradient=None, coefficients=(1., 1., 1.)):
    """Native Euler forward plus analytic discrete adjoint, including stopping.

    The default scalar objective sums the three declared workloads over paths.
    An arbitrary supplied path-state cotangent [B,K+1,G] can replace it. Return
    one initial gradient per realization and physical gradients [B,3]; sum the
    latter over B to differentiate a shared (D,kappa,a2). No neural network or
    autograd operation is involved. Boundary-event derivatives are piecewise.
    """
    initial, increments = _validate(initial, increments, context, spec, dt)
    packet = numpy_euler(initial, increments, context, spec, dt)
    states, stops = packet["states"], packet["stopped_at"]
    gradient = (workload_state_gradient(states, spec, dt, coefficients=coefficients)
                if path_gradient is None else np.asarray(path_gradient, dtype=float))
    if gradient.shape != states.shape or not np.isfinite(gradient).all():
        raise ValueError("Finite complete path-state cotangent [B,K+1,G] required")
    moments, reconstruction = _geometry(spec.n_cells, spec.length)
    scale = _scales(context, spec)
    kscale = np.array((-np.sinc(spec.h/(2*np.pi)),)*2+(0., 0.))
    ascale = np.array((0., 0.)+(2*np.sinc(spec.h/np.pi),)*2)
    cotangent = gradient[:, -1].copy()
    physics_gradient = np.zeros((len(initial), 3))
    rows = np.arange(len(initial))
    for index in reversed(range(increments.shape[1])):
        current = states[:, index]
        active = stops > index*dt
        origin = np.where(active[:, None], current, 1/spec.n_cells)
        following = np.roll(origin, -1, axis=-1)
        face = (origin+following)/(2*spec.h)
        summary = origin @ moments
        velocity = (summary*scale) @ reconstruction
        flux = face*velocity-context.diffusion/spec.h**2*(following-origin)
        amplitude = np.sqrt(context.diffusion/spec.h**2*(origin+following))
        noise = amplitude*increments[:, index]/np.sqrt(context.n_particles)
        proposal = (current+dt*(np.roll(flux, 1, axis=-1)-flux)
                    +np.roll(noise, 1, axis=-1)-noise)
        proposal = np.where(active[:, None], proposal, current)
        descending = (proposal <= spec.mass_stop) & (proposal < current)
        denominator = np.where(descending, current-proposal, 1.)
        ratios = np.where(descending, (current-spec.mass_stop)/denominator, 1.)
        cells = ratios.argmin(-1)
        fraction = ratios[rows, cells]
        crossed = active & (proposal <= spec.mass_stop).any(-1)
        # Pull back first through F(m,p)=m+alpha(m,p)(p-m).
        cotp = np.where(active[:, None], fraction[:, None]*cotangent, 0.)
        cotm = np.where(active[:, None], (1-fraction[:, None])*cotangent, cotangent)
        factor = np.where(crossed, (cotangent*(proposal-current)).sum(-1)
                          /denominator[rows, cells], 0.)
        cotp[rows, cells] += factor*fraction
        cotm[rows, cells] += factor*(1-fraction)
        # Pull back Euler: B^T q = roll(q,-1)-q on periodic faces.
        btq = np.roll(cotp, -1, axis=-1)-cotp
        z = dt*btq
        zn = btq*increments[:, index]/np.sqrt(context.n_particles)
        face_cot = z*velocity/(2*spec.h)
        mobility_cot = zn*context.diffusion/(2*spec.h**2*amplitude)
        result = (cotp+context.diffusion/spec.h**2*(z-np.roll(z, 1, axis=-1))
                  +face_cot+np.roll(face_cot, 1, axis=-1)
                  +(((z*face) @ reconstruction.T)*scale) @ moments.T
                  +mobility_cot+np.roll(mobility_cot, 1, axis=-1))
        result = np.where(active[:, None], result, 0.)
        cotangent = gradient[:, index]+cotm+result
        physics_gradient[:, 0] += (z*(-(following-origin)/spec.h**2)
                                   +zn*amplitude/(2*context.diffusion)).sum(-1)
        physics_gradient[:, 1] += (z*face*((summary*kscale) @ reconstruction)).sum(-1)
        physics_gradient[:, 2] += (z*face*((summary*ascale) @ reconstruction)).sum(-1)
    return {**packet, "initial_gradient": cotangent, "physics_gradient": physics_gradient}


def many_observables(states, spec: NativeDKSpec, *, n_times=16):
    """All resolved sub-Nyquist sine/cosine modes at evenly spaced stored times."""
    if n_times < 1:
        raise ValueError("Positive observation-time count required")
    indices = np.unique(np.linspace(0, states.shape[1]-1, n_times).round().astype(int))
    modes = np.arange(1, spec.n_cells//2)
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    phase = modes[:, None]*centers
    scale = np.sqrt(2)*np.sinc(modes*spec.h/(2*np.pi))[:, None]
    weights = np.stack((scale*np.cos(phase), scale*np.sin(phase)), axis=1).reshape(-1, spec.n_cells)
    if isinstance(states, torch.Tensor):
        return (states[:, indices] @ states.new_tensor(weights.T)).flatten(1)
    return (states[:, indices] @ weights.T).reshape(len(states), -1)


def _timing(call, repeats, warmup):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter()-start)
    return {"median_seconds": float(np.median(samples)),
            "samples_seconds": samples, "minimum_seconds": min(samples),
            "maximum_seconds": max(samples)}


def probe(*, steps=(16, 64, 128), batches=(1, 64, 256), horizon=1/16,
          repeats=7, warmup=2, seed=2026102601):
    """Independent development latency probe; no fit/checkpoint data are loaded."""
    torch.set_num_threads(1)
    context, spec = Context(kappa=.75, a2=.1, n_particles=2048), NativeDKSpec(n_cells=32)
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    probabilities = np.exp(.2*np.cos(centers))
    probabilities /= probabilities.sum()
    direction = np.cos(3*centers)/spec.n_cells
    direction -= direction.mean()
    records = []
    for k in steps:
        dt = horizon/k
        for b in batches:
            rng = np.random.default_rng(np.random.SeedSequence((seed, k, b)))
            initial = rng.multinomial(context.n_particles, probabilities, size=b)/context.n_particles
            increments = rng.normal(0, np.sqrt(dt), (b, k, spec.n_cells))
            tensor_initial, tensor_noise = torch.tensor(initial), torch.tensor(increments)
            baseline = numpy_euler(initial, increments, context, spec, dt)
            check = torch_fourier_euler(tensor_initial, tensor_noise, context, spec, dt)
            disagreement = float(np.max(np.abs(baseline["states"]-check["states"].numpy())))
            if disagreement > 2e-12:
                raise ValueError("NumPy/Torch physical trajectories disagree")
            calls = {
                "numpy_full_path": lambda: numpy_euler(initial, increments, context, spec, dt),
                "numpy_many_observables": lambda: many_observables(
                    numpy_euler(initial, increments, context, spec, dt)["states"], spec),
                "numpy_path_functionals": lambda: workload_values(
                    numpy_euler(initial, increments, context, spec, dt)["states"], spec, dt),
                "numpy_initial_jvp": lambda: numpy_euler(
                    initial, increments, context, spec, dt, initial_tangent=direction),
                "numpy_forward_plus_initial_reverse": lambda: numpy_adjoint(
                    initial, increments, context, spec, dt),
            }

            def reverse_mode():
                m = tensor_initial.detach().requires_grad_(True)
                output = torch_fourier_euler(m, tensor_noise, context, spec, dt)["states"]
                # One scalar scalarization of the three tasks, per path. Full
                # Jacobians for every output require additional reverse passes.
                value = workload_values(output, spec, dt).sum()
                return torch.autograd.grad(value, m)[0]

            calls["torch_forward_plus_initial_reverse"] = reverse_mode
            with torch.no_grad():
                calls["torch_full_path"] = lambda: torch_fourier_euler(
                    tensor_initial, tensor_noise, context, spec, dt)
                for name, call in calls.items():
                    if name == "torch_forward_plus_initial_reverse":
                        with torch.enable_grad():
                            timing = _timing(call, repeats, warmup)
                    else:
                        timing = _timing(call, repeats, warmup)
                    records.append({"method": name, "steps": k, "batch_paths": b,
                                    "dt": dt, "horizon": horizon, **timing,
                                    "paths_per_second": b/timing["median_seconds"],
                                    "conservation_error": float(np.abs(
                                        baseline["states"].sum(-1)-1).max()),
                                    "stop_count": int(np.isfinite(baseline["stopped_at"]).sum()),
                                    "numpy_torch_max_difference": disagreement})
    return {"kind": "classical-path-workload-probe-v1", "records": records,
            "seed": seed, "cpu_threads": torch.get_num_threads(),
            "numpy_version": np.__version__, "torch_version": torch.__version__,
            "platform": platform.platform(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing_scope": "Warm complete path+stopping+declared workload; supplied initial/noise; no I/O",
            "claim": "Finite-Euler development timings; no NN/statistical/continuous-DK superiority claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(16, 64, 128))
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 64, 256))
    parser.add_argument("--horizon", type=float, default=1/16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    result = probe(steps=args.steps, batches=args.batches, horizon=args.horizon,
                   repeats=args.repeats, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(result["records"])}, indent=2))


if __name__ == "__main__":
    main()
