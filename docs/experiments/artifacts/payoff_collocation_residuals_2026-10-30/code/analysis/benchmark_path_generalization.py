"""Independent complete-path, new-observable and sensitivity runtime benchmark.

All methods receive the same fresh initial histograms and physical Brownian
increments.  Reference discrepancies below concern the same finite Euler law;
they are development checks, not held-out accuracy or continuous-DK evidence.
Run final timings only after other fitting jobs have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.data import source_hash  # noqa: E402
from nn_dk.frozen_models import current_context, load_frozen_model  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.path_model import CausalPathOperator  # noqa: E402
from nn_dk.path_numerics import torch_stop_paths  # noqa: E402
from nn_dk.path_runtime import PreparedPathRuntime  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

_probe_spec = importlib.util.spec_from_file_location(
    "generalization_classical_probe", Path(__file__).with_name("probe_path_workloads.py"))
probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(probe)
_compiled_spec = importlib.util.spec_from_file_location(
    "generalization_compiled_native", Path(__file__).with_name("compiled_native.py"))
compiled_control = importlib.util.module_from_spec(_compiled_spec)
_compiled_spec.loader.exec_module(compiled_control)

DEFAULT_SEED = 2026102612
SHORT_COMMIT = "26ac405"
LONG_COMMIT = "041eade"
SHORT_STEPS = {2 ** -8: "e30d58b3a0", 2 ** -9: "645459a4bd", 2 ** -10: "699c7c5bf9"}


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def timing(call, repeats=21, warmup=3):
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("Positive repetitions and nonnegative warmup required")
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return {"samples_seconds": samples, "median_seconds": float(np.median(samples)),
            "minimum_seconds": min(samples), "maximum_seconds": max(samples),
            "p10_seconds": float(np.percentile(samples, 10)),
            "p90_seconds": float(np.percentile(samples, 90)),
            "p95_seconds": float(np.percentile(samples, 95)),
            "warmup": warmup, "repeats": repeats}


def projected_initial_gradient(value):
    gradient = np.asarray(value, dtype=np.float64)
    if gradient.ndim != 2 or not np.isfinite(gradient).all():
        raise ValueError("Finite per-path initial gradients [B,G] required")
    return gradient - gradient.mean(-1, keepdims=True)


def errors(value, reference):
    actual, expected = np.asarray(value), np.asarray(reference)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Finite matching numerical arrays required")
    rms = float(np.sqrt(np.mean(np.square(actual - expected))))
    scale = float(np.sqrt(np.mean(np.square(expected))))
    return {"rms_error": rms, "reference_rms": scale,
            "relative_rms_error": rms / scale if scale > 1e-14 else None,
            "maximum_absolute_error": float(np.max(np.abs(actual - expected)))}


def cell_probabilities(context, spec):
    """Native-cell integrals of the initial profile, using fixed 32-point Gauss quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(32)
    x = (np.arange(spec.n_cells)[:, None] + (nodes[None] + 1) / 2) * spec.h
    modes = np.arange(1, len(context.cos_coeffs) + 1)
    logarithm = (np.cos(x[..., None] * modes) @ np.asarray(context.cos_coeffs)
                 + np.sin(x[..., None] * modes) @ np.asarray(context.sin_coeffs))
    unnormalized = np.exp(logarithm) @ weights
    return unnormalized / unnormalized.sum()


def fresh_inputs(context, spec, steps, batch, dt, *, seed, case, conditional=False):
    """Separate initial/noise RNG families; initialization is shared by every method."""
    identity = stable_hash({"seed": seed, "case": case, "steps": steps, "batch": batch,
                            "dt": dt, "role": "timing-only-independent-of-accuracy",
                            "conditioning": "fixed-histogram" if conditional else "IID-initial-law"})
    entropy = (seed, *np.frombuffer(bytes.fromhex(identity), dtype="<u4").tolist())
    initial_sequence, noise_sequence = np.random.SeedSequence(entropy).spawn(2)
    initial_rng = np.random.default_rng(initial_sequence)
    noise_rng = np.random.default_rng(noise_sequence)
    started = time.perf_counter()
    probabilities = cell_probabilities(context, spec)
    probability_seconds = time.perf_counter() - started
    started = time.perf_counter()
    initial = initial_rng.multinomial(context.n_particles, probabilities,
                                     size=1 if conditional else batch) / context.n_particles
    if conditional:
        initial = np.repeat(initial, batch, axis=0)
    initial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    noise = noise_rng.normal(0, np.sqrt(dt), (batch, steps, spec.n_cells))
    noise_seconds = time.perf_counter() - started
    return initial, noise, {"identity": identity, "initial_hash": array_hash(initial),
            "brownian_hash": array_hash(noise), "initialization": "shared-multinomial-native-cell-histogram",
            "initial_probability_rule": "fixed-32-point-per-cell-Gauss-Legendre-normalized-profile",
            "probability_seconds": probability_seconds, "initialization_seconds": initial_seconds,
            "physical_noise_seconds": noise_seconds, "seed": seed,
            "conditioning": "fixed-histogram" if conditional else "IID-initial-law",
            "rng_roles": ["initial-multinomial", "physical-face-noise"],
            "trajectory_families": [stable_hash([identity, i])[:24] for i in range(batch)],
            "conditional_inputs_supplied_to_every_method": True}


def positive_time_observables(states, spec):
    """The literal evaluation bank: 30 modes at T*j/16, j=1,...,16.

    Small unit-test grids with fewer than 16 steps use every positive node.
    All scientific short/long grids nest the declared 16 physical times.
    """
    count = min(17, states.shape[1])
    indices = np.unique(np.linspace(0, states.shape[1] - 1, count).round().astype(int))
    values = probe.many_observables(states[:, indices], spec, n_times=len(indices))
    return values[:, 2 * (spec.n_cells // 2 - 1):]


def empirical_moments(values, n_particles):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite complete-family bank outcomes [P,T,O] required")
    mean = values.mean(0)
    centered = values - mean
    return mean, n_particles * np.einsum("pti,ptj->tij", centered, centered) / (len(values) - 1)


def benchmark_moment_tasks(model, context, dt, horizon, *, case, arrays_directory, compiled=None,
                           n_paths=16384, chunk_size=64, repeats=7, warmup=1, seed=DEFAULT_SEED):
    """Measured complete Monte Carlo queries, not extrapolated path throughput."""
    if type(n_paths) is not int or n_paths < 2 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("At least two paths and a positive chunk size required")
    directory = Path(arrays_directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = round(horizon / dt)
    times = np.arange(steps + 1) * dt
    queries = np.asarray((horizon / 4, horizon / 2, horizon))
    indices = np.rint(queries / dt).astype(int)
    if not np.allclose(indices * dt, queries, atol=1e-12, rtol=0):
        raise ValueError("Literal payoff query times must lie on the integration grid")
    banks = (PayoffBank.terminal_high_modes(model.spec, horizon),
             PayoffBank.mixed_terminal_integral(model.spec, horizon))
    runtimes = {label: PreparedPathRuntime(model, context, dt, horizon, learned=learned,
        mode="dense-cached", depth=3) for label, learned in (("learned_dense_path", True),
                                                            ("exact_kernel_picard_depth3", False))}
    with torch.no_grad():
        for runtime in runtimes.values():
            runtime.prepare(steps)
    records, pools = [], []
    for role in ("conditional", "IID-ensemble"):
        initial, noise, pool = fresh_inputs(context, model.spec, steps, n_paths, dt, seed=seed,
            case=case + ":matched-moment-query:" + role, conditional=role == "conditional")
        input_path = directory / f"{case}_{role}_p{n_paths}_inputs.npz"
        np.savez_compressed(input_path, initial=initial, brownian=noise)
        pool.update(role=role, case=case, paths=n_paths, artifact_path=str(input_path.resolve()),
                    artifact_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    input_scope="fresh-independent-timing-family; shared-across-banks-and-methods")
        pools.append(pool)
        methods = {"compiled_native_euler" if compiled is not None else "numpy_native_euler":
            lambda origin, increments: (compiled(origin, increments, context, model.spec, dt)
                if compiled is not None else probe.numpy_euler(origin, increments, context, model.spec, dt))}
        methods.update({label: (lambda origin, increments, runtime=runtime: runtime(
            origin, increments, return_raw=True)[0]) for label, runtime in runtimes.items()})
        for bank in banks:
            reference = None
            for label, method in methods.items():
                components = []

                def query():
                    simulation = reduction = 0.
                    values = np.empty((n_paths, len(queries), 4), dtype=np.float64)
                    for start in range(0, n_paths, chunk_size):
                        end = min(start + chunk_size, n_paths)
                        tick = time.perf_counter()
                        with torch.no_grad():
                            packet = method(initial[start:end], noise[start:end])
                        simulated = time.perf_counter()
                        states = _tensor_array(packet["states"])
                        stops = _tensor_array(packet["stopped_at"])
                        values[start:end] = bank.evaluate_paths(states, times, stops)[:, indices]
                        reduced = time.perf_counter()
                        simulation += simulated - tick
                        reduction += reduced - simulated
                    tick = time.perf_counter()
                    mean, covariance = empirical_moments(values, context.n_particles)
                    aggregation = time.perf_counter() - tick
                    components.append((simulation, reduction, aggregation))
                    return mean, covariance

                prediction = query()
                if reference is None:
                    reference = prediction
                components.clear()
                measured = timing(query, repeats=repeats, warmup=warmup)
                measured_components = np.asarray(components[warmup:])
                output_path = directory / f"{case}_{role}_{bank.name}_{label}_moments.npz"
                np.savez_compressed(output_path, mean=prediction[0], scaled_covariance=prediction[1])
                records.append({"case": case, "role": role, "bank": bank.name, "method": label,
                    "paths": n_paths, "chunk_size": chunk_size, "dt": dt, "horizon": horizon,
                    "query_times": queries.tolist(), "optimizer_seed": 0 if label == "learned_dense_path" else None,
                    "input_identity": pool["identity"], "initial_hash": pool["initial_hash"],
                    "brownian_hash": pool["brownian_hash"], "initialization_seconds": pool["initialization_seconds"],
                    "physical_noise_seconds": pool["physical_noise_seconds"],
                    "mean_error_vs_native_development_pool": errors(prediction[0], reference[0]),
                    "covariance_error_vs_native_development_pool": errors(prediction[1], reference[1]),
                    "simulation_component_seconds": measured_components[:, 0].tolist(),
                    "new_bank_readout_component_seconds": measured_components[:, 1].tolist(),
                    "aggregation_component_seconds": measured_components[:, 2].tolist(),
                    "components_sum_to_end_to_end_up_to_clock_and_allocation_overhead": True,
                    "sample_pool_reuse_scope": "Readout component measured while each full-path chunk is available; persistent-path I/O excluded",
                    "cost_scope": "measured complete fixed-input Monte Carlo query: all paths+stopping+literal bank+mean+N-covariance; no throughput extrapolation",
                    "accuracy_scope": "timing-only development pool; independent final accuracy measured separately",
                    "output_artifact": {"path": str(output_path.resolve()), "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()},
                    **measured})
    return records, pools


def _tensor_array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _ad(call, initial, noise, context, spec, dt):
    origin = initial.detach().requires_grad_(True)
    physics = origin.new_tensor((context.diffusion, context.kappa, context.a2), requires_grad=True)
    states = call(origin, noise, physics)
    value = probe.workload_values(states, spec, dt).sum()
    initial_gradient, physics_gradient = torch.autograd.grad(value, (origin, physics))
    return states, initial_gradient, physics_gradient


def _jvp(call, initial, noise, direction):
    return torch.autograd.functional.jvp(lambda value: call(value, noise, None),
                                         initial, direction, create_graph=False, strict=False)


def _legacy_call(model, context, dt):
    def call(initial, noise):
        raw = model(initial, noise, dt, context)
        return torch_stop_paths(initial, raw[:, 1:], model.spec, dt,
                                stopping=model.problem == "dk")["states"]
    return call


def benchmark_case(model, context, dt, horizon, *, batch, seed=DEFAULT_SEED, case="development",
                   repeats=21, warmup=3, derivatives=True, finite_difference=True, compiled=None):
    """Benchmark one fixed fitted model; useful independently for numerical smoke tests.

    Returns summary records and raw numerical arrays.  No checkpoint is fitted,
    selected or modified here.  Model serialization and input generation are
    outside the warm timings; cold kernel/runtime preparation is recorded.
    """
    torch.set_num_threads(1)
    steps = round(horizon / dt)
    if steps < 1 or not math_close(steps * dt, horizon) or type(batch) is not int or batch < 1:
        raise ValueError("Positive batch and horizon on the declared step grid required")
    spec = model.spec
    initial, noise, input_record = fresh_inputs(context, spec, steps, batch, dt, seed=seed, case=case)
    conversion_started = time.perf_counter()
    tensor_initial, tensor_noise = torch.tensor(initial), torch.tensor(noise)
    input_record["torch_input_conversion_seconds"] = time.perf_counter() - conversion_started
    input_record["input_cost_measurement"] = "single-shared-pool-materialization-separate-from-repeated-warm-solver-timing"
    arrays = {"initial": initial, "brownian": noise}
    reference = probe.numpy_euler(initial, noise, context, spec, dt)
    arrays["reference_states"] = reference["states"]
    records, preparation = [], []
    common = {"case": case, "steps": steps, "batch_paths": batch, "dt": dt, "horizon": horizon,
              "input_identity": input_record["identity"], "native_state_cells": spec.n_cells,
              "accuracy_scope": "same-step-finite-Euler-development-pool",
              "training_cost_included": False}
    methods = {
        "numpy_native_euler": {"forward": lambda: probe.numpy_euler(initial, noise, context, spec, dt)["states"],
                               "family": "classical-sequential-native"},
        "torch_native_euler": {"forward": lambda: probe.torch_fourier_euler(
            tensor_initial, tensor_noise, context, spec, dt)["states"],
            "ad": lambda origin, forcing, physics: probe.torch_fourier_euler(
                origin, forcing, context, spec, dt, physics=physics)["states"],
            "family": "classical-sequential-native"},
        "legacy_learned_torch": {"forward": lambda: _legacy_call(model, context, dt)(tensor_initial, tensor_noise),
                                 "family": "learned-original-triangular"},
    }
    if compiled is not None:
        check = compiled(initial, noise, context, spec, dt)
        np.testing.assert_allclose(check["states"], reference["states"], atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(check["stopped_at"], reference["stopped_at"], atol=2e-13, rtol=2e-12)
        methods["compiled_native_euler"] = {"forward": lambda: compiled(
            initial, noise, context, spec, dt)["states"], "family": "classical-serial-compiled-native"}
    depths = tuple(sorted({1, 2, 3, model.depth}))
    for learned, mode, depth in ([(True, mode, model.depth) for mode in ("dense-cached", "causal-block-fft")]
                                + [(False, mode, depth) for mode in ("dense-cached", "causal-block-fft")
                                   for depth in depths]):
        label = ("learned" if learned else "exact_kernel") + "_" + mode.replace("-", "_") + f"_depth{depth}"
        runtime = PreparedPathRuntime(model, context, dt, horizon, mode=mode, learned=learned, depth=depth)

        def prepare(mode=mode, learned=learned, depth=depth):
            fresh = PreparedPathRuntime(model, context, dt, horizon, mode=mode,
                                         learned=learned, depth=depth)
            return fresh.prepare(steps)

        with torch.no_grad():
            cost = timing(prepare, repeats=repeats, warmup=warmup)
            runtime.prepare(steps)
        preparation.append({**common, "method": label, "scope": "runtime-geometry-plus-kernel-and-cache",
                            "repeated_per_new_physics_step_horizon": True, **cost})
        methods[label] = {"forward": lambda runtime=runtime: runtime(tensor_initial, tensor_noise),
                          "ad": lambda origin, forcing, physics, runtime=runtime: runtime(
                              origin, forcing, physics=physics),
                          "family": "learned-causal-picard" if learned else "classical-exact-kernel-picard",
                          "depth": depth, "backend": mode,
                          "uses_fitted_parameters": learned}
    reference_observables = positive_time_observables(reference["states"], spec)
    for label, method in methods.items():
        with torch.no_grad():
            states = method["forward"]()
            array = _tensor_array(states)
            arrays[label + "_states"] = array
            diagnostic = {"states": errors(array, reference["states"]),
                          "many_observables": errors(_tensor_array(positive_time_observables(states, spec)),
                                                       reference_observables),
                          "mass_error": float(np.abs(array.sum(-1) - 1).max()),
                          "minimum_mass": float(array.min())}
            for task, call in (("full_density_path", method["forward"]),
                               ("many_new_observables", lambda method=method: positive_time_observables(
                                   method["forward"](), spec))):
                cost = timing(call, repeats=repeats, warmup=warmup)
                records.append({**common, "method": label, "family": method["family"], "task": task,
                                "output_dimension": (int(np.prod(array.shape[1:])) if task == "full_density_path"
                                                     else reference_observables.shape[1]),
                                "timing_scope": "complete-forward-plus-stopping-plus-declared-readout",
                                "diagnostics": diagnostic, "paths_per_second": batch / cost["median_seconds"],
                                **cost})
    derivative_records = []
    if derivatives:
        baseline = probe.numpy_adjoint(initial, noise, context, spec, dt)
        reference_initial = projected_initial_gradient(baseline["initial_gradient"])
        reference_physics = baseline["physics_gradient"].sum(0)
        arrays["native_projected_initial_gradient"] = reference_initial
        arrays["native_shared_physics_gradient"] = reference_physics
        native_cost = timing(lambda: probe.numpy_adjoint(initial, noise, context, spec, dt),
                             repeats=repeats, warmup=warmup)
        derivative_records.append({**common, "method": "numpy_native_analytic_adjoint",
            "task": "scalar_initial_and_physics_reverse", "physics_scope": "shared-physics-gradient-summed-over-paths",
            "initial_projection": "subtract-per-path-cell-mean", "scalarization": [1., 1., 1.],
            "initial_error": errors(reference_initial, reference_initial),
            "physics_error": errors(reference_physics, reference_physics), **native_cost})
        if compiled is not None:
            compiled_reverse = compiled.adjoint(initial, noise, context, spec, dt)
            compiled_initial = projected_initial_gradient(compiled_reverse["initial_gradient"])
            compiled_physics = compiled_reverse["physics_gradient"].sum(0)
            np.testing.assert_allclose(compiled_initial, reference_initial, atol=5e-12, rtol=5e-11)
            np.testing.assert_allclose(compiled_physics, reference_physics, atol=5e-12, rtol=5e-11)
            arrays["compiled_native_projected_initial_gradient"] = compiled_initial
            arrays["compiled_native_shared_physics_gradient"] = compiled_physics
            compiled_reverse_cost = timing(lambda: compiled.adjoint(initial, noise, context, spec, dt),
                                           repeats=repeats, warmup=warmup)
            derivative_records.append({**common, "method": "compiled_native_analytic_adjoint",
                "family": "classical-serial-compiled-native", "task": "scalar_initial_and_physics_reverse",
                "initial_projection": "subtract-per-path-cell-mean", "scalarization": [1., 1., 1.],
                "physics_scope": "shared-physics-gradient-summed-over-paths",
                "physics_interpretation": "Derivative of the known physical coefficients",
                "initial_error": errors(compiled_initial, reference_initial),
                "physics_error": errors(compiled_physics, reference_physics),
                "physics_gradient": compiled_physics.tolist(), "reference_physics_gradient": reference_physics.tolist(),
                "timing_scope": "complete-compiled-forward-stopping-plus-identical-NumPy-workload-cotangent-plus-compiled-adjoint",
                **compiled_reverse_cost})
        centers = (np.arange(spec.n_cells) + .5) * spec.h
        direction = np.cos(3 * centers) / spec.n_cells
        direction -= direction.mean()
        tensor_direction = tensor_initial.new_tensor(direction).expand_as(tensor_initial)
        tangent_reference = probe.numpy_euler(initial, noise, context, spec, dt, initial_tangent=direction)
        arrays["native_initial_jvp"] = tangent_reference["tangent"]
        arrays["initial_jvp_direction"] = direction
        tangent_cost = timing(lambda: probe.numpy_euler(initial, noise, context, spec, dt,
                                                       initial_tangent=direction),
                               repeats=repeats, warmup=warmup)
        derivative_records.append({**common, "method": "numpy_native_analytic_jvp", "task": "initial_path_jvp",
            "tangent_error": errors(tangent_reference["tangent"], tangent_reference["tangent"]), **tangent_cost})
        if compiled is not None:
            compiled_tangent = compiled(initial, noise, context, spec, dt, initial_tangent=direction)
            np.testing.assert_allclose(compiled_tangent["tangent"], tangent_reference["tangent"],
                                       atol=2e-13, rtol=2e-12)
            arrays["compiled_native_initial_jvp"] = compiled_tangent["tangent"]
            compiled_cost = timing(lambda: compiled(initial, noise, context, spec, dt,
                initial_tangent=direction), repeats=repeats, warmup=warmup)
            derivative_records.append({**common, "method": "compiled_native_analytic_jvp",
                "task": "initial_path_jvp", "family": "classical-serial-compiled-native",
                "tangent_error": errors(compiled_tangent["tangent"], tangent_reference["tangent"]),
                "timing_scope": "serial-compiled-complete-forward-stopping-plus-conservative-initial-JVP",
                **compiled_cost})
        for label, method in methods.items():
            if "ad" not in method:
                continue
            with torch.enable_grad():
                value = _ad(method["ad"], tensor_initial, tensor_noise, context, spec, dt)
                origin_gradient = projected_initial_gradient(_tensor_array(value[1]))
                physics_gradient = _tensor_array(value[2])
                arrays[label + "_projected_initial_gradient"] = origin_gradient
                arrays[label + "_shared_physics_gradient"] = physics_gradient
                cost = timing(lambda method=method: _ad(method["ad"], tensor_initial, tensor_noise,
                                                        context, spec, dt), repeats=repeats, warmup=warmup)
                derivative_records.append({**common, "method": label, "family": method["family"],
                    "task": "scalar_initial_and_physics_reverse", "initial_projection": "subtract-per-path-cell-mean",
                    "physics_scope": "shared-physics-gradient-summed-over-paths", "scalarization": [1., 1., 1.],
                    "physics_interpretation": ("Fixed-physics trained NN parametrization derivative only; not a validated parameter-conditioned solver"
                        if method["family"] == "learned-causal-picard" else "Derivative of the known physical coefficients"),
                    "initial_error": errors(origin_gradient, reference_initial),
                    "physics_error": errors(physics_gradient, reference_physics),
                    "physics_gradient": physics_gradient.tolist(), "reference_physics_gradient": reference_physics.tolist(),
                    "timing_scope": "complete-forward-stopping-workloads-plus-both-reverse-gradients-new-physics-plan",
                    **cost})
                jvp = _jvp(method["ad"], tensor_initial, tensor_noise, tensor_direction)
                arrays[label + "_initial_jvp"] = _tensor_array(jvp[1])
                jvp_cost = timing(lambda method=method: _jvp(method["ad"], tensor_initial,
                                                            tensor_noise, tensor_direction),
                                  repeats=repeats, warmup=warmup)
                derivative_records.append({**common, "method": label, "family": method["family"],
                    "task": "initial_path_jvp", "tangent_error": errors(_tensor_array(jvp[1]), tangent_reference["tangent"]),
                    "timing_scope": "complete-forward-plus-full-path-conservative-initial-JVP", **jvp_cost})
    finite = []
    if derivatives and finite_difference and batch == 1:
        direction = arrays["initial_jvp_direction"][None]
        tensor_direction = tensor_initial.new_tensor(direction)
        epsilon = 1e-5
        for label, method in methods.items():
            if "ad" not in method or ("exact_kernel" in label and not label.endswith("dense_cached_depth3")):
                continue
            physics = tensor_initial.new_tensor((context.diffusion, context.kappa, context.a2))

            def objective(origin, parameters):
                return float(probe.workload_values(method["ad"](origin, tensor_noise, parameters), spec, dt).sum())

            with torch.no_grad():
                initial_fd = (objective(tensor_initial + epsilon * tensor_direction, physics)
                              - objective(tensor_initial - epsilon * tensor_direction, physics)) / (2 * epsilon)
                physics_fd = []
                for coordinate in range(3):
                    shift = torch.zeros_like(physics)
                    shift[coordinate] = epsilon
                    physics_fd.append((objective(tensor_initial, physics + shift)
                                       - objective(tensor_initial, physics - shift)) / (2 * epsilon))
            initial_ad = float((arrays[label + "_projected_initial_gradient"] * direction).sum())
            finite.append({"method": label, "epsilon": epsilon, "initial_direction_ad": initial_ad,
                           "initial_direction_fd": initial_fd, "initial_absolute_error": abs(initial_fd - initial_ad),
                           "physics_fd": physics_fd, "physics_ad": arrays[label + "_shared_physics_gradient"].tolist(),
                           "physics_error": errors(physics_fd, arrays[label + "_shared_physics_gradient"])})
    return {"input": input_record, "records": records, "preparation": preparation,
            "derivative_records": derivative_records, "finite_difference": finite,
            "reference_stop_count": int(np.isfinite(reference["stopped_at"]).sum()),
            "context": asdict(context), "spec": asdict(spec)}, arrays


def math_close(left, right):
    return abs(left - right) <= 1e-12


def _load_current_model(path, commit):
    started = time.perf_counter()
    frozen, provenance = load_frozen_model(path, commit, kind="path")
    model = CausalPathOperator(NativeDKSpec(**asdict(frozen.spec)), frozen.problem,
                               frozen.width, frozen.depth)
    model.load_state_dict(frozen.state_dict())
    context = current_context(frozen.path_config.context(frozen.case))
    return model, context, frozen.dt, frozen.path_config.horizon, provenance, time.perf_counter() - started


def run_benchmark(output, *, checkpoint_root=None, include_long=False, long_checkpoint=None,
                   smoke=False, repeats=21, warmup=3, seed=DEFAULT_SEED, final_timing=False,
                   use_compiled=True, moment_paths=16384, moment_repeats=7, moment_warmup=1):
    """Write incremental JSON and raw arrays, rejecting any source change during timing."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    array_directory = output.parent / (output.stem + "_arrays")
    array_directory.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(checkpoint_root or root / "runs/interacting-followup-2026-10-25/paths/checkpoints")
    before_source = source_hash()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    probe_hash = hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    compiled = None
    compiled_metadata = {"available": False, "reason": "Explicitly disabled"}
    if use_compiled:
        if compiled_control.compiler_available():
            compile_started = time.perf_counter()
            compiled = compiled_control.compile_native(output.parent / "compiled_native_cache")
            compiled_metadata = {"available": True, **compiled.metadata,
                                 "complete_setup_seconds": time.perf_counter() - compile_started}
        else:
            compiled_metadata = {"available": False, "reason": "Optional C++ compiler unavailable"}
    jobs = [(case, dt, checkpoint_root / f"{case}_{suffix}_seed0.pt", SHORT_COMMIT)
            for case in ("dk_interacting_uniform", "dk_interacting") for dt, suffix in SHORT_STEPS.items()]
    if smoke:
        jobs = jobs[:1]
    if include_long:
        path = Path(long_checkpoint or root / "runs/path-first-batch-2026-10-25-v2/checkpoints/dk_interacting_f24ccb4888_seed0.pt")
        jobs.append(("dk_interacting_long", 2 ** -7, path, LONG_COMMIT))
    result = {"kind": "independent-general-path-workload-benchmark-v1", "status": "running",
              "seed": seed, "timing_pool_independent_of_accuracy": True,
              "final_timing_after_external_fit_completion": bool(final_timing),
              "claim": "Finite-Euler independent runtime/task probe; no continuous-DK or universal superiority claim",
              "source_hash": before_source, "script_sha256": script_hash, "classical_probe_sha256": probe_hash,
              "compiled_classical_control": compiled_metadata,
              "environment": {"platform": platform.platform(), "processor": platform.processor(),
                  "torch": torch.__version__, "numpy": np.__version__, "cpu_threads": 1,
                  "thread_environment": {name: os.environ.get(name) for name in
                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                  "load_average_start": os.getloadavg()},
              "timing_scope": "Warm complete paths/readouts/derivatives with fixed supplied inputs; preparation and initialization separate",
              "derivative_scope": "Piecewise pathwise derivatives of fixed sum of three path tasks; shared physics aggregated over batch",
              "many_observable_bank": {"times": "T*j/16,j=1,...,16", "zero_time_included": False,
                  "modes": "normalized exact-cell-integrated sine/cosine modes 1,...,15", "dimension": 480},
              "moment_task_design": {"paths": 8 if smoke else moment_paths, "chunk_size": 64,
                  "repeats": 1 if smoke else moment_repeats, "warmup": 0 if smoke else moment_warmup,
                  "roles": ["conditional", "IID-ensemble"], "banks": ["terminal-modes34", "terminal3-integral1"],
                  "methods": ["learned_dense_path", "exact_kernel_picard_depth3", "compiled_native_euler-or-NumPy-fallback"],
                  "short_step": 2 ** -10, "optimizer_seed": 0, "input_seed": seed,
                  "sampling_scope": "all complete families; uncensored simulation plus declared stopping; no survivor averages"},
              "moment_tasks": [], "moment_input_pools": [],
              "learned_physics_derivative_limitation": "Checkpoints were trained at fixed D,kappa,a2. Their physics AD is a diagnostic derivative of the parametrization, not validated inverse-problem or parameter-generalization capability. Finite differences check implementation only.",
              "unsupported": {"legacy_learned_torch": "No tensor-physics argument; only forward comparison",
                              "old_fixed_four_moment_statistics_network": "Does not supply density paths, arbitrary observables or pathwise gradients"},
              "cases": []}
    started = time.perf_counter()

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    for case, expected_step, path, commit in jobs:
        model, context, dt, horizon, provenance, loading_seconds = _load_current_model(path, commit)
        if dt != expected_step:
            raise ValueError("Checkpoint suffix/timestep mapping differs from the declared benchmark")
        for batch in ((1,) if smoke else (1, 64, 256)):
            diagnostics, arrays = benchmark_case(model, context, dt, horizon, batch=batch, seed=seed,
                case=case, repeats=repeats, warmup=warmup,
                derivatives=smoke or round(horizon / dt) >= 64, compiled=compiled)
            name = f"{case}_k{round(horizon / dt)}_b{batch}.npz"
            artifact = array_directory / name
            np.savez_compressed(artifact, **arrays)
            diagnostics.update(checkpoint=provenance, checkpoint_loading_seconds=loading_seconds,
                               raw_arrays=str(artifact.resolve()), raw_arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            result["cases"].append(diagnostics)
            result["elapsed_seconds"] = time.perf_counter() - started
            save()
            print(json.dumps({"case": case, "steps": round(horizon / dt), "batch": batch,
                              "completed_cases": len(result["cases"])}), flush=True)
        if smoke or (horizon == 1 / 16 and dt == 2 ** -10):
            moment_rows, input_pools = benchmark_moment_tasks(model, context, dt, horizon, case=case,
                arrays_directory=array_directory / "moment_queries", compiled=compiled,
                n_paths=8 if smoke else moment_paths, repeats=1 if smoke else moment_repeats,
                warmup=0 if smoke else moment_warmup, seed=seed)
            for row in moment_rows:
                row["checkpoint"] = provenance if row["method"] == "learned_dense_path" else {
                    "training_required": False, "physical_context": asdict(context)}
            result["moment_tasks"].extend(moment_rows)
            result["moment_input_pools"].extend(input_pools)
            result["elapsed_seconds"] = time.perf_counter() - started
            save()
            print(json.dumps({"case": case, "complete_moment_query_rows": len(result["moment_tasks"])}), flush=True)
    if (source_hash() != before_source or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
            or hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest() != probe_hash
            or (compiled is not None and (
                hashlib.sha256(Path(compiled_control.__file__).read_bytes()).hexdigest() != compiled.metadata["python_wrapper_sha256"]
                or hashlib.sha256(Path(compiled_control.__file__).with_name("native_euler_control.cpp").read_bytes()).hexdigest() != compiled.metadata["cpp_sha256"]))):
        result["status"] = "invalid-source-changed"
        save()
        raise RuntimeError("Package or benchmark source changed during the independent timing")
    result["status"] = "complete"
    result["environment"]["load_average_end"] = os.getloadavg()
    result["elapsed_seconds"] = time.perf_counter() - started
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--include-long", action="store_true")
    parser.add_argument("--long-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-compiled", action="store_true",
                        help="Explicitly omit the optional serial C++ Euler/JVP control")
    parser.add_argument("--moment-paths", type=int, default=16384)
    parser.add_argument("--moment-repeats", type=int, default=7)
    parser.add_argument("--moment-warmup", type=int, default=1)
    parser.add_argument("--final-timing", action="store_true",
                        help="Record that the caller confirmed other fitting jobs have finished")
    args = parser.parse_args()
    run_benchmark(args.output, checkpoint_root=args.checkpoint_root, include_long=args.include_long,
                  long_checkpoint=args.long_checkpoint, smoke=args.smoke, repeats=args.repeats,
                  warmup=args.warmup, seed=args.seed, final_timing=args.final_timing,
                  use_compiled=not args.no_compiled, moment_paths=args.moment_paths,
                  moment_repeats=args.moment_repeats, moment_warmup=args.moment_warmup)


if __name__ == "__main__":
    main()
