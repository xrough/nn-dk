"""Independent warm density-path timings on validation inputs only.

This script does not modify checkpoints, input banks or scientific reports.
Run after training processes finish: CPU contention invalidates clean latency
comparisons. It times complete fixed-weight trajectories and their declared
stopping wrapper, independently of generation, fitting, bootstrap or HDF5 I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# Set before NumPy/Torch imports, rather than assuming a BLAS thread setting.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.data import source_hash, write_json  # noqa: E402
from nn_dk.native import initialize_masses, simulate_native  # noqa: E402
from nn_dk.path_config import PathConfig  # noqa: E402
from nn_dk.path_data import load_path_bank  # noqa: E402
from nn_dk.path_numerics import torch_euler, torch_stop_paths  # noqa: E402
from nn_dk.path_training import load_path_model  # noqa: E402
from nn_dk.types import seeded_rng  # noqa: E402


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy_control(initial, increments, context, spec, dt, problem):
    """Vectorized NumPy Euler stencil, with no Torch/autograd wrapper overhead."""
    current = initial.copy()
    states = np.empty((len(initial), len(increments[0]) + 1, spec.n_cells))
    states[:, 0] = current
    diffusion = dt * context.diffusion / spec.h**2
    amplitude = (0. if problem == "heat" else
                 np.sqrt(2 * context.diffusion / (spec.n_cells * spec.h**2 * context.n_particles)))
    for index in range(increments.shape[1]):
        current = current + diffusion * (
            np.roll(current, 1, axis=-1) + np.roll(current, -1, axis=-1) - 2 * current)
        if problem == "additive":
            face = amplitude * increments[:, index]
            current = current + np.roll(face, 1, axis=-1) - face
        states[:, index + 1] = current
    return states


def _latency_summary(samples):
    samples = np.asarray(samples)
    median = float(np.median(samples))
    return {"median_seconds": median, "minimum_seconds": float(samples.min()),
            "maximum_seconds": float(samples.max()),
            "p05_seconds": float(np.quantile(samples, .05)),
            "p95_seconds": float(np.quantile(samples, .95)),
            "samples_seconds": samples.tolist()}


def _summary(samples, n_paths):
    summary = _latency_summary(samples)
    median = summary["median_seconds"]
    return {**summary,
            "batch_paths_per_second": n_paths / median,
            "seconds_per_path_in_batch": median / n_paths}


def _restoration_timing(path, warmup, repeats):
    tick = time.perf_counter()
    model = load_path_model(path)
    first_call = time.perf_counter() - tick
    for _ in range(warmup):
        model = load_path_model(path)
    samples = []
    for _ in range(repeats):
        tick = time.perf_counter()
        model = load_path_model(path)
        samples.append(time.perf_counter() - tick)
    return model, {
        **_latency_summary(samples), "first_measured_restoration_seconds": first_call,
        "scope": "Checkpoint deserialization, provenance/state verification, network allocation and eval restoration",
        "cache_scope": "Same-process repeated loads; checkpoint was already hash-read. OS/file caches uncontrolled; not cold-start latency.",
    }


def _initialization_timing(config, case, count, warmup, repeats):
    # This namespace is independent of every stored physical input bank. The
    # generated development samples are discarded, not appended to an experiment.
    rng = seeded_rng(config.seed, "warm-benchmark-initialization-development", config.config_hash,
                     case, count)
    seeds = rng.integers(0, 2**63 - 1, 1 + warmup + repeats)

    def draw(seed):
        return initialize_masses(config.context(case), config.spec, count, int(seed))

    tick = time.perf_counter()
    initial = draw(seeds[0])
    first_call = time.perf_counter() - tick
    for seed in seeds[1:1 + warmup]:
        initial = draw(seed)
    samples = []
    for seed in seeds[1 + warmup:]:
        tick = time.perf_counter()
        initial = draw(seed)
        samples.append(time.perf_counter() - tick)
        if initial.shape != (count, config.n_cells) or not np.allclose(initial.sum(-1), 1):
            raise ValueError("Initialization did not return the complete IID histogram batch")
    return {"case": case, "batch_paths": count, "n_particles": config.n_particles,
            **_summary(samples, count), "first_measured_initialization_seconds": first_call,
            "scope": "IID particle-profile draws and finite-volume histogram construction; no NN inference",
            "rng_scope": "Independent discarded development draws, distinct seed for each repetition",
            "cache_scope": "First measured call follows package imports; warmed same-process repetitions. OS/allocator cache state uncontrolled; not a fresh-process cold benchmark."}


def benchmark(run, input_run=None, *, batches=(1, 64), warmup=3, repeats=21):
    run = Path(run).resolve()
    input_run = run if input_run is None else Path(input_run).resolve()
    config, input_config = (PathConfig.load(path / "config.json") for path in (run, input_run))
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("source_hash") != source_hash()
            or manifest.get("config_hash") != config.config_hash
            or manifest.get("fit_lock", {}).get("expected_models") != config.expected_models):
        raise ValueError("Benchmark a completed, locked run with unchanged source/configuration")
    if (any(type(n) is not int or n < 1 or n > input_config.validation_paths for n in batches)
            or type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 3):
        raise ValueError("Use distinct available validation inputs and positive warmups / >=3 repeats")
    if config.spec != input_config.spec or input_config.horizon < config.horizon:
        raise ValueError("Input validation bank needs the same grid and a sufficient horizon")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    locked = manifest["fit_lock"]["models"]
    if len(locked) != config.expected_models:
        raise ValueError("Every configured fitted model must be included in the benchmark")
    records, initialization = [], {}
    rng = np.random.default_rng(2026102507)  # Method-order randomization only.
    for item in locked:
        case, dt = item["case"], item["dt"]
        if (case not in input_config.cases or dt not in input_config.step_sizes
                or config.context(case) != input_config.context(case)):
            raise ValueError("Validation input physics and numerical step must match the fitted target")
        if _sha(item["checkpoint_path"]) != item["checkpoint_sha256"]:
            raise ValueError("A locked fitted checkpoint changed")
        if _sha(item["metadata_path"]) != item["metadata_sha256"]:
            raise ValueError("A locked fitted checkpoint metadata file changed")
        model, restoration = _restoration_timing(item["checkpoint_path"], warmup, repeats)
        bank = load_path_bank(input_run, input_config, "validation", case, dt)
        n_steps = round(config.horizon / dt)
        context, problem = config.context(case), config.problem(case)
        for count in batches:
            if (case, count) not in initialization:
                initialization[(case, count)] = _initialization_timing(
                    config, case, count, warmup, repeats)
            initial = np.ascontiguousarray(bank["initial"][:count])
            increments = np.ascontiguousarray(bank["brownian"][:count, :n_steps])
            families = tuple(bank["family_ids"][:count])
            times = np.arange(n_steps + 1) * dt

            def neural():
                with torch.no_grad():
                    raw = model(initial, increments, dt, context)
                    return torch_stop_paths(initial, raw[:, 1:], config.spec, dt,
                                            stopping=problem == "dk")["states"].numpy()

            def torch_reference():
                with torch.no_grad():
                    return torch_euler(initial, increments, context, config.spec, dt, problem)["states"].numpy()

            def numpy_reference():
                if problem == "dk":
                    return simulate_native(
                        context, config.spec, times, dt, 0, count,
                        initial_masses=initial, brownian_increments=increments,
                        family_ids=families).states
                return _numpy_control(initial, increments, context, config.spec, dt, problem)

            functions = {"neural_full_path": neural, "native_numpy_euler": numpy_reference,
                         "torch_euler_diagnostic": torch_reference}
            prediction, reference = neural(), numpy_reference()
            energy = context.n_particles * np.sum((prediction - reference)**2, axis=-1) / (
                1 - 1 / config.n_cells)
            strong = float(np.sqrt(energy.max(axis=1).mean()))
            oracle_difference = float(np.max(np.abs(torch_reference() - reference)))
            if oracle_difference > 1e-12:
                raise ValueError("Independent NumPy and Torch Euler implementations disagree")
            for _ in range(warmup):
                for name in rng.permutation(list(functions)):
                    functions[name]()
            samples = {name: [] for name in functions}
            for _ in range(repeats):
                for name in rng.permutation(list(functions)):
                    tick = time.perf_counter()
                    result = functions[name]()
                    samples[name].append(time.perf_counter() - tick)
                    if result.shape != prediction.shape or not np.all(np.isfinite(result)):
                        raise ValueError("A timed method did not return complete finite density paths")
            summaries = {name: _summary(values, count) for name, values in samples.items()}
            matched = strong <= config.tolerance
            row = {"case": case, "problem": problem, "seed": item["seed"], "dt": dt,
                   "horizon": config.horizon, "steps": n_steps, "batch_paths": count,
                   "n_unique_validation_inputs": count,
                   "checkpoint_path": item["checkpoint_path"],
                   "checkpoint_sha256": item["checkpoint_sha256"],
                   "input_arrays_sha256": hashlib.sha256(
                       initial.tobytes() + increments.tobytes()).hexdigest(),
                   "validation_strong_supremum_rms": strong,
                   "validation_within_same_step_research_tolerance": matched,
                   "numpy_torch_oracle_maximum_difference": oracle_difference,
                   "checkpoint_restoration": restoration,
                   "timings": summaries,
                   "neural_over_native_median_cost_ratio": (
                       summaries["neural_full_path"]["median_seconds"]
                       / summaries["native_numpy_euler"]["median_seconds"]),
                   "validation_strong_tolerance_only_native_over_neural_speedup": (
                       summaries["native_numpy_euler"]["median_seconds"]
                       / summaries["neural_full_path"]["median_seconds"] if matched else None)}
            records.append(row)
            print(f"[timing] {case}, seed={item['seed']}, P={count}: "
                  f"NN/native={row['neural_over_native_median_cost_ratio']:.4g}; "
                  f"validation strong={100 * strong:.4g}%", flush=True)
    return {"kind": "warm-validation-full-path-timing-v1", "source_hash": source_hash(),
            "benchmark_script_sha256": _sha(Path(__file__)),
            "run": str(run), "config_hash": config.config_hash,
            "input_run": str(input_run), "input_config_hash": input_config.config_hash,
            "input_role": "validation", "validation_bank_sha256": _sha(input_run / "path_validation.h5"),
            "final_inputs_read": False, "warmup_repetitions": warmup,
            "timed_repetitions": repeats, "dtype": "float64", "device": "cpu",
            "torch_threads": torch.get_num_threads(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "scope": "Complete warm density paths; same-step strong path accuracy on these validation inputs only",
            "excluded_costs": ["data generation", "model loading", "training", "HDF5 I/O", "bootstrap", "initialization sampling"],
            "included_costs": ["fixed-weight model/kernel evaluation", "physical-noise response", "full-state output", "declared stopping wrapper"],
            "limits": "Validation strong point errors are not joint weak/uncertainty acceptance gates. No matched continuous-time, marginal-law or continuum certification. Single-process one-thread settings do not prevent external CPU contention. Training cost is excluded and reported separately.",
            "initialization_timings": list(initialization.values()),
            "comparisons": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--input-run", help="Optional different completed validation bank; physical cases and step must match")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 64))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args(argv)
    before = source_hash()
    script_before = _sha(Path(__file__))
    result = benchmark(args.run, args.input_run, batches=tuple(args.batches),
                       warmup=args.warmup, repeats=args.repeats)
    if source_hash() != before or _sha(Path(__file__)) != script_before:
        raise RuntimeError("Package or benchmark source changed during the independent timing")
    write_json(Path(args.output), result)


if __name__ == "__main__":
    main()
