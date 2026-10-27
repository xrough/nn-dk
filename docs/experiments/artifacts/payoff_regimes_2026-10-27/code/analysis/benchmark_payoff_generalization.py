"""Same-output costs for new payoff heads and a native Gaussian baseline.

Final timings require the twelve-fit lock and an explicit --final-timing flag
after other workers finish. Both methods return four physical means and a
4x4 N-scaled covariance at T/4,T/2,T. The classical Gaussian law is approximate
and unstopped; cost measurements alone make no accuracy or superiority claim.
Histogram-only initialization uses its direct multinomial IID law, without
constructing microscopic particle positions. Its integration cost is included
for ensemble NN queries; the Gaussian baseline integrates that initial law
analytically through diag(p)-p p^T and needs no initial Monte Carlo pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.evaluate_path_generalization import mixture_moments  # noqa: E402
from analysis.native_gaussian_baseline import (  # noqa: E402
    LAW,
    drift_jacobian_covariance,
    iid_scaled_covariance,
)
from analysis.path_generalization_reference import (  # noqa: E402
    ReferenceConfig,
    profile_probabilities,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_learning import PayoffBank, load_payoff_model  # noqa: E402
from nn_dk.types import seeded_rng  # noqa: E402

SEED = 2026102615
SOURCE_FREEZE = "1a274fcf4d2b9c9b19f4605f514f38d8465807216a99419504f77d7796442df1"
KIND = "same-output-new-payoff-head-and-native-Gaussian-cost-v1"
SCRIPT = Path(__file__).resolve()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _npz(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def fresh_histograms(cfg, case, count, *, role="fixed-ensemble", repetition=0):
    """Independent timing-only native IID law; no physical path outcomes."""
    probabilities, _ = profile_probabilities(cfg, case)
    rng = seeded_rng(SEED, KIND, cfg.config_hash, case, role, repetition)
    return rng.multinomial(cfg.n_particles, probabilities, size=count)/cfg.n_particles


def _check_output(mean, covariance):
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    if (mean.shape[-2:] != (3, 4) or covariance.shape != (*mean.shape, 4)
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-11, rtol=0)):
        raise ValueError("Same-output benchmark requires finite three-time four-payoff means/covariances")
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue < -1e-10:
        raise ValueError("A covariance output is not positive semidefinite")
    return {"mean_sha256": array_hash(mean), "scaled_covariance_sha256": array_hash(covariance),
            "mean_shape": list(mean.shape), "scaled_covariance_shape": list(covariance.shape),
            "minimum_covariance_eigenvalue": minimum_eigenvalue}


def lga_marginals(initial, context, spec, times, step, bank, initial_scaled_covariance=None):
    """RK4 native linear-noise marginals, omitting unused transition matrices.

    This is the same mean/covariance ODE as native_gaussian_baseline, projected
    to the declared bank. Only requested marginal outputs are computed: no
    joint-time propagators, Gaussian path samples, PSD repair or stopping.
    """
    tick = time.perf_counter()
    initial, times = np.asarray(initial, dtype=float), np.asarray(times, dtype=float)
    g = spec.n_cells
    if (initial.shape != (g,) or not np.isfinite(initial).all() or (initial < 0).any()
            or not np.isclose(initial.sum(), 1, atol=1e-10, rtol=0)
            or times.shape != (3,) or not np.isfinite(times).all() or min(times) <= 0
            or not (np.diff(times) > 0).all() or times[-1] > bank.horizon
            or not np.isfinite(step) or step <= 0 or bank.n_cells != g):
        raise ValueError("Valid native initial state, declared three-time bank and positive RK4 step required")
    rows = np.flatnonzero(bank.integral_rows)
    r, d = len(rows), g+len(rows)
    mean = np.concatenate((initial, np.zeros(r)))
    covariance = np.zeros((d, d))
    if initial_scaled_covariance is not None:
        initial_covariance = np.asarray(initial_scaled_covariance, dtype=float)
        if (initial_covariance.shape != (g, g) or not np.isfinite(initial_covariance).all()
                or not np.allclose(initial_covariance, initial_covariance.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(initial_covariance).min() < -1e-12):
            raise ValueError("Finite PSD N-scaled native initial covariance required")
        covariance[:g, :g] = initial_covariance
    projection = np.zeros((4, d))
    projection[:, :g] = bank.terminal_weights
    for column, row in enumerate(rows):
        projection[row, g+column] = 1.
    running = bank.running_weights[rows]
    prep_seconds = time.perf_counter()-tick
    minimum_mass = float(initial.min())

    def rhs(mu, cov):
        nonlocal minimum_mass
        minimum_mass = min(minimum_mass, float(mu[:g].min()))
        drift, jacobian, forcing = drift_jacobian_covariance(mu[:g], context, spec)
        if r:
            a, q = np.zeros((d, d)), np.zeros((d, d))
            a[:g, :g], a[g:, :g], q[:g, :g] = jacobian, running, forcing
            drift = np.concatenate((drift, running @ mu[:g]))
        else:
            a, q = jacobian, forcing
        return drift, a @ cov+cov @ a.T+q

    means, covariances = [], []
    previous, count_total = 0., 0
    tick = time.perf_counter()
    for target in times:
        count = int(np.ceil((target-previous)/step))
        dt = (target-previous)/count
        velocity = abs(context.kappa)+2*abs(context.a2)
        if dt*(4*context.diffusion/spec.h**2+2*velocity/spec.h) > 2.5:
            raise ValueError("RK4 covariance stability condition violated")
        for _ in range(count):
            k1 = rhs(mean, covariance)
            k2 = rhs(mean+dt/2*k1[0], covariance+dt/2*k1[1])
            k3 = rhs(mean+dt/2*k2[0], covariance+dt/2*k2[1])
            k4 = rhs(mean+dt*k3[0], covariance+dt*k3[1])
            mean = mean+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            covariance = covariance+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        minimum_mass = min(minimum_mass, float(mean[:g].min()))
        means.append(projection @ mean)
        covariances.append(projection @ covariance @ projection.T)
        previous = target
        count_total += count
    integration_seconds = time.perf_counter()-tick
    return np.asarray(means), np.asarray(covariances), {
        "law": LAW, "return_joint": False, "unused_transition_matrices_computed": False,
        "preparation_seconds": prep_seconds, "integration_seconds": integration_seconds,
        "integration_steps": count_total, "minimum_deterministic_mass": minimum_mass,
        "deterministic_boundary_encountered": minimum_mass <= spec.mass_stop,
        "stopped_or_killed_law_equivalence": False,
        "linearization_and_stopping_bias": "unresolved", "no_density_or_PSD_repair": True}


def lga_batch(initial, context, spec, times, step, bank):
    predictions = [lga_marginals(state, context, spec, times, step, bank) for state in initial]
    return np.stack([item[0] for item in predictions]), np.stack([item[1] for item in predictions])


def predict_chunks(model, initial, times, chunk_size=1024):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Positive prediction chunk size required")
    mean, covariance = np.empty((len(initial), len(times), 4)), np.empty((len(initial), len(times), 4, 4))
    for start in range(0, len(initial), chunk_size):
        prediction = model.predict_conditional(initial[start:start+chunk_size], times)
        mean[start:start+len(prediction.mean)] = prediction.mean
        covariance[start:start+len(prediction.mean)] = prediction.scaled_covariance
    return mean, covariance


def _time(call, repeats, warmup):
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("Positive repeats and nonnegative warmup required")
    for _ in range(warmup):
        call()
    durations, value = [], None
    for _ in range(repeats):
        tick = time.perf_counter()
        value = call()
        durations.append(time.perf_counter()-tick)
    return {"samples_seconds": durations, "median_seconds": float(np.median(durations)),
            "minimum_seconds": min(durations), "maximum_seconds": max(durations),
            "p05_seconds": float(np.quantile(durations, .05)),
            "p95_seconds": float(np.quantile(durations, .95)),
            "repeats": repeats, "warmup": warmup}, value


def _lock(root, cfg=ReferenceConfig()):
    path = Path(root)/"payoffs"/"fit_lock.json"
    lock = _read(path)
    if (lock.get("expected_models") != 12 or len(lock.get("models", [])) != 12
            or lock.get("complete") is not True or lock.get("source_hash") != source_hash()):
        raise ValueError("Final fixed-bank costs require the complete twelve-head source-strict fit lock")
    slots = set()
    for item in lock["models"]:
        slot = item["bank"], item["seed"], item["init"]
        if (slot in slots or item.get("complete") is not True
                or item.get("source_hash") != source_hash()
                or _sha(item["checkpoint_path"]) != item["checkpoint_sha256"]
                or _sha(item["metadata_path"]) != item["metadata_sha256"]):
            raise ValueError("Duplicate, incomplete or mutated fixed-bank fit lock")
        model = load_payoff_model(item["checkpoint_path"])
        if (not model.complete or model.provenance.get("source_hash") != source_hash()
                or model.bank.bank_hash != item["bank_hash"] or model.target_id != item["target_id"]):
            raise ValueError("A locked new-payoff model did not restore strictly")
        if (model.spec != cfg.spec or model.horizon != cfg.horizon
                or (model.context.n_particles, model.context.diffusion, model.context.kappa, model.context.a2)
                != (cfg.n_particles, cfg.diffusion, cfg.kappa, cfg.a2)
                or model.bank.bank_hash not in {bank.bank_hash for bank in _banks(cfg)}):
            raise ValueError("Locked model target/physics/payoff bank differs from the benchmark")
        slots.add(slot)
    expected = {(bank, seed, init) for bank in ("terminal-modes34", "terminal3-integral1")
                for seed in (0, 1, 2) for init in ("cold", "trunk-transfer")}
    if slots != expected:
        raise ValueError("Locked new-payoff allocations differ from the twelve predeclared slots")
    return lock, _sha(path)


def prepare(root, cfg=ReferenceConfig(), *, repeats=21, warmup=3, batches=(1, 64, 256),
            ensemble_paths=16384, inference_chunk=1024):
    if source_hash() != SOURCE_FREEZE:
        raise ValueError("Benchmark requires the frozen package source")
    if (not batches or any(type(b) is not int or b < 1 for b in batches)
            or type(ensemble_paths) is not int or ensemble_paths < 2
            or type(inference_chunk) is not int or inference_chunk < 1
            or type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0):
        raise ValueError("Invalid predeclared benchmark allocation")
    directory = Path(root)/"timing"
    directory.mkdir(parents=True, exist_ok=True)
    plan_path, inputs_path = directory/"payoff_benchmark_plan.json", directory/"payoff_benchmark_inputs.npz"
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": _sha(SCRIPT),
                "helper_sha256": {name: _sha(SCRIPT.with_name(name)) for name in
                    ("native_gaussian_baseline.py", "path_generalization_reference.py",
                     "evaluate_path_generalization.py")},
                "seed": SEED, "config": json.loads(json.dumps(asdict(cfg))), "config_hash": cfg.config_hash,
                "banks": {bank.name: bank.to_dict() for bank in _banks(cfg)},
                "repeats": repeats, "warmup": warmup, "conditional_batches": list(batches),
                "ensemble_initial_histograms": ensemble_paths, "inference_chunk": inference_chunk,
                "lga_coarse_step": 2**-7, "lga_fine_step": 2**-10,
                "lga_fine_cost_conditional_batches": [1],
                "outputs": "4means+4x4N-scaled-covariance-at-three-times; no joint-time covariance",
                "initialization": "direct-IID-multinomial-native-histogram",
                "ensemble_lga_initialization": "analytic-mu=p,C_N=diag(p)-ppT; no initial MC",
                "timing_pool_independent_of_scientific_references": True,
                "accuracy_claim": False}
    if plan_path.exists():
        saved = _read(plan_path)
        if saved["identity"] != identity or saved.get("inputs_sha256") != _sha(inputs_path):
            raise ValueError("Frozen payoff benchmark plan/input artifact changed")
        return directory, saved
    arrays = {}
    for case in cfg.cases:
        probabilities, _ = profile_probabilities(cfg, case)
        arrays[case+"_probabilities"] = probabilities
        arrays[case+"_conditional"] = fresh_histograms(cfg, case, max(batches), role="fixed-conditional")
        arrays[case+"_ensemble"] = fresh_histograms(cfg, case, ensemble_paths)
    _npz(inputs_path, arrays)
    saved = {"identity": identity, "inputs_path": str(inputs_path.resolve()),
             "inputs_sha256": _sha(inputs_path),
             "input_arrays": {name: {"sha256": array_hash(array), "shape": list(array.shape)}
                              for name, array in arrays.items()}}
    _json(plan_path, saved)
    return directory, saved


def run_benchmark(root, cfg=ReferenceConfig(), *, final_timing=False, smoke=False,
                  repeats=21, warmup=3, batches=(1, 64, 256), ensemble_paths=16384, inference_chunk=1024):
    """Run only after caller confirms workers are quiet, or as an explicit smoke."""
    if not final_timing and not smoke:
        raise ValueError("Use --final-timing after workers finish, or an explicit development smoke")
    torch.set_num_threads(1)
    directory, plan = prepare(root, cfg, repeats=repeats, warmup=warmup, batches=batches,
                              ensemble_paths=ensemble_paths, inference_chunk=inference_chunk)
    lock, lock_sha = _lock(root, cfg)
    output = directory/"payoff_benchmark.json"
    run_identity = {"plan_sha256": _sha(directory/"payoff_benchmark_plan.json"),
                    "fit_lock_sha256": lock_sha, "final_timing": final_timing,
                    "smoke": smoke, "source_hash": source_hash()}
    if output.exists():
        result = _read(output)
        if result["identity"] != run_identity:
            raise ValueError("Existing payoff timing run has a different immutable identity")
        for row in result["records"]:
            artifact = row["output_artifact"]
            if _sha(artifact["path"]) != artifact["sha256"]:
                raise ValueError("A committed payoff timing output artifact changed")
    else:
        result = {"identity": run_identity, "status": "running", "records": [],
                  "cold_restoration": [], "initialization": [], "lga_refinement": [],
                  "platform": platform.platform(), "numpy_version": np.__version__,
                  "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
                  "thread_environment": {name: os.environ.get(name) for name in (
                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                  "scope": "same-bank marginal moments; LGA approximate unstopped law; no accuracy or superiority assertion",
                  "timing_excludes": ["imports", "HDF5/file-I/O", "future reference generation", "bootstrap"],
                  "final_timing_after_external_worker_completion": final_timing,
                  "cold_cache_scope": "first measured restore follows integrity reads; OS caches uncontrolled"}
    completed = {row["key"] for row in result["records"]}
    inputs = np.load(plan["inputs_path"])
    artifacts = directory/"payoff_arrays"
    artifacts.mkdir(exist_ok=True)

    def publish(row, arrays):
        array_path = artifacts/(row["key"]+".npz")
        _npz(array_path, arrays)
        row["output_artifact"] = {"path": str(array_path.resolve()), "sha256": _sha(array_path)}
        row["output_hashes"] = _check_output(arrays["mean"], arrays["scaled_covariance"])
        result["records"].append(row)
        completed.add(row["key"])
        _json(output, result)

    for case in cfg.cases:
        probabilities = inputs[case+"_probabilities"]
        if case not in {item["case"] for item in result["initialization"]}:
            counter = iter(range(repeats+warmup))

            def draw_only():
                return fresh_histograms(cfg, case, ensemble_paths, role="fresh-ensemble-repetition",
                                        repetition=next(counter))

            timing, initial = _time(draw_only, repeats, warmup)
            result["initialization"].append({"case": case, "paths": ensemble_paths,
                "method": "direct-native-IID-multinomial-histogram", **timing,
                "last_initial_sha256": array_hash(initial), "includes_probability_setup": False})
            _json(output, result)
        for bank in _banks(cfg):
            context = cfg.context(case)
            coarse_first = fine_first = None
            for step in (2**-7, 2**-10):
                allowed_batches = batches if step == 2**-7 else (1,)
                for batch in allowed_batches:
                    key = f"lga_{case}_{bank.name}_step{round(-np.log2(step))}_conditional_B{batch}"
                    initial = inputs[case+"_conditional"][:batch]
                    if key in completed:
                        continue
                    timing, prediction = _time(lambda: lga_batch(initial, context, cfg.spec,
                        cfg.query_times, step, bank), repeats, warmup)
                    mean, covariance = prediction
                    if batch == 1:
                        if step == 2**-7:
                            coarse_first = prediction
                        else:
                            fine_first = prediction
                    publish({"key": key, "method": "native-linear-Gaussian-RK4", "case": case,
                             "bank": bank.name, "role": "conditional", "batch_states": batch,
                             "step": step, "initial_pool_sha256": array_hash(initial),
                             "law": LAW, "stopping_and_linearization_bias": "unresolved",
                             "initial_law_integration": False, **timing},
                            {"mean": mean, "scaled_covariance": covariance})
                key = f"lga_{case}_{bank.name}_step{round(-np.log2(step))}_ensemble"
                if key not in completed:
                    tick = time.perf_counter()
                    initial_covariance = iid_scaled_covariance(probabilities)
                    initialization_seconds = time.perf_counter()-tick
                    def call():
                        return lga_marginals(probabilities, context, cfg.spec, cfg.query_times,
                                             step, bank, iid_scaled_covariance(probabilities))

                    timing, prediction = _time(call, repeats, warmup)
                    mean, covariance, diagnostic = prediction
                    publish({"key": key, "method": "native-linear-Gaussian-RK4", "case": case,
                             "bank": bank.name, "role": "IID-ensemble", "step": step,
                             "initial_histograms": 0, "initial_law_integration": "analytic-first-two-moments",
                             "analytic_initialization_first_call_seconds": initialization_seconds,
                             "initial_scaled_covariance_sha256": array_hash(initial_covariance),
                             "diagnostics": diagnostic, **timing},
                            {"mean": mean, "scaled_covariance": covariance})
            for step in (7, 10):
                key = f"lga_{case}_{bank.name}_step{step}_conditional_B1"
                if key not in completed:
                    continue
                if (step == 7 and coarse_first is None) or (step == 10 and fine_first is None):
                    row = next(row for row in result["records"] if row["key"] == key)
                    with np.load(row["output_artifact"]["path"]) as arrays:
                        value = arrays["mean"].copy(), arrays["scaled_covariance"].copy()
                    if step == 7:
                        coarse_first = value
                    else:
                        fine_first = value
            already_audited = {(item["case"], item["bank"]) for item in result["lga_refinement"]}
            if (coarse_first is not None and fine_first is not None
                    and (case, bank.name) not in already_audited):
                result["lga_refinement"].append({"case": case, "bank": bank.name,
                    "scope": "one fresh timing histogram; numerical ODE output audit only, not LGA law accuracy",
                    "sqrtN_mean_difference_over_fixed_bank_scales": float(np.max(
                        np.abs(coarse_first[0]-fine_first[0])*np.sqrt(cfg.n_particles)/bank.scales)),
                    "covariance_relative_frobenius_difference": float(np.linalg.norm(
                        coarse_first[1]-fine_first[1])/np.linalg.norm(fine_first[1]))})
                _json(output, result)
    preregistration = _read(Path(root)/"payoffs"/"preregistration.json")
    transfers = {item["optimizer_seed"]: item["original_training_seconds"]
                 for item in preregistration.get("historical_transfers", [])}
    for item in lock["models"]:
        name = f"{item['bank']}_{item['init']}_seed{item['seed']}"
        tick = time.perf_counter()
        model = load_payoff_model(item["checkpoint_path"])
        first_restore = time.perf_counter()-tick
        if name not in {entry["name"] for entry in result["cold_restoration"]}:
            restore_timing, model = _time(lambda: load_payoff_model(item["checkpoint_path"]), repeats, warmup)
            result["cold_restoration"].append({"name": name, "first_measured_seconds": first_restore,
                                                **restore_timing})
            _json(output, result)
        training = _read(item["metadata_path"])["training_seconds"]
        for case in cfg.cases:
            for batch in batches:
                key = f"statistics_{case}_{name}_conditional_B{batch}"
                if key in completed:
                    continue
                initial = inputs[case+"_conditional"][:batch]
                timing, prediction = _time(lambda: predict_chunks(model, initial, cfg.query_times,
                                                                  inference_chunk), repeats, warmup)
                mean, covariance = prediction
                publish({"key": key, "method": "new-payoff-statistics-head", "case": case,
                         "bank": item["bank"], "init": item["init"], "optimizer_seed": item["seed"],
                         "role": "conditional", "batch_states": batch,
                         "checkpoint_sha256": item["checkpoint_sha256"],
                         "initial_pool_sha256": array_hash(initial),
                         "fresh_head_training_seconds": training,
                         "historical_trunk_training_seconds": transfers.get(item["seed"], 0.)
                         if item["init"] == "trunk-transfer" else 0., **timing},
                        {"mean": mean, "scaled_covariance": covariance})
            key = f"statistics_{case}_{name}_ensemble"
            if key in completed:
                continue
            initial = inputs[case+"_ensemble"]
            inference_timing, conditional = _time(lambda: predict_chunks(model, initial, cfg.query_times,
                                                                        inference_chunk), repeats, warmup)
            aggregation_timing, mixture = _time(lambda: mixture_moments(*conditional,
                n_particles=cfg.n_particles), repeats, warmup)
            counter = iter(range(repeats+warmup))
            components = []

            def end_to_end():
                repetition = next(counter)
                tick = time.perf_counter()
                origin = fresh_histograms(cfg, case, ensemble_paths,
                    role="fresh-ensemble-repetition", repetition=repetition)
                drawn = time.perf_counter()
                conditional_prediction = predict_chunks(model, origin, cfg.query_times, inference_chunk)
                inferred = time.perf_counter()
                value = mixture_moments(*conditional_prediction, n_particles=cfg.n_particles)
                aggregated = time.perf_counter()
                components.append((drawn-tick, inferred-drawn, aggregated-inferred))
                return value

            total_timing, total_prediction = _time(end_to_end, repeats, warmup)
            components = np.asarray(components[warmup:])
            mean, covariance = mixture
            publish({"key": key, "method": "new-payoff-statistics-head", "case": case,
                     "bank": item["bank"], "init": item["init"], "optimizer_seed": item["seed"],
                     "role": "IID-ensemble", "initial_histograms": ensemble_paths,
                     "inference_chunk": inference_chunk, "checkpoint_sha256": item["checkpoint_sha256"],
                     "shared_fixed_initial_pool_sha256": array_hash(initial),
                     "fixed_pool_inference": inference_timing, "fixed_pool_aggregation": aggregation_timing,
                     "fresh_initialization_inference_aggregation": total_timing,
                     "fresh_component_samples_seconds": {name: components[:, index].tolist()
                         for index, name in enumerate(("initialization", "inference", "aggregation"))},
                     "fresh_repeat_input_stream": {"seed": SEED, "role": "fresh-ensemble-repetition",
                         "warmup_repetitions": list(range(warmup)),
                         "measured_repetitions": list(range(warmup, warmup+repeats)),
                         "same_draws_across_all_heads": True},
                     "fresh_total_last_output_hashes": _check_output(*total_prediction),
                     "initial_law_integration_uncertainty": "not estimated by latency repetitions; scientific evaluator reports separately",
                     "fresh_head_training_seconds": training,
                     "historical_trunk_training_seconds": transfers.get(item["seed"], 0.)
                     if item["init"] == "trunk-transfer" else 0.},
                    {"mean": mean, "scaled_covariance": covariance})
    inputs.close()
    if source_hash() != plan["identity"]["source_hash"] or _sha(SCRIPT) != plan["identity"]["script_sha256"]:
        raise ValueError("Source changed during timing")
    result["status"] = "complete"
    result["record_count"] = len(result["records"])
    result["training_amortization"] = "Fresh head fitting and optional historical trunk fitting recorded separately; no reuse-count assumption"
    _json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/path-generalization-2026-10-26"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--final-timing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    kwargs = {"repeats": args.repeats, "warmup": args.warmup}
    if args.smoke:
        kwargs.update(batches=(1, 2), ensemble_paths=8, inference_chunk=4)
    if args.prepare_only:
        directory, plan = prepare(args.root, **kwargs)
        print(json.dumps({"status": "prepared-only", "directory": str(directory),
                          "inputs_sha256": plan["inputs_sha256"]}, indent=2))
    else:
        result = run_benchmark(args.root, final_timing=args.final_timing, smoke=args.smoke, **kwargs)
        print(json.dumps({"status": result["status"], "record_count": result["record_count"]}, indent=2))


if __name__ == "__main__":
    main()
