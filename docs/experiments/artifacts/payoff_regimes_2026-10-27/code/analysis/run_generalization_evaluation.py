"""Resumable matched-task evaluation after all new-payoff fits are locked.

Prepared inference, classical Picard and Euler consume the same physical face
increments. Only reduced payoffs are retained per candidate; the checkpoint and
shared input bank reproduce complete density trajectories. Large 480-query
bootstrap statistics are accumulated in two transactional slots instead of
persisting every candidate's 480-column path table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import beta, ks_2samp, wasserstein_distance  # noqa: E402

from analysis.evaluate_path_generalization import (  # noqa: E402
    array_sha,
    bootstrap_counts,
    diagonal_comparison,
    distribution_comparison,
    empirical,
    mixture_moments,
    moment_comparison,
    reduce_paths,
)
from analysis.native_gaussian_baseline import (  # noqa: E402
    iid_scaled_covariance,
    linear_gaussian,
    sample_gaussian_process,
)
from analysis.path_generalization_reference import ReferenceConfig, load_reference  # noqa: E402
from analysis.probe_path_workloads import numpy_euler, workload_values  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.path_model import CausalPathOperator  # noqa: E402
from nn_dk.path_runtime import PreparedPathRuntime  # noqa: E402
from nn_dk.payoff_learning import PayoffBank, load_payoff_model  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

SCRIPT = Path(__file__).resolve()
KIND = "broad-task-paired-density-and-payoff-comparison-v1"
SEED = 2026102614
N_BOOTSTRAP = 128
BATCH_SIZE = 64
STEP_KEYS = {2**-8: "e30d58b3a0", 2**-9: "645459a4bd", 2**-10: "699c7c5bf9"}
SOURCE_FREEZE = "1a274fcf4d2b9c9b19f4605f514f38d8465807216a99419504f77d7796442df1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _immutable(path, value):
    if Path(path).exists():
        if _read(path) != value:
            raise ValueError(f"Frozen evaluation identity changed: {path}")
    else:
        _json(path, value)


def _banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def _seed(key, namespace="families"):
    return int(stable_hash([SEED, key, namespace])[:15], 16)


def candidate_plan(cfg):
    result = []
    for case in cfg.cases:
        path_case = "dk_interacting_uniform" if case == "uniform" else "dk_interacting"
        for dt, key in STEP_KEYS.items():
            for seed in (0, 1, 2):
                path = REPO/"runs"/"interacting-followup-2026-10-25"/"paths"/"checkpoints"/f"{path_case}_{key}_seed{seed}.pt"
                result.append({"name": f"path_dt{round(-np.log2(dt))}_seed{seed}",
                               "method": "prepared-learned-path", "case": case, "dt": dt,
                               "seed": seed, "depth": 3, "checkpoint_path": str(path),
                               "checkpoint_sha256": _sha(path), "historical_commit": "26ac405",
                               "metadata_sha256": _sha(path.with_suffix(".json")),
                               "runtime_mode": "dense-cached", "batch_size": BATCH_SIZE})
        for depth in (1, 2, 3):
            result.append({"name": f"classical_picard_depth{depth}", "method": "exact-kernel-picard",
                           "case": case, "dt": 2**-10, "depth": depth,
                           "runtime_mode": "dense-cached", "batch_size": BATCH_SIZE,
                           "training_required": False})
        for dt in STEP_KEYS:
            result.append({"name": f"native_euler_dt{round(-np.log2(dt))}", "method": "native-euler",
                           "case": case, "dt": dt, "batch_size": BATCH_SIZE,
                           "training_required": False})
    return result


def prepare(root, cfg=ReferenceConfig()):
    """Freeze all method choices before opening any final reference outcomes."""
    root, output = Path(root).resolve(), Path(root).resolve()/"evaluation"
    if source_hash() != SOURCE_FREEZE:
        raise ValueError("Evaluation requires the declared complete package source freeze")
    if cfg.reference_factors != (1, 2, 4):
        raise ValueError("This fixed experiment preregisters exactly three nested reference levels")
    helpers = ("evaluate_path_generalization.py", "native_gaussian_baseline.py",
               "probe_path_workloads.py", "path_generalization_reference.py")
    historical = REPO/"runs"/"interacting-followup-2026-10-25"
    historical_inputs = [historical/"paths"/f"path_{role}.h5" for role in ("train", "validation")]
    historical_inputs += [historical/"unscaled"/f"interacting_{role}.h5" for role in ("train", "validation")]
    identity = {"kind": KIND, "source_hash": source_hash(), "source_freeze_commit": "0599cf3",
                "script_sha256": _sha(SCRIPT),
                "helpers": {name: _sha(SCRIPT.with_name(name)) for name in helpers},
                "reference_config": json.loads(json.dumps(asdict(cfg))),
                "reference_config_hash": cfg.config_hash, "candidate_plan": candidate_plan(cfg),
                "banks": {bank.name: bank.to_dict() for bank in _banks(cfg)},
                "groups": list(cfg.group_keys), "paths_per_group": cfg.paths_per_group,
                "optimizer_seeds": [0, 1, 2], "statistical_fit_count": 12,
                "lga": {"law": "unstopped-linear-noise-approximation",
                        "requested_rk4_steps": [2**-7, 2**-10],
                        "actual_coarse_substep_on_common_grid": cfg.common_step,
                        "nonlinear_samples": cfg.paths_per_group},
                "bootstrap": {"unit": "whole-trajectory-family", "seed": SEED,
                              "count": N_BOOTSTRAP, "shared_counts_within_group": True},
                "task_normalization": "reference empirical standard deviations; covariance products",
                "reference_bias": "coupled refinement diagnostics; continuous hitting bias unresolved",
                "stats_joint_time_support": False, "selection_using_final_results": False,
                "candidate_storage": "reduced banks/nonlinear/stops/strong summaries; no raw 480-query tables"}
    identity["hypotheses"] = {
        "reuse": "A reusable full-path operator may reduce adaptation cost for newly declared payoffs",
        "specialization": "A separately fitted statistical head may win after repeated conditional queries",
        "learning_contribution": "Exact-kernel Picard tests whether gains require a fitted kernel",
        "classical_statistics": "A no-training linear-Gaussian approximation may suffice in this short weak-noise regime"}
    identity["acceptance_interpretation"] = (
        "5% normalized error and 1.25% uncertainty are research targets; simultaneous-in-bank "
        "sampling diagnostics are separate from numerical-reference and stopping bias. "
        "No best seed, initialization, or time step is selected using final outcomes.")
    identity["historical_input_banks"] = [{"path": str(path.resolve()), "sha256": _sha(path)}
                                           for path in historical_inputs]
    identity["historical_path_config"] = _read(historical/"paths"/"config.json")
    identity["allocation_interpretation"] = (
        "fixed previously trained width16 path operators versus fresh width32 payoff heads; "
        "data/capacity differ, so this compares deployed/adaptation accuracy-cost, not sample efficiency")
    _immutable(output/"preregistration.json", identity)
    return output, identity


def _verify_fits(root, expected=12):
    path = Path(root)/"payoffs"/"fit_lock.json"
    lock = _read(path)
    if (not lock.get("complete") or lock.get("source_hash") != source_hash()
            or lock.get("expected_models") != expected or len(lock.get("models", [])) != expected):
        raise ValueError("Evaluation requires all declared immutable payoff fits")
    slots = set()
    for item in lock["models"]:
        slot = (item["bank"], item["seed"], item["init"])
        if slot in slots:
            raise ValueError("Duplicate payoff fit slot")
        slots.add(slot)
        if (_sha(item["checkpoint_path"]) != item["checkpoint_sha256"]
                or _sha(item["metadata_path"]) != item["metadata_sha256"]):
            raise ValueError("A locked payoff fit changed")
        model = load_payoff_model(item["checkpoint_path"])
        if (not model.complete or model.target_id != item["target_id"]
                or model.bank.bank_hash != item["bank_hash"]):
            raise ValueError("A payoff fit could not be strictly restored")
    expected_slots = {(bank.name, seed, initialization) for bank in _banks(ReferenceConfig())
                      for seed in (0, 1, 2) for initialization in ("cold", "trunk-transfer")}
    if slots != expected_slots:
        raise ValueError("The complete predeclared bank/seed/initialization inventory is required")
    for packet in lock["input_banks"].values():
        if _sha(packet["path"]) != packet["sha256"] or _sha(packet["metadata_path"]) != packet["metadata_sha256"]:
            raise ValueError("The locked current-state training/validation pool changed")
    return lock, _sha(path)


def _training_families(lock, historical_inputs=()):
    pools = []
    for packet in lock["input_banks"].values():
        with h5py.File(packet["path"], "r") as store:
            families = {family.decode() for case in store for family in store[case]["family_ids"][:]}
        pools.append(families)
    if pools[0] & pools[1]:
        raise ValueError("Training and validation families overlap")
    result = set.union(*pools)
    for packet in historical_inputs:
        if _sha(packet["path"]) != packet["sha256"]:
            raise ValueError("A frozen historical learner input bank changed")
        with h5py.File(packet["path"], "r") as store:
            result.update(family.decode() for case in store for family in store[case]["family_ids"][:])
    return result


def _input_identity(root, cfg):
    lock, fit_sha = _verify_fits(root)
    directory = Path(root)/"reference"
    manifest = _read(directory/"manifest.json")
    reference_path = directory/"reference.h5"
    if (manifest.get("status") != "complete" or manifest.get("reference_sha256") != _sha(reference_path)
            or manifest["identity"]["config_hash"] != cfg.config_hash):
        raise ValueError("A complete immutable independent reference is required")
    return {"fit_lock_sha256": fit_sha, "reference_sha256": manifest["reference_sha256"],
            "reference_manifest_sha256": _sha(directory/"manifest.json"),
            "reference_config_hash": cfg.config_hash}, lock


def _many_reference(values, counts):
    center = values.mean(0)
    z = values-center
    n = len(z)
    variance = (z*z).sum(0)/(n-1)
    delta = counts@z/n
    return {"center": center, "variance": variance, "boot_mean": center+delta,
            "boot_variance": (counts@(z*z)-n*delta**2)/(n-1)}


def _many_from_accumulator(reference, summary, counts, accumulator, n, n_particles):
    mean = summary["center"]+accumulator["sum"]/n
    variance = (accumulator["square"]-accumulator["sum"]**2/n)/(n-1)
    boot_mean = summary["center"]+accumulator["boot_sum"]/n
    boot_variance = (accumulator["boot_square"]-accumulator["boot_sum"]**2/n)/(n-1)
    result = diagonal_comparison(reference, predicted_mean=mean,
        predicted_scaled_variance=n_particles*variance, n_particles=n_particles, counts=counts)
    std, rv = np.sqrt(summary["variance"]), summary["variance"]
    dm, dv = (mean-summary["center"])/std, (variance-rv)/rv
    bdm = (boot_mean-summary["boot_mean"])/std
    bdv = (boot_variance-summary["boot_variance"])/rv
    result["reference_only_mean_sampling95_radius"] = result["mean_sampling95_simultaneous_radius"]
    result["reference_only_variance_sampling95_radius"] = result["variance_sampling95_simultaneous_radius"]
    result["mean_sampling95_simultaneous_radius"] = float(np.quantile(abs(bdm-dm).max(-1), .95))
    result["variance_sampling95_simultaneous_radius"] = float(np.quantile(abs(bdv-dv).max(-1), .95))
    result["paired_whole_family_candidate_uncertainty_included"] = True
    return result, {"many_bootstrap_mean": boot_mean, "many_bootstrap_variance": boot_variance,
                    "many_mean": mean, "many_variance": variance}


def _runtime(method, cfg):
    started = time.perf_counter()
    context = cfg.context(method["case"])
    if method["method"] == "native-euler":
        return None, {"restoration_and_preparation_seconds": 0.}
    if method["method"] == "prepared-learned-path":
        runtime = PreparedPathRuntime.from_checkpoint(method["checkpoint_path"], "26ac405",
                                                      mode="dense-cached")
    else:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = CausalPathOperator(cfg.spec, "dk", width=2, depth=method["depth"])
        runtime = PreparedPathRuntime(model, context, method["dt"], cfg.horizon,
                                      mode="dense-cached", learned=False, depth=method["depth"])
    if runtime.context != context or runtime.spec != cfg.spec or runtime.horizon != cfg.horizon:
        raise ValueError("Density-path runtime physics/grid/horizon differs from the reference")
    runtime.prepare(round(cfg.horizon/method["dt"]))
    return runtime, {"restoration_and_preparation_seconds": time.perf_counter()-started,
                     "runtime_kind": runtime.kind, "runtime_provenance": runtime.provenance}


def _cache_shapes(n, q, b=N_BOOTSTRAP):
    return {"bank_b": (n, 3, 4), "bank_c": (n, 3, 4), "nonlinear": (n, 3),
            "stopped_at": (n,), "sup_error_sq": (n,), "time_error_sq": (n,),
            "minimum_mass": (n,), "mass_error": (n,),
            "acc_sum": (2, q), "acc_square": (2, q),
            "acc_boot_sum": (2, b, q), "acc_boot_square": (2, b, q)}


def _raw_chunk_hash(store, start, end):
    return stable_hash({name: array_sha(store[name][start:end]) for name in (
        "bank_b", "bank_c", "nonlinear", "stopped_at", "sup_error_sq", "time_error_sq",
        "minimum_mass", "mass_error")})


def path_candidate(path, method, cfg, data, counts, many_reference, identity, deadline=None):
    """Commit fixed-size batches with two-slot large-bank bootstrap accumulation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, q = len(data["initial"]), data["many_observables"].shape[1]
    chunks = (n+BATCH_SIZE-1)//BATCH_SIZE
    if not path.exists():
        with h5py.File(path, "w") as store:
            store.attrs.update(identity_json=json.dumps(identity, sort_keys=True), complete=False,
                               simulation_seconds=0., reduction_seconds=0.)
            for name, shape in _cache_shapes(n, q, len(counts)).items():
                store.create_dataset(name, shape, dtype="f8", compression="lzf")
            store.create_dataset("progress", data=0)
            store.create_dataset("chunk_sha256", (chunks,), dtype="S64")
            store.create_dataset("accumulator_sha256", (2,), dtype="S64")
    runtime, preparation = _runtime(method, cfg)
    dt = method["dt"]
    factor = round(dt/cfg.model_step)
    with h5py.File(path, "r+") as store:
        if json.loads(store.attrs["identity_json"]) != identity:
            raise ValueError("Candidate cache method/source/input/bootstrap identity changed")
        done = int(store["progress"][()])
        if not 0 <= done <= chunks:
            raise ValueError("Candidate cache has an invalid committed progress marker")
        for chunk in range(done):
            start, end = chunk*BATCH_SIZE, min(n, (chunk+1)*BATCH_SIZE)
            if _raw_chunk_hash(store, start, end) != store["chunk_sha256"][chunk].decode():
                raise ValueError("A committed reduced candidate chunk changed")
        accumulator = {key: store["acc_"+key][done % 2] for key in (
            "sum", "square", "boot_sum", "boot_square")}
        if done and stable_hash({key: array_sha(value) for key, value in accumulator.items()}) != store["accumulator_sha256"][done % 2].decode():
            raise ValueError("Committed many-query bootstrap accumulator changed")
        for chunk in range(done, chunks):
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Reduced candidate cache is resumable from its committed chunk")
            start, end = chunk*BATCH_SIZE, min(n, (chunk+1)*BATCH_SIZE)
            initial = data["initial"][start:end]
            noise = data["model_increments"][start:end]
            noise = noise.reshape(len(initial), noise.shape[1]//factor, factor, cfg.n_cells).sum(2)
            tick = time.perf_counter()
            if runtime is None:
                packet = numpy_euler(initial, noise, cfg.context(method["case"]), cfg.spec, dt)
            else:
                with torch.no_grad():
                    packet, _ = runtime(initial, noise, return_raw=True)
            store.attrs["simulation_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            values = reduce_paths(packet, cfg.spec, dt, cfg.horizon, cfg.common_step)
            states = packet["states"]
            states = states.detach().numpy() if isinstance(states, torch.Tensor) else states
            if (not np.isfinite(states).all() or states.min() < -1e-12
                    or not np.allclose(states.sum(-1), 1., atol=1e-10, rtol=0)):
                raise ValueError("Invalid density path prevents every survivor-conditioned estimate")
            difference = values["common_states"]-data["common_paths"][start:end]
            energy = np.square(difference).sum(-1)
            raw = {"bank_b": values["banks"]["terminal-modes34"],
                   "bank_c": values["banks"]["terminal3-integral1"],
                   "nonlinear": values["nonlinear"], "stopped_at": values["stopped_at"],
                   "sup_error_sq": energy.max(-1), "time_error_sq": energy.mean(-1),
                   "minimum_mass": states.min(axis=(1, 2)),
                   "mass_error": np.abs(states.sum(-1)-1).max(-1)}
            for name, value in raw.items():
                store[name][start:end] = value
            z = values["many"]-many_reference["center"]
            accumulator["sum"] += z.sum(0)
            accumulator["square"] += (z*z).sum(0)
            accumulator["boot_sum"] += counts[:, start:end]@z
            accumulator["boot_square"] += counts[:, start:end]@(z*z)
            next_slot = (chunk+1) % 2
            for name, value in accumulator.items():
                store["acc_"+name][next_slot] = value
            store["accumulator_sha256"][next_slot] = stable_hash({
                name: array_sha(value) for name, value in accumulator.items()}).encode()
            store["chunk_sha256"][chunk] = _raw_chunk_hash(store, start, end).encode()
            store.attrs["reduction_seconds"] += time.perf_counter()-tick
            store.flush()
            store["progress"][()] = chunk+1
            store.flush()
        store.attrs["complete"] = True
        store.flush()
        result = {name: store[name][:] for name in (
            "bank_b", "bank_c", "nonlinear", "stopped_at", "sup_error_sq", "time_error_sq",
            "minimum_mass", "mass_error")}
        result["accumulator"] = accumulator
        result["cost"] = {**preparation, "simulation_seconds": float(store.attrs["simulation_seconds"]),
                          "reduction_seconds": float(store.attrs["reduction_seconds"]),
                          "scope": "scientific evaluation under possible contention; separate warm benchmark required"}
    return result


def _strong(energy, counts, n_particles, n_cells):
    scale = n_particles/(1-1/n_cells)
    point = float(np.sqrt(scale*np.mean(energy)))
    samples = np.sqrt(scale*(counts@energy)/len(energy))
    return {"normalized_rms": point, "bootstrap95_interval": np.quantile(samples, [.025, .975]).tolist(),
            "normalization": "uniform-IID-native-histogram-fluctuation-energy",
            "all_cells_and_common_physical_times": True}


def _stop_comparison(reference, candidate, horizon, counts):
    r, c = np.isfinite(reference), np.isfinite(candidate)

    def interval(hits):
        k, n = int(hits.sum()), len(hits)
        return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                1. if k == n else float(beta.ppf(.975, k+1, n-k))]

    difference = c.astype(float)-r
    rt, ct = np.minimum(reference, horizon), np.minimum(candidate, horizon)
    return {"reference_probability": float(r.mean()), "candidate_probability": float(c.mean()),
            "reference_exact_binomial95_interval": interval(r),
            "candidate_exact_binomial95_interval": interval(c),
            "paired_probability_difference": float(difference.mean()),
            "paired_difference_bootstrap95_interval": np.quantile(counts@difference/len(r), [.025, .975]).tolist(),
            "censored_at_horizon_ks": float(ks_2samp(rt, ct).statistic),
            "censored_wasserstein_over_horizon": float(wasserstein_distance(rt, ct)/horizon),
            "zero_observed_stops_is_not_zero_population_probability": True}


def _raw_bank_metrics(data, candidate, cfg, counts, boot_output):
    result = {}
    for name, label in (("bank_b", "terminal-modes34"), ("bank_c", "terminal3-integral1")):
        reference, values = data[name], candidate[name]
        result[label] = {"same_time": moment_comparison(reference, values, counts=counts,
                                                       n_particles=cfg.n_particles),
                         "joint_time_12": moment_comparison(
                             reference.reshape(len(reference), 1, -1), values.reshape(len(values), 1, -1),
                             counts=counts, n_particles=cfg.n_particles),
                         "scalar_distributions": distribution_comparison(reference, values)}
        bm, bc = empirical(values, counts)
        boot_output[label+"_bootstrap_mean"] = bm
        boot_output[label+"_bootstrap_scaled_covariance"] = cfg.n_particles*bc
    result["nonlinear3"] = {"moments": moment_comparison(data["workloads"], candidate["nonlinear"],
                                                        counts=counts, n_particles=cfg.n_particles),
                            "scalar_distributions": distribution_comparison(data["workloads"], candidate["nonlinear"]),
                            "definitions": ["terminal-normalized-cos3", "density-energy-common-grid-integral",
                                            "temperature-.05-log-mean-exp-cos3-common-grid"]}
    return result


def _publish(directory, result, boot_arrays, cache_path=None):
    directory = Path(directory)
    bootstrap_path = directory.with_suffix(".npz")
    _npz(bootstrap_path, boot_arrays)
    result["bootstrap_artifact"] = {"path": str(bootstrap_path.resolve()), "sha256": _sha(bootstrap_path)}
    if cache_path is not None:
        result["reduced_artifact"] = {"path": str(Path(cache_path).resolve()), "sha256": _sha(cache_path)}
    _json(directory.with_suffix(".json"), result)
    return result


def _replay(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    result = _read(path)
    if result.get("identity") != identity or not result.get("complete"):
        raise ValueError("Cached method result has incompatible source/input/method identity")
    for name in ("bootstrap_artifact", "reduced_artifact"):
        if name in result and _sha(result[name]["path"]) != result[name]["sha256"]:
            raise ValueError("A completed method evidence artifact changed")
    return result


def evaluate_path(directory, method, cfg, data, counts, many_reference, identity, deadline=None):
    path = Path(directory)/method["name"]
    replay = _replay(path.with_suffix(".json"), identity)
    if replay is not None:
        return replay
    cache = path.with_suffix(".h5")
    values = path_candidate(cache, method, cfg, data, counts, many_reference, identity, deadline)
    bootstrap = {}
    metrics = _raw_bank_metrics(data, values, cfg, counts, bootstrap)
    many, many_bootstrap = _many_from_accumulator(data["many_observables"], many_reference,
        counts, values["accumulator"], len(data["initial"]), cfg.n_particles)
    metrics["many480"] = many
    bootstrap.update(many_bootstrap)
    result = {"identity": identity, "complete": True, "method": method,
              "metrics": metrics,
              "strong_common_grid_sup": _strong(values["sup_error_sq"], counts, cfg.n_particles, cfg.n_cells),
              "strong_common_grid_time_rms": _strong(values["time_error_sq"], counts, cfg.n_particles, cfg.n_cells),
              "stopping": _stop_comparison(data["stopped_at"], values["stopped_at"], cfg.horizon, counts),
              "conservation": {"minimum_mass": float(values["minimum_mass"].min()),
                               "maximum_mass_error": float(values["mass_error"].max()), "failed_paths": 0},
              "cost": values["cost"], "continuous_reference_bias": "unresolved"}
    if "checkpoint_path" in method:
        metadata = _read(Path(method["checkpoint_path"]).with_suffix(".json"))
        result["cost"]["historical_training_seconds"] = metadata["training_seconds"]
    else:
        result["cost"]["historical_training_seconds"] = 0.
    return _publish(path, result, bootstrap, cache)


def _integration_uncertainty(mean, covariance, bm, bc, reference):
    _, rc = empirical(reference)
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scales = std[..., :, None]*std[..., None, :]
    # Covariances passed here are already N-scaled. Convert their uncertainty
    # using N times the reference physical variance outside this helper.
    return {"mean95_radius_in_reference_std_by_time": np.quantile(
        np.max(abs(bm-mean)/std, axis=-1), .95, axis=0).tolist(),
        "scaled_covariance95_absolute_radius_by_time": np.quantile(
            np.max(abs(bc-covariance), axis=(-2, -1)), .95, axis=0).tolist(),
        "reference_covariance_products": scales.tolist(),
        "scope": "standalone IID initial-law Monte Carlo integration; no future-path sampling"}


def evaluate_statistics(directory, item, cfg, data, counts, identity):
    name = f"statistics_{item['bank']}_{item['init']}_seed{item['seed']}"
    path = Path(directory)/name
    replay = _replay(path.with_suffix(".json"), identity)
    if replay is not None:
        return replay
    started = time.perf_counter()
    model = load_payoff_model(item["checkpoint_path"])
    restoration = time.perf_counter()-started
    conditional = data["metadata"]["role"] == "conditional"
    model.require_physics(cfg.context(data["metadata"]["case"]))
    if model.spec != cfg.spec or model.horizon != cfg.horizon:
        raise ValueError("Statistics head grid/horizon differs from the reference")
    if conditional and not np.array_equal(data["initial"], np.broadcast_to(
            data["initial"][0], data["initial"].shape)):
        raise ValueError("Conditional reference group contains different initial histograms")
    initial = data["initial"][:1] if conditional else data["initial"]
    tick = time.perf_counter()
    prediction = model.predict_conditional(initial, cfg.query_times)
    inference = time.perf_counter()-tick
    if conditional:
        mean, covariance = prediction.mean[0], prediction.scaled_covariance[0]
        bm, bc = None, None
    else:
        mean, covariance = mixture_moments(prediction.mean, prediction.scaled_covariance,
                                           n_particles=cfg.n_particles)
        bm, bc = mixture_moments(prediction.mean, prediction.scaled_covariance,
                                 n_particles=cfg.n_particles, counts=counts)
    reference_name = "bank_b" if item["bank"] == "terminal-modes34" else "bank_c"
    reference = data[reference_name]
    metrics = moment_comparison(reference, predicted_mean=mean, predicted_scaled_covariance=covariance,
        bootstrap_predicted_mean=bm, bootstrap_predicted_scaled_covariance=bc,
        n_particles=cfg.n_particles, counts=counts)
    bootstrap = {"predicted_mean": mean, "predicted_scaled_covariance": covariance}
    if not conditional:
        bootstrap.update(bootstrap_predicted_mean=bm, bootstrap_predicted_scaled_covariance=bc)
        integration = _integration_uncertainty(mean, covariance, bm, bc, reference)
        _, rc = empirical(reference)
        denominator = cfg.n_particles*np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))[..., :, None] \
            * np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))[..., None, :]
        integration["covariance95_normalized_radius_by_time"] = np.quantile(
            np.max(abs(bc-covariance)/denominator, axis=(-2, -1)), .95, axis=0).tolist()
        integration["initial_histograms"] = len(initial)
        integration["comparison_coupling"] = "shared-reference-IID-initial-families; paired difference bootstrap"
        integration["standalone_uncertainty"] = "bootstrap variability of deployed initial-law integration itself"
    else:
        integration = {"initial_histograms": 1, "initial_law_integration_required": False,
                       "mean_sampling_uncertainty": 0., "covariance_sampling_uncertainty": 0.}
    raw_path = path.with_suffix(".h5")
    with h5py.File(raw_path, "w") as store:
        store.attrs.update(identity_json=json.dumps(identity, sort_keys=True), complete=True)
        store.create_dataset("conditional_mean", data=prediction.mean, compression="lzf")
        store.create_dataset("conditional_scaled_covariance", data=prediction.scaled_covariance, compression="lzf")
        store.create_dataset("initial", data=initial, compression="lzf")
    metadata = _read(item["metadata_path"])
    result = {"identity": identity, "complete": True,
              "method": {**item, "name": name, "method": "statistics-head"}, "bank": item["bank"],
              "metrics": {"same_time": metrics}, "integration_uncertainty": integration,
              "unsupported": ["joint-time covariance", "scalar distributions", "full density paths", "stopping probabilities"],
              "cost": {"restoration_seconds": restoration, "inference_seconds": inference,
                       "fresh_payoff_training_seconds": metadata["training_seconds"],
                       "scope": "scientific prediction; separate uncontended warm timings required"},
              "continuous_reference_bias": "unresolved"}
    return _publish(path, result, bootstrap, raw_path)


def _lga_projection(bank, n_cells):
    projection = np.zeros((4, n_cells+2))
    projection[:, :n_cells] = bank.terminal_weights
    if bank.name == "terminal3-integral1":
        projection[2, n_cells], projection[3, n_cells+1] = 1., 1.
    return projection


def _gaussian_prediction(result, projection, qi):
    mean = result["mean_augmented"][qi]@projection.T
    covariance = np.einsum("ai,tij,bj->tab", projection,
                           result["scaled_covariance_augmented"][qi], projection)
    joint = result["joint_scaled_covariance_augmented"][qi][:, :, qi, :]
    projected = np.einsum("ai,tisj,bj->tasb", projection, joint, projection)
    return mean, covariance, projected.reshape(12, 12)


def _lga_many(result, cfg):
    centers = (np.arange(cfg.n_cells)+.5)*cfg.spec.h
    weights = np.sqrt(2)*np.asarray([function(mode*centers)*np.sinc(mode/cfg.n_cells)
        for mode in range(1, cfg.n_cells//2) for function in (np.cos, np.sin)])
    mean = result["mean_states"][1:]@weights.T
    variance = np.einsum("oi,tij,oj->to", weights, result["scaled_covariance_states"][1:], weights)
    return mean.reshape(-1), variance.reshape(-1)


def evaluate_lga(directory, cfg, data, counts, identity, step):
    name = "linear_gaussian_rk4_"+str(round(-np.log2(step)))
    path = Path(directory)/name
    replay = _replay(path.with_suffix(".json"), identity)
    if replay is not None:
        return replay
    conditional = data["metadata"]["role"] == "conditional"
    initial = data["initial"][0] if conditional else data["probabilities"]
    initial_covariance = None if conditional else iid_scaled_covariance(initial)
    bank_b, bank_c = _banks(cfg)
    result = linear_gaussian(initial, cfg.context(data["metadata"]["case"]), cfg.spec,
        cfg.common_times, step, initial_scaled_covariance=initial_covariance,
        terminal_weights=np.eye(cfg.n_cells), running_weights=bank_c.running_weights[2:], return_joint=True)
    qi = np.rint(np.asarray(cfg.query_times)/cfg.common_step).astype(int)
    metrics, bootstrap, predictions = {}, {}, {}
    for bank, key in ((bank_b, "bank_b"), (bank_c, "bank_c")):
        projection = _lga_projection(bank, cfg.n_cells)
        mean, covariance, joint = _gaussian_prediction(result, projection, qi)
        metrics[bank.name] = {"same_time": moment_comparison(data[key], predicted_mean=mean,
            predicted_scaled_covariance=covariance, n_particles=cfg.n_particles, counts=counts),
            "joint_time_12": moment_comparison(data[key].reshape(len(data[key]), 1, 12),
                predicted_mean=mean.reshape(1, 12), predicted_scaled_covariance=joint[None],
                n_particles=cfg.n_particles, counts=counts)}
        predictions[bank.name] = {"mean": mean.tolist(), "scaled_covariance": covariance.tolist(),
                                  "joint_scaled_covariance": joint.tolist()}
        bootstrap[bank.name+"_mean"] = mean
        bootstrap[bank.name+"_scaled_covariance"] = covariance
        bootstrap[bank.name+"_joint_scaled_covariance"] = joint
    many_mean, many_variance = _lga_many(result, cfg)
    metrics["many480"] = diagonal_comparison(data["many_observables"], predicted_mean=many_mean,
        predicted_scaled_variance=many_variance, n_particles=cfg.n_particles, counts=counts)
    # Sampling this explicitly declared Gaussian process gives a real classical
    # nonlinear-payoff comparator, with independent whole-family uncertainty.
    tick = time.perf_counter()
    sampled = sample_gaussian_process(result, cfg.n_particles, len(data["initial"]),
                                     seed=_seed(identity["group"], "LGA-nonlinear-"+str(step)))
    sample_seconds = time.perf_counter()-tick
    states = sampled["samples_augmented"][:, :, :cfg.n_cells]
    nonlinear = workload_values(states, cfg.spec, cfg.common_step)
    metrics["nonlinear3"] = {"moments": moment_comparison(data["workloads"], nonlinear,
        n_particles=cfg.n_particles, counts=counts, paired=False,
        seed=_seed(identity["group"], "LGA-nonlinear-bootstrap-"+str(step))),
        "scalar_distributions": distribution_comparison(data["workloads"], nonlinear),
        "law": "unstopped-Gaussian-approximation; no density repair or DK-stop claim"}
    for bank, key in ((bank_b, "bank_b"), (bank_c, "bank_c")):
        values = sampled["samples_augmented"][:, qi]@_lga_projection(bank, cfg.n_cells).T
        metrics[bank.name]["scalar_distributions"] = distribution_comparison(data[key], values)
    bootstrap.update(many_mean=many_mean, many_scaled_variance=many_variance,
                     lga_mean_augmented=result["mean_augmented"],
                     lga_scaled_covariance_augmented=result["scaled_covariance_augmented"],
                     lga_joint_scaled_covariance_augmented=result["joint_scaled_covariance_augmented"])
    raw_path = path.with_suffix(".h5")
    with h5py.File(raw_path, "w") as store:
        store.attrs.update(identity_json=json.dumps(identity, sort_keys=True), complete=True,
                           law=result["law"], gaussian_sampling_seed=sampled["diagnostics"]["seed"])
        store.create_dataset("nonlinear", data=nonlinear, compression="lzf")
    record = {"identity": identity, "complete": True,
              "method": {"name": name, "method": "linear-Gaussian", "requested_max_step": step},
              "metrics": metrics, "predictions": predictions,
              "law_diagnostics": result["diagnostics"],
              "gaussian_sampling_diagnostics": {**sampled["diagnostics"],
                  "negative_native_mass_fraction": float(np.any(states < 0, axis=(1, 2)).mean()),
                  "minimum_native_mass": float(states.min()),
                  "below_stopping_threshold_fraction": float(np.any(states <= cfg.spec.mass_stop, axis=(1, 2)).mean())},
              "unsupported": ["same-Brownian strong density-path error", "stopped-DK equivalence"],
              "cost": {"deterministic_integration_seconds": result["diagnostics"]["integration_seconds"],
                       "Gaussian_sampling_seconds": sample_seconds, "training_seconds": 0.,
                       "scope": "scientific evaluation; separate warm uncontended benchmark required"},
              "continuous_reference_bias": "unresolved", "linearization_and_stopping_bias": "unresolved"}
    return _publish(path, record, bootstrap, raw_path)


def reference_audit(directory, cfg, data, identity):
    path = Path(directory)/"reference_refinement.json"
    replay = _replay(path, identity)
    if replay is not None:
        return replay
    n = len(data["audit_bank_b"])
    counts = bootstrap_counts(n, _seed(identity["group"], "reference-audit"), N_BOOTSTRAP)
    records, bootstrap = [], {"counts": counts.astype(np.uint16)}
    for coarse, fine in ((0, 1), (1, 2), (0, 2)):
        reference = {"bank_b": data["audit_bank_b"][:, fine],
                     "bank_c": data["audit_bank_c"][:, fine],
                     "workloads": data["audit_workloads"][:, fine]}
        candidate = {"bank_b": data["audit_bank_b"][:, coarse],
                     "bank_c": data["audit_bank_c"][:, coarse],
                     "nonlinear": data["audit_workloads"][:, coarse]}
        prefix = f"levels_{coarse}_{fine}_"
        raw_bootstrap = {}
        metrics = _raw_bank_metrics(reference, candidate, cfg, counts, raw_bootstrap)
        bootstrap.update({prefix+key: value for key, value in raw_bootstrap.items()})
        metrics["many480"] = diagonal_comparison(data["audit_many_observables"][:, fine],
            data["audit_many_observables"][:, coarse], counts=counts, n_particles=cfg.n_particles)
        delta = data["audit_common_paths"][:, coarse]-data["audit_common_paths"][:, fine]
        records.append({"coarse_factor": cfg.reference_factors[coarse],
                        "fine_factor": cfg.reference_factors[fine], "metrics": metrics,
                        "strong": _strong(np.square(delta).sum(-1).max(-1), counts, cfg.n_particles, cfg.n_cells),
                        "stopping": _stop_comparison(data["audit_stopped_at"][:, fine],
                            data["audit_stopped_at"][:, coarse], cfg.horizon, counts)})
    return _publish(path.with_suffix(""), {"identity": identity, "complete": True,
        "records": records, "n_audited_families": n,
        "interpretation": "coupled finite-reference differences, not a continuous-boundary bias enclosure"}, bootstrap)


def _read_group(path, key):
    # Full integrity checking has already been performed once per invocation
    # through load_reference. Keep only one group's large input arrays resident.
    with h5py.File(path, "r") as store:
        group = store[key]
        result = {name: group[name][:] for name in group if name not in (
            "chunk_complete", "chunk_sha256", "chunk_seconds")}
        result["metadata"] = dict(group.attrs)
    return result


def _group_bootstrap(directory, data, counts, many_reference):
    path = Path(directory)/"reference_bootstrap.npz"
    values = {"counts": counts.astype(np.uint16), "many_reference_mean": many_reference["center"],
              "many_reference_variance": many_reference["variance"],
              "many_reference_bootstrap_mean": many_reference["boot_mean"],
              "many_reference_bootstrap_variance": many_reference["boot_variance"]}
    for name in ("bank_b", "bank_c", "workloads"):
        mean, covariance = empirical(data[name], counts)
        values[name+"_bootstrap_mean"] = mean
        values[name+"_bootstrap_covariance"] = covariance
    if path.exists():
        with np.load(path) as prior:
            if set(prior) != set(values) or any(not np.array_equal(prior[name], value)
                                               for name, value in values.items()):
                raise ValueError("Group reference bootstrap artifact changed")
    else:
        _npz(path, values)
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _lga_refinement(results, cfg):
    coarse, fine = results
    record = {"coarse_requested_max_step": 2**-7, "fine_requested_max_step": 2**-10,
              "coarse_actual_step_on_common_grid": cfg.common_step,
              "interpretation": "deterministic Gaussian-ODE integration audit; linearization/stopping bias separate",
              "banks": {}}
    for bank in _banks(cfg):
        c, f = coarse["predictions"][bank.name], fine["predictions"][bank.name]
        cm, fm = np.asarray(c["mean"]), np.asarray(f["mean"])
        cc, fc = np.asarray(c["scaled_covariance"]), np.asarray(f["scaled_covariance"])
        std = np.sqrt(np.diagonal(fc, axis1=-2, axis2=-1))
        record["banks"][bank.name] = {
            "mean_max_in_fine_Gaussian_std": float(np.max(np.sqrt(cfg.n_particles)*abs(cm-fm)/std)),
            "covariance_max_normalized": float(np.max(abs(cc-fc)/(std[..., :, None]*std[..., None, :])))}
    return record


def run(root, *, groups=None, deadline=None):
    root = Path(root).resolve()
    cfg = ReferenceConfig()
    output, design = prepare(root, cfg)
    inputs, fit_lock = _input_identity(root, cfg)
    learner_families = _training_families(fit_lock, design["historical_input_banks"])
    _immutable(output/"input_lock.json", inputs)
    selected = cfg.group_keys if groups is None else tuple(groups)
    if not selected or not set(selected) <= set(cfg.group_keys) or len(set(selected)) != len(selected):
        raise ValueError("Select distinct declared reference groups")
    # This validates every committed reference chunk, source/config, and family
    # grouping once, before optimized direct per-group reads below.
    checked = load_reference(root/"reference", cfg, group_key=selected[0])
    reference_path = root/"reference"/"reference.h5"
    completed_groups = []
    for index, key in enumerate(selected):
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("Broad-task evaluation is resumable by group and method")
        actual_output, actual_design = prepare(root, cfg)
        if actual_output != output or actual_design != design:
            raise ValueError("Frozen evaluator/helper identities changed")
        data = checked if index == 0 else _read_group(reference_path, key)
        if index == 0:
            checked = None
        reference_families = {family.decode() if isinstance(family, bytes) else str(family)
                              for family in data["family_ids"]}
        if reference_families & learner_families:
            raise ValueError("Final reference families overlap a learner training/validation family")
        directory = output/key
        directory.mkdir(parents=True, exist_ok=True)
        input_arrays = {name: array_sha(data[name]) for name in (
            "initial", "family_ids", "model_increments", "common_paths", "bank_b", "bank_c",
            "many_observables", "workloads", "stopped_at")}
        counts = bootstrap_counts(len(data["initial"]), _seed(key), N_BOOTSTRAP)
        many = _many_reference(data["many_observables"], counts)
        base = {"kind": KIND, "source_hash": source_hash(), "script_sha256": design["script_sha256"],
                "helper_hashes": design["helpers"], "input_lock": inputs, "group": key,
                "array_hashes": input_arrays, "bootstrap_counts_sha256": array_sha(counts)}
        bootstrap_artifact = _group_bootstrap(directory, data, counts, many)
        records = []
        print(f"[group {key}] reference audit", flush=True)
        audit = reference_audit(directory, cfg, data, {**base, "method": "reference-refinement"})
        case = key.split("__")[0]
        for method in design["candidate_plan"]:
            if method["case"] != case:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Broad-task evaluator checkpointed before its next method")
            print(f"[group {key}] {method['name']}", flush=True)
            records.append(evaluate_path(directory, method, cfg, data, counts, many,
                                         {**base, "method": method}, deadline))
        for item in fit_lock["models"]:
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Broad-task evaluator checkpointed before its next statistics head")
            print(f"[group {key}] statistics {item['bank']} {item['init']} seed{item['seed']}", flush=True)
            records.append(evaluate_statistics(directory, item, cfg, data, counts,
                                               {**base, "method": item}))
        gaussian = []
        for step in (2**-7, 2**-10):
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Broad-task evaluator checkpointed before Gaussian baseline")
            print(f"[group {key}] linear-Gaussian RK4 maxstep={step:g}", flush=True)
            value = evaluate_lga(directory, cfg, data, counts,
                                 {**base, "method": {"kind": "linear-Gaussian", "step": step}}, step)
            gaussian.append(value)
            records.append(value)
        group_record = {"identity": base, "complete": True,
                        "reference_bootstrap": bootstrap_artifact,
                        "reference_audit_path": str((directory/"reference_refinement.json").resolve()),
                        "reference_audit_sha256": _sha(directory/"reference_refinement.json"),
                        "reference_audit_complete": audit["complete"],
                        "methods": [{"name": record["method"]["name"],
                                     "method": record["method"],
                                     "json_path": str((directory/record["method"]["name"]).with_suffix(".json").resolve())}
                                    for record in records],
                        "gaussian_ODE_refinement": _lga_refinement(gaussian, cfg),
                        "optimizer_seeds_pooled": False, "continuous_reference_bias": "unresolved"}
        _json(directory/"group_results.json", group_record)
        completed_groups.append(key)
        del data, many, counts, records, gaussian
    # Final input SHA verification detects mutation during a long evaluation.
    if _sha(reference_path) != inputs["reference_sha256"] or _sha(root/"payoffs"/"fit_lock.json") != inputs["fit_lock_sha256"]:
        raise ValueError("A frozen input mutated during broad-task evaluation")
    all_completed = []
    for key in cfg.group_keys:
        path = output/key/"group_results.json"
        if path.exists():
            group = _read(path)
            if (group.get("complete") and group["identity"].get("input_lock") == inputs
                    and group["identity"].get("source_hash") == source_hash()
                    and group["identity"].get("script_sha256") == design["script_sha256"]):
                all_completed.append(key)
    index = {"kind": KIND, "source_hash": source_hash(), "complete": set(all_completed) == set(cfg.group_keys),
             "groups_this_invocation": completed_groups, "declared_groups": list(cfg.group_keys),
             "completed_groups": all_completed,
             "preregistration_sha256": _sha(output/"preregistration.json"), "input_lock": inputs,
             "all_stopped_paths_retained": True, "continuous_reference_bias": "unresolved"}
    _json(output/"evaluation_index.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"runs"/"path-generalization-2026-10-26")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args(argv)
    if not np.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 8*60*60:
        parser.error("Budget must be positive and at most eight hours")
    torch.set_num_threads(1)
    try:
        if args.prepare_only:
            output, design = prepare(args.root)
            print(json.dumps({"prepared": str(output), "groups": design["groups"]}), flush=True)
        else:
            result = run(args.root, groups=args.groups, deadline=time.monotonic()+args.budget_seconds)
            print(json.dumps(result), flush=True)
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
