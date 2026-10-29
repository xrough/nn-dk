"""Locked harder-regime moment accuracy and literal training/inference costs.

Every fitted seed is retained.  Native Monte Carlo uses new forcing and new
ensemble initials, independently of the locked reference.  The neural and LGA
algorithms timed below are exactly those producing the accuracy predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.benchmark_path_generalization import timing  # noqa: E402
from analysis.benchmark_payoff_generalization import lga_marginals, predict_chunks  # noqa: E402
from analysis.compiled_native import compile_native  # noqa: E402
from analysis.evaluate_path_generalization import (  # noqa: E402
    bootstrap_counts,
    empirical,
    mixture_moments,
    moment_comparison,
)
from analysis.native_gaussian_baseline import iid_scaled_covariance  # noqa: E402
from analysis.run_payoff_repair import _json, _read, _sha  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "harder-regime-independent-moment-accuracy-cost-v1"
SEED = 2026102819


def _runner():
    from analysis import run_payoff_regimes

    return run_payoff_regimes


def _immutable(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != value:
            raise ValueError("The preregistered accuracy/cost identity changed")
    else:
        _json(path, value)
    return value


def prepare(root):
    """Freeze comparison settings before opening any final-reference outcomes."""
    root = Path(root).resolve()
    runner = _runner()
    study = runner.load_plan(root)
    helpers = [Path(__file__), REPO / "analysis/run_payoff_regimes.py",
               REPO / "analysis/benchmark_payoff_generalization.py",
               REPO / "analysis/evaluate_path_generalization.py",
               REPO / "analysis/native_gaussian_baseline.py",
               REPO / "analysis/compiled_native.py", REPO / "analysis/native_euler_control.cpp"]
    cases = runner.case_names(study)
    plan = {"kind": KIND, "source_hash": source_hash(), "seed": SEED,
            "helpers": {str(p): _sha(p) for p in helpers},
            "cases": list(cases), "configs": {c: asdict(runner.reference_configuration(c, study)) for c in cases},
            "banks": {c: runner.payoff_bank(c, study).to_dict() for c in cases},
            "all_optimizer_seeds": list(study.optimizer_seeds), "untrained_repaired_control": True,
            "LGA_max_steps": [2**-7, 2**-10], "primary_LGA_max_step": 2**-10,
            "native_MC_step": 2**-10, "native_MC_chunk": 32,
            "native_MC_independent_initials_and_forcing": True,
            "bootstrap": {"count": 128, "seed": SEED, "unit": "whole-trajectory-family"},
            "neural_conditional_batches": [1, 64, 256], "conditional_repeats": 21,
            "conditional_warmup": 3, "ensemble_repeats": 5, "ensemble_warmup": 1,
            "native_MC_repeats": 3, "native_MC_warmup": 1,
            "known_cost_query_counts": [1, 100, 10000, 100000],
            "accuracy_target": .05, "uncertainty_component_target": .0125,
            "selection_using_test": False, "post_test_refitting": False,
            "advantage": "bootstrap contrast of maximum absolute normalized moment errors; every seed reported",
            "cost_qualification": "same statistic, accuracy and uncertainty gates; no timing-only crossover claim",
            "scope": "four means and same-time 4x4 covariance at three horizons; no sampled-law prediction",
            "stopping": "all stopped native paths included; LGA is unstopped, NN stopping probability unsupported",
            "timing_after_all_fitting_reference_accuracy_workers_finish": True}
    path = root / "comparison/preregistration.json"
    if not path.exists() and any((root / "reference" / c / "reference.h5").exists() for c in cases):
        raise ValueError("Comparison must be preregistered before final reference outcomes")
    return _immutable(path, plan)


def _verified(root):
    root = Path(root).resolve()
    runner = _runner()
    study = runner.load_plan(root)
    plan = prepare(root)
    lock = runner.verify_fit_lock(root)
    if not isinstance(lock, dict) or not lock.get("complete"):
        raise ValueError("A complete all-regime fit lock is required")
    expected = len(plan["all_optimizer_seeds"]) * len(plan["cases"])
    if len(lock.get("models", [])) != expected:
        raise ValueError("Every declared optimizer seed is required before evaluation")
    return root, runner, study, plan, lock


def model_entries(lock, regime, seeds=(0, 1, 2)):
    models = [{**m, "family": "repaired"} for m in lock["models"] if m["case"] == regime]
    if len(models) != len(seeds) or {m["seed"] for m in models} != set(seeds):
        raise ValueError("All three independently fitted seeds (or every smoke-plan seed) must be retained")
    return models + [{"name": regime + "_untrained", "case": regime, "seed": 0,
                      "family": "untrained-repaired", "untrained": True}]


def _model(item, runner, study):
    if not item.get("untrained"):
        return runner.load_declared_model(item)
    regime = item["case"]
    cfg = runner.reference_configuration(regime, study)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        repair = runner.repaired_configuration(regime, study)
        model = RepairPayoffModel(cfg.spec, cfg.context("uniform"), cfg.horizon,
                                  runner.payoff_bank(regime, study), width=repair.study.width,
                                  quadrature_nodes=repair.quadrature_nodes)
    return model.eval()


def _prediction(model, initial, times):
    return predict_chunks(model, initial, times, chunk_size=512)


def _checked_lga(*args):
    mean, covariance, diagnostics = lga_marginals(*args)
    symmetry = float(np.max(abs(covariance-np.swapaxes(covariance, -1, -2))))
    eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if (not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or symmetry > 1e-9 or eigenvalue < -1e-9):
        raise ValueError("Invalid Gaussian moments; no covariance repair permitted")
    return mean, covariance, {**diagnostics, "minimum_covariance_eigenvalue": eigenvalue,
                              "covariance_symmetry_error": symmetry}


def _save_result(path, identity, metrics, arrays, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    npz = path.with_suffix(".npz")
    temporary = npz.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(npz)
    result = {"complete": True, "identity": identity, "metrics": metrics,
              "arrays": {"path": str(npz), "sha256": _sha(npz)}, **extra}
    _json(path, result)
    return result


def _replay(path, identity):
    if not path.exists():
        return None
    packet = _read(path)
    if (not packet.get("complete") or packet.get("identity") != identity
            or _sha(packet["arrays"]["path"]) != packet["arrays"]["sha256"]):
        raise ValueError("A completed accuracy record or its numeric evidence changed")
    return packet


def standalone_sampling(mean, covariance, bootstrap_mean, bootstrap_covariance):
    """Candidate-only uncertainty using physical (not N-scaled) covariances."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    bm, bc = np.asarray(bootstrap_mean), np.asarray(bootstrap_covariance)
    std = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    if np.any(std <= 0):
        raise ValueError("Positive candidate fluctuation scales required")
    scale = std[..., :, None] * std[..., None, :]
    return {"mean95_in_candidate_std": np.quantile(
                np.max(abs(bm-mean) / std, axis=-1), .95, axis=0).tolist(),
            "covariance95_normalized": np.quantile(
                np.max(abs(bc-covariance) / scale, axis=(-2, -1)), .95, axis=0).tolist(),
            "scope": "candidate-only whole-family sampling; not paired bias"}


def bootstrap_advantage(reference, neural_mean, neural_covariance, gaussian_mean,
                        gaussian_covariance, counts, n_particles,
                        neural_bootstrap_mean=None, neural_bootstrap_covariance=None):
    """Positive contrast means smaller NN error, with shared-reference uncertainty.

    Scales are the fixed empirical reference standard deviations.  The interval
    compares this fitted network with one fixed Gaussian algorithm; it does not
    pool optimizer seeds or make an across-regime simultaneous guarantee.
    """
    rm, rc = empirical(reference)
    bm, bc = empirical(reference, counts)
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scale = std[..., :, None] * std[..., None, :]
    if np.any(std <= 1e-14):
        raise ValueError("Nondegenerate reference fluctuation scales required")
    nm, nc = np.asarray(neural_mean), np.asarray(neural_covariance) / n_particles
    gm, gc = np.asarray(gaussian_mean), np.asarray(gaussian_covariance) / n_particles
    if (neural_bootstrap_mean is None) != (neural_bootstrap_covariance is None):
        raise ValueError("Both neural integration bootstrap arrays are required")
    bnm = nm[None] if neural_bootstrap_mean is None else np.asarray(neural_bootstrap_mean)
    bnc = nc[None] if neural_bootstrap_covariance is None else np.asarray(neural_bootstrap_covariance)/n_particles
    result = []
    for i in range(len(rm)):
        mean_point = float(np.max(abs(gm[i]-rm[i])/std[i]) - np.max(abs(nm[i]-rm[i])/std[i]))
        cov_point = float(np.max(abs(gc[i]-rc[i])/scale[i]) - np.max(abs(nc[i]-rc[i])/scale[i]))
        md = np.max(abs(gm[i]-bm[:, i])/std[i], axis=-1) - np.max(abs(bnm[:, i]-bm[:, i])/std[i], axis=-1)
        cd = np.max(abs(gc[i]-bc[:, i])/scale[i], axis=(-2, -1)) - np.max(
            abs(bnc[:, i]-bc[:, i])/scale[i], axis=(-2, -1))
        result.append({"time_index": i, "mean_error_advantage": mean_point,
                       "covariance_error_advantage": cov_point,
                       "bootstrap_mean_advantages": md.tolist(),
                       "bootstrap_covariance_advantages": cd.tolist(),
                       "mean_advantage_percentile95": np.quantile(md, [.025, .975]).tolist(),
                       "covariance_advantage_percentile95": np.quantile(cd, [.025, .975]).tolist(),
                       "mean_advantage_resolved": bool(np.quantile(md, .025) > 0),
                       "covariance_advantage_resolved": bool(np.quantile(cd, .025) > 0)})
    return {"records": result, "positive_means_NN_smaller_error": True,
            "interval_scope": "pointwise per fitted seed/horizon; no family-wise regime or reference-bias certificate"}


def near_boundary(path_minimum_mass, spec):
    """All-path minimum diagnostics, including each absorbed trajectory."""
    values = np.asarray(path_minimum_mass, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.min() < -1e-12:
        raise ValueError("Finite nonnegative minimum for every path required")
    return {"minimum_mass": float(values.min()),
            "minimum_mass_quantiles_1_5_50_percent": np.quantile(values, [.01, .05, .5]).tolist(),
            "mass_stop": spec.mass_stop,
            "fractions_at_most_stop_multiple": {str(k): float(np.mean(values <= k*spec.mass_stop*(1+1e-12)))
                                                for k in (1, 2, 5, 10)},
            "unit": "whole-path minimum, including stopped paths"}


def _native_mc(runner, cfg, bank, profile, initial, probabilities, count, seed,
               backend, *, step=2**-10, chunk=32):
    """Literal MC workload: fresh initials/noise, complete stopped solve, moments."""
    rng = seeded_rng(seed, KIND, "candidate-native-MC")
    outcomes, stops, path_minima = [], [], []
    minimum, mass_error = 1., 0.
    for start in range(0, count, chunk):
        size = min(chunk, count-start)
        origin = (np.broadcast_to(initial, (size, cfg.n_cells)).copy() if initial is not None else
                  rng.multinomial(cfg.n_particles, probabilities, size=size) / cfg.n_particles)
        dw = rng.normal(scale=np.sqrt(step), size=(size, round(cfg.horizon/step), cfg.n_cells))
        result = runner.native_payoff(origin, dw, cfg.context(profile), cfg.spec, step,
                                      bank, cfg.query_times, backend=backend)
        values = np.asarray(result["outcomes"])
        stopped = np.asarray(result["stopped_at"])
        if (values.shape != (size, 3, 4) or stopped.shape != (size,)
                or not np.isfinite(values).all() or np.any(np.isnan(stopped))
                or np.any(stopped < 0) or np.any(stopped[np.isfinite(stopped)] > cfg.horizon+1e-12)
                or not np.isfinite([result["minimum_mass"], result["mass_error"]]).all()
                or result["minimum_mass"] < -1e-12 or result["mass_error"] > 1e-10):
            raise ValueError("Invalid native path blocks every estimate; no survivor filtering")
        outcomes.append(values)
        stops.append(stopped)
        if "path_minimum_mass" in result:
            path_minima.append(np.asarray(result["path_minimum_mass"]))
        minimum = min(minimum, result["minimum_mass"])
        mass_error = max(mass_error, result["mass_error"])
    y, stopped = np.concatenate(outcomes), np.concatenate(stops)
    return y, {"minimum_mass": float(minimum), "mass_error": float(mass_error),
               "stopped_paths": int(np.isfinite(stopped).sum()), "paths": count,
               "stopped_at": stopped,
               "path_minimum_mass": np.concatenate(path_minima) if path_minima else np.empty(0)}


def _reference_manifest(root, regime):
    path = root / "reference" / regime / "manifest.json"
    value = _read(path)
    if value.get("status") != "complete":
        raise ValueError("Complete immutable native reference required")
    h5 = path.with_name("reference.h5")
    if value.get("reference_sha256") != _sha(h5):
        raise ValueError("Native reference evidence changed")
    return value


def evaluate(root, cases=None, groups=None):
    root, runner, study, plan, lock = _verified(root)
    torch.set_num_threads(1)
    backend = compile_native(root / "comparison/compiled_cache")
    selected = plan["cases"] if cases is None else cases
    if not set(selected) <= set(plan["cases"]):
        raise ValueError("Unknown preregistered regime")
    all_records = []
    for regime in selected:
        cfg = runner.reference_configuration(regime, study)
        bank = runner.payoff_bank(regime, study)
        manifest = _reference_manifest(root, regime)
        base = {"source_hash": source_hash(), "comparison_plan_sha256": _sha(root/"comparison/preregistration.json"),
                "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"), "regime": regime,
                "reference_sha256": manifest["reference_sha256"], "bank_hash": bank.bank_hash}
        for key in cfg.group_keys if groups is None else groups:
            if key not in cfg.group_keys:
                raise ValueError("Unknown reference group")
            data = runner.load_reference(root, regime, key)
            profile, role, _ = key.split("__")
            conditional = role == "conditional"
            reference = np.asarray(data["outcomes"])
            if len(reference) != cfg.paths_per_group:
                raise ValueError("Reference allocation changed")
            initials, probabilities = data["initial"], data["probabilities"]
            if conditional and not np.all(initials == initials[:1]):
                raise ValueError("Conditional group must use one supplied initial histogram")
            counts = bootstrap_counts(len(reference), int(stable_hash([SEED, regime, key])[:12], 16), 128)
            dest = root / "comparison/accuracy" / regime / key
            records = []
            for item in model_entries(lock, regime, plan["all_optimizer_seeds"]):
                identity = {**base, "group": key, "name": item["name"]}
                path = dest / (item["name"] + ".json")
                replay = _replay(path, identity)
                if replay is not None:
                    records.append(replay)
                    continue
                tick = time.perf_counter()
                model = _model(item, runner, study)
                mean, cov = _prediction(model, initials[:1] if conditional else initials, cfg.query_times)
                if conditional:
                    pm, pc, bm, bc = mean[0], cov[0], None, None
                else:
                    pm, pc = mixture_moments(mean, cov, n_particles=cfg.n_particles)
                    bm, bc = mixture_moments(mean, cov, n_particles=cfg.n_particles, counts=counts)
                metrics = moment_comparison(reference, predicted_mean=pm, predicted_scaled_covariance=pc,
                    bootstrap_predicted_mean=bm, bootstrap_predicted_scaled_covariance=bc,
                    n_particles=cfg.n_particles, counts=counts)
                arrays = {"mean": pm, "scaled_covariance": pc}
                if bm is not None:
                    arrays.update(bootstrap_mean=bm, bootstrap_scaled_covariance=bc)
                extra = {"method": item, "scientific_prediction_seconds": time.perf_counter()-tick,
                         "stopping_probability_prediction": "unsupported", "initial_integration_MC": not conditional}
                if bm is not None:
                    extra["standalone_initial_integration_sampling"] = standalone_sampling(
                        pm, pc/cfg.n_particles, bm, bc/cfg.n_particles)
                records.append(_save_result(path, identity, metrics, arrays, extra))
                print(f"accuracy {regime} {key} {item['name']}", flush=True)
            for step in plan["LGA_max_steps"]:
                name = f"LGA_dt{round(-np.log2(step))}"
                identity = {**base, "group": key, "name": name}
                path = dest / (name + ".json")
                replay = _replay(path, identity)
                if replay is not None:
                    records.append(replay)
                    continue
                origin = initials[0] if conditional else probabilities
                pm, pc, diagnostics = _checked_lga(origin, cfg.context(profile), cfg.spec,
                    cfg.query_times, step, bank, None if conditional else iid_scaled_covariance(origin))
                metrics = moment_comparison(reference, predicted_mean=pm, predicted_scaled_covariance=pc,
                                             n_particles=cfg.n_particles, counts=counts)
                records.append(_save_result(path, identity, metrics, {"mean": pm, "scaled_covariance": pc},
                    {"method": {"name": name, "family": "LGA", "step": step},
                     "law_diagnostics": diagnostics, "stopping_probability_prediction": "unsupported",
                     "law": "unstopped-linear-noise-approximation"}))
            name = "native-MC-dt10"
            identity = {**base, "group": key, "name": name}
            path = dest / (name + ".json")
            replay = _replay(path, identity)
            if replay is None:
                candidate_seed = int(stable_hash([SEED, regime, key, "independent-candidate"])[:12], 16)
                y, diagnostics = _native_mc(runner, cfg, bank, profile,
                    initials[0] if conditional else None, probabilities, cfg.paths_per_group,
                    candidate_seed, backend)
                metrics = moment_comparison(reference, y, paired=False, n_particles=cfg.n_particles,
                                             counts=counts, seed=SEED+1)
                pm, covariance = empirical(y)
                candidate_counts = bootstrap_counts(len(y), SEED+2, 128)
                bm, bc = empirical(y, candidate_counts)
                stopped = diagnostics.pop("stopped_at")
                minima = diagnostics.pop("path_minimum_mass")
                if len(minima):
                    diagnostics["near_boundary"] = near_boundary(minima, cfg.spec)
                replay = _save_result(path, identity, metrics, {"outcomes": y, "stopped_at": stopped,
                    "path_minimum_mass": minima,
                    "mean": pm, "scaled_covariance": cfg.n_particles*covariance,
                    "bootstrap_mean": bm, "bootstrap_scaled_covariance": cfg.n_particles*bc,
                    "family_ids": np.array([stable_hash([SEED, regime, key, "independent-candidate", i])
                                            for i in range(len(y))], dtype="S64")},
                    {"method": {"name": name, "family": "native-MC", "step": 2**-10},
                     "diagnostics": diagnostics, "independent_reference_and_candidate": True,
                     "rng": {"seed": candidate_seed, "stream": [KIND, "candidate-native-MC"],
                             "chunk": plan["native_MC_chunk"], "forcing": "independent-PCG64-normal"},
                     "standalone_MC_sampling": standalone_sampling(pm, covariance, bm, bc)})
            records.append(replay)
            gaussian = next(r for r in records if r["method"].get("name") == "LGA_dt10")
            with np.load(gaussian["arrays"]["path"]) as arrays:
                gm, gc = arrays["mean"], arrays["scaled_covariance"]
            contrasts = {}
            for record in records:
                if record["method"].get("family") != "repaired":
                    continue
                with np.load(record["arrays"]["path"]) as a:
                    contrasts[record["identity"]["name"]] = bootstrap_advantage(reference,
                        a["mean"], a["scaled_covariance"], gm, gc, counts, cfg.n_particles,
                        a["bootstrap_mean"] if "bootstrap_mean" in a else None,
                        a["bootstrap_scaled_covariance"] if "bootstrap_scaled_covariance" in a else None)
            refinement = {}
            audit = np.asarray(data["audit_outcomes"])
            for level in range(audit.shape[1]-1):
                refinement[f"level{level}_versus_finest"] = moment_comparison(audit[:, -1], audit[:, level],
                    n_particles=cfg.n_particles, paired=True, seed=SEED+101)
            stopped = np.asarray(data["stopped_at"])
            _json(dest/"group.json", {"complete": True, "identity": base, "group": key,
                "records": [{"path": str(dest/(r["identity"]["name"]+".json")),
                             "sha256": _sha(dest/(r["identity"]["name"]+".json"))} for r in records],
                "NN_versus_fine_LGA": contrasts, "reference_refinement": refinement,
                "stopped_reference_paths": int(np.isfinite(stopped).sum()),
                "reference_paths": len(stopped), "reference_stopping_probability": float(np.isfinite(stopped).mean()),
                "near_boundary_diagnostics": near_boundary(data["path_minimum_mass"], cfg.spec)})
            all_records.extend(records)
    expected = [(c, k) for c in plan["cases"] for k in runner.reference_configuration(c, study).group_keys]
    complete = [(c, k) for c, k in expected if (root/"comparison/accuracy"/c/k/"group.json").exists()]
    _json(root/"comparison/accuracy_index.json", {"complete": len(complete) == len(expected),
        "groups": complete, "expected_groups": expected, "records_per_group": len(plan["all_optimizer_seeds"])+4,
        "source_hash": source_hash(), "plan_sha256": _sha(root/"comparison/preregistration.json")})
    return all_records


def benchmark(root, *, final_timing=False):
    if not final_timing:
        raise ValueError("Final timing requires all competing CPU workers finished")
    root, runner, study, plan, lock = _verified(root)
    if not _read(root/"comparison/accuracy_index.json").get("complete"):
        raise ValueError("Complete accuracy comparison required before final timings")
    torch.set_num_threads(1)
    identity = {"plan_sha256": _sha(root/"comparison/preregistration.json"),
                "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"), "source_hash": source_hash()}
    output = root/"comparison/timing.json"
    previous = _read(output) if output.exists() else None
    if previous and previous["identity"] != identity:
        raise ValueError("Timing identity changed")
    if previous and previous.get("complete"):
        return previous
    packet = previous or {"identity": identity, "complete": False, "records": [], "loading": [],
        "environment": {"platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
                        "cpu_threads": 1}, "final_timing_after_all_workers_finished": True}
    done = {r["slot"] for r in packet["records"]}
    tick = time.perf_counter()
    backend = compile_native(root/"comparison/compiled_cache")
    packet["compiled_control"] = {"setup_seconds_this_process": time.perf_counter()-tick, **backend.metadata}

    def save(slot, call, repeats, warmup, extra):
        if slot in done:
            return
        tick = time.perf_counter()
        pm, pc = call()
        first = time.perf_counter()-tick
        if (not np.isfinite(pm).all() or not np.isfinite(pc).all()
                or np.linalg.eigvalsh(pc).min() < -1e-9):
            raise ValueError("Invalid literal cost prediction")
        packet["records"].append({"slot": slot, **extra, **timing(call, repeats, warmup),
                                  "first_prediction_seconds": first})
        done.add(slot)
        _json(output, packet)
        print(f"timing {slot}", flush=True)

    for regime in plan["cases"]:
        cfg = runner.reference_configuration(regime, study)
        bank = runner.payoff_bank(regime, study)
        for profile in cfg.cases:
            data = runner.load_reference(root, regime, f"{profile}__ensemble__0")
            probabilities = data["probabilities"]
            rng = seeded_rng(SEED, KIND, "timing-only", regime, profile)
            origins = rng.multinomial(cfg.n_particles, probabilities, size=256)/cfg.n_particles
            for item in model_entries(lock, regime, plan["all_optimizer_seeds"]):
                tick = time.perf_counter()
                model = _model(item, runner, study)
                packet["loading"].append({"regime": regime, "profile": profile, "name": item["name"],
                                           "seconds": time.perf_counter()-tick})
                common = {"regime": regime, "profile": profile, "name": item["name"],
                          "family": item["family"], "seed": item["seed"], "bank": bank.name}
                for batch in plan["neural_conditional_batches"]:
                    save(f"{regime}/{profile}/{item['name']}/conditional/{batch}",
                        lambda m=model, b=batch: _prediction(m, origins[:b], cfg.query_times),
                        plan["conditional_repeats"], plan["conditional_warmup"],
                        {**common, "role": "conditional", "batch": batch,
                         "cache_state": "B1 first query cold; larger batches follow warmed horizon coefficients",
                         "scope": "three-horizon-four-mean-and-same-time-4x4-covariance-per-initial"})

                def ensemble(m=model):
                    initial = rng.multinomial(cfg.n_particles, probabilities, size=cfg.paths_per_group)/cfg.n_particles
                    mu, cov = _prediction(m, initial, cfg.query_times)
                    return mixture_moments(mu, cov, n_particles=cfg.n_particles)

                save(f"{regime}/{profile}/{item['name']}/ensemble", ensemble,
                    plan["ensemble_repeats"], plan["ensemble_warmup"],
                    {**common, "role": "ensemble", "batch": cfg.paths_per_group,
                     "cache_state": "warm after conditional calls; charge B1 first-query probe for known cold total",
                     "scope": "fresh-initial-histograms-inference-total-covariance"})
            for step in plan["LGA_max_steps"]:
                for role in ("conditional", "ensemble"):
                    origin = origins[0] if role == "conditional" else probabilities
                    cov = None if role == "conditional" else iid_scaled_covariance(probabilities)
                    save(f"{regime}/{profile}/LGA/{step}/{role}",
                        lambda o=origin, c=cov, dt=step: _checked_lga(o, cfg.context(profile), cfg.spec,
                            cfg.query_times, dt, bank, c)[:2],
                        plan["conditional_repeats"], plan["conditional_warmup"],
                        {"regime": regime, "profile": profile, "name": f"LGA_dt{round(-np.log2(step))}",
                         "family": "LGA", "role": role, "batch": 1, "step": step,
                         "scope": "literal-three-query-marginals-analytic-IID-covariance-for-ensemble"})
            for role in ("conditional", "ensemble"):
                def native(ro=role):
                    y, _ = _native_mc(runner, cfg, bank, profile, origins[0] if ro == "conditional" else None,
                        probabilities, cfg.paths_per_group, int(rng.integers(1, 2**62)), backend)
                    pm, covariance = empirical(y)
                    return pm, cfg.n_particles*covariance

                save(f"{regime}/{profile}/native-MC/{role}", native,
                    plan["native_MC_repeats"], plan["native_MC_warmup"],
                    {"regime": regime, "profile": profile, "name": "native-MC-dt10", "family": "native-MC",
                     "role": role, "batch": cfg.paths_per_group, "step": plan["native_MC_step"],
                     "scope": "fresh-initials-and-forcing-stopped-native-solve-requested-payoff-moments"})
    packet["complete"] = True
    _json(output, packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "benchmark"))
    parser.add_argument("--root", type=Path, default=REPO/"runs/payoff-regimes-2026-10-28")
    parser.add_argument("--case", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--final-timing", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "evaluate":
        evaluate(args.root, args.case, args.group)
    else:
        benchmark(args.root, final_timing=args.final_timing)


if __name__ == "__main__":
    main()
