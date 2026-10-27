"""Fresh, locked accuracy and literal cost comparison of repaired payoff heads.

No model is selected using this file's independent reference outcomes. Historical
models retain their original source-strict loader in an isolated Git snapshot.
Only the supported same-time four-output means/covariances are compared here.
"""
from __future__ import annotations

import argparse
import importlib
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

import h5py  # noqa: E402
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
from analysis.path_generalization_reference import profile_probabilities  # noqa: E402
from analysis.run_payoff_repair import (  # noqa: E402
    _json,
    _read,
    _sha,
    configuration,
    load_declared_model,
    reference_configuration,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.frozen_models import frozen_package  # noqa: E402
from nn_dk.path_model import CausalPathOperator  # noqa: E402
from nn_dk.path_runtime import PreparedPathRuntime  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

OLD = REPO / "runs/path-generalization-2026-10-26"
KIND = "independent-repaired-payoff-accuracy-cost-v1"
SEED = 2026102714


def banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Immutable repair-study identity changed: {path}")
    else:
        _json(path, value)


def prepare(root):
    root = Path(root).resolve()
    cfg = reference_configuration()
    old = _read(OLD / "payoffs/fit_lock.json")
    candidates = _read(OLD / "evaluation/preregistration.json")["candidate_plan"]
    paths = [item for item in candidates
             if item["method"] == "prepared-learned-path" and item["dt"] == cfg.model_step]
    if len(old["models"]) != 12 or len(paths) != 6:
        raise ValueError("Keep every historical head and all six finest path checkpoints")
    helpers = [Path(__file__), REPO / "analysis/run_payoff_repair.py",
               REPO / "analysis/benchmark_payoff_generalization.py",
               REPO / "analysis/path_generalization_reference.py",
               REPO / "analysis/compiled_native.py", REPO / "analysis/native_euler_control.cpp",
               REPO / "analysis/evaluate_path_generalization.py"]
    plan = {"kind": KIND, "source_hash": source_hash(), "seed": SEED,
            "helpers": {str(p): _sha(p) for p in helpers},
            "reference_config": asdict(cfg), "reference_config_hash": cfg.config_hash,
            "banks": [b.to_dict() for b in banks(cfg)],
            "old_fit_lock": {"path": str(OLD / "payoffs/fit_lock.json"),
                             "sha256": _sha(OLD / "payoffs/fit_lock.json")},
            "old_heads": old["models"], "old_paths": paths,
            "historical_package_commit": "9d31018",
            "reference_seed_distinct_from_previous_study": True,
            "bootstrap": {"count": 128, "seed": SEED, "unit": "whole-trajectory-family"},
            "controls": ["untrained-repaired-analytic-baseline", "all-12-old-heads",
                         "three-fine-path-seeds", "native-Euler", "exact-kernel-Picard-depth3",
                         "literal-three-query-LGA-2^-7-and-2^-10"],
            "neural_conditional_timing_batches": [1, 64, 256],
            "neural_conditional_repeats": 21, "neural_conditional_warmup": 3,
            "LGA_conditional_timing_batch": 1, "LGA_repeats": 21,
            "ensemble_initials_and_MC_paths": cfg.paths_per_group,
            "ensemble_repeats": 5, "ensemble_warmup": 1,
            "path_MC_repeats": 3, "path_MC_warmup": 1,
            "path_timing_seed": 0, "path_chunk_size": 64,
            "cold_cost": "constructor/restoration and first prediction recorded separately from warm calls",
            "selection_using_test": False, "post_test_refitting": False,
            "scope": "same-time-4x4-only; not full joint-time moments or a continuum claim",
            "reference_bias": "coupled finite-step differences, continuous hitting bias unresolved",
            "timing_after_all_fitting_and_evaluation_workers_finish": True}
    # Round-trip tuples before immutable comparisons.
    plan = json.loads(json.dumps(plan))
    dest = root / "comparison/preregistration.json"
    if not dest.exists() and (root / "reference/reference.h5").exists():
        raise ValueError("Accuracy/cost comparison must be frozen before test outcomes")
    _immutable(dest, plan)
    return plan


def _verified(root):
    root = Path(root).resolve()
    plan = prepare(root)
    lock = _read(root / "payoffs/fit_lock.json")
    if not lock.get("complete") or len(lock["models"]) != 10:
        raise ValueError("All ten declared fits must be complete")
    for item in lock["models"]:
        for key in ("checkpoint", "metadata"):
            if _sha(item[f"{key}_path"]) != item[f"{key}_sha256"]:
                raise ValueError("A locked fitted model changed")
    for item in plan["old_heads"] + plan["old_paths"]:
        if (_sha(item["checkpoint_path"]) != item["checkpoint_sha256"]
                or _sha(Path(item["checkpoint_path"]).with_suffix(".json")) != item["metadata_sha256"]):
            raise ValueError("A historical comparison checkpoint or cost sidecar changed")
    return root, plan, lock, reference_configuration()


def _load_old(item):
    alias, _, _, _, _ = frozen_package("9d31018")
    return importlib.import_module(f"{alias}.payoff_learning").load_payoff_model(item["checkpoint_path"])


def _model_entries(lock, plan):
    result = []
    for item in lock["models"]:
        result.append({**item, "name": item["name"], "family": item["variant"], "old": False})
    for item in plan["old_heads"]:
        result.append({**item, "name": f"old_{item['bank']}_{item['init']}_seed{item['seed']}",
                       "family": "old-" + item["init"], "old": True,
                       "metadata_path": str(Path(item["checkpoint_path"]).with_suffix(".json"))})
    for bank in plan["banks"]:
        result.append({"name": f"untrained_{bank['name']}", "bank": bank["name"],
                       "family": "untrained-repaired", "seed": 0, "untrained": True})
    return result


def _model(item, cfg):
    if item.get("untrained"):
        from nn_dk.payoff_repair import RepairPayoffModel
        bank = next(b for b in banks(cfg) if b.name == item["bank"])
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = RepairPayoffModel(cfg.spec, cfg.context("uniform"), cfg.horizon, bank,
                                      width=configuration().width)
        return model.eval()
    return _load_old(item) if item.get("old") else load_declared_model(item)


def _prediction(model, initial, times):
    return predict_chunks(model, initial, times, chunk_size=512)


def _path_runtime(item, cfg, case):
    if item["family"] == "native-Euler":
        return None
    if item["family"] == "learned-path":
        runtime = PreparedPathRuntime.from_checkpoint(item["checkpoint_path"], "26ac405")
    else:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            m = CausalPathOperator(cfg.spec, "dk", width=2, depth=3)
        runtime = PreparedPathRuntime(m, cfg.context(case), cfg.model_step, cfg.horizon,
                                      learned=False, depth=3)
    runtime.prepare(round(cfg.horizon / cfg.model_step))
    return runtime


def _path_values(runtime, compiled, initial, noise, cfg, case, bank_filter=None):
    selected_banks = [b for b in banks(cfg) if bank_filter is None or b.name == bank_filter]
    if not selected_banks:
        raise ValueError("Unknown physical payoff bank")
    outputs = {b.name: [] for b in selected_banks}
    stops, mass_error, minimum = [], 0., 1.
    times = np.arange(noise.shape[1] + 1) * cfg.model_step
    indices = np.rint(np.asarray(cfg.query_times) / cfg.model_step).astype(int)
    for start in range(0, len(initial), 64):
        args = (initial[start:start + 64], noise[start:start + 64])
        with torch.no_grad():
            packet = (compiled(*args, cfg.context(case), cfg.spec, cfg.model_step) if runtime is None
                      else runtime(*args, return_raw=True)[0])
        states, stop = packet["states"], packet["stopped_at"]
        if isinstance(states, torch.Tensor):
            states, stop = states.numpy(), stop.numpy()
        if not np.isfinite(states).all() or states.min() < -1e-12:
            raise ValueError("Invalid path prohibits survivor-conditioned estimates")
        mass_error = max(mass_error, float(abs(states.sum(-1) - 1).max()))
        minimum = min(minimum, float(states.min()))
        for b in selected_banks:
            outputs[b.name].append(b.evaluate_paths(states, times, stop)[:, indices])
        stops.append(stop)
    return {k: np.concatenate(v) for k, v in outputs.items()}, {
        "mass_error": mass_error, "minimum_mass": minimum,
        "stopped_paths": int(np.isfinite(np.concatenate(stops)).sum())}


def _save_result(path, identity, metrics, arrays, extra):
    npz = path.with_suffix(".npz")
    with npz.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    npz.with_suffix(".tmp").replace(npz)
    result = {"complete": True, "identity": identity, "metrics": metrics,
              "arrays": {"path": str(npz), "sha256": _sha(npz)}, **extra}
    _json(path, result)
    return result


def _replay(path, identity):
    if not path.exists():
        return None
    value = _read(path)
    if (not value.get("complete") or value["identity"] != identity
            or _sha(value["arrays"]["path"]) != value["arrays"]["sha256"]):
        raise ValueError("A completed independent comparison changed")
    return value


def evaluate(root, groups=None):
    root, plan, lock, cfg = _verified(root)
    torch.set_num_threads(1)
    manifest = _read(root / "reference/manifest.json")
    if manifest.get("status") != "complete" or _sha(root / "reference/reference.h5") != manifest["reference_sha256"]:
        raise ValueError("Complete immutable fresh reference required")
    family_sets = []
    with h5py.File(root / "reference/reference.h5", "r") as store:
        for key in cfg.group_keys:
            family_sets.append(set(store[key]["family_ids"][:].astype(str)))
    if any(len(s) != cfg.paths_per_group for s in family_sets) or any(
            a & b for i, a in enumerate(family_sets) for b in family_sets[i + 1:]):
        raise ValueError("Reference trajectory families overlap")
    learner_ids = set()
    for role in ("train", "validation"):
        with h5py.File(root / f"payoffs/interacting_{role}.h5", "r") as store:
            for case in cfg.cases:
                learner_ids.update(store[case]["family_ids"][:].astype(str))
    if any(s & learner_ids for s in family_sets):
        raise ValueError("Reference and learner trajectory families overlap")
    identity_base = {"source_hash": source_hash(), "plan_sha256": _sha(root / "comparison/preregistration.json"),
                     "fit_lock_sha256": _sha(root / "payoffs/fit_lock.json"),
                     "reference_sha256": manifest["reference_sha256"]}
    compiled = compile_native(root / "comparison/compiled_cache")
    all_records = []
    wanted = cfg.group_keys if groups is None else groups
    for key in wanted:
        case, role, _ = key.split("__")
        dest = root / "comparison/accuracy" / key
        dest.mkdir(parents=True, exist_ok=True)
        with h5py.File(root / "reference/reference.h5", "r") as store:
            g = store[key]
            data = {name: g[name][:] for name in ("initial", "model_increments", "bank_b", "bank_c",
                    "stopped_at", "probabilities", "audit_bank_b", "audit_bank_c")}
        conditional = role == "conditional"
        counts = bootstrap_counts(cfg.paths_per_group, int(stable_hash([SEED, key])[:12], 16), 128)
        references = dict(zip((b.name for b in banks(cfg)), (data["bank_b"], data["bank_c"]), strict=True))
        records = []
        for item in _model_entries(lock, plan):
            identity = {**identity_base, "group": key, "name": item["name"], "bank": item["bank"]}
            path = dest / (item["name"] + ".json")
            replay = _replay(path, identity)
            if replay is not None:
                records.append(replay)
                continue
            tick = time.perf_counter()
            model = _model(item, cfg)
            if asdict(model.spec) != asdict(cfg.spec) or model.horizon != cfg.horizon:
                raise ValueError("Model grid/horizon mismatch")
            mean, covariance = _prediction(model, data["initial"][:1] if conditional else data["initial"], cfg.query_times)
            elapsed = time.perf_counter() - tick
            if conditional:
                pm, pc, bm, bc = mean[0], covariance[0], None, None
            else:
                pm, pc = mixture_moments(mean, covariance, n_particles=cfg.n_particles)
                bm, bc = mixture_moments(mean, covariance, n_particles=cfg.n_particles, counts=counts)
            metric = moment_comparison(references[item["bank"]], predicted_mean=pm,
                predicted_scaled_covariance=pc, bootstrap_predicted_mean=bm,
                bootstrap_predicted_scaled_covariance=bc, n_particles=cfg.n_particles, counts=counts)
            arrays = {"mean": pm, "scaled_covariance": pc}
            if bm is not None:
                arrays.update(bootstrap_mean=bm, bootstrap_scaled_covariance=bc)
            records.append(_save_result(path, identity, metric, arrays,
                {"method": item, "scientific_prediction_seconds": elapsed,
                 "initial_integration_MC": not conditional, "joint_time_supported": False}))
            print(f"accuracy {key} {item['name']}", flush=True)
        # Literal algorithm used in BOTH Gaussian accuracy and timing.
        for step in (2**-7, 2**-10):
            for bank in banks(cfg):
                name = f"LGA_dt{round(-np.log2(step))}_{bank.name}"
                identity = {**identity_base, "group": key, "name": name, "bank": bank.name}
                path = dest / (name + ".json")
                replay = _replay(path, identity)
                if replay is not None:
                    records.append(replay)
                    continue
                origin = data["initial"][0] if conditional else data["probabilities"]
                pm, pc, diag = lga_marginals(origin, cfg.context(case), cfg.spec, cfg.query_times,
                    step, bank, None if conditional else iid_scaled_covariance(origin))
                metric = moment_comparison(references[bank.name], predicted_mean=pm,
                    predicted_scaled_covariance=pc, n_particles=cfg.n_particles, counts=counts)
                records.append(_save_result(path, identity, metric, {"mean": pm, "scaled_covariance": pc},
                    {"method": {"name": name, "bank": bank.name, "family": "LGA", "dt": step},
                     "law_diagnostics": diag, "law": "unstopped-linear-Gaussian-approximation"}))
        path_methods = [{**p, "family": "learned-path"} for p in plan["old_paths"] if p["case"] == case]
        path_methods.extend({"name": name, "family": name, "seed": None} for name in ("native-Euler", "Picard"))
        for item in path_methods:
            name = item["name"]
            identity = {**identity_base, "group": key, "name": name}
            path = dest / (name + ".json")
            replay = _replay(path, identity)
            if replay is not None:
                records.append(replay)
                continue
            runtime = _path_runtime(item, cfg, case)
            values, diag = _path_values(runtime, compiled, data["initial"], data["model_increments"], cfg, case)
            metric = {b: moment_comparison(references[b], values[b], n_particles=cfg.n_particles,
                                          counts=counts, paired=True) for b in references}
            arrays = {"outcomes_" + b: v for b, v in values.items()}
            records.append(_save_result(path, identity, metric, arrays, {"method": item, "diagnostics": diag}))
            print(f"accuracy {key} {name}", flush=True)
        audits = {}
        for bank, field in zip(banks(cfg), ("audit_bank_b", "audit_bank_c"), strict=True):
            values = data[field]
            audits[bank.name] = moment_comparison(values[:, -1], values[:, 0],
                n_particles=cfg.n_particles, paired=True, seed=SEED + 1)
        _json(dest / "group.json", {"complete": True, "identity": identity_base,
            "group": key, "records": [{"path": str(dest / (r["identity"]["name"] + ".json")),
                "sha256": _sha(dest / (r["identity"]["name"] + ".json"))} for r in records],
            "reference_refinement": audits, "stopped_reference_paths": int(np.isfinite(data["stopped_at"]).sum())})
        all_records.extend(records)
    complete_groups = [key for key in cfg.group_keys if (root / "comparison/accuracy" / key / "group.json").exists()]
    _json(root / "comparison/accuracy_index.json", {"complete": set(complete_groups) == set(cfg.group_keys),
        "identity": identity_base, "groups": complete_groups, "expected_groups": list(cfg.group_keys),
        "unique_reference_families": sum(map(len, family_sets)), "disjoint_learner_families": len(learner_ids),
        "declared_method_records_per_group": 33})
    return all_records


def benchmark(root, *, final_timing=False):
    if not final_timing:
        raise ValueError("Confirm all training/evaluation workers are finished with --final-timing")
    root, plan, lock, cfg = _verified(root)
    if not _read(root / "comparison/accuracy_index.json").get("complete"):
        raise ValueError("Complete accuracy evaluation before final timings")
    torch.set_num_threads(1)
    output = root / "comparison/timing.json"
    identity = {"plan_sha256": _sha(root / "comparison/preregistration.json"),
                "fit_lock_sha256": _sha(root / "payoffs/fit_lock.json"), "source_hash": source_hash()}
    if output.exists():
        previous = _read(output)
        if previous["identity"] != identity:
            raise ValueError("Timing identity changed")
        if previous.get("complete"):
            return previous
        records, loading = previous["records"], previous["loading"]
    else:
        records, loading = [], []
    complete_slots = {r["slot"] for r in records}
    packet = {"identity": identity, "complete": False, "records": records, "loading": loading,
              "environment": {"platform": platform.platform(), "numpy": np.__version__,
                              "torch": torch.__version__, "cpu_threads": 1},
              "final_timing_after_all_workers_finished": True}

    def save(slot, call, repeats, warmup, extra):
        if slot in complete_slots:
            return
        first_started = time.perf_counter()
        result = call()
        first_prediction_seconds = time.perf_counter() - first_started
        mean, covariance = result
        if (not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                or np.linalg.eigvalsh(covariance).min() < -1e-10):
            raise ValueError("Invalid prediction in literal cost comparison")
        measured = timing(call, repeats, warmup)
        records.append({"slot": slot, **extra, **measured,
                        "first_prediction_seconds": first_prediction_seconds})
        complete_slots.add(slot)
        _json(output, packet)
        print(f"timing {slot}", flush=True)

    for case in cfg.cases:
        probabilities, _ = profile_probabilities(cfg, case)
        rng = seeded_rng(SEED, KIND, "timing-only", case)
        conditional_origins = rng.multinomial(cfg.n_particles, probabilities, size=256) / cfg.n_particles
        for item in _model_entries(lock, plan):
            tick = time.perf_counter()
            model = _model(item, cfg)
            loading.append({"name": item["name"], "case": case, "seconds": time.perf_counter() - tick})
            for batch in (1, 64, 256):
                save(f"{case}/{item['name']}/conditional/{batch}",
                    lambda m=model, b=batch: _prediction(m, conditional_origins[:b], cfg.query_times), 21, 3,
                    {"case": case, "name": item["name"], "family": item["family"], "bank": item["bank"],
                     "role": "conditional", "batch": batch, "seed": item["seed"],
                     "scope": "three-horizon-four-mean-and-same-time-4x4-covariance-per-initial"})
            def ensemble(m=model):
                origin = rng.multinomial(cfg.n_particles, probabilities, size=cfg.paths_per_group) / cfg.n_particles
                pm, pc = _prediction(m, origin, cfg.query_times)
                return mixture_moments(pm, pc, n_particles=cfg.n_particles)
            save(f"{case}/{item['name']}/ensemble", ensemble, 5, 1,
                {"case": case, "name": item["name"], "family": item["family"], "bank": item["bank"],
                 "role": "ensemble", "batch": cfg.paths_per_group, "seed": item["seed"],
                 "scope": "fresh-histograms-plus-inference-plus-total-covariance-aggregation"})
        for step in (2**-7, 2**-10):
            for bank in banks(cfg):
                for role in ("conditional", "ensemble"):
                    origin = conditional_origins[0] if role == "conditional" else probabilities
                    initial_cov = None if role == "conditional" else iid_scaled_covariance(probabilities)
                    save(f"{case}/LGA/{bank.name}/{step}/{role}",
                        lambda o=origin, s=step, b=bank, c=initial_cov: lga_marginals(
                            o, cfg.context(case), cfg.spec, cfg.query_times, s, b, c)[:2], 21, 3,
                        {"case": case, "name": f"LGA_dt{round(-np.log2(step))}", "family": "LGA",
                         "bank": bank.name, "role": role, "batch": 1, "dt": step,
                         "scope": "literal-three-query-marginals-analytic-IID-covariance-for-ensemble"})
        tick = time.perf_counter()
        compiled = compile_native(root / "comparison/compiled_cache")
        packet.setdefault("compiled_control", []).append({"case": case,
            "setup_seconds_this_process": time.perf_counter() - tick, **compiled.metadata})
        path_methods = [{**p, "family": "learned-path"} for p in plan["old_paths"]
                        if p["case"] == case and p["seed"] == 0]
        path_methods.extend({"name": name, "family": name, "seed": None} for name in ("native-Euler", "Picard"))
        for item in path_methods:
            tick = time.perf_counter()
            runtime = _path_runtime(item, cfg, case)
            packet.setdefault("path_preparation", []).append({"case": case, "name": item["name"],
                "restoration_and_kernel_preparation_seconds": time.perf_counter() - tick})
            for role in ("conditional", "ensemble"):
                for bank in banks(cfg):
                    def path_call(r=runtime, ro=role, b=bank):
                        origin = (np.broadcast_to(conditional_origins[0], (cfg.paths_per_group, cfg.n_cells)).copy()
                                  if ro == "conditional" else rng.multinomial(
                                      cfg.n_particles, probabilities, size=cfg.paths_per_group) / cfg.n_particles)
                        noise = rng.normal(scale=np.sqrt(cfg.model_step), size=(cfg.paths_per_group,
                            round(cfg.horizon / cfg.model_step), cfg.n_cells))
                        vals, _ = _path_values(r, compiled, origin, noise, cfg, case, b.name)
                        mu, covariance = empirical(vals[b.name])
                        return mu, cfg.n_particles * covariance
                    save(f"{case}/{item['name']}/{bank.name}/{role}", path_call, 3, 1,
                        {"case": case, "name": item["name"], "family": item["family"], "bank": bank.name,
                         "role": role, "batch": cfg.paths_per_group, "seed": item.get("seed"),
                         "scope": "fresh-initials-and-forcing-plus-complete-paths-and-bank-aggregation"})
    packet["complete"] = True
    _json(output, packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "benchmark"))
    parser.add_argument("--root", type=Path, default=REPO / "runs/payoff-repair-2026-10-27")
    parser.add_argument("--group", action="append")
    parser.add_argument("--final-timing", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "evaluate":
        evaluate(args.root, args.group)
    else:
        benchmark(args.root, final_timing=args.final_timing)


if __name__ == "__main__":
    main()
