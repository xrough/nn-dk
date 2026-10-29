"""Independent IID-initial integration for the harder-regime neural heads.

Preparation saves only planned family identifiers and stream addresses. Initial
histograms are drawn after the complete fit lock; their integration bootstrap
is independent of the native-reference bootstrap. The original paired-initial
records remain untouched as diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import evaluate_payoff_regimes as original  # noqa: E402
from analysis import run_payoff_regimes as runner  # noqa: E402
from analysis.evaluate_path_generalization import (  # noqa: E402
    array_sha,
    bootstrap_counts,
    mixture_moments,
    moment_comparison,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "independent-hard-regime-neural-initial-integration-v1"
SEED = 2026102827
BOOTSTRAPS = 128


def pool_address(regime, profile):
    return int(stable_hash([SEED, KIND, regime, profile, "independent-initial-pool"])[:12], 16)


def initial_bootstrap_address(regime, profile):
    return int(stable_hash([SEED, KIND, regime, profile, "initial-bootstrap"])[:12], 16)


def planned_families(regime, profile, count):
    return np.array([stable_hash([SEED, KIND, regime, profile, i]) for i in range(count)], dtype="S64")


def _pool_key(regime, profile):
    return regime+"__"+profile


def prepare(root):
    """Preregister the independent integration pools without drawing histograms."""
    root = Path(root).resolve()
    study = runner.load_plan(root)
    cases = runner.case_names(study)
    helpers = [Path(__file__), Path(original.__file__), Path(runner.__file__),
               REPO/"analysis/evaluate_path_generalization.py",
               REPO/"analysis/benchmark_payoff_generalization.py"]
    pools, arrays = [], {}
    for regime in cases:
        cfg = runner.reference_configuration(regime, study)
        for profile in cfg.cases:
            key = _pool_key(regime, profile)
            families = planned_families(regime, profile, cfg.paths_per_group)
            arrays[key] = families
            pools.append({"key": key, "regime": regime, "profile": profile,
                "group": profile+"__ensemble__0", "paths": cfg.paths_per_group,
                "family_ids_array_sha256": array_sha(families),
                "initial_seed": pool_address(regime, profile),
                "initial_stream": [KIND, "multinomial-initials"],
                "bootstrap_seed": initial_bootstrap_address(regime, profile)})
    value = {"kind": KIND, "seed": SEED, "source_hash": source_hash(),
        "helpers": {str(path): runner._sha(path) for path in helpers},
        "study_plan_hash": study.plan_hash, "study_plan_sha256": runner._sha(root/"plan.json"),
        "training_preregistration_sha256": runner._sha(root/"payoffs/preregistration.json"),
        "original_comparison_preregistration_sha256": runner._sha(root/"comparison/preregistration.json"),
        "cases": list(cases), "optimizer_seeds": list(study.optimizer_seeds), "pools": pools,
        "models_per_pool": len(study.optimizer_seeds)+1, "expected_records": len(pools)*(len(study.optimizer_seeds)+1),
        "shared_pool_across_optimizer_seeds_and_untrained_control": True,
        "initials_drawn_during_preparation": False,
        "all_declared_fits_strictly_locked_before_initial_draw": True,
        "pool_family_formula": "stable_hash([SEED,KIND,regime,profile,index])",
        "pool_seed_formula": "int(stable_hash([SEED,KIND,regime,profile,'independent-initial-pool'])[:12],16)",
        "pool_rng": "seeded_rng(initial_seed,KIND,'multinomial-initials'); one multinomial call",
        "bootstrap_count": BOOTSTRAPS,
        "reference_bootstrap": {"canonical_evaluator_seed": original.SEED,
            "seed_formula": "int(stable_hash([original.SEED,regime,profile+'__ensemble__0'])[:12],16)"},
        "independent_NN_and_reference_bootstrap": True,
        "all_records_primary_initial_ensemble_predictions": True,
        "original_paired_initial_records": "preserved diagnostic; no numerical edits",
        "matching_timing": "same model names, regime/profile/ensemble role; unchanged fresh-IID histogram integration algorithm",
        "LGA_contrast": "reuse locked original LGA_dt10; identical reference resample, independent NN integration resample",
        "pretest_address_audit": "audit/deployment_pretest_independence.json",
        "required_audit_checks": ["planned_families_disjoint", "address_streams_disjoint"],
        "selection_using_final_reference": False, "post_test_refitting": False}
    destination = root/"deployment/preregistration.json"
    if not destination.exists() and any((root/"reference"/c/"reference.h5").exists() for c in cases):
        raise ValueError("Independent deployment pools must be preregistered before final reference draws")
    plan = original._immutable(destination, value)
    families_path = root/"deployment/planned_initial_families.npz"
    if not families_path.exists():
        with families_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        families_path.with_suffix(".tmp").replace(families_path)
    with np.load(families_path) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise ValueError("Planned independent initial family inventory changed")
    original._immutable(root/"deployment/preparation.json", {
        "complete": True, "initials_drawn": False, "plan_sha256": runner._sha(destination),
        "families_path": str(families_path), "families_sha256": runner._sha(families_path),
        "planned_pools": len(pools), "planned_independent_families": sum(p["paths"] for p in pools)})
    return plan


def _verified(root, *, require_complete_accuracy=True):
    root = Path(root).resolve()
    plan = prepare(root)
    study = runner.load_plan(root)
    lock = runner.verify_fit_lock(root)
    if not lock.get("complete") or len(lock.get("models", [])) != len(runner.slots(study)):
        raise ValueError("All declared fits must strictly restore before independent initial draws")
    audit = runner._read(root/plan["pretest_address_audit"])
    if (not audit.get("complete") or audit.get("source_hash") != source_hash()
            or audit.get("deployment_plan_sha256") != runner._sha(root/"deployment/preregistration.json")
            or any(audit.get(k) is not True for k in plan["required_audit_checks"])):
        raise ValueError("Pre-outcome disjoint family and entropy audit required")
    if require_complete_accuracy:
        index = runner._read(root/"comparison/accuracy_index.json")
        if (not index.get("complete") or index.get("source_hash") != source_hash()
                or index.get("plan_sha256") != runner._sha(root/"comparison/preregistration.json")):
            raise ValueError("Complete original accuracy comparison required")
    return root, plan, study, lock


def _initial_pool(root, pool, cfg, plan_sha, lock_sha):
    path = root/"deployment/initials"/pool["regime"]/(pool["profile"]+".json")
    identity = {"kind": KIND, "source_hash": source_hash(), "plan_sha256": plan_sha,
                "fit_lock_sha256": lock_sha, "pool": pool}
    previous = original._replay(path, identity)
    if previous is not None:
        with np.load(previous["arrays"]["path"]) as arrays:
            return {k: arrays[k] for k in arrays.files}, previous
    probabilities, quadrature = runner.profile_probabilities(cfg, pool["profile"])
    rng = seeded_rng(pool["initial_seed"], KIND, "multinomial-initials")
    initial = rng.multinomial(cfg.n_particles, probabilities, size=pool["paths"])/cfg.n_particles
    families = planned_families(pool["regime"], pool["profile"], pool["paths"])
    arrays = {"initial": initial, "probabilities": probabilities, "family_ids": families}
    if array_sha(families) != pool["family_ids_array_sha256"]:
        raise ValueError("Independent family address mismatch")
    record = original._save_result(path, identity, {}, arrays,
        {"initials_drawn_after_complete_fit_lock": True, "initial_probability_audit": quadrature,
         "no_native_forcing_or_reference_payoff_used": True})
    return arrays, record


def independent_bootstraps(regime, profile, reference_paths, initial_paths):
    """Independent product resampling, common across the four heads in one pool."""
    group = profile+"__ensemble__0"
    reference_seed = int(stable_hash([original.SEED, regime, group])[:12], 16)
    initial_seed = initial_bootstrap_address(regime, profile)
    if reference_seed == initial_seed:
        raise ValueError("Reference and initial-pool bootstrap addresses must differ")
    return (bootstrap_counts(reference_paths, reference_seed, BOOTSTRAPS),
            bootstrap_counts(initial_paths, initial_seed, BOOTSTRAPS))


def _original_gaussian(root, regime, group, base):
    path = root/"comparison/accuracy"/regime/group/"LGA_dt10.json"
    packet = runner._read(path)
    identity = packet.get("identity", {})
    if (not packet.get("complete") or packet["method"].get("family") != "LGA"
            or identity.get("regime") != regime or identity.get("group") != group
            or identity.get("reference_sha256") != base["reference_sha256"]
            or identity.get("fit_lock_sha256") != base["fit_lock_sha256"]
            or identity.get("comparison_plan_sha256") != base["comparison_plan_sha256"]
            or runner._sha(packet["arrays"]["path"]) != packet["arrays"]["sha256"]):
        raise ValueError("Original fine Gaussian comparator identity or evidence changed")
    with np.load(packet["arrays"]["path"]) as arrays:
        return arrays["mean"], arrays["scaled_covariance"], packet, runner._sha(path)


def evaluate(root, cases=None):
    root, plan, study, lock = _verified(root)
    torch.set_num_threads(1)
    selected = plan["cases"] if cases is None else cases
    if not set(selected) <= set(plan["cases"]):
        raise ValueError("Unknown preregistered physical regime")
    plan_sha, lock_sha = runner._sha(root/"deployment/preregistration.json"), runner._sha(root/"payoffs/fit_lock.json")
    all_records = []
    for pool in plan["pools"]:
        regime, profile, group = pool["regime"], pool["profile"], pool["group"]
        if regime not in selected:
            continue
        cfg, bank = runner.reference_configuration(regime, study), runner.payoff_bank(regime, study)
        manifest = original._reference_manifest(root, regime)
        reference_packet = runner.load_reference(root, regime, group)
        reference = np.asarray(reference_packet["outcomes"])
        initials, initial_record = _initial_pool(root, pool, cfg, plan_sha, lock_sha)
        if set(initials["family_ids"].astype(str)) & set(reference_packet["family_ids"].astype(str)):
            raise ValueError("Independent initial-integration and reference families overlap")
        reference_counts, initial_counts = independent_bootstraps(regime, profile, len(reference), len(initials["initial"]))
        base = {"source_hash": source_hash(), "deployment_plan_sha256": plan_sha,
            "comparison_plan_sha256": runner._sha(root/"comparison/preregistration.json"),
            "fit_lock_sha256": lock_sha, "regime": regime, "group": group,
            "reference_sha256": manifest["reference_sha256"], "bank_hash": bank.bank_hash,
            "independent_initial_pool_sha256": initial_record["arrays"]["sha256"]}
        gm, gc, gaussian, gaussian_sha = _original_gaussian(root, regime, group, base)
        records, contrasts = [], {}
        dest = root/"deployment/accuracy"/regime/group
        for item in original.model_entries(lock, regime, plan["optimizer_seeds"]):
            identity = {**base, "name": item["name"]}
            path = dest/(item["name"]+".json")
            record = original._replay(path, identity)
            if record is None:
                tick = time.perf_counter()
                model = original._model(item, runner, study)
                cm, cc = original._prediction(model, initials["initial"], cfg.query_times)
                pm, pc = mixture_moments(cm, cc, n_particles=cfg.n_particles)
                bm, bc = mixture_moments(cm, cc, n_particles=cfg.n_particles, counts=initial_counts)
                metrics = moment_comparison(reference, predicted_mean=pm, predicted_scaled_covariance=pc,
                    bootstrap_predicted_mean=bm, bootstrap_predicted_scaled_covariance=bc,
                    n_particles=cfg.n_particles, counts=reference_counts)
                record = original._save_result(path, identity, metrics,
                    {"mean": pm, "scaled_covariance": pc, "bootstrap_mean": bm,
                     "bootstrap_scaled_covariance": bc},
                    {"method": item, "scientific_prediction_seconds": time.perf_counter()-tick,
                     "initial_integration_MC": True, "initial_pool_independent_of_reference": True,
                     "reference_and_initial_bootstrap_independent": True,
                     "bootstrap_counts": {"reference_sha256": array_sha(reference_counts),
                                          "initial_sha256": array_sha(initial_counts)},
                     "standalone_initial_integration_sampling": original.standalone_sampling(
                         pm, pc/cfg.n_particles, bm, bc/cfg.n_particles),
                     "stopping_probability_prediction": "unsupported",
                     "cost_join": {"regime": regime, "profile": profile, "name": item["name"], "role": "ensemble"}})
            records.append(record)
            if item["family"] == "repaired":
                with np.load(record["arrays"]["path"]) as arrays:
                    contrasts[item["name"]] = original.bootstrap_advantage(reference, arrays["mean"],
                        arrays["scaled_covariance"], gm, gc, reference_counts, cfg.n_particles,
                        arrays["bootstrap_mean"], arrays["bootstrap_scaled_covariance"])
            print(f"independent-deployment {regime} {profile} {item['name']}", flush=True)
        runner._json(dest/"group.json", {"complete": True, "identity": base, "group": group,
            "regime": regime, "independent_initials": True,
            "records": [{"path": str(dest/(r["identity"]["name"]+".json")),
                         "sha256": runner._sha(dest/(r["identity"]["name"]+".json"))} for r in records],
            "NN_versus_fine_LGA": contrasts, "LGA_record_sha256": gaussian_sha,
            "LGA_numeric_arrays_sha256": gaussian["arrays"]["sha256"],
            "initial_pool_record_path": str(root/"deployment/initials"/regime/(profile+".json")),
            "initial_pool_record_sha256": runner._sha(root/"deployment/initials"/regime/(profile+".json")),
            "bootstrap_counts": {"reference_sha256": array_sha(reference_counts),
                                 "initial_sha256": array_sha(initial_counts)}})
        all_records.extend(records)
    expected = [(p["regime"], p["group"]) for p in plan["pools"]]
    completed = [(r, g) for r, g in expected if (root/"deployment/accuracy"/r/g/"group.json").exists()]
    runner._json(root/"deployment/accuracy_index.json", {"complete": len(completed) == len(expected),
        "kind": KIND, "source_hash": source_hash(), "deployment_plan_sha256": plan_sha,
        "fit_lock_sha256": lock_sha, "groups": completed, "expected_groups": expected,
        "records_per_group": plan["models_per_pool"], "expected_records": plan["expected_records"],
        "new_independent_initial_families": sum(p["paths"] for p in plan["pools"]),
        "original_paired_comparison_untouched": True, "inference_algorithm_unchanged": True,
        "selection_using_reference": False, "post_test_refitting": False})
    return all_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--root", type=Path, default=REPO/"runs/payoff-regimes-2026-10-27")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    result = prepare(args.root) if args.command == "prepare" else evaluate(args.root, args.case)
    print(json.dumps({"complete": True, "prepared": args.command == "prepare",
                      "records": None if args.command == "prepare" else len(result)}))


if __name__ == "__main__":
    main()
