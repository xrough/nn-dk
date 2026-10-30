"""Literal three-query Gaussian quality bridge for the completed payoff costs.

The broader path evaluator places seventeen common observation knots in its
Gaussian ODE solve. A requested maxstep 2^-7 therefore actually advances at
2^-8 there. The fixed-bank cost script returns only three query times, using
its literal 2^-7 steps. This preregistered supplement evaluates that exact
three-query algorithm at both predeclared steps on every held-out group.
No new physical data, fitted parameters, payoff definitions, or model selection
are introduced. Linearization and unstopped-versus-killed-law bias stay explicit.
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

from analysis.benchmark_payoff_generalization import (  # noqa: E402
    SOURCE_FREEZE,
    lga_marginals,
)
from analysis.evaluate_path_generalization import (  # noqa: E402
    array_sha,
    bootstrap_counts,
    empirical,
    moment_comparison,
)
from analysis.native_gaussian_baseline import iid_scaled_covariance  # noqa: E402
from analysis.path_generalization_reference import (  # noqa: E402
    ReferenceConfig,
    load_reference,
    prepare_reference,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "literal-three-query-Gaussian-quality-cost-bridge-v1"
SCRIPT = Path(__file__).resolve()
STEPS = (2**-7, 2**-10)
BOOTSTRAP_SEED = 2026102614
N_BOOTSTRAP = 128


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
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def _group_seed(group):
    # Literal same family counts as run_generalization_evaluation._seed(group).
    return int(stable_hash([BOOTSTRAP_SEED, group, "families"])[:15], 16)


def prepare(root, cfg=ReferenceConfig()):
    """Freeze the bridge before any physical reference outcome is generated."""
    root = Path(root).resolve()
    if source_hash() != SOURCE_FREEZE:
        raise ValueError("Gaussian payoff audit requires the frozen package source")
    directory = root/"evaluation"/"gaussian_marginals"
    path = directory/"preregistration.json"
    reference_plan = prepare_reference(root/"reference", cfg)
    cost_plan = root/"timing"/"payoff_benchmark_plan.json"
    cost_result = root/"timing"/"payoff_benchmark.json"
    fit_lock = root/"payoffs"/"fit_lock.json"
    if (not cost_plan.is_file() or not fit_lock.is_file()
            or _read(fit_lock).get("complete") is not True):
        raise ValueError("Prepared literal cost plan and complete fit lock required before preregistration")
    helpers = ("benchmark_payoff_generalization.py", "native_gaussian_baseline.py",
               "evaluate_path_generalization.py", "path_generalization_reference.py",
               "run_generalization_evaluation.py")
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": _sha(SCRIPT),
                "helper_hashes": {name: _sha(SCRIPT.with_name(name)) for name in helpers},
                "config": json.loads(json.dumps(asdict(cfg))), "config_hash": cfg.config_hash,
                "groups": list(cfg.group_keys), "steps": list(STEPS),
                "banks": {bank.name: bank.to_dict() for bank in _banks(cfg)},
                "fit_lock_sha256": _sha(fit_lock), "cost_plan_sha256": _sha(cost_plan),
                "cost_result_sha256": _sha(cost_result) if cost_result.exists() else None,
                "reference_plan_identity_hash": stable_hash(reference_plan["identity"]),
                "literal_grid": "three-queries-no-forced-common-knots",
                "queries": list(cfg.query_times), "joint_time_support": False,
                "bootstrap": {"count": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                              "unit": "whole-trajectory-family",
                              "same_counts_as_broader_evaluator": True},
                "initialization": "conditional-fixed-histogram-or-analytic-IID-first-two-moments",
                "law": "unstopped-native-linear-noise-approximation-v1",
                "killed_integral_and_stopping_equivalence": False,
                "linearization_and_stopping_bias": "unresolved",
                "selection_using_reference_outcomes": False}
    if path.exists():
        if _read(path) != identity:
            raise ValueError("Frozen Gaussian payoff audit identity changed")
    else:
        if ((root/"reference"/"reference.h5").exists()
                or reference_plan["status"] != "waiting-for-complete-fit-lock"):
            raise ValueError("The literal cost-quality bridge must be preregistered before reference outcomes")
        _json(path, identity)
    return directory, identity


def _verified_inputs(root, cfg, plan):
    from analysis.benchmark_payoff_generalization import _lock

    _, fit_sha = _lock(root, cfg)
    if fit_sha != plan["fit_lock_sha256"]:
        raise ValueError("Payoff fit lock changed after bridge preregistration")
    manifest = _read(Path(root)/"reference"/"manifest.json")
    evaluation = _read(Path(root)/"evaluation"/"evaluation_index.json")
    reference_path = Path(root)/"reference"/"reference.h5"
    if (manifest.get("status") != "complete"
            or manifest.get("reference_sha256") != _sha(reference_path)
            or manifest["identity"]["config_hash"] != cfg.config_hash
            or stable_hash(manifest["identity"]) != plan["reference_plan_identity_hash"]
            or evaluation.get("complete") is not True or evaluation.get("source_hash") != source_hash()
            or evaluation["input_lock"]["reference_sha256"] != manifest["reference_sha256"]
            or evaluation["input_lock"]["fit_lock_sha256"] != fit_sha):
        raise ValueError("Complete immutable reference and broader evaluation are required")
    inputs = {"fit_lock_sha256": fit_sha, "reference_sha256": manifest["reference_sha256"],
              "reference_manifest_sha256": _sha(Path(root)/"reference"/"manifest.json"),
              "reference_config_hash": cfg.config_hash,
              "broader_evaluation_index_sha256": _sha(Path(root)/"evaluation"/"evaluation_index.json")}
    # This verifies all committed chunks and family separation once. Later reads
    # are small direct projections protected by the same immutable whole-file SHA.
    checked = load_reference(Path(root)/"reference", cfg, group_key=cfg.group_keys[0])
    return inputs, checked


def _read_group(root, key):
    with h5py.File(Path(root)/"reference"/"reference.h5", "r") as store:
        group = store[key]
        data = {name: group[name][:] for name in (
            "initial", "family_ids", "bank_b", "bank_c", "probabilities")}
        data["metadata"] = dict(group.attrs)
    return data


def evaluate_group(data, cfg, key, counts):
    """Literal coarse/fine cost algorithms and whole-family same-time metrics."""
    conditional = data["metadata"]["role"] == "conditional"
    case = data["metadata"]["case"]
    initial = data["initial"][0] if conditional else data["probabilities"]
    initial_covariance = None if conditional else iid_scaled_covariance(initial)
    results = []
    for step in STEPS:
        metrics, predictions, arrays, diagnostics = {}, {}, {}, {}
        for bank, reference_name in zip(_banks(cfg), ("bank_b", "bank_c"), strict=True):
            mean, covariance, diagnostic = lga_marginals(initial, cfg.context(case), cfg.spec,
                cfg.query_times, step, bank, initial_covariance)
            metrics[bank.name] = {"same_time": moment_comparison(data[reference_name],
                predicted_mean=mean, predicted_scaled_covariance=covariance,
                n_particles=cfg.n_particles, counts=counts)}
            predictions[bank.name] = {"mean": mean.tolist(), "scaled_covariance": covariance.tolist()}
            arrays[bank.name+"_mean"] = mean
            arrays[bank.name+"_scaled_covariance"] = covariance
            diagnostics[bank.name] = diagnostic
        results.append({"complete": True, "method": {
            "name": f"linear_gaussian_marginals_rk4_{round(-np.log2(step))}",
            "method": "linear-Gaussian-marginals", "requested_max_step": step,
            "literal_grid": "three-queries-no-forced-common-knots"},
            "group": key, "metrics": metrics, "predictions": predictions,
            "law_diagnostics": diagnostics, "bootstrap_arrays": arrays,
            "initialization": "conditional" if conditional else "analytic-IID-moments",
            "initial_pool_integration_MC": False,
            "unsupported": ["joint-time covariance", "density-path samples", "stopping law",
                            "exact killed-integral law"],
            "continuous_reference_bias": "unresolved",
            "linearization_and_stopping_bias": "unresolved"})
    numerical = {}
    for bank, reference_name in zip(_banks(cfg), ("bank_b", "bank_c"), strict=True):
        coarse, fine = results[0]["predictions"][bank.name], results[1]["predictions"][bank.name]
        mean_delta = np.asarray(coarse["mean"])-np.asarray(fine["mean"])
        covariance_delta = np.asarray(coarse["scaled_covariance"])-np.asarray(fine["scaled_covariance"])
        _, reference_covariance = empirical(data[reference_name])
        std = np.sqrt(cfg.n_particles*np.diagonal(reference_covariance, axis1=-2, axis2=-1))
        numerical[bank.name] = {
            "maximum_sqrtN_mean_difference_in_reference_scaled_std": float(np.max(
                abs(mean_delta)*np.sqrt(cfg.n_particles)/std)),
            "maximum_covariance_difference_in_reference_products": float(np.max(
                abs(covariance_delta)/(std[..., :, None]*std[..., None, :]))),
            "relative_covariance_frobenius_difference": float(np.linalg.norm(covariance_delta)
                /np.linalg.norm(fine["scaled_covariance"])),
            "scope": "deterministic ODE step difference; no linearization/stopping-bias bound"}
    for result in results:
        result["deterministic_coarse_fine"] = numerical
    return results


def run(root, cfg=ReferenceConfig(), *, groups=None):
    root = Path(root).resolve()
    directory, plan = prepare(root, cfg)
    inputs, checked = _verified_inputs(root, cfg, plan)
    selected = cfg.group_keys if groups is None else tuple(groups)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(cfg.group_keys):
        raise ValueError("Distinct predeclared audit groups required")
    for key in selected:
        data = checked if key == cfg.group_keys[0] else _read_group(root, key)
        counts = bootstrap_counts(len(data["initial"]), _group_seed(key), N_BOOTSTRAP)
        identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": plan["script_sha256"],
                    "helper_hashes": plan["helper_hashes"], "input_lock": inputs, "group": key,
                    "bootstrap_counts_sha256": array_sha(counts),
                    "array_hashes": {name: array_sha(data[name]) for name in (
                        "initial", "family_ids", "bank_b", "bank_c", "probabilities")}}
        records = evaluate_group(data, cfg, key, counts)
        for record in records:
            arrays = record.pop("bootstrap_arrays")
            arrays["bootstrap_counts"] = counts.astype(np.uint16)
            record["identity"] = {**identity, "method": record["method"]}
            path = directory/key/(record["method"]["name"]+".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                previous = _read(path)
                if previous.get("identity") != record["identity"]:
                    raise ValueError("Completed Gaussian bridge record identity changed")
                artifact = previous["bootstrap_artifact"]
                if artifact["sha256"] != _sha(artifact["path"]):
                    raise ValueError("Gaussian bridge numerical/bootstrap artifact changed")
                continue
            array_path = path.with_suffix(".npz")
            _npz(array_path, arrays)
            record["bootstrap_artifact"] = {"path": str(array_path.resolve()), "sha256": _sha(array_path)}
            _json(path, record)
        del data, counts, records
    index_records = []
    for key in cfg.group_keys:
        for step in STEPS:
            path = directory/key/f"linear_gaussian_marginals_rk4_{round(-np.log2(step))}.json"
            if path.exists():
                record = _read(path)
                if record.get("complete") and record["identity"].get("input_lock") == inputs:
                    index_records.append({"group": key, "name": record["method"]["name"],
                        "method": record["method"], "path": str(path.resolve()),
                        "json_path": str(path.resolve()), "sha256": _sha(path)})
    # A final reference hash prevents changing inputs during a resumed evaluation.
    if _sha(root/"reference"/"reference.h5") != inputs["reference_sha256"]:
        raise ValueError("Reference mutated during the Gaussian payoff audit")
    if source_hash() != plan["source_hash"] or _sha(SCRIPT) != plan["script_sha256"]:
        raise ValueError("Source changed during the Gaussian payoff audit")
    complete = len(index_records) == len(cfg.group_keys)*len(STEPS)
    index = {"kind": KIND, "source_hash": source_hash(), "status": "complete" if complete else "partial",
             "complete": complete, "groups": list(cfg.group_keys), "records": index_records,
             "expected_records": len(cfg.group_keys)*len(STEPS), "completed_records": len(index_records),
             "input_lock": inputs, "reference_sha256": inputs["reference_sha256"],
             "fit_lock_sha256": inputs["fit_lock_sha256"],
             "preregistration": {"path": str((directory/"preregistration.json").resolve()),
                                  "sha256": _sha(directory/"preregistration.json")},
             "literal_grid": "three-queries-no-forced-common-knots",
             "continuous_reference_bias": "unresolved", "linearization_and_stopping_bias": "unresolved"}
    _json(directory/"index.json", index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"runs/path-generalization-2026-10-26")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--groups", nargs="+")
    args = parser.parse_args()
    if args.prepare_only:
        directory, plan = prepare(args.root)
        print(json.dumps({"status": "prepared-only", "directory": str(directory),
                          "groups": plan["groups"], "steps": plan["steps"]}, indent=2))
    else:
        started = time.monotonic()
        result = run(args.root, groups=args.groups)
        print(json.dumps({"status": result["status"], "completed_records": result["completed_records"],
                          "seconds": time.monotonic()-started}, indent=2))


if __name__ == "__main__":
    main()
