"""Fresh-test accuracy and cost for the separately identified boundary repair.

The estimator and literal timing loops are loaded unchanged in a private
namespace.  Explicit hooks select the new locked study/model and new random
streams; importing this adapter never mutates the historical evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import payoff_legacy_control as legacy  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402

KIND = "boundary-consistent-balanced-residual-fresh-moment-accuracy-cost-v1"
SEED = 2026102919


def _runner():
    from analysis import run_payoff_boundary_regimes

    return run_payoff_boundary_regimes


def _model(item, runner, study):
    if item.get("family") == "legacy-v2":
        return legacy.load_model(item)
    if item.get("untrained"):
        return runner.untrained_model(item["case"], study)
    return runner.load_declared_model(item)


_implementation = isolated_module(
    "evaluate_payoff_regimes.py", "analysis._boundary_regimes_evaluator_impl")
_original_immutable = _implementation._immutable
_original_entries = _implementation.model_entries
_original_evaluate = _implementation.evaluate


def _immutable(path, value):
    """Bind historical controls and literal expanded allocations to each plan."""
    path = Path(path)
    if path.name == "preregistration.json" and value.get("kind") in (
            KIND, "independent-boundary-repair-neural-initial-integration-v1"):
        root = path.parents[1]
        study = _runner().load_plan(root)
        controls = legacy.inventory(value["cases"], study.optimizer_seeds)
        legacy.verify_targets(controls, _runner(), study)
        control_path = root/"comparison/legacy_controls.json"
        _original_immutable(control_path, controls)
        value = {**value, "legacy_controls_sha256": _implementation._sha(control_path),
                 "legacy_control_family": "legacy-v2",
                 "legacy_control_optimizer_seeds": list(study.optimizer_seeds)}
        if value["kind"] == KIND:
            value["records_per_group"] = 2*len(study.optimizer_seeds)+4
            value["new_versus_legacy_contrast"] = (
                "matched optimizer-seed, common new reference bootstrap; independent "
                "IID initial pool and its common integration bootstrap for both predictors")
        else:
            value["models_per_pool"] = 2*len(study.optimizer_seeds)+1
            value["expected_records"] = len(value["pools"])*value["models_per_pool"]
    return _original_immutable(path, value)


def model_entries(lock, regime, seeds=(0, 1, 2)):
    current = _original_entries(lock, regime, seeds)
    controls = legacy.inventory((regime,), seeds)["models"]
    return current[:-1]+controls+current[-1:]


def predictor_advantage(reference, candidate, control, counts, n_particles):
    """Common-reference contrast retaining BOTH predictors' pool uncertainty."""
    rm, rc = _implementation.empirical(reference)
    bm, bc = _implementation.empirical(reference, counts)
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scale = std[..., :, None]*std[..., None, :]
    if np.any(std <= 1e-14):
        raise ValueError("Nondegenerate reference fluctuation scales required")
    records = []
    for i in range(len(rm)):
        record = {"time_index": i}
        for statistic, key, ref, bref, denominator in (
                ("mean", "mean", rm, bm, std),
                ("covariance", "scaled_covariance", n_particles*rc,
                 n_particles*bc, n_particles*scale)):
            axes = -1 if statistic == "mean" else (-2, -1)
            first, second = candidate[key], control[key]
            bfirst = candidate.get("bootstrap_"+key, first[None])
            bsecond = control.get("bootstrap_"+key, second[None])
            point = float(np.max(abs(second[i]-ref[i])/denominator[i])
                          - np.max(abs(first[i]-ref[i])/denominator[i]))
            samples = (np.max(abs(bsecond[:, i]-bref[:, i])/denominator[i], axis=axes)
                       - np.max(abs(bfirst[:, i]-bref[:, i])/denominator[i], axis=axes))
            interval = np.quantile(samples, [.025, .975])
            record.update({statistic+"_error_advantage": point,
                           "bootstrap_"+statistic+"_advantages": samples.tolist(),
                           statistic+"_advantage_percentile95": interval.tolist(),
                           statistic+"_advantage_resolved": bool(interval[0] > 0)})
        records.append(record)
    return {"records": records, "positive_means_NN_smaller_error": True,
            "both_predictor_initial_pool_uncertainties_included": True,
            "interval_scope": "pointwise matched optimizer seed; no family-wise certificate"}


def add_legacy_contrasts(root, regime, group, destination):
    runner = _runner()
    study = runner.load_plan(root)
    cfg = runner.reference_configuration(regime, study)
    data = runner.load_reference(root, regime, group)
    counts = _implementation.bootstrap_counts(len(data["outcomes"]),
        int(_implementation.stable_hash([SEED, regime, group])[:12], 16), 128)
    packet_path = Path(destination)/"group.json"
    packet = runner._read(packet_path)
    contrasts = {}
    for seed in study.optimizer_seeds:
        arrays = []
        for name in (f"{regime}_boundary_seed{seed}", f"{regime}_legacy_v2_seed{seed}"):
            metadata = runner._read(Path(destination)/(name+".json"))
            if _implementation._sha(metadata["arrays"]["path"]) != metadata["arrays"]["sha256"]:
                raise ValueError("Predictor evidence changed before legacy contrast")
            with np.load(metadata["arrays"]["path"]) as saved:
                arrays.append({key: saved[key] for key in saved.files})
        contrasts[f"{regime}_boundary_seed{seed}"] = predictor_advantage(
            data["outcomes"], arrays[0], arrays[1], counts, cfg.n_particles)
    packet["NN_versus_legacy_v2"] = contrasts
    runner._json(packet_path, packet)


def evaluate(root, cases=None, groups=None):
    records = _original_evaluate(root, cases, groups)
    runner = _runner()
    root = Path(root).resolve()
    study = runner.load_plan(root)
    for regime in runner.case_names(study) if cases is None else cases:
        for group in (runner.reference_configuration(regime, study).group_keys
                      if groups is None else groups):
            destination = root/"comparison/accuracy"/regime/group
            add_legacy_contrasts(root, regime, group, destination)
    index_path = root/"comparison/accuracy_index.json"
    index = runner._read(index_path)
    index["records_per_group"] = 2*len(study.optimizer_seeds)+4
    runner._json(index_path, index)
    return records


bind_study(_implementation, __file__, KIND=KIND, SEED=SEED,
           _runner=_runner, _model=_model, _immutable=_immutable,
           model_entries=model_entries, evaluate=evaluate)


def __getattr__(name):
    return getattr(_implementation, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "benchmark"))
    parser.add_argument("--root", type=Path, default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--case", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--final-timing", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = _implementation.prepare(args.root)
    elif args.command == "evaluate":
        result = _implementation.evaluate(args.root, args.case, args.group)
    else:
        result = _implementation.benchmark(args.root, final_timing=args.final_timing)
    print(json.dumps({"complete": True, "command": args.command,
                      "records": len(result) if isinstance(result, list) else None}))


if __name__ == "__main__":
    main()
