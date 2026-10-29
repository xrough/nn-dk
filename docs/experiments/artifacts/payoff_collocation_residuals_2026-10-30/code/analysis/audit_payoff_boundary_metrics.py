"""Independent moments, product bootstrap and v3/v2 contrast replay.

Empirical moments and total covariance are reconstructed by explicit centered
sums in the frozen independent mathematical audit. Both predictors' initial
integration uncertainty enters the repaired-versus-historical contrast.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import erf  # noqa: E402

from analysis import audit_payoff_regimes_metrics as mathematics  # noqa: E402
from analysis import payoff_legacy_control as legacy  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import native_coefficients  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "independent-boundary-repair-v2-v3-moment-product-bootstrap-replay-v1"


def contrast_both(reference, reference_bootstrap, first, second, n_particles):
    rm, rc = reference
    brm, brc = reference_bootstrap
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scale = std[..., :, None]*std[..., None, :]
    result = []
    for t in range(len(rm)):
        row = {"time_index": t}
        for statistic, key, truth, draws, normalizer in (
                ("mean", "mean", rm, brm, std),
                ("covariance", "scaled_covariance", n_particles*rc,
                 n_particles*brc, n_particles*scale)):
            axes = -1 if statistic == "mean" else (-2, -1)
            a, b = first[key], second[key]
            ba = first.get("bootstrap_"+key, a[None])
            bb = second.get("bootstrap_"+key, b[None])
            point = float(abs((b[t]-truth[t])/normalizer[t]).max()
                          - abs((a[t]-truth[t])/normalizer[t]).max())
            samples = (abs((bb[:, t]-draws[:, t])/normalizer[t]).max(axis=axes)
                       - abs((ba[:, t]-draws[:, t])/normalizer[t]).max(axis=axes))
            interval = np.quantile(samples, [.025, .975])
            row.update({statistic+"_error_advantage": point,
                        "bootstrap_"+statistic+"_advantages": samples.tolist(),
                        statistic+"_advantage_percentile95": interval.tolist(),
                        statistic+"_advantage_resolved": bool(interval[0] > 0)})
        result.append(row)
    return result


def analytic_untrained_predictions(model, initials, times):
    """Independent zero-head heat/native-drift algebra, no neural layers."""
    state = np.asarray(initials)
    time_grid = np.asarray(times)
    mc, _, cc, _ = model._time_coefficients(torch.tensor(times, dtype=torch.float64))
    phi = model.bank.terminal_weights/model.bank.scales[:, None]
    start = np.sqrt(model.context.n_particles)*(state @ phi.T)
    heat_drift = model.context.diffusion/model.spec.h**2*(
        np.roll(state, 1, -1)+np.roll(state, -1, -1)-2*state)
    native_drift = native_coefficients(state, model.context, model.spec).drift
    interaction = np.sqrt(model.context.n_particles)*((native_drift-heat_drift) @ phi.T)
    distance = state-model.spec.mass_stop
    gate = np.prod(erf(distance[:, None, :]*np.sqrt(
        model.horizon/(model.boundary_width**2*time_grid))[None, :, None]), axis=-1)
    active = (state > model.spec.mass_stop).all(-1)
    baseline = np.einsum("tog,pg->pto", mc.numpy(), state)
    mean = start[:, None]+gate[:, :, None]*(baseline-start[:, None]
        + time_grid[None, :, None]*interaction[:, None])
    covariance = gate[:, :, None, None]*np.einsum("tijg,pg->ptij", cc.numpy(), state)
    mean = np.where(active[:, None, None], mean, start[:, None])
    covariance = np.where(active[:, None, None, None], covariance, 0.)
    return (mean*model.bank.scales/np.sqrt(model.context.n_particles),
            covariance*np.outer(model.bank.scales, model.bank.scales))


def run(root):
    torch.set_num_threads(1)
    root = Path(root).resolve()
    started, cpu = time.monotonic(), time.process_time()
    study, _ = runner._assert_frozen(root)
    lock = runner.verify_fit_lock(root)
    comparison = runner._read(root/"comparison/preregistration.json")
    deployment = runner._read(root/"deployment/preregistration.json")
    if any(not runner._read(root/directory/"accuracy_index.json")["complete"]
           for directory in ("comparison", "deployment")):
        raise ValueError("Complete frozen accuracy and independent deployment required")
    controls = runner._read(root/"comparison/legacy_controls.json")
    if (runner._sha(root/"comparison/legacy_controls.json") != comparison["legacy_controls_sha256"]
            or controls != legacy.inventory(runner.case_names(study), study.optimizer_seeds)):
        raise ValueError("Declared historical controls changed")
    records, contrasts, legacy_contrasts, direct = [], [], [], []
    maximum = 0.
    for case in runner.case_names(study):
        cfg = runner.reference_configuration(case, study)
        selected = [dict(row, family="repaired") for row in lock["models"] if row["case"] == case]
        selected += [row for row in controls["models"] if row["case"] == case]
        selected += [{"name": case+"_untrained", "family": "untrained-repaired", "untrained": True}]
        for group in cfg.group_keys:
            profile, role, _ = group.split("__")
            reference = runner.load_reference(root, case, group)["outcomes"]
            rm, rc = mathematics.empirical(reference)
            counts = mathematics._counts(len(reference), int(stable_hash(
                [comparison["seed"], case, group])[:12], 16))
            brm, brc = mathematics.empirical(reference, counts)
            base = root/("comparison" if role == "conditional" else "deployment")/"accuracy"/case/group
            group_packet = runner._read(base/"group.json")
            if role == "ensemble":
                pool = next(p for p in deployment["pools"] if p["regime"] == case and p["profile"] == profile)
                initial_packet, initial = mathematics._numeric(root/"deployment/initials"/case/(profile+".json"))
                initial_counts = mathematics._counts(len(initial["initial"]), pool["bootstrap_seed"])
            values = {}
            for item in selected:
                packet, arrays = mathematics._numeric(base/(item["name"]+".json"))
                values[item["name"]] = arrays
                if packet["identity"]["reference_sha256"] != runner._read(root/"reference"/case/"manifest.json")["reference_sha256"]:
                    raise ValueError("Accuracy record reference identity changed")
                expected = mathematics.metric_records(rm, rc, brm, brc, arrays["mean"],
                    arrays["scaled_covariance"], cfg.n_particles,
                    arrays.get("bootstrap_mean"), arrays.get("bootstrap_scaled_covariance"))
                error = mathematics._compare(expected, packet["metrics"]["records"])
                if role == "ensemble":
                    error = max(error, mathematics._compare(mathematics.standalone_sampling(
                        arrays["mean"], arrays["scaled_covariance"], arrays["bootstrap_mean"],
                        arrays["bootstrap_scaled_covariance"], cfg.n_particles),
                        packet["standalone_initial_integration_sampling"]))
                    replay = item.get("untrained") or (cfg.horizon == 1. and cfg.kappa == 6.)
                    if replay:
                        if item.get("untrained"):
                            model = runner.untrained_model(case, study)
                            cm, cc = analytic_untrained_predictions(model, initial["initial"], cfg.query_times)
                        else:
                            model = (legacy.load_model(item) if item["family"] == "legacy-v2"
                                     else runner.load_declared_model(item))
                            cm, cc = mathematics.direct_predictions(model, initial["initial"], cfg.query_times)
                        m, c = mathematics.mixture(cm, cc, cfg.n_particles)
                        bm, bc = mathematics.mixture(cm, cc, cfg.n_particles, initial_counts)
                        direct_error = mathematics._compare({"mean": m, "scaled_covariance": c,
                            "bootstrap_mean": bm, "bootstrap_scaled_covariance": bc}, arrays)
                        error = max(error, direct_error)
                        direct.append({"case": case, "profile": profile, "name": item["name"],
                            "family": item["family"], "algebraic_untrained": bool(item.get("untrained")),
                            "pool_sha256": initial_packet["arrays"]["sha256"], "maximum_error": direct_error})
                records.append({"case": case, "group": group, "role": role, "name": item["name"],
                    "family": item["family"], "maximum_numeric_replay_error": error,
                    "source_record_sha256": runner._sha(base/(item["name"]+".json")),
                    "physical_payoff_bounds": mathematics.physical_bounds(runner.payoff_bank(case, study),
                        cfg.query_times, arrays["mean"], arrays["scaled_covariance"], cfg.n_particles)})
                maximum = max(maximum, error)
            _, gaussian = mathematics._numeric(root/"comparison/accuracy"/case/group/"LGA_dt10.json")
            for seed in study.optimizer_seeds:
                name, oldname = f"{case}_boundary_seed{seed}", f"{case}_legacy_v2_seed{seed}"
                arrays = values[name]
                expected = mathematics.contrast_records(rm, rc, brm, brc, arrays["mean"],
                    arrays["scaled_covariance"], gaussian["mean"], gaussian["scaled_covariance"],
                    cfg.n_particles, arrays.get("bootstrap_mean"), arrays.get("bootstrap_scaled_covariance"))
                maximum = max(maximum, mathematics._compare(expected, group_packet["NN_versus_fine_LGA"][name]["records"]))
                contrasts.append({"case": case, "group": group, "role": role, "name": name, "records": expected})
                expected = contrast_both((rm, rc), (brm, brc), arrays, values[oldname], cfg.n_particles)
                maximum = max(maximum, mathematics._compare(expected, group_packet["NN_versus_legacy_v2"][name]["records"]))
                legacy_contrasts.append({"case": case, "group": group, "role": role, "name": name, "records": expected})
        print(case, "independent new/legacy metrics replayed", flush=True)
    expected = sum(len(runner.reference_configuration(case, study).group_keys)
                   *(2*len(study.optimizer_seeds)+1) for case in runner.case_names(study))
    if len(records) != expected:
        raise AssertionError("Mathematical audit omitted a declared new/legacy neural head")
    result = {"complete": True, "kind": KIND, "source_hash": source_hash(),
        "script_sha256": runner._sha(__file__), "fit_lock_sha256": runner._sha(root/"payoffs/fit_lock.json"),
        "records": records, "conditional_contrasts": contrasts, "legacy_contrasts": legacy_contrasts,
        "all_seed_consistency": mathematics.seed_consistency(contrasts, study.optimizer_seeds),
        "legacy_seed_consistency": mathematics.seed_consistency(legacy_contrasts, study.optimizer_seeds),
        "direct_mixture_replays": direct, "direct_full_pool_replays": len(direct),
        "independent_ensemble_metric_records": sum(r["role"] == "ensemble" for r in records),
        "conditional_metric_records": sum(r["role"] == "conditional" for r in records),
        "max_numeric_replay_error": maximum, "audit_wall_seconds": time.monotonic()-started,
        "audit_cpu_seconds": time.process_time()-cpu, "seeds_pooled": False,
        "selection_or_refitting": False, "timing_workers_may_start_after_process_exit": True}
    output = root/"audit/metrics_replay.json"
    if output.exists():
        raise ValueError("Preserve the earlier independent mathematical audit")
    runner._json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.root)
    print(json.dumps({key: result[key] for key in ("complete", "independent_ensemble_metric_records",
        "conditional_metric_records", "direct_full_pool_replays", "max_numeric_replay_error")}))


if __name__ == "__main__":
    main()
