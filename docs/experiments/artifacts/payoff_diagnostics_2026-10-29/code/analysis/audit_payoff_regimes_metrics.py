"""Independent statistical replay of conditional and deployment moment results.

This audit reimplements the estimators without calling the production moment,
mixture, sampling-radius or advantage functions.  No fit is changed or selected.
Direct prediction replays are fixed in advance: every untrained heat control,
and all three T=1, kappa=6 fitted seeds for both independent initial pools.
Run the numerical phase before uncontended timing, and the cost phase afterward.
"""
from __future__ import annotations

import argparse
import hashlib
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

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.audit_payoff_regimes import _json, _sha  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "independent-regime-moment-mixture-product-bootstrap-replay-v1"


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def empirical(values, counts=None):
    """Explicit weighted centered sums, independent of flattened BLAS code."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    center = np.mean(values, axis=0)
    z = values-center
    if counts is None:
        return center, np.stack([np.cov(values[:, t].T, ddof=1) for t in range(values.shape[1])])
    means, covariances = [], []
    for weights in np.asarray(counts):
        if not np.isclose(weights.sum(), n, atol=0, rtol=0):
            raise ValueError("Whole-family bootstrap counts must sum to the pool size")
        delta = np.einsum("p,pti->ti", weights, z)/n
        weighted = weights[:, None, None]*z
        second = np.einsum("pti,ptj->tij", weighted, z, optimize=True)
        means.append(center+delta)
        covariances.append((second-n*delta[:, :, None]*delta[:, None, :])/(n-1))
    return np.asarray(means), np.asarray(covariances)


def mixture(mean, scaled_covariance, n_particles, counts=None):
    location, between = empirical(mean, counts)
    if counts is None:
        within = np.mean(scaled_covariance, axis=0)
    else:
        within = np.stack([np.einsum("p,ptij->tij", w, scaled_covariance)/len(mean) for w in counts])
    return location, within+n_particles*between


def metric_records(rm, rc, brm, brc, pm, pc, n_particles, bpm=None, bpc=None):
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scale = std[:, :, None]*std[:, None, :]
    pm, pc = np.asarray(pm), np.asarray(pc)/n_particles
    bpm = pm[None] if bpm is None else np.asarray(bpm)
    bpc = pc[None] if bpc is None else np.asarray(bpc)/n_particles
    dm, dc = (pm-rm)/std, (pc-rc)/scale
    bdm, bdc = (bpm-brm)/std, (bpc-brc)/scale
    result = []
    for t in range(len(rm)):
        mean_error, cov_error = float(abs(dm[t]).max()), float(abs(dc[t]).max())
        mean_radius = float(np.quantile(abs(bdm[:, t]-dm[t]).max(-1), .95))
        cov_radius = float(np.quantile(abs(bdc[:, t]-dc[t]).max(axis=(-2, -1)), .95))
        result.append({"time_index": t, "max_mean_error_in_reference_std": mean_error,
            "max_normalized_covariance_error": cov_error,
            "mean_sampling95_simultaneous_radius": mean_radius,
            "covariance_sampling95_simultaneous_radius": cov_radius,
            "mean_error_plus_sampling95": mean_error+mean_radius,
            "covariance_error_plus_sampling95": cov_error+cov_radius,
            "reference_mean": rm[t].tolist(), "candidate_mean": pm[t].tolist(),
            "reference_scaled_covariance": (n_particles*rc[t]).tolist(),
            "candidate_scaled_covariance": (n_particles*pc[t]).tolist(),
            "normalized_mean_difference": dm[t].tolist(),
            "normalized_covariance_difference": dc[t].tolist()})
    return result


def contrast_records(rm, rc, brm, brc, nm, nc, gm, gc, n_particles, bnm=None, bnc=None):
    std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
    scale = std[:, :, None]*std[:, None, :]
    nm, nc, gm, gc = np.asarray(nm), np.asarray(nc)/n_particles, np.asarray(gm), np.asarray(gc)/n_particles
    bnm = nm[None] if bnm is None else np.asarray(bnm)
    bnc = nc[None] if bnc is None else np.asarray(bnc)/n_particles
    result = []
    for t in range(len(rm)):
        mean_point = float(abs((gm[t]-rm[t])/std[t]).max()-abs((nm[t]-rm[t])/std[t]).max())
        cov_point = float(abs((gc[t]-rc[t])/scale[t]).max()-abs((nc[t]-rc[t])/scale[t]).max())
        mean_draws = abs((gm[t]-brm[:, t])/std[t]).max(-1)-abs((bnm[:, t]-brm[:, t])/std[t]).max(-1)
        cov_draws = abs((gc[t]-brc[:, t])/scale[t]).max(axis=(-2, -1))-abs(
            (bnc[:, t]-brc[:, t])/scale[t]).max(axis=(-2, -1))
        mi, ci = np.quantile(mean_draws, [.025, .975]), np.quantile(cov_draws, [.025, .975])
        result.append({"time_index": t, "mean_error_advantage": mean_point,
            "covariance_error_advantage": cov_point, "bootstrap_mean_advantages": mean_draws.tolist(),
            "bootstrap_covariance_advantages": cov_draws.tolist(),
            "mean_advantage_percentile95": mi.tolist(), "covariance_advantage_percentile95": ci.tolist(),
            "mean_advantage_resolved": bool(mi[0] > 0), "covariance_advantage_resolved": bool(ci[0] > 0)})
    return result


def standalone_sampling(pm, pc, bpm, bpc, n_particles):
    covariance, bcov = np.asarray(pc)/n_particles, np.asarray(bpc)/n_particles
    std = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    scale = std[:, :, None]*std[:, None, :]
    return {"mean95_in_candidate_std": np.quantile(abs((bpm-pm)/std).max(-1), .95, axis=0).tolist(),
            "covariance95_normalized": np.quantile(abs((bcov-covariance)/scale).max(axis=(-2, -1)), .95, axis=0).tolist()}


def physical_bounds(bank, times, mean, scaled_covariance, n_particles):
    clock = np.asarray(times)[:, None]
    lower = bank.terminal_weights.min(-1)+clock*np.minimum(0., bank.running_weights.min(-1))
    upper = bank.terminal_weights.max(-1)+clock*np.maximum(0., bank.running_weights.max(-1))
    inside = (mean >= lower-1e-12) & (mean <= upper+1e-12)
    variance = np.diagonal(scaled_covariance, axis1=-2, axis2=-1)/n_particles
    variance_inside = inside & (variance >= -1e-12) & (variance <= (upper-mean)*(mean-lower)+1e-12)
    return {"mean_inside": inside.tolist(), "variance_inside": variance_inside.tolist(),
            "mean_violation_count": int((~inside).sum()),
            "variance_violation_count": int((~variance_inside).sum()),
            "minimum_scaled_covariance_eigenvalue": float(np.linalg.eigvalsh(scaled_covariance).min())}


def seed_consistency(contrasts, seeds):
    groups = {}
    for packet in contrasts:
        for row in packet["records"]:
            for statistic in ("mean", "covariance"):
                key = packet["case"], packet["group"], packet["role"], row["time_index"], statistic
                groups.setdefault(key, []).append((packet["name"], row[statistic+"_advantage_percentile95"]))
    result = []
    for key, entries in sorted(groups.items()):
        if len(entries) != len(seeds) or len({name for name, _ in entries}) != len(seeds):
            raise ValueError("An advantage summary omitted an optimizer seed")
        intervals = np.asarray([interval for _, interval in entries])
        result.append({"case": key[0], "group": key[1], "role": key[2], "time_index": key[3],
                       "statistic": key[4], "all_seeds_NN_closer_pointwise95": bool(np.all(intervals[:, 0] > 0)),
                       "all_seeds_LGA_closer_pointwise95": bool(np.all(intervals[:, 1] < 0)),
                       "minimum_lower95": float(intervals[:, 0].min()),
                       "maximum_upper95": float(intervals[:, 1].max()),
                       "scope": "all fixed optimizer seeds, pointwise intervals; not a family-wise grid certificate"})
    return result


def _compare(expected, actual):
    """Compare an independently reconstructed numeric subtree."""
    maximum = 0.
    if isinstance(expected, dict):
        for key, value in expected.items():
            maximum = max(maximum, _compare(value, actual[key]))
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        if len(expected) != len(actual):
            raise AssertionError("Saved mathematical record count changed")
        for first, second in zip(expected, actual, strict=True):
            maximum = max(maximum, _compare(first, second))
    elif isinstance(expected, (bool, str)):
        if expected != actual:
            raise AssertionError("Saved nonnumeric audit field differs")
    else:
        first, second = np.asarray(expected), np.asarray(actual)
        np.testing.assert_allclose(first, second, rtol=2e-10, atol=2e-10)
        if first.size:
            maximum = float(abs(first-second).max())
    return maximum


def _numeric(path):
    packet = json.loads(Path(path).read_text())
    artifact = packet["arrays"]
    if not packet.get("complete") or _sha(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Incomplete or changed accuracy numeric evidence")
    with np.load(artifact["path"]) as saved:
        values = {key: saved[key] for key in saved.files}
    return packet, values


def _new_model(runner, study, case):
    config, bank = runner.repaired_configuration(case, study), runner.payoff_bank(case, study)
    model = RepairPayoffModel(config.study.spec, config.study.physics, config.study.horizon,
                              bank, config.study.width, config.quadrature_nodes)
    return model.eval()


def direct_predictions(model, initials, times):
    """Call value equations separately by physical time, not the evaluator API."""
    mean, covariance = [], []
    with torch.no_grad():
        for t in times:
            means, covariances = [], []
            for start in range(0, len(initials), 512):
                state = torch.tensor(initials[start:start+512], dtype=torch.float64)
                m, c = model.moment_values(state, torch.full((len(state),), float(t), dtype=torch.float64))
                means.append(m.numpy()*model.bank.scales/np.sqrt(model.context.n_particles))
                covariances.append(c.numpy()*np.outer(model.bank.scales, model.bank.scales))
            mean.append(np.concatenate(means))
            covariance.append(np.concatenate(covariances))
    return np.stack(mean, axis=1), np.stack(covariance, axis=1)


def analytic_untrained_predictions(model, initials, times):
    """Zero-correction heat algebra; no neural-layer evaluation."""
    state = np.asarray(initials)
    coefficients = model._time_coefficients(torch.tensor(times, dtype=torch.float64))
    mc, cc = coefficients[0].numpy(), coefficients[2].numpy()
    start = np.sqrt(model.context.n_particles)*(state @ (model.bank.terminal_weights/model.bank.scales[:, None]).T)
    gate = np.prod(np.tanh((state-model.spec.mass_stop)/model.boundary_width), axis=-1)
    active = (state > model.spec.mass_stop).all(-1)
    baseline = np.einsum("tog,pg->pto", mc, state)
    covariance = np.einsum("tijg,pg->ptij", cc, state)
    mean = start[:, None]+gate[:, None, None]*(baseline-start[:, None])
    covariance *= gate[:, None, None, None]
    mean = np.where(active[:, None, None], mean, start[:, None])
    covariance = np.where(active[:, None, None, None], covariance, 0.)
    return (mean*model.bank.scales/np.sqrt(model.context.n_particles),
            covariance*np.outer(model.bank.scales, model.bank.scales))


def _counts(n, seed, repetitions=128):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=repetitions).astype(float)


def run(root):
    from analysis import run_payoff_regimes as runner

    torch.set_num_threads(1)
    wall, cpu = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    study, _ = runner._assert_frozen(root)
    lock = runner.verify_fit_lock(root)
    original_index = runner._read(root/"comparison/accuracy_index.json")
    deployment_index = runner._read(root/"deployment/accuracy_index.json")
    if not original_index.get("complete") or not deployment_index.get("complete"):
        raise ValueError("All frozen conditional and independent-initial results must complete before this audit")
    comparison = runner._read(root/"comparison/preregistration.json")
    deployment = runner._read(root/"deployment/preregistration.json")
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": _sha(__file__),
                "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"),
                "original_index_sha256": _sha(root/"comparison/accuracy_index.json"),
                "deployment_index_sha256": _sha(root/"deployment/accuracy_index.json"),
                "predeclared_direct_fitted_replays": "T=1,kappa=6; every optimizer seed and both initial laws",
                "untrained_replays": "all twelve independent-initial heat controls, algebra without network layers"}
    result = {"complete": False, "identity": identity, "records": [], "direct_mixture_replays": [],
              "conditional_contrasts": [], "reference_hashes": {}, "max_numeric_replay_error": 0.,
              "reference_refinement": [], "selection_or_refitting": False, "seeds_pooled": False}
    models = {row["name"]: row for row in lock["models"]}
    for case in runner.case_names(study):
        cfg = runner.reference_configuration(case, study)
        manifest = runner._read(root/"reference"/case/"manifest.json")
        reference_path = root/"reference"/case/"reference.h5"
        if _sha(reference_path) != manifest["reference_sha256"]:
            raise ValueError("Reference HDF5 changed after its completed manifest")
        result["reference_hashes"][case] = manifest["reference_sha256"]
        for key in cfg.group_keys:
            profile, role, _ = key.split("__")
            with h5py.File(reference_path, "r") as store:
                reference = store[key]["outcomes"][:]
                refined_outcomes = store[key]["audit_outcomes"][:]
            rm, rc = empirical(reference)
            reference_seed = int(stable_hash([comparison["seed"], case, key])[:12], 16)
            counts = _counts(len(reference), reference_seed)
            brm, brc = empirical(reference, counts)
            gaussian, gaussian_arrays = _numeric(root/"comparison/accuracy"/case/key/"LGA_dt10.json")
            gm, gc = gaussian_arrays["mean"], gaussian_arrays["scaled_covariance"]
            original_group = runner._read(root/"comparison/accuracy"/case/key/"group.json")
            refinement_counts = _counts(len(refined_outcomes), comparison["seed"]+101)
            frm, frc = empirical(refined_outcomes[:, -1])
            fbrm, fbrc = empirical(refined_outcomes[:, -1], refinement_counts)
            for level in range(refined_outcomes.shape[1]-1):
                cm, cc = empirical(refined_outcomes[:, level])
                cbm, cbc = empirical(refined_outcomes[:, level], refinement_counts)
                replayed = metric_records(frm, frc, fbrm, fbrc, cm, cfg.n_particles*cc,
                                           cfg.n_particles, cbm, cfg.n_particles*cbc)
                replay_error = _compare(replayed, original_group["reference_refinement"][f"level{level}_versus_finest"]["records"])
                result["max_numeric_replay_error"] = max(result["max_numeric_replay_error"], replay_error)
                result["reference_refinement"].append({"case": case, "group": key,
                    "level": f"level{level}_versus_finest", "maximum_replay_error": replay_error,
                    "records": [{k: row[k] for k in ("time_index", "max_mean_error_in_reference_std",
                        "max_normalized_covariance_error", "mean_sampling95_simultaneous_radius",
                        "covariance_sampling95_simultaneous_radius", "mean_error_plus_sampling95",
                        "covariance_error_plus_sampling95")} for row in replayed]})
            if role == "conditional":
                directory = root/"comparison/accuracy"/case/key
                group = runner._read(directory/"group.json")
                selected = [models[row["name"]] for row in lock["models"] if row["case"] == case]
            else:
                directory = root/"deployment/accuracy"/case/key
                group = runner._read(directory/"group.json")
                selected = [models[row["name"]] for row in lock["models"] if row["case"] == case]
                selected += [{"name": case+"_untrained", "untrained": True, "seed": 0}]
                pool = next(p for p in deployment["pools"] if p["regime"] == case and p["profile"] == profile)
                initial_record, initial_arrays = _numeric(root/"deployment/initials"/case/(profile+".json"))
                initial_counts = _counts(len(initial_arrays["initial"]), pool["bootstrap_seed"])
                if array_hash(initial_counts) == array_hash(counts):
                    raise AssertionError("Independent initial and reference bootstrap matrices coincide")
                _compare({"reference_sha256": array_hash(counts), "initial_sha256": array_hash(initial_counts)},
                         group["bootstrap_counts"])
            for item in selected:
                path = directory/(item["name"]+".json")
                packet, arrays = _numeric(path)
                if (packet["identity"]["reference_sha256"] != manifest["reference_sha256"]
                        or packet["identity"]["fit_lock_sha256"] != identity["fit_lock_sha256"]):
                    raise ValueError("Metric record source reference or fit inventory changed")
                pm, pc = arrays["mean"], arrays["scaled_covariance"]
                bpm, bpc = arrays.get("bootstrap_mean"), arrays.get("bootstrap_scaled_covariance")
                expected = metric_records(rm, rc, brm, brc, pm, pc, cfg.n_particles, bpm, bpc)
                std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
                scale = std[:, :, None]*std[:, None, :]
                standalone_mean = (np.quantile(abs((bpm-pm)/std).max(-1), .95, axis=0)
                                   if bpm is not None else np.zeros(len(rm)))
                standalone_covariance = (np.quantile(abs((bpc-pc)/cfg.n_particles/scale).max(axis=(-2, -1)), .95, axis=0)
                                         if bpc is not None else np.zeros(len(rm)))
                error = _compare(expected, packet["metrics"]["records"])
                if packet["metrics"]["bootstrap"]["counts_sha256"] != array_hash(counts):
                    raise AssertionError("Metric reference bootstrap count identity changed")
                if role == "ensemble":
                    error = max(error, _compare(standalone_sampling(pm, pc, bpm, bpc, cfg.n_particles),
                                               packet["standalone_initial_integration_sampling"]))
                    _compare({"reference_sha256": array_hash(counts), "initial_sha256": array_hash(initial_counts)},
                             packet["bootstrap_counts"])
                    replay = item.get("untrained") or (cfg.horizon == 1. and cfg.kappa == 6.)
                    if replay:
                        if item.get("untrained"):
                            model = _new_model(runner, study, case)
                            cm, cc = analytic_untrained_predictions(model, initial_arrays["initial"], cfg.query_times)
                        else:
                            model = runner.load_declared_model(item)
                            cm, cc = direct_predictions(model, initial_arrays["initial"], cfg.query_times)
                        m, c = mixture(cm, cc, cfg.n_particles)
                        bm, bc = mixture(cm, cc, cfg.n_particles, initial_counts)
                        direct_error = _compare({"mean": m, "scaled_covariance": c,
                                                 "bootstrap_mean": bm, "bootstrap_scaled_covariance": bc}, arrays)
                        error = max(error, direct_error)
                        result["direct_mixture_replays"].append({"case": case, "profile": profile,
                            "name": item["name"], "algebraic_untrained": bool(item.get("untrained")),
                            "initial_pool_sha256": initial_record["arrays"]["sha256"],
                            "paths": len(initial_arrays["initial"]), "maximum_error": direct_error})
                if not item.get("untrained"):
                    contrasts = contrast_records(rm, rc, brm, brc, pm, pc, gm, gc, cfg.n_particles, bpm, bpc)
                    contrast_error = _compare(contrasts, group["NN_versus_fine_LGA"][item["name"]]["records"])
                    error = max(error, contrast_error)
                    result["conditional_contrasts"].append({"case": case, "group": key, "name": item["name"],
                                                            "role": role, "records": contrasts})
                result["records"].append({"case": case, "group": key, "name": item["name"], "role": role,
                    "source_record_sha256": _sha(path), "numeric_arrays_sha256": packet["arrays"]["sha256"],
                    "maximum_numeric_replay_error": error,
                    "worst_mean_error": max(r["max_mean_error_in_reference_std"] for r in expected),
                    "worst_covariance_error": max(r["max_normalized_covariance_error"] for r in expected),
                    "worst_mean_error_plus_sampling": max(r["mean_error_plus_sampling95"] for r in expected),
                    "worst_covariance_error_plus_sampling": max(r["covariance_error_plus_sampling95"] for r in expected),
                    "physical_payoff_bounds": physical_bounds(runner.payoff_bank(case, study), cfg.query_times,
                                                               pm, pc, cfg.n_particles),
                    "standalone_sampling_in_reference_units": {"mean95_radius": standalone_mean.tolist(),
                                                                "covariance95_radius": standalone_covariance.tolist()},
                    "scalar_metric_records": [{k: row[k] for k in ("time_index", "max_mean_error_in_reference_std",
                        "max_normalized_covariance_error", "mean_sampling95_simultaneous_radius",
                        "covariance_sampling95_simultaneous_radius", "mean_error_plus_sampling95",
                        "covariance_error_plus_sampling95")} for row in expected],
                    "accuracy_and_sampling_gate_without_reference_refinement": all(r["mean_error_plus_sampling95"] <= .05
                        and r["covariance_error_plus_sampling95"] <= .05
                        and r["mean_sampling95_simultaneous_radius"] <= .0125
                        and r["covariance_sampling95_simultaneous_radius"] <= .0125 for r in expected)})
                result["max_numeric_replay_error"] = max(result["max_numeric_replay_error"], error)
        print(case, "independent metrics and contrasts replayed", flush=True)
    expected_ensemble = deployment["expected_records"]
    actual_ensemble = sum(row["role"] == "ensemble" for row in result["records"])
    expected_conditional = len(runner.case_names(study))*2*study.conditional_anchors*len(study.optimizer_seeds)
    if actual_ensemble != expected_ensemble or len(result["records"])-actual_ensemble != expected_conditional:
        raise AssertionError("Mathematical audit omitted a declared model/group")
    result["all_seed_consistency"] = seed_consistency(result["conditional_contrasts"], study.optimizer_seeds)
    result.update(complete=True, independent_ensemble_metric_records=actual_ensemble,
                  conditional_metric_records=expected_conditional,
                  direct_full_pool_replays=len(result["direct_mixture_replays"]),
                  audit_wall_seconds=time.monotonic()-wall, audit_cpu_seconds=time.process_time()-cpu,
                  timing_workers_may_start_after_process_exit=True,
                  limitations=["Pointwise bootstrap contrast intervals are not family-wise across regimes or seeds.",
                               "5% and1.25% gates remain finite-reference sampling diagnostics, not continuous-bias certificates."])
    _json(root/"audit/metrics_replay.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.root)
    print(json.dumps({key: result[key] for key in ("complete", "independent_ensemble_metric_records",
        "conditional_metric_records", "direct_full_pool_replays", "max_numeric_replay_error", "audit_cpu_seconds")}))


if __name__ == "__main__":
    main()
