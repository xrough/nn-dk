"""Fresh-path mean/raw-second-moment residual transport for frozen v3 heads.

The backward-generator identities are continuum identities. This audit reports
their finite-Euler/stopping/quadrature defect rather than treating a covariance
PDE residual as a prediction-error certificate. No previous final outcomes are
opened, no model is fitted, and all continuation families remain in the audit.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.compiled_native import compile_native  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "fresh-native-mean-raw-second-residual-transport-audit-v1"
HELPERS = ("audit_payoff_mean_gradients.py", "compiled_native.py", "native_euler_control.cpp",
           "run_payoff_boundary_regimes.py", "run_payoff_regimes.py", "payoff_study_adapter.py",
           "audit_payoff_regimes_validation.py")


@dataclass(frozen=True)
class TransportPlan:
    cases: tuple[str, ...] = ()
    anchors_per_profile: int = 2
    query_fractions: tuple[float, ...] = (.25, .5, 1.)
    paths: int = 64
    chunk_paths: int = 8
    step: float = 2**-10
    refinement_factor: int = 2
    residual_stride: int = 16
    jet_batch_size: int = 128
    n_bootstrap: int = 128
    seed: int = 2026103003

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "query_fractions", tuple(self.query_fractions))
        names = ("anchors_per_profile", "paths", "chunk_paths", "refinement_factor",
                 "residual_stride", "jet_batch_size", "n_bootstrap", "seed")
        if (any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in names)
                or self.paths < 2 or self.paths % self.chunk_paths or self.refinement_factor < 2
                or self.residual_stride < 2 or self.residual_stride % 2 or self.n_bootstrap < 2
                or not np.isfinite(self.step) or self.step <= 0
                or not self.query_fractions or not np.isfinite(self.query_fractions).all()
                or min(self.query_fractions) <= 0 or max(self.query_fractions) > 1
                or any(a >= b for a, b in zip(self.query_fractions, self.query_fractions[1:]))
                or len(set(self.cases)) != len(self.cases)):
            raise ValueError("Complete families, even residual stride, nested grids and ordered queries required")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_arrays(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _grid(states, times, stops):
    states, times, stops = map(lambda x: np.asarray(x, dtype=float), (states, times, stops))
    if (states.ndim != 3 or states.shape[0] < 1 or states.shape[1] < 2
            or times.shape != (states.shape[1],) or times[0] != 0
            or not np.isfinite(states).all() or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any() or stops.shape != (len(states),)
            or np.isnan(stops).any() or (stops < 0).any()
            or (np.isfinite(stops) & (stops > times[-1]+1e-12)).any()):
        raise ValueError("Complete finite polygons, increasing zero-origin times and hit times required")
    return states, times, stops


def polygon_running(states, times, stopped_at, running_weights):
    """Exact killed linear-reward integral on each supplied stopped polygon.

    A crossing endpoint is already the interpolated hit state; the segment's
    exposure is the fractional hit duration, not the full numerical time step.
    Frozen terminal states and initially absorbed paths are retained.
    """
    states, times, stops = _grid(states, times, stopped_at)
    weights = np.asarray(running_weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != states.shape[2] or not np.isfinite(weights).all():
        raise ValueError("Finite running weights [output,cell] required")
    exposure = np.minimum(np.diff(times)[None], np.maximum(stops[:, None]-times[None, :-1], 0.))
    values = states @ weights.T
    increments = .5*exposure[..., None]*(values[:, :-1]+values[:, 1:])
    accumulated = np.concatenate((np.zeros((len(states), 1, len(weights))),
                                  np.cumsum(increments, axis=1)), axis=1)
    return accumulated, exposure


def raw_second_residual(mean, accumulated_running, mean_residual, covariance_residual):
    """r_C + (u+A)r_mu^T + r_mu(u+A)^T; this is not r_C alone."""
    u, a, r, c = (np.asarray(v, dtype=float) for v in
                  (mean, accumulated_running, mean_residual, covariance_residual))
    if (u.shape != a.shape or u.shape != r.shape or u.ndim < 1
            or c.shape != (*u.shape, u.shape[-1])
            or any(not np.isfinite(v).all() for v in (u, a, r, c))):
        raise ValueError("Compatible finite means, accumulated rewards and residuals required")
    total = u+a
    return c + total[..., :, None]*r[..., None, :] + r[..., :, None]*total[..., None, :]


def integrate_residual(values, times, stopped_at):
    """Left-generator quadrature on explicit nodes, ending at each hit time.

    ``values`` contains the left value for every supplied interval. This does
    not claim that sparse-node quadrature is exact along the path.
    """
    values, times, stops = (np.asarray(v, dtype=float) for v in (values, times, stopped_at))
    if (values.ndim < 2 or len(values) < 1 or times.ndim != 1 or len(times) != values.shape[1]+1
            or len(times) < 2 or times[0] != 0 or (np.diff(times) <= 0).any()
            or stops.shape != (len(values),) or np.isnan(stops).any() or (stops < 0).any()
            or not np.isfinite(values).all() or not np.isfinite(times).all()):
        raise ValueError("Finite left residuals, complete increasing nodes and per-family hit times required")
    exposure = np.minimum(np.diff(times)[None], np.maximum(stops[:, None]-times[None, :-1], 0.))
    return np.sum(values*exposure.reshape(*exposure.shape, *((1,)*(values.ndim-2))), axis=1)


def _moments(outcomes, counts):
    n = counts.sum()
    mean = np.einsum("r,rqo->qo", counts, outcomes)/n
    second = np.einsum("r,rqi,rqj->qij", counts, outcomes, outcomes)/n
    covariance = n/(n-1)*(second-mean[..., :, None]*mean[..., None, :])
    return mean, second, covariance


def summarize_transport(initial_mean, initial_covariance, outcomes, mean_integral,
                        raw_second_integral, covariance_integral, *, bootstrap_counts=None,
                        return_bootstrap=False):
    """Whole-family paired errors, transported residuals and observed defect.

    Raw second moments use E[YY^T]; the separately labelled sample covariance
    uses the n-1 denominator. Centering raw moments is not silently substituted
    for transporting r_C. No error enclosure or accuracy certificate is made.
    """
    u, c, y, im, ir, ic = (np.asarray(v, dtype=float) for v in
                          (initial_mean, initial_covariance, outcomes, mean_integral,
                           raw_second_integral, covariance_integral))
    if (y.ndim != 3 or len(y) < 2 or u.shape != y.shape[1:]
            or c.shape != (*u.shape, u.shape[-1]) or im.shape != y.shape
            or ir.shape != (*y.shape, y.shape[-1]) or ic.shape != ir.shape
            or any(not np.isfinite(v).all() for v in (u, c, y, im, ir, ic))):
        raise ValueError("At least two complete families and compatible finite transport arrays required")
    if not np.allclose(c, c.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError("Symmetric initial model covariance required")

    def metrics(counts):
        mean, second, covariance = _moments(y, counts)
        mean_error = u-mean
        raw_error = c+u[..., :, None]*u[..., None, :]-second
        transported_mean = np.einsum("r,rqo->qo", counts, im)/counts.sum()
        transported_second = np.einsum("r,rqij->qij", counts, ir)/counts.sum()
        covariance_only = np.einsum("r,rqij->qij", counts, ic)/counts.sum()
        return {"mean_discrepancy": mean_error, "raw_second_discrepancy": raw_error,
                "sample_covariance_discrepancy_unbiased": c-covariance,
                "mean_residual_integral": transported_mean,
                "raw_second_residual_integral": transported_second,
                "covariance_residual_integral_alone": covariance_only,
                "mean_transport_defect": mean_error-transported_mean,
                "raw_second_transport_defect": raw_error-transported_second}

    point = metrics(np.ones(len(y)))
    result = {"families": len(y), "point": {k:v.tolist() for k,v in point.items()},
              "units": "internal normalized sqrt(N) payoffs; covariance/raw seconds in corresponding products",
              "covariance_residual_alone_is_prediction_error_certificate": False,
              "transport_defect": "observed finite-Euler/generator, stopping and quadrature remainder plus sampling error"}
    if bootstrap_counts is not None:
        counts = np.asarray(bootstrap_counts)
        if (counts.ndim != 2 or counts.shape[1] != len(y) or len(counts) < 2
                or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.equal(counts, np.rint(counts)).all() or not np.all(counts.sum(-1) == len(y))):
            raise ValueError("Whole-family bootstrap frequency rows must each retain n draws")
        boot = [metrics(row) for row in counts]
        result["pointwise_percentile95"] = {
            key: np.quantile(np.stack([row[key] for row in boot]), [.025, .975], axis=0).tolist()
            for key in point}
        result["bootstrap_scope"] = "whole continuation families; all queries/outputs/residual terms paired; pointwise only"
        if return_bootstrap:
            result["bootstrap_samples"] = {key:np.stack([row[key] for row in boot]) for key in point}
    return result


def _study(study_root, cases):
    from analysis.audit_payoff_mean_gradients import _inventory

    return _inventory(study_root, cases)


def prepare(root, study_root, plan=None):
    """Freeze validation-only anchors and all model/RNG/source identities."""
    from analysis.audit_payoff_mean_gradients import select_anchors
    from analysis.audit_payoff_regimes_validation import validation_metadata

    root, study_root = Path(root).resolve(), Path(study_root).resolve()
    plan = TransportPlan() if plan is None else plan
    if not isinstance(plan, TransportPlan):
        raise TypeError("TransportPlan required")
    runner, study, cases, models = _study(study_root, plan.cases)
    identity = {"kind": KIND, "plan": asdict(plan), "cases": list(cases), "models": models,
                "study_root": str(study_root), "source_hash": source_hash(), "script_sha256": sha(__file__),
                "helpers": {name:sha(Path(__file__).with_name(name)) for name in HELPERS},
                "fit_lock_sha256": sha(study_root/"payoffs/fit_lock.json"),
                "stream": "fresh-residual-transport-continuations-no-old-final-outcomes-v1",
                "old_final_reference_outcomes_opened": False, "fitting_or_model_selection": False,
                "anchor_policy": "validation-only active unique origins; profile-wise minimum-to-median mass ranks",
                "residual_quadrature": "left sums with fractional absorption exposure; stride and half-stride",
                "refinement_quadrature": "same physical residual nodes on coupled coarse/fine Euler polygons"}
    identity = json.loads(json.dumps(identity))
    if (root/"plan.json").exists():
        saved = _validate(root)
        if saved["identity"] != identity:
            raise ValueError("Frozen diagnostic identity changed")
        return saved
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty diagnostic directory")
    anchors = {}
    for case in cases:
        cfg = runner.configuration(case, study)
        queries = cfg.horizon*np.asarray(plan.query_fractions)
        if not np.allclose(queries/plan.step, np.rint(queries/plan.step), atol=1e-10, rtol=0):
            raise ValueError("Queries must lie on the native grid")
        bank = runner.load_state_bank(study_root, case, "validation")
        values = select_anchors(bank, validation_metadata(bank), cfg.spec, plan.anchors_per_profile)
        path = root/"anchors"/(case+".npz")
        save_arrays(path, values)
        anchors[case] = {"path": str(path), "sha256": sha(path), "rows": len(values["states"]),
                         "queries": queries.tolist(), "state_bank_path": str(bank["path"]),
                         "state_bank_sha256": sha(bank["path"])}
    saved = {"identity": identity, "identity_hash": stable_hash(identity), "anchors": anchors}
    write(root/"plan.json", saved)
    return saved


def _validate(root):
    saved = read(Path(root)/"plan.json")
    identity = saved["identity"]
    if (identity["kind"] != KIND or stable_hash(identity) != saved["identity_hash"]
            or identity["source_hash"] != source_hash() or identity["script_sha256"] != sha(__file__)
            or any(sha(Path(__file__).with_name(n)) != h for n,h in identity["helpers"].items())
            or sha(Path(identity["study_root"])/"payoffs/fit_lock.json") != identity["fit_lock_sha256"]):
        raise ValueError("Frozen diagnostic source, helpers or fit lock changed")
    for item in identity["models"]:
        if sha(item["checkpoint_path"]) != item["checkpoint_sha256"] or sha(item["metadata_path"]) != item["metadata_sha256"]:
            raise ValueError("Frozen checkpoint changed")
    for item in saved["anchors"].values():
        if sha(item["path"]) != item["sha256"] or sha(item["state_bank_path"]) != item["state_bank_sha256"]:
            raise ValueError("State-only anchors or validation bank changed")
    return saved


def _jet_values(model, states, remaining, batch_size):
    rows = {key:[] for key in ("mean", "covariance", "r_mean", "r_covariance")}
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            x = torch.tensor(states[start:start+batch_size], dtype=torch.float64)
            t = torch.tensor(remaining[start:start+batch_size], dtype=torch.float64)
            mean, covariance = model._moment_jets(x, t)
            source = mean.face_derivatives@mean.face_derivatives.transpose(-1, -2)
            running = math.sqrt(model.context.n_particles)*(x@model.normalized_running_weights.T)
            running = torch.where(model._active(x)[:, None], running, 0.)
            values = (mean.value, covariance.value, mean.time_derivative-mean.generator-running,
                      covariance.time_derivative-covariance.generator-source)
            for key, value in zip(rows, values, strict=True):
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Every family's transported model residual must remain finite")
                rows[key].append(value.numpy())
    return {key:np.concatenate(values) for key,values in rows.items()}


def transport_on_paths(model, packet, dt, queries, stride, batch_size=128):
    """Evaluate both residual strides on a single complete native polygon."""
    if type(stride) is not int or stride < 2 or stride % 2 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Even residual stride and positive jet batch size required")
    states, stops = np.asarray(packet["states"]), np.asarray(packet["stopped_at"])
    times = np.arange(states.shape[1])*dt
    _grid(states, times, stops)
    if not np.array_equal(states[:, 0], np.broadcast_to(states[0, 0], states[:, 0].shape)):
        raise ValueError("One supplied conditional anchor per transport polygon batch required")
    queries = np.asarray(queries, dtype=float)
    indices = np.rint(queries/dt).astype(int)
    if (queries.ndim != 1 or not len(queries) or (indices < 1).any() or (indices >= len(times)).any()
            or (np.diff(queries) <= 0).any() or not np.allclose(indices*dt, queries, atol=1e-12, rtol=0)):
        raise ValueError("Positive increasing queries on the complete polygon grid required")
    weights = math.sqrt(model.context.n_particles)*model.normalized_running_weights.detach().numpy()
    accumulated, _ = polygon_running(states, times, stops, weights)
    result = {key:[] for key in ("initial_mean", "initial_covariance", "mean_integral",
                                "raw_second_integral", "covariance_integral", "half_mean_integral",
                                "half_raw_second_integral", "half_covariance_integral")}
    for query, last in zip(queries, indices, strict=True):
        coarse_nodes = np.unique(np.r_[np.arange(0, last, stride), last]).astype(int)
        nodes = np.unique(np.r_[np.arange(0, last, stride//2), last]).astype(int)
        rows = states[:, nodes[:-1]].reshape(-1, states.shape[-1])
        remaining = np.broadcast_to(query-times[nodes[:-1]], (len(states), len(nodes)-1)).reshape(-1)
        values = _jet_values(model, rows, remaining, batch_size)
        values = {key: v.reshape(len(states), len(nodes)-1, *v.shape[1:]) for key,v in values.items()}
        raw = raw_second_residual(values["mean"], accumulated[:, nodes[:-1]],
                                  values["r_mean"], values["r_covariance"])
        result["initial_mean"].append(values["mean"][0, 0])
        result["initial_covariance"].append(values["covariance"][0, 0])
        positions = np.searchsorted(nodes, coarse_nodes[:-1])
        for prefix, positions_, time_nodes in (("", positions, coarse_nodes),
                                               ("half_", np.arange(len(nodes)-1), nodes)):
            for key, value in (("mean_integral", values["r_mean"]),
                               ("raw_second_integral", raw),
                               ("covariance_integral", values["r_covariance"])):
                result[prefix+key].append(integrate_residual(value[:, positions_], times[time_nodes], stops))
    return {key:np.stack(value, axis=0 if key.startswith("initial_") else 1)
            for key,value in result.items()}


def _native_chunk(saved, plan, case, anchor, state, cfg, models, queries, start, backend):
    factor, count, steps = plan.refinement_factor, plan.chunk_paths, round(queries[-1]/plan.step)
    fine, families = [], []
    for family in range(start, start+count):
        key = [KIND, saved["identity_hash"], case, anchor, family, "new-physical-continuation"]
        rng = seeded_rng(plan.seed, *key)
        fine.append(rng.normal(0, math.sqrt(plan.step/factor), (steps*factor, cfg.n_cells)))
        families.append(stable_hash([plan.seed, *key]))
    fine = np.asarray(fine)
    coarse = fine.reshape(count, steps, factor, cfg.n_cells).sum(2)
    initial = np.broadcast_to(state, (count, cfg.n_cells)).copy()
    result = {"family_ids":np.asarray(families), "fine_brownian":fine, "initial":initial}
    for name, noise, dt, stride in (("coarse", coarse, plan.step, plan.residual_stride),
                                    ("fine", fine, plan.step/factor, plan.residual_stride*factor)):
        packet = backend(initial, noise, cfg.physics, cfg.spec, dt)
        path, stops = packet["states"], packet["stopped_at"]
        times = np.arange(path.shape[1])*dt
        result["states_"+name], result["stopped_at_"+name] = path, stops
        for seed, model in models.items():
            if (model.spec != cfg.spec or any(getattr(model.context,key)!=getattr(cfg.physics,key)
                                             for key in ("n_particles","diffusion","kappa","a2"))):
                raise ValueError("Historical model and native path physics differ")
            full = model.bank.evaluate_paths(path, times, stops)[:, np.rint(queries/dt).astype(int)]
            result[f"seed{seed}_outcomes_"+name] = full*math.sqrt(cfg.n_particles)/model.bank.scales
            values = transport_on_paths(model, packet, dt, queries, stride, plan.jet_batch_size)
            result.update({f"seed{seed}_{key}_{name}":v for key,v in values.items()})
    return result


def _summary(root, saved, plan, chunks):
    records = []
    for case, item in saved["anchors"].items():
        with np.load(item["path"]) as origins:
            profiles = origins["profiles"].copy()
        seeds = [v["seed"] for v in saved["identity"]["models"] if v["case"] == case]
        for anchor in range(item["rows"]):
            selected = [v for v in chunks if (v["case"],v["anchor"]) == (case,anchor)]
            stores = []
            for chunk in selected:
                with np.load(chunk["path"]) as store:
                    stores.append({key:store[key] for key in store.files})
            ids = np.concatenate([v["family_ids"] for v in stores])
            if len(ids) != plan.paths or len(set(ids.tolist())) != plan.paths:
                raise ValueError("Independent family allocation is incomplete or duplicated")
            rng = seeded_rng(plan.seed, KIND, case, anchor, "paired-whole-family-bootstrap")
            counts = rng.multinomial(plan.paths, np.full(plan.paths,1/plan.paths), size=plan.n_bootstrap)
            for seed in seeds:
                row = {"case":case, "anchor":anchor, "profile":str(profiles[anchor]), "seed":seed,
                       "queries":item["queries"], "families":plan.paths, "levels":{}}
                bootstraps = {}
                for level in ("coarse", "fine"):
                    def joined(key):
                        values = [v[f"seed{seed}_{key}_{level}"] for v in stores]
                        if key.startswith("initial_"):
                            if any(not np.array_equal(value,values[0]) for value in values[1:]):
                                raise ValueError("Frozen initial prediction differs between chunks")
                            return values[0]
                        return np.concatenate(values)

                    summaries = {}
                    for prefix in ("", "half_"):
                        summaries[prefix or "declared"] = summarize_transport(
                            joined("initial_mean"), joined("initial_covariance"), joined("outcomes"),
                            joined(prefix+"mean_integral"), joined(prefix+"raw_second_integral"),
                            joined(prefix+"covariance_integral"), bootstrap_counts=counts,
                            return_bootstrap=True)
                        bootstraps[(level,prefix or "declared")] = summaries[prefix or "declared"].pop("bootstrap_samples")
                    row["levels"][level] = summaries
                    row[level+"_stopped_families"] = int(sum(np.isfinite(v["stopped_at_"+level]).sum() for v in stores))
                    row[level+"_maximum_mass_error"] = float(max(abs(v["states_"+level].sum(-1)-1).max() for v in stores))
                coarse_stops = np.concatenate([v["stopped_at_coarse"] for v in stores])
                fine_stops = np.concatenate([v["stopped_at_fine"] for v in stores])
                both = np.isfinite(coarse_stops)&np.isfinite(fine_stops)
                row["paired_stopping_diagnostics"] = {
                    "coarse_only":int((np.isfinite(coarse_stops)&~np.isfinite(fine_stops)).sum()),
                    "fine_only":int((~np.isfinite(coarse_stops)&np.isfinite(fine_stops)).sum()),
                    "both":int(both.sum()),
                    "maximum_hit_time_difference_on_both":float(abs(coarse_stops[both]-fine_stops[both]).max()) if both.any() else None}
                row["paired_refinement_and_halfstride"] = {}
                for key in ("mean_discrepancy", "raw_second_discrepancy", "mean_residual_integral",
                            "raw_second_residual_integral", "mean_transport_defect", "raw_second_transport_defect"):
                    a = np.asarray(row["levels"]["coarse"]["declared"]["point"][key])
                    b = np.asarray(row["levels"]["fine"]["declared"]["point"][key])
                    c = np.asarray(row["levels"]["coarse"]["half_"]["point"][key])
                    d = np.asarray(row["levels"]["fine"]["half_"]["point"][key])
                    row["paired_refinement_and_halfstride"][key] = {
                        "fine_minus_coarse_same_physical_residual_nodes":(b-a).tolist(),
                        "halfstride_minus_stride_on_coarse_Euler":(c-a).tolist(),
                        "halfstride_minus_stride_on_fine_Euler":(d-b).tolist(),
                        "fine_minus_coarse_pointwise_percentile95":np.quantile(
                            bootstraps[("fine","declared")][key]-bootstraps[("coarse","declared")][key],
                            [.025,.975],axis=0).tolist(),
                        "coarse_halfstride_difference_pointwise_percentile95":np.quantile(
                            bootstraps[("coarse","half_")][key]-bootstraps[("coarse","declared")][key],
                            [.025,.975],axis=0).tolist(),
                        "fine_halfstride_difference_pointwise_percentile95":np.quantile(
                            bootstraps[("fine","half_")][key]-bootstraps[("fine","declared")][key],
                            [.025,.975],axis=0).tolist()}
                records.append(row)
    result = {"complete":True, "kind":KIND, "identity_hash":saved["identity_hash"], "records":records,
              "mean_identity":"u0-E[Y] = E integral r_mu only for the continuum ideal",
              "raw_second_identity":"C0+u0*u0^T-E[Y*Y^T] = E integral [r_C+(u+A)r_mu^T+r_mu(u+A)^T] only for the continuum ideal",
              "scope":"fresh diagnostic continuations from validation-only anchors, all frozen optimizer seeds",
              "sampling":"whole independent families paired across both Euler levels, both residual strides, and all seeds",
              "not_certified":["continuous-time stopping bias", "generator-versus-Euler defect enclosure",
                               "residual quadrature enclosure", "simultaneous accuracy", "covariance residual alone as prediction error"],
              "old_final_reference_outcomes_opened":False, "fitting_or_selection":False}
    write(root/"summary.json",result)
    return result


def _run(root, max_chunks, budget_seconds):
    saved = _validate(root)
    plan = TransportPlan(**saved["identity"]["plan"])
    runner, study, _, _ = _study(Path(saved["identity"]["study_root"]), plan.cases)
    deadline, chunks, created, backend, stopped_by = time.monotonic()+budget_seconds, [], 0, None, None
    for case,item in saved["anchors"].items():
        cfg, models = runner.configuration(case,study), None
        with np.load(item["path"]) as origins:
            states = origins["states"].copy()
        for anchor,state in enumerate(states):
            for start in range(0,plan.paths,plan.chunk_paths):
                path = root/"chunks"/case/f"anchor{anchor}_{start}.npz"
                metadata = path.with_suffix(".json")
                expected = {"path":str(path), "case":case, "anchor":anchor, "start":start,
                            "families":plan.chunk_paths, "identity_hash":saved["identity_hash"]}
                if metadata.exists():
                    record = read(metadata)
                    if any(record.get(k)!=v for k,v in expected.items()) or sha(path)!=record["sha256"]:
                        raise ValueError("Committed transport chunk changed")
                    chunks.append(record)
                    continue
                if path.exists():
                    raise ValueError("Uncommitted transport chunk cannot be accepted")
                if time.monotonic() >= deadline or (max_chunks is not None and created >= max_chunks):
                    stopped_by = "time-budget" if time.monotonic() >= deadline else "chunk-budget"
                    continue
                if backend is None:
                    backend = compile_native(root/"compiled")
                if models is None:
                    models = {v["seed"]:runner.load_declared_model(v) for v in saved["identity"]["models"] if v["case"]==case}
                values = _native_chunk(saved,plan,case,anchor,state,cfg,models,np.asarray(item["queries"]),start,backend)
                save_arrays(path,values)
                record = {**expected,"sha256":sha(path)}
                write(metadata,record)
                chunks.append(record)
                created += 1
    expected_count = sum(v["rows"] for v in saved["anchors"].values())*(plan.paths//plan.chunk_paths)
    index = {"complete":len(chunks)==expected_count,"identity_hash":saved["identity_hash"],
             "expected_chunks":expected_count,"chunks":chunks,"stopped_by":stopped_by}
    if index["complete"]:
        if (root/"index.json").exists() and read(root/"index.json").get("complete"):
            load_complete(root)
            return read(root/"index.json")
        _summary(root,saved,plan,chunks)
        index["summary_sha256"] = sha(root/"summary.json")
    write(root/"index.json",index)
    return index


def run(root, max_chunks=None, budget_seconds=8*60*60):
    root = Path(root).resolve()
    if (max_chunks is not None and (type(max_chunks) is not int or max_chunks<0)
            or not np.isfinite(budget_seconds) or not 0<budget_seconds<=8*60*60):
        raise ValueError("Nonnegative chunk budget and finite positive <=8h time budget required")
    if not (root/"plan.json").exists():
        raise ValueError("Prepare the transport diagnostic first")
    with (root/".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this transport diagnostic") from error
        try:
            return _run(root,max_chunks,budget_seconds)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def load_complete(root):
    root = Path(root).resolve()
    saved = _validate(root)
    index = read(root/"index.json")
    plan = TransportPlan(**saved["identity"]["plan"])
    expected = {(case,anchor,start) for case,item in saved["anchors"].items()
                for anchor in range(item["rows"]) for start in range(0,plan.paths,plan.chunk_paths)}
    actual = [(v["case"],v["anchor"],v["start"]) for v in index["chunks"]]
    if (not index["complete"] or index["identity_hash"]!=saved["identity_hash"]
            or len(actual)!=len(expected) or set(actual)!=expected or index["expected_chunks"]!=len(expected)):
        raise ValueError("All declared transport chunks must complete")
    for item in index["chunks"]:
        if sha(item["path"])!=item["sha256"]:
            raise ValueError("Committed transport chunk changed")
    if sha(root/"summary.json")!=index["summary_sha256"]:
        raise ValueError("Transport summary changed")
    return read(root/"summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","verify"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--study-root",type=Path,default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--max-chunks",type=int)
    parser.add_argument("--budget-seconds",type=float,default=8*60*60)
    args = parser.parse_args()
    if args.stage=="prepare":
        result = prepare(args.root,args.study_root,TransportPlan(**read(args.plan)) if args.plan else None)
        print(json.dumps({"prepared":True,"identity_hash":result["identity_hash"]}))
    elif args.stage=="run":
        result = run(args.root,args.max_chunks,args.budget_seconds)
        print(json.dumps({k:result[k] for k in ("complete","expected_chunks","stopped_by")}))
        return 75 if result.get("stopped_by")=="time-budget" and not result["complete"] else 0
    else:
        result = load_complete(args.root)
        print(json.dumps({"complete":result["complete"],"records":len(result["records"])}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
