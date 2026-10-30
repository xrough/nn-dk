"""Frozen natural-validation residual decomposition for all crossed fits.

This audit reads learner validation states and completed checkpoints only. It
draws no paths, reads no final outcomes, and never changes a fitted network.
Row weights are uniform over the original natural validation population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import run_payoff_collocation_study as study_runner  # noqa: E402
from analysis.payoff_collocation_sampling import boundary_time_strata  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "natural-validation-all-fit-collocation-residual-decomposition-v1"
MEAN_KEYS = ("mu_dt", "Lmu", "running_source", "rmu", "mu_transport")
COVARIANCE_KEYS = ("C_dt", "LC", "Gamma", "rC", "C_transport", "Gamma_common", "rC_common")
COMPONENT_KEYS = MEAN_KEYS+COVARIANCE_KEYS
HELPERS = ("run_payoff_collocation_study.py", "run_payoff_regimes.py", "fit_payoff_collocation_ablation.py",
           "payoff_collocation_sampling.py", "report_payoff_collocation_residuals.py",
           "report_lna_payoff_correction.py", "run_lna_payoff_correction_study.py",
           "payoff_study_adapter.py", "report_payoff_regimes.py", "evaluate_payoff_regimes.py",
           "evaluate_path_generalization.py", "benchmark_path_generalization.py",
           "benchmark_payoff_generalization.py", "native_gaussian_baseline.py",
           "path_generalization_reference.py", "probe_path_workloads.py", "run_payoff_repair.py")
UNITS = {"native_mean": "u=sqrt(N)*physical_mean/bank.scales",
    "native_covariance": "K=N*physical_covariance/outer(bank.scales,bank.scales)",
    "physical_mean": "native term * bank.scales/sqrt(N)",
    "physical_covariance": "native term * outer(bank.scales,bank.scales)/N; NOT N-scaled physical covariance",
    "dimensionless": "T*native term/stored train-only stage loss scale",
    "validation_measure": "literal natural rows, weight1/Nrows; no proposal importance weights",
    "counterfactual": "same-seed uniform/pseudo-Huber selected mean Gamma substituted by subtraction only; diagnostic, not common-mean training",
    "covariance_MSE": "all16entries, including both symmetric off-diagonals",
    "inference": "descriptive validation population; no resampling intervals or significance"}


@dataclass(frozen=True)
class ResidualAuditPlan:
    chunk_rows: int = 128
    decomposition_atol: float = 1e-10
    decomposition_rtol: float = 1e-11

    def __post_init__(self):
        if (type(self.chunk_rows) is not int or self.chunk_rows < 1
                or not np.isfinite((self.decomposition_atol, self.decomposition_rtol)).all()
                or self.decomposition_atol <= 0 or self.decomposition_rtol < 0):
            raise ValueError("Positive row chunks and declared finite decomposition tolerances required")

    def to_dict(self):
        return asdict(self)


def strata_labels(config):
    if config.distance_edges != (.25, 1., 4.) or config.time_edges != (.25, .5, .75):
        raise ValueError("Retain the declared full nineteen-stratum distance/time design")
    return ("absorbed/zero-time", "absorbed/positive-time", "interior/zero-time", *(
        f"active/time-{t}/distance-{d}" for t in range(4) for d in range(4)))


def sha(path):
    return study_runner._sha(Path(path))


def read(path):
    return study_runner._read(Path(path))


def write(path, value):
    study_runner._json(Path(path), value)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def helper_paths():
    return tuple(Path(__file__).with_name(name) for name in HELPERS)


def _rows(root, case, study):
    config = study_runner.ablation_configuration(next(s for s in study_runner.slots(study)
                                                     if s["case"] == case), study)
    bank = study_runner.flat_bank(root, case, "validation", study, config)
    if set(bank["row_kinds"]) != {"native"} or bank["future_payoff_labels"]:
        raise ValueError("Residual audit accepts only the unchanged natural validation rows")
    labels = boundary_time_strata(bank["states"], bank["horizons"], config.strata)
    if not set(labels) <= set(strata_labels(config.strata)):
        raise ValueError("Undeclared residual-validation stratum")
    ids = [stable_hash([case, "validation-row", i, f, p, array_sha(m), float(t)])
        for i, (f, p, m, t) in enumerate(zip(bank["family_ids"], bank["parent_family_ids"],
            bank["states"], bank["horizons"], strict=True))]
    return {"states": bank["states"], "horizons": bank["horizons"],
        "row_ids": np.asarray(ids, dtype="S64"), "family_ids": np.asarray(bank["family_ids"], dtype="S64"),
        "parent_family_ids": np.asarray(bank["parent_family_ids"], dtype="S64"), "strata": labels}, bank, config


def _identity(study_root, plan):
    study_root = Path(study_root).resolve()
    study, _ = study_runner._prepared(study_root)
    locked = study_runner.verify_fit_lock(study_root)
    if len(locked["models"]) != len(study_runner.slots(study)):
        raise ValueError("Every declared crossed fit must finish before any residual audit")
    hashes = {}
    for path in helper_paths():
        if not path.is_file() or not path.read_bytes().strip():
            raise ValueError("Residual auditor and reporter helper sources must exist before freezing")
        hashes[path.name] = sha(path)
    pools = {}
    for case in study_runner.case_names(study):
        rows, bank, config = _rows(study_root, case, study)
        pools[case] = {"rows": len(rows["states"]), "bank_content_sha256": bank["content_sha256"],
            "arrays": {name: array_sha(value) for name, value in rows.items()},
            "strata_config": config.strata.to_dict(), "strata_labels": list(strata_labels(config.strata))}
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": sha(__file__),
        "helpers": hashes, "study_root": str(study_root), "plan": plan.to_dict(),
        "fit_lock_sha256": sha(study_root/"payoffs/fit_lock.json"),
        "data_lock_sha256": sha(study_root/"payoffs/data_lock.json"),
        "flat_pool_lock_sha256": sha(study_root/"payoffs/flat_pool_lock.json"),
        "study_preregistration_sha256": sha(study_root/"payoffs/preregistration.json"),
        "study_plan_sha256": sha(study_root/"plan.json"), "models": locked["models"], "pools": pools,
        "units": UNITS, "absolute_quantiles": [.5, .9, .99],
        "unique_covariance_indices": np.transpose(np.triu_indices(4)).tolist(),
        "unique_covariance_multiplicity": [1 if i == j else 2 for i, j in zip(*np.triu_indices(4), strict=True)],
        "covariance_canonical_entries": 16, "final_outcomes_or_references_read": False,
        "selection": "all locked slots, all original ordered natural validation rows; bins depend only on state/time"}
    return study, locked, identity


def _save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as sink:
        np.savez_compressed(sink, **arrays)
    temporary.replace(path)


def prepare(study_root, audit_root, plan=None):
    started, cpu = time.monotonic(), time.process_time()
    plan = ResidualAuditPlan() if plan is None else plan
    audit_root, study_root = Path(audit_root).resolve(), Path(study_root).resolve()
    if audit_root == study_root or study_root.is_relative_to(audit_root) or audit_root.is_relative_to(study_root):
        raise ValueError("Use a distinct residual-audit root outside the authoritative study")
    with study_runner._exclusive(audit_root/"execution/.prepare.lock"):
        costs = _prepare_costs(audit_root, study_root)
        attempt = {"started_unix": time.time(), "pid": os.getpid(), "finished": False, "complete": False}
        costs["attempts"].append(attempt)
        write(audit_root/"execution/prepare_costs.json", {**costs, "content_sha256": stable_hash(costs)})
        try:
            study, _, identity = _identity(study_root, plan)
            packet = {"identity": identity, "identity_hash": stable_hash(identity)}
            study_runner._immutable(audit_root/"plan.json", packet)
            records = {}
            for case in study_runner.case_names(study):
                rows, _, _ = _rows(study_root, case, study)
                path = audit_root/"inputs"/(case+".npz")
                if not path.exists():
                    _save_arrays(path, rows)
                with np.load(path, allow_pickle=False) as saved:
                    actual = {key: array_sha(saved[key]) for key in saved.files}
                if actual != identity["pools"][case]["arrays"]:
                    raise ValueError("Ordered residual input states, identifiers or strata changed")
                records[case] = {"path": str(path), "sha256": sha(path)}
            study_runner._immutable(audit_root/"inputs/manifest.json", records)
            attempt["complete"] = True
        except BaseException as error:
            attempt["error"] = type(error).__name__+": "+str(error)
            raise
        finally:
            attempt.update(finished=True, finished_unix=time.time(), wall_seconds=time.monotonic()-started,
                           cpu_seconds=time.process_time()-cpu)
            write(audit_root/"execution/prepare_costs.json", {**costs, "content_sha256": stable_hash(costs)})
    return packet


def _prepare_costs(root, study_root):
    path = Path(root)/"execution/prepare_costs.json"
    identity = {"study_root": str(Path(study_root).resolve()), "source_hash": source_hash(),
                "script_sha256": sha(__file__)}
    if not path.exists():
        return {"identity": identity, "attempts": [],
                "scope": "entire prepare API invocation including source/model/input integrity and snapshot IO; imports/startup excluded; separate validation audit overhead"}
    value = read(path)
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("identity") != identity or value.get("content_sha256") != stable_hash(payload):
        raise ValueError("Residual preparation cost identity or checksum changed")
    return payload


def _checked_prepare_costs(root, study_root):
    costs = _prepare_costs(root, study_root)
    if (not costs["attempts"] or not any(a.get("complete") for a in costs["attempts"])
            or any(a.get("finished") is not True or any(not isinstance(a.get(k), (float, int))
                or not np.isfinite(a[k]) or a[k] < 0 for k in ("wall_seconds", "cpu_seconds"))
                for a in costs["attempts"])):
        raise ValueError("Every residual preparation attempt requires its measured paid cost")
    return costs


def _prepared(root):
    root = Path(root).resolve()
    packet = read(root/"plan.json")
    plan = ResidualAuditPlan(**packet["identity"]["plan"])
    study, locked, identity = _identity(packet["identity"]["study_root"], plan)
    if packet != {"identity": identity, "identity_hash": stable_hash(identity)}:
        raise ValueError("Frozen residual source, model inventory, scale, pool or row identity changed")
    _checked_prepare_costs(root, identity["study_root"])
    rows = read(root/"inputs/manifest.json")
    if set(rows) != set(identity["pools"]):
        raise ValueError("Incomplete natural-validation input inventory")
    expected_inputs = {"manifest.json"}|{case+".npz" for case in rows}
    actual_inputs = {str(p.relative_to(root/"inputs")) for p in (root/"inputs").rglob("*") if p.is_file()}
    if actual_inputs != expected_inputs:
        raise ValueError("Orphan or undeclared residual input snapshot")
    for case, record in rows.items():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Natural residual row snapshot changed")
        with np.load(record["path"], allow_pickle=False) as arrays:
            if {key: array_sha(arrays[key]) for key in arrays.files} != identity["pools"][case]["arrays"]:
                raise ValueError("Natural residual row content changed")
    return root, study, locked, packet, rows


def convert_components(native, model):
    result = dict(native)
    scales = np.asarray(model.bank.scales)
    n, horizon = model.context.n_particles, model.horizon
    for name in COMPONENT_KEYS:
        is_mean = name in MEAN_KEYS
        physical = scales/math.sqrt(n) if is_mean else np.outer(scales, scales)/n
        loss_scale = np.asarray(model.loss_scaling[("mean" if is_mean else "covariance")+"_scale"])
        result["physical_"+name] = native[name]*physical
        result["dimensionless_"+name] = horizon*native[name]/loss_scale
    return result


def extract_components(model, states, horizons, common_mean_model, plan=None):
    plan = ResidualAuditPlan() if plan is None else plan
    before = study_runner.engine._digest(model.state_dict())
    common_before = study_runner.engine._digest(common_mean_model.state_dict())
    initial = torch.as_tensor(states, dtype=torch.float64)
    times = torch.as_tensor(horizons, dtype=torch.float64)
    with torch.no_grad():
        initial, times = model._inputs(initial, times)
        origins, coefficients = model._origins(initial, times), model._time_coefficients(times)
        mean = model._mean_jet(initial, times, origins, coefficients)
        covariance = model._covariance_jet(initial, times, origins, coefficients)
        running = math.sqrt(model.context.n_particles)*(initial @ model.normalized_running_weights.T)
        running = torch.where(model._active(initial)[..., None], running, 0.)
        gamma = mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2)
        common = mean if common_mean_model is model else common_mean_model._mean_jet(initial, times)
        common_gamma = common.face_derivatives @ common.face_derivatives.transpose(-1, -2)
        native = {"mu_dt": mean.time_derivative, "Lmu": mean.generator, "running_source": running,
            "rmu": mean.time_derivative-mean.generator-running,
            "mu_transport": mean.time_derivative-mean.generator,
            "C_dt": covariance.time_derivative, "LC": covariance.generator, "Gamma": gamma,
            "rC": covariance.time_derivative-covariance.generator-gamma,
            "C_transport": covariance.time_derivative-covariance.generator,
            "Gamma_common": common_gamma,
            "rC_common": covariance.time_derivative-covariance.generator-common_gamma}
        canonical = {"mean": model.mean_residual(initial, times), "covariance": model.covariance_residual(initial, times)}
        errors = {}
        for stage, key in (("mean", "rmu"), ("covariance", "rC")):
            torch.testing.assert_close(native[key], canonical[stage], atol=plan.decomposition_atol,
                                       rtol=plan.decomposition_rtol)
            errors[stage] = float((native[key]-canonical[stage]).abs().max())
        arrays = {key: value.detach().cpu().numpy() for key, value in native.items()}
    if (before != study_runner.engine._digest(model.state_dict())
            or common_before != study_runner.engine._digest(common_mean_model.state_dict())):
        raise ValueError("Residual/source extraction changed fitted network weights or buffers")
    result = convert_components(arrays, model)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite residual component; every validation row must remain")
    return result, errors


def natural_metrics(normalized, delta):
    value = np.asarray(normalized, dtype=float)
    squared = value*value
    huber = 2*squared/(np.sqrt(1+(value/delta)**2)+1)
    return {"natural_mse": float(squared.mean()), "natural_huber": float(huber.mean())}


def _chunk_path(root, name, start):
    return Path(root)/"chunks"/name/f"rows{start}.npz"


def _chunk_record(root, name, start, count, packet):
    return {"audit_identity_hash": packet["identity_hash"], "model": name, "start": start, "count": count}


def _read_chunk(path, expected):
    path = Path(path)
    record = path.with_suffix(".json")
    if not path.exists() and not record.exists():
        if path.with_suffix(path.suffix+".tmp").exists():
            raise ValueError("Orphan residual chunk remains; never redraw or silently replace it")
        return None
    if not path.is_file() or not record.is_file():
        raise ValueError("Orphan or incomplete residual chunk")
    saved = read(record)
    payload = {key: value for key, value in saved.items() if key != "content_sha256"}
    if (saved.get("identity") != expected or saved.get("content_sha256") != stable_hash(payload)
            or saved.get("arrays_sha256") != sha(path)):
        raise ValueError("Residual chunk source, allocation or checksum changed")
    with np.load(path, allow_pickle=False) as arrays:
        keys = {prefix+name for name in COMPONENT_KEYS for prefix in ("", "physical_", "dimensionless_")}|{"row_ids"}
        if set(arrays.files) != keys or saved["array_hashes"] != {key: array_sha(arrays[key]) for key in arrays.files}:
            raise ValueError("Residual component inventory changed")
        for name in COMPONENT_KEYS:
            shape = (expected["count"], 4) if name in MEAN_KEYS else (expected["count"], 4, 4)
            for prefix in ("", "physical_", "dimensionless_"):
                if arrays[prefix+name].shape != shape or not np.isfinite(arrays[prefix+name]).all():
                    raise ValueError("Incomplete or nonfinite residual component rows")
    return saved


def load_record_arrays(record):
    if sha(record["rows"]["path"]) != record["rows"]["sha256"]:
        raise ValueError("Residual row metadata checksum changed")
    with np.load(record["rows"]["path"], allow_pickle=False) as rows:
        result = {key: rows[key] for key in rows.files}
    collected = {}
    cursor = 0
    for chunk in record["array_chunks"]:
        if chunk["start"] != cursor or sha(chunk["path"]) != chunk["sha256"]:
            raise ValueError("Residual model chunk rows are missing or changed")
        with np.load(chunk["path"], allow_pickle=False) as arrays:
            if not np.array_equal(arrays["row_ids"], result["row_ids"][cursor:cursor+chunk["count"]]):
                raise ValueError("Residual component rows are not the original ordered validation rows")
            for key in arrays.files:
                if key != "row_ids":
                    collected.setdefault(key, []).append(arrays[key])
        cursor += chunk["count"]
    if cursor != len(result["states"]):
        raise ValueError("Incomplete residual model row allocation")
    result.update({key: np.concatenate(values) for key, values in collected.items()})
    return result


def _record(root, model, packet, rows_record, *, create=False):
    size = packet["identity"]["pools"][model["case"]]["rows"]
    chunk_rows = packet["identity"]["plan"]["chunk_rows"]
    chunks, errors = [], {"mean": 0., "covariance": 0.}
    for start in range(0, size, chunk_rows):
        count = min(chunk_rows, size-start)
        path = _chunk_path(root, model["name"], start)
        saved = _read_chunk(path, _chunk_record(root, model["name"], start, count, packet))
        if saved is None:
            return None
        errors = {stage: max(errors[stage], saved["maximum_decomposition_error"][stage]) for stage in errors}
        chunks.append({"path": str(path.resolve()), "sha256": sha(path), "start": start, "count": count,
                       "record_path": str(path.with_suffix(".json").resolve()), "record_sha256": sha(path.with_suffix(".json"))})
    provisional = {"model": model, "rows": rows_record, "array_chunks": chunks}
    arrays = load_record_arrays(provisional)
    metadata = read(model["metadata_path"])
    config = study_runner.engine.CollocationAblationConfig.from_dict(metadata["identity"]["config"])
    scaling = study_runner.load_declared_model(model).loss_scaling
    counts = [{"label": label, "rows": int(np.sum(arrays["strata"] == label)),
        "parent_families": len(set(arrays["parent_family_ids"][arrays["strata"] == label].astype(str)))}
        for label in strata_labels(config.strata)]
    metrics = {stage: natural_metrics(arrays["dimensionless_"+key], config.repair.pseudo_huber_delta)
               for stage, key in (("mean", "rmu"), ("covariance", "rC"))}
    for stage in metrics:
        stored = model["natural_validation"][stage]
        for key in ("natural_mse", "natural_huber"):
            np.testing.assert_allclose(metrics[stage][key], stored[key], atol=1e-10, rtol=1e-10)
    summary = {"model_name": model["name"], "case": model["case"], "rows": size,
        "parent_families": len(set(arrays["parent_family_ids"].astype(str))), "strata_counts": counts,
        "canonical_validation": metrics, "maximum_decomposition_error": errors,
        "loss_scaling": scaling, "strata_config": config.strata.to_dict(),
        "horizon": config.repair.study.horizon, "n_particles": config.repair.study.n_particles,
        "bank_scales": np.asarray(study_runner.payoff_bank(model["case"], study_runner.load_plan(
            packet["identity"]["study_root"])).scales).tolist(),
        "delta": config.repair.pseudo_huber_delta, "units": UNITS,
        "common_source_model": f"{model['case']}_uniform_pseudo_huber_seed{model['seed']}",
        "weights_and_buffers_unchanged": True, "covariance_canonical_entries": 16}
    summary["normalization"] = {"horizon": summary["horizon"], "n_particles": summary["n_particles"],
        "bank_scales": summary["bank_scales"], "mean_scale": scaling["mean_scale"],
        "covariance_scale": scaling["covariance_scale"], "pseudo_huber_delta": summary["delta"]}
    summary["observable_labels"] = list(study_runner.payoff_bank(model["case"], study_runner.load_plan(
        packet["identity"]["study_root"])).labels)
    path = Path(root)/"models"/(model["name"]+".json")
    if not create and not path.is_file():
        raise ValueError("Complete residual chunks lack their immutable model summary")
    study_runner._immutable(path, summary)
    return {**provisional, "summary": summary, "summary_path": str(path.resolve()), "summary_sha256": sha(path)}


def _execution(root):
    path = Path(root)/"execution/attempts.json"
    if not path.exists():
        return {"attempts": [], "failures": []}
    saved = read(path)
    value = {key: item for key, item in saved.items() if key != "content_sha256"}
    if saved.get("content_sha256") != stable_hash(value):
        raise ValueError("Residual audit paid-cost or failure inventory changed")
    return value


def _save_execution(root, value):
    write(Path(root)/"execution/attempts.json", {**value, "content_sha256": stable_hash(value)})


def run(audit_root, *, max_chunks=None, deadline=None):
    tick, cpu = time.monotonic(), time.process_time()
    root = Path(audit_root).resolve()
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Nonnegative row-chunk budget required")
    with study_runner._exclusive(root/"execution/.run.lock"):
        execution = _execution(root)
        if execution["failures"]:
            raise ValueError("Residual failure inventory is nonempty; use a separately versioned audit after review")
        attempt = {"finished": False, "complete": False, "pid": os.getpid(), "chunks_committed": 0,
                   "started_unix": time.time()}
        execution["attempts"].append(attempt)
        _save_execution(root, execution)
        models = {}
        active_group = None
        try:
            root, study, locked, packet, rows = _prepared(root)
            for item in locked["models"]:
                group = item["case"], item["seed"]
                if group != active_group:
                    models.clear()  # Bound transient per-model heat-coefficient caches.
                    active_group = group
                with np.load(rows[item["case"]]["path"], allow_pickle=False) as saved:
                    states, horizons, row_ids = saved["states"], saved["horizons"], saved["row_ids"]
                common_name = f"{item['case']}_uniform_pseudo_huber_seed{item['seed']}"
                for start in range(0, len(states), packet["identity"]["plan"]["chunk_rows"]):
                    count = min(packet["identity"]["plan"]["chunk_rows"], len(states)-start)
                    path = _chunk_path(root, item["name"], start)
                    expected = _chunk_record(root, item["name"], start, count, packet)
                    if _read_chunk(path, expected) is not None:
                        continue
                    if max_chunks is not None and attempt["chunks_committed"] >= max_chunks:
                        return verify(root, require_complete=False)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise BudgetExhausted("Residual audit saved its completed chunks before the deadline")
                    for name in (item["name"], common_name):
                        if name not in models:
                            models[name] = study_runner.load_declared_model(next(m for m in locked["models"] if m["name"] == name))
                    started, ccpu = time.monotonic(), time.process_time()
                    arrays, errors = extract_components(models[item["name"]], states[start:start+count],
                        horizons[start:start+count], models[common_name], ResidualAuditPlan(**packet["identity"]["plan"]))
                    arrays["row_ids"] = row_ids[start:start+count]
                    _save_arrays(path, arrays)
                    record = {"identity": expected, "arrays_sha256": sha(path),
                        "array_hashes": {key: array_sha(value) for key, value in arrays.items()},
                        "maximum_decomposition_error": errors, "weights_and_buffers_unchanged": True,
                        "wall_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-ccpu}
                    write(path.with_suffix(".json"), {**record, "content_sha256": stable_hash(record)})
                    attempt["chunks_committed"] += 1
                _record(root, item, packet, rows[item["case"]], create=True)
            attempt["complete"] = True
        except BudgetExhausted:
            raise
        except BaseException as error:
            execution["failures"].append({"type": type(error).__name__, "message": str(error),
                                          "at_unix": time.time()})
            raise
        finally:
            attempt.update(finished=True, finished_unix=time.time(), wall_seconds=time.monotonic()-tick,
                           cpu_seconds=time.process_time()-cpu)
            _save_execution(root, execution)
    return verify(root, require_complete=False)


def verify(audit_root, *, require_complete=True):
    root, study, locked, packet, rows = _prepared(audit_root)
    execution = _execution(root)
    if execution["failures"]:
        raise ValueError("Residual audit failure inventory is nonempty")
    records, expected_files = [], set()
    for item in locked["models"]:
        size = packet["identity"]["pools"][item["case"]]["rows"]
        for start in range(0, size, packet["identity"]["plan"]["chunk_rows"]):
            path = _chunk_path(root, item["name"], start)
            expected_files.update((str(path.resolve()), str(path.with_suffix(".json").resolve())))
        record = _record(root, item, packet, rows[item["case"]])
        if record is not None:
            records.append(record)
    actual = {str(p.resolve()) for p in (root/"chunks").rglob("*") if p.is_file()} if (root/"chunks").exists() else set()
    if not actual <= expected_files:
        raise ValueError("Orphan or undeclared residual chunk inventory")
    paid = all(a.get("finished") is True and all(isinstance(a.get(k), (float, int)) and np.isfinite(a[k])
               and a[k] >= 0 for k in ("wall_seconds", "cpu_seconds")) for a in execution["attempts"])
    complete = len(records) == len(locked["models"]) and paid and bool(execution["attempts"])
    if require_complete and not complete:
        raise ValueError("All declared model/row chunks and measured audit attempts must complete")
    inputs = {str((root/"plan.json").resolve()): sha(root/"plan.json"),
              str((root/"inputs/manifest.json").resolve()): sha(root/"inputs/manifest.json")}
    for value in rows.values():
        inputs[value["path"]] = value["sha256"]
    prepare_costs = _checked_prepare_costs(root, packet["identity"]["study_root"])
    inputs[str((root/"execution/prepare_costs.json").resolve())] = sha(root/"execution/prepare_costs.json")
    expected_summaries = {record["summary_path"] for record in records}
    actual_summaries = {str(p.resolve()) for p in (root/"models").rglob("*") if p.is_file()} if (root/"models").exists() else set()
    if actual_summaries != expected_summaries:
        raise ValueError("Orphan or undeclared residual model summary")
    for record in records:
        inputs[record["summary_path"]] = record["summary_sha256"]
        for chunk in record["array_chunks"]:
            inputs[chunk["path"]], inputs[chunk["record_path"]] = chunk["sha256"], chunk["record_sha256"]
    if (root/"execution/attempts.json").exists():
        inputs[str((root/"execution/attempts.json").resolve())] = sha(root/"execution/attempts.json")
    return {"root": root, "study_root": Path(packet["identity"]["study_root"]), "study": study,
        "plan": packet, "fit_lock": locked, "records": records, "inputs": inputs,
        "complete": complete, "execution": execution, "preparation_costs": prepare_costs}


def load_complete(audit_root):
    return verify(audit_root, require_complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args()
    if (args.stage == "prepare") != (args.study_root is not None) or not 0 < args.budget_seconds <= 8*60*60:
        parser.error("study-root only for prepare; budget positive and <=8h")
    torch.set_num_threads(1)
    try:
        result = (prepare(args.study_root, args.root, ResidualAuditPlan(args.chunk_rows)) if args.stage == "prepare"
                  else run(args.root, max_chunks=args.max_chunks, deadline=time.monotonic()+args.budget_seconds)
                  if args.stage == "run" else load_complete(args.root))
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75
    print(json.dumps({"complete": result.get("complete", False), "models": len(result.get("records", [])),
                      "root": str(args.root.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
