"""Validation-only component/stratum residual supplement for all crossed fits.

These are diagnostics of validation-selected models on their selection pool.
They are neither independent accuracy measurements nor error certificates.
Per-row numerical arrays, weights and trajectory banks remain local.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import report_lna_payoff_correction as qa  # noqa: E402
from analysis import run_payoff_collocation_study as study_runner  # noqa: E402
from analysis.payoff_collocation_sampling import BoundaryStrataConfig, boundary_time_strata  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "all-model-validation-residual-component-stratum-supplement-v1"
QUANTILES = (.5, .9, .99)
UNITS = {"solver_raw": "", "physical_raw": "physical_", "train_normalized": "dimensionless_"}
TERMS = {"mean": ("mu_dt", "Lmu", "running_source", "rmu", "mu_transport"),
    "covariance": ("C_dt", "LC", "Gamma", "rC", "C_transport", "Gamma_common", "rC_common")}
ATOL, RTOL = 1e-11, 2e-9
COMPONENT_FIELDS = ("case", "name", "seed", "proposal", "loss", "stratum", "stage", "unit", "term",
    "component_i", "component_j", "unique_triangle", "plot_multiplicity", "rows", "parent_families",
    "population_weight", "metric_defined", "signed_mean", "rms", "maximum_absolute", "absolute_q50",
    "absolute_q90", "absolute_q99", "natural_mse_contribution", "natural_huber_contribution", "counterfactual_source_only")


def _auditor():
    from analysis import audit_payoff_collocation_residuals

    return audit_payoff_collocation_residuals


def read(path):
    return qa.read(path)


def sha(path):
    return qa.sha(path)


def write(path, value):
    return qa.write(path, value)


def artifact(path, inputs, expected=None):
    return qa.artifact(path, inputs, expected)


def stratum_names(config):
    if not isinstance(config, BoundaryStrataConfig):
        config = BoundaryStrataConfig.from_dict(config)
    if config.distance_edges != (.25, 1., 4.) or config.time_edges != (.25, .5, .75):
        raise ValueError("The frozen supplement requires all 19 prescribed boundary/time strata")
    return ("absorbed/zero-time", "absorbed/positive-time", "interior/zero-time", *(
        f"active/time-{time}/distance-{distance}" for time in range(4) for distance in range(4)))


def _close(actual, expected, label, *, scale=None):
    actual, expected = np.asarray(actual), np.asarray(expected)
    reference = np.abs(expected) if scale is None else np.asarray(scale)
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or np.any(abs(actual-expected) > ATOL+RTOL*reference)):
        raise ValueError(f"Residual {label} decomposition, units or canonical averaging changed")


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for key in ("states", "horizons", "row_ids", "family_ids", "parent_family_ids", "strata"):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode()+str(value.dtype).encode()+str(value.shape).encode()+value.tobytes())
    return digest.hexdigest()


def _huber(values, delta):
    return 2*values**2/(np.sqrt(1+(values/delta)**2)+1)


def validate_model(record, arrays):
    """Reconstruct unit transforms/decompositions and canonical natural losses."""
    summary = record["summary"]
    model = record["model"]
    if (summary["model_name"] != model["name"] or summary["case"] != model["case"]
            or summary.get("weights_and_buffers_unchanged") is not True
            or summary.get("covariance_canonical_entries") != 16
            or summary.get("common_source_model") != f"{model['case']}_uniform_pseudo_huber_seed{model['seed']}"):
        raise ValueError("Residual model/source identity or unchanged-weight contract changed")
    normalization = summary["normalization"]
    horizon, particles = normalization["horizon"], normalization["n_particles"]
    scales = np.asarray(normalization["bank_scales"], dtype=float)
    mean_scale = np.asarray(normalization["mean_scale"], dtype=float)
    covariance_scale = np.asarray(normalization["covariance_scale"], dtype=float)
    delta = normalization["pseudo_huber_delta"]
    if (not math.isfinite(horizon) or horizon <= 0 or type(particles) is not int or particles < 2
            or scales.shape != (4,) or mean_scale.shape != (4,) or covariance_scale.shape != (4, 4)
            or any(not np.isfinite(value).all() or np.any(value <= 0) for value in (scales, mean_scale, covariance_scale))
            or not math.isfinite(delta) or delta <= 0):
        raise ValueError("Frozen finite positive physical/train normalization is required")
    config = BoundaryStrataConfig.from_dict(summary["strata_config"])
    labels = stratum_names(config)
    states, horizons = np.asarray(arrays["states"]), np.asarray(arrays["horizons"])
    n = len(states)
    if (n < 1 or states.ndim != 2 or horizons.shape != (n,) or config.horizon != horizon
            or any(np.asarray(arrays[key]).shape != (n,) for key in ("row_ids", "family_ids", "parent_family_ids", "strata"))
            or len(set(arrays["row_ids"].astype(str))) != n or summary["rows"] != n
            or any(not value for key in ("row_ids", "family_ids", "parent_family_ids") for value in arrays[key].astype(str))):
        raise ValueError("Complete ordered unique validation row IDs and parent groups required")
    actual_labels = boundary_time_strata(states, horizons, config)
    if not np.array_equal(actual_labels, arrays["strata"].astype(str)):
        raise ValueError("Frozen training-input-only stratum labels changed")
    parents = arrays["parent_family_ids"].astype(str)
    if summary["parent_families"] != len(set(parents)):
        raise ValueError("Distinct parent-family count changed")
    counts = {name: int(np.sum(actual_labels == name)) for name in labels}
    expected_counts = [{"label": name, "rows": counts[name],
        "parent_families": len(set(parents[actual_labels == name]))} for name in labels]
    if summary["strata_counts"] != expected_counts:
        raise ValueError("All 19 occupied and unoccupied stratum counts must remain")
    for stage, terms in TERMS.items():
        shape = (n, 4) if stage == "mean" else (n, 4, 4)
        physical = scales/math.sqrt(particles) if stage == "mean" else np.outer(scales, scales)/particles
        train = mean_scale if stage == "mean" else covariance_scale
        for term in terms:
            native = np.asarray(arrays[term])
            if native.shape != shape or not np.isfinite(native).all():
                raise ValueError("Every full 4/16-entry residual component must remain finite and aligned")
            _close(arrays["physical_"+term], native*physical, "physical "+term)
            _close(arrays["dimensionless_"+term], horizon*native/train, "train-normalized "+term)
    expected = {"rmu": arrays["mu_dt"]-arrays["Lmu"]-arrays["running_source"],
        "mu_transport": arrays["mu_dt"]-arrays["Lmu"],
        "rC": arrays["C_dt"]-arrays["LC"]-arrays["Gamma"],
        "C_transport": arrays["C_dt"]-arrays["LC"],
        "rC_common": arrays["C_dt"]-arrays["LC"]-arrays["Gamma_common"]}
    for term, value in expected.items():
        stage = "mean" if term in ("rmu", "mu_transport") else "covariance"
        magnitude = sum(abs(arrays[key]) for key in (("mu_dt", "Lmu", "running_source")
                        if stage == "mean" else ("C_dt", "LC", "Gamma", "Gamma_common")))
        _close(arrays[term], value, term, scale=magnitude)
    canonical = {}
    for stage, term in (("mean", "rmu"), ("covariance", "rC")):
        values = arrays["dimensionless_"+term]
        canonical[stage] = {"natural_mse": float(np.mean(values**2)), "natural_huber": float(np.mean(_huber(values, delta)))}
        for metric, value in canonical[stage].items():
            _close(np.array(value), np.array(summary["canonical_validation"][stage][metric]), "canonical "+stage+" "+metric)
            _close(np.array(value), np.array(record["model"]["natural_validation"][stage][metric]), "selected-model "+stage+" "+metric)
    return {"rows": n, "labels": labels, "parents": parents, "strata": actual_labels,
        "canonical": canonical, "normalization": normalization, "metadata_sha256": _fingerprint(arrays)}


def component_rows(record, arrays, checked):
    """Stream every component; empty-bin metrics stay undefined (None)."""
    model, n = record["model"], checked["rows"]
    origin = {key: model[key] for key in ("case", "name", "seed", "proposal", "loss")}
    for label in ("all", *checked["labels"]):
        mask = np.ones(n, dtype=bool) if label == "all" else checked["strata"] == label
        count = int(mask.sum())
        parents = len(set(checked["parents"][mask]))
        for stage, terms in TERMS.items():
            components = ((i, None) for i in range(4)) if stage == "mean" else ((i, j) for i in range(4) for j in range(4))
            components = tuple(components)
            for unit, prefix in UNITS.items():
                for term in terms:
                    for i, j in components:
                        values = arrays[prefix+term][mask, i] if j is None else arrays[prefix+term][mask, i, j]
                        statistics = {key: None for key in ("signed_mean", "rms", "maximum_absolute", "absolute_q50", "absolute_q90", "absolute_q99", "natural_mse_contribution", "natural_huber_contribution")}
                        if count:
                            absolute = abs(values)
                            q50, q90, q99 = np.quantile(absolute, QUANTILES)
                            statistics.update(signed_mean=float(values.mean()), rms=float(np.sqrt(np.mean(values**2))),
                                maximum_absolute=float(absolute.max()), absolute_q50=float(q50), absolute_q90=float(q90), absolute_q99=float(q99))
                            if unit == "train_normalized" and term in ("rmu", "rC"):
                                statistics["natural_mse_contribution"] = float(np.sum(values**2)/(n*(4 if j is None else 16)))
                                statistics["natural_huber_contribution"] = float(np.sum(_huber(values, checked["normalization"]["pseudo_huber_delta"]))/(
                                    n*(4 if j is None else 16)))
                        yield {**origin, "stratum": label, "stage": stage, "unit": unit, "term": term,
                            "component_i": i, "component_j": j, "unique_triangle": j is None or i <= j,
                            "plot_multiplicity": 1 if j is None or i == j else 2,
                            "rows": count, "parent_families": parents, "population_weight": count/n,
                            "metric_defined": bool(count), **statistics, "counterfactual_source_only": term.endswith("_common")}


def loss_rows(record, arrays, checked):
    model, n = record["model"], checked["rows"]
    origin = {key: model[key] for key in ("case", "name", "seed", "proposal", "loss")}
    output = []
    for stage, term in (("mean", "rmu"), ("covariance", "rC")):
        values = arrays["dimensionless_"+term]
        total, total_huber = 0., 0.
        for label in ("all", *checked["labels"]):
            mask = np.ones(n, dtype=bool) if label == "all" else checked["strata"] == label
            count = int(mask.sum())
            mse = float(np.mean(values[mask]**2)) if count else None
            huber = float(np.mean(_huber(values[mask], checked["normalization"]["pseudo_huber_delta"]))) if count else None
            contribution = count/n*mse if count else None
            huber_contribution = count/n*huber if count else None
            if label != "all" and contribution is not None:
                total += contribution
                total_huber += huber_contribution
            output.append({**origin, "stage": stage, "stratum": label, "rows": count,
                "parent_families": len(set(checked["parents"][mask])), "population_weight": count/n,
                "metric_defined": bool(count), "common_unit_mse": mse, "common_unit_huber": huber,
                "natural_mse_contribution": contribution, "natural_huber_contribution": huber_contribution})
        _close(np.array(total), np.array(checked["canonical"][stage]["natural_mse"]), "stratum reconstruction "+stage)
        _close(np.array(total_huber), np.array(checked["canonical"][stage]["natural_huber"]), "stratum Huber reconstruction "+stage)
    return output


def load_complete(root):
    evidence = _auditor().load_complete(Path(root).resolve())
    expected = {(m["case"], m["seed"], m["proposal"], m["loss"], m["name"]) for m in study_runner.slots(evidence["study"])}
    actual = [tuple(r["model"][key] for key in ("case", "seed", "proposal", "loss", "name")) for r in evidence["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Every declared crossed model/seed is required; no best-seed selection")
    return evidence


def _helpers(evidence):
    helpers = {str((REPO/"analysis"/name).resolve()): digest for name, digest in evidence["plan"]["identity"]["helpers"].items()}
    for path in (Path(__file__).resolve(), Path(qa.__file__).resolve()):
        if str(path) in helpers and helpers[str(path)] != sha(path):
            raise ValueError("Residual reporter/helper changed after the audit choices froze")
        helpers[str(path)] = sha(path)
    return helpers


def _record_inputs(record, inputs, study_root):
    artifact(record["rows"]["path"], inputs, record["rows"]["sha256"])
    artifact(record["summary_path"], inputs, record["summary_sha256"])
    for chunk in record["array_chunks"]:
        artifact(chunk["path"], inputs, chunk["sha256"])
        artifact(chunk["record_path"], inputs, chunk["record_sha256"])
    for kind in ("checkpoint", "metadata"):
        path = Path(record["model"][kind+"_path"]).resolve()
        if not path.is_relative_to(study_root/"payoffs"):
            raise ValueError("Residual model evidence must belong to the declared locked study")
        artifact(path, inputs, record["model"][kind+"_sha256"])


def _lock_inputs(evidence, inputs):
    identity = evidence["plan"]["identity"]
    for key, relative in (("fit_lock_sha256", "payoffs/fit_lock.json"), ("data_lock_sha256", "payoffs/data_lock.json"),
            ("flat_pool_lock_sha256", "payoffs/flat_pool_lock.json"), ("study_preregistration_sha256", "payoffs/preregistration.json"),
            ("study_plan_sha256", "plan.json")):
        artifact(evidence["study_root"]/relative, inputs, identity[key])


def _figures(output, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case in sorted({row["model"]["case"] for row in records}):
        items = [row for row in records if row["model"]["case"] == case]
        names = [f"{r['model']['proposal']}/{r['model']['loss']}/s{r['model']['seed']}" for r in items]
        fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(items)*.3)), constrained_layout=True)
        for axis, stage in zip(axes, ("mean", "covariance"), strict=True):
            values = np.array([[next(v["common_unit_mse"] for v in r["losses"] if v["stage"] == stage and v["stratum"] == label)
                                for label in r["labels"]] for r in items], dtype=float)
            plotted = np.ma.masked_invalid(np.arcsinh(np.sqrt(values)))
            cmap = plt.get_cmap("viridis").with_extremes(bad="lightgray")
            image = axis.imshow(plotted, aspect="auto", cmap=cmap)
            axis.set_yticks(range(len(names)), names, fontsize=6)
            axis.set_xticks(range(19), ["abs/0", "abs/+", "int/0", *[f"t{t}/d{d}" for t in range(4) for d in range(4)]], rotation=90, fontsize=6)
            axis.set_title(stage+"; gray = unoccupied; zero = measured zero")
            fig.colorbar(image, ax=axis, label="asinh(train-normalized residual RMS)")
        fig.suptitle(case+" — validation-selected residual diagnostics, not independent accuracy")
        fig.savefig(output/(case+"_strata.png"), dpi=150)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(9, max(4, len(items)*.3)), constrained_layout=True)
        pairs = [(i, j) for i in range(4) for j in range(i, 4)]
        values = np.array([[r["covariance_components"][i, j] for i, j in pairs] for r in items])
        image = axis.imshow(np.arcsinh(values), aspect="auto")
        axis.set_yticks(range(len(names)), names, fontsize=6)
        axis.set_xticks(range(10), [f"C{i}{j} ×{1 if i == j else 2}" for i, j in pairs], rotation=45, fontsize=7)
        axis.set_title(case+"; unique 10 shown, all 16 entries used in loss (off-diagonal multiplicity 2)")
        fig.colorbar(image, ax=axis, label="asinh(global train-normalized component RMS)")
        fig.savefig(output/(case+"_covariance_components.png"), dpi=150)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, max(4, len(items)*.3)), constrained_layout=True)
        image = axis.imshow(np.arcsinh(np.array([r["balance"] for r in items])), aspect="auto")
        axis.set_yticks(range(len(names)), names, fontsize=6)
        axis.set_xticks(range(4), ("∂t C", "L C", "Γ(u)", "residual"))
        axis.set_title(case+"; global term RMS magnitudes are not additive loss contributions")
        fig.colorbar(image, ax=axis, label="asinh(train-normalized term RMS)")
        fig.savefig(output/(case+"_covariance_balance.png"), dpi=150)
        plt.close(fig)


def _csv(path, rows):
    if not rows:
        raise ValueError("A complete nonempty summary is required")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_costs(evidence):
    rows = []
    for phase, record in (("prepare", evidence["preparation_costs"]), ("residual-extraction", evidence["execution"])):
        if not record["attempts"]:
            raise ValueError("Complete measured residual preparation/extraction attempts required")
        for index, attempt in enumerate(record["attempts"]):
            if (attempt.get("finished") is not True or any(not isinstance(attempt.get(key), (float, int))
                    or not math.isfinite(attempt[key]) or attempt[key] < 0 for key in ("wall_seconds", "cpu_seconds"))):
                raise ValueError("Interrupted residual audit costs cannot be invented as zero")
            rows.append({"phase": phase, "attempt": index, "complete": attempt.get("complete", False),
                "wall_seconds": attempt["wall_seconds"], "cpu_seconds": attempt["cpu_seconds"],
                "scope": "separate diagnostic overhead; nested chunk durations are not added again; summed invocations are not concurrent elapsed wall time"})
    return rows


def _text(allocation):
    return ("# Matched collocation residual supplement\n\n"
        f"All {allocation['models']} declared models remain, with all three optimizer seeds and all four proposal/loss variants. "
        "The component table retains all four means and all sixteen covariance entries in solver-raw, physical-raw and stored training-normalized units.\n\n"
        "These models were selected on this natural validation pool. These are descriptive selection-pool diagnostics, not independent accuracy measurements, "
        "field bias estimates, 95% significance tests or an error/continuum certificate. No new physical continuations, labels or final Monte Carlo data are used.\n\n"
        "All nineteen boundary/time strata appear. Unoccupied bins have count zero and undefined metrics, not evidence of zero residual. "
        "Stratification only rebalances occupied states; it cannot create absent boundary, corner, absorbed or zero-time states. Empty strata therefore supply no evidence for those states. "
        "Distinct parent counts describe grouping; snapshots and remaining-time replicas are not independent families, and parent counts must not be summed across bins.\n\n"
        "Natural-population MSE and pseudo-Huber contributions use row occupancy and all four/sixteen entries. Their sums reconstruct both canonical global validation losses; "
        "an unweighted average of occupied-bin losses would change the measure. Both common-unit MSE and pseudo-Huber loss are retained for every variant.\n\n"
        "The physical-raw mean residual is the solver residual times bank scale/√N; physical covariance residual uses bank-scale outer product/N. "
        "Training-normalized terms use T times the solver term divided by its original stored stage scale; no new fit, inverse-time weighting, clipping or normalization is introduced.\n\n"
        "Stored loss units are common across the crossed variants within each case/stage; magnitudes across different physical regimes are not independently measured fluctuation errors. "
        "Preparation and residual-extraction attempt costs are separate diagnostic overhead in audit_costs.csv.\n\n"
        "Covariance source Γ(u), operator L C and time derivative are shown separately. Their RMS magnitudes do not add to the residual loss. "
        "Γ_common and rC_common substitute the matched-seed uniform/pseudo-Huber mean only as a counterfactual source diagnostic; other variants were not trained with that common mean.\n\n"
        "The ten unique covariance entries are a plot convenience with diagonal multiplicity one and off-diagonal multiplicity two. All sixteen signed entries remain in the compressed CSV. "
        "`component_statistics.csv.gz` is streamed losslessly; raw per-row arrays, state banks and checkpoints remain local and checksum-indexed.\n")


def generate(root, *, output=None, documentation=None):
    evidence = load_complete(root)
    root, study_root = evidence["root"], evidence["study_root"]
    output = root/"report" if output is None else Path(output).resolve()
    if (output == root or root.is_relative_to(output) or output.is_relative_to(study_root)
            or output.is_relative_to(root) and output != root/"report"):
        raise ValueError("Use the audit report directory or a separate output outside authoritative inputs")
    if documentation is not None and (Path(documentation).suffix != ".md" or any(
            Path(documentation).resolve().is_relative_to(p) for p in (root, study_root))):
        raise ValueError("Documentation must be a Markdown file outside authoritative study/audit roots")
    inputs = dict(evidence["inputs"])
    _lock_inputs(evidence, inputs)
    for record in evidence["records"]:
        _record_inputs(record, inputs, study_root)
    helpers = _helpers(evidence)
    for path, digest in helpers.items():
        artifact(path, inputs, digest)
    identity = {"kind": KIND, "source_hash": source_hash(), "reporter_sha256": sha(__file__),
        "helpers": helpers, "inputs": inputs, "quantiles": list(QUANTILES), "empty_metrics": None,
        "covariance_loss_entries": 16, "unique_plot_entries": 10, "normalization_population": "fixed-natural-validation-rows"}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old = load_report(output)
        if old["identity"] != identity:
            raise ValueError("Residual report inputs/source changed; use a new output directory")
        if documentation is not None:
            Path(documentation).parent.mkdir(parents=True, exist_ok=True)
            Path(documentation).write_text((output/"report.md").read_text())
        return old
    with tempfile.TemporaryDirectory(prefix=".residual-report-", dir=output.parent) as temporary:
        stage = Path(temporary)/"report"
        stage.mkdir()
        losses, models, plots, seen = [], [], [], {}
        count = 0
        with gzip.open(stage/"component_statistics.csv.gz", "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COMPONENT_FIELDS)
            writer.writeheader()
            for record in evidence["records"]:
                arrays = _auditor().load_record_arrays(record)
                checked = validate_model(record, arrays)
                model = record["model"]
                if model["case"] in seen and seen[model["case"]] != checked["metadata_sha256"]:
                    raise ValueError("Crossed variants/seeds must use the same literal ordered validation rows")
                seen[model["case"]] = checked["metadata_sha256"]
                for row in component_rows(record, arrays, checked):
                    writer.writerow(row)
                    count += 1
                local = loss_rows(record, arrays, checked)
                losses.extend(local)
                models.append({**{key: model[key] for key in ("case", "name", "seed", "proposal", "loss")},
                    "rows": checked["rows"], "parent_families": len(set(checked["parents"])),
                    **{stage+"_"+metric: value for stage, values in checked["canonical"].items() for metric, value in values.items()},
                    "canonical_reconstruction_passed": True})
                plots.append({"model": model, "losses": local, "labels": checked["labels"],
                    "covariance_components": np.sqrt(np.mean(arrays["dimensionless_rC"]**2, axis=0)),
                    "balance": [float(np.sqrt(np.mean(arrays["dimensionless_"+term]**2))) for term in ("C_dt", "LC", "Gamma", "rC")]})
        allocation = {"models": len(models), "cases": len(seen), "strata_per_model": 19, "component_statistic_rows": count,
            "stratum_loss_rows": len(losses), "all_seeds_and_variants_retained": True}
        _csv(stage/"stratum_losses.csv", losses)
        _csv(stage/"model_summary.csv", models)
        costs = audit_costs(evidence)
        _csv(stage/"audit_costs.csv", costs)
        occupancy = [{"case": case, "rows": next(r["rows"] for r in models if r["case"] == case),
            "parent_families": next(r["parent_families"] for r in models if r["case"] == case),
            "occupied_strata": sum(r["rows"] > 0 for r in losses if r["case"] == case and r["name"] == next(
                m["name"] for m in models if m["case"] == case) and r["stage"] == "mean" and r["stratum"] != "all")}
            for case in sorted(seen)]
        _csv(stage/"occupancy_summary.csv", occupancy)
        _figures(stage, plots)
        occupancy_text = "\n| Case | Validation rows | Distinct parents | Occupied / 19 |\n| --- | ---: | ---: | ---: |\n"+"\n".join(
            f"| {r['case']} | {r['rows']} | {r['parent_families']} | {r['occupied_strata']} / 19 |" for r in occupancy)+"\n"
        (stage/"report.md").write_text(_text(allocation)+occupancy_text)
        write(stage/"report.json", {"kind": KIND, "complete": True, "identity": identity, "allocation": allocation,
            "models": models, "audit_costs": costs, "occupancy": occupancy,
            "independent_accuracy_claim": False, "validation_selection_pool": True,
            "counterfactual_is_training_policy": False, "active_readme_goal_complete": False})
        manifest = {"kind": KIND, "complete": True, "identity": identity, "allocation": allocation,
            "outputs": {path.name: sha(path) for path in stage.iterdir() if path.is_file()}}
        write(stage/"manifest.json", manifest)
        stage.rename(output)
    if documentation is not None:
        Path(documentation).parent.mkdir(parents=True, exist_ok=True)
        Path(documentation).write_text((output/"report.md").read_text())
    return manifest


def load_report(output):
    output = Path(output).resolve()
    manifest = read(output/"manifest.json")
    identity = manifest["identity"]
    if (manifest.get("kind") != KIND or manifest.get("complete") is not True or identity.get("source_hash") != source_hash()
            or identity.get("reporter_sha256") != sha(__file__)):
        raise ValueError("Complete current-source residual supplement required")
    for path, digest in identity["inputs"].items():
        if sha(path) != digest:
            raise ValueError("Residual supplement input changed")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p != output/"manifest.json"}
    if actual != set(manifest["outputs"]) or any(sha(output/name) != digest for name, digest in manifest["outputs"].items()):
        raise ValueError("Residual supplement exact output inventory changed")
    result = read(output/"report.json")
    if not result.get("complete") or result["identity"] != identity or result["allocation"] != manifest["allocation"]:
        raise ValueError("Residual supplement result/manifest differ")
    return manifest


def archive(root, destination, *, test_root=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    manifest = load_report(root/"report")
    evidence = load_complete(root)
    test_root = root if test_root is None else Path(test_root).resolve()
    tested, logs = qa.verify_final_tests(test_root, helpers=manifest["identity"]["helpers"])
    study_root = evidence["study_root"]
    if destination.exists() or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in (root, study_root, test_root)):
        raise ValueError("Use a new archive outside authoritative audit/study/test roots")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".residual-evidence-", dir=destination.parent) as temporary:
        stage = Path(temporary)/"archive"
        stage.mkdir()
        copied, inventory = {}, {}

        def copy_file(path, logical):
            target = stage/logical
            compressed = path.suffix in (".json", ".csv") and path.stat().st_size > 100000
            if compressed:
                target = target.with_name(target.name+".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            if compressed:
                with path.open("rb") as source, target.open("wb") as sink:
                    with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
                        shutil.copyfileobj(source, zipped)
            else:
                shutil.copyfile(path, target)
            copied[str(target.relative_to(stage))] = {"sha256": sha(target), "original_sha256": sha(path),
                "logical_path": logical, "compressed": compressed}

        # Deliberately exclude reference/comparison/deployment trees: this
        # supplement never opens final MC evidence, even merely to hash it.
        selected = [("audit/"+str(path.relative_to(root)), path) for path in root.rglob("*") if path.is_file()]
        selected += [("study/plan.json", study_root/"plan.json")]
        selected += [("study/"+str(path.relative_to(study_root)), path) for path in (study_root/"payoffs").rglob("*") if path.is_file()]
        for logical, path in sorted(selected):
            inventory[logical] = {"sha256": sha(path), "bytes": path.stat().st_size}
            if path.suffix in (".json", ".md", ".csv", ".png") or path.name == "component_statistics.csv.gz":
                copy_file(path, logical)
        for name, digest in tested["source_inputs"].items():
            path = REPO/name
            if sha(path) != digest:
                raise ValueError("Maintained source changed after full test acceptance")
            copy_file(path, "code/"+name)
        copy_file(test_root/"audit/final_tests.json", "qa/final_tests.json")
        log_paths = {}
        for command in tested["commands"]:
            path = Path(command["log_path"]).resolve()
            logical = "qa/"+command["kind"]+".log"
            copy_file(path, logical)
            log_paths[command["log_path"]] = logical
        if set(logs) != {Path(c["log_path"]).resolve() for c in tested["commands"]}:
            raise ValueError("Final test log allocation changed")
        (stage/"README.md").write_text("# Validation residual supplement evidence\n\n"
            "See [the supplement](audit/report/report.md). All crossed models, optimizer seeds and 19 bins remain. "
            "This is validation-selected descriptive residual evidence, not independent accuracy or an error certificate. "
            "Final MC reference/comparison/deployment trees are deliberately neither copied nor indexed here. "
            "Per-row NPZ, checkpoints and learner HDF5 banks stay local and checksum-indexed. Tested maintained sources, "
            "configuration, licensing and QA logs are copied exactly; this is a partial repository snapshot. "
            "See [license coverage](code/LICENSING.md), the [canonical repository](https://github.com/xrough/nn-dk) "
            "and [canonical documentation](https://github.com/xrough/nn-dk/tree/main/docs) for omitted Git history/documents.\n")
        helpers = {"code/"+str(Path(path).relative_to(REPO)): digest for path, digest in manifest["identity"]["helpers"].items()}
        write(stage/"archive_manifest.json", {"kind": KIND, "complete": True, "allocation": manifest["allocation"],
            "copied": copied, "raw_inventory": inventory, "source_helpers": helpers,
            "tested_source_inputs": tested["source_inputs"], "test_receipt_sha256": sha(test_root/"audit/final_tests.json"),
            "test_log_paths": log_paths, "report_manifest_sha256": sha(root/"report/manifest.json"),
            "generated": {"README.md": sha(stage/"README.md")}, "large_raw_artifacts_retained_locally": True,
            "final_MC_evidence_opened": False, "active_readme_goal_complete": False})
        verify_archive(stage)
        stage.rename(destination)
    return verify_archive(destination)


def verify_archive(destination):
    """Offline checks use archive bytes only, not original runs or live source."""
    destination = Path(destination).resolve()
    manifest = read(destination/"archive_manifest.json")
    if (manifest.get("kind") != KIND or manifest.get("complete") is not True
            or manifest.get("active_readme_goal_complete") is not False or manifest.get("final_MC_evidence_opened") is not False
            or manifest.get("large_raw_artifacts_retained_locally") is not True):
        raise ValueError("Complete scoped residual supplement archive required")
    expected = set(manifest["copied"])|set(manifest["generated"])|{"archive_manifest.json"}
    if {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()} != expected:
        raise ValueError("Residual archive exact file inventory changed")
    logical = {}
    for name, item in manifest["copied"].items():
        path = destination/name
        if not path.resolve().is_relative_to(destination) or sha(path) != item["sha256"]:
            raise ValueError("Residual archive checksum/path changed")
        data = gzip.decompress(path.read_bytes()) if item["compressed"] else path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["original_sha256"] or item["logical_path"] in logical:
            raise ValueError("Lossless residual evidence checksum changed")
        logical[item["logical_path"]] = (item, data)
    for name, digest in manifest["generated"].items():
        if not (destination/name).resolve().is_relative_to(destination) or sha(destination/name) != digest:
            raise ValueError("Residual archive wrapper changed")
    for name, digest in {**manifest["source_helpers"], **{"code/"+k: v for k, v in manifest["tested_source_inputs"].items()}}.items():
        if name not in logical or logical[name][0]["original_sha256"] != digest:
            raise ValueError("Archived helper or tested maintained source changed")
    for name, value in manifest["raw_inventory"].items():
        if (Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(value.get("sha256"), str)
                or len(value["sha256"]) != 64 or type(value.get("bytes")) is not int or value["bytes"] < 0):
            raise ValueError("Invalid local raw-artifact inventory")
        if name in logical and (logical[name][0]["original_sha256"] != value["sha256"] or len(logical[name][1]) != value["bytes"]):
            raise ValueError("Copied residual evidence disagrees with its local-only index")
        if name.startswith(("study/reference/", "study/comparison/", "study/deployment/")):
            raise ValueError("Validation-only residual evidence cannot import final MC trees")
    qa_item = logical["qa/final_tests.json"]
    tested = json.loads(qa_item[1])
    if (qa_item[0]["original_sha256"] != manifest["test_receipt_sha256"] or tested.get("kind") != qa.TEST_GATE_KIND
            or tested.get("complete") is not True or tested.get("source_inputs") != manifest["tested_source_inputs"]
            or tested.get("before_sha256") != tested.get("after_sha256")):
        raise ValueError("Archived current-source test receipt changed")
    if tested["before_sha256"] != stable_hash(tested["source_inputs"]):
        raise ValueError("Archived test source-map digest changed")
    commands = tested.get("commands", [])
    if len(commands) != 2 or {c["kind"] for c in commands} != set(qa.TEST_COMMANDS) or len({c["log_path"] for c in commands}) != 2:
        raise ValueError("Archived complete Ruff/pytest command allocation changed")
    for command in commands:
        name = manifest["test_log_paths"].get(command["log_path"])
        if (command["argv"] != qa.TEST_COMMANDS[command["kind"]] or command["exit_code"] != 0
                or name not in logical or logical[name][0]["original_sha256"] != command["log_sha256"]):
            raise ValueError("Archived successful full-suite log/command changed")
        if command["kind"] == "pytest":
            import re
            counts = re.findall(r"\b(\d+) passed\b", logical[name][1].decode())
            if not counts or command.get("test_count") != int(counts[-1]):
                raise ValueError("Archived pytest count changed")
    report_item = logical["audit/report/manifest.json"]
    report = json.loads(report_item[1])
    result = json.loads(logical["audit/report/report.json"][1])
    if (report_item[0]["original_sha256"] != manifest["report_manifest_sha256"] or not report.get("complete")
            or report.get("kind") != KIND or result["identity"] != report["identity"] or not result.get("complete")
            or result["allocation"] != report["allocation"] or report["allocation"] != manifest["allocation"]):
        raise ValueError("Archived report identity/allocation differs")
    for name, digest in report["outputs"].items():
        if logical["audit/report/"+name][0]["original_sha256"] != digest:
            raise ValueError("Archived residual report output changed")
    return {"complete": True, "allocation": manifest["allocation"], "copied_artifacts_verified": len(manifest["copied"]),
        "indexed_local_raw_files": len(manifest["raw_inventory"]), "manifest_sha256": sha(destination/"archive_manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "archive", "verify-report", "verify-archive"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--documentation", type=Path)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args(argv)
    if ((args.action in ("report", "archive") and args.root is None)
            or (args.action != "report" and args.output is None)):
        parser.error("report/archive need --root; archive/verification need --output")
    result = (generate(args.root, output=args.output, documentation=args.documentation) if args.action == "report" else
        archive(args.root, args.output, test_root=args.test_root) if args.action == "archive" else
        load_report(args.output) if args.action == "verify-report" else verify_archive(args.output))
    print(json.dumps({k: v for k, v in result.items() if k not in ("identity", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
