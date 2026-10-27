#!/usr/bin/env python3
"""Audit and compact a COMPLETE BSDE coverage/architecture experiment.

Usage: .venv/bin/python analysis/summarize_bsde_study.py runs/bsde-anchor-study

Reads saved JSON, observation banks and checkpoint bytes.  It never deserializes
a checkpoint, fits a model, simulates a forward path, or selects a best seed.
The complete-status gate runs before any held-out outcome is opened.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

_HELPER_PATH = Path(__file__).with_name("summarize_bsde_batch.py")
_SPEC = importlib.util.spec_from_file_location("_bsde_artifact_helpers", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)
_read, _require = _HELPER._read, _HELPER._require
_sha, _stable_hash = _HELPER._hash_file, _HELPER._stable_hash


def _model_key(record):
    factors = record["factors"]
    return (record["evaluation"]["problem"], factors["architecture"],
            factors["train_anchors"], factors["train_continuations"], factors["optimizer_seed"])


def _checkpoint(directory, key):
    problem, architecture, anchors, children, seed = key
    return directory / "checkpoints" / problem / architecture / f"a{anchors}_c{children}" / f"seed_{seed}.pt"


def _validate_design(directory, manifest, allow_smoke):
    from nn_dk.bsde_study import _design
    from nn_dk.bsde_study_config import BSDEStudyConfig, bsde_study_preset

    config = _read(directory / "config.json")
    _require(config == manifest.get("config") and _stable_hash(config) == manifest.get("config_hash"),
             "Study configuration content/hash differs from the manifest")
    name = config.get("name")
    _require(name == "bsde-anchor-study" or allow_smoke and name == "bsde-study-smoke",
             "Primary summary requires the preregistered full allocation; use --allow-smoke only for software checks")
    _require(config == bsde_study_preset(name).to_dict(),
             "Study configuration silently reduced or altered the preregistered allocation")
    study = BSDEStudyConfig(**config)
    base = study.base_config
    _require(_read(directory / "base_config.json") == base.to_dict(), "Saved common base configuration mismatch")
    preregistration = _read(directory / "preregistration.json")
    _require(preregistration == _design(study)
             and _stable_hash(preregistration) == manifest.get("preregistration_hash"),
             "Study preregistration content/hash mismatch")
    _require(preregistration.get("independent_test") is True and base.seed == 20261025
             and base.seed != preregistration.get("previous_observed_test_seed"),
             "Study held-out streams are not fresh relative to the inspected pilot")
    _require(manifest.get("expected_models") == study.expected_models
             and manifest.get("completed_evaluations") == study.expected_models,
             "Study manifest has an incomplete model/evaluation allocation")
    return study, preregistration


def _validate_calibration(calibration, base, source):
    config = base.to_dict()
    _require(calibration.get("config_hash") == base.config_hash
             and calibration.get("source_hash") == source,
             "Study calibration source/common-configuration mismatch")
    _require(calibration.get("complete") is True and calibration.get("passed") is True,
             "Study calibration is incomplete or failed")
    for problem in ("brownian", "ou"):
        _require(calibration[problem].get("complete") is True
                 and calibration[problem].get("passed") is True,
                 f"Incomplete/failed {problem} analytic calibration")
    _require(set(calibration["native_refinement"]) == set(base.cases), "Missing native calibration profile")
    for case, audit in calibration["native_refinement"].items():
        _require(audit.get("complete") is True and audit.get("passed") is True
                 and audit.get("n_independent_families") == base.refinement_paths
                 and audit.get("all_stopped_paths_included") is True,
                 f"Native calibration allocation/status mismatch: {case}")
        _require(audit.get("reference_steps") == [base.reference_dt / d for d in (1, 2, 4)],
                 "Native calibration refinement steps changed")
        _require(0 <= audit["maximum_mass_error"] < 1e-10
                 and audit["minimum_saved_mass"] >= -1e-12, "Native calibration conservation/validity failed")
        _require(len(audit["family_ids"]) == len(set(audit["family_ids"])) == config["refinement_paths"],
                 "Native calibration trajectory-family allocation mismatch")


def _validate_inventory(directory, manifest, study, records):
    base = study.base_config
    expected = [(problem, architecture, anchors, children, seed)
                for problem in ("ou", "dk") for anchors, children in study.allocations
                for architecture in study.architectures for seed in base.optimizer_seeds]
    by_key = {_model_key(record): record for record in records}
    _require(len(by_key) == len(records) == len(expected) and set(by_key) == set(expected),
             "Study is missing, duplicating or introducing model factors")
    unlock = manifest.get("test_unlocked_unix")
    created = manifest.get("created_unix")
    _require(isinstance(unlock, (int, float)) and isinstance(created, (int, float))
             and math.isfinite(unlock) and math.isfinite(created) and 0 < created <= unlock,
             "Study is missing a valid all-fits-before-test lock timestamp")
    _require((directory / "preregistration.json").stat().st_mtime <= unlock + 1.,
             "Study preregistration was modified after the test bank unlocked")
    inventory, models, hashes = [], [], {}
    for key in expected:
        record = by_key[key]
        problem, architecture, anchors, children, seed = key
        checkpoint = _checkpoint(directory, key)
        metadata_path = checkpoint.with_suffix(".json")
        _require(checkpoint.is_file() and metadata_path.is_file(), "Study is missing a completed checkpoint")
        metadata = _read(metadata_path)
        sha = _sha(checkpoint)
        _require(checkpoint.stat().st_mtime <= unlock + 1.
                 and metadata_path.stat().st_mtime <= unlock + 1.,
                 "A study fit was written after the final-test lock")
        expected_kind = (("native-dk" if problem == "dk" else "linear-ou")
                         + ("-bsde-structured-moments-v1" if architecture == "structured" else "-bsde-moments-v1"))
        _require(metadata.get("complete") is True and metadata.get("kind") == expected_kind
                 and (metadata.get("problem"), metadata.get("architecture"), metadata.get("seed"))
                 == (problem, architecture, seed), "Checkpoint kind/problem/architecture/seed mismatch")
        counts = {"training_anchors": anchors, "training_paths": anchors*children,
                  "validation_anchors": study.validation[0],
                  "validation_paths": study.validation[0]*study.validation[1]}
        _require(all(metadata.get(k) == v for k, v in counts.items())
                 and metadata.get("allocation") == [anchors, children]
                 and metadata.get("study_config_hash") == study.config_hash,
                 "Checkpoint metadata silently reduced the declared training/validation allocation")
        _require(metadata.get("training_view_fingerprint") == record.get("training_view_fingerprint")
                 and metadata.get("validation_fingerprint") == record.get("validation_fingerprint"),
                 "Checkpoint/evaluation training-view fingerprint mismatch")
        epochs = metadata.get("epochs")
        _require(type(epochs) is int and 1 <= epochs <= base.training["epochs"]
                 and len(metadata.get("history", ())) == epochs, "Incomplete checkpoint epoch history")
        evaluation = record["evaluation"]
        _require(evaluation.get("training") == metadata,
                 "Individual evaluation training metadata differs from its saved checkpoint")
        _require(record.get("checkpoint_sha256") == sha
                 and evaluation["model"].get("checkpoint_sha256") == sha,
                 "Study checkpoint SHA256 mismatch")
        _require((evaluation["model"].get("problem"), evaluation["model"].get("seed")) == (problem, seed),
                 "Individual evaluation model identity mismatch")
        individual = (directory / "common" / "evaluation"
                      / f"{problem}_seed{seed}_{_stable_hash(evaluation['model'])[:20]}.json")
        _require(individual.is_file() and _read(individual) == evaluation,
                 "Aggregate and immutable individual study evaluations differ")
        _require(individual.stat().st_mtime + 1. >= unlock,
                 "An individual held-out evaluation predates the all-fits lock")
        inventory.append({"key": list(key), "checkpoint_sha256": sha,
                          "training_fingerprint": record["training_view_fingerprint"],
                          "validation_fingerprint": record["validation_fingerprint"]})
        for path in (checkpoint, metadata_path, individual):
            hashes[str(path.relative_to(directory))] = _sha(path)
        models.append({"key": list(key), "study_model_id": record["study_model_id"],
                       "checkpoint_sha256": sha, **counts,
                       "training_view_fingerprint": record["training_view_fingerprint"],
                       "validation_fingerprint": record["validation_fingerprint"],
                       "epochs": epochs, "training_seconds": metadata.get("training_seconds"),
                       "first_epoch_seconds": metadata.get("first_epoch_seconds")})
    _require(inventory == manifest.get("fit_lock"), "Saved fit-lock inventory/hash differs from completed checkpoints")
    return models, hashes


def _validate_comparisons(records, base):
    for record in records:
        evaluation = record["evaluation"]
        if evaluation["problem"] == "ou":
            _require(evaluation.get("n_test_anchors") == base.ou_test_anchors,
                     "OU analytic test allocation mismatch")
            config = {**base.to_dict(), "conditional_anchors": base.ou_test_anchors}
            for component in ("neural_error", "total_error", "time_error"):
                _HELPER._comparison(evaluation[component], config, True, f"ou/{component}")
            _HELPER._decomposition(evaluation["total_error"], evaluation["neural_error"], evaluation["time_error"])
        else:
            for case in base.cases:
                _HELPER._validate_native_case(evaluation["cases"][case], base.to_dict(), case)


def _validate_saved_banks(directory, base, records):
    from nn_dk.bsde_evaluation import _stopping_summary
    from nn_dk.bsde_study_evaluation import _read_reference_samples

    first = next(record["evaluation"] for record in records if record["evaluation"]["problem"] == "dk")
    rows, families, hashes = [], set(), {}
    for case, scope in product(base.cases, ("conditional", "ensemble")):
        bank = _read_reference_samples(directory / "common", base, case, scope)
        children = base.conditional_paths if scope == "conditional" else 1
        anchors = base.conditional_anchors if scope == "conditional" else base.outer_paths
        count = anchors*children
        expected_ids = [f"bsde-test/{base.config_hash}/{case}/{scope}/anchor{j//children}/child{j%children}"
                        for j in range(count)]
        _require(bank["family_ids"].tolist() == expected_ids
                 and not families.intersection(expected_ids), "Common reference family identities overlap or changed")
        families.update(expected_ids)
        for level in ("coarse", "fine"):
            _require(np.all(np.isfinite(bank[level])), "Nonfinite coarse/fine reference observations")
            stop = bank[f"{level}_stopped_at"]
            _require(stop.shape == (count,) and np.all(~np.isnan(stop)) and np.all(stop >= 0),
                     "Invalid saved reference stopping times")
            computed = _stopping_summary(stop, base.horizon, base.query_times)
            declared = first["cases"][case][f"{scope}_stopping"][level]
            _require(computed == declared, "Reference stopping JSON differs from the saved bank")
        for record in records:
            if record["evaluation"]["problem"] == "dk":
                result = record["evaluation"]["cases"][case]
                for suffix in ("stopping", "reference_diagnostics"):
                    _require(result[f"{scope}_{suffix}"] == first["cases"][case][f"{scope}_{suffix}"],
                             "Study models did not share identical declared reference diagnostics")
        paths = sorted((directory / "common" / "evaluation_cache" / case / scope).glob("chunk_*.npz"))
        bank_hashes = {str(path.relative_to(directory)): _sha(path) for path in paths}
        hashes.update(bank_hashes)
        rows.append({"case": case, "scope": scope, "independent_families": count,
                     "initial_anchors": anchors, "continuations_per_anchor": children,
                     "bank_chunk_count": len(paths), "bank_files_sha256": _stable_hash(bank_hashes),
                     "fine_stopping": first["cases"][case][f"{scope}_stopping"]["fine"],
                     "coarse_stopping": first["cases"][case][f"{scope}_stopping"]["coarse"]})
    return rows, hashes


def _validate_prediction_caches(directory, base, records):
    from nn_dk.bsde_evaluation import _load_chunk, aggregate_conditional_moments
    from nn_dk.types import seeded_rng

    hashes = {}
    for case in base.cases:
        path = directory / "common" / "study_initial_cache" / f"{case}_ensemble.npz"
        initial, metadata = _load_chunk(path, base, {"case": case, "kind": "ensemble",
            "n_initial": base.outer_paths, "target_id": base.target_id("dk")})
        masses = initial["initial"]
        initial_seed = int(seeded_rng(base.seed, "bsde-test-initial", case, "ensemble").integers(0, 2**63-1))
        _require(metadata.get("initial_seed") == initial_seed
                 and masses.shape == (base.outer_paths, base.n_cells)
                 and np.all(np.isfinite(masses)) and np.min(masses) >= 0
                 and np.max(np.abs(masses.sum(-1)-1)) < 1e-12,
                 "Shared initial-state prediction bank changed")
        hashes[str(path.relative_to(directory))] = _sha(path)
    for record in records:
        if record["evaluation"]["problem"] != "dk":
            continue
        _require(set(record.get("prediction_caches", {})) == set(base.cases),
                 "Missing per-initial prediction cache for paired ensemble comparisons")
        for case, location in record["prediction_caches"].items():
            relative = Path(location["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts,
                     "Study prediction cache path must remain inside the common directory")
            path = directory / "common" / relative
            arrays, _ = _load_chunk(path, base, {"study_model_id": record["study_model_id"],
                "case": case, "kind": "ensemble", "target_id": base.target_id("dk")})
            mean, covariance = arrays["mean"], arrays["scaled_covariance"]
            _require(mean.shape == (base.outer_paths, len(base.query_times), 4)
                     and covariance.shape == (*mean.shape, 4)
                     and np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))
                     and np.max(np.abs(covariance-np.swapaxes(covariance, -1, -2))) < 1e-10
                     and np.linalg.eigvalsh(covariance).min() >= -1e-9,
                     "Invalid per-initial prediction cache for paired bootstrap")
            predicted_mean, predicted_covariance = aggregate_conditional_moments(mean, covariance,
                                                                                base.n_particles)
            comparison = record["evaluation"]["cases"][case]["ensemble_neural_error"]
            _require(np.allclose(predicted_mean, comparison["prediction_mean"], atol=1e-12, rtol=1e-10)
                     and np.allclose(predicted_covariance, comparison["prediction_scaled_covariance"],
                                     atol=1e-12, rtol=1e-10),
                     "Per-initial prediction cache disagrees with evaluated ensemble moments")
            hashes[str(path.relative_to(directory))] = _sha(path)
    return hashes


def _validate_contrasts(contrasts, records, base):
    from nn_dk.bsde_study_evaluation import paired_metric_contrasts

    expected = paired_metric_contrasts(records)
    key_fields = ("problem", "case", "scope", "optimizer_seed", "factor",
                  "baseline_model_id", "candidate_model_id")
    def key(row):
        return tuple(row[field] for field in key_fields)
    actual = {key(row): row for row in contrasts}
    _require(len(actual) == len(contrasts) == len(expected)
             and set(actual) == {key(row) for row in expected}, "Missing/duplicate/undeclared paired study contrast")
    effects = []
    for reference in expected:
        row = actual[key(reference)]
        for field, value in reference.items():
            if field != "uncertainty":
                _require(row.get(field) == value, "Paired contrast factors or point estimates changed")
        uncertainty = row.get("paired_uncertainty", {})
        _require(uncertainty.get("anchor_generalization_uncertainty_included")
                 is (row["problem"] == "dk" and row["scope"] == "ensemble"),
                 "Paired contrast confuses fixed-anchor and population sampling")
        if row["problem"] == "dk":
            _require(uncertainty.get("n_bootstrap") == base.bootstrap_samples
                     and uncertainty.get("confidence_level") == .95
                     and uncertainty.get("optimizer_uncertainty_included") is False
                     and uncertainty.get("reference_bias_certified") is False,
                     "Paired bootstrap allocation/confidence/scope mismatch")
        else:
            _require(uncertainty.get("reference_sampling_error") == 0.,
                     "OU analytic contrasts must have zero reference Monte Carlo error")
        intervals = uncertainty.get("candidate_minus_baseline_ci95", {})
        _require(set(intervals) == set(row["candidate_minus_baseline"]), "Missing paired metric intervals")
        metrics = {}
        for metric, value in row["candidate_minus_baseline"].items():
            interval = intervals[metric]
            _require(len(interval) == 2 and all(math.isfinite(v) for v in interval)
                     and interval[0] <= interval[1], "Invalid paired pointwise interval")
            if row["problem"] == "ou":
                _require(interval == [value, value], "Fixed-bank analytic OU interval changed")
            decision = ("improved_on_declared_bank" if interval[1] < 0 else
                        "worse_on_declared_bank" if interval[0] > 0 else
                        "equal_on_fixed_analytic_bank" if interval == [0., 0.] and row["problem"] == "ou"
                        else "unresolved")
            metrics[metric] = {"candidate_minus_baseline": value, "pointwise_ci95": interval,
                               "comparison": decision}
        effects.append({**{field: row[field] for field in key_fields},
                        **{field: row[field] for field in ("baseline_architecture", "candidate_architecture",
                             "baseline_train_anchors", "candidate_train_anchors", "fixed_train_paths")},
                        "metrics": metrics, "sampling_scope": uncertainty["method"],
                        "optimizer_seed_pooled": False, "multiplicity_adjusted": False})
    return effects


def summarize(directory, output_dir=None, *, allow_smoke=False):
    """Reject partial/tampered runs before producing any derived results."""
    from nn_dk.bsde_study_reporting import _comparison_rows, _validate_records

    directory = Path(directory).resolve()
    manifest = _read(directory / "manifest.json")
    _require(manifest.get("status") == "complete",
             "Study summary requires manifest status='complete'; held-out outcomes remain unopened")
    _require(manifest.get("workflow") == "conditional-bsde-anchor-study-v1", "Expected the BSDE factorial-study workflow")
    _require(_HELPER._source_hash() == manifest.get("source_hash"),
             "Current package source differs from the frozen study source hash")
    study, preregistration = _validate_design(directory, manifest, allow_smoke)
    base = study.base_config
    calibration = _read(directory / "common" / "calibration.json")
    _validate_calibration(calibration, base, manifest["source_hash"])
    records = _read(directory / "evaluations.json")
    _validate_records(directory, study, calibration, records)
    models, input_hashes = _validate_inventory(directory, manifest, study, records)
    _validate_comparisons(records, base)
    banks, bank_hashes = _validate_saved_banks(directory, base, records)
    input_hashes.update(bank_hashes)
    input_hashes.update(_validate_prediction_caches(directory, base, records))
    contrasts = _read(directory / "contrasts.json")
    effects = _validate_contrasts(contrasts, records, base)
    rows = _comparison_rows(records)
    expected_rows = len(base.optimizer_seeds)*len(study.allocations)*len(study.architectures)*(3+5*len(base.cases))
    expected_contrasts = len(base.optimizer_seeds)*(len(study.allocations)+len(study.architectures))*(1+2*len(base.cases))
    _require(len(rows) == expected_rows and len(effects) == expected_contrasts,
             "Comparison/contrast row count differs from the preregistered factorial allocation")
    for relative in ("manifest.json", "config.json", "base_config.json", "preregistration.json",
                     "common/calibration.json", "evaluations.json", "contrasts.json"):
        input_hashes[relative] = _sha(directory / relative)
    result = {"schema_version": 1, "status": "complete-artifact-summary",
              "run_directory": str(directory), "primary_scientific_allocation": study.name == "bsde-anchor-study",
              "model_count": len(models), "comparison_count": len(rows), "paired_contrast_count": len(effects),
              "optimizer_seeds": list(base.optimizer_seeds), "allocations": [list(v) for v in study.allocations],
              "validation": list(study.validation), "training_paths_per_fit": study.training_paths,
              "accuracy_target": {"normalized_tolerance": base.tolerance,
                  "each_uncertainty_budget": base.uncertainty_budget,
                  "gate": "Maximum pointwise errors and uncertainty intervals; RMSE below 5% alone does not pass",
                  "reference_bias": "Residual fine-reference DK bias remains uncertified"},
              "source_hash": manifest["source_hash"], "study_config_hash": study.config_hash,
              "base_config_hash": base.config_hash, "preregistration_hash": manifest["preregistration_hash"],
              "fit_lock_sha256": _stable_hash(manifest["fit_lock"]), "test_seed": base.seed,
              "test_bank_ids": preregistration["test_bank_ids"],
              "run_elapsed_seconds": manifest.get("elapsed_seconds"),
              "training_seconds_all_distinct_fits": sum(model["training_seconds"] for model in models),
              "shared_forward_pool_generation_seconds": manifest.get("shared_pool_generation_seconds", {}),
              "calibration_passed": True, "calibration_families_per_profile": base.refinement_paths,
              "models": models, "rows": rows, "paired_effects": effects, "reference_banks": banks,
              "provenance": {"analysis_script_sha256": _sha(Path(__file__)),
                             "analysis_helpers_sha256": _sha(_HELPER_PATH), "input_sha256": input_hashes,
                             "scope": "JSON, checkpoint bytes and saved observable banks; checkpoints are not deserialized; HDF5 pools and full forward histories are not reread",
                             "chronology": "Recorded unlock time and local artifact modification times support lock-before-test ordering; this is not cryptographic attestation"},
              "interpretation": {"allocation": "Coverage and replication change together at a fixed total training-path budget",
                  "conditional_intervals": "Paired whole-trajectory resampling within each fixed anchor; conditional population generalization uncertainty is excluded",
                  "ensemble_intervals": "Paired whole initial-histogram and forcing families",
                  "optimizer_seeds": "All reported individually; no best-seed selection or pooling as Monte Carlo samples",
                  "paired_intervals": "Pointwise 95% intervals; multiplicity unadjusted; negative differences favor the candidate",
                  "error_components": "Neural errors use the training-step Euler law; total errors use fine DK or continuous OU references; signed components add but RMSEs do not",
                  "reference_bias": "Finite stopped-DK refinement contrasts do not certify residual continuous-time bias",
                  "covariance": "Same-time 4x4 covariance only; no joint temporal covariance or density-path convergence claim",
                  "speedup": "No matched-accuracy speed advantage is inferred"}}
    output = Path(output_dir).resolve() if output_dir else directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    summary_path, table_path = output / "study_summary.json", output / "study_summary_table.csv"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(summary_path)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = table_path.with_suffix(".csv.tmp")
    temporary.write_text(stream.getvalue())
    temporary.replace(table_path)
    return {"summary": str(summary_path), "table": str(table_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-smoke", action="store_true", help="Permit the exact software-smoke preset; never certify it as the scientific allocation")
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.output, allow_smoke=args.allow_smoke), indent=2))


if __name__ == "__main__":
    main()
