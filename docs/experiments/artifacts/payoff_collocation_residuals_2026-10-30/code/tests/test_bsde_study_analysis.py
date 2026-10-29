"""Artifact-only study audits reject partial, reduced and altered experiments."""

import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from nn_dk.bsde_evaluation import _cache_chunk, _provenance, _stopping_summary
from nn_dk.bsde_study import _design
from nn_dk.bsde_study_config import bsde_study_preset
from nn_dk.bsde_study_evaluation import paired_metric_contrasts, test_bank_identity
from nn_dk.types import seeded_rng

_PATH = Path(__file__).parents[1] / "analysis" / "summarize_bsde_study.py"
_SPEC = importlib.util.spec_from_file_location("bsde_study_analysis", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _comparison(base, anchors=None):
    shape = (() if anchors is None else (anchors,)) + (len(base.query_times),)
    return {"errors": {"mean_normalized_signed": np.zeros((*shape, 4)).tolist(),
            "scaled_covariance_normalized_signed": np.zeros((*shape, 4, 4)).tolist(),
            "mean_normalized_rmse": 0., "scaled_covariance_normalized_rmse": 0.,
            "maximum_mean_normalized_error": 0., "maximum_scaled_covariance_normalized_error": 0.,
            "scaled_variance_normalized_rmse": 0.},
            "uncertainty": {"available": True, "n_bootstrap": base.bootstrap_samples,
                "maximum_sampling_normalized_halfwidth": .1,
                "maximum_contrast_normalized_ci_endpoint": .1},
            "accuracy_gate": {"status": "unresolved", "reference_bias_resolved": False},
            "minimum_predicted_conditional_covariance_eigenvalue": 0.}


def _native_case(base):
    result = {key: _comparison(base, base.conditional_anchors if key.startswith("conditional") else None)
              for key in ("conditional_total_error", "conditional_neural_error", "ensemble_total_error",
                          "ensemble_neural_error", "ensemble_time_error")}
    for scope, count in (("conditional", base.conditional_anchors*base.conditional_paths),
                         ("ensemble", base.outer_paths)):
        result[f"{scope}_stopping"] = {level: _stopping_summary(np.full(count, np.inf), base.horizon,
                                                                base.query_times)
                                        for level in ("coarse", "fine")}
        result[f"{scope}_reference_diagnostics"] = {"all_stopped_paths_included": True,
            "maximum_mass_error": 0., "minimum_saved_mass": 1/base.n_cells}
    result["cost"] = {"reference_paths": base.outer_paths, "n_initial_anchors": base.outer_paths,
        "conditional_reference": {"reference_paths": base.conditional_anchors*base.conditional_paths,
                                   "n_initial_anchors": base.conditional_anchors}}
    result["ensemble_neural_error"].update(
        prediction_mean=np.zeros((len(base.query_times), 4)).tolist(),
        prediction_scaled_covariance=np.zeros((len(base.query_times), 4, 4)).tolist())
    return result


def _banks(directory, base):
    for case in base.cases:
        for scope in ("conditional", "ensemble"):
            children = base.conditional_paths if scope == "conditional" else 1
            count = base.conditional_anchors*children if scope == "conditional" else base.outer_paths
            initial_seed = int(seeded_rng(base.seed, "bsde-test-initial", case, scope).integers(0, 2**63-1))
            for start in range(0, count, base.reference_batch_size):
                end = min(count, start+base.reference_batch_size)
                shape = (end-start, len(base.query_times), 4)
                arrays = {"coarse": np.zeros(shape), "fine": np.zeros(shape),
                    "coarse_stopped_at": np.full(end-start, np.inf),
                    "fine_stopped_at": np.full(end-start, np.inf),
                    "family_ids": np.asarray([f"bsde-test/{base.config_hash}/{case}/{scope}/anchor{j//children}/child{j%children}"
                                               for j in range(start, end)])}
                metadata = {**_provenance(base), "case": case, "kind": scope,
                    "start": start, "end": end, "target_id": base.spec.target_id,
                    "dt": base.dt, "reference_dt": base.reference_dt,
                    "initial_seed": initial_seed, "all_stopped_paths_included": True}
                _cache_chunk(directory/"common"/"evaluation_cache"/case/scope/f"chunk_{start:08d}.npz",
                             arrays, metadata)
        _cache_chunk(directory/"common"/"study_initial_cache"/f"{case}_ensemble.npz",
                     {"initial": np.full((base.outer_paths, base.n_cells), 1/base.n_cells)},
                     {**_provenance(base), "case": case, "kind": "ensemble",
                      "n_initial": base.outer_paths, "target_id": base.target_id("dk"),
                      "initial_seed": initial_seed})


def _save_records(directory, records):
    _write(directory/"evaluations.json", records)
    for record in records:
        evaluation = record["evaluation"]
        identity = analysis._stable_hash(evaluation["model"])[:20]
        _write(directory/"common"/"evaluation"/f"{evaluation['problem']}_seed{evaluation['seed']}_{identity}.json",
               evaluation)


def _completed_fixture(directory):
    study = bsde_study_preset("bsde-study-smoke")
    base = study.base_config
    source = analysis._HELPER._source_hash()
    preregistration = _design(study)
    _write(directory/"config.json", study.to_dict())
    _write(directory/"base_config.json", base.to_dict())
    _write(directory/"preregistration.json", preregistration)
    calibration = {**_provenance(base), "complete": True, "passed": True,
        "brownian": {"complete": True, "passed": True}, "ou": {"complete": True, "passed": True},
        "native_refinement": {case: {"complete": True, "passed": True,
            "n_independent_families": base.refinement_paths,
            "family_ids": [f"audit/{case}/{j}" for j in range(base.refinement_paths)],
            "reference_steps": [base.reference_dt/d for d in (1, 2, 4)],
            "all_stopped_paths_included": True, "maximum_mass_error": 0., "minimum_saved_mass": 1/base.n_cells}
            for case in base.cases}}
    _write(directory/"common"/"calibration.json", calibration)
    _banks(directory, base)
    records, inventory = [], []
    for problem in ("ou", "dk"):
        for anchors, children in study.allocations:
            for architecture in study.architectures:
                for seed in base.optimizer_seeds:
                    key = (problem, architecture, anchors, children, seed)
                    checkpoint = analysis._checkpoint(directory, key)
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(f"opaque checkpoint {key}".encode())
                    sha = analysis._sha(checkpoint)
                    prefix = "native-dk" if problem == "dk" else "linear-ou"
                    kind = prefix + ("-bsde-structured-moments-v1" if architecture == "structured" else "-bsde-moments-v1")
                    train_fp = analysis._stable_hash((problem, anchors, children))
                    valid_fp = analysis._stable_hash((problem, "validation"))
                    metadata = {"complete": True, "problem": problem, "architecture": architecture,
                        "seed": seed, "kind": kind, "allocation": [anchors, children],
                        "study_config_hash": study.config_hash, "training_anchors": anchors,
                        "training_paths": anchors*children, "validation_anchors": study.validation[0],
                        "validation_paths": study.validation[0]*study.validation[1],
                        "training_view_fingerprint": train_fp, "validation_fingerprint": valid_fp,
                        "epochs": 1, "history": [{"epoch": 1}], "training_seconds": .01}
                    _write(checkpoint.with_suffix(".json"), metadata)
                    identity = {"problem": problem, "seed": seed, "checkpoint_sha256": sha}
                    evaluation = {**_provenance(base), "complete": True, "problem": problem,
                        "seed": seed, "times": list(base.query_times), "step_size": base.dt,
                        "model": identity, "training": metadata}
                    if problem == "ou":
                        evaluation.update(n_test_anchors=base.ou_test_anchors,
                            **{component: _comparison(base, base.ou_test_anchors)
                               for component in ("neural_error", "total_error", "time_error")})
                    else:
                        evaluation["cases"] = {case: _native_case(base) for case in base.cases}
                    factors = {"architecture": architecture, "train_anchors": anchors,
                               "train_continuations": children, "train_paths": anchors*children,
                               "optimizer_seed": seed}
                    bank = test_bank_identity(base, problem)
                    record = {"factors": factors, "test_bank_id": bank,
                        "study_model_id": analysis._stable_hash({"bank": bank, "factors": factors, "checkpoint_sha256": sha}),
                        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha,
                        "study_config_hash": study.config_hash, "evaluation_config_hash": base.config_hash,
                        "training_view_fingerprint": train_fp, "validation_fingerprint": valid_fp,
                        "evaluation": evaluation}
                    records.append(record)
                    if problem == "dk":
                        record["prediction_caches"] = {}
                        for case in base.cases:
                            relative = Path("study_prediction_cache") / record["study_model_id"] / f"{case}_ensemble.npz"
                            _cache_chunk(directory/"common"/relative,
                                {"mean": np.zeros((base.outer_paths, len(base.query_times), 4)),
                                 "scaled_covariance": np.zeros((base.outer_paths, len(base.query_times), 4, 4))},
                                {**_provenance(base), "case": case, "kind": "ensemble",
                                 "study_model_id": record["study_model_id"], "target_id": base.target_id("dk")})
                            record["prediction_caches"][case] = {"path": str(relative)}
                    inventory.append({"key": list(key), "checkpoint_sha256": sha,
                                      "training_fingerprint": train_fp, "validation_fingerprint": valid_fp})
    unlocked = time.time()
    _save_records(directory, records)
    contrasts = paired_metric_contrasts(records)
    for contrast in contrasts:
        uncertainty = {"method": "analytic reference" if contrast["problem"] == "ou" else "paired trajectories",
            "anchor_generalization_uncertainty_included": contrast["problem"] == "dk" and contrast["scope"] == "ensemble",
            "candidate_minus_baseline_ci95": {key: [value, value] for key, value in contrast["candidate_minus_baseline"].items()}}
        if contrast["problem"] == "ou":
            uncertainty["reference_sampling_error"] = 0.
        else:
            uncertainty.update(n_bootstrap=base.bootstrap_samples, confidence_level=.95,
                               optimizer_uncertainty_included=False, reference_bias_certified=False)
        contrast["paired_uncertainty"] = uncertainty
    _write(directory/"contrasts.json", contrasts)
    manifest = {"workflow": "conditional-bsde-anchor-study-v1", "status": "complete",
        "config": study.to_dict(), "config_hash": study.config_hash, "source_hash": source,
        "expected_models": study.expected_models, "completed_evaluations": study.expected_models,
        "preregistration_hash": analysis._stable_hash(preregistration), "fit_lock": inventory,
        "created_unix": unlocked-10., "test_unlocked_unix": unlocked, "elapsed_seconds": 1.}
    _write(directory/"manifest.json", manifest)
    return records


def test_partial_study_stops_before_opening_test_outcomes(tmp_path):
    _write(tmp_path/"manifest.json", {"status": "fitting"})
    with pytest.raises(ValueError, match="status='complete'.*unopened"):
        analysis.summarize(tmp_path)
    assert not (tmp_path/"report").exists()


def test_complete_artifact_summary_preserves_all_arms_and_pointwise_intervals(tmp_path):
    _completed_fixture(tmp_path)
    paths = analysis.summarize(tmp_path, allow_smoke=True)
    result = json.loads(Path(paths["summary"]).read_text())
    assert (result["model_count"], result["comparison_count"], result["paired_contrast_count"]) == (12, 78, 25)
    assert result["primary_scientific_allocation"] is False
    assert result["test_seed"] == 20261025
    assert len(result["reference_banks"]) == 4
    assert {tuple(model["key"][1:4]) for model in result["models"]} == {
        (architecture, anchors, children) for architecture in ("mlp", "structured")
        for anchors, children in ((4, 4), (8, 2), (16, 1))}
    assert all(effect["optimizer_seed_pooled"] is False and effect["multiplicity_adjusted"] is False
               for effect in result["paired_effects"])
    assert {effect["factor"] for effect in result["paired_effects"]} == {"architecture", "independent_initial_states"}
    assert Path(paths["table"]).read_text().count("\n") == 79
    assert b"\r\n" not in Path(paths["table"]).read_bytes()


def test_primary_analysis_rejects_the_smoke_allocation(tmp_path):
    _completed_fixture(tmp_path)
    with pytest.raises(ValueError, match="full allocation"):
        analysis.summarize(tmp_path)


def test_changed_checkpoint_or_lock_is_rejected(tmp_path):
    records = _completed_fixture(tmp_path)
    checkpoint = Path(records[0]["checkpoint_path"])
    checkpoint.write_bytes(b"modified model bytes")
    with pytest.raises(ValueError, match="SHA mismatch"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_missing_paired_contrasts_is_rejected(tmp_path):
    _completed_fixture(tmp_path)
    contrasts = analysis._read(tmp_path/"contrasts.json")
    _write(tmp_path/"contrasts.json", contrasts[:-1])
    with pytest.raises(ValueError, match="paired study contrast"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_reduced_reference_allocation_cannot_be_hidden_by_complete_status(tmp_path):
    records = _completed_fixture(tmp_path)
    records[6]["evaluation"]["cases"]["uniform"]["ensemble_stopping"]["fine"]["n_paths"] -= 1
    _save_records(tmp_path, records)
    with pytest.raises(ValueError, match="Reference path allocation"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_saved_reference_bank_tamper_is_detected(tmp_path):
    _completed_fixture(tmp_path)
    path = tmp_path/"common"/"evaluation_cache"/"uniform"/"conditional"/"chunk_00000000.npz"
    with np.load(path, allow_pickle=False) as stream:
        values = {key: stream[key] for key in stream.files}
    values["fine_stopped_at"] = values["fine_stopped_at"].copy()
    values["fine_stopped_at"][0] = 0.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="stopping JSON differs"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_individual_evaluation_tamper_and_stale_preregistration_are_rejected(tmp_path):
    records = _completed_fixture(tmp_path)
    records[0]["evaluation"]["n_test_anchors"] -= 1
    _write(tmp_path/"evaluations.json", records)
    with pytest.raises(ValueError, match="immutable individual"):
        analysis.summarize(tmp_path, allow_smoke=True)
    preregistration = analysis._read(tmp_path/"preregistration.json")
    preregistration["independent_test"] = False
    _write(tmp_path/"preregistration.json", preregistration)
    with pytest.raises(ValueError, match="preregistration content/hash"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_changed_fit_lock_fingerprint_is_rejected(tmp_path):
    _completed_fixture(tmp_path)
    manifest = analysis._read(tmp_path/"manifest.json")
    manifest["fit_lock"][0]["training_fingerprint"] = "changed view"
    _write(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError, match="fit-lock inventory/hash"):
        analysis.summarize(tmp_path, allow_smoke=True)


def test_effect_signs_and_optimizer_seeds_remain_separate(tmp_path):
    records = _completed_fixture(tmp_path)
    second_seed = copy.deepcopy(records)
    for record in second_seed:
        record["factors"]["optimizer_seed"] = 1
        record["evaluation"]["seed"] = 1
        record["study_model_id"] += "/independent-seed1"
    records.extend(second_seed)
    for record in records:
        architecture = record["factors"]["architecture"]
        seed = record["factors"]["optimizer_seed"]
        error = .2 if architecture == "mlp" else .1 if seed == 0 else .3
        evaluation = record["evaluation"]
        comparisons = ([evaluation["neural_error"]] if evaluation["problem"] == "ou" else
                       [case[f"{scope}_neural_error"] for case in evaluation["cases"].values()
                        for scope in ("conditional", "ensemble")])
        for comparison in comparisons:
            comparison["errors"].update(mean_normalized_rmse=error,
                scaled_covariance_normalized_rmse=error, scaled_variance_normalized_rmse=error)
    contrasts = paired_metric_contrasts(records)
    base = bsde_study_preset("bsde-study-smoke").base_config
    for row in contrasts:
        point = row["candidate_minus_baseline"]
        analytic = row["problem"] == "ou"
        uncertainty = {"method": "analytic" if analytic else "paired whole trajectories",
            "anchor_generalization_uncertainty_included": not analytic and row["scope"] == "ensemble",
            "candidate_minus_baseline_ci95": {metric: [value, value] if analytic else [value-.01, value+.01]
                                               for metric, value in point.items()}}
        if analytic:
            uncertainty["reference_sampling_error"] = 0.
        else:
            uncertainty.update(n_bootstrap=base.bootstrap_samples, confidence_level=.95,
                               optimizer_uncertainty_included=False, reference_bias_certified=False)
        row["paired_uncertainty"] = uncertainty
    effects = analysis._validate_contrasts(contrasts, records, base)
    architecture = [row for row in effects if row["factor"] == "architecture"]
    assert len(architecture) == 30
    for row in architecture:
        expected = "improved_on_declared_bank" if row["optimizer_seed"] == 0 else "worse_on_declared_bank"
        assert row["metrics"]["mean_rmse"]["comparison"] == expected
        assert row["optimizer_seed_pooled"] is False


def test_primary_preset_requires_36_fits_234_rows_and_75_paired_contrasts():
    study = bsde_study_preset("bsde-anchor-study")
    base = study.base_config
    assert study.expected_models == 36
    assert len(base.optimizer_seeds)*len(study.allocations)*len(study.architectures)*(3+5*len(base.cases)) == 234
    assert len(base.optimizer_seeds)*(len(study.allocations)+len(study.architectures))*(1+2*len(base.cases)) == 75
