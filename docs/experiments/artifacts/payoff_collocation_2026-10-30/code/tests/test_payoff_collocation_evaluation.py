"""Complete crossed inventories, fresh IID pools and strict numeric replay."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import evaluate_payoff_collocation_study as evaluation  # noqa: E402
from analysis import evaluate_payoff_regimes as historical  # noqa: E402
from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis.compiled_native import compiler_available  # noqa: E402


def tiny_plan():
    return runner.StudyPlan(horizons=(1/64,), kappas=(.75,), n_cells=8, n_particles=512,
        train_paths=1, validation_paths=1, horizon_samples=1, width=2,
        adam_steps=1, lbfgs_steps=0, lbfgs_max_iter=1, validation_every=1,
        batch_size=4, quadrature_nodes=2, state_step=1/1024, reference_step=1/1024,
        conditional_anchors=1, paths_per_group=8, refinement_paths=4, chunk_paths=4)


def fit_locked(root):
    torch.set_num_threads(1)
    plan = tiny_plan()
    runner.prepare(root, plan)
    for seed in plan.optimizer_seeds:
        runner.fit_seed(root, seed)
    runner.lock(root)
    return plan


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    root = tmp_path_factory.mktemp("crossed-evaluation")/"study"
    plan = fit_locked(root)
    patch = pytest.MonkeyPatch()
    # This software fixture has no historical studies. Do not scan the user's
    # real large run banks while proving the temporary workflow's contract.
    patch.setattr(evaluation, "historical_identifier_inventory", lambda *args: (set(), []))
    prepared = evaluation.prepare(root)
    assert prepared["contract"]["counts"]["trained_models"] == 12
    assert not list((root/"reference").rglob("reference.h5"))
    assert not (root/"deployment/initials").exists()
    runner.freeze_evaluation(root)
    runner.generate_reference(root)
    evaluation.evaluate(root)
    yield root, plan
    patch.undo()


def test_successor_estimators_are_isolated_and_all_variants_seeds_retained():
    assert historical.SEED == 2026102819 and historical.KIND != evaluation.KIND
    assert evaluation._base._adapter_origin == historical.__file__
    assert evaluation._base.prepare.__globals__["SEED"] == evaluation.SEED
    models = [{**slot, "checkpoint_sha256": "literal-test"} for slot in runner.slots()]
    assert len(models) == 72
    for case in runner.case_names():
        entries = evaluation.model_entries({"models": models}, case)
        assert len(entries) == 13
        assert len({(row["proposal"], row["loss"], row["seed"]) for row in entries[:-1]}) == 12
        assert entries[-1]["untrained"] and entries[-1]["family"] == "untrained-repaired"
        removed = [row for row in models if row["name"] != entries[0]["name"]]
        with pytest.raises(ValueError, match="Every crossed"):
            evaluation.model_entries({"models": removed}, case)
        duplicated = removed+[entries[1]]
        with pytest.raises(ValueError, match="Every crossed"):
            evaluation.model_entries({"models": duplicated}, case)


def test_primary_evidence_uses_all_twelve_actual_fits_and_independent_iid_initials(completed):
    root, plan = completed
    evidence = evaluation.primary_records(root, require_timing=False)
    assert len(evidence["lock"]["models"]) == 12
    assert evidence["original_accuracy_records"] == len(evidence["records"]) == 64
    assert evidence["independent_initial_records"] == len(evidence["paired_initial_records"]) == 26
    assert len(evidence["groups"]) == 4 and len(evidence["paired_initial_groups"]) == 2
    for group in evidence["groups"]:
        assert len(group["NN_versus_uniform_huber"]) == 9
        for contrast in group["NN_versus_uniform_huber"].values():
            assert len(contrast["records"]) == 3
            assert contrast["both_predictor_initial_pool_uncertainties_included"]
            assert "pointwise" in contrast["interval_scope"]
        if "__ensemble__" in group["group"]:
            assert group["primary_NN_initial_integration"] == "independent-of-reference"
    for record in evidence["records"]:
        if "__ensemble__" in record["identity"]["group"] and record["method"]["family"] in (
                "repaired", "untrained-repaired"):
            assert record["initial_pool_independent_of_reference"]
            assert record["reference_and_initial_bootstrap_independent"]
            assert record["bootstrap_counts"]["reference_sha256"] != record["bootstrap_counts"]["initial_sha256"]
    for pool in evaluation._deployment_prepare(root)["pools"]:
        packet = runner._read(root/"deployment/initials"/pool["regime"]/(pool["profile"]+".json"))
        reference = runner.load_reference(root, pool["regime"], pool["group"])
        with np.load(packet["arrays"]["path"]) as arrays:
            assert not set(arrays["family_ids"].astype(str)) & set(reference["family_ids"].astype(str))
            assert arrays["initial"].shape == (plan.paths_per_group, plan.n_cells)


def test_completed_evaluation_replays_without_new_native_draws(completed, monkeypatch):
    root, _ = completed
    evidence = evaluation.primary_records(root, require_timing=False)
    original = {path: Path(path).read_bytes() for path in evidence["inputs"] if path.endswith((".pt", ".npz"))}

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed numeric predictions must replay without new native draws")

    monkeypatch.setattr(evaluation._base, "_native_mc", forbidden)
    monkeypatch.setattr(evaluation._deployment, "seeded_rng", forbidden)
    evaluation.evaluate(root)
    for path, data in original.items():
        assert Path(path).read_bytes() == data


def test_corrupt_numeric_data_and_missing_variant_block_complete_report(completed):
    root, plan = completed
    evidence = evaluation.primary_records(root, require_timing=False)
    data = Path(evidence["records"][0]["arrays"]["path"])
    original = data.read_bytes()
    try:
        data.write_bytes(original[:-1]+bytes([original[-1] ^ 1]))
        with pytest.raises(ValueError, match="checksum"):
            evaluation.primary_records(root, require_timing=False)
    finally:
        data.write_bytes(original)
    case = runner.case_names(plan)[0]
    group = runner.reference_configuration(case, plan).group_keys[0]
    path = root/"comparison/accuracy"/case/group/"group.json"
    index_path = root/"comparison/accuracy_index.json"
    group_bytes, index_bytes = path.read_bytes(), index_path.read_bytes()
    try:
        packet = runner._read(path)
        packet["records"].pop(0)
        runner._json(path, packet)
        index = runner._read(index_path)
        index["group_sha256"][case+"/"+group] = runner._sha(path)
        runner._json(index_path, index)
        with pytest.raises(ValueError, match="method/group"):
            evaluation.primary_records(root, require_timing=False)
    finally:
        path.write_bytes(group_bytes)
        index_path.write_bytes(index_bytes)


def test_initial_and_reference_bootstraps_are_independent_but_common_across_variants():
    reference, initials = evaluation._deployment.independent_bootstraps("T1_k6", "uniform", 64, 64)
    again = evaluation._deployment.independent_bootstraps("T1_k6", "uniform", 64, 64)
    assert not np.array_equal(reference, initials)
    np.testing.assert_array_equal(reference, again[0])
    np.testing.assert_array_equal(initials, again[1])
    values = np.random.default_rng(71).normal(size=(64, 3, 4))
    mean, covariance = evaluation._base.empirical(values)
    candidate = {"mean": mean, "scaled_covariance": covariance,
        "bootstrap_mean": np.broadcast_to(mean, (128, *mean.shape)).copy(),
        "bootstrap_scaled_covariance": np.broadcast_to(covariance, (128, *covariance.shape)).copy()}
    candidate["bootstrap_mean"][::2] += 10
    packet = evaluation.predictor_advantage(values, candidate, copy.deepcopy(candidate), reference, 1)
    assert all(row["mean_advantage_percentile95"] == [0., 0.] for row in packet["records"])
    assert all(row["covariance_advantage_percentile95"] == [0., 0.] for row in packet["records"])


def test_final_timing_requires_all_worker_leases_and_explicit_attestation(completed):
    root, _ = completed
    with pytest.raises(ValueError, match="heavy workers"):
        evaluation.benchmark(root)
    with runner._exclusive(root/"payoffs/execution/.seed0.lock"):
        with pytest.raises(RuntimeError, match="Another process owns"):
            evaluation.benchmark(root, final_timing=True)


def test_literal_final_timings_keep_all_variants_and_same_call_components(completed):
    root, _ = completed
    packet = evaluation.benchmark(root, final_timing=True)
    assert packet["complete"] and len(packet["records"]) == 116
    assert len(packet["loading"]) == 26
    assert packet["all_study_worker_leases_held"]
    for row in packet["records"]:
        assert len(row["samples_seconds"]) == row["repeats"]
        assert all(len(values) == row["repeats"] for values in row["warm_components"].values())
        if row["family"] == "repaired" and row["role"] == "ensemble":
            assert set(row["warm_components"]) == {"initialization_seconds", "prediction_seconds", "aggregation_seconds"}
    evidence = evaluation.primary_records(root)
    assert evidence["timing"] == packet
    original_bytes = (root/"comparison/timing.json").read_bytes()
    assert evaluation.benchmark(root, final_timing=True) == packet
    assert (root/"comparison/timing.json").read_bytes() == original_bytes
    original = runner._read(root/"comparison/timing.json")
    try:
        corrupted = copy.deepcopy(original)
        corrupted["records"][0]["samples_seconds"][0] += 1
        runner._json(root/"comparison/timing.json", corrupted)
        with pytest.raises(ValueError, match="checksum"):
            evaluation.primary_records(root)
    finally:
        runner._json(root/"comparison/timing.json", original)


def test_missing_fit_lock_prevents_preparation_and_outside_evidence_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluation.prepare(tmp_path/"not-a-study")
    outside = tmp_path/"older-final-data.npz"
    outside.write_bytes(b"must not reuse old outcomes")
    with pytest.raises(ValueError, match="older study"):
        evaluation._artifact(outside, tmp_path/"new-study", {})
