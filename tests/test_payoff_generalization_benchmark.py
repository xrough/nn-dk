"""Same-output costs, direct IID initialization and Gaussian marginal equivalence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.native_gaussian_baseline import iid_scaled_covariance, linear_gaussian  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "benchmark_payoff_generalization", Path(__file__).parents[1]/"analysis"/"benchmark_payoff_generalization.py")
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("ensemble", [False, True])
def test_marginal_optimized_gaussian_solver_matches_full_audited_ode(mixed, ensemble):
    cfg = benchmark.ReferenceConfig(n_cells=16)
    bank = (PayoffBank.mixed_terminal_integral if mixed else PayoffBank.terminal_high_modes)(
        cfg.spec, cfg.horizon)
    probabilities, _ = benchmark.profile_probabilities(cfg, "nonuniform")
    initial = probabilities if ensemble else benchmark.fresh_histograms(cfg, "nonuniform", 1)[0]
    covariance = iid_scaled_covariance(initial) if ensemble else None
    mean, scaled, metadata = benchmark.lga_marginals(
        initial, cfg.context("nonuniform"), cfg.spec, cfg.query_times, 1/128, bank, covariance)
    running_rows = np.flatnonzero(bank.integral_rows)
    full = linear_gaussian(initial, cfg.context("nonuniform"), cfg.spec, cfg.query_times, 1/128,
                           initial_scaled_covariance=covariance,
                           running_weights=bank.running_weights[running_rows], return_joint=False)
    projection = np.zeros((4, cfg.n_cells+len(running_rows)))
    projection[:, :cfg.n_cells] = bank.terminal_weights
    for column, row in enumerate(running_rows):
        projection[row, cfg.n_cells+column] = 1
    expected_mean = full["mean_augmented"] @ projection.T
    expected_covariance = np.einsum("ai,tij,bj->tab", projection,
                                    full["scaled_covariance_augmented"], projection)
    np.testing.assert_allclose(mean, expected_mean, atol=2e-16, rtol=1e-14)
    np.testing.assert_allclose(scaled, expected_covariance, atol=2e-16, rtol=1e-14)
    assert metadata["return_joint"] is False
    assert metadata["unused_transition_matrices_computed"] is False
    assert metadata["linearization_and_stopping_bias"] == "unresolved"


def test_direct_iid_histograms_are_independent_reproducible_and_state_law_correct():
    cfg = benchmark.ReferenceConfig()
    a = benchmark.fresh_histograms(cfg, "uniform", 32768)
    b = benchmark.fresh_histograms(cfg, "uniform", 32768)
    c = benchmark.fresh_histograms(cfg, "uniform", 32768, role="other-independent-namespace")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(a.sum(-1), 1)
    np.testing.assert_array_equal(a*cfg.n_particles, np.rint(a*cfg.n_particles))
    np.testing.assert_allclose(a.mean(0), 1/32, atol=7e-5)
    empirical = cfg.n_particles*np.cov(a, rowvar=False)
    np.testing.assert_allclose(empirical, iid_scaled_covariance(np.full(32, 1/32)), atol=.001)


class FakeModel:
    def __init__(self, bank):
        self.bank, self.complete, self.target_id = bank, True, "fake-target-"+bank.name
        cfg = benchmark.ReferenceConfig(n_cells=bank.n_cells)
        self.spec, self.context, self.horizon = cfg.spec, cfg.context("uniform"), bank.horizon
        self.provenance = {"source_hash": benchmark.SOURCE_FREEZE}

    def predict_conditional(self, initial, times):
        mean = ((initial @ self.bank.terminal_weights.T)[:, None]
                + np.asarray(times)[None, :, None]*(initial @ self.bank.running_weights.T)[:, None])
        covariance = np.broadcast_to(np.eye(4)[None, None]*(.1+np.asarray(times))[None, :, None, None],
                                      (*mean.shape, 4)).copy()
        return SimpleNamespace(mean=mean, scaled_covariance=covariance)


def test_chunked_predictions_and_total_covariance_match_direct_output():
    cfg = benchmark.ReferenceConfig(n_cells=16)
    model = FakeModel(PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))
    initial = benchmark.fresh_histograms(cfg, "nonuniform", 13)
    chunked = benchmark.predict_chunks(model, initial, cfg.query_times, chunk_size=4)
    direct = model.predict_conditional(initial, cfg.query_times)
    np.testing.assert_array_equal(chunked[0], direct.mean)
    np.testing.assert_array_equal(chunked[1], direct.scaled_covariance)
    mean, covariance = benchmark.mixture_moments(*chunked, n_particles=cfg.n_particles)
    centered = direct.mean-direct.mean.mean(0)
    expected = direct.scaled_covariance.mean(0)+cfg.n_particles*np.einsum(
        "bti,btj->tij", centered, centered)/(len(initial)-1)
    np.testing.assert_allclose(covariance, expected, atol=3e-16)
    np.testing.assert_array_equal(mean, direct.mean.mean(0))


def _fake_lock(root, cfg, monkeypatch):
    directory = root/"payoffs"
    directory.mkdir(parents=True)
    models, loaded = [], {}
    for bank in benchmark._banks(cfg):
        for seed in (0, 1, 2):
            for init in ("cold", "trunk-transfer"):
                name = f"{bank.name}_{seed}_{init}"
                checkpoint, metadata = directory/(name+".pt"), directory/(name+".json")
                checkpoint.write_bytes(name.encode())
                metadata.write_text(json.dumps({"training_seconds": 10.+seed}))
                loaded[str(checkpoint)] = FakeModel(bank)
                models.append({"checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "metadata_path": str(metadata),
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "complete": True, "source_hash": benchmark.SOURCE_FREEZE,
                    "bank": bank.name, "bank_hash": bank.bank_hash, "seed": seed, "init": init,
                    "target_id": "fake-target-"+bank.name})
    lock = {"complete": True, "source_hash": benchmark.SOURCE_FREEZE,
            "expected_models": 12, "models": models}
    (directory/"fit_lock.json").write_text(json.dumps(lock))
    (directory/"preregistration.json").write_text(json.dumps({"historical_transfers": [
        {"optimizer_seed": seed, "original_training_seconds": 100.+seed} for seed in (0, 1, 2)]}))
    monkeypatch.setattr(benchmark, "load_payoff_model", lambda path: loaded[str(path)])
    monkeypatch.setattr(benchmark, "source_hash", lambda: benchmark.SOURCE_FREEZE)
    return lock


def test_prepare_creates_timing_inputs_without_fit_results_or_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "source_hash", lambda: benchmark.SOURCE_FREEZE)
    cfg = benchmark.ReferenceConfig(n_cells=16)
    directory, plan = benchmark.prepare(tmp_path, cfg, repeats=1, warmup=0,
                                        batches=(1, 2), ensemble_paths=8, inference_chunk=4)
    assert (directory/"payoff_benchmark_inputs.npz").is_file()
    assert not (directory/"payoff_benchmark.json").exists()
    assert plan["identity"]["seed"] == 2026102615
    assert plan["identity"]["timing_pool_independent_of_scientific_references"]
    with pytest.raises(ValueError, match="plan"):
        benchmark.prepare(tmp_path, cfg, repeats=2, warmup=0,
                           batches=(1, 2), ensemble_paths=8, inference_chunk=4)


def test_complete_cost_smoke_contains_ensemble_initialization_and_analytic_classical_law(tmp_path, monkeypatch):
    cfg = benchmark.ReferenceConfig(n_cells=16)
    _fake_lock(tmp_path, cfg, monkeypatch)
    result = benchmark.run_benchmark(tmp_path, cfg, smoke=True, repeats=1, warmup=0,
                                     batches=(1, 2), ensemble_paths=8, inference_chunk=4)
    assert result["status"] == "complete" and result["record_count"] == 92
    assert not result["final_timing_after_external_worker_completion"]
    statistics = [row for row in result["records"] if row["method"] == "new-payoff-statistics-head"]
    ensembles = [row for row in statistics if row["role"] == "IID-ensemble"]
    assert len(ensembles) == 24
    for row in ensembles:
        assert row["initial_histograms"] == 8
        assert row["fresh_initialization_inference_aggregation"]["median_seconds"] > 0
        assert len(row["fresh_component_samples_seconds"]["initialization"]) == 1
        assert row["fresh_repeat_input_stream"]["same_draws_across_all_heads"]
        assert row["output_hashes"]["mean_shape"] == [3, 4]
        assert "not estimated" in row["initial_law_integration_uncertainty"]
    lga = [row for row in result["records"] if row["method"] == "native-linear-Gaussian-RK4"]
    assert all(row["initial_histograms"] == 0 for row in lga if row["role"] == "IID-ensemble")
    assert not any(row["step"] == 2**-10 and row.get("batch_states", 1) > 1 for row in lga)
    assert len(result["cold_restoration"]) == 12 and len(result["initialization"]) == 2
    assert len(result["lga_refinement"]) == 4
    for row in result["records"]:
        assert Path(row["output_artifact"]["path"]).is_file()
    replay = benchmark.run_benchmark(tmp_path, cfg, smoke=True, repeats=1, warmup=0,
                                     batches=(1, 2), ensemble_paths=8, inference_chunk=4)
    assert replay["record_count"] == 92


def test_incomplete_lock_and_unconfirmed_final_timing_are_rejected(tmp_path, monkeypatch):
    cfg = benchmark.ReferenceConfig(n_cells=16)
    lock = _fake_lock(tmp_path, cfg, monkeypatch)
    lock["models"][0]["complete"] = False
    (tmp_path/"payoffs"/"fit_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="incomplete"):
        benchmark._lock(tmp_path, cfg)
    with pytest.raises(ValueError, match="final-timing"):
        benchmark.run_benchmark(tmp_path, cfg)
