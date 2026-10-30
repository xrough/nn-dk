"""A0 reports distinguish moment calibration from uncertified DK bias audits."""

import json
import time
from dataclasses import replace

import h5py
import numpy as np
import pytest

from nn_dk import dk_calibration
from nn_dk.convergence_config import ConvergenceConfig


def tiny_config(**changes):
    base = ConvergenceConfig(
        n_cells=8, n_particles=2048, cases=("uniform",), horizon=.0625,
        query_times=(.03125, .0625), reference_dt=1 / 256,
        step_sizes=(1 / 64, 1 / 128), proposal_dt=1 / 64,
        train_path_counts=(2,), widths=(2,), base_width=2, optimizer_seeds=(0,),
        states_per_path=2, validation_paths=2, coefficient_test_paths=2,
        diagnostic_paths=2, final_paths=2, refinement_paths=3,
        trajectory_batch_size=2, evaluation_batch_size=2, bootstrap_samples=8)
    return replace(base, **changes)


@pytest.fixture(autouse=True)
def stable_source_during_parallel_development(monkeypatch):
    # Other agents may edit unrelated source while these tests execute.
    monkeypatch.setattr(dk_calibration, "source_hash", lambda: "test-source-fingerprint")


def test_native_calibration_independent_matrix_moments_and_uncertified_refinement(tmp_path):
    config = tiny_config(cases=("uniform", "nonuniform", "interacting"))
    result = dk_calibration.calibrate_native(config, tmp_path)
    assert result["complete"]
    assert result["linear_gaussian"]["status"] == "passed"
    assert result["linear_gaussian"]["conserved_null_mode"]["passed"]
    assert result["linear_gaussian"]["exact"]["metadata"]["stopped"] is False
    rows = result["linear_gaussian"]["euler_refinement"]
    assert [row["step_size"] for row in rows] == list(config.step_sizes)
    assert all(row["matrix_spectral_max_abs_difference"] < 1e-12 for row in rows)
    assert rows[1]["errors"]["scaled_covariance_normalized_rmse"] < rows[0]["errors"]["scaled_covariance_normalized_rmse"]
    for case in config.cases:
        refinement = result["native_refinement"][case]
        assert refinement["completed_paths"] == config.refinement_paths
        assert refinement["all_stopped_paths_included"]
        assert refinement["temporal_order"] is None
        assert set(refinement["comparisons"]) == {"coarse_vs_half", "half_vs_quarter", "coarse_vs_quarter"}
        bias = result["reference_bias"][case]
        assert bias["status"] == "uncertified"
        assert bias["normalized_bound"] >= 0
        assert np.isfinite(bias["empirical_normalized_difference"])
        assert bias["empirical_contrast_ci95"]
        comparison = refinement["comparisons"]["half_vs_quarter"]
        assert comparison["paired"]
        assert comparison["prediction"]["n_independent_paths"] == config.refinement_paths
        assert not comparison["accuracy_gate"]["reference_bias_resolved"]
    assert json.loads((tmp_path / "calibration.json").read_text()) == result
    json.dumps(result, allow_nan=False)


def test_native_calibration_cached_resume_and_source_config_validation(tmp_path, monkeypatch):
    config = tiny_config()
    result = dk_calibration.calibrate_native(config, tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("Completed calibration must not regenerate trajectories")

    monkeypatch.setattr(dk_calibration, "coupled_native_refinement", unexpected)
    assert dk_calibration.calibrate_native(config, tmp_path) == result
    with pytest.raises(ValueError, match="source/configuration mismatch"):
        dk_calibration.calibrate_native(replace(config, seed=config.seed + 1), tmp_path)
    monkeypatch.setattr(dk_calibration, "source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="source/configuration mismatch"):
        dk_calibration.calibrate_native(config, tmp_path)


def test_native_calibration_deadline_commits_each_coupled_family_and_resumes(tmp_path, monkeypatch):
    config = tiny_config()
    # Expire after linear calibration and exactly one complete coupled family.
    counter = iter(range(100))
    real_monotonic = time.monotonic
    monkeypatch.setattr(dk_calibration.time, "monotonic", lambda: next(counter))
    partial = dk_calibration.calibrate_native(config, tmp_path, deadline=5)
    assert not partial["complete"]
    assert partial["native_refinement"]["uniform"]["completed_paths"] == 1
    with h5py.File(tmp_path / "calibration_refinement_uniform.h5", "r") as store:
        initial_saved = store["quarter/states"][0]
        np.testing.assert_array_equal(store["completed"][:], [True, False, False])
    monkeypatch.setattr(dk_calibration.time, "monotonic", real_monotonic)
    actual_refinement = dk_calibration.coupled_native_refinement
    generated = []

    def counted(*args, **kwargs):
        generated.append(kwargs["family_ids"])
        return actual_refinement(*args, **kwargs)

    monkeypatch.setattr(dk_calibration, "coupled_native_refinement", counted)
    resumed = dk_calibration.calibrate_native(config, tmp_path)
    assert resumed["complete"]
    assert len(generated) == 2
    with h5py.File(tmp_path / "calibration_refinement_uniform.h5", "r") as store:
        np.testing.assert_array_equal(store["quarter/states"][0], initial_saved)


def test_native_calibration_stopped_paths_cannot_certify_reference_bias(tmp_path):
    # Two particles in eight cells guarantee initial boundary exits on every path.
    config = tiny_config(n_particles=2)
    result = dk_calibration.calibrate_native(config, tmp_path)
    comparison = result["native_refinement"]["uniform"]["comparisons"]["half_vs_quarter"]
    assert comparison["prediction"]["diagnostics"]["initially_stopped"] == config.refinement_paths
    assert comparison["prediction"]["n_independent_paths"] == config.refinement_paths
    assert comparison["prediction"]["stopping"]["probability_by_horizon"] == 1
    assert result["reference_bias"]["uniform"]["normalized_bound"] == 0
    assert result["reference_bias"]["uniform"]["status"] == "uncertified"
    assert comparison["accuracy_gate"]["status"] == "unresolved"


def test_native_calibration_expired_budget_and_existing_manifest(tmp_path):
    config = tiny_config()
    result = dk_calibration.calibrate_native(config, tmp_path, deadline=time.monotonic() - 1)
    assert not result["complete"]
    assert not result["linear_gaussian"]
    assert not result["native_refinement"]
    assert list(tmp_path.glob("*.h5")) == []
    (tmp_path / "manifest.json").write_text(json.dumps({
        "config_hash": config.config_hash, "source_hash": "different-source"}))
    with pytest.raises(ValueError, match="source/configuration mismatch"):
        dk_calibration.calibrate_native(config, tmp_path)
