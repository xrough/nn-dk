"""Artifact-only summaries must preserve distinctions and reject partial runs."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "analysis" / "summarize_first_batch.py"
_SPEC = importlib.util.spec_from_file_location("findings_analysis", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def test_partial_run_rejected_before_reading_aggregate(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "budget-exhausted"}))
    # Neither config nor records exists: the manifest guard must run first.
    with pytest.raises(ValueError, match="status='complete'"):
        analysis.summarize(tmp_path)
    assert not (tmp_path / "report").exists()


def test_allocations_come_from_manifest_axes():
    config = {
        "cases": ["uniform", "interacting"], "step_sizes": [0.1, 0.05, 0.025],
        "train_path_counts": [8, 16], "widths": [16, 32], "base_width": 32,
        "training_modes": ["oracle", "increments"], "optimizer_seeds": [3, 7],
    }
    keys = analysis._expected_keys(config)
    # Per case: 3 oracle fits and 1 increment fit, each with two seeds.
    assert len(keys) == 2 * (3 + 4 * 2 * 3 * 3)
    assert ("uniform", "increments_n16_w32_s7", "both", 0.025) in keys
    assert ("uniform", "increments_n8_w32_s7", "both", 0.025) not in keys


def test_variance_and_cross_time_covariance_are_separate():
    comparison = {
        "errors": {
            "mean_normalized_rmse": 0.1, "scaled_covariance_normalized_rmse": 0.2,
            "mean_normalized": [0.1, 0.3], "scaled_covariance_normalized": [[0.2, 0.4], [0.4, 0.6]],
            "per_query": [
                {"index": 0, "time": 0.25, "scaled_variance_normalized_error": 0.2},
                {"index": 1, "time": 0.5, "scaled_variance_normalized_error": 0.6},
            ],
            "covariance_pairs": [
                {"i": 0, "j": 0, "two_time": False, "normalized_error": 0.2},
                {"i": 0, "j": 1, "two_time": True, "normalized_error": 0.4},
                {"i": 1, "j": 1, "two_time": False, "normalized_error": 0.6},
            ],
        },
        "uncertainty": {"mean_normalized_ci95": [[-0.2, 0.4], [-0.1, 0.2]]},
        "accuracy_gate": {"status": "unresolved"},
    }
    summary = analysis.component_summary(comparison)
    assert summary["variance_error_rmse_by_time"] == {"0.25": 0.2, "0.5": 0.6}
    assert summary["two_time_covariance_error_rmse"] == 0.4
    assert summary["two_time_covariance_error_rmse_by_times"] == {"0.25|0.5": 0.4}
    assert summary["mean_ci95_max_abs_endpoint"] == 0.4
    assert summary["accuracy_gate"]["status"] == "unresolved"


def test_no_stopping_events_do_not_create_zero_upper_bound():
    summary = analysis._outcome({"diagnostics": {"n_paths": 8192, "n_stopped": 0}})
    assert summary["zero_event_one_sided95_probability_upper"] == pytest.approx(0.0003656232)
