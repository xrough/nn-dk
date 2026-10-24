"""The reference precision diagnostic remains descriptive and artifact-only."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nn_dk.bsde_study_config import bsde_study_preset

_PATH = Path(__file__).parents[1]/"analysis"/"diagnose_bsde_reference_noise.py"
_SPEC = importlib.util.spec_from_file_location("bsde_reference_noise_analysis", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def test_reference_mean_and_variance_sampling_formulas_on_known_samples():
    base = bsde_study_preset("bsde-study-smoke").base_config
    normalized = np.broadcast_to(np.asarray((-2., -1., 1., 2.))[None, :, None, None], (2, 4, 3, 4))
    physical = normalized*np.asarray(base.observable_scale)/np.sqrt(base.n_particles)
    result = analysis.estimate_reference_noise(physical, base)
    variance = 10/3
    fourth = 8.5
    mean_floor = np.sqrt(variance/4)
    variance_floor = np.sqrt((fourth-(4-3)/(4-1)*variance**2)/4)
    assert result["n_fixed_anchors"] == 2
    assert result["independent_children_per_anchor"] == 4
    assert result["mean_rmse_noise_estimate"] == pytest.approx(mean_floor)
    assert result["variance_rmse_noise_estimate"] == pytest.approx(variance_floor)
    assert [row["time"] for row in result["by_time"]] == list(base.query_times)
    assert all(row["mean_rmse_noise_estimate"] == pytest.approx(mean_floor) for row in result["by_time"])


def test_time_dependent_rescaling_keeps_mean_and_variance_units_distinct():
    base = bsde_study_preset("bsde-study-smoke").base_config
    normalized = (np.asarray((-2., -1., 1., 2.))[None, :, None, None]
                  * np.asarray((1., 2., 3.))[None, None, :, None])
    normalized = np.broadcast_to(normalized, (1, 4, 3, 4))
    physical = normalized*np.asarray(base.observable_scale)/np.sqrt(base.n_particles)
    result = analysis.estimate_reference_noise(physical, base)
    times = result["by_time"]
    assert times[1]["mean_rmse_noise_estimate"] == pytest.approx(2*times[0]["mean_rmse_noise_estimate"])
    assert times[1]["variance_rmse_noise_estimate"] == pytest.approx(4*times[0]["variance_rmse_noise_estimate"])
    assert times[2]["mean_rmse_noise_estimate"] == pytest.approx(3*times[0]["mean_rmse_noise_estimate"])
    assert times[2]["variance_rmse_noise_estimate"] == pytest.approx(9*times[0]["variance_rmse_noise_estimate"])


def test_partial_run_rejected_before_reading_held_out_data(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"status": "fitting"}))
    with pytest.raises(ValueError, match="status='complete'.*unopened"):
        analysis.diagnose(tmp_path)
    assert not (tmp_path/"report").exists()


def test_precision_diagnostic_requires_an_unchanged_audited_snapshot(tmp_path):
    path = tmp_path/"immutable.json"
    path.write_text('{"value":1}')
    summary = {"provenance": {"input_sha256": {path.name: analysis._sha(path)}}}
    analysis.validate_snapshot_hashes(tmp_path, summary)
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="audited study artifact changed"):
        analysis.validate_snapshot_hashes(tmp_path, summary)


def test_invalid_or_survivor_only_reference_array_cannot_be_silently_used():
    base = bsde_study_preset("bsde-study-smoke").base_config
    with pytest.raises(ValueError, match="Finite anchor/child/time/four-mode"):
        analysis.estimate_reference_noise(np.zeros((2, 1, 3, 4)), base)
    values = np.zeros((2, 4, 3, 4))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite anchor/child/time/four-mode"):
        analysis.estimate_reference_noise(values, base)
