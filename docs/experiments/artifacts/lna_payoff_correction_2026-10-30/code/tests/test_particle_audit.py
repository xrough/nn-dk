"""Cross-law particle comparisons retain physical observables and every stop."""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_dk.convergence_config import convergence_preset
from nn_dk.dk_model import DKCoefficientModel
from nn_dk.particle_audit import audit_particles, particle_physical_observables


def tiny_config(**kwargs):
    defaults = dict(horizon=1 / 32, query_times=(1 / 128, 1 / 64, 1 / 32),
                    reference_dt=1 / 512, proposal_dt=1 / 128,
                    step_sizes=(1 / 128, 1 / 256), n_particles=128,
                    particle_numbers=(64, 128), diagnostic_paths=4,
                    evaluation_batch_size=2, states_per_path=4, bootstrap_samples=8)
    defaults.update(kwargs)
    return replace(convergence_preset("a-smoke"), **defaults)


def test_direct_particle_observables_are_the_declared_continuous_functions():
    positions = np.array([[0., np.pi / 2], [np.pi, 3 * np.pi / 2]])
    expected = np.sqrt(2) * np.stack([f(k * positions).mean(-1)
                                     for k in (1, 2) for f in (np.cos, np.sin)], axis=-1)
    np.testing.assert_allclose(particle_physical_observables(positions), expected, atol=1e-15)
    assert particle_physical_observables(positions[:, None]).shape == (2, 1, 4)
    with pytest.raises(ValueError, match="Finite"):
        particle_physical_observables(np.empty((2, 0)))


def test_particle_audit_is_resumable_and_separates_neural_time_and_particle_errors(tmp_path):
    torch.set_num_threads(1)
    config = tiny_config()
    model = DKCoefficientModel(config.spec, width=4)
    context = config.context("uniform")
    interrupted = audit_particles(config, context, model, tmp_path, deadline=time.monotonic() - 1)
    assert not interrupted["complete"]
    assert all(value["completed_paths"] == 0 for value in interrupted["counts"])
    report = audit_particles(config, context, model, tmp_path)
    assert report["complete"]
    assert len(report["counts"]) == 2
    paths = sorted(tmp_path.glob("n*/chunks/*.npz"))
    assert len(paths) == 4
    before = [path.stat().st_mtime_ns for path in paths]
    repeated = audit_particles(config, context, model, tmp_path)
    assert repeated == report
    assert before == [path.stat().st_mtime_ns for path in paths]
    for entry in report["counts"]:
        assert entry["all_paths_included"] and entry["n_independent_families"] == 4
        assert entry["particle_target"]["law"] != entry["native_target"]["law"]
        assert set(entry["comparisons"]) == {
            "nn_minus_native_at_same_step", "native_time_discretization_difference",
            "native_minus_particles", "nn_minus_particles"}
        assert entry["comparisons"]["native_minus_particles"]["uncertainty"]["available"]
        assert np.shape(entry["estimates"]["particles"]["scaled_covariance"]) == (12, 12)
        assert entry["shared_initialization"]["maximum_absolute_initial_projection_error"] > 0
        assert entry["costs"]["particles_seconds"] >= 0
        assert "separate refinement audits" in entry["interpretation"]["bias_certification"]
    with torch.no_grad():
        model.output.bias[0] += 0.1
    with pytest.raises(ValueError, match="different data/model"):
        audit_particles(config, context, model, tmp_path)


def test_all_initially_stopped_histogram_paths_remain_in_particle_comparison(tmp_path):
    torch.set_num_threads(1)
    config = tiny_config(n_particles=8, particle_numbers=(8,), diagnostic_paths=8)
    model = DKCoefficientModel(config.spec, width=4)
    report = audit_particles(config, config.context("uniform"), model, tmp_path)
    entry = report["counts"][0]
    assert report["complete"] and entry["n_independent_families"] == 8
    for method in ("native_reference", "native_nn_step", "nn"):
        assert entry["estimates"][method]["n_independent_paths"] == 8
        assert entry["stopping"][method]["initial_stopping_probability"] > 0
        assert entry["stopping"][method]["all_stopped_paths_included"]
    np.testing.assert_allclose(entry["estimates"]["native_reference"]["mean"],
                               entry["estimates"]["native_nn_step"]["mean"], atol=1e-15)


def test_particle_audit_rejects_source_and_config_changes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = tiny_config(particle_numbers=(64,))
    model = DKCoefficientModel(config.spec, width=4)
    context = config.context("uniform")
    audit_particles(config, context, model, tmp_path, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="different data/model"):
        audit_particles(replace(config, bootstrap_samples=10), context, model, tmp_path)
    monkeypatch.setattr("nn_dk.particle_audit.source_hash", lambda: "different-source")
    with pytest.raises(ValueError, match="different data/model"):
        audit_particles(config, context, model, tmp_path)
