"""General path benchmark numerical checks; these are not final timing runs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_dk.native_types import NativeDKSpec
from nn_dk.path_model import CausalPathOperator
from nn_dk.types import Context

_spec = importlib.util.spec_from_file_location(
    "benchmark_path_generalization", Path(__file__).parents[1] / "analysis/benchmark_path_generalization.py")
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _model():
    torch.manual_seed(52)
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "dk", width=4, depth=3)
    with torch.no_grad():
        model.kernel_network[-1].weight.normal_(0, .1)
        model.kernel_network[-1].bias.normal_(0, .1)
    return model, Context(kappa=.75, a2=.1, n_particles=2048)


def test_cell_probabilities_and_timing_rng_are_reproducible_and_independent():
    spec = NativeDKSpec()
    uniform = Context(n_particles=2048)
    nonuniform = Context(cos_coeffs=(.2, 0, 0), n_particles=2048)
    np.testing.assert_allclose(benchmark.cell_probabilities(uniform, spec), 1 / 32, atol=1e-16)
    probabilities = benchmark.cell_probabilities(nonuniform, spec)
    assert probabilities.sum() == pytest.approx(1.)
    assert probabilities.min() > 0
    assert probabilities.max() - probabilities.min() > .01
    first = benchmark.fresh_inputs(nonuniform, spec, 16, 4, 1 / 256, seed=2026102612, case="nonuniform")
    second = benchmark.fresh_inputs(nonuniform, spec, 16, 4, 1 / 256, seed=2026102612, case="nonuniform")
    changed = benchmark.fresh_inputs(nonuniform, spec, 16, 4, 1 / 256, seed=2026102613, case="nonuniform")
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[1], changed[1])
    assert len(set(first[2]["trajectory_families"])) == 4
    assert set(first[2]["trajectory_families"]).isdisjoint(changed[2]["trajectory_families"])
    assert first[2]["initial_hash"] == benchmark.array_hash(first[0])
    assert first[2]["brownian_hash"] == benchmark.array_hash(first[1])
    assert benchmark.SHORT_STEPS[1 / 256] == "e30d58b3a0"
    assert benchmark.SHORT_STEPS[1 / 512] == "645459a4bd"
    assert benchmark.SHORT_STEPS[1 / 1024] == "699c7c5bf9"


def test_projected_gradient_and_error_metrics_preserve_tangent_derivative():
    values = np.arange(24.).reshape(3, 8)
    projected = benchmark.projected_initial_gradient(values)
    np.testing.assert_allclose(projected.sum(-1), 0., atol=1e-15)
    direction = np.array([1, -1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(values @ direction, projected @ direction)
    diagnostics = benchmark.errors(np.array([2., 4.]), np.array([1., 2.]))
    assert diagnostics["relative_rms_error"] == pytest.approx(1.)
    assert benchmark.errors(np.zeros(3), np.zeros(3))["relative_rms_error"] is None
    with pytest.raises(ValueError):
        benchmark.errors(np.ones(2), np.ones(3))


def test_numerical_smoke_compares_native_and_neural_paths_and_derivatives():
    model, context = _model()
    result, arrays = benchmark.benchmark_case(model, context, 1 / 256, 1 / 64,
        batch=1, repeats=1, warmup=0, derivatives=True, finite_difference=True)
    records = result["records"]
    assert len(records) == 22
    assert {record["task"] for record in records} == {"full_density_path", "many_new_observables"}
    assert all(record["median_seconds"] > 0 for record in records)
    assert all(len(record["samples_seconds"]) == 1 for record in records)
    assert all(record["diagnostics"]["mass_error"] < 1e-14 for record in records)
    assert len(result["preparation"]) == 8
    assert all(record["median_seconds"] > 0 for record in result["preparation"])
    assert arrays["initial"].shape == (1, 8)
    assert arrays["brownian"].shape == (1, 4, 8)
    np.testing.assert_allclose(arrays["learned_dense_cached_depth3_states"],
                               arrays["legacy_learned_torch_states"], atol=2e-16, rtol=2e-14)
    np.testing.assert_allclose(arrays["learned_causal_block_fft_depth3_states"],
                               arrays["legacy_learned_torch_states"], atol=2e-16, rtol=2e-14)
    native = [record for record in result["derivative_records"] if record["method"] == "torch_native_euler"]
    assert len(native) == 2
    for record in native:
        for key in ("initial_error", "physics_error", "tangent_error"):
            if key in record:
                assert record[key]["maximum_absolute_error"] < 1e-11
    assert len(result["finite_difference"]) == 4
    for record in result["finite_difference"]:
        assert record["initial_absolute_error"] < 1e-8
        assert record["physics_error"]["maximum_absolute_error"] < 1e-8
    np.testing.assert_allclose(arrays["initial_jvp_direction"].sum(), 0., atol=1e-16)


def test_incremental_benchmark_writes_audited_raw_arrays(tmp_path, monkeypatch):
    def fake_load(path, commit):
        model, context = _model()
        return model, context, 1 / 256, 1 / 64, {"historical_commit": commit,
            "checkpoint_path": str(path), "checkpoint_sha256": "unit-test"}, .001

    monkeypatch.setattr(benchmark, "_load_current_model", fake_load)
    monkeypatch.setattr(benchmark, "source_hash", lambda: "unit-test-source-freeze")
    output = tmp_path / "timing.json"
    result = benchmark.run_benchmark(output, smoke=True, repeats=1, warmup=0)
    assert result["status"] == "complete"
    assert result["seed"] == 2026102612
    assert not result["final_timing_after_external_fit_completion"]
    assert result["timing_pool_independent_of_accuracy"]
    saved = json.loads(output.read_text())
    assert len(saved["cases"]) == 1
    artifact = Path(saved["cases"][0]["raw_arrays"])
    assert artifact.is_file()
    with np.load(artifact) as arrays:
        assert "native_projected_initial_gradient" in arrays
        assert "learned_dense_cached_depth3_shared_physics_gradient" in arrays
    assert "universal superiority" in result["claim"]
    assert "physics" in result["derivative_scope"]
    assert "fixed D" in result["learned_physics_derivative_limitation"]
    if result["compiled_classical_control"]["available"]:
        assert any(record["method"] == "compiled_native_euler"
                   for record in saved["cases"][0]["records"])
        assert any(record["method"] == "compiled_native_analytic_jvp"
                   for record in saved["cases"][0]["derivative_records"])
        reverse = [record for record in saved["cases"][0]["derivative_records"]
                   if record["method"] == "compiled_native_analytic_adjoint"]
        assert len(reverse) == 1
        assert reverse[0]["initial_error"]["maximum_absolute_error"] < 1e-11
        assert reverse[0]["physics_error"]["maximum_absolute_error"] < 1e-11


@pytest.mark.parametrize("repeats,warmup", [(0, 0), (1, -1), (True, 0)])
def test_timing_rejects_invalid_allocations(repeats, warmup):
    with pytest.raises(ValueError):
        benchmark.timing(lambda: None, repeats=repeats, warmup=warmup)
