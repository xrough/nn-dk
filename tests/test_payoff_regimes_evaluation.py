"""Hard-regime comparisons retain every seed, path and source of uncertainty."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluate_payoff_regimes", Path(__file__).parents[1]/"analysis/evaluate_payoff_regimes.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)
bootstrap_counts, empirical = study.bootstrap_counts, study.empirical


def test_all_seeds_are_retained_and_incomplete_or_duplicate_allocation_rejected():
    models = [{"case": "r", "name": f"r_seed{s}", "seed": s} for s in (0, 1, 2)]
    result = study.model_entries({"models": models}, "r")
    assert [x["seed"] for x in result[:3]] == [0, 1, 2]
    assert result[-1]["untrained"]
    with pytest.raises(ValueError, match="three"):
        study.model_entries({"models": models[:-1]}, "r")
    with pytest.raises(ValueError, match="three"):
        study.model_entries({"models": models[:-1]+[models[0]]}, "r")


def test_joint_reference_resampling_resolves_large_improvement_but_not_identical_predictions():
    rng = np.random.default_rng(71)
    y = rng.normal(size=(512, 3, 4))
    mean, covariance = empirical(y)
    counts = bootstrap_counts(len(y), 73, 128)
    contrast = study.bootstrap_advantage(y, mean, 2048*covariance,
        mean+1, 2048*(covariance+np.eye(4)[None]), counts, 2048)
    assert all(row["mean_advantage_resolved"] for row in contrast["records"])
    assert all(row["covariance_advantage_resolved"] for row in contrast["records"])
    same = study.bootstrap_advantage(y, mean, 2048*covariance, mean,
                                    2048*covariance, counts, 2048)
    assert all(not row["mean_advantage_resolved"] for row in same["records"])
    assert all(row["mean_advantage_percentile95"] == [0, 0] for row in same["records"])
    assert all(row["covariance_advantage_percentile95"] == [0, 0] for row in same["records"])


def test_advantage_keeps_initial_integration_uncertainty_instead_of_paired_bias_only():
    rng = np.random.default_rng(81)
    y = rng.normal(size=(128, 3, 4))
    mean, covariance = empirical(y)
    counts = bootstrap_counts(len(y), 83, 128)
    exact = study.bootstrap_advantage(y, mean, covariance, mean+.1, covariance, counts, 1)
    bmean = np.broadcast_to(mean, (128, *mean.shape)).copy()
    bmean[::2] += 3
    bcov = np.broadcast_to(covariance, (128, *covariance.shape)).copy()
    noisy = study.bootstrap_advantage(y, mean, covariance, mean+.1, covariance,
        counts, 1, bmean, bcov)
    assert all(a["mean_advantage_percentile95"][0] > b["mean_advantage_percentile95"][0]
               for a, b in zip(exact["records"], noisy["records"], strict=True))
    with pytest.raises(ValueError, match="Both"):
        study.bootstrap_advantage(y, mean, covariance, mean, covariance, counts, 1, bmean)


def test_standalone_sampling_uses_physical_covariance_normalization():
    mean = np.zeros((3, 4))
    covariance = np.broadcast_to(np.eye(4), (3, 4, 4))
    bm = np.ones((128, 3, 4))*.1
    bc = np.broadcast_to(covariance, (128, 3, 4, 4))
    result = study.standalone_sampling(mean, covariance, bm, bc)
    np.testing.assert_allclose(result["mean95_in_candidate_std"], .1)
    np.testing.assert_allclose(result["covariance95_normalized"], 0)


def test_near_boundary_diagnostics_include_absorbed_paths_and_expose_boundary_strip():
    result = study.near_boundary([.01, .01, .03, .2], SimpleNamespace(mass_stop=.01))
    assert result["minimum_mass"] == .01
    assert result["fractions_at_most_stop_multiple"]["1"] == .5
    assert result["fractions_at_most_stop_multiple"]["5"] == .75
    with pytest.raises(ValueError, match="every path"):
        study.near_boundary([.01, np.nan], SimpleNamespace(mass_stop=.01))


def _cfg():
    return SimpleNamespace(n_cells=4, n_particles=32, horizon=1/64,
        spec=object(), query_times=(1/256, 1/128, 1/64), context=lambda profile: profile)


def test_native_candidate_uses_fresh_forcing_keeps_stopped_paths_and_is_reproducible():
    seen = []

    def native(initial, dw, context, spec, step, bank, times, backend=None):
        seen.append((initial.copy(), dw.copy()))
        values = np.repeat(initial[:, None], 3, axis=1)
        return {"outcomes": values, "stopped_at": np.zeros(len(initial)),
                "minimum_mass": float(initial.min()), "mass_error": 0.}

    runner = SimpleNamespace(native_payoff=native)
    cfg = _cfg()
    p = np.full(4, .25)
    y, diagnostic = study._native_mc(runner, cfg, None, "uniform", None, p, 9, 91, None, chunk=4)
    assert y.shape == (9, 3, 4)
    assert diagnostic["stopped_paths"] == 9
    assert diagnostic["paths"] == 9
    assert [len(x[0]) for x in seen] == [4, 4, 1]
    first = [(a.copy(), b.copy()) for a, b in seen]
    seen.clear()
    y2, _ = study._native_mc(runner, cfg, None, "uniform", None, p, 9, 91, None, chunk=4)
    np.testing.assert_array_equal(y, y2)
    for a, b in zip(first, seen, strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    seen.clear()
    study._native_mc(runner, cfg, None, "uniform", None, p, 9, 92, None, chunk=4)
    assert not np.array_equal(first[0][1], seen[0][1])


@pytest.mark.parametrize("defect", ("mass", "nonfinite", "stop", "negative"))
def test_any_native_failure_blocks_ensemble_without_survivor_conditioning(defect):
    def native(initial, dw, context, spec, step, bank, times, backend=None):
        values = np.zeros((len(initial), 3, 4))
        stops = np.full(len(initial), np.inf)
        if defect == "nonfinite":
            values[0, 0, 0] = np.nan
        if defect == "stop":
            stops[0] = np.nan
        return {"outcomes": values, "stopped_at": stops,
                "minimum_mass": -1e-3 if defect == "negative" else .1,
                "mass_error": 1e-3 if defect == "mass" else 0.}

    with pytest.raises(ValueError, match="no survivor"):
        study._native_mc(SimpleNamespace(native_payoff=native), _cfg(), None, "uniform",
                         None, np.full(4, .25), 8, 91, None)


def test_conditional_mc_never_resamples_the_supplied_histogram():
    supplied = np.array([.1, .2, .3, .4])
    origins = []

    def native(initial, dw, context, spec, step, bank, times, backend=None):
        origins.append(initial.copy())
        return {"outcomes": np.repeat(initial[:, None], 3, axis=1),
                "stopped_at": np.full(len(initial), np.inf), "minimum_mass": .1, "mass_error": 0.}

    study._native_mc(SimpleNamespace(native_payoff=native), _cfg(), None, "uniform",
                     supplied, np.full(4, .25), 9, 91, None, chunk=4)
    np.testing.assert_array_equal(np.concatenate(origins), np.broadcast_to(supplied, (9, 4)))


def test_completed_record_detects_modified_prediction_bytes_and_identity(tmp_path):
    path = tmp_path/"candidate.json"
    identity = {"reference": "fixed", "seed": 2}
    study._save_result(path, identity, {}, {"mean": np.zeros((3, 4))}, {})
    assert study._replay(path, identity)["complete"]
    with pytest.raises(ValueError, match="changed"):
        study._replay(path, {**identity, "seed": 1})
    path.with_suffix(".npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        study._replay(path, identity)


def test_benchmark_requires_explicit_quiet_worker_gate_before_loading_anything():
    with pytest.raises(ValueError, match="workers"):
        study.benchmark("not-a-run")
