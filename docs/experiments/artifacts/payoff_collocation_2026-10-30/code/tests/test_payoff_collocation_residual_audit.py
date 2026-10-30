"""Residual decomposition, units, population weights and frozen chunk evidence."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_collocation_residuals as audit  # noqa: E402
from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis.compiled_native import compiler_available  # noqa: E402
from analysis.payoff_collocation_sampling import boundary_time_strata  # noqa: E402


def tiny_plan():
    return runner.StudyPlan(horizons=(1/64,), kappas=(.75,), n_cells=8, n_particles=128,
        train_paths=2, validation_paths=2, horizon_samples=1, width=4,
        adam_steps=1, lbfgs_steps=0, lbfgs_max_iter=1, validation_every=1,
        batch_size=4, quadrature_nodes=2, state_step=1/1024, reference_step=1/1024,
        conditional_anchors=1, paths_per_group=4, refinement_paths=2, chunk_paths=2)


@pytest.fixture(scope="module")
def complete_study(tmp_path_factory):
    if not compiler_available():
        pytest.skip("Optional native compiler unavailable")
    torch.set_num_threads(1)
    base = tmp_path_factory.mktemp("residual-software")
    root, plan = base/"study", tiny_plan()
    runner.prepare(root, plan)
    for seed in (0, 1, 2):
        runner.fit_seed(root, seed)
    runner.lock(root)
    return root, plan


def test_physical_units_distinguish_sqrtN_and_N_and_stored_loss_scales():
    scales = np.array([1., 2., 3., 5.])
    model = SimpleNamespace(bank=SimpleNamespace(scales=scales), horizon=2.,
        context=SimpleNamespace(n_particles=100), loss_scaling={
            "mean_scale": np.array([2., 4., 6., 10.]), "covariance_scale": np.outer(scales, scales)*3})
    native = {key: np.ones((2, 4))*(i+1) if key in audit.MEAN_KEYS else np.ones((2, 4, 4))*(i+1)
              for i, key in enumerate(audit.COMPONENT_KEYS)}
    result = audit.convert_components(native, model)
    for key in audit.COMPONENT_KEYS:
        is_mean = key in audit.MEAN_KEYS
        factor = scales/10 if is_mean else np.outer(scales, scales)/100
        loss_scale = model.loss_scaling[("mean" if is_mean else "covariance")+"_scale"]
        np.testing.assert_allclose(result["physical_"+key], native[key]*factor, atol=0, rtol=0)
        np.testing.assert_allclose(result["dimensionless_"+key], 2*native[key]/loss_scale, atol=0, rtol=0)
    assert result["physical_Gamma"][0, 0, 0] == native["Gamma"][0, 0, 0]/100
    assert not np.allclose(result["physical_Gamma"], native["Gamma"]*np.outer(scales, scales))


def test_covariance_mse_uses_all_sixteen_entries_not_unweighted_unique_ten():
    values = np.zeros((2, 4, 4))
    values[:, 0, 1], values[:, 1, 0], values[:, 2, 2] = 2., 6., 1.
    actual = audit.natural_metrics(values, 3.)
    assert actual["natural_mse"] == (4+36+1)/16
    unique = values[:, *np.triu_indices(4)]
    assert actual["natural_mse"] != np.mean(unique*unique)
    expected_huber = sum(18*(math.sqrt(1+(value/3)**2)-1) for value in (2., 6., 1.))/16
    assert np.isclose(actual["natural_huber"], expected_huber, rtol=1e-14)


def test_all_nineteen_strata_are_defined_from_state_time_only():
    study = tiny_plan()
    config = runner.ablation_configuration(runner.slots(study)[0], study).strata
    states, times = [], []
    for time_fraction in (.1, .3, .6, .9):
        for distance in (.1, .5, 2., 8.):
            minimum = config.mass_stop+distance*config.boundary_width*np.sqrt(time_fraction)
            state = np.full(8, (1-minimum)/7)
            state[0] = minimum
            states.append(state)
            times.append(time_fraction*config.horizon)
    interior = np.full(8, 1/8)
    absorbed = interior.copy()
    absorbed[0] = config.mass_stop
    absorbed[1] += 1/8-config.mass_stop
    states += [absorbed, absorbed, interior]
    times += [0., config.horizon/2, 0.]
    labels = boundary_time_strata(np.asarray(states), np.asarray(times), config)
    assert len(audit.strata_labels(config)) == 19
    assert set(labels) == set(audit.strata_labels(config))
    changed = type(config)(config.horizon, config.mass_stop, config.boundary_width,
                           distance_edges=(.5, 1., 4.))
    with pytest.raises(ValueError, match="nineteen-stratum"):
        audit.strata_labels(changed)


def test_true_jets_decompose_at_zero_absorption_and_positive_times_without_mutation():
    torch.set_num_threads(1)
    study = tiny_plan()
    config = runner.ablation_configuration(runner.slots(study)[0], study)
    bank = runner.payoff_bank(runner.case_names(study)[0], study)
    model = runner.engine._model(config, bank, 0)
    common = runner.engine._model(config, bank, 1)
    model.loss_scaling = {"mean_scale": [2., 3., 4., 5.], "covariance_scale": np.full((4, 4), 7.).tolist()}
    with torch.no_grad():
        common.mean_network[-1].weight.fill_(.05)
    states = np.full((4, 8), 1/8)
    states[2, 0] = config.strata.mass_stop
    states[2, 1] += 1/8-config.strata.mass_stop
    horizons = np.array([0., study.horizons[0]/4, study.horizons[0]/2, study.horizons[0]])
    before = runner.engine._digest(model.state_dict())
    arrays, errors = audit.extract_components(model, states, horizons, common)
    assert max(errors.values()) < 1e-12
    assert before == runner.engine._digest(model.state_dict())
    np.testing.assert_allclose(arrays["rmu"], arrays["mu_dt"]-arrays["Lmu"]-arrays["running_source"], atol=0, rtol=0)
    np.testing.assert_allclose(arrays["rC"], arrays["C_dt"]-arrays["LC"]-arrays["Gamma"], atol=0, rtol=0)
    np.testing.assert_allclose(arrays["rC_common"], arrays["rC"]+arrays["Gamma"]-arrays["Gamma_common"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(arrays["rmu"][[0, 2]], 0., atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["rC"][[0, 2]], 0., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(arrays["running_source"][2], 0.)
    assert np.max(abs(arrays["Gamma"]-arrays["Gamma_common"])) > 1e-8


def test_twelve_actual_fits_audit_resumes_and_reconstructs_global_population(complete_study, tmp_path, monkeypatch):
    study_root, _ = complete_study
    root = tmp_path/"audit"
    packet = audit.prepare(study_root, root, audit.ResidualAuditPlan(chunk_rows=8))
    assert len(packet["identity"]["models"]) == 12
    assert audit.prepare(study_root, root, audit.ResidualAuditPlan(chunk_rows=8)) == packet
    partial = audit.run(root, max_chunks=1)
    assert not partial["complete"]
    with pytest.raises(ValueError, match="All declared model"):
        audit.load_complete(root)
    original = audit.extract_components
    calls = []

    def resumed(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "extract_components", resumed)
    result = audit.run(root)
    assert result["complete"] and len(result["records"]) == 12
    assert len(calls) == 23  #16rows/model,chunk8; first completed chunk was reused.
    for record in result["records"]:
        arrays = audit.load_record_arrays(record)
        summary = record["summary"]
        assert summary["rows"] == 16 and summary["parent_families"] == 4
        assert len(summary["strata_counts"]) == 19
        assert sum(item["rows"] for item in summary["strata_counts"]) == 16
        assert any(item["rows"] == 0 and item["parent_families"] == 0 for item in summary["strata_counts"])
        for stage, key in (("mean", "rmu"), ("covariance", "rC")):
            values = arrays["dimensionless_"+key]
            contributions = sum(float(np.square(values[arrays["strata"] == item["label"]]).sum())
                /(len(values)*np.prod(values.shape[1:])) for item in summary["strata_counts"])
            assert np.isclose(contributions, summary["canonical_validation"][stage]["natural_mse"], rtol=1e-12)
    assert audit.load_complete(root)["complete"]
    preparations = audit.load_complete(root)["preparation_costs"]["attempts"]
    assert len(preparations) == 2
    assert all(item["finished"] and item["complete"] and item["wall_seconds"] > 0 for item in preparations)
    assert not list(root.rglob("*.h5")) and not list(root.rglob("*.pt"))


def test_orphan_tamper_and_missing_summary_are_never_blessed(complete_study, tmp_path):
    study_root, _ = complete_study
    root = tmp_path/"audit"
    audit.prepare(study_root, root, audit.ResidualAuditPlan(chunk_rows=16))
    result = audit.run(root)
    record = result["records"][0]
    path = Path(record["array_chunks"][0]["path"])
    original = path.read_bytes()
    path.write_bytes(original+b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        audit.load_complete(root)
    path.write_bytes(original)
    summary = Path(record["summary_path"])
    original_summary = summary.read_bytes()
    summary.unlink()
    with pytest.raises(ValueError, match="immutable model summary"):
        audit.load_complete(root)
    assert not summary.exists()
    summary.write_bytes(original_summary)
    extra = root/"chunks/undeclared.npz"
    extra.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="undeclared residual"):
        audit.load_complete(root)
    extra.unlink()
    extra = root/"models/undeclared.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="undeclared residual model summary"):
        audit.load_complete(root)
    extra.unlink()
    (root/"inputs/undeclared.npz").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="undeclared residual input"):
        audit.load_complete(root)


def test_all_fit_lock_and_frozen_plan_are_required(complete_study, tmp_path, monkeypatch):
    study_root, _ = complete_study
    root = tmp_path/"audit"
    audit.prepare(study_root, root)
    saved = (root/"plan.json").read_bytes()
    packet = audit.read(root/"plan.json")
    packet["identity"]["units"]["physical_covariance"] = "incorrect N-scaled claim"
    audit.write(root/"plan.json", packet)
    with pytest.raises(ValueError, match="Frozen residual source"):
        audit.run(root)
    failed = audit._execution(root)
    assert failed["attempts"][-1]["finished"] and failed["attempts"][-1]["wall_seconds"] > 0
    assert failed["failures"]
    (root/"plan.json").write_bytes(saved)
    verify = runner.verify_fit_lock

    def incomplete(path):
        value = copy.deepcopy(verify(path))
        value["models"].pop()
        return value

    monkeypatch.setattr(runner, "verify_fit_lock", incomplete)
    with pytest.raises(ValueError, match="Every declared crossed fit"):
        audit.prepare(study_root, tmp_path/"not-allowed")
    attempts = audit._prepare_costs(tmp_path/"not-allowed", study_root)["attempts"]
    assert len(attempts) == 1 and attempts[0]["finished"] and not attempts[0]["complete"]
    assert attempts[0]["wall_seconds"] > 0 and "error" in attempts[0]


def test_failed_chunk_has_a_paid_failure_inventory(complete_study, tmp_path, monkeypatch):
    study_root, _ = complete_study
    root = tmp_path/"audit"
    audit.prepare(study_root, root)

    def failure(*args, **kwargs):
        raise FloatingPointError("controlled software extraction failure")

    monkeypatch.setattr(audit, "extract_components", failure)
    with pytest.raises(FloatingPointError, match="controlled software"):
        audit.run(root)
    costs = audit._execution(root)
    assert costs["failures"] and costs["attempts"][-1]["finished"]
    assert costs["attempts"][-1]["wall_seconds"] > 0
    with pytest.raises(ValueError, match="failure inventory"):
        audit.load_complete(root)


def test_unfinished_prepare_cost_is_not_invented_as_zero_or_ignored(complete_study, tmp_path, monkeypatch):
    study_root, _ = complete_study
    root = tmp_path/"audit"
    audit.prepare(study_root, root)
    costs = audit._prepare_costs(root, study_root)
    costs["attempts"].append({"finished": False, "complete": False, "pid": 123})
    audit.write(root/"execution/prepare_costs.json", {**costs, "content_sha256": audit.stable_hash(costs)})

    def forbidden(*args, **kwargs):
        pytest.fail("Residuals were evaluated despite unknown preparation costs")

    monkeypatch.setattr(audit, "extract_components", forbidden)
    with pytest.raises(ValueError, match="measured paid cost"):
        audit.run(root)
    assert not (root/"chunks").exists()
