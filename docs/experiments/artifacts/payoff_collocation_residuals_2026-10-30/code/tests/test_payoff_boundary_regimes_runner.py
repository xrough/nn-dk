"""Fresh independent streams and strict native-reference unlocking for v2."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from analysis import run_payoff_boundary_regimes as runner
from analysis import run_payoff_regimes as historical
from analysis.compiled_native import compiler_available
from nn_dk.native import simulate_native
from nn_dk.payoff_repair import RepairPayoffModel


def smoke_plan():
    return runner.StudyPlan(
        horizons=(1/64,), kappas=(.75,), optimizer_seeds=(0,), n_cells=16,
        train_paths=2, validation_paths=1, state_step=1/256, reference_step=1/512,
        paths_per_group=4, refinement_paths=2, chunk_paths=2, adam_steps=1,
        lbfgs_steps=0, validation_every=1, width=2, batch_size=16, quadrature_nodes=4)


def test_successor_keeps_allocations_and_uses_distinct_model_and_stream_identity():
    plan = runner.StudyPlan()
    assert plan.train_seed == 2026102901 and plan.reference_seed == 2026102902
    assert len(runner.case_names(plan)) == 6 and len(runner.slots(plan)) == 18
    assert runner.StudyPlan.from_dict(plan.to_dict()) == plan
    assert historical.StudyPlan().train_seed == 2026102801
    assert historical.WORKFLOW != runner.WORKFLOW
    assert historical.STREAM != runner.STREAM
    for case in runner.case_names(plan):
        cfg = runner.configuration(case, plan)
        previous = historical.configuration(case)
        repaired = runner.repaired_configuration(case, plan)
        assert repaired.study == cfg and repaired.quadrature_nodes == 16
        assert cfg.name == f"payoff-boundary-regimes-{case}"
        assert cfg.config_hash != previous.config_hash
        assert cfg.train_paths == 256 and cfg.validation_paths == 128
        assert runner.reference_configuration(case, plan).paths_per_group == 16384
        assert cfg.query_times == previous.query_times
    assert {item["variant"] for item in runner.slots(plan)} == {"boundary-repaired"}
    assert all("_boundary_seed" in item["name"] for item in runner.slots(plan))
    with pytest.raises(ValueError, match="Independent"):
        replace(plan, reference_seed=plan.train_seed)


def test_new_training_validation_reference_anchor_streams_are_distinct_from_old_study():
    new_ids, old_ids, new_entropy, old_entropy = set(), set(), set(), set()
    for case in runner.case_names():
        for module, ids, entropy in ((runner, new_ids, new_entropy),
                                     (historical, old_ids, old_entropy)):
            configurations = ((module.configuration(case), "train"),
                              (module.configuration(case), "validation"),
                              (module.reference_configuration(case), "reference-conditional"),
                              (module.reference_configuration(case), "reference-ensemble"),
                              (module.reference_configuration(case), "held-out-anchor"))
            for cfg, role in configurations:
                for profile in cfg.cases:
                    family = module._family(cfg, role, profile, 0, 0)
                    assert family not in ids
                    ids.add(family)
                    for purpose in ("initial", "physical-forcing", "residual-horizons"):
                        sample = module._rng(cfg, role, profile, 0, 0, purpose).bytes(32)
                        assert sample not in entropy
                        entropy.add(sample)
    assert not new_ids & old_ids
    assert not new_entropy & old_entropy


def test_design_freezes_both_adapter_and_underlying_native_machinery():
    design = runner._design(smoke_plan())
    assert design["workflow"] == runner.WORKFLOW
    assert design["fresh_training_validation_reference_and_deployment_streams"]
    assert not design["historical_analysis_modules_mutated"]
    assert design["runner_sha256"] == runner._sha(Path(runner.__file__))
    required = {"run_payoff_regimes.py", "payoff_study_adapter.py",
                "evaluate_payoff_boundary_regimes.py", "evaluate_payoff_boundary_deployment.py",
                "evaluate_payoff_regimes.py", "evaluate_payoff_regimes_deployment.py",
                "benchmark_path_generalization.py", "benchmark_payoff_generalization.py",
                "evaluate_path_generalization.py", "probe_path_workloads.py",
                "path_generalization_reference.py", "run_payoff_repair.py",
                "compiled_native.py", "native_euler_control.cpp", "native_gaussian_baseline.py"}
    assert required <= set(design["helpers"])
    assert design["boundary_model_configurations"] == design["repaired_configs"]


def test_untrained_factory_has_new_model_kind_and_preserves_torch_random_stream():
    import torch

    plan = smoke_plan()
    case = runner.case_names(plan)[0]
    torch.manual_seed(13)
    state = torch.get_rng_state().clone()
    model = runner.untrained_model(case, plan)
    assert torch.equal(torch.get_rng_state(), state)
    assert model.kind != RepairPayoffModel.kind
    assert model.horizon == 1/64 and model.spec.n_cells == 16
    assert not model.training


def test_checkpoint_hash_and_variant_are_checked_before_restoration(tmp_path, monkeypatch):
    path = tmp_path/"checkpoint.pt"
    path.write_bytes(b"deliberately-not-a-model")
    monkeypatch.setattr(runner, "load_boundary_payoff_model",
                        lambda path: pytest.fail("must verify first"))
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.load_declared_model({"checkpoint_path": path, "checkpoint_sha256": "old"})
    with pytest.raises(ValueError, match="Only declared boundary"):
        runner.load_declared_model({"checkpoint_path": path, "variant": "repaired"})


def test_all_fit_lock_blocks_backend_and_test_histogram_draws(tmp_path, monkeypatch):
    implementation = runner._implementation
    monkeypatch.setattr(implementation, "_assert_frozen", lambda root: (smoke_plan(), {}))

    def reject(root):
        raise ValueError("all successor fits required")

    monkeypatch.setattr(implementation, "verify_fit_lock", reject)
    monkeypatch.setattr(implementation, "compile_native",
                        lambda *args: pytest.fail("must not compile or draw"))
    monkeypatch.setattr(implementation, "_create_reference",
                        lambda *args: pytest.fail("must not draw held-out anchors"))
    with pytest.raises(ValueError, match="all successor fits"):
        runner.generate_reference(tmp_path)
    assert not list(tmp_path.rglob("reference.h5"))


@pytest.mark.skipif(not compiler_available(), reason="Native C++ control needs compiler")
def test_native_successor_chunk_preserves_stopping_payoffs_and_coupled_levels(tmp_path):
    plan = smoke_plan()
    case = runner.case_names(plan)[0]
    cfg = runner.reference_configuration(case, plan)
    backend = runner.compile_native(tmp_path/"compiled")
    initial, finest, factor, families = runner.reference_draw(
        cfg, "nonuniform__ensemble__0", 0, 2)
    saved = runner.simulate_reference_chunk(
        cfg, "nonuniform__ensemble__0", 0, 2, backend=backend)
    np.testing.assert_array_equal(saved["family_ids"], families)
    for level, refinement in enumerate(cfg.reference_factors):
        dw = runner._aggregate(finest, factor//refinement)
        dt = cfg.reference_step/refinement
        times = np.arange(dw.shape[1]+1)*dt
        packet = simulate_native(
            cfg.context("nonuniform"), cfg.spec, times, dt, 0, 2,
            initial_masses=initial, brownian_increments=dw,
            family_ids=tuple(families.astype(str)))
        bank = runner.payoff_bank(case, plan)
        values = bank.evaluate_paths(packet.states, times, packet.stopped_at)
        indices = np.rint(np.asarray(cfg.query_times)/dt).astype(int)
        np.testing.assert_allclose(saved["audit_outcomes"][:, level], values[:, indices],
                                   atol=3e-15, rtol=0)


@pytest.mark.skipif(not compiler_available(), reason="Native C++ control needs compiler")
def test_complete_small_successor_run_strictly_restores_before_fresh_references(tmp_path):
    root, plan = tmp_path/"successor", smoke_plan()
    runner.prepare(root, plan)
    assert not list(root.rglob("reference.h5"))
    fit = runner.fit_seed(root, 0)
    assert fit["complete"] and len(fit["models"]) == 1
    locked = runner.lock(root)
    assert locked == runner.verify_fit_lock(root)
    assert locked["models"][0]["model_kind"] == runner.untrained_model(
        runner.case_names(plan)[0], plan).kind
    references = runner.generate_reference(root)
    case = runner.case_names(plan)[0]
    assert references[case]["status"] == "complete"
    with h5py.File(root/"reference"/case/"reference.h5", "r") as store:
        assert set(store) == set(runner.reference_configuration(case, plan).group_keys) | {
            "query_times", "reference_factors"}
    assert runner.generate_reference(root) == references
    with pytest.raises(ValueError, match="fitting is closed"):
        runner.fit_seed(root, 0)
