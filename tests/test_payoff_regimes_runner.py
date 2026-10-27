"""Physical regime declaration, independent streams and native payoff storage."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_regimes_runner", Path(__file__).parents[1]/"analysis"/"run_payoff_regimes.py")
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)
from analysis.compiled_native import compiler_available  # noqa: E402
from nn_dk.interacting_training import _bank  # noqa: E402
from nn_dk.native import simulate_native  # noqa: E402
from nn_dk.types import seeded_rng  # noqa: E402


def smoke_plan():
    return runner.StudyPlan(horizons=(1/64,), kappas=(.75,), optimizer_seeds=(0,),
                            n_cells=16, train_paths=2, validation_paths=1,
                            state_step=1/256, reference_step=1/512,
                            paths_per_group=4, refinement_paths=2, chunk_paths=2,
                            adam_steps=1, lbfgs_steps=0, validation_every=1,
                            width=2, batch_size=16, quadrature_nodes=4)


def test_default_design_declares_all_six_regimes_and_eighteen_seeds():
    plan = runner.StudyPlan()
    names, slots = runner.case_names(plan), runner.slots(plan)
    assert len(names) == 6 and len(slots) == 18
    assert len({row["name"] for row in slots}) == 18
    assert {runner.configuration(name, plan).horizon for name in names} == {.25, 1.}
    assert {runner.configuration(name, plan).kappa for name in names} == {.75, 3., 6.}
    for name in names:
        cfg = runner.configuration(name, plan)
        assert {row["seed"] for row in slots if row["case"] == name} == {0, 1, 2}
        assert cfg.train_paths == 256 and cfg.validation_paths == 128
        assert len(cfg.state_times)*cfg.horizon_samples == 16
        assert runner.repaired_configuration(name, plan).quadrature_nodes == 16
        ref = runner.reference_configuration(name, plan)
        assert ref.seed != cfg.seed and len(ref.group_keys) == 6
        assert ref.paths_per_group == 16384 and ref.refinement_paths == 1024
        assert ref.reference_step == 2**-12 and ref.reference_factors == (1, 2, 4)
    assert runner.StudyPlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(ValueError, match="Unknown"):
        runner.configuration("selected-after-test", plan)
    with pytest.raises(ValueError, match="Complete"):
        runner.StudyPlan.from_dict({})
    with pytest.raises(ValueError, match="Independent"):
        replace(plan, reference_seed=plan.train_seed)


def test_every_role_regime_and_purpose_has_independent_addressable_streams():
    plan = runner.StudyPlan()
    families, samples = set(), set()
    for case in runner.case_names(plan):
        cfg = runner.configuration(case, plan)
        ref = runner.reference_configuration(case, plan)
        for config, role in ((cfg, "train"), (cfg, "validation"), (ref, "reference-conditional"),
                             (ref, "reference-ensemble"), (ref, "held-out-anchor")):
            for profile in cfg.cases:
                family = runner._family(config, role, profile, 0, 0)
                assert family not in families
                families.add(family)
                for purpose in ("initial", "physical-forcing", "residual-horizons"):
                    value = runner._rng(config, role, profile, 0, 0, purpose).bytes(32)
                    assert value not in samples
                    samples.add(value)
                    assert value == runner._rng(config, role, profile, 0, 0, purpose).bytes(32)


def test_brownian_nested_sums_preserve_exact_family_coupling():
    fine = seeded_rng(17, "brownian").normal(size=(3, 32, 16))
    np.testing.assert_allclose(runner._aggregate(runner._aggregate(fine, 2), 2),
                               runner._aggregate(fine, 4), atol=2e-15, rtol=0)
    with pytest.raises(ValueError, match="nested"):
        runner._aggregate(fine, 3)


@pytest.mark.skipif(not compiler_available(), reason="Optional serial C++ native control needs compiler")
def test_compiled_state_collocation_is_valid_grouped_resumable_and_corruption_rejected(tmp_path, monkeypatch):
    plan = smoke_plan()
    case = runner.case_names(plan)[0]
    cfg = runner.configuration(case, plan)
    backend = runner.compile_native(tmp_path/"compiled")
    root = tmp_path/"run"
    runner._json(root/"plan.json", plan.to_dict())
    runner._json(root/"payoffs"/"preregistration.json", {"test": "state-only"})
    monkeypatch.setattr(runner, "_assert_frozen", lambda root: (plan, {}))
    directory = root/"payoffs"/case
    for role in ("train", "validation"):
        packet = runner.ensure_state_bank(directory, cfg, role, plan, backend)
        assert packet["complete"] and not packet["future_payoff_labels"]
        assert runner.ensure_state_bank(directory, cfg, role, plan, backend) == packet
        bank = runner.load_state_bank(root, case, role)
        tensors, fingerprint = _bank(bank, cfg, role)
        count = cfg.train_paths if role == "train" else cfg.validation_paths
        assert tensors["states"].shape == (2*count*16, cfg.n_cells)
        assert fingerprint["independent_families"] == count*2
        assert fingerprint["samples_per_family"] == 16
    lock = runner._data_lock(root, plan)
    assert lock["all_family_splits_disjoint"]
    assert runner.calibration(root)["held_out_inputs_read"] is False
    path = directory/"states_train.h5"
    with h5py.File(path, "r+") as store:
        store["uniform"]["states"][0, 0, 0] += 1e-8
    with pytest.raises(ValueError, match="chunk changed"):
        runner.load_state_bank(root, case, "train")


@pytest.mark.skipif(not compiler_available(), reason="Optional serial C++ native control needs compiler")
def test_reference_chunk_matches_package_native_euler_and_literal_killed_bank(tmp_path):
    plan, backend = smoke_plan(), runner.compile_native(tmp_path/"compiled")
    case = runner.case_names(plan)[0]
    cfg = runner.reference_configuration(case, plan)
    key = "nonuniform__ensemble__0"
    initial, finest, factor, families = runner.reference_draw(cfg, key, 0, 2)
    saved = runner.simulate_reference_chunk(cfg, key, 0, 2, backend=backend)
    assert factor == 4 and np.array_equal(saved["family_ids"], families)
    assert saved["outcomes"].shape == (2, 3, 4)
    assert saved["audit_outcomes"].shape == (2, 3, 3, 4)
    bank = runner.payoff_bank(case, plan)
    for level, refinement in enumerate(cfg.reference_factors):
        dw = runner._aggregate(finest, factor//refinement)
        dt = cfg.reference_step/refinement
        times = np.arange(dw.shape[1]+1)*dt
        packet = simulate_native(cfg.context("nonuniform"), cfg.spec, times, dt, 0, 2,
                                 initial_masses=initial, brownian_increments=dw,
                                 family_ids=tuple(families.astype(str)))
        values = bank.evaluate_paths(packet.states, times, packet.stopped_at)
        indices = np.rint(np.asarray(cfg.query_times)/dt).astype(int)
        np.testing.assert_allclose(saved["audit_outcomes"][:, level], values[:, indices],
                                   atol=3e-15, rtol=0)
    assert set(saved) == set(runner._reference_shapes(cfg))


def test_reference_strict_fit_unlock_precedes_backend_and_anchor_draw(tmp_path, monkeypatch):
    plan = smoke_plan()
    monkeypatch.setattr(runner, "_assert_frozen", lambda root: (plan, {}))

    def reject(root):
        raise ValueError("all eighteen strict fits required")

    monkeypatch.setattr(runner, "verify_fit_lock", reject)
    monkeypatch.setattr(runner, "compile_native", lambda *args: pytest.fail("must not compile or draw"))
    monkeypatch.setattr(runner, "_create_reference", lambda *args: pytest.fail("must not create anchors"))
    with pytest.raises(ValueError, match="eighteen"):
        runner.generate_reference(tmp_path)
    assert not list(tmp_path.rglob("reference.h5"))


def test_native_reduction_rejects_failed_paths_without_survivor_averaging():
    plan = smoke_plan()
    case = runner.case_names(plan)[0]
    cfg = runner.configuration(case, plan)
    initial, noise = np.ones((2, 16))/16, np.zeros((2, 4, 16))
    packet = {"states": np.tile(initial[:, None], (1, 5, 1)), "stopped_at": np.full(2, np.inf)}
    packet["states"][1, 2, 0] = np.nan
    with pytest.raises(ValueError, match="survivor"):
        runner.native_payoff(initial, noise, cfg.physics, cfg.spec, 1/256,
                             runner.payoff_bank(case, plan), cfg.query_times,
                             backend=lambda *args: packet)


def test_declared_model_rejects_changed_checksum_before_load(tmp_path, monkeypatch):
    path = tmp_path/"checkpoint.pt"
    path.write_bytes(b"mutated")
    monkeypatch.setattr(runner, "load_repaired_payoff_model", lambda path: pytest.fail("must verify first"))
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.load_declared_model({"checkpoint_path": path, "checkpoint_sha256": "old"})


def test_fit_inventory_verification_is_exact(tmp_path, monkeypatch):
    plan = smoke_plan()
    slots = runner.slots(plan)
    runner._json(tmp_path/"payoffs"/"preregistration.json", {"frozen": True})
    runner._json(tmp_path/"payoffs"/"data_lock.json", {"banks": "frozen"})
    monkeypatch.setattr(runner, "_prepared", lambda root: (plan, {"slots": slots}))
    records = [{**row, "checkpoint_sha256": "sha"} for row in slots]
    monkeypatch.setattr(runner, "_model_record", lambda root, slot, plan: records[0])
    saved = {"complete": True, "source_hash": runner.source_hash(), "expected_models": 1,
             "models": records, "plan_hash": plan.plan_hash,
             "preregistration_sha256": runner._sha(tmp_path/"payoffs"/"preregistration.json"),
             "data_lock_sha256": runner._sha(tmp_path/"payoffs"/"data_lock.json")}
    runner._json(tmp_path/"payoffs"/"fit_lock.json", saved)
    assert runner.verify_fit_lock(tmp_path) == saved
    runner._json(tmp_path/"payoffs"/"fit_lock.json", {**saved, "expected_models": 0})
    with pytest.raises(ValueError, match="strictly restore"):
        runner.verify_fit_lock(tmp_path)


def test_worker_lock_is_exclusive_and_releases(tmp_path):
    path = tmp_path/"worker.lock"
    with runner._exclusive(path):
        with pytest.raises(RuntimeError, match="Another process"):
            with runner._exclusive(path):
                raise AssertionError("Second lock must fail")
    with runner._exclusive(path):
        assert Path(path).read_text().strip().isdigit()


def test_prepared_lock_rejects_literal_pool_mutation(tmp_path, monkeypatch):
    plan = smoke_plan()
    monkeypatch.setattr(runner, "_assert_frozen", lambda root: (plan, {"slots": []}))
    monkeypatch.setattr(runner, "_data_lock", lambda root, plan: {"changed": True})
    runner._json(tmp_path/"payoffs"/"data_lock.json", {"changed": False})
    with pytest.raises(ValueError, match="collocation data lock"):
        runner._prepared(tmp_path)


def test_zero_chunk_limit_does_not_draw_any_held_out_outcomes(tmp_path, monkeypatch):
    plan = smoke_plan()
    case = runner.case_names(plan)[0]
    cfg = runner.reference_configuration(case, plan)
    inventory = {"models": [], "source_hash": runner.source_hash(), "expected_models": 1}
    monkeypatch.setattr(runner, "_assert_frozen", lambda root: (plan, {}))
    monkeypatch.setattr(runner, "verify_fit_lock", lambda root: inventory)
    monkeypatch.setattr(runner, "compile_native", lambda *args: SimpleNamespace())
    monkeypatch.setattr(runner, "simulate_reference_chunk", lambda *args: pytest.fail("no outcome draws"))
    result = runner.generate_reference(tmp_path, max_chunks=0)
    assert result[case]["status"] == "partial"
    assert result[case]["committed_chunks"] == 0
    path = tmp_path/"reference"/case/"reference.h5"
    with h5py.File(path, "r") as store:
        assert not any(store[key]["chunk_complete"][:].any() for key in cfg.group_keys)
