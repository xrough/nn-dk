"""Tiny software-only labels, immutable banks and all-seed correction locks."""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import run_lna_payoff_correction_study as runner  # noqa: E402
from analysis.compiled_native import compile_native, compiler_available  # noqa: E402


def tiny_plan(**changes):
    values = dict(horizons=(1/64,), kappas=(.75,), n_cells=8, n_particles=2048,
        train_natural_per_profile=2, train_stress_per_profile=1, validation_per_profile=1,
        train_continuations=4, validation_continuations=4, label_chunk=2, label_step=1/1024,
        width=4, depth=1, adam_steps=1, lbfgs_steps=0, batch_size=4, validation_every=1)
    return runner.StudyPlan(**(values | changes))


@pytest.fixture
def anchors(tmp_path):
    torch.set_num_threads(1)
    root, plan = tmp_path/"candidate", tiny_plan()
    runner.prepare(root, plan)
    return root, plan


@pytest.fixture
def data(anchors):
    if not compiler_available():
        pytest.skip("Optional native C++ compiler unavailable")
    root, plan = anchors
    assert runner.generate_data(root)["complete"]
    return root, plan


def test_defaults_prespecify_all_six_cases_and_eighteen_fits_without_outcomes():
    plan = runner.StudyPlan()
    assert len(runner.cases(plan)) == 6 and len(runner.slots(plan)) == 18
    assert plan.optimizer_seeds == (0, 1, 2)
    assert plan.train_natural_per_profile == 40 and plan.train_stress_per_profile == 20
    assert plan.validation_per_profile == 20
    assert plan.train_continuations == 256 and plan.validation_continuations == 512
    assert plan.label_step == 2**-12 and plan.baseline_step == 2**-7
    assert plan.batch_size == 64 and plan.adam_steps == 1000 and plan.lbfgs_steps == 30
    assert (plan.train_initial_seed, plan.train_label_seed, plan.validation_initial_seed,
            plan.validation_label_seed) == (2026103001, 2026103002, 2026103004, 2026103005)
    assert runner.StudyPlan.from_dict(plan.to_dict()) == plan
    for case in runner.cases(plan):
        assert runner.configuration(case, plan).label_step == plan.label_step
        assert tuple(runner.query_times(case, plan)) == runner.configuration(case, plan).query_times


@pytest.mark.parametrize("change", [{"optimizer_seeds": (0,)}, {"train_natural_per_profile": 3},
    {"validation_label_seed": 2026103002}, {"train_continuations": 3}, {"stress_distance_multipliers": (.25,)}])
def test_invalid_or_truncated_designs_are_rejected(change):
    with pytest.raises(ValueError):
        tiny_plan(**change)


def test_stress_rows_keep_parent_groups_and_exact_declared_width_units(anchors):
    root, plan = anchors
    case = runner.cases(plan)[0]
    train, _ = runner._anchors(root, case, "train", plan)
    validation, _ = runner._anchors(root, case, "validation", plan)
    assert train["states"].shape == (6, 8) and validation["states"].shape == (2, 8)
    assert list(train["row_kinds"].astype(str)).count("native") == 4
    assert list(train["row_kinds"].astype(str)).count("stress") == 2
    assert not set(train["parent_family_ids"]) & set(validation["parent_family_ids"])
    stress = train["row_kinds"].astype(str) == "stress"
    np.testing.assert_allclose(train["states"][stress].min(-1),
        runner.spec(case, plan).mass_stop+.25*.05/plan.n_cells, atol=1e-18, rtol=0)
    assert all(value in train["family_ids"][~stress] for value in train["parent_family_ids"][stress])
    assert not list(root.rglob("*.h5"))
    assert not (root/"data/data_lock.json").exists()
    assert runner.prepare(root, plan) == root/"data"
    larger = tiny_plan(train_natural_per_profile=6, train_stress_per_profile=3)
    generated = runner._draw_anchors(case, "train", larger)
    selected = generated["row_kinds"].astype(str) == "stress"
    np.testing.assert_allclose(generated["target_distances"][selected],
                              np.tile(np.array([.25, 1., 4.])*.05/8, 2), atol=0, rtol=0)


def test_label_chunks_resume_without_redrawing_committed_families(anchors, monkeypatch):
    if not compiler_available():
        pytest.skip("Optional native C++ compiler unavailable")
    root, plan = anchors
    case = runner.cases(plan)[0]
    assert runner.generate_data(root, max_chunks=1) == {"complete": False, "committed_chunks": 1}
    first = root/"data"/case/"train/labels/anchor0/chunk0/arrays.npz"
    original = first.read_bytes()
    draw = runner._draw_labels
    called = []

    def resumed(state, case, role, profile, anchor, start, count, plan, backend):
        assert (role, anchor, start) != ("train", 0, 0)
        called.append((role, anchor, start))
        return draw(state, case, role, profile, anchor, start, count, plan, backend)

    monkeypatch.setattr(runner, "_draw_labels", resumed)
    assert runner.generate_data(root)["complete"]
    assert len(called) == 15 and first.read_bytes() == original
    assert runner.prepared(root)[0] == plan

    def forbidden(*args, **kwargs):
        pytest.fail("Completed baseline/label evidence was regenerated")

    monkeypatch.setattr(runner, "_draw_labels", forbidden)
    monkeypatch.setattr(runner, "baseline_batch", forbidden)
    assert runner.generate_data(root)["complete"]
    assert runner.load_bank(root, case, "train")["supervised_native_MC_labels"]


def test_label_reduction_is_unbiased_covariance_and_preserves_every_family(data):
    root, plan = data
    case = runner.cases(plan)[0]
    bank = runner.load_bank(root, case, "train")
    draws = []
    families, forcing = set(), set()
    for path in sorted((root/"data"/case).rglob("chunk*/arrays.npz")):
        with np.load(path) as arrays:
            ids = set(arrays["family_ids"].astype(str))
            entropy = set(arrays["forcing_entropy_sha256"].astype(str))
            assert not families & ids and not forcing & entropy
            families.update(ids)
            forcing.update(entropy)
            if path.parent.parent.name == "anchor0" and "train" in path.parts:
                draws.append(arrays["outcomes"])
    values = np.concatenate(draws)
    np.testing.assert_allclose(bank["label_mean"][0], values.mean(0), atol=0, rtol=0)
    expected = np.stack([np.cov(values[:, q], rowvar=False, ddof=1) for q in range(3)])*plan.n_particles
    np.testing.assert_allclose(bank["label_scaled_covariance"][0], expected, atol=1e-15, rtol=1e-14)
    assert len(families) == len(forcing) == 32
    lock = runner._read(root/"data/data_lock.json")
    assert len(lock["identifier_inventory"]) == 18
    assert all("outcomes" not in row["fields"] for row in lock["identifier_inventory"])
    assert bank["provenance"]["training_mixture"] == {"native": 2/3, "stress": 1/3}
    assert bank["provenance"]["all_stopped_paths_retained"]


def test_initially_absorbed_label_paths_remain_in_zero_covariance_target(anchors):
    if not compiler_available():
        pytest.skip("Optional native C++ compiler unavailable")
    _, plan = anchors
    case = runner.cases(plan)[0]
    state = np.full(plan.n_cells, 1/plan.n_cells)
    state[0] = runner.spec(case, plan).mass_stop
    state[1] += 1/plan.n_cells-state[0]
    arrays = runner._draw_labels(state, case, "train", "uniform", 0, 0, 4, plan, compile_native())
    assert len(arrays["family_ids"]) == 4
    np.testing.assert_array_equal(arrays["stopped_at"], 0)
    np.testing.assert_array_equal(arrays["outcomes"][..., 2:], 0)
    np.testing.assert_allclose(arrays["outcomes"], np.broadcast_to(
        state@runner.payoff_bank(case, plan).terminal_weights.T, (4, 3, 4)), atol=2e-15, rtol=0)


def test_corruption_and_missing_reduced_evidence_are_rejected(data):
    root, plan = data
    case = runner.cases(plan)[0]
    path = root/"data"/case/"train/labels/anchor0/chunk0/arrays.npz"
    saved = path.read_bytes()
    path.write_bytes(saved+b"corrupt")
    with pytest.raises(ValueError, match="checksum changed"):
        runner.prepared(root)
    path.write_bytes(saved)
    reduced = root/"data"/case/"train/bank/arrays.npz"
    reduced.unlink()
    with pytest.raises(ValueError, match="Incomplete reduced"):
        runner.prepared(root)
    assert not reduced.exists()


def test_source_plan_and_incomplete_chunk_guards_precede_generation(anchors):
    root, plan = anchors
    changed = plan.to_dict()
    changed["width"] += 1
    original = (root/"plan.json").read_bytes()
    runner._json(root/"plan.json", changed)
    with pytest.raises(ValueError, match="source, helper, plan"):
        runner.generate_data(root)
    (root/"plan.json").write_bytes(original)
    case = runner.cases(plan)[0]
    broken = root/"data"/case/"train/labels/anchor0/chunk0"
    broken.mkdir(parents=True)
    (broken/"arrays.npz").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Incomplete correction evidence"):
        runner.generate_data(root)


def test_all_three_tiny_fits_restore_and_lock_without_final_access(data, monkeypatch):
    root, plan = data
    with pytest.raises(FileNotFoundError):
        runner.lock(root)
    for seed in plan.optimizer_seeds:
        worker = runner.fit_seed(root, seed)
        assert worker["complete"] and len(worker["models"]) == 1
        assert worker["held_out_reference_inputs_read"] is False
        model = worker["models"][0]
        assert model["full_invocation_cost_complete"]
        assert model["full_invocation_seconds"] >= model["training_seconds"] > 0
        assert model["baseline_step"] == plan.baseline_step
    locked = runner.lock(root)
    assert locked == runner.verify_fit_lock(root)
    assert len(locked["cost_records"]) == 5
    worker_path = root/"payoffs/execution/worker_seed0.json"
    saved_worker = worker_path.read_bytes()
    worker_path.write_bytes(saved_worker+b" ")
    with pytest.raises(ValueError, match="strictly restore"):
        runner.verify_fit_lock(root)
    worker_path.write_bytes(saved_worker)
    assert locked["expected_models"] == 3
    assert {row["seed"] for row in locked["models"]} == {0, 1, 2}
    assert len({row["method_id"] for row in locked["models"]}) == 3
    lock_path = root/"payoffs/fit_lock.json"
    original = lock_path.read_bytes()
    changed = copy.deepcopy(locked)
    changed["collocation_root"] = str(root/"unrelated-study")
    runner._json(lock_path, changed)
    with pytest.raises(ValueError, match="strictly restore"):
        runner.verify_fit_lock(root)
    lock_path.write_bytes(original)
    for action in (runner.generate_reference, runner.load_reference):
        with pytest.raises(ValueError, match="No final"):
            action(root)
    with pytest.raises(FileNotFoundError):
        runner.assert_joint_fit_locks(root)
    from analysis import run_payoff_collocation_study as other

    other_plan = other.StudyPlan(horizons=plan.horizons, kappas=plan.kappas,
        n_cells=plan.n_cells, n_particles=plan.n_particles, state_step=plan.label_step,
        reference_step=plan.label_step)
    with monkeypatch.context() as patch:
        patch.setattr(other, "verify_fit_lock", lambda path: {"complete": True, "expected_models": 12})
        patch.setattr(other, "load_plan", lambda path: other_plan)
        joint = runner.assert_joint_fit_locks(root)
        assert joint["candidate"] == locked and joint["collocation"]["complete"]
        patch.setattr(other, "load_plan", lambda path: replace(other_plan, n_particles=4096))
        with pytest.raises(ValueError, match="target, stopping spec"):
            runner.assert_joint_fit_locks(root)
    with pytest.raises(ValueError, match="fitting is closed"):
        runner.fit_seed(root, 0)
    with pytest.raises(ValueError, match="preparation is closed"):
        runner.prepare(root, plan)
    with pytest.raises(ValueError, match="data generation is closed"):
        runner.generate_data(root)
    path = Path(locked["models"][0]["checkpoint_path"])
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="strictly restore"):
        runner.verify_fit_lock(root)


def test_seed_ownership_deadlines_and_costs_are_explicit(data):
    root, _ = data
    with runner.utilities._exclusive(root/"payoffs/execution/.seed0.lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            runner.fit_seed(root, 0)
    with pytest.raises(runner.BudgetExhausted, match="resumable deadline"):
        runner.fit_seed(root, 0, deadline=time.monotonic()-1)
    worker = runner._read(root/"payoffs/execution/worker_seed0.json")
    assert not worker["complete"] and not worker["attempts"][-1]["complete"]
    assert runner.fit_seed(root, 0)["complete"]
    costs = runner._read(root/"execution/data_costs.json")
    assert costs["attempts"][0]["complete"]
    changed = copy.deepcopy(costs)
    changed["attempts"][0]["wall_seconds"] = 0
    runner._json(root/"execution/data_costs.json", changed)
    with pytest.raises(ValueError, match="cost record identity"):
        runner.generate_data(root)
