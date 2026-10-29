"""Fresh matched pools, complete crossed allocations and strict reference unlock."""
from __future__ import annotations

import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis import run_payoff_regimes as historical  # noqa: E402
from analysis.compiled_native import compiler_available  # noqa: E402
from analysis.evaluate_path_generalization import array_sha  # noqa: E402


def tiny_plan():
    return runner.StudyPlan(horizons=(1/64,), kappas=(.75,), n_cells=8, n_particles=64,
        train_paths=2, validation_paths=2, horizon_samples=1, width=4,
        adam_steps=1, lbfgs_steps=0, lbfgs_max_iter=1, validation_every=1,
        batch_size=4, quadrature_nodes=2, state_step=1/1024, reference_step=1/1024,
        conditional_anchors=1, paths_per_group=4, refinement_paths=2, chunk_paths=2)


@pytest.fixture
def prepared(tmp_path):
    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    torch.set_num_threads(1)
    root, plan = tmp_path/"study", tiny_plan()
    runner.prepare(root, plan)
    return root, plan


def comparison_stub(root, monkeypatch):
    """Real checksum-bound identifier files, without any outcomes or draws."""
    helper_directory = root.parent/"future-helpers"
    helper_directory.mkdir(exist_ok=True)
    helpers = tuple(helper_directory/name for name in (
        "evaluate_payoff_collocation_study.py", "report_payoff_collocation_study.py"))
    for path in helpers:
        path.write_text("# temporary software-test comparison helper\n")
    monkeypatch.setattr(runner, "evaluation_helper_paths", lambda: helpers)
    hashes = {str(path.resolve()): runner._sha(path) for path in helpers}
    contract = runner.evaluation_contract(root)
    comparison = {"contract": contract, "accuracy_target": .05,
        "uncertainty_component_target": .0125, "bootstrap": {"count": 128},
        "selection_using_test": False, "post_test_refitting": False, "helpers": hashes}
    runner._json(root/"comparison/preregistration.json", comparison)
    pools, arrays = [], {}
    for index, item in enumerate(contract["independent_IID_integration"]):
        key = "pool"+str(index)
        values = np.asarray([hashlib.sha256(f"test-independent-{index}-{i}".encode()).hexdigest()
                             for i in range(item["families"])], dtype="S64")
        arrays[key] = values
        pools.append({"key": key, "regime": item["case"], "profile": item["profile"],
            "group": item["group"], "paths": item["families"],
            "family_ids_array_sha256": array_sha(values)})
    deployment = {"helpers": hashes, "pools": pools, "models_per_pool": 13,
                  "expected_records": contract["counts"]["independent_IID_replacement_records"]}
    runner._json(root/"deployment/preregistration.json", deployment)
    np.savez(root/"deployment/planned_initial_families.npz", **arrays)
    refresh_address_stub(root)
    return helpers


def refresh_address_stub(root):
    runner._json(root/"audit/deployment_pretest_independence.json", {
        "complete": True, "source_hash": runner.source_hash(), "planned_families_disjoint": True,
        "address_streams_disjoint": True,
        "deployment_plan_sha256": runner._sha(root/"deployment/preregistration.json"),
        "planned_family_npz_sha256": runner._sha(root/"deployment/planned_initial_families.npz"),
        "original_comparison_preregistration_sha256": runner._sha(root/"comparison/preregistration.json"),
        "training_preregistration_sha256": runner._sha(root/"payoffs/preregistration.json")})


def test_scientific_defaults_retain_full_crossing_and_new_streams():
    plan = runner.StudyPlan()
    design = runner._design(plan)
    assert len(runner.slots(plan)) == 72
    assert len(design["configs"]) == 6
    assert plan.train_seed == 2026103101 and plan.reference_seed == 2026103102
    assert plan.train_paths == 256 and plan.validation_paths == 128
    assert plan.paths_per_group == 16384 and plan.refinement_paths == 1024
    assert plan.reference_step == 2**-12 and plan.reference_factors == (1, 2, 4)
    assert design["stress_augmentation"] is False
    assert "natural-occupancy" in design["selection"]
    assert historical.STREAM != runner.STREAM
    assert historical.StudyPlan().train_seed == 2026102801
    assert historical.WORKFLOW == "harder-native-dk-repaired-payoff-comparison-v1"
    for case in runner.case_names(plan):
        cfgs = [runner.ablation_configuration(slot, plan) for slot in runner.slots(plan)
                if slot["case"] == case and slot["seed"] == 0]
        assert len({cfg.config_hash for cfg in cfgs}) == 4
        assert len({cfg.repair.study.config_hash for cfg in cfgs}) == 1


@pytest.mark.parametrize("change", [{"optimizer_seeds": (0,)}, {"proposals": ("uniform",)},
                                   {"losses": ("squared",)}])
def test_crossed_controls_cannot_be_silently_removed(change):
    with pytest.raises(ValueError, match="complete fixed 2x2"):
        runner.StudyPlan(**change)


def test_literal_flat_pool_matches_every_variant_and_parent_split(prepared):
    root, plan = prepared
    case = runner.case_names(plan)[0]
    slots = [slot for slot in runner.slots(plan) if slot["seed"] == 0]
    bank_digests = {role: [] for role in ("train", "validation")}
    pools = {}
    for slot in slots:
        cfg = runner.ablation_configuration(slot, plan)
        for role in bank_digests:
            bank = runner.flat_bank(root, case, role, plan, cfg)
            bank_digests[role].append(bank["content_sha256"])
            assert bank["provenance"]["stress_augmentation"] is False
            assert set(bank["row_kinds"]) == {"native"}
            assert bank["future_payoff_labels"] is False
            pools[role] = bank
    assert all(len(set(values)) == 1 for values in bank_digests.values())
    assert not set(pools["train"]["parent_family_ids"]) & set(pools["validation"]["parent_family_ids"])
    assert not list((root/"reference").rglob("reference.h5"))
    assert runner.prepare(root, plan) == root/"payoffs"
    costs = runner._cost_log(root/"payoffs/prepare_costs.json", {
        "workflow": runner.WORKFLOW, "source_hash": runner.source_hash(), "plan_hash": plan.plan_hash})
    assert len(costs["attempts"]) == 2
    assert all(value["finished"] and value["wall_seconds"] > 0 for value in costs["attempts"])


def test_all_twelve_tiny_fits_match_controls_before_reference_unlock(prepared, monkeypatch):
    root, plan = prepared
    with pytest.raises(ValueError, match="Freeze comparative evaluation"):
        runner.generate_reference(root)
    assert not list((root/"reference").rglob("reference.h5"))
    rng_before = torch.get_rng_state().clone()
    for seed in (0, 1, 2):
        packet = runner.fit_seed(root, seed)
        assert packet["complete"] and len(packet["models"]) == 4
        assert packet["held_out_reference_inputs_read"] is False
        assert set(packet["pool_preparation"]) == set(runner.case_names(plan))
        assert packet["preflight_integrity_wall_seconds"] > 0
        assert packet["attempts"][-1]["pool_preparation"] == packet["pool_preparation"]
        for model in packet["models"]:
            assert model["full_invocation_cost_complete"]
            assert model["full_invocation_seconds"] >= model["training_seconds"] > 0
            assert model["full_invocation_cpu_seconds"] > 0
            assert runner._sha(Path(model["invocation_record_path"])) == model["invocation_record_sha256"]
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
    locked = runner.lock(root)
    assert locked["complete"] and locked["expected_models"] == 12
    assert runner.verify_fit_lock(root) == locked
    controls = runner._implementation._read(root/"payoffs/matched_controls.json")
    assert len(controls["records"]) == 3
    assert controls["all_variant_pool_initialization_and_scale_controls_matched"]
    assert len({row["method_id"] for row in locked["models"]}) == 12
    for seed in (0, 1, 2):
        rows = [row for row in locked["models"] if row["seed"] == seed]
        assert len({row["initial_state_sha256"] for row in rows}) == 1
        assert len({row["loss_scaling_sha256"] for row in rows}) == 1
    assert not list((root/"reference").rglob("reference.h5"))
    with pytest.raises(ValueError, match="Freeze comparative evaluation"):
        runner.generate_reference(root, max_chunks=0)
    helpers = comparison_stub(root, monkeypatch)
    frozen = runner.freeze_evaluation(root)
    assert frozen == runner.verify_evaluation_lock(root) == runner.freeze_evaluation(root)
    assert frozen["contract"]["counts"] == {"trained_models": 12, "primary_records": 64,
        "independent_IID_replacement_records": 26, "raw_retained_records": 90}
    assert set(frozen["helpers"]) == {str(path.resolve()) for path in helpers}
    assert not list((root/"reference").rglob("reference.h5"))
    partial = runner.generate_reference(root, max_chunks=1)
    assert next(iter(partial.values()))["status"] == "partial"
    with pytest.raises(ValueError, match="complete, unmutated"):
        runner.load_reference(root, runner.case_names(plan)[0])
    complete = runner.generate_reference(root)
    assert next(iter(complete.values()))["status"] == "complete"
    reference = runner.load_reference(root, runner.case_names(plan)[0])
    assert len(reference["groups"]) == 4
    ids = [family.decode() for group in reference["groups"].values() for family in group["family_ids"]]
    assert len(ids) == 16 and len(set(ids)) == 16
    with pytest.raises(ValueError, match="before any held-out"):
        runner.freeze_evaluation(root)
    with pytest.raises(ValueError, match="fitting is closed"):
        runner.fit_seed(root, 0)
    checkpoint = Path(locked["models"][0]["checkpoint_path"])
    checkpoint.write_bytes(checkpoint.read_bytes()+b"stale model inventory")
    with pytest.raises(ValueError, match="strictly restore"):
        runner.verify_evaluation_lock(root)


def test_budget_resume_and_active_seed_owner_are_enforced(prepared):
    root, _ = prepared
    with runner._implementation._exclusive(root/"payoffs/execution/.seed0.lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            runner.fit_seed(root, 0)
    with pytest.raises(runner.BudgetExhausted, match="resumable deadline"):
        runner.fit_seed(root, 0, deadline=time.monotonic()-1)
    assert not runner._implementation._read(root/"payoffs/execution/worker_seed0.json")["complete"]
    assert runner.fit_seed(root, 0)["complete"]
    attempts = runner._read(root/"payoffs/execution/worker_seed0.json")["attempts"]
    assert len(attempts) == 2 and not attempts[0]["complete"] and attempts[1]["complete"]
    assert attempts[0]["pool_preparation"] == {}
    assert len(attempts[1]["pool_preparation"]) == 1
    with pytest.raises(FileNotFoundError):
        runner.lock(root)
    assert not list((root/"reference").rglob("reference.h5"))


def test_pool_lock_or_checkpoint_tampering_rejects_unlock(prepared):
    root, plan = prepared
    flat_path = root/"payoffs/flat_pool_lock.json"
    original = flat_path.read_bytes()
    changed = runner._implementation._read(flat_path)
    case = runner.case_names(plan)[0]
    changed["banks"][case]["train"]["content_sha256"] = "0"*64
    runner._implementation._json(flat_path, changed)
    with pytest.raises(ValueError, match="flat train/validation pool lock"):
        runner.fit_seed(root, 0)
    flat_path.write_bytes(original)
    for seed in (0, 1, 2):
        runner.fit_seed(root, seed)
    result = runner.lock(root)
    checkpoint = Path(result["models"][0]["checkpoint_path"])
    checkpoint.write_bytes(checkpoint.read_bytes()+b"changed")
    with pytest.raises(ValueError):
        runner.generate_reference(root, max_chunks=0)
    assert not list((root/"reference").rglob("reference.h5"))


def test_evaluation_gate_rejects_missing_stale_or_post_outcome_freezes(prepared, monkeypatch):
    root, _ = prepared
    for seed in (0, 1, 2):
        runner.fit_seed(root, seed)
    runner.lock(root)
    helper = root.parent/"missing-future-evaluator.py"
    monkeypatch.setattr(runner, "evaluation_helper_paths", lambda: (helper,))
    with pytest.raises(ValueError, match="missing or empty"):
        runner.freeze_evaluation(root)
    helpers = comparison_stub(root, monkeypatch)
    reference = root/"reference"/runner.case_names(runner.load_plan(root))[0]/"reference.h5"
    reference.write_bytes(b"no final data may predate an evaluation freeze")
    with pytest.raises(ValueError, match="before any held-out"):
        runner.freeze_evaluation(root)
    reference.unlink()
    runner.freeze_evaluation(root)
    original = helpers[0].read_bytes()
    helpers[0].write_bytes(original+b"# stale implementation\n")
    with pytest.raises(ValueError, match="helper hash"):
        runner.generate_reference(root, max_chunks=0)
    helpers[0].write_bytes(original)
    comparison_path = root/"comparison/preregistration.json"
    original = comparison_path.read_bytes()
    comparison = runner._read(comparison_path)
    comparison["accuracy_target"] = .10
    runner._json(comparison_path, comparison)
    refresh_address_stub(root)
    with pytest.raises(ValueError, match="criteria changed"):
        runner.verify_evaluation_lock(root)
    comparison_path.write_bytes(original)
    refresh_address_stub(root)
    assert runner.verify_evaluation_lock(root)["complete"]
    deployment_path = root/"deployment/preregistration.json"
    original = deployment_path.read_bytes()
    deployment = runner._read(deployment_path)
    deployment["pools"].pop()
    runner._json(deployment_path, deployment)
    refresh_address_stub(root)
    with pytest.raises(ValueError, match="Incomplete independent"):
        runner.verify_evaluation_lock(root)
    deployment_path.write_bytes(original)
    refresh_address_stub(root)
    assert runner.verify_evaluation_lock(root)["complete"]
    assert not list((root/"reference").rglob("reference.h5"))


def test_reference_gate_precedes_native_initial_anchor_creation(prepared, monkeypatch):
    root, _ = prepared

    def forbidden(*args, **kwargs):
        pytest.fail("Native held-out initial anchors were drawn before evaluation freeze")

    monkeypatch.setattr(runner._implementation, "_create_reference", forbidden)
    with pytest.raises(ValueError, match="Freeze comparative evaluation"):
        runner.generate_reference(root, max_chunks=1)
    assert not list((root/"reference").rglob("reference.h5"))


def test_full_invocation_cost_log_cannot_change_after_fit_lock(prepared):
    root, _ = prepared
    for seed in (0, 1, 2):
        runner.fit_seed(root, seed)
    locked = runner.lock(root)
    path = Path(locked["models"][0]["invocation_record_path"])
    packet = runner._read(path)
    packet["attempts"][0]["wall_seconds"] *= 2
    runner._json(path, packet)
    with pytest.raises(ValueError, match="cost provenance or content"):
        runner.verify_fit_lock(root)


def test_matched_initialization_and_pool_comparison_rejects_a_confounded_fit(prepared):
    root, plan = prepared
    for seed in (0, 1, 2):
        runner.fit_seed(root, seed)
    rows = [runner._model_record(root, slot, plan) for slot in runner.slots(plan)]
    bad = copy.deepcopy(rows)
    bad[0]["initial_state_sha256"] = "0"*64
    with pytest.raises(ValueError, match="identical initialization"):
        runner._matched_controls(bad, plan)
    bad = copy.deepcopy(rows)
    bad[0]["data_fingerprints"]["train"]["rows"] += 1
    with pytest.raises(ValueError, match="identical initialization"):
        runner._matched_controls(bad, plan)
