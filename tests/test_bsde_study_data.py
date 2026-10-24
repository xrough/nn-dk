"""Coverage ablations preserve path budgets, nested anchors and split isolation."""

import time
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nn_dk import bsde_study_data
from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_data import generate_bsde_pool, load_bsde_pool
from nn_dk.bsde_study_data import (
    generate_study_pool,
    load_study_pool,
    study_pool_identity,
)
from nn_dk.types import seeded_rng

ALLOCATIONS = ((2, 4), (4, 2), (8, 1))
VALIDATION = (4, 2)


def config():
    return replace(bsde_preset("bsde-smoke"), train_anchors=2, validation_anchors=2,
                   horizon=1 / 32, query_times=(1 / 64, 3 / 128, 1 / 32))


def options():
    return {"allocations": ALLOCATIONS, "validation": VALIDATION}


@pytest.mark.parametrize("problem", ["ou", "dk"])
def test_union_views_are_fixed_budget_nested_disjoint_and_reproducible(
        tmp_path, monkeypatch, problem):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    identity = study_pool_identity(cfg, problem, **options())
    assert identity["training_path_budget"] == 8
    assert identity["training_union_paths"] == 16
    generated = generate_study_pool(tmp_path / "pool", cfg, problem, **options())
    assert generated["complete"] and generated["generated_this_call"] == 12
    assert generated["paths_this_call"] == 24
    assert generate_study_pool(tmp_path / "pool", cfg, problem, **options())["paths_this_call"] == 0
    generate_study_pool(tmp_path / "fresh", cfg, problem, **options())
    views = [load_study_pool(tmp_path / "pool", cfg, problem, "train", allocation=a,
                            **options()) for a in ALLOCATIONS]
    validation = load_study_pool(tmp_path / "pool", cfg, problem, "validation", **options())
    for allocation, view in zip(ALLOCATIONS, views):
        assert view.n_paths == 8
        assert len(set(view.anchor_ids)) == allocation[0]
        assert view.diagnostics["allocation"] == list(allocation)
        assert view.proposals is None
        fresh = load_study_pool(tmp_path / "fresh", cfg, problem, "train",
                                allocation=allocation, **options())
        assert fresh.fingerprint == view.fingerprint
        assert not set(view.anchor_ids) & set(validation.anchor_ids)
        assert not set(view.family_ids) & set(validation.family_ids)
        if problem == "dk":
            assert view.diagnostics["cases"].count("uniform") == 4
            assert view.diagnostics["cases"].count("nonuniform") == 4
    assert set(views[0].anchor_ids) < set(views[1].anchor_ids) < set(views[2].anchor_ids)
    # Shared continuations are literally the same samples, rather than redraws.
    for family in set(views[0].family_ids) & set(views[1].family_ids):
        a, b = views[0].family_ids.index(family), views[1].family_ids.index(family)
        assert np.array_equal(views[0].states[a], views[1].states[b])
        assert np.array_equal(views[0].increments[a], views[1].increments[b])
    oracle_view = load_study_pool(tmp_path / "pool", cfg, problem, "train",
                                 allocation=ALLOCATIONS[0], include_proposals=True, **options())
    assert oracle_view.proposals is not None
    assert oracle_view.fingerprint == views[0].fingerprint
    if problem == "dk":
        assert validation.diagnostics["cases"].count("uniform") == 4
        assert validation.diagnostics["cases"].count("nonuniform") == 4
    with h5py.File(tmp_path / "pool" / f"{problem}_study_pool.h5", "r") as store:
        assert len(store["train"]) == 8
        assert [store[f"train/anchor_{i:06d}"].attrs["n_paths"] for i in range(8)] == [
            4, 4, 2, 2, 1, 1, 1, 1]
        assert "proposals" not in store["train/anchor_000000/learner"]
        assert store["train/anchor_000000/oracle"].attrs["not_network_inputs"]
        assert store["train/anchor_000000/learner/states"].chunks is not None


def test_study_domain_separates_previous_pool_and_explicit_seed(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    monkeypatch.setattr("nn_dk.bsde_data.source_hash", lambda: "study-test-source")
    generate_bsde_pool(tmp_path / "old", cfg, "ou")
    previous = load_bsde_pool(tmp_path / "old", cfg, "ou", "train")
    generate_study_pool(tmp_path / "new", cfg, "ou", **options())
    current = load_study_pool(tmp_path / "new", cfg, "ou", "train",
                              allocation=ALLOCATIONS[0], **options())
    generate_study_pool(tmp_path / "new-seed", cfg, "ou", seed=cfg.seed + 1, **options())
    new_seed = load_study_pool(tmp_path / "new-seed", cfg, "ou", "train",
                               allocation=ALLOCATIONS[0], seed=cfg.seed + 1, **options())
    assert not set(previous.anchor_ids) & set(current.anchor_ids)
    assert not set(previous.family_ids) & set(current.family_ids)
    assert not set(current.anchor_ids) & set(new_seed.anchor_ids)
    assert not np.array_equal(current.initial, new_seed.initial)


def test_study_pool_recovers_pending_and_rejects_plan_source_changes(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    expired = generate_study_pool(tmp_path, cfg, "ou", deadline=time.monotonic() - 1, **options())
    assert not expired["complete"] and expired["paths_this_call"] == 0
    with pytest.raises(ValueError, match="incomplete|absent"):
        load_study_pool(tmp_path, cfg, "ou", "train", allocation=ALLOCATIONS[0], **options())
    with h5py.File(tmp_path / "ou_study_pool.h5", "a") as store:
        store["train"].create_group("_pending_anchor_000000")
    assert generate_study_pool(tmp_path, cfg, "ou", **options())["complete"]
    with h5py.File(tmp_path / "ou_study_pool.h5", "r") as store:
        assert not any(name.startswith("_pending_") for name in store["train"])
    for changed in ({**options(), "seed": cfg.seed + 1},
                    {**options(), "validation": (2, 2)},
                    {**options(), "allocations": ((2, 4), (8, 1))}):
        with pytest.raises(ValueError, match="changed"):
            generate_study_pool(tmp_path, cfg, "ou", **changed)
    with pytest.raises(ValueError, match="changed"):
        generate_study_pool(tmp_path, replace(cfg, seed=cfg.seed + 1), "ou", **options())
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="changed"):
        load_study_pool(tmp_path, cfg, "ou", "train", allocation=ALLOCATIONS[0], **options())


def test_committed_anchor_survives_interruption_and_matches_fresh_pool(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    readings = iter((0.0, 2.0))
    monkeypatch.setattr(bsde_study_data, "time", SimpleNamespace(
        monotonic=lambda: next(readings), perf_counter=time.perf_counter))
    partial = generate_study_pool(tmp_path / "resumed", cfg, "ou", deadline=1.0, **options())
    assert not partial["complete"]
    assert partial["generated_this_call"] == 1 and partial["paths_this_call"] == 4
    with h5py.File(tmp_path / "resumed" / "ou_study_pool.h5", "r") as store:
        original = store["train/anchor_000000/learner/states"][:]
        assert store["train/anchor_000000"].attrs["complete"]
    monkeypatch.setattr(bsde_study_data, "time", time)
    completed = generate_study_pool(tmp_path / "resumed", cfg, "ou", **options())
    assert completed["complete"] and completed["paths_this_call"] == 20
    with h5py.File(tmp_path / "resumed" / "ou_study_pool.h5", "r") as store:
        assert np.array_equal(original, store["train/anchor_000000/learner/states"][:])
    generate_study_pool(tmp_path / "fresh", cfg, "ou", **options())
    for allocation in ALLOCATIONS:
        resumed = load_study_pool(tmp_path / "resumed", cfg, "ou", "train",
                                  allocation=allocation, **options())
        fresh = load_study_pool(tmp_path / "fresh", cfg, "ou", "train",
                                allocation=allocation, **options())
        assert resumed.fingerprint == fresh.fingerprint


def test_scientific_default_plan_uses_5120_training_and_1024_validation_paths():
    identity = study_pool_identity(bsde_preset("bsde-first-batch"), "dk")
    assert identity["allocations"] == [[128, 16], [512, 4], [2048, 1]]
    assert identity["training_path_budget"] == 2048
    assert identity["training_union_paths"] == 5120
    assert identity["validation"] == [256, 4]


def test_study_initial_and_forcing_seeds_are_separate_from_fixed_test_namespaces(
        tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    generate_study_pool(tmp_path, cfg, "dk", **options())
    with h5py.File(tmp_path / "dk_study_pool.h5", "r") as store:
        study_initial = {int(group.attrs["initial_seed"])
                         for split in ("train", "validation") for group in store[split].values()}
        study_forcing = {int(group.attrs["forward_seed"])
                         for split in ("train", "validation") for group in store[split].values()}
        # These are the evaluator's fixed reference stream keys; derived child
        # samples remain in that test family and are never pool continuations.
        for case in cfg.cases:
            for kind in ("conditional", "ensemble"):
                initial_seed = int(seeded_rng(cfg.seed, "bsde-test-initial", case, kind).integers(
                    0, 2**63 - 1))
                assert initial_seed not in study_initial
                for start in range(0, 16, cfg.reference_batch_size):
                    forcing_seed = int(seeded_rng(cfg.seed, "bsde-test-forcing", case, kind,
                                                  start).integers(0, 2**63 - 1))
                    assert forcing_seed not in study_forcing
        assert not study_initial & study_forcing


def test_view_selection_and_family_identity_are_checked(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr("nn_dk.bsde_study_data.source_hash", lambda: "study-test-source")
    generate_study_pool(tmp_path, cfg, "ou", **options())
    for kwargs in ({"split": "test"}, {"split": "train"},
                   {"split": "train", "allocation": (2, 2)},
                   {"split": "validation", "allocation": (4, 2)}):
        with pytest.raises(ValueError):
            load_study_pool(tmp_path, cfg, "ou", **kwargs, **options())
    with h5py.File(tmp_path / "ou_study_pool.h5", "a") as store:
        store["train/anchor_000000/learner/family_ids"][0] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        load_study_pool(tmp_path, cfg, "ou", "train", allocation=ALLOCATIONS[0], **options())
    with pytest.raises(ValueError, match="identity"):
        generate_study_pool(tmp_path, cfg, "ou", **options())


@pytest.mark.parametrize("allocations,validation,seed", [
    ((), (2, 2), 0), (((2, 0),), (2, 2), 0),
    (((2, 4), (4, 4)), (2, 2), 0), (((4, 2), (2, 4)), (2, 2), 0),
    (((2, 4), (4, 1)), (2, 2), 0), (((3, 2),), (2, 2), 0),
    (((2, 4),), (3, 2), 0), (((2, 4),), (2, 2), -1),
    (((2, 4),), (2, 2), True),
])
def test_invalid_study_plans_are_rejected(allocations, validation, seed):
    with pytest.raises(ValueError):
        study_pool_identity(config(), "dk", allocations=allocations,
                             validation=validation, seed=seed)
