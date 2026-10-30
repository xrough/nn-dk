"""Independent deployment preparation and resampling do not reuse test initials."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_regimes_deployment", Path(__file__).parents[1]/"analysis/evaluate_payoff_regimes_deployment.py")
deployment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deployment)


def _preparation_root(tmp_path, monkeypatch):
    root = tmp_path/"study"
    plan = SimpleNamespace(plan_hash="frozen-physical-plan", optimizer_seeds=(0, 1, 2))
    cfg = SimpleNamespace(cases=("uniform", "nonuniform"), paths_per_group=8)
    monkeypatch.setattr(deployment.runner, "load_plan", lambda root: plan)
    monkeypatch.setattr(deployment.runner, "case_names", lambda study: ("regime",))
    monkeypatch.setattr(deployment.runner, "reference_configuration", lambda case, study: cfg)
    for path in (root/"plan.json", root/"payoffs/preregistration.json", root/"comparison/preregistration.json"):
        deployment.runner._json(path, {"frozen": True})
    return root


def test_preparation_writes_only_family_addresses_without_rng_or_reference_draws(tmp_path, monkeypatch):
    root = _preparation_root(tmp_path, monkeypatch)

    def prohibited(*args, **kwargs):
        raise AssertionError("No random initial or reference draw during preparation")

    monkeypatch.setattr(deployment, "seeded_rng", prohibited)
    monkeypatch.setattr(deployment.runner, "load_reference", prohibited)
    plan = deployment.prepare(root)
    assert not plan["initials_drawn_during_preparation"]
    assert len(plan["pools"]) == 2 and plan["expected_records"] == 8
    assert not (root/"deployment/initials").exists()
    assert not (root/"reference").exists()
    with np.load(root/"deployment/planned_initial_families.npz") as arrays:
        assert set(arrays.files) == {p["key"] for p in plan["pools"]}
        assert len(set(np.concatenate(list(arrays.values())))) == 16
    assert deployment.prepare(root) == plan


def test_preparation_rejects_post_reference_registration_and_changed_family_inventory(tmp_path, monkeypatch):
    root = _preparation_root(tmp_path, monkeypatch)
    reference = root/"reference/regime/reference.h5"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference creation already started")
    with pytest.raises(ValueError, match="before final reference"):
        deployment.prepare(root)
    reference.unlink()
    deployment.prepare(root)
    path = root/"deployment/planned_initial_families.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, replaced=np.array(["different family"]))
    with pytest.raises(ValueError, match="inventory changed"):
        deployment.prepare(root)


def test_every_new_initial_and_bootstrap_address_and_family_is_distinct():
    families, initial_seeds, bootstrap_seeds = set(), set(), set()
    for regime in ("T0p25_k0p75", "T0p25_k3", "T0p25_k6", "T1_k0p75", "T1_k3", "T1_k6"):
        for profile in ("uniform", "nonuniform"):
            seed = deployment.pool_address(regime, profile)
            bootstrap = deployment.initial_bootstrap_address(regime, profile)
            new = set(deployment.planned_families(regime, profile, 16))
            assert not families & new
            assert seed not in initial_seeds and bootstrap not in bootstrap_seeds
            assert seed != bootstrap
            families.update(new)
            initial_seeds.add(seed)
            bootstrap_seeds.add(bootstrap)
    assert len(families) == 192 and len(initial_seeds) == len(bootstrap_seeds) == 12


def test_reference_bootstrap_is_canonical_but_initial_bootstrap_is_independent():
    regime, profile = "T1_k6", "nonuniform"
    reference, initial = deployment.independent_bootstraps(regime, profile, 64, 64)
    address = int(deployment.stable_hash([deployment.original.SEED, regime, profile+"__ensemble__0"])[:12], 16)
    canonical = deployment.bootstrap_counts(64, address, deployment.BOOTSTRAPS)
    np.testing.assert_array_equal(reference, canonical)
    assert reference.shape == initial.shape == (128, 64)
    assert not np.array_equal(reference, initial)
    for counts in (reference, initial):
        np.testing.assert_array_equal(counts.sum(-1), 64)


def test_pool_initials_are_reproducible_and_do_not_use_reference_histograms(tmp_path, monkeypatch):
    cfg = SimpleNamespace(n_particles=32)
    monkeypatch.setattr(deployment.runner, "profile_probabilities",
                        lambda cfg, profile: (np.array([.1, .2, .3, .4]), {"known_profile": True}))
    pool = {"regime": "r", "profile": "uniform", "paths": 32,
            "initial_seed": deployment.pool_address("r", "uniform"),
            "family_ids_array_sha256": deployment.array_sha(deployment.planned_families("r", "uniform", 32))}
    arrays, packet = deployment._initial_pool(tmp_path, pool, cfg, "fixed-plan", "all-fits-locked")
    rng = deployment.seeded_rng(pool["initial_seed"], deployment.KIND, "multinomial-initials")
    expected = rng.multinomial(32, [.1, .2, .3, .4], size=32)/32
    np.testing.assert_array_equal(arrays["initial"], expected)
    assert packet["initials_drawn_after_complete_fit_lock"]
    again, restored = deployment._initial_pool(tmp_path, pool, cfg, "fixed-plan", "all-fits-locked")
    np.testing.assert_array_equal(again["initial"], expected)
    assert packet == restored
    with pytest.raises(ValueError, match="changed"):
        deployment._initial_pool(tmp_path, pool, cfg, "changed-plan", "all-fits-locked")


def test_missing_fit_lock_blocks_initial_pool_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, "prepare", lambda root: {})
    monkeypatch.setattr(deployment.runner, "load_plan", lambda root: object())

    def reject(root):
        raise ValueError("All fits incomplete")

    monkeypatch.setattr(deployment.runner, "verify_fit_lock", reject)
    with pytest.raises(ValueError, match="incomplete"):
        deployment._verified(tmp_path)
    assert not (tmp_path/"deployment/initials").exists()
