from dataclasses import replace

import numpy as np
import pytest

from nn_dk.config import ExperimentConfig, contexts_for, preset, smoke_config
from nn_dk.types import Context, QueryBank, TargetSpec, TrajectoryBatch, seeded_rng


def test_config_roundtrip_and_presets(tmp_path):
    import json

    for name in ("smoke", "pilot", "n2048", "resolution64", "long-horizon"):
        config = preset(name)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config.to_dict()))
        assert ExperimentConfig.load(path) == config
        assert ExperimentConfig.load(path).config_hash == config.config_hash
    for path in __import__("pathlib").Path("configs").glob("*.toml"):
        if path.stem.startswith("path_"):
            from nn_dk.path_config import PathConfig, path_preset

            assert PathConfig.load(path) == path_preset(path.stem.replace("_", "-"))
            continue
        if path.stem.startswith("generator_"):
            from nn_dk.generator_config import GeneratorStudyConfig, generator_preset

            assert GeneratorStudyConfig.load(path) == generator_preset(path.stem.replace("_", "-"))
            continue
        if path.stem in ("bsde_study_smoke", "bsde_anchor_study"):
            from nn_dk.bsde_study_config import BSDEStudyConfig, bsde_study_preset

            assert BSDEStudyConfig.load(path) == bsde_study_preset(path.stem.replace("_", "-"))
            continue
        if path.stem.startswith("bsde_"):
            from nn_dk.bsde_config import BSDEConfig, bsde_preset

            assert BSDEConfig.load(path) == bsde_preset(path.stem.replace("_", "-"))
            continue
        if path.stem.startswith("a_"):
            from nn_dk.convergence_config import ConvergenceConfig, convergence_preset

            assert ConvergenceConfig.load(path) == convergence_preset(path.stem.replace("_", "-"))
            continue
        assert ExperimentConfig.load(path) == preset(path.stem)


def test_config_rejects_non_nested_grids_and_odd_pairs():
    with pytest.raises(ValueError, match="nested"):
        replace(smoke_config(), lag=0.013)
    with pytest.raises(ValueError, match="Even"):
        replace(smoke_config(), train_replicates=3)
    with pytest.raises(ValueError, match="Queries"):
        replace(smoke_config(), query_times=(0.123,))


def test_independent_streams_context_groups_and_serialization():
    c = smoke_config()
    groups = [contexts_for(c, split) for split in ("train", "validation", "test")]
    identifiers = [ctx.context_id for group in groups for ctx in group]
    assert len(identifiers) == len(set(identifiers)) == 12
    assert groups[0] == contexts_for(c, "train")
    a = seeded_rng(10, "train", 1).normal(size=20)
    assert np.array_equal(a, seeded_rng(10, "train", 1).normal(size=20))
    assert not np.array_equal(a, seeded_rng(10, "test", 1).normal(size=20))
    assert Context(cos_coeffs=[0, 0, 0]).context_id == Context().context_id


def test_targets_queries_and_negative_mass_contracts():
    target = TargetSpec(8)
    with pytest.raises(ValueError, match="Incompatible"):
        target.require_compatible(TargetSpec(16))
    with pytest.raises(ValueError):
        TargetSpec(regime="U")
    times = np.array([0, 0.25])
    states = np.ones((2, 2, 8)) / 8
    batch = TrajectoryBatch(times, states, target)
    assert QueryBank((0.25,)).extract(batch).shape == (2, 4)
    with pytest.raises(ValueError, match="stored time"):
        QueryBank((0.5,)).extract(batch)
    states[0, -1, 0] = -0.1
    with pytest.raises(ValueError, match="marked invalid"):
        TrajectoryBatch(times, states, target)
