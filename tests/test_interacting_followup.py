"""Reference gates and process ownership for the interacting follow-up batch."""
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nn_dk.data import write_json
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.path_config import INTERACTING_CASES, PathConfig

MODULE = Path(__file__).resolve().parents[1]/"analysis"/"run_interacting_followup.py"
SPEC = importlib.util.spec_from_file_location("interacting_followup_analysis", MODULE)
followup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(followup)


@pytest.fixture
def design(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "source_hash", lambda: "frozen-test-source")
    paths = replace(PathConfig(), cases=INTERACTING_CASES,
                    step_sizes=(2**-8, 2**-9, 2**-10),
                    query_times=(1/64, 1/32, 1/16), optimizer_seeds=(0, 1, 2))
    unscaled = replace(InteractingStudyConfig(), optimizer_seeds=(0, 1, 2))
    balanced = replace(unscaled, name="balanced-test", loss_balancing="training-initial-rms")
    config = tmp_path/"config.json"
    write_json(config, {"paths": paths.to_dict(), "unscaled": unscaled.to_dict(),
                        "balanced": balanced.to_dict(), "budget_seconds": 28800.})
    output = tmp_path/"run"
    _, configs, _, identity = followup.prepare_batch(config, output)
    return config, output, configs, identity


def _inventories(design, monkeypatch):
    config, output, configs, identity = design
    path_rows, stat_rows = [], {"unscaled": [], "balanced": []}
    by_seed = {seed: [] for seed in configs["paths"].optimizer_seeds}
    for seed in by_seed:
        for case in configs["paths"].cases:
            for dt in configs["paths"].step_sizes:
                checkpoint = followup._checkpoint(output/"paths", case, dt, seed)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(f"path-{case}-{dt}-{seed}")
                write_json(checkpoint.with_suffix(".json"), {"complete": True})
                path_rows.append({"checkpoint_path": str(checkpoint), "case": case,
                                  "dt": dt, "seed": seed})
                by_seed[seed].append(followup._model_record("paths", checkpoint, case=case, dt=dt, seed=seed))
        for study in stat_rows:
            checkpoint = output/study/"checkpoints"/f"moments_seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(f"{study}-{seed}")
            write_json(checkpoint.with_suffix(".json"), {"complete": True})
            stat_rows[study].append({"checkpoint_path": str(checkpoint.relative_to(output/study)), "seed": seed})
            by_seed[seed].append(followup._model_record(study, checkpoint, seed=seed))
        write_json(output/"execution"/f"seed_worker_{seed}.json", {"identity": identity,
                   "seed": seed, "pid": 999000+seed, "complete": True, "models": by_seed[seed]})
    monkeypatch.setattr(followup, "_pid_running", lambda pid: False)
    monkeypatch.setattr(followup, "_inventory", lambda directory, cfg: {"models": path_rows})
    monkeypatch.setattr(followup, "_fit_inventory", lambda directory, cfg: stat_rows[Path(directory).name])
    return by_seed


def test_generate_opens_only_training_workflows(design, monkeypatch):
    config, output, _, _ = design
    calls = []
    monkeypatch.setattr(followup, "run_path_study", lambda directory, cfg, **kwargs:
                        calls.append((Path(directory).name, kwargs["stage"])))
    monkeypatch.setattr(followup, "run_interacting_study", lambda directory, cfg, **kwargs:
                        calls.append((Path(directory).name, kwargs["stage"])))
    result = followup.run_batch(config, output, stage="generate")
    assert result["complete"]
    assert calls == [("paths", "generate"), ("unscaled", "generate"), ("balanced", "generate")]
    assert not (output/"execution"/"fit_lock.json").exists()
    assert not (output/"execution"/"completed.json").exists()


@pytest.mark.parametrize("corruption", ["config", "source", "script", "snapshot"])
def test_frozen_changes_reject_before_work(design, monkeypatch, corruption):
    config, output, _, _ = design
    if corruption == "config":
        config.write_text(config.read_text()+"\n")
    elif corruption == "source":
        monkeypatch.setattr(followup, "source_hash", lambda: "mutated-source")
    elif corruption == "script":
        mutated = output/"fake-script.py"
        mutated.write_text("# changed orchestrator\n")
        monkeypatch.setattr(followup, "SCRIPT", mutated)
    else:
        (output/"study_config.json").write_text("{}")
    calls = []
    monkeypatch.setattr(followup, "run_path_study", lambda *args, **kwargs: calls.append("work"))
    with pytest.raises(ValueError, match="Frozen"):
        followup.run_batch(config, output, stage="generate")
    assert calls == []
    assert not (output/"execution"/"fit_lock.json").exists()


def test_global_inventory_of_twenty_four_fits_precedes_any_final_reference(design, monkeypatch):
    config, output, configs, identity = design
    _inventories(design, monkeypatch)
    assert identity["expected_models"] == 24
    calls = []

    def native(directory, cfg, *, deadline, stage):
        study = Path(directory).name
        lock = followup._read(output/"execution"/"fit_lock.json")
        assert lock["expected_models"] == 24
        assert len(lock["models"]) == 24
        if stage in ("evaluate", "all"):
            assert calls[:2] == [("unscaled", "fit"), ("balanced", "fit")]
            write_json(Path(directory)/"manifest.json", {"status": "complete"})
        calls.append((study, stage))

    monkeypatch.setattr(followup, "run_interacting_study", native)
    monkeypatch.setattr(followup, "run_path_study", native)
    followup._evaluate(config, output, configs, identity, time.monotonic()+60)
    assert calls == [("unscaled", "fit"), ("balanced", "fit"), ("paths", "evaluate"), ("balanced", "all")]
    assert followup._read(output/"execution"/"completed.json")["complete"]
    assert not (output/"unscaled"/"interacting_reference.h5").exists()


@pytest.mark.parametrize("failure", ["missing-fit", "live-worker", "ownership-lock", "bad-exit", "changed-fit"])
def test_incomplete_or_running_workers_cannot_unlock_references(design, monkeypatch, failure):
    config, output, configs, identity = design
    by_seed = _inventories(design, monkeypatch)
    if failure == "missing-fit":
        path = output/"execution"/"seed_worker_0.json"
        packet = followup._read(path)
        packet["models"].pop()
        write_json(path, packet)
    elif failure == "live-worker":
        monkeypatch.setattr(followup, "_pid_running", lambda pid: pid == 999000)
    elif failure == "ownership-lock":
        (output/"execution"/".seed-worker-0.lock").write_text("999000")
    elif failure == "bad-exit":
        write_json(output/"execution"/"worker_exits.json", {"identity": identity,
                   "workers": [{"seed": seed, "returncode": 75 if seed == 0 else 0}
                               for seed in configs["paths"].optimizer_seeds]})
    else:
        Path(by_seed[0][0]["checkpoint_path"]).write_text("corrupted model")
    calls = []
    monkeypatch.setattr(followup, "run_interacting_study", lambda *args, **kwargs: calls.append("reference"))
    monkeypatch.setattr(followup, "run_path_study", lambda *args, **kwargs: calls.append("reference"))
    with pytest.raises(ValueError):
        followup._evaluate(config, output, configs, identity, time.monotonic()+60)
    assert calls == []
    assert not (output/"execution"/"fit_lock.json").exists()


def test_worker_declared_seed_ownership_and_only_training_banks(design, monkeypatch):
    config, output, configs, identity = design
    write_json(output/"execution"/"generated.json", {"identity": identity, "complete": True})
    monkeypatch.setattr(followup, "prepare_path_run", lambda *args: {})
    monkeypatch.setattr(followup, "prepare_interacting_run", lambda *args: {})
    banks, slots = [], []

    def path_bank(directory, cfg, role, case, dt):
        banks.append(role)
        return {"role": role}

    def moment_bank(directory, cfg, role):
        banks.append(role)
        return {"role": role}

    def path_fit(cfg, case, dt, seed, path, train, validation, deadline):
        assert seed == 1 and (train["role"], validation["role"]) == ("train", "validation")
        slots.append(("paths", case, dt, seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        return {"complete": True, "seed": seed}

    def moment_fit(cfg, seed, path, train, validation, deadline):
        assert seed == 1
        slots.append(("moments", cfg.loss_balancing, seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        return {"complete": True, "seed": seed}

    monkeypatch.setattr(followup, "load_path_bank", path_bank)
    monkeypatch.setattr(followup, "load_interacting_bank", moment_bank)
    monkeypatch.setattr(followup, "fit_path", path_fit)
    monkeypatch.setattr(followup, "fit_interacting", moment_fit)
    packet = followup.fit_seed(config, output, configs, identity, 1, time.monotonic()+60)
    assert packet["complete"] and len(packet["models"]) == 8
    assert len(slots) == 8 and set(banks) == {"train", "validation"}
    assert packet["final_references_read"] is False
    lock = output/"execution"/".seed-worker-1.lock"
    assert not lock.exists()
    lock.write_text("another worker")
    with pytest.raises(FileExistsError):
        followup.fit_seed(config, output, configs, identity, 1, time.monotonic()+60)
    assert lock.read_text() == "another worker"
    with pytest.raises(ValueError, match="declared"):
        followup.fit_seed(config, output, configs, identity, 99, time.monotonic()+60)


def test_partial_launch_waits_for_launched_process_before_releasing_writers(design, monkeypatch):
    config, output, configs, identity = design
    events = []
    worker = SimpleNamespace(pid=12345, wait=lambda: events.append("actual-exit") or 0)

    def launch(*args, **kwargs):
        if events:
            raise OSError("launch failed")
        events.append("launched")
        return worker

    monkeypatch.setattr(followup.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="launch failed"):
        followup._launch_workers(config, output, configs, identity, time.monotonic()+60)
    assert events == ["launched", "actual-exit"]
    exits = followup._read(output/"execution"/"worker_exits.json")
    assert exits["workers"] == [{"seed": 0, "pid": 12345, "returncode": 0}]


def test_locked_evaluation_resume_does_not_restart_generation_or_fitting(design, monkeypatch):
    config, output, _, _ = design
    write_json(output/"execution"/"fit_lock.json", {"locked": True})
    calls = []
    monkeypatch.setattr(followup, "_generate", lambda *args: pytest.fail("Regenerated locked inputs"))
    monkeypatch.setattr(followup, "_launch_workers", lambda *args: pytest.fail("Restarted locked fitting"))
    monkeypatch.setattr(followup, "_evaluate", lambda *args: calls.append("evaluation-resumed"))
    assert followup.run_batch(config, output, stage="all")["complete"]
    assert calls == ["evaluation-resumed"]


def test_candidate_comparison_rejects_hidden_data_or_physics_changes(design):
    config, _, _, _ = design
    body = json.loads(config.read_text())
    body["balanced"]["seed"] += 1
    write_json(config, body)
    with pytest.raises(ValueError, match="differ only"):
        followup.load_design(config)
