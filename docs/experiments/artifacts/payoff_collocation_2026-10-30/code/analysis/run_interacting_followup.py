"""Run the preregistered interacting refinement and loss-balancing batch.

Seed workers write only their own checkpoints. Final references remain locked
until all workers have actually exited and all declared fits restore correctly.
The unscaled statistics candidate receives no standalone final reference bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from nn_dk.data import source_hash, write_json  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.interacting_data import load_interacting_bank  # noqa: E402
from nn_dk.interacting_study import (  # noqa: E402
    _fit_inventory,
    prepare_interacting_run,
    run_interacting_study,
)
from nn_dk.interacting_training import fit_interacting  # noqa: E402
from nn_dk.path_config import INTERACTING_CASES, PathConfig  # noqa: E402
from nn_dk.path_data import load_path_bank  # noqa: E402
from nn_dk.path_study import (  # noqa: E402
    _checkpoint,
    _inventory,
    prepare_path_run,
    run_path_study,
)
from nn_dk.path_training import fit_path  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

WORKFLOW = "interacting-path-refinement-and-balanced-moments-v1"
SCRIPT = Path(__file__).resolve()
EXIT_CHECKPOINTED = 75


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_design(path):
    body = _read(path)
    configs = {"paths": PathConfig(**body["paths"]),
               "unscaled": InteractingStudyConfig(**body["unscaled"]),
               "balanced": InteractingStudyConfig(**body["balanced"])}
    paths, unscaled, balanced = (configs[key] for key in ("paths", "unscaled", "balanced"))
    if (paths.cases != INTERACTING_CASES
            or paths.optimizer_seeds != unscaled.optimizer_seeds
            or paths.optimizer_seeds != balanced.optimizer_seeds
            or unscaled.loss_balancing != "none"
            or balanced.loss_balancing != "training-initial-rms"):
        raise ValueError("Require both interacting path cases and matched declared candidate optimizer seeds")
    ignored = {"name", "loss_balancing", "loss_scale_floor"}
    if ({key: value for key, value in unscaled.to_dict().items() if key not in ignored}
            != {key: value for key, value in balanced.to_dict().items() if key not in ignored}):
        raise ValueError("Statistics candidates must differ only in name and declared loss scaling")
    if ((paths.n_cells, paths.n_particles, paths.diffusion, paths.rho_stop, paths.horizon,
         paths.interaction_kappa, paths.interaction_a2, paths.query_times)
            != (balanced.n_cells, balanced.n_particles, balanced.diffusion, balanced.rho_stop,
                balanced.horizon, balanced.kappa, balanced.a2, balanced.query_times)):
        raise ValueError("Both solver families require identical physical targets and observation times")
    budget = body.get("budget_seconds", min(cfg.budget_seconds for cfg in configs.values()))
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 8*60*60:
        raise ValueError("Follow-up active budget must be positive and at most eight hours")
    return body, configs, float(budget)


def _identity(config_path, configs):
    return {"workflow": WORKFLOW, "source_hash": source_hash(), "script_sha256": _sha(SCRIPT),
            "configuration_sha256": _sha(config_path),
            "configuration_hashes": {key: cfg.config_hash for key, cfg in configs.items()},
            "expected_models": sum(cfg.expected_models for cfg in configs.values())}


def prepare_batch(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    body, configs, budget = load_design(config_path)
    identity = _identity(config_path, configs)
    frozen = output/"execution"/"frozen.json"
    snapshot = output/"study_config.json"
    if frozen.exists():
        if _read(frozen) != identity or not snapshot.is_file() or _sha(snapshot) != _sha(config_path):
            raise ValueError("Frozen follow-up source, runner or configuration changed; use a fresh output")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("An unrecognized nonempty output cannot become a new follow-up batch")
        output.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and _sha(snapshot) != _sha(config_path):
            raise ValueError("Existing follow-up configuration snapshot differs")
        snapshot.write_bytes(config_path.read_bytes())
        write_json(frozen, identity)
        write_json(output/"execution"/"budget.json", {"active_seconds": 0., "budget_seconds": budget,
                   "cost_scope": "Cumulative coordinator monotonic wall time; concurrent worker CPU recorded separately"})
    return body, configs, budget, identity


def _assert_frozen(config_path, output, expected):
    _, configs, _ = load_design(config_path)
    actual = _identity(config_path, configs)
    if (actual != expected or _read(Path(output)/"execution"/"frozen.json") != expected
            or _sha(Path(output)/"study_config.json") != expected["configuration_sha256"]):
        raise ValueError("Frozen follow-up source, runner or configuration changed")


def _remaining(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExhausted("Follow-up checkpointed at its active deadline; no fresh stage started")


@contextmanager
def _exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink()


def _generate(config_path, output, configs, identity, deadline):
    for study, cfg in configs.items():
        _remaining(deadline)
        _assert_frozen(config_path, output, identity)
        runner = run_path_study if study == "paths" else run_interacting_study
        runner(output/study, cfg, deadline=deadline, stage="generate")
    write_json(output/"execution"/"generated.json", {"identity": identity, "complete": True})


def _model_record(study, path, **slot):
    return {"study": study, **slot, "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": _sha(path), "metadata_path": str(path.with_suffix(".json").resolve()),
            "metadata_sha256": _sha(path.with_suffix(".json"))}


def fit_seed(config_path, output, configs, identity, seed, deadline):
    if type(seed) is not int or seed not in configs["paths"].optimizer_seeds:
        raise ValueError("Worker must exclusively own one declared optimizer seed")
    generated = output/"execution"/"generated.json"
    if not generated.is_file() or _read(generated) != {"identity": identity, "complete": True}:
        raise ValueError("Workers require every training and validation bank to be generated")
    _assert_frozen(config_path, output, identity)
    record_path = output/"execution"/f"seed_worker_{seed}.json"
    lock_path = output/"execution"/f".seed-worker-{seed}.lock"
    with _exclusive(lock_path):
        for study, cfg in configs.items():
            manifest = (prepare_path_run if study == "paths" else prepare_interacting_run)(output/study, cfg)
            if manifest.get("fit_lock") is not None:
                raise ValueError("Seed workers cannot run after final-reference unlock")
        started, cpu = time.monotonic(), time.process_time()
        packet = {"identity": identity, "seed": seed, "pid": os.getpid(), "threads": 1,
                  "started_unix": time.time(), "started_monotonic": started,
                  "final_references_read": False, "complete": False, "models": []}
        write_json(record_path, packet)
        try:
            cfg = configs["paths"]
            for case in cfg.cases:
                for dt in cfg.step_sizes:
                    _remaining(deadline)
                    _assert_frozen(config_path, output, identity)
                    print(f"[seed {seed}] paths {case} dt={dt:g}", flush=True)
                    path = _checkpoint(output/"paths", case, dt, seed)
                    train, validation = [load_path_bank(output/"paths", cfg, role, case, dt)
                                         for role in ("train", "validation")]
                    metadata = fit_path(cfg, case, dt, seed, path, train, validation, deadline=deadline)
                    if not path.with_suffix(".json").exists() or _read(path.with_suffix(".json")) != metadata:
                        write_json(path.with_suffix(".json"), metadata)
                    if not metadata["complete"]:
                        raise BudgetExhausted("Path seed worker checkpointed before allocation completed")
                    packet["models"].append(_model_record("paths", path, case=case, dt=dt, seed=seed))
                    write_json(record_path, packet)
            for study in ("unscaled", "balanced"):
                _remaining(deadline)
                _assert_frozen(config_path, output, identity)
                cfg = configs[study]
                print(f"[seed {seed}] statistics {study}", flush=True)
                train, validation = [load_interacting_bank(output/study, cfg, role)
                                     for role in ("train", "validation")]
                path = output/study/"checkpoints"/f"moments_seed{seed}.pt"
                metadata = fit_interacting(cfg, seed, path, train, validation, deadline=deadline)
                if not path.with_suffix(".json").exists() or _read(path.with_suffix(".json")) != metadata:
                    write_json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    raise BudgetExhausted("Statistics seed worker checkpointed before allocation completed")
                packet["models"].append(_model_record(study, path, seed=seed))
                write_json(record_path, packet)
            _assert_frozen(config_path, output, identity)
            packet["complete"] = True
            return packet
        finally:
            packet.update(finished_unix=time.time(), wall_seconds=time.monotonic()-started,
                          process_cpu_seconds=time.process_time()-cpu)
            write_json(record_path, packet)


def _pid_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _launch_workers(config_path, output, configs, identity, deadline):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    write_json(output/"execution"/"worker_deadline.json", {"identity": identity,
               "deadline_monotonic": deadline, "issued_unix": time.time()})
    workers, streams, exits = [], [], []
    try:
        for seed in configs["paths"].optimizer_seeds:
            _remaining(deadline)
            _assert_frozen(config_path, output, identity)
            stream = (output/"execution"/f"seed_worker_{seed}.log").open("a")
            streams.append(stream)
            command = [sys.executable, "-u", str(SCRIPT), "--config", str(config_path),
                       "--output", str(output), "--stage", "worker", "--seed", str(seed)]
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            workers.append((seed, process))
            write_json(output/"execution"/"worker_launch.json", {"identity": identity,
                       "workers": [{"seed": key, "pid": worker.pid} for key, worker in workers]})
        # Native optimizers checkpoint between atomic steps; wait for actual
        # process exit rather than killing a checkpoint write at the deadline.
        for seed, process in workers:
            code = process.wait()
            exits.append({"seed": seed, "pid": process.pid, "returncode": code})
        write_json(output/"execution"/"worker_exits.json", {"identity": identity, "workers": exits})
        if any(item["returncode"] != 0 for item in exits):
            raise BudgetExhausted("A worker exited incomplete; final references remain locked")
    finally:
        # Even a launch/configuration failure must not leave writers orphaned
        # behind a released coordinator lock. Their shared deadline is bounded.
        waited = {item["seed"] for item in exits}
        for seed, process in workers:
            if seed not in waited:
                exits.append({"seed": seed, "pid": process.pid, "returncode": process.wait()})
        if exits:
            write_json(output/"execution"/"worker_exits.json", {"identity": identity, "workers": exits})
        for stream in streams:
            stream.close()


def verify_complete_fits(config_path, output, configs, identity):
    """Require actual worker exit and every independently restorable model."""
    _assert_frozen(config_path, output, identity)
    records = []
    for seed in configs["paths"].optimizer_seeds:
        path = output/"execution"/f"seed_worker_{seed}.json"
        if (output/"execution"/f".seed-worker-{seed}.lock").exists() or not path.is_file():
            raise ValueError("Every declared worker must exit and release its ownership lock")
        worker = _read(path)
        if (worker.get("identity") != identity or worker.get("seed") != seed
                or worker.get("complete") is not True or _pid_running(worker["pid"])):
            raise ValueError("Every declared worker must complete and actually exit before reference unlock")
        records.extend(worker["models"])
    exit_path = output/"execution"/"worker_exits.json"
    if exit_path.exists():
        exits = _read(exit_path)
        expected = set(configs["paths"].optimizer_seeds)
        if (exits.get("identity") != identity
                or {item["seed"] for item in exits["workers"]} != expected
                or any(item["returncode"] != 0 for item in exits["workers"])):
            raise ValueError("Actual seed worker exit status prevents final-reference unlock")
    inventory = []
    for item in _inventory(output/"paths", configs["paths"])["models"]:
        inventory.append(_model_record("paths", Path(item["checkpoint_path"]),
                                      case=item["case"], dt=item["dt"], seed=item["seed"]))
    for study in ("unscaled", "balanced"):
        for item in _fit_inventory(output/study, configs[study]):
            inventory.append(_model_record(study, output/study/item["checkpoint_path"], seed=item["seed"]))
    if (len(inventory) != identity["expected_models"] or len(records) != len(inventory)
            or sorted(records, key=stable_hash) != sorted(inventory, key=stable_hash)):
        raise ValueError("Worker final checkpoint inventory does not match every restored declared fit")
    return {"identity": identity, "models": inventory, "expected_models": len(inventory)}


def _evaluate(config_path, output, configs, identity, deadline):
    _remaining(deadline)
    inventory = verify_complete_fits(config_path, output, configs, identity)
    lock_path = output/"execution"/"fit_lock.json"
    if lock_path.exists() and _read(lock_path) != inventory:
        raise ValueError("A follow-up model changed after the global fit lock")
    write_json(lock_path, inventory)
    # Replaying complete immutable fits creates the native candidate fit locks.
    # Both candidates lock before either solver sees a final reference.
    for study in ("unscaled", "balanced"):
        _remaining(deadline)
        _assert_frozen(config_path, output, identity)
        run_interacting_study(output/study, configs[study], deadline=deadline, stage="fit")
    _remaining(deadline)
    verify_complete_fits(config_path, output, configs, identity)
    run_path_study(output/"paths", configs["paths"], deadline=deadline, stage="evaluate")
    _remaining(deadline)
    _assert_frozen(config_path, output, identity)
    run_interacting_study(output/"balanced", configs["balanced"], deadline=deadline, stage="all")
    _assert_frozen(config_path, output, identity)
    verify_complete_fits(config_path, output, configs, identity)
    if any(_read(output/study/"manifest.json").get("status") != "complete"
           for study in ("paths", "balanced")):
        raise ValueError("Final follow-up workflows have not completed their native artifact locks")
    if (output/"unscaled"/"interacting_reference.h5").exists():
        raise ValueError("Unscaled candidate unexpectedly generated a standalone final reference")
    write_json(output/"execution"/"completed.json", {"identity": identity, "complete": True,
               "finished_unix": time.time(), "comparison_benchmark_run_separately": True,
               "continuous_time_accuracy_certified": False})


def run_batch(config_path, output, *, stage="all", seed=None):
    if stage not in ("all", "generate", "worker", "evaluate"):
        raise ValueError("Unsupported interacting follow-up stage")
    output, config_path = Path(output).resolve(), Path(config_path).resolve()
    if stage == "worker" and not (output/"execution"/"frozen.json").is_file():
        raise ValueError("Workers require an existing prepared follow-up batch")
    _, configs, budget, identity = prepare_batch(config_path, output)
    if stage == "worker":
        issued = output/"execution"/"worker_deadline.json"
        if issued.is_file():
            packet = _read(issued)
            if packet["identity"] != identity:
                raise ValueError("Worker deadline has incompatible frozen provenance")
            deadline = packet["deadline_monotonic"]
        else:
            deadline = time.monotonic()+budget-_read(output/"execution"/"budget.json")["active_seconds"]
        return fit_seed(config_path, output, configs, identity, seed, deadline)
    if seed is not None:
        raise ValueError("Only a seed worker may receive --seed")
    with _exclusive(output/"execution"/".coordinator.lock"):
        ledger = _read(output/"execution"/"budget.json")
        started, cpu, unix = time.monotonic(), time.process_time(), time.time()
        deadline = started+budget-ledger["active_seconds"]
        packet = {"identity": identity, "stage": stage, "started_unix": unix,
                  "pid": os.getpid(), "complete": False}
        try:
            _remaining(deadline)
            complete = output/"execution"/"completed.json"
            if complete.exists():
                if _read(complete).get("identity") != identity:
                    raise ValueError("Completed follow-up provenance changed")
                verify_complete_fits(config_path, output, configs, identity)
                packet["complete"] = True
                return packet
            locked = (output/"execution"/"fit_lock.json").is_file()
            if stage in ("all", "generate") and not locked:
                _generate(config_path, output, configs, identity, deadline)
            if stage == "all" and not locked:
                _launch_workers(output/"study_config.json", output, configs, identity, deadline)
            if stage in ("all", "evaluate"):
                _evaluate(config_path, output, configs, identity, deadline)
            packet["complete"] = True
            return packet
        except Exception as error:
            packet["reason"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            elapsed = time.monotonic()-started
            ledger["active_seconds"] += elapsed
            write_json(output/"execution"/"budget.json", ledger)
            packet.update(finished_unix=time.time(), wall_seconds=elapsed,
                          coordinator_process_cpu_seconds=time.process_time()-cpu)
            write_json(output/"execution"/f"coordinator_{time.time_ns()}.json", packet)
            write_json(output/"execution"/"latest.json", packet)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "generate", "worker", "evaluate"), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        result = run_batch(args.config, args.output, stage=args.stage, seed=args.seed)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except (BudgetExhausted, ValueError, OSError) as error:
        print(f"Follow-up preserved without a new reference unlock: {error}", file=sys.stderr, flush=True)
        return EXIT_CHECKPOINTED


if __name__ == "__main__":
    raise SystemExit(main())
