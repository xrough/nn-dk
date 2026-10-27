"""Prepare, fit per-seed, and lock a preregistered new-payoff comparison.

The current-state banks are generated once, independently of held-out references.
All twelve fresh statistical heads see literally the same train/validation
arrays. Historical bank-A weights are validated under their committed source;
only their hidden trunk is transferred, with a freshly initialized payoff head.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))


from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.frozen_models import load_frozen_model  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.interacting_data import (  # noqa: E402
    ensure_interacting_bank,
    load_interacting_bank,
)
from nn_dk.payoff_learning import (  # noqa: E402
    PAYOFF_MODEL_KIND,
    PayoffBank,
    fit_payoff,
    load_payoff_model,
)
from nn_dk.types import stable_hash  # noqa: E402

WORKFLOW = "new-terminal-and-killed-integral-payoff-generalization-v1"
SCRIPT = Path(__file__).resolve()
HISTORICAL_COMMIT = "26ac405"
SOURCE_FREEZE_COMMIT = "0599cf3"
SOURCE_FREEZE_HASH = "1a274fcf4d2b9c9b19f4605f514f38d8465807216a99419504f77d7796442df1"
INITIALIZATIONS = ("cold", "trunk-transfer")


def configuration():
    return InteractingStudyConfig(
        name="path-generalization-new-payoffs", seed=2026102613,
        n_cells=32, n_particles=2048, diffusion=1., kappa=.75, a2=.1,
        horizon=1/16, query_times=(1/64, 1/32, 1/16),
        optimizer_seeds=(0, 1, 2), width=32,
        state_step=2**-9, train_paths=128, validation_paths=64,
        state_fractions=(0., .25, .5, 1.), horizon_samples=4,
        adam_steps=1000, lbfgs_steps=30, lbfgs_max_iter=5,
        batch_size=128, learning_rate=1e-3, validation_every=20,
        loss_balancing="none")


def payoff_banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n")
    temporary.replace(path)


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Frozen new-payoff artifact changed: {path}")
    else:
        _json(path, value)


@contextmanager
def _exclusive(path):
    # Advisory locks are released by the OS even after a worker crash. The
    # persistent filename is not itself evidence of a live process.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid())+"\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _transfer_path(seed):
    return REPO/"runs"/"interacting-followup-2026-10-25"/"unscaled"/"checkpoints"/f"moments_seed{seed}.pt"


def _load_transfer(cfg, seed):
    path = _transfer_path(seed)
    model, record = load_frozen_model(path, HISTORICAL_COMMIT, kind="moments")
    physics = (model.spec.n_cells, model.spec.length, model.spec.rho_stop,
               model.context.n_particles, model.context.diffusion,
               model.context.kappa, model.context.a2, model.horizon, model.width)
    expected = (cfg.spec.n_cells, cfg.spec.length, cfg.spec.rho_stop,
                cfg.n_particles, cfg.diffusion, cfg.kappa, cfg.a2, cfg.horizon, cfg.width)
    if physics != expected or model.provenance["seed"] != seed:
        raise ValueError("Historical statistics trunk has incompatible physics, width, horizon or seed")
    metadata = path.with_suffix(".json")
    original_metadata = _read(metadata)
    if (original_metadata.get("identity_hash") != model.provenance["identity_hash"]
            or original_metadata.get("target_id") != model.target_id
            or original_metadata.get("seed") != seed or not original_metadata.get("complete")
            or not math.isfinite(original_metadata.get("training_seconds", -1))
            or original_metadata["training_seconds"] <= 0):
        raise ValueError("Historical transfer metadata is inconsistent with its strictly restored checkpoint")
    record = {**record, "metadata_path": str(metadata.resolve()),
              "metadata_sha256": _sha(metadata), "optimizer_seed": seed,
              "original_training_seconds": original_metadata["training_seconds"],
              "original_config_hash": original_metadata["config_hash"],
              "original_data_hash": original_metadata["data_hash"],
              "transferred": "network hidden layers 0,2,4 only; final head reset",
              "selected_state": "strict-original-loader-best-validation-state"}
    return {key: value.detach().clone() for key, value in model.state_dict().items()}, record


def _design(cfg, transfers):
    if source_hash() != SOURCE_FREEZE_HASH:
        raise ValueError("New-payoff experiment requires the declared complete source freeze")
    banks = payoff_banks(cfg)
    return {"workflow": WORKFLOW, "kind": PAYOFF_MODEL_KIND,
            "source_hash": source_hash(), "runner_sha256": _sha(SCRIPT),
            "source_freeze_commit": SOURCE_FREEZE_COMMIT,
            "config": cfg.to_dict(), "config_hash": cfg.config_hash,
            "payoff_banks": [bank.to_dict() for bank in banks],
            "payoff_bank_hashes": {bank.name: bank.bank_hash for bank in banks},
            "optimizer_seeds": list(cfg.optimizer_seeds),
            "initializations": list(INITIALIZATIONS),
            "expected_models": len(banks)*len(INITIALIZATIONS)*len(cfg.optimizer_seeds),
            "historical_transfers": transfers,
            "data": "one literal current-state/horizon pool shared by all banks, seeds and initializations",
            "split": "independent whole trajectory families; all derived samples remain grouped",
            "training": "known-generator residual; no future payoff labels",
            "reference_inputs_read": False,
            "final_reference_lock": "all twelve complete checkpoints and input artifacts strictly restored",
            "selection": "minimum-validation-residual; no selection using held-out final outcomes",
            "cost": "training incl validation and checkpointing; historical restoration recorded separately",
            "continuous_reference_bias": "unresolved"}


def _assert_frozen(directory, cfg):
    design = _read(directory/"preregistration.json")
    # Compare the current script/source/config and actual immutable historical
    # files without loading or rerunning any held-out reference calculation.
    transfers = design["historical_transfers"]
    for record in transfers:
        for key in ("checkpoint", "metadata"):
            if _sha(record[f"{key}_path"]) != record[f"{key}_sha256"]:
                raise ValueError("A frozen historical transfer input changed")
    if _design(cfg, transfers) != design or _read(directory/"config.json") != cfg.to_dict():
        raise ValueError("New-payoff source, runner, design or configuration drifted; use a fresh run")
    return design


def _data_identity(directory, cfg):
    records = {}
    arrays = {}
    for role in ("train", "validation"):
        bank = load_interacting_bank(directory, cfg, role)
        arrays[role] = bank
        records[role] = {"path": str((directory/f"interacting_{role}.h5").resolve()),
                         "sha256": _sha(directory/f"interacting_{role}.h5"),
                         "metadata_path": str((directory/f"interacting_{role}.json").resolve()),
                         "metadata_sha256": _sha(directory/f"interacting_{role}.json"),
                         "content_hash": bank["content_hash"],
                         "independent_families": len(set(bank["family_ids"])),
                         "collocation_rows": len(bank["states"]),
                         "family_ids_hash": stable_hash(list(bank["family_ids"]))}
    if set(arrays["train"]["family_ids"]) & set(arrays["validation"]["family_ids"]):
        raise ValueError("New-payoff training and validation families overlap")
    return records, arrays


def prepare(root, deadline=None):
    """Freeze design and transfer SHAs before generating any current-state data."""
    directory, cfg = Path(root).resolve()/"payoffs", configuration()
    with _exclusive(directory/"execution"/".prepare.lock"):
        if (directory/"preregistration.json").exists():
            design = _assert_frozen(directory, cfg)
        else:
            if any(path.name not in {"execution"} for path in directory.iterdir()):
                raise ValueError("Unrecognized payoff output cannot be adopted; choose a fresh root")
            transfers = [_load_transfer(cfg, seed)[1] for seed in cfg.optimizer_seeds]
            design = _design(cfg, transfers)
            _immutable(directory/"config.json", cfg.to_dict())
            _immutable(directory/"preregistration.json", design)
        for role in ("train", "validation"):
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("New-payoff data generation is resumable at the fixed deadline")
            _assert_frozen(directory, cfg)
            packet = ensure_interacting_bank(directory, cfg, role, deadline=deadline)
            if not packet["complete"]:
                raise BudgetExhausted("New-payoff current-state bank checkpointed before completion")
        records, _ = _data_identity(directory, cfg)
        _immutable(directory/"data_lock.json", {"workflow": WORKFLOW,
                   "source_hash": design["source_hash"], "config_hash": cfg.config_hash,
                   "preregistration_sha256": _sha(directory/"preregistration.json"),
                   "banks": records, "all_models_share_literal_pool": True,
                   "train_validation_families_disjoint": True})
    return directory


def _prepared(root):
    directory, cfg = Path(root).resolve()/"payoffs", configuration()
    design = _assert_frozen(directory, cfg)
    data = _read(directory/"data_lock.json")
    records, arrays = _data_identity(directory, cfg)
    if data != {"workflow": WORKFLOW, "source_hash": source_hash(), "config_hash": cfg.config_hash,
                "preregistration_sha256": _sha(directory/"preregistration.json"),
                "banks": records, "all_models_share_literal_pool": True,
                "train_validation_families_disjoint": True}:
        raise ValueError("Shared new-payoff training/validation inputs changed")
    return directory, cfg, design, arrays


def _checkpoint(directory, bank, initialization, seed):
    return directory/"checkpoints"/bank.name/initialization/f"seed{seed}.pt"


def _model_record(path, cfg, bank, initialization, seed):
    model = load_payoff_model(path)
    if (model.config.config_hash != cfg.config_hash or model.bank.bank_hash != bank.bank_hash
            or model.provenance["seed"] != seed
            or (model.provenance["transfer_trunk_hash"] is None) != (initialization == "cold")):
        raise ValueError("Payoff fitted slot has incompatible configuration/bank/initialization")
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "checkpoint_path": str(path.resolve()), "checkpoint_sha256": _sha(path),
            "metadata_path": str(path.with_suffix(".json").resolve()),
            "metadata_sha256": _sha(path.with_suffix(".json")),
            "complete": True, "source_hash": model.provenance["source_hash"],
            "config_hash": cfg.config_hash, "target_id": model.target_id,
            "bank": bank.name, "bank_hash": bank.bank_hash,
            "seed": seed, "init": initialization,
            "transfer_trunk_hash": model.provenance["transfer_trunk_hash"]}


def fit_seed(root, seed, deadline=None):
    directory, cfg, design, arrays = _prepared(root)
    if type(seed) is not int or seed not in cfg.optimizer_seeds:
        raise ValueError("Use one predeclared optimizer seed: 0, 1 or 2")
    if (directory/"fit_lock.json").exists():
        raise ValueError("New-payoff final fit lock already exists; fitting is closed")
    with _exclusive(directory/"execution"/f".seed{seed}.lock"):
        if (directory/"fit_lock.json").exists():
            raise ValueError("New-payoff final fit lock already exists; fitting is closed")
        tick, cpu = time.monotonic(), time.process_time()
        state, transfer = _load_transfer(cfg, seed)
        restoration = time.monotonic()-tick
        if transfer != next(row for row in design["historical_transfers"] if row["optimizer_seed"] == seed):
            raise ValueError("Historical trunk derivation changed since preregistration")
        output = directory/"execution"/f"worker_seed{seed}.json"
        attempts = _read(output).get("attempts", []) if output.exists() else []
        attempt = {"pid": os.getpid(), "started_unix": time.time(), "complete": False}
        packet = {"workflow": WORKFLOW, "seed": seed, "source_hash": source_hash(),
                  "config_hash": cfg.config_hash, "pid": os.getpid(), "complete": False,
                  "transfer_restoration_seconds": restoration, "models": [],
                  "attempts": [*attempts, attempt], "held_out_reference_inputs_read": False}
        _json(output, packet)
        try:
            for bank in payoff_banks(cfg):
                for initialization in INITIALIZATIONS:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise BudgetExhausted("Payoff seed worker reached its resumable deadline")
                    _assert_frozen(directory, cfg)
                    path = _checkpoint(directory, bank, initialization, seed)
                    print(f"[seed {seed}] {bank.name} {initialization}", flush=True)
                    metadata = fit_payoff(cfg, bank, seed, path, arrays["train"], arrays["validation"],
                                          transfer_state=state if initialization == "trunk-transfer" else None,
                                          deadline=deadline)
                    _json(path.with_suffix(".json"), metadata)
                    if not metadata["complete"]:
                        raise BudgetExhausted("Payoff checkpoint saved before optimizer allocation completed")
                    packet["models"].append(_model_record(path, cfg, bank, initialization, seed))
                    _json(output, packet)
            packet["complete"] = True
        finally:
            attempt.update(complete=packet["complete"], finished_unix=time.time(),
                           wall_seconds=time.monotonic()-tick, cpu_seconds=time.process_time()-cpu)
            packet.update(wall_seconds=attempt["wall_seconds"], cpu_seconds=attempt["cpu_seconds"])
            _json(output, packet)
    return packet


def lock(root):
    """Strictly restore every declared fit and freeze their exact input/weight files."""
    directory, cfg, design, _ = _prepared(root)
    with _exclusive(directory/"execution"/".fit-lock.lock"):
        models = []
        # Holding every worker lock prevents publication while a fit is being
        # written or resumed. Files/records alone never prove a worker exited.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for seed in cfg.optimizer_seeds:
                stack.enter_context(_exclusive(directory/"execution"/f".seed{seed}.lock"))
                worker = _read(directory/"execution"/f"worker_seed{seed}.json")
                if (not worker["complete"] or worker["source_hash"] != source_hash()
                        or worker["config_hash"] != cfg.config_hash):
                    raise ValueError("Every predeclared seed worker must complete before fit-lock publication")
            for bank in payoff_banks(cfg):
                for initialization in INITIALIZATIONS:
                    for seed in cfg.optimizer_seeds:
                        path = _checkpoint(directory, bank, initialization, seed)
                        models.append(_model_record(path, cfg, bank, initialization, seed))
            if len(models) != design["expected_models"]:
                raise ValueError("Incomplete or duplicated payoff fit inventory")
            result = {"workflow": WORKFLOW, "kind": PAYOFF_MODEL_KIND, "complete": True,
                      "source_hash": source_hash(), "config_hash": cfg.config_hash,
                      "runner_sha256": _sha(SCRIPT), "expected_models": len(models), "models": models,
                      "preregistration_path": str((directory/"preregistration.json").resolve()),
                      "preregistration_sha256": _sha(directory/"preregistration.json"),
                      "data_lock_path": str((directory/"data_lock.json").resolve()),
                      "data_lock_sha256": _sha(directory/"data_lock.json"),
                      "input_banks": _read(directory/"data_lock.json")["banks"],
                      "all_models_shared_literal_pool": True,
                      "training_families_independent_of_held_out_references": True,
                      "seeds_pooled": False}
            _immutable(directory/"fit_lock.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("prepare", "fit", "lock"))
    parser.add_argument("--stage", dest="stage_option", choices=("prepare", "fit", "lock"))
    parser.add_argument("--root", type=Path, default=REPO/"runs"/"path-generalization-2026-10-26")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args(argv)
    stage = args.stage_option or args.stage
    if stage is None or (args.stage and args.stage_option and args.stage != args.stage_option):
        parser.error("Specify one consistent stage: prepare, fit or lock")
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 8*60*60:
        parser.error("Budget must be positive and at most eight hours")
    if (stage == "fit") != (args.seed is not None):
        parser.error("Only fit requires --seed 0, 1 or 2")
    deadline = time.monotonic()+args.budget_seconds
    try:
        if stage == "prepare":
            directory = prepare(args.root, deadline)
            print(f"Prepared shared payoff pools: {directory}", flush=True)
        elif stage == "fit":
            result = fit_seed(args.root, args.seed, deadline)
            print(f"Seed {args.seed} completed {len(result['models'])} new-payoff fits", flush=True)
        else:
            result = lock(args.root)
            print(f"Locked {result['expected_models']} complete new-payoff fits", flush=True)
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
