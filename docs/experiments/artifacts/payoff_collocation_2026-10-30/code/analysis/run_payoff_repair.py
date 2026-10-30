"""Locked training and fresh references for the payoff formulation repair.

The original raw-loss head, its balanced-loss ablation and the staged repaired
head consume identical collocation arrays. Every declared fit is retained; no
final-reference result chooses a formulation, seed or training allocation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from analysis.path_generalization_reference import ReferenceConfig  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.interacting_data import (  # noqa: E402
    collocation_data_id,
    ensure_interacting_bank,
    load_interacting_bank,
)
from nn_dk.payoff_learning import PayoffBank, fit_payoff, load_payoff_model  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

WORKFLOW = "independent-terminal-integral-payoff-formulation-repair-v1"
SCRIPT = Path(__file__).resolve()
VARIANTS = ("v1-none", "v1-balanced", "repaired")
EXPECTED_MODELS = 10
HELPERS = ("evaluate_payoff_repair.py", "path_generalization_reference.py",
           "evaluate_path_generalization.py", "benchmark_payoff_generalization.py",
           "native_gaussian_baseline.py", "probe_path_workloads.py")


def configuration(variant="repaired"):
    """Fixed optimizer/data allocation; repaired fitting has two such stages."""
    if variant not in VARIANTS:
        raise ValueError("Unknown payoff repair variant")
    return InteractingStudyConfig(
        name="payoff-formulation-repair", seed=2026102713,
        n_cells=32, n_particles=2048, diffusion=1., kappa=.75, a2=.1,
        horizon=1 / 16, query_times=(1 / 64, 1 / 32, 1 / 16),
        optimizer_seeds=(0, 1, 2), width=32,
        state_step=2**-9, train_paths=128, validation_paths=64,
        state_fractions=(0., .25, .5, 1.), horizon_samples=4,
        adam_steps=1000, lbfgs_steps=30, lbfgs_max_iter=5,
        batch_size=128, learning_rate=1e-3, validation_every=20,
        loss_balancing="training-initial-rms" if variant == "v1-balanced" else "none")


def repaired_configuration():
    from nn_dk.payoff_repair import RepairConfig
    return RepairConfig(study=configuration(), quadrature_nodes=8)


def reference_configuration():
    return ReferenceConfig(seed=2026102711, expected_models=EXPECTED_MODELS)


def payoff_banks(cfg=None):
    cfg = configuration() if cfg is None else cfg
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def slots():
    """All ten preregistered slots; the four v1 ablations use seed zero."""
    return [{"bank": bank.name, "variant": variant, "seed": seed,
             "name": f"{bank.name}_{variant}_seed{seed}"}
            for bank in payoff_banks() for variant in VARIANTS
            for seed in ((0, 1, 2) if variant == "repaired" else (0,))]


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _immutable(path, value):
    if Path(path).exists():
        if _read(path) != value:
            raise ValueError(f"Frozen payoff-repair artifact changed: {path}")
    else:
        _json(path, value)


@contextmanager
def _exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _design():
    cfg = configuration()
    return {
        "workflow": WORKFLOW, "source_hash": source_hash(),
        "runner_sha256": _sha(SCRIPT),
        "helpers": {name: _sha(SCRIPT.with_name(name)) for name in HELPERS},
        "config": cfg.to_dict(), "config_hash": cfg.config_hash,
        "variant_configs": {variant: configuration(variant).to_dict() for variant in VARIANTS},
        "repaired_config": repaired_configuration().to_dict(),
        "reference_config": json.loads(json.dumps(asdict(reference_configuration()))),
        "reference_config_hash": reference_configuration().config_hash,
        "payoff_banks": {bank.name: bank.to_dict() for bank in payoff_banks(cfg)},
        "slots": slots(), "expected_models": EXPECTED_MODELS,
        "data": "one literal current-state/horizon pool shared by all ten fits",
        "split": "whole independent trajectory families; all derived samples grouped",
        "training": "known generator residuals only; no future-payoff labels",
        "repaired_training": "mean stage then covariance stage with frozen mean; each has the declared optimizer allocation",
        "ablations": "raw v1 and training-initial-RMS v1, both banks, seed zero; all retained",
        "selection": "within-model independent validation only; no final-test-based selection or refitting",
        "final_reference_inputs_read": False,
        "final_reference_lock": "all ten complete checkpoints strictly restored before outcomes or anchors",
        "historical_test_reuse": False,
        "cost": "all ablation and seed costs retained; training and inference separately recorded",
        "continuous_reference_bias": "unresolved"}


def _assert_frozen(directory):
    design = _read(Path(directory) / "preregistration.json")
    if design != _design() or _read(Path(directory) / "config.json") != configuration().to_dict():
        raise ValueError("Payoff-repair source, helper, runner or configuration changed; use a fresh run")
    return design


def _data_identity(directory):
    cfg, records, arrays = configuration(), {}, {}
    for role in ("train", "validation"):
        bank = load_interacting_bank(directory, cfg, role)
        arrays[role] = bank
        path = Path(directory) / f"interacting_{role}.h5"
        metadata = path.with_suffix(".json")
        records[role] = {
            "path": str(path.resolve()), "sha256": _sha(path),
            "metadata_path": str(metadata.resolve()), "metadata_sha256": _sha(metadata),
            "content_hash": bank["content_hash"],
            "independent_families": len(set(bank["family_ids"])),
            "collocation_rows": len(bank["states"]),
            "family_ids_hash": stable_hash(list(bank["family_ids"]))}
    if set(arrays["train"]["family_ids"]) & set(arrays["validation"]["family_ids"]):
        raise ValueError("Payoff-repair training and validation families overlap")
    return records, arrays


def _data_lock(directory, records):
    return {"workflow": WORKFLOW, "source_hash": source_hash(),
            "config_hash": configuration().config_hash,
            "preregistration_sha256": _sha(Path(directory) / "preregistration.json"),
            "banks": records, "all_models_share_literal_pool": True,
            "train_validation_families_disjoint": True,
            "variant_rebinding": "optimizer-only metadata changes after physical collocation-data identity check"}


def _reference_backend():
    """Isolated original backend with a current, strict two-kind loader adapter.

    Only this private module's loader is replaced; the historical reference
    module and all committed experiments keep their own unchanged globals.
    """
    path = SCRIPT.with_name("path_generalization_reference.py")
    name = "_nn_dk_payoff_repair_reference_backend"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(root, deadline=None):
    directory = Path(root).resolve() / "payoffs"
    with _exclusive(directory / "execution" / ".prepare.lock"):
        if (directory / "preregistration.json").exists():
            _assert_frozen(directory)
        else:
            if any(path.name != "execution" for path in directory.iterdir()):
                raise ValueError("Unrecognized payoff-repair outputs; choose a fresh root")
            design = _design()
            _immutable(directory / "config.json", configuration().to_dict())
            _immutable(directory / "preregistration.json", design)
        for role in ("train", "validation"):
            _assert_frozen(directory)
            if deadline is not None and time.monotonic() >= deadline:
                raise BudgetExhausted("Payoff-repair data checkpointed at its deadline")
            packet = ensure_interacting_bank(directory, configuration(), role, deadline=deadline)
            if not packet["complete"]:
                raise BudgetExhausted("Payoff-repair shared collocation generation is resumable")
        records, _ = _data_identity(directory)
        _immutable(directory / "data_lock.json", _data_lock(directory, records))
        _reference_backend().prepare_reference(Path(root).resolve() / "reference",
                                                reference_configuration())
    return directory


def _prepared(root):
    directory = Path(root).resolve() / "payoffs"
    design = _assert_frozen(directory)
    records, arrays = _data_identity(directory)
    if _read(directory / "data_lock.json") != _data_lock(directory, records):
        raise ValueError("Shared payoff-repair training/validation artifacts changed")
    return directory, design, arrays


def _rebound(bank, cfg):
    original = configuration()
    if (bank["config_hash"] != original.config_hash
            or bank["source_hash"] != source_hash()
            or collocation_data_id(original) != collocation_data_id(cfg)):
        raise ValueError("Variant does not describe the identical physical collocation pool")
    return {**bank, "config_hash": cfg.config_hash,
            "shared_pool_original_config_hash": original.config_hash,
            "shared_pool_physical_data_id": collocation_data_id(original)}


def _checkpoint(directory, slot):
    return Path(directory) / "checkpoints" / slot["bank"] / slot["variant"] / f"seed{slot['seed']}.pt"


def load_declared_model(item):
    """Strictly restore a named current-source v1 or repaired checkpoint."""
    if item["variant"] not in VARIANTS:
        raise ValueError("Unknown declared repair-model variant")
    path = item["checkpoint_path"]
    if "checkpoint_sha256" in item and _sha(path) != item["checkpoint_sha256"]:
        raise ValueError("A declared payoff checkpoint changed")
    if item["variant"] == "repaired":
        from nn_dk.payoff_repair import load_repaired_payoff_model
        return load_repaired_payoff_model(path)
    return load_payoff_model(path)


def _model_record(directory, slot):
    path = _checkpoint(directory, slot)
    model = load_declared_model({**slot, "checkpoint_path": str(path)})
    cfg = configuration(slot["variant"])
    expected_config = repaired_configuration().config_hash if slot["variant"] == "repaired" else cfg.config_hash
    bank = next(bank for bank in payoff_banks() if bank.name == slot["bank"])
    provenance = model.provenance
    metadata = _read(path.with_suffix(".json"))
    seconds = metadata.get("training_seconds", -1)
    if (not model.complete or provenance.get("source_hash") != source_hash()
            or provenance.get("config_hash") != expected_config
            or provenance.get("seed") != slot["seed"] or model.bank.bank_hash != bank.bank_hash
            or not metadata.get("complete") or metadata.get("identity_hash") != provenance["identity_hash"]
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Repair-model slot metadata, configuration, bank or seed mismatch")
    return {**slot, "init": slot["variant"], "path": str(path.resolve()),
            "checkpoint_path": str(path.resolve()), "checkpoint_sha256": _sha(path),
            "metadata_path": str(path.with_suffix(".json").resolve()),
            "metadata_sha256": _sha(path.with_suffix(".json")),
            "complete": True, "source_hash": source_hash(), "config_hash": expected_config,
            "target_id": model.target_id, "bank_hash": bank.bank_hash, "model_kind": model.kind,
            "training_seconds": seconds, "selected_step": metadata.get("best_step")}


def fit_seed(root, seed, deadline=None):
    directory, design, arrays = _prepared(root)
    if type(seed) is not int or seed not in configuration().optimizer_seeds:
        raise ValueError("Use one declared optimizer seed: 0, 1 or 2")
    with _exclusive(directory / "execution" / f".seed{seed}.lock"):
        if (directory / "fit_lock.json").exists():
            raise ValueError("Final payoff-repair fit lock exists; fitting is closed")
        started, cpu = time.monotonic(), time.process_time()
        output = directory / "execution" / f"worker_seed{seed}.json"
        attempts = _read(output).get("attempts", []) if output.exists() else []
        attempt = {"pid": os.getpid(), "started_unix": time.time(), "complete": False}
        packet = {"workflow": WORKFLOW, "seed": seed, "source_hash": source_hash(),
                  "preregistration_sha256": _sha(directory / "preregistration.json"),
                  "pid": os.getpid(), "complete": False, "models": [],
                  "attempts": [*attempts, attempt], "held_out_reference_inputs_read": False}
        _json(output, packet)
        try:
            for slot in design["slots"]:
                if slot["seed"] != seed:
                    continue
                _assert_frozen(directory)
                if deadline is not None and time.monotonic() >= deadline:
                    raise BudgetExhausted("Payoff-repair seed worker reached its resumable deadline")
                cfg = configuration(slot["variant"])
                train, validation = (_rebound(arrays[role], cfg) for role in ("train", "validation"))
                bank = next(bank for bank in payoff_banks(cfg) if bank.name == slot["bank"])
                path = _checkpoint(directory, slot)
                print(f"[seed {seed}] {slot['bank']} {slot['variant']}", flush=True)
                if slot["variant"] == "repaired":
                    from nn_dk.payoff_repair import fit_repaired_payoff
                    metadata = fit_repaired_payoff(repaired_configuration(), bank, seed, path,
                                                   train, validation, deadline=deadline)
                else:
                    metadata = fit_payoff(cfg, bank, seed, path, train, validation, deadline=deadline)
                _json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    raise BudgetExhausted("Payoff-repair optimizer checkpointed before completion")
                packet["models"].append(_model_record(directory, slot))
                _json(output, packet)
            packet["complete"] = True
        finally:
            attempt.update(complete=packet["complete"], finished_unix=time.time(),
                           wall_seconds=time.monotonic() - started,
                           cpu_seconds=time.process_time() - cpu)
            packet.update(wall_seconds=attempt["wall_seconds"], cpu_seconds=attempt["cpu_seconds"])
            _json(output, packet)
    return packet


def lock(root):
    directory, design, _ = _prepared(root)
    with _exclusive(directory / "execution" / ".fit-lock.lock"), ExitStack() as stack:
        for seed in configuration().optimizer_seeds:
            stack.enter_context(_exclusive(directory / "execution" / f".seed{seed}.lock"))
            worker = _read(directory / "execution" / f"worker_seed{seed}.json")
            if (not worker.get("complete") or worker.get("source_hash") != source_hash()
                    or worker.get("preregistration_sha256") != _sha(directory / "preregistration.json")):
                raise ValueError("All predeclared seed workers must complete before the final fit lock")
        models = [_model_record(directory, slot) for slot in design["slots"]]
        if len(models) != EXPECTED_MODELS or len({item["name"] for item in models}) != EXPECTED_MODELS:
            raise ValueError("Incomplete or duplicated payoff-repair fit inventory")
        result = {"workflow": WORKFLOW, "complete": True, "source_hash": source_hash(),
                  "expected_models": EXPECTED_MODELS, "models": models,
                  "config_hash": configuration().config_hash, "runner_sha256": _sha(SCRIPT),
                  "preregistration_path": str((directory / "preregistration.json").resolve()),
                  "preregistration_sha256": _sha(directory / "preregistration.json"),
                  "data_lock_path": str((directory / "data_lock.json").resolve()),
                  "data_lock_sha256": _sha(directory / "data_lock.json"),
                  "input_banks": _read(directory / "data_lock.json")["banks"],
                  "all_models_shared_literal_pool": True, "seeds_pooled": False,
                  "selection_using_final_outcomes": False,
                  "total_recorded_model_training_seconds": sum(item["training_seconds"] for item in models)}
        _immutable(directory / "fit_lock.json", result)
    return result


def verify_fit_lock(root):
    """Verify the exact complete declared inventory without writing or refitting."""
    directory, design, _ = _prepared(root)
    saved = _read(directory / "fit_lock.json")
    models = [_model_record(directory, slot) for slot in design["slots"]]
    if (not saved.get("complete") or saved.get("source_hash") != source_hash()
            or saved.get("expected_models") != EXPECTED_MODELS or saved.get("models") != models
            or saved.get("data_lock_sha256") != _sha(directory / "data_lock.json")
            or saved.get("preregistration_sha256") != _sha(directory / "preregistration.json")):
        raise ValueError("Complete payoff-repair fit lock is missing, changed or incompatible")
    return saved


def generate_reference(root, deadline=None, max_chunks=None):
    root = Path(root).resolve()
    with _exclusive(root / "reference" / "execution" / ".generation.lock"):
        fit_lock = verify_fit_lock(root)
        backend = _reference_backend()
        by_path = {str(Path(item["checkpoint_path"]).resolve()): item for item in fit_lock["models"]}

        def strict_loader(path):
            item = by_path.get(str(Path(path).resolve()))
            if item is None:
                raise ValueError("Reference unlock requested an undeclared checkpoint")
            return load_declared_model(item)

        backend.load_payoff_model = strict_loader
        return backend.generate_reference(root / "reference", reference_configuration(),
                                           fit_lock=fit_lock, max_chunks=max_chunks, deadline=deadline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "lock", "reference"))
    parser.add_argument("--root", type=Path, default=REPO / "runs" / "payoff-repair-2026-10-27")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8 * 60 * 60)
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 8 * 60 * 60:
        parser.error("Budget must be positive and at most eight hours")
    if (args.stage == "fit") != (args.seed is not None):
        parser.error("Only fit requires --seed 0, 1 or 2")
    if args.max_chunks is not None and (args.stage != "reference" or args.max_chunks < 0):
        parser.error("Only reference accepts a nonnegative --max-chunks")
    deadline = time.monotonic() + args.budget_seconds
    try:
        if args.stage == "prepare":
            print(f"Prepared shared payoff-repair banks: {prepare(args.root, deadline)}", flush=True)
        elif args.stage == "fit":
            result = fit_seed(args.root, args.seed, deadline)
            print(f"Completed seed {args.seed}: {len(result['models'])} declared fits", flush=True)
        elif args.stage == "lock":
            result = lock(args.root)
            print(f"Locked {result['expected_models']} complete payoff-repair fits", flush=True)
        else:
            result = generate_reference(args.root, deadline, args.max_chunks)
            print(json.dumps({"status": result["status"], "committed_chunks": result["committed_chunks"]}),
                  flush=True)
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
