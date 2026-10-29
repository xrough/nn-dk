"""Fresh matched-pool 2x2 proposal/loss study of the fixed v3 architecture.

The isolated historical runner supplies native state banks and independent,
strictly locked Euler references. New slot-aware fitting retains every crossed
variant, seed and regime without changing historical modules or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import fit_payoff_collocation_ablation as engine  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_boundary import BoundaryRepairConfig  # noqa: E402

_implementation = isolated_module("run_payoff_regimes.py", "analysis._collocation_study_runner_impl")
_base_configuration = _implementation.configuration
_base_design = _implementation._design
_base_prepared = _implementation._prepared
WORKFLOW = "fresh-matched-natural-collocation-proposal-loss-study-v1"
STREAM = "fresh-matched-collocation-pools-and-reference-families-2026103101-v1"
PROPOSALS = ("uniform", "stratified")
LOSSES = ("pseudo_huber", "squared")
EVALUATION_HELPERS = (
    "evaluate_payoff_collocation_study.py", "report_payoff_collocation_study.py",
    "evaluate_payoff_regimes.py", "evaluate_payoff_regimes_deployment.py",
    "report_payoff_regimes.py", "audit_payoff_regimes_deployment.py",
    "audit_payoff_regimes.py", "evaluate_path_generalization.py",
    "benchmark_payoff_generalization.py", "native_gaussian_baseline.py",
    "payoff_study_adapter.py", "compiled_native.py", "native_euler_control.cpp")


@dataclass(frozen=True)
class StudyPlan(_implementation.StudyPlan):
    train_seed: int = 2026103101
    reference_seed: int = 2026103102
    proposals: tuple[str, ...] = PROPOSALS
    losses: tuple[str, ...] = LOSSES

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "proposals", tuple(self.proposals))
        object.__setattr__(self, "losses", tuple(self.losses))
        if (self.optimizer_seeds != (0, 1, 2) or self.proposals != PROPOSALS or self.losses != LOSSES):
            raise ValueError("Retain all three seeds and the complete fixed 2x2 proposal/loss crossing")


def configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return replace(_base_configuration(case, plan), name=f"payoff-collocation-study-{case}")


def repaired_configuration(case, plan=None):
    return BoundaryRepairConfig(configuration(case, plan),
                                quadrature_nodes=(StudyPlan() if plan is None else plan).quadrature_nodes)


def ablation_configuration(slot, plan=None):
    plan = StudyPlan() if plan is None else plan
    if slot.get("proposal") not in plan.proposals or slot.get("loss") not in plan.losses:
        raise ValueError("Unknown crossed collocation variant")
    return engine.CollocationAblationConfig(repaired_configuration(slot["case"], plan),
                                           proposal=slot["proposal"], loss=slot["loss"])


def slots(plan=None):
    plan = StudyPlan() if plan is None else plan
    return [{"case": case, "seed": seed, "proposal": proposal, "loss": loss,
             "bank": _implementation.payoff_bank(case, plan).name,
             "variant": f"{proposal}__{loss}",
             "name": f"{case}_{proposal}_{loss}_seed{seed}"}
            for case in _implementation.case_names(plan) for seed in plan.optimizer_seeds
            for proposal in plan.proposals for loss in plan.losses]


def _checkpoint(root, slot):
    return Path(root)/"payoffs"/slot["case"]/"checkpoints"/slot["variant"]/f"seed{slot['seed']}.pt"


def _design(plan):
    return {**_base_design(plan),
        "learner_kind": engine.KIND, "architecture": "unchanged-moving-boundary-native-slope-v3",
        "crossed_design": {"proposals": list(plan.proposals), "losses": list(plan.losses)},
        "variant_configs": {slot["name"]: ablation_configuration(slot, plan).to_dict() for slot in slots(plan)},
        "literal_pool_policy": "one shared fresh native-occupancy train/validation pool per case; identical ordered rows and parent groups across all variants",
        "stress_augmentation": False,
        "selection": "same complete natural-occupancy fixed-unit validation MSE for all variants",
        "initialization": "same architecture and literal initial state per case/optimizer seed across all four variants",
        "training": "known-generator staged residual; uniform or importance-corrected stratified Adam proposals; natural-row full-pool LBFGS",
        "loss_scaling": "shared known-heat full-horizon scales computed from training rows only",
        "fresh_streams": {"train_validation_seed": plan.train_seed, "final_reference_seed": plan.reference_seed,
                          "namespace": STREAM},
        "future_evaluation_policy": "freeze comparative evaluation helpers and criteria before the first held-out reference outcome; physical/family allocations already fixed here",
        "historical_modules_mutated": False,
        "inference": "all seed/variant outcomes retained; no best-seed or post-test refit selection"}


def flat_bank(root, case, role, plan, config=None):
    """Exact ordered natural rows, bound to their HDF5 parent-pool checksum."""
    config = (engine.CollocationAblationConfig(repaired_configuration(case, plan))
              if config is None else config)
    bank = _implementation.load_state_bank(root, case, role)
    return engine.make_flat_bank(bank["states"], bank["horizons"], bank["family_ids"],
        role=role, config=config, parent_family_ids=bank["family_ids"],
        provenance={"stream_namespace": STREAM, "parent_pool_sha256": bank["sha256"],
                    "stress_augmentation": False})


def _flat_lock(root, plan):
    banks = {}
    for case in _implementation.case_names(plan):
        config = engine.CollocationAblationConfig(repaired_configuration(case, plan))
        banks[case] = {}
        for role in ("train", "validation"):
            value = flat_bank(root, case, role, plan, config)
            _, fingerprint = engine._bank(value, config, role)
            banks[case][role] = {"content_sha256": value["content_sha256"], "fingerprint": fingerprint}
    return {"workflow": WORKFLOW, "kind": engine.BANK_KIND, "source_hash": source_hash(),
            "plan_hash": plan.plan_hash, "data_lock_sha256": _implementation._sha(Path(root)/"payoffs/data_lock.json"),
            "future_payoff_labels": False, "stress_augmentation": False, "banks": banks}


def prepare(root, plan=None, deadline=None):
    started, cpu = time.monotonic(), time.process_time()
    path = _implementation.prepare(root, plan, deadline)
    frozen_plan = _implementation.load_plan(root)
    _implementation._immutable(Path(root)/"payoffs/flat_pool_lock.json", _flat_lock(root, frozen_plan))
    cost_path = Path(root)/"payoffs/prepare_costs.json"
    identity = {"workflow": WORKFLOW, "source_hash": source_hash(), "plan_hash": frozen_plan.plan_hash}
    costs = _cost_log(cost_path, identity)
    costs["attempts"].append({"finished": True, "wall_seconds": time.monotonic()-started,
                             "cpu_seconds": time.process_time()-cpu})
    costs["scope"] = "complete prepare invocation including native pools, integrity, compilation and flat locks; native generation is a component, not an extra additive charge"
    _save_cost_log(cost_path, costs)
    return path


def _prepared(root):
    plan, design = _base_prepared(root)
    if _implementation._read(Path(root)/"payoffs/flat_pool_lock.json") != _flat_lock(root, plan):
        raise ValueError("Shared literal flat train/validation pool lock changed")
    return plan, design


def load_declared_model(item):
    path = Path(item["checkpoint_path"])
    if (item.get("proposal") not in PROPOSALS or item.get("loss") not in LOSSES
            or item.get("variant") != item["proposal"]+"__"+item["loss"]
            or item.get("checkpoint_sha256", _implementation._sha(path)) != _implementation._sha(path)):
        raise ValueError("Only an unchanged declared crossed variant may be loaded")
    model = engine.load_collocation_ablation_model(path)
    config = model.config
    if config.proposal != item["proposal"] or config.loss != item["loss"]:
        raise ValueError("Checkpoint proposal/loss does not match its slot")
    return model


def _cost_log(path, identity):
    if not path.exists():
        return {"identity": identity, "attempts": []}
    saved = _implementation._read(path)
    payload = {key: value for key, value in saved.items() if key != "content_sha256"}
    if saved.get("content_sha256") != engine._digest(payload) or saved.get("identity") != identity:
        raise ValueError("Invocation cost provenance or content changed")
    return payload


def _save_cost_log(path, payload):
    _implementation._json(path, {**payload, "content_sha256": engine._digest(payload)})


def _model_record(root, slot, plan):
    path = _checkpoint(root, slot)
    model = load_declared_model({**slot, "checkpoint_path": str(path)})
    metadata = _implementation._read(path.with_suffix(".json"))
    expected = ablation_configuration(slot, plan)
    identity = model.provenance
    flat_lock = _implementation._read(Path(root)/"payoffs/flat_pool_lock.json")
    seconds = metadata.get("training_seconds", -1)
    if (not model.complete or not metadata.get("complete") or metadata.get("kind") != engine.KIND
            or model.method_kind != engine.KIND or identity["sources"]["package_source_hash"] != source_hash()
            or identity["config_hash"] != expected.config_hash or identity["seed"] != slot["seed"]
            or model.bank.bank_hash != _implementation.payoff_bank(slot["case"], plan).bank_hash
            or metadata.get("identity") != identity or metadata.get("method_id") != model.method_id
            or identity["selection"] != "natural-validation-fixed-unit-MSE"
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Crossed model source, configuration, seed, selection or metadata mismatch")
    for role in ("train", "validation"):
        if identity["data"][role] != flat_lock["banks"][slot["case"]][role]["fingerprint"]:
            raise ValueError("A crossed model was fitted to a different literal pool")
    invocation_path = path.with_suffix(".invocations.json")
    invocations = _cost_log(invocation_path, {"source_hash": source_hash(), "config_hash": expected.config_hash,
                                            "seed": slot["seed"], "slot": slot["name"]})
    if not invocations["attempts"]:
        raise ValueError("Every fitted model requires its full invocation cost record")
    known = [value for value in invocations["attempts"] if value.get("finished")]
    return {**slot, "checkpoint_path": str(path.resolve()), "path": str(path.resolve()),
        "checkpoint_sha256": _implementation._sha(path), "metadata_path": str(path.with_suffix(".json").resolve()),
        "metadata_sha256": _implementation._sha(path.with_suffix(".json")), "complete": True,
        "source_hash": source_hash(), "config_hash": expected.config_hash, "target_id": model.target_id,
        "bank_hash": model.bank.bank_hash, "model_kind": engine.KIND, "architecture_kind": model.kind,
        "method_id": model.method_id, "training_seconds": seconds, "stage_seconds": metadata["stage_seconds"],
        "selected_step": metadata["best_step"], "natural_validation": metadata["best_validation"],
        "initial_state_sha256": identity["initial_state_sha256"],
        "data_fingerprints": identity["data"], "loss_scaling_sha256": engine._digest(model.loss_scaling),
        "full_invocation_seconds": sum(value["wall_seconds"] for value in known),
        "full_invocation_cpu_seconds": sum(value["cpu_seconds"] for value in known),
        "full_invocation_cost_complete": len(known) == len(invocations["attempts"]),
        "invocation_record_path": str(invocation_path.resolve()),
        "invocation_record_sha256": _implementation._sha(invocation_path),
        "invocation_cost_scope": "all timed engine entry-through-return attempts including preloop checks/scales/initial validation/saves; runner pool preparation and strict restoration are separate"}


def _matched_controls(models, plan):
    """Prove this is a crossed study of losses/proposals on identical inputs."""
    proof = []
    for case in _implementation.case_names(plan):
        for seed in plan.optimizer_seeds:
            rows = [r for r in models if r["case"] == case and r["seed"] == seed]
            if ({(r["proposal"], r["loss"]) for r in rows} != set(
                    (p, loss) for p in plan.proposals for loss in plan.losses) or len(rows) != 4
                    or len({r["initial_state_sha256"] for r in rows}) != 1
                    or len({engine._digest(r["data_fingerprints"]) for r in rows}) != 1
                    or len({r["loss_scaling_sha256"] for r in rows}) != 1):
                raise ValueError("All crossed variants require identical initialization, literal pools and scales")
            proof.append({"case": case, "seed": seed, "variants": 4,
                          "initial_state_sha256": rows[0]["initial_state_sha256"],
                          "data_fingerprints_sha256": engine._digest(rows[0]["data_fingerprints"]),
                          "loss_scaling_sha256": rows[0]["loss_scaling_sha256"]})
    return proof


def fit_seed(root, seed, deadline=None):
    invocation_started, invocation_cpu = time.monotonic(), time.process_time()
    plan, design = _prepared(root)
    if type(seed) is not int or seed not in plan.optimizer_seeds:
        raise ValueError("Use one preregistered optimizer seed")
    root = Path(root).resolve()
    with _implementation._exclusive(root/"payoffs/execution"/f".seed{seed}.lock"):
        if (root/"payoffs/fit_lock.json").exists():
            raise ValueError("Final fit lock exists; fitting is closed")
        output = root/"payoffs/execution"/f"worker_seed{seed}.json"
        previous = _implementation._read(output).get("attempts", []) if output.exists() else []
        started, cpu = invocation_started, invocation_cpu
        attempt = {"pid": os.getpid(), "started_unix": time.time(), "complete": False}
        packet = {"workflow": WORKFLOW, "seed": seed, "source_hash": source_hash(), "pid": os.getpid(),
            "preregistration_sha256": _implementation._sha(root/"payoffs/preregistration.json"),
            "complete": False, "models": [], "attempts": [*previous, attempt],
            "held_out_reference_inputs_read": False, "selection": "natural-validation-fixed-unit-MSE",
            "pool_preparation": {}, "preflight_integrity_wall_seconds": time.monotonic()-started,
            "worker_cost_scope": "full current invocation including preflight pool/source integrity, cached pool conversion, fits and strict restoration; attempts may overlap other seed workers"}
        _implementation._json(output, packet)
        try:
            pools = {}
            for slot in design["slots"]:
                if slot["seed"] != seed:
                    continue
                _implementation._assert_frozen(root)
                if deadline is not None and time.monotonic() >= deadline:
                    raise _implementation.BudgetExhausted("Collocation worker reached its resumable deadline")
                cfg = ablation_configuration(slot, plan)
                if slot["case"] not in pools:
                    pool_started, pool_cpu = time.monotonic(), time.process_time()
                    pools[slot["case"]] = {role: flat_bank(root, slot["case"], role, plan, cfg)
                                          for role in ("train", "validation")}
                    packet["pool_preparation"][slot["case"]] = {
                        "seconds": time.monotonic()-pool_started, "cpu_seconds": time.process_time()-pool_cpu,
                        "scope": "flat native train/validation pool conversion once within this seed invocation; shared by all four variants, excludes native generation"}
                path = _checkpoint(root, slot)
                print(f"[seed {seed}] {slot['case']} {slot['variant']}", flush=True)
                costs_path = path.with_suffix(".invocations.json")
                costs_path.parent.mkdir(parents=True, exist_ok=True)
                costs = _cost_log(costs_path, {"source_hash": source_hash(), "config_hash": cfg.config_hash,
                                             "seed": seed, "slot": slot["name"]})
                cost_attempt = {"started_unix": time.time(), "pid": os.getpid(), "finished": False}
                costs["attempts"].append(cost_attempt)
                _save_cost_log(costs_path, costs)
                fit_started, fit_cpu = time.monotonic(), time.process_time()
                metadata = None
                try:
                    metadata = engine.fit_collocation_ablation(cfg, _implementation.payoff_bank(slot["case"], plan),
                        seed, path, pools[slot["case"]]["train"], pools[slot["case"]]["validation"], deadline=deadline)
                finally:
                    cost_attempt.update(finished=True, finished_unix=time.time(),
                        wall_seconds=time.monotonic()-fit_started, cpu_seconds=time.process_time()-fit_cpu,
                        optimizer_allocation_complete=bool(metadata and metadata.get("complete")))
                    _save_cost_log(costs_path, costs)
                _implementation._json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    raise _implementation.BudgetExhausted("Collocation optimizer checkpointed before completion")
                packet["models"].append(_model_record(root, slot, plan))
                _implementation._json(output, packet)
            packet["complete"] = True
        finally:
            attempt.update(complete=packet["complete"], finished_unix=time.time(),
                           wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
                           pool_preparation=packet["pool_preparation"],
                           preflight_integrity_wall_seconds=packet["preflight_integrity_wall_seconds"])
            packet.update(wall_seconds=attempt["wall_seconds"], cpu_seconds=attempt["cpu_seconds"])
            _implementation._json(output, packet)
    return packet


_base_lock = _implementation.lock
_base_verify_lock = _implementation.verify_fit_lock
_base_generate_reference = _implementation.generate_reference
_base_load_reference = _implementation.load_reference


def lock(root):
    result = _base_lock(root)
    proof = _matched_controls(result["models"], _implementation.load_plan(root))
    _implementation._immutable(Path(root)/"payoffs/matched_controls.json", {
        "workflow": WORKFLOW, "complete": True, "fit_lock_sha256": _implementation._sha(Path(root)/"payoffs/fit_lock.json"),
        "records": proof, "all_variant_pool_initialization_and_scale_controls_matched": True})
    return result


def verify_fit_lock(root):
    result = _base_verify_lock(root)
    proof = _matched_controls(result["models"], _implementation.load_plan(root))
    expected = {"workflow": WORKFLOW, "complete": True,
        "fit_lock_sha256": _implementation._sha(Path(root)/"payoffs/fit_lock.json"), "records": proof,
        "all_variant_pool_initialization_and_scale_controls_matched": True}
    if _implementation._read(Path(root)/"payoffs/matched_controls.json") != expected:
        raise ValueError("Crossed literal-pool controls changed before reference unlock")
    return result


def evaluation_contract(root, fit_lock=None):
    """Exact method/group/query and diagnostic criteria, without outcome draws."""
    fit_lock = verify_fit_lock(root) if fit_lock is None else fit_lock
    plan = _implementation.load_plan(root)
    methods, inventory, integration = {}, [], []
    for case in _implementation.case_names(plan):
        cfg = _implementation.reference_configuration(case, plan)
        methods[case] = [{"name": row["name"], "role": "trained", "seed": row["seed"],
            "proposal": row["proposal"], "loss": row["loss"], "variant": row["variant"],
            "method_id": row["method_id"], "checkpoint_sha256": row["checkpoint_sha256"]}
            for row in fit_lock["models"] if row["case"] == case]
        methods[case] += [
            {"name": case+"_untrained", "role": "untrained-v3-zero-correction", "seed": 0},
            {"name": "LGA_dt7", "role": "unstopped-Gaussian", "step": 2**-7},
            {"name": "LGA_dt10", "role": "unstopped-Gaussian", "step": 2**-10},
            {"name": "native-MC-dt10", "role": "independent-native-MC", "step": 2**-10}]
        for group in cfg.group_keys:
            inventory.extend({"case": case, "group": group, "method": method["name"],
                              "queries": list(cfg.query_times)} for method in methods[case])
        for profile in cfg.cases:
            integration.append({"case": case, "profile": profile, "group": profile+"__ensemble__0",
                                "families": cfg.paths_per_group,
                                "methods": [m["name"] for m in methods[case] if m["role"] in
                                            ("trained", "untrained-v3-zero-correction")]})
    independent = sum(len(row["methods"]) for row in integration)
    return {"schema_version": 1, "workflow": WORKFLOW, "source_hash": source_hash(),
        "plan_hash": plan.plan_hash, "all_optimizer_seeds": list(plan.optimizer_seeds),
        "cases": list(_implementation.case_names(plan)), "methods": methods,
        "method_group_query_inventory": inventory, "independent_IID_integration": integration,
        "counts": {"trained_models": len(fit_lock["models"]), "primary_records": len(inventory),
                   "independent_IID_replacement_records": independent,
                   "raw_retained_records": len(inventory)+independent},
        "criteria": {"accuracy_target": .05, "uncertainty_component_target": .0125,
            "confidence": .95, "bootstrap_replicates": 128, "bootstrap_unit": "whole independent trajectory family",
            "intervals": "pointwise only; no simultaneous grid-wide confidence",
            "mean_units": "reference standard deviations",
            "covariance_units": "products of reference standard deviations",
            "supported_outputs": "four means and full same-time4x4covariance at each declared physical query",
            "finite_reference_gate": "5% finite-reference diagnostic only; reference refinement and sampling uncertainties remain separate",
            "continuum_or_continuous_stopping_certificate": False,
            "all_optimizer_seeds_retained": True, "selection_using_final_outcomes": False,
            "post_test_refitting": False, "all_stopped_paths_retained": True,
            "failed_paths": "mark complete method/group unavailable; no survivor-only estimates or silent path removal",
            "independent_initials_primary_for_IID": True,
            "native_MC": "independent initial/forcing families; finite-step error and standalone sampling must qualify any cost advantage",
            "cost_gate": "same tasks, finite-reference accuracy and uncertainty requirements; no timing-only or continuum-qualified crossover",
            "timing": "literal workloads only after all fitting/reference/accuracy workers exit"}}


def evaluation_helper_paths():
    """Separate from training helpers: future evaluation code may be finished during fitting."""
    return tuple(Path(name) if Path(name).is_absolute() else Path(__file__).with_name(name)
                 for name in EVALUATION_HELPERS)


def _evaluation_packet(root, fit_lock):
    from analysis.evaluate_path_generalization import array_sha

    root = Path(root).resolve()
    contract = evaluation_contract(root, fit_lock)
    hashes = {}
    for path in evaluation_helper_paths():
        if not path.is_file() or not path.read_bytes().strip():
            raise ValueError(f"Required future evaluation/report helper is missing or empty: {path.name}")
        hashes[str(path.resolve())] = _implementation._sha(path)
    names = ("comparison/preregistration.json", "deployment/preregistration.json",
             "deployment/planned_initial_families.npz", "audit/deployment_pretest_independence.json")
    for name in names:
        if not (root/name).is_file():
            raise ValueError(f"Prepare evaluation criteria and identifier plans before freeze-evaluation: {name}")
    comparison = _implementation._read(root/names[0])
    deployment = _implementation._read(root/names[1])
    address = _implementation._read(root/names[3])
    if comparison.get("contract") != contract:
        raise ValueError("Exact comparative method/group/query inventory or criteria differ from the frozen contract")
    if (comparison.get("accuracy_target") != .05 or comparison.get("uncertainty_component_target") != .0125
            or comparison.get("bootstrap", {}).get("count") != 128
            or comparison.get("selection_using_test") is not False or comparison.get("post_test_refitting") is not False):
        raise ValueError("Comparative accuracy, uncertainty or selection criteria changed")
    for packet in (comparison, deployment):
        if not isinstance(packet.get("helpers"), dict) or not packet["helpers"]:
            raise ValueError("Evaluation helpers must be explicitly checksum-indexed")
        for name, digest in packet["helpers"].items():
            path = Path(name).resolve()
            if not path.is_file() or _implementation._sha(path) != digest:
                raise ValueError("Current evaluation dependency differs from its declared helper hash")
            hashes[str(path)] = digest
    if (not address.get("complete") or address.get("planned_families_disjoint") is not True
            or address.get("address_streams_disjoint") is not True
            or address.get("source_hash") != source_hash()
            or address.get("deployment_plan_sha256") != _implementation._sha(root/names[1])
            or address.get("planned_family_npz_sha256") != _implementation._sha(root/names[2])
            or address.get("original_comparison_preregistration_sha256") != _implementation._sha(root/names[0])
            or address.get("training_preregistration_sha256") != _implementation._sha(root/"payoffs/preregistration.json")):
        raise ValueError("Independent initial-integration identifier/address audit is missing or stale")
    pools = deployment.get("pools", [])
    expected = {(item["case"], item["profile"]): item for item in contract["independent_IID_integration"]}
    actual = [(item.get("regime"), item.get("profile")) for item in pools]
    if (len(actual) != len(expected) or set(actual) != set(expected)
            or deployment.get("models_per_pool") != 13
            or deployment.get("expected_records") != contract["counts"]["independent_IID_replacement_records"]):
        raise ValueError("Incomplete independent initial-integration method/pool allocation")
    with np.load(root/names[2]) as arrays:
        if set(arrays.files) != {item["key"] for item in pools}:
            raise ValueError("Independent initial family identifier arrays are incomplete")
        seen = set()
        for item in pools:
            values = arrays[item["key"]]
            wanted = expected[(item["regime"], item["profile"])]
            ids = values.astype(str).tolist()
            if (values.shape != (wanted["families"],) or item.get("paths") != wanted["families"]
                    or item.get("group") != wanted["group"] or len(set(ids)) != len(ids) or seen.intersection(ids)
                    or item.get("family_ids_array_sha256") != array_sha(values)):
                raise ValueError("Independent initial family inventory changed or overlaps")
            seen.update(ids)
    return {"kind": "locked-all-model-comparative-evaluation-before-reference-v1", "complete": True,
        "source_hash": source_hash(), "workflow": WORKFLOW, "contract": contract,
        "fit_lock_sha256": _implementation._sha(root/"payoffs/fit_lock.json"),
        "study_plan_sha256": _implementation._sha(root/"plan.json"),
        "training_preregistration_sha256": _implementation._sha(root/"payoffs/preregistration.json"),
        "planning_artifacts": {name: _implementation._sha(root/name) for name in names},
        "helpers": hashes}


def _heldout_outcomes(root):
    found = []
    for directory in (Path(root)/"reference", Path(root)/"deployment/initials",
                      Path(root)/"comparison/accuracy", Path(root)/"deployment/accuracy"):
        for path in directory.rglob("*") if directory.exists() else ():
            name = path.name.lower()
            if path.is_file() and (path.suffix.lower() in (".h5", ".hdf5", ".npz", ".npy")
                                   or "outcome" in name or name.startswith("chunk")):
                found.append(str(path))
    return found


def freeze_evaluation(root):
    """Lock exact criteria/helpers only after every crossed fit strictly restores."""
    root = Path(root).resolve()
    with _implementation._exclusive(root/"comparison/.evaluation-freeze.lock"):
        fit_lock = verify_fit_lock(root)
        if _heldout_outcomes(root):
            raise ValueError("Evaluation must freeze before any held-out reference HDF5, initial draws or outcome chunks")
        packet = _evaluation_packet(root, fit_lock)
        _implementation._immutable(root/"comparison/evaluation_lock.json", packet)
        return packet


def verify_evaluation_lock(root):
    root = Path(root).resolve()
    path = root/"comparison/evaluation_lock.json"
    if not path.is_file():
        raise ValueError("Freeze comparative evaluation before any held-out reference anchor or outcome draw")
    packet = _evaluation_packet(root, verify_fit_lock(root))
    if _implementation._read(path) != packet:
        raise ValueError("Evaluation lock, model inventory, criteria, identifier plans or helper source changed")
    return packet


def generate_reference(root, case=None, deadline=None, max_chunks=None):
    verify_evaluation_lock(root)
    return _base_generate_reference(root, case, deadline, max_chunks)


def load_reference(root, case, group_key=None):
    verify_evaluation_lock(root)
    return _base_load_reference(root, case, group_key)


bind_study(_implementation, __file__, WORKFLOW=WORKFLOW, STREAM=STREAM, SCRIPT=Path(__file__).resolve(),
    HELPERS=("run_payoff_regimes.py", "payoff_study_adapter.py", "fit_payoff_collocation_ablation.py",
             "payoff_collocation_sampling.py", "compiled_native.py", "native_euler_control.cpp"),
    StudyPlan=StudyPlan, configuration=configuration, repaired_configuration=repaired_configuration,
    slots=slots, _checkpoint=_checkpoint, _design=_design, _prepared=_prepared,
    _model_record=_model_record, fit_seed=fit_seed, load_declared_model=load_declared_model,
    lock=lock, verify_fit_lock=verify_fit_lock, generate_reference=generate_reference,
    load_reference=load_reference)


def __getattr__(name):
    return getattr(_implementation, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "lock", "freeze-evaluation", "reference", "calibrate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args()
    if ((args.stage == "fit") != (args.seed is not None) or (args.plan is not None and args.stage != "prepare")
            or (args.case is not None and args.stage != "reference")
            or (args.max_chunks is not None and (args.stage != "reference" or args.max_chunks < 0))
            or not 0 < args.budget_seconds <= 8*60*60):
        parser.error("Seed only for fit, plan only for prepare, case/chunk budget only for reference, positive <=8h budget")
    deadline = time.monotonic()+args.budget_seconds
    torch.set_num_threads(1)
    try:
        if args.stage == "prepare":
            plan = StudyPlan.from_dict(_implementation._read(args.plan)) if args.plan else None
            print(prepare(args.root, plan, deadline), flush=True)
        elif args.stage == "fit":
            result = fit_seed(args.root, args.seed, deadline)
            print(json.dumps({"complete": result["complete"], "models": len(result["models"])}), flush=True)
        elif args.stage == "lock":
            result = lock(args.root)
            print(json.dumps({"complete": True, "models": len(result["models"])}), flush=True)
        elif args.stage == "freeze-evaluation":
            result = freeze_evaluation(args.root)
            print(json.dumps({"complete": result["complete"], "records": len(result["contract"]["method_group_query_inventory"])}), flush=True)
        elif args.stage == "reference":
            result = _implementation.generate_reference(args.root, args.case, deadline, args.max_chunks)
            print(json.dumps({case: value["status"] for case, value in result.items()}), flush=True)
        else:
            print(json.dumps(_implementation.calibration(args.root)), flush=True)
    except _implementation.BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
