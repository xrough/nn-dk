"""Locked all-variant moment comparison with independent IID integration.

The historical numerical/uncertainty algorithms run in private namespaces.
All crossed methods share new reference resamples; IID neural integration uses
a separate new initial pool and independent resampling. Paired-initial outputs
remain diagnostics. This module generates no final data during preparation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis.audit_payoff_regimes import (  # noqa: E402
    entropy_digest,
    historical_identifier_inventory,
    read_identifiers,
)
from analysis.audit_payoff_regimes_deployment import (  # noqa: E402
    check_bootstrap_seed_independence,
    verify_pool_contracts,
)
from analysis.evaluate_payoff_boundary_regimes import predictor_advantage  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.payoff_boundary import BoundaryPayoffModel  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "matched-collocation-all-variant-fresh-moment-comparison-v1"
SEED = 2026103121
DEPLOYMENT_KIND = "matched-collocation-independent-initial-integration-v1"
DEPLOYMENT_SEED = 2026103127

_base = isolated_module("evaluate_payoff_regimes.py", "analysis._collocation_comparison_impl")
_deployment = isolated_module("evaluate_payoff_regimes_deployment.py", "analysis._collocation_deployment_impl")
_base_prepare, _base_evaluate = _base.prepare, _base.evaluate
_deployment_prepare, _deployment_evaluate = _deployment.prepare, _deployment.evaluate
_base_immutable = _base._immutable


def _helper_files():
    names = ("evaluate_payoff_collocation_study.py", "report_payoff_collocation_study.py",
        "run_payoff_collocation_study.py", "run_payoff_regimes.py", "fit_payoff_collocation_ablation.py",
        "payoff_collocation_sampling.py", "payoff_study_adapter.py", "evaluate_payoff_regimes.py",
        "evaluate_payoff_regimes_deployment.py", "evaluate_payoff_boundary_regimes.py",
        "report_payoff_regimes.py",
        "payoff_legacy_control.py", "audit_payoff_regimes.py", "audit_payoff_regimes_deployment.py",
        "benchmark_payoff_generalization.py", "benchmark_path_generalization.py",
        "evaluate_path_generalization.py", "native_gaussian_baseline.py", "compiled_native.py",
        "native_euler_control.cpp")
    paths = [REPO/"analysis"/name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError("All future evaluator/reporter and numerical helper sources must exist before freezing")
    return {str(path): runner._sha(path) for path in paths}


def _immutable(path, value):
    path = Path(path)
    if value.get("kind") in (KIND, DEPLOYMENT_KIND):
        study = runner.load_plan(path.parents[1])
        count = len(study.proposals)*len(study.losses)*len(study.optimizer_seeds)
        value = {**value, "helpers": _helper_files()}
        if value["kind"] == KIND:
            value.update(contract=runner.evaluation_contract(path.parents[1]),
                records_per_group=count+4, trained_models_per_case=count,
                primary_NN_ensemble_initials="independent-of-reference deployment pool",
                paired_initial_predictions="diagnostic only; never primary IID deployment",
                matched_seed_contrast="each nonbaseline crossed variant versus uniform pseudo_huber at the same seed",
                bootstrap_scope="pointwise whole-family; common reference and independent initial resampling",
                simultaneous_grid_confidence=False)
        else:
            value.update(models_per_pool=count+1, expected_records=len(value["pools"])*(count+1),
                all_variants_and_optimizer_seeds_retained=True,
                shared_pool_across_all_variants_and_seeds=True)
    return _base_immutable(path, value)


def _comparison_prepare(root):
    runner.verify_fit_lock(root)
    return _base_prepare(root)


def model_entries(lock, regime, seeds=(0, 1, 2)):
    rows = [row for row in lock["models"] if row["case"] == regime]
    expected = {(proposal, loss, seed) for proposal in runner.PROPOSALS
                for loss in runner.LOSSES for seed in seeds}
    if (len(rows) != len(expected) or len({row["name"] for row in rows}) != len(rows)
            or {(row.get("proposal"), row.get("loss"), row.get("seed")) for row in rows} != expected
            or any(row.get("variant") != row["proposal"]+"__"+row["loss"]
                   or row["name"] != f"{regime}_{row['proposal']}_{row['loss']}_seed{row['seed']}" for row in rows)):
        raise ValueError("Every crossed variant and optimizer seed is required, without duplicates")
    return [{**row, "family": "repaired"} for row in rows] + [{"case": regime, "seed": 0,
        "name": regime+"_untrained", "family": "untrained-repaired", "untrained": True}]


def _model(item, module, study):
    if not item.get("untrained"):
        return module.load_declared_model(item)
    cfg, repair = module.reference_configuration(item["case"], study), module.repaired_configuration(item["case"], study)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return BoundaryPayoffModel(cfg.spec, cfg.context("uniform"), cfg.horizon,
            module.payoff_bank(item["case"], study), repair.study.width, repair.quadrature_nodes).eval()


def _all_references(root, study):
    manifests = {}
    for case in runner.case_names(study):
        packet = _base._reference_manifest(root, case)
        identity = packet["identity"]
        if (identity.get("stream") != runner.STREAM or identity.get("source_hash") != source_hash()
                or identity.get("runner_sha256") != runner._sha(runner.__file__)
                or packet.get("groups") != list(runner.reference_configuration(case, study).group_keys)):
            raise ValueError("A complete fresh-reference source/stream/group identity is required")
        manifests[case] = packet
    return manifests


def _verified(root):
    root = Path(root).resolve()
    study = runner.load_plan(root)
    plan = _comparison_prepare(root)
    lock = runner.verify_fit_lock(root)
    runner.verify_evaluation_lock(root)
    if len(lock["models"]) != len(runner.slots(study)):
        raise ValueError("Every declared fit must restore before evaluation")
    _all_references(root, study)
    return root, runner, study, plan, lock


def _initial_only_audit(root, deployment_plan):
    """Check identifier/address separation without opening any future outcomes."""
    root = Path(root).resolve()
    output = root/"audit/deployment_pretest_independence.json"
    plan_sha = runner._sha(root/"deployment/preregistration.json")
    families_path = root/"deployment/planned_initial_families.npz"
    if output.exists():
        result = runner._read(output)
        if (not result.get("complete") or result.get("source_hash") != source_hash()
                or result.get("deployment_plan_sha256") != plan_sha
                or result.get("planned_family_npz_sha256") != runner._sha(families_path)
                or result.get("training_preregistration_sha256") != runner._sha(root/"payoffs/preregistration.json")
                or result.get("original_comparison_preregistration_sha256") != runner._sha(root/"comparison/preregistration.json")
                or result.get("planned_families_disjoint") is not True
                or result.get("address_streams_disjoint") is not True):
            raise ValueError("Completed pre-outcome family/address audit changed")
        return result
    study = runner.load_plan(root)
    if any((root/"reference"/case/"reference.h5").exists() for case in runner.case_names(study)):
        raise ValueError("Family/address preparation must precede final outcomes")
    if any((root/"deployment/initials").rglob("*.npz")):
        raise ValueError("Family/address preparation must precede independent initial draws")
    with np.load(families_path) as stored:
        new_ids, new_addresses, bootstrap_seeds, pools = verify_pool_contracts(
            deployment_plan, {key: stored[key] for key in stored.files}, _deployment.array_sha)
    future, candidates, learner, addresses, reference_seeds = set(), set(), set(), set(), set()
    learner_files = []
    for case in runner.case_names(study):
        cfg = runner.configuration(case, study)
        for role in ("train", "validation"):
            path = root/"payoffs"/case/f"states_{role}.h5"
            ids, datasets = read_identifiers(path)
            learner.update(ids)
            learner_files.append({"path": str(path), "datasets": datasets})
            count = cfg.train_paths if role == "train" else cfg.validation_paths
            for profile in cfg.cases:
                for index in range(count):
                    for purpose in ("initial", "physical-forcing", "residual-horizons"):
                        addresses.add(entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                                    role, profile, 0, index, purpose))
        cfg = runner.reference_configuration(case, study)
        for group in cfg.group_keys:
            profile, role, anchor = group.split("__")
            reference_seeds.add(int(stable_hash([SEED, case, group])[:12], 16))
            if role == "conditional":
                addresses.add(entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                            "held-out-anchor", profile, int(anchor), 0, "initial"))
            for index in range(cfg.paths_per_group):
                future.add(runner._family(cfg, "reference-"+role, profile, int(anchor), index))
                candidates.add(stable_hash([SEED, case, group, "independent-candidate", index]))
                for purpose in (("initial", "physical-forcing") if role == "ensemble" else ("physical-forcing",)):
                    addresses.add(entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                                 "reference-"+role, profile, int(anchor), index, purpose))
            seed = int(stable_hash([SEED, case, group, "independent-candidate"])[:12], 16)
            addresses.add(entropy_digest(seed, KIND, "candidate-native-MC"))
    if (new_ids & (future | candidates | learner) or future & (candidates | learner)
            or candidates & learner or new_addresses & addresses or SEED+2 in reference_seeds):
        raise ValueError("New learner/reference/candidate/initial families or physical streams overlap")
    historical, history = historical_identifier_inventory(REPO/"runs", root, new_ids | future | candidates)
    bootstrap = check_bootstrap_seed_independence(bootstrap_seeds, reference_seeds, SEED+2)
    result = {"complete": True, "kind": KIND+"-pretest-identifiers",
        "source_hash": source_hash(), "deployment_plan_sha256": plan_sha,
        "planned_family_npz_sha256": runner._sha(families_path),
        "training_preregistration_sha256": runner._sha(root/"payoffs/preregistration.json"),
        "original_comparison_preregistration_sha256": runner._sha(root/"comparison/preregistration.json"),
        "planned_families_disjoint": True, "address_streams_disjoint": True,
        "planned_pools": pools, "learner_identifier_files": learner_files,
        "historical_identifier_files": history, "historical_identifiers": len(historical),
        "future_reference_families": len(future), "native_candidate_families": len(candidates),
        "independent_initial_families": len(new_ids), "bootstrap_seed_audit": bootstrap,
        "identifier_inventory_sha256": stable_hash(sorted(new_ids | future | candidates | learner)),
        "no_histogram_reference_forcing_or_bootstrap_draw_during_audit": True}
    runner._immutable(output, result)
    return result


def prepare(root):
    """Freeze criteria/helpers and initial identifiers after all fits, before outcomes."""
    root = Path(root).resolve()
    plan = _comparison_prepare(root)
    deployment_plan = _deployment_prepare(root)
    _initial_only_audit(root, deployment_plan)
    return plan


def _deployment_verified(root, *, require_complete_accuracy=True):
    root, _, study, plan, lock = _verified(root)
    deployment_plan = _deployment_prepare(root)
    _initial_only_audit(root, deployment_plan)
    if require_complete_accuracy:
        _validate_index(root, "comparison", plan, study, lock, require_complete=True)
    return root, deployment_plan, study, lock


def _expected_methods(lock, case, study, *, deployment=False):
    models = model_entries(lock, case, study.optimizer_seeds)
    if not deployment:
        models += [{"name": "LGA_dt7", "family": "LGA", "step": 2**-7},
                   {"name": "LGA_dt10", "family": "LGA", "step": 2**-10},
                   {"name": "native-MC-dt10", "family": "native-MC", "step": 2**-10}]
    return {row["name"]: row for row in models}


def _artifact(path, root, inputs, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("New study evidence cannot silently reference an older study")
    digest = runner._sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Changed scientific record or numeric evidence checksum")
    inputs[str(path)] = digest
    return path


def _group_records(root, branch, case, group, study, lock, inputs):
    path = _artifact(root/branch/"accuracy"/case/group/"group.json", root, inputs)
    packet = runner._read(path)
    deployment = branch == "deployment"
    expected = _expected_methods(lock, case, study, deployment=deployment)
    plan_sha = runner._sha(root/"comparison/preregistration.json")
    base = {"source_hash": source_hash(), "comparison_plan_sha256": plan_sha,
        "fit_lock_sha256": runner._sha(root/"payoffs/fit_lock.json"), "regime": case,
        "reference_sha256": _base._reference_manifest(root, case)["reference_sha256"],
        "bank_hash": runner.payoff_bank(case, study).bank_hash}
    if deployment:
        initial_path = _artifact(packet["initial_pool_record_path"], root, inputs,
                                 packet["initial_pool_record_sha256"])
        initial = runner._read(initial_path)
        arrays_path = _artifact(initial["arrays"]["path"], root, inputs, initial["arrays"]["sha256"])
        deployment_plan = runner._read(root/"deployment/preregistration.json")
        pool = next(row for row in deployment_plan["pools"] if row["regime"] == case and row["group"] == group)
        initial_identity = {"kind": DEPLOYMENT_KIND, "source_hash": source_hash(),
            "plan_sha256": runner._sha(root/"deployment/preregistration.json"),
            "fit_lock_sha256": base["fit_lock_sha256"], "pool": pool}
        cfg = runner.reference_configuration(case, study)
        with np.load(arrays_path) as arrays:
            family_ids = arrays["family_ids"]
            values = arrays["initial"]
            if (not initial.get("complete") or initial.get("identity") != initial_identity
                    or set(arrays.files) != {"family_ids", "initial", "probabilities"}
                    or values.shape != (pool["paths"], cfg.n_cells) or not np.isfinite(values).all()
                    or np.any(values < 0) or not np.allclose(values.sum(-1), 1, rtol=0, atol=1e-12)
                    or not np.array_equal(family_ids, _deployment.planned_families(case, pool["profile"], pool["paths"]))
                    or _deployment.array_sha(family_ids) != pool["family_ids_array_sha256"]
                    or not np.array_equal(arrays["probabilities"], runner.profile_probabilities(cfg, pool["profile"])[0])
                    or set(family_ids.astype(str)) & set(runner.load_reference(root, case, group)["family_ids"].astype(str))):
                raise ValueError("Independent initial-pool identity, simplex data or planned families changed")
        base.update(group=group, deployment_plan_sha256=runner._sha(root/"deployment/preregistration.json"),
                    independent_initial_pool_sha256=initial["arrays"]["sha256"])
    if (not packet.get("complete") or packet.get("identity") != base
            or packet.get("group") != group or len(packet.get("records", [])) != len(expected)
            or (deployment and packet.get("independent_initials") is not True)):
        raise ValueError("Complete exact method/group/reference identity required")
    records, found = [], set()
    for entry in packet["records"]:
        record = runner._read(_artifact(entry["path"], root, inputs, entry["sha256"]))
        method = record["method"]
        name = method["name"]
        if (name not in expected or name in found or not record.get("complete")
                or record["identity"] != {**base, **({} if deployment else {"group": group}), "name": name}
                or method != expected[name]
                or len(record["metrics"].get("records", [])) != len(runner.reference_configuration(case, study).query_times)
                or (deployment and (record.get("initial_pool_independent_of_reference") is not True
                    or record.get("reference_and_initial_bootstrap_independent") is not True))):
            raise ValueError("An all-seed/all-variant method record changed or disappeared")
        _artifact(record["arrays"]["path"], root, inputs, record["arrays"]["sha256"])
        found.add(name)
        records.append(record)
    if found != set(expected):
        raise ValueError("Complete crossed method inventory required")
    return {**packet, "regime": case}, records


def _validate_index(root, branch, plan, study, lock, *, require_complete=False):
    pairs = [(case, group) for case in plan["cases"]
             for group in runner.reference_configuration(case, study).group_keys
             if branch != "deployment" or "__ensemble__" in group]
    path = root/branch/"accuracy_index.json"
    if not path.exists():
        if require_complete:
            raise ValueError("Complete independently initialized accuracy comparison required")
        return [], [], {}
    index = runner._read(path)
    if index.get("kind") != KIND+"-"+branch:
        if require_complete:
            raise ValueError("Complete checked comparison index required")
        # A crash between the inherited evaluator and the final atomic index is
        # recoverable only by verifying every actual group/record below.
        return [], [], {}
    if (index.get("source_hash") != source_hash()
            or index.get("plan_sha256") != runner._sha(root/"comparison/preregistration.json")
            or index.get("fit_lock_sha256") != runner._sha(root/"payoffs/fit_lock.json")
            or {tuple(pair) for pair in index.get("expected_groups", [])} != set(pairs)
            or len(index.get("groups", [])) != len({tuple(pair) for pair in index.get("groups", [])})):
        raise ValueError("Checked accuracy index identity/inventory changed")
    inputs, groups, records = {}, [], []
    for case, group in index["groups"]:
        if (case, group) not in pairs:
            raise ValueError("Undeclared group in accuracy index")
        group_path = root/branch/"accuracy"/case/group/"group.json"
        _artifact(group_path, root, inputs, index["group_sha256"][case+"/"+group])
        packet, rows = _group_records(root, branch, case, group, study, lock, inputs)
        groups.append(packet)
        records.extend(rows)
    complete = len(groups) == len(pairs)
    if index.get("complete") != complete or (require_complete and not complete):
        raise ValueError("All declared comparison groups must be complete")
    return groups, records, inputs


def _refresh_index(root, branch, plan, study, lock):
    expected = [(case, group) for case in plan["cases"]
        for group in runner.reference_configuration(case, study).group_keys
        if branch != "deployment" or "__ensemble__" in group]
    completed, hashes = [], {}
    for case, group in expected:
        path = root/branch/"accuracy"/case/group/"group.json"
        if path.exists():
            _group_records(root, branch, case, group, study, lock, {})
            completed.append((case, group))
            hashes[case+"/"+group] = runner._sha(path)
    runner._json(root/branch/"accuracy_index.json", {"kind": KIND+"-"+branch,
        "complete": len(completed) == len(expected), "groups": completed, "expected_groups": expected,
        "group_sha256": hashes, "source_hash": source_hash(),
        "plan_sha256": runner._sha(root/"comparison/preregistration.json"),
        "fit_lock_sha256": runner._sha(root/"payoffs/fit_lock.json"),
        "records_per_group": plan["records_per_group"]-(3 if branch == "deployment" else 0)})


def _add_contrasts(root, branch, case, group, study):
    destination = root/branch/"accuracy"/case/group
    if not (destination/"group.json").exists():
        return
    data = runner.load_reference(root, case, group)
    cfg = runner.reference_configuration(case, study)
    counts = _base.bootstrap_counts(len(data["outcomes"]), int(stable_hash([SEED, case, group])[:12], 16), 128)
    result = {}
    for seed in study.optimizer_seeds:
        baseline_name = f"{case}_uniform_pseudo_huber_seed{seed}"
        baseline_packet = runner._read(destination/(baseline_name+".json"))
        with np.load(baseline_packet["arrays"]["path"]) as values:
            baseline = {key: values[key] for key in values.files}
        for proposal in study.proposals:
            for loss in study.losses:
                name = f"{case}_{proposal}_{loss}_seed{seed}"
                if name == baseline_name:
                    continue
                packet = runner._read(destination/(name+".json"))
                if runner._sha(packet["arrays"]["path"]) != packet["arrays"]["sha256"]:
                    raise ValueError("Changed matched-contrast predictor evidence")
                with np.load(packet["arrays"]["path"]) as values:
                    candidate = {key: values[key] for key in values.files}
                result[name] = {**predictor_advantage(data["outcomes"], candidate, baseline, counts, cfg.n_particles),
                    "candidate_name": name, "baseline_name": baseline_name, "optimizer_seed": seed,
                    "proposal": proposal, "loss": loss}
    packet = runner._read(destination/"group.json")
    packet["NN_versus_uniform_huber"] = result
    runner._json(destination/"group.json", packet)


def evaluate(root, cases=None, groups=None):
    root, _, study, plan, lock = _verified(root)
    with runner._exclusive(root/"comparison/execution/.accuracy.lock"):
        for branch in ("comparison", "deployment"):
            _validate_index(root, branch, plan, study, lock)
        records = _base_evaluate(root, cases, groups)
        for case in plan["cases"] if cases is None else cases:
            for group in runner.reference_configuration(case, study).group_keys if groups is None else groups:
                _add_contrasts(root, "comparison", case, group, study)
        _refresh_index(root, "comparison", plan, study, lock)
        index = runner._read(root/"comparison/accuracy_index.json")
        if index["complete"]:
            records += _deployment_evaluate(root, cases)
            for case in plan["cases"] if cases is None else cases:
                for profile in runner.reference_configuration(case, study).cases:
                    _add_contrasts(root, "deployment", case, profile+"__ensemble__0", study)
            _refresh_index(root, "deployment", plan, study, lock)
    return records


def primary_records(root, *, require_timing=True):
    """Strict complete evidence; replace only IID NN rows with independent pools."""
    root, _, study, plan, lock = _verified(root)
    main_groups, main_records, inputs = _validate_index(root, "comparison", plan, study, lock, require_complete=True)
    dep_groups, dep_records, dep_inputs = _validate_index(root, "deployment", plan, study, lock, require_complete=True)
    inputs.update(dep_inputs)
    replacements = {(row["identity"]["regime"], row["identity"]["group"], row["identity"]["name"]): row for row in dep_records}
    records, paired = [], []
    for row in main_records:
        identity = row["identity"]
        key = identity["regime"], identity["group"], identity["name"]
        if "__ensemble__" in identity["group"] and row["method"]["family"] in ("repaired", "untrained-repaired"):
            paired.append(row)
            records.append(replacements.pop(key))
        else:
            records.append(row)
    if replacements:
        raise ValueError("Undeclared independent initial method record")
    replacement_groups = {(g["regime"], g["group"]): g for g in dep_groups}
    groups, paired_groups = [], []
    for group in main_groups:
        replacement = replacement_groups.get((group["regime"], group["group"]))
        if replacement:
            paired_groups.append(group)
            groups.append({**group, "NN_versus_fine_LGA": replacement["NN_versus_fine_LGA"],
                "NN_versus_uniform_huber": replacement["NN_versus_uniform_huber"],
                "primary_NN_initial_integration": "independent-of-reference"})
        else:
            groups.append(group)
    for relative in ("plan.json", "payoffs/preregistration.json", "payoffs/data_lock.json",
        "payoffs/flat_pool_lock.json", "payoffs/fit_lock.json", "payoffs/matched_controls.json",
        "comparison/preregistration.json", "comparison/accuracy_index.json", "comparison/evaluation_lock.json",
        "deployment/preregistration.json", "deployment/preparation.json", "deployment/planned_initial_families.npz",
        "deployment/accuracy_index.json", "audit/deployment_pretest_independence.json"):
        _artifact(root/relative, root, inputs)
    for item in lock["models"]:
        for kind in ("checkpoint", "metadata"):
            _artifact(item[kind+"_path"], root, inputs, item[kind+"_sha256"])
    timing = None
    if require_timing:
        timing = _read_timing(root)
        if not timing.get("complete"):
            raise ValueError("Complete final isolated timing required")
        _artifact(root/"comparison/timing.json", root, inputs)
    return {"root": root, "study": study, "plan": plan, "lock": lock, "timing": timing,
        "groups": groups, "records": records, "references": _all_references(root, study), "inputs": inputs,
        "paired_initial_records": paired, "paired_initial_groups": paired_groups,
        "original_accuracy_records": len(main_records), "independent_initial_records": len(dep_records)}


def _timing_identity(root):
    return {"plan_sha256": runner._sha(root/"comparison/preregistration.json"),
            "fit_lock_sha256": runner._sha(root/"payoffs/fit_lock.json"), "source_hash": source_hash()}


def _read_timing(root):
    packet = runner._read(root/"comparison/timing.json")
    if (packet.get("identity") != _timing_identity(root)
            or packet.get("payload_sha256") != stable_hash({k: v for k, v in packet.items() if k != "payload_sha256"})
            or packet.get("final_timing_after_all_workers_finished") is not True
            or len({row["slot"] for row in packet["records"]}) != len(packet["records"])):
        raise ValueError("Final timing identity, checksum or workload inventory changed")
    study, plan = runner.load_plan(root), runner._read(root/"comparison/preregistration.json")
    lock = runner.verify_fit_lock(root)
    expected, loads = {}, set()
    for case in plan["cases"]:
        for profile in runner.reference_configuration(case, study).cases:
            for item in model_entries(lock, case, study.optimizer_seeds):
                loads.add((case, profile, item["name"]))
                common = {"regime": case, "profile": profile, "name": item["name"],
                    "family": item["family"], "seed": item["seed"], "bank": runner.payoff_bank(case, study).name,
                    **{key: item[key] for key in ("proposal", "loss", "variant") if key in item}}
                expected[f"{case}/{profile}/{item['name']}/ensemble"] = {
                    **common, "role": "ensemble", "batch": runner.reference_configuration(case, study).paths_per_group}
                for batch in plan["neural_conditional_batches"]:
                    expected[f"{case}/{profile}/{item['name']}/conditional/{batch}"] = {
                        **common, "role": "conditional", "batch": batch}
            for step in plan["LGA_max_steps"]:
                for role in ("conditional", "ensemble"):
                    expected[f"{case}/{profile}/LGA/{step}/{role}"] = {
                        "regime": case, "profile": profile, "name": f"LGA_dt{round(-np.log2(step))}",
                        "family": "LGA", "role": role, "step": step, "batch": 1}
            for role in ("conditional", "ensemble"):
                expected[f"{case}/{profile}/native-MC/{role}"] = {
                    "regime": case, "profile": profile, "name": "native-MC-dt10", "family": "native-MC",
                    "role": role, "step": plan["native_MC_step"],
                    "batch": runner.reference_configuration(case, study).paths_per_group}
    found = {row["slot"] for row in packet["records"]}
    found_loads = {(row["regime"], row["profile"], row["name"]) for row in packet["loading"]}
    if (not found <= set(expected) or not found_loads <= loads or len(found_loads) != len(packet["loading"])
            or (packet.get("complete") and (found != set(expected) or found_loads != loads))
            or packet.get("environment", {}).get("cpu_threads") != 1
            or packet.get("all_study_worker_leases_held") is not True
            or any(not np.isfinite(row["seconds"]) or row["seconds"] < 0 for row in packet["loading"])):
        raise ValueError("Complete literal timing and restoration allocations required")
    for row in packet["records"]:
        samples = np.asarray(row["samples_seconds"])
        components = row["warm_components"]
        repeats = (plan["native_MC_repeats"] if row["family"] == "native-MC" else
                   plan["ensemble_repeats"] if row["family"] in ("repaired", "untrained-repaired")
                   and row["role"] == "ensemble" else plan["conditional_repeats"])
        if (any(row.get(key) != value for key, value in expected[row["slot"]].items())
                or samples.shape != (repeats,) or not np.isfinite(samples).all() or np.any(samples < 0)
                or row["repeats"] != repeats or not components
                or set(row["cold_components"]) != set(components)
                or any(not np.isscalar(value) or not np.isfinite(value) or value < 0
                       for value in row["cold_components"].values())
                or any(np.shape(values) != (repeats,) or not np.isfinite(values).all()
                       or np.any(np.asarray(values) < 0) for values in components.values())
                or sum(row["cold_components"].values()) > row["first_prediction_seconds"]+1e-6
                or np.any(np.sum(list(components.values()), axis=0) > samples+1e-6)
                or not np.isfinite(row["first_prediction_seconds"]) or row["first_prediction_seconds"] < 0
                or not np.isclose(row["median_seconds"], np.median(samples), rtol=1e-12, atol=1e-15)):
            raise ValueError("Timing/component raw samples or medians changed")
    return packet


def benchmark(root, *, final_timing=False):
    """Measure literal queries while holding every known study worker lease.

    The explicit flag also attests that unrelated competing CPU work has ended;
    actual known fitting/reference/evaluation locks are acquired, not inferred
    from old PID or status files. Component samples come from the same calls.
    """
    if not final_timing:
        raise ValueError("Final isolated timing requires all heavy workers finished")
    root = Path(root).resolve()
    evidence = primary_records(root, require_timing=False)
    study, plan, lock = evidence["study"], evidence["plan"], evidence["lock"]
    with ExitStack() as leases:
        paths = [root/"comparison/execution/.accuracy.lock", root/"comparison/execution/.timing.lock",
                 root/"payoffs/execution/.prepare.lock", root/"payoffs/execution/.fit-lock.lock"]
        paths += [root/"payoffs/execution"/f".seed{seed}.lock" for seed in study.optimizer_seeds]
        paths += [root/"reference"/case/"execution/.generation.lock" for case in plan["cases"]]
        for path in paths:
            leases.enter_context(runner._exclusive(path))
        torch.set_num_threads(1)
        output = root/"comparison/timing.json"
        previous = _read_timing(root) if output.exists() else None
        if previous and previous.get("complete"):
            return previous
        packet = previous or {"identity": _timing_identity(root), "complete": False,
            "records": [], "loading": [], "resume_preparations": [],
            "environment": {"platform": platform.platform(), "numpy": np.__version__,
                            "torch": str(torch.__version__), "cpu_threads": 1},
            "final_timing_after_all_workers_finished": True, "all_study_worker_leases_held": True,
            "external_competing_work_finished_attested": True}
        done = {row["slot"] for row in packet["records"]}
        loaded = {(row["regime"], row["profile"], row["name"]) for row in packet["loading"]}
        tick = time.perf_counter()
        backend = _base.compile_native(root/"comparison/compiled_cache")
        packet["compiled_control"] = {"setup_seconds_this_process": time.perf_counter()-tick, **backend.metadata}

        def persist():
            packet["payload_sha256"] = stable_hash({k: v for k, v in packet.items() if k != "payload_sha256"})
            runner._json(output, packet)

        def save(slot, call, repeats, warmup, extra):
            if slot in done:
                return
            components = []

            def measured():
                mean, covariance, parts = call()
                if (not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                        or np.linalg.eigvalsh(covariance).min() < -1e-9):
                    raise ValueError("Invalid literal timing prediction; no repair or survivor filtering")
                components.append(parts)
                return mean, covariance

            started = time.perf_counter()
            measured()
            cold = time.perf_counter()-started
            sample = _base.timing(measured, repeats, warmup)
            warm = components[1+warmup:]
            if len(warm) != repeats:
                raise ValueError("Literal component sample count differs from total timing calls")
            packet["records"].append({"slot": slot, **extra, **sample,
                "first_prediction_seconds": cold, "cold_components": components[0],
                "warm_components": {key: [row[key] for row in warm] for key in warm[0]},
                "component_scope": "same actual calls as warm total samples; no subtraction of medians"})
            done.add(slot)
            persist()

        for case in plan["cases"]:
            cfg, bank = runner.reference_configuration(case, study), runner.payoff_bank(case, study)
            for profile in cfg.cases:
                probabilities = runner.profile_probabilities(cfg, profile)[0]
                rng = seeded_rng(SEED, KIND, "timing-only", case, profile)
                origins = rng.multinomial(cfg.n_particles, probabilities, size=max(plan["neural_conditional_batches"]))/cfg.n_particles
                for item in model_entries(lock, case, study.optimizer_seeds):
                    started = time.perf_counter()
                    model = _model(item, runner, study)
                    key = case, profile, item["name"]
                    if key not in loaded:
                        packet["loading"].append({"regime": case, "profile": profile,
                            "name": item["name"], "seconds": time.perf_counter()-started})
                        loaded.add(key)
                    common = {"regime": case, "profile": profile, "name": item["name"],
                        "family": item["family"], "seed": item["seed"], "bank": bank.name,
                        **{key: item[key] for key in ("proposal", "loss", "variant") if key in item}}
                    if any(slot.startswith(f"{case}/{profile}/{item['name']}/") for slot in done):
                        tick = time.perf_counter()
                        _base._prediction(model, origins[:1], cfg.query_times)
                        packet["resume_preparations"].append({**common,
                            "seconds": time.perf_counter()-tick, "scope": "restore warm horizon coefficients on resume"})
                    for batch in plan["neural_conditional_batches"]:
                        def conditional(m=model, b=batch):
                            tick = time.perf_counter()
                            mean, covariance = _base._prediction(m, origins[:b], cfg.query_times)
                            return mean, covariance, {"prediction_seconds": time.perf_counter()-tick}
                        save(f"{case}/{profile}/{item['name']}/conditional/{batch}", conditional,
                             plan["conditional_repeats"], plan["conditional_warmup"],
                             {**common, "role": "conditional", "batch": batch,
                              "cache_state": "B1 first query cold; larger batches warm",
                              "scope": "supplied-state three-time four means and full same-time covariance"})

                    def ensemble(m=model):
                        start = time.perf_counter()
                        initial = rng.multinomial(cfg.n_particles, probabilities, size=cfg.paths_per_group)/cfg.n_particles
                        generated = time.perf_counter()
                        mean, covariance = _base._prediction(m, initial, cfg.query_times)
                        predicted = time.perf_counter()
                        mean, covariance = _base.mixture_moments(mean, covariance, n_particles=cfg.n_particles)
                        return mean, covariance, {"initialization_seconds": generated-start,
                            "prediction_seconds": predicted-generated,
                            "aggregation_seconds": time.perf_counter()-predicted}
                    save(f"{case}/{profile}/{item['name']}/ensemble", ensemble,
                         plan["ensemble_repeats"], plan["ensemble_warmup"],
                         {**common, "role": "ensemble", "batch": cfg.paths_per_group,
                          "scope": "fresh independent IID histograms, neural prediction and total covariance"})
                for step in plan["LGA_max_steps"]:
                    for role in ("conditional", "ensemble"):
                        def gaussian(dt=step, ro=role):
                            tick = time.perf_counter()
                            covariance = _base.iid_scaled_covariance(probabilities) if ro == "ensemble" else None
                            prepared = time.perf_counter()
                            mean, covariance, _ = _base._checked_lga(
                                origins[0] if ro == "conditional" else probabilities,
                                cfg.context(profile), cfg.spec, cfg.query_times, dt, bank, covariance)
                            return mean, covariance, {"analytic_initial_covariance_seconds": prepared-tick,
                                "gaussian_evolution_seconds": time.perf_counter()-prepared}
                        save(f"{case}/{profile}/LGA/{step}/{role}", gaussian,
                            plan["conditional_repeats"], plan["conditional_warmup"],
                            {"regime": case, "profile": profile, "name": f"LGA_dt{round(-np.log2(step))}",
                             "family": "LGA", "role": role, "batch": 1, "step": step,
                             "scope": "literal marginal ODE including analytic IID covariance construction"})
                for role in ("conditional", "ensemble"):
                    def native(ro=role):
                        tick = time.perf_counter()
                        outcomes, _ = _base._native_mc(runner, cfg, bank, profile,
                            origins[0] if ro == "conditional" else None, probabilities, cfg.paths_per_group,
                            int(rng.integers(1, 2**62)), backend)
                        solved = time.perf_counter()
                        mean, covariance = _base.empirical(outcomes)
                        return mean, cfg.n_particles*covariance, {
                            "initial_noise_and_native_solve_seconds": solved-tick,
                            "aggregation_seconds": time.perf_counter()-solved}
                    save(f"{case}/{profile}/native-MC/{role}", native,
                         plan["native_MC_repeats"], plan["native_MC_warmup"],
                         {"regime": case, "profile": profile, "name": "native-MC-dt10", "family": "native-MC",
                          "role": role, "batch": cfg.paths_per_group, "step": plan["native_MC_step"],
                          "scope": "fresh initials and forcing, all stopped native paths and empirical moments"})
        packet["complete"] = True
        persist()
        return packet


bind_study(_base, __file__, KIND=KIND, SEED=SEED, _runner=lambda: runner, _immutable=_immutable,
           prepare=_comparison_prepare, _verified=_verified, model_entries=model_entries, _model=_model)
bind_study(_deployment, __file__, KIND=DEPLOYMENT_KIND, SEED=DEPLOYMENT_SEED,
           original=_base, runner=runner, _verified=_deployment_verified)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "benchmark"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--group", action="append")
    parser.add_argument("--final-timing", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "evaluate":
        result = evaluate(args.root, args.case, args.group)
    else:
        result = benchmark(args.root, final_timing=args.final_timing)
    print(json.dumps({"complete": True, "command": args.command,
                      "records": len(result) if isinstance(result, list) else None}))


if __name__ == "__main__":
    main()

