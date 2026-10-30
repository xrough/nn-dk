"""Pre-outcome independence audit for the new neural initial-law integration.

Only declared identifiers and RNG addresses are inspected.  This helper never
draws a histogram, physical forcing, reference outcome or bootstrap count.
It preserves the original paired-initial comparison as a separate diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis.audit_payoff_regimes import (  # noqa: E402
    _json,
    _sha,
    entropy_digest,
    historical_identifier_inventory,
    read_identifiers,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "independent-neural-initial-integration-pretest-address-audit-v1"


def expected_pool_contract(seed, kind, pool):
    """Reconstruct each formula independently of the deployment helper."""
    regime, profile, count = pool["regime"], pool["profile"], pool["paths"]
    initial_seed = int(stable_hash([seed, kind, regime, profile, "independent-initial-pool"])[:12], 16)
    bootstrap_seed = int(stable_hash([seed, kind, regime, profile, "initial-bootstrap"])[:12], 16)
    families = np.array([stable_hash([seed, kind, regime, profile, i]) for i in range(count)], dtype="S64")
    return {"initial_seed": initial_seed, "bootstrap_seed": bootstrap_seed,
            "initial_entropy_digest": entropy_digest(initial_seed, kind, "multinomial-initials"),
            "family_ids": families}


def verify_pool_contracts(plan, arrays, array_hash):
    if set(arrays) != {pool["key"] for pool in plan["pools"]}:
        raise ValueError("Planned initial family NPZ inventory differs from its declaration")
    all_ids, initial_addresses, bootstrap_seeds, records = set(), set(), set(), []
    for pool in plan["pools"]:
        expected = expected_pool_contract(plan["seed"], plan["kind"], pool)
        actual = np.asarray(arrays[pool["key"]])
        if (pool["initial_seed"] != expected["initial_seed"]
                or pool["bootstrap_seed"] != expected["bootstrap_seed"]
                or pool["initial_stream"] != [plan["kind"], "multinomial-initials"]
                or not np.array_equal(actual, expected["family_ids"])
                or array_hash(actual) != pool["family_ids_array_sha256"]):
            raise ValueError("A planned independent initial family or RNG formula changed")
        ids = set(actual.astype(str))
        if len(ids) != pool["paths"] or ids & all_ids:
            raise ValueError("Independent neural initial pools have duplicate trajectory families")
        address, bootstrap = expected["initial_entropy_digest"], expected["bootstrap_seed"]
        if address in initial_addresses or bootstrap in bootstrap_seeds:
            raise ValueError("Independent neural pools share an initial/bootstrap address")
        all_ids.update(ids)
        initial_addresses.add(address)
        bootstrap_seeds.add(bootstrap)
        records.append({"key": pool["key"], "regime": pool["regime"], "profile": pool["profile"],
                        "paths": pool["paths"], "family_ids_array_sha256": array_hash(actual),
                        "initial_seed": expected["initial_seed"], "initial_entropy_digest": address,
                        "bootstrap_default_rng_integer_seed": bootstrap})
    return all_ids, initial_addresses, bootstrap_seeds, records


def check_bootstrap_seed_independence(new_seeds, reference_seeds, candidate_seed):
    new_seeds, reference_seeds = set(new_seeds), set(reference_seeds)
    if new_seeds & reference_seeds or candidate_seed in new_seeds:
        raise ValueError("Neural integration and reference/candidate bootstrap streams overlap")
    return {"new_neural_integer_seeds": sorted(new_seeds),
            "reference_integer_seeds": sorted(reference_seeds),
            "native_candidate_integer_seed": candidate_seed,
            "NumPy_API": "np.random.default_rng(integer_seed), distinct from SHA-addressed physical streams",
            "bootstrap_count_drawn_during_audit": False}


def run(root):
    from analysis import evaluate_payoff_regimes as original
    from analysis import evaluate_payoff_regimes_deployment as deployment
    from analysis import run_payoff_regimes as runner

    root = Path(root).resolve()
    study, _ = runner._prepared(root)
    cases = runner.case_names(study)
    if any((root/"reference"/case/"reference.h5").exists() for case in cases):
        raise ValueError("Deployment independence audit must precede every reference outcome")
    if any((root/"deployment/initials").rglob("*.npz")):
        raise ValueError("Deployment independence audit must precede independent histogram draws")
    plan = deployment.prepare(root)
    preparation = runner._read(root/"deployment/preparation.json")
    plan_path = root/"deployment/preregistration.json"
    planned_path = root/"deployment/planned_initial_families.npz"
    if (preparation["initials_drawn"] is not False or plan["initials_drawn_during_preparation"] is not False
            or preparation["plan_sha256"] != _sha(plan_path)
            or preparation["families_sha256"] != _sha(planned_path)
            or plan["source_hash"] != source_hash()):
        raise ValueError("Independent deployment preparation identity changed")
    with np.load(planned_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    new_ids, new_addresses, new_bootstrap_seeds, pool_records = verify_pool_contracts(plan, arrays, deployment.array_sha)
    historical, history = historical_identifier_inventory(REPO/"runs", root, new_ids)
    learner_ids, learner_address_count = set(), 0
    future_family_digest, candidate_family_digest = hashlib.sha256(), hashlib.sha256()
    future_families, candidate_families, future_addresses, candidate_addresses = 0, 0, 0, 0
    reference_bootstrap_seeds = set()
    learner_files = []
    for case in cases:
        cfg = runner.configuration(case, study)
        for role in ("train", "validation"):
            path = root/"payoffs"/case/f"states_{role}.h5"
            ids, datasets = read_identifiers(path)
            if ids & new_ids:
                raise ValueError("New neural integration families overlap with learner families")
            learner_ids.update(ids)
            learner_files.append({"path": str(path.resolve()), "identifiers": len(ids),
                                  "datasets_read": datasets})
            count = cfg.train_paths if role == "train" else cfg.validation_paths
            for profile in cfg.cases:
                for index in range(count):
                    for purpose in ("initial", "physical-forcing", "residual-horizons"):
                        address = entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                                 role, profile, 0, index, purpose)
                        if address in new_addresses:
                            raise ValueError("New neural initial stream overlaps with learner streams")
                        learner_address_count += 1
        cfg = runner.reference_configuration(case, study)
        for key in cfg.group_keys:
            profile, role, anchor = key.split("__")
            anchor = int(anchor)
            reference_bootstrap_seeds.add(int(stable_hash([original.SEED, case, key])[:12], 16))
            if role == "conditional":
                address = entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                         "held-out-anchor", profile, anchor, 0, "initial")
                if address in new_addresses:
                    raise ValueError("New neural initial stream overlaps with a reference anchor stream")
                future_addresses += 1
            for index in range(cfg.paths_per_group):
                family = stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                                      "reference-"+role, profile, anchor, index])
                if family in new_ids:
                    raise ValueError("New neural initial family overlaps with future reference family")
                future_family_digest.update(family.encode())
                future_families += 1
                candidate_family = stable_hash([original.SEED, case, key, "independent-candidate", index])
                if candidate_family in new_ids:
                    raise ValueError("New neural initial family overlaps with future native candidate family")
                candidate_family_digest.update(candidate_family.encode())
                candidate_families += 1
                for purpose in (("initial", "physical-forcing") if role == "ensemble" else ("physical-forcing",)):
                    address = entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash,
                                             "reference-"+role, profile, anchor, index, purpose)
                    if address in new_addresses:
                        raise ValueError("New neural initial stream overlaps with a future reference stream")
                    future_addresses += 1
            candidate_seed = int(stable_hash([original.SEED, case, key, "independent-candidate"])[:12], 16)
            candidate_address = entropy_digest(candidate_seed, original.KIND, "candidate-native-MC")
            if candidate_address in new_addresses:
                raise ValueError("New neural initial stream overlaps with an independent native MC stream")
            candidate_addresses += 1
    bootstrap_audit = check_bootstrap_seed_independence(new_bootstrap_seeds, reference_bootstrap_seeds,
                                                        original.SEED+2)
    if (any((root/"reference"/case/"reference.h5").exists() for case in cases)
            or any((root/"deployment/initials").rglob("*.npz"))):
        raise ValueError("A held-out reference or histogram draw appeared during this pretest audit")
    output = root/"audit/deployment_pretest_independence.json"
    if output.exists():
        raise ValueError("Do not overwrite a completed deployment pretest snapshot")
    result = {"complete": True, "kind": KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_hash": source_hash(), "audit_script_sha256": _sha(__file__),
              "deployment_plan_sha256": _sha(plan_path),
              "deployment_helper_sha256": _sha(deployment.__file__),
              "planned_family_npz_sha256": _sha(planned_path),
              "original_pretest_audit_sha256": _sha(root/"audit/pretest_audit.json"),
              "training_preregistration_sha256": _sha(root/"payoffs/preregistration.json"),
              "original_comparison_preregistration_sha256": _sha(root/"comparison/preregistration.json"),
              "planned_families_disjoint": True, "address_streams_disjoint": True,
              "planned_independent_neural_initial_families": len(new_ids), "planned_pools": pool_records,
              "current_learner_families": len(learner_ids), "learner_identifier_files": learner_files,
              "historical_identifiers": len(historical), "historical_identifier_files": history,
              "future_reference_families_checked": future_families,
              "future_reference_family_enumeration_sha256": future_family_digest.hexdigest(),
              "future_native_candidate_families_checked": candidate_families,
              "future_native_candidate_family_enumeration_sha256": candidate_family_digest.hexdigest(),
              "learner_entropy_addresses_checked": learner_address_count,
              "future_reference_entropy_addresses_checked": future_addresses,
              "future_native_candidate_entropy_addresses_checked": candidate_addresses,
              "independent_neural_initial_entropy_addresses": sorted(new_addresses),
              "bootstrap_default_rng_seed_audit": bootstrap_audit,
              "no_histogram_reference_forcing_or_bootstrap_draw_during_audit": True,
              "no_selection_refitting_or_numerical_comparison_change": True,
              "all_final_reference_H5_and_independent_initial_NPZ_absent": True,
              "limitations": ["Declared pseudorandom stream separation is not a mathematical independence proof.",
                              "The new independent-initial ensemble comparison is primary; original paired results remain diagnostic."]}
    _json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.root)
    print(json.dumps({"complete": True, "planned_initial_families": result["planned_independent_neural_initial_families"],
                      "reference_families_checked": result["future_reference_families_checked"],
                      "native_candidate_families_checked": result["future_native_candidate_families_checked"]}))


if __name__ == "__main__":
    main()
