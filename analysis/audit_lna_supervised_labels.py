"""Replay a fixed 24-family prefix of saved learner labels with NumPy Euler.

Only previously committed training families are replayed. Missing producer
chunks leave the receipt partial. No label generation, fitting or final-data
access is performed, and existing study sources/metadata are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import run_lna_payoff_correction_study as producer  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import simulate_native  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "independent-numpy-replay-of-saved-supervised-LNA-labels-v2"
ANCHORS = (0, 40, 60, 100)
ATOL = 3e-12  # Existing independent native-reference replay tolerance.
RECEIPT = "execution/native_label_replay_audit.json"
ERROR_FIELDS = {"maximum_outcome_error", "stopping_time_error", "minimum_mass_error",
                "maximum_replayed_mass_error"}
HELPERS = ("analysis/run_lna_payoff_correction_study.py", "analysis/run_payoff_regimes.py",
           "src/nn_dk/native.py", "src/nn_dk/native_types.py", "src/nn_dk/payoff_learning.py",
           "src/nn_dk/types.py")


def allocation(plan):
    natural, stress = plan.train_natural_per_profile, plan.train_stress_per_profile
    indices = (0, natural, natural+stress, 2*natural+stress)
    return [{"case": case, "role": "train", "anchor": anchor, "continuation": 0}
            for case in producer.cases(plan) for anchor in indices]


def _identity(root):
    plan, _ = producer._assert_prepared(root)
    expected = allocation(plan)
    return plan, {"kind": KIND, "study_root": str(Path(root).resolve()), "source_hash": source_hash(),
        "script_sha256": producer._sha(__file__),
        "helpers": {name: producer._sha(REPO/name) for name in HELPERS},
        "study_plan_sha256": producer._sha(Path(root)/"plan.json"),
        "producer_preregistration_sha256": producer._sha(Path(root)/"data/preregistration.json"),
        "allocation": expected, "expected_rows": len(expected), "scientific_default_rows": 24,
        "allocation_rule": "first natural and first stress anchor per initial profile, every declared case",
        "selection": "fixed cases and indices before reading any selected outcome",
        "absolute_tolerance": ATOL, "relative_tolerance": 0.,
        "scope": "saved learner-family NumPy replay only; no new families, models, final data or continuum certificate"}


def _key(row):
    return row["case"], row["role"], row["anchor"], row["continuation"]


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _inputs(root, plan, item):
    case, role, anchor = item["case"], item["role"], item["anchor"]
    anchors, record = producer._anchors(root, case, role, plan)
    directory = Path(root)/"data"/case/role
    chunk = directory/"labels"/f"anchor{anchor}"/"chunk0"
    if not (chunk/"record.json").exists():
        return None
    loaded = producer._load_unit(chunk, producer._label_identity(plan, case, role, anchor, 0, record))
    if loaded is None:
        return None
    arrays, _ = loaded
    files = (directory/"anchors/arrays.npz", directory/"anchors/record.json",
             chunk/"arrays.npz", chunk/"record.json")
    return anchors, arrays, {str(path.resolve()): producer._sha(path) for path in files}


def _forcing(plan, item, profile):
    # Independently spell out the producer's frozen address formula. This is
    # the existing continuation zero, with its original physical forcing.
    seed = plan.train_label_seed
    keys = (producer.STREAM, plan.plan_hash, item["case"], "train", profile,
            item["anchor"], item["continuation"], "label-forcing")
    entropy = hashlib.sha256(json.dumps([seed, *keys], separators=(",", ":")).encode()).hexdigest()
    horizon = producer.query_times(item["case"], plan)[-1]
    steps = round(horizon/plan.label_step)
    increments = seeded_rng(seed, *keys).normal(size=(steps, plan.n_cells))*np.sqrt(plan.label_step)
    family = stable_hash([producer.STREAM, "continuation-family", entropy])
    return increments, entropy, family


def _stop(value):
    return float(value) if np.isfinite(value) else None


def _replay(root, plan, item, inputs):
    anchors, arrays, files = inputs
    anchor = item["anchor"]
    state = anchors["states"][anchor]
    profile = anchors["profiles"][anchor].decode()
    increments, entropy, family = _forcing(plan, item, profile)
    if (arrays["family_ids"][0].decode() != family
            or arrays["forcing_entropy_sha256"][0].decode() != entropy
            or arrays["brownian_sha256"][0].decode() != _array_sha(increments)):
        raise ValueError("Saved learner family, entropy address or Brownian SHA does not replay")
    times = np.arange(len(increments)+1)*plan.label_step
    tick, cpu = time.monotonic(), time.process_time()
    native = simulate_native(producer.context(item["case"], plan), producer.spec(item["case"], plan),
        times, plan.label_step, 0, 1, state[None], increments[None], family_ids=(family,))
    outcomes = producer.payoff_bank(item["case"], plan).evaluate_paths(native.states, times, native.stopped_at)
    indices = np.rint(producer.query_times(item["case"], plan)/plan.label_step).astype(int)
    reduced = outcomes[0, indices]
    saved_stop, native_stop = arrays["stopped_at"][0], native.stopped_at[0]
    same_stopped = bool(np.isfinite(saved_stop) == np.isfinite(native_stop))
    stop_error = abs(float(saved_stop-native_stop)) if np.isfinite(saved_stop) and np.isfinite(native_stop) else 0.
    errors = {"maximum_outcome_error": float(abs(reduced-arrays["outcomes"][0]).max()),
        "stopping_time_error": stop_error,
        "minimum_mass_error": float(abs(native.states.min()-arrays["minimum_mass"][0])),
        "maximum_replayed_mass_error": float(abs(native.states.sum(-1)-1).max())}
    passed = same_stopped and all(np.isfinite(value) and value <= ATOL for value in errors.values())
    return {**item, "verified": bool(passed), "profile": profile,
        "row_kind": anchors["row_kinds"][anchor].decode(),
        "parent_family_id": anchors["parent_family_ids"][anchor].decode(), "family_id": family,
        "initial_state_sha256": _array_sha(state), "forcing_entropy_sha256": entropy,
        "brownian_sha256": _array_sha(increments), "steps": len(increments), "step": plan.label_step,
        "query_times": producer.query_times(item["case"], plan).tolist(),
        "saved_outcomes_sha256": _array_sha(arrays["outcomes"][0]), "replayed_outcomes_sha256": _array_sha(reduced),
        "saved_stopped_at": _stop(saved_stop), "replayed_stopped_at": _stop(native_stop),
        "stopped_membership_matches": same_stopped,
        "saved_minimum_mass": float(arrays["minimum_mass"][0]), "replayed_minimum_mass": float(native.states.min()),
        "input_files": files, "errors": errors, "compiled_backend_called": False,
        "wall_seconds": time.monotonic()-tick, "cpu_seconds": time.process_time()-cpu}


def _write(root, packet):
    packet = {key: value for key, value in packet.items() if key != "content_sha256"}
    producer._json(Path(root)/RECEIPT, {**packet, "content_sha256": stable_hash(packet)})


def _prior_costs(root):
    folder = Path(root)/"execution/superseded_replay_audit_v1"
    if not folder.exists():
        return []
    manifest = producer._read(folder/"manifest.json")
    names = {"audit_lna_supervised_labels.py", "native_label_replay_audit.json"}
    if (manifest.get("kind") != "preserved-native-label-replay-audit-v1"
            or set(manifest.get("files", {})) != names
            or any(producer._sha(folder/name) != manifest["files"][name] for name in names)):
        raise ValueError("Preserved prior audit source or receipt checksum changed")
    old = producer._read(folder/"native_label_replay_audit.json")
    old_payload = {key: value for key, value in old.items() if key != "content_sha256"}
    wall = sum(attempt["wall_seconds"] for attempt in old["attempts"])
    cpu = sum(attempt["cpu_seconds"] for attempt in old["attempts"])
    if (old.get("content_sha256") != stable_hash(old_payload)
            or old["identity"]["script_sha256"] != manifest["files"]["audit_lna_supervised_labels.py"]
            or manifest["paid_wall_seconds"] != wall or manifest["paid_cpu_seconds"] != cpu
            or manifest["verified_rows"] != len(old["records"]) or manifest["complete"] != old["complete"]):
        raise ValueError("Preserved prior audit identity, allocation or paid costs changed")
    return [{"manifest_path": str((folder/"manifest.json").resolve()),
        "manifest_sha256": producer._sha(folder/"manifest.json"),
        "receipt_path": str((folder/"native_label_replay_audit.json").resolve()),
        "receipt_sha256": manifest["files"]["native_label_replay_audit.json"],
        "auditor_source_path": str((folder/"audit_lna_supervised_labels.py").resolve()),
        "auditor_source_sha256": manifest["files"]["audit_lna_supervised_labels.py"],
        "verified_rows": len(old["records"]), "complete": old["complete"],
        "wall_seconds": wall, "cpu_seconds": cpu,
        "scope": "superseded v1 audit paid overhead; its rows are preserved and are not relabelled as v2 verification"}]


def verify(root, *, require_complete=True):
    root = Path(root).resolve()
    plan, identity = _identity(root)
    saved = producer._read(root/RECEIPT)
    payload = {key: value for key, value in saved.items() if key != "content_sha256"}
    if saved.get("identity") != identity or saved.get("content_sha256") != stable_hash(payload):
        raise ValueError("Native-label audit source, plan, allocation or receipt checksum changed")
    if saved.get("prior_audit_costs") != _prior_costs(root):
        raise ValueError("Prior audit evidence or paid overhead changed")
    if (saved.get("new_scientific_families") != 0 or saved.get("final_outcomes_or_models_accessed") is not False
            or not isinstance(saved.get("attempts"), list) or not saved["attempts"]):
        raise ValueError("Complete separate audit cost and model-free scope declarations required")
    if any(attempt.get("finished") is not True or any(not isinstance(attempt.get(key), (float, int))
               or not np.isfinite(attempt[key]) or attempt[key] < 0 for key in ("wall_seconds", "cpu_seconds"))
           for attempt in saved["attempts"]):
        raise ValueError("Incomplete measured replay-audit invocation costs")
    expected = {_key(item): item for item in identity["allocation"]}
    actual = [_key(row) for row in saved["records"]]
    if len(set(actual)) != len(actual) or not set(actual) <= set(expected):
        raise ValueError("Duplicate or undeclared learner-label replay allocation")
    for row in saved["records"]:
        inputs = _inputs(root, plan, expected[_key(row)])
        if inputs is None or inputs[2] != row["input_files"]:
            raise ValueError("Previously audited raw learner inputs are missing or changed")
        anchors, arrays, _ = inputs
        profile = anchors["profiles"][row["anchor"]].decode()
        increments, entropy, family = _forcing(plan, row, profile)
        if (row["family_id"] != family or row["forcing_entropy_sha256"] != entropy
                or row["brownian_sha256"] != _array_sha(increments)
                or arrays["family_ids"][0].decode() != family
                or arrays["forcing_entropy_sha256"][0].decode() != entropy
                or arrays["brownian_sha256"][0].decode() != row["brownian_sha256"]
                or row["saved_outcomes_sha256"] != _array_sha(arrays["outcomes"][0])
                or row["initial_state_sha256"] != _array_sha(anchors["states"][row["anchor"]])
                or row["steps"] != len(increments) or row["step"] != plan.label_step
                or row["query_times"] != producer.query_times(row["case"], plan).tolist()
                or row["profile"] != profile or row["row_kind"] != anchors["row_kinds"][row["anchor"]].decode()
                or row["parent_family_id"] != anchors["parent_family_ids"][row["anchor"]].decode()
                or row["saved_stopped_at"] != _stop(arrays["stopped_at"][0])
                or row["saved_minimum_mass"] != float(arrays["minimum_mass"][0])
                or set(row["errors"]) != ERROR_FIELDS
                or row["compiled_backend_called"] is not False or not row["verified"]
                or not row["stopped_membership_matches"]
                or any(not np.isfinite(v) or v > ATOL or v < 0 for v in row["errors"].values())):
            raise ValueError("Learner replay address, forcing, control scope or numeric verification changed")
    complete = len(actual) == len(expected) and all(row["verified"] for row in saved["records"])
    if saved["complete"] != complete or require_complete and not complete:
        raise ValueError("All predetermined learner-label replays must verify before audit completion")
    return saved


def run(root, *, max_cases=None):
    root = Path(root).resolve()
    tick, cpu = time.monotonic(), time.process_time()
    plan, identity = _identity(root)
    declared = list(dict.fromkeys(item["case"] for item in identity["allocation"]))
    if max_cases is not None and (type(max_cases) is not int or not 1 <= max_cases <= len(declared)):
        raise ValueError("A positive prefix of declared cases is required")
    with producer.utilities._exclusive(root/"execution/.native-label-replay.lock"):
        if (root/RECEIPT).exists():
            saved = verify(root, require_complete=False)
        else:
            saved = {"identity": identity, "complete": False, "records": [], "attempts": [],
                     "prior_audit_costs": _prior_costs(root),
                     "new_scientific_families": 0, "final_outcomes_or_models_accessed": False}
        packet = {key: value for key, value in saved.items() if key != "content_sha256"}
        done = {_key(row) for row in packet["records"]}
        attempt = {"complete": False, "finished": False, "newly_verified": 0, "max_cases": max_cases,
                   "pid": os.getpid(), "started_unix": time.time()}
        packet["attempts"].append(attempt)
        try:
            allowed = set(declared if max_cases is None else declared[:max_cases])
            for item in identity["allocation"]:
                if item["case"] not in allowed or _key(item) in done:
                    continue
                inputs = _inputs(root, plan, item)
                if inputs is None:
                    continue
                row = _replay(root, plan, item, inputs)
                packet["records"].append(row)
                if not row["verified"]:
                    raise AssertionError("Independent NumPy replay exceeds the predeclared tolerance")
                attempt["newly_verified"] += 1
                if len(packet["records"]) < identity["expected_rows"]:
                    _write(root, packet)
            packet["records"].sort(key=lambda row: list(map(_key, identity["allocation"])).index(_key(row)))
            packet["complete"] = len(packet["records"]) == identity["expected_rows"]
            attempt["complete"] = packet["complete"]
        finally:
            attempt.update(finished=True, finished_unix=time.time(),
                wall_seconds=time.monotonic()-tick, cpu_seconds=time.process_time()-cpu,
                scope="whole replay-audit invocation including integrity, RNG reconstruction and receipt updates; separate from producer preparation")
            _write(root, packet)
    return verify(root, require_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.max_cases is not None and args.stage != "run" or args.allow_partial and args.stage != "verify":
        parser.error("max-cases only for run; allow-partial only for verify")
    packet = run(args.root, max_cases=args.max_cases) if args.stage == "run" else verify(
        args.root, require_complete=not args.allow_partial)
    print(json.dumps({"complete": packet["complete"], "verified": len(packet["records"]),
                      "expected": packet["identity"]["expected_rows"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
