"""Independent completed-reference, native-candidate and IID-pool audit.

This is a post-outcome evidence check, never a model-selection step. Native
replays use the package's NumPy Euler implementation, physical cell-integrated
weights and an independent killed-integral reduction. No compiled reference
wrapper or fitted network is used to reproduce the selected native outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import i0  # noqa: E402

from analysis import audit_payoff_regimes as pretest  # noqa: E402
from analysis import evaluate_payoff_regimes as evaluator  # noqa: E402
from analysis import evaluate_payoff_regimes_deployment as deployment  # noqa: E402
from analysis import run_payoff_regimes as runner  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import simulate_native  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "independent-completed-harder-regime-native-evidence-audit-v1"
REPLAY_INDICES = (0, 7, 255, 1023)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _typed_array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _entropy(seed, *keys):
    return hashlib.sha256(json.dumps([seed, *keys], separators=(",", ":")).encode()).hexdigest()


def _chunk_digest(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def physical_weights(spec):
    """Exact averages of fixed physical sine/cosine functions over FV cells."""
    centers = (np.arange(spec.n_cells)+.5)*spec.h
    omega = 2*np.pi/spec.length
    return np.asarray([np.sin(3*omega*centers)*np.sinc(3/spec.n_cells),
                       np.cos(3*omega*centers)*np.sinc(3/spec.n_cells),
                       np.sin(omega*centers)*np.sinc(1/spec.n_cells),
                       np.cos(omega*centers)*np.sinc(1/spec.n_cells)])


def reduce_native_polygon(states, times, stopped_at, spec, horizon, query_times):
    """Independent frozen terminal / killed linear-segment integral readout."""
    states, times, stops = np.asarray(states), np.asarray(times), np.asarray(stopped_at)
    if (states.ndim != 3 or states.shape[-1] != spec.n_cells
            or times.shape != (states.shape[1],) or len(times) < 2 or times[0] != 0
            or not np.isfinite(states).all() or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any() or stops.shape != (len(states),)
            or np.isnan(stops).any() or (stops < 0).any()
            or (np.isfinite(stops) & (stops > times[-1]+1e-12)).any()
            or states.min() < -1e-12 or not np.allclose(states.sum(-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Every complete conservative stopped path is required")
    weights = physical_weights(spec)
    projected = states@weights.T
    duration = np.minimum(np.diff(times)[None], np.maximum(stops[:, None]-times[None, :-1], 0))
    integrated = np.concatenate((np.zeros((len(states), 1, 2)), np.cumsum(
        .5*duration[..., None]*(projected[:, :-1, 2:]+projected[:, 1:, 2:])/horizon, axis=1)), axis=1)
    result = np.concatenate((projected[..., :2], integrated), axis=-1)
    indices = np.searchsorted(times, np.asarray(query_times))
    if (indices >= len(times)).any() or not np.allclose(times[indices], query_times, atol=1e-12, rtol=0):
        raise ValueError("Every requested physical time must be present")
    return {"outcomes": result[:, indices], "stopped_at": stops,
            "path_minimum_mass": states.min(axis=(1, 2)),
            "maximum_mass_error": float(abs(states.sum(-1)-1).max())}


def _profile_probabilities(cfg, profile):
    nodes, weights = np.polynomial.legendre.leggauss(128)
    centers = (np.arange(cfg.n_cells)+.5)*cfg.spec.h
    x = centers[:, None]+cfg.spec.h/2*nodes
    alpha = .2 if profile == "nonuniform" else 0.
    return cfg.spec.h/2*(np.exp(alpha*np.cos(x))@weights)/(2*np.pi*i0(alpha))


def _stop_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        raise AssertionError("Stopping event membership changed")
    finite = np.isfinite(actual)
    return float(abs(actual[finite]-expected[finite]).max()) if finite.any() else 0.


def _check_path_values(values, stops, minima, count, horizon, mass_stop):
    if (values.shape != (count, 3, 4) or stops.shape != (count,) or minima.shape != (count,)
            or not np.isfinite(values).all() or not np.isfinite(minima).all() or minima.min() < -1e-12
            or np.isnan(stops).any() or (stops < 0).any()
            or (np.isfinite(stops) & (stops > horizon+1e-12)).any()):
        raise ValueError("All configured stopped realizations must remain in every native estimate")
    stopped = np.isfinite(stops) & (stops > 0)
    initially_stopped = stops == 0
    if ((stopped.any() and not np.allclose(minima[stopped], mass_stop, atol=2e-12, rtol=0))
            or (initially_stopped.any() and np.any(minima[initially_stopped] > mass_stop+2e-12))):
        raise ValueError("A stopped native path did not freeze at its first threshold crossing")


def _reference_shapes(cfg):
    n, r, f = cfg.paths_per_group, cfg.refinement_paths, len(cfg.reference_factors)
    return {"initial": ((n, cfg.n_cells), "f8"), "family_ids": ((n,), "S64"),
            "base_brownian_sha256": ((n,), "S64"), "drawn_brownian_sha256": ((n,), "S64"),
            "outcomes": ((n, 3, 4), "f8"), "stopped_at": ((n,), "f8"),
            "path_minimum_mass": ((n,), "f8"), "audit_outcomes": ((r, f, 3, 4), "f8"),
            "audit_stopped_at": ((r, f), "f8"), "audit_path_minimum_mass": ((r, f), "f8"),
            "audit_brownian_sha256": ((r, f), "S64")}


def _reference_replay(cfg, key, group, indices):
    """Reproduce addressed noise without runner.reference_draw or compiled code."""
    profile, role, anchor = key.split("__")
    anchor, factor = int(anchor), cfg.reference_factors[-1]
    initial, finest, families = [], [], []
    probabilities = group["probabilities"][:]
    conditional = (np.asarray(json.loads(group.attrs["conditional_initial_json"]))
                   if role == "conditional" else None)
    for index in indices:
        family = stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                              "reference-"+role, profile, anchor, index])
        initial.append(conditional if role == "conditional" else seeded_rng(
            cfg.seed, runner.STREAM, cfg.config_hash, "reference-"+role, profile, anchor,
            index, "initial").multinomial(cfg.n_particles, probabilities)/cfg.n_particles)
        finest.append(seeded_rng(cfg.seed, runner.STREAM, cfg.config_hash, "reference-"+role,
            profile, anchor, index, "physical-forcing").normal(
                0, np.sqrt(cfg.reference_step/factor),
                (round(cfg.horizon/cfg.reference_step)*factor, cfg.n_cells)))
        families.append(family)
    initial, finest = np.asarray(initial), np.asarray(finest)
    np.testing.assert_array_equal(initial, group["initial"][indices])
    if any(_array_sha(row) != group["drawn_brownian_sha256"][index].decode()
           for row, index in zip(finest, indices, strict=True)):
        raise AssertionError("Native reference fine Brownian forcing did not replay")
    records = []
    for level, refinement in enumerate(cfg.reference_factors):
        ratio = factor//refinement
        dw = finest.reshape(len(initial), -1, ratio, cfg.n_cells).sum(axis=2)
        dt = cfg.reference_step/refinement
        if any(_array_sha(row) != group["audit_brownian_sha256"][index, level].decode()
               for row, index in zip(dw, indices, strict=True)):
            raise AssertionError("Coupled native reference Brownian sum did not replay")
        if level == 0 and any(_array_sha(row) != group["base_brownian_sha256"][index].decode()
                              for row, index in zip(dw, indices, strict=True)):
            raise AssertionError("Native reference base Brownian sum did not replay")
        times = np.arange(dw.shape[1]+1)*dt
        packet = simulate_native(cfg.context(profile), cfg.spec, times, dt, 0, len(initial),
                                 initial, dw, family_ids=tuple(families))
        reduced = reduce_native_polygon(packet.states, times, packet.stopped_at,
                                         cfg.spec, cfg.horizon, cfg.query_times)
        stored = group["audit_outcomes"][indices, level]
        error = float(abs(reduced["outcomes"]-stored).max())
        minimum_error = float(abs(reduced["path_minimum_mass"]-group["audit_path_minimum_mass"][indices, level]).max())
        stop_error = _stop_error(reduced["stopped_at"], group["audit_stopped_at"][indices, level])
        if error > 3e-12 or stop_error > 3e-12 or minimum_error > 3e-12:
            raise AssertionError("Independent native Euler replay differs from saved reference")
        if level == 0:
            np.testing.assert_allclose(reduced["outcomes"], group["outcomes"][indices], atol=3e-12, rtol=0)
            _stop_error(reduced["stopped_at"], group["stopped_at"][indices])
        records.append({"reference_factor": refinement, "step": dt,
                        "maximum_outcome_error": error, "maximum_minimum_mass_error": minimum_error,
                        "maximum_stopping_time_error": stop_error,
                        "maximum_mass_error": reduced["maximum_mass_error"],
                        "stopped_replayed_paths": int(np.isfinite(packet.stopped_at).sum())})
    return {"group": key, "indices": indices, "independent_paths": len(indices),
            "levels": records, "physical_weights_independent_of_PayoffBank": True,
            "compiled_backend_called": False}


def _candidate_prefix_replay(cfg, key, packet, arrays, probabilities, conditional):
    seed = int(stable_hash([evaluator.SEED, packet["identity"]["regime"], key, "independent-candidate"])[:12], 16)
    if packet.get("rng") != {"seed": seed, "stream": [evaluator.KIND, "candidate-native-MC"],
                              "chunk": 32, "forcing": "independent-PCG64-normal"}:
        raise ValueError("Native candidate RNG metadata or chunk order changed")
    step, chunk, count = 2**-10, 32, min(2, cfg.paths_per_group)
    size = min(chunk, cfg.paths_per_group)
    rng = seeded_rng(seed, evaluator.KIND, "candidate-native-MC")
    initial = (np.broadcast_to(conditional, (size, cfg.n_cells)).copy() if conditional is not None else
               rng.multinomial(cfg.n_particles, probabilities, size=size)/cfg.n_particles)
    dw = rng.normal(scale=np.sqrt(step), size=(size, round(cfg.horizon/step), cfg.n_cells))
    times = np.arange(dw.shape[1]+1)*step
    native = simulate_native(cfg.context(key.split("__")[0]), cfg.spec, times, step, 0, count,
        initial[:count], dw[:count], family_ids=tuple(arrays["family_ids"][:count].astype(str)))
    reduced = reduce_native_polygon(native.states, times, native.stopped_at, cfg.spec, cfg.horizon, cfg.query_times)
    error = float(abs(reduced["outcomes"]-arrays["outcomes"][:count]).max())
    stop_error = _stop_error(reduced["stopped_at"], arrays["stopped_at"][:count])
    minimum_error = float(abs(reduced["path_minimum_mass"]-arrays["path_minimum_mass"][:count]).max())
    if max(error, stop_error, minimum_error) > 3e-12:
        raise AssertionError("Independent deployment native candidate prefix did not replay")
    return {"group": key, "prefix_paths": count, "source_seed": seed,
            "maximum_outcome_error": error, "maximum_stopping_time_error": stop_error,
            "maximum_minimum_mass_error": minimum_error,
            "first_chunk_initial_sha256": _array_sha(initial), "first_chunk_forcing_sha256": _array_sha(dw),
            "compiled_backend_called": False, "conditional_initial_fixed": conditional is not None}


def _audit_deployment_pools(root, study, seen, entropy):
    plan_path, pretest_path = root/"deployment/preregistration.json", root/"audit/deployment_pretest_independence.json"
    plan, proof = deployment.prepare(root), _read(pretest_path)
    if (not proof.get("complete") or proof.get("source_hash") != source_hash()
            or proof.get("deployment_plan_sha256") != _sha(plan_path)
            or proof.get("planned_families_disjoint") is not True
            or proof.get("address_streams_disjoint") is not True):
        raise ValueError("Complete pre-outcome deployment address proof required")
    index = _read(root/"deployment/accuracy_index.json")
    if not index.get("complete") or index.get("deployment_plan_sha256") != _sha(plan_path):
        raise ValueError("Complete independent deployment prediction inventory required")
    planned_path = root/"deployment/planned_initial_families.npz"
    preparation = _read(root/"deployment/preparation.json")
    if preparation.get("families_sha256") != _sha(planned_path):
        raise ValueError("Planned initial family inventory changed")
    records, total = [], 0
    with np.load(planned_path) as declared:
        for pool in plan["pools"]:
            regime, profile, count = pool["regime"], pool["profile"], pool["paths"]
            cfg = runner.reference_configuration(regime, study)
            path = root/"deployment/initials"/regime/(profile+".json")
            packet = _read(path)
            expected_identity = {"kind": deployment.KIND, "source_hash": source_hash(),
                "plan_sha256": _sha(plan_path), "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"),
                "pool": pool}
            if (not packet.get("complete") or packet.get("identity") != expected_identity
                    or _sha(packet["arrays"]["path"]) != packet["arrays"]["sha256"]):
                raise ValueError("Independent initial integration evidence changed")
            with np.load(packet["arrays"]["path"]) as saved:
                families = saved["family_ids"].astype(str)
                expected = np.asarray([stable_hash([deployment.SEED, deployment.KIND, regime, profile, i])
                                       for i in range(count)])
                np.testing.assert_array_equal(families, expected)
                np.testing.assert_array_equal(saved["family_ids"], declared[regime+"__"+profile])
                if _typed_array_sha(saved["family_ids"]) != pool["family_ids_array_sha256"]:
                    raise ValueError("Independent deployment pool family hash changed")
                current = set(families)
                if len(current) != count or current & seen:
                    raise ValueError("Independent deployment families overlap a previous native/learner/history family")
                seen.update(current)
                expected_seed = int(stable_hash([deployment.SEED, deployment.KIND, regime, profile,
                                                "independent-initial-pool"])[:12], 16)
                if pool["initial_seed"] != expected_seed:
                    raise ValueError("Independent initial pool seed changed")
                address = _entropy(expected_seed, deployment.KIND, "multinomial-initials")
                if address in entropy:
                    raise ValueError("Independent initial pool reuses native/learner entropy")
                entropy.add(address)
                probabilities = saved["probabilities"]
                np.testing.assert_allclose(probabilities, _profile_probabilities(cfg, profile), atol=2e-14, rtol=0)
                initial = seeded_rng(expected_seed, deployment.KIND, "multinomial-initials").multinomial(
                    cfg.n_particles, probabilities, size=count)/cfg.n_particles
                np.testing.assert_array_equal(initial, saved["initial"])
            records.append({"regime": regime, "profile": profile, "paths": count,
                            "families_all_replayed": True, "initial_histograms_all_replayed": True,
                            "record_sha256": _sha(path), "arrays_sha256": packet["arrays"]["sha256"],
                            "entropy_address": address})
            total += count
    if total != sum(p["paths"] for p in plan["pools"]):
        raise ValueError("Independent initial pool allocation changed")
    return {"planned_pretest_audit_sha256": _sha(pretest_path), "plan_sha256": _sha(plan_path),
            "complete_pools": len(records), "independent_families": total,
            "all_histograms_reproduced": True, "records": records}


def audit(root, *, output=None):
    """Read only complete outcomes; persist an immutable, bounded replay audit."""
    started = time.monotonic()
    root = Path(root).resolve()
    output = root/"audit/native_reference_candidate_audit.json" if output is None else Path(output)
    if output.exists():
        raise ValueError("Do not overwrite an already saved native evidence audit")
    study, design = runner._assert_frozen(root)
    fit_lock = runner.verify_fit_lock(root)
    comparison = evaluator.prepare(root)
    if not _read(root/"comparison/accuracy_index.json").get("complete"):
        raise ValueError("Every completed accuracy group is required before native evidence auditing")
    seen, learner_entropy = set(), set()
    for case in runner.case_names(study):
        cfg = runner.configuration(case, study)
        for role in ("train", "validation"):
            bank = runner.load_state_bank(root, case, role)
            current = set(bank["family_ids"])
            if current & seen:
                raise ValueError("Learner trajectory families overlap")
            seen.update(current)
            count = cfg.train_paths if role == "train" else cfg.validation_paths
            for profile in cfg.cases:
                for index in range(count):
                    for purpose in ("initial", "physical-forcing", "residual-horizons"):
                        learner_entropy.add(_entropy(cfg.seed, runner.STREAM, cfg.config_hash,
                                                     role, profile, 0, index, purpose))
    learner_count = len(seen)
    historical, history = pretest.historical_identifier_inventory(REPO/"runs", root, seen)
    seen.update(historical)
    entropy, reference_ids, base_hashes, drawn_hashes = set(learner_entropy), set(), set(), set()
    reference_records, reference_replays, candidate_records, candidate_replays = [], [], [], []
    selected_keys = {}
    for case_index, case in enumerate(runner.case_names(study)):
        cfg = runner.reference_configuration(case, study)
        manifest_path = root/"reference"/case/"manifest.json"
        manifest, path = _read(manifest_path), manifest_path.with_name("reference.h5")
        if (manifest.get("status") != "complete" or manifest.get("failed_paths") != 0
                or manifest.get("reference_sha256") != _sha(path)
                or manifest["identity"]["source_hash"] != source_hash()
                or manifest["identity"]["config_hash"] != cfg.config_hash):
            raise ValueError("Complete current-source native reference with unchanged raw SHA required")
        selected = ("uniform__conditional__0" if case_index % 2 == 0 else "nonuniform__ensemble__0")
        selected_keys[case] = selected
        shape_map, chunks_total = _reference_shapes(cfg), 0
        with h5py.File(path, "r") as store:
            if (json.loads(store.attrs["identity_json"]) != manifest["identity"]
                    or store.attrs["fit_inventory_hash"] != stable_hash(manifest["fit_lock"])
                    or set(store) != set(cfg.group_keys)|{"query_times", "reference_factors"}
                    or not np.array_equal(store["query_times"][:], cfg.query_times)
                    or not np.array_equal(store["reference_factors"][:], cfg.reference_factors)):
                raise ValueError("Reference identity/fit lock HDF5 header changed")
            expected_inventory = {"source_hash": source_hash(), "expected_models": len(design["slots"]),
                "models": [{key: row[key] for key in ("name", "checkpoint_path", "checkpoint_sha256", "complete")}
                           for row in fit_lock["models"]]}
            if manifest["fit_lock"] != expected_inventory:
                raise ValueError("Native reference was not unlocked by the complete declared fit inventory")
            for key in cfg.group_keys:
                group = store[key]
                profile, role, anchor = key.split("__")
                anchor = int(anchor)
                if set(group) != set(shape_map)|{"probabilities", "chunk_complete", "chunk_sha256", "chunk_seconds"}:
                    raise ValueError("Native reference group dataset inventory changed")
                for name, (shape, dtype) in shape_map.items():
                    if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
                        raise ValueError("Native reference array shape/dtype changed")
                initial = group["initial"][:]
                if (np.any(initial < 0) or not np.isfinite(initial).all()
                        or not np.allclose(initial.sum(-1), 1, atol=1e-12, rtol=0)
                        or not np.allclose(initial*cfg.n_particles, np.rint(initial*cfg.n_particles), atol=1e-10, rtol=0)):
                    raise ValueError("Native reference initial masses are not IID histogram states")
                probabilities = group["probabilities"][:]
                np.testing.assert_allclose(probabilities, _profile_probabilities(cfg, profile), atol=2e-14, rtol=0)
                if role == "conditional":
                    expected = seeded_rng(cfg.seed, runner.STREAM, cfg.config_hash, "held-out-anchor",
                        profile, anchor, 0, "initial").multinomial(cfg.n_particles, probabilities)/cfg.n_particles
                    if not np.array_equal(initial, np.broadcast_to(expected, initial.shape)):
                        raise ValueError("Conditional initial histogram changed within its native ensemble")
                    address = _entropy(cfg.seed, runner.STREAM, cfg.config_hash, "held-out-anchor",
                                       profile, anchor, 0, "initial")
                    if address in entropy:
                        raise ValueError("Held-out anchor reuses a learner/reference entropy address")
                    entropy.add(address)
                families = group["family_ids"].asstr()[:]
                expected = [stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                                        "reference-"+role, profile, anchor, index])
                            for index in range(cfg.paths_per_group)]
                if list(families) != expected:
                    raise ValueError("Native reference family addressing changed")
                current = set(families)
                if len(current) != cfg.paths_per_group or current & seen or current & reference_ids:
                    raise ValueError("Native reference families overlap learner/history/reference families")
                reference_ids.update(current)
                for index in range(cfg.paths_per_group):
                    for purpose in (("initial", "physical-forcing") if role == "ensemble" else ("physical-forcing",)):
                        address = _entropy(cfg.seed, runner.STREAM, cfg.config_hash, "reference-"+role,
                                           profile, anchor, index, purpose)
                        if address in entropy:
                            raise ValueError("Native reference reuses a physical RNG entropy address")
                        entropy.add(address)
                values, stops, minima = group["outcomes"][:], group["stopped_at"][:], group["path_minimum_mass"][:]
                _check_path_values(values, stops, minima, cfg.paths_per_group, cfg.horizon, cfg.spec.mass_stop)
                for level in range(len(cfg.reference_factors)):
                    _check_path_values(group["audit_outcomes"][:, level], group["audit_stopped_at"][:, level],
                                       group["audit_path_minimum_mass"][:, level], cfg.refinement_paths,
                                       cfg.horizon, cfg.spec.mass_stop)
                np.testing.assert_array_equal(group["audit_outcomes"][:, 0], values[:cfg.refinement_paths])
                np.testing.assert_array_equal(group["audit_stopped_at"][:, 0], stops[:cfg.refinement_paths])
                chunks = math.ceil(cfg.paths_per_group/cfg.chunk_paths)
                if not group["chunk_complete"][:].all() or len(group["chunk_complete"]) != chunks:
                    raise ValueError("Incomplete native reference chunks cannot produce an estimate")
                for chunk in range(chunks):
                    start = chunk*cfg.chunk_paths
                    arrays = {name: group[name][start:min(start+cfg.chunk_paths, shape[0])]
                              for name, (shape, _) in shape_map.items() if start < shape[0]}
                    if _chunk_digest(arrays) != group["chunk_sha256"][chunk].decode():
                        raise ValueError("Native reference committed chunk hash changed")
                if stable_hash(list(group["chunk_sha256"].asstr()[:])) != manifest["group_content_hashes"][key]:
                    raise ValueError("Native reference combined content hash changed")
                base, drawn = set(group["base_brownian_sha256"].asstr()[:]), set(group["drawn_brownian_sha256"].asstr()[:])
                if (len(base) != cfg.paths_per_group or len(drawn) != cfg.paths_per_group
                        or base & base_hashes or drawn & drawn_hashes):
                    raise ValueError("Different native reference families share forcing content")
                base_hashes.update(base)
                drawn_hashes.update(drawn)
                if int(np.isfinite(stops).sum()) != manifest["stopped_paths_by_group"][key]:
                    raise ValueError("Stopped paths were removed or changed in the reference manifest")
                if key == selected:
                    indices = [i for i in REPLAY_INDICES if i < cfg.refinement_paths]
                    if not indices:
                        indices = [0]
                    reference_replays.append({"case": case, **_reference_replay(cfg, key, group, indices)})
                reference_records.append({"case": case, "group": key, "paths": cfg.paths_per_group,
                    "refinement_paths": cfg.refinement_paths, "committed_chunks": chunks,
                    "stopped_paths": int(np.isfinite(stops).sum()),
                    "all_stopped_paths_retained": True,
                    "reference_file_sha256": manifest["reference_sha256"],
                    "manifest_sha256": _sha(manifest_path), "families_hash": stable_hash(sorted(current))})
                chunks_total += chunks
            if chunks_total != manifest["committed_chunks"]:
                raise ValueError("Reference manifest committed-chunk allocation changed")
    seen.update(reference_ids)
    candidate_ids = set()
    for case in runner.case_names(study):
        cfg = runner.reference_configuration(case, study)
        manifest = _read(root/"reference"/case/"manifest.json")
        with h5py.File(root/"reference"/case/"reference.h5", "r") as reference:
            for key in cfg.group_keys:
                path = root/"comparison/accuracy"/case/key/"native-MC-dt10.json"
                packet = _read(path)
                if (not packet.get("complete") or packet.get("independent_reference_and_candidate") is not True
                        or _sha(packet["arrays"]["path"]) != packet["arrays"]["sha256"]
                        or packet["identity"].get("source_hash") != source_hash()
                        or packet["identity"].get("regime") != case or packet["identity"].get("group") != key
                        or packet["identity"].get("reference_sha256") != manifest["reference_sha256"]
                        or packet["identity"].get("fit_lock_sha256") != _sha(root/"payoffs/fit_lock.json")
                        or packet["identity"].get("comparison_plan_sha256") != _sha(root/"comparison/preregistration.json")):
                    raise ValueError("Independent native candidate evidence changed")
                with np.load(packet["arrays"]["path"]) as saved:
                    arrays = {name: saved[name] for name in saved.files}
                families = arrays["family_ids"].astype(str)
                expected = [stable_hash([evaluator.SEED, case, key, "independent-candidate", i])
                            for i in range(cfg.paths_per_group)]
                if list(families) != expected:
                    raise ValueError("Native candidate family addressing changed")
                current = set(families)
                if len(current) != cfg.paths_per_group or current & seen or current & candidate_ids:
                    raise ValueError("Native candidate families overlap prior reference/learner/history families")
                candidate_ids.update(current)
                seed = int(stable_hash([evaluator.SEED, case, key, "independent-candidate"])[:12], 16)
                address = _entropy(seed, evaluator.KIND, "candidate-native-MC")
                if address in entropy:
                    raise ValueError("Native candidate physical forcing stream is not independent")
                entropy.add(address)
                _check_path_values(arrays["outcomes"], arrays["stopped_at"], arrays["path_minimum_mass"],
                                   cfg.paths_per_group, cfg.horizon, cfg.spec.mass_stop)
                mean = arrays["outcomes"].mean(axis=0)
                centered = arrays["outcomes"]-mean
                covariance = cfg.n_particles*np.einsum("nti,ntj->tij", centered, centered)/(cfg.paths_per_group-1)
                mean_error, cov_error = float(abs(mean-arrays["mean"]).max()), float(abs(covariance-arrays["scaled_covariance"]).max())
                if max(mean_error, cov_error) > 3e-12:
                    raise ValueError("Native candidate moments do not include every configured realization")
                if packet["diagnostics"].get("stopped_paths") != int(np.isfinite(arrays["stopped_at"]).sum()):
                    raise ValueError("Native candidate stopped-path count changed")
                if key == selected_keys[case]:
                    group = reference[key]
                    conditional = group["initial"][0] if "__conditional__" in key else None
                    candidate_replays.append({"case": case, **_candidate_prefix_replay(
                        cfg, key, packet, arrays, group["probabilities"][:], conditional)})
                candidate_records.append({"case": case, "group": key, "paths": cfg.paths_per_group,
                    "source_seed": seed, "entropy_address": address,
                    "stopped_paths": int(np.isfinite(arrays["stopped_at"]).sum()),
                    "maximum_mean_recalculation_error": mean_error,
                    "maximum_scaled_covariance_recalculation_error": cov_error,
                    "arrays_sha256": packet["arrays"]["sha256"], "record_sha256": _sha(path),
                    "all_stopped_paths_retained": True})
    seen.update(candidate_ids)
    pool_audit = _audit_deployment_pools(root, study, seen, entropy)
    result = {"kind": KIND, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": _sha(__file__), "source_hash": source_hash(),
              "training_preregistration_sha256": _sha(root/"payoffs/preregistration.json"),
              "comparison_preregistration_sha256": _sha(root/"comparison/preregistration.json"),
              "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"), "expected_models": len(fit_lock["models"]),
              "unique_learner_families": learner_count, "unique_historical_families": len(historical),
              "historical_identifier_files": len(history), "unique_reference_families": len(reference_ids),
              "unique_native_candidate_families": len(candidate_ids),
              "unique_base_reference_forcing_hashes": len(base_hashes),
              "unique_drawn_reference_forcing_hashes": len(drawn_hashes),
              "reference_candidate_learner_historical_overlap": 0,
              "physical_entropy_address_overlap": 0, "reference_groups": reference_records,
              "entropy_scope": "current learners, reference, native candidate and independent NN initial pools; historical families checked separately",
              "reference_replays": reference_replays, "native_candidate_groups": candidate_records,
              "native_candidate_replays": candidate_replays, "independent_NN_initial_pools": pool_audit,
              "maximum_native_reference_replay_error": max(
                  level["maximum_outcome_error"] for row in reference_replays for level in row["levels"]),
              "maximum_native_candidate_replay_error": max(row["maximum_outcome_error"] for row in candidate_replays),
              "seconds": time.monotonic()-started,
              "selection_using_audit": False, "survivor_conditioned_estimates": False,
              "independent_replay_backend": "nn_dk.native.simulate_native NumPy Euler; no compiled reference wrapper",
              "comparison_settings": {key: comparison[key] for key in ("native_MC_step", "native_MC_chunk")},
              "limitations": ["Addressed pseudorandom streams are not a mathematical independence proof.",
                              "Selected bounded replays do not independently simulate every trajectory.",
                              "All claims retain finite-Euler-reference bias and Monte Carlo uncertainty."]}
    _json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, output=args.output)
    print(json.dumps({key: result[key] for key in ("complete", "unique_reference_families",
                                                  "unique_native_candidate_families", "seconds")}))


if __name__ == "__main__":
    main()
