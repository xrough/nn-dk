"""Frozen-mean gradient/source diagnostics with fresh native JVP families.

This is a diagnostic of the finite-Euler conditional mean, not a continuum
derivative certificate. Historical final reference outcomes are never opened.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.compiled_native import compile_native  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import native_coefficients  # noqa: E402
from nn_dk.observations import incidence_matrix  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "fresh-native-conditional-mean-face-gradient-source-audit-v1"
HELPERS = ("compiled_native.py", "native_euler_control.cpp", "run_payoff_boundary_regimes.py",
           "run_payoff_regimes.py", "payoff_study_adapter.py")


@dataclass(frozen=True)
class GradientPlan:
    cases: tuple[str, ...] = ()
    anchors_per_profile: int = 2
    query_fractions: tuple[float, ...] = (.25, .5, 1.)
    paths: int = 512
    chunk_paths: int = 16
    step: float = 2**-10
    refinement_paths: int = 64
    refinement_factor: int = 2
    n_bootstrap: int = 128
    seed: int = 2026103001

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "query_fractions", tuple(self.query_fractions))
        integers = ("anchors_per_profile", "paths", "chunk_paths", "refinement_factor", "n_bootstrap", "seed")
        if any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in integers):
            raise ValueError("Positive integer allocations required")
        if (self.paths < 4 or self.paths % 2 or self.paths % self.chunk_paths
                or type(self.refinement_paths) is not int or self.refinement_paths < 0
                or self.refinement_paths > self.paths or self.refinement_paths % self.chunk_paths
                or (self.refinement_paths and (self.refinement_paths < 4 or self.refinement_paths % 2))
                or self.refinement_factor < 2 or self.n_bootstrap < 2
                or not np.isfinite(self.step) or self.step <= 0
                or not self.query_fractions or not np.isfinite(self.query_fractions).all()
                or min(self.query_fractions) <= 0 or max(self.query_fractions) > 1
                or any(a >= b for a, b in zip(self.query_fractions, self.query_fractions[1:]))
                or len(set(self.cases)) != len(self.cases)):
            raise ValueError("Even families, complete chunks, nested refinement and ordered queries required")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _inventory(study_root, cases):
    from analysis import run_payoff_boundary_regimes as runner

    study, _ = runner._assert_frozen(study_root)
    lock = runner.verify_fit_lock(study_root)
    names = tuple(cases) or tuple(runner.case_names(study))
    if not names or set(names) - set(runner.case_names(study)):
        raise ValueError("Unknown or empty case inventory")
    models = [row for row in lock["models"] if row["case"] in names]
    if any({row["seed"] for row in models if row["case"] == case} != {0, 1, 2}
           or len([row for row in models if row["case"] == case]) != 3 for case in names):
        raise ValueError("All three frozen optimizer seeds must be retained per case")
    return runner, study, names, models


def select_anchors(bank, metadata, spec, per_profile):
    """Minimum and median active mass ranks; current state/metadata only."""
    states = np.asarray(bank["states"])
    profiles = np.asarray(bank["case_ids"], dtype=str)
    families = np.asarray(bank["family_ids"], dtype=str)
    times = np.asarray(bank["snapshot_times"])
    active = np.asarray(metadata["physical_active"], dtype=bool) & (states > spec.mass_stop).all(-1)
    indices, fractions, positions, candidate_counts = [], [], [], []
    for profile in sorted(set(profiles)):
        unique = {}
        for i in np.flatnonzero((profiles == profile) & active):
            unique.setdefault((families[i], float(times[i])), int(i))
        candidates = np.array(list(unique.values()), dtype=int)
        if len(candidates) < per_profile:
            raise ValueError("Insufficient distinct active validation anchors")
        ordered = candidates[np.lexsort((candidates, states[candidates].min(-1)))]
        ranks = np.linspace(0., .5, per_profile)
        rank_indices = np.rint(ranks*(len(ordered)-1)).astype(int)
        indices.extend(ordered[rank_indices].tolist())
        fractions.extend(ranks.tolist())
        positions.extend(rank_indices.tolist())
        candidate_counts.extend([len(ordered)]*len(ranks))
    if len(set(indices)) != len(indices):
        raise ValueError("Anchor selection duplicates an origin")
    chosen = np.array(indices, dtype=int)
    return {"states": states[chosen], "indices": chosen, "profiles": profiles[chosen],
            "family_ids": families[chosen], "snapshot_times": times[chosen],
            "selection_rank_fraction": np.asarray(fractions), "minimum_mass_rank": np.asarray(positions),
            "candidate_count": np.asarray(candidate_counts)}


def prepare(root, study_root, plan=None):
    """Freeze all checkpoints, state-only anchors, source and RNG allocations."""
    root, study_root = Path(root).resolve(), Path(study_root).resolve()
    plan = GradientPlan() if plan is None else plan
    if not isinstance(plan, GradientPlan):
        raise TypeError("GradientPlan required")
    runner, study, cases, models = _inventory(study_root, plan.cases)
    identity = {"kind": KIND, "plan": asdict(plan), "cases": list(cases), "models": models,
                "study_root": str(study_root), "source_hash": source_hash(),
                "script_sha256": sha(__file__),
                "helpers": {name: sha(Path(__file__).with_name(name)) for name in HELPERS},
                "fit_lock_sha256": sha(study_root/"payoffs/fit_lock.json"),
                "anchor_policy": "profile-wise unique active origins; minimum-to-median mass ranks",
                "stream": "new-mean-gradient-diagnostic-independent-continuations-v1",
                "old_final_reference_outcomes_opened": False, "refitting_or_selection": False}
    identity = json.loads(json.dumps(identity))
    path = root/"plan.json"
    if path.exists():
        saved = _validate(root)
        if saved["identity"] != identity:
            raise ValueError("Frozen diagnostic identity changed")
        return saved
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty diagnostic directory")
    from analysis.audit_payoff_regimes_validation import validation_metadata

    anchors = {}
    for case in cases:
        cfg = runner.configuration(case, study)
        query = cfg.horizon*np.asarray(plan.query_fractions)
        if not np.allclose(query/plan.step, np.rint(query/plan.step), atol=1e-10, rtol=0):
            raise ValueError("Queries must lie on the declared native grid")
        bank = runner.load_state_bank(study_root, case, "validation")
        values = select_anchors(bank, validation_metadata(bank), cfg.spec, plan.anchors_per_profile)
        anchor_path = root/"anchors"/(case+".npz")
        npz(anchor_path, values)
        anchors[case] = {"path": str(anchor_path), "sha256": sha(anchor_path),
                         "rows": len(values["states"]), "queries": query.tolist(),
                         "state_bank_path": str(bank["path"]), "state_bank_sha256": sha(bank["path"])}
    result = {"identity": identity, "anchors": anchors}
    result["identity_hash"] = stable_hash(result)
    write(path, result)
    return result


def _validate(root):
    saved = read(Path(root)/"plan.json")
    identity = saved["identity"]
    if (identity["kind"] != KIND
            or stable_hash({"identity": identity, "anchors": saved["anchors"]}) != saved["identity_hash"]
            or identity["source_hash"] != source_hash() or identity["script_sha256"] != sha(__file__)
            or any(sha(Path(__file__).with_name(name)) != digest for name, digest in identity["helpers"].items())
            or sha(Path(identity["study_root"])/"payoffs/fit_lock.json") != identity["fit_lock_sha256"]):
        raise ValueError("Frozen source, helpers or fit lock changed")
    for item in identity["models"]:
        if (sha(item["checkpoint_path"]) != item["checkpoint_sha256"]
                or sha(item["metadata_path"]) != item["metadata_sha256"]):
            raise ValueError("Frozen checkpoint changed")
    for item in saved["anchors"].values():
        if sha(item["path"]) != item["sha256"] or sha(item["state_bank_path"]) != item["state_bank_sha256"]:
            raise ValueError("State-only anchors or validation bank changed")
    return saved


def payoff_jvp(bank, packet, dt, queries):
    """Differentiate the complete frozen-terminal/killed-integral polygon.

    The exposure derivative at the first crossing is essential. Differentiating
    only the stopped states silently omits the running reward's boundary term.
    """
    states, tangent = np.asarray(packet["states"]), np.asarray(packet["tangent"])
    stops, dstops = np.asarray(packet["stopped_at"]), np.asarray(packet["stopped_at_tangent"])
    times = np.arange(states.shape[1])*dt
    indices = np.rint(np.asarray(queries)/dt).astype(int)
    if (states.shape != tangent.shape or states.ndim != 3 or states.shape[-1] != bank.n_cells
            or not np.isfinite(tangent).all() or not np.isfinite(dstops).all()
            or stops.shape != (len(states),) or dstops.shape != stops.shape
            or (indices < 0).any() or (indices >= len(times)).any()
            or not np.allclose(indices*dt, queries, atol=1e-12, rtol=0)):
        raise ValueError("Compatible complete native JVP polygons and grid queries required")
    values = tangent @ bank.terminal_weights.T
    exposure = np.minimum(dt, np.maximum(stops[:, None]-times[None, :-1], 0.))
    dexposure = np.where((stops[:, None] > times[None, :-1])
                         & (stops[:, None] < times[None, 1:]), dstops[:, None], 0.)
    running, derivative = states @ bank.running_weights.T, tangent @ bank.running_weights.T
    segments = .5*(exposure[..., None]*(derivative[:, :-1]+derivative[:, 1:])
                     + dexposure[..., None]*(running[:, :-1]+running[:, 1:]))
    values[:, 1:] += np.cumsum(segments, axis=1)
    return values[:, indices]


def source_estimates(gradients):
    """Cross-fit source avoids the positive quadratic Monte Carlo plug-in bias."""
    values = np.asarray(gradients, dtype=float)
    if values.ndim != 4 or len(values) < 4 or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError("Even complete [family,query,output,face] gradients required")
    mean = values.mean(0)
    left, right = values[:len(values)//2].mean(0), values[len(values)//2:].mean(0)
    plugin = mean @ mean.swapaxes(-1, -2)
    cross = .5*(left @ right.swapaxes(-1, -2)+right @ left.swapaxes(-1, -2))
    return mean, plugin, cross


def bootstrap_sources(gradients, count, seed):
    values = np.asarray(gradients)
    source_estimates(values)
    rng = np.random.default_rng(seed)
    half = len(values)//2
    means, sources = [], []
    for _ in range(count):
        indices = np.concatenate((rng.integers(0, half, half), half+rng.integers(0, half, half)))
        mean, _, cross = source_estimates(values[indices])
        means.append(mean)
        sources.append(cross)
    return np.asarray(means), np.asarray(sources)


def refinement_comparison(coarse, fine, n_bootstrap, seed):
    """Pair both grids within each family, including both cross-Gram halves."""
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if coarse.shape != fine.shape:
        raise ValueError("Paired coarse and fine family gradients must have equal shapes")
    coarse_mean, _, coarse_source = source_estimates(coarse)
    fine_mean, _, fine_source = source_estimates(fine)
    coarse_boot, coarse_source_boot = bootstrap_sources(coarse, n_bootstrap, seed)
    fine_boot, fine_source_boot = bootstrap_sources(fine, n_bootstrap, seed)
    difference, source_difference = fine_mean-coarse_mean, fine_source-coarse_source
    boot_difference, boot_source_difference = fine_boot-coarse_boot, fine_source_boot-coarse_source_boot
    arrays = {"gradient_difference": difference, "cross_source_difference": source_difference,
              "paired_bootstrap_gradient_difference": boot_difference,
              "paired_bootstrap_cross_source_difference": boot_source_difference}
    summary = {"families": len(coarse), "paired_whole_family_resampling": True,
        "gradient_frobenius_change_by_query": np.linalg.norm(difference, axis=(-2, -1)).tolist(),
        "gradient_change_pointwise_interval": np.quantile(np.linalg.norm(
            boot_difference, axis=(-2, -1)), [.025, .975], axis=0).T.tolist(),
        "source_cross_frobenius_change_by_query": np.linalg.norm(source_difference, axis=(-2, -1)).tolist(),
        "source_change_pointwise_interval": np.quantile(np.linalg.norm(
            boot_source_difference, axis=(-2, -1)), [.025, .975], axis=0).T.tolist(),
        "interpretation": "paired finite-grid derivative/source difference only; remaining time and stopping bias unresolved"}
    return summary, arrays


def _draw(saved, plan, case, anchor, start, count, steps, factor, cells):
    noise, ids = [], []
    for family in range(start, start+count):
        key = [KIND, saved["identity_hash"], "fresh-native-continuation", case, anchor, family]
        rng = seeded_rng(plan.seed, *key)
        noise.append(rng.normal(0, math.sqrt(plan.step/factor), (steps*factor, cells)))
        ids.append(stable_hash([plan.seed, *key]))
    return np.asarray(noise), np.asarray(ids)


def _native_chunk(saved, plan, case, anchor, state, cfg, bank, queries, start, backend):
    count = plan.chunk_paths
    refined = start < plan.refinement_paths
    factor = plan.refinement_factor if refined else 1
    steps = round(queries[-1]/plan.step)
    fine, families = _draw(saved, plan, case, anchor, start, count, steps, factor, cfg.n_cells)
    noise = fine.reshape(count, steps, factor, cfg.n_cells).sum(2)
    initial = np.broadcast_to(state, (count, cfg.n_cells)).copy()
    coefficients = native_coefficients(state, cfg.context("uniform"), cfg.spec)
    faces = incidence_matrix(cfg.n_cells)*coefficients.face_amplitudes/math.sqrt(cfg.n_particles)
    result = {"family_ids": families}
    levels = [("gradient", noise, plan.step)]
    if refined:
        levels.append(("refined_gradient", fine, plan.step/factor))
    for name, increments, dt in levels:
        directions = []
        for face in range(cfg.n_cells):
            packet = backend(initial, increments, cfg.context("uniform"), cfg.spec, dt,
                             initial_tangent=faces[:, face])
            derivative = payoff_jvp(bank, packet, dt, queries)
            directions.append(derivative*math.sqrt(cfg.n_particles)/bank.scales)
            if face == 0 and name == "gradient":
                result["stopped_at"] = packet["stopped_at"]
                direction = faces[:, 0]
                nonzero = abs(direction) > 0
                epsilon = min(1e-4, .1*float(((state-cfg.spec.mass_stop)[nonzero]
                                               /abs(direction[nonzero])).min()))
                finite = []
                for sign in (1, -1):
                    changed = backend(initial+sign*epsilon*direction, increments,
                                      cfg.context("uniform"), cfg.spec, dt)
                    all_values = bank.evaluate_paths(changed["states"],
                        np.arange(steps+1)*dt, changed["stopped_at"])
                    finite.append(all_values[:, np.rint(queries/dt).astype(int)])
                estimate = (finite[0]-finite[1])/(2*epsilon)
                result["finite_difference_error"] = estimate-derivative
                result["finite_difference_epsilon"] = np.array(epsilon)
        result[name] = np.stack(directions, -1)
    return result


def _controls(model, state):
    point = torch.tensor(state[None], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    actual = model._mean_jet(point, zero).face_derivatives[0].detach().numpy()
    coefficients = native_coefficients(state, model.context, model.spec)
    expected = model.normalized_weights.numpy() @ (incidence_matrix(model.spec.n_cells)
                                                   *coefficients.face_amplitudes)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    absorbed = state.copy()
    excess = absorbed[0]-model.spec.mass_stop
    absorbed[0] = model.spec.mass_stop
    absorbed[1] += excess
    jet = model._mean_jet(torch.tensor(absorbed[None]), torch.tensor([model.horizon]))
    np.testing.assert_array_equal(jet.face_derivatives.detach().numpy(), 0.)
    return {"zero_time_face_maximum_error": float(abs(actual-expected).max()),
            "absorbed_face_source_exact_zero": True}


def _summary(root, saved, plan, runner, study, chunks):
    rows, evidence = [], {}
    for case, item in saved["anchors"].items():
        anchors = np.load(item["path"])
        queries = np.asarray(item["queries"])
        for anchor, state in enumerate(anchors["states"]):
            selected = [row for row in chunks if row["case"] == case and row["anchor"] == anchor]
            arrays = [np.load(row["path"]) for row in selected]
            gradient = np.concatenate([v["gradient"] for v in arrays])
            family_ids = np.concatenate([v["family_ids"] for v in arrays])
            if len(set(family_ids.tolist())) != plan.paths:
                raise ValueError("Independent continuation family allocation duplicated")
            mean, plugin, cross = source_estimates(gradient)
            seed = int(stable_hash([plan.seed, case, anchor, "bootstrap"])[:15], 16)
            boot_mean, boot_source = bootstrap_sources(gradient, plan.n_bootstrap, seed)
            key = f"{case}_anchor{anchor}"
            evidence[key+"_reference_mean_gradient"] = mean
            evidence[key+"_reference_plugin_source"] = plugin
            evidence[key+"_reference_cross_source"] = cross
            evidence[key+"_bootstrap_mean_gradient"] = boot_mean
            evidence[key+"_bootstrap_cross_source"] = boot_source
            refinement_rows = [v for v in arrays if "refined_gradient" in v]
            refinement = None
            if refinement_rows:
                refinement, refinement_arrays = refinement_comparison(
                    np.concatenate([v["gradient"] for v in refinement_rows]),
                    np.concatenate([v["refined_gradient"] for v in refinement_rows]),
                    plan.n_bootstrap, seed)
                evidence.update({key+"_refinement_"+name: value for name, value in refinement_arrays.items()})
            for entry in saved["identity"]["models"]:
                if entry["case"] != case:
                    continue
                model = runner.load_declared_model(entry)
                controls = _controls(model, state)
                point = torch.tensor(np.broadcast_to(state, (len(queries), len(state))).copy())
                with torch.no_grad():
                    nn_gradient = model._mean_jet(point, torch.tensor(queries)).face_derivatives.numpy()
                nn_source = nn_gradient @ nn_gradient.swapaxes(-1, -2)
                difference = nn_gradient-mean
                source_distance = np.linalg.norm(nn_source-cross, axis=(-2, -1))
                source_interval = np.quantile(np.linalg.norm(nn_source[None]-boot_source,
                                                             axis=(-2, -1)), [.025, .975], axis=0).T
                evidence[key+f"_seed{entry['seed']}_gradient"] = nn_gradient
                rows.append({"case": case, "anchor": anchor, "profile": str(anchors["profiles"][anchor]),
                    "seed": entry["seed"], "queries": queries.tolist(), "controls": controls,
                    "minimum_mass": float(state.min()), "distance_to_boundary": float(state.min()-model.spec.mass_stop),
                    "minimum_mass_over_stopping_mass": float(state.min()/model.spec.mass_stop),
                    "boundary_stratum": ("one_to_two_thresholds" if state.min()/model.spec.mass_stop <= 2 else
                        "two_to_five_thresholds" if state.min()/model.spec.mass_stop <= 5 else
                        "five_to_ten_thresholds" if state.min()/model.spec.mass_stop <= 10 else
                        "ten_to_25_thresholds" if state.min()/model.spec.mass_stop <= 25 else "above_25_thresholds"),
                    "boundary_width": model.boundary_width,
                    "distance_over_boundary_width": float((state.min()-model.spec.mass_stop)/model.boundary_width),
                    "distance_over_moving_width_by_query": ((state.min()-model.spec.mass_stop)
                        /(model.boundary_width*np.sqrt(queries/model.horizon))).tolist(),
                    "selection_rank_fraction": float(anchors["selection_rank_fraction"][anchor]),
                    "minimum_mass_rank": int(anchors["minimum_mass_rank"][anchor]),
                    "candidate_count": int(anchors["candidate_count"][anchor]),
                    "gradient_absolute_rms_by_query": np.sqrt(np.mean(difference**2, axis=(-2, -1))).tolist(),
                    "reference_gradient_sampling_rms_by_query": np.sqrt(np.mean(
                        (boot_mean-mean)**2, axis=(0, -2, -1))).tolist(),
                    "source_cross_frobenius_error_by_query": source_distance.tolist(),
                    "source_error_pointwise_bootstrap_interval": source_interval.tolist(),
                    "plugin_minus_cross_frobenius_by_query": np.linalg.norm(plugin-cross, axis=(-2, -1)).tolist(),
                    "cross_source_minimum_eigenvalue_by_query": np.linalg.eigvalsh(cross).min(-1).tolist(),
                    "paired_refinement": refinement,
                    "finite_difference_pathwise_maximum_error": max(float(abs(v["finite_difference_error"]).max()) for v in arrays),
                    "stopped_families": int(sum(np.isfinite(v["stopped_at"]).sum() for v in arrays)),
                    "families": plan.paths})
    path = root/"gradient_evidence.npz"
    npz(path, evidence)
    result = {"complete": True, "kind": KIND, "identity_hash": saved["identity_hash"],
        "records": rows, "numeric_evidence": {"path": str(path), "sha256": sha(path)},
        "scope": "finite-Euler conditional mean derivatives at state-only validation anchors",
        "sampling_unit": "whole independent continuation family; all queries/outputs/faces paired",
        "source_estimators": "plugin has positive quadratic MC bias; independent-half symmetric cross-Gram is unbiased if pathwise mean differentiation is valid and may be indefinite",
        "uncertainty": "pointwise bootstrap only; no simultaneous or continuum certificate",
        "remaining_bias": "time discretization, derivative interchange, finite-difference branch changes, and unbounded-variance near-hit sensitivities remain unresolved",
        "fitting_or_final_test_access": False}
    write(root/"summary.json", result)
    return result


def _run(root, max_chunks=None):
    root = Path(root).resolve()
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Nonnegative optional chunk budget required")
    saved = _validate(root)
    plan = GradientPlan(**saved["identity"]["plan"])
    runner, study, _, _ = _inventory(Path(saved["identity"]["study_root"]), plan.cases)
    backend = None
    chunks, new = [], 0
    for case, item in saved["anchors"].items():
        cfg = runner.configuration(case, study)
        bank = runner.payoff_bank(case, study)
        with np.load(item["path"]) as store:
            states = store["states"]
        for anchor, state in enumerate(states):
            for start in range(0, plan.paths, plan.chunk_paths):
                path = root/"chunks"/case/f"anchor{anchor}_{start}.npz"
                record = path.with_suffix(".json")
                if record.exists():
                    result = read(record)
                    if (result["identity_hash"] != saved["identity_hash"] or sha(path) != result["sha256"]
                            or (result["case"], result["anchor"], result["start"], result["families"])
                            != (case, anchor, start, plan.chunk_paths)
                            or result["path"] != str(path)):
                        raise ValueError("Committed JVP chunk changed")
                    chunks.append(result)
                    continue
                if path.exists():
                    raise ValueError("Uncommitted JVP chunk cannot be accepted")
                if max_chunks is not None and new >= max_chunks:
                    continue
                if backend is None:
                    backend = compile_native(root/"compiled")
                values = _native_chunk(saved, plan, case, anchor, state, cfg, bank,
                                       np.asarray(item["queries"]), start, backend)
                npz(path, values)
                result = {"path": str(path), "sha256": sha(path), "case": case, "anchor": anchor,
                          "start": start, "families": plan.chunk_paths, "identity_hash": saved["identity_hash"]}
                write(record, result)
                chunks.append(result)
                new += 1
    expected = sum(v["rows"] for v in saved["anchors"].values())*(plan.paths//plan.chunk_paths)
    index = {"complete": len(chunks) == expected, "identity_hash": saved["identity_hash"],
             "expected_chunks": expected, "chunks": chunks}
    if index["complete"]:
        if (root/"index.json").exists() and read(root/"index.json").get("complete"):
            load_complete(root)
            return read(root/"index.json")
        _summary(root, saved, plan, runner, study, chunks)
        index["summary_sha256"] = sha(root/"summary.json")
    write(root/"index.json", index)
    return index


def run(root, max_chunks=None):
    """One owner per diagnostic root; a live owner is never inferred from a PID file."""
    root = Path(root).resolve()
    if not (root/"plan.json").exists():
        raise ValueError("Prepare the diagnostic before running")
    with (root/".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A diagnostic owner is already running") from error
        try:
            return _run(root, max_chunks)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_complete(root):
    root = Path(root)
    saved = _validate(root)
    index = read(root/"index.json")
    if not index.get("complete") or index["identity_hash"] != saved["identity_hash"]:
        raise ValueError("Every diagnostic chunk must complete before reading a summary")
    plan = GradientPlan(**saved["identity"]["plan"])
    expected = {(case, anchor, start) for case, item in saved["anchors"].items()
                for anchor in range(item["rows"]) for start in range(0, plan.paths, plan.chunk_paths)}
    actual = [(row["case"], row["anchor"], row["start"]) for row in index["chunks"]]
    if (len(actual) != len(expected) or set(actual) != expected
            or index["expected_chunks"] != len(expected)):
        raise ValueError("Incomplete diagnostic chunk inventory")
    for item in index["chunks"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Committed JVP chunk changed")
    if sha(root/"summary.json") != index["summary_sha256"]:
        raise ValueError("Diagnostic summary changed")
    result = read(root/"summary.json")
    if sha(result["numeric_evidence"]["path"]) != result["numeric_evidence"]["sha256"]:
        raise ValueError("Diagnostic numeric evidence changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--study-root", type=Path, default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.root, args.study_root, GradientPlan(**read(args.plan)) if args.plan else None)
        print(json.dumps({"prepared": True, "identity_hash": result["identity_hash"]}))
    else:
        result = run(args.root, args.max_chunks)
        print(json.dumps({"complete": result["complete"], "chunks": len(result["chunks"]),
                          "expected_chunks": result["expected_chunks"]}))


if __name__ == "__main__":
    main()
