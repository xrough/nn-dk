"""Immutable supervised correction study for conditional nonlinear-LNA payoffs.

Only fresh learner initials/continuations are drawn here. Final-reference access
belongs to a separately frozen joint evaluator after both studies finish fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_PROCESS_WALL_START = time.monotonic()

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import run_payoff_regimes as utilities  # noqa: E402
from analysis.compiled_native import compile_native  # noqa: E402
from analysis.payoff_collocation_sampling import (  # noqa: E402
    BoundaryStrataConfig,
    BoundaryStressConfig,
    make_boundary_stress_pool,
)
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context, seeded_rng, stable_hash  # noqa: E402

WORKFLOW = "supervised-conditional-nonlinear-LNA-payoff-correction-study-v1"
STREAM = "lna-supervised-correction-2026-10-30-v1"
HELPERS = ("fit_lna_payoff_correction.py", "payoff_collocation_sampling.py",
           "batched_native_gaussian.py", "native_gaussian_baseline.py",
           "compiled_native.py", "native_euler_control.cpp", "run_payoff_regimes.py")
NORMALIZER_POLICY = "fixed physical payoff-bank scales; training-only engine normalization; no validation/final normalization"
PROFILES = ("uniform", "nonuniform")


@dataclass(frozen=True)
class StudyPlan:
    horizons: tuple[float, ...] = (.25, 1.)
    kappas: tuple[float, ...] = (.75, 3., 6.)
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    a2: float = .1
    rho_stop: float = .01/(2*np.pi)
    train_natural_per_profile: int = 40
    train_stress_per_profile: int = 20
    validation_per_profile: int = 20
    train_continuations: int = 256
    validation_continuations: int = 512
    label_step: float = 2**-12
    label_chunk: int = 32
    baseline_step: float = 2**-7
    stress_distance_multipliers: tuple[float, ...] = (.25, 1., 4.)
    boundary_width_per_cell: float = .05
    train_initial_seed: int = 2026103001
    train_label_seed: int = 2026103002
    validation_initial_seed: int = 2026103004
    validation_label_seed: int = 2026103005
    width: int = 32
    depth: int = 2
    adam_steps: int = 1000
    lbfgs_steps: int = 30
    lbfgs_max_iter: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64
    validation_every: int = 20
    collocation_root: str = "runs/payoff-collocation-study-2026-10-29"

    def __post_init__(self):
        for name in ("horizons", "kappas", "optimizer_seeds", "stress_distance_multipliers"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.optimizer_seeds != (0, 1, 2) or not self.horizons or not self.kappas
                or len(set(self.horizons)) != len(self.horizons) or len(set(self.kappas)) != len(self.kappas)
                or not np.isfinite((*self.horizons, *self.kappas)).all()
                or min(self.horizons) <= 0 or min(self.kappas) < 0
                or self.stress_distance_multipliers != (.25, 1., 4.)):
            raise ValueError("Distinct regimes, all three seeds and prescribed stress distances required")
        for name in ("n_cells", "n_particles", "train_natural_per_profile", "train_stress_per_profile",
                     "validation_per_profile", "train_continuations", "validation_continuations",
                     "label_chunk", "width", "depth", "adam_steps", "lbfgs_steps", "lbfgs_max_iter",
                     "batch_size", "validation_every", "train_initial_seed", "train_label_seed",
                     "validation_initial_seed", "validation_label_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (
                    0 if name in ("adam_steps", "lbfgs_steps") else 1):
                raise ValueError(f"Invalid integer allocation: {name}")
        seeds = (self.train_initial_seed, self.train_label_seed,
                 self.validation_initial_seed, self.validation_label_seed)
        if (len(set(seeds)) != 4 or self.adam_steps+self.lbfgs_steps < 1
                or self.train_natural_per_profile != 2*self.train_stress_per_profile
                or min(self.train_continuations, self.validation_continuations) < 2
                or self.train_continuations % self.label_chunk or self.validation_continuations % self.label_chunk
                or not self.collocation_root or not np.isfinite((self.diffusion, self.a2, self.rho_stop,
                    self.label_step, self.baseline_step, self.boundary_width_per_cell, self.learning_rate)).all()
                or min(self.diffusion, self.rho_stop, self.label_step, self.baseline_step,
                       self.boundary_width_per_cell, self.learning_rate) <= 0):
            raise ValueError("Independent seeds, fixed 2:1 mixture and complete valid allocations required")
        for horizon in self.horizons:
            queries = horizon*np.asarray((.25, .5, 1.))
            if not np.allclose(queries/self.label_step, np.rint(queries/self.label_step), atol=1e-10, rtol=0):
                raise ValueError("All physical label queries must lie on the native Euler grid")
        NativeDKSpec(self.n_cells, rho_stop=self.rho_stop)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete supervised correction StudyPlan required")
        return cls(**value)

    @property
    def plan_hash(self):
        return stable_hash(self.to_dict())


def _engine():
    from analysis import fit_lna_payoff_correction

    return fit_lna_payoff_correction


def _number(value):
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def cases(plan=None):
    plan = StudyPlan() if plan is None else plan
    return tuple(f"T{_number(t)}_k{_number(k)}" for t in plan.horizons for k in plan.kappas)


case_names = cases


def _physical(case, plan):
    values = {f"T{_number(t)}_k{_number(k)}": (t, k) for t in plan.horizons for k in plan.kappas}
    if case not in values:
        raise ValueError("Unknown predeclared correction regime")
    return values[case]


def spec(case, plan):
    _physical(case, plan)
    return NativeDKSpec(plan.n_cells, rho_stop=plan.rho_stop)


def context(case, plan):
    _, kappa = _physical(case, plan)
    return Context(n_particles=plan.n_particles, diffusion=plan.diffusion, kappa=kappa, a2=plan.a2)


def query_times(case, plan):
    return _physical(case, plan)[0]*np.asarray((.25, .5, 1.))


def payoff_bank(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return PayoffBank.mixed_terminal_integral(spec(case, plan), _physical(case, plan)[0])


def configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    horizon, kappa = _physical(case, plan)
    return _engine().LNAPayoffCorrectionConfig(horizon=horizon, kappa=kappa,
        n_cells=plan.n_cells, n_particles=plan.n_particles, diffusion=plan.diffusion, a2=plan.a2,
        rho_stop=plan.rho_stop, width=plan.width, depth=plan.depth,
        mean_adam_steps=plan.adam_steps, covariance_adam_steps=plan.adam_steps,
        mean_lbfgs_steps=plan.lbfgs_steps, covariance_lbfgs_steps=plan.lbfgs_steps,
        lbfgs_max_iter=plan.lbfgs_max_iter, learning_rate=plan.learning_rate,
        batch_size=plan.batch_size, validation_every=plan.validation_every, label_step=plan.label_step)


def slots(plan):
    return [{"case": case, "seed": seed, "name": f"{case}_lna_correction_seed{seed}",
             "variant": "trained-LNA-correction"} for case in cases(plan) for seed in plan.optimizer_seeds]


def _sha(path):
    return utilities._sha(Path(path))


def _read(path):
    return utilities._read(Path(path))


def _json(path, value):
    utilities._json(Path(path), value)


def _immutable(path, value):
    utilities._immutable(Path(path), value)


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Supervised correction work checkpointed at its resumable deadline")


def load_plan(root):
    return StudyPlan.from_dict(_read(Path(root)/"plan.json"))


def _design(plan):
    return {"kind": WORKFLOW, "stream": STREAM, "source_hash": source_hash(),
        "runner_sha256": _sha(__file__), "helpers": {name: _sha(Path(__file__).with_name(name)) for name in HELPERS},
        "plan": plan.to_dict(), "plan_hash": plan.plan_hash,
        "configs": {case: configuration(case, plan).to_dict() for case in cases(plan)},
        "banks": {case: payoff_bank(case, plan).to_dict() for case in cases(plan)}, "slots": slots(plan),
        "training_mixture": {"native": 2/3, "stress": 1/3, "physical_initial_law_claim": False,
            "stress_construction": "first prescribed natural parents, argmin cell, cyclic distance multiplier times .05/G at T"},
        "labels": "independent conditional native Euler means and unbiased N-scaled covariance; all stopped paths retained",
        "normalizer_policy": NORMALIZER_POLICY,
        "baseline": {"kind": "conditional nonlinear mean-field plus native linear-noise covariance",
            "step": plan.baseline_step, "initial_scaled_covariance": "zero conditional covariance",
            "analytic_IID_replacement": False, "stopping_equivalence": False, "PSD_or_density_repair": False},
        "collocation_root": str(Path(plan.collocation_root).resolve()),
        "final_reference_policy": "NO final outcomes here; joint evaluator must lock every collocation and correction fit and freeze choices/identifiers/helpers before any final initial or outcome draw",
        "selection": "natural-only independent validation; all optimizer seeds retained; no final-data refit"}


def _assert_prepared(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    design = _read(root/"data/preregistration.json")
    if design != _design(plan):
        raise ValueError("Frozen correction source, helper, plan or architecture changed")
    return plan, design


def _array_sha(value):
    return utilities._array_sha(np.asarray(value))


def _save_npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as sink:
        np.savez_compressed(sink, **values)
    temporary.replace(path)


def _store_unit(directory, arrays, identity, costs=None, extra=None):
    directory = Path(directory)
    if (directory/"record.json").exists() or (directory/"arrays.npz").exists():
        raise ValueError("Refuse to overwrite committed or incomplete correction evidence")
    _save_npz(directory/"arrays.npz", arrays)
    record = {"identity": identity, "complete": True, "arrays_sha256": _sha(directory/"arrays.npz"),
        "arrays": {key: _array_sha(value) for key, value in arrays.items()},
        "costs": {} if costs is None else costs, "diagnostics": {} if extra is None else extra}
    record["content_sha256"] = stable_hash(record)
    _json(directory/"record.json", record)
    return record


def _load_unit(directory, identity):
    directory = Path(directory)
    if not directory.exists():
        return None
    if not (directory/"record.json").is_file() or not (directory/"arrays.npz").is_file():
        raise ValueError("Incomplete correction evidence; never redraw or replace a saved chunk")
    record = _read(directory/"record.json")
    payload = {key: value for key, value in record.items() if key != "content_sha256"}
    if (record.get("identity") != identity or record.get("complete") is not True
            or record.get("content_sha256") != stable_hash(payload)
            or record["arrays_sha256"] != _sha(directory/"arrays.npz")):
        raise ValueError("Correction evidence identity or checksum changed")
    with np.load(directory/"arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if record["arrays"] != {key: _array_sha(value) for key, value in arrays.items()}:
        raise ValueError("Correction array inventory or content changed")
    return arrays, record


def _address(plan, case, role, profile, anchor, index, purpose):
    seed = getattr(plan, ("train" if role == "train" else "validation")+
                   ("_initial_seed" if purpose == "initial" else "_label_seed"))
    keys = (STREAM, plan.plan_hash, case, role, profile, anchor, index, purpose)
    digest = hashlib.sha256(json.dumps([seed, *keys], separators=(",", ":")).encode()).hexdigest()
    return seed, keys, digest


@dataclass(frozen=True)
class _ProfileConfig:
    n_cells: int
    rho_stop: float
    cases: tuple[str, ...] = PROFILES

    @property
    def spec(self):
        return NativeDKSpec(self.n_cells, rho_stop=self.rho_stop)


def _anchor_identity(plan, case, role):
    return {"kind": WORKFLOW+"-anchors", "plan_hash": plan.plan_hash, "case": case,
            "role": role, "source_hash": source_hash()}


def profile_probabilities(case, profile, plan):
    _physical(case, plan)
    return utilities.profile_probabilities(_ProfileConfig(plan.n_cells, plan.rho_stop), profile)


def _draw_anchors(case, role, plan):
    states, families, parents, kinds, profiles, transforms, distances, entropy = [], [], [], [], [], [], [], []
    natural_count = plan.train_natural_per_profile if role == "train" else plan.validation_per_profile
    physical = spec(case, plan)
    horizon = _physical(case, plan)[0]
    for profile in PROFILES:
        probabilities, _ = profile_probabilities(case, profile, plan)
        natural, ids = [], []
        for index in range(natural_count):
            seed, keys, digest = _address(plan, case, role, profile, index, 0, "initial")
            state = seeded_rng(seed, *keys).multinomial(plan.n_particles, probabilities)/plan.n_particles
            family = stable_hash([STREAM, "initial-family", digest])
            natural.append(state)
            ids.append(family)
            states.append(state)
            families.append(family)
            parents.append(family)
            kinds.append("native")
            profiles.append(profile)
            transforms.append("")
            distances.append(-1.)
            entropy.append(digest)
        if role == "train":
            for index in range(plan.train_stress_per_profile):
                distance = plan.stress_distance_multipliers[index % 3]*plan.boundary_width_per_cell/plan.n_cells
                cfg = BoundaryStressConfig(BoundaryStrataConfig(horizon, physical.mass_stop,
                    plan.boundary_width_per_cell/plan.n_cells), (distance,),
                    ((int(np.argmin(natural[index])),),), (horizon,))
                stress = make_boundary_stress_pool(np.asarray(natural[index:index+1]), [ids[index]], cfg)
                transform = str(stress.transform_ids[0])
                states.append(stress.states[0])
                families.append(stable_hash([STREAM, "stress-anchor", ids[index], transform]))
                parents.append(ids[index])
                kinds.append("stress")
                profiles.append(profile)
                transforms.append(transform)
                distances.append(distance)
                entropy.append("")
    return {"states": np.asarray(states), "family_ids": np.asarray(families, dtype="S64"),
        "parent_family_ids": np.asarray(parents, dtype="S64"), "row_kinds": np.asarray(kinds, dtype="S8"),
        "profiles": np.asarray(profiles, dtype="S16"), "transform_ids": np.asarray(transforms, dtype="S64"),
        "target_distances": np.asarray(distances), "initial_entropy_sha256": np.asarray(entropy, dtype="S64")}


def _anchors(root, case, role, plan):
    unit = _load_unit(Path(root)/"data"/case/role/"anchors", _anchor_identity(plan, case, role))
    if unit is None:
        raise ValueError("Prepare complete learner anchor pools first")
    arrays, record = unit
    expected = 2*((plan.train_natural_per_profile+plan.train_stress_per_profile) if role == "train"
                  else plan.validation_per_profile)
    if (arrays["states"].shape != (expected, plan.n_cells)
            or len(set(arrays["family_ids"].astype(str))) != expected
            or not np.isfinite(arrays["states"]).all() or (arrays["states"] < 0).any()
            or not np.allclose(arrays["states"].sum(-1), 1., atol=1e-10, rtol=0)
            or (role == "validation" and set(arrays["row_kinds"].astype(str)) != {"native"})):
        raise ValueError("Anchor allocation, states or natural validation inventory changed")
    return arrays, record


def _costs(path, identity):
    if not Path(path).exists():
        return {"identity": identity, "attempts": []}
    packet = _read(path)
    payload = {k: v for k, v in packet.items() if k != "content_sha256"}
    if packet.get("identity") != identity or packet.get("content_sha256") != stable_hash(payload):
        raise ValueError("Correction cost record identity or content changed")
    return payload


def _save_costs(path, packet):
    _json(path, {**packet, "content_sha256": stable_hash(packet)})


def prepare(root, plan=None, deadline=None):
    started, cpu = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    plan = (load_plan(root) if (root/"plan.json").exists() else StudyPlan()) if plan is None else plan
    with utilities._exclusive(root/"execution/.prepare.lock"):
        if (root/"payoffs/fit_lock.json").exists():
            raise ValueError("Correction fit lock exists; preparation is closed")
        if (root/"plan.json").exists():
            if load_plan(root) != plan:
                raise ValueError("Frozen correction StudyPlan changed")
            _assert_prepared(root)
        else:
            if any(p.name != "execution" for p in root.iterdir()):
                raise ValueError("Choose a fresh correction study root")
            _immutable(root/"plan.json", plan.to_dict())
            _immutable(root/"data/preregistration.json", _design(plan))
        costs = _costs(root/"execution/prepare_costs.json", {"plan_hash": plan.plan_hash, "stage": "anchors"})
        attempt = {"pid": os.getpid(), "complete": False}
        costs["attempts"].append(attempt)
        try:
            seen = set()
            for case in cases(plan):
                for role in ("train", "validation"):
                    _deadline(deadline)
                    directory = root/"data"/case/role/"anchors"
                    if _load_unit(directory, _anchor_identity(plan, case, role)) is None:
                        tick, tcpu = time.monotonic(), time.process_time()
                        arrays = _draw_anchors(case, role, plan)
                        _store_unit(directory, arrays, _anchor_identity(plan, case, role),
                                    {"wall_seconds": time.monotonic()-tick, "cpu_seconds": time.process_time()-tcpu})
                    arrays, _ = _anchors(root, case, role, plan)
                    ids = set(arrays["parent_family_ids"].astype(str))
                    if seen & ids:
                        raise ValueError("Learner parent families overlap across cases or train/validation")
                    seen.update(ids)
            attempt["complete"] = True
        finally:
            attempt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            _save_costs(root/"execution/prepare_costs.json", costs)
    return root/"data"


def _unit_identity(plan, case, role, anchor, kind, anchors_record):
    return {"kind": WORKFLOW+"-"+kind, "plan_hash": plan.plan_hash, "case": case, "role": role,
            "anchor": anchor, "source_hash": source_hash(), "anchor_pool_sha256": anchors_record["arrays_sha256"]}


def baseline_batch(states, case, plan, *, step=None):
    from analysis.batched_native_gaussian import conditional_gaussian_batch

    bank = payoff_bank(case, plan)
    result = conditional_gaussian_batch(states, context(case, plan), spec(case, plan), query_times(case, plan),
        plan.baseline_step if step is None else step, bank.terminal_weights, bank.running_weights,
        batch_size=64, return_native=False)
    mean, covariance, diagnostics = result["mean"], result["scaled_covariance"], result["diagnostics"]
    if (mean.shape != (len(states), 3, 4) or covariance.shape != (len(states), 3, 4, 4)
            or not np.asarray(diagnostics["valid"]).all() or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-11, rtol=0)
            or np.linalg.eigvalsh(covariance).min() < -1e-10):
        raise ValueError("Invalid physical LNA covariance or means; no clipping or PSD repair")
    return {"mean": mean, "scaled_covariance": covariance}, diagnostics


def _draw_labels(state, case, role, profile, anchor, start, count, plan, backend):
    physical, physics = spec(case, plan), context(case, plan)
    horizon = _physical(case, plan)[0]
    steps = round(horizon/plan.label_step)
    increments, families, addresses, hashes = [], [], [], []
    for index in range(start, start+count):
        seed, keys, digest = _address(plan, case, role, profile, anchor, index, "label-forcing")
        dw = seeded_rng(seed, *keys).normal(size=(steps, plan.n_cells))*np.sqrt(plan.label_step)
        increments.append(dw)
        addresses.append(digest)
        hashes.append(_array_sha(dw))
        families.append(stable_hash([STREAM, "continuation-family", digest]))
    packet = backend(np.broadcast_to(state, (count, plan.n_cells)), np.asarray(increments), physics,
                     physical, plan.label_step)
    utilities._validate_paths(packet, horizon)
    times = np.arange(steps+1)*plan.label_step
    values = payoff_bank(case, plan).evaluate_paths(packet["states"], times, packet["stopped_at"])
    indices = np.rint(query_times(case, plan)/plan.label_step).astype(int)
    outcomes = values[:, indices]
    if outcomes.shape != (count, 3, 4) or not np.isfinite(outcomes).all():
        raise ValueError("Every declared stopped learner continuation must produce complete physical payoffs")
    return {"outcomes": outcomes, "family_ids": np.asarray(families, dtype="S64"),
        "forcing_entropy_sha256": np.asarray(addresses, dtype="S64"),
        "brownian_sha256": np.asarray(hashes, dtype="S64"), "stopped_at": packet["stopped_at"],
        "minimum_mass": packet["states"].min(axis=(1, 2))}


def _label_identity(plan, case, role, anchor, start, anchors_record):
    return {**_unit_identity(plan, case, role, anchor, "label-chunk", anchors_record),
            "start": start, "count": plan.label_chunk, "step": plan.label_step}


def _finalize_bank(root, case, role, plan, *, create=False):
    directory = Path(root)/"data"/case/role
    anchors, anchors_record = _anchors(root, case, role, plan)
    total = plan.train_continuations if role == "train" else plan.validation_continuations
    baseline_path = directory/"baseline"
    baseline = _load_unit(baseline_path, _unit_identity(plan, case, role, -1, "baseline-cache", anchors_record))
    if baseline is None:
        return None
    baseline_arrays, baseline_record = baseline
    labels, means, covariances = [], [], []
    seen = set()
    for anchor in range(len(anchors["states"])):
        draws = []
        for start in range(0, total, plan.label_chunk):
            path = directory/"labels"/f"anchor{anchor}"/f"chunk{start}"
            chunk = _load_unit(path, _label_identity(plan, case, role, anchor, start, anchors_record))
            if chunk is None:
                return None
            arrays, record = chunk
            ids = arrays["family_ids"].astype(str).tolist()
            profile = anchors["profiles"][anchor].decode()
            wanted = [stable_hash([STREAM, "continuation-family", _address(plan, case, role, profile,
                anchor, i, "label-forcing")[2]]) for i in range(start, start+plan.label_chunk)]
            if (ids != wanted or len(set(ids)) != len(ids) or seen.intersection(ids)
                    or arrays["outcomes"].shape != (plan.label_chunk, 3, 4)
                    or not np.isfinite(arrays["outcomes"]).all()
                    or arrays["stopped_at"].shape != (plan.label_chunk,) or np.isnan(arrays["stopped_at"]).any()
                    or np.any(arrays["stopped_at"] < 0)):
                raise ValueError("Label family allocation, stopping inclusion or payoff evidence changed")
            seen.update(ids)
            draws.append(arrays["outcomes"])
            labels.append({"path": str(path.resolve()), "record_sha256": _sha(path/"record.json"),
                           "arrays_sha256": record["arrays_sha256"], "families": len(ids),
                           "stopped_families": int(np.isfinite(arrays["stopped_at"]).sum())})
        outcomes = np.concatenate(draws)
        mean = outcomes.mean(0)
        centered = outcomes-mean
        means.append(mean)
        covariances.append(plan.n_particles*np.einsum("nqi,nqj->qij", centered, centered)/(len(outcomes)-1))
    if not create and any(not p.exists() for p in (directory/"label_manifest.json",
            directory/"baseline_manifest.json", directory/"bank/record.json", directory/"bank/arrays.npz")):
        raise ValueError("Incomplete reduced supervised bank; verification never recreates evidence")
    _immutable(directory/"label_manifest.json", {"complete": True, "case": case, "role": role,
        "plan_hash": plan.plan_hash, "paths_per_anchor": total, "anchors": len(anchors["states"]),
        "all_stopped_paths_retained": True, "unbiased_sample_covariance": True, "chunks": labels})
    _immutable(directory/"baseline_manifest.json", {"complete": True, "case": case, "role": role,
        "plan_hash": plan.plan_hash, "conditional_initial_covariance": "zero", "step": plan.baseline_step,
        "analytic_IID_replacement": False, "cache": {"path": str(baseline_path.resolve()),
            "record_sha256": _sha(baseline_path/"record.json"), "arrays_sha256": baseline_record["arrays_sha256"]}})
    raw = {**anchors, "times": query_times(case, plan), "baseline_mean": baseline_arrays["mean"],
        "baseline_scaled_covariance": baseline_arrays["scaled_covariance"], "label_mean": np.asarray(means),
        "label_scaled_covariance": np.asarray(covariances)}
    provenance = {"stream_namespace": STREAM, "role": role, "parent_pool_sha256": anchors_record["arrays_sha256"],
        "label_manifest_sha256": _sha(directory/"label_manifest.json"),
        "baseline_manifest_sha256": _sha(directory/"baseline_manifest.json"),
        "normalizer_policy": NORMALIZER_POLICY, "training_mixture": {"native": 2/3, "stress": 1/3},
        "all_stopped_paths_retained": True}
    bank = _make_bank(raw, role, case, plan, provenance)
    identity = {"kind": WORKFLOW+"-supervised-bank", "case": case, "role": role,
        "plan_hash": plan.plan_hash, "provenance": provenance, "bank_content_sha256": bank["content_sha256"]}
    saved = _load_unit(directory/"bank", identity)
    if saved is None:
        _store_unit(directory/"bank", raw, identity)
    elif any(not np.array_equal(saved[0][key], value) for key, value in raw.items()):
        raise ValueError("Reduced supervised bank disagrees with all saved raw learner families")
    return bank


def _make_bank(raw, role, case, plan, provenance):
    return _engine().make_supervised_bank(raw["states"], raw["times"], raw["baseline_mean"],
        raw["baseline_scaled_covariance"], raw["label_mean"], raw["label_scaled_covariance"],
        raw["family_ids"].astype(str), parent_family_ids=raw["parent_family_ids"].astype(str),
        row_kinds=raw["row_kinds"].astype(str), role=role, config=configuration(case, plan),
        bank=payoff_bank(case, plan), provenance=provenance)


def _data_inventory(root, plan):
    inventory, identifier_files = {}, []
    seen = set()
    for case in cases(plan):
        inventory[case] = {}
        for role in ("train", "validation"):
            bank = _finalize_bank(root, case, role, plan)
            if bank is None:
                raise ValueError("All learner baselines and label families must complete before fitting")
            directory = Path(root)/"data"/case/role
            anchors, _ = _anchors(root, case, role, plan)
            parents = set(anchors["parent_family_ids"].astype(str))
            if parents & seen:
                raise ValueError("Natural/stress parent groups overlap cases or train/validation splits")
            seen.update(parents)
            identifier_files.append({"path": str((directory/"anchors/arrays.npz").resolve()),
                "sha256": _sha(directory/"anchors/arrays.npz"),
                "fields": ["family_ids", "parent_family_ids", "profiles", "initial_entropy_sha256"],
                "case": case, "role": role, "kind": "learner-anchors"})
            labels = _read(directory/"label_manifest.json")
            for chunk in labels["chunks"]:
                identifier_files.append({"path": str(Path(chunk["path"])/"arrays.npz"),
                    "sha256": chunk["arrays_sha256"],
                    "fields": ["family_ids", "forcing_entropy_sha256", "brownian_sha256"],
                    "case": case, "role": role, "kind": "learner-continuation-labels"})
            inventory[case][role] = {"content_sha256": bank["content_sha256"],
                "bank_record_sha256": _sha(directory/"bank/record.json"),
                "bank_npz_sha256": _sha(directory/"bank/arrays.npz"),
                "label_manifest_sha256": _sha(directory/"label_manifest.json"),
                "baseline_manifest_sha256": _sha(directory/"baseline_manifest.json")}
    return {"complete": True, "plan_hash": plan.plan_hash, "source_hash": source_hash(), "banks": inventory,
            "identifier_inventory": identifier_files,
            "parent_splits_disjoint": True, "all_stopped_paths_retained": True, "final_outcomes_read": False}


def generate_data(root, case=None, role=None, deadline=None, max_chunks=None):
    started, cpu = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    plan, _ = _assert_prepared(root)
    if case is not None and case not in cases(plan) or role not in (None, "train", "validation"):
        raise ValueError("Use a declared learner regime and role")
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Nonnegative bounded software chunk allocation required")
    with utilities._exclusive(root/"execution/.data.lock"):
        if (root/"payoffs/fit_lock.json").exists():
            raise ValueError("Correction fit lock exists; learner data generation is closed")
        costs = _costs(root/"execution/data_costs.json", {"plan_hash": plan.plan_hash, "stage": "labels-and-baselines"})
        attempt = {"pid": os.getpid(), "complete": False, "committed_chunks": 0}
        costs["attempts"].append(attempt)
        try:
            _deadline(deadline)
            backend = compile_native(root/"compiled")
            for current in cases(plan) if case is None else (case,):
                for current_role in ("train", "validation") if role is None else (role,):
                    anchors, record = _anchors(root, current, current_role, plan)
                    directory = root/"data"/current/current_role
                    total = plan.train_continuations if current_role == "train" else plan.validation_continuations
                    identity = _unit_identity(plan, current, current_role, -1, "baseline-cache", record)
                    path = directory/"baseline"
                    if _load_unit(path, identity) is None:
                        tick, tcpu = time.monotonic(), time.process_time()
                        baseline, diagnostic = baseline_batch(anchors["states"], current, plan)
                        diagnostic = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in diagnostic.items()}
                        _store_unit(path, baseline, identity,
                            {"wall_seconds": time.monotonic()-tick, "cpu_seconds": time.process_time()-tcpu,
                             "scope": "complete conditional ODE cache for every literal anchor; no analytic IID replacement"}, diagnostic)
                    for index, state in enumerate(anchors["states"]):
                        _deadline(deadline)
                        for start in range(0, total, plan.label_chunk):
                            _deadline(deadline)
                            identity = _label_identity(plan, current, current_role, index, start, record)
                            path = directory/"labels"/f"anchor{index}"/f"chunk{start}"
                            if _load_unit(path, identity) is not None:
                                continue
                            if max_chunks is not None and attempt["committed_chunks"] >= max_chunks:
                                return {"complete": False, "committed_chunks": attempt["committed_chunks"]}
                            tick, tcpu = time.monotonic(), time.process_time()
                            arrays = _draw_labels(state, current, current_role,
                                anchors["profiles"][index].decode(), index, start, plan.label_chunk, plan, backend)
                            _store_unit(path, arrays, identity,
                                {"wall_seconds": time.monotonic()-tick, "cpu_seconds": time.process_time()-tcpu})
                            attempt["committed_chunks"] += 1
                    _finalize_bank(root, current, current_role, plan, create=True)
            try:
                inventory = _data_inventory(root, plan)
            except ValueError as error:
                if "must complete before fitting" not in str(error):
                    raise
                return {"complete": False, "committed_chunks": attempt["committed_chunks"]}
            _immutable(root/"data/data_lock.json", inventory)
            attempt["complete"] = True
            return inventory
        finally:
            attempt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            _save_costs(root/"execution/data_costs.json", costs)


def prepared(root):
    plan, design = _assert_prepared(root)
    if _read(Path(root)/"data/data_lock.json") != _data_inventory(root, plan):
        raise ValueError("Complete supervised data lock changed")
    return plan, design


def load_bank(root, case, role):
    plan, _ = _assert_prepared(root)
    if case not in cases(plan) or role not in ("train", "validation"):
        raise ValueError("Declared learner bank required")
    bank = _finalize_bank(root, case, role, plan)
    if bank is None:
        raise ValueError("Incomplete supervised learner bank")
    return bank


def _checkpoint(root, slot):
    return Path(root)/"payoffs"/slot["case"]/"checkpoints"/f"seed{slot['seed']}.pt"


def load_declared_model(item):
    path = Path(item["checkpoint_path"])
    if item.get("checkpoint_sha256", _sha(path)) != _sha(path):
        raise ValueError("Declared correction checkpoint changed")
    return _engine().load_lna_correction(path)


def _model_record(root, slot, plan):
    path = _checkpoint(root, slot)
    model = load_declared_model({"checkpoint_path": str(path)})
    metadata = _read(path.with_suffix(".json"))
    identity = model.provenance
    config = configuration(slot["case"], plan)
    locked = _read(Path(root)/"data/data_lock.json")["banks"][slot["case"]]
    if (not model.complete or metadata.get("complete") is not True or metadata.get("identity") != identity
            or metadata.get("kind") != model.method_kind or metadata.get("method_id") != model.method_id
            or identity["config_hash"] != config.config_hash or identity["seed"] != slot["seed"]
            or identity["sources"]["package_source_hash"] != source_hash()
            or identity["bank_hash"] != payoff_bank(slot["case"], plan).bank_hash):
        raise ValueError("Correction model source, bank, seed or complete metadata differs")
    for role in ("train", "validation"):
        if identity["data"][role]["content_sha256"] != locked[role]["content_sha256"]:
            raise ValueError("Correction model was trained on different learner banks")
    cost_path = path.with_suffix(".invocations.json")
    costs = _costs(cost_path, {"slot": slot["name"], "config_hash": config.config_hash,
                              "data_lock_sha256": _sha(Path(root)/"data/data_lock.json")})
    finished = [v for v in costs["attempts"] if v.get("finished")]
    if (not costs["attempts"] or not finished or not math.isfinite(metadata["training_seconds"])
            or metadata["training_seconds"] <= 0):
        raise ValueError("Completed correction fit requires finite full invocation costs")
    return {**slot, "complete": True, "checkpoint_path": str(path.resolve()), "checkpoint_sha256": _sha(path),
        "metadata_path": str(path.with_suffix(".json").resolve()), "metadata_sha256": _sha(path.with_suffix(".json")),
        "source_hash": source_hash(), "config_hash": config.config_hash, "target_id": model.target_id,
        "baseline_step": plan.baseline_step,
        "bank_hash": identity["bank_hash"], "method_id": model.method_id, "model_kind": model.method_kind,
        "initial_state_sha256": identity["initial_state_sha256"], "data_fingerprints": identity["data"],
        "loss_scaling": identity["loss_scaling"], "training_seconds": metadata["training_seconds"],
        "stage_seconds": metadata["stage_seconds"], "selected_step": metadata["best_step"],
        "natural_validation": metadata["best_validation"],
        "full_invocation_seconds": sum(v["wall_seconds"] for v in finished),
        "full_invocation_cpu_seconds": sum(v["cpu_seconds"] for v in finished),
        "full_invocation_cost_complete": len(finished) == len(costs["attempts"]),
        "invocation_cost_scope": "all engine entry-through-return attempts including initialization/validation/saves; cached-bank integrity/loading, sidecar writes and strict restoration are worker overhead",
        "invocation_record_path": str(cost_path.resolve()), "invocation_record_sha256": _sha(cost_path)}


def fit_seed(root, seed, deadline=None):
    root = Path(root).resolve()
    started, cpu = time.monotonic(), time.process_time()
    plan, design = prepared(root)
    if type(seed) is not int or seed not in plan.optimizer_seeds:
        raise ValueError("One preregistered optimizer seed per owner required")
    with utilities._exclusive(root/"payoffs/execution"/f".seed{seed}.lock"):
        if (root/"payoffs/fit_lock.json").exists():
            raise ValueError("Correction fit lock exists; fitting is closed")
        output = root/"payoffs/execution"/f"worker_seed{seed}.json"
        previous = _read(output).get("attempts", []) if output.exists() else []
        attempt = {"pid": os.getpid(), "complete": False, "pool_preparation": {}}
        packet = {"workflow": WORKFLOW, "seed": seed, "complete": False, "models": [],
            "source_hash": source_hash(), "preregistration_sha256": _sha(root/"data/preregistration.json"),
            "data_lock_sha256": _sha(root/"data/data_lock.json"), "attempts": [*previous, attempt],
            "held_out_reference_inputs_read": False}
        try:
            for slot in design["slots"]:
                if slot["seed"] != seed:
                    continue
                _deadline(deadline)
                tick, tcpu = time.monotonic(), time.process_time()
                train, validation = (load_bank(root, slot["case"], role) for role in ("train", "validation"))
                attempt["pool_preparation"][slot["case"]] = {
                    "seconds": time.monotonic()-tick, "cpu_seconds": time.process_time()-tcpu,
                    "scope": "load and verify cached baseline/label banks; generation charged separately"}
                config = configuration(slot["case"], plan)
                path = _checkpoint(root, slot)
                cost_path = path.with_suffix(".invocations.json")
                costs = _costs(cost_path, {"slot": slot["name"], "config_hash": config.config_hash,
                    "data_lock_sha256": _sha(root/"data/data_lock.json")})
                entry = {"finished": False, "started_unix": time.time(), "pid": os.getpid()}
                costs["attempts"].append(entry)
                _save_costs(cost_path, costs)
                tick, tcpu = time.monotonic(), time.process_time()
                metadata = None
                try:
                    metadata = _engine().fit_lna_correction(config, payoff_bank(slot["case"], plan), seed,
                        path, train, validation, deadline=deadline)
                finally:
                    entry.update(finished=True, wall_seconds=time.monotonic()-tick,
                        cpu_seconds=time.process_time()-tcpu, allocation_complete=bool(metadata and metadata.get("complete")))
                    _save_costs(cost_path, costs)
                _json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    raise BudgetExhausted("Correction engine saved an incomplete resumable optimizer allocation")
                packet["models"].append(_model_record(root, slot, plan))
                _json(output, packet)
            packet["complete"] = attempt["complete"] = True
        finally:
            attempt.update(wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            _json(output, packet)
    return packet


def lock(root):
    root = Path(root).resolve()
    plan, design = prepared(root)
    from contextlib import ExitStack

    with utilities._exclusive(root/"payoffs/execution/.fit-lock.lock"), ExitStack() as stack:
        stack.enter_context(utilities._exclusive(root/"execution/.prepare.lock"))
        stack.enter_context(utilities._exclusive(root/"execution/.data.lock"))
        for seed in plan.optimizer_seeds:
            stack.enter_context(utilities._exclusive(root/"payoffs/execution"/f".seed{seed}.lock"))
            worker = _read(root/"payoffs/execution"/f"worker_seed{seed}.json")
            if (not worker.get("complete") or worker.get("source_hash") != source_hash()
                    or worker.get("preregistration_sha256") != _sha(root/"data/preregistration.json")
                    or worker.get("data_lock_sha256") != _sha(root/"data/data_lock.json")):
                raise ValueError("Every correction worker must complete its fixed allocation")
        models = [_model_record(root, slot, plan) for slot in design["slots"]]
        result = _fit_lock_packet(root, plan, design, models)
        _immutable(root/"payoffs/fit_lock.json", result)
    return result


def _fit_lock_packet(root, plan, design, models):
    root = Path(root)
    costs = [root/"execution/prepare_costs.json", root/"execution/data_costs.json"]
    for stage, path in zip(("anchors", "labels-and-baselines"), costs, strict=True):
        _costs(path, {"plan_hash": plan.plan_hash, "stage": stage})
    costs.extend(root/"payoffs/execution"/f"worker_seed{seed}.json" for seed in plan.optimizer_seeds)
    costs.extend(p for p in sorted((root/"execution").glob("cli_*.json"))
                 if _read(p)["stage"] in ("prepare", "data", "fit"))
    return {"workflow": WORKFLOW, "complete": True, "source_hash": source_hash(),
        "plan_hash": plan.plan_hash, "expected_models": len(slots(plan)), "models": models,
        "runner_sha256": _sha(__file__), "preregistration_sha256": _sha(Path(root)/"data/preregistration.json"),
        "data_lock_sha256": _sha(Path(root)/"data/data_lock.json"),
        "cost_records": {str(path.resolve()): _sha(path) for path in costs},
        "collocation_root": design["collocation_root"], "selection_using_final_outcomes": False,
        "seeds_pooled": False, "all_stopped_paths_retained": True}


def verify_fit_lock(root):
    root = Path(root).resolve()
    plan, design = prepared(root)
    saved = _read(root/"payoffs/fit_lock.json")
    expected = _fit_lock_packet(root, plan, design,
                               [_model_record(root, slot, plan) for slot in design["slots"]])
    if saved != expected:
        raise ValueError("Every declared correction fit must strictly restore with its original data")
    return saved


def assert_joint_fit_locks(root):
    from analysis import run_payoff_collocation_study

    candidate = verify_fit_lock(root)
    plan = load_plan(root)
    collocation_root = Path(candidate["collocation_root"]).resolve()
    collocation = run_payoff_collocation_study.verify_fit_lock(collocation_root)
    old_plan = run_payoff_collocation_study.load_plan(collocation_root)
    if cases(plan) != run_payoff_collocation_study.case_names(old_plan):
        raise ValueError("Joint studies must retain exactly the same declared physical regimes")
    for case in cases(plan):
        reference = run_payoff_collocation_study.reference_configuration(case, old_plan)
        horizon, kappa = _physical(case, plan)
        if ((plan.n_cells, plan.n_particles, plan.diffusion, kappa, plan.a2, horizon)
                != (reference.n_cells, reference.n_particles, reference.diffusion,
                    reference.kappa, reference.a2, reference.horizon)
                or spec(case, plan) != reference.spec
                or tuple(query_times(case, plan)) != tuple(reference.query_times)
                or payoff_bank(case, plan).bank_hash != run_payoff_collocation_study.payoff_bank(case, old_plan).bank_hash):
            raise ValueError("Joint comparison target, stopping spec, queries or payoff bank differs")
    return {"candidate": candidate, "collocation": collocation, "collocation_root": str(collocation_root)}


def generate_reference(*args, **kwargs):
    raise ValueError("No final draws in the learner runner; use the frozen joint evaluator after both fit locks")


def load_reference(*args, **kwargs):
    raise ValueError("No final outcomes may be opened by the learner runner")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "data", "fit", "lock", "verify-lock"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case")
    parser.add_argument("--role", choices=("train", "validation"))
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args()
    if ((args.stage == "fit") != (args.seed is not None) or args.plan is not None and args.stage != "prepare"
            or any(v is not None for v in (args.case, args.role, args.max_chunks)) and args.stage != "data"
            or not 0 < args.budget_seconds <= 8*60*60):
        parser.error("Plan only for prepare, seed only for fit, case/role/chunks only for data; budget <=8h")
    deadline = time.monotonic()+args.budget_seconds
    torch.set_num_threads(1)
    status = 0
    try:
        if args.stage == "prepare":
            result = str(prepare(args.root, StudyPlan.from_dict(_read(args.plan)) if args.plan else None, deadline))
        elif args.stage == "data":
            result = generate_data(args.root, args.case, args.role, deadline, args.max_chunks)
        elif args.stage == "fit":
            result = fit_seed(args.root, args.seed, deadline)
        elif args.stage == "lock":
            result = lock(args.root)
        else:
            result = verify_fit_lock(args.root)
        print(json.dumps({"stage": args.stage, "root": str(args.root.resolve()),
            "complete": True if isinstance(result, str) else result.get("complete", False),
            "models": len(result.get("models", [])) if isinstance(result, dict) else 0}), flush=True)
    except BudgetExhausted as error:
        status = 75
        print(str(error), file=sys.stderr, flush=True)
        return 75
    except BaseException:
        status = 1
        raise
    finally:
        if (args.root/"plan.json").exists():
            _json(args.root/"execution"/f"cli_{args.stage}_{os.getpid()}.json", {
                "stage": args.stage, "pid": os.getpid(), "exit_code": status,
                "wall_seconds": time.monotonic()-_PROCESS_WALL_START,
                "process_cpu_seconds": time.process_time(), "source_hash": source_hash(),
                "scope": "wall from module startup after standard-library imports; CPU from entire process lifetime; includes import, integrity, compilation, caches and saves; function/unit costs are nonadditive components"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
