"""Frozen harder-regime payoff fits and independent native Monte Carlo.

The learned package and previous experiments are unchanged. All declared fits
must restore before any held-out anchor or outcome is drawn. State-only
collocation and final references use independent, addressable trajectory
families. Native paths are transient; only requested payoff outcomes are saved.
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
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import i0  # noqa: E402

from analysis.compiled_native import compile_native  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.experiments import BudgetExhausted  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.payoff_repair import RepairConfig, fit_repaired_payoff, load_repaired_payoff_model  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

WORKFLOW = "harder-native-dk-repaired-payoff-comparison-v1"
STREAM = "independent-harder-regime-payoff-families-2026102801-v1"
SCRIPT = Path(__file__).resolve()
HELPERS = ("evaluate_payoff_regimes.py", "compiled_native.py", "native_euler_control.cpp",
           "native_gaussian_baseline.py", "audit_payoff_regimes.py")


@dataclass(frozen=True)
class StudyPlan:
    """A full pre-data plan; changing any field requires a fresh output root."""

    horizons: tuple[float, ...] = (.25, 1.)
    kappas: tuple[float, ...] = (.75, 3., 6.)
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    a2: float = .1
    train_seed: int = 2026102801
    reference_seed: int = 2026102802
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 32
    train_paths: int = 256
    validation_paths: int = 128
    state_step: float = 2**-10
    state_fractions: tuple[float, ...] = (0., .25, .5, 1.)
    horizon_samples: int = 4
    adam_steps: int = 1000
    lbfgs_steps: int = 30
    lbfgs_max_iter: int = 5
    validation_every: int = 20
    batch_size: int = 128
    quadrature_nodes: int = 16
    reference_step: float = 2**-12
    reference_factors: tuple[int, ...] = (1, 2, 4)
    paths_per_group: int = 16384
    refinement_paths: int = 1024
    conditional_anchors: int = 2
    chunk_paths: int = 16

    def __post_init__(self):
        for name in ("horizons", "kappas", "optimizer_seeds", "state_fractions", "reference_factors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (not self.horizons or not self.kappas or len(set(self.horizons)) != len(self.horizons)
                or len(set(self.kappas)) != len(self.kappas)
                or not np.isfinite((*self.horizons, *self.kappas)).all()
                or min(self.horizons) <= 0 or min(self.kappas) < 0):
            raise ValueError("Distinct finite positive horizons and nonnegative interactions required")
        integers = ("n_cells", "n_particles", "train_seed", "reference_seed", "width", "train_paths",
                    "validation_paths", "horizon_samples", "adam_steps", "lbfgs_steps", "lbfgs_max_iter",
                    "validation_every", "batch_size", "quadrature_nodes", "paths_per_group",
                    "refinement_paths", "conditional_anchors", "chunk_paths")
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < (
                    0 if name in ("train_seed", "reference_seed", "adam_steps", "lbfgs_steps") else 1):
                raise ValueError(f"Invalid integer plan setting: {name}")
        if (self.train_seed == self.reference_seed or self.refinement_paths > self.paths_per_group
                or not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.optimizer_seeds)
                or self.chunk_paths > self.paths_per_group or self.quadrature_nodes < 2):
            raise ValueError("Independent seeds and valid complete allocation required")
        # The package validates all physical, optimizer and temporal settings.
        for case in case_names(self):
            configuration(case, self)
            reference_configuration(case, self)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def plan_hash(self):
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete StudyPlan field inventory required")
        return cls(**value)


def _number(value):
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def case_names(plan=None):
    plan = StudyPlan() if plan is None else plan
    return tuple(f"T{_number(horizon)}_k{_number(kappa)}"
                 for horizon in plan.horizons for kappa in plan.kappas)


def _physical(case, plan):
    values = {f"T{_number(horizon)}_k{_number(kappa)}": (horizon, kappa)
              for horizon in plan.horizons for kappa in plan.kappas}
    if case not in values:
        raise ValueError("Unknown predeclared physical regime")
    return values[case]


def configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    horizon, kappa = _physical(case, plan)
    # Same named seed across regimes still has independent config-addressed RNGs.
    return InteractingStudyConfig(
        name=f"payoff-regimes-{case}", seed=plan.train_seed,
        n_cells=plan.n_cells, n_particles=plan.n_particles, diffusion=plan.diffusion,
        kappa=kappa, a2=plan.a2, horizon=horizon,
        query_times=(horizon/4, horizon/2, horizon), optimizer_seeds=plan.optimizer_seeds,
        width=plan.width, train_paths=plan.train_paths, validation_paths=plan.validation_paths,
        state_step=plan.state_step, reference_step=plan.reference_step,
        state_fractions=plan.state_fractions, horizon_samples=plan.horizon_samples,
        adam_steps=plan.adam_steps, lbfgs_steps=plan.lbfgs_steps,
        lbfgs_max_iter=plan.lbfgs_max_iter, validation_every=plan.validation_every,
        batch_size=plan.batch_size, learning_rate=1e-3, loss_balancing="none",
        conditional_anchors=plan.conditional_anchors, conditional_paths=plan.paths_per_group,
        ensemble_paths=plan.paths_per_group, ensemble_initial_paths=plan.paths_per_group,
        refinement_paths=plan.refinement_paths, reference_factors=plan.reference_factors,
        reference_batch_size=plan.chunk_paths, n_bootstrap=128)


def repaired_configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return RepairConfig(configuration(case, plan), quadrature_nodes=plan.quadrature_nodes)


def payoff_bank(case, plan=None):
    cfg = configuration(case, plan)
    return PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)


@dataclass(frozen=True)
class RegimeReferenceConfig:
    study: InteractingStudyConfig
    seed: int
    paths_per_group: int
    refinement_paths: int
    chunk_paths: int
    reference_step: float
    reference_factors: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "reference_factors", tuple(self.reference_factors))
        if (self.refinement_paths > self.paths_per_group or self.paths_per_group < 2
                or self.refinement_paths < 2 or self.chunk_paths < 1
                or self.reference_factors != self.study.reference_factors):
            raise ValueError("Valid reference allocation and identical refinement factors required")

    def __getattr__(self, name):
        return getattr(self.study, name)

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    def to_dict(self):
        return {"study": self.study.to_dict(), "seed": self.seed,
                "paths_per_group": self.paths_per_group, "refinement_paths": self.refinement_paths,
                "chunk_paths": self.chunk_paths, "reference_step": self.reference_step,
                "reference_factors": list(self.reference_factors)}

    @property
    def group_keys(self):
        return tuple(f"{profile}__conditional__{anchor}" for profile in self.cases
                     for anchor in range(self.conditional_anchors)) + tuple(
            f"{profile}__ensemble__0" for profile in self.cases)


def reference_configuration(case, plan=None):
    plan = StudyPlan() if plan is None else plan
    return RegimeReferenceConfig(configuration(case, plan), plan.reference_seed,
                                 plan.paths_per_group, plan.refinement_paths, plan.chunk_paths,
                                 plan.reference_step, plan.reference_factors)


def slots(plan=None):
    plan = StudyPlan() if plan is None else plan
    return [{"case": case, "seed": seed, "bank": payoff_bank(case, plan).name,
             "name": f"{case}_repaired_seed{seed}", "variant": "repaired"}
            for case in case_names(plan) for seed in plan.optimizer_seeds]


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _immutable(path, value):
    if Path(path).exists():
        if _read(path) != value:
            raise ValueError(f"Frozen harder-regime artifact changed: {path}")
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
        stream.write(str(os.getpid())+"\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_plan(root):
    return StudyPlan.from_dict(_read(Path(root)/"plan.json"))


def _design(plan):
    return {"workflow": WORKFLOW, "source_hash": source_hash(), "runner_sha256": _sha(SCRIPT),
            "helpers": {name: _sha(SCRIPT.with_name(name)) for name in HELPERS},
            "plan": plan.to_dict(), "plan_hash": plan.plan_hash,
            "configs": {case: configuration(case, plan).to_dict() for case in case_names(plan)},
            "repaired_configs": {case: repaired_configuration(case, plan).to_dict()
                                 for case in case_names(plan)},
            "reference_configs": {case: reference_configuration(case, plan).to_dict()
                                  for case in case_names(plan)},
            "banks": {case: payoff_bank(case, plan).to_dict() for case in case_names(plan)},
            "slots": slots(plan), "expected_models": len(slots(plan)),
            "selection": "independent within-fit validation only; all regimes and seeds retained",
            "reference_unlock": "every declared fit complete and strictly restored before held-out draws",
            "training": "known-generator residual; state-only native collocation; no future-payoff labels",
            "noise": "physical Brownian forcing only; exact all-face generator trace; no auxiliary trace noise",
            "stopping": "first crossing of Euler polygon; frozen terminal; killed integral; all paths retained",
            "native_reference": "unchanged serial FP64 compiled native backend; bounded transient full paths",
            "cost": "all fits and state generation charged separately from cold and warm inference",
            "continuous_reference_bias": "unresolved; coupled Euler refinements estimated separately"}


def _assert_frozen(root):
    root, plan = Path(root), load_plan(root)
    design = _read(root/"payoffs"/"preregistration.json")
    if design != _design(plan):
        raise ValueError("Harder-regime source, runner, helper or configuration changed; use a fresh run")
    return plan, design


@lru_cache(maxsize=64)
def profile_probabilities(cfg, profile):
    if profile not in cfg.cases:
        raise ValueError("Unknown initial profile")

    def calculate(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        centers = (np.arange(cfg.n_cells)+.5)*cfg.spec.h
        positions = centers[:, None]+cfg.spec.h/2*nodes
        alpha = .2 if profile == "nonuniform" else 0.
        return cfg.spec.h/2*(np.exp(alpha*np.cos(positions))@weights)/(2*np.pi*i0(alpha))

    coarse, fine = calculate(32), calculate(64)
    if not np.isclose(fine.sum(), 1., atol=2e-14, rtol=0):
        raise ValueError("Initial profile cell quadrature failed")
    return fine, {"method": "32/64-point-cell-Gauss-Legendre-with-analytic-von-Mises-normalizer",
                  "normalization_error": float(abs(fine.sum()-1)),
                  "maximum_probability_refinement_difference": float(abs(fine-coarse).max()),
                  "probabilities_sha256": _array_sha(fine)}


def _family(cfg, role, profile, anchor, index):
    return stable_hash([STREAM, cfg.seed, cfg.config_hash, role, profile, anchor, index])


def _rng(cfg, role, profile, anchor, index, purpose):
    return seeded_rng(cfg.seed, STREAM, cfg.config_hash, role, profile, anchor, index, purpose)


def _chunk_digest(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _datasets(group, shapes, chunk):
    for name, (shape, dtype) in shapes.items():
        group.create_dataset(name, shape, dtype=dtype,
                             chunks=(min(chunk, shape[0]), *shape[1:]), compression="lzf", shuffle=True)


def _state_shapes(cfg, role):
    count = cfg.train_paths if role == "train" else cfg.validation_paths
    return {"states": ((count, len(cfg.state_times), cfg.n_cells), "f8"),
            "horizons": ((count, len(cfg.state_times), cfg.horizon_samples), "f8"),
            "stopped_at": ((count,), "f8"), "path_minimum_mass": ((count,), "f8"),
            "family_ids": ((count,), "S64"),
            "initial_sha256": ((count,), "S64"), "brownian_sha256": ((count,), "S64")}


def _state_identity(cfg, role):
    return {"kind": WORKFLOW+"-state-only-bank", "source_hash": source_hash(),
            "config_hash": cfg.config_hash, "target_id": cfg.target_id, "role": role,
            "stream": STREAM, "future_payoff_labels": False}


def _state_chunk(cfg, role, profile, start, count, backend):
    steps, probabilities = round(cfg.horizon/cfg.state_step), profile_probabilities(cfg, profile)[0]
    initial, noise, horizons, families = [], [], [], []
    for index in range(start, start+count):
        initial.append(_rng(cfg, role, profile, 0, index, "initial").multinomial(
            cfg.n_particles, probabilities)/cfg.n_particles)
        noise.append(_rng(cfg, role, profile, 0, index, "physical-forcing").normal(
            0, np.sqrt(cfg.state_step), (steps, cfg.n_cells)))
        horizons.append(_rng(cfg, role, profile, 0, index, "residual-horizons").uniform(
            0, cfg.horizon, (len(cfg.state_times), cfg.horizon_samples)))
        families.append(_family(cfg, role, profile, 0, index))
    initial, noise = np.asarray(initial), np.asarray(noise)
    packet = backend(initial, noise, cfg.context(profile), cfg.spec, cfg.state_step)
    _validate_paths(packet, cfg.horizon)
    indices = np.rint(np.asarray(cfg.state_times)/cfg.state_step).astype(int)
    return {"states": packet["states"][:, indices], "horizons": np.asarray(horizons),
            "stopped_at": packet["stopped_at"],
            "path_minimum_mass": packet["states"].min(axis=(1, 2)),
            "family_ids": np.asarray(families, dtype="S64"),
            "initial_sha256": np.asarray([_array_sha(row) for row in initial], dtype="S64"),
            "brownian_sha256": np.asarray([_array_sha(row) for row in noise], dtype="S64")}


def _validate_paths(packet, horizon):
    states, stops = np.asarray(packet["states"]), np.asarray(packet["stopped_at"])
    if (not np.isfinite(states).all() or states.min() < -1e-12
            or not np.allclose(states.sum(-1), 1., atol=1e-10, rtol=0)
            or stops.shape != (len(states),) or np.isnan(stops).any() or (stops < 0).any()
            or (np.isfinite(stops) & (stops > horizon+1e-12)).any()):
        raise ValueError("Invalid native paths: survivor-conditioned estimates are forbidden")


def _verify_chunks(group, shapes, chunk_paths):
    expected = set(shapes)|{"chunk_complete", "chunk_sha256", "chunk_seconds"}
    if set(group) != expected:
        raise ValueError("Stored dataset inventory changed")
    for name, (shape, dtype) in shapes.items():
        if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
            raise ValueError("Stored dataset shape/dtype changed")
    count = next(iter(shapes.values()))[0][0]
    chunks = math.ceil(count/chunk_paths)
    if any(group[name].shape != (chunks,) for name in ("chunk_complete", "chunk_sha256", "chunk_seconds")):
        raise ValueError("Stored chunk inventory changed")
    for chunk in np.flatnonzero(group["chunk_complete"][:]):
        start, stop = int(chunk)*chunk_paths, min((int(chunk)+1)*chunk_paths, count)
        values = {name: group[name][start:stop] for name in shapes}
        if _chunk_digest(values) != group["chunk_sha256"][chunk].decode():
            raise ValueError("A committed native trajectory chunk changed")


def _add_chunk_metadata(group, count, chunk):
    chunks = math.ceil(count/chunk)
    group.create_dataset("chunk_complete", (chunks,), dtype="?")
    group.create_dataset("chunk_sha256", (chunks,), dtype="S64")
    group.create_dataset("chunk_seconds", (chunks,), dtype="f8")


def _state_packet(path, cfg, role, chunk_paths):
    identity = _state_identity(cfg, role)
    with h5py.File(path, "r") as store:
        if json.loads(store.attrs["identity_json"]) != identity or set(store) != set(cfg.cases):
            raise ValueError("State bank identity changed")
        for profile in cfg.cases:
            _verify_chunks(store[profile], _state_shapes(cfg, role), chunk_paths)
        complete = all(bool(store[p]["chunk_complete"][:].all()) for p in cfg.cases)
        hashes = {p: [x.decode() for x in store[p]["chunk_sha256"][:]] for p in cfg.cases}
        seconds = sum(float(store[p]["chunk_seconds"][:].sum()) for p in cfg.cases)
    return {**identity, "complete": complete, "content_hash": stable_hash(hashes),
            "generation_seconds": seconds, "sha256": _sha(path), "path": str(Path(path).resolve()),
            "all_paths_retained": True}


def ensure_state_bank(directory, cfg, role, plan, backend, deadline=None):
    if role not in ("train", "validation"):
        raise ValueError("Only train/validation current states are learner inputs")
    path = Path(directory)/f"states_{role}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = cfg.train_paths if role == "train" else cfg.validation_paths
    if not path.exists():
        with h5py.File(path, "w") as store:
            store.attrs["identity_json"] = json.dumps(_state_identity(cfg, role), sort_keys=True)
            for profile in cfg.cases:
                group = store.create_group(profile)
                _datasets(group, _state_shapes(cfg, role), plan.chunk_paths)
                _add_chunk_metadata(group, count, plan.chunk_paths)
    _state_packet(path, cfg, role, plan.chunk_paths)
    with h5py.File(path, "r+") as store:
        for profile in cfg.cases:
            group = store[profile]
            for chunk, complete in enumerate(group["chunk_complete"][:]):
                if complete:
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    _json(path.with_suffix(".json"), _state_packet(path, cfg, role, plan.chunk_paths))
                    raise BudgetExhausted("State collocation saved at its resumable deadline")
                start, begun = chunk*plan.chunk_paths, time.monotonic()
                values = _state_chunk(cfg, role, profile, start, min(plan.chunk_paths, count-start), backend)
                for name, array in values.items():
                    group[name][start:start+len(array)] = array
                group["chunk_sha256"][chunk] = _chunk_digest(values).encode()
                group["chunk_seconds"][chunk] = time.monotonic()-begun
                store.flush()
                group["chunk_complete"][chunk] = True
                store.flush()
    packet = _state_packet(path, cfg, role, plan.chunk_paths)
    _json(path.with_suffix(".json"), packet)
    return packet


def load_state_bank(root, case, role):
    plan, _ = _assert_frozen(root)
    cfg, path = configuration(case, plan), Path(root)/"payoffs"/case/f"states_{role}.h5"
    packet = _state_packet(path, cfg, role, plan.chunk_paths)
    if not packet["complete"] or _read(path.with_suffix(".json")) != packet:
        raise ValueError("Only complete unmutated state banks may enter training")
    states, horizons, families, profiles, snapshot_times = [], [], [], [], []
    with h5py.File(path, "r") as store:
        for profile in cfg.cases:
            group = store[profile]
            states.append(np.repeat(group["states"][:], cfg.horizon_samples, axis=1).reshape(-1, cfg.n_cells))
            horizons.append(group["horizons"][:].reshape(-1))
            repeats = len(cfg.state_times)*cfg.horizon_samples
            families.extend(np.repeat(group["family_ids"][:].astype(str), repeats))
            shape = group["horizons"].shape
            profiles.extend([profile]*int(np.prod(shape)))
            snapshot_times.append(np.broadcast_to(np.asarray(cfg.state_times)[None, :, None], shape).reshape(-1))
    return {**packet, "states": np.concatenate(states), "horizons": np.concatenate(horizons),
            "family_ids": tuple(families), "case_ids": tuple(profiles),
            "snapshot_times": np.concatenate(snapshot_times), "split_unit": "independent-trajectory-family"}


def _data_lock(root, plan):
    records, seen = {}, set()
    for case in case_names(plan):
        records[case] = {}
        for role in ("train", "validation"):
            bank = load_state_bank(root, case, role)
            families = set(bank["family_ids"])
            if seen.intersection(families):
                raise ValueError("State bank trajectory families overlap across splits/regimes")
            seen.update(families)
            path = Path(bank["path"])
            records[case][role] = {"path": str(path), "sha256": bank["sha256"],
                                   "metadata_path": str(path.with_suffix(".json")),
                                   "metadata_sha256": _sha(path.with_suffix(".json")),
                                   "content_hash": bank["content_hash"],
                                   "independent_families": len(families),
                                   "collocation_rows": len(bank["states"]),
                                   "family_ids_hash": stable_hash(sorted(families)),
                                   "generation_seconds": bank["generation_seconds"]}
    return {"workflow": WORKFLOW, "source_hash": source_hash(), "plan_hash": plan.plan_hash,
            "preregistration_sha256": _sha(Path(root)/"payoffs"/"preregistration.json"),
            "banks": records, "all_family_splits_disjoint": True, "future_payoff_labels": False}


def calibration(root):
    """Training-only reachability/stopping diagnostics, never test selection."""
    plan, _ = _assert_frozen(root)
    output = {"workflow": WORKFLOW, "source_hash": source_hash(), "plan_hash": plan.plan_hash,
              "selection_using_diagnostics": False, "held_out_inputs_read": False, "cases": {}}
    for case in case_names(plan):
        cfg, records = configuration(case, plan), {}
        path = Path(root)/"payoffs"/case/"states_train.h5"
        with h5py.File(path, "r") as store:
            for profile in cfg.cases:
                group = store[profile]
                states, stops = group["states"][:], group["stopped_at"][:]
                weights = PayoffBank._weights(cfg.spec, ((1, "cos"), (1, "sin")))
                amplitudes = np.linalg.norm(states@weights.T, axis=-1)
                records[profile] = {"independent_families": len(states),
                                    "stopped_paths": int(np.isfinite(stops).sum()),
                                    "stopped_fraction": float(np.isfinite(stops).mean()),
                                    "minimum_density": float(states.min()/cfg.spec.h),
                                    "path_minimum_mass_quantiles": np.quantile(
                                        group["path_minimum_mass"][:], [0., .01, .05, .5, 1.]).tolist(),
                                    "fraction_path_minimum_below_stop_multiples": {
                                        str(multiple): float(np.mean(group["path_minimum_mass"][:]
                                                                    <= multiple*cfg.spec.mass_stop))
                                        for multiple in (1., 2., 5., 10.)},
                                    "active_fraction_by_snapshot": [float(np.mean(stops > t))
                                                                    for t in cfg.state_times],
                                    "residual_horizon_quantiles": np.quantile(
                                        group["horizons"][:], [0., .01, .25, .5, .75, .99, 1.]).tolist(),
                                    "maximum_mass_error": float(abs(states.sum(-1)-1).max()),
                                    "mode1_amplitude_quantiles_by_snapshot": np.quantile(
                                        amplitudes, [0., .25, .5, .75, 1.], axis=0).tolist(),
                                    "state_times": list(cfg.state_times),
                                    "state_file_sha256": _sha(path)}
        output["cases"][case] = records
    _immutable(Path(root)/"payoffs"/"training_calibration.json", output)
    return output


def prepare(root, plan=None, deadline=None):
    root = Path(root).resolve()
    plan = (load_plan(root) if (root/"plan.json").exists() else StudyPlan()) if plan is None else plan
    with _exclusive(root/"payoffs"/"execution"/".prepare.lock"):
        if (root/"payoffs"/"preregistration.json").exists():
            if load_plan(root) != plan:
                raise ValueError("Frozen StudyPlan changed")
            _assert_frozen(root)
        else:
            if any(path.name != "execution" for path in (root/"payoffs").iterdir()):
                raise ValueError("Unrecognized existing payoff outputs; choose a fresh root")
            _immutable(root/"plan.json", plan.to_dict())
            _immutable(root/"payoffs"/"preregistration.json", _design(plan))
        backend = compile_native(root/"compiled")
        _immutable(root/"payoffs"/"compiled_backend.json", {
            key: value for key, value in backend.metadata.items()
            if key not in ("compile_seconds_this_process", "cache_hit")})
        for case in case_names(plan):
            for role in ("train", "validation"):
                _assert_frozen(root)
                ensure_state_bank(root/"payoffs"/case, configuration(case, plan), role, plan,
                                  backend, deadline)
        _immutable(root/"payoffs"/"data_lock.json", _data_lock(root, plan))
        calibration(root)
        for case in case_names(plan):
            prepare_reference(root, case)
    return root/"payoffs"


def _prepared(root):
    plan, design = _assert_frozen(root)
    if _read(Path(root)/"payoffs"/"data_lock.json") != _data_lock(root, plan):
        raise ValueError("State collocation data lock changed")
    return plan, design


def _checkpoint(root, slot):
    return Path(root)/"payoffs"/slot["case"]/"checkpoints"/f"seed{slot['seed']}.pt"


def load_declared_model(item):
    path = Path(item["checkpoint_path"])
    if item.get("variant", "repaired") != "repaired":
        raise ValueError("Only declared repaired models belong to this study")
    if item.get("checkpoint_sha256", _sha(path)) != _sha(path):
        raise ValueError("Declared repaired checkpoint changed")
    return load_repaired_payoff_model(path)


def _model_record(root, slot, plan):
    path = _checkpoint(root, slot)
    model, metadata = load_declared_model({**slot, "checkpoint_path": str(path)}), _read(path.with_suffix(".json"))
    expected = repaired_configuration(slot["case"], plan)
    seconds = metadata.get("training_seconds", -1)
    if (not model.complete or not metadata.get("complete")
            or model.provenance.get("source_hash") != source_hash()
            or model.provenance.get("config_hash") != expected.config_hash
            or model.provenance.get("seed") != slot["seed"]
            or model.bank.bank_hash != payoff_bank(slot["case"], plan).bank_hash
            or metadata.get("identity_hash") != model.provenance.get("identity_hash")
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Repaired model slot configuration, bank, source or seed mismatch")
    return {**slot, "checkpoint_path": str(path.resolve()), "checkpoint_sha256": _sha(path),
            "path": str(path.resolve()), "metadata_path": str(path.with_suffix(".json").resolve()),
            "metadata_sha256": _sha(path.with_suffix(".json")), "complete": True,
            "source_hash": source_hash(), "config_hash": expected.config_hash,
            "target_id": model.target_id, "bank_hash": model.bank.bank_hash,
            "model_kind": model.kind, "training_seconds": seconds,
            "selected_step": metadata.get("best_step")}


def fit_seed(root, seed, deadline=None):
    plan, design = _prepared(root)
    if type(seed) is not int or seed not in plan.optimizer_seeds:
        raise ValueError("Use one preregistered optimizer seed")
    root = Path(root)
    with _exclusive(root/"payoffs"/"execution"/f".seed{seed}.lock"):
        if (root/"payoffs"/"fit_lock.json").exists():
            raise ValueError("Final fit lock exists; fitting is closed")
        output = root/"payoffs"/"execution"/f"worker_seed{seed}.json"
        previous = _read(output).get("attempts", []) if output.exists() else []
        started, cpu = time.monotonic(), time.process_time()
        attempt = {"pid": os.getpid(), "started_unix": time.time(), "complete": False}
        packet = {"workflow": WORKFLOW, "seed": seed, "source_hash": source_hash(),
                  "preregistration_sha256": _sha(root/"payoffs"/"preregistration.json"),
                  "pid": os.getpid(), "complete": False, "models": [], "attempts": [*previous, attempt],
                  "held_out_reference_inputs_read": False}
        _json(output, packet)
        try:
            for slot in design["slots"]:
                if slot["seed"] != seed:
                    continue
                _assert_frozen(root)
                if deadline is not None and time.monotonic() >= deadline:
                    raise BudgetExhausted("Regime seed worker reached its resumable deadline")
                train, validation = (load_state_bank(root, slot["case"], role)
                                     for role in ("train", "validation"))
                path = _checkpoint(root, slot)
                print(f"[seed {seed}] {slot['case']} repaired", flush=True)
                metadata = fit_repaired_payoff(repaired_configuration(slot["case"], plan),
                                               payoff_bank(slot["case"], plan), seed, path,
                                               train, validation, deadline=deadline)
                _json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    raise BudgetExhausted("Regime optimizer checkpointed before completion")
                packet["models"].append(_model_record(root, slot, plan))
                _json(output, packet)
            packet["complete"] = True
        finally:
            attempt.update(complete=packet["complete"], finished_unix=time.time(),
                           wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu)
            packet.update(wall_seconds=attempt["wall_seconds"], cpu_seconds=attempt["cpu_seconds"])
            _json(output, packet)
    return packet


def lock(root):
    plan, design = _prepared(root)
    root = Path(root)
    with _exclusive(root/"payoffs"/"execution"/".fit-lock.lock"), ExitStack() as stack:
        for seed in plan.optimizer_seeds:
            stack.enter_context(_exclusive(root/"payoffs"/"execution"/f".seed{seed}.lock"))
            worker = _read(root/"payoffs"/"execution"/f"worker_seed{seed}.json")
            if (not worker.get("complete") or worker.get("source_hash") != source_hash()
                    or worker.get("preregistration_sha256") != _sha(root/"payoffs"/"preregistration.json")):
                raise ValueError("All preregistered optimizer workers must complete first")
        models = [_model_record(root, slot, plan) for slot in design["slots"]]
        if len({row["name"] for row in models}) != len(slots(plan)):
            raise ValueError("Duplicated or incomplete model inventory")
        result = {"workflow": WORKFLOW, "complete": True, "source_hash": source_hash(),
                  "expected_models": len(slots(plan)), "models": models,
                  "plan_hash": plan.plan_hash, "runner_sha256": _sha(SCRIPT),
                  "preregistration_path": str((root/"payoffs"/"preregistration.json").resolve()),
                  "preregistration_sha256": _sha(root/"payoffs"/"preregistration.json"),
                  "data_lock_path": str((root/"payoffs"/"data_lock.json").resolve()),
                  "data_lock_sha256": _sha(root/"payoffs"/"data_lock.json"),
                  "selection_using_final_outcomes": False, "seeds_pooled": False,
                  "total_recorded_model_training_seconds": sum(row["training_seconds"] for row in models)}
        _immutable(root/"payoffs"/"fit_lock.json", result)
    return result


def verify_fit_lock(root):
    plan, design = _prepared(root)
    root = Path(root)
    saved = _read(root/"payoffs"/"fit_lock.json")
    models = [_model_record(root, slot, plan) for slot in design["slots"]]
    if (not saved.get("complete") or saved.get("source_hash") != source_hash()
            or saved.get("expected_models") != len(slots(plan)) or saved.get("models") != models
            or saved.get("plan_hash") != plan.plan_hash
            or saved.get("data_lock_sha256") != _sha(root/"payoffs"/"data_lock.json")
            or saved.get("preregistration_sha256") != _sha(root/"payoffs"/"preregistration.json")):
        raise ValueError("All declared complete fits must strictly restore before reference unlock")
    return saved


def native_payoff(initial, increments, context, spec, dt, bank, query_times, backend=None):
    """Complete native paths and exact killed-polygon payoff reduction.

    The timing caller supplies fresh initialization/noise separately, permitting
    explicit inclusion of those costs. No trajectory is filtered or repaired.
    """
    backend = compile_native() if backend is None else backend
    packet = backend(initial, increments, context, spec, dt)
    horizon = increments.shape[1]*dt
    _validate_paths(packet, horizon)
    times = np.arange(increments.shape[1]+1)*dt
    queries = np.asarray(query_times, dtype=float)
    indices = np.rint(queries/dt).astype(int)
    if (queries.ndim != 1 or not len(queries) or not np.isfinite(queries).all()
            or (queries < 0).any() or (queries > horizon+1e-12).any()
            or not np.allclose(indices*dt, queries, atol=1e-11, rtol=0)):
        raise ValueError("Requested payoff times must lie on the complete native grid")
    values = bank.evaluate_paths(packet["states"], times, packet["stopped_at"])
    return {"outcomes": values[:, indices], "stopped_at": packet["stopped_at"],
            "mass_error": float(abs(packet["states"].sum(-1)-1).max()),
            "path_minimum_mass": packet["states"].min(axis=(1, 2)),
            "minimum_mass": float(packet["states"].min())}


def _reference_identity(cfg):
    return {"kind": WORKFLOW+"-reference", "config": cfg.to_dict(), "config_hash": cfg.config_hash,
            "source_hash": source_hash(), "runner_sha256": _sha(SCRIPT), "stream": STREAM,
            "payoff_bank": PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon).to_dict(),
            "cpp_sha256": _sha(SCRIPT.with_name("native_euler_control.cpp")),
            "wrapper_sha256": _sha(SCRIPT.with_name("compiled_native.py")),
            "rng": "NumPy-PCG64-SHA256-addressed-independent-family-streams-v1"}


def prepare_reference(root, case):
    plan, _ = _assert_frozen(root)
    cfg = reference_configuration(case, plan)
    directory = Path(root)/"reference"/case
    path, identity = directory/"manifest.json", _reference_identity(cfg)
    if path.exists():
        manifest = _read(path)
        if manifest.get("identity") != identity:
            raise ValueError("Reference source, configuration or backend changed")
        return manifest
    manifest = {"identity": identity, "status": "waiting-for-complete-fit-lock",
                "groups": list(cfg.group_keys), "committed_chunks": 0, "generation_seconds": 0.,
                "continuous_reference_bias": "unresolved", "sampling_accuracy_certificate": "unresolved"}
    _json(path, manifest)
    return manifest


def _reference_shapes(cfg):
    n, r, g, q, f = cfg.paths_per_group, cfg.refinement_paths, cfg.n_cells, len(cfg.query_times), len(cfg.reference_factors)
    return {"initial": ((n, g), "f8"), "family_ids": ((n,), "S64"),
            "base_brownian_sha256": ((n,), "S64"), "drawn_brownian_sha256": ((n,), "S64"),
            "outcomes": ((n, q, 4), "f8"), "stopped_at": ((n,), "f8"),
            "path_minimum_mass": ((n,), "f8"),
            "audit_outcomes": ((r, f, q, 4), "f8"), "audit_stopped_at": ((r, f), "f8"),
            "audit_path_minimum_mass": ((r, f), "f8"),
            "audit_brownian_sha256": ((r, f), "S64")}


def _reference_chunk_arrays(group, cfg, start, count):
    return {name: group[name][start:min(start+count, shape[0])]
            for name, (shape, _) in _reference_shapes(cfg).items() if start < shape[0]}


def _reference_digest(group, cfg, start, count):
    return _chunk_digest(_reference_chunk_arrays(group, cfg, start, count))


def _create_reference(path, cfg, manifest, lock):
    with h5py.File(path, "w") as store:
        store.attrs["identity_json"] = json.dumps(manifest["identity"], sort_keys=True)
        store.attrs["fit_inventory_hash"] = stable_hash(lock)
        store.create_dataset("query_times", data=cfg.query_times)
        store.create_dataset("reference_factors", data=cfg.reference_factors)
        for key in cfg.group_keys:
            profile, role, anchor = key.split("__")
            group = store.create_group(key)
            probabilities, audit = profile_probabilities(cfg, profile)
            group.attrs.update(case=profile, role=role, anchor_id=int(anchor), context_id=cfg.context(profile).context_id)
            group.attrs["profile_probability_audit_json"] = json.dumps(audit, sort_keys=True)
            group.create_dataset("probabilities", data=probabilities)
            if role == "conditional":
                initial = _rng(cfg, "held-out-anchor", profile, int(anchor), 0, "initial").multinomial(
                    cfg.n_particles, probabilities)/cfg.n_particles
                group.attrs["conditional_initial_json"] = json.dumps(initial.tolist())
                group.attrs["conditional_initial_sha256"] = _array_sha(initial)
            _datasets(group, _reference_shapes(cfg), cfg.chunk_paths)
            _add_chunk_metadata(group, cfg.paths_per_group, cfg.chunk_paths)


def _aggregate(increments, factor):
    if type(factor) is not int or factor < 1 or increments.shape[1] % factor:
        raise ValueError("Integer nested Brownian aggregation required")
    return increments.reshape(len(increments), increments.shape[1]//factor, factor, increments.shape[-1]).sum(2)


def reference_draw(cfg, key, start, count, conditional_initial=None):
    profile, role, anchor = key.split("__")
    anchor = int(anchor)
    probabilities, factor = profile_probabilities(cfg, profile)[0], cfg.reference_factors[-1] if start < cfg.refinement_paths else 1
    steps = round(cfg.horizon/cfg.reference_step)*factor
    initial, noise, families = [], [], []
    for index in range(start, start+count):
        initial.append(_rng(cfg, "reference-"+role, profile, anchor, index, "initial").multinomial(
            cfg.n_particles, probabilities)/cfg.n_particles if role == "ensemble" else np.asarray(conditional_initial))
        noise.append(_rng(cfg, "reference-"+role, profile, anchor, index, "physical-forcing").normal(
            0, np.sqrt(cfg.reference_step/factor), (steps, cfg.n_cells)))
        families.append(_family(cfg, "reference-"+role, profile, anchor, index))
    return np.asarray(initial), np.asarray(noise), int(factor), np.asarray(families, dtype="S64")


def simulate_reference_chunk(cfg, key, start, count, conditional_initial=None, backend=None):
    backend = compile_native() if backend is None else backend
    initial, finest, finest_factor, families = reference_draw(cfg, key, start, count, conditional_initial)
    base = _aggregate(finest, finest_factor)
    profile = key.split("__")[0]
    bank = PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)
    reduced = native_payoff(initial, base, cfg.context(profile), cfg.spec, cfg.reference_step,
                            bank, cfg.query_times, backend)
    result = {"initial": initial, "family_ids": families, "outcomes": reduced["outcomes"],
              "stopped_at": reduced["stopped_at"],
              "path_minimum_mass": reduced["path_minimum_mass"],
              "base_brownian_sha256": np.asarray([_array_sha(x) for x in base], dtype="S64"),
              "drawn_brownian_sha256": np.asarray([_array_sha(x) for x in finest], dtype="S64")}
    audited = min(count, max(0, cfg.refinement_paths-start))
    if audited:
        values, stops, minima, hashes = [], [], [], []
        for factor in cfg.reference_factors:
            dw = _aggregate(finest[:audited], finest_factor//factor)
            level = (reduced if factor == 1 else native_payoff(
                initial[:audited], dw, cfg.context(profile), cfg.spec, cfg.reference_step/factor,
                bank, cfg.query_times, backend))
            values.append(level["outcomes"][:audited])
            stops.append(level["stopped_at"][:audited])
            minima.append(level["path_minimum_mass"][:audited])
            hashes.append(np.asarray([_array_sha(x) for x in dw], dtype="S64"))
        result.update(audit_outcomes=np.stack(values, axis=1), audit_stopped_at=np.stack(stops, axis=1),
                      audit_path_minimum_mass=np.stack(minima, axis=1),
                      audit_brownian_sha256=np.stack(hashes, axis=1))
    return result


def _verify_reference_store(store, cfg, identity, lock):
    if (json.loads(store.attrs["identity_json"]) != identity
            or store.attrs["fit_inventory_hash"] != stable_hash(lock)
            or set(store) != set(cfg.group_keys)|{"query_times", "reference_factors"}
            or not np.array_equal(store["query_times"][:], cfg.query_times)
            or not np.array_equal(store["reference_factors"][:], cfg.reference_factors)):
        raise ValueError("Native reference inventory, target or complete fit lock changed")
    shapes = _reference_shapes(cfg)
    for key in cfg.group_keys:
        group = store[key]
        if set(group) != set(shapes)|{"probabilities", "chunk_complete", "chunk_sha256", "chunk_seconds"}:
            raise ValueError("Native reference group inventory changed")
        for name, (shape, dtype) in shapes.items():
            if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
                raise ValueError("Native reference shape/dtype changed")
        profile, role, anchor = key.split("__")
        probabilities, _ = profile_probabilities(cfg, profile)
        if not np.array_equal(group["probabilities"][:], probabilities):
            raise ValueError("Native reference initial-law probabilities changed")
        if role == "conditional":
            initial = np.asarray(json.loads(group.attrs["conditional_initial_json"]))
            expected = _rng(cfg, "held-out-anchor", profile, int(anchor), 0, "initial").multinomial(
                cfg.n_particles, probabilities)/cfg.n_particles
            if not np.array_equal(initial, expected) or _array_sha(initial) != group.attrs["conditional_initial_sha256"]:
                raise ValueError("Native held-out anchor changed")
        chunks = math.ceil(cfg.paths_per_group/cfg.chunk_paths)
        if any(group[name].shape != (chunks,) for name in ("chunk_complete", "chunk_sha256", "chunk_seconds")):
            raise ValueError("Native reference chunk metadata changed")
        for chunk in np.flatnonzero(group["chunk_complete"][:]):
            start = int(chunk)*cfg.chunk_paths
            count = min(cfg.chunk_paths, cfg.paths_per_group-start)
            if _reference_digest(group, cfg, start, count) != group["chunk_sha256"][chunk].decode():
                raise ValueError("A committed native reference trajectory chunk changed")


def generate_reference(root, case=None, deadline=None, max_chunks=None):
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Nonnegative integer reference chunk limit required")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("Finite absolute monotonic deadline required")
    root = Path(root).resolve()
    plan, _ = _assert_frozen(root)
    cases = case_names(plan) if case is None else (case,)
    if any(item not in case_names(plan) for item in cases):
        raise ValueError("Unknown predeclared reference regime")
    # This strict restoration happens before backend construction or anchor draw.
    fit_lock = verify_fit_lock(root)
    inventory = {"source_hash": source_hash(), "expected_models": len(slots(plan)),
                 "models": [{k: row[k] for k in ("name", "checkpoint_path", "checkpoint_sha256", "complete")}
                            for row in fit_lock["models"]]}
    output, completed_now = {}, 0
    backend = compile_native(root/"compiled")
    for name in cases:
        cfg, directory = reference_configuration(name, plan), root/"reference"/name
        with _exclusive(directory/"execution"/".generation.lock"):
            manifest = prepare_reference(root, name)
            if manifest.get("fit_lock") not in (None, inventory):
                raise ValueError("Reference fit lock changed after unlock")
            manifest["fit_lock"] = inventory
            path = directory/"reference.h5"
            if not path.exists():
                _create_reference(path, cfg, manifest, inventory)
            if "failure" in manifest:
                manifest.setdefault("recovered_attempt_failures", []).append(manifest.pop("failure"))
            manifest["status"] = "generating"
            _json(directory/"manifest.json", manifest)
            with h5py.File(path, "r+") as store:
                _verify_reference_store(store, cfg, manifest["identity"], inventory)
                for key in cfg.group_keys:
                    group = store[key]
                    conditional = json.loads(group.attrs["conditional_initial_json"]) if "__conditional__" in key else None
                    for chunk, complete in enumerate(group["chunk_complete"][:]):
                        if complete:
                            continue
                        if ((max_chunks is not None and completed_now >= max_chunks)
                                or (deadline is not None and time.monotonic() >= deadline)):
                            manifest["status"] = "partial"
                            _json(directory/"manifest.json", manifest)
                            output[name] = manifest
                            return output
                        start, begun = chunk*cfg.chunk_paths, time.monotonic()
                        count = min(cfg.chunk_paths, cfg.paths_per_group-start)
                        try:
                            values = simulate_reference_chunk(cfg, key, start, count, conditional, backend)
                            for key_array, array in values.items():
                                group[key_array][start:start+len(array)] = array
                            group["chunk_sha256"][chunk] = _reference_digest(group, cfg, start, count).encode()
                            group["chunk_seconds"][chunk] = time.monotonic()-begun
                            store.flush()
                            group["chunk_complete"][chunk] = True
                            store.flush()
                        except Exception as error:
                            manifest.update(status="failed", failure={"group": key, "chunk": chunk,
                                "type": type(error).__name__, "message": str(error),
                                "survivor_conditioned_estimates_allowed": False})
                            _json(directory/"manifest.json", manifest)
                            raise
                        completed_now += 1
                        manifest.update(committed_chunks=sum(int(store[k]["chunk_complete"][:].sum()) for k in cfg.group_keys),
                                        generation_seconds=sum(float(store[k]["chunk_seconds"][:].sum()) for k in cfg.group_keys))
                        _json(directory/"manifest.json", manifest)
                manifest.update(status="complete", failed_paths=0,
                                stopped_paths_by_group={key: int(np.isfinite(store[key]["stopped_at"][:]).sum())
                                                        for key in cfg.group_keys},
                                group_content_hashes={key: stable_hash([x.decode() for x in store[key]["chunk_sha256"][:]])
                                                      for key in cfg.group_keys})
            manifest.update(reference_path=str(path.resolve()), reference_sha256=_sha(path))
            _json(directory/"manifest.json", manifest)
            output[name] = manifest
    return output


def load_reference(root, case, group_key=None):
    plan, _ = _assert_frozen(root)
    cfg, directory = reference_configuration(case, plan), Path(root)/"reference"/case
    manifest = prepare_reference(root, case)
    path = directory/"reference.h5"
    if (manifest.get("status") != "complete" or not path.is_file()
            or _sha(path) != manifest.get("reference_sha256")):
        raise ValueError("Only complete, unmutated independent reference ensembles may produce estimates")
    if group_key is not None and group_key not in cfg.group_keys:
        raise ValueError("Unknown reference group")
    keys, groups, seen = cfg.group_keys if group_key is None else (group_key,), {}, set()
    with h5py.File(path, "r") as store:
        _verify_reference_store(store, cfg, manifest["identity"], manifest["fit_lock"])
        for key in cfg.group_keys:
            families = [x.decode() for x in store[key]["family_ids"][:]]
            if len(set(families)) != len(families) or seen.intersection(families):
                raise ValueError("Independent reference trajectory families overlap")
            seen.update(families)
        for key in keys:
            group = store[key]
            groups[key] = {name: group[name][:] for name in _reference_shapes(cfg)}
            groups[key].update(probabilities=group["probabilities"][:], metadata=dict(group.attrs))
    result = {"manifest": manifest, "query_times": np.asarray(cfg.query_times),
              "reference_factors": np.asarray(cfg.reference_factors), "groups": groups}
    if group_key is not None:
        result.update(groups[group_key])
    return result


def iter_reference_groups(root, case):
    plan = load_plan(root)
    for key in reference_configuration(case, plan).group_keys:
        yield key, load_reference(root, case, group_key=key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibrate", "fit", "lock", "reference"))
    parser.add_argument("--root", type=Path, default=REPO/"runs"/"payoff-regimes-2026-10-28")
    parser.add_argument("--plan", type=Path, help="Complete StudyPlan JSON; prepare only")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--budget-seconds", type=float, default=8*60*60)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 8*60*60
            or (args.stage == "fit") != (args.seed is not None)
            or (args.plan is not None and args.stage != "prepare")
            or (args.case is not None and args.stage != "reference")
            or (args.max_chunks is not None and (args.stage != "reference" or args.max_chunks < 0))):
        parser.error("Use a positive <=8h budget, seed only for fit, plan only for prepare, case/chunk limit only for reference")
    deadline = time.monotonic()+args.budget_seconds
    try:
        if args.stage == "prepare":
            plan = StudyPlan.from_dict(_read(args.plan)) if args.plan else None
            print(prepare(args.root, plan, deadline), flush=True)
        elif args.stage == "calibrate":
            print(json.dumps(calibration(args.root)), flush=True)
        elif args.stage == "fit":
            result = fit_seed(args.root, args.seed, deadline)
            print(f"Completed seed {args.seed}: {len(result['models'])} fits", flush=True)
        elif args.stage == "lock":
            print(f"Locked {lock(args.root)['expected_models']} fits", flush=True)
        else:
            result = generate_reference(args.root, args.case, deadline, args.max_chunks)
            print(json.dumps({key: {name: value[name] for name in ("status", "committed_chunks")}
                              for key, value in result.items()}), flush=True)
    except BudgetExhausted as error:
        print(str(error), file=sys.stderr, flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
