"""A5 comparison of native stopped DK and NN with direct particle observables.

This intentionally compares different laws. It does not reinterpret particles as
native DK states, certify a continuum limit, or discard absorbed trajectories.
Only small query vectors survive each chunk; microscopic paths are discarded.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .data import source_hash
from .evaluation import json_safe
from .native import extract_observables, physical_observable_weights, simulate_native
from .particles import sample_initial, simulate_particles
from .types import TargetSpec, seeded_rng, stable_hash

_METHODS = ("particles", "native_reference", "native_nn_step", "nn")


def particle_physical_observables(positions, modes=(1, 2)):
    """Empirical averages of fixed sqrt(2) cos/sin(k x), never spline filters."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim < 1 or not positions.shape[-1] or not np.isfinite(positions).all():
        raise ValueError("Finite particle positions and a nonempty particle axis required")
    return np.sqrt(2) * np.stack([function(k * positions).mean(-1)
                                 for k in modes for function in (np.cos, np.sin)], axis=-1)


def _histograms(positions, spec):
    cells = np.floor((positions % spec.length) / spec.h).astype(int)
    masses = np.zeros((len(positions), spec.n_cells))
    np.add.at(masses, (np.arange(len(positions))[:, None], cells), 1 / positions.shape[-1])
    return masses


def _atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _model_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _moments(values, n_particles):
    mean = values.mean(0)
    centered = values - mean
    return mean, n_particles * centered.T @ centered / (len(values) - 1)


def _ci(values):
    return np.moveaxis(np.quantile(values, [0.025, 0.975], axis=0), 0, -1)


def _contrast(left, right, n_particles, bootstrap_indices):
    lm, lc = _moments(left, n_particles)
    rm, rc = _moments(right, n_particles)
    result = dict(sqrt_n_mean_difference=np.sqrt(n_particles) * (lm - rm),
                  scaled_covariance_difference=lc - rc,
                  scaled_variance_difference=np.diag(lc - rc),
                  sqrt_n_wasserstein=[np.sqrt(n_particles) * wasserstein_distance(left[:, i], right[:, i])
                                      for i in range(left.shape[1])],
                  normalization="continuous uniform-IID physical-mode standard deviations equal one",
                  uncertainty=dict(available=False))
    if bootstrap_indices is not None:
        means, covariances = [], []
        for indices in bootstrap_indices:
            lmean, lcov = _moments(left[indices], n_particles)
            rmean, rcov = _moments(right[indices], n_particles)
            means.append(np.sqrt(n_particles) * (lmean - rmean))
            covariances.append(lcov - rcov)
        result["uncertainty"] = dict(available=True, confidence_level=0.95,
            method="paired whole independent trajectory family percentile bootstrap",
            sqrt_n_mean_difference_ci95=_ci(means),
            scaled_covariance_difference_ci95=_ci(covariances),
            wasserstein="descriptive empirical distance; not bootstrapped",
            interpretation="pointwise sampling uncertainty only; excludes numerical and model bias")
    return result


def _summarize(chunks, context, config):
    values = {method: np.concatenate([chunk[method] for chunk in chunks]) for method in _METHODS}
    n = len(values["particles"])
    families = np.concatenate([chunk["family_ids"] for chunk in chunks])
    if len(set(families)) != n:
        raise ValueError("Particle-audit trajectory families must be independent and unique")
    counts = int(config.bootstrap_samples)
    rng = seeded_rng(config.seed, "particle-audit-bootstrap", context.context_id)
    indices = (rng.integers(0, n, (counts, n)) if n >= 2 and counts >= 2 else None)
    estimates = {}
    for method, values_ in values.items():
        mean, covariance = _moments(values_, context.n_particles)
        estimates[method] = dict(mean=mean, scaled_covariance=covariance,
                                 scaled_variances=np.diag(covariance), n_independent_paths=n)
    comparisons = {
        "nn_minus_native_at_same_step": _contrast(values["nn"], values["native_nn_step"], context.n_particles, indices),
        "native_time_discretization_difference": _contrast(values["native_nn_step"], values["native_reference"], context.n_particles, indices),
        "native_minus_particles": _contrast(values["native_reference"], values["particles"], context.n_particles, indices),
        "nn_minus_particles": _contrast(values["nn"], values["particles"], context.n_particles, indices),
    }
    stopping = {}
    times = np.asarray(config.queries.times)
    for method in _METHODS[1:]:
        stop = np.concatenate([chunk[f"{method}_stopped_at"] for chunk in chunks])
        flags = stop[:, None] <= times
        frozen_mask = np.repeat(flags, len(config.queries.modes) * 2, axis=1)
        stopping[method] = dict(times=times, cdf=flags.mean(0),
            initial_stopping_probability=float(np.mean(stop == 0)),
            stopping_probability=float(np.mean(stop <= times[-1])),
            restricted_mean_stopping_time=float(np.minimum(stop, times[-1]).mean()),
            stopped_mean_contribution=np.mean(values[method] * frozen_mask, axis=0),
            all_stopped_paths_included=True,
            interpretation="Frozen-path contribution, not a survivor-conditioned estimate or an unstopped counterfactual")
        if indices is not None:
            stopping[method]["cdf_ci95"] = _ci([flags[i].mean(0) for i in indices])
    initialization = np.concatenate([chunk["initial_projection_error"] for chunk in chunks])
    costs = {key: sum(float(chunk["costs"][i]) for chunk in chunks)
             for i, key in enumerate(("initialization_seconds", *[f"{m}_seconds" for m in _METHODS]))}
    return dict(estimates=estimates, comparisons=comparisons, stopping=stopping,
                n_independent_families=n, all_paths_included=True,
                shared_initialization=dict(law="IID particles with exact profile rejection sampling",
                    dk_state="histogram of the same particle sample",
                    initial_histogram_observation_minus_direct_particle_mean=initialization.mean(0),
                    sqrt_n_initial_projection_rmse=np.sqrt(context.n_particles * (initialization**2).mean(0)),
                    maximum_absolute_initial_projection_error=float(np.abs(initialization).max())),
                costs=costs,
                interpretation=dict(
                    neural_error="NN minus native DK at the same integration step",
                    native_particle_discrepancy="Includes spatial projection, stopped versus unstopped target differences, and numerical bias",
                    coupling="Same initial sample; shared face increments for the three native solvers; independent particle innovations",
                    physical_test_functions="Identical continuous sqrt(2) sine/cosine modes; field evaluations use exact cell integrals",
                    convergence_claim="Descriptive fixed-grid comparison only; no continuum DK convergence claim",
                    bias_certification="Particle timestep and DK reference bias require separate refinement audits"))


def audit_particles(config, context, model, output_dir, n_paths=None, deadline=None):
    """Run/resume A5 for configured particle counts and one selected fitted model.

    Chunk seeds/identities are stable. Deadlines are checked between atomic chunks;
    a completed chunk always includes particles and every stopped native path.
    Changing model/target/integration settings rejects an existing cached audit.
    """
    spec = config.spec
    spec.require_compatible(model.spec)
    count = config.diagnostic_paths if n_paths is None else n_paths
    if not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("At least two independent particle-audit paths required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    particle_numbers = tuple(config.particle_numbers) or (context.n_particles,)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in particle_numbers) or len(set(particle_numbers)) != len(particle_numbers):
        raise ValueError("Unique positive integer particle counts required")
    neural_dt = min(config.step_sizes)
    ratio = round(neural_dt / config.reference_dt)
    if ratio < 1 or not np.isclose(ratio * config.reference_dt, neural_dt, atol=1e-12, rtol=0):
        raise ValueError("NN integration timestep must be an integer multiple of the reference step")
    steps = round(config.times[-1] / config.reference_dt)
    if steps % ratio:
        raise ValueError("NN and reference Brownian grids must nest")
    batch_size = int(config.evaluation_batch_size)
    model_hash = _model_hash(model)
    source_fingerprint = source_hash()
    results = []
    complete = True
    for n_particles in particle_numbers:
        audit_context = replace(context, n_particles=int(n_particles))
        directory = output_dir / f"n{n_particles}"
        chunk_dir = directory / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        identity = dict(schema_version=1, stage="A5-fixed-grid-particle-audit-v1", context=asdict(audit_context),
                        native_target=asdict(spec), query_bank=asdict(config.queries), n_paths=int(count),
                        batch_size=batch_size, model_hash=model_hash, seed=int(config.seed),
                        source_hash=source_fingerprint, config_hash=config.config_hash,
                        particle_dt=config.reference_dt, native_reference_dt=config.reference_dt,
                        nn_dt=neural_dt, horizon=config.horizon)
        identity_hash = stable_hash(identity)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("identity_hash") != identity_hash:
                raise ValueError("Existing particle audit belongs to different data/model/settings")
        manifest = dict(identity=identity, identity_hash=identity_hash, complete=False,
                        target_laws=dict(particles="unstopped empirical interacting-particle law",
                                         native="stopped finite-volume DK law",
                                         neural="stopped learned finite-volume coefficient law"))
        _atomic_json(manifest, manifest_path)
        chunks = []
        for offset in range(0, count, batch_size):
            size = min(batch_size, count - offset)
            chunk_path = chunk_dir / f"chunk_{offset:08d}.npz"
            if not chunk_path.exists():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                seed = int(seeded_rng(config.seed, "particle-audit", audit_context.context_id, offset).integers(0, 2**63 - 1))
                families = tuple(stable_hash(["particle-audit-family", audit_context.context_id, config.seed, offset + i])[:24]
                                 for i in range(size))
                started = time.perf_counter()
                initial = sample_initial(audit_context, size, seeded_rng(seed, "audit-initial"))
                initial_masses = _histograms(initial, spec)
                init_error = initial_masses @ physical_observable_weights(spec, config.queries).T - particle_physical_observables(initial, config.queries.modes)
                initialization_seconds = time.perf_counter() - started
                # Only query snapshots are retained, then microscopic positions
                # and legacy spline observations are immediately discarded.
                particle_batch = simulate_particles(audit_context, TargetSpec(n_cells=spec.n_cells),
                    config.times, config.reference_dt, seed, size, initial_particles=initial, retain_particles=True)
                particle_y = particle_physical_observables(particle_batch.particle_states[:, 1:], config.queries.modes).reshape(size, -1)
                costs = [initialization_seconds, particle_batch.diagnostics["total_seconds"]]
                del particle_batch
                increments = seeded_rng(seed, "audit-native-face-noise").normal(
                    0, np.sqrt(config.reference_dt), (size, steps, spec.n_cells))
                fine = simulate_native(audit_context, spec, config.times, config.reference_dt, seed,
                    size, initial_masses=initial_masses, brownian_increments=increments, family_ids=families)
                coarse_dw = increments.reshape(size, steps // ratio, ratio, spec.n_cells).sum(2)
                coarse = simulate_native(audit_context, spec, config.times, neural_dt, seed,
                    size, initial_masses=initial_masses, brownian_increments=coarse_dw, family_ids=families)
                neural = simulate_native(audit_context, spec, config.times, neural_dt, seed,
                    size, initial_masses=initial_masses, brownian_increments=coarse_dw,
                    coefficients=model.sampling_coefficients, family_ids=families)
                costs.extend(x.diagnostics["total_seconds"] for x in (fine, coarse, neural))
                arrays = dict(particles=particle_y, initial_projection_error=init_error,
                              family_ids=np.asarray(families), costs=np.asarray(costs), identity_hash=identity_hash)
                for method, native in zip(_METHODS[1:], (fine, coarse, neural), strict=True):
                    arrays[method] = extract_observables(native, config.queries)
                    arrays[f"{method}_stopped_at"] = native.stopped_at
                temporary = chunk_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(chunk_path)
            with np.load(chunk_path, allow_pickle=False) as data:
                if str(data["identity_hash"]) != identity_hash or data["particles"].shape != (size, config.queries.dimension):
                    raise ValueError("Particle audit chunk identity/shape mismatch")
                chunks.append({key: data[key].copy() for key in data.files})
        completed_paths = sum(len(chunk["particles"]) for chunk in chunks)
        manifest.update(complete=completed_paths == count, completed_paths=completed_paths,
                        requested_paths=int(count), chunks=len(chunks))
        _atomic_json(manifest, manifest_path)
        result = dict(n_particles=int(n_particles), complete=manifest["complete"],
                      completed_paths=completed_paths, manifest_path=str(manifest_path),
                      particle_target=dict(law="unstopped interacting particles", observation="direct-continuous-physical-Fourier-modes-v1"),
                      native_target=asdict(spec), query_bank=asdict(config.queries))
        if completed_paths >= 2:
            result.update(_summarize(chunks, audit_context, config))
        _atomic_json(result, directory / "report.json")
        results.append(result)
        complete = complete and manifest["complete"]
    report = dict(stage="A5", complete=complete, counts=results,
                  status="descriptive comparison; independent numerical bias audits still required")
    _atomic_json(report, output_dir / "particle_audit.json")
    return json_safe(report)
