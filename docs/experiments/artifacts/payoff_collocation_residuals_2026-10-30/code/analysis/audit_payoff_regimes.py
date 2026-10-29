"""Independent pretest and analytic controls for the harder-regime study.

This helper never generates or reads held-out DK outcomes.  The pretest audit
reads historical identifier datasets only, replays learner streams, and checks
future RNG *addresses* without drawing them.  Its free-heat control uses a
sparse lifted matrix exponential, independently of the model's quadrature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

from analysis.compiled_native import compile_native  # noqa: E402
from analysis.native_gaussian_baseline import drift_jacobian_covariance  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import boundary_step, native_coefficients, simulate_native  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.observations import incidence_matrix  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402
from nn_dk.types import Context, seeded_rng, stable_hash  # noqa: E402

KIND = "independent-harder-regime-pretest-and-analytic-audit-v1"
IDENTIFIER_NAMES = {"family_ids", "integration_family_ids", "trajectory_ids"}


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def entropy_digest(seed, *keys):
    """Identify seeded_rng's SeedSequence entropy without instantiating it."""
    return hashlib.sha256(json.dumps([seed, *keys], separators=(",", ":")).encode()).hexdigest()


def read_identifiers(path):
    """Read only named identifier datasets, never a historical payoff/state."""
    identifiers, inventory = set(), {}
    with h5py.File(path, "r") as store:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in IDENTIFIER_NAMES:
                values = obj.asstr()[:].reshape(-1)
                unique = {str(value) for value in values if value}
                identifiers.update(unique)
                inventory[name] = {"rows": len(values), "unique_identifiers": len(unique),
                                   "identifiers_hash": stable_hash(sorted(unique))}
        store.visititems(visit)
    return identifiers, inventory


def historical_identifier_inventory(runs, excluded_root, current):
    runs, excluded_root, current = Path(runs).resolve(), Path(excluded_root).resolve(), set(current)
    all_ids, records = set(), []
    for path in sorted(runs.rglob("*.h5")):
        if path.resolve().is_relative_to(excluded_root):
            continue
        identifiers, datasets = read_identifiers(path)
        overlap = identifiers & current
        if overlap:
            raise ValueError(f"Historical trajectory-family overlap: {path}")
        all_ids.update(identifiers)
        records.append({"path": str(path.resolve()), "unique_identifiers": len(identifiers),
                        "identifiers_hash": stable_hash(sorted(identifiers)),
                        "datasets_read": datasets, "overlap_with_current": 0})
    return all_ids, records


def exact_free_heat_moments(initial, spec, context, bank, horizon):
    """Linear native first/second moments from an independent sparse lift.

    X=[native masses, four running integrals] has a constant linear drift A.
    Q is linear in the mean mass.  Propagating [E[m], vec(N Cov(X))] is one
    constant matrix exponential.  No quadrature, model jet, or fitted tensor
    enters this calculation.  These are unstopped analytic controls only.
    """
    initial = np.asarray(initial, dtype=float)
    g, d = spec.n_cells, spec.n_cells + 4
    if initial.shape != (g,) or np.any(initial < 0) or not np.isclose(initial.sum(), 1):
        raise ValueError("A nonnegative unit-mass native state is required")
    heat = context.diffusion / spec.h**2 * (
        np.roll(np.eye(g), 1, axis=0) + np.roll(np.eye(g), -1, axis=0) - 2 * np.eye(g))
    a = np.zeros((d, d))
    a[:g, :g], a[g:, :g] = heat, bank.running_weights
    q_map = np.zeros((d*d, g))
    incidence = incidence_matrix(g)
    for cell in range(g):
        basis = np.eye(g)[cell]
        intensity = context.diffusion / spec.h**2 * (basis + np.roll(basis, -1))
        qv = np.zeros((d, d))
        qv[:g, :g] = (incidence * intensity) @ incidence.T
        q_map[:, cell] = qv.reshape(-1)
    a, heat, q_map = sparse.csr_matrix(a), sparse.csr_matrix(heat), sparse.csr_matrix(q_map)
    lift = sparse.bmat([[heat, sparse.csr_matrix((g, d*d))],
                       [q_map, sparse.kron(a, sparse.eye(d)) + sparse.kron(sparse.eye(d), a)]],
                      format="csr")
    x0 = np.zeros(g + d*d)
    x0[:g] = initial
    result = expm_multiply(float(horizon) * lift, x0)
    covariance = result[g:].reshape(d, d)
    full_mean = expm_multiply(float(horizon) * a, np.r_[initial, np.zeros(4)])
    readout = np.concatenate((bank.terminal_weights, np.eye(4)), axis=1)
    derivative = (a @ covariance + covariance @ a.T + (q_map @ result[:g]).reshape(d, d))
    return {"mean": readout @ full_mean,
            "scaled_covariance": readout @ covariance @ readout.T,
            "scaled_covariance_time_derivative": readout @ derivative @ readout.T,
            "method": "independent-sparse-augmented-first-second-moment-matrix-exponential"}


def analytic_controls(horizons=(.25, 1.), n_cells=32, quadrature_nodes=16):
    spec = NativeDKSpec(n_cells)
    context = Context(n_particles=2048, diffusion=1., kappa=.75, a2=.1)
    centers = (np.arange(n_cells) + .5) * spec.h
    probabilities = np.exp(.2 * np.cos(centers))
    probabilities /= probabilities.sum()
    states = {"uniform": np.full(n_cells, 1/n_cells), "smooth-nonuniform": probabilities,
              "synthetic-IID-histogram": np.random.default_rng(2026102803).multinomial(
                  context.n_particles, probabilities) / context.n_particles}
    records = []
    for horizon in horizons:
        bank = PayoffBank.mixed_terminal_integral(spec, horizon)
        models = {q: RepairPayoffModel(spec, context, horizon, bank, width=4, quadrature_nodes=q)
                  for q in sorted({quadrature_nodes, 32})}
        for fraction in (.25, .5, 1.):
            t = float(horizon * fraction)
            coefficients = {q: model._time_coefficients(torch.tensor([t], dtype=torch.float64))
                            for q, model in models.items()}
            for state_name, state in states.items():
                exact = exact_free_heat_moments(state, spec, context, bank, t)
                scale = np.sqrt(np.diag(exact["scaled_covariance"]))
                product = np.outer(scale, scale)
                predictions = {}
                for q, c in coefficients.items():
                    mean = (c[0][0].numpy() @ state) * bank.scales / np.sqrt(context.n_particles)
                    covariance = np.einsum("ijg,g->ij", c[2][0].numpy(), state) * np.outer(
                        bank.scales, bank.scales)
                    derivative = np.einsum("ijg,g->ij", c[3][0].numpy(), state) * np.outer(
                        bank.scales, bank.scales)
                    predictions[str(q)] = {
                        "maximum_normalized_covariance_error": float(np.max(abs(
                            covariance-exact["scaled_covariance"]) / product)),
                        "maximum_scaled_mean_error": float(np.max(abs(mean-exact["mean"]))),
                        "maximum_normalized_time_derivative_error": float(t*np.max(abs(
                            derivative-exact["scaled_covariance_time_derivative"]) / product))}
                records.append({"horizon": float(horizon), "query_time": t, "state": state_name,
                                "quadrature": predictions})
    growth = []
    mode = np.cos(centers)
    for kappa in (.75, 3., 6.):
        physics = Context(n_particles=2048, diffusion=1., kappa=kappa, a2=.1)
        drift, jacobian, qv = drift_jacobian_covariance(states["uniform"], physics, spec)
        exact = (kappa/2-1.) * np.sinc(spec.h/(2*np.pi))**2
        observed = float(mode @ jacobian @ mode / (mode @ mode))
        native = native_coefficients(states["uniform"], physics, spec)
        if not np.isclose(observed, exact, atol=1e-12, rtol=0):
            raise AssertionError("Uniform instability control disagrees with native Jacobian")
        np.testing.assert_allclose(drift, native.drift, atol=1e-13, rtol=0)
        np.testing.assert_allclose(qv, native.covariance, atol=1e-13, rtol=0)
        growth.append({"kappa": kappa, "exact_mode1_growth_rate": exact,
                       "observed_mode1_growth_rate": observed,
                       "maximum_mode1_eigenvector_error": float(abs(jacobian @ mode-exact*mode).max()),
                       "centered_cell_Peclet": (kappa+.2)*spec.h,
                       "Euler_CFL_at_step_2^-10": 2**-10*(2/spec.h**2+(kappa+.2)/spec.h)})
    return {"complete": True, "kind": KIND+"-analytic-controls", "source_hash": source_hash(),
            "quadrature_nodes": quadrature_nodes, "records": records, "uniform_linearization": growth,
            "target": "unstopped-free-heat-control-not-interacting-stopped-truth",
            "no_held_out_DK_outcomes_generated_or_read": True}


def stopped_native_controls(backend=None, n_cells=32):
    """Synthetic controlled hit, already-absorbed path, and complete integral."""
    backend = compile_native() if backend is None else backend
    spec = NativeDKSpec(n_cells)
    context = Context(n_particles=2048, diffusion=1., kappa=6., a2=.1)
    dt, steps = 2**-10, 4
    initial = np.full((3, n_cells), 1/n_cells)
    initial[2, 0] = spec.mass_stop
    initial[2, 1] += 1/n_cells-spec.mass_stop
    dw = np.zeros((3, steps, n_cells))
    dw[1, 0, 0] = 3.
    times = np.arange(steps+1)*dt
    families = tuple(f"synthetic-control-{i}" for i in range(3))
    exact = simulate_native(context, spec, times, dt, 2026102804, 3, initial,
                            dw, family_ids=families)
    compiled = backend(initial, dw, context, spec, dt)
    np.testing.assert_allclose(compiled["states"], exact.states, atol=2e-14, rtol=0)
    np.testing.assert_allclose(compiled["stopped_at"], exact.stopped_at, atol=2e-14, rtol=0)
    if not (np.isinf(exact.stopped_at[0]) and 0 < exact.stopped_at[1] < dt
            and exact.stopped_at[2] == 0):
        raise AssertionError("The synthetic crossing/absorption control did not stop as designed")
    drift, amplitude = native_coefficients(initial[1], context, spec).drift, native_coefficients(
        initial[1], context, spec).face_amplitudes
    noise = amplitude * dw[1, 0] / np.sqrt(context.n_particles)
    proposal = initial[1] + dt*drift + np.roll(noise, 1)-noise
    stopped, crossed, stop = boundary_step(initial[1], proposal, 0., dt, spec)
    np.testing.assert_allclose(stopped, compiled["states"][1, 1], atol=2e-14, rtol=0)
    assert crossed and np.isclose(stop, exact.stopped_at[1], atol=2e-14, rtol=0)
    bank = PayoffBank.mixed_terminal_integral(spec, steps*dt)
    observed = bank.evaluate_paths(compiled["states"], times, compiled["stopped_at"])
    manual = compiled["states"] @ bank.terminal_weights.T
    accumulated = np.zeros((3, 4))
    for step in range(steps):
        exposure = np.minimum(dt, np.maximum(exact.stopped_at-times[step], 0))
        endpoints = (compiled["states"][:, step]+compiled["states"][:, step+1]) / 2
        accumulated += exposure[:, None] * (endpoints @ bank.running_weights.T)
        manual[:, step+1] += accumulated
    np.testing.assert_allclose(observed, manual, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(observed[2], np.broadcast_to(observed[2, 0], observed[2].shape))
    np.testing.assert_array_equal(observed[1, 1:], np.broadcast_to(observed[1, 1], observed[1, 1:].shape))
    stop_error = float(abs(compiled["stopped_at"][1:]-exact.stopped_at[1:]).max())
    return {"complete": True, "paths": 3, "all_paths_retained": True,
            "never_stopped": 1, "first_segment_crossing": 1, "already_absorbed": 1,
            "maximum_cpp_vs_native_state_error": float(abs(compiled["states"]-exact.states).max()),
            "maximum_cpp_vs_native_stopping_time_error": stop_error,
            "maximum_killed_integral_manual_error": float(abs(observed-manual).max()),
            "mass_error": float(abs(compiled["states"].sum(-1)-1).max()),
            "backend_fingerprint": backend.metadata["fingerprint"]}


def _rng_address(runner, cfg, role, profile, anchor, index, purpose):
    return entropy_digest(cfg.seed, runner.STREAM, cfg.config_hash, role, profile, anchor, index, purpose)


def pretest_audit(root):
    """Verify locked pretest design without materializing final initials/noise."""
    from analysis import evaluate_payoff_regimes as evaluator
    from analysis import run_payoff_regimes as runner

    root = Path(root).resolve()
    plan, design = runner._assert_frozen(root)
    comparison = evaluator.prepare(root)
    cases = runner.case_names(plan)
    reference_snapshots = {}
    for case in cases:
        directory = root/"reference"/case
        if (directory/"reference.h5").exists():
            raise ValueError("Pretest audit must precede every held-out reference draw")
        manifest = runner._read(directory/"manifest.json")
        if manifest["status"] != "waiting-for-complete-fit-lock" or manifest["committed_chunks"] != 0:
            raise ValueError("Fresh reference must still be waiting for the complete fit lock")
        reference_snapshots[case] = {"reference_h5_present": False,
                                     "manifest_sha256": _sha(directory/"manifest.json"),
                                     "status": manifest["status"], "committed_chunks": 0}
    if (root/"payoffs/fit_lock.json").exists():
        raise ValueError("Run pretest independence checks before the final model lock")
    try:
        runner.verify_fit_lock(root)
    except (FileNotFoundError, ValueError):
        unlock_rejected = True
    else:
        raise AssertionError("Missing all-model fit lock did not block reference unlocking")
    backend = compile_native(root/"compiled")
    current_ids, current_entropy, horizon_max, states_max = set(), set(), 0., 0.
    split_records, replay_records = [], []
    for case in cases:
        cfg = runner.configuration(case, plan)
        for role in ("train", "validation"):
            bank = runner.load_state_bank(root, case, role)
            families = set(bank["family_ids"])
            if current_ids & families:
                raise ValueError("Learner families overlap across fixed regimes or splits")
            current_ids.update(families)
            count = cfg.train_paths if role == "train" else cfg.validation_paths
            repeats = len(cfg.state_times)*cfg.horizon_samples
            if len(families) != 2*count or len(bank["states"]) != 2*count*repeats:
                raise ValueError("Learner independent/derived sample allocation changed")
            with h5py.File(root/"payoffs"/case/f"states_{role}.h5", "r") as store:
                for profile in cfg.cases:
                    group = store[profile]
                    for index in range(count):
                        family = stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                                              role, profile, 0, index])
                        if group["family_ids"][index].decode() != family:
                            raise AssertionError("Learner family addressing changed")
                        expected = seeded_rng(cfg.seed, runner.STREAM, cfg.config_hash,
                                              role, profile, 0, index, "residual-horizons").uniform(
                            0, cfg.horizon, (len(cfg.state_times), cfg.horizon_samples))
                        error = float(abs(expected-group["horizons"][index]).max())
                        if error != 0:
                            raise AssertionError("Learner residual-horizon stream failed replay")
                        horizon_max = max(horizon_max, error)
                        for purpose in ("initial", "physical-forcing", "residual-horizons"):
                            address = _rng_address(runner, cfg, role, profile, 0, index, purpose)
                            if address in current_entropy:
                                raise ValueError("Learner physical streams share an entropy address")
                            current_entropy.add(address)
                    index = 0
                    family = stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                                          role, profile, 0, index])
                    rng = seeded_rng(cfg.seed, runner.STREAM, cfg.config_hash, role, profile, 0, index, "initial")
                    probabilities = runner.profile_probabilities(cfg, profile)[0]
                    initial = rng.multinomial(cfg.n_particles, probabilities)/cfg.n_particles
                    dw = seeded_rng(cfg.seed, runner.STREAM, cfg.config_hash, role, profile, 0, index,
                                    "physical-forcing").normal(0, np.sqrt(cfg.state_step),
                                                               (round(cfg.horizon/cfg.state_step), cfg.n_cells))
                    if (_sha_array(initial) != group["initial_sha256"][index].decode()
                            or _sha_array(dw) != group["brownian_sha256"][index].decode()):
                        raise AssertionError("Learner initial/physical forcing digest failed replay")
                    packet = simulate_native(cfg.context(profile), cfg.spec, cfg.state_times,
                        cfg.state_step, 0, 1, initial[None], dw[None], family_ids=(family,))
                    error = float(abs(packet.states[0]-group["states"][index]).max())
                    if error > 2e-13:
                        raise AssertionError("Compiled learner path differs from independent native replay")
                    states_max = max(states_max, error)
                    stored_stop = float(group["stopped_at"][index])
                    if not (np.isinf(stored_stop) and np.isinf(packet.stopped_at[0])):
                        np.testing.assert_allclose(packet.stopped_at[0], stored_stop, atol=2e-13, rtol=0)
                    replay_records.append({"case": case, "role": role, "profile": profile,
                                           "index": index, "maximum_state_error": error,
                                           "initial_and_forcing_hashes_reproduced": True})
            split_records.append({"case": case, "role": role, "families": len(families),
                                  "collocation_rows": len(bank["states"]),
                                  "derived_rows_per_family": repeats, "h5_sha256": bank["sha256"],
                                  "content_hash": bank["content_hash"]})
    historical, history = historical_identifier_inventory(REPO/"runs", root, current_ids)
    future_ids, future_entropy = set(), set()
    for case in cases:
        cfg = runner.reference_configuration(case, plan)
        for key in cfg.group_keys:
            profile, role, anchor = key.split("__")
            anchor = int(anchor)
            if role == "conditional":
                address = _rng_address(runner, cfg, "held-out-anchor", profile, anchor, 0, "initial")
                future_entropy.add(address)
            for index in range(cfg.paths_per_group):
                family = stable_hash([runner.STREAM, cfg.seed, cfg.config_hash,
                                      "reference-"+role, profile, anchor, index])
                if family in future_ids or family in current_ids or family in historical:
                    raise ValueError("Declared future reference family is not disjoint")
                future_ids.add(family)
                for purpose in (("initial", "physical-forcing") if role == "ensemble" else ("physical-forcing",)):
                    address = _rng_address(runner, cfg, "reference-"+role, profile, anchor, index, purpose)
                    if address in future_entropy or address in current_entropy:
                        raise ValueError("Declared future reference entropy address is not disjoint")
                    future_entropy.add(address)
    candidate_entropy, candidate_ids = [], set()
    for case in cases:
        cfg = runner.reference_configuration(case, plan)
        for key in cfg.group_keys:
            seed = int(stable_hash([evaluator.SEED, case, key, "independent-candidate"])[:12], 16)
            address = entropy_digest(seed, evaluator.KIND, "candidate-native-MC")
            if address in future_entropy or address in current_entropy or address in candidate_entropy:
                raise ValueError("Native candidate MC stream address is not independent")
            candidate_entropy.append(address)
            for index in range(cfg.paths_per_group):
                family = stable_hash([evaluator.SEED, case, key, "independent-candidate", index])
                if (family in candidate_ids or family in future_ids
                        or family in current_ids or family in historical):
                    raise ValueError("Native candidate MC family is not independent")
                candidate_ids.add(family)
    if any((root/"reference"/case/"reference.h5").exists() for case in cases):
        raise ValueError("A reference appeared during the pretest audit")
    result = {"complete": True, "kind": KIND, "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": _sha(__file__), "source_hash": source_hash(),
              "git_commit_at_audit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "preregistration_sha256": _sha(root/"payoffs/preregistration.json"),
              "comparison_preregistration_sha256": _sha(root/"comparison/preregistration.json"),
              "plan_hash": plan.plan_hash, "expected_models": len(runner.slots(plan)),
              "declared_slots": design["slots"], "reference_snapshots": reference_snapshots,
              "reference_unlock_rejected": unlock_rejected,
              "learner_splits": split_records, "unique_learner_families": len(current_ids),
              "unique_learner_entropy_addresses": len(current_entropy),
              "all_learner_horizon_streams_replayed": True,
              "maximum_horizon_replay_error": horizon_max, "state_replays": replay_records,
              "maximum_native_vs_compiled_learner_state_error": states_max,
              "historical_inventory": history, "unique_historical_identifiers": len(historical),
              "historical_current_overlap": 0, "future_reference_families": len(future_ids),
              "future_reference_entropy_addresses": len(future_entropy),
              "future_native_candidate_families": len(candidate_ids),
              "candidate_MC_entropy_addresses": candidate_entropy,
              "future_reference_current_historical_family_overlap": 0,
              "candidate_reference_learner_historical_family_overlap": 0,
              "learner_reference_candidate_entropy_overlap": 0,
              "native_stopping_controls": stopped_native_controls(backend, plan.n_cells),
              "analytic_controls": analytic_controls(plan.horizons, plan.n_cells, plan.quadrature_nodes),
              "comparison_settings": {key: comparison[key] for key in (
                  "primary_LGA_max_step", "native_MC_step", "bootstrap", "all_optimizer_seeds")},
              "held_out_outcomes_or_initials_opened_drawn_or_generated": False,
              "limitations": ["Addressed pseudorandom streams are not a mathematical independence proof.",
                              "Free-heat checks do not certify the interacting stopped law.",
                              "Continuous boundary-hitting bias remains separate from Euler refinements."]}
    output = root/"audit/pretest_audit.json"
    if output.exists():
        raise ValueError("Do not overwrite an already recorded pretest snapshot")
    _json(output, result)
    return result


def _sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analytic", "pretest"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "pretest":
        if args.root is None or args.output is not None:
            parser.error("Pretest requires --root and writes its fixed audit path")
        result = pretest_audit(args.root)
        print(json.dumps({"complete": result["complete"], "source_hash": result["source_hash"],
                          "learner_families": result["unique_learner_families"],
                          "future_reference_families": result["future_reference_families"]}))
    else:
        if args.root is not None or args.output is None:
            parser.error("Analytic controls require --output only")
        result = analytic_controls()
        _json(args.output, result)
        print(json.dumps({"complete": True, "output": str(args.output.resolve()), "sha256": _sha(args.output)}))


if __name__ == "__main__":
    main()
