"""Independent analytic controls for the moving-boundary payoff ansatz.

Only deterministic controlled states and randomly initialized network heads
are used. No trajectory pool, checkpoint, fit, or future outcome is opened.
Dense state Hessians, native NumPy coefficients, and finite time differences
independently check the implementation's generator jets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import native_coefficients  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.observations import incidence_matrix  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context  # noqa: E402

KIND = "controlled-moving-boundary-mean-covariance-generator-identities-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_for_control(*, cells=32, horizon=1., kappa=6., same_modes=False):
    from nn_dk.payoff_boundary import BoundaryPayoffModel

    spec = NativeDKSpec(cells)
    physics = Context(n_particles=2048, diffusion=1., kappa=kappa, a2=.1)
    if same_modes:
        weights = PayoffBank._weights(spec, ((1, "sin"), (1, "cos"),
                                           (1, "sin"), (1, "cos")))
        phi, running = weights.copy(), weights.copy()/horizon
        phi[2:], running[:2] = 0., 0.
        bank = PayoffBank("moving-boundary-shared-mode-control", phi, running,
                          horizon, spec.length)
    else:
        bank = PayoffBank.mixed_terminal_integral(spec, horizon)
    with torch.random.fork_rng():
        torch.manual_seed(2026102937)
        model = BoundaryPayoffModel(spec, physics, horizon, bank, width=4,
                                     quadrature_nodes=16)
    return model


def controlled_state(model, distance=None):
    result = np.full(model.spec.n_cells, 1/model.spec.n_cells)
    if distance is not None:
        result[0] = model.spec.mass_stop+float(distance)
        result[model.spec.n_cells//2] += 1/model.spec.n_cells-result[0]
    np.testing.assert_allclose(result.sum(), 1., rtol=0, atol=2e-15)
    return torch.tensor(result, dtype=torch.float64)


def arbitrary_heads(model):
    """Nonzero trainable heads make the identities independent of initialization."""
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(2026102939)
        for network in (model.mean_network, model.covariance_network):
            network[-1].weight.normal_(0., .08)
            network[-1].bias.normal_(0., .3)


def _error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return float(np.max(abs(actual-expected), initial=0.))


def initial_corner_control():
    """Every strict-interior zero-time slope is known under native coefficients."""
    model = model_for_control()
    phi = model.bank.terminal_weights/model.bank.scales[:, None]
    running = model.bank.running_weights/model.bank.scales[:, None]
    records = []
    for head_setting in ("zero_correction", "arbitrary_nonzero_heads"):
        if head_setting == "arbitrary_nonzero_heads":
            arbitrary_heads(model)
        for distance in (None, model.boundary_width*.4, model.boundary_width*1e-8):
            state = controlled_state(model, distance)
            initial = state[None]
            zero = torch.zeros(1, dtype=torch.float64)
            mean, covariance = model._moment_jets(initial, zero)
            values = model.moment_values(initial, zero)
            native = native_coefficients(state.numpy(), model.context, model.spec)
            expected_start = math.sqrt(model.context.n_particles)*(phi@state.numpy())
            expected_generator = math.sqrt(model.context.n_particles)*(phi@native.drift)
            expected_slope = expected_generator+math.sqrt(model.context.n_particles)*(
                running@state.numpy())
            faces = incidence_matrix(model.spec.n_cells)*native.face_amplitudes
            expected_face = phi@faces
            expected_covariance_slope = phi@native.covariance@phi.T
            np.testing.assert_allclose(mean.value[0].detach(), expected_start, atol=2e-12, rtol=2e-13)
            np.testing.assert_allclose(mean.time_derivative[0].detach(), expected_slope,
                                       atol=3e-11, rtol=2e-12)
            np.testing.assert_allclose(mean.generator[0].detach(), expected_generator,
                                       atol=3e-11, rtol=2e-12)
            np.testing.assert_allclose(mean.face_derivatives[0].detach(), expected_face,
                                       atol=3e-12, rtol=2e-12)
            np.testing.assert_allclose(covariance.time_derivative[0].detach(), expected_covariance_slope,
                                       atol=3e-11, rtol=2e-12)
            assert torch.count_nonzero(covariance.value) == 0
            assert torch.count_nonzero(covariance.generator) == 0
            assert torch.count_nonzero(covariance.face_derivatives) == 0
            for actual, jet in zip(values, (mean, covariance), strict=True):
                torch.testing.assert_close(actual, jet.value, atol=2e-12, rtol=2e-13)
            residuals = model.residual(initial, zero)
            for residual in residuals:
                np.testing.assert_allclose(residual.detach(), 0., atol=4e-11, rtol=0)
            records.append({"heads": head_setting, "minimum_mass": float(state.min()),
                            "distance_to_boundary": float(state.min()-model.spec.mass_stop),
                            "mean_value_maximum_error": _error(mean.value[0].detach(), expected_start),
                            "native_mean_slope_maximum_error": _error(mean.time_derivative[0].detach(),
                                                                      expected_slope),
                            "native_covariance_slope_maximum_error": _error(
                                covariance.time_derivative[0].detach(), expected_covariance_slope),
                            "mean_residual_maximum": float(residuals[0].detach().abs().max()),
                            "covariance_residual_maximum": float(residuals[1].detach().abs().max())})
    return {"records": records, "arbitrary_heads": True,
            "native_drift_and_covariance": "independent-numpy-finite-volume-coefficients"}


def small_time_covariance_control():
    """Shared modes ensure terminal/integral cross terms are genuinely nonzero."""
    model = model_for_control(same_modes=True)
    arbitrary_heads(model)
    state = controlled_state(model, model.boundary_width*.4)
    native = native_coefficients(state.numpy(), model.context, model.spec)
    phi = model.bank.terminal_weights/model.bank.scales[:, None]
    q = model.bank.running_weights/model.bank.scales[:, None]
    expected = (phi@native.covariance@phi.T
                + .5*(phi@native.covariance@q.T+q@native.covariance@phi.T)
                + q@native.covariance@q.T/3)
    powers = (1+model.bank.integral_rows[:, None].astype(int)
              + model.bank.integral_rows[None, :].astype(int))
    assert abs(expected[0, 2]) > .1
    records = []
    for time_value in (1e-7, 1e-8, 1e-9):
        _, covariance = model.moment_values(state[None], torch.tensor([time_value], dtype=torch.float64))
        coefficient = covariance[0].detach().numpy()/time_value**powers
        relative = np.linalg.norm(coefficient-expected)/np.linalg.norm(expected)
        records.append({"horizon": time_value, "leading_coefficient_relative_error": float(relative),
                        "terminal_integral_cross_expected": float(expected[0, 2]),
                        "terminal_integral_cross_observed": float(coefficient[0, 2])})
    assert records[-1]["leading_coefficient_relative_error"] < 2e-6
    assert records[-1]["leading_coefficient_relative_error"] < records[0]["leading_coefficient_relative_error"]
    return {"records": records, "orders": "TT=tau; TI=tau^2; II=tau^3",
            "shared_mode_cross_is_nonzero": True}


def dense_derivative_control():
    model = model_for_control(cells=8, horizon=.25, kappa=.75)
    arbitrary_heads(model)
    time_value = .03
    distance = model.boundary_width*math.sqrt(time_value/model.horizon)*math.sqrt(.7)
    point = controlled_state(model, distance).requires_grad_()
    times = torch.tensor([time_value], dtype=torch.float64)
    jets = model._moment_jets(point[None], times)
    native = native_coefficients(point.detach().numpy(), model.context, model.spec)
    faces = torch.tensor(incidence_matrix(model.spec.n_cells)*native.face_amplitudes
                         /math.sqrt(model.context.n_particles))
    qv = faces@faces.T
    drift = torch.tensor(native.drift)
    records = []
    for component, jet in enumerate(jets):
        def value(masses, horizon=time_value):
            return model.moment_values(masses[None], torch.tensor([horizon], dtype=torch.float64))[
                component][0]

        jacobian = torch.autograd.functional.jacobian(value, point)
        hessian = torch.autograd.functional.hessian(lambda masses: value(masses).sum(), point)
        dense_generator_sum = (jacobian@drift).sum()+.5*(hessian*qv).sum()
        dense_faces = jacobian@faces
        epsilon = 1e-6
        numerical_time = (-value(point, time_value+2*epsilon)+8*value(point, time_value+epsilon)
                          -8*value(point, time_value-epsilon)+value(point, time_value-2*epsilon))/(12*epsilon)
        torch.testing.assert_close(jet.generator.sum(), dense_generator_sum, atol=2e-8, rtol=4e-10)
        torch.testing.assert_close(jet.face_derivatives[0], dense_faces, atol=2e-10, rtol=3e-10)
        torch.testing.assert_close(jet.time_derivative[0], numerical_time, atol=3e-8, rtol=3e-7)
        torch.testing.assert_close(jet.value[0], value(point), atol=2e-12, rtol=2e-13)
        records.append({"component": "mean" if component == 0 else "covariance",
                        "generator_sum_absolute_error": float(abs(jet.generator.sum()-dense_generator_sum).detach()),
                        "face_derivatives_maximum_error": float((jet.face_derivatives[0]-dense_faces).detach().abs().max()),
                        "time_derivatives_maximum_error": float((jet.time_derivative[0]-numerical_time).detach().abs().max())})
    return {"records": records, "minimum_mass": float(point.detach().min()),
            "remaining_horizon": time_value,
            "oracle": "dense-state-autodiff-Hessian-and-fourth-order-time-difference"}


def native_interaction_anchor_control():
    """Native NumPy stencils independently validate the quadratic jet anchor."""
    model = model_for_control()
    arbitrary_heads(model)
    centers = (np.arange(model.spec.n_cells)+.5)*model.spec.h
    initial = np.exp(.5*np.cos(centers)+.1*np.sin(2*centers))
    initial /= initial.sum()
    state = torch.tensor(initial[None])
    native = native_coefficients(initial, model.context, model.spec)
    heat = model.context.diffusion/model.spec.h**2*(
        np.roll(np.eye(model.spec.n_cells), 1, axis=0)
        + np.roll(np.eye(model.spec.n_cells), -1, axis=0)-2*np.eye(model.spec.n_cells))

    def interaction(point):
        return native_coefficients(point, model.context, model.spec).drift-heat@point

    origins = model._origins(state, torch.tensor([.03], dtype=torch.float64))
    jet = model._interaction_jet(origins[1])
    expected_value = interaction(initial)
    epsilon = 1e-3
    faces = incidence_matrix(model.spec.n_cells)*native.face_amplitudes/math.sqrt(model.context.n_particles)
    expected_faces = np.stack([(interaction(initial+epsilon*direction)
                                -interaction(initial-epsilon*direction))/(2*epsilon)
                               for direction in faces.T], axis=-1)
    expected_generator = (interaction(initial+epsilon*native.drift)
                          -interaction(initial-epsilon*native.drift))/(2*epsilon)
    expected_generator += .5*sum((interaction(initial+epsilon*direction)-2*expected_value
                                 +interaction(initial-epsilon*direction))/epsilon**2
                                for direction in faces.T)
    np.testing.assert_allclose(jet.value[0].detach(), expected_value, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(model._interaction_value(state)[0].detach(), expected_value,
                               atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(jet.face_derivatives[0].detach(), expected_faces, atol=3e-11, rtol=3e-9)
    np.testing.assert_allclose(jet.generator[0].detach(), expected_generator, atol=8e-8, rtol=3e-6)
    phi = model.bank.terminal_weights/model.bank.scales[:, None]
    running = model.bank.running_weights/model.bank.scales[:, None]
    start = math.sqrt(model.context.n_particles)*(phi@initial)
    slope = math.sqrt(model.context.n_particles)*(phi@native.drift+running@initial)
    small_time = []
    for time_value in (1e-5, 1e-6, 1e-7):
        mean = model.moment_values(state, torch.tensor([time_value], dtype=torch.float64))[0][0]
        observed = (mean.detach().numpy()-start)/time_value
        relative = np.linalg.norm(observed-slope)/np.linalg.norm(slope)
        small_time.append({"horizon": time_value, "mean_slope_relative_error": float(relative)})
    assert small_time[-1]["mean_slope_relative_error"] < 5e-6
    assert small_time[-1]["mean_slope_relative_error"] < small_time[0]["mean_slope_relative_error"]
    return {"quadratic_value_maximum_error": _error(jet.value[0].detach(), expected_value),
            "quadratic_generator_maximum_error": _error(jet.generator[0].detach(), expected_generator),
            "quadratic_face_derivatives_maximum_error": _error(jet.face_derivatives[0].detach(), expected_faces),
            "small_time": small_time,
            "oracle": "mass-conserving native NumPy coefficient directional stencils"}


def absorption_and_finiteness_control():
    model = model_for_control()
    arbitrary_heads(model)
    records = []
    for distance, time_value in ((0., 0.), (0., .4), (-model.spec.mass_stop*.2, .4),
                                 (model.boundary_width*1e-8, 0.),
                                 (model.boundary_width*1e-8, 1e-30),
                                 (model.boundary_width*1e-8, 1e-12),
                                 (model.boundary_width*.4, 1e-12),
                                 (model.boundary_width*.4, 1.)):
        state = controlled_state(model, distance)[None]
        times = torch.tensor([time_value], dtype=torch.float64)
        model.zero_grad(set_to_none=True)
        jets = model._moment_jets(state, times)
        values = model.moment_values(state, times)
        for jet, value in zip(jets, values, strict=True):
            torch.testing.assert_close(jet.value, value, atol=2e-10, rtol=2e-12)
            for key in ("value", "time_derivative", "generator", "face_derivatives"):
                assert bool(torch.isfinite(getattr(jet, key)).all())
        residuals = model.residual(state, times)
        for residual in residuals:
            assert bool(torch.isfinite(residual).all())
        loss = sum(torch.logaddexp(torch.zeros_like(value), value).sum() for value in residuals)
        if loss.requires_grad:
            loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert bool(torch.isfinite(parameter.grad).all())
        eigenvalues = torch.linalg.eigvalsh(values[1].detach())[0]
        assert float(eigenvalues.min()) > -2e-10
        if distance <= 0:
            torch.testing.assert_close(values[0], model.scaled_observables(state), atol=2e-12, rtol=2e-13)
            assert torch.count_nonzero(values[1]) == 0
            assert all(torch.count_nonzero(value) == 0 for value in residuals)
        records.append({"distance_to_boundary": float(state.min()-model.spec.mass_stop),
                        "horizon": time_value, "absorbed": distance <= 0,
                        "minimum_covariance_eigenvalue": float(eigenvalues.min()),
                        "mean_residual_maximum": float(residuals[0].detach().abs().max()),
                        "covariance_residual_maximum": float(residuals[1].detach().abs().max()),
                        "all_jets_residuals_parameter_gradients_finite": True})
    return {"records": records, "no_state_clipping_or_renormalization": True}


def gate_boundary_order_control():
    """The survival gate has linear boundary order and flat interior time jets."""
    model = model_for_control(cells=8, horizon=.25, kappa=.75)
    records = []
    for time_value in (1e-12, .03, .25):
        for argument in (1e-4, .2, 1., 4.):
            distance = argument*model.boundary_width*math.sqrt(time_value/model.horizon)
            state = controlled_state(model, distance)[None]
            value, derivative, first, second = model._gate_terms(
                state, torch.tensor([time_value], dtype=torch.float64))
            # Direct calculus in ordinary coordinates is independent of the
            # implementation's stabilized logarithmic evaluation.
            actual_distance = float(state[0, 0])-model.spec.mass_stop
            c = model.horizon/model.boundary_width**2
            z = actual_distance*math.sqrt(c/time_value)
            density = math.exp(-z*z)/math.sqrt(math.pi)
            expected = (math.erf(z), -z*density/time_value,
                        2*math.sqrt(c/time_value)*density,
                        -4*actual_distance*(c/time_value)**1.5*density)
            actual = tuple(float(component[0, 0]) for component in (value, derivative, first, second))
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=2e-12)
            records.append({"horizon": time_value, "erf_argument": z,
                            "maximum_scaled_gate_partial_error": float(np.max(
                                abs(np.asarray(actual)-expected)/(1+abs(np.asarray(expected)))))})
    slopes = []
    for distance_fraction in (1e-3, 1e-4, 1e-5):
        state = controlled_state(model, model.boundary_width*distance_fraction)
        actual_distance = float(state.min())-model.spec.mass_stop
        time_value = .03
        covariance = model.moment_values(state[None], torch.tensor([time_value], dtype=torch.float64))[1]
        slope = float(covariance[0].detach().trace())/actual_distance
        slopes.append({"distance": actual_distance, "covariance_trace_over_distance": slope})
    assert slopes[-1]["covariance_trace_over_distance"] > 1.
    assert abs(slopes[-1]["covariance_trace_over_distance"]/
               slopes[-2]["covariance_trace_over_distance"]-1) < 1e-3
    zero_terms = model._gate_terms(controlled_state(model, model.boundary_width*.4)[None],
                                  torch.zeros(1, dtype=torch.float64))
    assert torch.all(zero_terms[0] == 1)
    assert all(torch.count_nonzero(partial) == 0 for partial in zero_terms[1:])
    return {"records": records, "linear_boundary_covariance_control": slopes,
            "zero_time_cell_factors_one_all_partials_zero": True,
            "boundary_order": "linear in distance at fixed positive time"}


def loss_scaling_control():
    """Literal equal residual rows have equal influence at all physical times."""
    from nn_dk.payoff_boundary import _normalized_residual, _objective

    model = model_for_control(cells=8, horizon=.25, kappa=.75)
    scales = np.array([.5, 1., 2., 4.])
    scaling = {"mean_scale": scales.tolist(), "covariance_scale": np.outer(scales, scales).tolist(),
               "small_time_powers": [0, 0, 1, 1], "scale_time_floor": 1., "pseudo_huber_delta": 3.}
    times = torch.tensor([0., 1e-30, .03, .25], dtype=torch.float64)
    states = controlled_state(model)[None].expand(4, -1)
    records = []
    for stage in ("mean", "covariance"):
        row = torch.tensor([[-1e4, -.1, .1, 1e4]], dtype=torch.float64)
        if stage == "covariance":
            row = row[..., :, None]*torch.tensor([[[1., -.3, .4, 2.]]], dtype=torch.float64)
        residual = row.expand(4, *row.shape[1:]).clone().requires_grad_()
        expected_z = model.horizon*residual/torch.tensor(scaling[f"{stage}_scale"])
        normalized = _normalized_residual(model, residual, scaling, stage)
        torch.testing.assert_close(normalized, expected_z, atol=0, rtol=0)
        with patch.object(model, f"{stage}_residual", return_value=residual):
            loss, observed = _objective(model, states, times, scaling, stage)
        delta = scaling["pseudo_huber_delta"]
        expected_loss = (2*delta**2*(torch.hypot(torch.ones_like(expected_z), expected_z/delta)-1)).mean()
        torch.testing.assert_close(loss, expected_loss, atol=1e-9, rtol=2e-14)
        torch.testing.assert_close(observed, residual, atol=0, rtol=0)
        gradient = torch.autograd.grad(loss, residual)[0]
        expected_gradient = (2*expected_z/torch.hypot(torch.ones_like(expected_z), expected_z/delta)
                             *model.horizon/torch.tensor(scaling[f"{stage}_scale"])/residual.numel())
        torch.testing.assert_close(gradient, expected_gradient, atol=2e-14, rtol=2e-13)
        assert bool(torch.isfinite(gradient).all())
        for index in range(1, 4):
            torch.testing.assert_close(gradient[index], gradient[0], atol=0, rtol=0)
        influence = 2*expected_z/torch.hypot(torch.ones_like(expected_z), expected_z/delta)
        assert float(influence.detach().abs().max()) <= 2*delta
        records.append({"stage": stage, "literal_loss": float(loss.detach()),
                        "independent_formula_absolute_error": float(abs(loss-expected_loss).detach()),
                        "maximum_normalized_influence": float(influence.detach().abs().max()),
                        "influence_bound": 2*delta, "identical_rows_identical_influence_at_all_times": True})
    return {"records": records, "inverse_time_weighting": False,
            "scales": "fixed synthetic physical units; no fitted or held-out outcomes",
            "pseudo_huber": "2 delta^2 (sqrt(1+(z/delta)^2)-1)"}


def run_audit():
    torch.set_num_threads(1)
    controls = {"initial_corner": initial_corner_control(),
                "small_time_covariance": small_time_covariance_control(),
                "dense_derivatives": dense_derivative_control(),
                "native_interaction_anchor": native_interaction_anchor_control(),
                "absorption_finiteness_psd": absorption_and_finiteness_control(),
                "gate_boundary_order": gate_boundary_order_control(),
                "loss_scaling": loss_scaling_control()}
    return {"complete": True, "kind": KIND, "source_hash": source_hash(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": _sha(__file__),
            "controls": controls, "no_fitting_checkpoint_trajectory_or_final_test_data_opened": True,
            "scope": "Software and analytic consistency only; no finite-horizon accuracy certification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("A formulation audit never overwrites prior evidence")
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"complete": result["complete"], "output": str(args.output.resolve()),
                      "sha256": _sha(args.output)}))


if __name__ == "__main__":
    main()
