"""Certified references for the noninteracting, stopped native DK generator.

The interior generator preserves polynomials of degree at most two.  We solve
that finite-dimensional moment problem independently by Poisson uniformization.
It is NOT silently identified with the stopped law: a positive affine-exponential
martingale bounds the first boundary hitting probability, and compact-state
bounds propagate it to every requested mean/covariance entry.

For alpha=D/h² and u(0)=lambda*e_i, the backward exponential is
exp(-N*u(H-t).m), where
u'_i=alpha*(u_{i-1}-2u_i+u_{i+1})
     -alpha/2*((u_i-u_{i-1})²+(u_i-u_{i+1})²).
For 0<lambda<=1 the cube [0,1]^G is invariant, its Jacobian is Metzler with
zero row sums, and its flow is nonexpansive in the maximum norm.  A fourth-order
Taylor enclosure therefore accumulates local remainder/roundoff without an
exponential Gronwall factor.  The stopping certificate uses optional stopping
on each time window; see the returned proof metadata for the explicit bound.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np

_UNIT = np.finfo(float).eps/2
_DEFAULTS = {"ode_step": 2**-16, "window_step": 2**-16,
             "tilt": 1.0, "certificate_tolerance": 1e-8}


def _gamma(count):
    return count*_UNIT/(1-count*_UNIT)


def _up(value):
    return np.nextafter(value, np.inf)


def _inflate_bound(value, operations):
    """Cover roundoff while constructing a nonnegative arithmetic error bound."""
    return _up(np.asarray(value)*(1+_gamma(operations)))


def _base(config):
    return getattr(config, "base_config", config)


def _settings(settings):
    result = {**_DEFAULTS, **(settings or {})}
    if set(result) != set(_DEFAULTS) or any(not np.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("Known positive finite generator-reference settings required")
    if result["tilt"] > 1:
        raise ValueError("The cooperative Riccati certificate requires 0 < tilt <= 1")
    ratio = result["window_step"]/min(result["ode_step"], result["window_step"])
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10):
        raise ValueError("Riccati integration steps must nest in the stopping-window grid")
    return result


def _physics(base, context=None):
    if context is not None and (context.kappa != 0 or context.a2 != 0
                               or context.diffusion != base.diffusion
                               or context.n_particles != base.n_particles):
        raise ValueError("Certified generator references require matching noninteracting physics")
    if base.diffusion <= 0 or base.n_particles < 1:
        raise ValueError("Positive diffusion and particle count required")


@lru_cache(maxsize=1)
def _pi():
    # Machin's identity and an alternating-series tail enclose pi.  Sixty
    # arctangent terms and 80 decimal digits leave error below 1e-74.
    with localcontext() as ctx:
        ctx.prec = 80
        def atan_inverse(q):
            x = Decimal(1)/Decimal(q)
            return sum(((-1 if k % 2 else 1)*x**(2*k+1)/Decimal(2*k+1)
                        for k in range(60)), Decimal(0))
        return +(16*atan_inverse(5)-4*atan_inverse(239))


def _geometry_parameters(base):
    alpha = base.diffusion/base.spec.h**2
    mass_stop = base.spec.mass_stop
    with localcontext() as ctx:
        ctx.prec = 80
        h = 2*_pi()/base.n_cells
        exact_alpha = Decimal.from_float(float(base.diffusion))/(h*h)
        exact_stop = Decimal.from_float(float(base.spec.rho_stop))*h
        alpha_error = _up(float(abs(exact_alpha-Decimal.from_float(alpha))+Decimal("1e-65")))
        stop_error = _up(float(abs(exact_stop-Decimal.from_float(mass_stop))+Decimal("1e-65")))
    return alpha, float(alpha_error), mass_stop, float(stop_error)


def _trig(q):
    """Decimal sine/cosine of q*pi, with absolute error below 1e-68."""
    q = q-2*((q+1)//2)
    if q > 1:
        q -= 2
    if q < -1:
        q += 2
    x = q*_pi()
    sine, cosine = x, Decimal(1)
    st, ct = x, Decimal(1)
    for k in range(1, 42):
        st *= -x*x/Decimal((2*k)*(2*k+1))
        ct *= -x*x/Decimal((2*k-1)*(2*k))
        sine += st
        cosine += ct
    return sine, cosine


@lru_cache(maxsize=16)
def _weights(n_cells):
    weights = np.empty((4, n_cells))
    errors = np.empty_like(weights)
    with localcontext() as ctx:
        ctx.prec = 80
        root_two = Decimal(2).sqrt()
        for mode in (1, 2):
            sinc = _trig(Decimal(mode)/n_cells)[0]/(_pi()*mode/n_cells)
            for i in range(n_cells):
                sine, cosine = _trig(Decimal(mode*(2*i+1))/n_cells)
                for component, value in enumerate((cosine, sine)):
                    value = root_two*sinc*value
                    j = 2*(mode-1)+component
                    weights[j, i] = float(value)
                    errors[j, i] = float(abs(value-Decimal.from_float(weights[j, i]))+Decimal("1e-65"))
    return weights, _up(errors)


@lru_cache(maxsize=16)
def _scale_upper(n_cells):
    with localcontext() as ctx:
        ctx.prec = 80
        values = []
        for k in (1, 1, 2, 2):
            value = _trig(Decimal(k)/n_cells)[0]/(_pi()*k/n_cells)
            values.append(_up(float(value+Decimal("1e-65"))))
        return np.asarray(values)


def profile_probabilities(config, context):
    """Cell probabilities and rigorous componentwise integration-error bounds.

    Composite Simpson quadrature uses an explicit fourth-derivative bound for
    the declared three-mode exponential profile, plus decimal arithmetic error.
    No Monte Carlo initial sample or learned parameter enters this calculation.
    """
    base = _base(config)
    _physics(base, context)
    if not any(context.cos_coeffs) and not any(context.sin_coeffs):
        p = np.full(base.n_cells, 1/base.n_cells)
        error = np.zeros_like(p) if base.n_cells & (base.n_cells-1) == 0 else np.spacing(p)
        return p, error
    with localcontext() as ctx:
        ctx.prec = 80
        coefficients = [(Decimal.from_float(float(a)), Decimal.from_float(float(b)))
                        for a, b in zip(context.cos_coeffs, context.sin_coeffs, strict=True)]
        derivative = [sum((Decimal(k+1)**r*(abs(a)+abs(b))
                          for k, (a, b) in enumerate(coefficients)), Decimal(0)) for r in range(5)]
        fourth_bound = derivative[0].exp()*(derivative[4]+4*derivative[1]*derivative[3]
            +3*derivative[2]**2+6*derivative[1]**2*derivative[2]+derivative[1]**4)
        subdivisions = 128
        h = 2*_pi()/base.n_cells
        arithmetic_error = h*derivative[0].exp()*(1+derivative[0])*Decimal("1e-60")
        error = h**5*fourth_bound/(180*subdivisions**4)+arithmetic_error
        integrals = []
        for cell in range(base.n_cells):
            total = Decimal(0)
            for j in range(subdivisions+1):
                q = Decimal(2*(cell*subdivisions+j))/Decimal(base.n_cells*subdivisions)
                exponent = Decimal(0)
                for k, (a, b) in enumerate(coefficients, 1):
                    sine, cosine = _trig(k*q)
                    exponent += a*cosine+b*sine
                factor = 1 if j in (0, subdivisions) else 4 if j % 2 else 2
                total += factor*exponent.exp()
            integrals.append(total*h/(3*subdivisions))
        normalizer = sum(integrals, Decimal(0))
        normalizer_error = base.n_cells*error+Decimal("1e-60")
        if normalizer <= normalizer_error:
            raise ValueError("Initial-profile normalization could not be enclosed")
        probabilities, errors = [], []
        for integral in integrals:
            lower = max(Decimal(0), (integral-error)/(normalizer+normalizer_error))
            upper = (integral+error)/(normalizer-normalizer_error)
            center = float(integral/normalizer)
            probabilities.append(center)
            errors.append(float(max(Decimal.from_float(center)-lower, upper-Decimal.from_float(center))
                                +Decimal("1e-60")))
        return np.asarray(probabilities), _up(np.asarray(errors))


def _poisson_weights(rate):
    """Finite Poisson weights with decimal-to-binary enclosures and tail bounds."""
    count = int(math.ceil(rate+20*math.sqrt(rate+1)+80))
    with localcontext() as ctx:
        ctx.prec = 80
        mu = Decimal.from_float(float(rate))
        value = (-mu).exp()
        decimal_error = Decimal("1e-70")*max(value, Decimal("1e-300"))
        values, errors = [], []
        for n in range(count+1):
            if n:
                value *= mu/n
                decimal_error = decimal_error*mu/n+abs(value)*Decimal("1e-74")
            center = float(value)
            values.append(center)
            errors.append(float(abs(Decimal.from_float(center)-value)+decimal_error))
        p_last_upper = value+decimal_error
        tail = p_last_upper*mu/Decimal(count+1)/(1-mu/Decimal(count+2))
        first_tail = mu*(p_last_upper+tail)
        return np.asarray(values), _up(np.asarray(errors)), float(_up(float(tail))), float(_up(float(first_tail)))


@lru_cache(maxsize=64)
def _moment_operator(n_cells, n_particles, alpha, horizon, alpha_error=0.):
    """Backward degree-two operator, independent of neural Fourier formulas."""
    weights, weight_error = _weights(n_cells)
    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    h = np.zeros((14, n_cells))
    h[:4] = weights
    quadratic = np.zeros((14, n_cells, n_cells))
    h_error = np.zeros(14)
    h_error[:4] = weight_error.max(axis=1)
    quadratic_error = np.zeros(14)
    for k, (i, j) in enumerate(pairs, 4):
        quadratic[k] = .5*(np.outer(weights[i], weights[j])+np.outer(weights[j], weights[i]))
        quadratic_error[k] = (np.max(np.abs(weights[i]))*np.max(weight_error[j])
            +np.max(np.abs(weights[j]))*np.max(weight_error[i])
            +np.max(weight_error[i])*np.max(weight_error[j])
            +_gamma(8)*np.max(np.abs(weights[i]))*np.max(np.abs(weights[j])))
    max_h, max_q = np.max(np.abs(h), axis=1), np.max(np.abs(quadratic), axis=(1, 2))
    initial_h, initial_q = max_h.copy(), max_q.copy()
    rate = 4*alpha*horizon
    poisson, poisson_error, tail, first_tail = _poisson_weights(rate)
    total_h, total_q = np.zeros_like(h), np.zeros_like(quadratic)
    total_h_error, total_q_error = np.zeros(14), np.zeros(14)
    indices = np.arange(n_cells)
    next_indices = (indices+1) % n_cells
    for n, (weight, weight_radius) in enumerate(zip(poisson, poisson_error, strict=True)):
        total_h += weight*h
        total_q += weight*quadratic
        total_h_error += weight*h_error+weight_radius*(max_h+h_error)
        total_q_error += weight*quadratic_error+weight_radius*(max_q+quadratic_error)
        if n == len(poisson)-1:
            break
        face = (quadratic[:, indices, indices]+quadratic[:, next_indices, next_indices]
                -quadratic[:, indices, next_indices]-quadratic[:, next_indices, indices])
        source = (face+np.roll(face, 1, axis=1))/(4*n_particles)
        h = .5*h+.25*(np.roll(h, 1, axis=1)+np.roll(h, -1, axis=1))+source
        quadratic = .25*(np.roll(quadratic, 1, axis=1)+np.roll(quadratic, -1, axis=1)
                          +np.roll(quadratic, 1, axis=2)+np.roll(quadratic, -1, axis=2))
        h_error = (h_error+2*quadratic_error/n_particles
                   +_gamma(16)*(max_h+2*max_q/n_particles))
        quadratic_error += _gamma(8)*(max_q+quadratic_error)
        max_h = initial_h+2*(n+1)*initial_q/n_particles+h_error
        max_q = initial_q+quadratic_error
    sum_round = _gamma(2*len(poisson)+10)
    # Include formation of alpha and rate from recorded geometry/diffusion.
    rate_roundoff = _gamma(16)*max(rate, 1.)
    total_h_error += (initial_h*tail+2*initial_q*first_tail/n_particles
                      +sum_round*(initial_h+2*(rate+1)*initial_q/n_particles)
                      +3*rate_roundoff*(initial_h+2*(rate+1)*initial_q/n_particles))
    total_q_error += initial_q*(tail+sum_round+2*rate_roundoff)
    total_q_error += alpha_error*horizon*8*initial_q
    total_h_error += alpha_error*horizon*(4*initial_h
        +32*(alpha+alpha_error)*horizon*initial_q/n_particles+8*initial_q/n_particles)
    # Each positive-radius recurrence/accumulation uses fewer than 256 scalar
    # operations per Poisson term; gamma also encloses evaluation of the bounds.
    radius_operations = 256*len(poisson)+2048
    return total_h, total_q, _inflate_bound(total_h_error, radius_operations), _inflate_bound(total_q_error, radius_operations), pairs, {
        "method": "degree-two backward generator Poisson uniformization",
        "poisson_rate": rate, "poisson_terms": len(poisson), "poisson_probability_tail_bound": tail,
        "arithmetic": "80-digit decimal Poisson enclosures; IEEE gamma bounds for backward coefficient operations and summation",
        "alpha_absolute_error": alpha_error, "radius_construction_operation_allowance": radius_operations,
    }


def _riccati_rhs(u, alpha):
    left, right = u-np.roll(u, 1), u-np.roll(u, -1)
    return -alpha*(left+right)-.5*alpha*(left*left+right*right)


def _riccati_jacobian_action(u, v, alpha):
    ul, ur = u-np.roll(u, 1), u-np.roll(u, -1)
    vl, vr = v-np.roll(v, 1), v-np.roll(v, -1)
    return -alpha*(vl+vr)-alpha*(ul*vl+ur*vr)


def _riccati_hessian(v, w, alpha):
    return -alpha*((v-np.roll(v, 1))*(w-np.roll(w, 1))
                   +(v-np.roll(v, -1))*(w-np.roll(w, -1)))


@lru_cache(maxsize=8)
def _riccati_grid(n_cells, alpha, horizon, ode_step, window_step, tilt, alpha_error=0.):
    step = min(ode_step, window_step)
    count = int(round(horizon/step))
    every = int(round(window_step/step))
    if not np.isclose(count*step, horizon, rtol=0, atol=1e-12) or count % every:
        raise ValueError("Stopping horizon must lie on the declared Riccati/window grid")
    u = np.zeros(n_cells)
    u[0] = tilt
    values = np.empty((count//every, n_cells))
    # ||u^(5)||_infinity <= 69888 alpha^5 on the invariant unit cube.
    remainder = 69888*alpha**5*step**5/120
    absolute_taylor_sum = (1+3*alpha*step+24*alpha**2*step**2/2
                          +264*alpha**3*step**3/6+3840*alpha**4*step**4/24)
    roundoff = _gamma(512)*absolute_taylor_sum
    local_error = _up(remainder+roundoff)
    for n in range(count):
        d1 = _riccati_rhs(u, alpha)
        d2 = _riccati_jacobian_action(u, d1, alpha)
        d3 = _riccati_jacobian_action(u, d2, alpha)+_riccati_hessian(d1, d1, alpha)
        d4 = _riccati_jacobian_action(u, d3, alpha)+3*_riccati_hessian(d1, d2, alpha)
        u = u+step*d1+step**2*d2/2+step**3*d3/6+step**4*d4/24
        # This projection concerns the auxiliary proof ODE only. Its exact flow
        # stays in the cube, so projection is nonexpansive and preserves enclosure.
        u = np.clip(u, 0., tilt)
        if (n+1) % every == 0:
            values[(n+1)//every-1] = u
    elapsed = np.arange(1, count//every+1)*every*step
    errors = _inflate_bound(np.arange(1, count//every+1)*every*local_error+3*alpha_error*elapsed, 2048)
    return values, errors, {"integration_step": step, "window_step": window_step,
        "tilt": tilt, "steps": count, "maximum_u_error": float(errors[-1]),
        "local_taylor_remainder": remainder, "local_roundoff_bound": roundoff,
        "fifth_derivative_bound": 69888*alpha**5,
        "alpha_absolute_error": alpha_error,
        "stability": "invariant unit cube, Metzler Jacobian with zero row sums, nonexpansive maximum-norm flow",
        "roundoff": "512-operation gamma allowance per projected Taylor step; elementary +,-,*,/ with IEEE round-to-nearest",
    }


def _exp_upper(x):
    """Upper enclosure using a short Taylor polynomial and outward squaring."""
    x = np.asarray(x, dtype=float)
    maximum = max(float(np.max(np.abs(x), initial=0)), 1.)
    squarings = max(1, int(math.ceil(math.log2(2*maximum))))
    z = x/(2**squarings)
    polynomial = np.full_like(z, 1/math.factorial(20))
    for k in range(19, -1, -1):
        polynomial = polynomial*z+1/math.factorial(k)
    remainder = 2*(.5**21)/math.factorial(21)
    result = _up(polynomial+2*_gamma(64)+remainder)
    for _ in range(squarings):
        result = _up(result*result)
    return result


def _stopping_bounds(base, initial, horizons, settings, probabilities=None, probability_error=None):
    alpha, alpha_error, mass_stop, stop_error = _geometry_parameters(base)
    u, u_error, proof = _riccati_grid(base.n_cells, alpha, max(horizons),
        settings["ode_step"], settings["window_step"], settings["tilt"], alpha_error)
    window = settings["window_step"]
    # Every off-diagonal heat-kernel entry <= alpha*t: its derivative is at
    # most alpha since the two distinct neighboring probabilities sum to <=1.
    initial = np.asarray(initial)
    mass_bound = (1. if probabilities is not None else
                  max(1., float(np.max(initial.sum(1)))*(1+_gamma(2*base.n_cells))))
    alpha_upper, stop_upper = _up(alpha+alpha_error), _up(mass_stop+stop_error)
    boundary = _inflate_bound(settings["tilt"]*(stop_upper
                   +alpha_upper*window*(mass_bound-stop_upper)), 128)
    count = len(initial)
    sums = np.zeros((count, len(u)))
    cyclic = np.stack([np.roll(initial, -shift, axis=1) for shift in range(base.n_cells)], axis=1)
    batch = 512
    for start in range(0, len(u), batch):
        end = min(len(u), start+batch)
        lower = np.maximum(0., np.nextafter(u[start:end]-u_error[start:end, None], -np.inf))
        if probabilities is None:
            dot = lower@cyclic.reshape(-1, base.n_cells).T
            dot_error = _inflate_bound(_gamma(2*base.n_cells+8)*np.maximum(np.abs(dot), 1.), 128)
            exponent = base.n_particles*(boundary-np.nextafter(dot-dot_error, -np.inf))
            exponent = _up(exponent+_gamma(8)*(np.abs(exponent)+1))
            values = _exp_upper(exponent).reshape(end-start, count, base.n_cells).sum(axis=-1).T
        else:
            exp_u = _exp_upper(-lower)
            p_upper = _up(probabilities+probability_error)
            folded = np.stack([exp_u@np.roll(p_upper, -shift) for shift in range(base.n_cells)], axis=1)
            folded = _up(folded*(1+_gamma(2*base.n_cells+8)))
            # Integer powers use outward binary multiplication, not library pow.
            power, factor, n = np.ones_like(folded), folded, base.n_particles
            while n:
                if n % 2:
                    power = _up(power*factor)
                n //= 2
                if n:
                    factor = _up(factor*factor)
            values = (_exp_upper(np.asarray(base.n_particles*boundary))*power).sum(axis=1)[None]
        sums[:, start:end] = values
    cumulative = np.cumsum(sums, axis=1)
    bounds = []
    for horizon in horizons:
        index = int(round(horizon/window))-1
        if index < 0 or not np.isclose((index+1)*window, horizon, atol=1e-12, rtol=0):
            raise ValueError("Reference query horizons must lie on the stopping-window grid")
        bounds.append(np.minimum(1., _up(cumulative[:, index]*(1+_gamma(4*(index+1)*base.n_cells)))))
    probability = np.stack(bounds, axis=1)
    proof.update(boundary_exponent_coefficient=boundary,
        time_windows=len(u), probability_bound=probability.tolist(),
        optional_stopping="P(first hit in window H-epsilon..H at cell i) <= exp(N*boundary_coefficient - N*u_i_shift(H).m0); sum all windows/cells",
        iid_initialization="E exp(-N*u.m0) = (sum_j p_j exp(-u_j))^N; cell-probability upper intervals used" if probabilities is not None else "fixed nonnegative histogram",
        valid=bool(proof["maximum_u_error"] <= settings["certificate_tolerance"]))
    return probability, proof


def _formal_moments(base, initial, horizons, ensemble=False, probability_error=None):
    initial = np.asarray(initial)
    alpha, alpha_error, _, _ = _geometry_parameters(base)
    means, covariances, mean_errors, covariance_errors, methods = [], [], [], [], []
    input_l1_error = 0. if probability_error is None else float(np.sum(probability_error))
    for horizon in horizons:
        h, quadratic, h_error, quadratic_error, pairs, method = _moment_operator(
            base.n_cells, base.n_particles, alpha, float(horizon), alpha_error)
        mean = initial@h[:4].T
        second = np.einsum("bi,kij,bj->bk", initial, quadratic[4:], initial)+initial@h[4:].T
        if ensemble:
            diagonal = np.diagonal(quadratic[4:], axis1=-2, axis2=-1)
            quadratic_part = np.einsum("bi,kij,bj->bk", initial, quadratic[4:], initial)
            second += (initial@diagonal.T-quadratic_part)/base.n_particles
        mean_error = (h_error[:4]+_gamma(2*base.n_cells+8)*np.max(np.abs(h[:4]), axis=1)
                      +np.max(np.abs(h[:4]), axis=1)*input_l1_error)
        second_error = (h_error[4:]+quadratic_error[4:]
            +_gamma(2*base.n_cells**2+32)*(np.max(np.abs(quadratic[4:]), axis=(1, 2))
                                        +np.max(np.abs(h[4:]), axis=1))
            +np.max(np.abs(quadratic[4:]), axis=(1, 2))*(2*input_l1_error+input_l1_error**2)
            +np.max(np.abs(h[4:]), axis=1)*input_l1_error)
        covariance = np.empty((len(initial), 4, 4))
        covariance_error = np.empty_like(covariance)
        for k, (i, j) in enumerate(pairs):
            value = base.n_particles*(second[:, k]-mean[:, i]*mean[:, j])
            error = base.n_particles*(second_error[k]+np.abs(mean[:, i])*mean_error[j]
                +np.abs(mean[:, j])*mean_error[i]+mean_error[i]*mean_error[j]
                +_gamma(4)*(np.abs(second[:, k])+np.abs(mean[:, i]*mean[:, j])))
            covariance[:, i, j] = covariance[:, j, i] = value
            covariance_error[:, i, j] = covariance_error[:, j, i] = _up(error)
        means.append(mean)
        covariances.append(covariance)
        # Positive-bound construction has <4096 operations per entry.  The
        # additional 1e-9 factor covers the accepted 1e-12 unit-mass tolerance.
        mean_errors.append(np.broadcast_to(_inflate_bound(mean_error, 4096)*(1+1e-9), mean.shape))
        covariance_errors.append(_inflate_bound(covariance_error, 4096)*(1+1e-9))
        methods.append(method)
    return (np.stack(means, 1), np.stack(covariances, 1),
            np.stack(mean_errors, 1), np.stack(covariance_errors, 1), methods)


def _packet(base, initial, horizons, settings, ensemble=False, probability_error=None):
    mean, covariance, mean_error, covariance_error, methods = _formal_moments(
        base, initial, horizons, ensemble, probability_error)
    stopping, proof = _stopping_bounds(base, initial, horizons, settings,
        probabilities=initial[0] if ensemble else None, probability_error=probability_error)
    std = _scale_upper(base.n_cells)
    numerical_mean_error, numerical_covariance_error = mean_error.copy(), covariance_error.copy()
    mass_bound = 1. if ensemble else max(1., float(np.max(initial.sum(1)))*(1+_gamma(2*base.n_cells)))
    mean_error += stopping[..., None]*(2*np.sqrt(2)*std*mass_bound)
    covariance_error += (stopping[..., None, None]*(12*base.n_particles*mass_bound**2+2*mass_bound)
                         *np.outer(std, std))
    _, _, mass_stop, stop_error = _geometry_parameters(base)
    initial_stopped = np.any(initial <= mass_stop-stop_error, axis=1) if not ensemble else np.zeros(1, bool)
    ambiguous_initial = ((~initial_stopped)&np.any(initial <= mass_stop+stop_error, axis=1)
                         if not ensemble else np.zeros(1, bool))
    if np.any(initial_stopped):
        weights, weight_error = _weights(base.n_cells)
        frozen = initial[initial_stopped]@weights.T
        mean[initial_stopped] = frozen[:, None]
        covariance[initial_stopped] = 0.
        mean_error[initial_stopped] = (weight_error.max(axis=1)
                                      +_gamma(2*base.n_cells+8)*np.abs(weights).max(axis=1))
        covariance_error[initial_stopped] = 0.
    mean_error, covariance_error = _inflate_bound(mean_error, 1024), _inflate_bound(covariance_error, 1024)
    audited_scope = (base.n_cells == 32 and base.n_particles == 2048
                     and base.diffusion == 1. and max(horizons) <= 1.)
    certificate = {"valid": bool(audited_scope and (proof["valid"] or np.all(initial_stopped))
                                  and not np.any(ambiguous_initial)
                                  and np.all(np.isfinite(mean_error))
                                  and np.all(np.isfinite(covariance_error))), "stopping": proof,
        "moment_methods": methods, "initially_stopped_conditional_states": int(initial_stopped.sum()),
        "manually_audited_scope": "G=32, N=2048, D=1, noninteracting, horizons<=1; other settings return an unresolved certificate",
        "within_manually_audited_scope": audited_scope,
        "ambiguous_initial_boundary_states": int(ambiguous_initial.sum()),
        "initial_mass_upper_bound": mass_bound,
        "input_mass_tolerance": "Nonnegative stored masses within 1e-12 of unit mass; compact bounds use the outward upper mass and numerical radii include a 1e-9 normalization allowance",
        "maximum_numerical_mean_error": float(np.max(numerical_mean_error)),
        "maximum_numerical_scaled_covariance_error": float(np.max(numerical_covariance_error)),
        "maximum_total_mean_error": float(np.max(mean_error)),
        "maximum_total_scaled_covariance_error": float(np.max(covariance_error)),
        "geometry": "Target coefficients use mathematical h=2*pi/G and recorded D,rho_stop. Machin/pi decimal enclosures bound their difference from binary64 alpha/mass_stop used by the numerical algorithms; observation weights use enclosed mathematical trigonometry.",
        "stopping_mean_bound_normalized": "2*sqrt(2*N)*P(stop by T)",
        "stopping_covariance_bound_normalized": "(12*N+2)*P(stop by T), from bounded raw moments and mean-product perturbations",
        "stopping_bias_proof": "Stopped Dynkin's identity gives the interior backward polynomial plus a boundary correction. In normalized units |Y|<=sqrt(2N), the formal heat mean has the same bound and its conditional covariance entries have magnitude<=2. Hence mean correction<=2sqrt(2N)P; raw-second correction<=(4N+2)P; the mean-product correction<=8NP, giving covariance bound(12N+2)P. No globally defined unstopped DK process is assumed.",
        "physical_target": {"law": "continuous stopped native noninteracting DK",
            "n_cells": base.n_cells, "n_particles": base.n_particles,
            "diffusion": base.diffusion, "mass_stop": base.spec.mass_stop,
            "stopping": "first continuous boundary hitting followed by freezing"},
        "interpretation": "Bounds enclose the stopped target; certificate validity does not imply neural accuracy or attainment of a requested tolerance"}
    result = {"mean": mean, "scaled_covariance": covariance,
        "mean_error_bound": mean_error, "scaled_covariance_error_bound": covariance_error,
        "horizons": tuple(horizons), "certificate": certificate}
    from .generator_config import generator_target_id

    result["target_id"] = generator_target_id(base, "dk")
    if ensemble:
        for key in ("mean", "scaled_covariance", "mean_error_bound", "scaled_covariance_error_bound"):
            result[key] = result[key][0]
    return result


def reference_conditional(config, initial, horizons, settings=None):
    """Stopped continuous DK moments conditional on complete initial histograms."""
    base = _base(config)
    _physics(base)
    initial = np.asarray(initial, dtype=float)
    if initial.ndim == 1:
        initial = initial[None]
    if (initial.ndim != 2 or initial.shape[1] != base.n_cells or not np.all(np.isfinite(initial))
            or np.any(initial < 0) or not np.allclose(initial.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Finite nonnegative normalized native initial masses required")
    horizons = _horizons(horizons)
    return _packet(base, initial, horizons, _settings(settings))


def reference_ensemble(config, context, horizons, settings=None):
    """Stopped continuous DK moments including the exact known IID histogram law."""
    base = _base(config)
    _physics(base, context)
    probabilities, error = profile_probabilities(base, context)
    packet = _packet(base, probabilities[None], _horizons(horizons),
                     _settings(settings), ensemble=True, probability_error=error)
    packet["certificate"]["profile_probability_error"] = error.tolist()
    audited_profile = (not any(context.sin_coeffs) and context.cos_coeffs in
                       ((0., 0., 0.), (.2, 0., 0.)))
    packet["certificate"]["within_manually_audited_profile_family"] = audited_profile
    packet["certificate"]["valid"] = bool(packet["certificate"]["valid"] and audited_profile)
    return packet


def _horizons(horizons):
    horizons = tuple(float(t) for t in horizons)
    if (not horizons or not all(np.isfinite(t) and t > 0 for t in horizons)
            or any(a >= b for a, b in zip(horizons, horizons[1:]))):
        raise ValueError("Positive increasing reference horizons required")
    return horizons


def _negative_exp_interval(lower, upper):
    """Enclose exp(-x) using positive Taylor terms, a geometric tail, reciprocal."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    count = int(math.ceil(2*float(np.max(upper))+50))
    term_lo, term_hi = np.ones_like(lower), np.ones_like(upper)
    total_lo, total_hi = term_lo.copy(), term_hi.copy()
    for n in range(1, count+1):
        term_lo = np.maximum(0., np.nextafter(np.nextafter(term_lo*lower, -np.inf)/n, -np.inf))
        term_hi = _up(_up(term_hi*upper)/n)
        total_lo = np.maximum(0., np.nextafter(total_lo+term_lo, -np.inf))
        total_hi = _up(total_hi+term_hi)
    next_term = _up(_up(term_hi*upper)/(count+1))
    ratio = _up(upper/(count+2))
    tail = _up(next_term/np.nextafter(1-ratio, -np.inf))
    exp_hi = _up(total_hi+tail)
    return np.nextafter(1/exp_hi, -np.inf), _up(1/total_lo)


def reference_ou(config, initial, horizons):
    """Certified continuous four-mode OU conditional moments, with no MC error."""
    from .generator_config import generator_target_id

    base, horizons = _base(config), _horizons(horizons)
    initial = np.asarray(initial, dtype=float)
    if initial.ndim == 1:
        initial = initial[None]
    if initial.ndim != 2 or initial.shape[1] != 4 or not np.all(np.isfinite(initial)):
        raise ValueError("Finite four-coordinate OU initial states required")
    lower, upper = np.empty((len(horizons), 4)), np.empty((len(horizons), 4))
    with localcontext() as ctx:
        ctx.prec = 80
        h = 2*_pi()/base.n_cells
        alpha = Decimal.from_float(float(base.diffusion))/(h*h)
        rates = [4*alpha*_trig(Decimal(k)/base.n_cells)[0]**2 for k in (1, 1, 2, 2)]
        for j, horizon in enumerate(horizons):
            for i, rate in enumerate(rates):
                value = rate*Decimal.from_float(horizon)
                lower[j, i] = max(0., np.nextafter(float(value-Decimal("1e-60")), -np.inf))
                upper[j, i] = _up(float(value+Decimal("1e-60")))
    if np.max(2*upper) > 600:
        raise ValueError("OU exponential certificate supports rate*horizon <=300")
    decay_lo, decay_hi = _negative_exp_interval(lower, upper)
    decay = .5*(decay_lo+decay_hi)
    decay_error = _up(np.maximum(decay-decay_lo, decay_hi-decay))
    mean = initial[:, None]*decay
    mean_error = _up(np.abs(initial[:, None])*decay_error+_gamma(4)*np.abs(mean))
    double_lo, double_hi = _negative_exp_interval(2*lower, 2*upper)
    variance_lo, variance_hi = np.nextafter(1-double_hi, -np.inf), _up(1-double_lo)
    variance = .5*(variance_lo+variance_hi)
    variance_error = _up(np.maximum(variance-variance_lo, variance_hi-variance))
    covariance = np.zeros((len(initial), len(horizons), 4, 4))
    covariance_error = np.zeros_like(covariance)
    for i in range(4):
        covariance[..., i, i] = variance[:, i]
        covariance_error[..., i, i] = variance_error[:, i]
    return {"mean": mean, "scaled_covariance": covariance,
        "mean_error_bound": mean_error, "scaled_covariance_error_bound": covariance_error,
        "horizons": horizons, "target_id": generator_target_id(base, "ou"),
        "certificate": {"valid": bool(base.n_cells == 32 and base.n_particles == 2048
                                       and base.diffusion == 1. and max(horizons) <= 1.),
            "manually_audited_scope": "G=32, N=2048, D=1, four declared OU modes and horizons<=1",
            "stopping_bias": 0., "sampling_uncertainty": 0.,
            "reference": "continuous diagonal OU conditional moments",
            "numerical_method": "80-digit Machin/pi and trigonometric Taylor bounds for rates; nonnegative exponential Taylor series with geometric tail, outward IEEE arithmetic and reciprocal intervals",
            "maximum_numerical_mean_error": float(np.max(mean_error)),
            "maximum_numerical_scaled_covariance_error": float(np.max(covariance_error))}}
