// Optional single-thread FP64 native Euler/JVP control.  No neural dependency.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

extern "C" int nn_dk_control_abi() { return 2; }

extern "C" int nn_dk_native_euler(
    int paths, int steps, int cells, double h, double mass_stop,
    double diffusion, double kappa, double a2, double n_particles, double dt,
    const double* initial, const double* brownian, const double* initial_tangent,
    const double* physics_tangent, int with_jvp, double* states, double* stops,
    double* tangents, double* stop_tangents) {
  if (paths < 1 || steps < 1 || cells < 2 || !(h > 0) || !(diffusion > 0) ||
      !(n_particles > 0) || !(dt > 0) || !initial || !brownian || !states || !stops)
    return 1;
  if (with_jvp && (!initial_tangent || !physics_tangent || !tangents || !stop_tangents))
    return 1;
  const double sinc1 = std::sin(h / 2) / (h / 2);
  const double sinc2 = std::sin(h) / h;
  const double scale[4] = {-kappa * sinc1, -kappa * sinc1, 2 * a2 * sinc2, 2 * a2 * sinc2};
  const double dp[3] = {with_jvp ? physics_tangent[0] : 0.,
                        with_jvp ? physics_tangent[1] : 0.,
                        with_jvp ? physics_tangent[2] : 0.};
  const double dscale[4] = {-dp[1] * sinc1, -dp[1] * sinc1,
                            2 * dp[2] * sinc2, 2 * dp[2] * sinc2};
  const double inverse_h2 = 1 / (h * h), inverse_sqrt_n = 1 / std::sqrt(n_particles);
  std::vector<double> moments(4 * cells), reconstruction(4 * cells);
  for (int cell = 0; cell < cells; ++cell) {
    const double center = (cell + .5) * h, face = (cell + 1) * h;
    moments[4 * cell] = std::cos(center);
    moments[4 * cell + 1] = std::sin(center);
    moments[4 * cell + 2] = std::cos(2 * center);
    moments[4 * cell + 3] = std::sin(2 * center);
    reconstruction[4 * cell] = std::sin(face);
    reconstruction[4 * cell + 1] = -std::cos(face);
    reconstruction[4 * cell + 2] = std::sin(2 * face);
    reconstruction[4 * cell + 3] = -std::cos(2 * face);
  }
  std::vector<double> m(cells), v(cells), flux(cells), noise(cells), proposal(cells),
      dflux(cells), dnoise(cells), proposed_v(cells), denominator(cells);
  for (int path = 0; path < paths; ++path) {
    const std::size_t origin_offset = static_cast<std::size_t>(path) * cells;
    const std::size_t state_offset = static_cast<std::size_t>(path) * (steps + 1) * cells;
    bool active = true;
    for (int cell = 0; cell < cells; ++cell) {
      m[cell] = initial[origin_offset + cell];
      if (!std::isfinite(m[cell]) || m[cell] < 0) return 2;
      active = active && m[cell] > mass_stop;
      states[state_offset + cell] = m[cell];
      if (with_jvp) {
        v[cell] = initial_tangent[origin_offset + cell];
        if (!std::isfinite(v[cell])) return 2;
        tangents[state_offset + cell] = v[cell];
      }
    }
    stops[path] = active ? std::numeric_limits<double>::infinity() : 0.;
    if (with_jvp) stop_tangents[path] = 0.;
    for (int step = 0; step < steps; ++step) {
      const std::size_t output_offset = state_offset + static_cast<std::size_t>(step + 1) * cells;
      if (active) {
        double summary[4] = {}, dsummary[4] = {};
        for (int cell = 0; cell < cells; ++cell) {
          for (int mode = 0; mode < 4; ++mode) {
            summary[mode] += m[cell] * moments[4 * cell + mode];
            if (with_jvp) dsummary[mode] += v[cell] * moments[4 * cell + mode];
          }
        }
        for (int cell = 0; cell < cells; ++cell) {
          const int following = (cell + 1) % cells;
          double velocity = 0., dvelocity = 0.;
          for (int mode = 0; mode < 4; ++mode) {
            velocity += summary[mode] * scale[mode] * reconstruction[4 * cell + mode];
            if (with_jvp)
              dvelocity += (dsummary[mode] * scale[mode] + summary[mode] * dscale[mode])
                           * reconstruction[4 * cell + mode];
          }
          const double face = (m[cell] + m[following]) / (2 * h);
          const double amplitude = std::sqrt(diffusion * inverse_h2 * (m[cell] + m[following]));
          const std::size_t noise_offset = (static_cast<std::size_t>(path) * steps + step) * cells + cell;
          const double increment = brownian[noise_offset];
          if (!std::isfinite(increment) || !std::isfinite(amplitude)) return 2;
          flux[cell] = face * velocity - diffusion * inverse_h2 * (m[following] - m[cell]);
          noise[cell] = amplitude * increment * inverse_sqrt_n;
          if (with_jvp) {
            dflux[cell] = ((v[cell] + v[following]) / (2 * h) * velocity + face * dvelocity
                - diffusion * inverse_h2 * (v[following] - v[cell])
                - dp[0] * inverse_h2 * (m[following] - m[cell]));
            const double da = (diffusion * (v[cell] + v[following])
                + dp[0] * (m[cell] + m[following])) * inverse_h2 / (2 * amplitude);
            dnoise[cell] = da * increment * inverse_sqrt_n;
          }
        }
        double fraction = 1.;
        int first_cell = 0;
        bool crossed = false;
        for (int cell = 0; cell < cells; ++cell) {
          const int preceding = (cell + cells - 1) % cells;
          proposal[cell] = m[cell] + dt * (flux[preceding] - flux[cell]) + noise[preceding] - noise[cell];
          if (!std::isfinite(proposal[cell])) return 2;
          if (with_jvp) {
            proposed_v[cell] = v[cell] + dt * (dflux[preceding] - dflux[cell])
                               + dnoise[preceding] - dnoise[cell];
            if (!std::isfinite(proposed_v[cell])) return 2;
          }
          crossed = crossed || proposal[cell] <= mass_stop;
          const bool descending = proposal[cell] <= mass_stop && proposal[cell] < m[cell];
          denominator[cell] = descending ? m[cell] - proposal[cell] : 1.;
          const double ratio = descending ? (m[cell] - mass_stop) / denominator[cell] : 1.;
          if (ratio < fraction) {
            fraction = ratio;
            first_cell = cell;
          }
        }
        double dfraction = 0.;
        if (with_jvp && crossed) {
          const double den = denominator[first_cell], numerator = m[first_cell] - mass_stop;
          dfraction = (v[first_cell] * den
              - numerator * (v[first_cell] - proposed_v[first_cell])) / (den * den);
          stop_tangents[path] = dt * dfraction;
        }
        for (int cell = 0; cell < cells; ++cell) {
          if (with_jvp)
            v[cell] = v[cell] + fraction * (proposed_v[cell] - v[cell])
                      + dfraction * (proposal[cell] - m[cell]);
          m[cell] = m[cell] + fraction * (proposal[cell] - m[cell]);
          if (!std::isfinite(m[cell]) || (with_jvp && !std::isfinite(v[cell]))) return 2;
        }
        if (crossed) {
          stops[path] = (step + fraction) * dt;
          active = false;
        }
      }
      for (int cell = 0; cell < cells; ++cell) {
        states[output_offset + cell] = m[cell];
        if (with_jvp) tangents[output_offset + cell] = v[cell];
      }
    }
  }
  return 0;
}

// Reverse one arbitrary complete path-state cotangent.  The caller has already
// run the forward solver once; no second forward integration is hidden here.
extern "C" int nn_dk_native_adjoint(
    int paths, int steps, int cells, double h, double mass_stop,
    double diffusion, double kappa, double a2, double n_particles, double dt,
    const double* states, const double* brownian, const double* stops,
    const double* path_gradient, double* initial_gradient, double* physics_gradient) {
  if (paths < 1 || steps < 1 || cells < 2 || !(h > 0) || !(diffusion > 0) ||
      !(n_particles > 0) || !(dt > 0) || !states || !brownian || !stops ||
      !path_gradient || !initial_gradient || !physics_gradient)
    return 1;
  const double sinc1 = std::sin(h / 2) / (h / 2), sinc2 = std::sin(h) / h;
  const double scale[4] = {-kappa * sinc1, -kappa * sinc1, 2 * a2 * sinc2, 2 * a2 * sinc2};
  const double kscale[4] = {-sinc1, -sinc1, 0., 0.};
  const double ascale[4] = {0., 0., 2 * sinc2, 2 * sinc2};
  const double inverse_h2 = 1 / (h * h), inverse_sqrt_n = 1 / std::sqrt(n_particles);
  std::vector<double> moments(4 * cells), reconstruction(4 * cells);
  for (int cell = 0; cell < cells; ++cell) {
    const double center = (cell + .5) * h, face = (cell + 1) * h;
    moments[4 * cell] = std::cos(center);
    moments[4 * cell + 1] = std::sin(center);
    moments[4 * cell + 2] = std::cos(2 * center);
    moments[4 * cell + 3] = std::sin(2 * center);
    reconstruction[4 * cell] = std::sin(face);
    reconstruction[4 * cell + 1] = -std::cos(face);
    reconstruction[4 * cell + 2] = std::sin(2 * face);
    reconstruction[4 * cell + 3] = -std::cos(2 * face);
  }
  std::vector<double> cotangent(cells), origin(cells), face(cells), velocity(cells),
      flux(cells), amplitude(cells), noise(cells), proposal(cells), denominator(cells),
      cotp(cells), cotm(cells), z(cells), zn(cells), face_cot(cells), mobility_cot(cells);
  for (int path = 0; path < paths; ++path) {
    const std::size_t state_offset = static_cast<std::size_t>(path) * (steps + 1) * cells;
    double* physical = physics_gradient + 3 * path;
    physical[0] = physical[1] = physical[2] = 0.;
    for (int cell = 0; cell < cells; ++cell) {
      cotangent[cell] = path_gradient[state_offset + static_cast<std::size_t>(steps) * cells + cell];
      if (!std::isfinite(cotangent[cell])) return 2;
    }
    for (int step = steps - 1; step >= 0; --step) {
      const std::size_t offset = state_offset + static_cast<std::size_t>(step) * cells;
      const bool active = stops[path] > step * dt;
      double summary[4] = {};
      for (int cell = 0; cell < cells; ++cell) {
        origin[cell] = active ? states[offset + cell] : 1. / cells;
        for (int mode = 0; mode < 4; ++mode)
          summary[mode] += origin[cell] * moments[4 * cell + mode];
      }
      for (int cell = 0; cell < cells; ++cell) {
        const int following = (cell + 1) % cells;
        velocity[cell] = 0.;
        for (int mode = 0; mode < 4; ++mode)
          velocity[cell] += summary[mode] * scale[mode] * reconstruction[4 * cell + mode];
        face[cell] = (origin[cell] + origin[following]) / (2 * h);
        flux[cell] = face[cell] * velocity[cell]
                     - diffusion * inverse_h2 * (origin[following] - origin[cell]);
        amplitude[cell] = std::sqrt(diffusion * inverse_h2 * (origin[cell] + origin[following]));
        const std::size_t noise_offset = (static_cast<std::size_t>(path) * steps + step) * cells + cell;
        noise[cell] = amplitude[cell] * brownian[noise_offset] * inverse_sqrt_n;
        if (!std::isfinite(amplitude[cell]) || !std::isfinite(noise[cell])) return 2;
      }
      bool crossed = false;
      double fraction = 1.;
      int first_cell = 0;
      for (int cell = 0; cell < cells; ++cell) {
        const int preceding = (cell + cells - 1) % cells;
        const double current = states[offset + cell];
        proposal[cell] = active ? current + dt * (flux[preceding] - flux[cell])
                                 + noise[preceding] - noise[cell] : current;
        crossed = crossed || (active && proposal[cell] <= mass_stop);
        const bool descending = proposal[cell] <= mass_stop && proposal[cell] < current;
        denominator[cell] = descending ? current - proposal[cell] : 1.;
        const double ratio = descending ? (current - mass_stop) / denominator[cell] : 1.;
        if (ratio < fraction) {
          fraction = ratio;
          first_cell = cell;
        }
      }
      double factor = 0.;
      if (crossed) {
        for (int cell = 0; cell < cells; ++cell)
          factor += cotangent[cell] * (proposal[cell] - states[offset + cell]);
        factor /= denominator[first_cell];
      }
      for (int cell = 0; cell < cells; ++cell) {
        cotp[cell] = active ? fraction * cotangent[cell] : 0.;
        cotm[cell] = active ? (1 - fraction) * cotangent[cell] : cotangent[cell];
      }
      cotp[first_cell] += factor * fraction;
      cotm[first_cell] += factor * (1 - fraction);
      for (int cell = 0; cell < cells; ++cell) {
        const int following = (cell + 1) % cells;
        const double btq = cotp[following] - cotp[cell];
        const std::size_t noise_offset = (static_cast<std::size_t>(path) * steps + step) * cells + cell;
        z[cell] = dt * btq;
        zn[cell] = btq * brownian[noise_offset] * inverse_sqrt_n;
        face_cot[cell] = z[cell] * velocity[cell] / (2 * h);
        mobility_cot[cell] = zn[cell] * diffusion * inverse_h2 / (2 * amplitude[cell]);
      }
      double force_cot[4] = {};
      for (int cell = 0; cell < cells; ++cell)
        for (int mode = 0; mode < 4; ++mode)
          force_cot[mode] += z[cell] * face[cell] * reconstruction[4 * cell + mode];
      for (int cell = 0; cell < cells; ++cell) {
        const int preceding = (cell + cells - 1) % cells, following = (cell + 1) % cells;
        double force_pullback = 0., kvelocity = 0., avelocity = 0.;
        for (int mode = 0; mode < 4; ++mode) {
          force_pullback += force_cot[mode] * scale[mode] * moments[4 * cell + mode];
          kvelocity += summary[mode] * kscale[mode] * reconstruction[4 * cell + mode];
          avelocity += summary[mode] * ascale[mode] * reconstruction[4 * cell + mode];
        }
        const double result = cotp[cell] + diffusion * inverse_h2 * (z[cell] - z[preceding])
            + face_cot[cell] + face_cot[preceding] + force_pullback
            + mobility_cot[cell] + mobility_cot[preceding];
        cotangent[cell] = path_gradient[offset + cell] + cotm[cell] + (active ? result : 0.);
        physical[0] += z[cell] * (-(origin[following] - origin[cell]) * inverse_h2)
                       + zn[cell] * amplitude[cell] / (2 * diffusion);
        physical[1] += z[cell] * face[cell] * kvelocity;
        physical[2] += z[cell] * face[cell] * avelocity;
        if (!std::isfinite(cotangent[cell])) return 2;
      }
    }
    for (int cell = 0; cell < cells; ++cell)
      initial_gradient[static_cast<std::size_t>(path) * cells + cell] = cotangent[cell];
    if (!std::isfinite(physical[0]) || !std::isfinite(physical[1]) || !std::isfinite(physical[2])) return 2;
  }
  return 0;
}
