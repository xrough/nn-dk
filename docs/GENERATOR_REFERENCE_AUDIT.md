# Continuous generator reference: independent mathematical audit

Audit date: 25 October 2026. This is a read-only review of the reference
mathematics and arithmetic enclosures, separate from implementation tests and
neural accuracy measurements. No final neural test outcomes were inspected.

Reviewed implementation: [generator_reference.py](../src/nn_dk/generator_reference.py),
SHA-256 `0f7e088e01390a72056679876e575a9904c306b0062b801c02f640059dc2d70c`.
Subsequent changes to its numerical formulas require renewed review.

The reviewed scope is noninteracting native DK with G=32, N=2048, D=1, T≤1,
the stated positive stopping threshold, and the uniform or exp(0.2 cos x)
IID initial laws. It concerns a continuous-time, fixed-grid stopped SDE.
It makes no assertion about continuum or unregularized density-valued DK.

## Mathematical checks

For a polynomial `h·m + mᵀQm`, the backward generator preserves degree at most
two. The Poisson uniformization recurrence at rate `4α`, α=D/h², is correct:
Q becomes the average of its four neighboring axis shifts, while h receives
its heat averaging and `(face(Q)+previous_face(Q))/(4N)`. This calculation
solves a formal interior polynomial problem. It does **not** assume that an
unstopped DK process exists after leaving the nonnegative simplex.

For the auxiliary exponential `exp(-N u(t)·m)`, the Riccati signs are correct.
With tilt at most one its invariant cube has a Metzler Jacobian with zero row
sums; the exact flow is nonexpansive in maximum norm. The derivative bounds
used for the Taylor remainder follow from `||J||∞≤8α` and the bilinear Hessian
bound `8α`: the constants 3, 24, 264, 3840 and 69888 are valid through the fifth
time derivative. Projecting the auxiliary numerical ODE onto its invariant
cube preserves its error enclosure; physical DK states are not projected.

The stopping-window argument is valid. The Riccati solution is bounded above
by the corresponding heat kernel. Every off-diagonal heat-kernel entry is at
most α times elapsed time. At a first-hit simplex face this supplies the
stated lower bound on the positive exponential martingale. Optional stopping
and a union over cells and time windows bound continuous first hitting,
including initial boundary hits and tied hit cells conservatively.

Stopped Dynkin's identity leaves a boundary correction to the interior
polynomial. For physical observation scale s, heat means are bounded by
√2 s and formal N-scaled covariance entries by `2 s_i s_j`. The latter follows
from noise variation `q_i≤4λ_i s_i²` and its heat-decay integral. Consequently
the mean correction is bounded by `2√2 s_i P(stop)` and the covariance
correction by `(12N+2)s_i s_j P(stop)`. The code also enlarges these constants
for its accepted input mass tolerance. This justifies the stopped reference
without a hidden globally unstopped process substitution.

## Numerical enclosures and limits

The updated implementation explicitly covers mathematical-π geometry,
trigonometric weights, Poisson tails, Simpson profile integration, coefficient
operations, Riccati truncation and roundoff, and final moment projections.
The alpha-sensitivity constants follow by scaling the same generator in time.
Upward probability endpoints and Poisson tails, downward Riccati lower bounds,
and conservative gamma inflation of radius construction address the issues
identified in the initial review. The OU positive-series exponential and
reciprocal intervals are also mathematically consistent.

No mathematical blocker was identified within the declared scope, assuming
the stated IEEE binary64/round-to-nearest and Decimal arithmetic behavior and
the documented conservative operation counts. This is an independently
reviewed numerical enclosure, not a machine-checked proof.

Function caveats remain explicit:

- `reference_conditional` requires finite nonnegative masses within its stated
  unit-mass tolerance. Inputs ambiguous within the threshold enclosure cannot
  receive a valid stopped certificate.
- `_riccati_grid` and the stopping queries must share the declared nested
  time/window grid. An excessive proof-ODE error invalidates the certificate.
- `profile_probabilities` was reviewed for the two prescribed moderate profile
  laws, not arbitrary enormous exponential-profile coefficients.
- `reference_ou` explicitly restricts rate times horizon to at most 300.
- A valid reference certificate does not establish neural accuracy. The full
  pointwise comparison must still include the reference bound and satisfy the
  unchanged error and uncertainty budgets. No observed stopping events, small
  RMSE or low training residual can substitute for those checks.
