# Audit of the causal density-path operator

Audited on **25 October 2026**, against source commit
`bd256de905497df821a5fcea57a333995c1781fd`.
The package source identity recorded by the completed smoke run is
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.
This is the repository's `source_hash()` identity, computed from all package
Python source texts; it is distinct from the individual file SHA-256 values
listed below.

This audit concerns the approved **density-path** implementation, not the
earlier moment predictor. It inspects mathematical consistency, implementation
boundaries, and provenance controls. It does not prove convergence of a trained
network or certify continuous-time DK dynamics.

## Finding and scope

The implementation is a causal, fixed-weight solution operator with explicit
physical Brownian inputs. It returns every native finite-volume cell mass,
including the initial state. The known DK interaction and mobility remain in
the equation; the neural network learns a propagator kernel. No per-realization
optimizer or learned statistical moment head substitutes for a trajectory.

No Itô-convention error was identified in the inspected ansatz or residual.
Known coefficients are evaluated at predictable left endpoints. The same
physical Brownian increments enter the operator and the residual. All grid
modes and all physical face drivers are retained.

The numerical target is the declared **stopped Euler polygon** at a fixed grid
and numerical step. Same-step agreement, finite Picard depth, or stable linear
poles alone cannot prove continuous-time boundary convergence, nonlinear
stochastic stability, or a continuum DK limit.

The completed `runs/path-smoke-2026-10-25-v2` manifest records five configured
fits and a completed workflow. This establishes that the actual trained
implementation can execute all five cases and save its artifacts. The smoke
horizon is only `1/16`; this audit does not extrapolate its scientific accuracy
to `T=1`. No locked final input or outcome from the ongoing first scientific
batch was inspected for this audit.

## Kernel parameterization is not successful learning

At spatial frequency `k`, write the known instantaneous heat eigenvalue as

\[
\lambda_k=\frac{4D}{h^2}\sin^2(\pi k/G),\qquad
\beta_k=\Delta\lambda_k.
\]

For nonzero modes, the pole parameterization in `kernel_values` is

\[
q_{\theta,k}=1-(1-\varepsilon)
\operatorname{sigmoid}\!\left[
\operatorname{logit}\!\left(\frac{\beta_k/2}{1-\varepsilon}\right)
+u_\theta(k,0,\text{physics},\Delta)\right],
\qquad\varepsilon=10^{-8}.
\]

Its lag response is

\[
G_\theta(n,k)=q_{\theta,k}^{\,n}
\left[1+0.1\frac{n\Delta}{1+n\Delta}
\tanh v_\theta(k,n\Delta,\text{physics},\Delta)\right].
\]

The constant mode is set to one, independently of the network. The three
hidden-layer SiLU network has seven declared features: frequency, squared sine
frequency, physical lag, `kappa`, `a2`, `D`, and `dt`.

At initialization, the final layer is zero, so the nonconstant poles are
`1 - beta_k/2`, and the lag correction is zero. Thus initialization uses known
instantaneous diffusion units but is **not** the correct future Euler solution.
Weights must change for its full trajectory predictions to become accurate.

The supported pole domain requires `beta_k < 1 - epsilon`. Within this domain,
the exact Euler pole `1-beta_k` is an admissible value of the per-mode pole
parameterization: its required logit correction is

\[
u_k^*=\operatorname{logit}\!\left(\frac{\beta_k}{1-\varepsilon}\right)
-\operatorname{logit}\!\left(\frac{\beta_k/2}{1-\varepsilon}\right).
\]

Setting the lag correction to zero then gives the Euler Green kernel. This
algebraic admissibility is distinct from a finite MLP jointly approximating
these values and an optimizer actually finding them. Exact-kernel control
tests establish the Green convolution's correctness; they do not establish
trained-network accuracy. Separate tests change weights and fit on independent
paths to verify that the neural prediction genuinely learns.

Each discrete-time resolution receives a separate fit. The network contains
`dt` and is not a step-independent coefficient model. At `G=32`, the configured
`2^-7, 2^-8, 2^-9` steps satisfy the pole domain. The larger native Euler
stability domain includes some negative-pole steps that this ansatz rejects.

## Causality, conservation, and nonlinear Picard blocks

Let `bar m_j^ell` be the preceding Picard path after its first-crossing/freezing
rule, and let `a_j^ell` denote its active-at-start indicator. The implemented
raw update is

\[
z^{\ell+1}_n=G_\theta(n)m_0+
\sum_{j<n}G_\theta(n-1-j)a_j^\ell
\left[\Delta f_{\mathrm{int}}(\bar m_j^\ell)
+N^{-1/2}S_h(\bar m_j^\ell)\Delta W_j\right].
\]

The zeroth iterate is the learned homogeneous heat response. In the heat
control, forcing is zero. In the additive control, the native uniform face
amplitude is constant, and one causal convolution suffices.

Every output at index `n` depends only on `m0` and increments with indices
strictly below `n`. Spatial FFTs act within one time slice. The temporal
contraction uses an explicitly lower-triangular matrix; it does not rely on
future terms canceling in a temporal FFT. Intermediate stopping also examines
only an already constructed prefix. Consequently each mobility multiplying
`Delta W_j` is adapted to the past of that physical Brownian path.

The periodic divergence satisfies `1^T B=0`. The response discards the forcing
zero mode, fixes the propagator zero mode, and adds a mass-zero update to `m0`.
The exact input is concatenated at time zero. This preserves total mass up to
floating-point roundoff without density renormalization.

For `G=32`, all 17 real-FFT bins are present: 15 complex nonzero frequencies
and the real Nyquist component give **31 real mass-zero directions**. There
are 32 independent physical face increments. Positive interior face amplitudes
give cell covariance rank 31. The uniform additive first-step Jacobian rank
test verifies this, rather than inferring field accuracy from four reported
Fourier observables.

For the exact Euler Green kernel, one Picard block recovers another correct
time-step prefix: a new proposal uses only previously correct origins. By
induction, `K` blocks recover the `K`-step stopped Euler states in exact
arithmetic. This statement concerns states and proposals up to absorption;
raw outputs after absorption are not promised to be physical Euler proposals.
It needs no global Picard contraction theorem. Useful accuracy at depth four
instead requires measurement; a learned kernel introduces an additional
approximation error.

A development-only probe replaced the learned kernel with the exact Euler
kernel on 64 fresh paths for each DK case. At `G=32, N=2048, T=1, dt=2^-7`,
depth-four normalized strong errors were approximately `1.01e-5`, `9.03e-6`,
and `7.80e-5` for uniform, nonuniform, and interacting cases. These numbers
isolate a Picard-depth floor on that development pool. They are **not trained
NN results**, stopping guarantees, or a replacement for the locked scientific
evaluation. The development record is
`/private/tmp/path-depth-development.json` and is not a permanent experiment
archive.

## Itô residual and stopping

`ito_residual` evaluates native coefficients at each final predicted path's
left endpoint. Before absorption its proposal residual is

\[
r_j=z_{j+1}^{\mathrm{proposal}}-\widehat m_j
-\Delta b_h(\widehat m_j)
-N^{-1/2}S_h(\widehat m_j)\Delta W_j.
\]

The cumulative residual sums these local residuals. The objective uses both
their full-cell energies, with fixed uniform-histogram fluctuation units and
a fixed allocated path/time denominator. Because the mass direction vanishes,
the cell Euclidean norm includes exactly the complete tangent-space norm.
There is no projection onto only selected statistical queries.

The crossing proposal contributes its complete uncensored residual. It is not
replaced by the frozen boundary state or by a fractional Brownian increment.
Later absorbed origins have zero exposure. Every path remains in the outer
average. At zero residual the active proposals obey the specified Euler
equation until its stopping rule; a merely small training residual is not an
accuracy certificate. In particular, artificial early absorption can alter
the available exposure, so held-out path errors and stopping diagnostics are
essential.

The same supplied `Delta W_j` is used on both sides. Independent minibatch
shuffling is computational randomness and does not add another SPDE forcing.
There is no differentiated smooth approximation of Brownian motion and no
midpoint stochastic integral. Network weights are fixed during inference, so
an SDE for evolving neural parameters and its Itô curvature correction are
not part of this implementation.

DK stopping is the first crossing of the Euler polygon at mass `0.01/G`,
followed by freezing. Intermediate Picard states are stopped before square
roots are evaluated. An initially absorbed zero histogram remains unchanged.
An arbitrary safe origin is evaluated only where its coefficient will be
multiplied by zero; this avoids masked `sqrt'(0)` in backpropagation and does
not repair a physical state. Segment-fraction bounds are intersection
geometry, not density clipping. Heat and additive controls remain unstopped.

Continuous boundary hitting between fine Euler steps remains a separate bias.
The reported strong supremum uses the declared stored grid; fine reference
excursions between those observation times are outside that metric. A
coarse/fine difference alone is not a rigorous bound on the continuous target.

## High-frequency Euler bias is distinct from neural instability

This section is an **analytic unstopped additive-noise control**, not a
nonlinear DK result or an additional scientific test ensemble. It needs no
trained checkpoint or final input bank.

In a real orthonormal Fourier direction with decay rate `lambda_k`, the
uniform-mobility additive control is an Ornstein--Uhlenbeck equation. Its
continuous stationary mass variance is `V0 = 1/(G*N)`. With
`beta_k = lambda_k*Delta`, explicit Euler instead has stationary variance

\[
V_{E,k}=\frac{V_0}{1-\beta_k/2}.
\]

At `G=32`, `D=1`, and `Delta=2^-7`, the Nyquist rate is
`103.75289205` and `beta_16 = 0.8105694691`. Its stable Euler multiplier is
`1-beta_16 = 0.1894305309`, but its stationary variance is **1.681476932 times**
the continuous variance: a **68.15% inflation**, or a standard-deviation
factor of `1.296717753`. Linear stability does not imply small statistical
discretization error. By comparison, the variance inflations of modes 1 and 2
are approximately `0.391%` and `1.567%`. A low-mode statistics check can miss
the high-frequency full-field error.

The common-driver error can also be computed without sampling. Compare
coarse Euler with fine Euler at `delta=Delta/r`, summing the fine Brownian
increments into the coarse driver. For one mode define

\[
a=1-\beta,\quad q=(1-\beta/r)^r,\quad
v_E=\frac1{1-\beta/2},\quad
v_F=\frac1{1-\beta/(2r)},\quad
c=\frac{2(1-q)}{1-aq}.
\]

Here `c*V0` is their stationary cross covariance under the shared physical
Brownian path. From a deterministic uniform initial field, the variance of
their difference at coarse time index `n`, divided by `V0`, is

\[
d_n=v_E(1-a^{2n})+v_F(1-q^{2n})
     -2c\bigl(1-(aq)^n\bigr).
\]

For `r=64` (`delta=2^-13`) and `T=1/16` (`n=8`), summing this exact expression
over all 31 real nonconstant directions gives the normalized **terminal**
full-field RMS error

\[
\left[\frac1{31}\sum_{k=1}^{31}d_{8,k}\right]^{1/2}
=0.295649301.
\]

The stationary single-Nyquist shared-driver error against the continuous OU
mode is approximately `0.505552372` in that mode's continuous fluctuation
units. The evaluation's pathwise maximum additionally examines every stored
time on each realization; it is not the terminal RMS statistic above.
Consequently a coarse/fine path-supremum error above 35% need not indicate a
neural instability. If the neural operator closely reproduces coarse Euler,
it inherits coarse Euler's time-discretization discrepancy. The three
comparisons in the workflow are therefore essential: neural versus same-step
Euler, same-step Euler versus fine Euler, and neural versus fine Euler.
Neither this linear calculation nor a same-step neural pass bounds nonlinear
DK or continuous stopping bias.

## Dispatch, independent inputs, and final locks

The `path-solver` CLI dispatches to the path workflow; a `path-*` preset cannot
silently run the legacy particle or moment workflow. Target IDs include the
path-law identifier, problem, native grid/stopping specification, declared
context, and discrete step. Source, configuration, and design changes require
a new run directory.

Input banks are addressed by configuration, role, case, and trajectory index.
Initial histogram sampling, physical face noise, and optimizer shuffling use
separate RNG namespaces. Training and validation family IDs must be disjoint.
Their banks contain initial histograms and noise, not future density labels.
Control development checks may simulate validation Euler trajectories; these
are development diagnostics, not locked final inputs.

The workflow trains controls first and verifies their independent validation
gate before proceeding. All declared fits must be complete and restored with
compatible provenance before the final inventory is byte-locked. The final
bank guard verifies this inventory and the checkpoint/metadata SHA-256 values
before it generates or reads final inputs. Fine reference drivers are summed
to construct the literal shared coarse increments. Completed bank rows,
predictions, caches, and final artifacts have content checks.

These controls protect the stated experimental separation. They do not turn
validation selection into a convergence proof. Reported bootstrap intervals
are paired, whole-trajectory, **pointwise** intervals, not simultaneous bands;
zero observed exits do not establish zero stopping probability.

### Parallel fitting orchestration

The later execution helper `analysis/fit_path_seed_batch.py` calls the same
frozen `fit_path` implementation in separate CPU processes, one owner per
declared optimizer seed. Its reviewed SHA-256 is
`35f436f156729e1c3f040824d52945e098e704784a9e45e4b603777bb5facd47`.
This analysis helper is outside the package `source_hash()` and must be
archived separately with execution provenance.

The helper does not change the declared data, optimizer allocation,
validation selection, or model source. Per-fit seeds, process-local Torch RNGs,
dedicated shuffle RNGs, and saved optimizer/RNG state preserve the numerical
procedure when an interrupted fit resumes. Read-only HDF5 access to completed
training and validation banks is shared; checkpoint and metadata paths are
distinct by seed. The helper does not request final inputs or mutate the
manifest/design. A separate small software experiment, described below,
compared actual sequential and parallel reruns for bitwise equality.

Safe execution requires the following operational conditions:

- Stop the sequential CLI before launching workers and resume it only after
  every worker has exited. The per-seed `O_EXCL` lock prevents duplicate worker
  ownership, but the sequential CLI does not consult these locks. A lock file
  or execution JSON alone is not proof of a live or terminated process.
- Complete all control development caches first. Workers read those existing
  caches; if a cache were absent, multiple workers could attempt the same
  temporary output path. Stale worker locks after an uncatchable termination
  require checking actual process ownership before recovery.
- Preserve execution records before restarting a worker. The helper's
  `seed_worker_N.json` destination is reused and otherwise overwrites the
  previous invocation's timing history.
- Account for the parallel segment explicitly. Workers leave
  `manifest.elapsed_seconds` unchanged. Active project wall time adds the
  **union** of worker intervals to the sequential intervals, rather than
  summing overlapping worker durations. Remaining invocation budgets must use
  that corrected accounting. Per-checkpoint `training_seconds` is fit wall
  time under concurrent CPU contention; it is neither total project wall time
  nor process CPU time. Worker process CPU fields cover only those workers.

These are orchestration and timing qualifications, not a change to the
stochastic ansatz. The original CLI remains responsible for validating the
complete checkpoint inventory, locking every fit, and unlocking final inputs
after the workers have finished.

### Actual sequential/worker software comparison

The [archived software result](experiments/artifacts/path_worker_repro_2026-10-25/README.md)
compares sequential fitting with two actual worker subprocesses on the same
small configuration and physical input arrays: `G=8`, `N=2048`, `T=1/16`,
step `1/128`, width eight, depth two, 12 training and eight validation paths,
all five cases, and optimizer seeds zero and one. Both copies fitted and passed
all control validation gates before worker fitting. Worker zero resumed an
incomplete checkpoint after **seven AdamW updates**.

All **10 model pairs** had bitwise identical current and selected weights,
weight hashes, AdamW/LBFGS states, Torch and shuffle RNG states, selected steps,
validation diagnostics, provenance, and complete training/validation histories
after excluding timing. Each subprocess's observed seed-lock PID matched its
actual child-process handle; both exited successfully. Pre-existing common
files remained unchanged, and no final input bank or fit lock was created.

The script is `analysis/audit_parallel_path_fitting.py`, SHA-256
`5f3a2a5fc922495726ff652c4166a6fdea165a477193052dd641a23122619ebd`.
The result JSON SHA-256 is
`d89a15b196c4322ae3c5853f3ca4bd9b29e0939d382a742fb3a50cd1250c8548`.
The archive retains its configuration, worker logs, and checksums. Local raw
software checkpoints and banks remain under
`runs/path-worker-repro-2026-10-25-v2`; they are not duplicated in the archive.

This establishes reproducibility for that software experiment with one
BLAS/Torch thread per process on the recorded machine and software versions.
It does not establish arbitrary-hardware bitwise reproducibility, validate
scientific `G=32, T=1` accuracy, or measure parallel speedup. No production run
file or locked scientific final input was used by this comparison.

## Source anchors and documentation qualifications

Individual SHA-256 values at the audited source commit:

```text
src/nn_dk/path_config.py
bd85da962dd435b48d6cd53d7bcc47d5ba3a7f12582f83220e857d4482b093ec
src/nn_dk/path_data.py
13813247f453e6a240929a428cce3d1816c596d2b9f87754f374b8f7e966845e
src/nn_dk/path_evaluation.py
1a915b4a55c996c630ac20a6486a2db938e2ed41047bba2f78db7c80b3183f65
src/nn_dk/path_model.py
7efbe4aa059e364ec9247b2a6c44ddc9ac2b56656ef53857fb2f2476bde01a4b
src/nn_dk/path_numerics.py
7cc7ea95bc3cf317db4c676210a8d376e0af6ab8813d2122cc8d9dd27460b48b
src/nn_dk/path_reporting.py
9258f0ae6a9e2a762d74e1682bdaddee8c31b43a4c64bc90663459db9d547534
src/nn_dk/path_study.py
c379a9058638a0b65ec6a9eebbab80436fe94fe14d1561f84e96c00306d2032b
src/nn_dk/path_training.py
8d05d81e94f4c0e582f355a31ff3308dc210d34fc97bcae3c3cf476c210a20bd
src/nn_dk/cli.py
4553985da32ebebd4e3a49b872c62715a8a8f4aa18df16db1fa24ba3de7acb92
```

Relevant executable evidence is in `tests/test_path_model.py`,
`tests/test_path_numerics.py`, `tests/test_path_training.py`,
`tests/test_path_data.py`, and `tests/test_path_workflow.py`. It covers
future-noise values and gradients, full noise rank, exact-kernel convolution,
actual held-out learning, boundary interpolation and uncensored proposals,
initial absorption, checkpoint continuation, data mutation, and final locks.
The earlier fit-grid nesting gap was corrected and has a configuration test.

The companion [method document](PATH_SOLVER.md) includes the two precise
qualifications identified by this audit: Euler-pole **admissibility is not
successful neural fitting**, and Picard forcing includes the active-at-start
mask used by the code. Its exact-prefix statement applies to stopped states
and active proposals. Its distinctions between discrete path simulation,
statistical prediction, and unproved continuous/continuum claims agree with
the inspected implementation. This audit does not reverify the method
document's external literature survey.
