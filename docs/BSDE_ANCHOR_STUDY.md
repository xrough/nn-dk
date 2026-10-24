# Initial-state coverage and structured values

This follow-up tests the conditional-generalization failures in the [first BSDE
batch](experiments/bsde_first_batch_2026-10-24.md). It crosses initial-state
coverage with value-network structure while keeping the number of training
trajectories fixed. Previous results are contextual evidence; every baseline is
refitted and evaluated on a fresh common test bank.

```bash
.venv/bin/nn-dk bsde-study --preset bsde-study-smoke --output runs/bsde-study-smoke
.venv/bin/nn-dk bsde-study --preset bsde-anchor-study --output runs/bsde-anchor-study
```

TOML equivalents are `configs/bsde_study_smoke.toml` and
`configs/bsde_anchor_study.toml`. Repeating a command resumes its saved work;
source and configuration hashes must match. The scientific invocation has an
eight-hour budget. Stage commands use `--stage calibrate`, `generate`, `fit`,
`evaluate`, or `report`. An exit code of 75 denotes saved progress at the budget
boundary.

## Fixed design

| Factor | Prespecified values |
|---|---|
| Training anchors × independent continuations | 128 × 16; 512 × 4; 2048 × 1 |
| Total training paths per fit | 2048 in every allocation |
| Value architecture | Existing MLP; structured mode-wise model |
| Integrand architecture | Existing three-layer width-64 MLP in both cases |
| Validation | 256 different anchors × 4 continuations, common to every arm |
| Optimizer seeds | 0, 1, 2, kept separate |
| Optimizer / width | Existing AdamW settings; width 64; at most 100 epochs; patience 10 |
| Physical target | Noninteracting native stopped DK; G=32, N=2048, D=1, T=1 |
| Initial laws | Uniform and density proportional to exp(0.2 cos x) |
| Forward / reference steps | 2^-7 / 2^-13; reference audits at 2^-14 and 2^-15 |
| Queries | Exact cell-integrated sine/cosine modes 1, 2 at 1/4, 1/2, 1 |
| New test seed | 20261025; prior inspected bank used 20261024 |
| OU analytic test | 1024 fresh initial states, exact Euler and continuous moments |
| Conditional DK test | 32 fresh anchors per profile × 512 continuations |
| Ensemble DK test | 8192 fresh initial histograms per profile |
| Bootstrap | 200 paired whole-trajectory resamples |

The factorial has **36 fits**: two forward problems, three allocations, two value
architectures and three seeds. A fit's 2048 paths contain respectively 128, 512 or
2048 independent initial states. This is a coverage-versus-replication experiment,
not a training-sample-size convergence curve. The 5120-path training union and
1024 validation paths are generated once per problem. Allocation views share
exact initial states and Brownian increments wherever their families overlap.

The validation distribution, maximum epoch count and early-stopping rule are
identical across arms. For a matched optimizer seed the integrand starts with
bitwise-identical parameters across architectures. Normalization uses each
view's unique training anchors, never validation or test outcomes.

`preregistration.json` saves the design before generation. All 36 checkpoints
must complete and their hashes must be locked before any final test evaluation.
Changing a fit after this lock is rejected. The old pilot's final bank is not
reused for selecting the new architecture.

## Structure being tested

The OU value head has four learned positive scalars: two decay rates and two
variance scales, shared within the cosine/sine pairs. It predicts

\[
\mu_k(T)=e^{-r_kT}y_{0,k},\qquad
\Sigma_{kk}(T)=v_k(1-e^{-2r_kT}).
\]

The parameters start at one and are fitted from the same trajectory objective
as the MLP. Analytic OU rates are not supplied as forecasts. This family contains
the stable Euler OU moments, so the analytic benchmark can distinguish an
architecture limitation from learning or martingale-control errors.

For DK, this learned mode-wise propagation is a backbone. A three-layer residual
network retains every mass-zero initial-state coordinate. Its mean and Cholesky
factor corrections have a fixed scale of 0.1 and start at zero. Conditional
covariance remains state-dependent and positive semidefinite. Initially stopped
states and time zero return their exact deterministic statistics. No additive
covariance floor is introduced.

The full-state residual matters: stopping generally destroys exact low-mode
closure. The backbone supplies an inductive bias rather than an exact formula
for the stopped dynamics. The integrand architecture, terminal objective,
physical forward process and Brownian bookkeeping remain unchanged. Forward
noise and the BSDE martingale use the same full increments with predictable
active-at-start masks, including the crossing step.

## Measurements and limits

The primary comparisons use the training-step Euler law, separating neural
error from temporal bias. Reports also retain total errors against fine DK or
continuous OU references. They show conditional means and variances first,
ensemble averages separately, full covariance and maximum pointwise errors,
stopping, and cost. Percentages use the fixed fluctuation normalization from the
first study; they are not relative density errors.

Prespecified paired contrasts compare structured minus MLP error at each
allocation, and 2048-anchor minus 128-anchor error within each architecture.
Negative RMSE differences indicate improvement. Conditional bootstraps resample
independent continuations within each fixed test anchor, preserving all times
and modes. Ensemble bootstraps resample matched initial-state/forcing families
and include the law of total covariance. Optimizer seeds remain separate.

The new conditional bank broadens initial-state coverage at a fixed reference
path budget. Its 512 continuations per anchor may leave small pointwise errors
unresolved. Paired improvements, low average RMSE and a successful pipeline do
not certify the 5% pointwise target. Sampling budgets and remaining fine-reference
bias are assessed separately. No learning-convergence rate, continuum SPDE
convergence, or accuracy-matched speed advantage follows from this fixed-grid
study alone.

Costs distinguish shared pool/reference generation, fitting, checkpoint loading,
initial histograms, value-only prediction, and extra prediction caches used for
paired comparisons. Shared costs repeated in model records must not be summed.
Results and plots are saved under `report/`; raw paths and checkpoints remain
under the run directory.
