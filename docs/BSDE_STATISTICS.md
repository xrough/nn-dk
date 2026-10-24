# Conditional DK statistics through Deep BSDE

This workflow implements the first stage of a comparison between learning stochastic density paths and learning their statistics directly. It predicts conditional means and **same-time** four-by-four covariances. Its forward dynamics use known coefficients; the value network replaces repeated forecast simulation at inference.

The existing particle benchmark and native coefficient CNN remain separate workflows. This is a Deep BSDE terminal-matching formulation, following [E, Han and Jentzen](https://pmc.ncbi.nlm.nih.gov/articles/PMC6112690/); it is not a claim that the DK equation has an ordinary Deep Ritz energy.

## Run and resume

```bash
.venv/bin/nn-dk bsde --preset bsde-smoke --output runs/bsde-smoke
.venv/bin/nn-dk bsde --preset bsde-first-batch --output runs/bsde-first-batch
.venv/bin/nn-dk bsde --preset bsde-refinement --output runs/bsde-refinement
```

Equivalent TOML configurations are in `configs/bsde_smoke.toml`, `configs/bsde_first_batch.toml`, and `configs/bsde_refinement.toml`. The first-batch invocation has an eight-hour budget. A shorter budget can be set with `--budget-seconds`; exit code 75 means the current work is checkpointed and the identical command can resume it.

Stages can be executed independently:

```bash
.venv/bin/nn-dk bsde --preset bsde-smoke --stage calibrate --output runs/bsde-smoke
.venv/bin/nn-dk bsde --stage generate --output runs/bsde-smoke
.venv/bin/nn-dk bsde --stage fit --output runs/bsde-smoke
.venv/bin/nn-dk bsde --stage evaluate --output runs/bsde-smoke
.venv/bin/nn-dk bsde --stage report --output runs/bsde-smoke
```

Source and configuration hashes are immutable within a run. Use a new directory after changing numerical source, training settings, or allocations. Completed checkpoint replay preserves checkpoint bytes; a final report requires every configured model and evaluation. Numerical calibration must pass before generation or fitting.

## Targets and calibration

The scalar Brownian calibration uses \(dX=dW\), \(X_0=0\), and the payoff \((X_T,X_T^2)\). With the exact predictable first-order integrand, the quadratic residual is

\[
R=\sum_n[(\Delta W_n)^2-\Delta],\qquad \mathbb E[R^2]=2T\Delta.
\]

A nonzero terminal objective is therefore possible even with the correct initial value. The calibration checks the pathwise identity and its analytically known sampling error.

The neural analytic benchmark is four independent normalized OU coordinates with rates

\[
\lambda_k=4D h^{-2}\sin^2(\pi k/G),\qquad
dZ_k=-\lambda_k Z_k\,dt+\sqrt{2\lambda_k}\,dW_k,
\]

for cosine/sine modes 1 and 2. Matrix-exponential moments and an independent Euler covariance recursion check the closed forms. Neural predictions are compared against both the exact Euler law used in training and the continuous-time OU law. This auxiliary Gaussian process is explicitly unstopped and is not the DK target.

The DK target has \(G=32\), \(N=2048\), \(D=1\), and no interaction. Initial particles are IID uniform or have density proportional to \(e^{0.2\cos x}\), then are histogrammed into native cell masses. The density threshold is \(0.01/(2\pi)\). Every path freezes at the first crossing of its Euler polygon, and every frozen path remains in the statistics. No clipping or renormalization is performed.

Observations integrate \(\sqrt2\cos(kx),\sqrt2\sin(kx)\), \(k=1,2\), exactly against reconstructed cell densities. Pilot horizons are \(1/4,1/2,1\); time zero is exact. Other times and a joint covariance across times are outside this version.

The first fit uses forward step \(2^{-7}\). The refinement configuration uses \(2^{-8}\) and requires a separate fit. Fine DK references start at \(2^{-13}\), audited against \(2^{-14}\) and \(2^{-15}\) through summed Brownian increments. These finite differences are evidence about reference sensitivity, not bounds on remaining continuous-time bias.

## Learned objects and the terminal objective

Two independent float64 MLPs have three hidden layers of width 64 and SiLU activations. The value network receives the initial state and declared horizon. Native states are represented in an orthonormal mass-zero basis; normalization is fitted on unique training initial anchors. The integrand network receives only the pre-step state, remaining time, horizon, and a deterministic center known at initialization.

Let \(y_i=\sqrt N\phi_i/s_i\), where \(s_i\) are fixed uniform-IID histogram fluctuation scales. OU observables use identity coordinates and \(N=1\). For each initial state, define the known Euler heat-flow center \(c\). It is used only for second-moment training and integrand inputs; it is not a value-network feature or a supplied mean forecast.

The value head has a full learned mean and a Cholesky covariance:

\[
\mu_\theta=y(M_0)+T d_\theta,\qquad
\Sigma_\theta=T L_\theta L_\theta^\top.
\]

Positive diagonal factors use softplus, without an additive covariance floor. Initially absorbed states and time zero return the exact initial mean and zero conditional covariance.

Define the fourteen-dimensional terminal payoff and initial value by

\[
H=(y(M_T),\operatorname{svec}[(y(M_T)-c)(y(M_T)-c)^\top]),
\]
\[
V_\theta=(\mu_\theta,\operatorname{svec}[\Sigma_\theta+(\mu_\theta-c)(\mu_\theta-c)^\top]).
\]

Off-diagonal `svec` entries have weight \(\sqrt2\), preserving the Frobenius norm. The loss gives equal weight to the four first-moment entries and ten second-moment entries in

\[
H-V_\theta-\sum_n I_{n,\mathrm{active}}Z_\eta(t_n,M_n,T,c)\Delta W_n.
\]

All path/time integrand inputs are evaluated in vectorized batches. There is no network evaluation for each forward simulation step at inference. The known forward trajectories are data for training, so no differentiation through their numerical integrator or input Hessian is needed.

## Noise and causality

For this statistics solver, the forward state and BSDE martingale use the **same Brownian increments**. Introducing an independent Brownian motion only into this martingale sum would remove its useful correlation with the payoff. Adding another physical forcing would define a different equation.

The active mask refers to the start of the step. On the crossing step, the martingale uses the full increment. The crossing fraction depends on that increment and must not multiply it in the predictable stochastic sum. Later contributions are zero. Future states, raw proposals, crossing fractions and stopping times are not integrand features.

Initialization, independent physical continuations, optimization shuffling, validation and test sampling use distinct keyed RNG streams. HDF5 keeps learner traces separate from oracle proposals and stopping metadata. Independent continuations share their initial anchor; they do not count as independent initial states.

For a later pathwise SPDE solver, physical forcing and auxiliary Feynman–Kac Brownian motion may have separate, independent roles under a derived stochastic representation. [Backward doubly stochastic equations](https://www.i2m.univ-amu.fr/perso/etienne.pardoux/_media/epsp.pdf) provide one relevant framework. Applying such a construction to conservative multiplicative DK noise requires a separate regularization and causality specification; it is not implemented here.

## Prediction and ensemble statistics

```python
from nn_dk.bsde_training import load_bsde_model

model = load_bsde_model("runs/bsde-first-batch/checkpoints/dk/seed_0.pt")
conditional = model.predict_conditional(initial_masses, model.config.query_times)
ensemble = model.predict_ensemble(
    model.config.context("nonuniform"), model.config.queries,
    n_initial=8192, seed=123,
)
```

Conditional predictions contain physical means with shape `(initial_states, horizons, 4)` and \(N\)-scaled conditional covariances with shape `(initial_states, horizons, 4, 4)`. Ensemble predictions have the corresponding horizon-only shapes. Their target identifiers include the forward discretization.

Ensemble covariances use

\[
N\operatorname{Cov}(Y)
=\mathbb E[N\operatorname{Cov}(Y\mid M_0)]
+N\operatorname{Cov}(\mathbb E[Y\mid M_0]).
\]

Averaging conditional covariances alone would omit initial sampling fluctuations. Prediction needs no simulated future forcing and no deterministic-center evaluation. Initial histogram generation, complete value-network inference, covariance validation and checkpoint loading have separate cost accounting.

The conditional function can be evaluated on valid initial masses, but accuracy outside the trained initial-state laws is not certified. Ensemble prediction flags profiles outside the configured training family and rejects incompatible interactions, diffusion, particle counts, observation modes or horizons.

## Pilot allocation and interpretation

| Allocation | Default |
|---|---|
| Training | 128 independent initial anchors × 16 continuations |
| Validation | 32 different anchors × 32 continuations |
| DK profile balance | Half of training/validation anchors from each initial law |
| OU analytic test | 256 held-out initial states |
| DK conditional test | 8 anchors per profile × 2048 continuations |
| DK ensemble test | 8192 initial histograms per profile, matched to independent forward reference families |
| Coupled reference audit | 32 families per profile |
| Optimizer seeds | 0, 1, 2, reported individually |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-5 |
| Training | At most 100 epochs, patience 10, path batch 32, gradient norm clip 100 |
| Bootstrap | 200 whole-family resamples |

Forward prefixes at different horizons remain in the same split. Conditional intervals resample independent continuations within a fixed anchor; ensemble intervals resample whole initial-state/continuation families. Optimizer-seed spread is reported separately.

The scientific targets remain 5% normalized error and 1.25% for each uncertainty contribution. The conditional pilot allocation is diagnostic and may not resolve these budgets. Small losses, positive covariance or a completed pipeline do not certify accuracy.

Reports separate neural error against the same-step Euler law, total error against the fine reference, and coarse/fine time differences. Covariance residuals and stopping diagnostics remain available even when overall gates are unresolved. Costs include forward generation, first-epoch profiling, fitting, initialization, inference and reference work. Shared costs repeated across optimizer-seed rows must not be summed. A speed advantage requires matched, resolved accuracy and amortization over distinct initial-state queries.

Next stages add interactions, joint temporal statistics, then an independently specified pathwise SPDE solver. Spatial refinement is a resolution study until a continuum regularization/noise target is declared.
