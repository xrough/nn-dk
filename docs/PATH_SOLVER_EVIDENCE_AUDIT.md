# Independent density-path evidence audit

Audited locally on **25 October 2026 (Asia/Shanghai)** against frozen package
source at commit **`bd256de`**. This audit restores actual saved networks and
recomputes trajectories; it does not infer correctness from a green manifest or
from the earlier neural moment results. Package, configuration, checkpoint and
input-bank files were not edited during the audit.

The package source hash is
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.
Documentation and the independent timing script lie outside the package-source
hash and do not change the running fits.

## Scope and acceptance meaning

The implemented network maps initial finite-volume masses and supplied physical
face Brownian increments to **all 32 cell masses at every numerical time**. Its
weights remain fixed during a realization. Its real FFT representation retains
Nyquist; this is a density-path operator, rather than a four-observable moment
map. It uses known drift and mobility inside a learned causal Volterra/Picard
construction, with separate fits for different numerical steps.

For a shared initial state and physical forcing, the primary empirical strong
metric is

\[
\left[\frac1P\sum_{p=1}^{P}\max_{n}
 \frac{N\sum_{j=1}^{G}|\widehat m^{(p)}_{n,j}-m^{(p)}_{n,j}|^2}
 {1-1/G}\right]^{1/2}.
\]

This uses the exact uniform-IID histogram density-fluctuation energy
`(1−1/G)/(N h)` as its fixed normalization. The maximum is taken before the
trajectory average and includes every stored numerical time. Weak checks also
retain all cells and full same-time/two-time covariance, together with the
cell-integrated sine/cosine modes 1 and 2. All stopped paths remain in statistics.

The declared 5% gates concern **the same-step Euler numerical target**. Paired
whole-trajectory bootstrap intervals quantify prediction-minus-reference
contrasts. They are pointwise intervals, not simultaneous confidence bands or
certificates of each marginal-law Monte Carlo precision. A fine-Euler difference
is reported separately and is not a continuous-time reference-bias enclosure.

## Short five-case smoke run

Run: [`runs/path-smoke-2026-10-25-v2/`](../runs/path-smoke-2026-10-25-v2/).
It uses `G=32, N=2048, D=1, T=1/16`, step `2^-7`, diagnostic reference step
`2^-10`, one optimizer seed, and 64 test paths per case. The cases are heat,
additive noise, uniform DK, nonuniform DK, and interacting nonuniform DK.

Provenance anchors:

| Item | SHA-256 / declared hash |
|---|---|
| Configuration semantic hash | `ac392518dcf71e7478021b060da4cf8e45cb37b6f71f0c401583e64995f04276` |
| Preregistration semantic hash | `529affde0a6e5ecbbefa25f2096e7ed9e80c06db883e853d98fc88afee9c8ec5` |
| Artifact-lock semantic hash | `28256fc9090cc5b7de2c6afa0f18302206a9edaaa017de31b99bc58375c24cf3` |
| Manifest snapshot bytes at audit | `b6d7368a3ed4680ae74a54fe2bc1f45cddab390aca1087c91310a2d31f079e2f` |
| `path_test.h5` bytes | `cd4d9d034bfd024c1995ddc48f5bfd0e8eca0cafe32258959a46a671b84f2202` |
| `evaluations.json` bytes | `4991bece2ffed04d0a384cf91c100cbd840c1ee5e6f16d27623e2c629aa7ec37` |

All **29 artifact-lock entries** matched the actual files. All five checkpoint
and prediction hashes also matched their final records. The manifest snapshot
hash identifies the audited manifest bytes; a later completed-run replay can
update elapsed-time metadata while retaining the same locked scientific files.

The audit checked the following against actual arrays:

- All five checkpoints restored successfully under the current package hash.
  Their selected weights reproduced **every saved raw proposal, stopped state
  and stopping time byte for byte**, with shape `(64, 9, 32)` per case.
- NumPy stencil recurrences independently reproduced the heat/additive Euler
  reference. The existing native NumPy DK backend reproduced the Torch DK Euler
  reference. The largest cell-mass discrepancy was **`2.78e-17`**.
- Direct recomputation of the strong metric agreed within `2e-13`. Direct
  `N*np.cov(...)` recomputation of the complete mode/time joint covariance agreed
  within `1e-13` in its unnormalized entries.
- All 64 paths in each case passed future-noise perturbations at cuts
  **0, 1, 4 and 7**, with unchanged earlier states. Repeated inference with the
  same initial masses and physical increments was exact. These are regression
  checks; the lower-triangular architecture supplies the causal dependency
  structure at mesh points.
- Replacing the supplied Brownian increments with zeros changed every
  stochastic model's trajectories. The deterministic heat model correctly
  ignored this input. This confirms that the saved output is a forcing-dependent
  path, but is not by itself a distributional-accuracy proof.
- Train, validation and test contained **160, 80 and 320 distinct trajectory
  families**, respectively, with no overlap. These counts sum across the five
  cases. The independently addressed initial-state and physical-noise RNG
  namespaces were inspected and reconstructed.
- Regenerating all fine Brownian streams from their recorded configuration
  reproduced the stored coarse increments and fine-reference snapshots exactly.
  Fine empirical Brownian variance divided by its specified step ranged from
  **0.9963 to 1.0015** across cases; this is descriptive sampling evidence.
- Total mass error was at most **`2.22e-16`**. No path failed or stopped in this
  small batch. Minimum masses were well above the positive stopping threshold.

Rounded empirical strong errors, in the fixed units above:

| Case | NN / same-step Euler | NN / fine Euler |
|---|---:|---:|
| Heat | 0.0791% | 12.5447% |
| Additive | 0.2061% | 35.5867% |
| Uniform DK | 0.1829% | 35.5483% |
| Nonuniform DK | 0.2465% | 35.9483% |
| Interacting DK | 0.2691% | 35.6961% |

All five same-step gates pass. The stochastic fine-reference differences are
already about **35%**, so this result must not be presented as 5% accuracy for
the continuous-time stopped DK system. Here temporal discretization dominates
the neural approximation error.

Checkpoint byte hashes:

| Case | Checkpoint SHA-256 |
|---|---|
| Heat | `883862151cb7a8cfbae0878724fda9c896b5121be1bb4aa2a56ee34fcb3ff899` |
| Additive | `1402d3e8d665e6afb8f673e66de380db1bc893c0515c93f3fb721b7aacbc62ae` |
| Uniform DK | `6a33cb87fc807e13812cb26b639d690ac460fbd49a438eb4fc17e2339ebbb0cb` |
| Nonuniform DK | `efdd3de031c248b5f2f801cb748ec997bd6aa5eb96d27e39f02ec70afc66b4d5` |
| Interacting DK | `f585d528e3360dd292b56667f7d02236571c75ddb38c067066af77026f388ffd` |

## Realized quadratic variation

The smoke run has only **512 active innovations** per stochastic case
(`64 paths × 8 steps`). Its empirical full-matrix QV differs from the expected
integrated covariance by 17.7–22.5% in relative Frobenius norm. This should not
be mistaken for a neural noise-amplitude error:

| Case | NN QV / expected covariance gap | Exact Euler QV / expected covariance gap | NN / exact-Euler realized-QV gap |
|---|---:|---:|---:|
| Additive | 22.5336% | 22.5351% | 0.0267% |
| Uniform DK | 19.9633% | 19.9664% | 0.0271% |
| Nonuniform DK | 20.0061% | 20.0034% | 0.0286% |
| Interacting DK | 17.7006% | 17.6961% | 0.0341% |

The exact-Euler rows were independently recomputed using the **same** physical
increments. All uncensored active-at-start proposals, including exit proposals,
are retained in this diagnostic. No fractional stopped increment is treated as
an ordinary full-step Gaussian proposal.

For the additive uniform-face control, `C = B diag(σ²) Bᵀ` has effective rank
`(tr C)²/tr(C²) = 2G/3`. With `n` independent Gaussian innovations, the relative
RMS Frobenius sampling error of their second-moment estimator is
`sqrt((2G/3 + 1)/n)`, approximately **20.9%** here. This formula is an additive
constant-covariance calculation, not a confidence interval for the nonlinear DK
rows. It explains why a visible empirical QV/expectation gap can coexist with
very close NN/Euler QV. A precise expected-covariance check needs more sampling
and an explicit uncertainty calculation.

## Short three-step refinement workflow

Run: [`runs/path-refinement-smoke-2026-10-25/`](../runs/path-refinement-smoke-2026-10-25/),
using [`configs/path_refinement_smoke.toml`](../configs/path_refinement_smoke.toml).
It keeps `G=32, N=2048, T=1/16`, compares steps `2^-7, 2^-8, 2^-9` against
`δ=2^-10`, and uses seed `2026102504`, 64 test paths per case and a separate fit
at each step.

| Anchor | Hash |
|---|---|
| Configuration semantic hash | `00fa885ecbb10ca1e7d8483db01a722d1e6d78030f21524c669335ecd148ba08` |
| Artifact-lock semantic hash | `d864e075dce6f11e0da4da0718c2cf9da530e35142f3439af0178a17c7470f06` |
| Manifest snapshot bytes at audit | `60dc786af33cd9703d1539f97082394c3783a3945663d8d9a24bdeb31ccc8ada` |

All **73 artifact hashes** and all **15 final model slots** were verified.
Checkpoint and metadata files preceded the recorded final-input unlock.
Restored weights reproduced every raw and stopped array exactly: shapes are
`(64,9,32)`, `(64,17,32)` and `(64,33,32)` for the three steps. Initial masses
and ordered trajectory families were identical across grids. Families remained
disjoint across training, validation and final roles.

The common Brownian base grid was checked literally: every coarser increment
array equals its declared block sum of the stored `2^-9` array byte for byte.
Reconstructing the original `2^-10` stream and summing it in a single differently
grouped reduction changes some coarse increments by at most **`1.11e-16`**,
ordinary floating-point nonassociativity. Thus the hierarchy shares physical
noise; it does not resample noise per fit. The fine-reference trajectories and
stopping times reconstructed from the original fine stream exactly matched all
stored snapshots.

All 64 trajectories at every model/grid passed future-noise perturbations at
the first, middle and last increment, plus exact replay. Native NumPy DK Euler
agreed with Torch Euler within `1e-15`. The full-state strong metric and complete
cell/time covariance were independently recomputed. No path failed or stopped.

NN/fine-Euler empirical strong differences:

| Case | Δ=`2^-7` | Δ=`2^-8` | Δ=`2^-9` |
|---|---:|---:|---:|
| Heat | 12.6280% | 4.0698% | 1.2507% |
| Additive | 35.8859% | 14.5899% | 5.5117% |
| Uniform DK | 36.1436% | 14.9689% | 5.6928% |
| Nonuniform DK | 35.9430% | 14.6928% | 5.7011% |
| Interacting DK | 36.5255% | 15.1464% | 5.5941% |

All **15 same-step gates pass**. These finite-reference gaps decline, but the
finest stochastic gap is still above 5%. The reference is only one further
binary refinement at the finest level, its bias is not enclosed, the networks
are refitted per step, and the maximum is measured on each grid's stored times.
**No temporal convergence order or continuous-time accuracy is inferred.**

## Limits still requiring experiments

- No observed stop among 64 paths per profile certifies neither a zero stopping
  probability nor rare boundary-event accuracy. Unit tests of the boundary rule
  and this interior experiment have different scopes.
- Grid-point causality does not make the interpolated numerical boundary time
  an exact stopping time for the continuous Brownian filtration. The declared
  Euler-polygon target and continuous stopped SDE remain distinct.
- A stored-time supremum does not measure excursions between those times.
- Independent whole-family bootstrap intervals do not certify simultaneous
  coverage over all covariance entries or optimizer seeds.
- These experiments use fixed `G,N` and known physics. They do not establish a
  continuum DK limit, robustness to stronger noise or new physics, or a neural
  speed advantage.
- The recorded workflow `evaluation_seconds` includes reference work, bootstrap,
  regression checks and HDF5 output. It is not NN inference latency.

## Independent timing protocol

[`analysis/benchmark_path_solver.py`](../analysis/benchmark_path_solver.py) times
complete warm trajectories with **validation inputs only**, at `P=1` and `P=64`.
It uses float64 CPU execution and one Torch/BLAS thread, randomized interleaving,
three warmups and 21 repetitions per method. It measures the fixed neural
operator plus its stopping wrapper, the native NumPy Euler implementation, and
Torch Euler as a diagnostic backend comparison. Model loading, input generation,
training, bootstrap and file I/O are excluded. Medians, ranges, 5th/95th
percentiles and throughput are retained for each seed.

Checkpoint restoration and IID-particle histogram initialization are measured
separately, outside inference. Initialization uses independent discarded
development draws with different fixed seeds per repetition. First measured
calls and warmed repetitions are distinguished; neither is advertised as an OS
cold-start benchmark because file, allocator and OS cache state is uncontrolled.

Its accuracy check is the **strong point error on those validation inputs**.
It is not a joint weak/uncertainty gate or matched continuous-time/marginal-law
certification. Clean timing should run after all training jobs finish to avoid
resource contention. No latency result is claimed here before that execution.

For the completed full-horizon pilot, the intended command is:

```bash
.venv/bin/python analysis/benchmark_path_solver.py \
  --run runs/path-first-batch-2026-10-25-v2 \
  --output runs/path-first-batch-2026-10-25-v2/benchmark/warm_validation.json
```

The full `T=1`, three-optimizer-seed pilot remains outside this audited evidence
until its fits are locked and final artifacts are available. Its final input
bank was not inspected while locked.

The full-horizon fitting stage subsequently moved from the sequential CLI to
three separate **one-thread processes**, one exclusive optimizer-seed writer per
process, before returning to the original workflow for locking and evaluation.
This preserves the fitting code and physical input banks but changes host cost
accounting. Per-fit `training_seconds` are overlapping wall times; their sum is
neither host elapsed time nor a direct CPU-time measurement. The manifest's
`elapsed_seconds` covers its recorded CLI invocations and does not itself include
standalone-worker wall intervals. Worker execution records provide start/finish,
wall and process-CPU measurements and need separate archival. Any final cost
statement must distinguish those quantities. Latency timing remains deferred
until fitting and evaluation processes are quiet.
