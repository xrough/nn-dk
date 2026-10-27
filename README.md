# NN-DK

[![verify](https://github.com/xrough/nn-dk/actions/workflows/verify.yml/badge.svg)](https://github.com/xrough/nn-dk/actions/workflows/verify.yml)

Neural numerical experiments for **Dean–Kawasaki dynamics and coarse particle
systems**, by Xiaohao Ji. The project studies when neural solvers can reproduce
stochastic density paths or answer repeated statistical queries, using
conservative numerical references and explicit uncertainty diagnostics.

A causal density-path network maps an initial histogram and physical Brownian
increments to every cell mass along a realization. Separate statistics networks
predict selected conditional means and covariances. The code keeps their targets,
accuracy evidence, and timing scopes distinct.

Start with the [DK background guide](docs/DK_BACKGROUND.md), adapted from selected
chapters of my doctoral thesis, for the particle-to-field derivation and its
connection to these numerical models.

![An actual held-out full-grid density realization](docs/experiments/artifacts/path_first_batch_2026-10-25/density_realizations.png)

## Install and run

Requirements: **Python 3.12**, Git, and [uv](https://docs.astral.sh/uv/).
The committed `uv.lock` fixes the dependency environment. The smoke example runs
on CPU; a GPU is not required.

```bash
git clone https://github.com/xrough/nn-dk.git
cd nn-dk
uv sync --locked --group dev --python 3.12
uv run nn-dk path-solver --preset path-smoke --output runs/path-smoke
```

The smoke command calibrates the numerical setup, generates inputs, fits five
small control/DK models, evaluates them, and writes `runs/path-smoke/report/`
with Markdown/JSON results and a CSV table. It is an executable check, not the
full scientific study. Training time depends on the machine; use
`--budget-seconds 600` to limit each invocation. Exit **75** means work has been
checkpointed: repeat the identical command to resume.

For checks and additional configurations:

```bash
uv run ruff check .
uv run pytest
uv run nn-dk path-solver --config configs/path_refinement_smoke.toml \
  --output runs/path-refinement-smoke
```

Use a full Git clone. Frozen-model reproduction restores historical source
commits; a shallow checkout is insufficient. Existing environments can run
`.venv/bin/nn-dk`, `.venv/bin/pytest`, and `.venv/bin/ruff` directly. Source and
configuration hashes guard saved outputs, so choose a new output directory after
changing numerical source or configuration. See [contributing](CONTRIBUTING.md)
and the [verification guide](docs/verification.md).

## What is implemented?

| Workflow | Output and purpose | Guide |
|---|---|---|
| `path-solver` | Full stochastic finite-volume cell-mass paths from initial states and physical noise; learned causal Volterra/Picard response with known interaction and mobility | [Path solver](docs/PATH_SOLVER.md) |
| `interacting-moments` | Selected means and full same-time covariance, conditioned on the complete interacting state | [Interacting solvers](docs/INTERACTING_SOLVERS.md) |
| `generator` | Structured generator-residual moment predictions for the declared observable bank | [Generator moments](docs/GENERATOR_MOMENTS.md) |
| `bsde`, `bsde-study` | Conditional value/statistics maps and initial-state coverage studies | [BSDE statistics](docs/BSDE_STATISTICS.md) |
| `convergence` | Autonomous native-DK coefficient learning, ablations, and refinement experiments | [Coefficient convergence](docs/PROJECT_A_CONVERGENCE.md) |
| `run` | Original particle/coarse-dynamics A/B benchmark | [Numerical methods](docs/methods.md) |

Every workflow has its own target identifiers, configuration, checkpoints, and
reference conventions. Their fitted weights and accuracy claims are not
interchangeable. The [analysis guide](analysis/README.md) describes postprocessing,
timing and the completed independent generalization study. Reproducing its
historical models requires the full local study inputs.

## Current evidence

The [statistics formulation repair](docs/experiments/payoff_repair_2026-10-27.md)
uses known heat moments, separate mean/covariance networks and staged training
to resolve the earlier integral-variance collapse. Ten fresh fits and another
98,304 independent reference families retain every ablation and optimizer seed.
Repaired worst mean/covariance point discrepancies are about **2.4% / 3.3%**;
sampling intervals still leave the blanket 5% criterion unresolved.

One fixed-bank conditional query takes **0.586–0.683 ms**, after **230–276 s**
of fitting per model, versus **1.83–2.00 ms** for the no-training Gaussian
approximation. Initial-ensemble NN integration takes **265–289 ms**, versus
**1.87–2.04 ms** for the Gaussian approximation. Costs refer to four means and
same-time 4×4 covariances at three horizons. The
[repair archive](docs/experiments/artifacts/payoff_repair_2026-10-27/README.md)
contains the controlled diagnosis, all seeds, cold/warm costs and uncertainty
accounting. The [completion checks](docs/experiments/payoff_repair_2026-10-27_completion.json)
record the final tests and archive verification. No accuracy-qualified training-cost
crossover is certified.

The earlier [independent broader-task comparison](docs/experiments/path_task_advantages_2026-10-27.md)
is complete: 98,304 independent reference families, six initial-state groups,
174 main evaluations, twelve literal Gaussian checks and twelve newly fitted
statistics heads, retaining all optimizer seeds. It compares full paths, new
Fourier queries, time-integrated functionals, moments and sensitivities, with
fitting, preparation and inference costs recorded separately.

**Neither neural method establishes a general advantage in this comparison.**
Compiled native Euler is faster for complete paths and derivatives. The
linear-Gaussian approximation is the inexpensive statistical control in this
short, weak-noise setting. The original statistics heads have fast conditional inference
but miss the accuracy targets, particularly for integrated observables. The
earlier fixed-bank success below therefore remains specific to that bank.
The [compact evidence archive](docs/experiments/artifacts/path_task_advantages_2026-10-27/README.md)
retains the complete tables, figures, training metadata and audits. No
accuracy-qualified neural training-cost crossover is established.

The [completed interacting follow-up](docs/experiments/interacting_followup_2026-10-25.md)
covers **24 fits**, uniform/nonuniform initial laws, and three time steps at
G=32, N=2048, D=1, T=1/16. Full-field path errors against a coupled fine Euler
reference, measured on a common time grid, decrease as follows:

| Fitted step | NN / fine-Euler strong RMS |
|---|---:|
| 2^-8 | 16.51–16.73% |
| 2^-9 | 7.719–7.734% |
| 2^-10 | 3.597–3.647% |

All 18 same-step Euler gates passed. The unscaled statistics model's conditional
mean and covariance point errors were **1.22–2.42%** and **0.73–2.58%**, with
pointwise interval endpoints below 5% relative to the finite Euler reference.
The report defines the fixed fluctuation units used by these percentages;
they are not relative errors of possibly zero means.

Warm conditional statistics queries took **0.36–0.76 ms** in that experiment.
Ensemble initialization largely removed the prepared-input cost advantage, and
the path network was slower than native Euler for the matched moment tasks.
These measurements support a specific conditional-statistics use case, not a
general neural speedup claim. Continuous-time accuracy and the stricter reference
uncertainty budget remain unresolved.

The separate **T=1, three-seed first path batch is complete**: all 15 configured
fits and same-step accuracy gates passed. Its largest same-step strong RMS was
0.1104%, while stochastic NN/fine-Euler errors remained about 44–45% at the
selected coarse step. The [compact archive](docs/experiments/artifacts/path_first_batch_2026-10-25/README.md)
contains the complete per-seed report, figures, timing scopes, checkpoint metadata,
and checksums. This complements the earlier
[short-run path report](docs/experiments/path_solver_2026-10-25.md),
[smoke archive](docs/experiments/artifacts/path_smoke_2026-10-25/README.md), and
[refinement archive](docs/experiments/artifacts/path_refinement_smoke_2026-10-25/README.md).

Earlier tracks retain their own source freezes and results:

- [Structured generator-moment study](docs/experiments/generator_accuracy_2026-10-25.md): 15 declared statistical comparisons passed for selected moments.
- [BSDE coverage study](docs/experiments/bsde_anchor_study_2026-10-24.md): 36 fits; broader initial-state coverage helped, but 5% accuracy remained unresolved.
- [Autonomous coefficient study](docs/experiments/a_first_batch_2026-10-24.md) and [first BSDE batch](docs/experiments/bsde_first_batch_2026-10-24.md): separate numerical targets and diagnostics.

## Numerical interpretation and reproducibility

The native DK model is a **stopped finite-volume Itô system**. Euler segments
freeze at their first crossing of density `0.01/(2*pi)`. States are never clipped
or renormalized; crossing proposals remain uncensored, and stopped paths remain
in the statistics. Training uses initial histograms and the physical Brownian
inputs, with optimizer randomness kept separate.

Agreement with same-step Euler measures neural approximation of that numerical
target. A finer Euler comparison measures a separate finite-reference difference.
Neither establishes continuous-time stopped-DK convergence, a continuum SPDE
limit, or rare-event accuracy. The
[operator audit](docs/PATH_OPERATOR_AUDIT.md) and
[evidence audit](docs/PATH_SOLVER_EVIDENCE_AUDIT.md) document the distinctions.

Committed archives contain compact synthetic evidence. Full binary checkpoints
and trajectory banks remain in ignored local `runs/` directories and must be
regenerated to reproduce inference from those weights. Historical results should
be reproduced at their recorded source commit, with the source guards intact.
Test counts in historical experiment reports describe those recorded revisions.

## Attribution, citation, and licenses

The [mathematical provenance](docs/provenance.md) credits the thesis material,
research papers, and earlier simulation ideas. The
[DK guide](docs/DK_BACKGROUND.md) explains the relevant theory and assumptions.
Please cite [NN-DK](CITATION.cff) and the research papers relevant to your use.

Code is available under the [MIT License](LICENSE). Project-authored documentation,
figures, and archived synthetic evidence use [CC BY 4.0](LICENSE-DOCS), with
third-party attribution preserved. See [license coverage](LICENSING.md).
