# Source and mathematical provenance

The project follows `README_DK_Neural_Coarse_Dynamics_and_Statistics_v0_1.md`,
supplied on 23 October 2026, with Project A central and B1 as its statistical
comparator. NN-DK's backends are newly implemented one-dimensional models
with no runtime dependency on the thesis checkout.

## Thesis material adapted for the public documentation

The [DK background guide](DK_BACKGROUND.md) contains newly adapted exposition
from Xiaohao Ji's doctoral thesis, *Stochastic PDEs and Mean-Field Particle
Systems*, Institut für Mathematik, Freie Universität Berlin, 2026. It selects
the empirical-measure Itô calculation, conservative square-root noise,
regularization choices, and the weak-error result, then explains their relation
to NN-DK's finite-volume masses and stopping rule. It imports no thesis source
files, proofs, figures, or animation implementation.

The weak-error theory is joint work of **Ana Djurdjevac, Xiaohao Ji, and Nicolas
Perkowski**, published as *Weak Error of Dean–Kawasaki Equation with Smooth
Mean-Field Interactions*, *Journal de Mathématiques Pures et Appliquées*
**213** (2026), 103937,
[DOI: 10.1016/j.matpur.2026.103937](https://doi.org/10.1016/j.matpur.2026.103937)
([arXiv:2502.20929](https://arxiv.org/abs/2502.20929)). Citation of the thesis
does not replace attribution to those coauthors or the original papers below.

On 26 October 2026, the local thesis checkout was at commit
`9afa694a76ba54a3c1ab9422e5bbb0f839e7526f`, matching the inspected
[public thesis commit](https://github.com/xrough/spde-mf/tree/9afa694a76ba54a3c1ab9422e5bbb0f839e7526f),
but **both inspected chapters had uncommitted local revisions**. The guide
uses those revised passages, not a claim that they appear verbatim in that
public commit. The read-only inspection covered these checkout-relative files:

| File | Selected lines in the inspected local version | SHA-256 of the complete inspected file |
| --- | --- | --- |
| `thesis/Chapters/Introduction.tex` | 344–390 | `922742c4f4d13b66c6b1427e544aaaffff4e6771d7e3575490ca441fef72366d` |
| `thesis/Chapters/DK-error.tex` | 19–53, 107–126, 187–192, 603–606, 1008–1017, 1278–1283, 1493–1498 | `faa7ea2bb413df823294b88132c32f8c757e5f64dd305102de5c5c3c51fe64b9` |

These hashes identify the local source snapshot used for the adaptation; the
source files themselves are not distributed by NN-DK. The documentation's
license applies to this project's authored exposition, with cited works
retaining their own licenses and attribution.

## Earlier numerical reference

The legacy reference is the local thesis
`scripts/Dean-Kawasaki/deankawasaki_animation` implementation, inspected
read-only. It supplies useful grid, particle, conservative flux, and diagnostic
ideas. At the earlier inspection, that directory was untracked, and the public
commit above did not contain it. The old `jixh-KPZ-1020/spde-mf` URL redirects
to `xrough/spde-mf`; neither URL should be cited as containing that local DK
solver at the inspected commit.

Key corrections relative to the animation workflow are explicit cell/spline
conventions, exact finite-mode particle forces, matched IID initialization
diagnostics, no state clipping or renormalization, uncertainty-aware statistics,
and training/evaluation separation.

## Original literature and limits

- David S. Dean, *Langevin Equation for the Density of a System of Interacting
  Langevin Processes* (1996),
  [DOI: 10.1088/0305-4470/29/24/001](https://doi.org/10.1088/0305-4470/29/24/001)
  ([arXiv:cond-mat/9611104](https://arxiv.org/abs/cond-mat/9611104)). Source for
  the density-dependent conservative-noise derivation.
- Vitalii Konarovskyi, Tobias Lehmann, and Max-K. von Renesse,
  *Dean–Kawasaki Dynamics: Ill-posedness vs. Triviality* (2019),
  [DOI: 10.1214/19-ECP208](https://doi.org/10.1214/19-ECP208)
  ([arXiv:1806.05018](https://arxiv.org/abs/1806.05018)). Explains the
  restrictions on unregularized continuum martingale solutions.
- The Djurdjevac–Ji–Perkowski paper above supplies the conditional continuum
  weak-error framework. Its hypotheses do not automatically validate NN-DK's
  stopped discrete comparator or neural approximation.
- [Siddani et al.: history-dependent flux learning](https://arxiv.org/abs/2606.06658)
  remains relevant prior work for the deferred memory/non-Gaussian branch.

The projected particle law and stopped native finite-volume law are distinct
targets. No continuum convergence, universal transfer, or neural superiority
is claimed by the software.
