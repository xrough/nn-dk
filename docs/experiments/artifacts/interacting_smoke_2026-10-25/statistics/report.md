# Interacting full-state neural statistics experiment

Target: stopped native DK, G=32, N=2048, kappa=0.75, a2=0.1, T=0.0625.

One full-state conditional-moment network spans both initial laws. It learns from known-generator residuals, with no future payoff labels. Ensemble predictions use independent initial-state Monte Carlo and total covariance.

Reference sampling error, initial-integration error and coupled time-step differences are reported separately. Continuous stopping/reference bias remains unresolved. This run cannot certify continuous-DK convergence or a 5% accuracy claim.

| Seed | Best validation objective | Fit seconds |
|---|---:|---:|
| 0 | 0.00587024 | 17.903 |

Errors use fixed uniform-particle fluctuation units. The values below are maximum point estimates over all query times and moment entries, not certified bounds.

| Seed | Profile | Scope | Mean error | Covariance error | Reference sampling halfwidth | Initial MC halfwidth |
|---|---|---|---:|---:|---:|---:|
| 0 | uniform | conditional 0 | 9.607% | 6.635% | 10.425% | 0.000% |
| 0 | uniform | conditional 1 | 13.941% | 9.079% | 11.282% | 0.000% |
| 0 | uniform | ensemble | 10.763% | 17.560% | 15.838% | 13.722% |
| 0 | nonuniform | conditional 0 | 4.660% | 4.971% | 10.307% | 0.000% |
| 0 | nonuniform | conditional 1 | 5.737% | 7.894% | 14.123% | 0.000% |
| 0 | nonuniform | ensemble | 7.007% | 7.926% | 13.350% | 11.588% |

Full per-case, per-anchor mean/covariance errors and uncertainty diagnostics are saved in `evaluations.json`; optimizer seeds are never pooled.
