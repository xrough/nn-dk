The completed supplement retains all 72 fits (six cases, three seeds, four crossed variants) on the same natural selection pool: 4,096 rows and 256 distinct parent trajectories per case. These are descriptive validation residuals, not independent payoff accuracy, confidence intervals or an error certificate. All four mean components and all 16 covariance entries enter the losses. All 19 bins remain, including undefined empty-bin metrics.

The main case-dependent ordering is captured below. Each entry counts seeds with lower natural-population MSE under stratification than uniform sampling at the same loss (pseudo-Huber / squared); it is not a significance test.

| Case | Mean, seeds lower / 3 (PH / squared) | Covariance, seeds lower / 3 (PH / squared) |
|---|---:|---:|
| T0p25_k0p75 | 1 / 2 | 2 / 1 |
| T0p25_k3 | 0 / 2 | 0 / 1 |
| T0p25_k6 | 3 / 2 | 3 / 2 |
| T1_k0p75 | 2 / 2 | 1 / 2 |
| T1_k3 | 1 / 3 | 3 / 2 |
| T1_k6 | 2 / 3 | 2 / 3 |

In T1_k6, stratified squared loss reduces mean MSE by 36.1–73.7% and covariance MSE by 86.2–94.6% versus uniform squared loss in matched seeds. Its mean MSE is also lower than uniform pseudo-Huber in all seeds, while its mean pseudo-Huber loss remains higher. The stored common training units preserve that tradeoff:

| T1_k6 variant | Mean MSE | Mean Huber | Covariance MSE | Covariance Huber | Physical covariance residual RMS |
|---|---:|---:|---:|---:|---:|
| uniform/pseudo_huber | 2360–2600 | 39.03–42.11 | 1.176e+05–1.479e+05 | 67.29–82.87 | 0.05779–0.07011 |
| uniform/squared | 1821–3296 | 71.51–92.86 | 1.395e+05–4.139e+05 | 146.8–233.2 | 0.05273–0.1038 |
| stratified/pseudo_huber | 2279–2653 | 37.63–40.76 | 1.26e+05–1.486e+05 | 68.22–77.39 | 0.05871–0.06459 |
| stratified/squared | 868.5–1164 | 55.1–59.94 | 1.222e+04–4.009e+04 | 54.55–75.58 | 0.02252–0.03869 |

Raw solver and physical residual RMS values for every fit, plus the complete matched-seed MSE/Huber comparisons, are retained in the accompanying JSON. Physical mean terms use bank scale/√N; covariance terms use the bank-scale outer product/N. Training-normalized terms use T divided by the original stored stage scale. Cross-regime magnitudes are not measured fluctuation errors.

Five cases occupy only four distance-3/time bins. T1_k6 occupies 11 of 19: 11 distance-1 rows (0.2686%), 294 distance-2 rows and 3,791 distance-3 rows. No case contains absorbed, exact-zero-time or distance-0 validation rows. The three occupied distance-1 bins contain 1/3/7 rows and 1/2/5 distinct parents respectively; parent counts must not be summed across bins because rows can share parents. Stratification cannot supply evidence for absent states.

In T1_k6, the 11 distance-1 rows contribute 98.1–98.9% of covariance MSE for uniform pseudo-Huber and 98.6–98.9% for stratified pseudo-Huber, versus 85.2–89.2% for stratified squared loss. Their covariance Huber contributions are 62.6–67.1%, 65.4–67.9%, and 31.3–39.9%, respectively. Natural population contributions use occupancy weights; an equal-bin average would change the measure.

Replacing only Γ with the matched-seed uniform/pseudo-Huber mean source raises T1_k6 stratified-squared covariance MSE by factors 3.33–9.75 (to approximately 1.19×10⁵–1.49×10⁵). This removes much of its own-source residual advantage under that substitution. It does not identify a correct mean gradient, simulate common-mean training, or establish a causal explanation of independent covariance accuracy. Source/operator RMS magnitudes do not add as loss contributions.

