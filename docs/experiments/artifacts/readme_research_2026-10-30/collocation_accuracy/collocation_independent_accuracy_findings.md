The completed C accuracy evidence passes the strict frozen reader: all 72 fits, four variants and three seeds remain, with 576 primary predictions and 732 retained records including 156 paired-initial diagnostics. Primary neural IID estimates use the independently drawn integration pools. No timing or cost-advantage conclusion is made.

Worst discrepancies below cover both profiles, both conditional anchors or the independent IID query, and all three horizons for each model. Mean values are reference SD; covariance values use reference-SD products. Ranges preserve all three seeds.

| Case | Variant | Worst mean | Worst covariance |
|---|---|---:|---:|
| T0p25_k0p75 | uniform/pseudo_huber | 0.0166–0.0177 | 0.0244–0.0255 |
| T0p25_k0p75 | uniform/squared | 0.0174–0.0197 | 0.0244–0.0255 |
| T0p25_k0p75 | stratified/pseudo_huber | 0.0161–0.019 | 0.0246–0.0255 |
| T0p25_k0p75 | stratified/squared | 0.0162–0.02 | 0.0249–0.0256 |
| T0p25_k3 | uniform/pseudo_huber | 0.0232–0.0371 | 0.0256–0.0279 |
| T0p25_k3 | uniform/squared | 0.0312–0.0413 | 0.0254–0.027 |
| T0p25_k3 | stratified/pseudo_huber | 0.0277–0.0465 | 0.026–0.0293 |
| T0p25_k3 | stratified/squared | 0.0374–0.0461 | 0.0259–0.0278 |
| T0p25_k6 | uniform/pseudo_huber | 0.0997–0.111 | 0.0468–0.0519 |
| T0p25_k6 | uniform/squared | 0.104–0.124 | 0.0481–0.0533 |
| T0p25_k6 | stratified/pseudo_huber | 0.098–0.117 | 0.0425–0.0479 |
| T0p25_k6 | stratified/squared | 0.106–0.117 | 0.0455–0.0478 |
| T1_k0p75 | uniform/pseudo_huber | 0.106–0.14 | 0.0547–0.0844 |
| T1_k0p75 | uniform/squared | 0.106–0.141 | 0.0566–0.0838 |
| T1_k0p75 | stratified/pseudo_huber | 0.102–0.135 | 0.052–0.0818 |
| T1_k0p75 | stratified/squared | 0.104–0.136 | 0.0499–0.0768 |
| T1_k3 | uniform/pseudo_huber | 0.189–0.247 | 0.267–0.308 |
| T1_k3 | uniform/squared | 0.238–0.315 | 0.274–0.349 |
| T1_k3 | stratified/pseudo_huber | 0.204–0.307 | 0.254–0.283 |
| T1_k3 | stratified/squared | 0.192–0.307 | 0.193–0.252 |
| T1_k6 | uniform/pseudo_huber | 4.84–5.08 | 0.942–2.92 |
| T1_k6 | uniform/squared | 11.4–18.3 | 0.945–0.964 |
| T1_k6 | stratified/pseudo_huber | 4.98–5.44 | 0.942–0.943 |
| T1_k6 | stratified/squared | 8.43–11.1 | 3.54–6.97 |

All-seed signed pointwise comparisons count profile/role/anchor/time/statistic blocks, with three seeds retained in every block. Against matched-seed uniform/pseudo-Huber, the 648 blocks contain 12 all-seed candidate-closer, 61 all-seed control-closer and 575 unresolved or seed-inconsistent blocks. Against Gaussian dt10, the corresponding 864-block counts are 0, 377 and 487. These are pointwise consistency summaries, not simultaneous inference.

Accuracy-only gate statuses are retained in full in the JSON: error-plus-uncertainty-unresolved=270, point-error-above-5pct=312, sampling-component-above-1.25pct=186. Physical moment-bound diagnostics remain separate. No timing-dependent qualification is evaluated.

In T1_k6, the stratified-squared validation residual MSE reductions are distinct from the independent discrepancies. Its worst mean error is 8.43–11.1 SD and worst covariance discrepancy is 3.54–6.97 SD products; Gaussian dt10 gives 0.256 and 0.169, respectively. For comparison, uniform-squared worst covariance discrepancies are 0.945–0.964 SD products and uniform-pseudo-Huber worst means are 4.84–5.08 SD. All three seeds favor the uniform-pseudo-Huber control over stratified-squared in all 18 strong-case mean blocks and 6 of 18 covariance blocks, with 12 covariance blocks unresolved. All 144 strong-case profile/anchor/time/statistic/variant blocks favor Gaussian in all three seeds pointwise. The paired residual/accuracy values for every strong-case fit remain in the JSON. A smaller selection-pool MSE does not establish accurate fluctuation moments.

All failed or unresolved finite-reference gates remain visible. Reference refinement, Monte Carlo uncertainty, finite Euler/stopping bias and the fixed four-payoff/same-time scope remain separate limitations. No partial A data, new physical paths, refits or criterion changes were used.

