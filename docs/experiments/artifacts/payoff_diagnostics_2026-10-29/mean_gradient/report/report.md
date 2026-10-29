# Frozen mean-gradient and covariance-source diagnostic

This report compares frozen NN mean derivatives with fresh finite-Euler conditional-mean JVP estimates. Signed componentwise intervals use whole continuation families; all optimizer seeds remain separate.

A zero exclusion below is a **pointwise descriptive bootstrap result**, not a simultaneous accuracy or bias certificate. Positive RMS/Frobenius distances never decide an exclusion.

## Allocation and controls

- Cases: 6; frozen mean models: 18; fixed validation anchors: 24; retained model/anchor records: 72.
- Independent continuation families per anchor: 512; query fractions: [0.25, 0.5, 1.0]; coarse step: 0.0009765625.
- Paired refinement families per anchor: 64; step refinement factor: 2; bootstrap replicates: 128.
- Zero-time face controls: maximum error 4.44089e-16; absorbed face sources exactly zero: True.
- Native killed-payoff JVP finite-difference check: largest pathwise error 1.07558e-11. The derivative includes the stopping-exposure term.

## Signed NN-minus-native contrasts

Gradient RMS uses all four outputs and native face directions. Source RMS and exclusions count ten unique symmetric entries. Units are the stored normalized mean noise-face derivative and the source of normalized N-scaled covariance. The reference is the independent-half cross-Gram, not the quadratically biased plug-in Gram.

| Case | Seed | Profile / anchor | Mass / stop | Distance / width | τ | Gradient RMS | Grad exclusions | Source RMS | Source exclusions | Stopped / families |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0p25_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.0625 | 0.00155715 | 59/128 | 0.0112962 | 4/10 | 0/512 |
| T0p25_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.125 | 0.00293353 | 83/128 | 0.00734051 | 7/10 | 0/512 |
| T0p25_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.25 | 0.00215772 | 78/128 | 0.00522922 | 8/10 | 0/512 |
| T0p25_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.0625 | 0.0013313 | 47/128 | 0.0118467 | 4/10 | 0/512 |
| T0p25_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.125 | 0.00205864 | 68/128 | 0.00785451 | 7/10 | 0/512 |
| T0p25_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.25 | 0.00282557 | 75/128 | 0.0048487 | 8/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.0625 | 0.000908211 | 22/128 | 0.0133616 | 5/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.125 | 0.00130639 | 46/128 | 0.00886784 | 5/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.25 | 0.00180121 | 76/128 | 0.00299485 | 5/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.0010613 | 50/128 | 0.0124132 | 5/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.00178902 | 80/128 | 0.00828108 | 5/10 | 0/512 |
| T0p25_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.00368627 | 103/128 | 0.00642586 | 10/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.0625 | 0.00153534 | 60/128 | 0.0112966 | 6/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.125 | 0.00288418 | 85/128 | 0.00731743 | 7/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.25 | 0.00180335 | 71/128 | 0.0045019 | 8/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.0625 | 0.00135452 | 45/128 | 0.011868 | 4/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.125 | 0.00221873 | 72/128 | 0.00786925 | 5/10 | 0/512 |
| T0p25_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.25 | 0.00269902 | 76/128 | 0.00533478 | 6/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.0625 | 0.000913256 | 25/128 | 0.0133137 | 3/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.125 | 0.00129552 | 51/128 | 0.0087506 | 5/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.25 | 0.00193516 | 85/128 | 0.00299061 | 7/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.00105515 | 41/128 | 0.0124213 | 6/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.00166257 | 73/128 | 0.00826054 | 7/10 | 0/512 |
| T0p25_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.003046 | 97/128 | 0.00659252 | 7/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.0625 | 0.00150944 | 61/128 | 0.0114477 | 6/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.125 | 0.00281338 | 85/128 | 0.0076895 | 6/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 49.174 | 9.6348 | 0.25 | 0.00209862 | 79/128 | 0.00460855 | 8/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.0625 | 0.00133171 | 43/128 | 0.0120583 | 4/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.125 | 0.0021011 | 71/128 | 0.00828059 | 6/10 | 0/512 |
| T0p25_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 66.935 | 13.187 | 0.25 | 0.00232156 | 77/128 | 0.00512081 | 8/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.0625 | 0.000904449 | 27/128 | 0.013223 | 5/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.125 | 0.00130199 | 58/128 | 0.00859617 | 5/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 55.445 | 10.889 | 0.25 | 0.0018087 | 73/128 | 0.00256562 | 6/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.00105688 | 45/128 | 0.0127718 | 6/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.0016728 | 80/128 | 0.00902034 | 8/10 | 0/512 |
| T0p25_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.00328655 | 98/128 | 0.00663871 | 8/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.0625 | 0.00607743 | 98/128 | 0.0132625 | 6/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.125 | 0.0135437 | 108/128 | 0.0185535 | 9/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.25 | 0.01176 | 115/128 | 0.0174126 | 10/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.0625 | 0.00412766 | 78/128 | 0.0126697 | 8/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.125 | 0.00734306 | 88/128 | 0.0125113 | 10/10 | 0/512 |
| T0p25_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.25 | 0.011324 | 106/128 | 0.0115328 | 10/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.0625 | 0.00110107 | 54/128 | 0.0126353 | 6/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.125 | 0.00173873 | 81/128 | 0.00871587 | 9/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.25 | 0.00416242 | 101/128 | 0.013888 | 10/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.00228482 | 97/128 | 0.0128697 | 8/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.00622523 | 118/128 | 0.0124819 | 9/10 | 0/512 |
| T0p25_k3 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.0157593 | 119/128 | 0.0381978 | 10/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.0625 | 0.00605533 | 100/128 | 0.0123136 | 6/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.125 | 0.0136581 | 114/128 | 0.0168156 | 9/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.25 | 0.0128383 | 119/128 | 0.0190484 | 8/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.0625 | 0.00419878 | 83/128 | 0.0122067 | 7/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.125 | 0.00774303 | 92/128 | 0.011606 | 10/10 | 0/512 |
| T0p25_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.25 | 0.00982253 | 103/128 | 0.0121945 | 10/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.0625 | 0.00111259 | 55/128 | 0.012576 | 5/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.125 | 0.00181749 | 85/128 | 0.00869125 | 8/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.25 | 0.00411785 | 100/128 | 0.00883339 | 9/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.0020471 | 92/128 | 0.0125978 | 8/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.00522806 | 115/128 | 0.0109774 | 9/10 | 0/512 |
| T0p25_k3 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.0123043 | 122/128 | 0.0253114 | 10/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.0625 | 0.00600519 | 100/128 | 0.0132635 | 6/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.125 | 0.0133012 | 106/128 | 0.0181315 | 8/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 41.958 | 8.1915 | 0.25 | 0.0120468 | 112/128 | 0.0217191 | 10/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.0625 | 0.00418496 | 87/128 | 0.011658 | 9/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.125 | 0.0075704 | 100/128 | 0.0109219 | 9/10 | 0/512 |
| T0p25_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 65.168 | 12.834 | 0.25 | 0.010075 | 105/128 | 0.0143653 | 10/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.0625 | 0.00121156 | 65/128 | 0.0129967 | 8/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.125 | 0.00232721 | 98/128 | 0.00977325 | 9/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 2 (above_25_thresholds) | 56.25 | 11.05 | 0.25 | 0.00620218 | 119/128 | 0.0138726 | 10/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.0625 | 0.00212485 | 84/128 | 0.0130153 | 6/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.125 | 0.00549884 | 107/128 | 0.0115304 | 9/10 | 0/512 |
| T0p25_k3 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.01367 | 124/128 | 0.0307677 | 8/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.0625 | 0.0249677 | 125/128 | 0.0812185 | 9/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.125 | 0.0637133 | 126/128 | 0.276106 | 10/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.25 | 0.0972613 | 126/128 | 0.436642 | 10/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.0625 | 0.00888949 | 119/128 | 0.0237036 | 8/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.125 | 0.0181157 | 124/128 | 0.0402475 | 10/10 | 0/512 |
| T0p25_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0222381 | 116/128 | 0.0495215 | 10/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.0625 | 0.00451093 | 105/128 | 0.017089 | 8/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.125 | 0.0130887 | 118/128 | 0.0274151 | 10/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.25 | 0.0326267 | 120/128 | 0.095793 | 10/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.0625 | 0.0036616 | 99/128 | 0.0150828 | 7/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.125 | 0.00982136 | 112/128 | 0.0182375 | 10/10 | 0/512 |
| T0p25_k6 | 0 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.25 | 0.0249378 | 125/128 | 0.0655567 | 9/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.0625 | 0.0252367 | 125/128 | 0.0784276 | 8/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.125 | 0.0649741 | 127/128 | 0.275011 | 10/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.25 | 0.103992 | 126/128 | 0.474967 | 9/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.0625 | 0.00878628 | 114/128 | 0.0218732 | 8/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.125 | 0.0178353 | 122/128 | 0.0349943 | 10/10 | 0/512 |
| T0p25_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0224823 | 118/128 | 0.0432228 | 10/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.0625 | 0.00363446 | 103/128 | 0.0167344 | 8/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.125 | 0.00986178 | 120/128 | 0.0232128 | 10/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.25 | 0.0279278 | 120/128 | 0.0858435 | 10/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.0625 | 0.00404379 | 115/128 | 0.0140343 | 7/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.125 | 0.0110633 | 121/128 | 0.0167891 | 7/10 | 0/512 |
| T0p25_k6 | 1 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.25 | 0.0265132 | 118/128 | 0.0533935 | 10/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.0625 | 0.0259674 | 125/128 | 0.0782046 | 9/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.125 | 0.0679674 | 125/128 | 0.287175 | 9/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 0 (above_25_thresholds) | 45.975 | 8.995 | 0.25 | 0.116066 | 124/128 | 0.554611 | 10/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.0625 | 0.00897869 | 114/128 | 0.0206079 | 9/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.125 | 0.0188885 | 118/128 | 0.0338601 | 9/10 | 0/512 |
| T0p25_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0202209 | 116/128 | 0.0387794 | 10/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.0625 | 0.00418777 | 120/128 | 0.0152514 | 8/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.125 | 0.0120464 | 122/128 | 0.0244565 | 9/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 2 (above_25_thresholds) | 46.875 | 9.175 | 0.25 | 0.0311646 | 122/128 | 0.115908 | 10/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.0625 | 0.00378939 | 112/128 | 0.0134086 | 9/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.125 | 0.0101741 | 115/128 | 0.0155018 | 9/10 | 0/512 |
| T0p25_k6 | 2 | uniform / 3 (above_25_thresholds) | 74.751 | 14.75 | 0.25 | 0.0240149 | 124/128 | 0.0379253 | 8/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.25 | 0.0093355 | 120/128 | 0.00379365 | 10/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.5 | 0.00991 | 122/128 | 0.00690009 | 10/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 1 | 0.00923763 | 117/128 | 0.0112526 | 9/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.25 | 0.0066833 | 121/128 | 0.0028662 | 9/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.5 | 0.00712543 | 122/128 | 0.00608859 | 10/10 | 0/512 |
| T1_k0p75 | 0 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 1 | 0.00720751 | 123/128 | 0.0101038 | 10/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.25 | 0.00317616 | 119/128 | 0.00295831 | 8/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.5 | 0.00564526 | 121/128 | 0.00824068 | 10/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 1 | 0.0112927 | 126/128 | 0.0222264 | 10/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.0029984 | 118/128 | 0.0025275 | 8/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.5 | 0.00519778 | 124/128 | 0.00796353 | 9/10 | 0/512 |
| T1_k0p75 | 0 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 1 | 0.00848057 | 118/128 | 0.0244164 | 10/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.25 | 0.0082376 | 122/128 | 0.00307924 | 8/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.5 | 0.00710312 | 119/128 | 0.00546488 | 10/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 1 | 0.00752111 | 114/128 | 0.00793752 | 10/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.25 | 0.00538611 | 118/128 | 0.00236368 | 7/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.5 | 0.0041481 | 118/128 | 0.00596708 | 9/10 | 0/512 |
| T1_k0p75 | 1 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 1 | 0.00736665 | 124/128 | 0.0125866 | 10/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.25 | 0.00301712 | 117/128 | 0.00279527 | 8/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.5 | 0.00538723 | 117/128 | 0.00694438 | 9/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 1 | 0.0119727 | 122/128 | 0.0246815 | 10/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.00294011 | 111/128 | 0.00247203 | 7/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.5 | 0.00477672 | 123/128 | 0.0065941 | 10/10 | 0/512 |
| T1_k0p75 | 1 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 1 | 0.00817772 | 122/128 | 0.0181556 | 10/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.25 | 0.00685399 | 118/128 | 0.00247874 | 8/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 0.5 | 0.00456474 | 122/128 | 0.00505686 | 9/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 0 (above_25_thresholds) | 43.75 | 8.55 | 1 | 0.00594409 | 120/128 | 0.0116185 | 10/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.25 | 0.00493847 | 117/128 | 0.00241268 | 9/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 0.5 | 0.00393692 | 117/128 | 0.00596938 | 10/10 | 0/512 |
| T1_k0p75 | 2 | nonuniform / 1 (above_25_thresholds) | 68.697 | 13.539 | 1 | 0.00634761 | 122/128 | 0.0111892 | 10/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.25 | 0.00287855 | 114/128 | 0.00269188 | 8/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 0.5 | 0.00505043 | 117/128 | 0.00598677 | 9/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 2 (above_25_thresholds) | 52 | 10.2 | 1 | 0.0114217 | 121/128 | 0.0214337 | 9/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.25 | 0.00273069 | 112/128 | 0.00232448 | 9/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 0.5 | 0.00440047 | 115/128 | 0.00669004 | 10/10 | 0/512 |
| T1_k0p75 | 2 | uniform / 3 (above_25_thresholds) | 75 | 14.8 | 1 | 0.00857988 | 126/128 | 0.0215861 | 10/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.25 | 0.0763206 | 125/128 | 0.205776 | 10/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.5 | 0.07873 | 126/128 | 0.210519 | 10/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 1 | 0.110618 | 127/128 | 0.481392 | 10/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0429434 | 124/128 | 0.0668931 | 9/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.5 | 0.0434008 | 125/128 | 0.070318 | 10/10 | 0/512 |
| T1_k3 | 0 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 1 | 0.0423722 | 125/128 | 0.15805 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.25 | 0.0125456 | 125/128 | 0.0135405 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.5 | 0.0257357 | 128/128 | 0.0640304 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 1 | 0.056992 | 127/128 | 0.407554 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.25 | 0.0128572 | 118/128 | 0.0088191 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.5 | 0.0200424 | 121/128 | 0.0267105 | 10/10 | 0/512 |
| T1_k3 | 0 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 1 | 0.0388342 | 119/128 | 0.170881 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.25 | 0.0645672 | 126/128 | 0.155367 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.5 | 0.0526243 | 126/128 | 0.103276 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 1 | 0.0740826 | 127/128 | 0.256076 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0299781 | 123/128 | 0.0377555 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.5 | 0.0195712 | 124/128 | 0.0222305 | 10/10 | 0/512 |
| T1_k3 | 1 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 1 | 0.0314053 | 126/128 | 0.0960551 | 10/10 | 0/512 |
| T1_k3 | 1 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.25 | 0.0119496 | 124/128 | 0.013816 | 10/10 | 0/512 |
| T1_k3 | 1 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.5 | 0.0235766 | 126/128 | 0.0649637 | 10/10 | 0/512 |
| T1_k3 | 1 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 1 | 0.0501199 | 127/128 | 0.377266 | 10/10 | 0/512 |
| T1_k3 | 1 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.25 | 0.0114566 | 116/128 | 0.00776804 | 10/10 | 0/512 |
| T1_k3 | 1 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.5 | 0.0192698 | 122/128 | 0.0173235 | 9/10 | 0/512 |
| T1_k3 | 1 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 1 | 0.0508767 | 127/128 | 0.169946 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.25 | 0.0746123 | 123/128 | 0.19825 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 0.5 | 0.0743816 | 125/128 | 0.181242 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 0 (above_25_thresholds) | 34.481 | 6.6963 | 1 | 0.0947302 | 126/128 | 0.311097 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.25 | 0.0406817 | 126/128 | 0.0624028 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 0.5 | 0.0393885 | 122/128 | 0.058544 | 10/10 | 0/512 |
| T1_k3 | 2 | nonuniform / 1 (above_25_thresholds) | 62.5 | 12.3 | 1 | 0.0280377 | 124/128 | 0.102767 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.25 | 0.0118716 | 124/128 | 0.0122798 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 0.5 | 0.0224529 | 128/128 | 0.0523372 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 2 (above_25_thresholds) | 54.076 | 10.615 | 1 | 0.0452797 | 127/128 | 0.331833 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.25 | 0.0122741 | 124/128 | 0.00885719 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 0.5 | 0.0184195 | 126/128 | 0.0225726 | 10/10 | 0/512 |
| T1_k3 | 2 | uniform / 3 (above_25_thresholds) | 74.108 | 14.622 | 1 | 0.0408324 | 125/128 | 0.127037 | 10/10 | 0/512 |
| T1_k6 | 0 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.25 | 7.18068 | 127/128 | 1881.38 | 10/10 | 512/512 |
| T1_k6 | 0 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.5 | 10.7583 | 126/128 | 4348.78 | 10/10 | 512/512 |
| T1_k6 | 0 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 1 | 15.9459 | 128/128 | 8832.97 | 10/10 | 512/512 |
| T1_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.25 | 0.20128 | 128/128 | 1.40857 | 10/10 | 505/512 |
| T1_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.5 | 0.325468 | 128/128 | 2.43892 | 10/10 | 505/512 |
| T1_k6 | 0 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 1 | 0.638612 | 126/128 | 5.57984 | 10/10 | 505/512 |
| T1_k6 | 0 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.25 | 0.592219 | 127/128 | 11.1381 | 10/10 | 512/512 |
| T1_k6 | 0 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.5 | 1.2856 | 127/128 | 47.1319 | 10/10 | 512/512 |
| T1_k6 | 0 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 1 | 2.76905 | 127/128 | 219.863 | 10/10 | 512/512 |
| T1_k6 | 0 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.25 | 0.0520961 | 127/128 | 0.096852 | 10/10 | 0/512 |
| T1_k6 | 0 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.5 | 0.134096 | 128/128 | 0.596087 | 9/10 | 0/512 |
| T1_k6 | 0 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 1 | 0.48302 | 126/128 | 8.02363 | 10/10 | 0/512 |
| T1_k6 | 1 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.25 | 6.15153 | 127/128 | 1371.83 | 10/10 | 512/512 |
| T1_k6 | 1 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.5 | 7.28358 | 126/128 | 1965.08 | 10/10 | 512/512 |
| T1_k6 | 1 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 1 | 14.905 | 126/128 | 7724.75 | 10/10 | 512/512 |
| T1_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.25 | 0.18991 | 128/128 | 1.35668 | 10/10 | 505/512 |
| T1_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.5 | 0.345895 | 127/128 | 3.11747 | 10/10 | 505/512 |
| T1_k6 | 1 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 1 | 0.718858 | 128/128 | 8.32393 | 10/10 | 505/512 |
| T1_k6 | 1 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.25 | 0.710522 | 128/128 | 15.6205 | 10/10 | 512/512 |
| T1_k6 | 1 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.5 | 1.75569 | 126/128 | 90.6627 | 10/10 | 512/512 |
| T1_k6 | 1 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 1 | 4.48275 | 128/128 | 614.741 | 10/10 | 512/512 |
| T1_k6 | 1 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.25 | 0.0501692 | 126/128 | 0.0910792 | 9/10 | 0/512 |
| T1_k6 | 1 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.5 | 0.127232 | 126/128 | 0.571727 | 10/10 | 0/512 |
| T1_k6 | 1 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 1 | 0.46082 | 126/128 | 7.93358 | 10/10 | 0/512 |
| T1_k6 | 2 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.25 | 6.83673 | 126/128 | 1683.92 | 10/10 | 512/512 |
| T1_k6 | 2 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 0.5 | 9.12823 | 128/128 | 3014.15 | 10/10 | 512/512 |
| T1_k6 | 2 | nonuniform / 0 (two_to_five_thresholds) | 3.8706 | 0.57412 | 1 | 19.2993 | 128/128 | 13640.9 | 10/10 | 512/512 |
| T1_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.25 | 0.218457 | 127/128 | 1.65462 | 10/10 | 505/512 |
| T1_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 0.5 | 0.379662 | 128/128 | 3.56959 | 10/10 | 505/512 |
| T1_k6 | 2 | nonuniform / 1 (above_25_thresholds) | 50.738 | 9.9475 | 1 | 0.610836 | 126/128 | 4.85146 | 10/10 | 505/512 |
| T1_k6 | 2 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.25 | 0.592941 | 128/128 | 11.1916 | 10/10 | 512/512 |
| T1_k6 | 2 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 0.5 | 1.26479 | 126/128 | 45.5194 | 10/10 | 512/512 |
| T1_k6 | 2 | uniform / 2 (above_25_thresholds) | 29.372 | 5.6743 | 1 | 2.53943 | 128/128 | 183.23 | 10/10 | 512/512 |
| T1_k6 | 2 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.25 | 0.054132 | 127/128 | 0.101309 | 10/10 | 0/512 |
| T1_k6 | 2 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 0.5 | 0.133144 | 128/128 | 0.578201 | 9/10 | 0/512 |
| T1_k6 | 2 | uniform / 3 (above_25_thresholds) | 71.113 | 14.023 | 1 | 0.445295 | 126/128 | 7.6109 | 10/10 | 0/512 |

## Paired finite-grid refinement

Both grids reuse each physical Brownian family and each bootstrap draw, including the same independent-half partition. Rows are shared by all optimizer seeds. These finite-grid differences do not bound remaining bias.

| Case | Profile / anchor | τ | Gradient change RMS | Grad exclusions | Cross-source change RMS | Source exclusions | Families |
|---|---|---:|---:|---:|---:|---:|---:|
| T0p25_k0p75 | nonuniform / 0 | 0.0625 | 0.000358339 | 65/128 | 0.00617884 | 5/10 | 64 |
| T0p25_k0p75 | nonuniform / 0 | 0.125 | 0.000411461 | 67/128 | 0.00410801 | 8/10 | 64 |
| T0p25_k0p75 | nonuniform / 0 | 0.25 | 0.000274418 | 66/128 | 0.000916289 | 9/10 | 64 |
| T0p25_k0p75 | nonuniform / 1 | 0.0625 | 0.000357483 | 61/128 | 0.00617007 | 7/10 | 64 |
| T0p25_k0p75 | nonuniform / 1 | 0.125 | 0.000411778 | 61/128 | 0.00411208 | 7/10 | 64 |
| T0p25_k0p75 | nonuniform / 1 | 0.25 | 0.000275029 | 62/128 | 0.000918236 | 7/10 | 64 |
| T0p25_k0p75 | uniform / 2 | 0.0625 | 0.000354949 | 64/128 | 0.00612674 | 3/10 | 64 |
| T0p25_k0p75 | uniform / 2 | 0.125 | 0.000410555 | 64/128 | 0.00410136 | 5/10 | 64 |
| T0p25_k0p75 | uniform / 2 | 0.25 | 0.000274795 | 66/128 | 0.000916787 | 4/10 | 64 |
| T0p25_k0p75 | uniform / 3 | 0.0625 | 0.000354935 | 63/128 | 0.00612773 | 6/10 | 64 |
| T0p25_k0p75 | uniform / 3 | 0.125 | 0.000411001 | 63/128 | 0.00410657 | 10/10 | 64 |
| T0p25_k0p75 | uniform / 3 | 0.25 | 0.000274324 | 63/128 | 0.000916618 | 10/10 | 64 |
| T0p25_k3 | nonuniform / 0 | 0.0625 | 0.000360329 | 64/128 | 0.00614571 | 3/10 | 64 |
| T0p25_k3 | nonuniform / 0 | 0.125 | 0.000412268 | 65/128 | 0.00408061 | 5/10 | 64 |
| T0p25_k3 | nonuniform / 0 | 0.25 | 0.000274696 | 66/128 | 0.000909741 | 6/10 | 64 |
| T0p25_k3 | nonuniform / 1 | 0.0625 | 0.000356861 | 68/128 | 0.0061137 | 5/10 | 64 |
| T0p25_k3 | nonuniform / 1 | 0.125 | 0.000411624 | 66/128 | 0.004089 | 8/10 | 64 |
| T0p25_k3 | nonuniform / 1 | 0.25 | 0.00027427 | 63/128 | 0.000911231 | 8/10 | 64 |
| T0p25_k3 | uniform / 2 | 0.0625 | 0.000354371 | 66/128 | 0.00611809 | 4/10 | 64 |
| T0p25_k3 | uniform / 2 | 0.125 | 0.000410396 | 67/128 | 0.00409923 | 5/10 | 64 |
| T0p25_k3 | uniform / 2 | 0.25 | 0.000275148 | 67/128 | 0.000917104 | 7/10 | 64 |
| T0p25_k3 | uniform / 3 | 0.0625 | 0.000355439 | 66/128 | 0.00613713 | 5/10 | 64 |
| T0p25_k3 | uniform / 3 | 0.125 | 0.000410448 | 67/128 | 0.00410128 | 6/10 | 64 |
| T0p25_k3 | uniform / 3 | 0.25 | 0.000274708 | 67/128 | 0.000919365 | 7/10 | 64 |
| T0p25_k6 | nonuniform / 0 | 0.0625 | 0.000386839 | 62/128 | 0.00593005 | 10/10 | 64 |
| T0p25_k6 | nonuniform / 0 | 0.125 | 0.000425299 | 66/128 | 0.00383341 | 10/10 | 64 |
| T0p25_k6 | nonuniform / 0 | 0.25 | 0.000280604 | 76/128 | 0.000867107 | 9/10 | 64 |
| T0p25_k6 | nonuniform / 1 | 0.0625 | 0.000363633 | 65/128 | 0.00607893 | 7/10 | 64 |
| T0p25_k6 | nonuniform / 1 | 0.125 | 0.000414052 | 71/128 | 0.00402455 | 7/10 | 64 |
| T0p25_k6 | nonuniform / 1 | 0.25 | 0.00027704 | 85/128 | 0.000951733 | 6/10 | 64 |
| T0p25_k6 | uniform / 2 | 0.0625 | 0.000356384 | 72/128 | 0.00613481 | 7/10 | 64 |
| T0p25_k6 | uniform / 2 | 0.125 | 0.000410612 | 74/128 | 0.00409474 | 9/10 | 64 |
| T0p25_k6 | uniform / 2 | 0.25 | 0.000277048 | 89/128 | 0.000999941 | 7/10 | 64 |
| T0p25_k6 | uniform / 3 | 0.0625 | 0.000355258 | 65/128 | 0.00612147 | 6/10 | 64 |
| T0p25_k6 | uniform / 3 | 0.125 | 0.000410782 | 72/128 | 0.00409744 | 6/10 | 64 |
| T0p25_k6 | uniform / 3 | 0.25 | 0.000276235 | 89/128 | 0.000992168 | 6/10 | 64 |
| T1_k0p75 | nonuniform / 0 | 0.25 | 0.000274284 | 64/128 | 0.000916042 | 9/10 | 64 |
| T1_k0p75 | nonuniform / 0 | 0.5 | 6.15639e-05 | 67/128 | 2.47364e-05 | 8/10 | 64 |
| T1_k0p75 | nonuniform / 0 | 1 | 1.1385e-05 | 48/128 | 4.29899e-05 | 2/10 | 64 |
| T1_k0p75 | nonuniform / 1 | 0.25 | 0.000274161 | 64/128 | 0.00091612 | 9/10 | 64 |
| T1_k0p75 | nonuniform / 1 | 0.5 | 6.14931e-05 | 66/128 | 2.47686e-05 | 6/10 | 64 |
| T1_k0p75 | nonuniform / 1 | 1 | 1.02529e-05 | 48/128 | 4.23364e-05 | 2/10 | 64 |
| T1_k0p75 | uniform / 2 | 0.25 | 0.000274551 | 66/128 | 0.000916958 | 9/10 | 64 |
| T1_k0p75 | uniform / 2 | 0.5 | 6.17044e-05 | 68/128 | 2.49646e-05 | 5/10 | 64 |
| T1_k0p75 | uniform / 2 | 1 | 1.30355e-05 | 57/128 | 4.30472e-05 | 3/10 | 64 |
| T1_k0p75 | uniform / 3 | 0.25 | 0.000274426 | 68/128 | 0.000916474 | 7/10 | 64 |
| T1_k0p75 | uniform / 3 | 0.5 | 6.16065e-05 | 68/128 | 2.41177e-05 | 5/10 | 64 |
| T1_k0p75 | uniform / 3 | 1 | 1.06759e-05 | 47/128 | 4.23147e-05 | 3/10 | 64 |
| T1_k3 | nonuniform / 0 | 0.25 | 0.00027322 | 70/128 | 0.000897458 | 7/10 | 64 |
| T1_k3 | nonuniform / 0 | 0.5 | 6.18195e-05 | 69/128 | 2.76265e-05 | 4/10 | 64 |
| T1_k3 | nonuniform / 0 | 1 | 1.77644e-05 | 24/128 | 5.20011e-05 | 1/10 | 64 |
| T1_k3 | nonuniform / 1 | 0.25 | 0.00027444 | 63/128 | 0.000910757 | 6/10 | 64 |
| T1_k3 | nonuniform / 1 | 0.5 | 6.15454e-05 | 61/128 | 2.66067e-05 | 5/10 | 64 |
| T1_k3 | nonuniform / 1 | 1 | 1.4731e-05 | 11/128 | 6.75587e-05 | 2/10 | 64 |
| T1_k3 | uniform / 2 | 0.25 | 0.000274104 | 68/128 | 0.000914992 | 9/10 | 64 |
| T1_k3 | uniform / 2 | 0.5 | 6.19804e-05 | 67/128 | 2.68444e-05 | 7/10 | 64 |
| T1_k3 | uniform / 2 | 1 | 2.08505e-05 | 11/128 | 9.27257e-05 | 3/10 | 64 |
| T1_k3 | uniform / 3 | 0.25 | 0.000273944 | 61/128 | 0.000913666 | 7/10 | 64 |
| T1_k3 | uniform / 3 | 0.5 | 6.14607e-05 | 62/128 | 2.57706e-05 | 4/10 | 64 |
| T1_k3 | uniform / 3 | 1 | 1.59214e-05 | 21/128 | 9.65394e-05 | 4/10 | 64 |
| T1_k6 | nonuniform / 0 | 0.25 | 0.0190437 | 69/128 | 0.132407 | 4/10 | 64 |
| T1_k6 | nonuniform / 0 | 0.5 | 0.0190437 | 69/128 | 0.132407 | 4/10 | 64 |
| T1_k6 | nonuniform / 0 | 1 | 0.0190437 | 69/128 | 0.132407 | 4/10 | 64 |
| T1_k6 | nonuniform / 1 | 0.25 | 0.000281968 | 80/128 | 0.000838006 | 10/10 | 64 |
| T1_k6 | nonuniform / 1 | 0.5 | 0.000118626 | 81/128 | 0.000977056 | 10/10 | 64 |
| T1_k6 | nonuniform / 1 | 1 | 0.0065082 | 52/128 | 0.168351 | 6/10 | 64 |
| T1_k6 | uniform / 2 | 0.25 | 0.000609682 | 60/128 | 0.000908098 | 2/10 | 64 |
| T1_k6 | uniform / 2 | 0.5 | 0.00333402 | 15/128 | 0.0361636 | 1/10 | 64 |
| T1_k6 | uniform / 2 | 1 | 0.00421834 | 13/128 | 0.0414307 | 1/10 | 64 |
| T1_k6 | uniform / 3 | 0.25 | 0.000273981 | 91/128 | 0.000909998 | 7/10 | 64 |
| T1_k6 | uniform / 3 | 0.5 | 7.57202e-05 | 112/128 | 0.000360777 | 7/10 | 64 |
| T1_k6 | uniform / 3 | 1 | 0.000388304 | 113/128 | 0.0102911 | 9/10 | 64 |

## Interpretation and limits

- Pointwise percentile bootstrap intervals are not simultaneous confidence bands; exclusion counts do not certify a grid-wide bias or accuracy statement.
- Frobenius/RMS distances are descriptive. A positive norm or positive lower percentile of a norm is never evidence of a resolved discrepancy.
- The plug-in Gram of a Monte Carlo mean has positive quadratic sampling bias. The independent-half symmetric cross-Gram targets the source without that plug-in bias, assuming differentiation under expectation is valid; it may be indefinite and is never projected PSD.
- Coarse/fine derivative and cross-source contrasts use paired whole-family resampling. A finite refinement difference does not enclose remaining time-discretization, stopping, or derivative-interchange bias.
- Near-hit pathwise sensitivities can have heavy tails. Finite bootstrap samples and common-noise finite differences do not certify their variance or asymptotic coverage.
- Anchors come from state-only validation pools selected before new continuations. These diagnostics do not assess a new independent initial-state population.
- This report studies the frozen mean and its covariance source. It makes no covariance-head prediction-accuracy, fitting-convergence, neural-speedup, or continuum-DK claim.

All signed estimates, interval endpoints and component masks are in `contrasts.npz`; `report.json` retains model/anchor/boundary metadata and the full input identity. `output_manifest.json` verifies these outputs and their frozen inputs.
