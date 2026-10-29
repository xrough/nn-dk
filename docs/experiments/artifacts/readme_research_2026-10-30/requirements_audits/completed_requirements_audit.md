# Completed R1 and publication-provenance audit

This read-only audit verifies completed evidence and its scope. It does **not** mark the overall README research goal complete. No tests, simulations, fits, live final accuracy results or timing experiments were run or inspected.

R1 retains **6 cases, 24 state-only validation anchors, 18 frozen models, all 3 optimizer seeds, 72 model/anchor comparisons, 768 chunks and 12,288 fresh continuation families**. Each record retains three horizons, 128 face-gradient components/query and 10 unique symmetric source components/query. Paired refinement uses 64 families/anchor; bootstrap uses 128 whole-family resamples.

Verified **788 copied R1 files**, stored and restored SHA-256 values, report input/output manifests, the identity including its anchor manifest, and the exact case/anchor/seed inventory. Current archived package/R1 source matches in **71 files**. Raw JVP chunks and checkpoints were not replayed.

The derivative is a pathwise JVP of the stopped finite-Euler map, including interpolated crossing and killed-integral exposure. The covariance source uses expected face derivatives in normalized mean units. Independent-half cross-Gram removes quadratic plug-in bias only if differentiation and expectation may be interchanged; it may be indefinite and is not repaired.

Unproven assumptions remain explicit: derivative interchange, near-hit sensitivity variance, stopping/time bias, simultaneous coverage, all-state accuracy, covariance-head accuracy, learning convergence, neural speedup and continuum DK convergence. Positive norm ranges are descriptive, not significance tests.

Both local thesis chapter hashes match the recorded revised snapshot at commit `9afa694a76ba54a3c1ab9422e5bbb0f839e7526f`; both remain locally modified. The DK guide credits the thesis and all three coauthors, cites Dean and the ill-posedness results, and separates conditional continuum theory from the stopped finite-volume law. The [primary arXiv record](https://arxiv.org/abs/2502.20929) confirms the coauthors and publisher DOI `10.1016/j.matpur.2026.103937`.

The original 15-fit archive verifies all **26 listed payloads**, its exact 27-file inventory including the manifest, and unchanged manifest SHA `9950709b82101febaf3ca670046af63a64209efa283235ed0ff5cfff9b905099`. MIT code/CC BY 4.0 authored-documentation and synthetic-evidence coverage, third-party exclusions, CFF/repository metadata and full-history Python 3.12 locked Ruff/pytest/build CI are present. Required historical-loader commits remain ancestors.

Copied clean full-history clone/build/install/smoke evidence at `0112776` and all 10 successful command logs verify by checksum. These commands were not repeated. Original publication is completed historical work; follow-up public-main/Actions verification and unfinished R2-R6 remain separate. No repository recreation, package-registry publishing, website or historical experiment rerun is added.

The JSON companion records absolute evidence paths, actual SHA-256 values, file counts, source identities and the maintained-document snapshot. Later documentation edits require refreshed final-document hashes.
