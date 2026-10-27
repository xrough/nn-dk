#!/usr/bin/env python3
"""Estimate conditional-reference Monte Carlo noise from a completed study.

This is a posthoc precision diagnostic.  It does not refit models, create paths,
change primary errors, or turn noise subtraction into accuracy certification.
Requires an unchanged, already-audited study_summary.json snapshot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_SUMMARY_PATH = Path(__file__).with_name("summarize_bsde_study.py")
_SPEC = importlib.util.spec_from_file_location("_bsde_study_summary_helpers", _SUMMARY_PATH)
_SUMMARY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUMMARY)
_read, _require = _SUMMARY._read, _SUMMARY._require
_sha, _stable_hash = _SUMMARY._sha, _SUMMARY._stable_hash


def estimate_reference_noise(samples, base):
    """Plug-in noise of sample means and unbiased sample variances at fixed m0."""
    samples = np.asarray(samples)
    _require(samples.ndim == 4 and samples.shape[1] >= 2
             and samples.shape[2:] == (len(base.query_times), 4)
             and np.all(np.isfinite(samples)), "Finite anchor/child/time/four-mode reference bank required")
    count = samples.shape[1]
    std = np.asarray(base.observable_scale)
    values = np.sqrt(base.n_particles)*samples/std
    centered = values-values.mean(axis=1, keepdims=True)
    variance = np.var(values, axis=1, ddof=1)
    fourth = np.mean(centered**4, axis=1)
    mean_mse = variance/count
    variance_mse = (fourth-(count-3)/(count-1)*variance**2)/count
    # Nonnegative algebraically; this maximum tolerates floating-point roundoff.
    _require(np.min(variance_mse) >= -1e-12, "Negative plug-in sample-variance uncertainty")
    variance_mse = np.maximum(variance_mse, 0.)
    return {"n_fixed_anchors": len(samples), "independent_children_per_anchor": count,
            "mean_rmse_noise_estimate": float(np.sqrt(mean_mse.mean())),
            "variance_rmse_noise_estimate": float(np.sqrt(variance_mse.mean())),
            "mean_reference_standard_error_by_anchor_time_mode": np.sqrt(mean_mse).tolist(),
            "variance_reference_standard_error_by_anchor_time_mode": np.sqrt(variance_mse).tolist(),
            "by_time": [{"time": time,
                         "mean_rmse_noise_estimate": float(np.sqrt(mean_mse[:, index].mean())),
                         "variance_rmse_noise_estimate": float(np.sqrt(variance_mse[:, index].mean()))}
                        for index, time in enumerate(base.query_times)]}


def validate_snapshot_hashes(directory, summary):
    """A precision diagnostic must use the exact artifacts previously audited."""
    for relative, expected in summary["provenance"]["input_sha256"].items():
        path = Path(directory)/relative
        _require(path.is_file() and _sha(path) == expected,
                 f"An audited study artifact changed before the precision diagnostic: {relative}")


def diagnose(directory, output=None, *, allow_smoke=False):
    from nn_dk.bsde_study_evaluation import _read_reference_samples

    directory = Path(directory).resolve()
    manifest = _read(directory/"manifest.json")
    _require(manifest.get("status") == "complete",
             "Reference-noise diagnostic requires status='complete'; held-out outcomes remain unopened")
    _require(manifest.get("workflow") == "conditional-bsde-anchor-study-v1",
             "Expected the conditional BSDE study workflow")
    study, preregistration = _SUMMARY._validate_design(directory, manifest, allow_smoke)
    base = study.base_config
    summary_path = directory/"report"/"study_summary.json"
    summary = _read(summary_path)
    _require(summary.get("status") == "complete-artifact-summary"
             and summary.get("source_hash") == manifest["source_hash"]
             and summary.get("study_config_hash") == study.config_hash
             and summary.get("model_count") == study.expected_models,
             "A matching complete artifact summary is required before the precision diagnostic")
    _require(_SUMMARY._HELPER._source_hash() == manifest["source_hash"]
             and _sha(_SUMMARY_PATH) == summary["provenance"]["analysis_script_sha256"]
             and _sha(_SUMMARY._HELPER_PATH) == summary["provenance"]["analysis_helpers_sha256"],
             "Numerical source or the audited analysis implementation changed")
    validate_snapshot_hashes(directory, summary)
    profiles, input_hashes = {}, {}
    for case in base.cases:
        bank = _read_reference_samples(directory/"common", base, case, "conditional")
        paths = sorted((directory/"common"/"evaluation_cache"/case/"conditional").glob("chunk_*.npz"))
        hashes = {str(path.relative_to(directory)): _sha(path) for path in paths}
        input_hashes.update(hashes)
        profiles[case] = {"bank_files_sha256": _stable_hash(hashes), "bank_chunk_count": len(paths),
                          "levels": {level: estimate_reference_noise(bank[level], base)
                                     for level in ("coarse", "fine")}}
    result = {"schema_version": 1, "kind": "conditional-reference-monte-carlo-noise-diagnostic-v1",
        "source_hash": manifest["source_hash"], "study_config_hash": study.config_hash,
        "base_config_hash": base.config_hash, "test_bank_ids": preregistration["test_bank_ids"],
        "query_times": list(base.query_times),
        "observable_order": ["sqrt(2) cos(x)", "sqrt(2) sin(x)", "sqrt(2) cos(2x)", "sqrt(2) sin(2x)"],
        "normalization": "Y = sqrt(N) times physical cell-integrated observable divided by sinc(k/G); means and variances use these fixed fluctuation units",
        "formulas": {"sample_variance": "s2 = sum_c (Y_c - mean(Y))^2 / (C-1)",
            "mean_sampling_variance": "estimated Var(mean(Y)) = s2 / C",
            "variance_sampling_variance": "Var(s2) = [mu4 - ((C-3)/(C-1))*sigma^4] / C; plug in mu4=mean_c((Y_c-mean(Y))^4), sigma^4=(s2)^2",
            "reported_rmse": "sqrt(average estimated sampling variance over all fixed anchors, query times and four modes)"},
        "profiles": profiles,
        "provenance": {"analysis_script_sha256": _sha(Path(__file__)),
            "audited_summary_sha256": _sha(summary_path),
            "audited_summary_script_sha256": _sha(_SUMMARY_PATH),
            "reference_bank_inputs_sha256": input_hashes,
            "all_audited_snapshot_inputs_rechecked": True},
        "limits": ["Posthoc plug-in finite-sample diagnostics, not confidence intervals or certified error floors.",
            "The variance formula is exact with population central moments; empirical substitution introduces finite-sample bias.",
            "Model error and reference noise add only in expected squared error under independence; realized RMSEs cannot be separated exactly.",
            "No noise-corrected accuracy certification or model selection is performed.",
            "Coarse and fine references share Brownian families and are not extra independent replicas.",
            "These quantities omit neural training uncertainty, fixed-anchor population generalization, stopping approximation and residual reference timestep bias."]}
    output = Path(output).resolve() if output else directory/"report"/"reference_noise_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(output)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"reference_noise_diagnostic": diagnose(args.directory, args.output,
                                                             allow_smoke=args.allow_smoke)}, indent=2))


if __name__ == "__main__":
    main()
