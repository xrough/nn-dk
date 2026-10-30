"""Signed finite-grid gradient/source contrasts for a completed frozen audit.

Positive norms have a positive noise floor. They are descriptive distances,
never significance tests. Signed componentwise intervals below are pointwise
percentile bootstrap diagnostics, without multiplicity or continuum guarantees.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import audit_payoff_mean_gradients as audit  # noqa: E402

KIND = "signed-componentwise-frozen-mean-gradient-source-report-v1"
LIMITATIONS = [
    "Pointwise percentile bootstrap intervals are not simultaneous confidence bands; exclusion counts do not certify a grid-wide bias or accuracy statement.",
    "Frobenius/RMS distances are descriptive. A positive norm or positive lower percentile of a norm is never evidence of a resolved discrepancy.",
    "The plug-in Gram of a Monte Carlo mean has positive quadratic sampling bias. The independent-half symmetric cross-Gram targets the source without that plug-in bias, assuming differentiation under expectation is valid; it may be indefinite and is never projected PSD.",
    "Coarse/fine derivative and cross-source contrasts use paired whole-family resampling. A finite refinement difference does not enclose remaining time-discretization, stopping, or derivative-interchange bias.",
    "Near-hit pathwise sensitivities can have heavy tails. Finite bootstrap samples and common-noise finite differences do not certify their variance or asymptotic coverage.",
    "Anchors come from state-only validation pools selected before new continuations. These diagnostics do not assess a new independent initial-state population.",
    "This report studies the frozen mean and its covariance source. It makes no covariance-head prediction-accuracy, fitting-convergence, neural-speedup, or continuum-DK claim.",
]


def signed_contrast(point, bootstrap, *, reference=None):
    """Signed estimates and pointwise intervals, with no norm-based decision.

    With a fixed candidate, point is candidate-reference_estimate and bootstrap
    contains candidate-reference_bootstrap. For paired refinement both inputs
    are the fine-minus-coarse differences using the same resampled families.
    """
    point, bootstrap = np.asarray(point, dtype=float), np.asarray(bootstrap, dtype=float)
    if (bootstrap.ndim != point.ndim+1 or bootstrap.shape[1:] != point.shape
            or len(bootstrap) < 2 or not np.isfinite(point).all() or not np.isfinite(bootstrap).all()):
        raise ValueError("Compatible finite signed point and bootstrap component arrays required")
    interval = np.quantile(bootstrap, [.025, .975], axis=0)
    excluded = (interval[0] > 0) | (interval[1] < 0)
    result = {"point": point, "pointwise_interval": interval, "pointwise_excludes_zero": excluded}
    if reference is not None:
        reference = np.asarray(reference)
        if reference.shape != point.shape:
            raise ValueError("Reference and contrast component shapes differ")
    return result


def _check_shape(values, key, shape):
    if key not in values or values[key].shape != shape or not np.isfinite(values[key]).all():
        raise ValueError(f"Incomplete or incompatible numeric allocation: {key}")
    return values[key]


def _input_data(root):
    """Completion/hash guards run before summaries, records or bootstrap data."""
    root = Path(root).resolve()
    summary = audit.load_complete(root)
    prepared, index = audit.read(root/"plan.json"), audit.read(root/"index.json")
    if (summary.get("identity_hash") != prepared["identity_hash"]
            or index.get("identity_hash") != prepared["identity_hash"]
            or not summary.get("complete")):
        raise ValueError("Diagnostic summary and frozen preparation differ")
    evidence = summary["numeric_evidence"]
    if audit.sha(evidence["path"]) != evidence["sha256"]:
        raise ValueError("Diagnostic numeric evidence changed")
    with np.load(evidence["path"]) as store:
        values = {key: store[key] for key in store.files}
    return root, summary, prepared, index, values


def _allocation(summary, prepared, index, values):
    plan = prepared["identity"]["plan"]
    models = prepared["identity"]["models"]
    cases = tuple(prepared["identity"]["cases"])
    if set(cases) != set(prepared["anchors"]) or len(set(cases)) != len(cases):
        raise ValueError("Incomplete case allocation")
    expected, shapes, expected_arrays = set(), {}, set()
    for case in cases:
        selected = [item for item in models if item["case"] == case]
        if {item["seed"] for item in selected} != {0, 1, 2} or len(selected) != 3:
            raise ValueError("All three frozen optimizer seeds must be retained")
        item = prepared["anchors"][case]
        if audit.sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen anchor evidence changed")
        with np.load(item["path"]) as store:
            states = store["states"]
        if states.ndim != 2 or len(states) != item["rows"]:
            raise ValueError("Incomplete anchor allocation")
        queries = np.asarray(item["queries"])
        q, g, b = len(queries), states.shape[1], plan["n_bootstrap"]
        shapes[case] = (q, g, b)
        for anchor in range(len(states)):
            key = f"{case}_anchor{anchor}"
            for suffix, shape in (("reference_mean_gradient", (q, 4, g)),
                                  ("reference_plugin_source", (q, 4, 4)),
                                  ("reference_cross_source", (q, 4, 4)),
                                  ("bootstrap_mean_gradient", (b, q, 4, g)),
                                  ("bootstrap_cross_source", (b, q, 4, 4))):
                name = key+"_"+suffix
                _check_shape(values, name, shape)
                expected_arrays.add(name)
            for seed in (0, 1, 2):
                expected.add((case, anchor, seed))
                name = key+f"_seed{seed}_gradient"
                _check_shape(values, name, (q, 4, g))
                expected_arrays.add(name)
            if plan["refinement_paths"]:
                for suffix, shape in (("gradient_difference", (q, 4, g)),
                                      ("cross_source_difference", (q, 4, 4)),
                                      ("paired_bootstrap_gradient_difference", (b, q, 4, g)),
                                      ("paired_bootstrap_cross_source_difference", (b, q, 4, 4))):
                    name = key+"_refinement_"+suffix
                    _check_shape(values, name, shape)
                    expected_arrays.add(name)
    actual = [(row["case"], row["anchor"], row["seed"]) for row in summary["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated case/anchor/optimizer-seed allocation")
    if set(values) != expected_arrays:
        raise ValueError("Incomplete or unexpected numeric evidence inventory")
    expected_chunks = {(case, anchor, start) for case, item in prepared["anchors"].items()
                       for anchor in range(item["rows"])
                       for start in range(0, plan["paths"], plan["chunk_paths"])}
    actual_chunks = [(row["case"], row["anchor"], row["start"]) for row in index["chunks"]]
    if len(actual_chunks) != len(expected_chunks) or set(actual_chunks) != expected_chunks:
        raise ValueError("Incomplete or duplicated continuation chunk allocation")
    # Reconstruct saved means and source estimators directly from complete raw
    # family JVPs. This also proves the pairing/allocation used by the report.
    all_ids = set()
    family_audit = []
    for case, item in prepared["anchors"].items():
        q, g, _ = shapes[case]
        for anchor in range(item["rows"]):
            rows = sorted((row for row in index["chunks"] if row["case"] == case
                           and row["anchor"] == anchor), key=lambda row: row["start"])
            gradients, stops, ids = [], [], []
            for row in rows:
                if audit.sha(row["path"]) != row["sha256"]:
                    raise ValueError("Committed raw JVP evidence changed")
                with np.load(row["path"]) as store:
                    value = store["gradient"]
                    families, stop = store["family_ids"], store["stopped_at"]
                if (value.shape != (plan["chunk_paths"], q, 4, g)
                        or families.shape != (plan["chunk_paths"],) or stop.shape != families.shape
                        or not np.isfinite(value).all() or np.isnan(stop).any() or np.any(stop < 0)):
                    raise ValueError("Incompatible complete raw-family JVP allocation")
                gradients.append(value)
                stops.append(stop)
                ids.extend(families.tolist())
            if len(ids) != plan["paths"] or len(set(ids)) != len(ids) or all_ids.intersection(ids):
                raise ValueError("Continuation families missing, duplicated or reused across anchors")
            all_ids.update(ids)
            mean, plugin, cross = audit.source_estimates(np.concatenate(gradients))
            key = f"{case}_anchor{anchor}"
            for name, actual_value in (("reference_mean_gradient", mean),
                                       ("reference_plugin_source", plugin),
                                       ("reference_cross_source", cross)):
                np.testing.assert_allclose(actual_value, values[key+"_"+name], atol=2e-14, rtol=2e-13)
            family_audit.append({"case": case, "anchor": anchor, "families": len(ids),
                                 "stopped_families": int(np.isfinite(np.concatenate(stops)).sum())})
    return shapes, family_audit


def _component_summary(contrast, *, symmetric=False):
    point, excluded = contrast["point"], contrast["pointwise_excludes_zero"]
    if symmetric:
        indices = np.triu_indices(point.shape[-1])
        selected_point, selected_excluded = point[..., indices[0], indices[1]], excluded[..., indices[0], indices[1]]
    else:
        selected_point = point.reshape(point.shape[0], -1)
        selected_excluded = excluded.reshape(point.shape[0], -1)
    return {"absolute_rms_by_query": np.sqrt(np.mean(selected_point**2, axis=-1)).tolist(),
            "maximum_absolute_component_by_query": abs(selected_point).max(-1).tolist(),
            "pointwise_zero_exclusions_by_query": selected_excluded.sum(-1).tolist(),
            "components_per_query": selected_point.shape[-1]}


def _markdown(result):
    lines = ["# Frozen mean-gradient and covariance-source diagnostic", "",
        "This report compares frozen NN mean derivatives with fresh finite-Euler conditional-mean JVP estimates. "
        "Signed componentwise intervals use whole continuation families; all optimizer seeds remain separate.", "",
        "A zero exclusion below is a **pointwise descriptive bootstrap result**, not a simultaneous accuracy or bias certificate. "
        "Positive RMS/Frobenius distances never decide an exclusion.", "",
        "## Allocation and controls", "",
        f"- Cases: {len(result['allocation']['cases'])}; frozen mean models: {result['allocation']['models']}; "
        f"fixed validation anchors: {result['allocation']['anchors']}; retained model/anchor records: {len(result['records'])}.",
        f"- Independent continuation families per anchor: {result['allocation']['paths_per_anchor']}; "
        f"query fractions: {result['allocation']['query_fractions']}; coarse step: {result['allocation']['step']:.9g}.",
        f"- Paired refinement families per anchor: {result['allocation']['refinement_paths']}; "
        f"step refinement factor: {result['allocation']['refinement_factor']}; bootstrap replicates: {result['allocation']['bootstrap_replicates']}.",
        f"- Zero-time face controls: maximum error {result['controls']['zero_time_face_maximum_error']:.6g}; "
        f"absorbed face sources exactly zero: {result['controls']['all_absorbed_sources_zero']}.",
        f"- Native killed-payoff JVP finite-difference check: largest pathwise error "
        f"{result['controls']['finite_difference_pathwise_maximum_error']:.6g}. The derivative includes the stopping-exposure term.", "",
        "## Signed NN-minus-native contrasts", "",
        "Gradient RMS uses all four outputs and native face directions. Source RMS and exclusions count ten unique symmetric entries. "
        "Units are the stored normalized mean noise-face derivative and the source of normalized N-scaled covariance. "
        "The reference is the independent-half cross-Gram, not the quadratically biased plug-in Gram.", "",
        "| Case | Seed | Profile / anchor | Mass / stop | Distance / width | τ | Gradient RMS | Grad exclusions | Source RMS | Source exclusions | Stopped / families |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["records"]:
        for q, time in enumerate(row["queries"]):
            lines.append(f"| {row['case']} | {row['seed']} | {row['profile']} / {row['anchor']} "
                f"({row['boundary_stratum']}) | {row['minimum_mass_over_stopping_mass']:.5g} "
                f"| {row['distance_over_boundary_width']:.5g} | {time:.5g} "
                f"| {row['gradient']['absolute_rms_by_query'][q]:.6g} "
                f"| {row['gradient']['pointwise_zero_exclusions_by_query'][q]}/{row['gradient']['components_per_query']} "
                f"| {row['source']['absolute_rms_by_query'][q]:.6g} "
                f"| {row['source']['pointwise_zero_exclusions_by_query'][q]}/10 "
                f"| {row['stopped_families']}/{row['families']} |")
    lines += ["", "## Paired finite-grid refinement", "",
        "Both grids reuse each physical Brownian family and each bootstrap draw, including the same independent-half partition. "
        "Rows are shared by all optimizer seeds. These finite-grid differences do not bound remaining bias.", "",
        "| Case | Profile / anchor | τ | Gradient change RMS | Grad exclusions | Cross-source change RMS | Source exclusions | Families |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["refinement"]:
        for q, time in enumerate(row["queries"]):
            lines.append(f"| {row['case']} | {row['profile']} / {row['anchor']} | {time:.5g} "
                f"| {row['gradient']['absolute_rms_by_query'][q]:.6g} "
                f"| {row['gradient']['pointwise_zero_exclusions_by_query'][q]}/{row['gradient']['components_per_query']} "
                f"| {row['source']['absolute_rms_by_query'][q]:.6g} "
                f"| {row['source']['pointwise_zero_exclusions_by_query'][q]}/10 | {row['families']} |")
    if not result["refinement"]:
        lines += ["", "No paired refinement families were allocated."]
    lines += ["", "## Interpretation and limits", ""]
    lines += ["- "+value for value in LIMITATIONS]
    lines += ["", "All signed estimates, interval endpoints and component masks are in `contrasts.npz`; "
              "`report.json` retains model/anchor/boundary metadata and the full input identity. "
              "`output_manifest.json` verifies these outputs and their frozen inputs.", ""]
    return "\n".join(lines)


def report(root, output=None):
    root, summary, prepared, index, values = _input_data(root)
    shapes, family_audit = _allocation(summary, prepared, index, values)
    output = (root/"report" if output is None else Path(output).resolve())
    if output == root or output in [Path(v["path"]).parent for v in prepared["anchors"].values()]:
        raise ValueError("Report outputs must not replace input directories")
    plan = prepared["identity"]["plan"]
    inputs = {str(root/name): audit.sha(root/name) for name in ("plan.json", "index.json", "summary.json")}
    inputs[summary["numeric_evidence"]["path"]] = summary["numeric_evidence"]["sha256"]
    identity = {"kind": KIND, "audit_root": str(root), "audit_identity_hash": prepared["identity_hash"], "inputs": inputs,
                "report_script_sha256": audit.sha(__file__), "audit_script_sha256": audit.sha(audit.__file__)}
    manifest_path = output/"output_manifest.json"
    if manifest_path.exists():
        manifest = audit.read(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Report input or implementation identity changed; use a new output directory")
        return load_report(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty report directory")
    records, refinements, arrays = [], [], {}
    family_rows = {(r["case"], r["anchor"]): r for r in family_audit}
    for row in summary["records"]:
        case, anchor, seed = row["case"], row["anchor"], row["seed"]
        key = f"{case}_anchor{anchor}"
        family = family_rows[(case, anchor)]
        if (row["families"] != plan["paths"] or row["stopped_families"] != family["stopped_families"]
                or not np.array_equal(row["queries"], prepared["anchors"][case]["queries"])):
            raise ValueError("Summary row loses queries or whole continuation families")
        gradient = values[key+f"_seed{seed}_gradient"]
        gradient_contrast = signed_contrast(gradient-values[key+"_reference_mean_gradient"],
                                           gradient[None]-values[key+"_bootstrap_mean_gradient"])
        source = gradient @ gradient.swapaxes(-1, -2)
        source_contrast = signed_contrast(source-values[key+"_reference_cross_source"],
                                         source[None]-values[key+"_bootstrap_cross_source"])
        for component, contrast in (("gradient", gradient_contrast), ("source", source_contrast)):
            for suffix, value in contrast.items():
                arrays[key+f"_seed{seed}_{component}_{suffix}"] = value
        keep = ("case", "anchor", "seed", "profile", "queries", "families", "stopped_families", "controls",
                "minimum_mass", "distance_to_boundary", "minimum_mass_over_stopping_mass", "boundary_stratum",
                "boundary_width", "distance_over_boundary_width", "distance_over_moving_width_by_query",
                "selection_rank_fraction", "minimum_mass_rank", "candidate_count",
                "finite_difference_pathwise_maximum_error", "plugin_minus_cross_frobenius_by_query",
                "cross_source_minimum_eigenvalue_by_query")
        record = {name: row[name] for name in keep}
        record.update(gradient=_component_summary(gradient_contrast),
                      source=_component_summary(source_contrast, symmetric=True),
                      array_prefix=key+f"_seed{seed}_")
        records.append(record)
        if seed == 0 and plan["refinement_paths"]:
            prefix = key+"_refinement_"
            ref_grad = signed_contrast(values[prefix+"gradient_difference"],
                                      values[prefix+"paired_bootstrap_gradient_difference"])
            ref_source = signed_contrast(values[prefix+"cross_source_difference"],
                                        values[prefix+"paired_bootstrap_cross_source_difference"])
            for component, contrast in (("gradient", ref_grad), ("source", ref_source)):
                for suffix, value in contrast.items():
                    arrays[prefix+component+"_"+suffix] = value
            refinements.append({"case": case, "anchor": anchor, "profile": row["profile"],
                "queries": row["queries"], "families": plan["refinement_paths"],
                "gradient": _component_summary(ref_grad), "source": _component_summary(ref_source, symmetric=True),
                "array_prefix": prefix, "paired_whole_family_resampling": True})
    records.sort(key=lambda row: (row["case"], row["seed"], row["profile"], row["anchor"]))
    result = {"complete": True, "identity": identity, "allocation": {
        "cases": prepared["identity"]["cases"], "models": len(prepared["identity"]["models"]),
        "anchors": len(family_audit), "paths_per_anchor": plan["paths"], "query_fractions": plan["query_fractions"],
        "step": plan["step"], "refinement_paths": plan["refinement_paths"],
        "refinement_factor": plan["refinement_factor"], "bootstrap_replicates": plan["n_bootstrap"],
        "gradient_components_per_query_by_case": {case: 4*shape[1] for case, shape in shapes.items()},
        "unique_symmetric_source_components_per_query": 10},
        "controls": {"zero_time_face_maximum_error": max(r["controls"]["zero_time_face_maximum_error"] for r in records),
            "all_absorbed_sources_zero": all(r["controls"]["absorbed_face_source_exact_zero"] for r in records),
            "finite_difference_pathwise_maximum_error": max(r["finite_difference_pathwise_maximum_error"] for r in records)},
        "family_audit": family_audit, "records": records, "refinement": refinements,
        "contrasts": "candidate minus native mean/cross-source; refinement fine minus coarse",
        "inference": "pointwise signed percentile intervals only; no norm-based or simultaneous decisions",
        "limitations": LIMITATIONS, "new_predictions_fits_or_simulations": False}
    audit.npz(output/"contrasts.npz", arrays)
    result["numeric_contrasts"] = {"path": str(output/"contrasts.npz"), "sha256": audit.sha(output/"contrasts.npz")}
    audit.write(output/"report.json", result)
    (output/"report.md").write_text(_markdown(result))
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(("case", "seed", "profile", "anchor", "boundary_stratum", "minimum_mass_over_stopping_mass",
        "query", "gradient_rms", "gradient_pointwise_exclusions", "source_rms", "source_pointwise_exclusions",
        "stopped_families", "families", "finite_difference_pathwise_maximum_error"))
    for row in records:
        for q, query in enumerate(row["queries"]):
            writer.writerow((row["case"], row["seed"], row["profile"], row["anchor"], row["boundary_stratum"],
                row["minimum_mass_over_stopping_mass"], query, row["gradient"]["absolute_rms_by_query"][q],
                row["gradient"]["pointwise_zero_exclusions_by_query"][q], row["source"]["absolute_rms_by_query"][q],
                row["source"]["pointwise_zero_exclusions_by_query"][q], row["stopped_families"], row["families"],
                row["finite_difference_pathwise_maximum_error"]))
    (output/"contrast_table.csv").write_text(table.getvalue())
    manifest = {"complete": True, "identity": identity, "outputs": {
        name: audit.sha(output/name) for name in ("report.json", "report.md", "contrasts.npz", "contrast_table.csv")}}
    audit.write(manifest_path, manifest)
    return result


def load_report(output):
    output = Path(output)
    manifest = audit.read(output/"output_manifest.json")
    if (not manifest.get("complete") or manifest["identity"]["report_script_sha256"] != audit.sha(__file__)
            or manifest["identity"]["audit_script_sha256"] != audit.sha(audit.__file__)):
        raise ValueError("Report implementation identity changed or is incomplete")
    audit.load_complete(manifest["identity"]["audit_root"])
    for path, digest in manifest["identity"]["inputs"].items():
        if audit.sha(path) != digest:
            raise ValueError("Frozen report input changed")
    expected = {"report.json", "report.md", "contrasts.npz", "contrast_table.csv"}
    if set(manifest["outputs"]) != expected:
        raise ValueError("Incomplete report output inventory")
    for name, digest in manifest["outputs"].items():
        if audit.sha(output/name) != digest:
            raise ValueError("Report output evidence changed")
    return audit.read(output/"report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(json.dumps({"complete": True, "records": len(result["records"]),
                      "anchors": result["allocation"]["anchors"],
                      "numeric_contrasts": result["numeric_contrasts"]}))


if __name__ == "__main__":
    main()
