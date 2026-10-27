"""Lightweight final claims audit: JSON/CSV only, no numerical solver replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value is True or value == "True"


def number_range(values, factor=1.):
    values = [factor*float(v) for v in values]
    return f"{min(values):.3g}" if min(values) == max(values) else f"{min(values):.3g}–{max(values):.3g}"


def close(left, right, tolerance=1e-10):
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((left, right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/payoff-regimes-2026-10-27"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-generator")
    args = parser.parse_args()
    root, repo = args.run.resolve(), args.repo.resolve()
    report = root/"report"
    manifest_path = report/"manifest.json"
    manifest = read(manifest_path)
    document = repo/"docs/experiments/payoff_regimes_2026-10-27.md"
    readme = repo/"README.md"
    doc, introduction = document.read_text(), readme.read_text()
    current = introduction.split("## Current evidence\n", 1)[1].split(
        "The [statistics formulation repair]", 1)[0]
    generator = repo/"analysis/report_payoff_regimes.py"
    assert manifest["complete"] and manifest["generator_sha256"] == sha(generator)
    if args.expected_generator:
        assert manifest["generator_sha256"] == args.expected_generator
    source = manifest["source_hash"]
    sys.path.insert(0, str(repo/"src"))
    from nn_dk.data import source_hash
    assert source_hash() == source
    inputs = {str(p): sha(p) for p in (manifest_path, document, readme, generator, Path(__file__),
        root/"payoffs/fit_lock.json", root/"plan.json", root/"comparison/preregistration.json",
        root/"deployment/preregistration.json", root/"comparison/timing.json")}
    for file in (root/"comparison/preregistration.json", root/"deployment/preregistration.json"):
        for path, expected in read(file)["helpers"].items():
            assert sha(path) == expected, path
            inputs[path] = expected
    for name, expected in manifest["outputs"].items():
        path = report/name
        assert sha(path) == expected, path
        inputs[str(path)] = expected
    proof = read(root/"audit/cost_report_audit.json")
    assert proof["complete"] and proof["source_hash"] == source
    unchanged = {}
    for path, expected in proof["inputs"].items():
        if path.endswith(".csv"):
            actual = sha(path)
            assert actual == expected, path
            assert manifest["outputs"][Path(path).name] == actual
            unchanged[Path(path).name] = actual
    assert len(unchanged) == 26
    for name in ("cost_report_audit.json", "metrics_replay.json", "native_reference_candidate_audit.json",
                 "postfit_validation_diagnostic.json", "boundary_corner_control.json", "final_tests.json",
                 "deployment_pretest_independence.json"):
        path = root/"audit"/name
        packet = read(path)
        assert packet["complete"], name
        audit_source = packet.get("source_hash", packet.get("identity", {}).get("source_hash"))
        assert audit_source == source, name
        assert manifest["inputs"][str(path)] == sha(path), name
        inputs[str(path)] = sha(path)
    fit = read(root/"payoffs/fit_lock.json")
    assert fit["complete"] and fit["source_hash"] == source and len(fit["models"]) == 18
    cases = sorted({m["case"] for m in fit["models"]})
    assert len(cases) == 6
    for case in cases:
        assert {m["seed"] for m in fit["models"] if m["case"] == case} == {0, 1, 2}
    original, supplement, primary = [], [], []
    for case in cases:
        for group_path in sorted((root/"comparison/accuracy"/case).glob("*/group.json")):
            group = read(group_path)
            assert group["complete"] and len(group["records"]) == 7
            for entry in group["records"]:
                packet = read(entry["path"])
                assert sha(entry["path"]) == entry["sha256"]
                assert packet["identity"]["source_hash"] == source
                original.append(packet)
        for group_path in sorted((root/"deployment/accuracy"/case).glob("*/group.json")):
            group = read(group_path)
            assert group["complete"] and group["independent_initials"] and len(group["records"]) == 4
            assert group["bootstrap_counts"]["reference_sha256"] != group["bootstrap_counts"]["initial_sha256"]
            for entry in group["records"]:
                packet = read(entry["path"])
                assert sha(entry["path"]) == entry["sha256"]
                assert packet["initial_pool_independent_of_reference"]
                assert packet["reference_and_initial_bootstrap_independent"]
                supplement.append(packet)
    assert len(original) == 252 and len(supplement) == 48
    replacements = {(p["identity"]["regime"], p["identity"]["group"], p["identity"]["name"]): p for p in supplement}
    for packet in original:
        key = tuple(packet["identity"][k] for k in ("regime", "group", "name"))
        if key in replacements:
            primary.append(replacements.pop(key))
        else:
            primary.append(packet)
    assert not replacements and len(primary) == 252
    primary_nn = [p for p in primary if p["method"]["family"] == "repaired"]
    assert len(primary_nn) == 108
    for case in cases:
        for seed in (0, 1, 2):
            assert len([p for p in primary_nn if p["identity"]["regime"] == case and p["method"]["seed"] == seed]) == 6
    assert manifest["models"] == 18 and manifest["accuracy_records"] == 252
    assert manifest["total_retained_accuracy_records"] == 300 and manifest["independent_initial_records"] == 48
    assert manifest["primary_NN_ensemble_initials_independent"] and not manifest["test_based_refitting"]
    accuracy, summaries, consistency = rows(report/"accuracy.csv"), rows(report/"model_summary.csv"), rows(report/"seed_consistency.csv")
    assert len(accuracy) == 756 and len(consistency) == 216
    assert sum(truth(r["all_seeds_NN_closer_pointwise95"]) for r in consistency) == 0
    assert sum(truth(r["all_seeds_LGA_closer_pointwise95"]) for r in consistency) == 57
    assert "Of 216 fixed group/horizon/statistic comparisons, **0 favor the NN for all three optimizer seeds**" in doc
    assert "57 favor the fine Gaussian approximation for all seeds" in doc
    assert "216 group/horizon/statistic comparisons" in current and "57 favor the fine Gaussian" in current
    for case in cases:
        for family, name in (("repaired", None), ("untrained-repaired", None), ("LGA", "LGA_dt7"),
                             ("LGA", "LGA_dt10"), ("native-MC", None)):
            selected = [r for r in summaries if r["regime"] == case and r["family"] == family
                        and (name is None or r["name"] == name)]
            assert selected
            expected = "| "+" | ".join([case, name or family,
                number_range([r["worst_mean"] for r in selected], 100),
                number_range([r["worst_covariance"] for r in selected], 100),
                number_range([r["worst_covariance_plus_sampling95"] for r in selected], 100)])+" |"
            assert expected in doc, expected
    strongest = [r for r in summaries if r["regime"] == "T1_k6"]
    strong_nn = [r for r in strongest if r["family"] == "repaired"]
    strong_lga = next(r for r in strongest if r["name"] == "LGA_dt10")
    close(min(float(r["worst_mean"]) for r in strong_nn), 5.850639716956288)
    close(max(float(r["worst_mean"]) for r in strong_nn), 6.3035595812395515)
    close(strong_lga["worst_mean"], .2517405946192094)
    assert "5.85–6.3 reference standard deviations" in doc and "94.5–95.2% versus 18.2%" in current
    component = max((r for r in rows(report/"components.csv") if r["regime"] == "T1_k6" and r["family"] == "repaired"),
                    key=lambda r: abs(float(r["normalized_mean_difference"])))
    assert component["observable"] == "killed-integral-cos1-over-T" and component["group"] == "nonuniform__conditional__0"
    assert float(component["time"]) == 1
    assert f"prediction {float(component['candidate_mean']):.6g}, reference {float(component['reference_mean']):.6g}" in doc
    stopping = rows(report/"stopping.csv")
    final_stops = {r["group"]: int(r["stopped"]) for r in stopping if r["regime"] == "T1_k6" and int(r["time_index"]) == 2}
    assert final_stops == {"nonuniform__conditional__0": 105, "nonuniform__conditional__1": 1,
        "nonuniform__ensemble__0": 246, "uniform__conditional__0": 0, "uniform__conditional__1": 0,
        "uniform__ensemble__0": 0}
    assert all(int(r["stopped"]) == 0 for r in stopping if r["regime"] != "T1_k6")
    assert "No stopping was observed in the other 5 regimes" in doc
    costs, totals, amortization = rows(report/"costs.csv"), rows(report/"total_costs.csv"), rows(report/"amortization.csv")
    timing = read(root/"comparison/timing.json")
    assert len(costs) == len(timing["records"]) == manifest["timing_records"] == 264
    assert len(totals) == 1056 and len(amortization) == 432
    assert all(not r["accuracy_qualified_payback_calls"] for r in amortization)
    assert manifest["qualified_payback_rows"] == 0 and "**0 qualified payback rows**" in doc
    assert timing["complete"] and timing["final_timing_after_all_workers_finished"] and timing["environment"]["cpu_threads"] == 1
    for case in cases:
        for family, name in (("repaired", None), ("LGA", "LGA_dt7"), ("LGA", "LGA_dt10"), ("native-MC", None)):
            selected = [r for r in costs if r["regime"] == case and r["family"] == family
                        and (name is None or r["name"] == name)]
            conditional = [r for r in selected if r["role"] == "conditional" and (int(r["batch"]) == 1 or family == "native-MC")]
            ensemble = [r for r in selected if r["role"] == "ensemble"]
            expected = "| "+" | ".join([case, name or family,
                number_range([r["median_seconds"] for r in conditional], 1000),
                number_range([r["median_seconds"] for r in ensemble], 1000),
                number_range([r["training_seconds"] for r in selected])])+" |"
            assert expected in doc, expected
    fitting = sum(m["training_seconds"] for m in fit["models"])
    assert f"**{fitting:.1f} model fitting seconds**" in doc
    workers = rows(report/"workers.csv")
    worker_cpu = sum(float(r["cpu_seconds"]) for r in workers)
    worker_longest = max(float(r["wall_seconds"]) for r in workers)
    worker_span = max(float(r["finished_unix"]) for r in workers)-min(float(r["started_unix"]) for r in workers)
    data_seconds = sum(float(r["generation_seconds"]) for r in rows(report/"data.csv"))
    reference_seconds = sum(float(r["reference_generation_seconds"]) for r in rows(report/"study_costs.csv"))
    for claim in (f"recorded {worker_cpu:.1f} CPU seconds", f"execution was {worker_longest:.1f} seconds",
                  f"span was {worker_span:.1f} seconds", f"generation recorded {data_seconds:.1f} seconds",
                  f"chunks recorded {reference_seconds:.1f} seconds"):
        assert claim in doc, claim
    validation = rows(report/"validation_residuals.csv")
    for case in cases:
        selected = [r for r in validation if r["regime"] == case and r["stratum"] == "all"]
        expected = "| "+" | ".join([case,
            number_range([r["normalized_rms"] for r in selected if r["stage"] == "mean"]),
            number_range([r["normalized_rms"] for r in selected if r["stage"] == "covariance"])])+" |"
        assert expected in doc, expected
    concentration = [r for r in rows(report/"validation_loss_concentration.csv") if r["regime"] == "T1_k6"
                     and r["stratum"] == "minimum_mass/five_to_ten_thresholds"]
    assert {int(r["rows"]) for r in concentration} == {88}
    for stage in ("mean", "covariance"):
        assert number_range([r["fraction_of_full_squared_loss"] for r in concentration if r["stage"] == stage], 100)+"%" in doc
    corner = read(root/"audit/boundary_corner_control.json")
    assert corner["barrier"] == .5
    assert max(r["maximum_covariance_identity_error"] for r in corner["records"]) < 2e-14
    assert max(r["maximum_mean_identity_error"] for r in corner["records"]) < 2e-14
    metrics = {"conditional_NN_ms": [1000*float(r["median_seconds"]) for r in costs
                if r["family"] == "repaired" and r["role"] == "conditional" and int(r["batch"]) == 1],
        "ensemble_NN_ms": [1000*float(r["median_seconds"]) for r in costs if r["family"] == "repaired" and r["role"] == "ensemble"],
        "coarse_LGA_ms": [1000*float(r["median_seconds"]) for r in costs if r["name"] == "LGA_dt7"],
        "native_MC_seconds": [float(r["median_seconds"]) for r in costs if r["family"] == "native-MC"],
        "training_seconds": [m["training_seconds"] for m in fit["models"]]}
    scopes = ["does not predict stopping probabilities or sample paths", "not generalization to unseen physical parameters",
        "does not certify arbitrary unseen histograms", "do not establish generalization to new initial laws",
        "do not constitute a simultaneous grid-wide inference", "not be a continuous-DK theorem",
        "continuous boundary-hitting bias", "could be memoized by any method"]
    assert all(scope in doc for scope in scopes)
    assert "no accuracy-qualified NN cost crossover is established" in current
    tests = read(root/"audit/final_tests.json")
    assert tests["full_suite"]["passed"] == 1110 and tests["final_report_tests"]["passed"] == 10
    with_expected_links = (report/"report.md").read_text()
    for path in report.glob("*.png"):
        with_expected_links = with_expected_links.replace(
            f"]({path.name})", f"](artifacts/payoff_regimes_2026-10-27/report/{path.name})")
    assert doc.startswith(with_expected_links)
    result = {"complete": True, "kind": "final-json-csv-documentation-claims-review-v1", "source_hash": source,
        "report_generator_sha256": sha(generator), "report_manifest_sha256": sha(manifest_path),
        "documentation_sha256": sha(document), "README_sha256": sha(readme), "script_sha256": sha(__file__),
        "inputs": inputs, "original_accuracy_records": 252, "independent_initial_records": 48,
        "retained_records": 300, "primary_records": 252, "all_fitted_models": 18,
        "all_seed_comparison_cells": 216, "all_seed_NN_closer_cells": 0, "all_seed_LGA_closer_cells": 57,
        "cost_CSVs_unchanged_from_independent_audit": unchanged,
        "training_seconds": fitting, "reported_timing_ranges": {k: [min(v), max(v)] for k,v in metrics.items()},
        "recorded_worker_cpu_seconds": worker_cpu, "recorded_worker_longest_seconds": worker_longest,
        "worker_UTC_span_seconds": worker_span, "recorded_state_generation_seconds": data_seconds,
        "recorded_reference_chunk_seconds": reference_seconds,
        "strongest_case": {"NN_mean_SD": [min(float(r["worst_mean"]) for r in strong_nn), max(float(r["worst_mean"]) for r in strong_nn)],
            "NN_covariance_normalized": [min(float(r["worst_covariance"]) for r in strong_nn), max(float(r["worst_covariance"]) for r in strong_nn)],
            "LGA_mean_SD": float(strong_lga["worst_mean"]), "LGA_covariance_normalized": float(strong_lga["worst_covariance"])},
        "stopping_counts_T1_k6": final_stops, "all_qualifications_preserved": scopes,
        "no_source_or_report_change": True, "no_simulation_model_or_bootstrap_replay": True,
        "limitations": "Checks final frozen JSON/CSV/prose and existing independent audits; does not rerun solvers or establish continuum convergence."}
    destination = root/"audit/final_report_claims.json"
    destination.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "path": str(destination), "sha256": sha(destination),
                      "report_generator_sha256": sha(generator), "unchanged_CSVs": len(unchanged)}))


if __name__ == "__main__":
    main()
