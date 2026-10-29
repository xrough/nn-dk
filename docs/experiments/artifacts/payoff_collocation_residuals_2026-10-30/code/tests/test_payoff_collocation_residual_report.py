"""Full-entry loss reconstruction, empty evidence and compact residual archives."""

from __future__ import annotations

import copy
import csv
import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_collocation_residuals as auditor  # noqa: E402
from analysis import report_payoff_collocation_residuals as report  # noqa: E402
from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis.payoff_collocation_sampling import BoundaryStrataConfig, boundary_time_strata  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402


def sample_record(tmp_path, *, horizon=0.25):
    n, g, particles = 8, 8, 128
    cfg = BoundaryStrataConfig(horizon, 0.01 / g, 0.05 / g)
    states = np.full((n, g), 1 / g)
    horizons = horizon * np.array([0.0, 0.5, 0.0, 0.125, 0.5, 1.0, 1.0, 1.0])
    for row, distance in (
        (0, 0.0),
        (1, 0.0),
        (3, 0.1 * cfg.boundary_width * np.sqrt(0.125)),
        (4, 2 * cfg.boundary_width * np.sqrt(0.5)),
    ):
        states[row, 0] = cfg.mass_stop + distance
        states[row, 1:] = (1 - states[row, 0]) / (g - 1)
    strata = boundary_time_strata(states, horizons, cfg)
    parent = np.array([f"parent{i // 2}" for i in range(n)])
    rows = {
        "states": states,
        "horizons": horizons,
        "row_ids": np.array([f"row{i}" for i in range(n)]),
        "family_ids": np.array([f"family{i // 2}" for i in range(n)]),
        "parent_family_ids": parent,
        "strata": strata,
    }
    rng = np.random.default_rng(919)
    lmu, running = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
    residual = (np.arange(n) - 2)[:, None] * np.array([1.0, 2.0, 3.0, 4.0])[None]
    mu_dt = lmu + running + residual
    lc = rng.normal(size=(n, 4, 4))
    lc = (lc + lc.swapaxes(-1, -2)) / 2
    raw = rng.normal(size=(n, 4, 3))
    gamma = raw @ raw.swapaxes(-1, -2)
    rc = np.zeros((n, 4, 4))
    rc[:, 0, 1] = 2.0
    rc[:, 1, 0] = 2.0
    cdt = lc + gamma + rc
    native = {
        "mu_dt": mu_dt,
        "Lmu": lmu,
        "running_source": running,
        "rmu": mu_dt - lmu - running,
        "mu_transport": mu_dt - lmu,
        "C_dt": cdt,
        "LC": lc,
        "Gamma": gamma,
        "rC": cdt - lc - gamma,
        "C_transport": cdt - lc,
        "Gamma_common": 0.7 * gamma,
        "rC_common": cdt - lc - 0.7 * gamma,
    }
    scales = np.array([0.5, 1.0, 2.0, 3.0])
    arrays = {**rows, **native}
    for stage, terms in report.TERMS.items():
        physical = (
            scales / np.sqrt(particles) if stage == "mean" else np.outer(scales, scales) / particles
        )
        for term in terms:
            arrays["physical_" + term] = native[term] * physical
            arrays["dimensionless_" + term] = native[term].copy()
    canonical = {
        stage: {
            "natural_mse": float(np.mean(arrays["dimensionless_" + key] ** 2)),
            "natural_huber": float(np.mean(report._huber(arrays["dimensionless_" + key], 3.0))),
        }
        for stage, key in (("mean", "rmu"), ("covariance", "rC"))
    }
    model = {
        "case": "T0p25_k0p75",
        "name": "T0p25_k0p75_uniform_pseudo_huber_seed0",
        "seed": 0,
        "proposal": "uniform",
        "loss": "pseudo_huber",
        "natural_validation": copy.deepcopy(canonical),
    }
    summary = {
        "model_name": model["name"],
        "case": model["case"],
        "rows": n,
        "parent_families": 4,
        "strata_counts": [
            {
                "label": label,
                "rows": int(np.sum(strata == label)),
                "parent_families": len(set(parent[strata == label])),
            }
            for label in report.stratum_names(cfg)
        ],
        "canonical_validation": canonical,
        "strata_config": cfg.to_dict(),
        "observable_labels": [f"output{i}" for i in range(4)],
        "normalization": {
            "horizon": horizon,
            "n_particles": particles,
            "bank_scales": scales.tolist(),
            "mean_scale": [horizon] * 4,
            "covariance_scale": np.full((4, 4), horizon).tolist(),
            "pseudo_huber_delta": 3.0,
        },
        "maximum_decomposition_error": {"mean": 0.0, "covariance": 0.0},
        "weights_and_buffers_unchanged": True,
        "covariance_canonical_entries": 16,
        "common_source_model": model["name"],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    row_path = tmp_path / "rows.npz"
    np.savez_compressed(row_path, **rows)
    component_path = tmp_path / "components.npz"
    np.savez_compressed(
        component_path,
        row_ids=rows["row_ids"],
        **{k: v for k, v in arrays.items() if k not in rows},
    )
    chunk = tmp_path / "components.json"
    report.write(chunk, {"source": "synthetic components"})
    summary_path = tmp_path / "summary.json"
    report.write(summary_path, summary)
    record = {
        "model": model,
        "summary": summary,
        "rows": {"path": str(row_path), "sha256": report.sha(row_path)},
        "summary_path": str(summary_path),
        "summary_sha256": report.sha(summary_path),
        "array_chunks": [
            {
                "path": str(component_path),
                "sha256": report.sha(component_path),
                "start": 0,
                "count": n,
                "record_path": str(chunk),
                "record_sha256": report.sha(chunk),
            }
        ],
    }
    return record, arrays


def test_full_six_case_inventory_keeps_all_seventy_two_slots(tmp_path, monkeypatch):
    study = runner.StudyPlan()
    records = [{"model": slot} for slot in runner.slots(study)]
    data = {"study": study, "records": records}
    monkeypatch.setattr(
        report, "_auditor", lambda: SimpleNamespace(load_complete=lambda path: data)
    )
    assert len(report.load_complete(tmp_path)["records"]) == 72
    records.pop()
    with pytest.raises(ValueError, match="Every declared"):
        report.load_complete(tmp_path)


def test_exact_occupancy_weighted_mse_and_huber_reconstruct_global_canonical(tmp_path):
    record, arrays = sample_record(tmp_path)
    checked = report.validate_model(record, arrays)
    rows = report.loss_rows(record, arrays, checked)
    for stage in ("mean", "covariance"):
        local = [r for r in rows if r["stage"] == stage and r["stratum"] != "all"]
        assert len(local) == 19
        for metric in ("mse", "huber"):
            contribution = sum(
                r["natural_" + metric + "_contribution"] for r in local if r["metric_defined"]
            )
            assert contribution == pytest.approx(checked["canonical"][stage]["natural_" + metric])
    mean_bins = [
        r["common_unit_mse"]
        for r in rows
        if r["stage"] == "mean" and r["stratum"] != "all" and r["metric_defined"]
    ]
    assert np.mean(mean_bins) != pytest.approx(checked["canonical"]["mean"]["natural_mse"])


def test_component_signed_statistics_full_sixteen_entries_and_unique_plot_weights(tmp_path):
    record, arrays = sample_record(tmp_path)
    checked = report.validate_model(record, arrays)
    rows = list(report.component_rows(record, arrays, checked))
    covariance = [
        r
        for r in rows
        if r["stage"] == "covariance"
        and r["unit"] == "train_normalized"
        and r["term"] == "rC"
        and r["stratum"] == "all"
    ]
    assert len(covariance) == 16
    assert sum(r["natural_mse_contribution"] for r in covariance) == pytest.approx(0.5)
    unique = [r for r in covariance if r["unique_triangle"]]
    assert len(unique) == 10
    assert sum(
        r["plot_multiplicity"] * r["natural_mse_contribution"] for r in unique
    ) == pytest.approx(0.5)
    mean = next(
        r
        for r in rows
        if r["unit"] == "solver_raw"
        and r["term"] == "rmu"
        and r["stratum"] == "all"
        and r["component_i"] == 0
    )
    assert mean["signed_mean"] == pytest.approx(1.5)
    np.testing.assert_allclose(
        [mean["absolute_q50"], mean["absolute_q90"], mean["absolute_q99"]],
        np.quantile(abs(arrays["rmu"][:, 0]), [0.5, 0.9, 0.99]),
    )


def test_empty_strata_are_undefined_not_measured_zero_and_parents_are_grouped(tmp_path):
    record, arrays = sample_record(tmp_path)
    checked = report.validate_model(record, arrays)
    rows = report.loss_rows(record, arrays, checked)
    empty = [row for row in rows if not row["metric_defined"]]
    assert empty and all(row["rows"] == row["parent_families"] == 0 for row in empty)
    assert all(
        row[key] is None
        for row in empty
        for key in (
            "common_unit_mse",
            "common_unit_huber",
            "natural_mse_contribution",
            "natural_huber_contribution",
        )
    )
    assert checked["rows"] == 8 and len(set(checked["parents"])) == 4
    assert (
        sum(r["parent_families"] for r in rows if r["stage"] == "mean" and r["stratum"] != "all")
        > 4
    )


@pytest.mark.parametrize(
    "change",
    [
        "physical",
        "normalization",
        "transport",
        "canonical-mse",
        "canonical-huber",
        "row-id",
        "stratum",
        "count",
        "nonfinite",
        "missing-entry",
    ],
)
def test_inconsistent_rows_units_components_or_canonical_losses_fail(tmp_path, change):
    record, arrays = sample_record(tmp_path)
    if change == "physical":
        arrays["physical_rmu"] *= 2
    elif change == "normalization":
        arrays["dimensionless_rC"] *= 2
    elif change == "transport":
        arrays["C_transport"] = arrays["LC"] + arrays["Gamma"]
    elif change == "canonical-mse":
        record["summary"]["canonical_validation"]["mean"]["natural_mse"] += 0.01
    elif change == "canonical-huber":
        record["model"]["natural_validation"]["covariance"]["natural_huber"] += 0.01
    elif change == "row-id":
        arrays["row_ids"][0] = arrays["row_ids"][1]
    elif change == "stratum":
        arrays["strata"][0] = "interior/zero-time"
    elif change == "count":
        record["summary"]["strata_counts"].pop()
    elif change == "nonfinite":
        arrays["Gamma"][0, 0, 0] = np.nan
    else:
        arrays["LC"] = arrays["LC"][:, :3, :3]
    with pytest.raises(ValueError):
        report.validate_model(record, arrays)


def _test_receipt(root):
    inputs = report.qa.collect_test_inputs()
    commands = []
    for kind, argv in report.qa.TEST_COMMANDS.items():
        log = root / "audit" / (kind + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("20 passed in 1.0s\n" if kind == "pytest" else "All checks passed!\n")
        command = {
            "kind": kind,
            "argv": list(argv),
            "cwd": str(report.qa.REPO.resolve()),
            "exit_code": 0,
            "log_path": str(log.resolve()),
            "log_sha256": report.sha(log),
        }
        if kind == "pytest":
            command["test_count"] = 20
        commands.append(command)
    report.write(
        root / "audit/final_tests.json",
        {
            "kind": report.qa.TEST_GATE_KIND,
            "schema_version": 1,
            "complete": True,
            "source_hash": source_hash(),
            "source_inputs": inputs,
            "before_sha256": stable_hash(inputs),
            "after_sha256": stable_hash(inputs),
            "commands": commands,
            "scope": "synthetic QA receipt format fixture only",
        },
    )


@pytest.fixture
def complete(tmp_path, monkeypatch):
    root, study_root = tmp_path / "residuals", tmp_path / "study"
    study_root.mkdir()
    root.mkdir()
    study = runner.StudyPlan(
        horizons=(0.25,),
        kappas=(0.75,),
        n_cells=8,
        n_particles=128,
        paths_per_group=8,
        refinement_paths=4,
        conditional_anchors=1,
        chunk_paths=4,
    )
    record, arrays = sample_record(root / "data")
    records = []
    for slot in runner.slots(study):
        value = copy.deepcopy(record)
        value["model"].update(slot)
        value["summary"].update(
            model_name=slot["name"],
            case=slot["case"],
            common_source_model=f"{slot['case']}_uniform_pseudo_huber_seed{slot['seed']}",
        )
        checkpoint = study_root / "payoffs" / (slot["name"] + ".pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"local checkpoint, indexed not copied")
        metadata = checkpoint.with_suffix(".json")
        report.write(
            metadata, {"complete": True, "natural_validation": value["model"]["natural_validation"]}
        )
        value["model"].update(
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=report.sha(checkpoint),
            metadata_path=str(metadata),
            metadata_sha256=report.sha(metadata),
        )
        path = root / "models" / (slot["name"] + ".json")
        report.write(path, value["summary"])
        value.update(summary_path=str(path), summary_sha256=report.sha(path))
        records.append(value)
    report.write(study_root / "plan.json", study.to_dict())
    report.write(study_root / "payoffs/fit_lock.json", {"models": [r["model"] for r in records]})
    (study_root / "payoffs/weights.pt").write_bytes(b"large-local-weight-fixture")
    (study_root / "payoffs/states_validation.h5").write_bytes(b"large-local-state-fixture")
    forbidden = study_root / "reference/final.h5"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"DO NOT OPEN FINAL MC DATA")
    helpers = {
        Path(report.__file__).name: report.sha(report.__file__),
        Path(auditor.__file__).name: report.sha(auditor.__file__),
    }
    identity = {"helpers": helpers}
    for key, relative in (
        ("fit_lock_sha256", "payoffs/fit_lock.json"),
        ("data_lock_sha256", "payoffs/data_lock.json"),
        ("flat_pool_lock_sha256", "payoffs/flat_pool_lock.json"),
        ("study_preregistration_sha256", "payoffs/preregistration.json"),
        ("study_plan_sha256", "plan.json"),
    ):
        path = study_root / relative
        if not path.exists():
            report.write(path, {"fixture": "locked input"})
        identity[key] = report.sha(path)
    packet = {"identity": identity, "identity_hash": "synthetic"}
    report.write(root / "plan.json", packet)
    costs = {
        "attempts": [{"finished": True, "complete": True, "wall_seconds": 0.1, "cpu_seconds": 0.1}]
    }
    data = {
        "root": root,
        "study_root": study_root,
        "study": study,
        "plan": packet,
        "fit_lock": {"models": [r["model"] for r in records]},
        "records": records,
        "inputs": {str(root / "plan.json"): report.sha(root / "plan.json")},
        "preparation_costs": costs,
        "execution": {**costs, "failures": []},
    }
    monkeypatch.setattr(
        report,
        "_auditor",
        lambda: SimpleNamespace(
            load_complete=lambda path: data, load_record_arrays=auditor.load_record_arrays
        ),
    )
    _test_receipt(root)
    return data, arrays


def test_actual_tables_figures_manifests_and_strict_replay(complete):
    evidence, _ = complete
    manifest = report.generate(evidence["root"])
    assert (
        manifest["allocation"]["models"] == 12 and manifest["allocation"]["strata_per_model"] == 19
    )
    assert manifest["allocation"]["stratum_loss_rows"] == 12 * 20 * 2
    assert manifest["allocation"]["component_statistic_rows"] == 12 * 20 * 3 * (5 * 4 + 7 * 16)
    output = evidence["root"] / "report"
    with gzip.open(output / "component_statistics.csv.gz", "rt") as source:
        assert (
            sum(1 for _ in csv.DictReader(source))
            == manifest["allocation"]["component_statistic_rows"]
        )
    assert len(list(output.glob("*.png"))) == 3
    assert report.generate(evidence["root"]) == manifest
    assert "cannot create absent" in (output / "report.md").read_text()
    assert report.load_report(output) == manifest


def test_compact_archive_offline_verifies_without_copying_or_reading_final_data(
    complete, tmp_path, monkeypatch
):
    evidence, _ = complete
    report.generate(evidence["root"])
    original_sha = report.sha

    def guarded(path):
        assert not Path(path).is_relative_to(evidence["study_root"] / "reference")
        return original_sha(path)

    monkeypatch.setattr(report, "sha", guarded)
    destination = tmp_path / "archive"
    result = report.archive(evidence["root"], destination)
    assert result["complete"]
    assert (
        not list(destination.rglob("*.npz"))
        and not list(destination.rglob("*.pt"))
        and not list(destination.rglob("*.h5"))
    )
    assert (destination / "code/LICENSING.md").is_file()
    manifest = report.read(destination / "archive_manifest.json")
    assert "study/payoffs/weights.pt" in manifest["raw_inventory"]
    assert not any(name.startswith("study/reference/") for name in manifest["raw_inventory"])
    assert report.verify_archive(destination) == result
    Path(evidence["records"][0]["array_chunks"][0]["path"]).write_bytes(
        b"raw local data no longer available"
    )
    assert report.verify_archive(destination) == result
    extra = destination / "unexpected.txt"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        report.verify_archive(destination)


@pytest.mark.parametrize("change", ["missing-csv", "mutated-csv", "nested-manifest"])
def test_report_numeric_outputs_must_be_complete(complete, change):
    evidence, _ = complete
    report.generate(evidence["root"])
    output = evidence["root"] / "report"
    path = output / "component_statistics.csv.gz"
    if change == "missing-csv":
        path.unlink()
    elif change == "mutated-csv":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        extra = output / "nested/manifest.json"
        extra.parent.mkdir()
        extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        report.load_report(output)


def test_archive_rejects_stale_current_source_test_map(complete, tmp_path):
    evidence, _ = complete
    report.generate(evidence["root"])
    path = evidence["root"] / "audit/final_tests.json"
    receipt = report.read(path)
    receipt["source_inputs"]["analysis/report_payoff_collocation_residuals.py"] = "0" * 64
    report.write(path, receipt)
    with pytest.raises(ValueError, match="current analysis"):
        report.archive(evidence["root"], tmp_path / "rejected")
