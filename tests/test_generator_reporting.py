"""Complete final generator reports retain every seed and deterministic gate."""

import copy
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from nn_dk.generator_config import generator_preset
from nn_dk.generator_evaluation import compare_certified_moments
from nn_dk.generator_reporting import write_generator_report


def records_for(directory, config):
    records = []
    for problem, seed in product(("ou", "dk"), config.optimizer_seeds):
        checkpoint = directory / f"{problem}_seed{seed}.pt"
        checkpoint.write_bytes(f"{problem} optimizer {seed}".encode())
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cases = (("analytic", "conditional"),) if problem == "ou" else product(
            config.cases, ("conditional", "ensemble"))
        for case, scope in cases:
            mean = np.zeros((3, 4))
            covariance = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
            prediction = {"mean": mean + .002, "scaled_covariance": covariance,
                          "target_id": config.target_id(problem), "horizons": config.query_times}
            reference = {"mean": mean, "scaled_covariance": covariance,
                         "mean_error_bound": .001, "scaled_covariance_error_bound": .001,
                         "certificate": {"valid": True, "method": "synthetic unit-test enclosure"}}
            comparison = compare_certified_moments(prediction, reference, n_particles=1, std=np.ones(4))
            records.append({"problem": problem, "seed": seed, "case": case, "scope": scope,
                            "role": "locked-final",
                            "comparison": comparison, "target_id": config.target_id(problem),
                            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha,
                            "training_seconds": 2., "evaluation_seconds": .1})
    return records


@pytest.mark.parametrize("name,count", (("generator-smoke", 5), ("generator-accuracy", 15)))
def test_complete_reports_keep_all_scopes_seeds_and_bounds(tmp_path, monkeypatch, name, count):
    monkeypatch.setattr("nn_dk.generator_reporting.source_hash", lambda: "report-test-source")
    config = generator_preset(name)
    records = records_for(tmp_path, config)
    manifest = {"config_hash": config.config_hash, "source_hash": "report-test-source", "status": "reporting"}
    paths = write_generator_report(tmp_path, config, records, manifest)
    payload = json.loads(Path(paths["json"]).read_text())
    assert payload["all_declared_comparisons_passed"]
    assert len(payload["comparison_rows"]) == count
    assert payload["training_seconds_distinct_models"] == 2 * config.expected_models
    assert {row["optimizer_seed"] for row in payload["comparison_rows"]} == set(config.optimizer_seeds)
    assert {row["scope"] for row in payload["comparison_rows"]} == {"conditional", "ensemble"}
    text = Path(paths["markdown"]).read_text()
    assert "deterministic enclosures, not bootstrap 95% intervals" in text
    assert "not the previous trajectory-BSDE training objective" in text
    assert "no best seed is selected" in text
    with Path(paths["csv"]).open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == count
    assert b"\r\n" not in Path(paths["csv"]).read_bytes()
    assert Path(paths["figure"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_unresolved_final_result_is_reported_without_renaming_it_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("nn_dk.generator_reporting.source_hash", lambda: "report-test-source")
    config = generator_preset("generator-smoke")
    records = records_for(tmp_path, config)
    records[0]["comparison"]["reference_certificate"]["valid"] = False
    records[0]["comparison"]["accuracy_gate"].update(status="unresolved",
        reference_certificate_valid=False, reference_bias_resolved=False)
    manifest = {"config_hash": config.config_hash, "source_hash": "report-test-source"}
    paths = write_generator_report(tmp_path, config, records, manifest)
    payload = json.loads(Path(paths["json"]).read_text())
    assert not payload["all_declared_comparisons_passed"]
    assert payload["comparison_rows"][0]["accuracy_gate"] == "unresolved"


@pytest.mark.parametrize("change", ("missing", "duplicate", "checkpoint", "target", "false-pass", "source",
                                    "missing-time", "large-bound", "development", "nan-bound"))
def test_incomplete_or_false_provenance_rejected_before_export(tmp_path, monkeypatch, change):
    monkeypatch.setattr("nn_dk.generator_reporting.source_hash", lambda: "report-test-source")
    config = generator_preset("generator-smoke")
    records = records_for(tmp_path, config)
    manifest = {"config_hash": config.config_hash, "source_hash": "report-test-source"}
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif change == "checkpoint":
        Path(records[0]["checkpoint_path"]).write_bytes(b"changed")
    elif change == "target":
        records[0]["target_id"] = "Euler-target"
    elif change == "false-pass":
        records[0]["comparison"]["reference_certificate"]["valid"] = False
    elif change == "source":
        manifest["source_hash"] = "old-source"
    elif change == "missing-time":
        records[0]["comparison"]["errors"]["mean_normalized_signed"] = [[0.] * 4] * 2
    elif change == "large-bound":
        records[0]["comparison"]["accuracy_gate"]["maximum_reference_normalized_bound"] = .02
    elif change == "development":
        records[0]["role"] = "development"
    elif change == "nan-bound":
        records[0]["comparison"]["accuracy_gate"]["maximum_reference_normalized_bound"] = float("nan")
    with pytest.raises(ValueError):
        write_generator_report(tmp_path, config, records, manifest)
    assert not (tmp_path / "report").exists()
