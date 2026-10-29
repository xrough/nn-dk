"""All-case report contracts, negative gates and offline compact evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import evaluate_lna_payoff_correction as evaluator  # noqa: E402
from analysis import report_lna_payoff_correction as report  # noqa: E402
from analysis import run_lna_payoff_correction_study as runner  # noqa: E402
from analysis import run_payoff_collocation_study as old_runner  # noqa: E402
from analysis.evaluate_path_generalization import empirical, moment_comparison  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

REAL_LABEL_REPLAY = report._label_replay_audit


def save(path, value):
    report.write(path, value)
    return {"path": str(path), "sha256": report.sha(path)}


def cost_log(path, identity, attempt):
    value = {"identity": identity, "attempts": [attempt]}
    save(path, {**value, "content_sha256": stable_hash(value)})


def final_test_receipt(root, source_inputs=None):
    inputs = report.collect_test_inputs() if source_inputs is None else source_inputs
    commands = []
    for kind, argv in report.TEST_COMMANDS.items():
        log = root / "audit" / f"final_{kind}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("123 passed in 1.00s\n" if kind == "pytest" else "All checks passed!\n")
        command = {
            "kind": kind,
            "argv": list(argv),
            "cwd": str(report.REPO.resolve()),
            "exit_code": 0,
            "log_path": str(log.resolve()),
            "log_sha256": report.sha(log),
        }
        if kind == "pytest":
            command["test_count"] = 123
        commands.append(command)
    value = {
        "kind": report.TEST_GATE_KIND,
        "schema_version": 1,
        "complete": True,
        "source_hash": source_hash(),
        "source_inputs": inputs,
        "before_sha256": stable_hash(inputs),
        "after_sha256": stable_hash(inputs),
        "commands": commands,
        "scope": "synthetic receipt-format fixture only, not real full-suite acceptance",
    }
    save(root / "audit/final_tests.json", value)
    return value


def evaluation_receipt(root, stage, *, whole=True):
    module = report._evaluator()
    path = root / "evaluation/execution" / f"cli_{stage}_90_100.json"
    value = {
        "kind": module.INVOCATION_KIND,
        "schema_version": 1,
        "stage": stage,
        "root": str(root.resolve()),
        "pid": 90,
        "invocation_ns": 100,
        "source_hash": source_hash(),
        "script_sha256": report.sha(module.__file__),
        "finished": True,
        "status": "success",
        "exit_code": 0,
        "started_unix": 1.0,
        "finished_unix": 2.0,
        "wall_seconds": 1.0,
        "cpu_seconds": 0.8,
        "process_cpu_seconds_at_finish": 2.0,
        "whole_process": whole,
        "nonadditive_contains_nested_work": True,
        "args": {},
        "scope": "synthetic process-receipt contract fixture",
    }
    save(path, value)
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root, other = tmp_path / "candidate", tmp_path / "collocation"
    root.mkdir()
    other.mkdir()
    study = runner.StudyPlan(
        train_natural_per_profile=2,
        train_stress_per_profile=1,
        validation_per_profile=1,
        train_continuations=2,
        validation_continuations=2,
        label_chunk=2,
        adam_steps=1,
        lbfgs_steps=0,
        collocation_root=str(other),
    )
    old = old_runner.StudyPlan(paths_per_group=8, refinement_paths=4, chunk_paths=4)
    case_names = runner.cases(study)
    rng = np.random.default_rng(411)
    reference = rng.normal(0, 0.01, (8, 3, 4))
    mean, covariance = empirical(reference)
    counts = rng.multinomial(8, np.full(8, 0.125), size=128)
    bm = np.broadcast_to(mean, (128, 3, 4)).copy()
    bc = np.broadcast_to(old.n_particles * covariance, (128, 3, 4, 4)).copy()
    metrics = moment_comparison(
        reference,
        predicted_mean=mean,
        predicted_scaled_covariance=old.n_particles * covariance,
        bootstrap_predicted_mean=bm,
        bootstrap_predicted_scaled_covariance=bc,
        n_particles=old.n_particles,
        counts=counts,
    )
    arrays = root / "metrics.npz"
    np.savez_compressed(
        arrays,
        mean=mean,
        scaled_covariance=old.n_particles * covariance,
        bootstrap_mean=bm,
        bootstrap_scaled_covariance=bc,
    )
    array_record = {"path": str(arrays), "sha256": report.sha(arrays)}
    models = []
    for slot in runner.slots(study):
        path = root / "payoffs" / slot["case"] / f"seed{slot['seed']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"software-fixture")
        metadata = save(path.with_suffix(".json"), {"complete": True})
        invocation = save(path.with_suffix(".invocations.json"), {"complete": True})
        models.append(
            {
                **slot,
                "checkpoint_path": str(path),
                "checkpoint_sha256": report.sha(path),
                "metadata_path": metadata["path"],
                "metadata_sha256": metadata["sha256"],
                "invocation_record_path": invocation["path"],
                "invocation_record_sha256": invocation["sha256"],
                "full_invocation_cost_complete": True,
                "full_invocation_seconds": 1.0,
                "training_seconds": 0.8,
                "selected_step": {"mean": 1, "covariance": 1},
                "natural_validation": {
                    stage: {"natural_mse": 0.01} for stage in ("mean", "covariance")
                },
            }
        )
    data = {
        "root": root,
        "collocation_root": other,
        "study": study,
        "collocation_study": old,
        "cases": case_names,
        "candidate_lock": {"models": models},
        "joint_lock_sha256": "a" * 64,
        "inputs": {},
        "plan": evaluator.EvaluationPlan(chunk_paths=4),
        "lock": {},
        "packet": {"helpers": {str(Path(report.__file__).resolve()): report.sha(report.__file__)}},
        "groups": [],
        "records": [],
        "baseline": {
            "root": other,
            "study": old,
            "lock": {"models": old_runner.slots(old)},
            "groups": [],
            "records": [],
            "references": {},
        },
        "contrasts": {"records": []},
    }
    for case in case_names:
        cfg = old_runner.reference_configuration(case, old)
        refpath = other / "reference" / case / "reference.h5"
        refpath.parent.mkdir(parents=True)
        refpath.write_bytes(b"raw-indexed-only")
        save(refpath.with_name("manifest.json"), {"status": "complete"})
        data["baseline"]["references"][case] = {
            "status": "complete",
            "reference_sha256": report.sha(refpath),
        }
        methods = evaluator.methods(data, case)
        for group in cfg.group_keys:
            data["groups"].append({"case": case, "group": group})
            data["baseline"]["groups"].append(
                {
                    "regime": case,
                    "group": group,
                    "reference_refinement": {
                        "level0_versus_finest": metrics,
                        "level1_versus_finest": metrics,
                    },
                    "NN_versus_fine_LGA": {},
                }
            )
            for method in methods:
                data["records"].append(
                    {
                        "complete": True,
                        "identity": {
                            "case": case,
                            "group": group,
                            "method": method,
                            "joint_lock_sha256": "a" * 64,
                        },
                        "method": method,
                        "available": True,
                        "all_families_retained": True,
                        "failed_family_ids": [],
                        "metrics": metrics,
                        "initial_pool_independent_of_reference": "__ensemble__" in group,
                        "arrays": array_record,
                    }
                )
            old_methods = (
                [
                    {**m, "family": "repaired"}
                    for m in data["baseline"]["lock"]["models"]
                    if m["case"] == case
                ]
                + [{"name": case + "_untrained", "family": "untrained-repaired"}]
                + [
                    {"name": n, "family": "LGA" if n.startswith("LGA") else "native-MC"}
                    for n in ("LGA_dt7", "LGA_dt10", "native-MC-dt10")
                ]
            )
            for method in old_methods:
                data["baseline"]["records"].append(
                    {
                        "identity": {"regime": case, "group": group, "name": method["name"]},
                        "method": method,
                        "metrics": metrics,
                        "arrays": array_record,
                        "initial_pool_independent_of_reference": True,
                    }
                )
            for model in (m for m in models if m["case"] == case):
                controls = (
                    report.CONTROLS
                    | {"LGA_dt7", "LGA_dt10", "native-MC-dt10"}
                    | {
                        m["name"]
                        for m in data["baseline"]["lock"]["models"]
                        if m["case"] == case and m["seed"] == model["seed"]
                    }
                )
                for control in controls:
                    packet = {
                        "records": [],
                        "both_predictor_initial_pool_uncertainties_included": True,
                    }
                    for t in range(3):
                        item = {"time_index": t}
                        for statistic in ("mean", "covariance"):
                            samples = np.linspace(-0.02, 0.04, 128)
                            item.update(
                                {
                                    statistic + "_error_advantage": 0.01,
                                    statistic + "_advantage_percentile95": np.quantile(
                                        samples, [0.025, 0.975]
                                    ).tolist(),
                                    "bootstrap_" + statistic + "_advantages": samples.tolist(),
                                }
                            )
                        packet["records"].append(item)
                    data["contrasts"]["records"].append(
                        {
                            "case": case,
                            "group": group,
                            "candidate": model["name"],
                            "seed": model["seed"],
                            "control": control,
                            "available": True,
                            "contrast": packet,
                            "pointwise_not_simultaneous": True,
                            "same_initials_assumed": False,
                        }
                    )
            for start in range(0, cfg.paths_per_group if "__ensemble__" in group else 1, 4):
                for level in (7, 10):
                    save(
                        root / "evaluation/baselines" / case / group / f"dt{level}_{start}.json",
                        {
                            "identity": {"joint_lock_sha256": "a" * 64},
                            "baseline_ODE_seconds": 0.1,
                            "arrays": array_record,
                        },
                    )
    jobs = []
    for job in evaluator._timing_slots(data):
        record = {
            "identity": {"job": job, "joint_lock_sha256": "a" * 64},
            "available": True,
            "components": {
                "conditional_LNA_seconds": 0.1,
                "head_seconds": 0.01,
                "initialization_seconds": 0.02,
                "aggregation_seconds": 0.03,
            },
            "timing_mode": "uninterrupted-full-uncached-workload-v1",
            "workload_seconds": 0.17,
            "component_sum_seconds": 0.16,
            "paid_attempt_seconds": 0.18,
            "attempt_cost_complete": True,
            "cold_end_to_end_seconds": 0.18 if job["phase"] == "cold" else None,
            "all_families_retained": True,
            "full_uncached_LNA_per_histogram": True,
            "accuracy_cache_used": False,
            "restoration_and_resume_attempts": [
                {"restoration_seconds": 0.01, "warm_preparation_seconds": 0.0}
            ],
        }
        entry = save(root / "evaluation/timing" / (job["slot"] + "/record.json"), record)
        jobs.append({"slot": job["slot"], **entry})
    data["timing"] = {
        "complete": True,
        "jobs": jobs,
        "external_competing_heavy_work_finished_attested": True,
        "environment": {"cpu_threads": 1},
    }
    for stage, seconds in (("anchors", 2.0), ("labels-and-baselines", 10.0)):
        name = "prepare" if stage == "anchors" else "data"
        cost_log(
            root / "execution" / (name + "_costs.json"),
            {"plan_hash": study.plan_hash, "stage": stage},
            {
                "complete": True,
                "pid": 10 if name == "prepare" else 11,
                "wall_seconds": seconds,
                "cpu_seconds": seconds,
            },
        )
    for case in case_names:
        for role in ("train", "validation"):
            count = 6 if role == "train" else 2
            chunks = []
            directory = root / "data" / case / role
            for relative in (
                "anchors/record.json",
                "anchors/arrays.npz",
                "bank/record.json",
                "bank/arrays.npz",
            ):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic indexed input")
            for i in range(count):
                path = directory / "labels" / str(i)
                entry = save(path / "record.json", {"costs": {"wall_seconds": 0.01}})
                (path / "arrays.npz").write_bytes(b"raw-only")
                chunks.append(
                    {
                        "path": str(path),
                        "record_sha256": entry["sha256"],
                        "arrays_sha256": report.sha(path / "arrays.npz"),
                        "families": 2,
                        "stopped_families": 1,
                    }
                )
            save(
                directory / "label_manifest.json",
                {
                    "complete": True,
                    "anchors": count,
                    "paths_per_anchor": 2,
                    "all_stopped_paths_retained": True,
                    "unbiased_sample_covariance": True,
                    "chunks": chunks,
                },
            )
            path = directory / "baseline"
            entry = save(path / "record.json", {"costs": {"wall_seconds": 0.2}})
            (path / "arrays.npz").write_bytes(b"baseline-only")
            save(
                directory / "baseline_manifest.json",
                {
                    "conditional_initial_covariance": "zero",
                    "analytic_IID_replacement": False,
                    "cache": {
                        "path": str(path),
                        "record_sha256": entry["sha256"],
                        "arrays_sha256": report.sha(path / "arrays.npz"),
                    },
                },
            )
    for seed in (0, 1, 2):
        save(
            root / "payoffs/execution" / f"worker_seed{seed}.json",
            {
                "complete": True,
                "seed": seed,
                "attempts": [
                    {
                        "complete": True,
                        "pid": 20 + seed,
                        "wall_seconds": 10.0,
                        "cpu_seconds": 10.0,
                        "pool_preparation": {case: {"seconds": 0.1} for case in case_names},
                    }
                ],
            },
        )
    final_test_receipt(root)
    monkeypatch.setattr(
        report._base,
        "runner",
        SimpleNamespace(
            reference_configuration=old_runner.reference_configuration,
            payoff_bank=old_runner.payoff_bank,
            load_reference=lambda *args: {
                "groups": {
                    group: {"stopped_at": np.array([np.inf, 0.1] * 4)}
                    for group in old_runner.reference_configuration(case_names[0], old).group_keys
                }
            },
        ),
    )
    monkeypatch.setattr(report, "load_complete", lambda root: data)
    monkeypatch.setattr(
        report,
        "_label_replay_audit",
        lambda evidence: (
            {
                "required": True,
                "complete": False,
                "verified_families": 0,
                "wall_seconds": None,
                "scope": "synthetic report-only fixture, not a physical label audit",
            },
            {},
        ),
    )
    return data


def test_full_allocations_and_actual_accuracy_helper_keep_every_seed(evidence):
    allocation = report.validate_evidence(evidence)
    assert allocation["candidate_models"] == 18 and allocation["new_primary_records"] == 216
    assert allocation["collocation_primary_records"] == 576 and allocation["timing_jobs"] == 288
    assert allocation["contrast_packets"] == 1080
    tables = report.accuracy_tables(evidence)
    assert len(tables["accuracy"]) == (216 + 576) * 3
    assert len(tables["signed_contrasts"]) == 1080 * 3 * 2
    assert {r["seed"] for r in tables["accuracy"] if r["family"] == "LNA-correction"} == {0, 1, 2}


@pytest.mark.parametrize(
    "change", ["model", "record", "context", "contrast", "seed", "independence"]
)
def test_partial_or_relabelled_evidence_is_rejected(evidence, change):
    if change == "model":
        evidence["candidate_lock"]["models"].pop()
    elif change == "record":
        evidence["records"].pop()
    elif change == "context":
        evidence["baseline"]["records"].pop()
    elif change == "contrast":
        evidence["contrasts"]["records"].pop()
    elif change == "seed":
        evidence["contrasts"]["records"][0]["seed"] = 99
    else:
        next(r for r in evidence["records"] if "__ensemble__" in r["identity"]["group"])[
            "initial_pool_independent_of_reference"
        ] = False
    with pytest.raises(ValueError):
        report.validate_evidence(evidence)


def test_timing_requires_every_full_uncached_cold_warm_observation(evidence):
    entry = evidence["timing"]["jobs"][0]
    value = report.read(entry["path"])
    value["accuracy_cache_used"] = True
    save(Path(entry["path"]), value)
    entry["sha256"] = report.sha(entry["path"])
    with pytest.raises(ValueError, match="uncached"):
        report.timing_table(evidence)


@pytest.mark.parametrize("kind", ["segmented", "counter", "missing-warm"])
def test_segmented_or_partial_counters_cannot_replace_actual_latency(evidence, kind):
    if kind == "missing-warm":
        evidence["timing"]["jobs"].pop()
    else:
        entry = evidence["timing"]["jobs"][0]
        packet = report.read(entry["path"])
        if kind == "segmented":
            packet["timing_mode"] = "segmented-service"
        else:
            packet["workload_seconds"] = 0.001
        save(Path(entry["path"]), packet)
        entry["sha256"] = report.sha(entry["path"])
    with pytest.raises(ValueError):
        report.timing_table(evidence)


def test_cost_components_are_nested_and_standalone_does_not_divide_by_seeds(evidence):
    summary, components, pools = report.preparation_costs(evidence)
    assert summary["native_MC_label_families"] == 96
    assert summary["full_prepare_plus_worker_seconds"] == 42
    assert summary["all_fit_invocation_seconds"] == 18
    assert summary["learner_LNA_cache_component_seconds"] == pytest.approx(2.4)
    assert sum(r["seconds"] for r in components) < summary["full_shared_input_preparation_seconds"]
    costs, _ = report.deployment_costs(evidence, report.timing_table(evidence), summary, pools)
    fitted = next(r for r in costs if r["family"] == "LNA-correction")
    assert fitted["known_offline_seconds"] == pytest.approx(12 + 1 + 4 + 0.02)
    assert not summary["complete_process_cost_receipts"]


def test_missing_cache_or_pool_cost_never_becomes_zero(evidence):
    path = next((evidence["root"] / "evaluation/baselines").rglob("*.json"))
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="cache cost"):
        report.preparation_costs(evidence)
    path.write_bytes(original)
    path = evidence["root"] / "payoffs/execution/worker_seed0.json"
    value = report.read(path)
    value["attempts"][0]["pool_preparation"].pop(evidence["cases"][0])
    save(path, value)
    with pytest.raises(ValueError, match="pool-preparation"):
        report.preparation_costs(evidence)


def test_negative_or_unresolved_gates_never_produce_qualified_payback(evidence):
    summary, _, pools = report.preparation_costs(evidence)
    tables = report.accuracy_tables(evidence)
    tables["costs"], _ = report.deployment_costs(
        evidence, report.timing_table(evidence), summary, pools
    )
    result = report.amortization_tables(tables)
    assert result and all(r["accuracy_qualified_payback_calls"] is None for r in result)
    assert all(r["candidate_gate"] == "complete-process-cost-unresolved" for r in result)


def test_generate_manifest_and_offline_archive_roundtrip(evidence, tmp_path, monkeypatch):
    manifest = report.generate(evidence["root"])
    assert manifest["complete"] and len(manifest["outputs"]) >= 12
    assert report.load_report(evidence["root"] / "report") == manifest
    documentation = tmp_path / "documentation/report.md"
    assert report.generate(evidence["root"], documentation=documentation) == manifest
    assert documentation.read_text() == (evidence["root"] / "report/report.md").read_text()
    destination = tmp_path / "archive"
    monkeypatch.setattr(report, "load_complete", lambda root: evidence)
    created = report.archive(evidence["root"], destination)
    assert created["complete"] and created["allocation"]["new_primary_records"] == 216
    assert (destination / "code/LICENSING.md").is_file()
    assert (destination / "code/tests/test_lna_correction_report.py").is_file()
    assert "canonical documentation" in (destination / "README.md").read_text()
    assert (
        not list(destination.rglob("*.pt"))
        and not list(destination.rglob("*.h5"))
        and not list(destination.rglob("*.npz"))
    )
    assert report.verify_archive(destination) == created
    # Offline verification never reads the original study files.
    (evidence["root"] / "metrics.npz").write_bytes(b"live source no longer available")
    assert report.verify_archive(destination) == created
    extra = destination / "candidate/report/nested/manifest.json"
    extra.parent.mkdir()
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        report.verify_archive(destination)


def test_posthoc_verification_receipts_do_not_change_report_identity(evidence, tmp_path):
    original = report.generate(evidence["root"])
    producer = evidence["root"] / "execution/cli_verify-lock_999.json"
    producer.write_text("posthoc receipts are deliberately not parsed by the frozen work reader")
    evaluator_verify = evidence["root"] / "evaluation/execution/cli_verify_999_100.json"
    evaluator_verify.parent.mkdir(parents=True, exist_ok=True)
    evaluator_verify.write_text("separate posthoc verification, not a work input")
    assert report.generate(evidence["root"]) == original
    assert str(producer) not in original["identity"]["inputs"]
    assert str(evaluator_verify) not in original["identity"]["inputs"]
    destination = tmp_path / "posthoc-archive"
    assert report.archive(evidence["root"], destination)["complete"]
    assert report.verify_archive(destination)["complete"]


@pytest.mark.parametrize("namespace", ["evaluator", "producer"])
@pytest.mark.parametrize("change", ["added", "changed"])
def test_archive_rejects_stale_work_receipt_inventory_without_regenerating(
    evidence, tmp_path, monkeypatch, namespace, change
):
    def receipt():
        if namespace == "evaluator":
            return evaluation_receipt(evidence["root"], "benchmark")
        path = evidence["root"] / "execution/cli_fit_999.json"
        save(path, {"stage": "fit", "pid": 999, "wall_seconds": 1.0,
                    "process_cpu_seconds": 0.8, "source_hash": source_hash()})
        return path

    if change == "changed":
        path = receipt()
    report.generate(evidence["root"])
    outputs = {p: p.read_bytes() for p in (evidence["root"] / "report").iterdir()}
    if change == "added":
        path = receipt()
    else:
        value = report.read(path)
        value["wall_seconds"] += 0.5
        save(path, value)

    def forbidden(*args, **kwargs):
        pytest.fail("Archiving must never regenerate or replace a frozen report")

    monkeypatch.setattr(report, "generate", forbidden)
    destination = tmp_path / "stale-archive"
    with pytest.raises(ValueError, match="work-cost inventory|Frozen report input"):
        report.archive(evidence["root"], destination)
    assert not destination.exists()
    assert all(path.read_bytes() == original for path, original in outputs.items())


def test_archive_requires_an_existing_report_without_generating_one(evidence, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Archiving must not implicitly create a report")

    monkeypatch.setattr(report, "generate", forbidden)
    destination = tmp_path / "missing-report-archive"
    with pytest.raises(FileNotFoundError):
        report.archive(evidence["root"], destination)
    assert not (evidence["root"] / "report").exists()
    assert not destination.exists()


def test_evaluator_process_receipts_have_explicit_nonadditive_coverage(evidence):
    paths = [evaluation_receipt(evidence["root"], stage) for stage in report.EVALUATOR_WORK_STAGES]
    summary, _, _ = report.preparation_costs(evidence)
    assert summary["evaluation_complete_process_cost_receipts"]
    assert summary["evaluation_work_process_seconds_nonadditive"] == 6
    assert summary["evaluation_work_process_cpu_seconds_nonadditive"] == pytest.approx(4.8)
    assert summary["full_prepare_plus_worker_seconds"] == 42
    assert all(str(p) in evidence["inputs"] for p in paths)
    evaluation_receipt(evidence["root"], "benchmark", whole=False)
    assert not report.preparation_costs(evidence)[0]["evaluation_complete_process_cost_receipts"]


def test_package_only_final_test_receipt_is_rejected(tmp_path):
    root = tmp_path / "candidate"
    save(root / "audit/final_tests.json", {"complete": True, "source_hash": source_hash()})
    with pytest.raises(ValueError, match="package hash alone"):
        report.verify_final_tests(root)


def test_current_analysis_edit_invalidates_tests_even_when_package_hash_is_unchanged(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    for name in (
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "LICENSE-DOCS",
        "LICENSING.md",
        "analysis/helper.py",
        "src/nn_dk/example.py",
        "tests/test_example.py",
        "configs/study.toml",
    ):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserved fixture bytes\n")
    monkeypatch.setattr(report, "REPO", repository)
    root = tmp_path / "candidate"
    receipt = final_test_receipt(root)
    package = source_hash()
    assert report.verify_final_tests(root)[0] == receipt
    (repository / "analysis/helper.py").write_text("changed analysis; package bytes unchanged\n")
    assert source_hash() == package == receipt["source_hash"]
    with pytest.raises(ValueError, match="current analysis"):
        report.verify_final_tests(root)
    monkeypatch.setattr(report, "load_report", lambda path: {"identity": {"helpers": {}}})
    monkeypatch.setattr(report, "load_complete", lambda path: {})
    with pytest.raises(ValueError, match="current analysis"):
        report.archive(root, tmp_path / "rejected-archive")


@pytest.mark.parametrize(
    "change", ["log", "partial-command", "post-source", "test-count", "exit-code", "shared-log"]
)
def test_final_test_gate_binds_successful_full_commands_logs_and_counts(tmp_path, change):
    root = tmp_path / "candidate"
    value = final_test_receipt(root)
    if change == "log":
        Path(value["commands"][1]["log_path"]).write_text("different test evidence\n")
    elif change == "partial-command":
        value["commands"][1]["argv"] += ["-k", "smoke"]
    elif change == "post-source":
        value["after_sha256"] = "0" * 64
    elif change == "test-count":
        value["commands"][1]["test_count"] += 1
    elif change == "shared-log":
        value["commands"][0]["log_path"] = value["commands"][1]["log_path"]
        value["commands"][0]["log_sha256"] = value["commands"][1]["log_sha256"]
    else:
        value["commands"][0]["exit_code"] = 1
    save(root / "audit/final_tests.json", value)
    with pytest.raises(ValueError):
        report.verify_final_tests(root)


def test_report_rejects_unindexed_nested_manifest_and_input_mutation(evidence):
    report.generate(evidence["root"])
    output = evidence["root"] / "report"
    extra = output / "nested/manifest.json"
    extra.parent.mkdir()
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        report.load_report(output)
    extra.unlink()
    extra.parent.rmdir()
    path = next(
        Path(name)
        for name in report.read(output / "manifest.json")["identity"]["inputs"]
        if Path(name).is_relative_to(evidence["root"])
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="input"):
        report.load_report(output)


def test_failed_group_remains_in_report_without_successful_metrics(evidence):
    record = evidence["records"][0]
    record.update(available=False, metrics=None, failed_family_ids=["failed-software-family"])
    identity = record["identity"]
    for contrast in evidence["contrasts"]["records"]:
        if (contrast["case"], contrast["group"], contrast["candidate"]) == (
            identity["case"],
            identity["group"],
            record["method"]["name"],
        ):
            contrast.update(available=False, contrast=None)
    allocation = report.validate_evidence(evidence)
    assert allocation["unavailable_new_records"] == 1
    tables = report.accuracy_tables(evidence)
    assert len(tables["accuracy"]) == (216 + 576) * 3
    assert sum(not row["available"] for row in tables["accuracy"]) == 3


def test_complete_process_receipts_charge_import_and_outer_overhead_once(evidence):
    rows = [("prepare", 10, 3.0), ("data", 11, 12.0)] + [
        ("fit", 20 + seed, 11.0) for seed in (0, 1, 2)
    ]
    for stage, pid, seconds in rows:
        save(
            evidence["root"] / "execution" / f"cli_{stage}_{pid}.json",
            {
                "stage": stage,
                "pid": pid,
                "wall_seconds": seconds,
                "process_cpu_seconds": seconds,
                "source_hash": source_hash(),
            },
        )
    summary, _, pools = report.preparation_costs(evidence)
    assert summary["complete_process_cost_receipts"]
    assert summary["full_prepare_plus_worker_seconds"] == 42
    assert summary["all_recorded_CLI_process_seconds_nonadditive"] == 48
    assert summary["standalone_full_input_build_charge_seconds"] == 15
    assert set(summary["seed_shared_worker_overhead_seconds"].values()) == {5.0}
    costs, _ = report.deployment_costs(evidence, report.timing_table(evidence), summary, pools)
    fitted = next(row for row in costs if row["family"] == "LNA-correction")
    assert fitted["known_offline_seconds"] == pytest.approx(15 + 1 + 5 + 0.02)


def test_label_replay_is_required_and_binds_original_inputs(evidence, monkeypatch):
    import analysis

    def missing(*args, **kwargs):
        raise FileNotFoundError("required independent replay receipt")

    fake = SimpleNamespace(allocation=lambda plan: [{}], verify=missing)
    monkeypatch.setattr(analysis, "audit_lna_supervised_labels", fake, raising=False)
    with pytest.raises(FileNotFoundError, match="required"):
        REAL_LABEL_REPLAY(evidence)
    receipt = evidence["root"] / "execution/native_label_replay_audit.json"
    save(receipt, {"complete": True})
    original = evidence["root"] / "data" / evidence["cases"][0] / "train/anchors/arrays.npz"
    payload = {
        "identity": {"helpers": {}, "script_sha256": report.sha(report.__file__)},
        "records": [{"input_files": {str(original): report.sha(original)}}],
        "attempts": [{"wall_seconds": 0.1, "cpu_seconds": 0.1}],
    }
    prior = {"wall_seconds": 0.4, "cpu_seconds": 0.3}
    for prefix in ("manifest", "receipt", "auditor_source"):
        path = evidence["root"] / "execution/superseded_replay_audit_v1" / prefix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserved prior audit bytes")
        prior[prefix + "_path"], prior[prefix + "_sha256"] = str(path), report.sha(path)
    payload["prior_audit_costs"] = [prior]
    fake.verify = lambda *args, **kwargs: payload
    fake.RECEIPT, fake.__file__ = "execution/native_label_replay_audit.json", report.__file__
    audited, helpers = REAL_LABEL_REPLAY(evidence)
    assert audited["complete"] and audited["verified_families"] == 1
    assert audited["wall_seconds"] == pytest.approx(0.5)
    assert audited["prior_audit_wall_seconds"] == 0.4
    assert prior["auditor_source_path"] in evidence["inputs"]
    assert str(original) in evidence["inputs"] and helpers
    original.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="checksum"):
        REAL_LABEL_REPLAY(evidence)


def test_qualified_arithmetic_requires_accuracy_sampling_refinement_and_bounds():
    candidate = {
        "case": "case",
        "profile": "uniform",
        "role": "conditional",
        "name": "candidate",
        "seed": 0,
        "family": "LNA-correction",
        "available": True,
        "complete_process_cost_receipts": True,
        "evaluation_complete_process_cost_receipts": True,
        "timing_attempt_cost_complete": True,
        "native_label_replay_verified": True,
        "full_scientific_allocation": True,
        "known_offline_seconds": 10.0,
        "cold_observed_seconds": 1.0,
        "warm_observed_seconds": 1.0,
    }
    control = {
        **candidate,
        "name": "control",
        "family": "raw-conditional-LNA",
        "known_offline_seconds": 0.0,
        "cold_observed_seconds": 2.0,
        "warm_observed_seconds": 2.0,
    }
    accuracy, sampling, bounds, refinement = [], [], [], []
    for name in ("candidate", "control"):
        for t in range(3):
            origin = {
                "regime": "case",
                "profile": "uniform",
                "role": "conditional",
                "name": name,
                "time_index": t,
            }
            accuracy.append(
                {
                    **origin,
                    "available": True,
                    "max_mean_error_in_reference_std": 0.001,
                    "max_normalized_covariance_error": 0.001,
                    "mean_sampling95_simultaneous_radius": 0.001,
                    "covariance_sampling95_simultaneous_radius": 0.001,
                    "mean_error_plus_sampling95": 0.002,
                    "covariance_error_plus_sampling95": 0.002,
                }
            )
            sampling.append({**origin, "mean95_radius": 0.001, "covariance95_radius": 0.001})
            bounds.append(
                {
                    **origin,
                    "mean_inside_physical_bounds": True,
                    "variance_inside_physical_bounds": True,
                }
            )
    refinement.extend(accuracy[:3])
    tables = {
        "costs": [candidate, control],
        "accuracy": accuracy,
        "sampling": sampling,
        "components": bounds,
        "reference_refinement": refinement,
    }
    assert all(
        row["accuracy_qualified_payback_calls"] == 10 for row in report.amortization_tables(tables)
    )
    accuracy[0]["max_mean_error_in_reference_std"] = 0.2
    assert (
        next(row for row in report.amortization_tables(tables) if row["statistic"] == "mean")[
            "accuracy_qualified_payback_calls"
        ]
        is None
    )
    accuracy[0]["max_mean_error_in_reference_std"] = 0.001
    sampling[0]["mean95_radius"] = 0.2
    assert (
        next(row for row in report.amortization_tables(tables) if row["statistic"] == "mean")[
            "candidate_gate"
        ]
        == "standalone-MC-component-unresolved"
    )
    bounds[0]["variance_inside_physical_bounds"] = False
    assert all(
        row["accuracy_qualified_payback_calls"] is None
        for row in report.amortization_tables(tables)
    )
