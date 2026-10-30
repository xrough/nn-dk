"""Signed uncertainty, full allocations, and immutable gradient report inputs."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_mean_gradients as audit  # noqa: E402
from analysis import report_payoff_mean_gradients as reporting  # noqa: E402


def test_positive_norm_noise_never_implies_a_resolved_signed_discrepancy():
    point = np.zeros((2, 4, 8))
    signs = np.array([-1., 1.] * 4)[:, None, None, None]
    bootstrap = signs*np.ones((8, *point.shape))
    # Every resampled distance is strictly positive even though the signed
    # contrast is centered at zero and no component interval excludes zero.
    assert np.quantile(np.linalg.norm(bootstrap, axis=(-2, -1)), .025) > 0
    contrast = reporting.signed_contrast(point, bootstrap)
    assert not contrast["pointwise_excludes_zero"].any()
    np.testing.assert_array_equal(contrast["pointwise_interval"][0], -1.)
    np.testing.assert_array_equal(contrast["pointwise_interval"][1], 1.)


def test_paired_refinement_can_cancel_shared_uncertainty_exactly():
    coarse = np.random.default_rng(45).normal(size=(8, 3, 4, 8))
    _, arrays = audit.refinement_comparison(coarse, coarse, 16, 917)
    contrast = reporting.signed_contrast(arrays["gradient_difference"],
                                        arrays["paired_bootstrap_gradient_difference"])
    assert not contrast["pointwise_excludes_zero"].any()
    np.testing.assert_array_equal(contrast["pointwise_interval"], 0.)


@pytest.fixture
def complete_audit(tmp_path, monkeypatch):
    root = tmp_path/"audit"
    root.mkdir()
    q, g, b, paths = 2, 8, 8, 4
    queries = [.01, .02]
    identity_hash = "synthetic-format-fixture-not-a-scientific-audit"
    raw = np.random.default_rng(909).normal(scale=.1, size=(paths, q, 4, g))
    mean, plugin, cross = audit.source_estimates(raw)
    boot_mean, boot_source = audit.bootstrap_sources(raw, b, 118)
    _, refined = audit.refinement_comparison(raw, raw+.002, b, 118)
    key = "toy_anchor0"
    values = {key+"_reference_mean_gradient": mean, key+"_reference_plugin_source": plugin,
        key+"_reference_cross_source": cross, key+"_bootstrap_mean_gradient": boot_mean,
        key+"_bootstrap_cross_source": boot_source}
    values.update({key+"_refinement_"+name: value for name, value in refined.items()})
    anchor_path = root/"anchors.npz"
    audit.npz(anchor_path, {"states": np.full((1, g), 1/g)})
    prepared = {"identity_hash": identity_hash,
        "identity": {"cases": ["toy"], "models": [{"case": "toy", "seed": seed} for seed in (0, 1, 2)],
                     "plan": {"paths": paths, "chunk_paths": 2, "n_bootstrap": b,
                              "refinement_paths": paths, "refinement_factor": 2,
                              "step": .001, "query_fractions": [.5, 1.]}},
        "anchors": {"toy": {"rows": 1, "path": str(anchor_path), "sha256": audit.sha(anchor_path),
                             "queries": queries}}}
    chunks = []
    for start in (0, 2):
        path = root/f"chunk{start}.npz"
        audit.npz(path, {"gradient": raw[start:start+2],
                         "family_ids": np.array([f"f{start}", f"f{start+1}"]),
                         "stopped_at": np.array([np.inf, .005 if start else np.inf])})
        chunks.append({"case": "toy", "anchor": 0, "start": start,
                       "path": str(path), "sha256": audit.sha(path)})
    records = []
    for seed in (0, 1, 2):
        values[key+f"_seed{seed}_gradient"] = mean+.05*(seed+1)
        records.append({"case": "toy", "anchor": 0, "seed": seed, "profile": "uniform",
            "queries": queries, "families": paths, "stopped_families": 1,
            "controls": {"zero_time_face_maximum_error": 2e-16, "absorbed_face_source_exact_zero": True},
            "minimum_mass": .125, "distance_to_boundary": .124,
            "minimum_mass_over_stopping_mass": 125., "boundary_stratum": "above_25_thresholds",
            "boundary_width": .005, "distance_over_boundary_width": 24.8,
            "distance_over_moving_width_by_query": [35., 24.8],
            "selection_rank_fraction": 0., "minimum_mass_rank": 0, "candidate_count": 512,
            "finite_difference_pathwise_maximum_error": 2e-12,
            "plugin_minus_cross_frobenius_by_query": [0., 0.],
            "cross_source_minimum_eigenvalue_by_query": [0., 0.]})
    evidence = root/"evidence.npz"
    audit.npz(evidence, values)
    summary = {"complete": True, "identity_hash": identity_hash, "records": records,
               "numeric_evidence": {"path": str(evidence), "sha256": audit.sha(evidence)}}
    audit.write(root/"plan.json", prepared)
    audit.write(root/"index.json", {"complete": True, "identity_hash": identity_hash, "chunks": chunks})
    audit.write(root/"summary.json", summary)

    def load(_root):
        if not audit.read(Path(_root)/"index.json")["complete"]:
            raise ValueError("Every diagnostic chunk must complete")
        return audit.read(Path(_root)/"summary.json")

    monkeypatch.setattr(audit, "load_complete", load)
    return root, summary, prepared, values


def test_full_report_replays_raw_family_estimators_signed_arrays_and_output_hashes(complete_audit):
    root, _, _, values = complete_audit
    result = reporting.report(root)
    assert result["allocation"]["models"] == 3
    assert result["allocation"]["anchors"] == 1
    assert len(result["records"]) == 3
    assert result["refinement"][0]["paired_whole_family_resampling"]
    assert result["records"][0]["stopped_families"] == 1
    with np.load(root/"report/contrasts.npz") as store:
        np.testing.assert_allclose(store["toy_anchor0_seed0_gradient_point"], .05, atol=1e-16)
        expected = (values["toy_anchor0_seed0_gradient"]
                    @ values["toy_anchor0_seed0_gradient"].swapaxes(-1, -2)
                    - values["toy_anchor0_reference_cross_source"])
        np.testing.assert_allclose(store["toy_anchor0_seed0_source_point"], expected, atol=1e-16)
    assert result == reporting.report(root)
    assert reporting.load_report(root/"report") == result
    text = (root/"report/report.md").read_text()
    assert "not a simultaneous" in text
    assert "positive lower percentile" in text
    assert "covariance-head" in text
    (root/"report/contrasts.npz").write_bytes(b"mutated report evidence")
    with pytest.raises(ValueError, match="output evidence"):
        reporting.load_report(root/"report")


@pytest.mark.parametrize("kind", ["missing_seed", "duplicate_seed", "missing_query", "missing_family_chunk"])
def test_truncated_or_duplicated_allocations_rejected(complete_audit, kind):
    root, summary, _, _ = complete_audit
    summary = copy.deepcopy(summary)
    if kind == "missing_seed":
        summary["records"].pop()
    elif kind == "duplicate_seed":
        summary["records"][-1] = copy.deepcopy(summary["records"][0])
    elif kind == "missing_query":
        summary["records"][0]["queries"] = [.01]
    else:
        index = audit.read(root/"index.json")
        index["chunks"].pop()
        audit.write(root/"index.json", index)
    audit.write(root/"summary.json", summary)
    with pytest.raises(ValueError, match="allocation|queries"):
        reporting.report(root)


def test_original_mutated_numeric_evidence_rejected_before_reporting(complete_audit):
    root, summary, _, _ = complete_audit
    Path(summary["numeric_evidence"]["path"]).write_bytes(b"mutated original")
    with pytest.raises(ValueError, match="numeric evidence changed"):
        reporting.report(root)
    assert not (root/"report").exists()


def test_original_raw_jvp_corruption_is_not_hidden_by_a_valid_summary(complete_audit):
    root, _, _, _ = complete_audit
    index = audit.read(root/"index.json")
    Path(index["chunks"][0]["path"]).write_bytes(b"mutated original chunk")
    with pytest.raises(ValueError, match="raw JVP evidence changed"):
        reporting.report(root)


def test_incomplete_audit_rejected_before_any_outputs(complete_audit):
    root, _, _, _ = complete_audit
    index = audit.read(root/"index.json")
    index["complete"] = False
    audit.write(root/"index.json", index)
    with pytest.raises(ValueError, match="must complete"):
        reporting.report(root)
    assert not (root/"report").exists()
