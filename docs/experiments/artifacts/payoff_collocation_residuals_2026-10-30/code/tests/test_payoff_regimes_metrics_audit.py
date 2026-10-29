"""Independent bootstrap and exact heat-control aggregation checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_regimes_metrics as audit  # noqa: E402
from analysis import evaluate_payoff_regimes as production  # noqa: E402
from analysis.evaluate_path_generalization import mixture_moments, moment_comparison  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402
from nn_dk.types import Context  # noqa: E402


def test_weighted_bootstrap_matches_explicit_repeated_whole_families():
    rng = np.random.default_rng(117)
    values = rng.normal(size=(24, 3, 4))+np.array([2., 5., -3., 7.])
    counts = rng.multinomial(24, np.full(24, 1/24), size=8)
    mean, cov = audit.empirical(values, counts)
    for i, weights in enumerate(counts):
        repeated = values[np.repeat(np.arange(len(values)), weights)]
        np.testing.assert_allclose(mean[i], repeated.mean(0), atol=3e-15)
        np.testing.assert_allclose(cov[i], np.stack([np.cov(repeated[:, t].T, ddof=1) for t in range(3)]), atol=3e-15)


def test_independent_error_radii_advantages_and_standalone_sampling_match_saved_contract():
    rng = np.random.default_rng(118)
    reference = rng.normal(size=(32, 3, 4))*.02+.1
    reference_counts, initial_counts = audit._counts(32, 11, 16), audit._counts(32, 19, 16)
    mu = rng.normal(size=(32, 3, 4))*.005+.102
    cc = np.broadcast_to(np.eye(4), (32, 3, 4, 4)).copy()*.6
    pm, pc = audit.mixture(mu, cc, 2048)
    bpm, bpc = audit.mixture(mu, cc, 2048, initial_counts)
    rm, rc = audit.empirical(reference)
    brm, brc = audit.empirical(reference, reference_counts)
    expected = moment_comparison(reference, predicted_mean=pm, predicted_scaled_covariance=pc,
        bootstrap_predicted_mean=bpm, bootstrap_predicted_scaled_covariance=bpc,
        counts=reference_counts, n_particles=2048)
    audit._compare(audit.metric_records(rm, rc, brm, brc, pm, pc, 2048, bpm, bpc), expected["records"])
    gm, gc = pm+.001, pc*1.1
    expected_contrast = production.bootstrap_advantage(reference, pm, pc, gm, gc,
                                                       reference_counts, 2048, bpm, bpc)
    audit._compare(audit.contrast_records(rm, rc, brm, brc, pm, pc, gm, gc, 2048, bpm, bpc),
                   expected_contrast["records"])
    audit._compare(audit.standalone_sampling(pm, pc, bpm, bpc, 2048),
                   production.standalone_sampling(pm, pc/2048, bpm, bpc/2048))
    for counts in (None, initial_counts):
        actual = audit.mixture(mu, cc, 2048, counts)
        expected = mixture_moments(mu, cc, n_particles=2048, counts=counts)
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-16)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-14)


def test_untrained_heat_algebra_matches_direct_value_equations_and_covariance_units():
    torch.set_num_threads(1)
    spec, horizon = NativeDKSpec(8), .25
    model = RepairPayoffModel(spec, Context(n_particles=2048, kappa=.75, a2=.1), horizon,
                              PayoffBank.mixed_terminal_integral(spec, horizon), 4, 16)
    raw = np.random.default_rng(119).uniform(.8, 1.2, (24, 8))
    states = raw/raw.sum(-1, keepdims=True)
    times = (horizon/4, horizon/2, horizon)
    analytic = audit.analytic_untrained_predictions(model, states, times)
    direct = audit.direct_predictions(model, states, times)
    np.testing.assert_allclose(analytic[0], direct[0], atol=4e-16, rtol=1e-12)
    np.testing.assert_allclose(analytic[1], direct[1], atol=4e-14, rtol=1e-12)


def test_seed_consistency_requires_every_seed_and_keeps_pointwise_scope():
    rows = [{"case": "T1_k6", "group": "uniform__ensemble__0", "role": "ensemble", "name": f"seed{i}",
             "records": [{"time_index": 0, "mean_advantage_percentile95": [.1, .2],
                          "covariance_advantage_percentile95": [-.3, -.1]}]} for i in range(3)]
    result = audit.seed_consistency(rows, (0, 1, 2))
    assert len(result) == 2
    assert all("not a family-wise" in row["scope"] for row in result)
    assert next(r for r in result if r["statistic"] == "mean")["all_seeds_NN_closer_pointwise95"]
    assert next(r for r in result if r["statistic"] == "covariance")["all_seeds_LGA_closer_pointwise95"]
    with pytest.raises(ValueError, match="omitted"):
        audit.seed_consistency(rows[:2], (0, 1, 2))
