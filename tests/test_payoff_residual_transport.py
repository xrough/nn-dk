"""Residual transport distinguishes mean, raw seconds, covariance and defects."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nn_dk.generator_jets import GeneratorJet
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.payoff_boundary import BoundaryPayoffModel
from nn_dk.payoff_learning import PayoffBank

_spec = importlib.util.spec_from_file_location(
    "payoff_residual_transport", Path(__file__).parents[1]/"analysis/audit_payoff_residual_transport.py")
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit
_spec.loader.exec_module(audit)


class LinearControl:
    """One-dimensional exact heat/linear models with optional known mean bias."""

    def __init__(self, *, decay=0., running=0., sigma=0., mean_bias=0.):
        self.decay, self.running, self.sigma, self.bias = decay, running, sigma, mean_bias
        self.context = SimpleNamespace(n_particles=1)
        self.normalized_running_weights = torch.tensor([[running]], dtype=torch.float64)

    def _active(self, states):
        return torch.ones(len(states), dtype=torch.bool)

    def _moment_jets(self, states, times):
        a, q, sigma, bias = self.decay, self.running, self.sigma, self.bias
        if a:
            assert sigma == 0
            coefficient = torch.exp(a*times)+q*torch.expm1(a*times)/a
            derivative = (a+q)*torch.exp(a*times)
            variance = torch.zeros_like(times)
            variance_derivative = torch.zeros_like(times)
        else:
            coefficient, derivative = 1+q*times, torch.full_like(times,q)
            variance = sigma**2*(times+q*times.square()+q*q*times**3/3)
            variance_derivative = sigma**2*coefficient.square()
        mean = GeneratorJet(coefficient[:, None]*states+bias*times[:, None],
                            derivative[:, None]*states+bias,
                            coefficient[:, None]*a*states,
                            (coefficient*sigma)[:, None, None])
        covariance = GeneratorJet(variance[:, None, None], variance_derivative[:, None, None],
                                  torch.zeros((len(states),1,1),dtype=torch.float64),
                                  torch.zeros((len(states),1,1,1),dtype=torch.float64))
        return mean,covariance


def _summary_from_paths(model, states, dt, *, stride=4, stopped_at=None):
    stops = np.full(len(states),np.inf) if stopped_at is None else stopped_at
    query = (states.shape[1]-1)*dt
    packet = {"states":states,"stopped_at":stops}
    values = audit.transport_on_paths(model,packet,dt,[query],stride)
    accumulated,_ = audit.polygon_running(states,np.arange(states.shape[1])*dt,stops,
                                          model.normalized_running_weights.numpy())
    outcomes = states[:, -1:, :]+accumulated[:, -1:]
    results = []
    for prefix in ("","half_"):
        results.append(audit.summarize_transport(values["initial_mean"],values["initial_covariance"],
            outcomes,values[prefix+"mean_integral"],values[prefix+"raw_second_integral"],
            values[prefix+"covariance_integral"]))
    return values,results


def test_deterministic_heat_identity_has_zero_mean_and_raw_second_transport():
    dt = 1/16
    times = np.arange(17)*dt
    states = np.broadcast_to(np.exp(-2*times)[None,:,None],(4,17,1)).copy()
    _, summaries = _summary_from_paths(LinearControl(decay=-2),states,dt)
    for summary in summaries:
        for key in ("mean_discrepancy","raw_second_discrepancy","mean_residual_integral",
                    "raw_second_residual_integral","mean_transport_defect","raw_second_transport_defect"):
            np.testing.assert_allclose(summary["point"][key],0,atol=5e-16,rtol=0)


def test_nonzero_running_payoff_and_mean_bias_need_cross_terms_not_covariance_residual():
    model = LinearControl(running=.4,mean_bias=.2)
    values,summaries = _summary_from_paths(model,np.ones((4,17,1)),1/16)
    expected_mean = .2
    expected_second = 1.6**2-1.4**2
    for summary in summaries:
        np.testing.assert_allclose(summary["point"]["mean_discrepancy"],[[expected_mean]],atol=5e-16)
        np.testing.assert_allclose(summary["point"]["mean_residual_integral"],[[expected_mean]],atol=5e-16)
        np.testing.assert_allclose(summary["point"]["raw_second_discrepancy"],[[[expected_second]]],atol=1e-15)
        np.testing.assert_array_equal(summary["point"]["covariance_residual_integral_alone"],[[[0.]]])
        assert not summary["covariance_residual_alone_is_prediction_error_certificate"]
    defect = [abs(np.asarray(s["point"]["raw_second_transport_defect"]).item()) for s in summaries]
    assert defect[0] == pytest.approx(2*defect[1],abs=1e-15)
    assert values["raw_second_integral"].item(0) > expected_second


def test_balanced_linear_brownian_example_preserves_raw_second_not_sample_covariance():
    states = np.array([[[.8],[1.2]],[[.8],[.4]]])
    _,summaries = _summary_from_paths(LinearControl(sigma=.4),states,1.,stride=2)
    result = summaries[0]["point"]
    np.testing.assert_allclose(result["raw_second_discrepancy"],0,atol=3e-16)
    np.testing.assert_allclose(result["raw_second_transport_defect"],0,atol=3e-16)
    # Unbiased covariance uses n-1; raw second moments intentionally use n.
    np.testing.assert_allclose(result["sample_covariance_discrepancy_unbiased"],[[[-.16]]],atol=3e-16)


def test_fractional_absorption_polygon_running_and_left_exposure_keep_all_paths():
    states = np.array([[[1.],[.25],[.25]],[[.2],[.2],[.2]],[[1.],[1.],[1.]]])
    stops = np.array([.5,0.,np.inf])
    accumulated,exposure = audit.polygon_running(states,[0.,1.,2.],stops,[[2.]])
    np.testing.assert_array_equal(exposure,[[.5,0],[0,0],[1,1]])
    np.testing.assert_allclose(accumulated[:,:,0],[[0,.625,.625],[0,0,0],[0,2,4]])
    integrated = audit.integrate_residual(np.full((3,2,1),3.),[0.,1.,2.],stops)
    np.testing.assert_array_equal(integrated,[[1.5],[0.],[6.]])


def test_raw_second_residual_has_both_off_diagonal_cross_terms():
    u,a,r = np.array([[2.,3.]]),np.array([[.5,1.]]),np.array([[.2,-.3]])
    actual = audit.raw_second_residual(u,a,r,np.zeros((1,2,2)))
    expected = np.outer((u+a)[0],r[0])+np.outer(r[0],(u+a)[0])
    np.testing.assert_array_equal(actual[0],expected)


def test_whole_family_bootstrap_pairs_payoffs_and_residuals():
    y = np.arange(12,dtype=float).reshape(6,1,2)/10
    u = np.array([[2.,3.]])
    im = u[None]-y
    initial_second = u[:,:,None]*u[:,None,:]
    ir = initial_second[None]-y[:,:,:,None]*y[:,:,None,:]
    counts = np.array([[6,0,0,0,0,0],[0,0,0,0,0,6],[1,1,1,1,1,1]])
    result = audit.summarize_transport(u,np.zeros((1,2,2)),y,im,ir,np.zeros_like(ir),
                                     bootstrap_counts=counts,return_bootstrap=True)
    for key in ("mean_transport_defect","raw_second_transport_defect"):
        np.testing.assert_allclose(result["bootstrap_samples"][key],0,atol=2e-15)
        np.testing.assert_allclose(result["pointwise_percentile95"][key],0,atol=2e-15)
    with pytest.raises(ValueError,match="Whole-family"):
        audit.summarize_transport(u,np.zeros((1,2,2)),y,im,ir,np.zeros_like(ir),bootstrap_counts=counts+1)


@pytest.mark.parametrize("kwargs",[{"residual_stride":3},{"paths":3,"chunk_paths":2},
                                    {"refinement_factor":1},{"query_fractions":(.5,.25)},
                                    {"step":np.nan}])
def test_plan_rejects_incompatible_allocations(kwargs):
    with pytest.raises(ValueError):
        audit.TransportPlan(**kwargs)


def test_transport_rejects_mixed_conditional_origins_and_nonfinite_arrays():
    states = np.array([[[1.],[1.]],[[2.],[2.]]])
    with pytest.raises(ValueError,match="conditional anchor"):
        audit.transport_on_paths(LinearControl(),{"states":states,"stopped_at":np.full(2,np.inf)},1.,[1.],2)
    with pytest.raises(ValueError,match="Complete finite"):
        audit.polygon_running(np.full((2,2,1),np.nan),[0,1],[np.inf,np.inf],[[1.]])


def _fake_study(tmp_path,monkeypatch):
    import analysis.audit_payoff_regimes_validation as metadata

    cfg = replace(InteractingStudyConfig(),n_cells=8,width=4)
    study_root = tmp_path/"historical"
    (study_root/"payoffs").mkdir(parents=True)
    (study_root/"payoffs/fit_lock.json").write_text('{"complete":true}')
    bank_path = study_root/"validation.h5"
    bank_path.write_bytes(b"state-only-validation-fixture")
    states = np.full((2,8),.125)
    bank = {"states":states,"case_ids":np.array(["uniform","nonuniform"]),
            "family_ids":np.array(["validation0","validation1"]),"snapshot_times":np.zeros(2),
            "path":bank_path}
    entries = []
    for seed in (0,1,2):
        path = study_root/f"model{seed}.pt"
        sidecar = path.with_suffix(".json")
        path.write_bytes(f"frozen fixture {seed}".encode())
        sidecar.write_text('{}')
        entries.append({"case":"tiny","seed":seed,"checkpoint_path":str(path),
                        "checkpoint_sha256":audit.sha(path),"metadata_path":str(sidecar),
                        "metadata_sha256":audit.sha(sidecar)})
    model = BoundaryPayoffModel(cfg.spec,cfg.physics,cfg.horizon,
                               PayoffBank.mixed_terminal_integral(cfg.spec,cfg.horizon),width=4,
                               quadrature_nodes=4).eval()
    runner = SimpleNamespace(configuration=lambda case,study:cfg,
                             load_state_bank=lambda *args:bank,
                             load_declared_model=lambda item:model)
    monkeypatch.setattr(audit,"_study",lambda *args:(runner,None,("tiny",),entries))
    monkeypatch.setattr(metadata,"validation_metadata",lambda b:{"physical_active":np.ones(2,dtype=bool)})
    return study_root,entries


def test_immutable_prepare_chunk_resume_complete_bootstrap_and_tamper(tmp_path,monkeypatch):
    from analysis.compiled_native import compiler_available

    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    torch.set_num_threads(1)
    study_root,entries = _fake_study(tmp_path,monkeypatch)
    plan = audit.TransportPlan(cases=("tiny",),anchors_per_profile=1,query_fractions=(1.,),
        paths=2,chunk_paths=2,step=1/128,residual_stride=2,n_bootstrap=3)
    root = tmp_path/"transport"
    prepared = audit.prepare(root,study_root,plan)
    assert audit.prepare(root,study_root,plan)==prepared
    with pytest.raises(ValueError,match="identity changed"):
        audit.prepare(root,study_root,replace(plan,seed=plan.seed+1))
    partial = audit.run(root,max_chunks=1)
    assert not partial["complete"] and len(partial["chunks"])==1
    with pytest.raises(ValueError,match="All declared"):
        audit.load_complete(root)
    first = Path(partial["chunks"][0]["path"])
    before = first.read_bytes()
    assert audit.run(root)["complete"]
    assert first.read_bytes()==before
    result = audit.load_complete(root)
    assert len(result["records"])==6
    assert {r["seed"] for r in result["records"]}=={0,1,2}
    assert all(r["families"]==2 for r in result["records"])
    assert not result["fitting_or_selection"]
    assert not result["old_final_reference_outcomes_opened"]
    assert "fine_minus_coarse_pointwise_percentile95" in result["records"][0]["paired_refinement_and_halfstride"]["raw_second_transport_defect"]
    assert audit.run(root)["complete"]
    original = first.read_bytes()
    first.write_bytes(original+b"tamper")
    with pytest.raises(ValueError,match="chunk changed"):
        audit.load_complete(root)
    first.write_bytes(original)
    Path(entries[0]["checkpoint_path"]).write_bytes(b"changed frozen weights")
    with pytest.raises(ValueError,match="checkpoint changed"):
        audit.load_complete(root)


def test_plan_source_tamper_and_exclusive_owner_are_rejected(tmp_path,monkeypatch):
    import fcntl

    study_root,_ = _fake_study(tmp_path,monkeypatch)
    root = tmp_path/"transport"
    plan = audit.TransportPlan(cases=("tiny",),anchors_per_profile=1,query_fractions=(1.,),
        paths=2,chunk_paths=2,step=1/128,residual_stride=2,n_bootstrap=3)
    audit.prepare(root,study_root,plan)
    with (root/".run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match="Another process"):
            audit.run(root,max_chunks=0)
    saved = json.loads((root/"plan.json").read_text())
    saved["identity"]["source_hash"]="changed"
    (root/"plan.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError,match="Frozen diagnostic"):
        audit.run(root,max_chunks=0)
