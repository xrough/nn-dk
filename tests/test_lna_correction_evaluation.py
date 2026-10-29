"""Joint pre-draw gates and literal per-histogram nonlinear-LNA evaluation.

Format fixtures below are explicitly synthetic and fit no scientific models.
The production baseline implementation and total-covariance algebra run in
these fixtures; metadata mocks stand in for already locked model allocations.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from analysis import evaluate_lna_payoff_correction as ev  # noqa: E402
from analysis import run_payoff_collocation_study as actual_collocation  # noqa: E402
from analysis.fit_lna_payoff_correction import (  # noqa: E402
    LNAPayoffCorrection,
    LNAPayoffCorrectionConfig,
)
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402


class ZeroHead:
    def predict_conditional(self,initial,times,mean,covariance,*,kind):
        return {"mean":mean.copy(),"scaled_covariance":covariance.copy()}


@pytest.fixture
def joint(tmp_path,monkeypatch):
    root,other = tmp_path/"candidate-software-fixture",tmp_path/"collocation-software-fixture"
    root.mkdir()
    other.mkdir()
    study = actual_collocation.StudyPlan(horizons=(1/64,),kappas=(.75,),n_cells=8,n_particles=512,
        train_paths=1,validation_paths=1,adam_steps=1,lbfgs_steps=0,width=2,paths_per_group=8,
        refinement_paths=4,chunk_paths=4,conditional_anchors=1)
    case = actual_collocation.case_names(study)[0]
    cfg = actual_collocation.reference_configuration(case,study)
    candidate_study = SimpleNamespace(optimizer_seeds=(0,1,2),baseline_step=2**-7)
    configuration = LNAPayoffCorrectionConfig(horizon=cfg.horizon,kappa=cfg.kappa,n_cells=8,n_particles=512,
        width=2,mean_adam_steps=1,covariance_adam_steps=1,mean_lbfgs_steps=0,covariance_lbfgs_steps=0)
    models = [{"case":case,"name":f"{case}_lna_correction_seed{s}","seed":s,"checkpoint_sha256":str(s)*64} for s in (0,1,2)]
    locked = {"complete":True,"models":models,"collocation_root":str(other)}
    colloc_lock = {"complete":True,"models":actual_collocation.slots(study)}
    context = {"root":root,"collocation_root":other,"study":candidate_study,"collocation_study":study,
        "candidate_lock":locked,"collocation_lock":colloc_lock,"cases":(case,),"runner":SimpleNamespace(
            configuration=lambda c,s:configuration,payoff_bank=lambda c,s:actual_collocation.payoff_bank(case,study))}
    ev.write(root/"plan.json",{"synthetic":True})
    ev.write(root/"payoffs/fit_lock.json",locked)
    audit_receipt = root/"execution/native_label_replay_audit.json"
    ev.write(audit_receipt,{"complete":True,"synthetic_format_fixture_only":True})
    ev.write(other/"payoffs/fit_lock.json",colloc_lock)
    ids = root/"data/learner_ids.npz"
    ev.save_arrays(ids,{"family_ids":np.array(["R4-learner"],dtype="S64"),
                        "parent_family_ids":np.array(["R4-parent"],dtype="S64")})
    ev.write(root/"data/data_lock.json",{"identifier_inventory":[{"path":str(ids),"sha256":ev.sha(ids),
        "fields":["family_ids","parent_family_ids"]}]})
    for role in ("train","validation"):
        path = other/f"payoffs/{case}/states_{role}.h5"
        path.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(path,"w") as f:
            f.create_dataset("family_ids",data=np.array(["old-learner-"+role],dtype="S64"))
    calls = {"final_draws":0,"reference_loads":0,"conditional_histograms_evolved":0}
    reference_data = {}
    old_records = []

    def prepare_old(_):
        path = other/"deployment/planned_initial_families.npz"
        if not path.exists():
            ev.save_arrays(path,{"old":np.array(["old-independent-pool"],dtype="S64")})
        return {"complete":True}

    def freeze_old(_):
        return ev.immutable(other/"comparison/evaluation_lock.json",{"complete":True,"methods":colloc_lock["models"]})

    def verify_old(_):
        if not (other/"comparison/evaluation_lock.json").exists():
            raise ValueError("Old evaluation lock missing")

    def generate(_,case=None,deadline=None,max_chunks=None):
        assert (root/"evaluation/joint_lock.json").exists()
        calls["final_draws"] += 1
        bank = actual_collocation.payoff_bank(context["cases"][0],study)
        records = []
        for group in cfg.group_keys:
            profile,role,anchor = group.split("__")
            p = actual_collocation.profile_probabilities(cfg,profile)[0]
            initial = seeded_rng(747,"synthetic",group).multinomial(cfg.n_particles,p,size=1)/cfg.n_particles
            # This is synthetic nondegenerate statistical-format evidence,
            # not an experimental native-Euler accuracy claim.
            raw = ev.conditional_baseline(initial,cfg.context(profile),cfg.spec,cfg.query_times,2**-7,bank,batch_size=4)
            mean,cov = raw["mean"][0],raw["scaled_covariance"][0]
            outcomes = mean+seeded_rng(839,"synthetic",group).normal(scale=.02,size=(cfg.paths_per_group,3,4))
            families = np.array([stable_hash([actual_collocation.STREAM,cfg.seed,cfg.config_hash,
                "reference-"+role,profile,int(anchor),i]) for i in range(cfg.paths_per_group)],dtype="S64")
            path = other/"reference"/context["cases"][0]/"reference.h5"
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"synthetic reference placeholder; format fixture only")
            ev.write(path.with_name("manifest.json"),{"reference_path":str(path),"reference_sha256":ev.sha(path)})
            reference_data[group] = {"outcomes":outcomes,"family_ids":families,
                "metadata":{"conditional_initial_json":json.dumps(initial[0].tolist())},
                "stopped_at":np.array([.001]+[np.inf]*7),
                "manifest":{"reference_sha256":ev.sha(path)}}
            old_methods = [{"name":m["name"],"family":"repaired","seed":m["seed"]} for m in colloc_lock["models"]]
            old_methods += [{"name":context["cases"][0]+"_untrained","family":"untrained-repaired","seed":0},
                *[{"name":name,"family":"LGA" if name.startswith("LGA") else "native-MC"} for name in
                  ("LGA_dt7","LGA_dt10","native-MC-dt10")]]
            numeric = ev.save_arrays(other/"synthetic-baselines"/(group+".npz"),{
                "mean":mean,"scaled_covariance":cov,"bootstrap_mean":np.broadcast_to(mean,(128,*mean.shape)),
                "bootstrap_scaled_covariance":np.broadcast_to(cov,(128,*cov.shape))})
            records.extend({"complete":True,"identity":{"regime":context["cases"][0],"group":group},
                            "method":method,"arrays":numeric} for method in old_methods)
        old_records[:] = records
        ev.write(other/"comparison/accuracy_index.json",{"complete":True})
        ev.write(other/"deployment/accuracy_index.json",{"complete":True})
        return {"complete":True}

    def load(_,regime,group):
        calls["reference_loads"] += 1
        if not (root/"evaluation/joint_lock.json").exists():
            raise AssertionError("Unfrozen final reference read")
        return reference_data[group]

    fake_collocation = SimpleNamespace(STREAM=actual_collocation.STREAM,
        reference_configuration=lambda c,s:cfg,payoff_bank=lambda c,s:actual_collocation.payoff_bank(case,study),
        profile_probabilities=actual_collocation.profile_probabilities,freeze_evaluation=freeze_old,
        verify_evaluation_lock=verify_old,generate_reference=generate,load_reference=load,
        _exclusive=actual_collocation._exclusive)
    original_comparison = ev.comparison
    fake_comparison = SimpleNamespace(SEED=original_comparison.SEED,prepare=prepare_old,
        read_identifiers=original_comparison.read_identifiers,
        _deployment=original_comparison._deployment,
        primary_records=lambda *a,**k:{"records":old_records,"inputs":{},"timing":{"complete":True}})
    monkeypatch.setattr(ev,"joint_context",lambda r:context)
    monkeypatch.setattr(ev,"collocation",fake_collocation)
    monkeypatch.setattr(ev,"comparison",fake_comparison)
    monkeypatch.setattr(ev,"_helpers",lambda:{str(Path(ev.__file__).resolve()):ev.sha(ev.__file__)})
    monkeypatch.setattr(ev,"_label_audit",lambda context:{"path":str(audit_receipt),
        "sha256":ev.sha(audit_receipt),"replays":4,"complete":True})
    monkeypatch.setattr(ev,"_model",lambda *args:ZeroHead() if args[-1]["kind"]!="raw" else None)
    baseline = ev.conditional_baseline

    def counted(initial,*args,**kwargs):
        calls["conditional_histograms_evolved"] += len(initial)
        return baseline(initial,*args,**kwargs)
    monkeypatch.setattr(ev,"conditional_baseline",counted)
    return context,calls


def test_plan_has_fixed_controls_and_honest_two_observation_timing():
    plan = ev.EvaluationPlan()
    assert plan.raw_steps==(2**-7,2**-10)
    assert (plan.cold_calls,plan.warm_calls,plan.conditional_batch)==(1,1,1)
    for override in ({"bootstrap_replicates":8},{"warm_calls":3},{"conditional_batch":64},{"point_error_target":.1}):
        with pytest.raises(ValueError):
            ev.EvaluationPlan(**override)


def test_total_covariance_integrates_real_histograms_and_uses_N_scaling():
    mean = np.array([[[1.,2.]],[[3.,6.]],[[5.,10.]]])
    within = np.broadcast_to(np.eye(2)*4,(3,1,2,2))
    mu,cov = ev.integrate_iid(mean,within,np.ones(3,dtype=bool),n_particles=7)
    np.testing.assert_allclose(mu,[[3.,6.]])
    np.testing.assert_allclose(cov[0],np.eye(2)*4+7*np.cov(mean[:,0].T,ddof=1))
    with pytest.raises(ValueError,match="survivor"):
        ev.integrate_iid(mean,within,np.array([True,False,True]),n_particles=7)


def test_actual_batched_LNA_and_candidate_preserve_initial_and_absorbed_moments():
    config = LNAPayoffCorrectionConfig(horizon=1/64,kappa=.75,n_cells=8,n_particles=512,
        width=2,mean_adam_steps=1,covariance_adam_steps=1,mean_lbfgs_steps=0,covariance_lbfgs_steps=0)
    from nn_dk.payoff_learning import PayoffBank
    bank = PayoffBank.mixed_terminal_integral(config.spec,config.horizon)
    model = LNAPayoffCorrection(config,bank).eval()
    initial = np.ones((2,8))/8
    initial[1,0] = config.spec.mass_stop
    initial[1,1] += 1/8-config.spec.mass_stop
    times = np.array([0.,1/128,1/64])
    baseline = ev.conditional_baseline(initial,config.context,config.spec,times,2**-7,bank,batch_size=2)
    result = ev.corrected_predictions(model,{"kind":"trained"},initial,times,baseline)
    assert result["available"]
    expected = initial@bank.terminal_weights.T
    np.testing.assert_allclose(result["mean"][:,0],expected,atol=2e-15,rtol=0)
    np.testing.assert_allclose(result["mean"][1],np.broadcast_to(expected[1],(3,4)),atol=2e-15,rtol=0)
    np.testing.assert_array_equal(result["scaled_covariance"][:,0],0.)
    np.testing.assert_array_equal(result["scaled_covariance"][1],0.)


def test_failed_baseline_rows_are_retained_without_floors_or_eigenrepair():
    mean = np.zeros((3,2,4))
    cov = np.broadcast_to(np.eye(4),(3,2,4,4)).copy()
    cov[1,:,0,0] = -1.
    baseline = {"mean":mean,"scaled_covariance":cov,
        "diagnostics":{"valid":np.ones(3,dtype=bool),"errors":["","",""]}}
    out = ev.corrected_predictions(None,{"kind":"raw"},np.ones((3,8))/8,[.1,.2],baseline)
    assert not out["available"] and out["valid"].tolist()==[True,False,True]
    assert out["scaled_covariance"][1,0,0,0]==-1.
    assert len(out["errors"])==3 and "indefinite" in out["errors"][1]


def test_missing_joint_fit_lock_fails_before_metadata_or_future_access(tmp_path,monkeypatch):
    calls = []
    def missing(root):
        calls.append("strict-both-fit-locks")
        raise ValueError("all72+18fit locks required")
    monkeypatch.setattr(ev,"_runner",lambda:SimpleNamespace(assert_joint_fit_locks=missing))
    with pytest.raises(ValueError,match="all72"):
        ev.prepare(tmp_path)
    assert calls==["strict-both-fit-locks"] and not list(tmp_path.iterdir())


def test_prepare_freeze_and_shared_reference_wrapper_never_draw_early(joint):
    context,calls = joint
    root = context["root"]
    packet = ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    assert len(packet["inventory"])==24 and len(packet["pools"])==2
    assert calls["final_draws"]==calls["reference_loads"]==0
    assert not (root/"evaluation/initials").exists()
    with pytest.raises(FileNotFoundError):
        ev.generate_shared_references(root)
    assert calls["final_draws"]==0
    lock = ev.freeze(root)
    assert lock["before_any_final_reference_or_independent_initials"]
    assert ev.verify_joint_lock(root)["lock"]==lock
    ev.generate_shared_references(root)
    assert calls["final_draws"]==1


@pytest.mark.parametrize("branch",["reference","deployment/initials","comparison/accuracy"])
def test_preexisting_collocation_outcomes_prevent_retroactive_joint_freeze(joint,branch):
    context,_ = joint
    path = context["collocation_root"]/branch/"opened.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,value=np.ones(1))
    with pytest.raises(ValueError,match="before either study"):
        ev.prepare(context["root"])


def test_source_or_identifier_plan_changes_cannot_resume(joint):
    context,_ = joint
    root = context["root"]
    ev.prepare(root)
    ev.freeze(root)
    p = root/"evaluation/planned_initial_families.npz"
    p.write_bytes(b"changed future family plan")
    with pytest.raises(ValueError,match="evidence changed"):
        ev.verify_joint_lock(root)


def test_chunk_resumption_complete_allocations_and_whole_family_contrasts(joint):
    context,calls = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    # Two conditional groups+two eight-histogram pools,all six methods.
    for _ in range(8):
        out = ev.evaluate(root,max_chunks=1)
        if out["complete"]:
            break
    assert out["complete"]
    groups,records,_ = ev._read_accuracy(ev.verify_joint_lock(root))
    assert len(groups)==4 and len(records)==24
    assert {r["method"]["seed"] for r in records if r["method"]["kind"]=="trained"}=={0,1,2}
    assert all(r["available"] for r in records)
    assert all(r["initial_pool_independent_of_reference"] for r in records if "__ensemble__" in r["identity"]["group"])
    before = calls["conditional_histograms_evolved"]
    ev.evaluate(root)
    assert calls["conditional_histograms_evolved"]==before
    report = ev.contrasts(root)
    assert len(report["records"])==4*3*10
    assert all(r["contrast"]["both_predictor_initial_pool_uncertainties_included"] for r in report["records"])
    assert all(r["pointwise_not_simultaneous"] for r in report["records"])
    complete = ev.load_complete(root,require_timing=False)
    assert complete["allocation"]["candidate_primary_records"]==24
    assert complete["allocation"]["baseline_primary_records"]==64
    # A numerical payload mutation is refused without silently regenerating it.
    numeric = Path(records[0]["arrays"]["path"])
    numeric.write_bytes(b"changed prediction evidence")
    with pytest.raises(ValueError,match="evidence changed"):
        ev.load_complete(root,require_timing=False)


def test_uncached_timing_counts_every_histogram_and_charges_resume_preparation(joint):
    context,calls = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    with pytest.raises(ValueError,match="ALL heavy work"):
        ev.benchmark(root)
    # First two jobs are B1 cold+warm,each evolving a fresh supplied histogram.
    before = calls["conditional_histograms_evolved"]
    out = ev.benchmark(root,final_timing=True,max_jobs=2)
    assert not out["complete"]
    assert calls["conditional_histograms_evolved"]-before==2
    # Next job is one complete uninterrupted cold IID call.
    before = calls["conditional_histograms_evolved"]
    ev.benchmark(root,final_timing=True,max_jobs=1)
    assert calls["conditional_histograms_evolved"]-before==8
    before = calls["conditional_histograms_evolved"]
    ev.benchmark(root,final_timing=True,max_jobs=1)
    # A fresh process/model for warm needs one complete measured warm-up plus
    # the complete uncached warm observation; no stored ODE outputs are used.
    assert calls["conditional_histograms_evolved"]-before==16
    path = root/"evaluation/timing_index.json"
    index = ev.read(path)
    assert len(index["jobs"])==4
    warm = ev.read(index["jobs"][-1]["path"])
    assert warm["accuracy_cache_used"] is False
    assert warm["full_uncached_LNA_per_histogram"]
    assert warm["restoration_and_resume_attempts"][-1]["warm_preparation_seconds"]>0
    assert warm["timing_mode"]=="uninterrupted-full-uncached-workload-v1"
    assert sum(warm["components"].values())==pytest.approx(warm["component_sum_seconds"])
    assert warm["component_sum_seconds"]<=warm["workload_seconds"]+1e-6
    with pytest.raises(ValueError,match="Uninterrupted primary timing"):
        ev.benchmark(root,final_timing=True,max_chunks=1)


def test_complete_cold_warm_allocation_replays_and_missing_job_is_rejected(joint):
    context,calls = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    ev.contrasts(root)
    timing = ev.benchmark(root,final_timing=True)
    assert timing["complete"] and len(timing["jobs"])==48
    complete = ev.primary_records(root)
    assert complete["allocation"]["timing_jobs"]==48
    before = calls["conditional_histograms_evolved"]
    assert ev.benchmark(root,final_timing=True)==timing
    assert calls["conditional_histograms_evolved"]==before
    timing["jobs"].pop()
    ev.write(root/"evaluation/timing_index.json",timing)
    with pytest.raises(ValueError,match="Every literal R4"):
        ev.primary_records(root)


def test_interrupted_whole_job_is_paid_and_repeats_identical_timing_inputs(joint,monkeypatch):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    original = ev.run_uninterrupted_workload
    attempted = []
    def interrupted(*args):
        value = original(*args)
        attempted.append(value[0]["initial"].copy())
        if len(attempted)==1:
            raise KeyboardInterrupt("synthetic interrupted observation")
        return value
    monkeypatch.setattr(ev,"run_uninterrupted_workload",interrupted)
    with pytest.raises(KeyboardInterrupt):
        ev.benchmark(root,final_timing=True,max_jobs=1)
    index = ev.benchmark(root,final_timing=True,max_jobs=1)
    np.testing.assert_array_equal(attempted[0],attempted[1])
    record = ev.read(index["jobs"][0]["path"])
    attempts = record["restoration_and_resume_attempts"]
    assert len(attempts)==2 and not attempts[0]["successful"] and attempts[1]["successful"]
    assert attempts[0]["workload_seconds"]>0
    assert record["attempt_cost_complete"]
    assert record["paid_attempt_seconds"]==pytest.approx(sum(a["elapsed_seconds"] for a in attempts))
    assert record["paid_attempt_seconds"]>record["workload_seconds"]


def test_segmented_only_timing_record_cannot_qualify_as_primary(joint):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    index = ev.benchmark(root,final_timing=True,max_jobs=1)
    path = Path(index["jobs"][0]["path"])
    record = ev.read(path)
    identity = record["identity"]
    record["timing_mode"] = "segmented-service-only"
    ev.write(path,record)
    with pytest.raises(ValueError,match="actual uninterrupted"):
        ev._read_timing_job(ev.verify_joint_lock(root),path,identity)


def test_primary_counter_includes_recurring_configuration_law_and_rng_setup(joint,monkeypatch):
    context,_ = joint
    context = {**context,"plan":ev.EvaluationPlan(baseline_batch_size=4)}
    clock = [0.]
    def counted(function):
        def call(*args,**kwargs):
            clock[0] += 1.
            return function(*args,**kwargs)
        return call
    for name in ("reference_configuration","payoff_bank","profile_probabilities"):
        monkeypatch.setattr(ev.collocation,name,counted(getattr(ev.collocation,name)))
    monkeypatch.setattr(ev,"seeded_rng",counted(ev.seeded_rng))
    monkeypatch.setattr(ev.time,"perf_counter",lambda:clock[0])
    case = context["cases"][0]
    job = {"case":case,"profile":"uniform","role":"ensemble","families":4,
           "method":{"kind":"raw","baseline_step":2**-7}}
    arrays,parts,seconds,available = ev.run_uninterrupted_workload(context,job,None)
    assert available and len(arrays["initial"])==4
    assert seconds==4. and parts["initialization_seconds"]==4.


@pytest.mark.parametrize("kind",["initial","baseline","prediction"])
@pytest.mark.parametrize("missing",["metadata","arrays"])
def test_committed_numeric_pairs_never_redraw_or_overwrite_an_orphan(joint,monkeypatch,kind,missing):
    context,calls = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    directory,pattern = {"initial":("initials","*.json"),"baseline":("baselines","dt7_0.json"),
                         "prediction":("predictions","chunk0.json")}[kind]
    metadata = sorted((root/"evaluation"/directory).rglob(pattern))[0]
    numeric = Path(ev.read(metadata)["arrays"]["path"])
    remove,survivor = (metadata,numeric) if missing=="metadata" else (numeric,metadata)
    unchanged = survivor.read_bytes()
    remove.unlink()
    before = dict(calls)
    def forbidden(*args,**kwargs):
        raise AssertionError("No new initial draw,ODE or head call after lost committed evidence")
    monkeypatch.setattr(ev,"seeded_rng",forbidden)
    monkeypatch.setattr(ev,"conditional_baseline",forbidden)
    monkeypatch.setattr(ZeroHead,"predict_conditional",forbidden)
    expected = ValueError if missing=="metadata" else FileNotFoundError
    with pytest.raises(expected):
        ev.evaluate(root)
    assert calls["conditional_histograms_evolved"]==before["conditional_histograms_evolved"]
    assert calls["final_draws"]==before["final_draws"]
    assert survivor.read_bytes()==unchanged and not remove.exists()


def test_committed_whole_timing_arrays_cannot_be_overwritten_after_metadata_loss(joint,monkeypatch):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    index = ev.benchmark(root,final_timing=True,max_jobs=1)
    path = Path(index["jobs"][0]["path"])
    arrays = Path(ev.read(path)["arrays"]["path"])
    unchanged = arrays.read_bytes()
    path.unlink()
    def forbidden(*args,**kwargs):
        raise AssertionError("A committed timing payload cannot be silently rerun")
    monkeypatch.setattr(ev,"run_uninterrupted_workload",forbidden)
    with pytest.raises(ValueError,match="Orphan"):
        ev.benchmark(root,final_timing=True,max_jobs=1)
    assert arrays.read_bytes()==unchanged and not path.exists()


@pytest.mark.parametrize("stage",ev.CLI_STAGES)
def test_every_command_records_nonadditive_invocation_cost_in_separate_namespace(tmp_path,monkeypatch,stage):
    def dispatch(args,parser):
        return {"case":{"status":"complete","committed_chunks":1}} if stage=="reference" else {"complete":True}
    monkeypatch.setattr(ev,"_dispatch",dispatch)
    assert ev.main([stage,"--root",str(tmp_path)])==0
    records = ev.invocation_receipts(tmp_path)
    assert len(records)==1
    record = records[0]
    assert record["stage"]==stage and record["status"]=="success" and record["finished"]
    assert not record["whole_process"]  # Explicitly a software in-process API test.
    assert record["nonadditive_contains_nested_work"]
    assert not record["nested_prediction_or_cache_costs_added_again"]
    assert record["wall_seconds"]>=0 and record["cpu_seconds"]>=0
    assert record["process_cpu_seconds_at_finish"]>=record["cpu_seconds"]
    assert not list((tmp_path/"execution").glob("cli_*.json"))
    assert Path(record["receipt_path"]).parent==tmp_path/"evaluation/execution"


def test_actual_subprocess_records_imports_and_failed_preflight_without_future_draws(tmp_path):
    root = tmp_path/"no-fit-lock"
    result = subprocess.run([sys.executable,str(Path(ev.__file__).resolve()),"prepare","--root",str(root)],
        cwd=Path(ev.__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode!=0
    records = ev.invocation_receipts(root)
    assert len(records)==1 and records[0]["whole_process"]
    assert records[0]["status"]=="error" and records[0]["error"]
    assert records[0]["wall_seconds"]>0 and records[0]["cpu_seconds"]>0
    assert not (root/"reference").exists() and not (root/"evaluation/initials").exists()


def test_checkpointed_invocation_cost_is_retained_and_receipt_corruption_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(ev,"_dispatch",lambda *args:{"complete":False})
    assert ev.main(["evaluate","--root",str(tmp_path)])==75
    record = ev.invocation_receipts(tmp_path)[0]
    assert record["status"]=="checkpointed" and record["exit_code"]==75
    path = Path(record["receipt_path"])
    original = ev.read(path)
    original["wall_seconds"] = -1.
    ev.write(path,original)
    with pytest.raises(ValueError,match="invocation receipt"):
        ev.invocation_receipts(tmp_path)


def test_posthoc_verification_receipt_does_not_change_frozen_work_inputs(joint,monkeypatch):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    before = ev.load_complete(root,require_timing=False,require_contrasts=False)
    monkeypatch.setattr(ev,"_dispatch",lambda *args:{"complete":True})
    assert ev.main(["verify","--root",str(root)])==0
    assert len(ev.invocation_receipts(root))==1
    after = ev.load_complete(root,require_timing=False,require_contrasts=False)
    assert after["inputs"]==before["inputs"]
    assert after["invocations"]==before["invocations"]==[]
    assert ev.invocation_receipts(root,include_verification=False)==[]


def test_one_failed_initial_makes_group_unavailable_without_dropping_it(joint,monkeypatch):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    original = ev.conditional_baseline
    def failed(initial,*args,**kwargs):
        out = original(initial,*args,**kwargs)
        if len(initial)>1:
            out["mean"][0] = np.nan
            out["scaled_covariance"][0] = np.nan
            out["diagnostics"]["valid"][0] = False
            out["diagnostics"]["errors"][0] = "synthetic physical failure"
        return out
    monkeypatch.setattr(ev,"conditional_baseline",failed)
    ev.evaluate(root)
    evidence = ev.load_complete(root,require_timing=False,require_contrasts=False)
    unavailable = [r for r in evidence["records"] if not r["available"]]
    assert len(unavailable)==12
    for record in unavailable:
        assert record["metrics"] is None
        with np.load(record["arrays"]["path"]) as a:
            assert len(a["family_ids"])==8 and np.count_nonzero(~a["valid"])==2
        assert len(record["failed_family_ids"])==2


def test_changed_or_duplicated_contrast_cannot_hide_a_fixed_seed(joint):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    result = ev.contrasts(root)
    result["records"][-1] = result["records"][0]
    ev.write(root/"evaluation/contrasts.json",result)
    with pytest.raises(ValueError,match="All candidate/control"):
        ev.load_complete(root,require_timing=False)


def test_removed_prediction_chunk_blocks_full_evidence(joint):
    context,_ = joint
    root = context["root"]
    ev.prepare(root,ev.EvaluationPlan(chunk_paths=4,baseline_batch_size=4))
    ev.freeze(root)
    ev.generate_shared_references(root)
    ev.evaluate(root)
    _,records,_ = ev._read_accuracy(ev.verify_joint_lock(root))
    ensemble = next(r for r in records if "__ensemble__" in r["identity"]["group"])
    Path(ensemble["predictions"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        ev.load_complete(root,require_timing=False,require_contrasts=False)


def test_canonical_evaluators_remain_bound_to_their_original_studies():
    from analysis import evaluate_payoff_regimes, run_payoff_regimes
    assert evaluate_payoff_regimes._runner() is run_payoff_regimes
    assert ev.comparison._base._runner() is actual_collocation
