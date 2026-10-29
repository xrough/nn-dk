"""Jointly locked finite-Euler evaluation of nonlinear-LNA corrections.

No final reference or independent initial histogram is opened by preparation.
Both studies must lock every fit before choices, sources and identifier plans
freeze. IID prediction evolves each histogram's own conditional LNA, followed
by total covariance; analytic IID Gaussian propagation is a separate control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

_PROCESS_WALL_START = time.monotonic()
_PROCESS_CPU_START = time.process_time()
_PROCESS_UNIX_START = time.time()

for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/"src"))

from analysis import evaluate_payoff_collocation_study as comparison  # noqa: E402
from analysis import run_payoff_collocation_study as collocation  # noqa: E402
from analysis.batched_native_gaussian import conditional_gaussian_batch  # noqa: E402
from analysis.evaluate_path_generalization import (  # noqa: E402
    array_sha,
    bootstrap_counts,
    mixture_moments,
    moment_comparison,
)
from analysis.evaluate_payoff_boundary_regimes import predictor_advantage  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "joint-locked-independent-IID-nonlinear-LNA-correction-evaluation-v1"
SEED = 2026110719
CONTROL_NAMES = ("gated_zero","raw_conditional_LNA_dt7","raw_conditional_LNA_dt10")
CLI_STAGES = ("prepare","freeze","reference","evaluate","contrasts","benchmark","verify")
WORK_STAGES = tuple(stage for stage in CLI_STAGES if stage!="verify")
INVOCATION_KIND = KIND+"-whole-process-invocation-v1"


@dataclass(frozen=True)
class EvaluationPlan:
    chunk_paths: int = 256
    baseline_batch_size: int = 64
    bootstrap_replicates: int = 128
    seed: int = SEED
    raw_steps: tuple[float,...] = (2**-7,2**-10)
    psd_tolerance: float = 1e-9
    point_error_target: float = .05
    uncertainty_component_target: float = .0125
    cold_calls: int = 1
    warm_calls: int = 1
    conditional_batch: int = 1
    timing_mode: str = "uninterrupted-full-uncached-workload-v1"

    def __post_init__(self):
        object.__setattr__(self,"raw_steps",tuple(self.raw_steps))
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in (
                "chunk_paths","baseline_batch_size","bootstrap_replicates","cold_calls","warm_calls","conditional_batch")):
            raise ValueError("Positive integer chunk and literal workload allocations required")
        if (self.bootstrap_replicates!=128 or self.raw_steps!=(2**-7,2**-10)
                or (self.cold_calls,self.warm_calls,self.conditional_batch)!=(1,1,1)
                or (self.point_error_target,self.uncertainty_component_target)!=(.05,.0125)
                or self.timing_mode!="uninterrupted-full-uncached-workload-v1"
                or not math.isfinite(self.psd_tolerance) or self.psd_tolerance<=0 or type(self.seed) is not int):
            raise ValueError("Retain declared controls,128 bootstraps,one cold/one warm,B1 and5%/1.25% criteria")


def _runner():
    from analysis import run_lna_payoff_correction_study
    return run_lna_payoff_correction_study


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    def encode(item):
        if isinstance(item,np.ndarray):
            return item.tolist()
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=encode)+"\n")
    temp.replace(path)


def immutable(path,value):
    # JSON normalization includes tuple/list equivalence, never numeric changes.
    value = json.loads(json.dumps(value,allow_nan=False))
    path = Path(path)
    if path.exists():
        if read(path)!=value:
            raise ValueError(f"Frozen R4 evaluation metadata changed: {path}")
    else:
        write(path,value)
    return value


def save_arrays(path,values):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)
    return {"path":str(path.resolve()),"sha256":sha(path)}


def artifact(path,inputs=None,expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest!=expected:
        raise ValueError(f"R4 evidence changed: {path}")
    if inputs is not None:
        inputs[str(path)] = digest
    return path


def joint_context(root):
    root = Path(root).resolve()
    torch.set_num_threads(1)
    runner = _runner()
    locked = runner.assert_joint_fit_locks(root)
    other = Path(locked["collocation_root"]).resolve()
    study,old = runner.load_plan(root),collocation.load_plan(other)
    cases = tuple(runner.case_names(study))
    if cases!=tuple(collocation.case_names(old)) or tuple(study.optimizer_seeds)!=(0,1,2):
        raise ValueError("Both complete case grids and all three optimizer seeds must agree")
    expected = {(s["case"],s["seed"],s["name"]) for s in runner.slots(study)}
    actual = [(s["case"],s["seed"],s["name"]) for s in locked["candidate"]["models"]]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError("Every declared R4 fitted case/seed is required")
    for case in cases:
        cfg = collocation.reference_configuration(case,old)
        config = runner.configuration(case,study)
        if (runner.payoff_bank(case,study).bank_hash!=collocation.payoff_bank(case,old).bank_hash
                or config.spec.target_id!=cfg.spec.target_id or config.context.context_id!=cfg.context("uniform").context_id
                or tuple(config.query_times)!=tuple(cfg.query_times)):
            raise ValueError("R4 and collocation physical targets,bank and query times must agree")
    return {"root":root,"runner":runner,"study":study,"collocation_root":other,
            "collocation_study":old,"candidate_lock":locked["candidate"],"collocation_lock":locked["collocation"],"cases":cases}


def methods(context,case):
    rows = [m for m in context["candidate_lock"]["models"] if m["case"]==case]
    if len(rows)!=3 or {m["seed"] for m in rows}!={0,1,2}:
        raise ValueError("All three correction seeds must remain")
    step = context["study"].baseline_step
    return [{"name":m["name"],"family":"LNA-correction","seed":m["seed"],"kind":"trained",
             "baseline_step":step,"checkpoint_sha256":m["checkpoint_sha256"]} for m in rows]+[
        {"name":"gated_zero","family":"gated-LNA","kind":"gated_zero","seed":None,"baseline_step":step},
        *[{"name":f"raw_conditional_LNA_dt{round(-np.log2(s))}","family":"raw-conditional-LNA",
           "kind":"raw","seed":None,"baseline_step":s} for s in (2**-7,2**-10)]]


def _helpers():
    names = ("evaluate_lna_payoff_correction.py","run_lna_payoff_correction_study.py",
        "report_lna_payoff_correction.py","audit_lna_supervised_labels.py",
        "fit_lna_payoff_correction.py","batched_native_gaussian.py","native_gaussian_baseline.py",
        "run_payoff_collocation_study.py","evaluate_payoff_collocation_study.py",
        "report_payoff_collocation_study.py","report_payoff_regimes.py","run_payoff_regimes.py",
        "evaluate_payoff_regimes.py","evaluate_payoff_regimes_deployment.py",
        "evaluate_payoff_boundary_regimes.py","evaluate_path_generalization.py",
        "benchmark_payoff_generalization.py","payoff_study_adapter.py","compiled_native.py","native_euler_control.cpp")
    paths = [REPO/"analysis"/name for name in names]
    if any(not p.is_file() or not p.stat().st_size for p in paths):
        raise ValueError("All joint evaluation and numerical helper sources must exist before freezing")
    return {str(p):sha(p) for p in paths}


def _outcomes(context):
    found = []
    for root,branches in ((context["collocation_root"],("reference","deployment/initials","comparison/accuracy","deployment/accuracy")),
                          (context["root"],("evaluation/initials","evaluation/baselines","evaluation/predictions","evaluation/accuracy","evaluation/timing"))):
        for branch in branches:
            directory = root/branch
            for p in directory.rglob("*") if directory.exists() else ():
                if p.is_file() and p.suffix.lower() in (".h5",".hdf5",".npz",".npy"):
                    found.append(str(p))
    return found


def _inventory(context,plan):
    result,pools = [],[]
    for case in context["cases"]:
        cfg = collocation.reference_configuration(case,context["collocation_study"])
        for group in cfg.group_keys:
            result.extend({"case":case,"group":group,"method":m["name"],"queries":list(cfg.query_times)}
                          for m in methods(context,case))
        for profile in cfg.cases:
            key = case+"__"+profile
            pools.append({"key":key,"case":case,"profile":profile,"group":profile+"__ensemble__0",
                "families":cfg.paths_per_group,"seed":int(stable_hash([KIND,plan.seed,key,"initials"])[:12],16),
                "bootstrap_seed":int(stable_hash([KIND,plan.seed,key,"initial-bootstrap"])[:12],16)})
    return result,pools


def _planned_ids(pool,plan):
    return np.array([stable_hash([KIND,plan.seed,pool["case"],pool["profile"],i])
                     for i in range(pool["families"])],dtype="S64")


def _id_audit(context,pools,ids,plan):
    """Inspect identifiers only; never learner/final payoff values."""
    learner = set()
    # R4 anchors,parents and continuation-family identities are metadata arrays.
    declared = read(context["root"]/"data/data_lock.json")["identifier_inventory"]
    for item in declared:
        p = artifact(item["path"],expected=item["sha256"])
        with np.load(p) as a:
            for key in ("family_ids","parent_family_ids"):
                if key in item["fields"]:
                    learner.update(a[key].astype(str).reshape(-1).tolist())
    for case in context["cases"]:
        for role in ("train","validation"):
            p = context["collocation_root"]/f"payoffs/{case}/states_{role}.h5"
            found,_ = comparison.read_identifiers(p)
            learner.update(found)
    seen,reference,candidates = set(),set(),set()
    for case in context["cases"]:
        cfg = collocation.reference_configuration(case,context["collocation_study"])
        for group in cfg.group_keys:
            profile,role,anchor = group.split("__")
            reference.update(stable_hash([collocation.STREAM,cfg.seed,cfg.config_hash,"reference-"+role,
                                          profile,int(anchor),i]) for i in range(cfg.paths_per_group))
            candidates.update(stable_hash([comparison.SEED,case,group,"independent-candidate",i])
                              for i in range(cfg.paths_per_group))
    other = context["collocation_root"]/"deployment/planned_initial_families.npz"
    with np.load(other) as a:
        for key in a.files:
            seen.update(a[key].astype(str).tolist())
    excluded = learner|reference|candidates|seen
    seen = set()
    addresses = set()
    for pool in pools:
        values = ids[pool["key"]].astype(str).tolist()
        if len(values)!=pool["families"] or len(set(values))!=len(values) or seen.intersection(values) or excluded.intersection(values):
            raise ValueError("Future R4 integration families overlap learner/reference/comparison families")
        seen.update(values)
        addresses.add((pool["seed"],KIND,"independent-initials"))
        refseed = int(stable_hash([comparison.SEED,pool["case"],pool["group"]])[:12],16)
        old_pool = comparison._deployment.initial_bootstrap_address(pool["case"],pool["profile"])
        if pool["bootstrap_seed"] in (refseed,old_pool):
            raise ValueError("Reference and both independent integration bootstrap streams must differ")
    if len(addresses)!=len(pools):
        raise ValueError("Distinct independent R4 pool addresses required")
    return {"complete":True,"learner_identifiers":len(learner),"reference_identifiers":len(reference),
        "native_candidate_identifiers":len(candidates),"new_initial_identifiers":len(seen),
        "new_identifier_inventory_sha256":stable_hash(sorted(seen)),"families_disjoint":True,
        "reference_and_two_initial_bootstrap_streams_independent":True,
        "final_physical_histograms_or_outcomes_opened_or_drawn":False}


def _label_audit(context):
    from analysis import audit_lna_supervised_labels as audit
    value = audit.verify(context["root"],require_complete=True)
    expected = audit.allocation(context["study"])
    if not value["complete"] or len(value["records"])!=len(expected):
        raise ValueError("Complete predetermined native learner-label replay required before final freeze")
    path = context["root"]/audit.RECEIPT
    return {"path":str(path),"sha256":sha(path),"replays":len(expected),"complete":True}


def _packet(context,plan,pools,inventory):
    root,other = context["root"],context["collocation_root"]
    return {"kind":KIND,"source_hash":source_hash(),"plan":asdict(plan),"helpers":_helpers(),
        "candidate_root":str(root),"collocation_root":str(other),
        "candidate_fit_lock_sha256":sha(root/"payoffs/fit_lock.json"),
        "collocation_fit_lock_sha256":sha(other/"payoffs/fit_lock.json"),
        "collocation_evaluation_lock_sha256":sha(other/"comparison/evaluation_lock.json"),
        "collocation_evaluation_packet":read(other/"comparison/evaluation_lock.json"),
        "candidate_native_label_audit":_label_audit(context),
        "candidate_plan_sha256":sha(root/"plan.json"),"cases":list(context["cases"]),
        "methods":{case:methods(context,case) for case in context["cases"]},
        "inventory":inventory,"pools":pools,
        "cache":"accuracy-only per-family conditional LNA cache shared across optimizer seeds and controls; standalone timing recomputes all ODEs",
        "IID_integration":"fresh independent histograms; E[conditional covariance]+unbiased covariance of conditional means",
        "bootstrap":"128 whole-family reference resamples common with72 evaluator; independent fresh R4 initial resamples, existing72 initial resamples separate; pointwise",
        "baseline_law":"nonlinear mean plus linear-noise covariance,already nonlinear; raw controls unstopped",
        "cost":"one uninterrupted cold plus one uninterrupted warm uncached B1 and fullIID call per method/profile;histogram generation+all ODE/head/totalcovariance timed;restoration separate;two observations,not stable latency distribution",
        "whole_process_cost_receipts":{"schema_version":1,"kind":INVOCATION_KIND,
            "path_pattern":"evaluation/execution/cli_{stage}_{pid}_{invocation_ns}.json",
            "work_stages":list(WORK_STAGES),"posthoc_verification":"recorded separately;excluded from frozen work inputs/costs",
            "scope":"nonadditive whole-command imports,preflight and work;contains nested prediction/cache costs;future receipt bytes not frozen here"},
        "all_stopped_and_failed_families_retained":True,"post_test_selection_or_refitting":False,
        "covariance_floors_clipping_or_eigenrepair":False,"continuous_stopping_or_continuum_certificate":False}


def prepare(root,plan=None):
    """After both fit locks, freeze metadata/identifiers without final draws."""
    context = joint_context(root)
    root,other = context["root"],context["collocation_root"]
    output = root/"evaluation/preregistration.json"
    plan = EvaluationPlan(**read(output)["plan"]) if plan is None and output.exists() else EvaluationPlan() if plan is None else plan
    if (root/"evaluation/joint_lock.json").exists():
        return verify_joint_lock(root)["packet"]
    if _outcomes(context):
        raise ValueError("Joint choices must freeze before either study materializes any final outcomes or IID histograms")
    comparison.prepare(other)
    collocation.freeze_evaluation(other)
    inventory,pools = _inventory(context,plan)
    ids = {p["key"]:_planned_ids(p,plan) for p in pools}
    identity_path = root/"evaluation/planned_initial_families.npz"
    if identity_path.exists():
        with np.load(identity_path) as a:
            if set(a.files)!=set(ids) or any(not np.array_equal(a[k],v) for k,v in ids.items()):
                raise ValueError("Planned R4 independent family identifiers changed")
    else:
        save_arrays(identity_path,ids)
    for pool in pools:
        pool["family_ids_array_sha256"] = array_sha(ids[pool["key"]])
    immutable(root/"evaluation/identifier_audit.json",_id_audit(context,pools,ids,plan))
    return immutable(output,_packet(context,plan,pools,inventory))


def freeze(root):
    """The candidate joint gate is additional to the unchanged72-model gate."""
    context = joint_context(root)
    root = context["root"]
    if (root/"evaluation/joint_lock.json").exists():
        return verify_joint_lock(root)["lock"]
    with collocation._exclusive(root/"evaluation/execution/.freeze.lock"):
        if _outcomes(context):
            raise ValueError("All joint model/criterion/source choices must precede final materialization")
        packet = prepare(root)
        if _outcomes(context):
            raise ValueError("A final outcome appeared before the joint freeze")
        value = {"kind":KIND,"complete":True,"source_hash":source_hash(),
            "packet_sha256":sha(root/"evaluation/preregistration.json"),
            "planned_families_sha256":sha(root/"evaluation/planned_initial_families.npz"),
            "identifier_audit_sha256":sha(root/"evaluation/identifier_audit.json"),
            "candidate_fit_lock_sha256":packet["candidate_fit_lock_sha256"],
            "collocation_fit_lock_sha256":packet["collocation_fit_lock_sha256"],
            "collocation_evaluation_lock_sha256":packet["collocation_evaluation_lock_sha256"],
            "before_any_final_reference_or_independent_initials":True}
        return immutable(root/"evaluation/joint_lock.json",value)


def verify_joint_lock(root):
    context = joint_context(root)
    root,other = context["root"],context["collocation_root"]
    lock,packet = read(root/"evaluation/joint_lock.json"),read(root/"evaluation/preregistration.json")
    plan = EvaluationPlan(**packet["plan"])
    collocation.verify_evaluation_lock(other)
    inventory,pools = _inventory(context,plan)
    path = artifact(root/"evaluation/planned_initial_families.npz",expected=lock["planned_families_sha256"])
    with np.load(path) as ids:
        if set(ids.files)!={p["key"] for p in pools}:
            raise ValueError("Exact independent pool allocation required")
        for pool in pools:
            values = _planned_ids(pool,plan)
            if not np.array_equal(values,ids[pool["key"]]):
                raise ValueError("Independent family plan changed")
            pool["family_ids_array_sha256"] = array_sha(values)
    expected = json.loads(json.dumps(_packet(context,plan,pools,inventory)))
    if (not lock.get("complete") or lock.get("source_hash")!=source_hash() or packet!=expected
            or sha(root/"evaluation/preregistration.json")!=lock["packet_sha256"]
            or sha(root/"evaluation/identifier_audit.json")!=lock["identifier_audit_sha256"]
            or any(lock[k]!=packet[k] for k in ("candidate_fit_lock_sha256","collocation_fit_lock_sha256","collocation_evaluation_lock_sha256"))
            or lock.get("before_any_final_reference_or_independent_initials") is not True):
        raise ValueError("Exact joint source/model/criteria/identifier lock changed")
    return {**context,"plan":plan,"packet":packet,"lock":lock,"joint_lock_sha256":sha(root/"evaluation/joint_lock.json")}


def generate_shared_references(root,case=None,*,deadline=None,max_chunks=None):
    context = verify_joint_lock(root)
    return collocation.generate_reference(context["collocation_root"],case,deadline,max_chunks)


def conditional_baseline(initial,context,spec,times,step,bank,*,batch_size=64):
    return conditional_gaussian_batch(initial,context,spec,times,step,bank.terminal_weights,
        bank.running_weights,batch_size=batch_size,return_native=False)


def checked_moments(mean,covariance,tolerance=1e-9):
    mean,covariance = np.asarray(mean),np.asarray(covariance)
    if mean.ndim!=3 or covariance.shape!=(*mean.shape,mean.shape[-1]):
        raise ValueError("Aligned per-histogram three-dimensional means/full covariances required")
    finite = np.isfinite(mean).all(axis=(1,2))&np.isfinite(covariance).all(axis=(1,2,3))
    valid = np.zeros(len(mean),dtype=bool)
    for i in np.flatnonzero(finite):
        valid[i] = (np.max(abs(covariance[i]-covariance[i].swapaxes(-1,-2)))<=tolerance
                    and np.linalg.eigvalsh(covariance[i]).min()>=-tolerance)
    return valid


def _model(context,case,method):
    if method["kind"]=="raw":
        return None
    if method["kind"]=="trained":
        item = next(m for m in context["candidate_lock"]["models"] if m["case"]==case and m["name"]==method["name"])
        return context["runner"].load_declared_model(item).eval()
    from analysis.fit_lna_payoff_correction import LNAPayoffCorrection
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return LNAPayoffCorrection(context["runner"].configuration(case,context["study"]),
            context["runner"].payoff_bank(case,context["study"])).eval()


def corrected_predictions(model,method,initial,times,baseline,*,tolerance=1e-9):
    """Keep failed rows aligned; a single failure prohibits group aggregation."""
    mean = np.array(baseline["mean"],copy=True)
    covariance = np.array(baseline["scaled_covariance"],copy=True)
    valid = np.asarray(baseline["diagnostics"]["valid"],dtype=bool).copy()
    errors = list(baseline["diagnostics"]["errors"])
    if valid.shape!=(len(initial),) or len(errors)!=len(initial):
        raise ValueError("Per-anchor baseline validity flags must remain aligned")
    if method["kind"]!="raw":
        def execute(indices):
            if not len(indices):
                return
            try:
                with torch.no_grad():
                    prediction = model.predict_conditional(initial[indices],times,mean[indices],covariance[indices],kind=method["kind"])
                for key,target in (("mean",mean),("scaled_covariance",covariance)):
                    value = prediction[key]
                    target[indices] = value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
            except (ValueError,FloatingPointError) as error:
                if len(indices)>1:
                    middle = len(indices)//2
                    execute(indices[:middle])
                    execute(indices[middle:])
                else:
                    valid[indices[0]] = False
                    errors[int(indices[0])] = type(error).__name__+": "+str(error)
        execute(np.flatnonzero(valid))
    output_valid = checked_moments(mean,covariance,tolerance)
    for i in np.flatnonzero(valid&~output_valid):
        errors[int(i)] = "Nonfinite,asymmetric or materially indefinite prediction; no repair"
    valid &= output_valid
    for i in np.flatnonzero(~valid):
        if not errors[int(i)]:
            errors[int(i)] = "Conditional baseline or correction unavailable;family retained"
    return {"mean":mean,"scaled_covariance":covariance,"valid":valid,"errors":errors,
            "all_families_retained":True,"available":bool(valid.all())}


def integrate_iid(mean,covariance,valid,*,n_particles,counts=None):
    valid = np.asarray(valid,dtype=bool)
    if valid.shape!=(len(mean),) or not valid.all():
        raise ValueError("A failed family makes the whole group unavailable; survivor integration forbidden")
    return mixture_moments(mean,covariance,n_particles=n_particles,counts=counts)


def _unit(path,identity):
    path = Path(path)
    if not path.exists():
        if path.with_suffix(".npz").exists():
            raise ValueError("Orphan saved R4 arrays lack metadata;refuse redraw,recomputation or overwrite")
        return None
    value = read(path)
    if not value.get("complete") or value.get("identity")!=identity:
        raise ValueError("A completed R4 numeric job changed identity")
    artifact(value["arrays"]["path"],expected=value["arrays"]["sha256"])
    return value


def _save_unit(path,identity,arrays,**extra):
    evidence = save_arrays(Path(path).with_suffix(".npz"),arrays)
    value = {"complete":True,"identity":identity,"arrays":evidence,**extra}
    write(path,value)
    return value


def _initial_pool(context,pool):
    root = context["root"]
    identity = {"joint_lock_sha256":context["joint_lock_sha256"],"pool":pool,"kind":KIND+"-independent-initials"}
    path = root/"evaluation/initials"/pool["case"]/(pool["profile"]+".json")
    previous = _unit(path,identity)
    if previous is not None:
        return previous
    cfg = collocation.reference_configuration(pool["case"],context["collocation_study"])
    probabilities = collocation.profile_probabilities(cfg,pool["profile"])[0]
    initial = seeded_rng(pool["seed"],KIND,"independent-initials").multinomial(
        cfg.n_particles,probabilities,size=pool["families"])/cfg.n_particles
    ids = _planned_ids(pool,context["plan"])
    if array_sha(ids)!=pool["family_ids_array_sha256"]:
        raise ValueError("Independent family identifiers changed before materialization")
    return _save_unit(path,identity,{"initial":initial,"family_ids":ids},
        independent_of_reference=True,initials=pool["families"],all_initial_histograms_retained=True)


def _baseline_chunk(context,case,group,initial,ids,start,step):
    root = context["root"]
    identity = {"joint_lock_sha256":context["joint_lock_sha256"],"case":case,"group":group,
        "step":step,"start":start,"initial_sha256":array_sha(initial),"family_ids_sha256":array_sha(ids),
        "batch_size":context["plan"].baseline_batch_size,"purpose":"shared accuracy cache,not standalone inference"}
    path = root/"evaluation/baselines"/case/group/f"dt{round(-np.log2(step))}_{start}.json"
    previous = _unit(path,identity)
    if previous is not None:
        return previous
    cfg = collocation.reference_configuration(case,context["collocation_study"])
    bank = collocation.payoff_bank(case,context["collocation_study"])
    tick = time.perf_counter()
    baseline = conditional_baseline(initial,cfg.context(group.split("__")[0]),cfg.spec,cfg.query_times,
        step,bank,batch_size=context["plan"].baseline_batch_size)
    seconds = time.perf_counter()-tick
    diagnostic = baseline["diagnostics"]
    values = {"mean":baseline["mean"],"scaled_covariance":baseline["scaled_covariance"],"family_ids":ids,
        "valid":np.asarray(diagnostic["valid"],dtype=bool),"errors":np.asarray(diagnostic["errors"],dtype=str)}
    for key in ("minimum_deterministic_mass","deterministic_boundary_encountered",
                "minimum_covariance_eigenvalue","covariance_symmetry_error"):
        if key in diagnostic:
            values[key] = diagnostic[key]
    return _save_unit(path,identity,values,baseline_ODE_seconds=seconds,
        actual_conditional_histograms=len(initial),batch_size=context["plan"].baseline_batch_size)


def _predicted_chunk(context,case,group,initial,ids,start,method,model,base):
    identity = {"joint_lock_sha256":context["joint_lock_sha256"],"case":case,"group":group,
        "method":method,"start":start,"baseline_sha256":base["arrays"]["sha256"],
        "family_ids_sha256":array_sha(ids)}
    path = context["root"]/"evaluation/predictions"/case/group/method["name"]/f"chunk{start}.json"
    previous = _unit(path,identity)
    if previous is not None:
        return previous
    cfg = collocation.reference_configuration(case,context["collocation_study"])
    with np.load(base["arrays"]["path"]) as a:
        baseline = {"mean":a["mean"],"scaled_covariance":a["scaled_covariance"],
                    "diagnostics":{"valid":a["valid"],"errors":a["errors"].astype(str).tolist()}}
    tick = time.perf_counter()
    result = corrected_predictions(model,method,initial,cfg.query_times,baseline,tolerance=context["plan"].psd_tolerance)
    seconds = time.perf_counter()-tick
    return _save_unit(path,identity,{k:np.asarray(result[k]) for k in ("mean","scaled_covariance","valid","errors")}|
        {"family_ids":ids},available=result["available"],head_seconds=seconds,
        accuracy_cache_shared=True,standalone_cost_includes_full_ODE=True)


def _reference_counts(case,group,count):
    return bootstrap_counts(count,int(stable_hash([comparison.SEED,case,group])[:12],16),128)


def evaluate(root,*,cases=None,max_chunks=None):
    """Resumable accuracy only. Every method retains the complete family pool."""
    context = verify_joint_lock(root)
    requested = tuple(cases) if cases is not None else context["cases"]
    if not requested or set(requested)-set(context["cases"]):
        raise ValueError("Declared evaluation cases required")
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks<0):
        raise ValueError("Nonnegative resumable chunk budget required")
    root,done = context["root"],0
    with collocation._exclusive(root/"evaluation/execution/.accuracy.lock"):
        for case in requested:
            cfg = collocation.reference_configuration(case,context["collocation_study"])
            models = {m["name"]:_model(context,case,m) for m in methods(context,case)}
            for group in cfg.group_keys:
                profile,role,_ = group.split("__")
                reference = collocation.load_reference(context["collocation_root"],case,group)
                if role=="ensemble":
                    pool = next(p for p in context["packet"]["pools"] if (p["case"],p["profile"])==(case,profile))
                    record = _initial_pool(context,pool)
                    with np.load(record["arrays"]["path"]) as a:
                        initial,ids = a["initial"],a["family_ids"]
                    if set(ids.astype(str)).intersection(reference["family_ids"].astype(str)):
                        raise ValueError("Candidate IID pool overlaps physical reference families")
                else:
                    initial = np.asarray(reference["metadata"]["conditional_initial_json"])
                    if initial.ndim==0:
                        initial = np.asarray(json.loads(str(initial)))
                    initial = initial[None]
                    ids = np.array([stable_hash([KIND,case,group,"conditional-map"])],dtype="S64")
                chunks = {m["name"]:[] for m in methods(context,case)}
                for start in range(0,len(initial),context["plan"].chunk_paths):
                    retained = all((root/"evaluation/predictions"/case/group/m["name"]/f"chunk{start}.json").exists()
                                   for m in methods(context,case))
                    if max_chunks is not None and done>=max_chunks and not retained:
                        return {"complete":False,"chunks_processed":done,"root":str(root)}
                    x,identifiers = initial[start:start+context["plan"].chunk_paths],ids[start:start+context["plan"].chunk_paths]
                    bases = {step:_baseline_chunk(context,case,group,x,identifiers,start,step)
                             for step in context["plan"].raw_steps}
                    for method in methods(context,case):
                        chunks[method["name"]].append(_predicted_chunk(context,case,group,x,identifiers,start,
                            method,models[method["name"]],bases[method["baseline_step"]]))
                    done += not retained
                _finish_group(context,case,group,reference,ids,chunks)
        return _refresh_index(context)


def _finish_group(context,case,group,reference,ids,chunks):
    cfg = collocation.reference_configuration(case,context["collocation_study"])
    ref = reference["outcomes"]
    reference_counts = _reference_counts(case,group,len(ref))
    role = group.split("__")[1]
    pool = next((p for p in context["packet"]["pools"] if (p["case"],p["group"])==(case,group)),None)
    initial_counts = bootstrap_counts(len(ids),pool["bootstrap_seed"],128) if pool else None
    records = []
    for method in methods(context,case):
        pieces,paths = [],[]
        for chunk in chunks[method["name"]]:
            artifact(chunk["arrays"]["path"],expected=chunk["arrays"]["sha256"])
            with np.load(chunk["arrays"]["path"]) as a:
                pieces.append({k:a[k] for k in a.files})
            paths.append({"path":str(Path(chunk["arrays"]["path"]).with_suffix(".json")),
                          "sha256":sha(Path(chunk["arrays"]["path"]).with_suffix(".json"))})
        values = {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        if not np.array_equal(values["family_ids"],ids):
            raise ValueError("Every conditional/integration family must remain in its declared order")
        identity = {"joint_lock_sha256":context["joint_lock_sha256"],"case":case,"group":group,"method":method,
                    "reference_sha256":reference["manifest"]["reference_sha256"]}
        path = context["root"]/"evaluation/accuracy"/case/group/(method["name"]+".json")
        previous = _unit(path,identity)
        if previous is not None:
            records.append({"path":str(path),"sha256":sha(path)})
            continue
        available = bool(values["valid"].all())
        arrays = {"family_ids":ids,"valid":values["valid"],"errors":values["errors"]}
        metrics = None
        if available:
            if role=="ensemble":
                mean,cov = integrate_iid(values["mean"],values["scaled_covariance"],values["valid"],n_particles=cfg.n_particles)
                bm,bc = integrate_iid(values["mean"],values["scaled_covariance"],values["valid"],n_particles=cfg.n_particles,counts=initial_counts)
            else:
                mean,cov = values["mean"][0],values["scaled_covariance"][0]
                bm,bc = np.broadcast_to(mean,(128,*mean.shape)),np.broadcast_to(cov,(128,*cov.shape))
            arrays.update(mean=mean,scaled_covariance=cov,bootstrap_mean=bm,bootstrap_scaled_covariance=bc)
            metrics = moment_comparison(ref,predicted_mean=mean,predicted_scaled_covariance=cov,
                bootstrap_predicted_mean=bm,bootstrap_predicted_scaled_covariance=bc,
                n_particles=cfg.n_particles,counts=reference_counts)
        _save_unit(path,identity,arrays,available=available,metrics=metrics,method=method,
            all_families_retained=True,failed_family_ids=ids[~values["valid"]].astype(str).tolist(),
            predictions=paths,initial_pool_independent_of_reference=role=="ensemble",
            bootstrap_counts={"reference_sha256":array_sha(reference_counts),
                              "initial_sha256":None if initial_counts is None else array_sha(initial_counts)})
        records.append({"path":str(path),"sha256":sha(path)})
    write(context["root"]/"evaluation/accuracy"/case/group/"group.json",{
        "complete":True,"case":case,"group":group,"joint_lock_sha256":context["joint_lock_sha256"],
        "records":records,"all_methods_and_optimizer_seeds_retained":True,
        "all_stopped_native_reference_families_retained":True,"initials_shared_with_reference":False})


def _refresh_index(context):
    expected = {(r["case"],r["group"]) for r in context["packet"]["inventory"]}
    completed = []
    for case,group in sorted(expected):
        path = context["root"]/"evaluation/accuracy"/case/group/"group.json"
        if path.exists():
            completed.append({"case":case,"group":group,"path":str(path),"sha256":sha(path)})
    packet = {"kind":KIND,"joint_lock_sha256":context["joint_lock_sha256"],
        "complete":len(completed)==len(expected),"groups":completed,
        "expected_groups":len(expected),"expected_records":len(context["packet"]["inventory"])}
    write(context["root"]/"evaluation/accuracy_index.json",packet)
    return packet


def _read_accuracy(context,*,require_complete=True):
    inputs,records,groups = {},[],[]
    path = context["root"]/"evaluation/accuracy_index.json"
    index = read(artifact(path,inputs))
    expected = {(row["case"],row["group"],row["method"]) for row in context["packet"]["inventory"]}
    expected_groups = {(case,group) for case,group,_ in expected}
    actual_groups = [(r["case"],r["group"]) for r in index["groups"]]
    if (index.get("kind")!=KIND or index.get("joint_lock_sha256")!=context["joint_lock_sha256"]
            or len(actual_groups)!=len(set(actual_groups)) or set(actual_groups)-expected_groups
            or index["expected_groups"]!=len(expected_groups) or index["expected_records"]!=len(expected)
            or (require_complete and (not index.get("complete") or set(actual_groups)!=expected_groups))):
        raise ValueError("Every declared R4 group and method must be inventoried")
    actual = []
    for group_record in index["groups"]:
        packet = read(artifact(group_record["path"],inputs,group_record["sha256"]))
        case,group = group_record["case"],group_record["group"]
        if (not packet.get("complete") or packet.get("joint_lock_sha256")!=context["joint_lock_sha256"]
                or (packet["case"],packet["group"])!=(case,group)):
            raise ValueError("Group identity changed")
        names = []
        for item in packet["records"]:
            record = read(artifact(item["path"],inputs,item["sha256"]))
            method = record["method"]
            expected_method = next((m for m in methods(context,case) if m["name"]==method["name"]),None)
            if (not record.get("complete") or expected_method!=method
                    or record["identity"].get("joint_lock_sha256")!=context["joint_lock_sha256"]
                    or (record["identity"]["case"],record["identity"]["group"])!=(case,group)
                    or record.get("all_families_retained") is not True):
                raise ValueError("An R4 seed/control record changed or disappeared")
            cfg = collocation.reference_configuration(case,context["collocation_study"])
            families = cfg.paths_per_group if "__ensemble__" in group else 1
            pool = next((p for p in context["packet"]["pools"] if (p["case"],p["group"])==(case,group)),None)
            expected_ids = (_planned_ids(pool,context["plan"]) if pool else
                            np.array([stable_hash([KIND,case,group,"conditional-map"])],dtype="S64"))
            with np.load(artifact(record["arrays"]["path"],inputs,record["arrays"]["sha256"])) as a:
                if (a["family_ids"].shape!=(families,) or a["valid"].shape!=(families,)
                        or not np.array_equal(a["family_ids"],expected_ids)):
                    raise ValueError("Complete unique family allocation required,including failures")
                if bool(a["valid"].all())!=record["available"]:
                    raise ValueError("Unavailable groups cannot be reported as successful predictions")
                if a["family_ids"][~a["valid"]].astype(str).tolist()!=record["failed_family_ids"]:
                    raise ValueError("Exact failed-family IDs required")
                if record["available"]:
                    values = record["metrics"]["records"]
                    if len(values)!=len(cfg.query_times) or {v["time_index"] for v in values}!=set(range(len(cfg.query_times))):
                        raise ValueError("Every physical query must remain")
                    if (a["mean"].shape!=(len(cfg.query_times),4) or a["scaled_covariance"].shape!=(len(cfg.query_times),4,4)
                            or a["bootstrap_mean"].shape!=(128,len(cfg.query_times),4)
                            or a["bootstrap_scaled_covariance"].shape!=(128,len(cfg.query_times),4,4)):
                        raise ValueError("Full same-time4x4moments required")
                elif record["metrics"] is not None or not record["failed_family_ids"]:
                    raise ValueError("Failures require explicit IDs and unavailable metrics")
            starts,prediction_ids = [],[]
            for chunk in record["predictions"]:
                evidence = read(artifact(chunk["path"],inputs,chunk["sha256"]))
                identity = evidence["identity"]
                if (not evidence.get("complete") or identity.get("joint_lock_sha256")!=context["joint_lock_sha256"]
                        or (identity["case"],identity["group"],identity["method"])!=(case,group,method)):
                    raise ValueError("Prediction chunk source/method allocation changed")
                start = identity["start"]
                starts.append(start)
                with np.load(artifact(evidence["arrays"]["path"],inputs,evidence["arrays"]["sha256"])) as arrays:
                    values = arrays["family_ids"]
                    if not np.array_equal(values,expected_ids[start:start+context["plan"].chunk_paths]):
                        raise ValueError("Prediction chunks must preserve literal family order")
                    prediction_ids.extend(values.tolist())
                base_path = context["root"]/"evaluation/baselines"/case/group/f"dt{round(-np.log2(method['baseline_step']))}_{start}.json"
                baseline = read(artifact(base_path,inputs))
                artifact(baseline["arrays"]["path"],inputs,identity["baseline_sha256"])
            if (starts!=list(range(0,families,context["plan"].chunk_paths))
                    or prediction_ids!=expected_ids.tolist()):
                raise ValueError("Every ordered prediction chunk is required,without missing families")
            names.append(method["name"])
            actual.append((case,group,method["name"]))
            records.append(record)
        if len(names)!=6 or set(names)!={m["name"] for m in methods(context,case)}:
            raise ValueError("All three candidates and all three controls must remain")
        groups.append(packet)
    if len(actual)!=len(set(actual)) or (require_complete and set(actual)!=expected):
        raise ValueError("Full R4 method/group allocation required")
    return groups,records,inputs


def _arrays(record):
    with np.load(artifact(record["arrays"]["path"],expected=record["arrays"]["sha256"])) as a:
        return {k:a[k] for k in a.files if k in ("mean","scaled_covariance","bootstrap_mean","bootstrap_scaled_covariance")}


def contrasts(root):
    """Signed common-reference comparisons,including both integration pools."""
    context = verify_joint_lock(root)
    _,records,inputs = _read_accuracy(context)
    old = comparison.primary_records(context["collocation_root"],require_timing=False)
    inputs.update(old["inputs"])
    by_slot = {(r["identity"]["case"],r["identity"]["group"],r["method"]["name"]):r for r in records}
    old_slots = {(r["identity"]["regime"],r["identity"]["group"],r["method"]["name"]):r for r in old["records"]}
    rows = []
    for case in context["cases"]:
        cfg = collocation.reference_configuration(case,context["collocation_study"])
        for group in cfg.group_keys:
            data = collocation.load_reference(context["collocation_root"],case,group)
            ref = data["outcomes"]
            counts = _reference_counts(case,group,len(ref))
            for candidate in (m for m in methods(context,case) if m["kind"]=="trained"):
                first = by_slot[(case,group,candidate["name"])]
                controls = [(name,by_slot[(case,group,name)],"R4-control") for name in CONTROL_NAMES]
                controls += [(r["method"]["name"],r,"collocation-matched-seed") for (c,g,_),r in old_slots.items()
                             if (c,g)==(case,group) and r["method"].get("seed")==candidate["seed"]
                             and r["method"]["family"]=="repaired"]
                controls += [(name,old_slots[(case,group,name)],"analytic-IID-Gaussian-or-native-control")
                             for name in ("LGA_dt7","LGA_dt10","native-MC-dt10")]
                if len([v for v in controls if v[2]=="collocation-matched-seed"])!=4:
                    raise ValueError("Every matched-seed collocation variant is required")
                for name,second,kind in controls:
                    available = first["available"] and second.get("available",True)
                    contrast = predictor_advantage(ref,_arrays(first),_arrays(second),counts,cfg.n_particles) if available else None
                    rows.append({"case":case,"group":group,"candidate":candidate["name"],"seed":candidate["seed"],
                        "control":name,"control_family":kind,"available":available,"contrast":contrast,
                        "same_initials_assumed":False,"pointwise_not_simultaneous":True})
    identity = {"joint_lock_sha256":context["joint_lock_sha256"],"accuracy_index_sha256":sha(context["root"]/"evaluation/accuracy_index.json"),
        "collocation_accuracy_index_sha256":sha(context["collocation_root"]/"comparison/accuracy_index.json"),
        "collocation_independent_index_sha256":sha(context["collocation_root"]/"deployment/accuracy_index.json")}
    return immutable(context["root"]/"evaluation/contrasts.json",{
        "kind":KIND,"complete":True,"identity":identity,"records":rows,"inputs":inputs,
        "all_fixed_optimizer_seeds_retained":True,"negative_or_unresolved_conclusions_preserved":True})


def _timing_slots(context):
    slots = []
    for case in context["cases"]:
        cfg = collocation.reference_configuration(case,context["collocation_study"])
        for profile in cfg.cases:
            for method in methods(context,case):
                for role in ("conditional","ensemble"):
                    for phase in ("cold","warm"):
                        slots.append({"slot":f"{case}/{profile}/{method['name']}/{role}/{phase}",
                            "case":case,"profile":profile,"method":method,"role":role,"phase":phase,
                            "families":1 if role=="conditional" else cfg.paths_per_group,
                            "batch_size":context["plan"].baseline_batch_size})
    return slots


def _timing_ids(context,job):
    return np.array([stable_hash([KIND,context["plan"].seed,"timing-only",job["case"],
        job["profile"],job["role"],i]) for i in range(job["families"])],dtype="S64")


def run_uninterrupted_workload(context,job,model):
    """Time a complete prediction; metadata and persistence stay outside."""
    started = time.perf_counter()
    cfg = collocation.reference_configuration(job["case"],context["collocation_study"])
    bank = collocation.payoff_bank(job["case"],context["collocation_study"])
    probabilities = collocation.profile_probabilities(cfg,job["profile"])[0]
    rng = seeded_rng(context["plan"].seed,KIND,"timing-only",job["case"],job["profile"],job["role"])
    initial = rng.multinomial(cfg.n_particles,probabilities,size=job["families"])/cfg.n_particles
    generated = time.perf_counter()
    baseline = conditional_baseline(initial,cfg.context(job["profile"]),cfg.spec,cfg.query_times,
        job["method"]["baseline_step"],bank,batch_size=context["plan"].baseline_batch_size)
    evolved = time.perf_counter()
    value = corrected_predictions(model,job["method"],initial,cfg.query_times,baseline,
                                  tolerance=context["plan"].psd_tolerance)
    corrected = time.perf_counter()
    available = bool(value["valid"].all())
    if available and job["role"]=="ensemble":
        mean,covariance = integrate_iid(value["mean"],value["scaled_covariance"],value["valid"],
                                       n_particles=cfg.n_particles)
    elif available:
        mean,covariance = value["mean"][0],value["scaled_covariance"][0]
    else:
        mean,covariance = np.empty(0),np.empty(0)
    arrays = {"initial":initial,"mean":mean,"scaled_covariance":covariance,
        "conditional_mean":value["mean"],"conditional_scaled_covariance":value["scaled_covariance"],
        "valid":value["valid"],"errors":np.asarray(value["errors"],dtype=str)}
    finished = time.perf_counter()
    parts = {"initialization_seconds":generated-started,"conditional_LNA_seconds":evolved-generated,
             "head_seconds":corrected-evolved,"aggregation_seconds":finished-corrected}
    return arrays,parts,finished-started,available


def benchmark(root,*,final_timing=False,max_jobs=None,max_chunks=None):
    """One uninterrupted cold/warm call; resume only between completed jobs.

    Every call generates its full histogram pool and executes every conditional
    ODE,head and total-covariance reduction without metadata I/O. A restarted
    warm job pays a complete uncached warm-up outside its measured observation.
    Interrupted attempts stay charged and rerun identical timing inputs.
    """
    if not final_timing:
        raise ValueError("Timing requires explicit attestation that ALL heavy work has exited")
    if max_chunks is not None:
        raise ValueError("Uninterrupted primary timing resumes between jobs; use max_jobs,not max_chunks")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs<0):
        raise ValueError("Nonnegative complete-job budget required")
    context = verify_joint_lock(root)
    _read_accuracy(context)
    comparison.primary_records(context["collocation_root"],require_timing=False)
    root,other = context["root"],context["collocation_root"]
    with ExitStack() as leases:
        paths = [root/"evaluation/execution/.accuracy.lock",root/"evaluation/execution/.timing.lock",
            root/"execution/.prepare.lock",root/"execution/.data.lock",root/"payoffs/execution/.fit-lock.lock",
            other/"comparison/execution/.accuracy.lock",other/"comparison/execution/.timing.lock",
            other/"payoffs/execution/.prepare.lock",other/"payoffs/execution/.fit-lock.lock"]
        paths += [directory/"payoffs/execution"/f".seed{s}.lock" for directory in (root,other) for s in (0,1,2)]
        paths += [other/"reference"/case/"execution/.generation.lock" for case in context["cases"]]
        for path in paths:
            leases.enter_context(collocation._exclusive(path))
        completed_jobs = 0
        live_key,live_model = None,None
        for job in _timing_slots(context):
            path = root/"evaluation/timing/jobs"/(job["slot"]+"/record.json")
            identity = {"joint_lock_sha256":context["joint_lock_sha256"],"job":job}
            if _unit(path,identity) is not None:
                _read_timing_job(context,path,identity)
                continue
            if max_jobs is not None and completed_jobs>=max_jobs:
                break
            path.parent.mkdir(parents=True,exist_ok=True)
            attempts_path = path.parent/"attempts.json"
            attempts = read(attempts_path) if attempts_path.exists() else []
            attempt = {"started_unix":time.time(),"finished":False,"successful":False,
                "restoration_seconds":0.,"warm_preparation_seconds":0.,"workload_seconds":0.,
                "accuracy_baseline_cache_read":False}
            attempts.append(attempt)
            write(attempts_path,attempts)
            attempt_started,attempt_cpu = time.perf_counter(),time.process_time()
            stage,stage_started = "restoration",time.perf_counter()
            try:
                key = (job["case"],job["profile"],job["method"]["name"],job["role"])
                reused = job["phase"]=="warm" and live_key==key
                model = live_model if reused else _model(context,job["case"],job["method"])
                attempt["restoration_seconds"] = 0. if reused else time.perf_counter()-stage_started
                attempt["reused_live_model_after_complete_cold_workload"] = reused
                if job["phase"]=="warm" and not reused:
                    stage,stage_started = "warmup",time.perf_counter()
                    run_uninterrupted_workload(context,job,model)
                    attempt["warm_preparation_seconds"] = time.perf_counter()-stage_started
                    attempt["warm_preparation_scope"] = "complete uncached workload after resume"
                    write(attempts_path,attempts)
                stage,stage_started = "workload",time.perf_counter()
                arrays,parts,seconds,available = run_uninterrupted_workload(context,job,model)
                attempt["workload_seconds"] = seconds
                attempt["successful"] = True
            except BaseException:
                duration = time.perf_counter()-stage_started
                field = {"restoration":"restoration_seconds","warmup":"warm_preparation_seconds",
                         "workload":"workload_seconds"}[stage]
                attempt[field] = duration
                raise
            finally:
                attempt.update(finished=True,elapsed_seconds=time.perf_counter()-attempt_started,
                               cpu_seconds=time.process_time()-attempt_cpu,finished_unix=time.time())
                write(attempts_path,attempts)
            arrays["family_ids"] = _timing_ids(context,job)
            known = [a for a in attempts if a.get("finished")]
            _save_unit(path,identity,arrays,available=available,components=parts,
                component_sum_seconds=sum(parts.values()),workload_seconds=seconds,
                timing_mode="uninterrupted-full-uncached-workload-v1",
                cold_end_to_end_seconds=attempt["restoration_seconds"]+seconds if job["phase"]=="cold" else None,
                restoration_and_resume_attempts=attempts,paid_attempt_seconds=sum(a["elapsed_seconds"] for a in known),
                attempt_cost_complete=len(known)==len(attempts),
                paid_attempt_scope="known attempt elapsed totals include successful observation once;never add workload again",
                all_families_retained=True,full_uncached_LNA_per_histogram=True,accuracy_cache_used=False,
                timing_scope="uninterrupted histogram generation,all conditional ODEs,head,total covariance;restoration separate",
                observation_scope="one cold and one warm observation,not a stable latency distribution",
                timing_initials_shared_across_methods_and_phases=True)
            completed_jobs += 1
            live_key,live_model = key,model
        return _timing_index(context)


def _read_timing_job(context,path,identity,inputs=None):
    value = _unit(path,identity)
    if value is None:
        raise ValueError("Every timing workload must complete")
    artifact(path,inputs)
    job = identity["job"]
    parts = value.get("components",{})
    seconds = value.get("workload_seconds",float("nan"))
    if (value.get("timing_mode")!="uninterrupted-full-uncached-workload-v1"
            or value.get("all_families_retained") is not True
            or value.get("full_uncached_LNA_per_histogram") is not True or value.get("accuracy_cache_used") is not False
            or set(parts)!={"initialization_seconds","conditional_LNA_seconds","head_seconds","aggregation_seconds"}
            or not math.isfinite(seconds) or seconds<=0 or any(not math.isfinite(v) or v<0 for v in parts.values())
            or not np.isclose(sum(parts.values()),value["component_sum_seconds"],rtol=1e-12)
            or value["component_sum_seconds"]>seconds+1e-6):
        raise ValueError("Primary timing requires actual uninterrupted full uncached workloads")
    attempts_path = Path(path).parent/"attempts.json"
    attempts = read(artifact(attempts_path,inputs))
    known = [a for a in attempts if a.get("finished")]
    if (attempts!=value["restoration_and_resume_attempts"] or not attempts or not attempts[-1].get("successful")
            or any(not math.isfinite(a[k]) or a[k]<0 for a in known for k in
                   ("elapsed_seconds","restoration_seconds","warm_preparation_seconds","workload_seconds","cpu_seconds"))
            or value["attempt_cost_complete"]!=(len(known)==len(attempts))
            or not np.isclose(sum(a["elapsed_seconds"] for a in known),value["paid_attempt_seconds"],rtol=1e-12)
            or attempts[-1]["workload_seconds"]!=seconds):
        raise ValueError("Complete paid-attempt timing accounting required")
    if job["phase"]=="cold" and not np.isclose(value["cold_end_to_end_seconds"],
            attempts[-1]["restoration_seconds"]+seconds,rtol=1e-12):
        raise ValueError("Cold restoration and workload must be charged once")
    with np.load(artifact(value["arrays"]["path"],inputs,value["arrays"]["sha256"])) as a:
        expected_ids = _timing_ids(context,job)
        if (not np.array_equal(a["family_ids"],expected_ids) or a["valid"].shape!=(job["families"],)
                or a["initial"].shape!=(job["families"],collocation.reference_configuration(
                    job["case"],context["collocation_study"]).n_cells)
                or bool(a["valid"].all())!=value["available"]):
            raise ValueError("Every literal timing histogram must actually be evolved and retained")
    return value


def _timing_index(context):
    expected = _timing_slots(context)
    jobs = []
    for job in expected:
        path = context["root"]/"evaluation/timing/jobs"/(job["slot"]+"/record.json")
        if path.exists():
            _read_timing_job(context,path,{"joint_lock_sha256":context["joint_lock_sha256"],"job":job})
            jobs.append({"slot":job["slot"],"path":str(path),"sha256":sha(path)})
    value = {"kind":KIND,"joint_lock_sha256":context["joint_lock_sha256"],"complete":len(jobs)==len(expected),
        "expected_jobs":len(expected),"jobs":jobs,"environment":{"platform":platform.platform(),
            "numpy":np.__version__,"torch":str(torch.__version__),"cpu_threads":1},
        "external_competing_heavy_work_finished_attested":True}
    write(context["root"]/"evaluation/timing_index.json",value)
    return value


def load_complete(root,*,require_timing=True,require_contrasts=True):
    context = verify_joint_lock(root)
    groups,records,inputs = _read_accuracy(context)
    baseline = comparison.primary_records(context["collocation_root"],require_timing=require_timing)
    inputs.update(baseline["inputs"])
    for name in ("evaluation/preregistration.json","evaluation/joint_lock.json","evaluation/planned_initial_families.npz",
                 "evaluation/identifier_audit.json","payoffs/fit_lock.json","data/data_lock.json","plan.json"):
        artifact(context["root"]/name,inputs)
    for path,digest in context["packet"]["helpers"].items():
        artifact(path,inputs,digest)
    for model in context["candidate_lock"]["models"]:
        for kind in ("checkpoint","metadata"):
            if kind+"_path" in model:
                artifact(model[kind+"_path"],inputs,model[kind+"_sha256"])
    for pool in context["packet"]["pools"]:
        path = context["root"]/"evaluation/initials"/pool["case"]/(pool["profile"]+".json")
        value = read(artifact(path,inputs))
        artifact(value["arrays"]["path"],inputs,value["arrays"]["sha256"])
    audit = context["packet"]["candidate_native_label_audit"]
    artifact(audit["path"],inputs,audit["sha256"])
    commands = invocation_receipts(context["root"],include_verification=False)
    for command in commands:
        artifact(command["receipt_path"],inputs)
    for case in context["cases"]:
        path = artifact(context["collocation_root"]/"reference"/case/"manifest.json",inputs)
        reference = read(path)
        artifact(reference["reference_path"],inputs,reference["reference_sha256"])
    timing = None
    if require_timing:
        timing = read(artifact(context["root"]/"evaluation/timing_index.json",inputs))
        expected = {job["slot"]:job for job in _timing_slots(context)}
        actual = [v["slot"] for v in timing["jobs"]]
        if (not timing.get("complete") or timing["joint_lock_sha256"]!=context["joint_lock_sha256"]
                or timing["expected_jobs"]!=len(expected) or len(actual)!=len(expected) or set(actual)!=set(expected)
                or timing.get("external_competing_heavy_work_finished_attested") is not True
                or timing["environment"].get("cpu_threads")!=1):
            raise ValueError("Every literal R4 cold/warm workload required after all heavy work")
        for entry in timing["jobs"]:
            _read_timing_job(context,artifact(entry["path"],inputs,entry["sha256"]),
                {"joint_lock_sha256":context["joint_lock_sha256"],"job":expected[entry["slot"]]},inputs)
    contrast = None
    if require_contrasts:
        contrast = read(artifact(context["root"]/"evaluation/contrasts.json",inputs))
        expected_contrasts = set()
        for case in context["cases"]:
            for group in collocation.reference_configuration(case,context["collocation_study"]).group_keys:
                for candidate in (m for m in methods(context,case) if m["kind"]=="trained"):
                    controls = [*CONTROL_NAMES,"LGA_dt7","LGA_dt10","native-MC-dt10"]
                    controls += [m["name"] for m in context["collocation_lock"]["models"]
                                 if m["case"]==case and m["seed"]==candidate["seed"]]
                    expected_contrasts.update((case,group,candidate["name"],name) for name in controls)
        actual = [(r["case"],r["group"],r["candidate"],r["control"]) for r in contrast["records"]]
        if (not contrast.get("complete") or contrast["identity"]["joint_lock_sha256"]!=context["joint_lock_sha256"]
                or len(actual)!=len(expected_contrasts) or set(actual)!=expected_contrasts):
            raise ValueError("All candidate/control/matched-seed contrasts required")
        for path,digest in contrast["inputs"].items():
            artifact(path,inputs,digest)
    return {**context,"groups":groups,"records":records,"baseline":baseline,"timing":timing,"contrasts":contrast,
        "invocations":commands,
        "inputs":inputs,"allocation":{"candidate_models":len(context["candidate_lock"]["models"]),
            "candidate_primary_records":len(records),"baseline_primary_records":len(baseline["records"]),
            "unavailable_candidate_records":sum(not r["available"] for r in records),
            "timing_jobs":0 if timing is None else len(timing["jobs"])}}


def primary_records(root,*,require_timing=True,require_contrasts=True):
    return load_complete(root,require_timing=require_timing,require_contrasts=require_contrasts)


def invocation_receipts(root,*,include_verification=True):
    """Measured complete process invocations;these totals contain nested work."""
    root = Path(root).resolve()
    rows = []
    for path in sorted((root/"evaluation/execution").glob("cli_*.json")):
        if not include_verification and path.name.startswith("cli_verify_"):
            continue
        row = read(path)
        if (row.get("kind")!=INVOCATION_KIND or row.get("schema_version")!=1
                or row.get("stage") not in CLI_STAGES or row.get("root")!=str(root)
                or not path.name.startswith("cli_"+str(row.get("stage"))+"_")
                or row.get("source_hash")!=source_hash() or row.get("script_sha256")!=sha(__file__)
                or row.get("finished") is not True or row.get("status") not in ("success","checkpointed","error")
                or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<0
                       for k in ("wall_seconds","cpu_seconds","process_cpu_seconds_at_finish"))
                or row["finished_unix"]<row["started_unix"]):
            raise ValueError("Whole-process evaluator invocation receipt changed or is incomplete")
        rows.append({**row,"receipt_path":str(path),"receipt_sha256":sha(path)})
    return rows


def _dispatch(args,parser):
    if args.action=="prepare":
        result = prepare(args.root)
        return {"prepared":True,"records":len(result["inventory"]),"pools":len(result["pools"])}
    if args.action=="freeze":
        return freeze(args.root)
    if args.action=="reference":
        if args.case and len(args.case)>1:
            parser.error("reference accepts at most one --case")
        value = generate_shared_references(args.root,None if not args.case else args.case[0],max_chunks=args.max_chunks)
        return {case:{key:packet[key] for key in ("status","committed_chunks") if key in packet}
                for case,packet in value.items()}
    if args.action=="evaluate":
        return evaluate(args.root,cases=args.case,max_chunks=args.max_chunks)
    if args.action=="contrasts":
        value = contrasts(args.root)
        return {"complete":value["complete"],"contrasts":len(value["records"])}
    if args.action=="benchmark":
        return benchmark(args.root,final_timing=args.final_timing,max_jobs=args.max_jobs,max_chunks=args.max_chunks)
    return load_complete(args.root)["allocation"]


def main(argv=None):
    actual_process = __name__=="__main__" and argv is None
    wall_start = _PROCESS_WALL_START if actual_process else time.monotonic()
    cpu_start = _PROCESS_CPU_START if actual_process else time.process_time()
    unix_start = _PROCESS_UNIX_START if actual_process else time.time()
    invocation_ns = time.time_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=CLI_STAGES)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--case",action="append")
    parser.add_argument("--max-chunks",type=int)
    parser.add_argument("--max-jobs",type=int)
    parser.add_argument("--final-timing",action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    status,exit_code,error = "error",1,None
    try:
        result = _dispatch(args,parser)
        checkpointed = result.get("complete") is False
        if args.action=="reference":
            checkpointed = any(p.get("status")!="complete" for p in result.values())
        status,exit_code = ("checkpointed",75) if checkpointed else ("success",0)
        print(json.dumps(result,indent=2))
    except BaseException as exception:
        error = type(exception).__name__+": "+str(exception)
        exit_code = 130 if isinstance(exception,KeyboardInterrupt) else (
            exception.code if isinstance(exception,SystemExit) and isinstance(exception.code,int) else 1)
        raise
    finally:
        package_digest,script_digest = source_hash(),sha(__file__)
        record = {"kind":INVOCATION_KIND,"schema_version":1,"finished":True,"stage":args.action,
            "pid":os.getpid(),"invocation_ns":invocation_ns,"root":str(root),
            "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "source_hash":package_digest,"script_sha256":script_digest,
            "started_unix":unix_start,"finished_unix":time.time(),
            "wall_seconds":time.monotonic()-wall_start,"cpu_seconds":time.process_time()-cpu_start,
            "process_cpu_seconds_at_finish":time.process_time(),"whole_process":actual_process,
            "status":status,"exit_code":exit_code,"error":error,
            "scope":"actual CLI:from module entry after stdlib,before NumPy/Torch/project imports;API main call:entry only;final receipt persistence excluded",
            "nonadditive_contains_nested_work":True,"nested_prediction_or_cache_costs_added_again":False}
        immutable(root/"evaluation/execution"/f"cli_{args.action}_{os.getpid()}_{invocation_ns}.json",record)
    return exit_code


if __name__=="__main__":
    raise SystemExit(main())
