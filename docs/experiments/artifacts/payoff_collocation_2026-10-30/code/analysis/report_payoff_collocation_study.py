"""Complete, all-variant matched-pool proposal/loss accuracy and cost report.

Postprocessing never selects a model, optimizer seed, regime or test allocation.
Independent initial-law integration is primary; paired-initial results remain
diagnostics. Pointwise contrasts and arithmetic cost curves are not certificates
of a grid-wide accuracy or neural advantage.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import run_payoff_collocation_study as runner  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "complete-crossed-natural-collocation-proposal-loss-report-v1"
VARIANTS = tuple(proposal+"__"+loss for proposal in runner.PROPOSALS for loss in runner.LOSSES)
BASELINE = "uniform__pseudo_huber"
NEURAL = ("repaired","untrained-repaired")
_base = isolated_module("report_payoff_regimes.py","analysis._collocation_report_impl")
bind_study(_base,__file__,runner=runner)


def _evaluator():
    from analysis import evaluate_payoff_collocation_study

    return evaluate_payoff_collocation_study


def read(path):
    return _base.read(path)


def sha(path):
    return _base.sha(path)


def _artifact(path,inputs,expected=None):
    return _base._artifact(path,inputs,expected)


def _variant(model):
    return model.get("variant",model.get("proposal","")+"__"+model.get("loss",""))


def matched_inventory(models,slots):
    """A Cartesian allocation cannot drop a bad seed or crossed variant."""
    fields = ("case","seed","proposal","loss","variant","name")
    expected = {tuple(row[k] for k in fields) for row in slots}
    actual = [tuple(row[k] for k in fields) for row in models]
    if len(actual)!=len(expected) or set(actual)!=expected or len(set(actual))!=len(actual):
        raise ValueError("Every declared case/optimizer-seed/crossed-variant model is required")
    groups = defaultdict(list)
    for row in models:
        groups[(row["case"],row["seed"])].append(row)
    for rows in groups.values():
        if len(rows)!=4 or {_variant(row) for row in rows}!=set(VARIANTS):
            raise ValueError("All four crossed variants must remain")
        if (len({row["initial_state_sha256"] for row in rows})!=1
                or len({json.dumps(row["data_fingerprints"],sort_keys=True) for row in rows})!=1
                or len({row["loss_scaling_sha256"] for row in rows})!=1):
            raise ValueError("Matched variants must share literal pools, initialization and scales")
    return groups


def load_complete(root):
    """The evaluator checks all physical records; this adds report allocations."""
    evidence = _evaluator().primary_records(Path(root).resolve())
    study = evidence["study"]
    matched_inventory(evidence["lock"]["models"],runner.slots(study))
    validate_evidence(evidence)
    return evidence


def _names(evidence,case):
    return {m["name"] for m in evidence["lock"]["models"] if m["case"]==case}|{
        case+"_untrained","LGA_dt7","LGA_dt10","native-MC-dt10"}


def _prediction_queries(record,queries):
    values = record.get("metrics",{}).get("records",[])
    if (not record.get("complete") or len(values)!=len(queries)
            or {v["time_index"] for v in values}!=set(range(len(queries)))):
        raise ValueError("Every physical prediction query is required exactly once")
    fields = ("max_mean_error_in_reference_std","max_normalized_covariance_error",
              "mean_sampling95_simultaneous_radius","covariance_sampling95_simultaneous_radius",
              "mean_error_plus_sampling95","covariance_error_plus_sampling95")
    if any(not math.isfinite(item[k]) or item[k]<0 for item in values for k in fields):
        raise ValueError("Finite nonnegative physical prediction metrics required")


def validate_evidence(evidence):
    """Reject partial methods, seeds, groups, independent pools or timings."""
    study,plan = evidence["study"],evidence["plan"]
    models = evidence["lock"]["models"]
    matched_inventory(models,runner.slots(study))
    cases = tuple(runner.case_names(study))
    if tuple(plan["cases"])!=cases or tuple(study.optimizer_seeds)!=(0,1,2):
        raise ValueError("Every declared regime and all three optimizer seeds are required")
    expected_groups = {(case,key) for case in cases for key in runner.reference_configuration(case,study).group_keys}
    groups = [(r["regime"],r["group"]) for r in evidence["groups"]]
    if len(groups)!=len(expected_groups) or set(groups)!=expected_groups:
        raise ValueError("Every declared physical comparison group is required")
    expected = {(case,key,name) for case,key in expected_groups for name in _names(evidence,case)}
    actual = [(r["identity"]["regime"],r["identity"]["group"],r["method"]["name"]) for r in evidence["records"]]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError("Complete variant/seed/classical accuracy evaluation inventory required")
    slot_lookup = {(m["case"],m["name"]):m for m in models}
    for record in evidence["records"]:
        identity,method = record["identity"],record["method"]
        case,key,name = identity["regime"],identity["group"],method["name"]
        if not record.get("complete") or identity["name"]!=name:
            raise ValueError("Incomplete method prediction")
        _prediction_queries(record,runner.reference_configuration(case,study).query_times)
        if (case,name) in slot_lookup:
            slot = slot_lookup[(case,name)]
            if method["family"]!="repaired" or any(method.get(k)!=slot[k] for k in ("seed","proposal","loss","variant")):
                raise ValueError("An accuracy method's crossed variant identity changed")
        if "__ensemble__" in key and method["family"] in NEURAL:
            if (record.get("initial_pool_independent_of_reference") is not True
                    or record.get("reference_and_initial_bootstrap_independent") is not True):
                raise ValueError("Primary IID-law neural integration must be independent of reference")
    paired_expected = {(case,key,name) for case,key in expected_groups if "__ensemble__" in key
                       for name in _names(evidence,case)-{"LGA_dt7","LGA_dt10","native-MC-dt10"}}
    paired = [(r["identity"]["regime"],r["identity"]["group"],r["method"]["name"])
              for r in evidence["paired_initial_records"]]
    if (len(paired)!=len(paired_expected) or set(paired)!=paired_expected
            or evidence["independent_initial_records"]!=len(paired_expected)
            or evidence["original_accuracy_records"]!=len(expected)):
        raise ValueError("Every paired-initial diagnostic and independent replacement must remain")
    for record in evidence["paired_initial_records"]:
        _prediction_queries(record,runner.reference_configuration(record["identity"]["regime"],study).query_times)
    for group in evidence["groups"]:
        case = group["regime"]
        fitted = {m["name"] for m in models if m["case"]==case}
        candidate = {m["name"] for m in models if m["case"]==case and _variant(m)!=BASELINE}
        if (set(group.get("NN_versus_fine_LGA",{}))!=fitted
                or set(group.get("NN_versus_uniform_huber",{}))!=candidate):
            raise ValueError("All Gaussian and matched-seed uniform-Huber contrasts are required")
        cfg = runner.reference_configuration(case,study)
        count = len(cfg.query_times)
        packets = [*group["NN_versus_fine_LGA"].values(),*group["NN_versus_uniform_huber"].values(),
                   *group["reference_refinement"].values()]
        if set(group["reference_refinement"])!={f"level{i}_versus_finest" for i in range(len(cfg.reference_factors)-1)}:
            raise ValueError("Every declared coupled reference-refinement level is required")
        if any(len(p["records"])!=count or
                {r["time_index"] for r in p["records"]}!=set(range(count)) for p in packets):
            raise ValueError("Every contrast and reference-refinement query is required")
    if set(evidence["references"])!=set(cases) or any(v.get("status")!="complete" for v in evidence["references"].values()):
        raise ValueError("Every native reference must complete")
    _verify_timing(evidence)
    return {"cases":len(cases),"models":len(models),"primary_accuracy_records":len(expected),
            "independent_initial_records":len(paired_expected),"total_retained_accuracy_records":len(expected)+len(paired_expected),
            "timing_records":len(evidence["timing"]["records"]),"restoration_records":len(evidence["timing"]["loading"])}


def _verify_timing(evidence):
    timing,study,plan = evidence["timing"],evidence["study"],evidence["plan"]
    if (not timing.get("complete") or not timing.get("final_timing_after_all_workers_finished")
            or timing.get("environment",{}).get("cpu_threads")!=1):
        raise ValueError("Complete isolated single-thread final timings required")
    expected,loads = {},set()
    fitted = {(m["case"],m["name"]):m for m in evidence["lock"]["models"]}
    for case in plan["cases"]:
        cfg = runner.reference_configuration(case,study)
        heads = _names(evidence,case)-{"LGA_dt7","LGA_dt10","native-MC-dt10"}
        for profile in cfg.cases:
            for name in heads:
                loads.add((case,profile,name))
                common = dict(regime=case,profile=profile,name=name,
                              family="repaired" if (case,name) in fitted else "untrained-repaired")
                if (case,name) in fitted:
                    common.update({k:fitted[(case,name)][k] for k in ("seed","proposal","loss","variant")})
                for batch in plan["neural_conditional_batches"]:
                    expected[f"{case}/{profile}/{name}/conditional/{batch}"] = {
                        **common,"role":"conditional","batch":batch}
                expected[f"{case}/{profile}/{name}/ensemble"] = {
                    **common,"role":"ensemble","batch":cfg.paths_per_group}
            for step in plan["LGA_max_steps"]:
                for role in ("conditional","ensemble"):
                    expected[f"{case}/{profile}/LGA/{step}/{role}"] = dict(regime=case,profile=profile,
                        name=f"LGA_dt{round(-np.log2(step))}",family="LGA",role=role,batch=1,step=step)
            for role in ("conditional","ensemble"):
                expected[f"{case}/{profile}/native-MC/{role}"] = dict(regime=case,profile=profile,
                    name="native-MC-dt10",family="native-MC",role=role,batch=cfg.paths_per_group,step=plan["native_MC_step"])
    actual = [r["slot"] for r in timing["records"]]
    restore = [(r["regime"],r["profile"],r["name"]) for r in timing["loading"]]
    if len(actual)!=len(expected) or set(actual)!=set(expected) or len(restore)!=len(loads) or set(restore)!=loads:
        raise ValueError("Every declared literal timing workload and model restoration is required")
    for row in timing["records"]:
        if any(row.get(k)!=v for k,v in expected[row["slot"]].items()):
            raise ValueError("Timing metadata must describe its literal declared workload")
        if any(not math.isfinite(row[k]) or row[k]<0 for k in ("median_seconds","first_prediction_seconds")):
            raise ValueError("Finite nonnegative measured costs required")
        prefix = "native_MC" if row["family"]=="native-MC" else (
            "ensemble" if row["family"] in NEURAL and row["role"]=="ensemble" else "conditional")
        repeats = plan[prefix+"_repeats"]
        samples = row.get("samples_seconds",[])
        cold,warm = row.get("cold_components",{}),row.get("warm_components",{})
        if (len(samples)!=repeats or not np.isfinite(samples).all() or min(samples)<0
                or not cold or set(cold)!=set(warm) or any(not math.isfinite(v) or v<0 for v in cold.values())
                or any(len(v)!=repeats or not np.isfinite(v).all() or min(v)<0 for v in warm.values())):
            raise ValueError("All same-call cold/warm timing component samples are required")
        if (sum(cold.values())>row["first_prediction_seconds"]+1e-6 or
                np.any(np.sum(list(warm.values()),axis=0)>np.asarray(samples)+1e-6)):
            raise ValueError("Timing components cannot exceed their same-call measured totals")
        if not np.isclose(row["median_seconds"],np.median(samples),rtol=1e-12,atol=1e-12):
            raise ValueError("Warm medians must replay their actual samples")
    if any(not math.isfinite(r["seconds"]) or r["seconds"]<0 for r in timing["loading"]):
        raise ValueError("Finite nonnegative restoration costs required")


def accuracy_tables(evidence):
    tables = _base.accuracy_tables(evidence)
    methods = {(m["case"],m["name"]):m for m in evidence["lock"]["models"]}
    for key in ("accuracy","components","covariances","sampling","contrasts"):
        for row in tables[key]:
            model = methods.get((row["regime"],row["name"]))
            row.update(variant=_variant(model) if model else None,
                       proposal=model["proposal"] if model else None,loss=model["loss"] if model else None)
            if key!="contrasts" and row.get("family") in NEURAL and row["role"]=="ensemble":
                row["NN_initial_pool"]="independent-of-reference"
    crossed = []
    for group in evidence["groups"]:
        case,key = group["regime"],group["group"]
        profile,role,anchor = key.split("__")
        cfg = runner.reference_configuration(case,evidence["study"])
        for name,contrast in group["NN_versus_uniform_huber"].items():
            model = methods[(case,name)]
            baseline = next(m for m in methods.values() if m["case"]==case and m["seed"]==model["seed"] and _variant(m)==BASELINE)
            records = contrast["records"]
            if len(records)!=len(cfg.query_times) or {v["time_index"] for v in records}!=set(range(len(cfg.query_times))):
                raise ValueError("Every matched contrast requires all declared query times")
            for item in records:
                for statistic in ("mean","covariance"):
                    low,high = item[statistic+"_advantage_percentile95"]
                    point = item[statistic+"_error_advantage"]
                    if not np.isfinite([point,low,high]).all() or low>high:
                        raise ValueError("Finite ordered pointwise error contrasts required")
                    crossed.append({"regime":case,"group":key,"profile":profile,"role":role,"anchor":int(anchor),
                        "time_index":item["time_index"],"time":cfg.query_times[item["time_index"]],
                        "name":name,"seed":model["seed"],"variant":_variant(model),"proposal":model["proposal"],
                        "loss":model["loss"],"control":baseline["name"],"statistic":statistic,
                        "advantage":point,"lower95":low,"upper95":high,
                        "candidate_closer_pointwise95":low>0,"baseline_closer_pointwise95":high<0,
                        "scope":"matched seed/shared learner pools and independent integration; pointwise only, no family-wise claim"})
    tables["crossed_contrasts"] = crossed
    tables["crossed_seed_consistency"] = seed_consistency(crossed)
    tables["gaussian_seed_consistency"] = seed_consistency(tables["contrasts"])
    return tables


def seed_consistency(rows):
    """Keep variants separate and reject incomplete/duplicated seed blocks."""
    keys = ("regime","profile","role","group","time_index","time","statistic","variant")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key,group in sorted(grouped.items()):
        seeds = [r["seed"] if r.get("seed") is not None else int(r["name"].rsplit("seed",1)[1]) for r in group]
        if len(group)!=3 or set(seeds)!={0,1,2}:
            raise ValueError("A matched contrast cannot drop or duplicate optimizer seeds")
        result.append({**dict(zip(keys,key,strict=True)),"optimizer_seeds":"(0, 1, 2)",
            "all_seeds_candidate_closer_pointwise95":all(r["lower95"]>0 for r in group),
            "all_seeds_control_closer_pointwise95":all(r["upper95"]<0 for r in group),
            "minimum_advantage":min(r["advantage"] for r in group),"maximum_advantage":max(r["advantage"] for r in group),
            "scope":"consistency across fixed seeds; not simultaneous confidence over regimes or components"})
    return result


def cost_tables(timing,lock,data_seconds,*,preparation_seconds=None,pool_seconds=None,worker_seconds=None):
    """One-model deployments pay one pool once; study totals share it once."""
    fitting = []
    for model in lock["models"]:
        seconds = model.get("full_invocation_seconds")
        if (model.get("full_invocation_cost_complete") is not True or seconds is None
                or not math.isfinite(seconds) or seconds<model["training_seconds"]):
            raise ValueError("Full fit invocation cost cannot be below recorded optimization loop time")
        fitting.append({**model,"training_seconds":seconds})
    if (preparation_seconds is None or not math.isfinite(preparation_seconds) or preparation_seconds<0
            or pool_seconds is None or worker_seconds is None or not math.isfinite(worker_seconds)
            or worker_seconds<sum(m["training_seconds"] for m in fitting)):
        raise ValueError("Complete nonadditive prepare/worker and per-seed pool costs required")
    if any((m["case"],m["seed"]) not in pool_seconds for m in fitting) or any(
            not math.isfinite(v) or v<0 for v in pool_seconds.values()):
        raise ValueError("Every seed/case shared pool conversion must be measured")
    if worker_seconds+1e-6<sum(m["training_seconds"] for m in fitting)+sum(pool_seconds.values()):
        raise ValueError("Nested fit and pool durations cannot exceed full worker time")
    # Charge the full observed input build once to a stand-alone deployment;
    # per-case native generation below is a breakdown, not an additive charge.
    costs,_ = _base.cost_tables(timing,{**lock,"models":fitting},{c:preparation_seconds for c in data_seconds})
    lookup = {(m["case"],m["name"]):m for m in lock["models"]}
    for row in costs:
        model = lookup.get((row["regime"],row["name"]))
        row.update(variant=_variant(model) if model else None,
                   recorded_optimizer_loop_seconds=model["training_seconds"] if model else 0.,
                   fit_cost_scope="all measured fit invocations" if model else "no fitting",
                   shared_seed_case_pool_conversion_seconds=pool_seconds[(model["case"],model["seed"])] if model else 0.,
                   case_native_generation_component_seconds=data_seconds[row["regime"]] if model else 0.,
                   data_cost_scope="full measured study prepare charged once; native-generation component not additive" if model else "no fitting pool")
        row["known_offline_seconds"] += row["shared_seed_case_pool_conversion_seconds"]
    totals = [{**{k:row[k] for k in ("slot","regime","profile","name","family","role","batch")},
        "query_calls":q,"known_offline_seconds":row["known_offline_seconds"],
        "first_prediction_seconds":row["first_prediction_seconds"],
        "subsequent_warm_seconds":(q-1)*row["median_seconds"],
        "total_known_seconds":row["known_offline_seconds"]+row["first_prediction_seconds"]+(q-1)*row["median_seconds"]}
        for row in costs for q in (1,100,10000,100000)]
    study = {"all_model_optimizer_loop_seconds":sum(m["training_seconds"] for m in lock["models"]),
             "all_model_fit_invocation_seconds":sum(m["training_seconds"] for m in fitting),
             "shared_state_pool_generation_seconds":sum(data_seconds.values()),
             "full_shared_preparation_seconds":preparation_seconds,
             "all_shared_seed_case_pool_conversion_seconds":sum(pool_seconds.values()),
             "all_worker_invocation_seconds":worker_seconds,
             "fit_pool_and_restoration_worker_overhead_seconds":worker_seconds-sum(m["training_seconds"] for m in fitting),
             "full_prepare_plus_worker_seconds":preparation_seconds+worker_seconds,
             "scope":"prepare plus worker totals; native generation, fits, conversions and restoration are nested components, never added again; summed concurrent durations are not elapsed study time"}
    return costs,totals,study


def amortization_tables(tables):
    rows = []
    for candidate in tables["costs"]:
        if candidate["family"]!="repaired" or (candidate["role"]=="conditional" and candidate["batch"]!=1):
            continue
        keys = {k:candidate[k] for k in ("regime","profile","role")}
        def compatible(row):
            return all(row[k]==v for k,v in keys.items())
        refinement = [r for r in tables["reference_refinement"] if compatible(r) and r["level"]=="level0_versus_finest"]
        for control in tables["costs"]:
            classical = control["family"] in ("LGA","native-MC")
            baseline = (control["family"]=="repaired" and control.get("variant")==BASELINE
                        and control["seed"]==candidate["seed"] and candidate.get("variant")!=BASELINE
                        and (control["role"]!="conditional" or control["batch"]==1))
            if not compatible(control) or not (classical or baseline):
                continue
            for statistic in ("mean","covariance"):
                gates = []
                for item in (candidate,control):
                    accuracy = [r for r in tables["accuracy"] if compatible(r) and r["name"]==item["name"]]
                    sampling = [r for r in tables["sampling"] if compatible(r) and r["name"]==item["name"]]
                    gates.append(_base.statistic_gate(accuracy,sampling,refinement,statistic)
                                 if len(sampling)==len(accuracy) and accuracy else "unresolved-inventory")
                arithmetic = _base.payback(candidate,control)
                qualified = arithmetic if all(g=="finite-reference-diagnostic-pass" for g in gates) else None
                rows.append({**keys,"candidate":candidate["name"],"control":control["name"],
                    "variant":candidate["variant"],"seed":candidate["seed"],"statistic":statistic,
                    "candidate_gate":gates[0],"control_gate":gates[1],"descriptive_payback_calls":arithmetic,
                    "accuracy_qualified_payback_calls":qualified,
                    "scope":"one complete model and its one data pool; fixed bank/physics/query times; finite-reference gate only"})
    return rows


def training_tables(evidence):
    """Natural validation units remain common even when training losses differ."""
    rows = []
    for model in evidence["lock"]["models"]:
        for stage in ("mean","covariance"):
            value = model["natural_validation"][stage]
            if (value["rows"]<1 or any(not math.isfinite(value[k]) or value[k]<0
                                      for k in ("natural_mse","natural_huber"))):
                raise ValueError("Complete natural-occupancy validation diagnostics required")
            rows.append({"regime":model["case"],"name":model["name"],"seed":model["seed"],
                "variant":_variant(model),"proposal":model["proposal"],"loss":model["loss"],"stage":stage,
                "selected_step":model["selected_step"][stage],"natural_validation_rows":value["rows"],
                "natural_validation_mse":value["natural_mse"],"natural_validation_rms":math.sqrt(value["natural_mse"]),
                "natural_validation_huber":value["natural_huber"],"optimizer_stage_seconds":model["stage_seconds"][stage],
                "recorded_optimizer_loop_seconds":model["training_seconds"],
                "full_invocation_seconds":model.get("full_invocation_seconds"),
                "selection":"common full natural-occupancy fixed-unit validation MSE",
                "covariance_source":"selected mean frozen within each variant; different variants can have different fitted means"})
    return rows


def _data_workers(evidence):
    data,workers,pools = [],[],[]
    for case in evidence["plan"]["cases"]:
        for role in ("train","validation"):
            path = _artifact(evidence["root"]/f"payoffs/{case}/states_{role}.json",evidence["inputs"])
            value = read(path)
            if (not value.get("complete") or not math.isfinite(value["generation_seconds"])
                    or value["generation_seconds"]<0):
                raise ValueError("Finite shared state-pool generation costs required")
            _artifact(value["path"],evidence["inputs"],value["sha256"])
            data.append({"regime":case,"role":role,"generation_seconds":value["generation_seconds"]})
    for seed in evidence["study"].optimizer_seeds:
        path = _artifact(evidence["root"]/f"payoffs/execution/worker_seed{seed}.json",evidence["inputs"])
        value = read(path)
        if not value.get("complete") or value["seed"]!=seed:
            raise ValueError("All optimizer-seed workers must finish before reporting")
        for i,attempt in enumerate(value["attempts"]):
            if any(not math.isfinite(attempt.get(k,float("nan"))) or attempt[k]<0 for k in ("wall_seconds","cpu_seconds")):
                raise ValueError("Finite worker attempt cost records required")
            if "pool_preparation" not in attempt:
                raise ValueError("Every resumed worker attempt must retain its shared pool costs")
            workers.append({"seed":seed,"attempt":i,**{k:v for k,v in attempt.items() if k!="pool_preparation"}})
            for case,record in attempt["pool_preparation"].items():
                if case not in evidence["plan"]["cases"] or any(
                        not math.isfinite(record[k]) or record[k]<0 for k in ("seconds","cpu_seconds")):
                    raise ValueError("Finite declared shared pool conversion costs required")
                pools.append({"regime":case,"seed":seed,"attempt":i,**record})
    expected = {(case,seed) for case in evidence["plan"]["cases"] for seed in evidence["study"].optimizer_seeds}
    if {(p["regime"],p["seed"]) for p in pools}!=expected:
        raise ValueError("Every seed/case shared pool-preparation measurement is required")
    for model in evidence["lock"]["models"]:
        path = _artifact(model["invocation_record_path"],evidence["inputs"],model["invocation_record_sha256"])
        attempts = read(path)["attempts"]
        if (not attempts or any(not a.get("finished") or not math.isfinite(a.get("wall_seconds",float("nan")))
                or a["wall_seconds"]<0 for a in attempts) or model.get("full_invocation_cost_complete") is not True
                or not np.isclose(sum(a["wall_seconds"] for a in attempts),model["full_invocation_seconds"],rtol=1e-12)):
            raise ValueError("All model invocation attempts require complete cost evidence")
    path = _artifact(evidence["root"]/"payoffs/prepare_costs.json",evidence["inputs"])
    preparation = runner._cost_log(path,{"workflow":runner.WORKFLOW,"source_hash":source_hash(),
                                         "plan_hash":evidence["study"].plan_hash})
    attempts = preparation["attempts"]
    if (not attempts or any(not a.get("finished") or any(not math.isfinite(a.get(k,float("nan"))) or a[k]<0
            for k in ("wall_seconds","cpu_seconds")) for a in attempts)):
        raise ValueError("All shared prepare invocations must retain complete costs")
    return data,workers,pools,[{"attempt":i,**a,"scope":preparation["scope"]} for i,a in enumerate(attempts)]


def _reference_inputs(evidence):
    """Bind the exact numerical references, not only their descendant metrics."""
    for case,expected in evidence["references"].items():
        path = _artifact(evidence["root"]/"reference"/case/"manifest.json",evidence["inputs"])
        packet = read(path)
        if packet!=expected or Path(packet["reference_path"]).resolve()!=path.with_name("reference.h5"):
            raise ValueError("Reference input identity changed")
        _artifact(packet["reference_path"],evidence["inputs"],packet["reference_sha256"])


def model_summary(accuracy,models):
    grouped = defaultdict(list)
    lookup = {(m["case"],m["name"]):m for m in models}
    for row in accuracy:
        grouped[(row["regime"],row["name"])].append(row)
    return [{"regime":case,"name":name,"family":rows[0]["family"],
             "variant":_variant(lookup[(case,name)]) if (case,name) in lookup else None,
             "seed":lookup[(case,name)]["seed"] if (case,name) in lookup else None,
             "maximum_mean_error_in_reference_std":max(r["max_mean_error_in_reference_std"] for r in rows),
             "maximum_normalized_covariance_error":max(r["max_normalized_covariance_error"] for r in rows),
             "maximum_mean_error_plus_sampling95":max(r["mean_error_plus_sampling95"] for r in rows),
             "maximum_covariance_error_plus_sampling95":max(r["covariance_error_plus_sampling95"] for r in rows)}
            for (case,name),rows in sorted(grouped.items())]


def _range(values,scale=1.):
    values = [float(v)*scale for v in values if v is not None]
    return "—" if not values else f"{min(values):.4g}" if min(values)==max(values) else f"{min(values):.4g}–{max(values):.4g}"


def figures(output,tables,cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    methods = [*VARIANTS,"untrained","LGA_dt7","LGA_dt10","native-MC-dt10"]
    for ax,metric,label in zip(axes,("maximum_mean_error_in_reference_std","maximum_normalized_covariance_error"),
                               ("Mean / reference SD","Covariance / reference SD products"),strict=True):
        for j,method in enumerate(methods):
            x,y,low,high = [],[],[],[]
            for i,case in enumerate(cases):
                selected = [r[metric] for r in tables["model_summary"] if r["regime"]==case and (
                    r["variant"]==method if method in VARIANTS else
                    r["family"]=="untrained-repaired" if method=="untrained" else r["name"]==method)]
                if not selected:
                    raise ValueError("Accuracy figure cannot omit a declared method")
                x.append(i+(j-3.5)*.085)
                lo,hi = max(min(selected),1e-12),max(max(selected),1e-12)
                center = math.sqrt(lo*hi)
                y.append(center)
                low.append(center-lo)
                high.append(hi-center)
            ax.errorbar(x,y,yerr=[low,high],marker="o",linestyle="none",markersize=4,capsize=2,label=method)
        ax.axhline(.05,color="black",linestyle="--",linewidth=1,label="5% diagnostic")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(cases)),cases,rotation=35,ha="right")
        ax.grid(axis="y",alpha=.25)
    axes[1].legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    fig.suptitle("Worst group/time point errors; min–max across all optimizer seeds, not confidence intervals")
    fig.savefig(output/"accuracy_cases.png",dpi=160)
    plt.close(fig)


def report_text(evidence,tables,study_cost):
    allocation = validate_evidence(evidence)
    qualified = sum(r["accuracy_qualified_payback_calls"] is not None for r in tables["amortization"])
    lines = ["# Matched-pool proposal and loss comparison", "",
        f"All **{allocation['models']} fits**, all four crossed variants and all three optimizer seeds are retained. "
        f"The primary comparison has **{allocation['primary_accuracy_records']} predictions**; "
        f"**{allocation['total_retained_accuracy_records']} records** remain including paired-initial diagnostics.", "",
        "## Fixed comparison", "",
        "The architecture is the same moving-boundary/native-slope v3 model. Within each regime/seed, "
        "the four variants use byte-matched fresh natural-occupancy training/validation pools, initialization "
        "and training-only residual scales. Adam uses uniform or importance-corrected stratified proposals; "
        "full-pool LBFGS uses the natural row measure. All variants select by the complete natural-validation "
        "fixed-unit MSE. The two training losses are pseudo-Huber and squared residuals. "
        "No stress-augmented target measure is silently substituted.", "",
        "The output bank is four terminal/killed-integral means and same-time 4×4 covariance at three horizons. "
        "Native stopped finite-volume references retain every stopped family; Gaussian is an unstopped approximation. "
        "Both Gaussian integration steps, independent native Monte Carlo and the untrained v3 control remain. "
        "Primary IID-law neural predictions integrate fresh histograms independently of reference histograms; "
        "shared-initial predictions are retained only as paired diagnostics.", "",
        "## Independent accuracy", "", "![](accuracy_cases.png)", "",
        "Each maximum below covers all physical query times, initial profiles and conditional/IID groups. "
        "Ranges retain all three seeds; they are not confidence intervals. Mean percentages use reference SD, "
        "covariances use products of reference SDs, and neither is relative error of a possibly zero mean.", "",
        "| Regime | Method | Worst mean (% SD) | Worst covariance (% SD products) |",
        "|---|---|---:|---:|"]
    for case in evidence["plan"]["cases"]:
        for method in [*VARIANTS,"untrained","LGA_dt7","LGA_dt10","native-MC-dt10"]:
            selected = [r for r in tables["model_summary"] if r["regime"]==case and (
                r["variant"]==method if method in VARIANTS else
                r["family"]=="untrained-repaired" if method=="untrained" else r["name"]==method)]
            lines.append(f"| {case} | {method} | {_range([r['maximum_mean_error_in_reference_std'] for r in selected],100)} "
                         f"| {_range([r['maximum_normalized_covariance_error'] for r in selected],100)} |")
    lines += ["", "## Matched-seed proposal/loss contrasts", "",
        "Positive contrasts mean the candidate error is smaller than uniform/pseudo-Huber at the same "
        "optimizer seed. All four variants share fresh reference resamples; independent initial-integration "
        "resamples are common across the neural predictors and independent of reference resamples. "
        "Every declared seed, group, time and statistic remains in crossed_contrasts.csv. "
        "Counts below require the same pointwise conclusion for all three seeds; they are not a simultaneous "
        "grid-wide confidence statement or permission to select a favorable seed.", "",
        "| Regime | Candidate versus uniform/pseudo-Huber | Statistic | Candidate closer, all seeds | Baseline closer, all seeds | Comparisons |",
        "|---|---|---|---:|---:|---:|"]
    for case in evidence["plan"]["cases"]:
        for variant in VARIANTS:
            if variant==BASELINE:
                continue
            for statistic in ("mean","covariance"):
                selected = [r for r in tables["crossed_seed_consistency"] if
                            (r["regime"],r["variant"],r["statistic"])==(case,variant,statistic)]
                lines.append(f"| {case} | {variant} | {statistic} "
                    f"| {sum(r['all_seeds_candidate_closer_pointwise95'] for r in selected)} "
                    f"| {sum(r['all_seeds_control_closer_pointwise95'] for r in selected)} | {len(selected)} |")
    lines += ["", "## Training objective and validation", "",
        "natural_validation.csv reports both natural MSE/RMS and natural pseudo-Huber diagnostics in the "
        "same fixed training-derived units. A lower robust training objective is not evidence of lower "
        "fluctuation error. Covariance training freezes each variant's selected mean; different fitted "
        "variants can still have different mean-gradient sources. The crossed study compares complete "
        "fitted methods, not covariance heads with a uniquely fixed shared learned source.", "",
        "## Fitting, preparation and inference costs", "",
        f"Recorded optimizer-loop time totals {study_cost['all_model_optimizer_loop_seconds']:.6g} s. "
        f"Recorded fit-invocation time totals {study_cost['all_model_fit_invocation_seconds']:.6g} s. "
        f"Shared train/validation state-pool generation costs {study_cost['shared_state_pool_generation_seconds']:.6g} s "
        f"within {study_cost['full_shared_preparation_seconds']:.6g} s of complete preparation. "
        f"Full seed-worker invocations total {study_cost['all_worker_invocation_seconds']:.6g} s. "
        "The recorded preparation/training allocation adds full preparation and worker durations once. Native generation, model fits, "
        "shared pool conversion, preflight and restoration are nested components, never additional charges. "
        "Summed concurrent durations are not elapsed batch time; "
        "worker attempts retain CPU, monotonic durations and UTC endpoints separately.", "",
        f"Reference construction separately recorded {study_cost['reference_generation_seconds_not_deployment_charge']:.6g} s. "
        "It is a comparison-study expense and is not charged to a model deployment. Unrecorded analysis, "
        "test and bootstrap runtimes are not represented as measured costs.", "",
        "Each fixed-model deployment pays its own fit, the full measured study preparation once, its "
        "seed/case pool conversion once, restoration and "
        "the measured cold first query. The 1/100/10,000/100,000-call curves add subsequent warm measured calls. "
        "For known-horizon ensemble/batched calls, the separately charged B=1 cold probe is a complete "
        "discarded prediction, not a pure cache-creation microbenchmark. Charging the full observed "
        "multi-regime input preparation is conservative for a standalone model; a minimal one-regime "
        "build has not been timed. In-study shared pools are not "
        "multiplied by the number of variants, optimizer seeds, profiles or timing rows.", "",
        "Conditional predictions accept a supplied histogram. IID-law timings actually draw and integrate "
        "the declared complete histogram pool; Gaussian propagates initial covariance analytically and "
        "native MC draws physical forcing and simulates its complete declared ensemble. Initialization, "
        "prediction and aggregation components are reported where measured, without subtracting unrelated "
        "medians or extrapolating a small-batch timing into a large-ensemble result.", "",
        f"There are **{qualified} accuracy-qualified finite-reference payback rows**. Descriptive arithmetic "
        "payback remains labelled separately when accuracy, standalone sampling or refinement gates fail. "
        "The 5% error and 1.25% uncertainty-component gates do not certify remaining continuous-time "
        "stopping bias. A lower warm latency or an unqualified arithmetic crossover is not an equal-accuracy "
        "neural advantage.", "",
        "The result is specific to the declared finite-grid target, bank, fixed physics and tested initial "
        "states. Bootstrap intervals are pointwise; no continuum limit, rare-event accuracy, unseen-law "
        "generalization or universal neural improvement follows. All primary records, paired diagnostics, "
        "reference refinements, fitting failures/accuracy failures and unresolved gates are preserved.", ""]
    return "\n".join(lines)


def _json(path,value):
    def encode(item):
        if isinstance(item,np.generic):
            return item.item()
        if isinstance(item,np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported report value: {type(item)}")
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=encode)+"\n")
    temporary.replace(path)


def generate(root,documentation=None,output=None):
    evidence = load_complete(root)
    root = evidence["root"]
    for name in ("payoffs/matched_controls.json","payoffs/flat_pool_lock.json","comparison/evaluation_lock.json"):
        _artifact(root/name,evidence["inputs"])
    _reference_inputs(evidence)
    data,workers,pools,preparations = _data_workers(evidence)
    tables = accuracy_tables(evidence)
    tables["natural_validation"] = training_tables(evidence)
    tables["model_summary"] = model_summary(tables["accuracy"],evidence["lock"]["models"])
    if evidence["paired_initial_records"]:
        # Paired initial estimates are retained without replacing primary IID results.
        paired = _base.accuracy_tables({**evidence,"records":evidence["paired_initial_records"],
                                      "groups":evidence["paired_initial_groups"]})
        tables["paired_initial_accuracy"] = paired["accuracy"]
        tables["paired_initial_sampling"] = paired["sampling"]
    data_seconds = {case:sum(r["generation_seconds"] for r in data if r["regime"]==case)
                    for case in evidence["plan"]["cases"]}
    pool_seconds = {(case,seed):sum(p["seconds"] for p in pools if (p["regime"],p["seed"])==(case,seed))
                    for case in evidence["plan"]["cases"] for seed in evidence["study"].optimizer_seeds}
    tables["costs"],tables["total_costs"],study_cost = cost_tables(evidence["timing"],evidence["lock"],data_seconds,
        preparation_seconds=sum(p["wall_seconds"] for p in preparations),pool_seconds=pool_seconds,
        worker_seconds=sum(w["wall_seconds"] for w in workers))
    study_cost["reference_generation_seconds_not_deployment_charge"] = sum(
        v["generation_seconds"] for v in evidence["references"].values())
    tables["data"],tables["workers"],tables["pool_preparation"],tables["full_preparation"] = data,workers,pools,preparations
    tables["amortization"] = amortization_tables(tables)
    output = root/"report" if output is None else Path(output).resolve()
    if output==root or root.is_relative_to(output) or output.is_relative_to(root/"payoffs") or output.is_relative_to(root/"comparison") or output.is_relative_to(root/"deployment"):
        raise ValueError("Report output cannot replace authoritative study inputs")
    output.mkdir(parents=True,exist_ok=True)
    frozen_helpers = read(root/"comparison/evaluation_lock.json")["helpers"]
    helpers = {**frozen_helpers,**{str(REPO/"analysis"/name):sha(REPO/"analysis"/name) for name in (
        "run_payoff_collocation_study.py","evaluate_payoff_collocation_study.py",
        "report_payoff_regimes.py","payoff_study_adapter.py")}}
    for path,digest in helpers.items():
        _artifact(path,evidence["inputs"],digest)
    identity = {"kind":KIND,"root":str(root),"source_hash":source_hash(),"report_script_sha256":sha(__file__),
        "helpers":helpers,"inputs":evidence["inputs"]}
    if (output/"manifest.json").exists():
        previous = read(output/"manifest.json")
        if previous["identity"]!=identity:
            raise ValueError("Report inputs or source changed; use a new output directory")
        return load_report(output)
    if any(output.iterdir()):
        raise ValueError("A new empty report output directory is required")
    for name,rows in tables.items():
        _base.csv_write(output/(name+".csv"),rows)
    figures(output,tables,evidence["plan"]["cases"])
    text = report_text(evidence,tables,study_cost)
    (output/"report.md").write_text(text)
    _json(output/"study_costs.json",study_cost)
    result = {"complete":True,"identity":identity,"allocation":validate_evidence(evidence),
              "all_optimizer_seeds_and_variants_retained":True,"primary_IID_integration_independent":True,
              "paired_initial_results_diagnostic_only":True,"post_test_selection_or_refitting":False,
              "pointwise_intervals_not_simultaneous":True,"continuous_DK_claim":False,
              "accuracy_gate":{"point_error":.05,"uncertainty_components":.0125,
                               "scope":"finite-reference diagnostic; continuous stopping bias unresolved"},
              "study_costs":study_cost,"tables":tables}
    _json(output/"report.json",result)
    manifest = {"kind":KIND,"complete":True,"identity":identity,"source_hash":source_hash(),
        "generator_sha256":sha(__file__),"inputs":evidence["inputs"],"allocation":result["allocation"],
        "outputs":{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        "qualified_payback_rows":sum(r["accuracy_qualified_payback_calls"] is not None for r in tables["amortization"])}
    _json(output/"manifest.json",manifest)
    if documentation is not None:
        documentation = Path(documentation)
        prefix = f"artifacts/{documentation.stem}/report/"
        documentation.write_text(text.replace("](accuracy_cases.png)","]("+prefix+"accuracy_cases.png)"))
    return manifest


def load_report(output):
    output = Path(output).resolve()
    manifest = read(output/"manifest.json")
    identity = manifest["identity"]
    if (not manifest.get("complete") or manifest.get("kind")!=KIND
            or identity["source_hash"]!=source_hash() or identity["report_script_sha256"]!=sha(__file__)
            or manifest.get("source_hash")!=identity["source_hash"]
            or manifest.get("generator_sha256")!=identity["report_script_sha256"]
            or manifest.get("inputs")!=identity["inputs"]
            or any(sha(name)!=digest for name,digest in identity["helpers"].items())):
        raise ValueError("A complete source-matching crossed-study report is required")
    for path,digest in identity["inputs"].items():
        if sha(path)!=digest:
            raise ValueError("Frozen report input changed")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p!=output/"manifest.json"}
    if (any(Path(n).name!=n for n in manifest["outputs"]) or actual!=set(manifest["outputs"])
            or any(sha(output/name)!=digest for name,digest in manifest["outputs"].items())):
        raise ValueError("Report output evidence changed or is incomplete")
    result = read(output/"report.json")
    if not result.get("complete") or result["identity"]!=identity or result["allocation"]!=manifest["allocation"]:
        raise ValueError("Report result and manifest allocation differ")
    return manifest


def archive(root,destination):
    """Lossless compact report/source evidence; full weights and paths stay local."""
    root,destination = Path(root).resolve(),Path(destination).resolve()
    report = load_report(root/"report")
    load_complete(root)
    if destination.exists() or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Use a new archive outside the authoritative run")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collocation-archive-",dir=destination.parent) as temporary:
        staging = Path(temporary)/"archive"
        _base._archive_contents(root,staging)
        manifest = read(staging/"archive_manifest.json")
        for path,digest in report["identity"]["helpers"].items():
            original = Path(path).resolve()
            if not original.is_relative_to(REPO) or sha(staging/"code"/original.relative_to(REPO))!=digest:
                raise ValueError("An archived helper differs from its evaluation preregistration")
        for path in (REPO/"tests").glob("test_payoff_collocation*.py"):
            target = staging/"code"/path.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            manifest["copied"][str(target.relative_to(staging))] = {
                "sha256":sha(target),"original_sha256":sha(path),"compressed":False}
        (staging/"README.md").write_text("# Matched-pool proposal/loss comparison evidence\n\n"
            "See [the complete report](report/report.md). All four variants and optimizer seeds remain. "
            "Primary IID-law integration is independent of reference; paired-initial predictions are diagnostics. "
            "Large JSON is losslessly compressed; full checkpoint and trajectory arrays remain locally indexed "
            "by checksum. Exact code snapshots include source files that were untracked at execution; the "
            "original Git commit alone is not claimed to contain them. This finite-grid ablation is one "
            "research stage; the learned-corrections candidate and final study remain separate requirements. "
            "Code is MIT; authored prose, figures and tables are CC BY 4.0.\n")
        manifest.update(kind=KIND,report_identity=report["identity"],allocation=report["allocation"],
            active_readme_goal_complete=False,generated={"README.md":sha(staging/"README.md")})
        _json(staging/"archive_manifest.json",manifest)
        verify_archive(staging)
        staging.rename(destination)
    return verify_archive(destination)


def verify_archive(destination):
    result = _base.verify_archive(destination)
    manifest = read(Path(destination)/"archive_manifest.json")
    if (manifest.get("kind")!=KIND or manifest.get("active_readme_goal_complete") is not False
            or not manifest.get("complete")):
        raise ValueError("A complete scoped crossed-study archive is required")
    return {**result,"allocation":manifest["allocation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--documentation",type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify-report",type=Path)
    actions.add_argument("--archive",type=Path,help="Create a new compact archive after complete report and final test evidence")
    actions.add_argument("--verify-archive",type=Path)
    args = parser.parse_args()
    if not args.root and not (args.verify_report or args.verify_archive):
        parser.error("--root is required to generate a report or archive")
    result = (load_report(args.verify_report) if args.verify_report else verify_archive(args.verify_archive)
              if args.verify_archive else archive(args.root,args.archive) if args.archive else
              generate(args.root,args.documentation,args.output))
    print(json.dumps({k:v for k,v in result.items() if k not in ("identity","inputs","outputs")},indent=2))


if __name__=="__main__":
    main()
