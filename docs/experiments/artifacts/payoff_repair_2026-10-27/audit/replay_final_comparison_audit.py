"""Read-only final accuracy, training, deployment-cost and uncertainty audit."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from analysis import run_payoff_repair as runner
from nn_dk.data import source_hash
from nn_dk.payoff_repair import load_repaired_payoff_model,_validation

ROOT=Path('runs/payoff-repair-2026-10-27').resolve()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
def point_rows(record):
 metric=record['metrics']
 return metric['records'] if 'records' in metric else [v for b in metric.values() for v in b['records']]

torch.set_num_threads(1);begin=datetime.now(timezone.utc).isoformat();started=time.monotonic()
source=source_hash();assert source=='b7876b03f8f4fc6b5b030a3e631ed7404b35529f6ac3b391d4c4c1cdc90a8b15'
plan=read(ROOT/'comparison/preregistration.json');fit=runner.verify_fit_lock(ROOT)
identity={'source_hash':source,'plan_sha256':sha(ROOT/'comparison/preregistration.json'),
          'fit_lock_sha256':sha(ROOT/'payoffs/fit_lock.json'),
          'reference_sha256':sha(ROOT/'reference/reference.h5')}
index=read(ROOT/'comparison/accuracy_index.json');assert index['complete'] and index['identity']==identity
results=[];artifact_checks=[]
for key in runner.reference_configuration().group_keys:
 group=read(ROOT/f'comparison/accuracy/{key}/group.json');assert group['complete']
 assert len(group['records'])==33 and group['identity']==identity
 for item in group['records']:
  assert sha(item['path'])==item['sha256'];record=read(item['path'])
  assert record['complete'] and record['identity']['group']==key
  for name,value in identity.items():assert record['identity'][name]==value
  assert sha(record['arrays']['path'])==record['arrays']['sha256']
  results.append(record);artifact_checks.extend([{'path':item['path'],'sha256':item['sha256']},record['arrays']])
assert len(results)==198

summaries=[];groups=defaultdict(list)
for record in results:
 method=record['method']
 if method.get('bank'):
  groups[(method['family'],method['bank'],method.get('seed',None))].append(record)
for (family,bank,seed),records in sorted(groups.items(),key=lambda x:str(x[0])):
 rows=[v for record in records for v in point_rows(record)]
 item={'family':family,'bank':bank,'seed':seed,'group_count':len(records),
       'worst_mean_error_in_reference_std':max(v['max_mean_error_in_reference_std'] for v in rows),
       'worst_normalized_covariance_error':max(v['max_normalized_covariance_error'] for v in rows),
       'worst_mean_error_plus_sampling95':max(v['mean_error_plus_sampling95'] for v in rows),
       'worst_covariance_error_plus_sampling95':max(v['covariance_error_plus_sampling95'] for v in rows),
       'maximum_mean_sampling95_radius':max(v['mean_sampling95_simultaneous_radius'] for v in rows),
       'maximum_covariance_sampling95_radius':max(v['covariance_sampling95_simultaneous_radius'] for v in rows)}
 if bank=='terminal3-integral1':
  ratios=[v['candidate_scaled_covariance'][i][i]/v['reference_scaled_covariance'][i][i]
          for v in rows for i in (2,3)]
  integral_mean_errors=[abs(v['normalized_mean_difference'][i]) for v in rows for i in (2,3)]
  item.update(minimum_integral_variance_ratio=min(ratios),maximum_integral_variance_ratio=max(ratios),
              worst_integral_mean_error_in_reference_std=max(integral_mean_errors))
 summaries.append(item)

# Inspect literal optimizer allocations and the actual frozen-mean covariance
# source before covariance optimization, rather than an obsolete initial mean.
_,banks=runner._data_identity(ROOT/'payoffs')
validation={name:torch.tensor(banks['validation'][name],dtype=torch.float64) for name in ('states','horizons')}
phase_checks=[]
for item in fit['models']:
 saved=torch.load(item['checkpoint_path'],map_location='cpu',weights_only=True)
 assert not saved['provenance']['future_moment_labels']
 if item['variant']!='repaired':continue
 model=load_repaired_payoff_model(item['checkpoint_path'])
 assert saved['steps_done']=={'mean_adam':1000,'mean_lbfgs':30,'covariance_adam':1000,'covariance_lbfgs':30}
 assert all(torch.equal(saved['model_state'][k],saved['best_mean_state'][k]) for k in saved['model_state'] if k.startswith('mean_network.'))
 assert all(torch.equal(saved['best_state'][k],saved['best_mean_state'][k]) for k in saved['model_state'] if k.startswith('mean_network.'))
 model.load_state_dict(saved['best_mean_state'])
 initial_for_selected_mean=_validation(model,validation,model.config,model.loss_scaling,'covariance')
 best=saved['best_validation']['covariance']
 assert best['objective']<=initial_for_selected_mean['objective']*(1+1e-12)
 phase_checks.append({'name':item['name'],'steps':saved['steps_done'],'mean_frozen':True,
                      'initial_covariance_objective_before_mean_fit':saved['initial_validation']['covariance']['objective'],
                      'initial_covariance_objective_with_selected_frozen_mean':initial_for_selected_mean['objective'],
                      'selected_covariance_objective':best['objective'],
                      'mean_selected_validation_objective':saved['best_validation']['mean']['objective'],
                      'training_seconds':item['training_seconds']})

training_total=sum(item['training_seconds'] for item in fit['models'])
assert training_total==fit['total_recorded_model_training_seconds']
workers=[read(ROOT/f'payoffs/execution/worker_seed{seed}.json') for seed in (0,1,2)]
assert all(w['complete'] for w in workers)
attempts=[a for w in workers for a in w['attempts']]
worker_span=max(a['finished_unix'] for a in attempts)-min(a['started_unix'] for a in attempts)
training={'total_all_ten_fit_seconds':training_total,
          'repaired_six_fit_seconds':sum(item['training_seconds'] for item in fit['models'] if item['variant']=='repaired'),
          'per_repaired_fit_seconds':[item['training_seconds'] for item in fit['models'] if item['variant']=='repaired'],
          'aggregate_worker_cpu_seconds':sum(w['cpu_seconds'] for w in workers),
          'aggregate_worker_wall_seconds_not_elapsed':sum(w['wall_seconds'] for w in workers),
          'elapsed_worker_span_seconds':worker_span,
          'repaired_architecture':'two independent three-hidden-layer width32 networks',
          'repaired_optimizer_allocation':'each of mean and covariance:1000AdamW+30LBFGS outer steps,max_iter5',
          'ablation_allocation':'v1 raw/balanced:1000AdamW+30LBFGS outer steps,max_iter5;onejointnetwork',
          'interpretation':'Capacity, formulation and allocation changed together;only v1-none versus v1-balanced isolates loss balancing.'}

timing=read(ROOT/'comparison/timing.json');rows=timing['records']
assert timing['complete'] and timing['final_timing_after_all_workers_finished'] and len(rows)==232
assert timing['identity']=={name:value for name,value in identity.items() if name!='reference_sha256'}
assert len({r['slot'] for r in rows})==232
for r in rows:
 samples=np.asarray(r['samples_seconds'])
 assert len(samples)==r['repeats'] and np.isfinite(samples).all() and (samples>0).all()
 assert np.median(samples)==r['median_seconds']
 if r['family'] in ('native-Euler','learned-path','Picard'):
  assert r['batch']==16384 and r['repeats']==3
  assert r['scope']=='fresh-initials-and-forcing-plus-complete-paths-and-bank-aggregation'
 elif r['family']=='LGA':
  assert r['scope']=='literal-three-query-marginals-analytic-IID-covariance-for-ensemble'
 elif r['role']=='ensemble':
  assert r['batch']==16384 and r['repeats']==5
  assert r['scope']=='fresh-histograms-plus-inference-plus-total-covariance-aggregation'
 else:
  assert r['batch'] in (1,64,256) and r['repeats']==21
  assert r['scope']=='three-horizon-four-mean-and-same-time-4x4-covariance-per-initial'
# Frozen source explicitly selects the bank for path timing.
helper=(Path('analysis/evaluate_payoff_repair.py').read_text())
assert '_path_values(r, compiled, origin, noise, cfg, case, b.name)' in helper

cost_ranges=[]
for family in ('repaired','v1-none','v1-balanced','untrained-repaired','LGA','native-Euler','learned-path','Picard'):
 for role in ('conditional','ensemble'):
  selected=[r for r in rows if r['family']==family and r['role']==role
            and (family not in ('repaired','v1-none','v1-balanced','untrained-repaired') or role!='conditional' or r['batch']==1)
            and (family!='LGA' or r['dt']==2**-7)]
  if selected:cost_ranges.append({'family':family,'role':role,'records':len(selected),
    'median_seconds_min':min(r['median_seconds'] for r in selected),
    'median_seconds_max':max(r['median_seconds'] for r in selected),
    'first_call_seconds_min':min(r['first_prediction_seconds'] for r in selected),
    'first_call_seconds_max':max(r['first_prediction_seconds'] for r in selected)})
# Every repaired model is reloaded per case; only its first B1 probe is cold.
cold_checks=[]
for case in ('uniform','nonuniform'):
 for item in fit['models']:
  if item['variant']!='repaired':continue
  selected=[r for r in rows if r['case']==case and r['name']==item['name']]
  b1=next(r for r in selected if r['role']=='conditional' and r['batch']==1)
  cold_checks.append({'case':case,'name':item['name'],'cold_B1_first_prediction_seconds':b1['first_prediction_seconds'],
                      'warm_B1_median_seconds':b1['median_seconds'],
                      'later_batches_and_ensemble_share_primed_analytic_cache':True,
                      'required_known_probe_charge_for_later_cold_deployment':b1['first_prediction_seconds']})

repaired=[s for s in summaries if s['family']=='repaired']
assert len(repaired)==6 and all(s['worst_mean_error_in_reference_std']<.05 and s['worst_normalized_covariance_error']<.05 for s in repaired)
qualified=False # Reference continuous stopping bias is explicitly unresolved.
result={'kind':'independent-payoff-repair-final-comparison-audit-v1','complete':True,
        'started_utc':begin,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'audit_script_sha256':sha(__file__),'source_hash':source,'reference_and_fit_identity':identity,
        'accuracy_records_checked':198,'accuracy_artifact_checks':artifact_checks,'timing_rows_checked':232,
        'timing_sha256':sha(ROOT/'comparison/timing.json'),'per_seed_bank_worst_errors':summaries,
        'frozen_mean_covariance_phase_checks':phase_checks,'training_costs':training,
        'literal_cost_ranges':cost_ranges,'cold_vs_warm_cache_checks':cold_checks,
        'all_repaired_point_statistics_below_five_percent':True,
        'population_five_percent_accuracy_certificate':False,'accuracy_qualified_crossover_established':qualified,
        'all_comparisons_use_one_bank_three_horizons_and_same_time_four_by_four_covariance':True,
        'initial_ensemble_and_path_Monte_Carlo_are_actual_16384_samples_not_extrapolated':True,
        'source_helpers_checkpoints_or_references_modified':False,'post_test_refitting':False,
        'limitations':['Observed point improvement does not by itself certify5percent population accuracy.',
                      'Reference sampling uncertainty and continuous stopping bias remain separate.',
                      'Raw conditional latency payback is arithmetic only until an accuracy budget is resolved.',
                      'Repaired fitted architecture is compared to an unstopped Gaussian law approximation under favorable short-horizon weak-noise conditions.']}
dest=ROOT/'audit/final_comparison_audit.json';tmp=dest.with_suffix('.json.tmp')
tmp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
print(json.dumps({'complete':True,'path':str(dest),'sha256':sha(dest),'elapsed_seconds':result['elapsed_seconds'],
                  'all_ten_training_seconds':training_total,'per_seed_bank_rows':len(summaries),'timing_rows':232},indent=2))
