"""Independent final-reference integrity, stream, coupling and reduction audit."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np

from analysis import run_payoff_repair as runner
from analysis import path_generalization_reference as reference
from nn_dk.data import source_hash
from nn_dk.native import boundary_step, simulate_native
from nn_dk.payoff_learning import PayoffBank
from nn_dk.types import seeded_rng, stable_hash

ROOT=Path('runs/payoff-repair-2026-10-27').resolve()
REPO=Path.cwd()

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def families(path):
 result=set()
 with h5py.File(path,'r') as store:
  def visit(name,obj):
   if isinstance(obj,h5py.Dataset) and name.split('/')[-1] in ('family_ids','integration_family_ids'):
    result.update(v for v in obj[:].astype(str).ravel() if v)
  store.visititems(visit)
 return result

def aggregate(dw, factor):
 return dw.reshape(len(dw),-1,factor,dw.shape[-1]).sum(axis=2)

def independently_reduce(states,times,stops,bank,query_times):
 # Integrate each linear observation over the actually active piece of every
 # Euler segment; stored right endpoints are the interpolated stopped state.
 terminal=states@bank.terminal_weights.T
 running=states@bank.running_weights.T
 exposure=np.maximum(0,np.minimum(np.diff(times)[None,:],stops[:,None]-times[None,:-1]))
 increments=(running[:,:-1]+running[:,1:])*.5*exposure[:,:,None]
 output=terminal.copy()
 output[:,1:]+=np.cumsum(increments,axis=1)
 indices=np.rint(np.asarray(query_times)/(times[1]-times[0])).astype(int)
 return output[:,indices]

started=datetime.now(timezone.utc).isoformat()
cfg=runner.reference_configuration()
manifest=read(ROOT/'reference/manifest.json')
pretest=read(ROOT/'audit/pretest_audit.json')
assert pretest['source_hash']==source_hash()==manifest['identity']['source_hash']
assert manifest['status']=='complete' and manifest['committed_chunks']==1536
assert manifest['failed_paths']==0
path=ROOT/'reference/reference.h5'
assert sha(path)==manifest['reference_sha256']
assert manifest['identity']['config_hash']==cfg.config_hash
assert manifest['identity']==json.loads(json.dumps(reference._identity(cfg)))
fit_lock=runner.verify_fit_lock(ROOT)
normalized=[{'checkpoint_path':str(Path(item['checkpoint_path']).resolve()),
             'checkpoint_sha256':item['checkpoint_sha256'], 'source_hash':source_hash(), 'complete':True}
            for item in fit_lock['models']]
expected_fit={'expected_models':10,'models':normalized,'inventory_hash':stable_hash(normalized)}
assert manifest['fit_lock']==expected_fit

historical=set()
for item in pretest['historical_family_inventory']:
 if item.get('present'):
  old=families(item['path'])
  assert stable_hash(sorted(old))==item['family_identifiers_hash']
  historical.update(old)
assert len(historical)==pretest['unique_historical_families_checked']==148928
current=set()
for role in ('train','validation'):
 current.update(families(ROOT/f'payoffs/interacting_{role}.h5'))
assert len(current)==384 and not current&historical
old_ref=REPO/'runs/path-generalization-2026-10-26/reference/reference.h5'
old_hashes=set()
with h5py.File(old_ref,'r') as store:
 for key in cfg.group_keys:
  for name in ('base_brownian_sha256','drawn_brownian_sha256'):
   old_hashes.update(store[key][name][:].astype(str))

seen=set();base_seen=set();drawn_seen=set();group_summaries=[];replays=[]
max_state=max_bank=max_coupling=0.
banks=(PayoffBank.terminal_high_modes(cfg.spec,cfg.horizon),
       PayoffBank.mixed_terminal_integral(cfg.spec,cfg.horizon))
with h5py.File(path,'r') as store:
 assert json.loads(store.attrs['identity_json'])==manifest['identity']
 assert json.loads(store.attrs['fit_lock_json'])==expected_fit
 assert np.array_equal(store['common_times'][:],cfg.common_times)
 assert np.array_equal(store['query_times'][:],cfg.query_times)
 assert np.array_equal(store['reference_factors'][:],cfg.reference_factors)
 for key in cfg.group_keys:
  group=store[key];case,role,anchor=reference._group_plan(key)
  ids=group['family_ids'][:].astype(str)
  expected_ids=np.array([stable_hash([reference.STREAM,cfg.seed,cfg.config_hash,case,role,anchor,index])
                         for index in range(cfg.paths_per_group)])
  assert np.array_equal(ids,expected_ids)
  unique=set(ids)
  assert len(unique)==cfg.paths_per_group and not unique&seen
  assert not unique&current and not unique&historical
  seen.update(unique)
  base_hashes=set(group['base_brownian_sha256'][:].astype(str))
  drawn_hashes=set(group['drawn_brownian_sha256'][:].astype(str))
  assert len(base_hashes)==len(drawn_hashes)==cfg.paths_per_group
  assert not base_seen&base_hashes and not drawn_seen&drawn_hashes
  assert not (base_hashes|drawn_hashes)&old_hashes
  base_seen.update(base_hashes);drawn_seen.update(drawn_hashes)
  complete=group['chunk_complete'][:]
  assert complete.all() and len(complete)==256
  assert not np.isfinite(group['stopped_at'][:]).any()
  # Independently recompute every committed chunk digest over stored bytes.
  shapes=reference._shapes(cfg)
  for chunk in range(256):
   start=chunk*cfg.chunk_paths;stop=min(start+cfg.chunk_paths,cfg.paths_per_group)
   digest=hashlib.sha256()
   for name in sorted(shapes):
    end=min(stop,cfg.refinement_paths) if name.startswith('audit_') else stop
    if end<=start:continue
    digest.update(name.encode());digest.update(np.ascontiguousarray(group[name][start:end]).tobytes())
   assert digest.hexdigest()==group['chunk_sha256'][chunk].decode()
  assert stable_hash(list(group['chunk_sha256'][:].astype(str)))==manifest['group_content_hashes'][key]
  probabilities,audit=reference.profile_probabilities(cfg,case)
  np.testing.assert_array_equal(group['probabilities'][:],probabilities)
  assert json.loads(group.attrs['profile_probability_audit_json'])==audit
  conditional=None
  if role=='conditional':
   arng=seeded_rng(cfg.seed,reference.STREAM,cfg.config_hash,'held-out-anchor',case,anchor)
   conditional=arng.multinomial(cfg.n_particles,probabilities)/cfg.n_particles
   np.testing.assert_array_equal(json.loads(group.attrs['conditional_initial_json']),conditional)
   assert array_sha(conditional)==group.attrs['conditional_initial_sha256']
  for start in (0,cfg.refinement_paths):
   count=cfg.chunk_paths
   rng=seeded_rng(cfg.seed,reference.STREAM,cfg.config_hash,'reference-chunk',case,role,anchor,start//count)
   initial=(rng.multinomial(cfg.n_particles,probabilities,size=count)/cfg.n_particles
            if role=='ensemble' else np.tile(conditional,(count,1)))
   finest=cfg.reference_factors[-1] if start<cfg.refinement_paths else 1
   steps=round(cfg.horizon/cfg.reference_step)*finest
   dw=rng.normal(0,np.sqrt(cfg.reference_step/finest),(count,steps,cfg.n_cells))
   base=aggregate(dw,finest)
   np.testing.assert_array_equal(initial,group['initial'][start:start+count])
   assert np.array_equal(np.array([array_sha(row) for row in base]),group['base_brownian_sha256'][start:start+count].astype(str))
   assert np.array_equal(np.array([array_sha(row) for row in dw]),group['drawn_brownian_sha256'][start:start+count].astype(str))
   np.testing.assert_array_equal(aggregate(base,round(cfg.model_step/cfg.reference_step)),group['model_increments'][start:start+count])
   for factor in (cfg.reference_factors if finest>1 else (1,)):
    increments=aggregate(dw,finest//factor)
    dt=cfg.reference_step/factor
    times=np.arange(increments.shape[1]+1)*dt
    # Package-native Euler is independent of the vectorized analysis helper
    # that generated the final bank; selected physical paths must agree.
    batch=simulate_native(cfg.context(case),cfg.spec,times,dt,0,n_paths=2,
                          initial_masses=initial[:2],brownian_increments=increments[:2],
                          family_ids=tuple(ids[start:start+2]))
    states=batch.states
    common=states[:,np.rint(cfg.common_times/dt).astype(int)]
    expected_common=(group['common_paths'][start:start+2] if factor==1 else
                     group['audit_common_paths'][start:start+2,list(cfg.reference_factors).index(factor)])
    state_err=float(abs(common-expected_common).max());max_state=max(max_state,state_err)
    assert state_err<2e-13
    np.testing.assert_array_equal(batch.stopped_at,group['stopped_at'][start:start+2])
    errs=[]
    for bank,field in zip(banks,('bank_b','bank_c')):
     values=independently_reduce(states,times,batch.stopped_at,bank,cfg.query_times)
     target=(group[field][start:start+2] if factor==1 else
             group['audit_'+field][start:start+2,list(cfg.reference_factors).index(factor)])
     err=float(abs(values-target).max());max_bank=max(max_bank,err);errs.append(err)
     assert err<2e-13
    if finest>1:
     assert np.array_equal(np.array([array_sha(row) for row in increments]),
                           group['audit_brownian_sha256'][start:start+count,list(cfg.reference_factors).index(factor)].astype(str))
     if factor>1:
      # Floating summation tree changes can introduce last-bit roundoff only.
      err=float(abs(aggregate(increments,factor)-base).max())
      max_coupling=max(max_coupling,err);assert err<1e-15
    replays.append({'group':key,'chunk_start':start,'paths_replayed':2,'refinement_factor':factor,
                    'dt':dt,'state_error':state_err,'terminal_bank_error':errs[0],
                    'mixed_integral_bank_error':errs[1]})
  group_summaries.append({'group':key,'paths':len(ids),'committed_chunks':256,
                         'unique_family_ids':len(unique),'unique_base_brownian_hashes':len(base_hashes),
                         'unique_drawn_brownian_hashes':len(drawn_hashes),'stopped_paths':0,
                         'current_and_historical_family_overlap':0,'historical_brownian_overlap':0})

# Explicit crossing control: validate the killed integral on a segment that
# reaches the boundary between saved nodes, then remains frozen.
m=np.full(cfg.n_cells,1/cfg.n_cells);proposal=m.copy()
proposal[0]=-.01;proposal[1]+=m[0]+.01
hit,crossed,stop=boundary_step(m,proposal,0.,.01,cfg.spec)
assert crossed and 0<float(stop)<.01
states=np.array([[m,hit,hit]])
times=np.array([0.,.01,.02]);stops=np.array([float(stop)])
synthetic=[]
for bank in banks:
 actual=bank.evaluate_paths(states,times,stops)
 manual=independently_reduce(states,times,stops,bank,(.01,.02))
 np.testing.assert_allclose(actual[:,1:],manual,atol=2e-15,rtol=0)
 # Frozen terminal, killed integral: both outputs remain unchanged after hit.
 np.testing.assert_allclose(actual[:,1],actual[:,2],atol=2e-15,rtol=0)
 synthetic.append({'bank':bank.name,'stopping_time':float(stop),'error':float(abs(actual[:,1:]-manual).max())})

assert len(seen)==len(base_seen)==len(drawn_seen)==98304
result={'kind':'independent-payoff-repair-reference-audit-v1','complete':True,
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'audit_script_sha256':sha(__file__),'source_hash':source_hash(),'source_freeze_commit':'10a898d',
        'reference_path':str(path),'reference_sha256':sha(path),
        'reference_manifest_sha256':sha(ROOT/'reference/manifest.json'),
        'configuration_hash':cfg.config_hash,'fit_lock_sha256':sha(ROOT/'payoffs/fit_lock.json'),
        'strictly_restored_models':10,'unique_reference_families':len(seen),
        'unique_base_brownian_hashes':len(base_seen),'unique_drawn_brownian_hashes':len(drawn_seen),
        'current_training_validation_families':len(current),'historical_families_checked':len(historical),
        'historical_reference_brownian_hashes_checked':len(old_hashes),
        'reference_family_overlap_with_current_or_historical':0,'historical_reference_brownian_overlap':0,
        'validated_chunks':1536,'failed_paths':0,'stopped_paths':0,
        'groups':group_summaries,'selected_RNG_force_and_refinement_replays':replays,
        'maximum_state_replay_error':max_state,'maximum_terminal_integral_reduction_error':max_bank,
        'maximum_Brownian_sum_coupling_error':max_coupling,'synthetic_killed_integral_boundary_controls':synthetic,
        'limitations':['No observed reference stops; synthetic interpolation checks do not certify rare-exit statistics.',
                      'Coupled finite Euler refinements do not certify the continuous hitting-time law.',
                      'Hash-addressed pseudorandom streams provide reproducible independence design, not a mathematical independence proof.']}
dest=ROOT/'audit/reference_independence_audit.json'
tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
print(json.dumps({'complete':True,'audit':str(dest),'sha256':sha(dest),
                  'state_replay_error':max_state,'bank_reduction_error':max_bank,
                  'coupling_error':max_coupling,'reference_families':len(seen)},indent=2))
