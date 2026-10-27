"""Independent selected prediction and whole-family metric replay."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import h5py,numpy as np,torch
from analysis.run_payoff_repair import load_declared_model,reference_configuration
from nn_dk.data import source_hash
from nn_dk.path_runtime import PreparedPathRuntime
from nn_dk.payoff_learning import PayoffBank
from nn_dk.types import stable_hash
ROOT=Path('runs/payoff-repair-2026-10-27').resolve();cfg=reference_configuration();torch.set_num_threads(1)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def count_sha(a):return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
def empirical(y,counts=None):
 p=len(y);center=y.mean(axis=0);z=y-center;products=z[..., :,None]*z[...,None,:]
 if counts is None:return center,products.sum(axis=0)/(p-1)
 delta=(counts@z.reshape(p,-1)).reshape(len(counts),*y.shape[1:])/p
 sec=(counts@products.reshape(p,-1)).reshape(len(counts),*products.shape[1:])
 return center+delta,(sec-p*delta[..., :,None]*delta[...,None,:])/(p-1)
def mixture(mu,ncov,counts=None):
 mean,between=empirical(mu,counts)
 within=(ncov.mean(axis=0) if counts is None else (counts@ncov.reshape(len(mu),-1)).reshape(len(counts),*ncov.shape[1:])/len(mu))
 return mean,within+cfg.n_particles*between

def metric_check(record,ref,cm,cc,bcm,bcc,counts):
 rm,rc=empirical(ref);brm,brc=empirical(ref,counts)
 s=np.sqrt(np.diagonal(rc,axis1=-2,axis2=-1));sc=s[..., :,None]*s[...,None,:]
 dm=(cm-rm)/s;dc=(cc/cfg.n_particles-rc)/sc
 bdm=(bcm-brm)/s;bdc=(bcc/cfg.n_particles-brc)/sc
 errors=[]
 for i,saved in enumerate(record['records']):
  meanpoint=float(abs(dm[i]).max());covpoint=float(abs(dc[i]).max())
  mrad=float(np.quantile(abs(bdm[:,i]-dm[i]).max(axis=-1),.95))
  crad=float(np.quantile(abs(bdc[:,i]-dc[i]).max(axis=(-2,-1)),.95))
  expected={'max_mean_error_in_reference_std':meanpoint,'max_normalized_covariance_error':covpoint,
            'mean_sampling95_simultaneous_radius':mrad,'covariance_sampling95_simultaneous_radius':crad,
            'mean_error_plus_sampling95':meanpoint+mrad,'covariance_error_plus_sampling95':covpoint+crad,
            'reference_mean':rm[i],'candidate_mean':cm[i],
            'reference_scaled_covariance':cfg.n_particles*rc[i],'candidate_scaled_covariance':cc[i],
            'normalized_mean_difference':dm[i],'normalized_covariance_difference':dc[i]}
  for key,value in expected.items():
   err=float(np.max(abs(np.asarray(saved[key])-value)));errors.append(err)
   assert err<3e-12,(key,i,err)
 assert record['bootstrap']['counts_sha256']==count_sha(counts)
 return max(errors)

started=time.monotonic();begin=datetime.now(timezone.utc).isoformat();checks=[]
for key in ('nonuniform__conditional__0','uniform__ensemble__0'):
 with h5py.File(ROOT/'reference/reference.h5','r') as f:
  g=f[key];initial=g['initial'][:];ref=g['bank_c'][:]
 seed=int(stable_hash([2026102714,key])[:12],16)
 counts=np.random.default_rng(seed).multinomial(len(ref),np.full(len(ref),1/len(ref)),size=128).astype(float)
 names=(['terminal3-integral1_repaired_seed0','terminal3-integral1_v1-none_seed0','terminal3-integral1_v1-balanced_seed0']
        if 'conditional' in key else ['terminal3-integral1_repaired_seed0'])
 for name in names:
  path=ROOT/'comparison/accuracy'/key/(name+'.json');result=read(path)
  assert result['identity']['source_hash']==source_hash()
  assert sha(result['arrays']['path'])==result['arrays']['sha256']
  stored=np.load(result['arrays']['path']);model=load_declared_model(result['method'])
  if 'conditional' in key:
   pred=model.predict_conditional(initial[:1],cfg.query_times);cm,cc=pred.mean[0],pred.scaled_covariance[0]
   bcm,bcc=cm[None],cc[None]
  else:
   mus=[];covs=[]
   for offset in range(0,len(initial),1024):
    pred=model.predict_conditional(initial[offset:offset+1024],cfg.query_times)
    mus.append(pred.mean);covs.append(pred.scaled_covariance)
   mu,cov=np.concatenate(mus),np.concatenate(covs)
   cm,cc=mixture(mu,cov);bcm,bcc=mixture(mu,cov,counts)
   np.testing.assert_allclose(bcm,stored['bootstrap_mean'],atol=3e-13,rtol=0)
   np.testing.assert_allclose(bcc,stored['bootstrap_scaled_covariance'],atol=3e-13,rtol=0)
  pred_err=max(float(abs(cm-stored['mean']).max()),float(abs(cc-stored['scaled_covariance']).max()))
  assert pred_err<3e-13
  metric_err=metric_check(result['metrics'],ref,cm,cc,bcm,bcc,counts)
  checks.append({'group':key,'method':name,'strict_restoration':True,'prediction_maximum_error':pred_err,
                 'metric_maximum_error':metric_err,'bootstrap_seed':seed,'bootstrap_count':128,
                 'bootstrap_unit':'whole-trajectory-family','counts_sha256':count_sha(counts),
                 'checkpoint_sha256':result['method']['checkpoint_sha256'],'result_sha256':sha(path)})
 if 'conditional' in key:
  path=ROOT/'comparison/accuracy'/key/'path_dt10_seed0.json';result=read(path)
  assert sha(result['arrays']['path'])==result['arrays']['sha256']
  packet=np.load(result['arrays']['path']);outcomes=packet['outcomes_terminal3-integral1']
  cm,c=empirical(outcomes);bcm,bc=empirical(outcomes,counts);cc=cfg.n_particles*c;bcc=cfg.n_particles*bc
  metric_err=metric_check(result['metrics']['terminal3-integral1'],ref,cm,cc,bcm,bcc,counts)
  with h5py.File(ROOT/'reference/reference.h5','r') as f:noise=f[key]['model_increments'][:2]
  runtime=PreparedPathRuntime.from_checkpoint(result['method']['checkpoint_path'],'26ac405')
  runtime.prepare(round(cfg.horizon/cfg.model_step))
  with torch.no_grad():pred=runtime(initial[:2],noise,return_raw=True)[0]
  states,stops=pred['states'].numpy(),pred['stopped_at'].numpy()
  bank=PayoffBank.mixed_terminal_integral(cfg.spec,cfg.horizon)
  times=np.arange(states.shape[1])*cfg.model_step
  raw=bank.evaluate_paths(states,times,stops)[:,np.rint(np.array(cfg.query_times)/cfg.model_step).astype(int)]
  err=float(abs(raw-outcomes[:2]).max());assert err<3e-13
  checks.append({'group':key,'method':'path_dt10_seed0','strict_restoration':True,
                 'selected_path_count':2,'prediction_maximum_error':err,'metric_maximum_error':metric_err,
                 'bootstrap_seed':seed,'bootstrap_count':128,'counts_sha256':count_sha(counts),'result_sha256':sha(path)})

result={'kind':'independent-payoff-repair-selected-accuracy-audit-v1','complete':True,
        'started_utc':begin,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-started,'source_hash':source_hash(),'audit_script_sha256':sha(__file__),
        'checks':checks,'source_or_frozen_helpers_modified':False,'refitting_performed':False,
        'optimizer_seed_selection_using_test':False,
        'scope':'Selected exact predictions and all corresponding mean/covariance/128-family-bootstrap metrics; no population accuracy certificate.'}
dest=ROOT/'audit/selected_accuracy_replay.json';tmp=dest.with_suffix('.json.tmp')
tmp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(dest)
print(json.dumps({'complete':True,'path':str(dest),'sha256':sha(dest),'elapsed_seconds':result['elapsed_seconds'],
                  'checks':len(checks),'maximum_prediction_error':max(v['prediction_maximum_error'] for v in checks),
                  'maximum_metric_error':max(v['metric_maximum_error'] for v in checks)},indent=2))
