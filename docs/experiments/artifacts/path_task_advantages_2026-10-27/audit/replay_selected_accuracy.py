import json,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from analysis.path_generalization_reference import ReferenceConfig
from analysis.native_gaussian_baseline import linear_gaussian
from nn_dk.path_runtime import PreparedPathRuntime
from nn_dk.payoff_learning import load_payoff_model
r=Path('runs/path-generalization-2026-10-26');cfg=ReferenceConfig();key='uniform__conditional__0';directory=r/'evaluation'/key;N=cfg.n_particles
with h5py.File(r/'reference/reference.h5','r')as f:
 g=f[key];ref={k:g[k][:]for k in ('bank_b','bank_c')};initial=g['initial'][:1];noise=g['model_increments'][:1]
with np.load(directory/'reference_bootstrap.npz')as f:counts=f['counts'].astype(float)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def moments(y):
 y=np.asarray(y);mean=y.mean(0);z=y-mean;cov=np.einsum('nti,ntj->tij',z,z)/(len(y)-1);d=(counts@z.reshape(len(y),-1)).reshape(len(counts),*y.shape[1:])/len(y);prod=np.einsum('nti,ntj->ntij',z,z);bc=((counts@prod.reshape(len(y),-1)).reshape(len(counts),*prod.shape[1:])-len(y)*np.einsum('bti,btj->btij',d,d))/(len(y)-1);return mean,cov,mean+d,bc
reference={name:moments(y)for name,y in ref.items()}
def compare(pred,record,keyname,candidate=None):
 rm,rc,brm,brc=reference[keyname];std=np.sqrt(np.diagonal(rc,axis1=-2,axis2=-1));scale=std[:,:,None]*std[:,None,:]
 if candidate is not None:cm,cc,bcm,bcc=moments(candidate)
 else:cm,scaled=pred;cc=scaled/N;bcm=cm[None];bcc=cc[None]
 dm=(cm-rm)/std;dc=(cc-rc)/scale;bdm=(bcm-brm)/std;bdc=(bcc-brc)/scale
 actual=[]
 for t,row in enumerate(record['records']):
  values={'max_mean_error_in_reference_std':float(np.abs(dm[t]).max()),'max_normalized_covariance_error':float(np.abs(dc[t]).max()),'mean_sampling95_simultaneous_radius':float(np.quantile(np.abs(bdm[:,t]-dm[t]).max(-1),.95)),'covariance_sampling95_simultaneous_radius':float(np.quantile(np.abs(bdc[:,t]-dc[t]).max((-2,-1)),.95))}
  err=max(abs(values[k]-row[k])for k in values);assert err<2e-12,(keyname,t,err);actual.append({'time_index':t,'manual_metric_values':values,'maximum_metric_replay_difference':err})
 return actual
results=[]
name='path_dt10_seed0';packet=read(directory/(name+'.json'));method=packet['method'];runtime=PreparedPathRuntime.from_checkpoint(method['checkpoint_path'],'26ac405',mode='dense-cached');runtime.prepare(64)
with torch.no_grad():replayed,_=runtime(initial,noise,return_raw=True)
states=replayed['states'].numpy();centers=(np.arange(cfg.n_cells)+.5)*cfg.spec.h;qi=np.rint(np.array(cfg.query_times)/method['dt']).astype(int)
def weight(mode,fun):return np.sinc(mode/cfg.n_cells)*fun(mode*centers)
wb=np.stack([weight(3,np.sin),weight(3,np.cos),weight(4,np.sin),weight(4,np.cos)])
b=states[:,qi]@wb.T;wc=np.stack([weight(1,np.sin),weight(1,np.cos)])/cfg.horizon;vals=states@wc.T;trap=np.concatenate([np.zeros((1,1,2)),np.cumsum(.5*(vals[:,1:]+vals[:,:-1])*method['dt'],axis=1)],axis=1);c=np.concatenate([b[:,:,:2],trap[:,qi]],axis=-1)
with h5py.File(directory/(name+'.h5'),'r')as f:
 candidate={k:f[k][:]for k in ('bank_b','bank_c')};assert np.allclose(b,candidate['bank_b'][:1],rtol=0,atol=2e-13);assert np.allclose(c,candidate['bank_c'][:1],rtol=0,atol=2e-13)
 errors={bank:compare(None,packet['metrics'][label]['same_time'],bank,candidate=candidate[bank])for bank,label in [('bank_b','terminal-modes34'),('bank_c','terminal3-integral1')]}
results.append({'method':name,'json_sha256':sha(directory/(name+'.json')),'independent_single_path_native_weight_and_integral_reduction_replay':True,'full_pool_manual_moment_and_bootstrap_replay':errors})
print(name,'PASS',flush=True)
name='statistics_terminal3-integral1_cold_seed1';packet=read(directory/(name+'.json'));model=load_payoff_model(packet['method']['checkpoint_path']);prediction=model.predict_conditional(initial,cfg.query_times)
with h5py.File(directory/(name+'.h5'),'r')as f:
 assert np.array_equal(prediction.mean,f['conditional_mean'][:]);assert np.array_equal(prediction.scaled_covariance,f['conditional_scaled_covariance'][:])
values=compare((prediction.mean[0],prediction.scaled_covariance[0]),packet['metrics']['same_time'],'bank_c')
results.append({'method':name,'json_sha256':sha(directory/(name+'.json')),'checkpoint_prediction_bitwise_equal':True,'manual_moment_and_bootstrap_replay':values});print(name,'PASS',flush=True)
name='linear_gaussian_rk4_7';packet=read(directory/(name+'.json'));runweights=np.stack([weight(1,np.sin),weight(1,np.cos)])/cfg.horizon
lga=linear_gaussian(initial[0],cfg.context('uniform'),cfg.spec,cfg.common_times,2**-7,terminal_weights=np.eye(cfg.n_cells),running_weights=runweights,return_joint=True);lqi=np.rint(np.array(cfg.query_times)/cfg.common_step).astype(int);records={}
for bank,label in [('bank_b','terminal-modes34'),('bank_c','terminal3-integral1')]:
 projection=np.zeros((4,cfg.n_cells+2))
 if bank=='bank_b':projection[:,:cfg.n_cells]=wb
 else:projection[:2,:cfg.n_cells]=wb[:2];projection[2,cfg.n_cells]=1;projection[3,cfg.n_cells+1]=1
 mean=lga['mean_augmented'][lqi]@projection.T;cov=np.einsum('ai,tij,bj->tab',projection,lga['scaled_covariance_augmented'][lqi],projection)
 records[bank]=compare((mean,cov),packet['metrics'][label]['same_time'],bank)
results.append({'method':name,'json_sha256':sha(directory/(name+'.json')),'known_coefficient_Gaussian_reintegration':True,'manual_moment_and_bootstrap_replay':records});print(name,'PASS',flush=True)
output={'kind':'independent-broad-task-selected-accuracy-replay-v1','complete':True,'group':key,'reference_sha256':read(r/'reference/manifest.json')['reference_sha256'],'paths_per_group':cfg.paths_per_group,'whole_family_bootstrap_count':len(counts),'records':results,'maximum_allowed_manual_metric_replay_difference':2e-12,'scope':'Three preselected solver records, all3querytimes and all16384families. No seed selection or all-groups certification; rarestopping and continuum claims remain unsupported.'}
out=r/'audit/selected_accuracy_replay.json';data=(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
if out.exists()and out.read_bytes()!=data:raise ValueError('Immutable audit changed')
out.write_bytes(data);print(out,'SHA',hashlib.sha256(data).hexdigest(),flush=True)
