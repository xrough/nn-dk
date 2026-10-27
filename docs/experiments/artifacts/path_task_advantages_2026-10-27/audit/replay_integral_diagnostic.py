import hashlib,json,math
from pathlib import Path
from types import MethodType
import h5py,numpy as np,torch
from nn_dk.generator_jets import GeneratorJet
from nn_dk.native import native_coefficients
from nn_dk.native_types import NativeDKSpec
from nn_dk.observations import incidence_matrix
from nn_dk.payoff_learning import PayoffBank,PayoffMomentModel,load_payoff_model
from nn_dk.types import Context
r=Path('runs/path-generalization-2026-10-26');spec=NativeDKSpec(32);T=1/16;N=2048
context=Context(kappa=0.,a2=0.,n_particles=N);bank=PayoffBank.mixed_terminal_integral(spec,T);model=PayoffMomentModel(spec,context,T,bank,32)
centers=(np.arange(32)+.5)*spec.h
raw=1+.12*np.cos(centers)+.04*np.sin(2*centers);m=torch.tensor((raw/raw.sum())[None],dtype=torch.float64);tau=torch.tensor([.027],dtype=torch.float64)
# Plug independent exact interior free-heat jets into the ACTUAL residual source
# operation. This changes only this disposable audit object, never a checkpoint.
def exact_heat_jets(self,initial,horizons):
 initial=initial.detach().clone().requires_grad_(True);horizons=horizons.detach().clone().requires_grad_(True)
 ph=torch.tensor(bank.terminal_weights.copy()/bank.scales[:,None]);q=torch.tensor(bank.running_weights.copy()/bank.scales[:,None]);lam=torch.tensor(4*context.diffusion/spec.h**2*np.sin(np.pi*np.array([3,3,1,1])/spec.n_cells)**2)
 decay=torch.exp(-horizons[:,None]*lam);integral=-torch.expm1(-horizons[:,None]*lam)/lam
 weights=torch.where(torch.tensor(bank.integral_rows)[None,:,None],integral[:,:,None]*q[None],decay[:,:,None]*ph[None])
 value=math.sqrt(N)*torch.einsum('bi,boi->bo',initial,weights)
 dt=[];gen=[];faces=[]
 for b in range(len(initial)):
  coefficients=native_coefficients(initial[b].detach().numpy(),context,spec);drift=torch.tensor(coefficients.drift);B=torch.tensor(incidence_matrix(spec.n_cells)*coefficients.face_amplitudes[None,:]/math.sqrt(N))
  rt=[];rg=[];rf=[]
  for i in range(4):
   time=torch.autograd.grad(value[b,i],horizons,retain_graph=True)[0][b];gradient=torch.autograd.grad(value[b,i],initial,retain_graph=True)[0][b]
   rt.append(time);rg.append(gradient@drift);rf.append(gradient@B)
  dt.append(torch.stack(rt));gen.append(torch.stack(rg));faces.append(torch.stack(rf))
 z=value.new_zeros((len(initial),4,4));zf=value.new_zeros((len(initial),4,4,spec.n_cells))
 return GeneratorJet(value,torch.stack(dt),torch.stack(gen),torch.stack(faces)),GeneratorJet(z,z,z,zf)
model._moment_jets=MethodType(exact_heat_jets,model)
mean_residual,cov_residual=model.residual(m,tau);assert float(mean_residual.abs().max())<2e-11
analytic={'native_eigenvalues':[float(4/spec.h**2*np.sin(np.pi*k/spec.n_cells)**2)for k in (3,1)],'exact_free_heat_mean_residual_maximum':float(mean_residual.abs().max()),'mean_solution':'terminal: exp(-lambda*tau)*phi.m; integral: ((1-exp(-lambda*tau))/lambda)*q.m','generator_source':'sqrt(N)*(q/observable_scale).m subtracted once, q already includes1/T','stopping_scope':'Interior analytic control only; free-heat formula does not solve an absorbing-boundary problem.'}
# Independent physical covariance-leading check in public N-scaled output units.
model=PayoffMomentModel(spec,context,T,bank,32);uniform=np.full(spec.n_cells,1/spec.n_cells);qv=native_coefficients(uniform,context,spec).covariance;leading=bank.running_weights@qv@bank.running_weights.T/3;small=[]
for time in (1e-6,2e-6,1e-5):
 prediction=model.predict_conditional(uniform,[time]);C=prediction.scaled_covariance[0,0];expected=time**3*leading;error=float(np.max(np.abs(C[2:,2:]-expected[2:,2:])));relative=float(np.max(np.abs(C[2:,2:]-expected[2:,2:]))/np.max(np.abs(expected[2:,2:])));assert relative<2e-12
 small.append({'time':time,'maximum_absolute_error_integral_block':error,'maximum_relative_error_integral_block':relative,'predicted_integral_scaled_variance_diagonal':np.diag(C)[2:].tolist(),'leading_integral_scaled_variance_diagonal':np.diag(expected)[2:].tolist()})
# Frozen mixed-head fit histories and actual conditional output collapse.
fit_rows=[];conditional_keys=('uniform__conditional__0','uniform__conditional__1','nonuniform__conditional__0','nonuniform__conditional__1')
for p in sorted((r/'payoffs/checkpoints/terminal3-integral1').rglob('*.pt')):
 saved=torch.load(p,weights_only=True,map_location='cpu');metadata=json.loads(p.with_suffix('.json').read_text());loaded=load_payoff_model(p);fitted=[]
 for key in conditional_keys:
  directory=r/'evaluation'/key;name=f'statistics_terminal3-integral1_{p.parent.name}_seed{saved["provenance"]["seed"]}';result=json.loads((directory/(name+'.json')).read_text())
  ratios=[];means=[]
  for row in result['metrics']['same_time']['records']:
   ref=np.array(row['reference_scaled_covariance']);actual=np.array(row['candidate_scaled_covariance']);ratios.append((np.diag(actual)[2:]/np.diag(ref)[2:]).tolist());means.append(row['normalized_mean_difference'][2:])
  fitted.append({'group':key,'integral_variance_candidate_over_reference_by_time':ratios,'integral_mean_error_in_reference_std_by_time':means})
 history=[{'step':0,'validation':saved['initial_validation']}]+[{'step':x['step'],'phase':x['phase'],'validation':x['validation']}for x in saved['history']if x['validation']is not None]
 fit_rows.append({'initialization':p.parent.name,'seed':saved['provenance']['seed'],'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'best_step':saved['best_step'],'complete_steps':saved['adam_steps_done']+saved['lbfgs_steps_done'],'initial_validation':saved['initial_validation'],'best_validation':saved['best_validation'],'loss_scaling':saved['provenance']['loss_scaling'],'validation_history':history,'conditional_outputs':fitted})
packet={'kind':'frozen-pure-integral-moment-head-diagnostic-v1','complete':True,'analytic_free_heat_check':analytic,'small_time_N_scaled_integral_covariance_check':small,'source_review':{'mean_internal_units':'sqrt(N)*physical_mean/observable_scale','public_mean_unscales_once':True,'public_covariance_is_N_scaled_and_unscales_once':True,'mean_running_source_has_no_extra_tau_factor':True,'pure_integral_covariance_has_tau_over_T_on_each_index_once':True,'boundary_values':'terminal-frozen, integrals-killed; exact hard zero atinitialtau and absorbedstate','no_identified_basis_source_sign_or_time_factor_mismatch':True},'initial_ansatz_limitation':'Pure-integral mean starts at zero with a free tau*network correction, whereas the covariance head is initialized at the known tau^3 leading variance. Correct mean derivative at tau=0 is not hard-coded. This can affect joint training; it is not an identified equation or unit mismatch.','training_diagnosis':'All six fits use the unbalanced summed residual objective, select the last permitted step, strongly lower mean residual while increasing covariance residual. Lower total validation objective is not evidence of accurate learned moments. These experiments establish failure of this trained configuration under its fixed budget, not inferiority of statistical learning in general.','mixed_fit_records':fit_rows,'post_test_refitting':False}
out=r/'audit/integral_head_diagnostic.json';data=(json.dumps(packet,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
if out.exists()and out.read_bytes()!=data:raise ValueError('Immutable diagnostic changed')
out.write_bytes(data);print('freeheat maxmeanresidual',analytic['exact_free_heat_mean_residual_maximum']);print('smalltime maxrelative',max(x['maximum_relative_error_integral_block']for x in small));print(out,'SHA',hashlib.sha256(data).hexdigest())
