import argparse,datetime,hashlib,json,os,resource,subprocess,sys,time,zipfile,tarfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--branch',default='codex/public-release');ap.add_argument('--root',required=True);ap.add_argument('--record',required=True);ap.add_argument('--reference');ap.add_argument('--recover-clone',action='store_true');a=ap.parse_args()
root=Path(a.root);record=Path(a.record);record.mkdir(parents=True,exist_ok=a.recover_clone)
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r={'kind':'final-research-clean-public-full-history-clone-build-install-smoke-v1','commit':a.commit,'branch':a.branch,'clone':str(root),'started_at_utc':started,'complete':False,'commands':[],'package_registry_publication':False,'scientific_study_allocation':False}
env=dict(os.environ,UV_CACHE_DIR='/private/tmp/nn-dk-release-uv-cache',UV_PYTHON_INSTALL_DIR='/private/tmp/nn-dk-release-uv-python',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
uv='/private/tmp/nn-dk-release-tools-20261026/bin/uv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (record/'verification.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
def run(name,args,cwd=None):
 start=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_CHILDREN);log=record/(name+'.log')
 with log.open('w') as f: proc=subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 row={'argv':list(map(str,args)),'cwd':str(cwd) if cwd else None,'exit_code':proc.returncode,'log_path':str(log),'log_sha256':sha(log),'wall_seconds':time.monotonic()-start,'process_cpu_seconds':after.ru_utime+after.ru_stime-cpu.ru_utime-cpu.ru_stime}
 r['commands'].append(row);save();print(json.dumps({'stage':name,'exit_code':proc.returncode,'wall_seconds':row['wall_seconds']}),flush=True)
 if proc.returncode:raise RuntimeError(name+' failed; see '+str(log))
 return log.read_text()
try:
 clone_args=['git','clone','--branch',a.branch]
 if a.reference:clone_args+=['--reference',a.reference,'--dissociate'];r['local_git_object_cache']=a.reference
 clone_args+=['https://github.com/xrough/nn-dk.git',str(root)]
 if a.recover_clone:
  prior=record/'verification.json';(record/'initial_clone_checkout_failure.json').write_bytes(prior.read_bytes())
  r['clone_recovery']={'prior_receipt':'initial_clone_checkout_failure.json','reason':'Public clone completed and dissociated its full pack; initial checkout returned128, but HEAD became readable immediately. Restore fresh tree and fsck before all verification.'}
  run('restore_initial_checkout',['git','restore','--source=HEAD','--staged','--worktree',':/'],root)
  run('full_object_verification',['git','fsck','--full'],root)
 else:run('public_clone',clone_args)
 assert not (root/'.git/objects/info/alternates').exists()
 r['clone_independent_from_object_cache']=True
 assert run('commit',['git','rev-parse','HEAD'],root).strip()==a.commit
 assert run('history',['git','rev-parse','--is-shallow-repository'],root).strip()=='false'
 r['history_commits']=int(run('history_count',['git','rev-list','--count','HEAD'],root))
 for i,c in enumerate(['f85f53430637e0d391482ad74e4b796bbfe9c67a','011277684b540860379eb726b2440a3fd0385226','43dd46a']):run('ancestor_'+str(i),['git','merge-base','--is-ancestor',c,'HEAD'],root)
 assert not run('clean_before',['git','status','--porcelain'],root).strip()
 r['lock_sha256']=sha(root/'uv.lock')
 run('locked_install',[uv,'sync','--locked','--group','dev','--python','3.12'],root)
 run('build',[uv,'build'],root)
 wheel=root/'dist/nn_dk-0.1.0-py3-none-any.whl';sdist=root/'dist/nn_dk-0.1.0.tar.gz'
 r['build_artifacts']={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [wheel,sdist]}
 with zipfile.ZipFile(wheel) as z:
  names=z.namelist();meta=z.read(next(x for x in names if x.endswith('.dist-info/METADATA'))).decode()
  assert 'License-Expression: MIT AND CC-BY-4.0' in meta
  assert (root/'README.md').read_text().strip() in meta
  for n in ['LICENSE','LICENSE-DOCS','LICENSING.md']: assert any(x.endswith('/licenses/'+n) for x in names)
 with tarfile.open(sdist) as t:
  ns=t.getnames()
  assert not any('/runs/' in x or '/.venv/' in x for x in ns)
  for n in ['LICENSE','LICENSE-DOCS','LICENSING.md','docs/experiments/readme_research_2026-10-30_completion.json']:assert 'nn_dk-0.1.0/'+n in ns
 r['wheel_and_sdist_license_metadata_verified']=True
 run('documented_smoke',[uv,'run','nn-dk','path-solver','--preset','path-smoke','--output','runs/path-smoke'],root)
 report=root/'runs/path-smoke/report/path_results.json';d=json.loads(report.read_text());r['smoke']={'report':str(report),'sha256':sha(report),'models':len(d['comparison_rows']),'same_step_passes':d['same_step_comparisons_passed'],'all_configured_models_reported':d['all_configured_models_reported'],'all_same_step_comparisons_passed':d['all_same_step_comparisons_passed'],'continuous_time_convergence_certified':d['continuous_time_convergence_certified']}
 assert r['smoke']['models']==5 and d['all_configured_models_reported'] and d['all_same_step_comparisons_passed']
 py=str(root/'.venv/bin/python');check='import json,nn_dk;from nn_dk.data import source_hash;print(json.dumps({"path":nn_dk.__file__,"source_hash":source_hash()}))'
 for mode,artifact in [('wheel',wheel),('sdist',sdist)]:
  run(mode+'_install',[uv,'pip','install','--python',py,'--no-deps','--force-reinstall',str(artifact)],root)
  imp=json.loads(run(mode+'_import',[py,'-c',check],record));assert '/site-packages/nn_dk/' in imp['path'];assert imp['source_hash']=='0d9bda04e5da1cc00cc3ec5917d23a4352e6ceb4585dd1a066779b364cf79576';r[mode+'_import']=imp
  run(mode+'_cli',[str(root/'.venv/bin/nn-dk'),'--help'],record)
 assert not run('clean_after',['git','status','--porcelain'],root).strip()
 assert r['lock_sha256']==sha(root/'uv.lock')
 r['complete']=True
except BaseException as e:r['failure']=repr(e);raise
finally:r['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
