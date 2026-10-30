"""Coordinate the frozen full allocation; never change or retry numerical work."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

repo = Path('/Users/xiaohao/Projects/Finance/Covariance/NN-DK')
record_root = repo/'runs/readme-research-completion-2026-10-30'
collocation = repo/'runs/payoff-collocation-study-2026-10-29'
candidate = repo/'runs/lna-payoff-correction-2026-10-30'
receipt = record_root/'actual_final_stages.json'
if receipt.exists():
    raise SystemExit('Existing driver receipt: inspect its live owner and completed work before any continuation.')
sys.path[:0] = [str(repo), str(repo/'src')]
from analysis import evaluate_lna_payoff_correction as evaluator
from analysis import evaluate_payoff_collocation_study as comparison
from analysis import report_lna_payoff_correction as report
from analysis import report_payoff_collocation_study as old_report

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

reference_proof = record_root/'reference_integrity.json'
assert json.loads(reference_proof.read_text())['complete']
freeze_proof = record_root/'actual_joint_freeze_verified.json'
assert json.loads(freeze_proof.read_text())['complete']
for directory in (collocation, candidate):
    proof, _ = report.verify_final_tests(directory)
    assert proof['before_sha256'] == '1a5aa6c50e91d32795e2495e5286590194fdc70fda3c5bc5e84125259efa0a24'
context = evaluator.verify_joint_lock(candidate)
assert sha(candidate/'evaluation/joint_lock.json') == '26f5c64e6c62c2bba7cac735b7232bdbcc7f9144d196e220b0f25629ea1cacad'

value = {'kind':'actual-frozen-joint-final-stages-v1','complete':False,
    'scientific_allocation':True,'active_readme_goal_complete':False,
    'driver_pid':os.getpid(),'started_at_utc':now(),'phase':'waiting_for_accuracy',
    'reference_integrity_sha256':sha(reference_proof),'freeze_verification_sha256':sha(freeze_proof),
    'source_commit':'011277684b540860379eb726b2440a3fd0385226',
    'automatic_retry_permitted':False,'commands':[],
    'accuracy_workers':[{'pid':99319,'script':'analysis/evaluate_payoff_collocation_study.py'},
                        {'pid':99318,'script':'analysis/evaluate_lna_payoff_correction.py'}]}

def save():
    value['updated_at_utc'] = now()
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(receipt)

def pid_command(pid):
    result = subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode == 0 else ''

def assert_quiet():
    snapshot = subprocess.run(['ps','-axo','pid,ppid,etime,time,%cpu,command'],
        check=True,capture_output=True,text=True).stdout
    forbidden = ('analysis/run_payoff_collocation_study.py','analysis/run_lna_payoff_correction_study.py',
        'analysis/evaluate_payoff_collocation_study.py','analysis/evaluate_lna_payoff_correction.py',
        'analysis/audit_payoff_collocation_residuals.py','analysis/audit_payoff_mean_gradients.py',
        'analysis/audit_payoff_residual_transport.py','/bin/pytest',' -m pytest')
    active = [line for line in snapshot.splitlines() if any(name in line for name in forbidden)]
    if active:
        raise RuntimeError('Heavy worker still present before isolated timing: '+str(active))
    path = record_root/f'quiet_before_{len(value["commands"]):02d}.txt'
    path.write_text(snapshot)
    value.setdefault('quiet_attestations',[]).append({'at_utc':now(),'snapshot':str(path),
        'snapshot_sha256':sha(path),'known_study_workers_exited':True,
        'external_competing_heavy_work_finished_attested':True})
    save()

env = dict(os.environ,MPLCONFIGDIR=str(record_root/'.mplconfig'))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    env[name] = '1'

def run(stage,arguments,quiet=False):
    if quiet:
        assert_quiet()
    value['phase'] = stage
    log = record_root/(stage+'.log')
    if log.exists():
        raise RuntimeError('Preserve existing stage log before a deliberate continuation: '+str(log))
    argv = [str(repo/'.venv/bin/python'),'-u',str(repo/'analysis'/arguments[0]),*arguments[1:]]
    attempt = {'stage':stage,'argv':argv,'complete':False,'started_at_utc':now(),'log':str(log)}
    value['commands'].append(attempt)
    save()
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log.open('wb') as sink:
        child = subprocess.Popen(argv,cwd=repo,env=env,stdout=sink,stderr=subprocess.STDOUT)
        attempt['pid'] = child.pid
        value['active_child_pid'] = child.pid
        save()
        print('START',stage,'pid',child.pid,flush=True)
        status = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt.update(complete=True,exit_code=status,finished_at_utc=now(),
        outer_wall_seconds=time.monotonic()-start,
        outer_child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        outer_cost_scope='whole subprocess including imports; nonadditive with nested CLI/workload receipts',
        log_sha256=sha(log))
    value['active_child_pid'] = None
    save()
    print('FINISH',stage,'status',status,flush=True)
    if status:
        raise RuntimeError('Stage failed; no automatic numerical retry: '+stage)

started = time.monotonic()
save()
try:
    while True:
        live = []
        for worker in value['accuracy_workers']:
            command = pid_command(worker['pid'])
            worker['same_process_live'] = worker['script'] in command
            if worker['same_process_live']:
                live.append(worker['pid'])
        if not live:
            break
        save()
        time.sleep(30)
    value['accuracy_processes_exited_at_utc'] = now()
    baseline = comparison.primary_records(collocation,require_timing=False)
    candidate_evidence = evaluator.load_complete(candidate,require_timing=False,require_contrasts=False)
    assert len(baseline['records']) == 576 and len(baseline['paired_initial_records']) == 156
    assert len(candidate_evidence['records']) == 216
    cli = list((candidate/'evaluation/execution').glob('cli_evaluate_99318_*.json'))
    assert len(cli) == 1
    accuracy_cli = json.loads(cli[0].read_text())
    assert accuracy_cli['status'] == 'success' and accuracy_cli['exit_code'] == 0
    value['accuracy_verified'] = {'collocation_primary_records':576,'collocation_retained_records':732,
        'new_primary_records':216,'combined_primary_records':792,'combined_retained_records':948,
        'candidate_cli_receipt':str(cli[0]),'candidate_cli_receipt_sha256':sha(cli[0])}
    save()
    run('actual_contrasts',['evaluate_lna_payoff_correction.py','contrasts','--root',str(candidate)])
    run('actual_collocation_benchmark',['evaluate_payoff_collocation_study.py','benchmark',
        '--root',str(collocation),'--final-timing'],quiet=True)
    run('actual_correction_benchmark',['evaluate_lna_payoff_correction.py','benchmark',
        '--root',str(candidate),'--final-timing'],quiet=True)
    run('actual_joint_verify',['evaluate_lna_payoff_correction.py','verify','--root',str(candidate)])
    run('actual_collocation_report',['report_payoff_collocation_study.py','--root',str(collocation),
        '--documentation',str(repo/'docs/experiments/payoff_collocation_2026-10-30.md')])
    run('actual_correction_report',['report_lna_payoff_correction.py','report','--root',str(candidate),
        '--documentation',str(repo/'docs/experiments/lna_payoff_correction_2026-10-30.md')])
    old_archive = repo/'docs/experiments/artifacts/payoff_collocation_2026-10-30'
    new_archive = repo/'docs/experiments/artifacts/lna_payoff_correction_2026-10-30'
    run('actual_collocation_archive',['report_payoff_collocation_study.py','--root',str(collocation),
        '--archive',str(old_archive)])
    run('actual_collocation_archive_verify',['report_payoff_collocation_study.py','--verify-archive',str(old_archive)])
    run('actual_correction_archive',['report_lna_payoff_correction.py','archive','--root',str(candidate),
        '--output',str(new_archive)])
    run('actual_correction_archive_verify',['report_lna_payoff_correction.py','verify-archive',
        '--output',str(new_archive)])
    evidence = evaluator.load_complete(candidate)
    assert len(evidence['baseline']['timing']['records']) == 696
    assert len(evidence['baseline']['timing']['loading']) == 156
    assert len(evidence['timing']['jobs']) == 288
    assert len(evidence['contrasts']['records']) == 1080
    value.update(complete=True,phase='computational_reports_and_archives_complete',
        allocation=evidence['allocation'],collocation_archive=old_report.verify_archive(old_archive),
        correction_archive=report.verify_archive(new_archive),finished_at_utc=now(),
        driver_wall_seconds=time.monotonic()-started,
        remaining='Owner review of outcomes, final README documentation, composite R1-R6 completion and publication')
    save()
    print('ALL FROZEN FINAL COMPUTATIONAL STAGES COMPLETE',flush=True)
except BaseException as error:
    value.update(error=type(error).__name__+': '+str(error),driver_stopped_at_utc=now())
    save()
    raise
