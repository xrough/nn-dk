#!/usr/bin/env python3
"""Collect terminal actual-study evidence; never run, repair or retry study work.

Usage: /Users/xiaohao/Projects/Finance/Covariance/NN-DK/.venv/bin/python
       /private/tmp/nn_dk_collect_readme_final_evidence.py

Exit 75: final-stage receipt is not terminal. Exit 1: evidence check failed.
An existing output is never overwritten. No project import occurs before the
terminal-driver gate. This external operator script is not a scientific helper.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

REPO = Path('/Users/xiaohao/Projects/Finance/Covariance/NN-DK')
COMPLETION = REPO / 'runs/readme-research-completion-2026-10-30'
C = REPO / 'runs/payoff-collocation-study-2026-10-29'
A = REPO / 'runs/lna-payoff-correction-2026-10-30'
R = REPO / 'runs/payoff-collocation-residuals-2026-10-30'
DRIVER = Path('/private/tmp/nn_dk_actual_final_stages.py')
DRIVER_SHA256 = 'cea8f424d0563e3d8349d219fdc1ae40a7e33508cf2d0aa2e83355a6efb1ea0a'
RECEIPT = COMPLETION / 'actual_final_stages.json'
OUTPUT = COMPLETION / 'final_computational_evidence.json'
ARTIFACTS = REPO / 'docs/experiments/artifacts'
ARCHIVES = {
    'collocation': ARTIFACTS / 'payoff_collocation_2026-10-30',
    'correction': ARTIFACTS / 'lna_payoff_correction_2026-10-30',
    'residual_supplement': ARTIFACTS / 'payoff_collocation_residuals_2026-10-30',
    'historical_diagnostics': ARTIFACTS / 'payoff_diagnostics_2026-10-29',
}


class NotReady(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_stage_plan(driver_bytes):
    """Derive ordered calls and complete argv from the reviewed driver AST."""
    bindings = {'repo': REPO, 'collocation': C, 'candidate': A,
                'old_archive': ARCHIVES['collocation'],
                'new_archive': ARCHIVES['correction']}

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(literal(node.left)) / literal(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'str' and len(node.args) == 1 and not node.keywords):
            return str(literal(node.args[0]))
        raise ValueError('Unreviewed dynamic expression in driver stage arguments')

    calls = sorted((n for n in ast.walk(ast.parse(driver_bytes))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'run'), key=lambda n: n.lineno)
    result = []
    for call in calls:
        require(len(call.args) == 2 and isinstance(call.args[1], ast.List),
                'Driver run signature changed')
        stage = literal(call.args[0])
        arguments = [literal(n) for n in call.args[1].elts]
        require(arguments and all(isinstance(v, str) for v in arguments),
                'Driver arguments must resolve to the reviewed actual paths')
        result.append({'stage': stage,
                       'argv': [str(REPO / '.venv/bin/python'), '-u',
                                str(REPO / 'analysis' / arguments[0]), *arguments[1:]]})
    require(result and len({r['stage'] for r in result}) == len(result),
            'Missing or duplicated driver stages')
    return result


def terminal_gate():
    """First runtime evidence gate: metadata only, before any project import."""
    if not RECEIPT.is_file():
        raise NotReady('actual_final_stages.json does not exist')
    receipt = read(RECEIPT)
    if receipt.get('complete') is not True:
        raise NotReady('Actual final stages are still incomplete; no readers were imported')
    driver_bytes = DRIVER.read_bytes()
    require(hashlib.sha256(driver_bytes).hexdigest() == DRIVER_SHA256,
            'Final-stage driver changed; review this external collector again')
    planned = source_stage_plan(driver_bytes)
    commands = receipt.get('commands', [])
    require(receipt.get('kind') == 'actual-frozen-joint-final-stages-v1'
            and receipt.get('scientific_allocation') is True
            and receipt.get('active_readme_goal_complete') is False
            and receipt.get('phase') == 'computational_reports_and_archives_complete'
            and receipt.get('active_child_pid') is None and not receipt.get('error'),
            'Receipt is not an unambiguous terminal actual-study success')
    require(len(commands) == len(planned),
            f'Expected all {len(planned)} source-declared driver commands')
    for command, expected in zip(commands, planned, strict=True):
        require(command.get('stage') == expected['stage'] and command.get('argv') == expected['argv'],
                'Driver stage order or actual path/arguments changed')
        require(command.get('complete') is True and type(command.get('exit_code')) is int
                and command['exit_code'] == 0, 'Every final-stage command must finish successfully')
        log = Path(command['log'])
        require(log.resolve().is_relative_to(COMPLETION) and log.is_file()
                and sha(log) == command.get('log_sha256'), 'Terminal command log missing or changed')
        require(all(isinstance(command.get(k), (int, float)) and not isinstance(command[k], bool)
                    and math.isfinite(command[k]) and command[k] >= 0
                    for k in ('outer_wall_seconds', 'outer_child_cpu_seconds')),
                'Terminal command cost scope is missing')
    require(not OUTPUT.exists(), 'Preserve the existing final_computational_evidence.json before rerunning')
    return receipt, planned, sha(RECEIPT)


def normalize(value):
    return json.loads(json.dumps(value))


def collect(receipt, planned, receipt_sha, started, cpu_started):
    # The terminal metadata gate has passed. Only now import frozen readers.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.chdir(REPO)
    sys.path[:0] = [str(REPO), str(REPO / 'src')]
    import torch
    torch.set_num_threads(1)
    from analysis import evaluate_lna_payoff_correction as ae
    from analysis import evaluate_payoff_collocation_study as ce
    from analysis import report_lna_payoff_correction as ar
    from analysis import report_payoff_collocation_study as cr
    from analysis import report_payoff_collocation_residuals as rr
    from analysis import archive_payoff_diagnostics as hd
    from analysis import run_lna_payoff_correction_study as ra
    from analysis import run_payoff_collocation_study as rc
    from nn_dk.data import source_hash

    draft_path = COMPLETION / 'requirements_draft.json'
    draft = read(draft_path)
    require(draft.get('kind') == 'readme-research-completion-requirements-draft-v1'
            and draft.get('complete') is False, 'Requirements draft changed scope or became completion proof')
    for label, archive_label, root in (('collocation', 'collocation', C),
                                     ('correction', 'lna_correction', A),
                                     ('residual_supplement', 'residual_supplement', R)):
        require(Path(draft['authoritative_roots'][label]).resolve() == root,
                'Actual study roots changed; no alternate/tiny roots are allowed')
        require(Path(draft['destinations']['archives'][archive_label]).resolve() == ARCHIVES[label],
                'Archive destination differs from the declared actual allocation')

    # Missing required paths must fail before a reader can attempt restoration.
    expected_paths = sorted({Path(p) for requirement in draft['requirements']
                             for key in ('expected_paths', 'public_archive_manifests', 'public_final_notes')
                             for p in requirement.get(key, [])})
    require(all(p.exists() for p in expected_paths),
            'A requirements-draft computational/report/archive path is missing')

    a = ae.load_complete(A, require_timing=True, require_contrasts=True)
    c = ce.primary_records(C, require_timing=True)
    require(normalize(asdict(a['plan'])) == normalize(asdict(ae.EvaluationPlan())),
            'A evaluation batch/chunk/seed/PSD/criterion plan differs from the declared source default')
    c_allocation = cr.validate_evidence(c)
    a_allocation = ar.validate_evidence(a)
    require(normalize(c['study'].to_dict()) == normalize(rc.StudyPlan().to_dict()),
            'C is not the complete declared source-default actual plan')
    actual_a, default_a = normalize(a['study'].to_dict()), normalize(ra.StudyPlan().to_dict())
    require(Path(actual_a.pop('collocation_root')).resolve() == C
            and Path(default_a.pop('collocation_root')).resolve() == C and actual_a == default_a,
            'A is not the complete declared source-default actual plan')
    contract = rc.evaluation_contract(C, c['lock'])
    inventory, pools = ae._inventory(a, a['plan'])
    source_expected = {
        'C_fits': len(rc.slots(c['study'])), 'A_fits': len(ra.slots(a['study'])),
        'C_primary': contract['counts']['primary_records'],
        'C_retained': contract['counts']['raw_retained_records'],
        'C_independent_replacements': contract['counts']['independent_IID_replacement_records'],
        'A_primary': len(inventory), 'A_timing_jobs': len(ae._timing_slots(a)),
    }
    expected_c_times = expected_c_restores = 0
    expected_contrasts = 0
    for case in c['plan']['cases']:
        cfg = rc.reference_configuration(case, c['study'])
        heads = cr._names(c, case) - {'LGA_dt7', 'LGA_dt10', 'native-MC-dt10'}
        expected_c_restores += len(cfg.cases) * len(heads)
        expected_c_times += len(cfg.cases) * (len(heads) * (len(c['plan']['neural_conditional_batches']) + 1)
                            + len(c['plan']['LGA_max_steps']) * 2 + 2)
        for model in (m for m in a['candidate_lock']['models'] if m['case'] == case):
            controls = set(ae.CONTROL_NAMES) | {'LGA_dt7', 'LGA_dt10', 'native-MC-dt10'}
            controls |= {m['name'] for m in c['lock']['models']
                         if m['case'] == case and m['seed'] == model['seed']}
            expected_contrasts += len(cfg.group_keys) * len(controls)
    source_expected.update(C_timing_records=expected_c_times, C_restorations=expected_c_restores,
                           contrasts=expected_contrasts)
    fixed = {'C_fits': 72, 'A_fits': 18, 'C_primary': 576, 'C_retained': 732,
             'C_independent_replacements': 156, 'A_primary': 216, 'A_timing_jobs': 288,
             'C_timing_records': 696, 'C_restorations': 156, 'contrasts': 1080}
    require(source_expected == fixed, 'Source-defined expected actual allocation differs from the declared release')
    actual = {'C_fits': len(c['lock']['models']), 'A_fits': len(a['candidate_lock']['models']),
              'C_primary': len(c['records']), 'C_retained': c['original_accuracy_records'] + c['independent_initial_records'],
              'C_independent_replacements': c['independent_initial_records'],
              'A_primary': len(a['records']), 'A_timing_jobs': len(a['timing']['jobs']),
              'C_timing_records': len(c['timing']['records']), 'C_restorations': len(c['timing']['loading']),
              'contrasts': len(a['contrasts']['records'])}
    require(actual == source_expected and actual['C_fits'] + actual['A_fits'] == 90,
            'Actual complete inventories differ from source-defined expectations')
    actual['joint_primary'] = actual['C_primary'] + actual['A_primary']
    actual['joint_retained'] = actual['C_retained'] + actual['A_primary']
    require(actual['joint_primary'] == 792 and actual['joint_retained'] == 948,
            'Joint primary/retained allocation changed')

    # Both report loaders verify complete hashes/output inventories. Rebuild only
    # their input inventory with pure collectors; never call generate/archive.
    a_manifest = ar.load_report(A / 'report')
    c_manifest = cr.load_report(C / 'report')
    r_manifest = rr.load_report(R / 'report')
    cfg = rc.ablation_configuration(rc.slots(c['study'])[0], c['study'])
    strata = len(rr.stratum_names(cfg.strata))
    component_width = sum(len(terms) * (4 if stage == 'mean' else 16)
                          for stage, terms in rr.TERMS.items())
    expected_residual = {'models': source_expected['C_fits'], 'cases': len(c['plan']['cases']),
        'strata_per_model': strata,
        'component_statistic_rows': source_expected['C_fits'] * (strata + 1) * len(rr.UNITS) * component_width,
        'stratum_loss_rows': source_expected['C_fits'] * (strata + 1) * len(rr.TERMS),
        'all_seeds_and_variants_retained': True}
    require(r_manifest['allocation'] == expected_residual,
            'Residual supplement is not the full source-defined72-model/19-stratum allocation')
    fresh_a_allocation, a_costs, _, _, fresh_a_identity = ar._report_inputs(a)
    require(a_manifest['identity'] == fresh_a_identity and a_manifest['allocation'] == fresh_a_allocation,
            'A report omitted current evidence or paid-work receipts')
    cr._reference_inputs(c)
    cr._data_workers(c)
    c_helpers = {**read(C / 'comparison/evaluation_lock.json')['helpers'],
                 **{str(REPO / 'analysis' / name): sha(REPO / 'analysis' / name)
                    for name in ('run_payoff_collocation_study.py', 'evaluate_payoff_collocation_study.py',
                                 'report_payoff_regimes.py', 'payoff_study_adapter.py')}}
    for path, digest in c_helpers.items():
        cr._artifact(path, c['inputs'], digest)
    current_c_identity = {'kind': cr.KIND, 'root': str(C), 'source_hash': source_hash(),
                          'report_script_sha256': sha(cr.__file__), 'helpers': c_helpers, 'inputs': c['inputs']}
    require(c_manifest['identity'] == current_c_identity and c_manifest['allocation'] == c_allocation,
            'C report omitted current input/cost inventory')
    require(a_manifest['allocation'] == a_allocation, 'A report allocation differs from current strict evidence')
    require(a['plan'].bootstrap_replicates == 128
            and contract['criteria']['bootstrap_replicates'] == 128,
            'Declared whole-family bootstrap allocation changed')
    a_report, c_report = read(A / 'report/report.json'), read(C / 'report/report.json')
    require(a_report['allocation'] == a_allocation and c_report['allocation'] == c_allocation,
            'Saved report allocations differ from their verified manifests')
    signed_expected = sum(2 * len(ra.query_times(row['case'], a['study'])) if row['available'] else 1
                          for row in a['contrasts']['records'])
    require(len(a_report['tables']['signed_contrasts']) == signed_expected
            == a_allocation['contrast_statistic_rows'], 'Conditional signed-row expansion changed')
    # Unavailable predictions and unfavorable/unresolved gates are valid results.
    unavailable = [{'case': row['identity']['case'], 'group': row['identity']['group'],
                    'method': row['method']['name'], 'failed_families': len(row['failed_family_ids'])}
                   for row in a['records'] if not row['available']]

    verified_archives = {
        'collocation': cr.verify_archive(ARCHIVES['collocation']),
        'correction': ar.verify_archive(ARCHIVES['correction']),
        'residual_supplement': rr.verify_archive(ARCHIVES['residual_supplement']),
        'historical_diagnostics': hd.verify_archive(ARCHIVES['historical_diagnostics']),
    }
    require(all(v.get('complete') is True for v in verified_archives.values()), 'An offline archive failed')
    require(verified_archives['collocation']['allocation'] == c_allocation
            and verified_archives['correction']['allocation'] == a_allocation,
            'Archive allocation differs from the strict current reports')
    require(verified_archives['residual_supplement']['allocation'] == r_manifest['allocation'],
            'Residual archive/report allocation differs')
    archive_manifests = {name: read(directory / 'archive_manifest.json')
                         for name, directory in ARCHIVES.items()}
    for name, directory in (('collocation', C), ('correction', A), ('residual_supplement', R)):
        require(archive_manifests[name].get('report_manifest_sha256') == sha(directory / 'report/manifest.json'),
                'Offline-valid archive is stale relative to the current ' + name + ' report')
    require(archive_manifests['collocation'].get('report_identity') == c_manifest['identity'],
            'C archive does not bind the current complete report input/cost identity')

    qa_proofs = {}
    helpers = a_manifest['identity']['helpers']
    for label, directory in (('A', A), ('C', C)):
        proof, logs = ar.verify_final_tests(directory, helpers=helpers)
        require(len(proof['source_inputs']) == 287, 'Maintained QA source inventory changed')
        counts = [row['test_count'] for row in proof['commands'] if row['kind'] == 'pytest']
        require(counts == [1502], 'Current complete full-suite receipt must record all1502 tests')
        qa_proofs[label] = {'receipt': str(directory / 'audit/final_tests.json'),
                            'receipt_sha256': sha(directory / 'audit/final_tests.json'),
                            'source_count': len(proof['source_inputs']), 'source_inputs': proof['source_inputs'],
                            'before_sha256': proof['before_sha256'], 'after_sha256': proof['after_sha256'],
                            'commands': proof['commands'], 'verified_log_paths': [str(p) for p in logs]}
    require(qa_proofs['A']['source_inputs'] == qa_proofs['C']['source_inputs'], 'Both QA source maps must agree')
    for name in ('correction', 'residual_supplement'):
        require(archive_manifests[name].get('test_receipt_sha256') == qa_proofs['A']['receipt_sha256'],
                'Archive contains a different QA receipt from the current source-bound proof')
    c_archived_qa = archive_manifests['collocation']['copied'].get('audit/final_tests.json')
    if c_archived_qa is None:
        c_archived_qa = archive_manifests['collocation']['copied'].get('audit/final_tests.json.gz')
    require(c_archived_qa is not None
            and c_archived_qa.get('original_sha256') == qa_proofs['C']['receipt_sha256'],
            'C archive does not preserve its current source-bound QA receipt')

    freeze_path, reference_path = COMPLETION / 'actual_joint_freeze_verified.json', COMPLETION / 'reference_integrity.json'
    freeze, reference = read(freeze_path), read(reference_path)
    require(freeze.get('complete') is True and freeze.get('joint_lock_precedes_all_final_materialization') is True
            and freeze.get('full_verify_joint_lock_API_completed') is True,
            'Verified pre-draw joint-freeze provenance missing')
    require(reference.get('complete') is True and reference.get('reference_families') == 589824
            and reference.get('paired_refinement_families') == 36864
            and reference.get('all_family_IDs_unique_across_groups_and_cases') is True
            and reference.get('all_case_manifests_and_HDF_checksums_verified') is True,
            'Complete six-case reference integrity provenance missing')
    require(sha(freeze_path) == receipt['freeze_verification_sha256']
            and sha(reference_path) == receipt['reference_integrity_sha256']
            and sha(A / 'evaluation/joint_lock.json') == freeze['joint_lock_sha256'] == a['joint_lock_sha256'],
            'Driver/freeze/reference identities disagree')
    for path, digest in freeze['fit_locks'].items():
        require(sha(path) == digest, 'Pre-draw fit lock changed')
    require({row['case'] for row in reference['cases']} == set(a['cases']), 'Reference case allocation differs')
    for row in reference['cases']:
        directory = C / 'reference' / row['case']
        require(sha(directory / 'manifest.json') == row['manifest_sha256']
                and a['baseline']['references'][row['case']]['reference_sha256'] == row['reference_sha256'],
                'Reference integrity proof differs from the strict final reference')

    inputs = {}
    def add(path, expected=None, role=None):
        path = Path(path).resolve()
        require(path.exists(), 'Required computational evidence path missing: ' + str(path))
        if path.is_dir():
            entry = {'type': 'directory', 'sha256': None}
        else:
            digest = sha(path)
            require(expected is None or digest == expected, 'Evidence changed: ' + str(path))
            entry = {'type': 'file', 'sha256': digest, 'bytes': path.stat().st_size}
        if str(path) in inputs:
            require(all(inputs[str(path)].get(k) == v for k, v in entry.items()), 'Conflicting evidence identities')
        else:
            inputs[str(path)] = entry
        if role:
            inputs[str(path)].setdefault('roles', [])
            if role not in inputs[str(path)]['roles']:
                inputs[str(path)]['roles'].append(role)
    for path in expected_paths:
        add(path, role='requirements-draft-path')
    for label, manifest in (('A', a_manifest), ('C', c_manifest), ('R', r_manifest)):
        for path, digest in manifest['identity']['inputs'].items():
            add(path, digest, 'verified-' + label + '-report-input')
    for proof in qa_proofs.values():
        add(proof['receipt'], proof['receipt_sha256'], 'current-QA-receipt')
        for command in proof['commands']:
            add(command['log_path'], command['log_sha256'], 'current-QA-success-log')
        for relative, digest in proof['source_inputs'].items():
            add(REPO / relative, digest, 'current-tested-maintained-source')
    for path in (DRIVER, Path(__file__), RECEIPT, draft_path, freeze_path, reference_path,
                 COMPLETION / 'production_residual_stage.json', COMPLETION / 'completed_requirements_audit.json',
                 COMPLETION / 'requirements_path_schema_audit.json'):
        add(path, role='cross-stage-provenance')
    for command in receipt['commands']:
        add(command['log'], command['log_sha256'], 'terminal-final-stage-log')
    for directory in (A, C):
        for relative in ('plan.json', 'payoffs/fit_lock.json'):
            add(directory / relative, role='actual-fit-and-plan-identity')
    for name, directory in ARCHIVES.items():
        add(directory / 'archive_manifest.json', role=name + '-offline-archive-manifest')
    for path in a_manifest['identity']['helpers']:
        add(path, a_manifest['identity']['helpers'][path], 'frozen-final-helper')
    manual_document_snapshots = []
    for item in [*draft['final_documentation'], *draft.get('additional_documentation_actions', [])]:
        path = Path(item['path'])
        add(path, role='manual-documentation-snapshot-not-final-signoff')
        manual_document_snapshots.append({'path': str(path), 'sha256': inputs[str(path.resolve())]['sha256'],
                                         'final_owner_review_complete': False})

    # Read statuses from existing reports; never impose a scientific outcome.
    def gates(report):
        rows = report['tables']['amortization']
        return {'rows': len(rows), 'candidate_gate_counts': dict(Counter(str(r.get('candidate_gate')) for r in rows)),
                'control_gate_counts': dict(Counter(str(r.get('control_gate')) for r in rows)),
                'accuracy_qualified_payback_rows': sum(r.get('accuracy_qualified_payback_calls') is not None for r in rows)}
    timing_cost_flags = Counter()
    for entry in a['timing']['jobs']:
        job = read(entry['path'])
        timing_cost_flags[str(job.get('attempt_cost_complete'))] += 1
    cost_summary = {k: v for k, v in a_costs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    cost_summary['evaluation_CLI_stage_coverage'] = a_costs['evaluation_CLI_stage_coverage']
    require(sha(RECEIPT) == receipt_sha and sha(DRIVER) == DRIVER_SHA256,
            'Terminal driver evidence changed during collection')
    return {
        'kind': 'readme-research-terminal-computational-evidence-v1', 'complete': False,
        'computational_evidence_complete': True, 'overall_complete': False, 'active_readme_goal_complete': False,
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Posthoc read-only verification of the fixed actual computational allocation; manual interpretation, final documentation and public-main verification remain separate.',
        'driver_gate': {'receipt': str(RECEIPT), 'receipt_sha256': receipt_sha,
                        'driver': str(DRIVER), 'driver_sha256': DRIVER_SHA256,
                        'source_derived_command_count': len(planned), 'ordered_stage_plan': planned,
                        'all_complete_exit_zero_and_log_hashes_verified': True},
        'source_defined_expected_counts': source_expected, 'actual_counts': actual,
        'strict_API_allocations': {'A': a_allocation, 'C': c_allocation, 'R': r_manifest['allocation']},
        'actual_plan_identities': {'package_source_hash': source_hash(), 'A_plan_hash': a['study'].plan_hash,
                                   'C_plan_hash': c['study'].plan_hash, 'joint_lock_sha256': a['joint_lock_sha256'],
                                   'A_fit_lock_sha256': sha(A / 'payoffs/fit_lock.json'),
                                   'C_fit_lock_sha256': sha(C / 'payoffs/fit_lock.json')},
        'unavailable_A_records': unavailable, 'available_contrast_packets': sum(r['available'] for r in a['contrasts']['records']),
        'unavailable_contrast_packets': sum(not r['available'] for r in a['contrasts']['records']),
        'signed_contrast_statistic_rows': signed_expected,
        'cost_completeness_and_unknowns': {'A_preparation_and_process_summary': cost_summary,
            'A_timing_attempt_cost_complete_counts': dict(timing_cost_flags),
            'C_study_cost_summary': c_report['study_costs'],
            'A_qualification_statuses_from_frozen_logic': gates(a_report),
            'C_qualification_statuses_from_frozen_logic': gates(c_report),
            'qualification_values_are_observed_not_required_outcomes': True,
            'no_new_latency_cost_or_accuracy_claim_created': True},
        'current_source_QA': qa_proofs, 'offline_archive_verification': verified_archives,
        'report_manifest_paths': {'A': str(A / 'report/manifest.json'), 'C': str(C / 'report/manifest.json'),
                                  'R': str(R / 'report/manifest.json')},
        'requirements_draft_path': str(draft_path), 'requirements_draft_is_completion_proof': False,
        'verified_input_paths_and_sha256': inputs,
        'manual_documentation_snapshots_not_final_signoff': manual_document_snapshots,
        'manual_finalization_pending': ['Review scientific findings and unresolved/failed gates without changing frozen criteria',
            'Finalize maintained README/guide interpretations and their current hashes/links/licenses',
            'Finish composite R1-R6 completion and append terminal public cross-stage provenance',
            'Publish reviewed follow-up public main and verify Actions at its exact commit'],
        'limits_retained': draft['limits_to_retain'],
        'new_fits_predictions_simulations_benchmarks_or_tests': False,
        'posthoc_bookkeeping_cost': {'wall_seconds': time.monotonic() - started,
                                    'cpu_seconds': time.process_time() - cpu_started,
                                    'scope': 'This read-only collector only; separate from scientific preparation and inference timings'},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # No alternate study roots, allocations or fallback switches.
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        receipt, planned, receipt_sha = terminal_gate()
        evidence = collect(receipt, planned, receipt_sha, started, cpu_started)
        data = (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        # Publish the complete JSON atomically without replacing prior evidence.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=COMPLETION, prefix='.final-computational-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, OUTPUT)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(json.dumps({'output': str(OUTPUT), 'sha256': sha(OUTPUT),
                          'computational_evidence_complete': True, 'overall_complete': False}))
        return 0
    except NotReady as error:
        print('NOT READY: ' + str(error), file=sys.stderr)
        return 75
    except Exception as error:
        print('EVIDENCE CHECK FAILED: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
