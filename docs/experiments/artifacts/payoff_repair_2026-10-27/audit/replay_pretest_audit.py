"""Read-only preparation audit; writes only its own new audit outputs.

Never opens, draws, generates, or evaluates the fresh final-test outcomes.
Historical test files are inspected only for stored family identifiers.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from analysis import run_payoff_repair as runner
from analysis import path_generalization_reference as reference
from nn_dk.data import source_hash
from nn_dk.interacting_data import DATA_KEY_KIND, _family, _seed, collocation_data_id
from nn_dk.native import initialize_masses, simulate_native
from nn_dk.types import seeded_rng, stable_hash

ROOT = Path('runs/payoff-repair-2026-10-27').resolve()
REPO = Path.cwd()
SHA = 'b7876b03f8f4fc6b5b030a3e631ed7404b35529f6ac3b391d4c4c1cdc90a8b15'
FROZEN = '10a898d'

def read(path):
    return json.loads(Path(path).read_text())

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def entropy_id(seed, *keys):
    # Match seeded_rng's entropy digest without instantiating/drawing a stream.
    return hashlib.sha256(json.dumps([seed, *keys], separators=(',', ':')).encode()).hexdigest()

def family_ids(path):
    result = set()
    inventory = {}
    with h5py.File(path, 'r') as store:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.split('/')[-1] in ('family_ids', 'integration_family_ids'):
                values = obj[:].astype(str).reshape(-1)
                unique = {value for value in values if value}
                result.update(unique)
                inventory[name] = {'rows': len(values), 'unique': len(unique), 'sha256': stable_hash(sorted(unique))}
        store.visititems(visit)
    return result, inventory

started = datetime.now(timezone.utc).isoformat()
ref_path = ROOT/'reference/reference.h5'
pretest_snapshot = {'reference_h5_present': ref_path.exists(),
                    'fit_lock_present': (ROOT/'payoffs/fit_lock.json').exists(),
                    'manifest': read(ROOT/'reference/manifest.json')}
assert not pretest_snapshot['reference_h5_present']
assert not pretest_snapshot['fit_lock_present']
assert pretest_snapshot['manifest']['status'] == 'waiting-for-complete-fit-lock'
assert pretest_snapshot['manifest']['committed_chunks'] == 0

cfg, refcfg = runner.configuration(), runner.reference_configuration()
design = runner._assert_frozen(ROOT/'payoffs')
assert source_hash() == SHA == design['source_hash']
rev = subprocess.check_output(['git', 'rev-parse', FROZEN], text=True).strip()
package_files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', rev, 'src/nn_dk'], text=True).splitlines()
package_texts = {str(Path(name).relative_to('src/nn_dk')):
                 subprocess.check_output(['git', 'show', f'{rev}:{name}'], text=True)
                 for name in package_files if name.endswith('.py')}
assert stable_hash(package_texts) == SHA
helper_checks = []
for name, expected in design['helpers'].items():
    path = REPO/'analysis'/name
    actual = file_sha(path)
    assert actual == expected
    helper_checks.append({'path': str(path), 'sha256': actual})
assert file_sha(REPO/'analysis/run_payoff_repair.py') == design['runner_sha256']
comparison = read(ROOT/'comparison/preregistration.json')
assert comparison['source_hash'] == SHA
for name, expected in comparison['helpers'].items():
    assert file_sha(name) == expected
    helper_checks.append({'path': name, 'sha256': expected})

lock = read(ROOT/'payoffs/data_lock.json')
assert lock['source_hash'] == SHA
assert file_sha(ROOT/'payoffs/preregistration.json') == lock['preregistration_sha256']
assert design['slots'] == runner.slots() and len(runner.slots()) == 10
_, arrays = runner._data_identity(ROOT/'payoffs')
ids = {role: set(bank['family_ids']) for role, bank in arrays.items()}
assert len(ids['train']) == 256 and len(ids['validation']) == 128
assert not ids['train'] & ids['validation']
shared_pool = []
for role, bank in arrays.items():
    counts = np.unique(bank['family_ids'], return_counts=True)[1]
    assert np.all(counts == 16)
    for variant in runner.VARIANTS:
        bound = runner._rebound(bank, runner.configuration(variant))
        assert bound['states'] is bank['states'] and bound['horizons'] is bank['horizons']
        assert bound['family_ids'] is bank['family_ids']
        assert collocation_data_id(runner.configuration(variant)) == collocation_data_id(cfg)
        changed = {key for key in cfg.to_dict() if cfg.to_dict()[key] != runner.configuration(variant).to_dict()[key]}
        assert changed <= {'loss_balancing'}
        shared_pool.append({'role': role, 'variant': variant, 'same_array_objects': True,
                            'changed_config_fields': sorted(changed)})
    for key in ('path', 'metadata_path'):
        assert file_sha(lock['banks'][role][key]) == lock['banks'][role]['sha256' if key == 'path' else 'metadata_sha256']

# Independently reconstruct every family ID and residual-horizon stream.
seed_ids, entropy_ids, horizon_replay_max = [], [], 0.
state_replays = []
for role in ('train', 'validation'):
    count = cfg.train_paths if role == 'train' else cfg.validation_paths
    with h5py.File(ROOT/f'payoffs/interacting_{role}.h5', 'r') as store:
        for case in cfg.cases:
            group = store[case]
            assert not bool(store.attrs['future_payoff_labels'])
            for index in range(count):
                family, seed = _family(cfg, role, case, index), _seed(cfg, role, case, index)
                assert group['family_ids'][index].decode() == family
                expected = seeded_rng(seed, 'independent-residual-horizons').uniform(
                    0, cfg.horizon, (len(cfg.state_times), cfg.horizon_samples))
                error = float(np.max(np.abs(expected - group['horizons'][index])))
                horizon_replay_max = max(horizon_replay_max, error)
                assert error == 0
                seed_ids.append(seed)
                context = cfg.context(case)
                entropy_ids.extend([entropy_id(seed, 'native-initial', context.context_id, cfg.spec.target_id),
                                    entropy_id(seed, 'native-face-noise', context.context_id, cfg.spec.target_id),
                                    entropy_id(seed, 'independent-residual-horizons')])
            index = 0
            seed, family = _seed(cfg, role, case, index), _family(cfg, role, case, index)
            initial = initialize_masses(cfg.context(case), cfg.spec, 1, seed)
            batch = simulate_native(cfg.context(case), cfg.spec, cfg.state_times, cfg.state_step,
                                    seed, 1, initial_masses=initial, family_ids=(family,))
            err = float(abs(batch.states[0] - group['states'][index]).max())
            assert err < 1e-14
            state_replays.append({'role': role, 'case': case, 'family': family, 'maximum_state_error': err})
assert len(set(seed_ids)) == len(seed_ids) == 384
assert len(set(entropy_ids)) == len(entropy_ids) == 1152

# Reference *entropy addresses* only; no held-out anchor/forcing samples drawn.
reference_entropy_ids = []
for key in refcfg.group_keys:
    case, role, anchor = reference._group_plan(key)
    for chunk in range((refcfg.paths_per_group + refcfg.chunk_paths - 1)//refcfg.chunk_paths):
        reference_entropy_ids.append(entropy_id(refcfg.seed, reference.STREAM, refcfg.config_hash,
                                               'reference-chunk', case, role, anchor, chunk))
    if role == 'conditional':
        reference_entropy_ids.append(entropy_id(refcfg.seed, reference.STREAM, refcfg.config_hash,
                                               'held-out-anchor', case, anchor))
assert len(set(reference_entropy_ids)) == 1540
assert not set(reference_entropy_ids) & set(entropy_ids)
assert cfg.seed != refcfg.seed

# Historical IDs only. No historical outcome arrays or fresh test arrays read.
historical_paths = [
    REPO/'runs/path-generalization-2026-10-26/payoffs/interacting_train.h5',
    REPO/'runs/path-generalization-2026-10-26/payoffs/interacting_validation.h5',
    REPO/'runs/interacting-followup-2026-10-25/balanced/interacting_train.h5',
    REPO/'runs/interacting-followup-2026-10-25/balanced/interacting_validation.h5',
    REPO/'runs/interacting-followup-2026-10-25/paths/path_train.h5',
    REPO/'runs/interacting-followup-2026-10-25/paths/path_validation.h5',
    REPO/'runs/interacting-followup-2026-10-25/paths/path_test.h5',
    REPO/'runs/interacting-followup-2026-10-25/balanced/interacting_reference.h5',
    REPO/'runs/path-generalization-2026-10-26/reference/reference.h5',
]
historical = []
current = ids['train'] | ids['validation']
all_historical = set()
for path in historical_paths:
    if not path.exists():
        historical.append({'path': str(path), 'present': False})
        continue
    old_ids, inventory = family_ids(path)
    overlap = current & old_ids
    assert not overlap
    all_historical.update(old_ids)
    historical.append({'path': str(path), 'present': True, 'unique_families': len(old_ids),
                       'family_identifiers_hash': stable_hash(sorted(old_ids)),
                       'overlap_with_current_training_validation': 0, 'datasets_read': inventory})

# Validate the actual unlock path rejects the currently missing fit lock.
try:
    runner.verify_fit_lock(ROOT)
except FileNotFoundError:
    unlock_rejected = True
else:
    raise AssertionError('Reference was unexpectedly unlocked')
assert not ref_path.exists()

rolling = []
for path in sorted((ROOT/'payoffs/checkpoints').rglob('*.pt')):
    if path.name.endswith('.tmp'):
        continue
    saved = torch.load(path, map_location='cpu', weights_only=True)
    provenance = saved['provenance']
    assert provenance['source_hash'] == SHA
    assert not provenance.get('future_moment_labels')
    for role in ('training', 'validation'):
        assert provenance['data_fingerprints'][role]['content_hash'] == arrays['train' if role == 'training' else role]['content_hash']
    rolling.append({'path': str(path), 'kind': saved['kind'], 'complete_at_snapshot': saved['complete'],
                    'steps_done': saved.get('steps_done', {'adam': saved.get('adam_steps_done'),
                                                        'lbfgs': saved.get('lbfgs_steps_done')}),
                    'best_step': saved['best_step'], 'initial_validation': saved['initial_validation'],
                    'best_validation': saved['best_validation'],
                    'training_seconds': saved['training_seconds']})

result = {'kind': 'independent-payoff-repair-pretest-audit-v1', 'complete': True,
          'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
          'audit_script_sha256': file_sha(__file__), 'source_freeze_commit': rev, 'source_hash': SHA,
          'runner_sha256': design['runner_sha256'], 'helper_checks': helper_checks,
          'pretest_snapshot': pretest_snapshot, 'reference_unlock_rejected_until_complete_fit_lock': unlock_rejected,
          'fresh_test_data_opened_drawn_or_created': False,
          'declared_slots': design['slots'], 'independent_training_families': 256,
          'independent_validation_families': 128, 'derived_samples_per_family': 16,
          'literal_pool_checks': shared_pool, 'train_validation_overlap': 0,
          'horizon_replay_maximum_error': horizon_replay_max, 'state_replays': state_replays,
          'unique_collocation_parent_seeds': len(set(seed_ids)),
          'unique_collocation_initial_forcing_horizon_entropy_digests': len(set(entropy_ids)),
          'predeclared_reference_entropy_digests': len(reference_entropy_ids),
          'reference_entropy_overlap_with_collocation': 0,
          'training_seed': cfg.seed, 'reference_seed': refcfg.seed,
          'historical_family_inventory': historical, 'unique_historical_families_checked': len(all_historical),
          'current_historical_family_overlap': 0, 'rolling_training_only_diagnostics': rolling,
          'selection_using_final_test': False, 'post_test_refitting': False,
          'limitations': ['Hash-addressed pseudorandom streams, not a mathematical independence proof.',
                         'No final-reference outcome or held-out initial anchor was sampled or opened in this audit.',
                         'Rolling checkpoints may advance after the recorded snapshot; final model lock remains authoritative.']}
dest = ROOT/'audit/pretest_audit.json'
tmp = dest.with_suffix('.json.tmp')
tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
tmp.replace(dest)
print(json.dumps({'complete': True, 'audit': str(dest), 'sha256': file_sha(dest),
                  'current_families': len(current), 'historical_families': len(all_historical),
                  'rolling_checkpoints': len(rolling), 'reference_still_absent': not ref_path.exists()}, indent=2))
