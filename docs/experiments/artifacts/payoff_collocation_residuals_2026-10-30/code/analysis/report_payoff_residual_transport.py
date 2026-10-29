"""Report observed residual transport and its finite-grid defects, not a certificate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'src'))

import numpy as np  # noqa: E402

from analysis import audit_payoff_residual_transport as audit  # noqa: E402

KIND = 'frozen-v3-residual-transport-report-v1'


def _maximum(value):
    values = np.asarray(value, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Complete finite transport metrics required')
    return float(abs(values).max())


def summarize(record):
    """Retain both Euler grids and quadratures; use no norm significance test."""
    result = {key: record[key] for key in ('case', 'anchor', 'profile', 'seed', 'queries', 'families')}
    result['levels'] = {}
    for level in ('coarse', 'fine'):
        result['levels'][level] = {}
        for stride in ('declared', 'half_'):
            point = record['levels'][level][stride]['point']
            expected = {'mean_discrepancy', 'raw_second_discrepancy',
                'sample_covariance_discrepancy_unbiased', 'mean_residual_integral',
                'raw_second_residual_integral', 'covariance_residual_integral_alone',
                'mean_transport_defect', 'raw_second_transport_defect'}
            if set(point) != expected:
                raise ValueError('Complete transport metric inventory required')
            metrics = {key+'_maxabs': _maximum(point[key]) for key in sorted(expected)}
            full = np.asarray(point['raw_second_residual_integral'])
            alone = np.asarray(point['covariance_residual_integral_alone'])
            metrics['mean_cross_terms_integral_maxabs'] = _maximum(full-alone)
            result['levels'][level][stride] = metrics
        result[level+'_stopped_families'] = record[level+'_stopped_families']
    result['paired_changes'] = {
        key: {name: _maximum(value[name]) for name in (
            'fine_minus_coarse_same_physical_residual_nodes',
            'halfstride_minus_stride_on_coarse_Euler', 'halfstride_minus_stride_on_fine_Euler')}
        for key, value in record['paired_refinement_and_halfstride'].items()}
    result['paired_stopping_diagnostics'] = record['paired_stopping_diagnostics']
    return result


def report(root, output=None):
    root = Path(root).resolve()
    completed = audit.load_complete(root)
    saved = audit.read(root/'plan.json')
    expected = {(case, anchor, row['seed']) for case, item in saved['anchors'].items()
        for anchor in range(item['rows']) for row in saved['identity']['models'] if row['case'] == case}
    actual = [(row['case'], row['anchor'], row['seed']) for row in completed['records']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete case/anchor/seed allocation required')
    plan = saved['identity']['plan']
    for row in completed['records']:
        if (row['families'] != plan['paths']
                or row['queries'] != saved['anchors'][row['case']]['queries']):
            raise ValueError('Query or family allocation changed')
    records = [summarize(row) for row in completed['records']]
    output = (root/'report' if output is None else Path(output)).resolve()
    inputs = {str(root/name): audit.sha(root/name) for name in ('plan.json', 'index.json', 'summary.json')}
    result = {'kind': KIND, 'complete': True, 'records': records,
        'allocation': {'cases': sorted(saved['anchors']), 'models': len(saved['identity']['models']),
            'anchors': sum(item['rows'] for item in saved['anchors'].values()),
            'families_per_anchor': plan['paths'], 'model_anchor_records': len(records),
            'step': plan['step'], 'refinement_factor': plan['refinement_factor'],
            'residual_stride': plan['residual_stride']},
        'units': 'normalized sqrt(N) payoffs; raw seconds in their corresponding products',
        'interpretation': 'Observed prediction discrepancy, residual transport and remainder are distinct. '
            'Raw-second transport includes accumulated running rewards and both mean-residual cross terms. '
            'It is not covariance error, and r_C alone is not an error certificate.',
        'uncertainty': 'Signed pointwise whole-family bootstrap intervals remain in the original summary; '
            'maxima below are descriptive and never used as significance tests or error enclosures.',
        'unresolved': completed['not_certified']}
    audit.write(output/'report.json', result)
    lines = ['# Frozen-v3 residual transport diagnostic', '',
        'All planned cases, validation anchors and optimizer seeds are retained. '
        'This investigation uses fresh diagnostic continuations, not final candidate-selection outcomes.', '',
        result['interpretation'], '', result['uncertainty'], '',
        'The table shows maximum absolute entries on the finer Euler grid with half-stride residual '
        'quadrature. Units are normalized sqrt(N) payoffs and raw-second products; these are not '
        'relative covariance errors. Euler and half-stride contrasts are retained in `report.json`.', '',
        '| Case | Anchor / profile | Seed | Mean discrepancy | Mean residual integral | Mean defect | Raw-second discrepancy | Raw-second integral | Raw-second defect |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        metrics = row['levels']['fine']['half_']
        keys = ('mean_discrepancy', 'mean_residual_integral', 'mean_transport_defect',
                'raw_second_discrepancy', 'raw_second_residual_integral', 'raw_second_transport_defect')
        numbers = ' | '.join(f"{metrics[key+'_maxabs']:.6g}" for key in keys)
        lines.append(f"| {row['case']} | {row['anchor']} / {row['profile']} | {row['seed']} | {numbers} |")
    lines.extend(['', 'Unresolved: '+', '.join(result['unresolved'])+'.', '',
                  'Every stopped path remains in the statistics. No fit, final-data tuning, '
                  'continuum claim or accuracy-qualified cost claim follows from this diagnostic.', ''])
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.md').write_text('\n'.join(lines))
    manifest = {'kind': KIND, 'complete': True, 'identity': {'audit_root': str(root), 'inputs': inputs,
        'report_script_sha256': audit.sha(__file__), 'audit_script_sha256': audit.sha(audit.__file__)},
        'outputs': {name: audit.sha(output/name) for name in ('report.json', 'report.md')}}
    audit.write(output/'output_manifest.json', manifest)
    return result


def load_report(output):
    output = Path(output).resolve()
    manifest = audit.read(output/'output_manifest.json')
    if (manifest['kind'] != KIND or manifest['complete'] is not True
            or manifest['identity']['report_script_sha256'] != audit.sha(__file__)
            or manifest['identity']['audit_script_sha256'] != audit.sha(audit.__file__)
            or set(manifest['outputs']) != {'report.json', 'report.md'}):
        raise ValueError('Report identity or inventory changed')
    audit.load_complete(manifest['identity']['audit_root'])
    for path, digest in manifest['identity']['inputs'].items():
        if audit.sha(path) != digest:
            raise ValueError('Frozen report input changed')
    for name, digest in manifest['outputs'].items():
        if audit.sha(output/name) != digest:
            raise ValueError('Report output evidence changed')
    return audit.read(output/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(json.dumps({'complete': True, 'records': len(result['records']), 'allocation': result['allocation']}))


if __name__ == '__main__':
    main()
