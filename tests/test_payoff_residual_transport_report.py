"""Transport reports retain allocation, signed source coupling and provenance."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import report_payoff_residual_transport as report  # noqa: E402


def record(seed=0):
    values = {key: np.full((2, 4), 3.).tolist() for key in (
        'mean_discrepancy', 'mean_residual_integral', 'mean_transport_defect')}
    values.update({key: np.full((2, 4, 4), 2.).tolist() for key in (
        'raw_second_discrepancy', 'sample_covariance_discrepancy_unbiased',
        'raw_second_residual_integral', 'covariance_residual_integral_alone', 'raw_second_transport_defect')})
    values['raw_second_residual_integral'] = np.full((2, 4, 4), 12.).tolist()
    change = {key: np.zeros((2, 4)).tolist() for key in (
        'fine_minus_coarse_same_physical_residual_nodes',
        'halfstride_minus_stride_on_coarse_Euler', 'halfstride_minus_stride_on_fine_Euler')}
    return {'case': 'case', 'anchor': 0, 'profile': 'uniform', 'seed': seed,
        'queries': [.25, .5], 'families': 8,
        'levels': {level: {stride: {'point': copy.deepcopy(values)} for stride in ('declared', 'half_')}
                   for level in ('coarse', 'fine')},
        'coarse_stopped_families': 2, 'fine_stopped_families': 3,
        'paired_stopping_diagnostics': {'both': 2, 'fine_only': 1},
        'paired_refinement_and_halfstride': {'mean_discrepancy': change}}


def test_raw_second_mean_cross_terms_remain_distinct_from_covariance_integral():
    row = report.summarize(record())
    metrics = row['levels']['fine']['half_']
    assert metrics['raw_second_residual_integral_maxabs'] == 12
    assert metrics['covariance_residual_integral_alone_maxabs'] == 2
    assert metrics['mean_cross_terms_integral_maxabs'] == 10
    assert row['fine_stopped_families'] == 3
    assert set(row['levels']) == {'coarse', 'fine'}
    assert set(row['levels']['coarse']) == {'declared', 'half_'}


@pytest.fixture
def completed(tmp_path, monkeypatch):
    plan = {'identity': {'plan': {'paths': 8, 'step': 1/128, 'refinement_factor': 2,
                                 'residual_stride': 4},
                         'models': [{'case': 'case', 'seed': i} for i in range(3)]},
            'anchors': {'case': {'rows': 1, 'queries': [.25, .5]}}}
    result = {'records': [record(i) for i in range(3)], 'not_certified': ['continuous stopping bias']}
    for name, value in (('plan.json', plan), ('index.json', {'complete': True}), ('summary.json', result)):
        (tmp_path/name).write_text(json.dumps(value))
    monkeypatch.setattr(report.audit, 'load_complete', lambda root: json.loads((Path(root)/'summary.json').read_text()))
    return tmp_path


def test_report_roundtrip_and_tampered_output_rejected(completed):
    result = report.report(completed)
    assert result['allocation']['model_anchor_records'] == 3
    assert result == report.load_report(completed/'report')
    output = completed/'report/report.md'
    output.write_text(output.read_text()+'tampered')
    with pytest.raises(ValueError, match='output evidence'):
        report.load_report(completed/'report')


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'families', 'queries'])
def test_report_rejects_partial_or_changed_allocations(completed, change):
    path = completed/'summary.json'
    value = json.loads(path.read_text())
    if change == 'missing':
        value['records'].pop()
    elif change == 'duplicate':
        value['records'][-1] = value['records'][0]
    elif change == 'families':
        value['records'][0]['families'] = 4
    else:
        value['records'][0]['queries'] = [.5]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='allocation'):
        report.report(completed)


def test_report_rejects_mutated_input_after_rendering(completed):
    report.report(completed)
    (completed/'index.json').write_text('{"complete": false}')
    with pytest.raises(ValueError, match='input changed'):
        report.load_report(completed/'report')


def test_missing_nonfinite_transport_terms_are_rejected():
    row = record()
    del row['levels']['fine']['half_']['point']['raw_second_residual_integral']
    with pytest.raises(ValueError, match='inventory'):
        report.summarize(row)
    row = record()
    row['levels']['fine']['half_']['point']['mean_discrepancy'][0][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        report.summarize(row)
