"""Matched rest controls and reference layouts of the figure notebooks (src.notebook)."""
import types

import numpy as np
import pandas as pd
import pytest
from snaplab_tools.stats import paired_ttest_vs_reference

from src.data import ScanInfo, build_scan_string
from src.notebook import (control_output_type, control_tag, load_rest_controls, reference_key, reference_layout,
                          ttest_vs_references)
from src.timescales import condition_runs, convert_data_dict


# ── which controls are read ───────────────────────────────────────────────────

@pytest.mark.parametrize('dataset, modality, rest_source, filt, name', [
    ('HCP-YA', 'MRI3T', 'clean', None, 'timescale_auc_zero_crossing_restctrl'),
    ('HCP-YA', 'MRI3T', 'noFIX', None, 'timescale_auc_zero_crossing_noFIX_restctrl'),
    # 7T lists no noFIX source, so its controls are cut from clean rest, as its runs were denoised.
    ('HCP-YA', 'MRI7T', 'noFIX', None, 'timescale_auc_zero_crossing_restctrl'),
    ('HCP-D', 'MRI3T', 'clean', 'bp-0.01-0.08hz', 'timescale_auc_zero_crossing_bp-0.01-0.08hz_restctrl'),
])
def test_control_output_type(dataset, modality, rest_source, filt, name):
    assert control_output_type(dataset, modality, rest_source, filt) == name


def test_an_unknown_rest_source_raises():
    with pytest.raises(ValueError, match='rest_source'):
        control_output_type('HCP-YA', 'MRI3T', 'nofix')


def test_control_tag_keeps_the_existing_file_names():
    assert control_tag() == '_cmp-restctrl'
    assert control_tag('noFIX') == '_rest-noFIX_cmp-restctrl'


# ── layout and tests ──────────────────────────────────────────────────────────

def test_without_controls_the_layout_is_the_custom_order_with_one_shared_reference():
    order, ref = reference_layout(['task-Rest', 'task-Movies', 'task-WM'])
    assert order == ['task-Rest', 'task-Movies', 'task-WM']
    assert ref == {'task-Movies': 'task-Rest', 'task-WM': 'task-Rest'}


def test_with_controls_each_condition_is_paired_with_its_own():
    order, ref = reference_layout(['task-Rest', 'task-Movies', 'task-WM'], controls=True)
    assert order == [reference_key('task-Movies'), 'task-Movies', reference_key('task-WM'), 'task-WM']
    assert ref == {'task-Movies': reference_key('task-Movies'), 'task-WM': reference_key('task-WM')}


def test_ttest_with_a_shared_reference_is_the_library_call():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=['Rest', 'A', 'B'])
    pd.testing.assert_frame_equal(ttest_vs_references(df, {'A': 'Rest', 'B': 'Rest'}),
                                  paired_ttest_vs_reference(df, reference='Rest'))


def test_ttest_with_own_references_corrects_across_conditions():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=['rA', 'A', 'rB', 'B'])
    out = ttest_vs_references(df, {'A': 'rA', 'B': 'rB'})
    one = paired_ttest_vs_reference(df[['rA', 'A']], reference='rA', correction=None)
    assert out.loc['A', 'p'] == one.loc['A', 'p']
    assert (out['p_corr'] >= out['p']).all() and list(out.index) == ['A', 'B']


def test_condition_runs_is_the_inverse_of_the_grouping():
    runs = ['tfMRIWMLR', 'tfMRIWMRL', 'tfMRIMOVIE17TAP', 'tfMRICARITPA']
    assert condition_runs('task-WM', runs) == ['tfMRIWMLR', 'tfMRIWMRL']
    assert condition_runs('task-Movies', runs) == ['tfMRIMOVIE17TAP']
    assert condition_runs('tfMRICARITPA', runs) == ['tfMRICARITPA']
    with pytest.raises(ValueError):
        condition_runs('task-Tasks', runs)


# ── load_rest_controls, on a stub processor ───────────────────────────────────

def _key(sub, task, condition):
    return build_scan_string(sub=sub, ses='01', modality='MRI3T', task=task, space='fsLR', seg='Schaefer4007',
                             tr='0.72', condition=condition)


def _processor(cache):
    """A stand-in exposing what load_rest_controls reads: dataset and get_precomputed_outputs."""
    proc = types.SimpleNamespace(dataset='HCP-YA', data_dict=None)
    proc.get_precomputed_outputs = lambda output_type: setattr(proc, 'data_dict', dict(cache.get(output_type, {})))
    return proc


def _data(subjects, tasks=(('rfMRIREST1LR', 'rest'), ('tfMRIWMLR', 'task'), ('tfMRIWMRL', 'task'))):
    keys = {s: {t: _key(s, t, c) for t, c in tasks} for s in subjects}
    return keys, {k: np.ones(3) for d in keys.values() for k in d.values()}


def test_controls_are_read_per_task_run_and_a_missing_one_drops_the_subject():
    keys, data = _data(('1', '2'))
    cache = {'timescale_auc_zero_crossing_noFIX_restctrl': {
        keys['1']['tfMRIWMLR']: np.full(3, 2.0), keys['1']['tfMRIWMRL']: np.full(3, 3.0),
        keys['2']['tfMRIWMLR']: np.full(3, 4.0), keys['2']['tfMRIWMRL']: None}}
    out, controls, dropped = load_rest_controls(_processor(cache), data, ['task-WM'], rest_source='noFIX',
                                                verbose=False)
    assert dropped == {'task-WM': ['2']}
    assert all('sub-2' not in k for k in out) and len(out) == 3
    assert set(controls) == {keys['1']['tfMRIWMLR'], keys['1']['tfMRIWMRL']}
    np.testing.assert_array_equal(controls[keys['1']['tfMRIWMRL']], np.full(3, 3.0))


def test_only_the_named_conditions_runs_are_read():
    """A subject missing another task's control is kept: only the conditions analysed matter."""
    keys, data = _data(('1',), tasks=(('rfMRIREST1LR', 'rest'), ('tfMRIWMLR', 'task'), ('tfMRISOCIALLR', 'task')))
    cache = {'timescale_auc_zero_crossing_restctrl': {keys['1']['tfMRIWMLR']: np.ones(3)}}
    out, controls, dropped = load_rest_controls(_processor(cache), data, ['task-WM'], verbose=False)
    assert dropped == {'task-WM': []} and set(controls) == {keys['1']['tfMRIWMLR']} and out == data


def test_uncached_controls_raise_rather_than_dropping_everyone():
    keys, data = _data(('1', '2'))
    with pytest.raises(FileNotFoundError, match='no matched rest control'):
        load_rest_controls(_processor({}), data, ['task-WM'], verbose=False)


def test_convert_data_dict_leaves_a_task_without_scans_unbinned():
    scans = [ScanInfo(subject_id=s, task=t, space='fsLR', seg='Schaefer4007', modality='MRI3T', tr=0.72, session='01')
             for s in ('1', '2') for t in ('rfMRIREST1LR', 'tfMRIWMLR')]
    rng = np.random.default_rng(0)
    data = {str(s): rng.normal(size=20) for s in scans if s.task == 'tfMRIWMLR'}
    out = convert_data_dict(data, scans, brain_map=np.arange(20), verbose=False)
    assert out['rfMRIREST1LR'].size == 0 and out['tfMRIWMLR'].shape == (2, 10)
