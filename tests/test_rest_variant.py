"""Rest-source variants (DATASET_PARAMS' rest_variant): HCP-YA 3T rest read clean or uncleaned (noFIX).

The default source must leave every path, scan key and output name unchanged; a non-default source
travels in the rest scans' ``desc`` and tags every output its processor writes.
"""
import numpy as np
import pytest

from snaplab_tools.timescales import compute_acf
from src.config import DATASET_PARAMS, get_dataset_params
from src.data import (DataLoader, ScanInfo, REST_VARIANT_DEFAULT, build_scan_string,
                      load_dataset_config)
from src.timescales import Timescales, rest_variant_of, rest_variant_output_type
from tests.conftest import integration

SUB = '100206'
HCPYA_FUNC = ('/mnt/lindenmp_bignas/lindenmp/research_data/HCP_YA/derivatives/parcellated_data/'
              f'sub-{SUB}/ses-01/func')


def _loader(dataset='HCP-YA'):
    """A DataLoader carrying only what path resolution and registration read (no disk access)."""
    loader = DataLoader.__new__(DataLoader)
    loader.dataset = dataset
    loader.config = load_dataset_config(dataset)
    loader.scan_registry = []
    return loader


def _register(loader, rest_variant=REST_VARIANT_DEFAULT, modality='MRI3T',
              tasks=('rfMRIREST1LR', 'tfMRIWMLR')):
    return loader.register_scans(subject_ids=[SUB], sessions=['01'], tasks=list(tasks), space='fsLR',
                                 seg='Schaefer4007', modality=modality, rest_variant=rest_variant)


def _key(sub, task, condition, desc=None):
    return build_scan_string(sub=sub, ses='01', modality='MRI3T', task=task, space='fsLR',
                             seg='Schaefer4007', tr='0.72', condition=condition, desc=desc)


# ── configuration ─────────────────────────────────────────────────────────────

def test_hcpya_3t_lists_nofix_and_defaults_to_clean():
    assert load_dataset_config('HCP-YA').modalities['MRI3T'].rest_variants == ['noFIX']
    assert load_dataset_config('HCP-YA').modalities['MRI7T'].rest_variants == []
    assert DATASET_PARAMS['HCP-YA']['rest_variant'] == REST_VARIANT_DEFAULT == 'clean'


def test_get_dataset_params_always_carries_rest_variant():
    assert get_dataset_params('HCP-D')['rest_variant'] == 'clean'
    assert get_dataset_params('HCP-YA', rest_variant='noFIX')['rest_variant'] == 'noFIX'
    assert DATASET_PARAMS['HCP-YA']['rest_variant'] == 'clean'      # the override did not leak


# ── registration and paths ────────────────────────────────────────────────────

def test_default_source_leaves_scans_and_paths_unchanged():
    loader = _loader()
    rest, task = _register(loader)
    assert rest.desc is None and task.desc is None
    assert str(rest) == ('sub-100206_task-rfMRIREST1LR_space-fsLR_seg-Schaefer4007_ses-01_'
                         'modality-MRI3T_tr-0.72_condition-rest')
    assert loader.get_scan_filepath(rest).name == (
        'sub-100206_ses-01_task-rfMRIREST1LR_space-fsLR_seg-Schaefer4007_stat-mean_timeseries.ptseries.nii')


def test_nofix_tags_rest_scans_only():
    loader = _loader()
    rest, task = _register(loader, rest_variant='noFIX')
    assert rest.desc == 'noFIX' and str(rest).endswith('_desc-noFIX')
    assert task.desc is None
    assert loader.get_scan_filepath(rest).name == (
        'sub-100206_ses-01_task-rfMRIREST1LR_space-fsLR_seg-Schaefer4007_desc-noFIX_'
        'stat-mean_timeseries.ptseries.nii')
    assert loader.get_scan_filepath(task).name == (
        'sub-100206_ses-01_task-tfMRIWMLR_space-fsLR_seg-Schaefer4007_stat-mean_timeseries.ptseries.nii')


def test_desc_survives_the_registry_round_trip():
    rest = _register(_loader(), rest_variant='noFIX')[0]
    assert ScanInfo.from_dict(rest.to_dict()) == rest


@pytest.mark.parametrize('dataset, modality, variant', [
    ('HCP-YA', 'MRI7T', 'noFIX'),        # 7T has no uncleaned rest on disk
    ('HCP-YA', 'MRI3T', 'nofix'),        # spelling is exact
    ('HCP-D', 'MRI3T', 'noFIX'),
])
def test_an_unlisted_variant_raises(dataset, modality, variant):
    with pytest.raises(ValueError, match='has no rest_variant'):
        _register(_loader(dataset), rest_variant=variant, modality=modality,
                  tasks=('rfMRIREST17TPA',) if modality == 'MRI7T' else ('rfMRIREST1PA',))


@integration
def test_nofix_file_resolves_on_disk():
    loader = DataLoader('HCP-YA', load_behavior=False, verbose=False)
    rest = _register(loader, rest_variant='noFIX', tasks=('rfMRIREST1LR',))[0]
    path = loader.get_scan_filepath(rest)
    assert str(path.parent) == HCPYA_FUNC and path.exists()


# ── output naming ─────────────────────────────────────────────────────────────

def test_rest_variant_of_reads_the_rest_keys():
    clean = [_key(SUB, 'rfMRIREST1LR', 'rest'), _key(SUB, 'tfMRIWMLR', 'task')]
    nofix = [_key(SUB, 'rfMRIREST1LR', 'rest', 'noFIX'), _key(SUB, 'tfMRIWMLR', 'task')]
    assert rest_variant_of(clean) == 'clean'
    assert rest_variant_of(nofix) == 'noFIX'
    assert rest_variant_of([]) == 'clean'
    with pytest.raises(ValueError, match='more than one rest source'):
        rest_variant_of(clean + nofix)


def test_token_goes_after_the_stat_name():
    assert rest_variant_output_type('autocorr', 'clean') == 'autocorr'
    assert rest_variant_output_type('autocorr', 'noFIX') == 'autocorr_noFIX'
    assert rest_variant_output_type('timescale_auc_zero_crossing', 'noFIX') == \
        'timescale_auc_zero_crossing_noFIX'


def _processor(desc):
    """A Timescales holding one rest and one task scan (no DataLoader init)."""
    rng = np.random.default_rng(0)
    proc = Timescales.__new__(Timescales)
    proc.dataset = 'HCP-YA'
    keys = [_key(SUB, 'rfMRIREST1LR', 'rest', desc), _key(SUB, 'tfMRIWMLR', 'task')]
    proc.scans = keys
    proc.timeseries_data = {k: rng.standard_normal((4, 200)).cumsum(axis=1) for k in keys}
    proc.timeseries_residuals = {keys[0]: None, keys[1]: proc.timeseries_data[keys[1]]}
    proc._load_subject_output = lambda **kw: None
    proc.saved = []
    proc._save_subject_output = lambda **kw: proc.saved.append(kw)
    return proc


def _written(proc):
    return sorted({(s['output_type'], s['task']) for s in proc.saved})


def test_default_source_names_are_unchanged():
    proc = _processor(None)
    proc.compute_autocorrelation_function(use_fft=True, verbose=False)
    proc.compute_timescale_measures(verbose=False)
    proc.compute_timeseries_measures(method='var', verbose=False)
    assert {o for o, _ in _written(proc)} == {'autocorr', 'timescale_auc_zero_crossing', 'metric_var'}
    assert {t for _, t in _written(proc)} == {'rfMRIREST1LR', 'tfMRIWMLR'}


def test_nofix_writes_tagged_rest_outputs_and_skips_task_scans():
    proc = _processor('noFIX')
    proc.compute_autocorrelation_function(use_fft=True, verbose=False)
    proc.compute_timescale_measures(verbose=False)
    proc.compute_timeseries_measures(method='var', verbose=False)
    assert _written(proc) == [('autocorr_noFIX', 'rfMRIREST1LR'),
                              ('metric_var_noFIX', 'rfMRIREST1LR'),
                              ('timescale_auc_zero_crossing_noFIX', 'rfMRIREST1LR')]
    task = proc.scans[1]
    assert proc.autocorr_dict[task] is None and proc.data_dict[task] is None


def test_nofix_keeps_the_task_keyed_control():
    proc = _processor('noFIX')
    task = proc.scans[1]
    proc.autocorr_rest_control = {proc.scans[0]: None,
                                  task: np.atleast_2d(compute_acf(proc.timeseries_data[task]))}
    proc.rest_control_meta = {task: {}}
    proc.compute_autocorrelation_function(use_rest_control='full', verbose=False)
    proc.compute_timescale_measures(suffix='restctrl', verbose=False)
    assert _written(proc) == [('autocorr_noFIX_restctrl', 'tfMRIWMLR'),
                              ('timescale_auc_zero_crossing_noFIX_restctrl', 'tfMRIWMLR')]
    # No rest_variant key: the token is the record.
    assert 'rest_variant' not in proc.saved[0]['metadata']


@pytest.mark.parametrize('call', ['compute_autocorrelation_function', 'compute_timeseries_measures'])
def test_nofix_refuses_task_residuals(call):
    proc = _processor('noFIX')
    with pytest.raises(ValueError, match='does not depend on the rest source'):
        getattr(proc, call)(use_residuals=True, verbose=False)
    assert proc.saved == []
