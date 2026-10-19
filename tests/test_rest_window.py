"""Length-matched rest (``restlen-<task>``): one window per matched task of each covered rest run,
nothing projected.

"""
import inspect

import numpy as np
import pytest

from snaplab_tools.timescales import compute_acf
from src.data import build_scan_string
from src import timescales as timescales_module
from src.notebook import _resolve_rest_match, _substitute_rest_windows, load_notebook_data
from src.timescales import (REST_WINDOW_MATCH, REST_WINDOW_NOTE, REST_WINDOW_TOKEN, Timescales,
                            frequency_filter_token, rest_window_covered, rest_window_match_for, rest_window_matches,
                            rest_window_output_type, rest_window_token)

HCPD_TASKS = (('tfMRICARITPA', 290), ('tfMRIEMOTIONPA', 168), ('tfMRIGUESSINGPA', 270))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _key(sub, task, condition, modality='MRI3T', tr='0.72'):
    return build_scan_string(sub=sub, ses='01', modality=modality, task=task, space='fsLR',
                             seg='Schaefer4007', tr=tr, condition=condition)


def _stub(dataset, timeseries):
    """A Timescales carrying only what the window path reads, built through ``__new__`` so no
    DataLoader init (and so no data mount) is involved."""
    processor = Timescales.__new__(Timescales)
    processor.dataset = dataset
    processor.scans = list(timeseries)
    processor.timeseries_data = dict(timeseries)
    return processor


def _hcpya_subject(rng, sub='100206', rest_frames=1200, wm=(405, 405)):
    ts = {_key(sub, 'rfMRIREST1LR', 'rest'): rng.standard_normal((5, rest_frames)),
          _key(sub, 'rfMRIREST1RL', 'rest'): rng.standard_normal((5, rest_frames)),
          _key(sub, 'tfMRIEMOTIONLR', 'task'): rng.standard_normal((5, 176))}
    for task, n in zip(('tfMRIWMLR', 'tfMRIWMRL'), wm):
        ts[_key(sub, task, 'task')] = rng.standard_normal((5, n))
    return ts


def _hcpd_subject(rng, sub='HCD001', rest_frames=478, tasks=HCPD_TASKS):
    ts = {_key(sub, run, 'rest', tr='0.8'): rng.standard_normal((5, rest_frames))
          for run in ('rfMRIREST1PA', 'rfMRIREST2PA')}
    for task, n in tasks:
        ts[_key(sub, task, 'task', tr='0.8')] = rng.standard_normal((5, n))
    return ts


class TestTable:
    def test_scope_is_wm_for_hcpya_and_carit_emotion_guessing_for_hcpd(self):
        assert REST_WINDOW_MATCH == {
            ('HCP-YA', 'MRI3T'): {'EMOTION': ('tfMRIEMOTIONLR', 'tfMRIEMOTIONRL'),
                                  'GAMBLING': ('tfMRIGAMBLINGLR', 'tfMRIGAMBLINGRL'),
                                  'LANGUAGE': ('tfMRILANGUAGELR', 'tfMRILANGUAGERL'),
                                  'RELATIONAL': ('tfMRIRELATIONALLR', 'tfMRIRELATIONALRL'),
                                  'SOCIAL': ('tfMRISOCIALLR', 'tfMRISOCIALRL'),
                                  'WM': ('tfMRIWMLR', 'tfMRIWMRL')},
            ('HCP-D', 'MRI3T'): {'CARIT': ('tfMRICARITPA',), 'EMOTION': ('tfMRIEMOTIONPA',),
                                 'GUESSING': ('tfMRIGUESSINGPA',)}}

    def test_everything_else_is_uncovered(self):
        # HCP-YA 7T rest and movies are already matched; the XCP-D datasets are out of scope.
        for dataset, modality in [('HCP-YA', 'MRI7T'), ('WAND', 'MRI3T'), ('Penn_LEAD', 'MRI3T')]:
            assert not rest_window_covered(dataset, modality)
        assert rest_window_covered('HCP-D', 'MRI3T', 'EMOTION')
        assert not rest_window_covered('HCP-D', 'MRI3T', 'WM')

    def test_each_match_has_its_own_token_in_the_last_slot(self):
        assert REST_WINDOW_TOKEN == 'restlen'
        assert rest_window_token('CARIT') == 'restlen-CARIT'
        assert rest_window_output_type('autocorr', 'WM') == 'autocorr_restlen-WM'
        assert (rest_window_output_type('autocorr_bp-0.01-0.08hz', 'EMOTION')
                == 'autocorr_bp-0.01-0.08hz_restlen-EMOTION')
        assert (rest_window_output_type('timescale_auc_zero_crossing_bp-0.01-0.08hz', 'GUESSING')
                == 'timescale_auc_zero_crossing_bp-0.01-0.08hz_restlen-GUESSING')

    @pytest.mark.parametrize('bad', [None, '', True])
    def test_a_token_needs_a_match_name(self, bad):
        """The bare 'restlen' of the first build is retired: every window names its task."""
        with pytest.raises(ValueError, match='match name'):
            rest_window_token(bad)

    def test_matches_per_dataset(self):
        assert rest_window_matches('HCP-YA') == ('EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL',
                                                  'SOCIAL', 'WM')
        assert rest_window_matches('HCP-D') == ('CARIT', 'EMOTION', 'GUESSING')
        assert rest_window_matches('HCP-YA', 'MRI7T') == ()
        assert rest_window_matches('WAND') == ()

    @pytest.mark.parametrize('dataset,task,match', [
        ('HCP-D', 'tfMRICARITPA', 'CARIT'), ('HCP-D', 'tfMRIEMOTIONPA', 'EMOTION'),
        ('HCP-D', 'tfMRIGUESSINGPA', 'GUESSING'), ('HCP-YA', 'tfMRIWMLR', 'WM'),
        ('HCP-YA', 'tfMRIWMRL', 'WM'), ('HCP-YA', 'tfMRIEMOTIONLR', 'EMOTION'),
        ('HCP-YA', 'tfMRISOCIALRL', 'SOCIAL')])
    def test_the_task_being_analysed_picks_its_match(self, dataset, task, match):
        assert rest_window_match_for(dataset, task) == match

    @pytest.mark.parametrize('dataset,task', [('HCP-YA', 'tfMRIMOVIE17TAP'), ('HCP-D', 'tfMRIWMLR'),
                                              ('WAND', 'reversallearning')])
    def test_an_unmatched_task_raises(self, dataset, task):
        with pytest.raises(ValueError, match='no length-matched rest'):
            rest_window_match_for(dataset, task)

    def test_note_says_what_the_file_is(self):
        assert 'Nothing projected' in REST_WINDOW_NOTE


class TestResolveRestWindows:
    def test_hcpya_window_is_the_wm_control_window(self, rng):
        """[175, 580): the scan-onset offset, then WM's 405 frames -- the window restctrl uses."""
        ts = _hcpya_subject(rng)
        proc = _stub('HCP-YA', ts)
        spans = proc.resolve_rest_windows(verbose=False)
        rest = [k for k in ts if 'condition-rest' in k]
        assert set(spans) == set(REST_WINDOW_MATCH['HCP-YA', 'MRI3T'])
        assert set(spans['WM']) == set(rest)                # task scans get no window
        # The subject's 176-frame EMOTION LR run gets the window its restctrl uses, [175, 351).
        assert all(spans['EMOTION'][k] == (175, 351) for k in rest)
        assert all(spans[m] == {} for m in ('GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL'))
        for k in rest:
            assert spans['WM'][k] == (175, 580)
            meta = proc.rest_window_meta['WM'][k]
            assert meta['target_frames'] == 405 and meta['source_window'] == [175, 580]
            assert meta['offset_frames'] == 175 and meta['offset_seconds'] == 126.0
            assert meta['source_run_frames'] == 1200
            assert meta['rest_match'] == 'WM'
            assert meta['matched_task'] == ['tfMRIWMLR', 'tfMRIWMRL']

    def test_hcpd_gets_one_window_per_task_all_from_frame_zero(self, rng):
        ts = _hcpd_subject(rng)
        proc = _stub('HCP-D', ts)
        spans = proc.resolve_rest_windows(verbose=False)
        rest = {k for k in ts if 'condition-rest' in k}
        assert {m: sorted(set(s.values())) for m, s in spans.items()} == {
            'CARIT': [(0, 290)], 'EMOTION': [(0, 168)], 'GUESSING': [(0, 270)]}
        for match, n in (('CARIT', 290), ('EMOTION', 168), ('GUESSING', 270)):
            assert set(spans[match]) == rest                # every rest run, once per match
            for k in rest:
                meta = proc.rest_window_meta[match][k]
                assert meta['rest_match'] == match and meta['target_frames'] == n
                assert meta['matched_task'] == list(REST_WINDOW_MATCH[('HCP-D', 'MRI3T')][match])

    def test_target_is_read_from_the_data_not_assumed(self, rng):
        """Two matched runs of different lengths: the window takes the shorter, so it never
        exceeds either task run."""
        proc = _stub('HCP-YA', _hcpya_subject(rng, wm=(405, 400)))
        assert set(proc.resolve_rest_windows(verbose=False)['WM'].values()) == {(175, 575)}

    def test_7t_rest_is_left_alone(self, rng):
        ts = _hcpya_subject(rng)
        ts[_key('100206', 'rfMRIREST17TPA', 'rest', modality='MRI7T', tr='1.0')] = \
            rng.standard_normal((5, 900))
        spans = _stub('HCP-YA', ts).resolve_rest_windows(verbose=False)
        assert not any('MRI7T' in k for by_scan in spans.values() for k in by_scan)

    def test_a_missing_task_run_costs_that_match_only(self, rng):
        spans = _stub('HCP-D', _hcpd_subject(rng, tasks=HCPD_TASKS[:1])).resolve_rest_windows(
            verbose=False)
        assert len(spans['CARIT']) == 2
        assert spans['EMOTION'] == {} and spans['GUESSING'] == {}

    def test_no_matched_task_run_means_no_window_not_an_error(self, rng):
        ts = {_key('100206', 'rfMRIREST1LR', 'rest'): rng.standard_normal((5, 1200)),
              _key('100206', 'tfMRIEMOTIONLR', 'task'): rng.standard_normal((5, 176))}
        spans = _stub('HCP-YA', ts).resolve_rest_windows(verbose=False)
        rest = _key('100206', 'rfMRIREST1LR', 'rest')
        assert spans['WM'] == {} and spans['EMOTION'] == {rest: (175, 351)}
        assert all(spans[m] == {} for m in ('GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL'))

    def test_a_window_that_does_not_fit_raises(self, rng):
        """175 + 405 > 500: a wrong table, not a failed scan -- raised, never swallowed."""
        with pytest.raises(ValueError, match='exceeds'):
            _stub('HCP-YA', _hcpya_subject(rng, rest_frames=500)).resolve_rest_windows(verbose=False)

    def test_needs_loaded_timeseries(self):
        proc = _stub('HCP-YA', {})
        proc.timeseries_data = None
        with pytest.raises(ValueError, match='no timeseries data'):
            proc.resolve_rest_windows(verbose=False)

    def test_every_window_carries_complete_provenance(self, rng):
        proc = _stub('HCP-D', _hcpd_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        required = {'window_kind', 'window_note', 'rest_match', 'matched_task', 'target_frames',
                    'source_window', 'source_run_frames', 'offset_seconds', 'offset_frames', 'tr',
                    'modality', 'dataset'}
        for by_scan in proc.rest_window_meta.values():
            for meta in by_scan.values():
                assert required <= set(meta)


class TestWindowACF:
    def test_the_cached_acf_is_compute_acf_of_the_slice(self, rng):
        ts = _hcpya_subject(rng)
        proc = _stub('HCP-YA', ts)
        proc.resolve_rest_windows(verbose=False)
        proc.compute_autocorrelation_function(use_fft=True, use_rest_window='WM',
                                              force_recompute=True, save_outputs=False,
                                              verbose=False)
        for k, v in ts.items():
            if 'condition-rest' in k:
                np.testing.assert_array_equal(proc.autocorr_dict[k],
                                              np.atleast_2d(compute_acf(v[:, 175:580], use_fft=True)))
            else:
                assert proc.autocorr_dict[k] is None        # skipped, not failed

    def test_each_match_caches_its_own_slice_of_the_same_run(self, rng):
        ts = _hcpd_subject(rng)
        proc = _stub('HCP-D', ts)
        proc.resolve_rest_windows(verbose=False)
        for match, n in (('CARIT', 290), ('EMOTION', 168), ('GUESSING', 270)):
            proc.compute_autocorrelation_function(use_fft=True, use_rest_window=match,
                                                  force_recompute=True, save_outputs=False,
                                                  verbose=False)
            for k, v in ts.items():
                if 'condition-rest' in k:
                    np.testing.assert_array_equal(proc.autocorr_dict[k],
                                                  np.atleast_2d(compute_acf(v[:, :n], use_fft=True)))

    def test_a_whole_run_window_reproduces_full_length_rest_exactly(self, rng):
        """The parity guarantee: a window IS a full-length run when it spans the run, because both
        go through the same compute_acf call. HCP-D (offset 0) with CARIT as long as rest."""
        ts = _hcpd_subject(rng, rest_frames=290, tasks=HCPD_TASKS[:1])
        rest_key = next(k for k in ts if 'condition-rest' in k)
        proc = _stub('HCP-D', ts)
        proc.compute_autocorrelation_function(use_fft=True, force_recompute=True,
                                              save_outputs=False, verbose=False)
        full = proc.autocorr_dict[rest_key]
        proc.resolve_rest_windows(verbose=False)
        proc.compute_autocorrelation_function(use_fft=True, use_rest_window='CARIT',
                                              force_recompute=True, save_outputs=False,
                                              verbose=False)
        np.testing.assert_array_equal(proc.autocorr_dict[rest_key], full)

    def test_windows_must_be_resolved_first(self, rng):
        with pytest.raises(ValueError, match='resolve_rest_windows'):
            _stub('HCP-YA', _hcpya_subject(rng)).compute_autocorrelation_function(
                use_rest_window='WM', save_outputs=False, verbose=False)

    @pytest.mark.parametrize('bad', [True, 1])
    def test_the_window_is_named_by_its_match_not_switched_on(self, rng, bad):
        proc = _stub('HCP-YA', _hcpya_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        with pytest.raises(ValueError, match='name of a matched task'):
            proc.compute_autocorrelation_function(use_rest_window=bad, save_outputs=False,
                                                  verbose=False)

    def test_a_match_this_dataset_does_not_have_raises(self, rng):
        proc = _stub('HCP-YA', _hcpya_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        with pytest.raises(ValueError, match="no length-matched rest windows named 'CARIT'"):
            proc.compute_autocorrelation_function(use_rest_window='CARIT', save_outputs=False,
                                                  verbose=False)

    @pytest.mark.parametrize('other', [dict(use_residuals=True), dict(use_rest_control='full')])
    def test_one_source_per_file(self, rng, other):
        proc = _stub('HCP-YA', _hcpya_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        with pytest.raises(ValueError, match='mutually exclusive'):
            proc.compute_autocorrelation_function(use_rest_window='WM', save_outputs=False,
                                                  verbose=False, **other)

    def test_written_files_carry_the_window_provenance(self, rng):
        proc = _stub('HCP-YA', _hcpya_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        saved = []
        proc._save_subject_output = lambda **kw: saved.append(kw)
        proc.compute_autocorrelation_function(use_fft=True, use_rest_window='WM',
                                              force_recompute=True, verbose=False)
        assert len(saved) == 2 and all(s['task'].startswith('rfMRIREST') for s in saved)
        for s in saved:
            assert s['output_type'] == 'autocorr_restlen-WM'
            assert s['metadata']['output_type'] == 'autocorr_restlen-WM'
            assert s['metadata']['source_window'] == [175, 580]
            assert s['metadata']['window_kind'] == 'length_matched_rest'
            assert s['metadata']['rest_match'] == 'WM'

    def test_a_filtered_window_records_the_source_run_length(self, rng):
        """A bandpassed window inherits the bp- token and records the whole run it was cut from."""
        proc = _stub('HCP-D', _hcpd_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        saved = []
        proc._save_subject_output = lambda **kw: saved.append(kw)
        proc.compute_autocorrelation_function(use_fft=True, use_rest_window='EMOTION', lowpass=0.08,
                                              highpass=0.01, force_recompute=True, verbose=False)
        assert len(saved) == 2
        for s in saved:
            assert s['output_type'] == 'autocorr_bp-0.01-0.08hz_restlen-EMOTION'
            assert s['metadata']['source_run_frames'] == 478
            assert s['metadata']['source_window'] == [0, 168]

    def test_defaults_are_unchanged(self):
        assert inspect.signature(Timescales._run_for_all_scans).parameters[
            'use_rest_window'].default is None
        assert inspect.signature(Timescales.compute_autocorrelation_function).parameters[
            'use_rest_window'].default is None
        params = inspect.signature(load_notebook_data).parameters
        assert params['rest_length'].default == 'full' and params['rest_match'].default is None


class _LoaderStub:
    """What _substitute_rest_windows reads: the dataset and get_precomputed_outputs."""

    def __init__(self, dataset, windows):
        self.dataset, self._windows, self.requested = dataset, windows, []

    def get_precomputed_outputs(self, output_type):
        self.requested.append(output_type)
        self.data_dict = dict(self._windows)


class TestLoaderSubstitution:
    def _dicts(self):
        rest3t = _key('100206', 'rfMRIREST1LR', 'rest')
        rest7t = _key('100206', 'rfMRIREST17TPA', 'rest', modality='MRI7T', tr='1.0')
        task = _key('100206', 'tfMRIWMLR', 'task')
        full = {rest3t: 'full-3T', rest7t: 'full-7T', task: 'task-INT'}
        return rest3t, rest7t, task, full

    def test_swaps_covered_rest_only(self):
        rest3t, rest7t, task, full = self._dicts()
        proc = _LoaderStub('HCP-YA', {rest3t: 'window-3T', rest7t: None, task: None})
        out = _substitute_rest_windows(proc, dict(full), 'timescale_auc_zero_crossing', 'WM')
        assert out[rest3t] == 'window-3T'
        assert out[rest7t] == 'full-7T'                     # uncovered: full length kept
        assert out[task] == 'task-INT'
        assert proc.requested == ['timescale_auc_zero_crossing_restlen-WM']

    def test_a_missing_window_becomes_none_so_the_subject_is_dropped(self):
        rest3t, _, _, full = self._dicts()
        out = _substitute_rest_windows(_LoaderStub('HCP-YA', {}), dict(full),
                                       'timescale_auc_zero_crossing', 'WM')
        assert out[rest3t] is None

    def test_the_frequency_token_is_carried(self):
        rest3t, _, _, full = self._dicts()
        proc = _LoaderStub('HCP-YA', {rest3t: 'w'})
        _substitute_rest_windows(proc, dict(full), 'timescale_auc_zero_crossing_bp-0.01-0.08hz', 'WM')
        assert proc.requested == ['timescale_auc_zero_crossing_bp-0.01-0.08hz_restlen-WM']

    def test_the_requested_match_is_the_one_loaded(self):
        rest = _key('HCD001', 'rfMRIREST1PA', 'rest', tr='0.8')
        proc = _LoaderStub('HCP-D', {rest: 'emotion-window'})
        out = _substitute_rest_windows(proc, {rest: 'full'},
                                       'timescale_auc_zero_crossing_bp-0.01-0.08hz', 'EMOTION')
        assert out[rest] == 'emotion-window'
        assert proc.requested == ['timescale_auc_zero_crossing_bp-0.01-0.08hz_restlen-EMOTION']

    def test_no_covered_rest_for_the_match_raises(self):
        wand = {_key('97923', 'rest', 'rest', tr='2.0'): 'full'}
        with pytest.raises(ValueError, match='no WAND rest scan'):
            _substitute_rest_windows(_LoaderStub('WAND', {}), wand, 'timescale_auc_zero_crossing', 'WM')


class TestResolveRestMatch:
    def test_a_single_match_is_picked_automatically(self, monkeypatch):
        monkeypatch.setattr(timescales_module, 'REST_WINDOW_MATCH', {('X', 'MRI3T'): {'WM': ('t',)}})
        assert _resolve_rest_match('X', None) == 'WM'

    def test_hcpya_now_needs_an_explicit_choice_too(self):
        """Every HCP-YA 3T task is matched, so None no longer means WM."""
        with pytest.raises(ValueError, match='several tasks'):
            _resolve_rest_match('HCP-YA', None)

    def test_several_matches_need_an_explicit_choice(self):
        """HCP-D has three: guessing would load CARIT-length rest for an EMOTION analysis."""
        with pytest.raises(ValueError, match='several tasks'):
            _resolve_rest_match('HCP-D', None)

    def test_an_explicit_choice_is_checked(self):
        assert _resolve_rest_match('HCP-D', 'EMOTION') == 'EMOTION'
        with pytest.raises(ValueError, match="no rest windows matched to 'WM'"):
            _resolve_rest_match('HCP-D', 'WM')

    def test_an_uncovered_dataset_raises(self):
        with pytest.raises(ValueError, match='no length-matched rest'):
            _resolve_rest_match('WAND', None)


class TestWindowMetric:
    """Per-scan metrics on the same windows, named with the same tokens as the window's ACF."""

    @staticmethod
    def _saved_names(proc, **kwargs):
        saved = []
        proc._save_subject_output = lambda **kw: saved.append(kw)
        proc.compute_timeseries_measures(method='var', force_recompute=True, verbose=False, **kwargs)
        return {s['output_type'] for s in saved}

    def test_window_variance_is_the_variance_of_the_slice(self, rng):
        ts = _hcpya_subject(rng)
        proc = _stub('HCP-YA', ts)
        proc.resolve_rest_windows(verbose=False)
        proc.compute_timeseries_measures(method='var', use_rest_window='WM', force_recompute=True,
                                         save_outputs=False, verbose=False)
        for k, v in ts.items():
            if 'condition-rest' in k:
                np.testing.assert_array_equal(proc.data_dict[k], np.var(v[:, 175:580], axis=1))
            else:
                assert proc.data_dict[k] is None

    def test_names_mirror_the_acf_and_unfiltered_names_are_unchanged(self, rng):
        ts = _hcpya_subject(rng)
        proc = _stub('HCP-YA', ts)
        proc.timeseries_residuals = {k: v for k, v in ts.items() if 'condition-task' in k}
        proc.resolve_rest_windows(verbose=False)
        assert self._saved_names(proc) == {'metric_var'}
        assert self._saved_names(proc, use_residuals=True) == {'metric_var_resids'}
        assert self._saved_names(proc, use_rest_window='WM') == {'metric_var_restlen-WM'}
        bp = dict(highpass=0.01, lowpass=0.08)
        assert self._saved_names(proc, use_residuals=True, **bp) == {'metric_var_bp-0.01-0.08hz_resids'}
        assert (self._saved_names(proc, use_rest_window='WM', **bp)
                == {'metric_var_bp-0.01-0.08hz_restlen-WM'})

    def test_residual_and_window_are_mutually_exclusive(self, rng):
        proc = _stub('HCP-YA', _hcpya_subject(rng))
        proc.resolve_rest_windows(verbose=False)
        with pytest.raises(ValueError, match='mutually exclusive'):
            proc.compute_timeseries_measures(method='var', use_residuals=True, use_rest_window='WM',
                                             save_outputs=False, verbose=False)

    def test_windows_must_be_resolved_first(self, rng):
        with pytest.raises(ValueError, match='resolve_rest_windows'):
            _stub('HCP-YA', _hcpya_subject(rng)).compute_timeseries_measures(
                method='var', use_rest_window='WM', save_outputs=False, verbose=False)

    def test_the_filter_token_is_the_one_the_acf_uses(self):
        assert frequency_filter_token() is None
        assert frequency_filter_token(0.01, 0.08) == 'bp-0.01-0.08hz'
        assert frequency_filter_token(highpass=0.01) == 'hp-0.01hz'
        assert frequency_filter_token(lowpass=0.08) == 'lp-0.08hz'
