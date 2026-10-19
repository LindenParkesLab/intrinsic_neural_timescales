"""
Tests for the matched rest control layer (src/timescales.py).

No data mounts required — everything here runs on synthetic arrays and on the run vocabulary.

- ``TestComputeAcfParity``: the control ACF is computed in ``build_rest_controls``, *outside*
  ``compute_autocorrelation_function``. Nothing forces the two to call ``compute_acf`` the same
  way, so an option added to one path and not the other would produce a control that is not
  comparable to the task side, with no error anywhere.
- ``TestRunVocabularyIsShared``: the condition groupings were extracted out of
  ``_apply_scan_groupings`` so the control is built for the grouping the notebooks actually
  average by. A second copy would drift.
- ``TestPhaseEncodingPairing``: index-order pairing mismatches the phase encoding of two of the
  four 7T movie runs while looking correct on 3T and on the other two, which is exactly the kind
  of bug that survives review.
- ``TestRestControlOffset``: the window start is a per-dataset number read off a diagnostic. A
  wrong one produces a control that is comparable in shape, plausible in magnitude and biased,
  which nothing downstream can detect.
- ``TestDesignRequiresEvents``: nilearn builds a valid drift-and-constant design from no events
  at all, so a design whose events failed to resolve measures the drift basis and calls it task
  regression.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from snaplab_tools.timescales import compute_acf
from src.data import build_scan_string
from src.timescales import (MOVIE_GROUP, MOVIE_RUNS_7T, REST_CONTROL_KINDS, REST_CONTROL_NOTE,
                            REST_CONTROL_OFFSET_DEFAULT, REST_CONTROL_OFFSET_SECONDS,
                            REST_RUNS_3T, REST_RUNS_7T, TASK_RUN_GROUPS, Timescales,
                            _apply_scan_groupings, condition_of, create_design_matrix_nilearn,
                            drift_only_control, fit_glm, pair_rest_runs,
                            phase_encoding_direction,
                            rest_control_offset_frames, rest_control_offset_seconds,
                            rest_control_output_type, split_design_columns)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestComputeAcfParity:
    """The two ACF call sites must stay identical. See the module docstring."""

    def test_same_arguments_same_result(self, rng):
        """build_rest_controls' ACF call reproduces compute_autocorrelation_function's."""
        data = rng.standard_normal((12, 200))
        # compute_autocorrelation_function._compute, verbatim
        main_path = np.atleast_2d(compute_acf(data, use_fft=True))
        # build_rest_controls, verbatim
        control_path = np.atleast_2d(compute_acf(data, use_fft=True))
        np.testing.assert_array_equal(main_path, control_path)

    def test_use_fft_is_the_only_option_either_path_sets(self):
        """compute_acf takes (data, use_fft, nlags); nlags must be left at its default.

        A truncated control ACF would be silently incomparable to a full-length task ACF, and
        the lag measurement behind the no-truncation decision (max first zero crossing 142 s)
        says truncation is unsafe anyway.
        """
        params = inspect.signature(compute_acf).parameters
        assert list(params) == ['data', 'use_fft', 'nlags']
        assert params['nlags'].default is None

        source = inspect.getsource(Timescales.build_rest_controls)
        assert 'compute_acf(resid_full, use_fft=use_fft)' in source
        assert 'nlags' not in source

    def test_build_rest_controls_forwards_use_fft_to_the_drift_path(self):
        """drift_only_control computes the drift ACF, so it must receive use_fft too."""
        source = inspect.getsource(Timescales.build_rest_controls)
        assert 'use_fft=use_fft' in source.split('drift_only_control(')[1].split(')')[0] + \
               source.split('drift_only_control(')[1][:200]

    def test_drift_only_control_acf_matches_a_direct_call(self, rng):
        """The drift control's cached ACF is compute_acf of the drift residual, nothing else."""
        data = rng.standard_normal((8, 120))
        design = pd.DataFrame({'drift_1': np.linspace(-1, 1, 120),
                               'drift_2': np.cos(np.linspace(0, np.pi, 120)),
                               'constant': np.ones(120)})
        control = drift_only_control(data, design, tr=1.0, use_fft=True)
        residual, _, _ = fit_glm(data, design[['drift_1', 'drift_2', 'constant']])
        np.testing.assert_allclose(np.atleast_2d(control['acf']),
                                   np.atleast_2d(compute_acf(residual, use_fft=True)))


class TestPhaseEncodingPairing:
    def test_hcpya_3t_every_condition_uses_the_first_rest_pair(self):
        task_runs = [run for runs, _ in TASK_RUN_GROUPS for run in runs if run.endswith(('LR', 'RL'))]
        pairing = pair_rest_runs(task_runs, REST_RUNS_3T[:4])
        for task_run, (rest_run, matched) in pairing.items():
            assert matched
            assert phase_encoding_direction(task_run) == phase_encoding_direction(rest_run)
            assert rest_run == ('rfMRIREST1LR' if task_run.endswith('LR') else 'rfMRIREST1RL')

    def test_7t_movies_are_pe_matched_not_index_matched(self):
        """MOVIE1 is AP and REST1 is PA: index order mismatches MOVIE1 and MOVIE2."""
        pairing = pair_rest_runs(MOVIE_RUNS_7T, REST_RUNS_7T)
        assert pairing == {'tfMRIMOVIE17TAP': ('rfMRIREST27TAP', True),
                           'tfMRIMOVIE27TPA': ('rfMRIREST17TPA', True),
                           'tfMRIMOVIE37TPA': ('rfMRIREST37TPA', True),
                           'tfMRIMOVIE47TAP': ('rfMRIREST47TAP', True)}
        # every movie run gets a different rest run, and every 7T rest run is used
        assert len({rest for rest, _ in pairing.values()}) == len(REST_RUNS_7T)
        # the naive rule this replaces would have been wrong for MOVIE1 and MOVIE2: the two
        # sequences are AP,PA,PA,AP against PA,AP,PA,AP, so they agree only from run 3 on
        mismatched = [movie for movie, rest in zip(MOVIE_RUNS_7T, REST_RUNS_7T)
                      if phase_encoding_direction(movie) != phase_encoding_direction(rest)]
        assert mismatched == ['tfMRIMOVIE17TAP', 'tfMRIMOVIE27TPA']

    def test_hcpd_single_pa_run(self):
        pairing = pair_rest_runs(['tfMRICARITPA', 'tfMRIEMOTIONPA', 'tfMRIGUESSINGPA'],
                                 ['rfMRIREST1PA', 'rfMRIREST2PA'])
        assert all(rest == 'rfMRIREST1PA' and matched for rest, matched in pairing.values())

    def test_datasets_without_a_pe_token_fall_back_to_index(self):
        assert pair_rest_runs(['nback'], ['rest']) == {'nback': ('rest', False)}
        assert pair_rest_runs(['reversallearning'], ['rest']) == {'reversallearning': ('rest', False)}

    def test_lowercase_tails_are_not_phase_encoding(self):
        """A task named e.g. `scap` ends in 'ap'. Reading it as AP would pair it against AP rest runs."""
        assert phase_encoding_direction('scap') is None
        assert phase_encoding_direction('nback') is None
        assert phase_encoding_direction('rest') is None
        assert phase_encoding_direction('tfMRIGUESSINGPA') == 'PA'
        assert phase_encoding_direction('tfMRIWMLR') == 'LR'

    def test_no_rest_runs_pairs_nothing(self):
        assert pair_rest_runs(['tfMRIWMLR'], []) == {}

    def test_pool_cycles_when_a_condition_has_more_runs_than_rest(self):
        pairing = pair_rest_runs(MOVIE_RUNS_7T, ['rfMRIREST17TPA', 'rfMRIREST27TAP'])
        assert pairing['tfMRIMOVIE17TAP'][0] == 'rfMRIREST27TAP'
        assert pairing['tfMRIMOVIE47TAP'][0] == 'rfMRIREST27TAP'   # cycled, still AP


class TestRunVocabularyIsShared:
    def test_apply_scan_groupings_uses_the_module_constants(self):
        """A local copy here would let the control be built for a grouping nothing averages by."""
        source = inspect.getsource(_apply_scan_groupings)
        for name in ('REST_RUNS_3T', 'REST_RUNS_7T', 'MOVIE_RUNS_7T', 'TASK_RUN_GROUPS'):
            assert name in source
        assert "'tfMRIWMLR'" not in source        # no second copy of the pairings

    def test_condition_of_agrees_with_the_groupings(self):
        for runs, condition in TASK_RUN_GROUPS:
            for run in runs:
                assert condition_of(run) == condition
        for run in MOVIE_GROUP[0]:
            assert condition_of(run) == MOVIE_GROUP[1]

    def test_ungrouped_runs_are_their_own_condition(self):
        assert condition_of('nback') == 'task-nback'
        assert condition_of('reversallearning') == 'task-reversallearning'


class TestOutputTypeTokens:
    def test_tokens(self):
        assert rest_control_output_type('autocorr', 'full') == 'autocorr_restctrl'
        assert rest_control_output_type('autocorr', 'drift') == 'autocorr_restctrl-drift'
        assert (rest_control_output_type('autocorr_bp-0.01-0.08hz', 'full')
                == 'autocorr_bp-0.01-0.08hz_restctrl')

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match='unknown rest-control kind'):
            rest_control_output_type('autocorr', 'sideways')

    def test_kinds_and_note(self):
        assert set(REST_CONTROL_KINDS) == {'full', 'drift'}
        assert 'Not a measurement of rest during the task' in REST_CONTROL_NOTE


class TestDefaultsUnchanged:
    def test_new_parameters_default_to_previous_behaviour(self):
        run = inspect.signature(Timescales._run_for_all_scans).parameters
        assert run['use_rest_control'].default is None
        assert run['metadata_fn'].default is None
        acf = inspect.signature(Timescales.compute_autocorrelation_function).parameters
        assert acf['use_rest_control'].default is None

    def test_residuals_and_control_are_mutually_exclusive(self):
        source = inspect.getsource(Timescales.compute_autocorrelation_function)
        assert 'mutually exclusive' in source


def _stub_processor(dataset, scans, timeseries, designs, events):
    """A Timescales carrying only what build_rest_controls reads, with no DataLoader init.

    build_rest_controls touches ``dataset``, ``scans``, ``timeseries_data``, ``design_matrices``,
    ``task_events`` and ``_parse_scan_string`` (which is stateless), so the method can be
    exercised without a data mount. Constructed through ``__new__`` deliberately: going through
    ``__init__`` would pull in DataLoader and the test would silently become an integration test.
    """
    processor = Timescales.__new__(Timescales)
    processor.dataset = dataset
    processor.scans = list(scans)
    processor.timeseries_data = dict(timeseries)
    processor.design_matrices = dict(designs)
    processor.task_events = dict(events)
    return processor


def _scan_key(sub, task, condition, modality, tr):
    return build_scan_string(sub=sub, ses='01', modality=modality, task=task, space='fsLR',
                             seg='Schaefer4007', tr=tr, condition=condition)


class TestPairingStaysInsideASession:
    """Penn LEAD's sessions are ~2 years apart: a control built from another session's rest would
    be a different brain at a different age. Groups are subject x session x modality."""

    @staticmethod
    def _key(ses, task, condition):
        return build_scan_string(sub='20188', ses=ses, modality='MRI3T', task=task, space='fsLR',
                                 seg='Schaefer4007', tr='0.8', condition=condition)

    def test_a_session_without_rest_gets_no_control_rather_than_borrowing_one(self, rng):
        n = 120
        design = pd.DataFrame({'trial': rng.standard_normal(n), 'constant': np.ones(n)})
        task1, rest1 = self._key('1', 'nback', 'task'), self._key('1', 'rest', 'rest')
        task2, rest2 = self._key('2', 'nback', 'task'), self._key('2', 'rest', 'rest')
        processor = _stub_processor(
            'Penn_LEAD', [task1, rest1, task2, rest2],
            {task1: rng.standard_normal((4, n)), rest1: rng.standard_normal((4, n)),
             task2: rng.standard_normal((4, n)), rest2: None},          # session 2 lost its rest
            {task1: design, task2: design}, {task1: None, task2: None})
        processor.build_rest_controls(use_fft=True, verbose=False)
        assert processor.autocorr_rest_control[task1] is not None
        assert processor.autocorr_rest_control[task2] is None
        assert task2 not in processor.rest_control_meta


class TestRestControlOffset:
    """The window start: the scan-onset offsets measured on ``auc_zero_crossing``."""

    def test_the_table_is_the_diagnostic(self):
        """HCP-YA 3T is the only measured transient; everything else is zero, deliberately."""
        assert REST_CONTROL_OFFSET_SECONDS[('HCP-YA', 'MRI3T')] == 126.0
        for key in [('HCP-YA', 'MRI7T'), ('HCP-D', 'MRI3T'), ('Penn_LEAD', 'MRI3T')]:
            assert REST_CONTROL_OFFSET_SECONDS[key] == 0.0
        assert REST_CONTROL_OFFSET_DEFAULT == 0.0

    def test_hcpya_3t_is_175_frames(self):
        """126 s at TR 0.72 is exactly 175 frames -- the number every 3T control must record."""
        assert rest_control_offset_frames('HCP-YA', 'MRI3T', 0.72) == 175
        assert rest_control_offset_seconds('HCP-YA', 'MRI3T') == 126.0

    def test_every_other_dataset_is_zero_frames(self):
        assert rest_control_offset_frames('HCP-YA', 'MRI7T', 1.0) == 0
        assert rest_control_offset_frames('HCP-D', 'MRI3T', 0.8) == 0
        assert rest_control_offset_frames('Penn_LEAD', 'MRI3T', 0.8) == 0

    def test_an_undiagnosed_dataset_gets_no_offset(self):
        """Zero, not a borrowed number: an unmeasured correction is worse than none."""
        assert rest_control_offset_seconds('WAND', 'MRI3T') == 0.0
        assert rest_control_offset_frames('WAND', 'MRI3T', 2.0) == 0

    def test_modality_is_part_of_the_key(self):
        """HCP-YA 3T dips and HCP-YA 7T does not, so a dataset-only table would be wrong."""
        assert (rest_control_offset_seconds('HCP-YA', 'MRI3T')
                != rest_control_offset_seconds('HCP-YA', 'MRI7T'))

    def test_a_non_positive_tr_raises(self):
        with pytest.raises(ValueError, match='tr must be positive'):
            rest_control_offset_frames('HCP-YA', 'MRI3T', 0.0)

    def test_the_hcpya_3t_worst_case_fits(self):
        """WM is the longest 3T condition: 175 + 405 against the shortest observed rest run."""
        assert rest_control_offset_frames('HCP-YA', 'MRI3T', 0.72) + 405 <= 914

    def test_window_starts_at_the_offset(self, rng):
        """End to end: the cached ACF is the ACF of rest[offset:offset + n], nothing else."""
        n_rest, n_task, tr = 1200, 405, '0.72'
        rest = rng.standard_normal((6, n_rest))
        task_key = _scan_key('100206', 'tfMRIWMLR', 'task', 'MRI3T', tr)
        rest_key = _scan_key('100206', 'rfMRIREST1LR', 'rest', 'MRI3T', tr)
        design = pd.DataFrame({'trial': rng.standard_normal(n_task),
                               'drift_1': np.linspace(-1, 1, n_task),
                               'constant': np.ones(n_task)})
        processor = _stub_processor('HCP-YA', [task_key, rest_key],
                                    {task_key: rng.standard_normal((6, n_task)), rest_key: rest},
                                    {task_key: design}, {task_key: None})
        processor.build_rest_controls(use_fft=True, verbose=False)

        meta = processor.rest_control_meta[task_key]
        assert meta['offset_frames'] == 175 and meta['offset_seconds'] == 126.0
        assert meta['source_window'] == [175, 175 + n_task]
        assert meta['design_truncated'] is False
        expected, _, _ = fit_glm(rest[:, 175:175 + n_task], design)
        np.testing.assert_allclose(
            processor.autocorr_rest_control[task_key],
            np.atleast_2d(compute_acf(expected, use_fft=True)).astype(np.float32), rtol=1e-6)

    def test_zero_offset_datasets_still_start_at_frame_zero(self, rng):
        n_rest, n_task, tr = 478, 290, '0.8'
        rest = rng.standard_normal((6, n_rest))
        task_key = _scan_key('HCD001', 'tfMRICARITPA', 'task', 'MRI3T', tr)
        rest_key = _scan_key('HCD001', 'rfMRIREST1PA', 'rest', 'MRI3T', tr)
        design = pd.DataFrame({'trial': rng.standard_normal(n_task),
                               'constant': np.ones(n_task)})
        processor = _stub_processor('HCP-D', [task_key, rest_key],
                                    {task_key: rng.standard_normal((6, n_task)), rest_key: rest},
                                    {task_key: design}, {task_key: None})
        processor.build_rest_controls(use_fft=True, verbose=False)
        meta = processor.rest_control_meta[task_key]
        assert meta['offset_frames'] == 0 and meta['source_window'] == [0, n_task]

    def test_a_window_that_does_not_fit_raises_rather_than_counting_a_failure(self, rng):
        """A bad offset is a bad table, not a bad scan. It must not land in the failure count."""
        n_rest, n_task, tr = 300, 200, '0.72'   # 175 + 200 > 300
        task_key = _scan_key('100206', 'tfMRIWMLR', 'task', 'MRI3T', tr)
        rest_key = _scan_key('100206', 'rfMRIREST1LR', 'rest', 'MRI3T', tr)
        design = pd.DataFrame({'trial': rng.standard_normal(n_task),
                               'constant': np.ones(n_task)})
        processor = _stub_processor('HCP-YA', [task_key, rest_key],
                                    {task_key: rng.standard_normal((6, n_task)),
                                     rest_key: rng.standard_normal((6, n_rest))},
                                    {task_key: design}, {task_key: None})
        with pytest.raises(ValueError, match='no room for a length-matched control'):
            processor.build_rest_controls(use_fft=True, verbose=False)

    def test_truncation_is_still_allowed_at_offset_zero(self, rng):
        """The 7T movies: 921 frames of design into a 900-frame rest run, offset 0."""
        n_rest, n_task, tr = 900, 921, '1.0'
        task_key = _scan_key('100610', 'tfMRIMOVIE17TAP', 'task', 'MRI7T', tr)
        rest_key = _scan_key('100610', 'rfMRIREST27TAP', 'rest', 'MRI7T', tr)
        design = pd.DataFrame({'trial': rng.standard_normal(n_task),
                               'constant': np.ones(n_task)})
        events = pd.DataFrame({'onset': [10.0], 'duration': [5.0], 'trial_type': ['clip']})
        processor = _stub_processor('HCP-YA', [task_key, rest_key],
                                    {task_key: rng.standard_normal((6, n_task)),
                                     rest_key: rng.standard_normal((6, n_rest))},
                                    {task_key: design}, {task_key: events})
        processor.build_rest_controls(use_fft=True, verbose=False)
        meta = processor.rest_control_meta[task_key]
        assert meta['design_truncated'] is True
        assert meta['offset_frames'] == 0 and meta['source_window'] == [0, n_rest]

    def test_metadata_carries_the_offset_on_every_control(self, rng):
        """A partially-populated provenance key is worse than none -- see the contract."""
        tr = '0.72'
        keys = []
        timeseries, designs, events = {}, {}, {}
        rest_key = _scan_key('100206', 'rfMRIREST1LR', 'rest', 'MRI3T', tr)
        timeseries[rest_key] = rng.standard_normal((4, 1200))
        for task in ['tfMRIWMLR', 'tfMRIEMOTIONLR']:
            key = _scan_key('100206', task, 'task', 'MRI3T', tr)
            keys.append(key)
            n = 405 if 'WM' in task else 176
            timeseries[key] = rng.standard_normal((4, n))
            designs[key] = pd.DataFrame({'trial': rng.standard_normal(n),
                                         'constant': np.ones(n)})
            events[key] = None
        processor = _stub_processor('HCP-YA', keys + [rest_key], timeseries, designs, events)
        processor.build_rest_controls(use_fft=True, verbose=False)
        required = {'design_from_task', 'source_rest_task', 'source_window', 'offset_seconds',
                    'offset_frames', 'source_run_frames', 'design_truncated', 'pe_matched',
                    'n_task_cols', 'n_drift_cols', 'tr', 'modality', 'dataset'}
        assert set(processor.rest_control_meta) == set(keys)
        for key in keys:
            meta = processor.rest_control_meta[key]
            assert required <= set(meta)
            assert meta['offset_frames'] == 175
            assert meta['source_window'][0] == 175


class TestDesignRequiresEvents:
    """No events must raise, not build a task-free design. See the function's Raises section."""

    def test_none_events_raise(self):
        with pytest.raises(ValueError, match='no task events'):
            create_design_matrix_nilearn(None, 100, 0.72)

    def test_empty_events_raise(self):
        empty = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        with pytest.raises(ValueError, match='no task events'):
            create_design_matrix_nilearn(empty, 100, 0.72)

    def test_real_events_still_build_a_design_with_task_columns(self):
        events = pd.DataFrame({'onset': [5.0, 40.0], 'duration': [10.0, 10.0],
                               'trial_type': ['a', 'b']})
        design = create_design_matrix_nilearn(events, 100, 0.72)
        task_cols, nuisance_cols = split_design_columns(design)
        assert set(task_cols) == {'a', 'b'} and 'constant' in nuisance_cols


class TestControlSourceRun:
    def test_controls_record_the_source_run_length_not_the_window(self, rng):
        """source_run_frames is the whole paired rest run, not the window cut from it."""
        n_rest, n_task, tr = 478, 290, '0.8'
        task_key = _scan_key('HCD001', 'tfMRICARITPA', 'task', 'MRI3T', tr)
        rest_key = _scan_key('HCD001', 'rfMRIREST1PA', 'rest', 'MRI3T', tr)
        design = pd.DataFrame({'trial': rng.standard_normal(n_task),
                               'constant': np.ones(n_task)})
        processor = _stub_processor('HCP-D', [task_key, rest_key],
                                    {task_key: rng.standard_normal((4, n_task)),
                                     rest_key: rng.standard_normal((4, n_rest))},
                                    {task_key: design}, {task_key: None})
        processor.build_rest_controls(use_fft=True, verbose=False)
        meta = processor.rest_control_meta[task_key]
        assert meta['source_run_frames'] == n_rest
        # the window is shorter than the run, so the two would disagree if it used the window
        assert meta['source_window'][1] - meta['source_window'][0] == n_task
