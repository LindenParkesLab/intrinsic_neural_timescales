"""
Unit and integration tests for data.py — AOMIC-PIOP2 dataset.

Test tiers
----------
Unit tests (no disk access):
    TestConfig       — YAML config loading and field values
    TestScanInfo     — condition derivation and the phase-encoding reader

Integration tests (require the AOMIC-PIOP2 mount, auto-skipped if unavailable):
    TestPathConstruction — get_scan_filepath template rendering and exact path assertions
    TestTaskEvents       — events resolve for the three tasks and are absent for rest
    TestDesignMatrix     — the events actually build a nilearn design with task regressors
    TestTimeseries       — loaded array shape and orientation
    TestBehavior         — participants.tsv + XCP-D QC motion join

Notes
-----
AOMIC-PIOP2 (ds002790) is single-session, so scans are registered with session=None and the
filename carries no ses- entity. Every file carries ``acq-seq``, which is written literally into
the filename template rather than promoted to a ScanInfo entity — see the YAML for why.

Run lengths are deliberately asserted loosely for ``stopsignal``: it is self-paced and varies
210–250 frames across subjects, and 10 of 226 runs are longer than the 240-frame rest run. Any
code that assumes a nominal length for it is wrong.
"""

import numpy as np
import pytest
from pathlib import Path

from src.data import DataLoader, ScanInfo, load_dataset_config
from src.timescales import create_design_matrix_nilearn, phase_encoding_direction
from tests.conftest import integration_aomic as integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A = '0001'    # has restingstate, stopsignal, workingmemory (no emomatching)
SUBJECT_B = '0002'    # has all four runs; its stopsignal has no unsuccesful_stop trials
#: Subject to read each task's events from — one that has all of that task's trial types.
#: stopsignal cannot use SUBJECT_B: see TestDesignMatrix.
EVENTS_SUBJECT = {'stopsignal': SUBJECT_A, 'workingmemory': SUBJECT_B, 'emomatching': SUBJECT_B}
SEG       = 'Schaefer4007'
SPACE     = 'MNI152NLin2009cAsym'
TR        = 2.0

AOMIC_BASE = Path('/mnt/lindenmp_bignas/lindenmp/research_data/AOMIC_PIOP2')

REST_TASK  = 'restingstate'
TASK_NAMES = ('stopsignal', 'workingmemory', 'emomatching')

N_PARCELS = 400
#: Fixed-length runs only. stopsignal is omitted on purpose — see the module docstring.
N_FRAMES = {'restingstate': 240, 'workingmemory': 160, 'emomatching': 135}


def _make_scan(subject_id=SUBJECT_B, task=REST_TASK, modality='MRI3T',
               seg=SEG, space=SPACE, tr=TR):
    return ScanInfo(subject_id=subject_id, task=task, space=space, seg=seg,
                    modality=modality, tr=tr, session=None)


# ==============================================================================
# Unit tests
# ==============================================================================

class TestConfig:
    def test_config_loads(self):
        config = load_dataset_config('AOMIC_PIOP2')
        assert config.name == 'AOMIC_PIOP2'
        assert config.base_path == AOMIC_BASE
        assert config.rawdata_subdir == 'bids'
        assert config.derivatives_subdir == 'derivatives'
        assert config.behavior_handler == 'aomic_piop2'
        assert config.rbc is False
        assert config.subject_id_rawdata_transform is None

    def test_modality_fields(self):
        mod = load_dataset_config('AOMIC_PIOP2').modalities['MRI3T']
        assert mod.tr == 2.0
        assert mod.data_subdir == 'xcpd'
        assert mod.file_ext == '.tsv'
        assert mod.task_events_handler == 'bids_tsv'
        assert mod.task_events_use_glob is False
        assert mod.use_glob is False
        assert mod.filename_template_fallbacks == []

    def test_acq_seq_is_in_both_templates(self):
        """PIOP2 has one acquisition, written literally rather than as a ScanInfo entity.

        If this ever fails because acq became an entity, the cached output filenames and every
        exported scan registry in the project changed with it.
        """
        mod = load_dataset_config('AOMIC_PIOP2').modalities['MRI3T']
        assert 'acq-seq' in mod.filename_template
        assert 'acq-seq' in mod.task_events_file_template

    def test_params_match_the_yaml(self):
        from src.config import get_dataset_params
        params = get_dataset_params('AOMIC_PIOP2')
        assert params['sessions'] == [None]
        assert params['space'] == SPACE
        assert params['seg'] == SEG
        assert set(params['tasks']['MRI3T']) == {REST_TASK, *TASK_NAMES}
        # XCP-D already detrended and regressed 36P, as for UCLA / Penn_LEAD / WAND.
        assert params['detrend'] is False
        assert params['nuisance_regression'] is False
        assert params['task_regression'] is True


class TestScanInfo:
    def test_restingstate_is_rest(self):
        """'restingstate' contains 'rest', so condition derivation needs no special case."""
        assert _make_scan(task=REST_TASK).condition == 'rest'

    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_tasks_are_task(self, task):
        assert _make_scan(task=task).condition == 'task'

    @pytest.mark.parametrize('task', (REST_TASK, *TASK_NAMES))
    def test_no_phase_encoding_is_read(self, task):
        """None of these names ends in a PE token.

        The reader takes the last two characters case-sensitively, so 'te' / 'al' / 'ry' / 'ng'
        must not be mistaken for directions. This is the same guard that stops UCLA's 'scap'
        being read as an AP scan, and it is asserted here because a future loosening of the
        reader would silently give AOMIC a phase-encoding pairing it does not have.
        """
        assert phase_encoding_direction(task) is None


# ==============================================================================
# Integration tests
# ==============================================================================

@integration
class TestPathConstruction:
    def test_exact_path(self, aomic_piop2_loader):
        path = aomic_piop2_loader.get_scan_filepath(_make_scan(task=REST_TASK))
        assert path == (AOMIC_BASE / 'derivatives' / 'xcpd' / f'sub-{SUBJECT_B}' / 'func'
                        / f'sub-{SUBJECT_B}_task-{REST_TASK}_acq-seq_space-{SPACE}'
                          f'_seg-{SEG}_stat-mean_timeseries.tsv')

    def test_no_session_entity_in_the_path(self, aomic_piop2_loader):
        """Single-session dataset: a 'ses-' anywhere means sessions leaked into the template."""
        path = aomic_piop2_loader.get_scan_filepath(_make_scan())
        assert 'ses-' not in str(path)

    @pytest.mark.parametrize('task', (REST_TASK, *TASK_NAMES))
    def test_files_exist(self, aomic_piop2_loader, task):
        assert aomic_piop2_loader.get_scan_filepath(
            _make_scan(subject_id=SUBJECT_B, task=task)).exists()


@integration
class TestTaskEvents:
    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_task_events_resolve(self, aomic_piop2_loader, task):
        events = aomic_piop2_loader.get_task_events(
            _make_scan(subject_id=EVENTS_SUBJECT[task], task=task))
        assert events is not None and len(events) > 0
        for column in ('onset', 'duration', 'trial_type'):
            assert column in events.columns
            assert events[column].notna().all()

    def test_rest_has_no_events(self, aomic_piop2_loader):
        """Required by validate_scan_registry's rest branch: rest scans must have no events.

        AOMIC ships no restingstate events file at all, so this is a property of the dataset
        rather than of the handler — but if it ever changed, every rest scan would be dropped
        from the registry as 'missing (events)' and the failure would look like missing data.
        """
        assert aomic_piop2_loader.get_task_events(_make_scan(task=REST_TASK)) is None

    def test_trial_types_are_strings(self, aomic_piop2_loader):
        """Numeric trial_type codes become np.int64 design-matrix column labels downstream."""
        events = aomic_piop2_loader.get_task_events(_make_scan(task='emomatching'))
        assert set(events['trial_type']) == {'control', 'emotion'}


@integration
class TestDesignMatrix:
    """The events resolving is not the same as the events building a usable design."""

    #: The full trial-type set of each task. stopsignal's is not guaranteed per subject —
    #: see test_stopsignal_trial_types_are_not_guaranteed.
    FULL_TASK_COLUMNS = {
        'stopsignal':    {'go', 'succesful_stop', 'unsuccesful_stop'},   # dataset's spelling
        'workingmemory': {'active_change', 'active_nochange', 'passive'},
        'emomatching':   {'control', 'emotion'},
    }

    @staticmethod
    def _task_columns(design):
        return {c for c in design.columns if not c.startswith('drift_') and c != 'constant'}

    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_design_has_the_expected_task_regressors(self, aomic_piop2_loader, task):
        events = aomic_piop2_loader.get_task_events(
            _make_scan(subject_id=EVENTS_SUBJECT[task], task=task))
        n_frames = N_FRAMES.get(task, 224)
        design = create_design_matrix_nilearn(events, n_frames, TR,
                                              hrf_model='glover', drift_model='cosine')
        assert self._task_columns(design) == self.FULL_TASK_COLUMNS[task]
        assert design.shape[0] == n_frames
        assert design.notna().all().all()

    def test_stopsignal_trial_types_are_not_guaranteed(self, aomic_piop2_loader):
        """stopsignal's design column set varies by subject, and that is the data, not a bug.

        4 of 226 runs are missing one of the two stop conditions — a participant who stopped
        successfully every time has no ``unsuccesful_stop`` trials. sub-0002 is that case.
        nilearn builds a design from the conditions that exist, which is correct for the GLM, so
        task regression and the residual INT (all fig1 needs) are unaffected.

        What it does affect is ``Timescales.task_contrasts``, which is keyed by design column:
        those subjects simply have no beta map for the missing condition. Any consumer that
        assumes three stopsignal contrast maps per subject has to handle their absence.
        """
        events = aomic_piop2_loader.get_task_events(_make_scan(subject_id=SUBJECT_B,
                                                               task='stopsignal'))
        design = create_design_matrix_nilearn(events, 224, TR,
                                              hrf_model='glover', drift_model='cosine')
        columns = self._task_columns(design)
        assert columns < self.FULL_TASK_COLUMNS['stopsignal']    # strict subset
        assert 'go' in columns


@integration
class TestTimeseries:
    @pytest.mark.parametrize('task', sorted(N_FRAMES))
    def test_shape_is_parcels_by_time(self, aomic_piop2_loader, task):
        data = np.asarray(aomic_piop2_loader.load_scan_timeseries(
            _make_scan(task=task), detrend=False, nuisance_regression=False,
            frequency_filter=False))
        assert data.shape == (N_PARCELS, N_FRAMES[task])

    def test_stopsignal_length_is_not_assumed(self, aomic_piop2_loader):
        """Self-paced: 210-250 frames, and 10 of 226 runs exceed the 240-frame rest run."""
        data = np.asarray(aomic_piop2_loader.load_scan_timeseries(
            _make_scan(task='stopsignal'), detrend=False, nuisance_regression=False,
            frequency_filter=False))
        assert data.shape[0] == N_PARCELS
        assert 210 <= data.shape[1] <= 250


@integration
class TestBehavior:
    def test_table_shape_and_index(self):
        loader = DataLoader('AOMIC_PIOP2', load_behavior=True, verbose=False)
        df = loader.df_behavior
        assert df.index.name == 'subject_id'
        # Leading zeros are significant: 'sub-0001' -> '0001', never 1.
        assert SUBJECT_A in df.index
        assert all(isinstance(i, str) for i in df.index[:5])

    def test_expected_columns(self):
        loader = DataLoader('AOMIC_PIOP2', load_behavior=True, verbose=False)
        df = loader.df_behavior
        for column in ('age', 'sex', 'raven_score', 'NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C',
                       'meanFD_rest', 'meanFD_task'):
            assert column in df.columns

    def test_sex_is_coded_and_missing_stays_missing(self):
        """M -> 1 as elsewhere. A missing sex must be NaN, not the 0 an == would produce."""
        df = DataLoader('AOMIC_PIOP2', load_behavior=True, verbose=False).df_behavior
        assert set(df['sex'].dropna().unique()) <= {0, 1}
        assert df['sex'].isna().sum() == 1

    def test_age_is_a_young_adult_range(self):
        """Guards the fig3/fig4 exclusion: there is no developmental curve in this sample."""
        age = DataLoader('AOMIC_PIOP2', load_behavior=True, verbose=False).df_behavior['age']
        assert age.min() >= 18.0 and age.max() <= 26.0
