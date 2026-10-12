"""
Unit and integration tests for data.py — AOMIC-PIOP1 dataset.

Test tiers
----------
Unit tests (no disk access):
    TestConfig       — YAML config loading, per-task acquisition parameters
    TestTaskParams   — ModalityConfig.template_fields, and that other datasets are unaffected
    TestScanInfo     — condition derivation and the phase-encoding reader

Integration tests (require the AOMIC-PIOP1 mount, auto-skipped if unavailable):
    TestRegistration     — per-task TR reaches the registry and the scan string
    TestPathConstruction — get_scan_filepath resolves acq per task
    TestTaskEvents       — events resolve for the five tasks and are absent for rest
    TestDesignMatrix     — the events build a nilearn design at each task's own TR
    TestTimeseries       — loaded array shape and orientation
    TestBehavior         — participants.tsv + XCP-D QC motion join

Notes
-----
AOMIC-PIOP1 (ds002785) is single-session but mixes two acquisitions: ``restingstate`` and
``faces`` are multiband (``acq-mb3``, TR 0.75 s) and the other four tasks sequential
(``acq-seq``, TR 2.0 s). Each task has exactly one acq, so acq is resolved per task from the
YAML's ``task_params`` rather than promoted to a ScanInfo entity, and the per-task TR is written
into each scan's ``tr-`` token at registration. Everything downstream -- task regression, the
timescale estimators, the rest-control offset -- reads TR from that token, never from the
modality, which is what makes this safe.
"""

import numpy as np
import pytest
from pathlib import Path

from src.data import DataLoader, ModalityConfig, ScanInfo, load_dataset_config
from src.timescales import create_design_matrix_nilearn, phase_encoding_direction
from tests.conftest import integration_aomic_piop1 as integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A = '0001'    # has all six runs
SUBJECT_B = '0002'    # has all six runs
SEG       = 'Schaefer4007'
SPACE     = 'MNI152NLin2009cAsym'

AOMIC_BASE = Path('/mnt/lindenmp_bignas/lindenmp/research_data/AOMIC_PIOP1')

REST_TASK  = 'restingstate'
TASK_NAMES = ('faces', 'anticipation', 'emomatching', 'gstroop', 'workingmemory')
ALL_TASKS  = (REST_TASK, *TASK_NAMES)

#: task -> (acq, TR in seconds). Checked against RepetitionTime in the BIDS sidecars and in
#: XCP-D's denoised JSONs.
ACQ = {
    'restingstate':  ('mb3', 0.75),
    'faces':         ('mb3', 0.75),
    'anticipation':  ('seq', 2.0),
    'emomatching':   ('seq', 2.0),
    'gstroop':       ('seq', 2.0),
    'workingmemory': ('seq', 2.0),
}

N_PARCELS = 400
#: Every run is fixed-length -- verified across all subjects on disk.
N_FRAMES = {'restingstate': 480, 'faces': 330, 'anticipation': 200,
            'emomatching': 135, 'gstroop': 245, 'workingmemory': 162}


def _make_scan(subject_id=SUBJECT_B, task=REST_TASK, modality='MRI3T', seg=SEG, space=SPACE):
    return ScanInfo(subject_id=subject_id, task=task, space=space, seg=seg,
                    modality=modality, tr=ACQ[task][1], session=None)


# ==============================================================================
# Unit tests
# ==============================================================================

class TestConfig:
    def test_config_loads(self):
        config = load_dataset_config('AOMIC_PIOP1')
        assert config.name == 'AOMIC_PIOP1'
        assert config.base_path == AOMIC_BASE
        assert config.rawdata_subdir == 'bids'
        assert config.derivatives_subdir == 'derivatives'
        assert config.behavior_handler == 'aomic_piop1'
        assert config.rbc is False
        assert config.subject_id_rawdata_transform is None

    def test_modality_fields(self):
        mod = load_dataset_config('AOMIC_PIOP1').modalities['MRI3T']
        assert mod.data_subdir == 'xcpd'
        assert mod.file_ext == '.tsv'
        assert mod.task_events_handler == 'bids_tsv'
        assert mod.task_events_use_glob is False
        assert mod.use_glob is False
        assert mod.filename_template_fallbacks == []

    def test_acq_is_a_placeholder_in_both_templates(self):
        """Two acquisitions, so neither may be written literally, as PIOP2's acq-seq is."""
        mod = load_dataset_config('AOMIC_PIOP1').modalities['MRI3T']
        for template in (mod.filename_template, mod.task_events_file_template):
            assert 'acq-{acq}' in template
            assert 'acq-seq' not in template and 'acq-mb3' not in template

    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_task_params(self, task):
        mod = load_dataset_config('AOMIC_PIOP1').modalities['MRI3T']
        acq, tr = ACQ[task]
        assert mod.task_params[task] == {'acq': acq, 'tr': tr}

    def test_every_processed_task_has_task_params(self):
        """A task added to DATASET_PARAMS without a task_params row has no acq and no TR.

        It would fail validation as a missing file rather than silently take the fallback TR,
        but the failure would read as missing data. Catch it here instead.
        """
        from src.config import get_dataset_params
        mod = load_dataset_config('AOMIC_PIOP1').modalities['MRI3T']
        assert set(get_dataset_params('AOMIC_PIOP1')['tasks']['MRI3T']) == set(mod.task_params)

    def test_params_match_the_yaml(self):
        from src.config import get_dataset_params
        params = get_dataset_params('AOMIC_PIOP1')
        assert params['sessions'] == [None]
        assert params['space'] == SPACE
        assert params['seg'] == SEG
        assert set(params['tasks']['MRI3T']) == set(ALL_TASKS)
        # XCP-D already detrended and regressed 36P, as for PIOP2 / UCLA / Penn_LEAD / WAND.
        assert params['detrend'] is False
        assert params['nuisance_regression'] is False
        assert params['task_regression'] is True


class TestTaskParams:
    def test_template_fields_drop_tr(self):
        mod = load_dataset_config('AOMIC_PIOP1').modalities['MRI3T']
        assert mod.template_fields('faces') == {'acq': 'mb3'}
        assert mod.template_fields('gstroop') == {'acq': 'seq'}

    def test_unlisted_task_has_no_fields(self):
        mod = ModalityConfig(tr=1.0, data_subdir='x', dir_template='', filename_template='',
                             file_ext='.tsv')
        assert mod.task_params == {}
        assert mod.template_fields('anything') == {}

    @pytest.mark.parametrize('dataset', ('AOMIC_PIOP2', 'HCP-YA', 'HCP-D', 'Penn_LEAD', 'WAND'))
    def test_other_datasets_set_no_task_params(self, dataset):
        """task_params is opt-in; every other dataset keeps one TR per modality."""
        for mod in load_dataset_config(dataset).modalities.values():
            assert mod.task_params == {}


class TestScanInfo:
    def test_restingstate_is_rest(self):
        assert _make_scan(task=REST_TASK).condition == 'rest'

    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_tasks_are_task(self, task):
        assert _make_scan(task=task).condition == 'task'

    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_no_phase_encoding_is_read(self, task):
        """None of these names ends in a PE token ('te', 'es', 'on', 'ng', 'op', 'ry').

        All six runs are acquired at PhaseEncodingDirection j, so PE is matched in fact; the
        reader correctly returns None and anything recording pe_matched would understate it.
        """
        assert phase_encoding_direction(task) is None


# ==============================================================================
# Integration tests
# ==============================================================================

@integration
class TestRegistration:
    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_get_tr_is_per_task(self, aomic_piop1_loader, task):
        assert aomic_piop1_loader.get_tr('MRI3T', task) == ACQ[task][1]

    def test_registered_tr_is_per_task(self):
        """The TR must reach the tr- token: every estimator and the design read it from there."""
        loader = DataLoader('AOMIC_PIOP1', load_behavior=False, verbose=False)
        scans = loader.register_scans([SUBJECT_B], [None], list(ALL_TASKS), SPACE, SEG, 'MRI3T')
        for scan in scans:
            assert scan.tr == ACQ[scan.task][1]
            assert f'_tr-{ACQ[scan.task][1]}_' in str(scan)


@integration
class TestPathConstruction:
    def test_exact_path(self, aomic_piop1_loader):
        path = aomic_piop1_loader.get_scan_filepath(_make_scan(task=REST_TASK))
        assert path == (AOMIC_BASE / 'derivatives' / 'xcpd' / f'sub-{SUBJECT_B}' / 'func'
                        / f'sub-{SUBJECT_B}_task-{REST_TASK}_acq-mb3_space-{SPACE}'
                          f'_seg-{SEG}_stat-mean_timeseries.tsv')

    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_acq_resolves_per_task(self, aomic_piop1_loader, task):
        path = aomic_piop1_loader.get_scan_filepath(_make_scan(task=task))
        assert f'_task-{task}_acq-{ACQ[task][0]}_' in path.name

    def test_no_session_entity_in_the_path(self, aomic_piop1_loader):
        assert 'ses-' not in str(aomic_piop1_loader.get_scan_filepath(_make_scan()))

    @pytest.mark.parametrize('subject_id', (SUBJECT_A, SUBJECT_B))
    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_files_exist(self, aomic_piop1_loader, subject_id, task):
        assert aomic_piop1_loader.get_scan_filepath(
            _make_scan(subject_id=subject_id, task=task)).exists()


@integration
class TestTaskEvents:
    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_task_events_resolve(self, aomic_piop1_loader, task):
        events = aomic_piop1_loader.get_task_events(_make_scan(task=task))
        assert events is not None and len(events) > 0
        for column in ('onset', 'duration', 'trial_type'):
            assert column in events.columns
            assert events[column].notna().all()

    def test_rest_has_no_events(self, aomic_piop1_loader):
        """Required by validate_scan_registry's rest branch: rest scans must have no events.

        PIOP1 ships no restingstate events file at all. If that ever changed, every rest scan
        would be dropped from the registry as 'missing (events)'.
        """
        assert aomic_piop1_loader.get_task_events(_make_scan(task=REST_TASK)) is None


@integration
class TestDesignMatrix:
    """The events resolving is not the same as the events building a usable design."""

    TASK_COLUMNS = {
        'faces':         {'anger', 'contempt', 'joy', 'neutral', 'pride'},
        'anticipation':  {'cue_negative', 'cue_neutral', 'img_negative', 'img_neutral'},
        'emomatching':   {'control', 'emotion'},
        'gstroop':       {'congruent', 'incongruent'},
        'workingmemory': {'active_change', 'active_nochange', 'passive'},
    }

    @pytest.mark.parametrize('task', TASK_NAMES)
    def test_design_has_the_expected_task_regressors(self, aomic_piop1_loader, task):
        events = aomic_piop1_loader.get_task_events(_make_scan(task=task))
        design = create_design_matrix_nilearn(events, N_FRAMES[task], ACQ[task][1],
                                              hrf_model='glover', drift_model='cosine')
        columns = {c for c in design.columns if not c.startswith('drift_') and c != 'constant'}
        assert columns == self.TASK_COLUMNS[task]
        assert design.shape[0] == N_FRAMES[task]
        assert design.notna().all().all()

    def test_last_event_ends_inside_the_run_at_its_own_tr(self, aomic_piop1_loader):
        """faces at the wrong TR (2.0 s) would span 660 s; its events end near 247.5 s."""
        events = aomic_piop1_loader.get_task_events(_make_scan(task='faces'))
        run_seconds = N_FRAMES['faces'] * ACQ['faces'][1]
        assert (events['onset'] + events['duration']).max() <= run_seconds


@integration
class TestTimeseries:
    @pytest.mark.parametrize('task', ALL_TASKS)
    def test_shape_is_parcels_by_time(self, aomic_piop1_loader, task):
        data = np.asarray(aomic_piop1_loader.load_scan_timeseries(
            _make_scan(task=task), detrend=False, nuisance_regression=False,
            frequency_filter=False))
        assert data.shape == (N_PARCELS, N_FRAMES[task])


@integration
class TestBehavior:
    @pytest.fixture(scope='class')
    def df(self):
        return DataLoader('AOMIC_PIOP1', load_behavior=True, verbose=False).df_behavior

    def test_table_shape_and_index(self, df):
        assert df.index.name == 'subject_id'
        # Leading zeros are significant: 'sub-0001' -> '0001', never 1.
        assert SUBJECT_A in df.index
        assert all(isinstance(i, str) for i in df.index[:5])

    def test_expected_columns(self, df):
        for column in ('age', 'sex', 'raven_score', 'NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C',
                       'meanFD_rest', 'meanFD_task', *(f'meanFD_{t}' for t in ALL_TASKS)):
            assert column in df.columns

    def test_sex_is_coded_and_missing_stays_missing(self, df):
        """M -> 1 as elsewhere. A missing sex must be NaN, not the 0 an == would produce."""
        assert set(df['sex'].dropna().unique()) <= {0, 1}
        assert df['sex'].isna().sum() == 7

    def test_age_is_a_young_adult_range(self, df):
        """Guards the fig3/fig4 exclusion: there is no developmental curve in this sample."""
        assert df['age'].min() >= 18.0 and df['age'].max() <= 27.0

    def test_motion_cache_is_per_sample(self, df):
        """Every XCP-D dataset shares the loader; they must not share a cache file."""
        from src.data import XCPD_MOTION_CACHE, _find_project_root
        assert (_find_project_root() / 'data'
                / XCPD_MOTION_CACHE.format(dataset='aomic_piop1')).exists()
