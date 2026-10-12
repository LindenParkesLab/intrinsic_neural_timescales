"""
Unit and integration tests for data.py — TCP (Transdiagnostic Connectome Project).

Test tiers
----------
Unit tests (no disk access):
    TestConfig       — YAML config, task_params and DATASET_PARAMS agree
    TestPairing      — every task run pairs with the PRE-task rest run of its own PE direction
    TestMotionLabels — XCP-D QC rows are relabelled to registry names before collapsing

Integration tests (require the TCP mount, auto-skipped if unavailable):
    TestPathConstruction — registry names resolve to the right BIDS task and run
    TestTaskEvents       — events are recoded for stroop and hammer and absent for rest
    TestDesignMatrix     — the recoded events build a usable nilearn design
    TestTimeseries       — loaded array shape and orientation
    TestValidation       — an incomplete subject is dropped whole
    TestBehavior         — participants.tsv + demos.tsv + XCP-D QC motion join

Notes
-----
TCP (ds005237) is single-session. BIDS names both rest runs of one phase-encoding direction
``restAP`` / ``restPA`` and tells them apart only by ``run-``; the registry renames them
``rest1AP`` / ``rest2AP`` (HCP-YA's convention) so no two runs share a task name, and the YAML's
``task_params`` maps each name back. rest run-01 is acquired before the tasks, run-02 after.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import get_dataset_params
from src.data import (DataLoader, ScanInfo, TCP_STROOP_STIMULUS_SECONDS, load_dataset_config,
                      xcpd_motion_columns)
from src.timescales import (condition_of, create_design_matrix_nilearn, pair_rest_runs,
                            phase_encoding_direction, split_design_columns)
from tests.conftest import integration_tcp as integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A = 'NDARINVAG023WG3'   # all seven runs
SUBJECT_INCOMPLETE = 'NDARINVAG339WHH'   # rest1AP, rest2AP, rest2PA only
SEG   = 'Schaefer4007'
SPACE = 'MNI152NLin2009cAsym'
TR    = 0.8

TCP_BASE = Path('/mnt/lindenmp_bignas/lindenmp/research_data/TCP')

REST_TASKS = ('rest1AP', 'rest1PA', 'rest2AP', 'rest2PA')
TASK_NAMES = ('stroopAP', 'stroopPA', 'hammerAP')
BIDS = {'rest1AP': ('restAP', '01'), 'rest1PA': ('restPA', '01'),
        'rest2AP': ('restAP', '02'), 'rest2PA': ('restPA', '02'),
        'stroopAP': ('stroopAP', '01'), 'stroopPA': ('stroopPA', '01'),
        'hammerAP': ('hammerAP', '01')}

N_PARCELS = 400
#: Every run is fixed-length. Rest is SHORTER than both tasks, so a matched control's design is
#: truncated to 488 frames: 22 frames (4.3 %) for stroop, 5 (1.0 %) for hammer.
N_FRAMES = {'rest1AP': 488, 'rest1PA': 488, 'rest2AP': 488, 'rest2PA': 488,
            'stroopAP': 510, 'stroopPA': 510, 'hammerAP': 493}


def _make_scan(subject_id=SUBJECT_A, task='rest1AP'):
    return ScanInfo(subject_id=subject_id, task=task, space=SPACE, seg=SEG,
                    modality='MRI3T', tr=TR, session=None)


# ==============================================================================
# Unit tests
# ==============================================================================

class TestConfig:
    def test_config_loads(self):
        config = load_dataset_config('TCP')
        assert config.name == 'TCP'
        assert config.base_path == TCP_BASE
        assert config.behavior_handler == 'tcp'
        assert config.modalities['MRI3T'].tr == TR
        assert config.modalities['MRI3T'].task_events_handler == 'tcp_bids_tsv'

    def test_task_params_map_registry_names_to_bids(self):
        mod = load_dataset_config('TCP').modalities['MRI3T']
        for task, (bids_task, run) in BIDS.items():
            assert mod.template_fields(task) == {'bids_task': bids_task, 'run': run}

    def test_every_processed_task_has_task_params(self):
        """A task without a row has no {bids_task} / {run} and would read as missing data."""
        mod = load_dataset_config('TCP').modalities['MRI3T']
        assert set(get_dataset_params('TCP')['tasks']['MRI3T']) == set(mod.task_params)

    def test_no_two_registry_names_share_a_bids_run(self):
        mod = load_dataset_config('TCP').modalities['MRI3T']
        runs = [(p['bids_task'], p['run']) for p in mod.task_params.values()]
        assert len(runs) == len(set(runs))

    def test_params(self):
        params = get_dataset_params('TCP')
        assert params['sessions'] == [None]
        assert params['space'] == SPACE and params['seg'] == SEG
        assert params['subject_ids_file'] == 'subject_ids.txt'     # no motion cut, by decision
        # XCP-D already detrended and regressed 36P, as for the other XCP-D datasets.
        assert params['detrend'] is False and params['nuisance_regression'] is False
        assert params['task_regression'] is True

    def test_conditions(self):
        for task in REST_TASKS:
            assert _make_scan(task=task).condition == 'rest'
        for task in TASK_NAMES:
            assert _make_scan(task=task).condition == 'task'


class TestPairing:
    def test_phase_encoding_is_read_from_every_registry_name(self):
        for task in BIDS:
            assert phase_encoding_direction(task) == task[-2:]

    def test_every_task_pairs_with_pre_task_rest_of_its_direction(self):
        """Registry order puts rest1* before rest2*, and the pairing takes the first same-PE run."""
        tasks = get_dataset_params('TCP')['tasks']['MRI3T']
        rest = [t for t in tasks if 'rest' in t]
        task = [t for t in tasks if 'rest' not in t]
        assert pair_rest_runs(task, rest) == {'stroopAP': ('rest1AP', True),
                                              'stroopPA': ('rest1PA', True),
                                              'hammerAP': ('rest1AP', True)}

    def test_stroop_runs_are_one_condition(self):
        assert condition_of('stroopAP') == condition_of('stroopPA') == 'task-stroop'
        assert condition_of('hammerAP') == 'task-hammerAP'


class _MotionStub:
    dataset = 'TCP'
    config = load_dataset_config('TCP')


class TestMotionLabels:
    def test_rest_runs_stay_apart_and_aliases_resolve(self, monkeypatch):
        rows = [('S1', None, 'restAP', '01', None, 0.10), ('S1', None, 'restPA', '01', None, 0.20),
                ('S1', None, 'restAP', '02', None, 0.50), ('S1', None, 'restPA', '02', None, 0.70),
                ('S1', None, 'stroopAP', '01', None, 0.30), ('S1', None, 'stroopPA', '01', None, 0.40),
                ('S1', None, 'hammerAP', '01', None, 0.90)]
        long = pd.DataFrame(rows, columns=['subject_id', 'session', 'task', 'run', 'acq', 'mean_fd'])
        monkeypatch.setattr('src.data.load_xcpd_motion', lambda loader, force=False: long)
        out = xcpd_motion_columns(_MotionStub())
        assert out.loc['S1', 'meanFD_rest1AP'] == pytest.approx(0.10)
        assert out.loc['S1', 'meanFD_rest2AP'] == pytest.approx(0.50)
        assert 'meanFD_restAP' not in out.columns        # never a pre/post mean
        assert out.loc['S1', 'meanFD_rest'] == pytest.approx(0.15)       # pre-task rest only
        assert out.loc['S1', 'meanFD_rest_post'] == pytest.approx(0.60)
        assert out.loc['S1', 'meanFD_task'] == pytest.approx(0.35)       # stroop, not hammer
        assert out.loc['S1', 'meanFD_hammerAP'] == pytest.approx(0.90)


# ==============================================================================
# Integration tests
# ==============================================================================

@integration
class TestPathConstruction:
    def test_registry_names_resolve_to_bids_task_and_run(self, tcp_loader):
        for task, (bids_task, run) in BIDS.items():
            path = tcp_loader.get_scan_filepath(_make_scan(task=task))
            assert path.name == (f'sub-{SUBJECT_A}_task-{bids_task}_run-{run}_space-{SPACE}_'
                                 f'seg-{SEG}_stat-mean_timeseries.tsv')
            assert path.exists()

    def test_pre_and_post_rest_are_different_files(self, tcp_loader):
        assert (tcp_loader.get_scan_filepath(_make_scan(task='rest1AP'))
                != tcp_loader.get_scan_filepath(_make_scan(task='rest2AP')))


@integration
class TestTaskEvents:
    @pytest.mark.parametrize('task', REST_TASKS)
    def test_rest_has_no_events(self, tcp_loader, task):
        assert tcp_loader.get_task_events(_make_scan(task=task)) is None

    @pytest.mark.parametrize('task', ('stroopAP', 'stroopPA'))
    def test_stroop_is_recoded(self, tcp_loader, task):
        ev = tcp_loader.get_task_events(_make_scan(task=task))
        assert list(ev.columns) == ['onset', 'duration', 'trial_type']
        assert ev['trial_type'].value_counts().to_dict() == {'congruent': 54, 'incongruent': 24}
        assert (ev['duration'] == TCP_STROOP_STIMULUS_SECONDS).all()

    def test_hammer_gets_its_cues(self, tcp_loader):
        ev = tcp_loader.get_task_events(_make_scan(task='hammerAP'))
        assert ev['trial_type'].value_counts().to_dict() == {'face': 54, 'shape': 54, 'cue': 18}
        cues = ev[ev['trial_type'] == 'cue']
        assert cues['duration'].between(2.8, 3.1).all()
        assert ev['onset'].is_monotonic_increasing


@integration
class TestDesignMatrix:
    @pytest.mark.parametrize('task,columns', [('stroopAP', {'congruent', 'incongruent'}),
                                              ('hammerAP', {'face', 'shape', 'cue'})])
    def test_design_has_task_regressors_and_full_rank(self, tcp_loader, task, columns):
        ev = tcp_loader.get_task_events(_make_scan(task=task))
        design = create_design_matrix_nilearn(ev, N_FRAMES[task], TR,
                                              hrf_model='glover', drift_model='cosine')
        task_cols, _ = split_design_columns(design)
        assert set(task_cols) == columns
        # The release's tiling stroop durations would make the task columns sum to ~constant;
        # the stimulus-length model must leave the design well conditioned.
        assert np.linalg.matrix_rank(design.values) == design.shape[1]


@integration
class TestTimeseries:
    @pytest.mark.parametrize('task', list(N_FRAMES))
    def test_shape(self, tcp_loader, task):
        ts = tcp_loader.load_scan_timeseries(_make_scan(task=task), cache=False, detrend=False,
                                             nuisance_regression=False, frequency_filter=False)
        assert ts.shape == (N_PARCELS, N_FRAMES[task])


@integration
class TestValidation:
    def test_incomplete_subject_is_dropped_whole(self):
        loader = DataLoader('TCP', load_behavior=False, verbose=False)
        params = get_dataset_params('TCP')
        loader.register_scans([SUBJECT_A, SUBJECT_INCOMPLETE], params['sessions'],
                              params['tasks']['MRI3T'], SPACE, SEG, 'MRI3T')
        loader.validate_scan_registry(verbose=False, remove_incomplete_subjects=True)
        assert {s.subject_id for s in loader.scan_registry} == {SUBJECT_A}
        assert len(loader.scan_registry) == 7


@integration
class TestBehavior:
    @pytest.fixture(scope='class')
    def df(self):
        return DataLoader('TCP', load_behavior=True, verbose=False).df_behavior

    def test_every_participant_and_the_join(self, df):
        assert len(df) == 245 and df.index.is_unique
        assert SUBJECT_A in df.index
        assert set(df['group'].dropna()) == {'Patient', 'GenPop'}
        assert (df['patient'] == (df['group'] == 'Patient')).all()
        # Four participants self-describe (sex 'O'): NaN, never silently coded 0.
        assert df['sex'].dropna().isin([0, 1]).all() and df['sex'].isna().sum() == 4
        assert df['dx_primary'].notna().sum() > 100

    def test_medication_reading(self, df):
        """The inferred coding: nearly every control is off psychiatric medication."""
        ctrl = df.loc[df['group'] == 'GenPop', 'psych_meds'].dropna()
        assert (ctrl == 0).mean() > 0.9

    def test_motion_columns(self, df):
        row = df.loc[SUBJECT_A]
        assert row['meanFD_rest'] == pytest.approx((row['meanFD_rest1AP'] + row['meanFD_rest1PA']) / 2)
        assert row['meanFD_task'] == pytest.approx((row['meanFD_stroopAP'] + row['meanFD_stroopPA']) / 2)
        assert np.isfinite(row['meanFD_rest_post'])
