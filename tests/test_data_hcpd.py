"""
Unit and integration tests for data.py — HCP-D dataset.

Test tiers
----------
Unit tests (no disk access):
    TestConfig          — YAML config loading and field values
    TestScanInfo        — ScanInfo construction, condition derivation, round-trip serialisation

Integration tests (require HCP-D data mount, auto-skipped if unavailable):
    TestPathConstruction — get_scan_filepath template rendering and exact path assertions
    TestFileExistence    — expected files are present on disk
    TestTimeseries       — loaded array shape, dtype, NaN, cache, and nuisance regression
    TestRegistryRoundtrip — export → CSV → reload preserves ScanInfo objects exactly

Notes
-----
HCP-D subject IDs are stored without underscores (e.g. 'HCD0001305V1MR').
Session labels use zero-padded two-digit format ('01').
Timeseries files are CIFTI .ptseries.nii; nibabel loads them as
(n_timepoints, n_parcels) which data.py transposes to (n_parcels, n_timepoints).
"""

import pytest
import numpy as np
from pathlib import Path

from src.data import DataLoader, ScanInfo, load_dataset_config
from tests.conftest import integration_hcpd as integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A = 'HCD0001305V1MR'
SUBJECT_B = 'HCD0008117V1MR'
SESSION   = '01'
SEG       = 'Schaefer4007'
SPACE     = 'fsLR'

HCPD_DERIV = Path('/mnt/lindenmp_bignas/lindenmp/research_data/HCP_D/derivatives')

# Expected output shapes: (n_parcels, n_timepoints).
# 400 parcels for Schaefer4007; 478 TRs @ 0.80 s ≈ 6.4 min for REST1AP.
N_PARCELS  = 400
REST_SHAPE = (N_PARCELS, 478)   # rfMRIREST1AP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scan(subject_id=SUBJECT_A, task='rfMRIREST1AP', modality='MRI3T',
               session=SESSION, seg=SEG, space=SPACE, tr=0.80, rbc=False):
    return ScanInfo(subject_id=subject_id, task=task, space=space, seg=seg,
                    modality=modality, tr=tr, session=session, rbc=rbc)


# ==============================================================================
# Config  (pure unit — no disk access)
# ==============================================================================

class TestConfig:

    def test_loads_without_error(self):
        cfg = load_dataset_config('HCP-D')
        assert cfg.name == 'HCP-D'

    def test_modalities_present(self):
        cfg = load_dataset_config('HCP-D')
        assert set(cfg.modalities) == {'MRI3T'}

    def test_tr_mri3t(self):
        cfg = load_dataset_config('HCP-D')
        assert cfg.modalities['MRI3T'].tr == pytest.approx(0.80)

    def test_rbc_false(self):
        assert load_dataset_config('HCP-D').rbc is False

    def test_behavior_handler(self):
        assert load_dataset_config('HCP-D').behavior_handler == 'hcp_d'

    def test_file_ext(self):
        cfg = load_dataset_config('HCP-D')
        assert cfg.modalities['MRI3T'].file_ext == '.ptseries.nii'

    def test_task_events_handler(self):
        cfg = load_dataset_config('HCP-D')
        assert cfg.modalities['MRI3T'].task_events_handler == 'hcp_fmri'

    def test_subject_id_transform(self):
        assert load_dataset_config('HCP-D').subject_id_rawdata_transform == 'split_hcpd'


# ==============================================================================
# ScanInfo  (pure unit — no disk access)
# ==============================================================================

class TestScanInfo:

    def test_condition_rest(self):
        assert _make_scan(task='rfMRIREST1AP').condition == 'rest'

    def test_condition_rest_pa(self):
        assert _make_scan(task='rfMRIREST2PA').condition == 'rest'

    def test_condition_task_carit(self):
        assert _make_scan(task='tfMRICARITPA').condition == 'task'

    def test_condition_task_emotion(self):
        assert _make_scan(task='tfMRIEMOTIONPA').condition == 'task'

    def test_roundtrip_with_session(self):
        scan = _make_scan()
        assert ScanInfo.from_dict(scan.to_dict()) == scan

    def test_roundtrip_no_session(self):
        scan = _make_scan(session=None)
        reconstructed = ScanInfo.from_dict(scan.to_dict())
        assert reconstructed.session is None
        assert reconstructed == scan

    def test_roundtrip_preserves_rbc_false(self):
        scan = _make_scan(rbc=False)
        assert ScanInfo.from_dict(scan.to_dict()).rbc is False

    def test_str_contains_subject(self):
        assert f'sub-{SUBJECT_A}' in str(_make_scan())

    def test_str_contains_task(self):
        assert 'task-rfMRIREST1AP' in str(_make_scan())

    def test_str_contains_session(self):
        assert f'ses-{SESSION}' in str(_make_scan())

    def test_str_no_session_when_none(self):
        assert 'ses-' not in str(_make_scan(session=None))

    def test_to_dict_keys(self):
        d = _make_scan().to_dict()
        for key in ('sub', 'ses', 'task', 'space', 'seg', 'modality', 'tr',
                    'condition', 'rbc', 'scan_string'):
            assert key in d

    def test_different_subjects_not_equal(self):
        assert _make_scan(subject_id=SUBJECT_A) != _make_scan(subject_id=SUBJECT_B)

    def test_different_tasks_not_equal(self):
        assert _make_scan(task='rfMRIREST1AP') != _make_scan(task='tfMRICARITPA')


# ==============================================================================
# Path construction  (integration — needs DataLoader init, no file I/O)
# ==============================================================================

@integration
class TestPathConstruction:

    def test_subdir_is_parcellated_data(self, hcpd_loader):
        fp = hcpd_loader.get_scan_filepath(_make_scan())
        assert 'parcellated_data' in str(fp)

    def test_extension_is_ptseries_nii(self, hcpd_loader):
        fp = hcpd_loader.get_scan_filepath(_make_scan())
        assert str(fp).endswith('.ptseries.nii')

    def test_exact_path_rest(self, hcpd_loader):
        scan = _make_scan(subject_id=SUBJECT_A, task='rfMRIREST1AP')
        expected = (
            HCPD_DERIV
            / 'parcellated_data'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION}_task-rfMRIREST1AP'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.ptseries.nii'
        )
        assert hcpd_loader.get_scan_filepath(scan) == expected

    def test_exact_path_task(self, hcpd_loader):
        scan = _make_scan(subject_id=SUBJECT_A, task='tfMRICARITPA')
        expected = (
            HCPD_DERIV
            / 'parcellated_data'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION}_task-tfMRICARITPA'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.ptseries.nii'
        )
        assert hcpd_loader.get_scan_filepath(scan) == expected

    def test_path_contains_subject_id(self, hcpd_loader):
        fp = hcpd_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_A))
        assert f'sub-{SUBJECT_A}' in str(fp)

    def test_path_contains_task(self, hcpd_loader):
        fp = hcpd_loader.get_scan_filepath(_make_scan(task='tfMRICARITPA'))
        assert 'tfMRICARITPA' in str(fp)

    def test_different_subjects_give_different_paths(self, hcpd_loader):
        fp_a = hcpd_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_A))
        fp_b = hcpd_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_B))
        assert fp_a != fp_b

    def test_different_tasks_give_different_paths(self, hcpd_loader):
        fp_rest = hcpd_loader.get_scan_filepath(_make_scan(task='rfMRIREST1AP'))
        fp_task = hcpd_loader.get_scan_filepath(_make_scan(task='tfMRICARITPA'))
        assert fp_rest != fp_task


# ==============================================================================
# File existence  (integration)
# ==============================================================================

@integration
class TestFileExistence:

    @pytest.mark.parametrize("subject_id, task", [
        (SUBJECT_A, 'rfMRIREST1AP'),
        (SUBJECT_A, 'rfMRIREST1PA'),
        (SUBJECT_A, 'rfMRIREST2AP'),
        (SUBJECT_A, 'rfMRIREST2PA'),
        (SUBJECT_A, 'tfMRICARITPA'),
        (SUBJECT_A, 'tfMRIEMOTIONPA'),
        (SUBJECT_A, 'tfMRIGUESSINGPA'),
        (SUBJECT_B, 'rfMRIREST1AP'),
        (SUBJECT_B, 'tfMRICARITPA'),
    ])
    def test_file_exists(self, hcpd_loader, subject_id, task):
        fp = hcpd_loader.get_scan_filepath(_make_scan(subject_id=subject_id, task=task))
        assert fp.exists(), f"Expected file not found: {fp}"


# ==============================================================================
# Timeseries loading  (integration)
# ==============================================================================

@integration
class TestTimeseries:
    """
    Tests load scans with nuisance_regression=False (raw load) unless the test
    is explicitly about the regression step.
    """

    def test_rest_shape(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1AP'), nuisance_regression=False, cache=False
        )
        assert ts.shape == REST_SHAPE, f"Expected {REST_SHAPE}, got {ts.shape}"

    def test_parcels_axis_is_first(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_task_has_correct_parcels(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(task='tfMRICARITPA'), nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_dtype_float32(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.dtype == np.float32

    def test_no_nan_rest(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert not np.isnan(ts).any()

    def test_no_nan_task(self, hcpd_loader):
        ts = hcpd_loader.load_scan_timeseries(
            _make_scan(task='tfMRICARITPA'), nuisance_regression=False, cache=False
        )
        assert not np.isnan(ts).any()

    def test_different_subjects_differ(self, hcpd_loader):
        ts_a = hcpd_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_A), nuisance_regression=False, cache=False
        )
        ts_b = hcpd_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_B), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_a, ts_b), (
            "Two different subjects returned identical timeseries."
        )

    def test_different_runs_differ(self, hcpd_loader):
        ts_ap = hcpd_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1AP'), nuisance_regression=False, cache=False
        )
        ts_pa = hcpd_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1PA'), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_ap, ts_pa), (
            "AP and PA runs returned identical timeseries."
        )

    def test_cache_returns_identical_array(self, hcpd_loader):
        scan = _make_scan(task='rfMRIREST2AP')
        ts1 = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=True)
        ts2 = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=True)
        assert np.array_equal(ts1, ts2)
        hcpd_loader.clear_cache()

    def test_nuisance_regression_changes_data(self, hcpd_loader):
        scan = _make_scan()
        ts_raw = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=False)
        ts_reg = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=True,  cache=False)
        assert not np.allclose(ts_raw, ts_reg)

    def test_nuisance_regression_reduces_global_signal_variance(self, hcpd_loader):
        scan = _make_scan()
        ts_raw = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=False)
        ts_reg = hcpd_loader.load_scan_timeseries(scan, nuisance_regression=True,  cache=False)
        gs_raw = np.mean(ts_raw, axis=0)
        gs_reg = np.mean(ts_reg, axis=0)
        assert gs_reg.var() < gs_raw.var()


# ==============================================================================
# Registry round-trip  (integration)
# ==============================================================================

@integration
class TestRegistryRoundtrip:

    def test_export_and_reload_preserves_scans(self, tmp_path):
        loader = DataLoader('HCP-D', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)

        original_scans = loader.register_scans(
            subject_ids=[SUBJECT_A, SUBJECT_B],
            sessions=[SESSION],
            tasks=['rfMRIREST1AP', 'tfMRICARITPA'],
            space=SPACE,
            seg=SEG,
            modality='MRI3T',
        )

        loader.export_scan_registry('test_registry.csv')
        loader.scan_registry = []
        loader.load_scan_registry('test_registry.csv')

        assert len(loader.scan_registry) == len(original_scans)
        for original, reloaded in zip(original_scans, loader.scan_registry):
            assert original == reloaded, (
                f"Mismatch after round-trip:\n  original: {original}\n  reloaded: {reloaded}"
            )

    def test_export_csv_has_expected_columns(self, tmp_path):
        loader = DataLoader('HCP-D', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)
        loader.register_scans(
            subject_ids=[SUBJECT_A], sessions=[SESSION],
            tasks=['rfMRIREST1AP'], space=SPACE, seg=SEG, modality='MRI3T',
        )
        loader.export_scan_registry('test_registry.csv')

        import pandas as pd
        df = pd.read_csv(tmp_path / 'test_registry.csv', index_col=0)
        for col in ('sub', 'ses', 'task', 'modality', 'condition', 'tr', 'scan_string'):
            assert col in df.columns, f"Missing column: {col}"
