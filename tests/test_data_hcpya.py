"""
Unit and integration tests for data.py — HCP-YA dataset.

Test tiers
----------
Unit tests (no disk access):
    TestConfig          — YAML config loading and field values
    TestScanInfo        — ScanInfo construction, condition derivation, round-trip serialisation

Integration tests (require HCP-YA data mount, auto-skipped if unavailable):
    TestPathConstruction — get_scan_filepath template rendering and exact path assertions
    TestFileExistence    — expected files are present on disk
    TestTimeseries       — loaded array shape, dtype, NaN, cache, and nuisance regression
    TestRegistryRoundtrip — export → CSV → reload preserves ScanInfo objects exactly
"""

import pytest
import numpy as np
from pathlib import Path

from src.data import DataLoader, ScanInfo, load_dataset_config
from tests.conftest import integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A = '100206'
SUBJECT_B = '100307'
SESSION   = '01'
SEG       = 'Schaefer4007'
SPACE     = 'fsLR'

HCPYA_DERIV = Path('/mnt/lindenmp_bignas/lindenmp/research_data/HCP_YA/derivatives')

# Expected output shapes: (n_parcels, n_timepoints)
N_PARCELS      = 400
REST_SHAPE     = (N_PARCELS, 1200)   # 1200 TRs @ 0.72 s
WM_SHAPE       = (N_PARCELS, 405)    # tfMRIWMLR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scan(subject_id=SUBJECT_A, task='rfMRIREST1LR', modality='MRI3T',
               session=SESSION, seg=SEG, space=SPACE, tr=0.72):
    return ScanInfo(subject_id=subject_id, task=task, space=space, seg=seg,
                    modality=modality, tr=tr, session=session)


# ==============================================================================
# Config  (pure unit — no disk access)
# ==============================================================================

class TestConfig:

    def test_loads_without_error(self):
        cfg = load_dataset_config('HCP-YA')
        assert cfg.name == 'HCP-YA'

    def test_modalities_present(self):
        cfg = load_dataset_config('HCP-YA')
        assert set(cfg.modalities) == {'MRI3T', 'MRI7T'}

    def test_tr_mri3t(self):
        cfg = load_dataset_config('HCP-YA')
        assert cfg.modalities['MRI3T'].tr == pytest.approx(0.72)

    def test_tr_mri7t(self):
        cfg = load_dataset_config('HCP-YA')
        assert cfg.modalities['MRI7T'].tr == pytest.approx(1.0)


    def test_behavior_handler(self):
        assert load_dataset_config('HCP-YA').behavior_handler == 'hcp_unrestricted_restricted'

    def test_mri3t_file_ext(self):
        cfg = load_dataset_config('HCP-YA')
        assert cfg.modalities['MRI3T'].file_ext == '.ptseries.nii'

    def test_unknown_dataset_raises(self):
        with pytest.raises(FileNotFoundError, match="No config found"):
            load_dataset_config('NONEXISTENT_DATASET')


# ==============================================================================
# ScanInfo  (pure unit — no disk access)
# ==============================================================================

class TestScanInfo:

    def test_condition_rest(self):
        assert _make_scan(task='rfMRIREST1LR').condition == 'rest'

    def test_condition_rest_lowercase(self):
        assert _make_scan(task='rest').condition == 'rest'

    def test_condition_task(self):
        assert _make_scan(task='tfMRIWMLR').condition == 'task'

    def test_roundtrip_with_session(self):
        scan = _make_scan()
        assert ScanInfo.from_dict(scan.to_dict()) == scan

    def test_roundtrip_no_session(self):
        scan = _make_scan(session=None)
        reconstructed = ScanInfo.from_dict(scan.to_dict())
        assert reconstructed.session is None
        assert reconstructed == scan

    def test_from_dict_ignores_the_retired_rbc_column(self):
        """Registries exported before the rbc flag was removed still load."""
        scan = _make_scan()
        assert ScanInfo.from_dict({**scan.to_dict(), 'rbc': 'False'}) == scan

    def test_str_contains_subject(self):
        assert f'sub-{SUBJECT_A}' in str(_make_scan())

    def test_str_contains_task(self):
        assert 'task-rfMRIREST1LR' in str(_make_scan())

    def test_str_contains_session(self):
        assert f'ses-{SESSION}' in str(_make_scan())

    def test_str_no_session_when_none(self):
        assert 'ses-' not in str(_make_scan(session=None))

    def test_to_dict_keys(self):
        d = _make_scan().to_dict()
        for key in ('sub', 'ses', 'task', 'space', 'seg', 'modality', 'tr',
                    'condition', 'scan_string'):
            assert key in d

    def test_different_subjects_not_equal(self):
        assert _make_scan(subject_id=SUBJECT_A) != _make_scan(subject_id=SUBJECT_B)

    def test_different_tasks_not_equal(self):
        assert _make_scan(task='rfMRIREST1LR') != _make_scan(task='tfMRIWMLR')


# ==============================================================================
# Path construction  (integration — needs DataLoader init, no file I/O)
# ==============================================================================

@integration
class TestPathConstruction:

    @pytest.mark.parametrize("task, modality, expected_subdir, expected_ext", [
        ('rfMRIREST1LR',  'MRI3T', 'parcellated_data',     '.ptseries.nii'),
        ('tfMRIWMLR',     'MRI3T', 'parcellated_data',     '.ptseries.nii'),
        ('rfMRIREST17TPA','MRI7T', 'parcellated_data_7T',  '.ptseries.nii'),
    ])
    def test_subdir_and_ext(self, hcpya_loader, task, modality, expected_subdir, expected_ext):
        scan = _make_scan(task=task, modality=modality, tr=hcpya_loader.get_tr(modality))
        fp = hcpya_loader.get_scan_filepath(scan)
        assert expected_subdir in str(fp), f"Expected subdir '{expected_subdir}' in {fp}"
        assert str(fp).endswith(expected_ext), f"Expected extension '{expected_ext}', got {fp.name}"

    def test_exact_path_mri3t_rest(self, hcpya_loader):
        """Full path assertion for a known rest scan."""
        scan = _make_scan(subject_id=SUBJECT_A, task='rfMRIREST1LR')
        expected = (
            HCPYA_DERIV
            / 'parcellated_data'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION}_task-rfMRIREST1LR'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.ptseries.nii'
        )
        assert hcpya_loader.get_scan_filepath(scan) == expected

    def test_exact_path_mri3t_task(self, hcpya_loader):
        """Full path assertion for a known task scan."""
        scan = _make_scan(subject_id=SUBJECT_A, task='tfMRIWMLR')
        expected = (
            HCPYA_DERIV
            / 'parcellated_data'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION}_task-tfMRIWMLR'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.ptseries.nii'
        )
        assert hcpya_loader.get_scan_filepath(scan) == expected

    def test_path_contains_subject_id(self, hcpya_loader):
        fp = hcpya_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_A))
        assert f'sub-{SUBJECT_A}' in str(fp)

    def test_path_contains_task(self, hcpya_loader):
        fp = hcpya_loader.get_scan_filepath(_make_scan(task='tfMRIWMLR'))
        assert 'tfMRIWMLR' in str(fp)

    def test_different_subjects_give_different_paths(self, hcpya_loader):
        fp_a = hcpya_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_A))
        fp_b = hcpya_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_B))
        assert fp_a != fp_b

    def test_different_tasks_give_different_paths(self, hcpya_loader):
        fp_rest = hcpya_loader.get_scan_filepath(_make_scan(task='rfMRIREST1LR'))
        fp_task = hcpya_loader.get_scan_filepath(_make_scan(task='tfMRIWMLR'))
        assert fp_rest != fp_task


# ==============================================================================
# File existence  (integration)
# ==============================================================================

@integration
class TestFileExistence:

    @pytest.mark.parametrize("subject_id, task", [
        (SUBJECT_A, 'rfMRIREST1LR'),
        (SUBJECT_A, 'rfMRIREST1RL'),
        (SUBJECT_A, 'rfMRIREST2LR'),
        (SUBJECT_A, 'rfMRIREST2RL'),
        (SUBJECT_A, 'tfMRIWMLR'),
        (SUBJECT_A, 'tfMRIWMRL'),
        (SUBJECT_B, 'rfMRIREST1LR'),
        (SUBJECT_B, 'tfMRIWMLR'),
    ])
    def test_file_exists(self, hcpya_loader, subject_id, task):
        scan = _make_scan(subject_id=subject_id, task=task)
        fp = hcpya_loader.get_scan_filepath(scan)
        assert fp.exists(), f"Expected file not found: {fp}"


# ==============================================================================
# Timeseries loading  (integration)
# ==============================================================================

@integration
class TestTimeseries:
    """
    Tests load scans with nuisance_regression=False (raw load) unless the test
    is explicitly about the regression step, so that loading behaviour and
    regression behaviour are tested independently.
    """

    # ── Shape and dtype ───────────────────────────────────────────────────────

    def test_rest_shape(self, hcpya_loader):
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1LR'), nuisance_regression=False, cache=False
        )
        assert ts.shape == REST_SHAPE, (
            f"Expected {REST_SHAPE}, got {ts.shape}"
        )

    def test_task_shape(self, hcpya_loader):
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(task='tfMRIWMLR'), nuisance_regression=False, cache=False
        )
        assert ts.shape == WM_SHAPE, (
            f"Expected {WM_SHAPE}, got {ts.shape}"
        )

    def test_parcels_axis_is_first(self, hcpya_loader):
        """Confirm (n_parcels, n_timepoints) orientation, not transposed."""
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_dtype_float32(self, hcpya_loader):
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.dtype == np.float32

    # ── Data quality ─────────────────────────────────────────────────────────

    def test_no_nan_rest(self, hcpya_loader):
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert not np.isnan(ts).any()

    def test_no_nan_task(self, hcpya_loader):
        ts = hcpya_loader.load_scan_timeseries(
            _make_scan(task='tfMRIWMLR'), nuisance_regression=False, cache=False
        )
        assert not np.isnan(ts).any()

    # ── Identity: different inputs produce different outputs ──────────────────

    def test_different_subjects_differ(self, hcpya_loader):
        ts_a = hcpya_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_A), nuisance_regression=False, cache=False
        )
        ts_b = hcpya_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_B), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_a, ts_b), (
            "Two different subjects returned identical timeseries — "
            "likely the same file was loaded for both."
        )

    def test_different_runs_differ(self, hcpya_loader):
        ts_lr = hcpya_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1LR'), nuisance_regression=False, cache=False
        )
        ts_rl = hcpya_loader.load_scan_timeseries(
            _make_scan(task='rfMRIREST1RL'), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_lr, ts_rl), (
            "LR and RL runs returned identical timeseries — "
            "likely the same file was loaded for both."
        )

    # ── Cache consistency ─────────────────────────────────────────────────────

    def test_cache_returns_identical_array(self, hcpya_loader):
        scan = _make_scan(task='rfMRIREST2LR')   # use a different run to avoid
        ts1 = hcpya_loader.load_scan_timeseries(  # colliding with other tests
            scan, nuisance_regression=False, cache=True
        )
        ts2 = hcpya_loader.load_scan_timeseries(
            scan, nuisance_regression=False, cache=True
        )
        assert np.array_equal(ts1, ts2)
        hcpya_loader.clear_cache()

    # ── Nuisance regression ───────────────────────────────────────────────────

    def test_nuisance_regression_changes_data(self, hcpya_loader):
        scan = _make_scan()
        ts_raw = hcpya_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=False)
        ts_reg = hcpya_loader.load_scan_timeseries(scan, nuisance_regression=True,  cache=False)
        assert not np.allclose(ts_raw, ts_reg)

    def test_nuisance_regression_reduces_global_signal_variance(self, hcpya_loader):
        """Regressing out the global signal should reduce its temporal variance."""
        scan = _make_scan()
        ts_raw = hcpya_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=False)
        ts_reg = hcpya_loader.load_scan_timeseries(scan, nuisance_regression=True,  cache=False)
        gs_raw = np.mean(ts_raw, axis=0)   # global signal across parcels
        gs_reg = np.mean(ts_reg, axis=0)
        assert gs_reg.var() < gs_raw.var()


# ==============================================================================
# Registry round-trip  (integration)
# ==============================================================================

@integration
class TestRegistryRoundtrip:

    def test_export_and_reload_preserves_scans(self, tmp_path):
        """
        Register scans → export to CSV → clear registry → reload → compare.
        Uses a fresh DataLoader (function-scoped) so the session fixture is unaffected.
        """
        loader = DataLoader('HCP-YA', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)

        original_scans = loader.register_scans(
            subject_ids=[SUBJECT_A, SUBJECT_B],
            sessions=[SESSION],
            tasks=['rfMRIREST1LR', 'tfMRIWMLR'],
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
        loader = DataLoader('HCP-YA', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)
        loader.register_scans(
            subject_ids=[SUBJECT_A], sessions=[SESSION],
            tasks=['rfMRIREST1LR'], space=SPACE, seg=SEG, modality='MRI3T',
        )
        loader.export_scan_registry('test_registry.csv')

        import pandas as pd
        df = pd.read_csv(tmp_path / 'test_registry.csv', index_col=0)
        for col in ('sub', 'ses', 'task', 'modality', 'condition', 'tr', 'scan_string'):
            assert col in df.columns, f"Missing column: {col}"
