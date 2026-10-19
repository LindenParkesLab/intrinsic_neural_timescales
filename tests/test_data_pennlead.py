"""
Unit and integration tests for data.py — Penn_LEAD dataset.

Test tiers
----------
Unit tests (no disk access):
    TestConfig          — YAML config loading and field values
    TestScanInfo        — ScanInfo construction, condition derivation, round-trip serialisation

Integration tests (require Penn_LEAD data mount, auto-skipped if unavailable):
    TestPathConstruction — get_scan_filepath template rendering and exact path assertions
    TestFileExistence    — expected files are present on disk for both sessions
    TestTimeseries       — loaded array shape, dtype, NaN, cache, and nuisance regression
    TestLongitudinal     — both sessions of a subject resolve to files on disk
    TestRegistryRoundtrip — export → CSV → reload preserves ScanInfo objects exactly

Notes
-----
Penn_LEAD is a longitudinal dataset; subjects have up to two (or three) sessions
labelled 'ses-1', 'ses-2', etc. (single-digit, no zero padding).
Timeseries files are TSV; pandas reads them as (n_timepoints, n_parcels) which
data.py transposes to (n_parcels, n_timepoints).
N_PARCELS is 400 (Schaefer4007; the TSV has 400 parcel columns with no index column).
"""

import pytest
import numpy as np
from pathlib import Path

from src.data import DataLoader, ScanInfo, load_dataset_config
from tests.conftest import integration_pennlead as integration


# ── Shared constants ──────────────────────────────────────────────────────────

SUBJECT_A  = '19861'    # has ses-1 and ses-2
SUBJECT_B  = '20238'    # has ses-1 and ses-2
SESSION_1  = '1'
SESSION_2  = '2'
SEG        = 'Schaefer4007'
SPACE      = 'MNI152NLin2009cAsym'

PENNLEAD_DERIV = Path('/mnt/lindenmp_bignas/lindenmp/research_data/Penn_LEAD/derivatives')

# Expected output shapes: (n_parcels, n_timepoints).
# 400 parcels for Schaefer4007; the TSV has 400 parcel columns (no index column).
# 522 TRs @ 0.8 s ≈ 7 min for resting-state session 1.
N_PARCELS  = 400
REST_SHAPE = (N_PARCELS, 522)   # task-rest, ses-1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scan(subject_id=SUBJECT_A, task='rest', modality='MRI3T',
               session=SESSION_1, seg=SEG, space=SPACE, tr=0.8):
    return ScanInfo(subject_id=subject_id, task=task, space=space, seg=seg,
                    modality=modality, tr=tr, session=session)


# ==============================================================================
# Config  (pure unit — no disk access)
# ==============================================================================

class TestConfig:

    def test_loads_without_error(self):
        cfg = load_dataset_config('Penn_LEAD')
        assert cfg.name == 'Penn_LEAD'

    def test_modalities_present(self):
        cfg = load_dataset_config('Penn_LEAD')
        assert set(cfg.modalities) == {'MRI3T'}

    def test_tr(self):
        cfg = load_dataset_config('Penn_LEAD')
        assert cfg.modalities['MRI3T'].tr == pytest.approx(0.8)


    def test_behavior_handler(self):
        assert load_dataset_config('Penn_LEAD').behavior_handler == 'penn_lead'

    def test_file_ext(self):
        cfg = load_dataset_config('Penn_LEAD')
        assert cfg.modalities['MRI3T'].file_ext == '.tsv'

    def test_task_events_handler(self):
        cfg = load_dataset_config('Penn_LEAD')
        assert cfg.modalities['MRI3T'].task_events_handler == 'bids_tsv'

    def test_no_subject_id_transform(self):
        assert load_dataset_config('Penn_LEAD').subject_id_rawdata_transform is None


# ==============================================================================
# ScanInfo  (pure unit — no disk access)
# ==============================================================================

class TestScanInfo:

    def test_condition_rest(self):
        assert _make_scan(task='rest').condition == 'rest'

    def test_condition_task_nback(self):
        assert _make_scan(task='nback').condition == 'task'

    def test_roundtrip_session1(self):
        scan = _make_scan(session=SESSION_1)
        assert ScanInfo.from_dict(scan.to_dict()) == scan

    def test_roundtrip_session2(self):
        scan = _make_scan(session=SESSION_2)
        reconstructed = ScanInfo.from_dict(scan.to_dict())
        assert reconstructed.session == SESSION_2
        assert reconstructed == scan

    def test_roundtrip_no_session(self):
        scan = _make_scan(session=None)
        reconstructed = ScanInfo.from_dict(scan.to_dict())
        assert reconstructed.session is None
        assert reconstructed == scan

    def test_str_contains_subject(self):
        assert f'sub-{SUBJECT_A}' in str(_make_scan())

    def test_str_contains_task(self):
        assert 'task-rest' in str(_make_scan())

    def test_str_contains_session(self):
        assert f'ses-{SESSION_1}' in str(_make_scan())

    def test_str_session2(self):
        assert f'ses-{SESSION_2}' in str(_make_scan(session=SESSION_2))

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
        assert _make_scan(task='rest') != _make_scan(task='nback')

    def test_different_sessions_not_equal(self):
        assert _make_scan(session=SESSION_1) != _make_scan(session=SESSION_2)


# ==============================================================================
# Path construction  (integration — needs DataLoader init, no file I/O)
# ==============================================================================

@integration
class TestPathConstruction:

    def test_subdir_is_xcpd(self, penn_lead_loader):
        fp = penn_lead_loader.get_scan_filepath(_make_scan())
        assert 'xcpd' in str(fp)

    def test_extension_is_tsv(self, penn_lead_loader):
        fp = penn_lead_loader.get_scan_filepath(_make_scan())
        assert str(fp).endswith('.tsv')

    def test_filename_includes_run_01(self, penn_lead_loader):
        """Penn_LEAD filenames embed run-01 in the template."""
        fp = penn_lead_loader.get_scan_filepath(_make_scan())
        assert 'run-01' in fp.name

    def test_exact_path_rest_ses1(self, penn_lead_loader):
        scan = _make_scan(subject_id=SUBJECT_A, task='rest', session=SESSION_1)
        expected = (
            PENNLEAD_DERIV
            / 'xcpd'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION_1}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION_1}_task-rest_run-01'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.tsv'
        )
        assert penn_lead_loader.get_scan_filepath(scan) == expected

    def test_exact_path_nback_ses1(self, penn_lead_loader):
        scan = _make_scan(subject_id=SUBJECT_A, task='nback', session=SESSION_1)
        expected = (
            PENNLEAD_DERIV
            / 'xcpd'
            / f'sub-{SUBJECT_A}'
            / f'ses-{SESSION_1}'
            / 'func'
            / f'sub-{SUBJECT_A}_ses-{SESSION_1}_task-nback_run-01'
              f'_space-{SPACE}_seg-{SEG}_stat-mean_timeseries.tsv'
        )
        assert penn_lead_loader.get_scan_filepath(scan) == expected

    def test_session2_path_differs_from_session1(self, penn_lead_loader):
        fp_s1 = penn_lead_loader.get_scan_filepath(_make_scan(session=SESSION_1))
        fp_s2 = penn_lead_loader.get_scan_filepath(_make_scan(session=SESSION_2))
        assert fp_s1 != fp_s2
        assert f'ses-{SESSION_1}' in str(fp_s1)
        assert f'ses-{SESSION_2}' in str(fp_s2)

    def test_different_subjects_give_different_paths(self, penn_lead_loader):
        fp_a = penn_lead_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_A))
        fp_b = penn_lead_loader.get_scan_filepath(_make_scan(subject_id=SUBJECT_B))
        assert fp_a != fp_b

    def test_different_tasks_give_different_paths(self, penn_lead_loader):
        fp_rest  = penn_lead_loader.get_scan_filepath(_make_scan(task='rest'))
        fp_nback = penn_lead_loader.get_scan_filepath(_make_scan(task='nback'))
        assert fp_rest != fp_nback


# ==============================================================================
# File existence  (integration)
# ==============================================================================

@integration
class TestFileExistence:

    @pytest.mark.parametrize("subject_id, task, session", [
        (SUBJECT_A, 'rest',  SESSION_1),
        (SUBJECT_A, 'nback', SESSION_1),
        (SUBJECT_A, 'rest',  SESSION_2),
        (SUBJECT_A, 'nback', SESSION_2),
        (SUBJECT_B, 'rest',  SESSION_1),
        (SUBJECT_B, 'nback', SESSION_1),
        (SUBJECT_B, 'rest',  SESSION_2),
        (SUBJECT_B, 'nback', SESSION_2),
    ])
    def test_file_exists(self, penn_lead_loader, subject_id, task, session):
        fp = penn_lead_loader.get_scan_filepath(
            _make_scan(subject_id=subject_id, task=task, session=session)
        )
        assert fp.exists(), f"Expected file not found: {fp}"


# ==============================================================================
# Timeseries loading  (integration)
# ==============================================================================

@integration
class TestTimeseries:
    """
    Tests load scans with nuisance_regression=False (raw load).

    Note on NaN values: Penn_LEAD XCPD-processed TSV files contain NaN in a
    small number of edge-of-mask parcels (typically 3 out of 400). These are
    an inherent property of the XCPD pipeline and are not a loading bug.
    Tests therefore assert a bounded NaN count rather than zero NaN.

    Note on nuisance regression: nuisance regression is disabled for Penn_LEAD
    in DATASET_PARAMS (nuisance_regression=False), so no regression tests are
    included here — they would exercise infrastructure Penn_LEAD doesn't use.
    """

    def test_rest_shape_ses1(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='rest', session=SESSION_1),
            nuisance_regression=False, cache=False
        )
        assert ts.shape == REST_SHAPE, f"Expected {REST_SHAPE}, got {ts.shape}"

    def test_parcels_axis_is_first(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_task_has_correct_parcels(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='nback'), nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_session2_has_correct_parcels(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='rest', session=SESSION_2),
            nuisance_regression=False, cache=False
        )
        assert ts.shape[0] == N_PARCELS

    def test_dtype_float32(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        assert ts.dtype == np.float32

    def test_nan_parcels_bounded_rest_ses1(self, penn_lead_loader):
        """Penn_LEAD has a small number of NaN edge-of-mask parcels; assert it stays under 10."""
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(), nuisance_regression=False, cache=False
        )
        nan_parcels = np.isnan(ts).any(axis=1).sum()
        assert nan_parcels < 10, f"Too many NaN parcels in rest ses-1: {nan_parcels}"

    def test_nan_parcels_bounded_rest_ses2(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(session=SESSION_2), nuisance_regression=False, cache=False
        )
        nan_parcels = np.isnan(ts).any(axis=1).sum()
        assert nan_parcels < 10, f"Too many NaN parcels in rest ses-2: {nan_parcels}"

    def test_nan_parcels_bounded_task(self, penn_lead_loader):
        ts = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='nback'), nuisance_regression=False, cache=False
        )
        nan_parcels = np.isnan(ts).any(axis=1).sum()
        assert nan_parcels < 10, f"Too many NaN parcels in nback: {nan_parcels}"

    def test_different_subjects_differ(self, penn_lead_loader):
        ts_a = penn_lead_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_A), nuisance_regression=False, cache=False
        )
        ts_b = penn_lead_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_B), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_a, ts_b), (
            "Two different subjects returned identical timeseries."
        )

    def test_different_sessions_differ(self, penn_lead_loader):
        """Session 2 timeseries should differ from session 1 for the same subject."""
        ts_s1 = penn_lead_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_A, session=SESSION_1),
            nuisance_regression=False, cache=False
        )
        ts_s2 = penn_lead_loader.load_scan_timeseries(
            _make_scan(subject_id=SUBJECT_A, session=SESSION_2),
            nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_s1, ts_s2), (
            "Session 1 and session 2 returned identical timeseries."
        )

    def test_different_tasks_differ(self, penn_lead_loader):
        ts_rest  = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='rest'),  nuisance_regression=False, cache=False
        )
        ts_nback = penn_lead_loader.load_scan_timeseries(
            _make_scan(task='nback'), nuisance_regression=False, cache=False
        )
        assert not np.array_equal(ts_rest, ts_nback), (
            "Rest and nback tasks returned identical timeseries."
        )

    def test_cache_returns_identical_array(self, penn_lead_loader):
        scan = _make_scan(task='rest', session=SESSION_2)
        ts1 = penn_lead_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=True)
        ts2 = penn_lead_loader.load_scan_timeseries(scan, nuisance_regression=False, cache=True)
        # equal_nan=True: NaN == NaN is True when checking cache identity
        assert np.array_equal(ts1, ts2, equal_nan=True)
        penn_lead_loader.clear_cache()


# ==============================================================================
# Longitudinal  (integration)
# ==============================================================================

@integration
class TestLongitudinal:
    """
    Tests for the longitudinal data structure unique to Penn_LEAD.
    Verifies that multi-session data resolves correctly.
    """

    def test_both_sessions_load_for_subject_a(self, penn_lead_loader):
        """Both sessions must be independently loadable for SUBJECT_A."""
        for session in (SESSION_1, SESSION_2):
            fp = penn_lead_loader.get_scan_filepath(
                _make_scan(subject_id=SUBJECT_A, session=session)
            )
            assert fp.exists(), f"Session {session} file missing: {fp}"


# ==============================================================================
# Registry round-trip  (integration)
# ==============================================================================

@integration
class TestRegistryRoundtrip:

    def test_export_and_reload_preserves_scans(self, tmp_path):
        """Exercises multi-session round-trip (both session labels must survive CSV serialisation)."""
        loader = DataLoader('Penn_LEAD', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)

        original_scans = loader.register_scans(
            subject_ids=[SUBJECT_A, SUBJECT_B],
            sessions=[SESSION_1, SESSION_2],
            tasks=['rest', 'nback'],
            space=SPACE, seg=SEG, modality='MRI3T',
        )

        loader.export_scan_registry('test_registry.csv')
        loader.scan_registry = []
        loader.load_scan_registry('test_registry.csv')

        assert len(loader.scan_registry) == len(original_scans)
        for original, reloaded in zip(original_scans, loader.scan_registry):
            assert original == reloaded, (
                f"Mismatch after round-trip:\n  original: {original}\n  reloaded: {reloaded}"
            )

    def test_session_labels_preserved_after_roundtrip(self, tmp_path):
        """Session labels '1' and '2' must survive CSV serialisation without being cast to int."""
        loader = DataLoader('Penn_LEAD', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)
        loader.register_scans(
            subject_ids=[SUBJECT_A],
            sessions=[SESSION_1, SESSION_2],
            tasks=['rest'],
            space=SPACE, seg=SEG, modality='MRI3T',
        )
        loader.export_scan_registry('test_registry.csv')
        loader.scan_registry = []
        loader.load_scan_registry('test_registry.csv')

        sessions_reloaded = {s.session for s in loader.scan_registry}
        assert SESSION_1 in sessions_reloaded, f"Session '{SESSION_1}' lost after round-trip"
        assert SESSION_2 in sessions_reloaded, f"Session '{SESSION_2}' lost after round-trip"

    def test_export_csv_has_expected_columns(self, tmp_path):
        loader = DataLoader('Penn_LEAD', output_dir=str(tmp_path),
                            load_behavior=False, verbose=False)
        loader.register_scans(
            subject_ids=[SUBJECT_A], sessions=[SESSION_1],
            tasks=['rest'], space=SPACE, seg=SEG, modality='MRI3T',
        )
        loader.export_scan_registry('test_registry.csv')

        import pandas as pd
        df = pd.read_csv(tmp_path / 'test_registry.csv', index_col=0)
        for col in ('sub', 'ses', 'task', 'modality', 'condition', 'tr', 'scan_string'):
            assert col in df.columns, f"Missing column: {col}"


# ==============================================================================
# Longitudinal registry: per-session completeness, session 3, CuBIDS variants
# ==============================================================================

def _stub_loader(present):
    """A DataLoader carrying only what validate_scan_registry reads. ``present`` is the set of
    (subject, session, task) whose file exists; every task scan found also has events.

    Built through ``__new__`` so the test needs no data mount.
    """
    loader = DataLoader.__new__(DataLoader)
    loader.dataset = 'Penn_LEAD'
    loader.config = load_dataset_config('Penn_LEAD')
    loader.scan_registry = [_make_scan(subject_id=s, session=ses, task=t)
                            for s in ('A', 'B') for ses in ('1', '2', '3') for t in ('rest', 'nback')]
    loader.get_scan_filepath = lambda scan: type('P', (), {
        'exists': lambda self: (scan.subject_id, scan.session, scan.task) in present})()
    loader.get_task_events = lambda scan: object() if scan.condition == 'task' else None
    return loader


# A has sessions 1 and 2 complete and session 3 missing its n-back; B has only session 1.
PRESENT = {('A', '1', 'rest'), ('A', '1', 'nback'), ('A', '2', 'rest'), ('A', '2', 'nback'),
           ('A', '3', 'rest'), ('B', '1', 'rest'), ('B', '1', 'nback')}


class TestPerSessionCompleteness:

    def test_session_unit_drops_only_the_incomplete_session(self):
        loader = _stub_loader(PRESENT)
        res = loader.validate_scan_registry(verbose=False, remove_incomplete_subjects=True,
                                            incomplete_unit='session')
        kept = {(s.subject_id, s.session) for s in loader.scan_registry}
        assert kept == {('A', '1'), ('A', '2'), ('B', '1')}
        assert ('A', '3') in res['removed_sessions']
        assert res['removed_subjects'] == []            # nobody lost every session

    def test_subject_unit_is_the_unchanged_default(self):
        """Every other dataset: one missing scan removes the subject from every session."""
        loader = _stub_loader(PRESENT)
        res = loader.validate_scan_registry(verbose=False, remove_incomplete_subjects=True)
        assert loader.scan_registry == []                # both A and B miss something
        assert res['removed_subjects'] == ['A', 'B'] and res['removed_sessions'] == []

    def test_a_subject_with_no_complete_session_is_reported_removed(self):
        loader = _stub_loader({('A', '1', 'rest')})
        res = loader.validate_scan_registry(verbose=False, remove_incomplete_subjects=True,
                                            incomplete_unit='session')
        assert loader.scan_registry == [] and res['removed_subjects'] == ['A', 'B']

    def test_an_unknown_unit_raises(self):
        with pytest.raises(ValueError, match='incomplete_unit'):
            _stub_loader(PRESENT).validate_scan_registry(verbose=False, incomplete_unit='run')

    def test_config_registers_three_sessions_judged_per_session(self):
        from src.config import DATASET_PARAMS
        params = DATASET_PARAMS['Penn_LEAD']
        assert params['sessions'] == ['1', '2', '3']
        assert params['incomplete_unit'] == 'session'
        # every other dataset keeps the per-subject rule
        assert all(p.get('incomplete_unit', 'subject') == 'subject'
                   for name, p in DATASET_PARAMS.items() if name != 'Penn_LEAD')


class TestCubidsVariants:
    """CuBIDS renamed non-dominant scans ``acq-VARIANT<reason>``. VARIANTObliquity is read;
    VARIANTNumVolumes (wrong frame count) must never be."""

    def test_fallback_names_obliquity_and_nothing_else(self):
        mod = load_dataset_config('Penn_LEAD').modalities['MRI3T']
        assert len(mod.filename_template_fallbacks) == 1
        fallback = mod.filename_template_fallbacks[0]
        assert 'acq-VARIANTObliquity' in fallback
        assert '*' not in fallback and 'NumVolumes' not in fallback

    def test_events_are_globbed_so_a_variants_events_are_found(self):
        mod = load_dataset_config('Penn_LEAD').modalities['MRI3T']
        assert mod.task_events_use_glob
        assert mod.task_events_file_template.endswith('_run-01_events.tsv')

    @pytest.mark.parametrize('name, acq', [
        ('sub-1_ses-2_task-rest_acq-VARIANTObliquity_run-01_space-X_seg-Y_stat-mean_timeseries.tsv',
         'VARIANTObliquity'),
        ('sub-1_ses-2_task-rest_run-01_space-X_seg-Y_stat-mean_timeseries.tsv', None),
    ])
    def test_get_scan_acq_reads_the_resolved_name(self, name, acq):
        loader = DataLoader.__new__(DataLoader)
        loader.get_scan_filepath = lambda scan: Path('/x') / name
        assert loader.get_scan_acq(_make_scan()) == acq


@integration
class TestCubidsVariantsOnDisk:
    """Scans identified from the QC tables on 2026-09-24."""

    def test_an_obliquity_scan_resolves(self, penn_lead_loader):
        scan = _make_scan(subject_id='21553', session=SESSION_2, task='nback')
        path = penn_lead_loader.get_scan_filepath(scan)
        assert path.exists() and 'acq-VARIANTObliquity' in path.name
        assert penn_lead_loader.get_scan_acq(scan) == 'VARIANTObliquity'

    def test_an_obliquity_scans_events_share_its_acq(self, penn_lead_loader):
        """The events glob could in principle find a different run's file; it must find this one."""
        import glob as _glob
        scan = _make_scan(subject_id='21553', session=SESSION_2, task='nback')
        assert penn_lead_loader.get_task_events(scan) is not None
        func = penn_lead_loader.rawdata_dir / 'sub-21553' / 'ses-2' / 'func'
        matches = _glob.glob(str(func / 'sub-21553_ses-2_task-nback*_run-01_events.tsv'))
        assert len(matches) == 1 and 'acq-VARIANTObliquity' in matches[0]

    def test_a_numvolumes_scan_is_never_found(self, penn_lead_loader):
        """sub-20139 ses-1's only n-back is a 521-frame VARIANTNumVolumes run."""
        path = penn_lead_loader.get_scan_filepath(
            _make_scan(subject_id='20139', session=SESSION_1, task='nback'))
        assert not path.exists()

    def test_a_dominant_scan_reports_no_acq(self, penn_lead_loader):
        assert penn_lead_loader.get_scan_acq(_make_scan(subject_id=SUBJECT_A)) is None
