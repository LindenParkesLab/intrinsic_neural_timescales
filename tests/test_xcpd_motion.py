"""Shared XCP-D head motion (``load_xcpd_motion`` / ``xcpd_motion_columns``).

Both XCP-D datasets here -- WAND and Penn_LEAD -- carry per-run motion in ``*_desc-linc_qc.tsv``,
and each documents ``mean_fd`` identically in its ``*_motion.json`` ("Power et al. (2012)",
HeadRadius 50 mm), so one loader serves both and the values are comparable across them. What these tests pin: that the entities come from the FILENAME (pandas
drops the zero padding the scan registries use), that runs and acquisition variants collapse
before anything else, that only a genuinely multi-session dataset grows per-session columns, and
that the ``meanFD_rest`` / ``meanFD_task`` aliases ``build_nuisance`` reads resolve per dataset.

These run on stubs: no data mount, no NAS reads. The real-data joins are covered by each
dataset's own test_data_*.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import (XCPD_MOTION_SPEC, XCPD_QC_FIELDS, _qc_entities, load_xcpd_motion,
                      xcpd_motion_columns)
from src.gams import FD_COLUMN


class _Stub:
    """The attributes load_xcpd_motion reads, with a long table injected in place of the NAS."""

    def __init__(self, dataset, long=None, tmp_path=None):
        self.dataset = dataset
        self.output_dir = tmp_path
        self._long = long

    # xcpd_motion_columns goes through load_xcpd_motion, which we stub out per test.


@pytest.fixture
def long_table():
    """One row per run, as load_xcpd_motion returns it."""
    def _make(rows):
        return pd.DataFrame(rows, columns=['subject_id', 'session', 'task', 'run', 'acq', 'mean_fd'])
    return _make


def _columns(monkeypatch, dataset, df):
    monkeypatch.setattr('src.data.load_xcpd_motion', lambda loader, force=False: df)
    return xcpd_motion_columns(_Stub(dataset))


class TestEntitiesComeFromTheFilename:
    """pandas reads WAND's sub as 395 and Penn_LEAD's run as 1; the registries use 00395 and 01."""

    def test_zero_padding_survives(self):
        e = _qc_entities('sub-00395_ses-03_task-rest_space-MNI152NLin2009cAsym_desc-linc_qc.tsv')
        assert e['sub'] == '00395' and e['ses'] == '03' and e['task'] == 'rest'

    def test_run_and_acq_are_read(self):
        e = _qc_entities('sub-138950_ses-1_task-rest_run-01_space-MNI152NLin2009cAsym_desc-linc_qc.tsv')
        assert e['sub'] == '138950' and e['ses'] == '1' and e['run'] == '01'
        e = _qc_entities('sub-20188_ses-1_task-nback_acq-VARIANTObliquity_run-01_desc-linc_qc.tsv')
        assert e['sub'] == '20188' and e['acq'] == 'VARIANTObliquity'
        e = _qc_entities('sub-0001_task-rest_space-MNI152NLin2009cAsym_desc-linc_qc.tsv')
        assert e['sub'] == '0001' and 'ses' not in e

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        class _NoDir:
            dataset = 'WAND'
            output_dir = tmp_path
            class config:
                base_path = tmp_path / 'nope'
                derivatives_subdir = 'derivatives'
                modalities = {'MRI3T': type('M', (), {'data_subdir': 'xcpd'})()}
        assert load_xcpd_motion(_NoDir()).empty


class TestSpecCoversEveryXcpdDataset:
    def test_the_two_datasets(self):
        assert sorted(XCPD_MOTION_SPEC) == ['Penn_LEAD', 'WAND']

    def test_every_spec_dataset_has_an_fd_column_entry(self):
        """A dataset with motion but no FD_COLUMN entry silently gets NO nuisance covariates."""
        for dataset in XCPD_MOTION_SPEC:
            assert FD_COLUMN.get(dataset) == 'meanFD_rest'

    def test_fd_is_the_field_the_analyses_read(self):
        assert XCPD_QC_FIELDS[0] == 'mean_fd'
        for field in ('mean_relative_rms', 'num_censored_volumes'):
            assert field in XCPD_QC_FIELDS


class TestWideColumns:
    def test_single_session_dataset_gets_one_column_per_task(self, monkeypatch, long_table):
        df = long_table([('00395', '03', 'rest', None, None, 0.10),
                         ('00395', '03', 'reversallearning', None, None, 0.30)])
        out = _columns(monkeypatch, 'WAND', df)
        assert out.loc['00395', 'meanFD_reversallearning'] == pytest.approx(0.30)
        # meanFD_task averages the non-rest tasks, never rest itself
        assert out.loc['00395', 'meanFD_task'] == pytest.approx(0.30)
        assert not any('_ses' in c for c in out.columns)

    def test_runs_average_within_a_session_before_anything_else(self, monkeypatch, long_table):
        """The general rule. Penn_LEAD is the exception -- it reads run-01 only; see
        test_penn_lead_reads_only_the_analysed_run."""
        df = long_table([('00395', '02', 'rest', '01', None, 0.10),
                         ('00395', '02', 'rest', '02', None, 0.30),
                         ('00395', '02', 'reversallearning', '01', None, 0.20)])
        out = _columns(monkeypatch, 'WAND', df)
        assert out.loc['00395', 'meanFD_rest'] == pytest.approx(0.20)

    def test_penn_lead_gets_per_session_columns_and_session_one_aliases(self, monkeypatch, long_table):
        df = long_table([('138950', '1', 'rest', '01', None, 0.10),
                         ('138950', '2', 'rest', '01', None, 0.50),
                         ('138950', '1', 'nback', '01', None, 0.20),
                         ('138950', '2', 'nback', '01', None, 0.60)])
        out = _columns(monkeypatch, 'Penn_LEAD', df)
        assert out.loc['138950', 'meanFD_rest_ses1'] == pytest.approx(0.10)
        assert out.loc['138950', 'meanFD_rest_ses2'] == pytest.approx(0.50)
        # The alias is session 1, NOT the mean of the two: averaging would mix two timepoints of a
        # longitudinal sample into one covariate.
        assert out.loc['138950', 'meanFD_rest'] == pytest.approx(0.10)
        assert out.loc['138950', 'meanFD_task'] == pytest.approx(0.20)
        # meanFD_<task> carries the alias session too, so no column holds the across-session mean
        # (0.30 here). That is deliberate: nothing should read a covariate that pools timepoints.
        assert out.loc['138950', 'meanFD_rest'] != pytest.approx(0.30)
        assert out.loc['138950', 'meanFD_rest'] != out.loc['138950', 'meanFD_rest_ses2']

    def test_penn_lead_reads_only_the_analysed_run(self, monkeypatch, long_table):
        """Rest run-01 (522 frames, before the n-back) is the run analysed. Run-02 (383 frames,
        after it) and CuBIDS's wrong-length VARIANTNumVolumes runs must not enter the covariate."""
        df = long_table([('20188', '1', 'rest',  '01', None,                0.10),
                         ('20188', '1', 'rest',  '02', None,                0.90),
                         ('20188', '1', 'nback', '01', 'VARIANTObliquity',  0.20),
                         ('20188', '2', 'rest',  '01', 'VARIANTNumVolumes', 0.70),
                         ('20188', '2', 'nback', '01', None,                0.30)])
        out = _columns(monkeypatch, 'Penn_LEAD', df)
        assert out.loc['20188', 'meanFD_rest_ses1'] == pytest.approx(0.10)     # not (0.10+0.90)/2
        assert out.loc['20188', 'meanFD_nback_ses1'] == pytest.approx(0.20)    # Obliquity is kept
        assert 'meanFD_rest_ses2' not in out.columns or np.isnan(out.loc['20188', 'meanFD_rest_ses2'])
        assert out.loc['20188', 'meanFD_nback_ses2'] == pytest.approx(0.30)

    def test_run_filters_are_penn_lead_only(self):
        """Every other XCP-D dataset averages all its runs."""
        for dataset, spec in XCPD_MOTION_SPEC.items():
            if dataset != 'Penn_LEAD':
                assert spec.get('runs') is None and not spec.get('exclude_acq')


    def test_wand_task_alias_is_its_one_task(self, monkeypatch, long_table):
        df = long_table([('00395', '03', 'rest', None, None, 0.15),
                         ('00395', '03', 'reversallearning', None, None, 0.25)])
        out = _columns(monkeypatch, 'WAND', df)
        assert out.loc['00395', 'meanFD_rest'] == pytest.approx(0.15)
        assert out.loc['00395', 'meanFD_task'] == pytest.approx(0.25)
        assert out.index.name == 'subject_id'

    def test_a_subject_missing_a_task_keeps_the_others(self, monkeypatch, long_table):
        df = long_table([('00001', '03', 'rest', None, None, 0.10),
                         ('00002', '03', 'rest', None, None, 0.12),
                         ('00002', '03', 'reversallearning', None, None, 0.22)])
        out = _columns(monkeypatch, 'WAND', df)
        assert np.isnan(out.loc['00001', 'meanFD_reversallearning'])
        assert out.loc['00001', 'meanFD_rest'] == pytest.approx(0.10)

    def test_an_unknown_dataset_raises_rather_than_guessing(self, monkeypatch, long_table):
        monkeypatch.setattr('src.data.load_xcpd_motion', lambda loader, force=False: long_table([]))
        with pytest.raises(KeyError, match='XCPD_MOTION_SPEC'):
            xcpd_motion_columns(_Stub('HCP-YA'))

    def test_no_motion_on_disk_returns_empty_so_the_handler_can_warn(self, monkeypatch, long_table):
        monkeypatch.setattr('src.data.load_xcpd_motion',
                            lambda loader, force=False: pd.DataFrame())
        assert xcpd_motion_columns(_Stub('WAND')).empty
