import os
import glob
import re
import logging
import numpy as np
import scipy as sp
from scipy import stats as sp_stats
import pandas as pd
import nibabel as nib
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from fractions import Fraction
from sklearn.linear_model import LinearRegression
from collections import defaultdict

from snaplab_tools.signal import apply_frequency_filter

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / 'configs' / 'datasets'


def _find_project_root() -> Path:
    """Walk up from cwd looking for setup.py as a project root marker."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / 'setup.py').exists():
            return parent
    return Path.cwd()


# ==============================================================================
# Config dataclasses
# ==============================================================================

@dataclass
class ModalityConfig:
    """Per-modality settings loaded from a dataset YAML config."""
    tr: float
    data_subdir: str
    dir_template: str
    filename_template: str
    file_ext: str
    task_events_handler: Optional[str] = None
    task_events_file_template: Optional[str] = None
    task_events_use_glob: bool = False
    use_glob: bool = False
    filename_template_fallbacks: List[str] = field(default_factory=list)
    task_params: Dict[str, dict] = field(default_factory=dict)

    def template_fields(self, task: str) -> dict:
        """Extra template fields for ``task``: its ``task_params`` entry minus ``tr``."""
        return {k: v for k, v in self.task_params.get(task, {}).items() if k != 'tr'}


@dataclass
class DatasetConfig:
    """Top-level dataset configuration loaded from a YAML file."""
    name: str
    base_path: Path
    rawdata_subdir: str
    derivatives_subdir: str
    modalities: Dict[str, ModalityConfig]
    behavior_handler: Optional[str] = None
    rbc: bool = False
    subject_id_rawdata_transform: Optional[str] = None


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    """Load a DatasetConfig from the corresponding YAML file in configs/datasets/."""
    config_path = _CONFIG_DIR / f'{dataset_name}.yaml'
    if not config_path.exists():
        available = [p.stem for p in _CONFIG_DIR.glob('*.yaml')]
        raise FileNotFoundError(
            f"No config found for dataset '{dataset_name}'. "
            f"Expected: {config_path}\n"
            f"Available datasets: {available}"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    modalities = {}
    for mod_name, mod_raw in raw['modalities'].items():
        modalities[mod_name] = ModalityConfig(
            tr=float(mod_raw['tr']),
            data_subdir=mod_raw['data_subdir'],
            dir_template=mod_raw['dir_template'],
            filename_template=mod_raw['filename_template'],
            file_ext=mod_raw['file_ext'],
            task_events_handler=mod_raw.get('task_events_handler'),
            task_events_file_template=mod_raw.get('task_events_file_template'),
            task_events_use_glob=bool(mod_raw.get('task_events_use_glob', False)),
            use_glob=bool(mod_raw.get('use_glob', False)),
            filename_template_fallbacks=mod_raw.get('filename_template_fallbacks') or [],
            task_params={str(t): dict(p) for t, p in (mod_raw.get('task_params') or {}).items()},
        )

    return DatasetConfig(
        name=raw['name'],
        base_path=Path(raw['paths']['base']),
        rawdata_subdir=raw['paths']['rawdata_subdir'],
        derivatives_subdir=raw['paths']['derivatives_subdir'],
        modalities=modalities,
        behavior_handler=(raw.get('behavior') or {}).get('handler'),
        rbc=bool(raw.get('rbc', False)),
        subject_id_rawdata_transform=raw.get('subject_id_rawdata_transform'),
    )


# ==============================================================================
# ScanInfo
# ==============================================================================

@dataclass
class ScanInfo:
    """Container for preprocessed scan metadata matching BIDS naming conventions.

    Example filenames:
        sub-100206_ses-01_task-rfMRIREST1LR_space-fsLR_seg-Schaefer4007_stat-mean_timeseries.ptseries.nii
        sub-00395_ses-03_task-reversallearning_space-MNI152NLin2009cAsym_seg-Schaefer4007_stat-mean_timeseries.tsv
    """
    subject_id: str
    task: str
    space: str
    seg: str
    modality: str
    tr: float

    # Optional BIDS fields
    session: Optional[str] = None
    stat: Optional[str] = 'mean'
    run: Optional[str] = None
    desc: Optional[str] = None
    den: Optional[str] = None
    rbc: Optional[bool] = False

    def __post_init__(self):
        self.condition = 'rest' if 'rest' in self.task.lower() else 'task'

    def __str__(self):
        entities = [
            f"sub-{self.subject_id}",
            f"task-{self.task}",
            f"space-{self.space}",
            f"seg-{self.seg}",
        ]
        if self.session is not None:
            entities.append(f"ses-{self.session}")
        entities.append(f"modality-{self.modality}")
        entities.append(f"tr-{self.tr}")
        entities.append(f"condition-{self.condition}")
        if self.run is not None:
            entities.append(f"run-{self.run}")
        if self.den is not None:
            entities.append(f"den-{self.den}")
        if self.desc is not None:
            entities.append(f"desc-{self.desc}")
        return "_".join(entities)

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary suitable for CSV export."""
        return {
            'sub': self.subject_id,
            'ses': self.session,
            'task': self.task,
            'space': self.space,
            'seg': self.seg,
            'modality': self.modality,
            'tr': self.tr,
            'condition': self.condition,
            'run': self.run,
            'desc': self.desc,
            'den': self.den,
            'rbc': self.rbc,
            'stat': self.stat,
            'scan_string': str(self),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ScanInfo':
        """Reconstruct a ScanInfo from a dict (e.g. a CSV row).

        Safe against both numeric NaN (float) and the string 'nan' that pandas
        produces when a column is read with dtype=str and the cell was empty.
        Also handles bool fields correctly when they arrive as strings ('True'/'False').
        """
        def _opt(val) -> Optional[str]:
            """Return None for any flavour of missing, else the string value."""
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except (TypeError, ValueError):
                pass
            s = str(val)
            return None if s in ('nan', 'None', '') else s

        def _bool(val) -> bool:
            """Parse bool fields that may arrive as Python bool or as strings."""
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() == 'true'
            try:
                if pd.isna(val):
                    return False
            except (TypeError, ValueError):
                pass
            return bool(val)

        return cls(
            subject_id=str(d['sub']),
            task=str(d['task']),
            space=str(d['space']),
            seg=str(d['seg']),
            modality=str(d['modality']),
            tr=float(d['tr']),
            session=_opt(d.get('ses')),
            stat=_opt(d.get('stat')) or 'mean',
            run=_opt(d.get('run')),
            desc=_opt(d.get('desc')),
            den=_opt(d.get('den')),
            rbc=_bool(d.get('rbc', False)),
        )


# ==============================================================================
# Behavior handlers
# ==============================================================================

# HCP-YA head motion. The S1200 release writes one scalar per run to
# MNINonLinear/Results/<run>/Movement_RelativeRMS_mean.txt -- the mean *relative RMS
# displacement* in mm. That is not Power's framewise displacement, so the columns are named
# meanRelRMS_* rather than meanFD_* and should be reported as such.
HCPYA_MOVEMENT_FILE = 'Movement_RelativeRMS_mean.txt'
HCPYA_MOVEMENT_CACHE = 'hcpya_movement_relativerms.csv'


def load_hcpya_movement(loader: 'DataLoader', force: bool = False) -> pd.DataFrame:
    """Per-run mean relative RMS displacement for every HCP-YA subject on disk.

    Reading the ~15k one-line text files off the NAS takes ~30 s, which is too slow to pay on
    every DataLoader init, so the result is cached as a CSV under the project output directory
    and re-read from there. Delete the cache (or pass ``force``) after downloading more runs.

    Returns
    -------
    pd.DataFrame
        Indexed by subject_id, one column per HCP run directory (e.g. 'rfMRI_REST1_LR').
        Empty if the raw data directory is absent.
    """
    cache = loader.output_dir / 'data' / HCPYA_MOVEMENT_CACHE
    if cache.exists() and not force:
        return pd.read_csv(cache, index_col=0, dtype={'subject_id': str})

    raw = loader.base_dir / loader.config.rawdata_subdir
    if not raw.is_dir():
        logger.warning("No raw data directory at %s; skipping HCP-YA motion.", raw)
        return pd.DataFrame()

    rows = {}
    for sub_dir in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.isdigit()):
        results = sub_dir / 'MNINonLinear' / 'Results'
        if not results.is_dir():
            continue
        rec = {}
        for run_dir in results.iterdir():
            path = run_dir / HCPYA_MOVEMENT_FILE
            if path.exists():
                try:
                    rec[run_dir.name] = float(np.loadtxt(path))
                except Exception as e:      # a truncated download, not a reason to abort
                    logger.warning("Could not parse %s: %s", path, e)
        if rec:
            rows[sub_dir.name] = rec

    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    df.index.name = 'subject_id'
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        logger.info("Cached HCP-YA motion for %d subjects to %s", len(df), cache)
    return df


def _add_hcpya_motion(loader: 'DataLoader', df: pd.DataFrame) -> pd.DataFrame:
    """Add rest / task mean-relative-RMS columns to the HCP-YA behavior table.

    Rest and task motion are kept as separate columns rather than one grand mean. They are
    correlated (r ~ 0.82 -- motion is largely a subject trait) but not interchangeable, and the
    INT deflection is a task-minus-rest difference, so collapsing them would average over exactly
    the contrast the deflection is built on.

    Columns added (NaN where a subject has no runs of that kind):
      meanRelRMS_rest  -- mean over the 3T rest runs
      meanRelRMS_task  -- mean over all 3T task runs
      meanRelRMS_<TASK> -- mean over that task's runs, e.g. meanRelRMS_WM
    """
    mv = load_hcpya_movement(loader)
    if mv.empty:
        logger.warning("No HCP-YA motion data found; meanRelRMS columns not added.")
        return df

    rest_cols = [c for c in mv.columns if c.startswith('rfMRI_REST') and '7T' not in c]
    task_cols = [c for c in mv.columns if c.startswith('tfMRI_') and '7T' not in c]

    out = pd.DataFrame(index=mv.index)
    out['meanRelRMS_rest'] = mv[rest_cols].mean(axis=1) if rest_cols else np.nan
    out['meanRelRMS_task'] = mv[task_cols].mean(axis=1) if task_cols else np.nan
    for name in ('EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'):
        cols = [c for c in task_cols if c.startswith(f'tfMRI_{name}_')]
        if cols:
            out[f'meanRelRMS_{name}'] = mv[cols].mean(axis=1)

    return df.join(out, how='left')


def _behavior_hcp_unrestricted_restricted(loader: 'DataLoader') -> pd.DataFrame:
    base = loader.config.base_path
    unrest_files = list(base.glob('unrestricted_*.csv'))
    rest_files = list(base.glob('RESTRICTED_*.csv'))
    if not unrest_files:
        raise FileNotFoundError(f"No unrestricted behavior CSV found in {base}")
    if not rest_files:
        raise FileNotFoundError(f"No restricted behavior CSV found in {base}")

    df_u = pd.read_csv(unrest_files[0], index_col=0)
    df_r = pd.read_csv(rest_files[0], index_col=0)
    df = pd.merge(df_u, df_r, left_index=True, right_index=True)
    df['age'] = df['Age_in_Yrs']
    df['sex'] = (df['Gender'] == 'M').astype(int)
    df.index = df.index.astype(str)
    df.index.name = 'subject_id'
    words_to_convert = ['Emotion', 'Gambling', 'Language', 'Relational', 'Social']
    new_columns = df.columns.to_series()
    for word in words_to_convert:
        new_columns = new_columns.str.replace(word, word.upper(), regex=False)
    df.columns = new_columns
    # Added after the case fix above so the meanRelRMS_* names survive it verbatim.
    return _add_hcpya_motion(loader, df)

#: NDA's missing-value code.
HCPD_SENTINEL = 999

def _load_hcpd_beh_file(beh_dir: Path, fname: str, cols: list,
                         rename: dict = None, missing_vals: list = None,
                         row_filter: 'callable' = None,
                         filter_cols: list = None) -> pd.DataFrame:
    """Load one HCP-D behavioral TSV and return a tidy DataFrame indexed by subject_id.

    Parameters
    ----------
    cols : list of str
        Columns to extract. Missing ones are skipped silently (the release varies).
    rename : dict, optional
        Applied last, so ``cols`` always names the on-disk column.
    missing_vals : list, optional
        Extra values to mask *in addition to* :data:`HCPD_SENTINEL`, which is always masked.
    row_filter : callable, optional
        ``frame -> boolean Series``, applied to the raw rows **before** de-duplication.
        Several of these files hold more than one row per subject, and the extra rows are not
        repeat visits -- they are different instruments (self- vs caregiver-report, different
        age-band forms, different constructs). Without a filter, ``keep='first'`` picks whichever
        the file happens to list first, which silently mixes scales within a column. Use this to
        pin one respondent and one form.
    filter_cols : list of str, optional
        Extra columns ``row_filter`` needs (e.g. ``comqother``, ``version_form``). They are read
        but never returned.
    """
    path = beh_dir / fname
    try:
        hdr = pd.read_csv(path, sep='\t', nrows=0)
    except FileNotFoundError:
        logger.warning('HCP-D behavioral file not found: %s', path)
        return pd.DataFrame()
    available = [c for c in cols if c in hdr.columns]
    if not available:
        return pd.DataFrame()

    extra = [c for c in (filter_cols or []) if c in hdr.columns]
    df = pd.read_csv(path, sep='\t', skiprows=[1],
                     usecols=['src_subject_id'] + available + extra, low_memory=False)

    if row_filter is not None:
        keep = row_filter(df)
        n_dropped = int((~keep).sum())
        df = df[keep]
        logger.debug('%s: row_filter kept %d of %d rows', fname, len(df), len(df) + n_dropped)
    df = df.drop(columns=extra, errors='ignore')

    df['subject_id'] = df['src_subject_id'].astype(str) + 'V1MR'
    df = (df.drop(columns='src_subject_id')
            .set_index('subject_id')
            .loc[lambda x: ~x.index.duplicated(keep='first')])
    df = df.apply(pd.to_numeric, errors='coerce')

    for mv in [HCPD_SENTINEL] + list(missing_vals or []):
        df = df.replace(mv, np.nan)

    for c in df.columns:
        if df[c].notna().sum() == 0:
            logger.warning("HCP-D: column '%s' from %s is empty after masking %d "
                           "(no usable values in this release)", c, fname, HCPD_SENTINEL)

    if rename:
        df = df.rename(columns=rename)
    return df


def _hcpd_self_report(df: pd.DataFrame) -> pd.Series:
    """Rows a participant answered about themselves, excluding caregiver-report rows.

    """
    return ~df['comqother'].astype(str).str.contains('caregiver', case=False, na=False)


def _behavior_hcp_d(loader: 'DataLoader') -> pd.DataFrame:
    beh_dir  = loader.config.base_path / 'behavior'
    proc_dir = loader.rawdata_dir  # .../HCP_D/processed

    # --- Demographics ---
    df = pd.read_csv(beh_dir / 'socdem01.txt', skiprows=[1], delimiter='\t',
                     usecols=['src_subject_id', 'interview_age', 'sex'])
    df['subject_id'] = df['src_subject_id'].astype(str) + 'V1MR'
    df['age'] = df['interview_age'] / 12
    df['sex'] = (df['sex'] == 'M').astype(int)
    df = df.drop(columns=['src_subject_id', 'interview_age']).set_index('subject_id')

    # --- Mean framewise displacement ---
    df = df.join(pd.read_csv(loader.config.base_path / 'HCP-D_meanFD.csv', index_col=0),
                 how='left')

    # --- Behavioral measures ---
    lf = lambda *a, **kw: _load_hcpd_beh_file(beh_dir, *a, **kw)

    # NIH Toolbox — Cognition
    for frame in [
        lf('cogcomp01.txt', ['nih_fluidcogcomp_unadjusted', 'nih_crycogcomp_unadjusted']),
        lf('dccs01.txt',    ['nih_dccs_ageadjusted']),
        lf('flanker01.txt', ['nih_flanker_ageadjusted']),
        lf('lswmt01.txt',   ['tbx_ls']),
        lf('orrt01.txt',    ['tbx_reading_score']),
        lf('tpvt01.txt',    ['tbx_vocab_theta']),
        lf('pcps01.txt',    ['nih_patterncomp_ageadjusted']),
    ]:
        df = df.join(frame, how='left')

    # Emotion recognition
    df = df.join(
        lf('er4001.txt', ['er40_c_cr', 'er40_c_ang', 'er40_c_fear', 'er40_c_hap', 'er40_c_sad']),
        how='left')

    # Psychopathology
    for frame in [
        lf('cbcl01.txt', ['cbcl_internal_raw', 'cbcl_external_raw', 'cbcl_total_raw',
                           'cbcl_attention_raw', 'cbcl_anxiety_raw', 'cbcl_aggressive_raw']),
        lf('ysr01.txt',  ['cbcl_internal', 'cbcl_external', 'cbcl_total'],
                          rename={'cbcl_internal': 'ysr_internal',
                                  'cbcl_external': 'ysr_external',
                                  'cbcl_total':    'ysr_total'}),
    ]:
        df = df.join(frame, how='left')

    # Personality / temperament / impulsivity — self-report only; see _hcpd_self_report.
    for frame in [
        lf('bisbas01.txt', ['bissc_total', 'bas_drive', 'bas_fs', 'bas_rr'],
           row_filter=_hcpd_self_report, filter_cols=['comqother']),
        lf('upps01.txt',   ['premed_score', 'senseek_score', 'persev_score'],
           row_filter=_hcpd_self_report, filter_cols=['comqother']),
    ]:
        df = df.join(frame, how='left')

    # One subject has a bas_fs of 99. Prob coding error. NaN-ing it!
    if 'bas_fs' in df.columns:
        df.loc[df['bas_fs'] > 90, 'bas_fs'] = np.nan

    # --- Stress / affect ---
    # Perceived Stress
    df = df.join(lf('pss01.txt', ['nih_tlbx_tscore'],
                    rename={'nih_tlbx_tscore': 'pss_tscore'},
                    row_filter=_hcpd_self_report, filter_cols=['comqother']), how='left')

    # Fear/anxiety
    df = df.join(lf('tlbx_fearanx01.txt', ['nih_tlbx_tscore'],
                    rename={'nih_tlbx_tscore': 'fearanx_tscore'},
                    row_filter=lambda d: d['version_form'].astype(str)
                                          .str.contains('Fear FF Ages 8-17', na=False),
                    filter_cols=['version_form']), how='left')

    # Wellbeing
    df = df.join(lf('tlbx_wellbeing01.txt', ['nih_tlbx_tscore'],
                    rename={'nih_tlbx_tscore': 'life_satisfaction_tscore'},
                    row_filter=lambda d: (
                        d['version_form'].astype(str).str.contains(
                            'General Life Satisfaction', na=False)
                        & ~d['version_form'].astype(str).str.contains('Parent Report', na=False)),
                    filter_cols=['version_form']), how='left')

    # --- CARIT performance (from per-subject EVs) ---
    df = df.join(load_hcpd_carit(loader), how='left')

    return df

#: CARIT trial-outcome EVs
CARIT_EVS = ('go', 'miss', 'nogoCRWin', 'nogoCRLose', 'nogoFAWin', 'nogoFALose')
CARIT_RUN = 'PA' 
HCPD_CARIT_CACHE = 'hcpd_carit_performance_pa.csv'

def _carit_run_counts(ev_dir: Path) -> Optional[dict]:
    """Trial counts for one CARIT run, or None when the run is absent."""
    if not (ev_dir / 'go.txt').exists():
        return None
    counts = {}
    for ev in CARIT_EVS:
        path = ev_dir / f'{ev}.txt'
        try:
            with open(path) as f:
                counts[ev] = sum(1 for line in f if line.strip())
        except FileNotFoundError:
            counts[ev] = 0        # an outcome that never occurred gets no file
    return counts


def _carit_metrics(counts: dict) -> Optional[dict]:
    """Signal-detection and error measures from pooled CARIT trial counts.

    CARIT is a go/no-go task: frequent go trials build a prepotent response that rare no-go
    trials require withholding. That gives two independent error types, and they are not
    redundant -- a false alarm is a failure to inhibit, a miss is a lapse of attention.

    ``d_prime`` and ``criterion`` separate those from response bias.
    """
    go, miss = counts['go'], counts['miss']
    cr = counts['nogoCRWin'] + counts['nogoCRLose']
    fa = counts['nogoFAWin'] + counts['nogoFALose']
    n_go, n_nogo = go + miss, cr + fa
    if n_go == 0 or n_nogo == 0:
        return None

    hit_c = (go + 0.5) / (n_go + 1)
    fa_c = (fa + 0.5) / (n_nogo + 1)
    z_hit, z_fa = sp_stats.norm.ppf(hit_c), sp_stats.norm.ppf(fa_c)

    win_n, lose_n = counts['nogoCRWin'] + counts['nogoFAWin'], counts['nogoCRLose'] + counts['nogoFALose']
    return {
        'carit_nogo_acc':       cr / n_nogo,
        'carit_omission_rate':  miss / n_go,
        'carit_dprime':         z_hit - z_fa,
        'carit_criterion':      -0.5 * (z_hit + z_fa),
        'carit_nonresponder':   go == 0,
        'carit_fa_rate_win':    counts['nogoFAWin'] / win_n if win_n else np.nan,
        'carit_fa_rate_lose':   counts['nogoFALose'] / lose_n if lose_n else np.nan,
        'carit_perf_exclude':   (go / n_go < 0.50) or (cr / n_nogo < 0.25),
        'carit_n_go_trials':    n_go,
        'carit_n_nogo_trials':  n_nogo,
    }


def load_hcpd_carit(loader: 'DataLoader', force: bool = False) -> pd.DataFrame:
    """CARIT go/no-go performance per subject, from the PA run.

    PA is the run the imaging uses; see :data:`CARIT_RUN` for the measured comparison against
    the AP run and against pooling the two.

    Returns
    -------
    pd.DataFrame
        Indexed by subject_id, carrying the metrics from :func:`_carit_metrics`.
        Empty if the processed directory is absent.

    Notes
    -----
    No reaction time is available; see :data:`CARIT_EVS`.
    """
    cache = loader.output_dir / 'data' / HCPD_CARIT_CACHE
    if cache.exists() and not force:
        return pd.read_csv(cache, index_col=0, dtype={'subject_id': str})

    proc = loader.rawdata_dir
    if not proc.is_dir():
        logger.warning("No processed directory at %s; skipping HCP-D CARIT.", proc)
        return pd.DataFrame()

    rows = {}
    for sub_dir in sorted(p for p in proc.iterdir() if p.is_dir() and p.name.startswith('HCD')):
        ev_dir = (sub_dir / 'MNINonLinear' / 'Results'
                  / f'tfMRI_CARIT_{CARIT_RUN}' / 'EVs')
        counts = _carit_run_counts(ev_dir)
        if counts is None:
            continue
        metrics = _carit_metrics(counts)
        if metrics is None:
            continue
        rows[sub_dir.name.replace('_', '')] = metrics

    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    df.index.name = 'subject_id'
    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        logger.info("Wrote HCP-D CARIT %s performance for %d subjects to %s",
                    CARIT_RUN, len(df), cache)
    return df


def _behavior_pnc_rbc(loader: 'DataLoader') -> pd.DataFrame:
    base = loader.config.base_path
    df = pd.read_csv(base / 'study-PNC_desc-participants.tsv', index_col=0, delimiter='\t')
    df['sex'] = (df['sex'] == 'Male').astype(int)
    df.index = df.index.astype(str)
    df.index.name = 'subject_id'

    df_qc = pd.read_csv(base / 'study-PNC_desc-functional_qc.tsv', sep='\t', dtype={'participant_id': str})
    df_qc = df_qc[df_qc['task'] != 'idemo']
    exclude_cols = ['normCrossCorrExclude', 'motionExclude', 'fmriExclude']
    df_qc = df_qc.loc[(df_qc[exclude_cols] == 0).all(axis=1)]
    has_rest = set(df_qc.loc[df_qc['task'] == 'rest', 'participant_id'])
    has_frac2back = set(df_qc.loc[df_qc['task'] == 'frac2back', 'participant_id'])
    passing_ids = has_rest & has_frac2back

    df_qc = df_qc[df_qc['participant_id'].isin(passing_ids)]
    df_qc = df_qc.groupby(['participant_id', 'task'])['meanFD'].mean().reset_index()
    rest_rms = df_qc[df_qc['task'] == 'rest'].set_index('participant_id')['meanFD'].rename('meanRelRMS_rest')
    frac2back_rms = df_qc[df_qc['task'] == 'frac2back'].set_index('participant_id')['meanFD'].rename('meanRelRMS_frac2back')
    df_qc_final = pd.concat([rest_rms, frac2back_rms], axis=1)
    df_qc_final.index.name = 'subject_id'

    df = df.loc[df_qc_final.index]
    df = pd.merge(df, df_qc_final, left_index=True, right_index=True)

    # True framewise displacement, from the same per-run QC files every other XCP-D dataset uses.
    motion = xcpd_motion_columns(loader)
    if motion.empty:
        logger.warning("No PNC per-run motion found; meanFD columns not added (meanRelRMS_* "
                       "from the study QC table are still present).")
        return df
    return df.join(motion, how='left')


def _behavior_penn_lead(loader: 'DataLoader') -> pd.DataFrame:
    df = pd.read_csv(loader.config.base_path / 'participants.tsv', delimiter='\t')
    df = df[['participant_id', 'age_months', 'sex']]
    df.rename(columns={'participant_id': 'subject_id', 'age_months': 'age'}, inplace=True)
    df['subject_id'] = [s.split('-')[1] for s in df['subject_id']]
    df['age'] = df['age'] / 12
    df['sex'] = (df['sex'] == 'M').astype(int)
    df.set_index('subject_id', inplace=True)

    bids_dir = loader.config.base_path / 'bids'
    for subject_id in df.index:
        ses_tsv = bids_dir / f'sub-{subject_id}' / f'sub-{subject_id}_sessions.tsv'
        if not ses_tsv.exists():
            continue
        df_ses = pd.read_csv(ses_tsv, sep='\t')
        if 'acq_time' not in df_ses.columns:
            continue
        df_ses['acq_time'] = pd.to_datetime(df_ses['acq_time'], format='mixed')
        df_ses = df_ses.sort_values('session_id').reset_index(drop=True)
        base_time = df_ses.iloc[0]['acq_time']
        for _, row in df_ses.iterrows():
            ses_label = row['session_id'].replace('ses-', '')
            delta_years = (row['acq_time'] - base_time).days / 365.25
            df.loc[subject_id, f'age_ses{ses_label}'] = df.loc[subject_id, 'age'] + delta_years

    # Head motion, per session as well as pooled
    motion = xcpd_motion_columns(loader)
    if motion.empty:
        logger.warning("No Penn_LEAD motion data found; meanFD columns not added.")
        return df
    return df.join(motion, how='left')


XCPD_MOTION_CACHE = '{dataset}_motion_xcpd.csv'
XCPD_QC_FIELDS = ('mean_fd', 'mean_relative_rms', 'max_relative_rms',
                  'num_censored_volumes', 'num_retained_volumes')
#: Optional keys: ``runs`` keeps only these run labels, ``exclude_acq`` drops these acq labels, so
#: the covariate describes the runs the analysis reads. Penn_LEAD analyses rest run-01 only (522
#: frames, before the n-back); run-02 is 383 frames and acquired after it, and CuBIDS's
#: VARIANTNumVolumes runs are wrong-length runs the loader never picks up.
XCPD_MOTION_SPEC = {
    'AOMIC_PIOP1': dict(rest_task='restingstate', task_tasks=None,           per_session=False, alias_session=None),
    'AOMIC_PIOP2': dict(rest_task='restingstate', task_tasks=None,           per_session=False, alias_session=None),
    'WAND':        dict(rest_task='rest',         task_tasks=None,           per_session=False, alias_session=None),
    'PNC':         dict(rest_task='rest',         task_tasks=('frac2back',), per_session=False, alias_session=None),
    'Penn_LEAD':   dict(rest_task='rest',         task_tasks=('nback',),     per_session=True,  alias_session='1',
                        runs=('01',), exclude_acq=('VARIANTNumVolumes',)),
}


def _qc_entities(filename: str) -> dict:
    """BIDS entities of a QC filename, as strings.

    Read from the NAME, not the file's own columns: pandas turns WAND's ``sub`` into 395 and its
    ``ses`` into 3, and Penn_LEAD's ``run`` into 1, dropping the zero padding the scan registries
    use ('00395', '03', '01'). The filename keeps it.
    """
    return dict(re.findall(r'([a-zA-Z]+)-([a-zA-Z0-9]+)', filename))


def load_xcpd_motion(loader: 'DataLoader', force: bool = False) -> pd.DataFrame:
    """Per-run head motion for every subject of an XCP-D-processed dataset, one row per run.

    Shared by AOMIC-PIOP1/PIOP2, WAND, PNC and Penn_LEAD, whose XCP-D outputs carry the same
    ``*_desc-linc_qc.tsv`` file.

    Returns
    -------
    pd.DataFrame
        Long: one row per run, with ``subject_id``, ``session``, ``task``, ``run``, ``acq`` and
        the :data:`XCPD_QC_FIELDS`. Empty if the derivatives directory is absent.
    """
    cache = (loader.output_dir / 'data'
             / XCPD_MOTION_CACHE.format(dataset=loader.dataset.lower()))
    if cache.exists() and not force:
        return pd.read_csv(cache, dtype={'subject_id': str, 'session': str, 'run': str, 'acq': str})

    xcpd = (loader.config.base_path / loader.config.derivatives_subdir
            / loader.config.modalities['MRI3T'].data_subdir)
    if not xcpd.is_dir():
        logger.warning("No XCP-D directory at %s; skipping %s motion.", xcpd, loader.dataset)
        return pd.DataFrame()

    rows = []
    for pattern in ('sub-*/func/*_desc-linc_qc.tsv', 'sub-*/ses-*/func/*_desc-linc_qc.tsv'):
        for path in sorted(xcpd.glob(pattern)):
            try:
                qc = pd.read_csv(path, sep='\t', dtype=str)
            except Exception as e:      # one unreadable QC file is not a reason to abort
                logger.warning("Could not parse %s: %s", path, e)
                continue
            if qc.empty or 'mean_fd' not in qc.columns:
                continue
            ent = _qc_entities(path.name)
            rec = {'subject_id': ent.get('sub'), 'session': ent.get('ses'),
                   'task': ent.get('task'), 'run': ent.get('run'), 'acq': ent.get('acq')}
            for field in XCPD_QC_FIELDS:
                rec[field] = float(qc.iloc[0][field]) if field in qc.columns else np.nan
            rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['subject_id', 'session', 'task', 'run']).reset_index(drop=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        logger.info("Cached %s motion for %d runs (%d subjects) to %s",
                    loader.dataset, len(df), df['subject_id'].nunique(), cache)
    return df


def xcpd_motion_columns(loader: 'DataLoader', force: bool = False) -> pd.DataFrame:
    """Per-subject motion columns for an XCP-D dataset, ready to join onto its behaviour table.

    Columns
    -------
    ``meanFD_<task>``            that task's FD: the mean over its runs, and over sessions for a
                                 dataset without ``alias_session``. Where ``alias_session`` is set
                                 (Penn_LEAD) this column carries THAT SESSION's value, not the
                                 across-session mean -- pooling would put two timepoints of a
                                 longitudinal sample into one number. The per-session columns below
                                 are where the other sessions live.
    ``meanFD_<task>_ses<N>``     per session, for datasets with ``per_session`` set
    ``meanFD_rest``              the dataset's rest task (from ``alias_session`` when set)
    ``meanFD_task``              mean over its analysed non-rest tasks (likewise)

    ``meanFD_rest`` and ``meanFD_task`` are what :func:`src.gams.build_nuisance` reads by default.
    They are kept separate rather than averaged into one number for the reason given in
    :func:`_add_hcpya_motion`: the INT deflection is a task-minus-rest difference.
    """
    spec = XCPD_MOTION_SPEC.get(loader.dataset)
    if spec is None:
        raise KeyError(f"No XCPD_MOTION_SPEC entry for {loader.dataset!r}; "
                       f"have {sorted(XCPD_MOTION_SPEC)}")
    long = load_xcpd_motion(loader, force=force)
    if long.empty:
        return pd.DataFrame()

    if spec.get('runs') is not None:
        long = long[long['run'].isin(spec['runs'])]
    if spec.get('exclude_acq'):
        long = long[~long['acq'].isin(spec['exclude_acq'])]

    # Runs (and XCP-D's acq variants) collapse first, so a task has one value per session.
    per_ses = (long.groupby(['subject_id', 'session', 'task'], dropna=False)['mean_fd']
               .mean().reset_index())
    out = (per_ses.pivot_table(index='subject_id', columns='task', values='mean_fd')
           .rename(columns=lambda c: f'meanFD_{c}'))

    if spec['per_session']:
        wide_ses = per_ses.pivot_table(index='subject_id', columns=['task', 'session'],
                                       values='mean_fd')
        wide_ses.columns = [f'meanFD_{task}_ses{ses}' for task, ses in wide_ses.columns]
        out = out.join(wide_ses)

    def _column(task):
        """The column the aliases read: this dataset's alias session, or the across-session mean."""
        if spec['per_session'] and spec['alias_session'] is not None:
            named = f'meanFD_{task}_ses{spec["alias_session"]}'
            if named in out.columns:
                return named
        return f'meanFD_{task}'

    rest_col = _column(spec['rest_task'])
    if rest_col in out.columns:
        out['meanFD_rest'] = out[rest_col]
    task_names = (spec['task_tasks'] if spec['task_tasks'] is not None
                  else sorted({t for t in per_ses['task'].unique() if t != spec['rest_task']}))
    task_cols = [c for c in (_column(t) for t in task_names) if c in out.columns]
    if task_cols:
        out['meanFD_task'] = out[task_cols].mean(axis=1)

    out.index.name = 'subject_id'
    return out


def _behavior_aomic(loader: 'DataLoader') -> pd.DataFrame:
    """AOMIC (PIOP1 / PIOP2) demographics, questionnaires and per-run head motion.

    """
    df = pd.read_csv(loader.config.base_path / 'participants.tsv', delimiter='\t')
    df.rename(columns={'participant_id': 'subject_id'}, inplace=True)
    df['subject_id'] = [s.split('-')[1] if str(s).startswith('sub-') else str(s)
                        for s in df['subject_id']]
    # Match the coding every other handler uses (M -> 1). Missing sex stays NaN rather than
    # becoming 0, which an == comparison would silently produce.
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    df.set_index('subject_id', inplace=True)

    motion = xcpd_motion_columns(loader)
    if motion.empty:
        logger.warning("No %s motion data found; meanFD columns not added.", loader.dataset)
        return df
    return df.join(motion, how='left')


def _behavior_wand(loader: 'DataLoader') -> pd.DataFrame:
    """WAND demographics and per-run head motion.

    """
    df = pd.read_csv(loader.config.base_path / 'participants.tsv', delimiter='\t')
    df.rename(columns={'participant_id': 'subject_id'}, inplace=True)
    df['subject_id'] = [s.split('-')[1] if str(s).startswith('sub-') else str(s)
                        for s in df['subject_id']]
    # Match the coding every other handler uses (male -> 1). Missing sex stays NaN rather than
    # becoming 0, which an == comparison would silently produce.
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df.set_index('subject_id', inplace=True)

    motion = xcpd_motion_columns(loader)
    if motion.empty:
        logger.warning("No WAND motion data found; meanFD columns not added.")
        return df
    return df.join(motion, how='left')


def _behavior_hbn_rbc(loader: 'DataLoader') -> pd.DataFrame:
    df = pd.read_csv(
        loader.config.base_path / 'study-HBN_desc-participants.tsv',
        index_col=0, delimiter='\t'
    )
    df['sex'] = (df['sex'] == 'Male').astype(int)
    df.index = df.index.astype(str)
    df.index.name = 'subject_id'
    return df


BEHAVIOR_HANDLERS = {
    'hcp_unrestricted_restricted': _behavior_hcp_unrestricted_restricted,
    'hcp_d': _behavior_hcp_d,
    'pnc_rbc': _behavior_pnc_rbc,
    'penn_lead': _behavior_penn_lead,
    'aomic_piop1': _behavior_aomic,
    'aomic_piop2': _behavior_aomic,
    'wand': _behavior_wand,
    'hbn_rbc': _behavior_hbn_rbc,
}


# ==============================================================================
# Task events handlers
# ==============================================================================

def _rawdata_subject_id(loader: 'DataLoader', subject_id: str) -> str:
    """Apply the rawdata subject ID transform configured for this dataset."""
    if loader.config.subject_id_rawdata_transform == 'split_hcpd':
        return split_hcpd_string(subject_id)
    return subject_id


def _events_hcp_fmri(loader: 'DataLoader', scan: ScanInfo) -> pd.DataFrame:
    subject_id = _rawdata_subject_id(loader, scan.subject_id)
    rawdata_dir = loader.config.base_path / loader.config.rawdata_subdir
    results_dir = rawdata_dir / subject_id / 'MNINonLinear' / 'Results'

    contents = os.listdir(str(results_dir))
    contents_joined = [''.join(c.split('_')) for c in contents]
    try:
        idx = contents_joined.index(scan.task)
    except ValueError:
        raise FileNotFoundError(
            f"Task directory for '{scan.task}' not found in {results_dir}"
        )
    ev_dir = results_dir / contents[idx] / 'EVs'
    return get_task_events_fmri(str(ev_dir))


def _events_hcp_meg(loader: 'DataLoader', scan: ScanInfo) -> pd.DataFrame:
    rawdata_dir = loader.config.base_path / loader.config.rawdata_subdir
    task = scan.task
    subdir_str = f'{task[:1]}-{task[1:]}'
    pattern = str(
        rawdata_dir / scan.subject_id / 'unprocessed' / 'MEG'
        / subdir_str / 'EPRIME'
        / f'{scan.subject_id}_*_{task[1:]}_*.xlsx'
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"MEG task events file not found: {pattern}")
    events = get_task_events_meg(matches[0], verbose=False)
    return events[['onset', 'duration', 'trial_type']]


def _events_hcp_7t_movie(loader: 'DataLoader', scan: ScanInfo) -> pd.DataFrame:
    movie_dir = loader.output_dir / 'data' / 'supporting_data' / 'movie_data'
    timing_map = {
        'MOVIE1': '7T_MOVIE1_CC1_v2_Video_Timing.csv',
        'MOVIE2': '7T_MOVIE2_HO1_v2_Video_Timing.csv',
        'MOVIE3': '7T_MOVIE3_CC2_v2_Video_Timing.csv',
        'MOVIE4': '7T_MOVIE4_HO2_v2_Video_Timing.csv',
    }
    timing_file = next((v for k, v in timing_map.items() if k in scan.task), None)
    if timing_file is None:
        raise ValueError(f"Unknown 7T movie task: '{scan.task}'")

    block_design = pd.read_csv(movie_dir / timing_file)
    block_design = block_design.loc[block_design['movie.block'] != 0.0]
    return pd.DataFrame({
        'onset': block_design['block_start_time'],
        'duration': block_design['block_duration'],
        'trial_type': 'movie',
    }).dropna().sort_values('onset').reset_index(drop=True)


def _events_bids_tsv(loader: 'DataLoader', scan: ScanInfo) -> pd.DataFrame:
    """Generic BIDS events TSV loader driven by the modality config template."""
    mod_config = loader.config.modalities[scan.modality]
    rawdata_dir = loader.config.base_path / loader.config.rawdata_subdir

    func_dir = rawdata_dir / f'sub-{scan.subject_id}'
    if scan.session is not None:
        func_dir = func_dir / f'ses-{scan.session}'
    func_dir = func_dir / 'func'

    filename = mod_config.task_events_file_template.format(**{
        **mod_config.template_fields(scan.task),
        'subject_id': scan.subject_id,
        'session': scan.session,
        'task': scan.task,
    })

    if mod_config.task_events_use_glob:
        matches = glob.glob(str(func_dir / filename))
        if not matches:
            raise FileNotFoundError(
                f"No events file matched: {func_dir / filename}"
            )
        return pd.read_csv(matches[0], sep='\t')
    else:
        return pd.read_csv(func_dir / filename, sep='\t')


def _events_ucla_bids_tsv(loader: 'DataLoader', scan: ScanInfo) -> pd.DataFrame:
    """UCLA events loader with task-specific post-processing.

    Handled tasks: stopsignal, bart, scap, pamenc, pamret. All return the three
    BIDS columns onset/duration/trial_type with trial_type coerced to string so
    nilearn's design matrix yields string-named regressor columns (numeric
    trial_type codes, e.g. scap's load conditions, otherwise become np.int64
    column labels that break downstream contrast handling).
    """
    mod_config = loader.config.modalities[scan.modality]
    rawdata_dir = loader.config.base_path / loader.config.rawdata_subdir
    func_dir = rawdata_dir / f'sub-{scan.subject_id}' / 'func'

    filename = mod_config.task_events_file_template.format(
        subject_id=scan.subject_id,
        task=scan.task,
    )
    matches = glob.glob(str(func_dir / filename))
    if not matches:
        raise FileNotFoundError(f"No events file matched: {func_dir / filename}")

    df = pd.read_csv(matches[0], sep='\t')

    def _clean(events: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing timing/label and stringify trial_type."""
        events = events[['onset', 'duration', 'trial_type']].dropna(
            subset=['onset', 'duration', 'trial_type']
        ).reset_index(drop=True)
        events['trial_type'] = events['trial_type'].astype(str)
        return events

    if scan.task == 'bart':
        # BART events are split across multiple rows per balloon trial; collapse
        # each trial_number into a single event spanning its full duration.
        rows = []
        for _, trial in df[['onset', 'duration', 'trial_type', 'trial_number']].groupby('trial_number'):
            trial = trial.reset_index(drop=True)
            rows.append({
                'onset': trial['onset'].iloc[0],
                'duration': trial['duration'].sum(),
                'trial_type': trial['trial_type'].iloc[0],
            })
        return _clean(pd.DataFrame(rows))
    else:
        # stopsignal, scap, pamenc, pamret: one row per trial already.
        return _clean(df)


TASK_EVENTS_HANDLERS = {
    'hcp_fmri': _events_hcp_fmri,
    'hcp_meg': _events_hcp_meg,
    'hcp_7t_movie': _events_hcp_7t_movie,
    'bids_tsv': _events_bids_tsv,
    'ucla_bids_tsv': _events_ucla_bids_tsv,
}


# ==============================================================================
# DataLoader
# ==============================================================================

class DataLoader:
    """Load and manage preprocessed neural timeseries data across multiple datasets.

    Dataset-specific knowledge (file paths, naming conventions, TRs, behavior and
    event file locations) lives in YAML configs under src/configs/datasets/. This
    class is a generic engine that reads those configs.

    Attributes
    ----------
    dataset : str
        Name of the dataset (e.g. 'HCP-YA').
    config : DatasetConfig
        Parsed YAML configuration for this dataset.
    base_dir : Path
        Root data directory for this dataset.
    rawdata_dir : Path
        Raw BIDS data directory.
    derivdata_dir : Path
        Derivatives directory.
    output_dir : Path
        Directory for writing outputs.
    df_behavior : pd.DataFrame or None
        Subject-level behavioral/demographic data.
    scan_registry : list of ScanInfo
        Registered scans available for loading.
    """

    def __init__(
        self,
        dataset: str,
        output_dir: Optional[str] = None,
        load_behavior: bool = True,
        verbose: bool = True,
    ):
        """Initialise DataLoader for a specific dataset.

        Parameters
        ----------
        dataset : str
            Name of the dataset (must match a YAML file in configs/datasets/).
        output_dir : str or None
            Directory for saving outputs. Defaults to the project root (parent of src/).
        load_behavior : bool
            Whether to load behavioral/demographic data on init.
        verbose : bool
            Print initialisation summary.
        """
        self.dataset = dataset
        self.config = load_dataset_config(dataset)

        self.base_dir = self.config.base_path
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir}")

        self.rawdata_dir = self.base_dir / self.config.rawdata_subdir
        self.derivdata_dir = self.base_dir / self.config.derivatives_subdir

        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser()
        else:
            self.output_dir = _find_project_root()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scan_registry: List[ScanInfo] = []
        self.df_behavior: Optional[pd.DataFrame] = None
        self._data_cache: Dict[str, np.ndarray] = {}
        # When set (via enable_hcpya_3t_resampling), HCP-YA MRI3T timeseries are
        # resampled onto this TR grid at load time. None disables resampling.
        self.resample_to_tr: Optional[float] = None

        if load_behavior:
            self.load_behavioral_data()

        if verbose:
            print(f"Initialized DataLoader for dataset: {dataset}")
            print(f"  Base directory:        {self.base_dir}")
            print(f"  Raw data directory:    {self.rawdata_dir}")
            print(f"  Derivatives directory: {self.derivdata_dir}")
            print(f"  Output directory:      {self.output_dir}")
            if self.df_behavior is not None:
                print(f"  Behavioral data:       {len(self.df_behavior)} subjects")

    # --------------------------------------------------------------------------
    # Config helpers
    # --------------------------------------------------------------------------

    def get_tr(self, modality: str, task: Optional[str] = None) -> float:
        """Return the TR (seconds) for a modality, or for one task within it.

        """
        if modality not in self.config.modalities:
            raise ValueError(
                f"Unknown modality '{modality}' for dataset '{self.dataset}'. "
                f"Available: {list(self.config.modalities)}"
            )
        mod_config = self.config.modalities[modality]
        if task is not None and 'tr' in mod_config.task_params.get(task, {}):
            return float(mod_config.task_params[task]['tr'])
        return mod_config.tr

    def enable_hcpya_3t_resampling(self, target_modality: str = 'MRI7T') -> float:
        """Resample HCP-YA 3T timeseries onto the target modality's TR grid at load time.

        Sets ``self.resample_to_tr`` so that every subsequent load of an HCP-YA
        MRI3T scan (rest and task) is resampled from its native TR
        (``get_tr('MRI3T')``) onto the target modality's TR (default MRI7T = 1.0 s).
        The ``tr`` of every MRI3T scan already in the registry is rewritten to the
        target so that cache keys, the exported scan registry, and all output
        filenames consistently reflect the resampled sampling grid. The native TR
        used to compute the resampling ratio is always read from the config, so it
        remains recoverable after this rewrite.

        Only valid for HCP-YA. Returns the target TR (seconds).
        """
        if self.dataset != 'HCP-YA':
            raise ValueError("3T->7T resampling is only defined for HCP-YA.")
        target_tr = self.get_tr(target_modality)
        self.resample_to_tr = target_tr
        for scan in self.scan_registry:
            if scan.modality == 'MRI3T':
                scan.tr = target_tr
        return target_tr

    # --------------------------------------------------------------------------
    # Registration
    # --------------------------------------------------------------------------

    def register_scans(
        self,
        subject_ids: List[str],
        sessions: List[str],
        tasks: List[str],
        space: str,
        seg: str,
        modality: str,
    ) -> List[ScanInfo]:
        """Register scans for all combinations of subjects × sessions × tasks.

        Parameters
        ----------
        subject_ids : list of str
            Subject IDs (without 'sub-' prefix).
        sessions : list of str
            Session labels; use [None] for sessionless datasets.
        tasks : list of str
            Task names (BIDS task-entity values).
        space : str
            MNI/surface space label.
        seg : str
            Parcellation/segmentation label.
        modality : str
            One of the modalities defined in this dataset's config.

        Returns
        -------
        list of ScanInfo
            The newly registered scans.
        """
        # Per task, not per modality: a dataset may mix acquisitions within one modality.
        tr_of = {task: self.get_tr(modality, task) for task in tasks}
        registered = []
        for subject_id in subject_ids:
            for session in sessions:
                for task in tasks:
                    scan = ScanInfo(
                        subject_id=str(subject_id),
                        session=session,
                        task=task,
                        space=space,
                        seg=seg,
                        modality=modality,
                        tr=tr_of[task],
                        rbc=self.config.rbc,
                    )
                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
        return registered

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def validate_scan_registry(
        self,
        verbose: bool = True,
        remove_incomplete_subjects: bool = False,
        incomplete_unit: str = 'subject',
    ) -> Dict[str, List]:
        """Check which registered scans exist on disk and remove missing ones.

        A scan is considered valid when:
          - Its timeseries file exists, AND
          - If the modality has a task_events_handler: task scans must have
            event files; rest scans must not.
          - If task_events_handler is null: only file existence is checked
            (handles rest-only or events-unavailable datasets cleanly).

        Parameters
        ----------
        verbose : bool
            Print detailed validation report.
        remove_incomplete_subjects : bool
            If True, remove the remaining scans of any unit (see ``incomplete_unit``) that had at
            least one missing scan.
        incomplete_unit : {'subject', 'session'}
            What "incomplete" is judged over. ``'subject'`` (the default) removes every scan of a
            subject with any missing scan, in any session. ``'session'`` removes only the
            (subject, session) pairs with a missing scan and keeps that subject's complete
            sessions -- for a longitudinal dataset (Penn_LEAD), where requiring every session
            would discard every subject not scanned at every timepoint.

        Returns
        -------
        dict
            Keys 'existing', 'missing', 'removed_subjects' (subjects with no scans left),
            'removed_sessions' (``(subject_id, session)`` pairs removed as incomplete; empty under
            ``'subject'``).
        """
        if incomplete_unit not in ('subject', 'session'):
            raise ValueError(f"incomplete_unit must be 'subject' or 'session'; got {incomplete_unit!r}")
        if not self.scan_registry:
            if verbose:
                print("Scan registry is empty.")
            return {'existing': [], 'missing': [], 'removed_subjects': [], 'removed_sessions': []}

        initial_count = len(self.scan_registry)
        if verbose:
            print(f"Validating {initial_count} scans in registry...")

        existing_scans = []
        missing_scans = []

        for scan in self.scan_registry:
            try:
                filepath = self.get_scan_filepath(scan)
                mod_config = self.config.modalities[scan.modality]

                if not filepath.exists():
                    missing_scans.append((scan, filepath))
                    if verbose:
                        print(f"  Missing (file): {scan} -> {filepath}")
                    continue

                # If no event handler configured, file existence is sufficient.
                if mod_config.task_events_handler is None:
                    existing_scans.append(scan)
                    continue

                task_events = self.get_task_events(scan)
                if scan.condition == 'task' and task_events is not None:
                    existing_scans.append(scan)
                elif scan.condition == 'rest' and task_events is None:
                    existing_scans.append(scan)
                else:
                    missing_scans.append((scan, filepath))
                    if verbose:
                        cond = scan.condition
                        has_ev = task_events is not None
                        print(
                            f"  Missing (events): {scan} "
                            f"[condition={cond}, has_events={has_ev}]"
                        )

            except Exception as e:
                missing_scans.append((scan, f"Error: {e}"))
                if verbose:
                    print(f"  Error: {scan} -> {e}")

        # Remove missing scans from registry
        for scan, _ in missing_scans:
            if scan in self.scan_registry:
                self.scan_registry.remove(scan)

        removed_subjects, removed_sessions = [], []
        if remove_incomplete_subjects and missing_scans:
            by_session = incomplete_unit == 'session'
            unit_of = ((lambda s: (s.subject_id, s.session)) if by_session
                       else (lambda s: s.subject_id))
            if verbose:
                print(f"\nRemoving all scans for {'sessions' if by_session else 'subjects'} "
                      "with incomplete data...")
            subjects_before = {s.subject_id for s in self.scan_registry} | {
                s.subject_id for s, _ in missing_scans}
            units_with_missing = {unit_of(scan) for scan, _ in missing_scans}
            scans_to_remove = [s for s in self.scan_registry if unit_of(s) in units_with_missing]
            for scan in scans_to_remove:
                self.scan_registry.remove(scan)
                if verbose:
                    print(f"  Removed: {scan} ({incomplete_unit} {unit_of(scan)} incomplete)")
            subjects_left = {s.subject_id for s in self.scan_registry}
            removed_subjects = sorted(subjects_before - subjects_left)
            if by_session:
                removed_sessions = sorted(units_with_missing, key=lambda u: (u[0], str(u[1])))

        if verbose:
            print(f"\nValidation Summary:")
            print(f"  Initially:             {initial_count} scans")
            print(f"  Existing:              {len(existing_scans)}")
            print(f"  Missing/Removed:       {len(missing_scans)} scans")
            if remove_incomplete_subjects:
                if incomplete_unit == 'session':
                    print(f"  Incomplete sessions:   {len(removed_sessions)}")
                print(f"  Subjects removed:      {len(removed_subjects)}")
            print(f"  Final registry:        {len(self.scan_registry)} scans")

            if missing_scans or existing_scans:
                print("\nMissing data by task:")
                task_missing: Dict[str, set] = defaultdict(set)
                task_existing: Dict[str, set] = defaultdict(set)
                for scan, _ in missing_scans:
                    task_missing[scan.task].add(scan.subject_id)
                for scan in existing_scans:
                    task_existing[scan.task].add(scan.subject_id)
                for task in sorted(set(task_missing) | set(task_existing)):
                    n_miss = len(task_missing[task])
                    n_exist = len(task_existing[task])
                    n_total = n_miss + n_exist
                    if n_total > 0:
                        pct = (n_miss / n_total) * 100
                        print(f"  {task}: {n_miss}/{n_total} missing ({pct:.1f}%)")

                print("\nUniquely missing subjects by task:")
                all_tasks = set(task_missing) | set(task_existing)
                for task in sorted(task_missing):
                    other_tasks = all_tasks - {task}
                    uniquely_missing = {
                        subj for subj in task_missing[task]
                        if all(subj in task_existing[t] for t in other_tasks)
                    }
                    if uniquely_missing:
                        print(f"  {task}: {len(uniquely_missing)} subjects missing only for this task")
                        if len(uniquely_missing) <= 5:
                            print(f"    Subject IDs: {sorted(uniquely_missing)}")

        return {
            'existing': existing_scans,
            'missing': missing_scans,
            'removed_subjects': removed_subjects,
            'removed_sessions': removed_sessions,
        }

    # --------------------------------------------------------------------------
    # File path resolution
    # --------------------------------------------------------------------------

    def get_scan_filepath(self, scan: ScanInfo) -> Path:
        """Construct the expected filepath for a scan's timeseries file.

        Uses the filename_template and dir_template from the dataset YAML config.
        For datasets with use_glob=True (e.g. NKI), the glob is resolved here and
        the first match is returned; FileNotFoundError is raised if no match. For
        datasets with filename_template_fallbacks (e.g. PNC), fallbacks are tried
        in order after the primary template.

        Parameters
        ----------
        scan : ScanInfo
            The scan to locate.

        Returns
        -------
        Path
            Path to the timeseries file (resolved for glob patterns).

        Raises
        ------
        FileNotFoundError
            If use_glob=True and no file matches the pattern.
        ValueError
            If the modality is not recognised for this dataset.
        """
        if scan.modality not in self.config.modalities:
            raise ValueError(
                f"Modality '{scan.modality}' not in config for dataset '{self.dataset}'. "
                f"Available: {list(self.config.modalities)}"
            )
        mod_config = self.config.modalities[scan.modality]

        deriv_root = (
            self.config.base_path
            / self.config.derivatives_subdir
            / mod_config.data_subdir
        )
        template_kwargs = {
            **mod_config.template_fields(scan.task),
            'subject_id': scan.subject_id,
            'session': scan.session,
            'task': scan.task,
            'space': scan.space,
            'seg': scan.seg,
            'ext': mod_config.file_ext,
        }

        dir_path = deriv_root / mod_config.dir_template.format(**template_kwargs)
        filename = mod_config.filename_template.format(**template_kwargs)
        filepath = dir_path / filename

        # Glob-based resolution (e.g. NKI acq label varies)
        if mod_config.use_glob:
            matches = glob.glob(str(filepath))
            if not matches:
                raise FileNotFoundError(
                    f"No file matched glob pattern: {filepath}"
                )
            return Path(matches[0])

        # Fallback templates (e.g. PNC acq-singleband variant)
        if not filepath.exists() and mod_config.filename_template_fallbacks:
            for fallback_template in mod_config.filename_template_fallbacks:
                fallback = dir_path / fallback_template.format(**template_kwargs)
                if fallback.exists():
                    return fallback

        return filepath

    def get_scan_acq(self, scan: ScanInfo) -> Optional[str]:
        """The ``acq-`` label of the file a scan resolves to, or None when it has none.

        ``acq`` is not a ScanInfo entity -- adding one would rename every cached output -- so a
        scan found through a fallback template (PNC's ``acq-singleband``, Penn_LEAD's CuBIDS
        ``acq-VARIANTObliquity``) looks identical to a dominant-group scan everywhere else. This
        reads it back off the resolved filename, which is what a sensitivity analysis that drops
        the variant scans needs.
        """
        match = re.search(r'_acq-([a-zA-Z0-9]+)_', self.get_scan_filepath(scan).name)
        return match.group(1) if match else None

    # --------------------------------------------------------------------------
    # Task events
    # --------------------------------------------------------------------------

    def get_task_events(self, scan: ScanInfo) -> Optional[pd.DataFrame]:
        """
        Return task events for a scan, or None if no handler is configured.

        Dispatches to the handler named in the modality config's
        task_events_handler field. Returns None for rest-only datasets or
        modalities without event files.
        """
        handler_key = self.config.modalities[scan.modality].task_events_handler
        if handler_key is None:
            return None
        handler = TASK_EVENTS_HANDLERS.get(handler_key)
        if handler is None:
            raise ValueError(
                f"Unknown task_events_handler '{handler_key}' for dataset "
                f"'{self.dataset}'. Available: {list(TASK_EVENTS_HANDLERS)}"
            )
        try:
            return handler(self, scan)
        except Exception:
            return None

    # --------------------------------------------------------------------------
    # Behavioral data
    # --------------------------------------------------------------------------

    def load_behavioral_data(self):
        """Load behavioral/demographic data using the configured handler."""
        handler_key = self.config.behavior_handler
        if handler_key is None:
            logger.debug("No behavior handler configured for '%s'.", self.dataset)
            return

        handler = BEHAVIOR_HANDLERS.get(handler_key)
        if handler is None:
            raise ValueError(
                f"Unknown behavior_handler '{handler_key}'. "
                f"Available: {list(BEHAVIOR_HANDLERS)}"
            )
        try:
            self.df_behavior = handler(self)
        except FileNotFoundError as e:
            logger.warning("Could not load behavioral data for '%s': %s", self.dataset, e)

    # --------------------------------------------------------------------------
    # Subject ID helpers
    # --------------------------------------------------------------------------

    def load_subject_ids(self, filename: str = 'subject_ids.txt') -> List[str]:
        """Load subject IDs from a text file in base_dir.

        Automatically strips a 'sub-' prefix if present, and removes underscores
        for HCP-D (subject_id_rawdata_transform = split_hcpd).

        Parameters
        ----------
        filename : str
            Name of the text file relative to base_dir.

        Returns
        -------
        list of str
            Subject ID strings ready for register_scans().
        """
        path = self.base_dir / filename
        subject_ids = np.loadtxt(str(path), dtype=str).tolist()
        # Strip sub- prefix
        subject_ids = [s.split('-')[1] if s.startswith('sub-') else s for s in subject_ids]
        # HCP-D IDs are stored with underscores in the file but used without them
        if self.config.subject_id_rawdata_transform == 'split_hcpd':
            subject_ids = [s.replace('_', '') for s in subject_ids]
        return subject_ids

    # --------------------------------------------------------------------------
    # Timeseries loading
    # --------------------------------------------------------------------------

    def load_scan_timeseries(
        self,
        scan: ScanInfo,
        cache: bool = True,
        detrend: bool = True,
        nuisance_regression: bool = True,
        frequency_filter: bool = True,
        lowpass: Optional[float] = 0.08,
        highpass: Optional[float] = 0.01,
        standardize: bool = False,
        verbose: bool = False,
        dry_load: bool = False,
    ) -> Optional[np.ndarray]:
        """Load and preprocess the timeseries for a single scan.

        Parameters
        ----------
        scan : ScanInfo
            The scan to load.
        cache : bool
            Cache the loaded array (keyed by str(scan)).
        detrend : bool
            Remove linear trend along the time axis.
        nuisance_regression : bool
            Regress out the global signal (mean across parcels at each timepoint).
        frequency_filter : bool
            Apply a frequency filter via apply_frequency_filter. Requires scan.tr
            to be set correctly.
        lowpass : float or None
            Lowpass cutoff in Hz (default 0.08). Ignored if frequency_filter=False.
        highpass : float or None
            Highpass cutoff in Hz (default 0.01). Ignored if frequency_filter=False.
        standardize : bool
            Z-score across timepoints after regression.
        verbose : bool
            Print filepath and shape on load.
        dry_load : bool
            Return None without loading or caching.

        Returns
        -------
        np.ndarray or None
            Array of shape (n_parcels, n_timepoints), or None if dry_load=True.
        """
        if dry_load:
            return None

        cache_key = str(scan)
        if cache and cache_key in self._data_cache:
            if verbose:
                print(f"  Cached: {cache_key}")
            return self._data_cache[cache_key]

        filepath = self.get_scan_filepath(scan)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if verbose:
            print(f"  Loading: {filepath}")

        # Format-specific loading
        if filepath.suffix == '.npy':
            timeseries = np.load(filepath)
        elif '.nii' in filepath.suffixes:
            timeseries = nib.load(str(filepath)).get_fdata()
        elif filepath.suffix == '.tsv':
            timeseries = pd.read_csv(filepath, sep='\t').values
        elif filepath.suffix == '.1D':
            timeseries = pd.read_csv(filepath).values
        else:
            raise ValueError(f"Unrecognised file extension: {filepath}")

        timeseries = timeseries.astype(np.float32)

        # Non-finite parcels -- edge-of-mask parcels the parcellation could not fill -- are
        # carried through untouched rather than repaired or dropped. Both steps below raise on
        # non-finite input (scipy's detrend solves a least-squares fit; sklearn rejects NaN), and
        # load_batch turns any raise into a silently skipped scan, so without this guard a single
        # bad parcel would quietly remove a whole subject from the analysis. Downstream already
        # expects NaN parcels: fit_glm masks them out and compute_acf returns all-NaN.
        # Penn_LEAD carries 3 such parcels out of 400; HCP-YA and HCP-D sampled clean.
        #
        # The all-finite case keeps the original whole-array code verbatim. Operating on a
        # boolean-masked copy instead is algebraically identical but not bitwise identical --
        # float32 reductions depend on memory layout, which moved results by ~1e-7 relative.
        # Data that already worked must come out unchanged, so it takes the untouched path.
        finite_parcels = np.isfinite(timeseries).all(axis=0)

        if finite_parcels.all():
            if detrend:
                timeseries = sp.signal.detrend(timeseries, axis=0, type='linear')

            if nuisance_regression:
                gs = np.mean(timeseries, axis=1).reshape(-1, 1)
                # gs_diff = np.append(0, np.diff(gs, axis=0)).reshape(-1, 1)
                # nuis = np.concatenate([gs, gs_diff, gs ** 2, gs_diff ** 2], axis=1)
                nuis = gs
                regr = LinearRegression()
                regr.fit(nuis, timeseries)
                timeseries = timeseries - regr.predict(nuis)
        else:
            if verbose:
                print(f"    {int((~finite_parcels).sum())}/{finite_parcels.size} non-finite "
                      "parcels left untouched by detrend / nuisance regression")
            if finite_parcels.any():
                if detrend:
                    timeseries[:, finite_parcels] = sp.signal.detrend(
                        timeseries[:, finite_parcels], axis=0, type='linear')

                if nuisance_regression:
                    gs = np.mean(timeseries[:, finite_parcels], axis=1).reshape(-1, 1)
                    nuis = gs
                    regr = LinearRegression()
                    regr.fit(nuis, timeseries[:, finite_parcels])
                    timeseries[:, finite_parcels] = (timeseries[:, finite_parcels]
                                                     - regr.predict(nuis))

        # Optional TR resampling (HCP-YA 3T -> 7T grid only). Applied after nuisance
        # regression and before frequency filtering, so the filter uses the new
        # sampling rate. Only ever touches HCP-YA MRI3T scans; everything else is a
        # no-op. The native TR is read from the config so it survives the registry
        # tr-rewrite done by enable_hcpya_3t_resampling.
        effective_tr = scan.tr
        if (self.resample_to_tr is not None and self.dataset == 'HCP-YA'
                and scan.modality == 'MRI3T'):
            native_tr = self.get_tr(scan.modality, scan.task)
            if not np.isclose(native_tr, self.resample_to_tr):
                ratio = Fraction(native_tr / self.resample_to_tr).limit_denominator(1000)
                up, down = ratio.numerator, ratio.denominator
                n_before = timeseries.shape[0]
                timeseries = sp.signal.resample_poly(timeseries, up, down, axis=0)
                effective_tr = self.resample_to_tr
                if verbose:
                    print(f"    Resampled {native_tr}s -> {self.resample_to_tr}s "
                          f"(up={up}, down={down}): {n_before} -> {timeseries.shape[0]} frames")

        if frequency_filter:
            timeseries = apply_frequency_filter(
                timeseries.T, 1.0 / effective_tr, lowpass=lowpass, highpass=highpass
            ).T

        if standardize:
            timeseries = sp.stats.zscore(timeseries, axis=0)

        # (n_timepoints, n_parcels) -> (n_parcels, n_timepoints)
        timeseries = timeseries.T

        if verbose:
            print(f"    Shape: {timeseries.shape}")

        if cache:
            self._data_cache[cache_key] = timeseries

        return timeseries

    def load_batch(
        self,
        subjects: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        seg: Optional[str] = None,
        cache: bool = False,
        detrend: bool = True,
        nuisance_regression: bool = True,
        frequency_filter: bool = True,
        lowpass: Optional[float] = 0.08,
        highpass: Optional[float] = 0.01,
        standardize: bool = False,
        verbose: bool = True,
        dry_load: bool = False,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Load timeseries for all scans in the registry, with optional filters.

        Parameters
        ----------
        subjects : list of str or None
            Restrict to these subject IDs.
        modalities : list of str or None
            Restrict to these modalities.
        conditions : list of str or None
            Restrict to these conditions ('rest', 'task').
        seg : str or None
            Restrict to this segmentation label.
        cache : bool
            Cache loaded arrays.
        detrend : bool
            Remove linear trend along the time axis.
        nuisance_regression : bool
            Apply global-signal nuisance regression.
        frequency_filter : bool
            Apply frequency filter (lowpass/highpass/bandpass).
        lowpass : float or None
            Lowpass cutoff in Hz (default 0.08). Ignored if frequency_filter=False.
        highpass : float or None
            Highpass cutoff in Hz (default 0.01). Ignored if frequency_filter=False.
        standardize : bool
            Z-score timeseries after regression.
        verbose : bool
            Print progress.
        dry_load : bool
            Return {scan_key: None} without loading.

        Returns
        -------
        dict
            Maps scan string keys to (n_parcels, n_timepoints) arrays, or to None
            if dry_load=True.
        """
        subject_set = {str(x) for x in subjects} if subjects else None
        scans_to_load = [
            s for s in self.scan_registry
            if (subject_set is None or s.subject_id in subject_set)
            and (not modalities or s.modality in modalities)
            and (not conditions or s.condition in conditions)
            and (not seg or s.seg == seg)
        ]

        if not scans_to_load:
            raise ValueError("No scans match the specified filters.")

        if verbose:
            print(f"{'Dry loading' if dry_load else 'Loading'} {len(scans_to_load)} scans...")

        loaded: Dict[str, Optional[np.ndarray]] = {}
        failed = []

        for i, scan in enumerate(scans_to_load):
            try:
                if verbose and i % 10 == 0:
                    print(f"  [{i + 1}/{len(scans_to_load)}] {scan.subject_id} {scan.task}")
                loaded[str(scan)] = self.load_scan_timeseries(
                    scan,
                    cache=cache,
                    verbose=False,
                    nuisance_regression=nuisance_regression,
                    detrend=detrend,
                    frequency_filter=frequency_filter,
                    lowpass=lowpass,
                    highpass=highpass,
                    standardize=standardize,
                    dry_load=dry_load,
                )
            except Exception as e:
                failed.append((scan, str(e)))
                if verbose and len(failed) <= 5:
                    print(f"    Failed: {e}")

        if verbose:
            print(f"\nLoaded: {len(loaded)}/{len(scans_to_load)} scans")
            if failed:
                print(f"Failed: {len(failed)} scans")
                for scan, error in failed[:3]:
                    print(f"  {scan.subject_id}: {error[:100]}")

        return loaded

    # --------------------------------------------------------------------------
    # Registry serialisation
    # --------------------------------------------------------------------------

    def list_registered_scans(self) -> pd.DataFrame:
        """Return a DataFrame of all registered scans."""
        if not self.scan_registry:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.scan_registry])

    def export_scan_registry(self, file_name: str = 'scan_registry.csv'):
        """Write the scan registry to a CSV in output_dir."""
        out_path = self.output_dir / file_name
        self.list_registered_scans().to_csv(out_path)
        logger.info("Scan registry exported to %s", out_path)

    def load_scan_registry(self, file_name: str = 'scan_registry.csv'):
        """Load a previously exported scan registry, reconstructing ScanInfo objects."""
        path = self.output_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Scan registry not found: {path}")
        # dtype=str preserves leading zeros in string fields (e.g. session '01'
        # must not be read back as integer 1). from_dict handles 'nan' strings
        # and bool strings produced by this read mode.
        df = pd.read_csv(path, index_col=0, dtype=str)
        self.scan_registry = [ScanInfo.from_dict(row) for _, row in df.iterrows()]
        print(f"Loaded {len(self.scan_registry)} scans from {file_name}")

    # --------------------------------------------------------------------------
    # Cache management
    # --------------------------------------------------------------------------

    def clear_cache(self):
        """Free all cached timeseries arrays."""
        self._data_cache.clear()
        print("Cleared data cache.")

    def get_cache_info(self) -> Dict:
        """Return cache size and memory usage."""
        return {
            'n_cached': len(self._data_cache),
            'cached_scans': list(self._data_cache.keys()),
            'memory_mb': sum(
                a.nbytes / 1024 / 1024 for a in self._data_cache.values()
            ),
        }

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    def summary(self):
        """Print a summary of the current DataLoader state."""
        print(f"\n{'=' * 50}")
        print(f"Dataset:          {self.dataset}")
        print(f"Base directory:   {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Registered scans: {len(self.scan_registry)}")
        if self.scan_registry:
            df = self.list_registered_scans()
            print("\nScans by modality:")
            print(df['modality'].value_counts().to_string())
            print("\nScans by condition:")
            print(df['condition'].value_counts().to_string())
        if self.df_behavior is not None:
            print(f"\nBehavioral data: {self.df_behavior.shape}")
        cache_info = self.get_cache_info()
        if cache_info['n_cached'] > 0:
            print(f"\nCache: {cache_info['n_cached']} scans, {cache_info['memory_mb']:.1f} MB")
        print(f"{'=' * 50}\n")

    def __repr__(self):
        return (
            f"DataLoader(dataset='{self.dataset}', "
            f"n_registered_scans={len(self.scan_registry)})"
        )


# ==============================================================================
# Standalone utility functions (also used by timescales.py)
# ==============================================================================

def get_longitudinal_array(
    data_dict: Dict[str, Optional[np.ndarray]],
    scan_registry: List['ScanInfo'],
    sessions: List[str],
    task: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Reshape a flat data_dict into a longitudinal array.

    For each subject in the registry, looks up the data for each requested
    session and stacks the result into a single array.  Entries missing from
    data_dict (subject never had that session, or it failed to load) are
    filled with NaN so the subject axis is always fully populated.

    Parameters
    ----------
    data_dict:
        Flat dict mapping scan-key strings to 1-D region arrays (n_regions,).
        Typically ``timescales_processor.timescales_dict``.
    scan_registry:
        List of ScanInfo objects (e.g. ``data_loader.scan_registry``).
    sessions:
        Ordered list of session labels to include (e.g. ['1', '2']).
    task:
        Task name to select (e.g. 'rest' or 'nback').

    Returns
    -------
    array : np.ndarray, shape (n_subjects, n_sessions, n_regions)
        NaN where data is absent for a given subject/session combination.
    subject_ids : list of str
        Subject IDs corresponding to axis 0, in registry order.
    """
    # Unique subjects in registry order
    subject_ids = list(dict.fromkeys(
        s.subject_id for s in scan_registry if s.task == task
    ))

    # Infer n_regions from the first available entry
    n_regions = None
    for v in data_dict.values():
        if v is not None:
            n_regions = v.shape[0]
            break
    if n_regions is None:
        raise ValueError("data_dict contains no non-None entries.")

    array = np.full((len(subject_ids), len(sessions), n_regions), np.nan)

    # Build a lookup: (subject_id, session) -> scan key
    lookup: Dict[Tuple[str, str], str] = {}
    for scan in scan_registry:
        if scan.task == task and scan.session in sessions:
            lookup[(scan.subject_id, scan.session)] = str(scan)

    for i, subject_id in enumerate(subject_ids):
        for j, session in enumerate(sessions):
            key = lookup.get((subject_id, session))
            if key and key in data_dict and data_dict[key] is not None:
                array[i, j, :] = data_dict[key]

    return array, subject_ids


def parse_scan_string(scan_string: str) -> Dict[str, str]:
    """
    Parse a key-value formatted scan string into a dictionary.

    Example:
        "sub-100307_modality-MRI3T_task-rfMRIREST1LR_seg-Schaefer4007"
        -> {'sub': '100307', 'modality': 'MRI3T', 'task': 'rfMRIREST1LR', ...}
    """
    kv_dict = {}
    for part in scan_string.split('_'):
        if '-' in part:
            key, value = part.split('-', 1)
            kv_dict[key] = value
    if 'tr' in kv_dict:
        kv_dict['tr'] = float(kv_dict['tr'])
    return kv_dict


def build_scan_string(**kwargs) -> str:
    """
    Build a scan string from key-value pairs.

    Example:
        build_scan_string(sub='100307', modality='MRI3T', task='rfMRIREST1LR')
        -> "sub-100307_modality-MRI3T_task-rfMRIREST1LR"
    """
    key_order = ['sub', 'ses', 'modality', 'task', 'space', 'seg', 'run', 'stat']
    parts = [
        f"{key}-{kwargs[key]}"
        for key in key_order
        if key in kwargs and kwargs[key] is not None
    ]
    parts += [
        f"{key}-{value}"
        for key, value in kwargs.items()
        if key not in key_order and value is not None
    ]
    return '_'.join(parts)


def get_task_events_fmri(ev_dir: str) -> pd.DataFrame:
    """Load HCP-style EV text files from a directory into a BIDS events DataFrame."""
    task_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
    for condition in os.listdir(ev_dir):
        try:
            ev_data = np.genfromtxt(os.path.join(ev_dir, condition))
            rows = []
            if ev_data.ndim == 1:
                rows = [{'onset': ev_data[0], 'duration': ev_data[1],
                         'trial_type': condition.replace('.txt', '')}]
            elif ev_data.ndim == 2:
                rows = [
                    {'onset': ev_data[i, 0], 'duration': ev_data[i, 1],
                     'trial_type': condition.replace('.txt', '')}
                    for i in range(ev_data.shape[0])
                ]
            if rows:
                task_events = pd.concat(
                    [task_events, pd.DataFrame(rows)], axis=0, ignore_index=True
                )
        except Exception:
            pass
    return task_events.reset_index(drop=True)


def get_task_events_meg(
    task_events_file_path: str,
    onset_column: str = 'Stim.OnsetTime',
    duration_ms: float = 2000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract trial onsets and durations from an HCP MEG working-memory Excel file.

    Parameters
    ----------
    task_events_file_path : str
        Path to the MEG working memory Excel file.
    onset_column : str
        Column to use for trial onset timing (default: 'Stim.OnsetTime').
    duration_ms : float
        Duration of each trial in milliseconds (default: 2000).
    verbose : bool
        Print extraction summary.

    Returns
    -------
    pd.DataFrame
        Columns: onset, duration, trial_type, block_number (times in seconds).
    """
    try:
        data = pd.read_excel(task_events_file_path, sheet_name=0)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}") from e

    if onset_column not in data.columns:
        available = [c for c in data.columns if 'OnsetTime' in c]
        raise ValueError(
            f"Column '{onset_column}' not found. Available timing columns: {available}"
        )

    timing_data_count = data[onset_column].notna().sum()
    if verbose:
        print(f"Timing column '{onset_column}' has {timing_data_count}/{len(data)} non-null values")

    valid = data[onset_column].notna() & data['BlockType'].notna()
    data_clean = data[valid].copy()
    if len(data_clean) == 0:
        raise ValueError(
            f"No valid trials found with both {onset_column} and BlockType data"
        )
    if verbose:
        print(f"Found {len(data_clean)} valid trials "
              f"(removed {len(data) - len(data_clean)} non-task trials)")

    onsets = data_clean[onset_column].values / 1000.0
    duration_sec = duration_ms / 1000.0
    durations = np.full(len(onsets), duration_sec)
    trial_types = determine_trial_types_fixed(data_clean)
    block_numbers = extract_block_numbers(data_clean)

    events_df = pd.DataFrame({
        'onset': onsets,
        'duration': durations,
        'trial_type': trial_types,
        'block_number': block_numbers,
    }).sort_values('onset').reset_index(drop=True)

    if verbose:
        print(f"\nTrial extraction summary:")
        print(f"  Total trials: {len(events_df)}")
        print(f"  Trial duration: {duration_sec:.3f} s")
        print(f"  Onset range: {events_df['onset'].min():.1f} – {events_df['onset'].max():.1f} s")
        print(f"  Trial types: {events_df['trial_type'].value_counts().to_dict()}")
        print(f"  Block numbers: {sorted(events_df['block_number'].unique())}")

    return events_df


def determine_trial_types_fixed(data: pd.DataFrame) -> List[str]:
    """Map BlockType column entries to '0back' / '2back' / 'unknown'."""
    result = []
    for bt in data['BlockType'].values:
        if pd.isna(bt):
            result.append('unknown')
        elif '0-Back' in str(bt) or '0Back' in str(bt):
            result.append('0back')
        elif '2-Back' in str(bt) or '2Back' in str(bt):
            result.append('2back')
        else:
            result.append('unknown')
    return result


def extract_block_numbers(data: pd.DataFrame) -> List[int]:
    """Extract block numbers from BlockNumber column."""
    return [
        int(n) if not pd.isna(n) else -1
        for n in data['BlockNumber'].values
    ]


def split_hcpd_string(s: str) -> str:
    """Insert underscores before 'V' and 'M' in HCP-D subject IDs.

    e.g. 'HCD0001305V1MR' -> 'HCD0001305_V1_MR'
    """
    return re.sub(r'([VM])', r'_\1', s)
