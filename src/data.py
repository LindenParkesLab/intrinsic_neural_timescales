import os
import glob
import re
import logging
import numpy as np
import scipy as sp
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
    return df


def _load_hcpd_beh_file(beh_dir: Path, fname: str, cols: list,
                         rename: dict = None, missing_vals: list = None) -> pd.DataFrame:
    """Load one HCP-D behavioral TSV and return a tidy DataFrame indexed by subject_id."""
    path = beh_dir / fname
    try:
        hdr = pd.read_csv(path, sep='\t', nrows=0)
    except FileNotFoundError:
        logger.warning('HCP-D behavioral file not found: %s', path)
        return pd.DataFrame()
    available = [c for c in cols if c in hdr.columns]
    if not available:
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t', skiprows=[1], usecols=['src_subject_id'] + available)
    df['subject_id'] = df['src_subject_id'].astype(str) + 'V1MR'
    df = (df.drop(columns='src_subject_id')
            .set_index('subject_id')
            .loc[lambda x: ~x.index.duplicated(keep='first')])
    df = df.apply(pd.to_numeric, errors='coerce')
    if missing_vals:
        for mv in missing_vals:
            df = df.replace(mv, np.nan)
    if rename:
        df = df.rename(columns=rename)
    return df


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

    # Pubertal development — coalesce sex-specific scores into one column
    pds = lf('pds01.txt', ['pds_boy_rs', 'pds_girl_rs'], missing_vals=[999])
    if not pds.empty and 'pds_boy_rs' in pds.columns:
        pds['pds_score'] = pds['pds_boy_rs'].combine_first(pds['pds_girl_rs'])
        pds = pds[['pds_score']]
    df = df.join(pds, how='left')

    # Personality / temperament / impulsivity
    for frame in [
        lf('bisbas01.txt', ['bissc_total', 'bas_drive', 'bas_fs', 'bas_rr']),
        lf('upps01.txt',   ['premed_score', 'senseek_score', 'persev_score']),
    ]:
        df = df.join(frame, how='left')

    # Stress / affect / wellbeing
    for frame in [
        lf('pss01.txt',            ['pss_totalscore']),
        lf('tlbx_fearanx01.txt',   ['nih_tlbx_tscore'], rename={'nih_tlbx_tscore': 'fearanx_tscore'}),
        lf('tlbx_wellbeing01.txt', ['nih_tlbx_tscore'], rename={'nih_tlbx_tscore': 'wellbeing_tscore'}),
        lf('tlbx_sensation01.txt', ['nih_tlbx_tscore'], rename={'nih_tlbx_tscore': 'sensation_tscore'}),
    ]:
        df = df.join(frame, how='left')

    # Outlier filter on fluid cognition composite (known extreme outliers in raw data)
    if 'nih_fluidcogcomp_unadjusted' in df.columns:
        col = df['nih_fluidcogcomp_unadjusted']
        df['nih_fluidcogcomp_unadjusted'] = col.where(col.sub(col.mean()).abs() <= 3 * col.std())

    # --- CARIT nogo accuracy (from per-subject EVs) ---
    def _ev_count(path):
        try:
            with open(path) as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0

    carit = {}
    for subj in df.index:
        ev_dir = (proc_dir / split_hcpd_string(subj)
                  / 'MNINonLinear' / 'Results' / 'tfMRI_CARIT_PA' / 'EVs')
        n_fa = (_ev_count(ev_dir / 'nogoFAWin.txt') + _ev_count(ev_dir / 'nogoFALose.txt'))
        n_cr = (_ev_count(ev_dir / 'nogoCRWin.txt') + _ev_count(ev_dir / 'nogoCRLose.txt'))
        total = n_fa + n_cr
        carit[subj] = n_cr / total if total > 0 else np.nan
    df['carit_nogo_acc'] = pd.Series(carit)

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
    rest_fd = df_qc[df_qc['task'] == 'rest'].set_index('participant_id')['meanFD'].rename('meanFD_rest')
    frac2back_fd = df_qc[df_qc['task'] == 'frac2back'].set_index('participant_id')['meanFD'].rename('meanFD_frac2back')
    df_qc_final = pd.concat([rest_fd, frac2back_fd], axis=1)
    df_qc_final.index.name = 'subject_id'

    df = df.loc[df_qc_final.index]
    return pd.merge(df, df_qc_final, left_index=True, right_index=True)


def _behavior_penn_lead(loader: 'DataLoader') -> pd.DataFrame:
    df = pd.read_csv(loader.config.base_path / 'participants.tsv', delimiter='\t')
    df = df[['participant_id', 'age_months', 'sex']]
    df.rename(columns={'participant_id': 'subject_id', 'age_months': 'age'}, inplace=True)
    df['subject_id'] = [s.split('-')[1] for s in df['subject_id']]
    df['age'] = df['age'] / 12
    df['sex'] = (df['sex'] == 'M').astype(int)
    df.set_index('subject_id', inplace=True)

    # Compute session-specific ages from BIDS sessions.tsv acquisition times.
    # The acq_time values are anonymised (anchored to ~year 1800) but the
    # time *differences* between sessions are preserved, so we can derive each
    # subject's age at later sessions as:  age_sesN = age_ses1 + Δt(ses1→sesN).
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

    return df


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

    filename = mod_config.task_events_file_template.format(
        subject_id=scan.subject_id,
        session=scan.session,
        task=scan.task,
    )

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

    def get_tr(self, modality: str) -> float:
        """Return the TR (seconds) for a given modality."""
        if modality not in self.config.modalities:
            raise ValueError(
                f"Unknown modality '{modality}' for dataset '{self.dataset}'. "
                f"Available: {list(self.config.modalities)}"
            )
        return self.config.modalities[modality].tr

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
        tr = self.get_tr(modality)
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
                        tr=tr,
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
            If True, remove all scans for any subject that had at least one
            missing scan.

        Returns
        -------
        dict
            Keys 'existing', 'missing', 'removed_subjects'.
        """
        if not self.scan_registry:
            if verbose:
                print("Scan registry is empty.")
            return {'existing': [], 'missing': [], 'removed_subjects': []}

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

        removed_subjects = []
        if remove_incomplete_subjects and missing_scans:
            if verbose:
                print("\nRemoving all scans for subjects with incomplete data...")
            subjects_with_missing = {scan.subject_id for scan, _ in missing_scans}
            scans_to_remove = [
                s for s in self.scan_registry
                if s.subject_id in subjects_with_missing
            ]
            for scan in scans_to_remove:
                self.scan_registry.remove(scan)
                if verbose:
                    print(f"  Removed: {scan} (subject {scan.subject_id} incomplete)")
            removed_subjects = sorted(subjects_with_missing)

        if verbose:
            print(f"\nValidation Summary:")
            print(f"  Initially:             {initial_count} scans")
            print(f"  Existing:              {len(existing_scans)}")
            print(f"  Missing/Removed:       {len(missing_scans)} scans")
            if remove_incomplete_subjects:
                print(f"  Incomplete subjects:   {len(removed_subjects)}")
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

        # Optional TR resampling (HCP-YA 3T -> 7T grid only). Applied after nuisance
        # regression and before frequency filtering, so the filter uses the new
        # sampling rate. Only ever touches HCP-YA MRI3T scans; everything else is a
        # no-op. The native TR is read from the config so it survives the registry
        # tr-rewrite done by enable_hcpya_3t_resampling.
        effective_tr = scan.tr
        if (self.resample_to_tr is not None and self.dataset == 'HCP-YA'
                and scan.modality == 'MRI3T'):
            native_tr = self.get_tr(scan.modality)
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
