"""TCP (Transdiagnostic Connectome Project): subject-level assembly of the INT signals, and the
within-session reliability measures the clinical analyses are gated on.

Kept out of any one notebook because the reliability notebook and the clinical notebook read exactly
the same arrays. See the TCP section of CLAUDE.md for the data and its caveats.

Every run keeps its own array, named ``<signal>_<run>``:

``rest_<rest run>``            raw rest INT (``rest1AP``, ``rest1PA`` pre-task; ``rest2*`` post-task)
``task_<task run>``            task GLM-residual INT (``stroopAP``, ``stroopPA``, ``hammerAP``)
``control_<task run>``         the matched control of that task run (``restctrl``)
``control_drift_<task run>``   its drift-only control (``restctrl-drift``)
``deflection_<task run>``      ``task - control`` -- the primary deflection
``deflection_raw_<task run>``  ``task - rest`` of the rest run its control was cut from

Averages over runs are added as ``<signal>_stroop`` (AP and PA), ``rest_pre`` and ``rest_post``.
The pairing of task and rest runs is read from :func:`src.timescales.pair_rest_runs`, never restated.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

from src.longitudinal import two_session_icc

#: Output-type suffix per signal; the task side and both controls are keyed by the task scan.
SIGNAL_SUFFIX = {
    'rest':          '',
    'task':          '_resids',
    'control':       '_restctrl',
    'control_drift': '_restctrl-drift',
}
REST_PRE = ('rest1AP', 'rest1PA')
REST_POST = ('rest2AP', 'rest2PA')
STROOP = ('stroopAP', 'stroopPA')
TASKS = ('stroopAP', 'stroopPA', 'hammerAP')


#: Broad diagnostic class of the release's free-text ``Primary_Dx``, first matching rule wins (case-
#: insensitive regex on the stripped text). Decided 2026-09-25 as a first pass, to be replaced if finer
#: diagnostic detail is obtained from the TCP team. A primary diagnosis marked past is its own class:
#: such patients are in the Patient group with no current primary disorder.
DX_CLASS_RULES = (
    ('past only',  r'^past\b'),
    ('psychotic',  r'^sz'),                                   # SZ, SZA, 'SZ or SZA', 'SZ (ruleout SZA)'
    ('mood',       r'^(bp|cyclothymia|mdd|dysthymia|depression|pmdd)'),
    ('anxiety',    r'(gad|ptsd|anxiety|panic|phobia|ocd|agoraphobia)'),
    ('other',      r'(adhd|aud|cud|sud|polysub|eating|ed\b)'),
)
#: The classes compared with controls; 'past only' and 'other' are too heterogeneous to test as classes.
DX_CLASSES = ('mood', 'psychotic', 'anxiety')


def dx_class(primary):
    """Broad class of one ``dx_primary`` string (see :data:`DX_CLASS_RULES`); None when missing.

    Raises
    ------
    ValueError
        If a non-missing diagnosis matches no rule -- a new spelling must be classed deliberately.
    """
    import re
    if primary is None or pd.isna(primary) or not str(primary).strip():
        return None
    text = str(primary).strip().lower()
    for label, pattern in DX_CLASS_RULES:
        if re.search(pattern, text):
            return label
    raise ValueError(f'no diagnostic class for {primary!r}; add a rule to DX_CLASS_RULES')


@dataclass
class TCPData:
    """One row of ``df`` per subject; every array is ``(n_subjects, n_regions)`` in ``df`` order."""
    df: pd.DataFrame
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    pairing: Dict[str, str] = field(default_factory=dict)


def load_tcp(output_type='timescale_auc_zero_crossing', processor=None,
             registry_file='scan_registry_TCP.csv'):
    """Every TCP subject with all seven runs, both controls of every task run, and covariates.

    Parameters
    ----------
    output_type : str
        INT estimator, e.g. ``'timescale_auc_zero_crossing'``.
    processor : Timescales or None
        A metadata-only processor; built (with behaviour) when None.
    registry_file : str
        The exported registry the cache was built from.

    Returns
    -------
    TCPData
        ``df`` columns: ``sub``, ``age``, ``sex``, ``site``, ``group``, ``patient``, ``dx_primary``,
        ``dx_nonprimary``, ``active_psychosis``, ``meds_any``, ``psych_meds``, ``fd_rest`` (pre-task
        rest), ``fd_rest_post``, ``fd_task`` (stroop), and ``fd_<run>`` for every run. A subject is
        kept only if every signal of every run is present.
    """
    from src.config import get_dataset_params
    from src.timescales import Timescales, pair_rest_runs
    if processor is None:
        processor = Timescales(dataset='TCP', timeseries_data=None, verbose=False)
    processor.load_scan_registry(registry_file)
    scans = {(s.subject_id, s.task): str(s) for s in processor.scan_registry}

    runs = get_dataset_params('TCP')['tasks']['MRI3T']
    rests = [t for t in runs if 'rest' in t]
    tasks = [t for t in runs if 'rest' not in t]
    pairing = {t: r for t, (r, _) in pair_rest_runs(tasks, rests).items()}

    outputs = {}
    for signal, suffix in SIGNAL_SUFFIX.items():
        processor.get_precomputed_outputs(output_type=f'{output_type}{suffix}')
        outputs[signal] = processor.data_dict

    wanted = [('rest', r) for r in rests] + [(s, t) for t in tasks for s in SIGNAL_SUFFIX if s != 'rest']
    subjects = sorted({s for s, _ in scans})
    rows, stacks = [], {f'{sig}_{run}': [] for sig, run in wanted}
    for sub in subjects:
        vals = {}
        for sig, run in wanted:
            v = outputs[sig].get(scans.get((sub, run)))
            if v is None:
                break
            vals[f'{sig}_{run}'] = np.asarray(v, float)
        else:
            rows.append(sub)
            for k, v in vals.items():
                stacks[k].append(v)
    arrays = {k: np.vstack(v) for k, v in stacks.items()}
    for t in tasks:
        arrays[f'deflection_{t}'] = arrays[f'task_{t}'] - arrays[f'control_{t}']
        arrays[f'deflection_raw_{t}'] = arrays[f'task_{t}'] - arrays[f'rest_{pairing[t]}']
    # Plain means: a run average is NaN at a parcel unless every run has it, so an averaged value
    # is never one run's value standing in for two.
    for sig in ('task', 'control', 'control_drift', 'deflection', 'deflection_raw'):
        arrays[f'{sig}_stroop'] = np.mean([arrays[f'{sig}_{t}'] for t in STROOP], axis=0)
    arrays['rest_pre'] = np.mean([arrays[f'rest_{r}'] for r in REST_PRE], axis=0)
    arrays['rest_post'] = np.mean([arrays[f'rest_{r}'] for r in REST_POST], axis=0)

    beh = processor.df_behavior.reindex(rows)
    df = pd.DataFrame({'sub': rows})
    for col in ('age', 'sex', 'site', 'group', 'patient', 'dx_primary', 'dx_nonprimary',
                'active_psychosis', 'meds_any', 'psych_meds'):
        df[col] = beh[col].to_numpy()
    df['dx_class'] = [('control' if g == 'GenPop' else dx_class(p))
                      for g, p in zip(df['group'], df['dx_primary'])]
    df['fd_rest'] = beh['meanFD_rest'].to_numpy()
    df['fd_rest_post'] = beh['meanFD_rest_post'].to_numpy()
    df['fd_task'] = beh['meanFD_task'].to_numpy()
    for run in runs:
        df[f'fd_{run}'] = beh[f'meanFD_{run}'].to_numpy()
    return TCPData(df, arrays, pairing)


def paired_icc(a, b):
    """ICC(3,1) per column between two aligned ``(n_subjects, n)`` arrays, via
    :func:`src.longitudinal.two_session_icc` -- the function Penn LEAD's reliability used, so the two
    datasets' numbers are the same statistic."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = (a[:, None], b[:, None]) if a.ndim == 1 else (a, b)
    n = a.shape[0]
    df = pd.DataFrame({'sub': list(range(n)) * 2, 'ses': ['1'] * n + ['2'] * n})
    return two_session_icc(np.vstack([a, b]), df)


def spearman_brown(r, k=2):
    """Reliability of the mean of ``k`` parallel measurements, from the single-measurement one."""
    r = np.asarray(r, float)
    return k * r / (1 + (k - 1) * r)


def identification(a, b, keep=None):
    """Fraction of subjects whose map in ``a`` correlates best with their own map in ``b``.

    Each parcel is centred across subjects first: INT maps share a strong group pattern, which
    would otherwise make every subject resemble every other and hide who is who. Returned with
    chance (``1 / n``) and the mean own-minus-other correlation (differential identifiability).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.ones(a.shape[1], bool) if keep is None else np.asarray(keep, bool)
    a, b = a[:, keep], b[:, keep]
    a = a - np.nanmean(a, axis=0)
    b = b - np.nanmean(b, axis=0)
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    za = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    zb = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    r = za @ zb.T / a.shape[1]
    n = r.shape[0]
    own = np.diag(r)
    other = (r.sum(1) - own) / (n - 1)
    return dict(accuracy=float(np.mean(r.argmax(1) == np.arange(n))), chance=1.0 / n,
                own_r=float(np.mean(own)), other_r=float(np.mean(other)),
                differential=float(np.mean(own - other)))
