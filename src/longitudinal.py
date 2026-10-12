"""Longitudinal INT deflections: session-level assembly, within-person change, and map-level tests.

Built for Penn LEAD (1–3 sessions per subject, ~2 years apart) and kept out of any one notebook
because the clinical-group analysis (Phase B) reads exactly the same assembly. See the Penn LEAD
section of CLAUDE.md for the data and its caveats.

Two things this module does differently from the cross-sectional pipeline, deliberately:

- **Missing data is dropped per session, not per subject.** ``src.timescales.filter_data_dict``
  keys on ``sub-…`` alone and so removes a subject from every session when any one scan is None.
  For a longitudinal sample that throws away the complete sessions of anyone with one bad scan.
- **The deflection is signed and referenced to the matched rest control** (task residual INT minus
  the ``restctrl`` INT of the same session's rest with the same design projected out), so a change
  in it is a change in the task effect, not in processing. An absolute deflection is not used for
  change: a sign flip would read as a decrease.
- **But in Penn LEAD the deflection is not a usable longitudinal outcome**: its two-session ICC is
  ~0 at every spatial scale (CLAUDE.md, Penn LEAD section), because it is the difference of two
  modestly reliable, correlated measures. The analyses therefore model the task INT with the
  matched-control INT as a per-unit COVARIATE (``unit_covariate`` / ``unit_covariates`` below) --
  the standard replacement for an unreliable difference score, which keeps the deflection's
  meaning ("task INT beyond what the processed rest predicts") without subtracting the noise in.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from scipy import stats

#: Output-type suffixes read per session. The task side is the GLM residual; the controls are keyed
#: by the task scan (see the ``restctrl`` section of CLAUDE.md).
SIGNAL_SUFFIX = {
    'rest':          '',
    'task':          '_resids',
    'control':       '_restctrl',
    'control_drift': '_restctrl-drift',
}

#: Deflections built from those signals, all task minus a rest-side reference. ``deflection`` is
#: the primary; the other two are sensitivity analyses.
DEFLECTIONS = {
    'deflection':       ('task', 'control'),        # matched control: length + projection held equal
    'deflection_drift': ('task', 'control_drift'),  # drift + constant projected only
    'deflection_raw':   ('task', 'rest'),           # as fig1: residual task vs raw rest
}


@dataclass
class SessionData:
    """One row of ``df`` per (subject, session); every array is ``(n_obs, n_regions)`` in ``df`` order."""
    df: pd.DataFrame
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def subset(self, mask) -> 'SessionData':
        mask = np.asarray(mask, bool)
        return SessionData(self.df.loc[mask].reset_index(drop=True),
                           {k: v[mask] for k, v in self.arrays.items()})


def assemble_sessions(outputs, registry, rest_task='rest', task='nback'):
    """Pair each (subject, session)'s rest and task outputs into aligned arrays.

    Parameters
    ----------
    outputs : dict
        ``signal -> {(sub, ses, task) -> (n_regions,) array or None}``, one entry per
        :data:`SIGNAL_SUFFIX` key. ``rest`` is keyed by the rest scan; ``task`` and the controls by
        the task scan.
    registry : iterable of (sub, ses, task)
        The scans that exist.
    rest_task, task : str
        Task names of the two scans a session must have.

    Returns
    -------
    SessionData
        A (subject, session) is kept only if every signal is present for it -- dropped per session,
        never per subject. ``df`` has ``sub`` and ``ses``; ``arrays`` has every signal plus every
        deflection in :data:`DEFLECTIONS`.
    """
    sessions = sorted({(s, ses) for s, ses, _ in registry}, key=lambda x: (x[0], str(x[1])))
    rows, stacks = [], {k: [] for k in outputs}
    for sub, ses in sessions:
        vals = {}
        for signal, table in outputs.items():
            scan = (sub, ses, rest_task if signal == 'rest' else task)
            v = table.get(scan)
            if v is None:
                break
            vals[signal] = np.asarray(v, float)
        else:
            rows.append({'sub': sub, 'ses': ses})
            for k, v in vals.items():
                stacks[k].append(v)
    df = pd.DataFrame(rows, columns=['sub', 'ses'])
    arrays = {k: np.vstack(v) if v else np.empty((0, 0)) for k, v in stacks.items()}
    for name, (a, b) in DEFLECTIONS.items():
        if a in arrays and b in arrays and len(df):
            arrays[name] = arrays[a] - arrays[b]
    return SessionData(df, arrays)


def load_penn_longitudinal(output_type='timescale_auc_zero_crossing', processor=None,
                           registry_file='scan_registry_Penn_LEAD.csv'):
    """Every Penn LEAD session with rest, n-back residual and both matched controls, plus covariates.

    Parameters
    ----------
    output_type : str
        INT estimator, e.g. ``'timescale_auc_zero_crossing'`` (fig1's).
    processor : Timescales or None
        A metadata-only processor; built (with behaviour) when None.
    registry_file : str
        The exported registry the cache was built from.

    Returns
    -------
    SessionData
        ``df`` columns: ``sub``, ``ses``, ``age``, ``sex``, ``study_group``, the ``dx_*`` flags,
        ``fd_rest``, ``fd_task``, ``dprime_0back``, ``dprime_2back``, ``variant`` (True when either
        scan of the session came in as a CuBIDS ``acq-VARIANT*`` file), plus ``age_between`` /
        ``age_within`` from :func:`add_age_components`.
    """
    from src.timescales import Timescales
    if processor is None:
        processor = Timescales(dataset='Penn_LEAD', timeseries_data=None, verbose=False)
    processor.load_scan_registry(registry_file)
    scans = {(s.subject_id, str(s.session), s.task): s for s in processor.scan_registry}

    by_key = {str(s): k for k, s in scans.items()}
    outputs = {}
    for signal, suffix in SIGNAL_SUFFIX.items():
        processor.get_precomputed_outputs(output_type=f'{output_type}{suffix}')
        outputs[signal] = {by_key[key]: v for key, v in processor.data_dict.items() if key in by_key}
    data = assemble_sessions(outputs, scans.keys())

    beh = processor.df_behavior
    df = data.df
    get = lambda col_fmt: [beh.loc[s, col_fmt.format(ses=ses)]
                           if s in beh.index and col_fmt.format(ses=ses) in beh.columns else np.nan
                           for s, ses in zip(df['sub'], df['ses'])]
    df['age'] = np.asarray(get('age_ses{ses}'), float)
    df['fd_rest'] = np.asarray(get('meanFD_rest_ses{ses}'), float)
    df['fd_task'] = np.asarray(get('meanFD_nback_ses{ses}'), float)
    df['dprime_0back'] = np.asarray(get('dprime_0back_ses{ses}'), float)
    df['dprime_2back'] = np.asarray(get('dprime_2back_ses{ses}'), float)
    for col in ['sex', 'study_group'] + [c for c in beh.columns if c.startswith('dx_')]:
        df[col] = beh.reindex(df['sub'])[col].to_numpy()
    df['variant'] = [any(processor.get_scan_acq(scans[(s, ses, t)]) for t in ('rest', 'nback'))
                     for s, ses in zip(df['sub'], df['ses'])]
    data.df = add_age_components(df)
    return data


def add_age_components(df, subject='sub', age='age'):
    """Split age into a between-person and a within-person part.

    ``age_between`` is the subject's mean age over their sessions, centred on the mean of those
    subject means; ``age_within`` is the session's deviation from its own subject's mean (0 for a
    single-session subject). In a mixed model their coefficients are the cross-sectional and the
    longitudinal age effect respectively -- the split is what keeps an accelerated longitudinal
    design from blending the two.
    """
    df = df.copy()
    subj_mean = df.groupby(subject)[age].transform('mean')
    grand = df.groupby(subject)[age].mean().mean()
    df['age_between'] = subj_mean - grand
    df['age_within'] = df[age] - subj_mean
    df['n_sessions'] = df.groupby(subject)[age].transform('size')
    return df


def annualized_change(values, df, subject='sub', age='age', covariates=('fd_rest', 'fd_task'),
                      min_sessions=2):
    """Per-subject rate of change per year: the OLS slope of each column of ``values`` on age.

    Two sessions give the difference over the interval; three give the least-squares slope. The
    same slope is taken for every column of ``covariates``, so a motion covariate enters as its
    own annualised change.

    Returns
    -------
    change : (n_subjects, n_regions) ndarray
    df_sub : DataFrame
        One row per subject: ``sub``, ``n_sessions``, ``age_first``, ``span`` (years between first
        and last session), ``delta_<cov>`` for each covariate, and the first session's value of
        every other column of ``df`` (e.g. sex, group).
    """
    values = np.asarray(values, float)
    rows, slopes = [], []
    for sub, idx in df.groupby(subject, sort=True).indices.items():
        if len(idx) < min_sessions:
            continue
        t = df[age].to_numpy(float)[idx]
        tc = t - t.mean()
        denom = np.sum(tc ** 2)
        if denom <= 0:
            continue
        slopes.append(tc @ (values[idx] - values[idx].mean(axis=0)) / denom)
        first = df.iloc[idx[np.argmin(t)]]
        row = {c: first[c] for c in df.columns}
        row.update({subject: sub, 'n_sessions': len(idx), 'age_first': t.min(),
                    'span': t.max() - t.min()})
        for cov in covariates:
            c = df[cov].to_numpy(float)[idx]
            row[f'delta_{cov}'] = tc @ (c - c.mean()) / denom
        rows.append(row)
    return np.asarray(slopes), pd.DataFrame(rows).reset_index(drop=True)


def regress_maps(Y, X, names=None, unit_covariate=None, unit_covariate_name='unit_covariate'):
    """Independent OLS of every column of ``Y`` on the same design ``X``.

    Rows with a non-finite value in a region are dropped for that region only (Penn LEAD carries a
    few low-coverage parcels).

    Parameters
    ----------
    Y : (n, n_regions) array
    X : (n, k) array
        Include the intercept column yourself.
    names : sequence of str, optional
        Column names of ``X``; default ``x0..x{k-1}``.
    unit_covariate : (n, n_regions) array, optional
        A covariate that differs per region -- region r's model gets column r appended to ``X``
        (e.g. the matched-control INT of that region). Reported under ``unit_covariate_name``.

    Returns
    -------
    dict of DataFrame
        ``'beta'``, ``'se'``, ``'t'``, ``'p'`` -- each ``(n_regions, k)`` with columns ``names`` --
        and ``'n'`` (observations used per region).
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    names = list(names) if names is not None else [f'x{i}' for i in range(X.shape[1])]
    C = None if unit_covariate is None else np.asarray(unit_covariate, float)
    if C is not None:
        names = names + [unit_covariate_name]
    k = len(names)
    out = {key: np.full((Y.shape[1], k), np.nan) for key in ('beta', 'se', 't', 'p')}
    n_used = np.zeros(Y.shape[1], int)
    row_ok = np.all(np.isfinite(X), axis=1)
    for r in range(Y.shape[1]):
        Xfull = X if C is None else np.column_stack([X, C[:, r]])
        ok = row_ok & np.isfinite(Y[:, r]) & np.all(np.isfinite(Xfull), axis=1)
        n = int(ok.sum())
        n_used[r] = n
        if n <= k:
            continue
        Xr, yr = Xfull[ok], Y[ok, r]
        beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        resid = yr - Xr @ beta
        dof = n - k
        sigma2 = resid @ resid / dof
        cov = sigma2 * np.linalg.pinv(Xr.T @ Xr)
        se = np.sqrt(np.diag(cov))
        out['beta'][r], out['se'][r] = beta, se
        with np.errstate(divide='ignore', invalid='ignore'):
            out['t'][r] = beta / se
        out['p'][r] = 2 * stats.t.sf(np.abs(out['t'][r]), dof)
    res = {key: pd.DataFrame(v, columns=names) for key, v in out.items()}
    res['n'] = n_used
    return res


def design_matrix(df, centred=(), raw=(), categorical=None):
    """``[1, centred..., raw...]`` from ``df``. Centred columns are mean-centred, so the intercept is
    the fitted value at their mean and at ``raw == 0`` -- e.g. mean annual change at no change in
    motion. ``categorical`` (a column name) adds treatment-coded dummies, first level as reference.
    """
    cols, names = [np.ones(len(df))], ['intercept']
    for c in centred:
        v = df[c].to_numpy(float)
        cols.append(v - np.nanmean(v)); names.append(c)
    for c in raw:
        cols.append(df[c].to_numpy(float)); names.append(c)
    if categorical is not None:
        levels = sorted(df[categorical].dropna().unique())
        for lev in levels[1:]:
            v = (df[categorical] == lev).astype(float).to_numpy()
            cols.append(v - v.mean()); names.append(f'{categorical}[{lev}]')
    return np.column_stack(cols), names


def map_corr(a, b, method='pearson'):
    """Correlation of two maps over the parcels finite in both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    f = stats.pearsonr if method == 'pearson' else stats.spearmanr
    return float(f(a[ok], b[ok])[0])


def intercept_map(change, X, weights=None, unit_covariate=None):
    """Intercept of ``change[:, r] ~ X`` for every region r, dropping non-finite rows per region.

    Regions are grouped by their pattern of missing subjects so each pattern needs one small solve;
    that is what makes the subject bootstrap affordable when a few hundred parcels each miss a
    different handful of subjects (low-coverage parcels). ``weights`` are per-subject counts, as a
    bootstrap resample produces. With ``unit_covariate`` every region has its own design (its own
    covariate column appended), so regions are solved one at a time.
    """
    change = np.asarray(change, float)
    X = np.asarray(X, float)
    w = np.ones(len(change)) if weights is None else np.asarray(weights, float)
    if unit_covariate is not None:
        C = np.asarray(unit_covariate, float)
        out = np.full(change.shape[1], np.nan)
        for r in range(change.shape[1]):
            Xr = np.column_stack([X, C[:, r]])
            m = np.isfinite(change[:, r]) & np.all(np.isfinite(Xr), axis=1)
            wm = w * m
            if np.count_nonzero(wm) <= Xr.shape[1]:
                continue
            XtW = Xr.T * wm
            try:
                out[r] = np.linalg.solve(XtW @ Xr, XtW @ np.nan_to_num(change[:, r]))[0]
            except np.linalg.LinAlgError:
                pass
        return out
    ok = np.isfinite(change) & np.all(np.isfinite(X), axis=1)[:, None]
    out = np.full(change.shape[1], np.nan)
    patterns = {}
    for r in range(change.shape[1]):
        patterns.setdefault(ok[:, r].tobytes(), []).append(r)
    for key, regions in patterns.items():
        m = np.frombuffer(key, dtype=bool)
        wm = w * m
        if np.count_nonzero(wm) <= X.shape[1]:
            continue
        XtW = X.T * wm
        A = XtW @ X
        try:
            coef = np.linalg.solve(A, XtW @ np.nan_to_num(change[:, regions]))
        except np.linalg.LinAlgError:
            continue
        out[regions] = coef[0]
    return out


def bootstrap_intercept_corr(change, X, target, n_boot=5000, seed=0, method='pearson',
                             unit_covariate=None):
    """Subject bootstrap of r(intercept map, target map).

    Resamples subjects with replacement, refits the intercept of every region (missing values
    dropped per region, as in :func:`intercept_map`), and correlates the refitted map with
    ``target``. Answers "is the map's alignment with the target reliable given which subjects
    happened to be sampled?" -- blind to spatial autocorrelation, which the BrainSMASH null covers.

    Returns
    -------
    (n_boot,) ndarray of r
    """
    rng = np.random.default_rng(seed)
    n = len(change)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        rs[b] = map_corr(intercept_map(change, X, weights=counts, unit_covariate=unit_covariate),
                         target, method)
    return rs


def mixed_model_maps(values, df, fixed=('age_between', 'age_within', 'sex', 'fd_rest', 'fd_task'),
                     subject='sub', unit_covariates=None, progress=False):
    """Per-region random-intercept model: ``value ~ fixed + unit_covariates + (1 | subject)``.

    Random slopes are not fitted: with one to three sessions per subject they are not estimable.
    ``unit_covariates`` maps a name to an ``(n_obs, n_regions)`` array whose column r enters region
    r's model only -- the matched-control INT, which differs by region.

    Returns
    -------
    DataFrame
        One row per region: ``beta_<term>``, ``se_<term>``, ``p_<term>`` for every fixed term, plus
        ``diff_within_between`` / ``p_diff``: the Wald test that the within- and between-person age
        slopes are equal, which is what an accelerated longitudinal design assumes.
    """
    import warnings
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    values = np.asarray(values, float)
    base = sm.add_constant(df[list(fixed)].astype(float)).reset_index(drop=True)
    unit_covariates = {k: np.asarray(v, float) for k, v in (unit_covariates or {}).items()}
    terms = list(fixed) + list(unit_covariates)
    groups = df[subject].to_numpy()
    rows = []
    it = range(values.shape[1])
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc='MixedLM per region')
    for r in it:
        exog = base.copy()
        for name, arr in unit_covariates.items():
            exog[name] = arr[:, r]
        ok = np.isfinite(values[:, r]) & np.all(np.isfinite(exog.to_numpy()), axis=1)
        row = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                warnings.simplefilter('ignore', RuntimeWarning)
                fit = sm.MixedLM(values[ok, r], exog[ok], groups=groups[ok]).fit(reml=True)
            fe = fit.fe_params
            cov = fit.cov_params().loc[fe.index, fe.index]
            for term in terms:
                row[f'beta_{term}'] = fe[term]
                row[f'se_{term}'] = np.sqrt(cov.loc[term, term])
                row[f'p_{term}'] = fit.pvalues[term]
            row['n'] = int(ok.sum())
            if {'age_within', 'age_between'} <= set(fixed):
                d = fe['age_within'] - fe['age_between']
                var = (cov.loc['age_within', 'age_within'] + cov.loc['age_between', 'age_between']
                       - 2 * cov.loc['age_within', 'age_between'])
                row['diff_within_between'] = d
                row['p_diff'] = 2 * stats.norm.sf(abs(d) / np.sqrt(var)) if var > 0 else np.nan
            row['converged'] = bool(fit.converged)
        except Exception:                          # a singular region is reported, not fatal
            row['converged'] = False
        rows.append(row)
    return pd.DataFrame(rows)


def two_session_icc(values, df, subject='sub', session='ses', sessions=('1', '2')):
    """ICC(3,1) (consistency, two-way mixed) per region between two sessions.

    With ~2 years between Penn LEAD sessions this is a LOWER bound on test-retest reliability --
    true developmental change counts as disagreement. Consistency rather than absolute agreement,
    so a uniform shift between sessions (e.g. an age effect shared by everyone) does not count
    against it. A subject enters a region only if both of its values there are finite.

    Returns
    -------
    icc : (n_regions,) ndarray
    n : (n_regions,) ndarray of int
        Subjects contributing to each region.
    """
    values = np.asarray(values, float)
    a, b = sessions
    ia = {s: i for i, s in enumerate(df[subject]) if df[session].iloc[i] == a}
    ib = {s: i for i, s in enumerate(df[subject]) if df[session].iloc[i] == b}
    both = sorted(set(ia) & set(ib))
    A = values[[ia[s] for s in both]]
    B = values[[ib[s] for s in both]]
    icc = np.full(values.shape[1], np.nan)
    n_used = np.zeros(values.shape[1], int)
    for r in range(values.shape[1]):
        ok = np.isfinite(A[:, r]) & np.isfinite(B[:, r])
        n = int(ok.sum())
        n_used[r] = n
        if n < 3:
            continue
        Y = np.column_stack([A[ok, r], B[ok, r]])
        grand = Y.mean()
        ms_rows = 2 * np.sum((Y.mean(1) - grand) ** 2) / (n - 1)
        ss_cols = n * np.sum((Y.mean(0) - grand) ** 2)
        ss_err = np.sum((Y - grand) ** 2) - ms_rows * (n - 1) - ss_cols
        ms_err = ss_err / (n - 1)
        icc[r] = (ms_rows - ms_err) / (ms_rows + ms_err)
    return icc, n_used


def fdr(p):
    """Benjamini-Hochberg q-values, NaN-safe."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def sa_deciles(sa_axis, n_bins=10):
    """Decile (1..n_bins) of every parcel along the S-A axis; 1 = sensorimotor end."""
    ranks = pd.Series(np.asarray(sa_axis, float)).rank(pct=True)
    return np.ceil(ranks * n_bins).astype(int).to_numpy()


def coverage_mask(values, min_fraction=0.8):
    """Parcels finite in at least ``min_fraction`` of rows. Penn LEAD's low-coverage parcels
    (orbitofrontal, temporal pole) are missing in up to two thirds of sessions; a parcel mean over
    whichever third happens to survive is not comparable across people."""
    return np.isfinite(np.asarray(values, float)).mean(axis=0) >= min_fraction


def bin_means(values, bins, keep=None):
    """``(n_obs, n_bins)`` mean of each row over the parcels in each bin (NaN-skipping), using only
    parcels where ``keep`` is True."""
    values = np.asarray(values, float)
    keep = np.ones(values.shape[1], bool) if keep is None else np.asarray(keep, bool)
    labels = sorted(np.unique(bins))
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.column_stack([np.nanmean(values[:, (bins == b) & keep], axis=1)
                                    for b in labels])


def sa_gradient_model(task_bins, control_bins, df, subject='sub', session='ses',
                      fixed=('sex', 'fd_rest', 'fd_task')):
    """One mixed model for how the age effect on the task INT varies along the S-A axis.

    Long format, one row per (session, S-A bin)::

        task ~ control + C(bin) + age_between + age_within
                       + age_between:sa_pos + age_within:sa_pos + fixed
               + (1 | subject) + (1 | subject:session)

    ``sa_pos`` is the bin's position scaled to run -0.5 (sensorimotor) .. +0.5 (association), so an
    interaction coefficient is the difference in the yearly age effect between the two ends of the
    axis. The bins of one session share a session-level variance component, so the ten
    observations a session contributes are not counted as independent.

    Returns
    -------
    fit : statsmodels MixedLMResults
    summary : DataFrame
        ``beta``, ``se``, ``p`` for the four age terms and ``control``.
    """
    import warnings
    import statsmodels.formula.api as smf
    task_bins = np.asarray(task_bins, float)
    control_bins = np.asarray(control_bins, float)
    n_bins = task_bins.shape[1]
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        for j in range(n_bins):
            rows.append({'y': task_bins[i, j], 'control': control_bins[i, j], 'bin': j,
                         'sa_pos': j / (n_bins - 1) - 0.5, 'subject': row[subject],
                         'session': f'{row[subject]}_{row[session]}',
                         'age_between': row['age_between'], 'age_within': row['age_within'],
                         **{c: row[c] for c in fixed}})
    long = pd.DataFrame(rows).dropna()
    formula = ('y ~ control + C(bin) + age_between + age_within + age_between:sa_pos '
               '+ age_within:sa_pos + ' + ' + '.join(fixed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = smf.mixedlm(formula, long, groups=long['subject'],
                          vc_formula={'session': '0 + C(session)'}).fit(reml=True)
    terms = ['age_between', 'age_within', 'age_between:sa_pos', 'age_within:sa_pos', 'control']
    summary = pd.DataFrame({'beta': fit.fe_params[terms], 'se': fit.bse[terms],
                            'p': fit.pvalues[terms]})
    return fit, summary


def load_hcpd_task_control(task='tfMRICARITPA', output_type='timescale_auc_zero_crossing',
                           age_range=None, processor=None, registry_file='scan_registry_HCP-D.csv'):
    """HCP-D's task-residual and matched-control INTs for one task, one row per subject.

    The cross-sectional reference for Penn LEAD: the same estimator, the same signals (task residual
    and the ``restctrl`` of the subject's paired rest run), so the same covariate-adjusted model can
    be fitted. ``age_range`` restricts to the ages Penn LEAD covers.

    Returns
    -------
    SessionData
        ``df``: ``sub``, ``age``, ``sex``, ``fd`` (HCP-D's per-subject mean FD);
        ``arrays``: ``task``, ``control``.
    """
    from src.timescales import Timescales
    if processor is None:
        processor = Timescales(dataset='HCP-D', timeseries_data=None, verbose=False)
    processor.load_scan_registry(registry_file)
    keys = [str(s) for s in processor.scan_registry if s.task == task]
    subs = [s.subject_id for s in processor.scan_registry if s.task == task]
    signals = {}
    for signal in ('task', 'control'):
        processor.get_precomputed_outputs(output_type=f'{output_type}{SIGNAL_SUFFIX[signal]}')
        signals[signal] = [processor.data_dict.get(k) for k in keys]
    beh = processor.df_behavior
    keep = [i for i, s in enumerate(subs)
            if signals['task'][i] is not None and signals['control'][i] is not None and s in beh.index]
    df = pd.DataFrame({'sub': [subs[i] for i in keep]})
    df['age'] = beh.reindex(df['sub'])['age'].to_numpy(float)
    df['sex'] = beh.reindex(df['sub'])['sex'].to_numpy(float)
    df['fd'] = beh.reindex(df['sub'])['meanFD'].to_numpy(float)
    arrays = {sig: np.vstack([np.asarray(signals[sig][i], float) for i in keep])
              for sig in ('task', 'control')}
    data = SessionData(df, arrays)
    if age_range is not None:
        data = data.subset(((df['age'] >= age_range[0]) & (df['age'] <= age_range[1])).to_numpy())
    return data


def fit_age_gam(age, y, age_grid, nuisance=None, nuisance_at=None, n_splines=5, lam='gcv',
                lam_grid=None):
    """Penalised spline of ``y`` on age, with linear nuisance terms held at FIXED values.

    Fits exactly as ``snaplab_tools.gams.fit_gam`` does -- the same terms and the same penalty
    handling, including pygam's shared penalty across the smooth and the linear nuisance terms
    (CLAUDE.md, "Known, accepted"), so Penn LEAD's GAMs behave like fig3's. With ``nuisance_at``
    left at None it reproduces ``fit_gam``'s curve exactly (pinned in tests/test_longitudinal.py).

    The one difference is where the curve is evaluated. ``fit_gam`` holds the nuisance at the
    fitted sample's own means, which moves every group's curve (and every bootstrap resample) to a
    different covariate level; here it is held at ``nuisance_at`` -- pass the full-sample means so
    group curves and resamples all sit at the same control INT, sex mix and motion.

    Parameters
    ----------
    age, y : (n,) arrays
    age_grid : (n_grid,) array
    nuisance : (n, k) array or None
        Linear covariates (e.g. matched-control INT, sex, FD).
    nuisance_at : (k,) array or None
        Values the nuisance is held at on ``age_grid``; defaults to this sample's means.
    n_splines, lam : as fig3's ``gcv`` recipe (5, 'gcv'); a float fixes the penalty.

    Returns
    -------
    dict
        ``pred`` (n_grid,), ``adjusted`` (n,) -- each observation with its nuisance deviation from
        ``nuisance_at`` removed, i.e. on the same scale as ``pred`` -- ``lam`` (penalty used),
        ``edof`` (effective degrees of freedom), ``n``. NaN rows are dropped.
    """
    import pygam
    from snaplab_tools.gams import _DEFAULT_LAM_GRID
    age, y = np.asarray(age, float), np.asarray(y, float)
    N = None if nuisance is None else np.asarray(nuisance, float).reshape(len(age), -1)
    ok = np.isfinite(age) & np.isfinite(y)
    if N is not None:
        ok &= np.all(np.isfinite(N), axis=1)
    X = age[ok, None] if N is None else np.column_stack([age[ok], N[ok]])
    at = None
    if N is not None:
        at = N[ok].mean(axis=0) if nuisance_at is None else np.asarray(nuisance_at, float)
    # Same terms and penalty handling as snaplab_tools.gams.fit_gam (see the docstring).
    terms = pygam.s(0, n_splines=n_splines, lam=0.6 if lam == 'gcv' else float(lam))
    for i in range(1, X.shape[1]):
        terms = terms + pygam.l(i)
    gam = pygam.LinearGAM(terms)
    if isinstance(lam, str):
        gam.gridsearch(X, y[ok], lam=_DEFAULT_LAM_GRID if lam_grid is None else lam_grid,
                       progress=False)
    else:
        gam.fit(X, y[ok])
    grid = np.asarray(age_grid, float)
    X_grid = grid[:, None] if at is None else np.column_stack([grid, np.tile(at, (len(grid), 1))])
    adjusted = np.full(len(age), np.nan)
    if at is None:
        adjusted[ok] = y[ok]
    else:
        X_at = np.column_stack([age[ok], np.tile(at, (ok.sum(), 1))])
        adjusted[ok] = y[ok] - (gam.predict(X) - gam.predict(X_at))
    lam_used = np.ravel(gam.lam[0])[0] if isinstance(gam.lam, list) else float(np.ravel(gam.lam)[0])
    return {'pred': gam.predict(X_grid), 'adjusted': adjusted, 'lam': float(lam_used),
            'edof': float(gam.statistics_['edof']), 'n': int(ok.sum())}


def cluster_bootstrap_age_gam(age, y, subjects, age_grid, nuisance=None, nuisance_at=None,
                              lam=0.6, n_splines=5, n_boot=1000, seed=0):
    """Subject-level bootstrap of :func:`fit_age_gam` curves.

    A subject's sessions are resampled together -- pygam has no random effects and would otherwise
    treat a subject's two or three sessions as independent, giving bands that are too narrow. The
    penalty is frozen at ``lam`` (the full fit's GCV choice), as fig3 does, so the band reflects
    sampling of subjects, not re-selection of smoothness.

    Returns
    -------
    (n_boot, n_grid) ndarray of curves (NaN rows where a resample could not be fitted).
    """
    age, y = np.asarray(age, float), np.asarray(y, float)
    subjects = np.asarray(subjects)
    N = None if nuisance is None else np.asarray(nuisance, float).reshape(len(age), -1)
    rows_of = {s: np.flatnonzero(subjects == s) for s in np.unique(subjects)}
    ids = np.array(list(rows_of))
    rng = np.random.default_rng(seed)
    curves = np.full((n_boot, len(age_grid)), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([rows_of[s] for s in rng.choice(ids, len(ids), replace=True)])
        try:
            curves[b] = fit_age_gam(age[idx], y[idx], age_grid,
                                    None if N is None else N[idx], nuisance_at,
                                    n_splines=n_splines, lam=lam)['pred']
        except Exception:
            pass
    return curves
