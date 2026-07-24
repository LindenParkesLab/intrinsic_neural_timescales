"""GAM fitting and visualisation utilities for intrinsic neural timescales."""
from contextlib import nullcontext
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygam
from joblib import Parallel, delayed

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; the bootstrap runs without a progress bar
    tqdm = None

try:
    # Pin BLAS to one thread per worker so N parallel processes don't oversubscribe the
    # cores (each GAM's linear algebra is tiny; process-level parallelism is the win).
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


# =============================================================================
# Data helpers
# =============================================================================

def apply_region_mask(region_mask, *arrays, sa_axis=None):
    """Exclude flagged regions from all data arrays and optionally sa_axis.

    Parameters
    ----------
    region_mask : (n_regions,) bool ndarray or None
        True marks a region to exclude. If None, inputs are returned unchanged.
    *arrays : (n_subjects, n_regions) ndarrays
        Any number of data arrays to mask column-wise.
    sa_axis : (n_regions,) ndarray or None
        Optional SA-axis vector to mask alongside the arrays.

    Returns
    -------
    tuple
        Masked arrays in input order, followed by sa_axis (masked if provided,
        else None).
    """
    if region_mask is None:
        return (*arrays, sa_axis)
    valid  = ~np.asarray(region_mask, dtype=bool)
    masked = tuple(arr[:, valid] for arr in arrays)
    sa_out = sa_axis[valid] if sa_axis is not None else None
    print(f"region_mask: excluding {region_mask.sum()} regions, {valid.sum()} remaining.")
    return (*masked, sa_out)


# =============================================================================
# GAM fitting
# =============================================================================

# Grid searched by fit_gam when lam='gcv' (log-spaced smoothing penalties).
_DEFAULT_LAM_GRID = np.logspace(-3, 3, 25)
# _DEFAULT_LAM_GRID = np.logspace(-5, 1, 25)

def fit_gam(age, y, age_grid, n_splines=4, lam=0.1, show_ci=False, nuisance=None,
            lam_grid=_DEFAULT_LAM_GRID):
    """Fit a GAM to (age, y) and evaluate on age_grid.

    NaN values in y are excluded from fitting (per-subject missingness). A region
    with fewer than n_splines valid subjects cannot be fit and yields NaN curves
    and gam=None.

    Parameters
    ----------
    age : (n_subjects,) ndarray
        Subject ages.
    y : (n_subjects,) ndarray
        Outcome for a single region/series.
    age_grid : (n_grid,) ndarray
        Evaluation points.
    n_splines : int
        Spline basis dimension for the age smooth.
    lam : float or 'gcv'
        Smoothing penalty for the age smooth. A float fixes it (original behaviour).
        'gcv' selects it by grid search over lam_grid (generalized cross-validation);
        the chosen value is readable from the returned gam.lam.
    show_ci : bool
        Whether to compute 95% prediction intervals.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates entered as linear terms l() in the GAM. When predicting on
        age_grid, nuisance values are fixed at their within-sample means so the
        curve reflects the age smooth only.
    lam_grid : array-like
        Candidate penalties searched when lam='gcv'; ignored otherwise. Defaults to
        _DEFAULT_LAM_GRID.

    Returns
    -------
    pred : (n_grid,) ndarray
        Predicted values.
    ci_lo : (n_grid,) ndarray or None
        Lower 95% CI, or None when show_ci is False.
    ci_hi : (n_grid,) ndarray or None
        Upper 95% CI, or None when show_ci is False.
    deriv : (n_grid,) ndarray
        Backward-difference derivative (units / year); NaN at index 0.
    gam : pygam.LinearGAM or None
        Fitted model, or None if the region had too few valid subjects.
    """
    mask = ~np.isnan(y)
    Y    = y[mask]

    # Some parcellations (e.g. Schaefer-1000) contain regions with no usable
    # BOLD signal, so a region column can be entirely (or nearly) NaN. pygam
    # raises on an empty target, so skip these and return NaN curves; downstream
    # consumers (compute_deriv_sa_corr, compute_partial_r2_age) already tolerate
    # NaN regions.
    if mask.sum() < n_splines:
        nan_curve = np.full(len(age_grid), np.nan)
        return nan_curve, None, None, nan_curve.copy(), None

    # 'gcv' selects lam by grid search below; build the terms with a numeric placeholder.
    _is_gcv   = isinstance(lam, str) and lam == 'gcv'
    if isinstance(lam, str) and not _is_gcv:
        raise ValueError(f"lam must be a float or 'gcv'; got {lam!r}")
    _term_lam = 0.6 if _is_gcv else lam

    if nuisance is not None:
        nuisance        = np.asarray(nuisance)
        nuisance_masked = nuisance[mask]
        X_fit           = np.column_stack([age[mask], nuisance_masked])
        terms           = pygam.s(0, n_splines=n_splines, lam=_term_lam)
        for i in range(1, nuisance.shape[1] + 1):
            terms = terms + pygam.l(i)
        nuisance_means = nuisance_masked.mean(axis=0)
        X_pred = np.column_stack([
            age_grid,
            np.tile(nuisance_means, (len(age_grid), 1)),
        ])
    else:
        X_fit  = age[mask].reshape(-1, 1)
        terms  = pygam.s(0, n_splines=n_splines, lam=_term_lam)
        X_pred = age_grid.reshape(-1, 1)

    gam = pygam.LinearGAM(terms)
    if _is_gcv:
        gam.gridsearch(X_fit, Y, lam=lam_grid, progress=False)
    else:
        gam.fit(X_fit, Y)
    pred  = gam.predict(X_pred)
    dt    = np.diff(age_grid)
    deriv = np.concatenate([[np.nan], np.diff(pred) / dt])

    ci_lo, ci_hi = None, None
    if show_ci:
        ci    = gam.confidence_intervals(X_pred, width=0.95)
        ci_lo = ci[:, 0]
        ci_hi = ci[:, 1]

    return pred, ci_lo, ci_hi, deriv, gam


def fit_series(age, arr, age_grid,
               n_splines=4, lam=0.1, demean=True, show_ci=False, nuisance=None,
               lam_grid=_DEFAULT_LAM_GRID):
    """Fit GAMs for every column (series) in arr.

    Parameters
    ----------
    age : (n_subjects,) ndarray
        Subject ages.
    arr : (n_subjects, n_series) ndarray
        One column per series to fit.
    age_grid : (n_grid,) ndarray
        Evaluation points.
    n_splines, lam, show_ci, lam_grid : see fit_gam. lam may be 'gcv' to select the
        smoothing penalty per series by grid search over lam_grid.
    demean : bool
        If True, each series is mean-centred (ignoring NaNs) before fitting.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates passed through to fit_gam.

    Returns
    -------
    dict
        Keys 'pred' (n_series, n_grid), 'deriv' (n_series, n_grid),
        'ci_lo'/'ci_hi' (n_series, n_grid) or None, 'gams' (list of n_series
        fitted pygam.LinearGAM objects, None where a series had too few valid
        subjects), and 'lams' (n_series) -- the age-smooth penalty actually used
        for each series (the grid-searched value when lam='gcv'), NaN where unfit.
        Freeze these to reuse the same smoothing in the bootstrap.
    """
    n_ser  = arr.shape[1]
    preds  = np.full((n_ser, len(age_grid)), np.nan)
    derivs = np.full((n_ser, len(age_grid)), np.nan)
    ci_los = np.full((n_ser, len(age_grid)), np.nan) if show_ci else None
    ci_his = np.full((n_ser, len(age_grid)), np.nan) if show_ci else None
    lams   = np.full(n_ser, np.nan)
    gams   = []

    for i in range(n_ser):
        y = arr[:, i].copy()
        if demean:
            y = y - np.nanmean(y)
        pred, ci_lo, ci_hi, deriv, gam = fit_gam(
            age, y, age_grid, n_splines=n_splines, lam=lam,
            show_ci=show_ci, nuisance=nuisance, lam_grid=lam_grid)
        preds[i]  = pred
        derivs[i] = deriv
        if show_ci:
            ci_los[i] = ci_lo
            ci_his[i] = ci_hi
        if gam is not None:
            lams[i] = float(np.ravel(gam.lam)[0])   # age-smooth penalty actually used
        gams.append(gam)

    return {
        'pred':  preds,
        'deriv': derivs,
        'ci_lo': ci_los,
        'ci_hi': ci_his,
        'lams':  lams,
        'gams':  gams,
    }


def fit_all_panels(age, rest, task, deflection, age_grid,
                   n_splines=4, lam=0.1, demean=True, show_ci=False, nuisance=None,
                   lam_grid=_DEFAULT_LAM_GRID):
    """Fit GAMs for rest, task, and deflection across all series.

    Parameters
    ----------
    age : (n_subjects,) ndarray
        Subject ages.
    rest, task, deflection : (n_subjects, n_series) ndarray
        One condition each; every column is fit independently.
    age_grid : (n_grid,) ndarray
        Evaluation points.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates passed through to fit_gam.
    lam_grid : array-like
        Candidate penalties searched when lam='gcv' (passed to fit_gam).

    Returns
    -------
    dict
        Keys 'rest', 'task', 'deflection' (each a fit_series result dict) plus
        'age_grid' (the evaluation grid).
    """
    kw      = dict(n_splines=n_splines, lam=lam, demean=demean,
                   show_ci=show_ci, nuisance=nuisance, lam_grid=lam_grid)
    results = {
        name: fit_series(age, arr, age_grid, **kw)
        for name, arr in [('rest', rest), ('task', task), ('deflection', deflection)]
    }
    results['age_grid'] = age_grid
    return results


# =============================================================================
# Partial R²
# =============================================================================

def compute_partial_r2_age(condition_data, age, gams, nuisance=None, demean=True):
    """Compute partial R² of the age smooth for each region/series.

    Uses the already-fitted full GAMs from fit_all_panels and compares them
    against a reduced model that contains only the nuisance terms (or the
    intercept when nuisance is None):

        partial_R2_age = (R2_full - R2_reduced) / (1 - R2_reduced)

    Regions with near-zero variance or too few valid subjects (gam is None) are
    skipped and returned as NaN.

    Parameters
    ----------
    condition_data : (n_subjects, n_regions) ndarray
        Outcome values per region.
    age : (n_subjects,) ndarray
        Subject ages.
    gams : list of pygam.LinearGAM or None
        Fitted full models from fit_all_panels, one per region.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates; must match those used during fitting.
    demean : bool
        Must match the demean flag used during fitting.

    Returns
    -------
    partial_r2 : (n_regions,) ndarray
        Partial R² of the age smooth; NaN for skipped regions.
    """
    n_subs, n_ser = condition_data.shape
    partial_r2    = np.full(n_ser, np.nan)

    for i in range(n_ser):
        y = condition_data[:, i].copy()
        if demean:
            y = y - np.nanmean(y)
        mask  = ~np.isnan(y)
        Y     = y[mask]
        age_m = age[mask]

        SS_tot = np.sum((Y - Y.mean()) ** 2)
        if SS_tot < 1e-12:
            continue

        gam_full = gams[i]
        if gam_full is None:      # too few valid subjects to fit (see fit_gam)
            continue
        if nuisance is not None:
            nuis_m     = nuisance[mask]
            X_full     = np.column_stack([age_m, nuis_m])
            y_hat_full = gam_full.predict(X_full)

            # reduced model: nuisance only (linear terms, no age smooth)
            nuis_terms = pygam.l(0)
            for j in range(1, nuis_m.shape[1]):
                nuis_terms = nuis_terms + pygam.l(j)
            gam_red   = pygam.LinearGAM(nuis_terms).fit(nuis_m, Y)
            y_hat_red = gam_red.predict(nuis_m)
        else:
            y_hat_full = gam_full.predict(age_m.reshape(-1, 1))
            y_hat_red  = np.full(len(Y), Y.mean())

        SS_res_full = np.sum((Y - y_hat_full) ** 2)
        SS_res_red  = np.sum((Y - y_hat_red)  ** 2)

        R2_full = 1 - SS_res_full / SS_tot
        R2_red  = 1 - SS_res_red  / SS_tot
        denom   = 1 - R2_red
        if denom > 1e-10:
            partial_r2[i] = (R2_full - R2_red) / denom

    return partial_r2


# =============================================================================
# Plotting helpers
# =============================================================================

def get_series_colors(mode, n_series, cmap, sa_axis=None):
    """Return a list of RGBA colors, one per series.

    Parameters
    ----------
    mode : {'decile', 'region', 'wholebrain'}
        'decile' samples the colormap at symmetric half-steps around the white
        midpoint; 'region' maps continuously from sa_axis rank; 'wholebrain'
        returns an empty list (it has its own plot).
    n_series : int
        Number of series (colors) to produce.
    cmap : matplotlib colormap
        Colormap to sample.
    sa_axis : (n_regions,) ndarray or None
        SA rank values; required when mode='region'.

    Returns
    -------
    list
        RGBA tuples, one per series (empty when mode='wholebrain').
    """
    if mode == 'wholebrain':
        return []
    if mode == 'decile':
        half_step = 0.5 / n_series
        positions = np.linspace(half_step, 1 - half_step, n_series)
    elif mode == 'region':
        if sa_axis is None:
            raise ValueError("sa_axis must be provided when mode='region'.")
        sa        = np.asarray(sa_axis, dtype=float)
        positions = (sa - sa.min()) / (sa.max() - sa.min())
    else:
        raise ValueError(f"mode must be 'decile', 'region', or 'wholebrain', got '{mode}'.")
    return [cmap(p) for p in positions]


def get_draw_order(mode, n_series, sa_axis=None):
    """Return index order so high-SA series renders on top.

    Parameters
    ----------
    mode : {'decile', 'region', 'wholebrain'}
        'decile' and 'wholebrain' use ascending index order; 'region' orders by
        ascending SA rank (low-rank drawn first, high-rank on top).
    n_series : int
        Number of series.
    sa_axis : (n_regions,) ndarray or None
        SA values; required (used for argsort) when mode='region'.

    Returns
    -------
    ndarray
        Index order for drawing the series.
    """
    if mode in ('decile', 'wholebrain'):
        return np.arange(n_series)
    return np.argsort(sa_axis)


def add_colorbar(fig, ax, cmap, mode, n_series, fontsize=7):
    """Add a horizontal SA colorbar under ax.

    Parameters
    ----------
    fig : matplotlib Figure
    ax : matplotlib Axes
        Axes to attach the colorbar to.
    cmap : matplotlib colormap
    mode : {'decile', 'region'}
        Sets the label and ticks (integer deciles vs SA-rank range).
    n_series : int
        Number of series, used for the color normalization range.
    fontsize : int
        Label and tick font size.
    """
    norm = mcolors.Normalize(vmin=1, vmax=n_series)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2,
                      fraction=0.05, aspect=20)
    if mode == 'decile':
        cb.set_label('SA decile', fontsize=fontsize)
        cb.set_ticks(list(range(1, n_series + 1)))
    else:
        cb.set_label('SA rank (region)', fontsize=fontsize)
        cb.set_ticks(np.linspace(1, n_series, 5, dtype=int).tolist())
    cb.ax.tick_params(labelsize=fontsize)


def style_ax(ax, xlabel, ylabel, title, demean, fontsize=8):
    """Apply shared axis styling (labels, title, ticks, spines).

    Parameters
    ----------
    ax : matplotlib Axes
    xlabel, ylabel, title : str or None
        Axis labels and title.
    demean : bool
        If True, draw a dashed zero reference line.
    fontsize : int
        Base font size for labels, title, and ticks.
    """
    if demean:
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=1)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize, pad=0.25)
    ax.spines[['top', 'right']].set_visible(False)


def draw_gam_figure(results, series_colors, draw_order, key_cfg, data_key,
                    age_grid, mode, n_series, cmap, demean, show_ci):
    """Draw a GAM figure with 1–3 panels (rest / task / deflection).

    Parameters
    ----------
    results : dict
        Output of fit_all_panels.
    series_colors : list
        RGBA colors, one per series.
    draw_order : ndarray
        Index order controlling which series render on top.
    key_cfg : list of tuple
        (condition_key, title, ylabel) per panel; its length sets the number of
        panels (1, 2, or 3).
    data_key : {'pred', 'deriv'}
        Which curve to draw.
    age_grid : (n_grid,) ndarray
        Evaluation points (x-axis).
    mode : {'decile', 'region'}
        Colorbar style.
    n_series : int
        Number of series.
    cmap : matplotlib colormap
    demean : bool
        If True, draw a dashed zero reference line.
    show_ci : bool
        If True and data_key='pred', shade the CI band.

    Returns
    -------
    fig : matplotlib Figure
    """
    n_panels = len(key_cfg)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 2.5), sharey=False, dpi=300)
    if n_panels == 1:
        axes = [axes]
    lw    = 0.4 if n_series > 20 else 1.8
    alpha = 0.7 if n_series > 20 else 1.0

    for ax, (key, title, ylabel) in zip(axes, key_cfg):
        for i in draw_order:
            ax.plot(age_grid, results[key][data_key][i],
                    color=series_colors[i], lw=lw, alpha=alpha)
            if show_ci and results[key]['ci_lo'] is not None and data_key == 'pred':
                ax.fill_between(age_grid,
                                results[key]['ci_lo'][i],
                                results[key]['ci_hi'][i],
                                color=series_colors[i], alpha=0.10)
        ylab = ylabel if data_key == 'pred' else f'd/dt {ylabel}'
        style_ax(ax, 'Age (years)', ylab, None, demean, fontsize=8)
        add_colorbar(fig, ax, cmap, mode, n_series, fontsize=8)

    fig.tight_layout()
    return fig


def plot_wholebrain_gam(results, panels_cfg, data_key, age, demean, show_ci=False,
                        condition_data=None, scatter_color='#3B3B3B', scatter_alpha=0.6,
                        scatter_size=10, smooth_color='#2c7bb6', fontsize=9,
                        grid=True, grid_alpha=0.3):
    """3-panel figure for wholebrain (mean across regions) GAM results.

    When data_key='pred', subject-level scatter is overlaid on the smooth if
    condition_data is provided. When data_key='deriv', only the derivative curve
    is plotted (no scatter).

    Parameters
    ----------
    results : dict
        Output of fit_all_panels with n_series=1.
    panels_cfg : list of tuple
        (condition_key, title, ylabel) per panel.
    data_key : {'pred', 'deriv'}
        Which curve to draw.
    age : (n_subjects,) ndarray
        Subject ages (scatter x-axis).
    demean : bool
        If True, scatter values are demeaned to match the smooth.
    show_ci : bool
        If True and data_key='pred', shade the CI band.
    condition_data : dict or None
        {key: (n_subjects, 1)} of raw (un-demeaned) data; required for scatter
        when data_key='pred'.
    scatter_color : str
        Colour for scatter points.
    scatter_alpha : float
        Scatter opacity.
    scatter_size : float
        Scatter marker size.
    smooth_color : str
        Colour for the GAM smooth and CI band.
    fontsize : int
        Base font size for labels and ticks.
    grid : bool
        Whether to show the background grid.
    grid_alpha : float
        Grid transparency.

    Returns
    -------
    fig : matplotlib Figure
    """
    age_grid  = results['age_grid']
    fig, axes = plt.subplots(1, len(panels_cfg), figsize=(4 * len(panels_cfg), 4))
    if len(panels_cfg) == 1:
        axes = [axes]

    for ax, (key, title, ylabel) in zip(axes, panels_cfg):
        curve = results[key][data_key][0]   # (n_grid,) — single series

        # subject scatter (pred only)
        if data_key == 'pred' and condition_data is not None:
            y_raw = condition_data[key][:, 0].copy()
            if demean:
                y_raw -= np.nanmean(y_raw)
            ax.scatter(age, y_raw, s=scatter_size, color=scatter_color,
                       alpha=scatter_alpha, edgecolors='white', linewidths=0.5,
                       zorder=2)

        # GAM smooth
        ax.plot(age_grid, curve, color=smooth_color, lw=2.5, alpha=0.8, zorder=3)

        # CI band
        if show_ci and results[key]['ci_lo'] is not None and data_key == 'pred':
            ax.fill_between(age_grid,
                            results[key]['ci_lo'][0],
                            results[key]['ci_hi'][0],
                            color=smooth_color, alpha=0.15, zorder=1)

        # zero reference line (when demeaned)
        if demean:
            ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4, zorder=1)

        # grid
        if grid:
            ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

        # labels
        ylab = ylabel if data_key == 'pred' else f'd/dt {ylabel}'
        ttl  = title if data_key == 'pred' else f'{title}\n(temporal derivative)'
        ax.set_xlabel('Age (years)', fontsize=fontsize)
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_title(ttl, fontsize=fontsize, fontweight='bold', pad=10)

        # spine styling — matches style_correlation_axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # tick styling
        ax.tick_params(axis='both', which='major', labelsize=fontsize,
                       length=6, width=1.2, colors='black')

    fig.tight_layout()
    return fig


def plot_deriv_sa_correlation(results, sa_axis, method='spearman', deflections_only=True):
    """Correlate GAM temporal derivatives with the SA axis at each age point.

    For each age point in age_grid, correlates the per-region derivative with
    sa_axis and plots that correlation as a function of age, one line per
    condition. Only meaningful in region mode.

    Parameters
    ----------
    results : dict
        Output of fit_all_panels.
    sa_axis : (n_regions,) ndarray
        SA rank values in parcellation order.
    method : {'spearman', 'pearson'}
        Correlation type.
    deflections_only : bool
        If True, plot only the deflection condition; otherwise rest/task/deflection.

    Returns
    -------
    fig : matplotlib Figure
    """
    age_grid   = results['age_grid']
    if deflections_only:
        conditions = ['deflection',]
        labels     = ['Deflection INT',]
    else:
        conditions = ['rest',     'task',     'deflection']
        labels     = ['Rest INT', 'Task INT', 'Deflection INT']
    colors     = ['#4C72B0',  '#DD8452',  '#55A868']
    corr_label = 'Spearman ρ' if method == 'spearman' else 'Pearson r'

    fig, ax = plt.subplots(figsize=(2, 1.75), dpi=300)
    for cond, label, color in zip(conditions, labels, colors):
        r_vals = compute_deriv_sa_corr(results[cond]['deriv'], sa_axis, method=method)
        ax.plot(age_grid, r_vals, color=color, lw=1.8, label=label)

    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel('Age (years)', fontsize=8, labelpad=1)
    ax.set_ylabel(f'{corr_label}\n(derivative × SA axis)', fontsize=8, labelpad=1)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8, pad=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


# =============================================================================
# Data-driven developmental stage boundary (multivariate change point)
# =============================================================================

def build_stage_signal(pred, age_grid, signal):
    """Derive the signal a stage boundary is detected on from the fitted GAM level.

    Parameters
    ----------
    pred : (n_regions, n_grid) ndarray
        Fitted GAM level on age_grid.
    age_grid : (n_grid,) ndarray
    signal : {'pred', 'deriv1', 'deriv2'}
        'pred'   -> the fitted level itself,
        'deriv1' -> first derivative  (rate of change),
        'deriv2' -> second derivative (acceleration).
        Derivatives are central differences w.r.t. age (np.gradient), so there is
        no leading NaN and deriv2 is defined across the whole grid.

    Returns
    -------
    (n_regions, n_grid) ndarray
    """
    pred = np.asarray(pred, float)
    if signal == 'pred':
        return pred
    # edge_order=2 uses second-order one-sided differences at the grid ends; without it a
    # double np.gradient compounds the boundary error and deriv2 hooks sharply at the extremes.
    d1 = np.gradient(pred, age_grid, axis=1, edge_order=2)
    if signal == 'deriv1':
        return d1
    if signal == 'deriv2':
        return np.gradient(d1, age_grid, axis=1, edge_order=2)
    raise ValueError(f"signal must be 'pred', 'deriv1' or 'deriv2'; got {signal!r}")


def _normalize_regions(sig, mode):
    """Per-region normalization across the age axis (axis=1), NaN-aware."""
    if mode is None:
        return sig
    if mode == 'center':
        return sig - np.nanmean(sig, axis=1, keepdims=True)
    if mode == 'zscore':
        mu = np.nanmean(sig, axis=1, keepdims=True)
        sd = np.nanstd(sig, axis=1, keepdims=True)
        return (sig - mu) / np.where(sd < 1e-12, 1.0, sd)
    raise ValueError(f"normalize must be None, 'center' or 'zscore'; got {mode!r}")


def _segment_costs(X):
    """Single-change-point L2 cost, vectorized via cumulative sums.

    X : (T, d) observations (one age per row). Returns (taus, costs) where
    costs[k] is the within-segment SS of splitting into left=[0:taus[k]) and
    right=[taus[k]:T), for every interior boundary taus[k] in 1..T-1.
    """
    T = X.shape[0]
    S1 = np.cumsum(X, axis=0)
    S2 = np.cumsum(X ** 2, axis=0)
    tot1, tot2 = S1[-1], S2[-1]
    taus = np.arange(1, T)
    nL = taus[:, None].astype(float)
    nR = (T - taus)[:, None].astype(float)
    sumL, sqL = S1[taus - 1], S2[taus - 1]
    sumR, sqR = tot1 - sumL, tot2 - sqL
    ssL = (sqL - sumL ** 2 / nL).sum(axis=1)
    ssR = (sqR - sumR ** 2 / nR).sum(axis=1)
    return taus, ssL + ssR


def _dynp_l2_costs(X, kmax):
    """Exact DP segmentation (== ruptures Dynp, model='l2'). Best total within-segment
    SS for K=1..kmax contiguous segments. X: (T, d)."""
    T = X.shape[0]
    S1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    S2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X ** 2, 0)])

    def seg(i, j):
        n = j - i
        s, sq = S1[j] - S1[i], S2[j] - S2[i]
        return float((sq - s ** 2 / n).sum())

    dp = np.full((kmax + 1, T + 1), np.inf)
    dp[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for t in range(k, T + 1):
            for s in range(k - 1, t):
                c = dp[k - 1, s] + seg(s, t)
                if c < dp[k, t]:
                    dp[k, t] = c
    return dp[1:kmax + 1, T]


def _ruptures_bkps(X, model, min_size, n_bkps):
    """n_bkps ordered breakpoint indices via ruptures Dynp (lazy import); trailing endpoint dropped."""
    try:
        import ruptures as rpt
    except ImportError as e:  # ruptures is optional; only needed for cost != None or n_bkps > 1
        raise ImportError("this code path requires the optional 'ruptures' package "
                          "(pip install ruptures).") from e
    algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(np.ascontiguousarray(X))
    return [int(b) for b in algo.predict(n_bkps=n_bkps)[:-1]]   # drop the final endpoint (== T)


def detect_stage_boundary(pred, age_grid, *, signal='deriv1', normalize='zscore',
                          min_segment_frac=0.10, search_bounds=None, cost=None, n_bkps=1):
    """Data-driven developmental stage boundary via multivariate change-point detection.

    Splits age_grid into contiguous stages at the age(s) minimising the within-stage
    sum of squared deviations from each stage's per-region mean of ``signal``
    (multivariate L2 cost == 1D-ordered k-means with a contiguity constraint ==
    ruptures Dynp). Per-region z-scoring makes the boundary reflect reorganization of
    the spatial PATTERN, not a few high-amplitude regions.

    By default (n_bkps=1) it finds a single boundary (two stages). ``cost=None`` then
    uses a fast exact vectorised L2 detector -- provably identical to ruptures
    Dynp(model='l2', n_bkps=1) but ~9x faster, so it is the one used in the bootstrap
    hot path. n_bkps > 1 (or a non-L2 ``cost`` string) routes to ruptures Dynp.

    Parameters
    ----------
    pred : (n_regions, n_grid) ndarray
        Fitted GAM level on age_grid.
    age_grid : (n_grid,) ndarray
    signal : {'deriv1', 'deriv2', 'pred'}
        Detect on rate of change (default), acceleration, or level. See build_stage_signal.
    normalize : {'zscore', 'center', None}
        Per-region normalization across age before detection.
    min_segment_frac : float
        Minimum fraction of the grid each stage must span (guards the unreliable
        derivative tails at the age extremes).
    search_bounds : (age_lo, age_hi) or None
        Optional hard age window the boundary must fall within. Exact single-boundary
        (cost=None, n_bkps=1) backend only.
    cost : None or str
        None -> exact L2. A ruptures cost model string ('rbf', 'normal', 'cosine',
        'l1', ...) detects with that cost via ruptures instead.
    n_bkps : int
        Number of change points to detect (>= 1). n_bkps > 1 requires ruptures and
        returns that many ordered boundaries; the two-stage narrative uses n_bkps=1.

    Returns
    -------
    dict
        'boundary_age', 'boundary_idx' : the FIRST boundary (age and age_grid index),
                                         kept for the single-boundary callers,
        'boundary_ages', 'boundary_idxs' : all detected boundaries (length n_bkps),
        'var_explained'                : 1 - SS(all stages)/SS(1 stage) of the normalized
                                         signal (L2 readout, for comparability),
        'cost_curve'                   : within-stage L2 SS per candidate boundary
                                         (age_grid-aligned; None unless the exact single
                                         backend ran).
    """
    if n_bkps < 1:
        raise ValueError(f"n_bkps must be >= 1; got {n_bkps}")
    sig = _normalize_regions(build_stage_signal(pred, age_grid, signal), normalize)
    finite = np.all(np.isfinite(sig), axis=1)   # keep regions finite across the whole grid
    X = sig[finite].T                           # (T=n_grid, n_valid_regions)
    grid = np.asarray(age_grid)
    T = X.shape[0]
    min_size = max(1, int(np.ceil(min_segment_frac * T)))

    if cost is None and n_bkps == 1:
        taus, costs = _segment_costs(X)
        adm = (taus >= min_size) & (taus <= T - min_size)
        if search_bounds is not None:
            a0, a1 = search_bounds
            adm &= (grid[taus] >= a0) & (grid[taus] <= a1)
        if not adm.any():
            raise ValueError("No admissible boundary; relax min_segment_frac or search_bounds.")
        idxs = [int(taus[adm][int(np.argmin(costs[adm]))])]
        cost_curve = np.full(T, np.nan)
        cost_curve[taus] = costs
    else:
        if search_bounds is not None:
            raise ValueError("search_bounds is only supported for the exact L2 single-boundary "
                             "backend (cost=None, n_bkps=1).")
        idxs = sorted(_ruptures_bkps(X, cost or 'l2', min_size, n_bkps))
        cost_curve = None

    # L2 variance-explained of the full segmentation (descriptive; comparable across costs)
    tot_ss = float((X.var(axis=0) * T).sum())
    bnds = [0] + idxs + [T]
    split_ss = float(sum(((X[a:b] - X[a:b].mean(0)) ** 2).sum() for a, b in zip(bnds[:-1], bnds[1:])))
    ages = [float(grid[i]) for i in idxs]
    return dict(boundary_age=ages[0], boundary_idx=idxs[0],
                boundary_ages=ages, boundary_idxs=idxs,
                var_explained=float(1 - split_ss / tot_ss) if tot_ss > 0 else np.nan,
                cost_curve=cost_curve)


def stage_segmentation_costs(pred, age_grid, *, signal='deriv1', normalize='zscore', kmax=4):
    """Within-stage SS for K=1..kmax contiguous stages (exact DP; ruptures Dynp l2).

    Diagnostic for how many developmental stages the data support. On smooth GAM
    signals the SS falls off smoothly with K (no sharp elbow), so use this to
    confirm that K=2 captures the dominant regime shift -- not to *select* K, which
    would over-segment. Returns costs (kmax,) for K=1..kmax segments.
    """
    sig = _normalize_regions(build_stage_signal(pred, age_grid, signal), normalize)
    X = sig[np.all(np.isfinite(sig), axis=1)].T
    return np.asarray(_dynp_l2_costs(X, kmax))


def select_n_stages(pred, age_grid, *, signal='deriv1', normalize='zscore', cost='l2',
                    min_segment_frac=0.10, pen=None):
    """Penalty-based selection of the number of developmental stages (ruptures Pelt).

    Unlike stage_segmentation_costs (which returns the SS-vs-K curve for you to inspect),
    this lets a penalty choose the breakpoint count directly. On smooth GAM signals the
    result is penalty-sensitive and tends to over-segment at a plain BIC penalty, so treat
    it as an exploratory robustness check on the two-stage assumption rather than an
    authoritative K -- sweep ``pen`` and report the range. Requires the optional 'ruptures'
    package.

    Parameters
    ----------
    pred, age_grid, signal, normalize : as in detect_stage_boundary.
    cost : str
        ruptures cost model ('l2', 'rbf', 'normal', ...).
    min_segment_frac : float
        Minimum fraction of the grid per segment.
    pen : float or None
        Penalty per breakpoint. None -> a BIC-like default (n_regions * log(T)).

    Returns
    -------
    dict
        'n_stages'      : number of stages selected (breakpoints + 1),
        'boundary_ages' : ages of the selected breakpoints,
        'pen'           : penalty used.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("select_n_stages requires the optional 'ruptures' package "
                          "(pip install ruptures).") from e
    sig = _normalize_regions(build_stage_signal(pred, age_grid, signal), normalize)
    X = sig[np.all(np.isfinite(sig), axis=1)].T
    T = X.shape[0]
    min_size = max(1, int(np.ceil(min_segment_frac * T)))
    if pen is None:
        pen = X.shape[1] * np.log(T)      # BIC-like default (dimension * log n)
    algo = rpt.Pelt(model=cost, min_size=min_size, jump=1).fit(np.ascontiguousarray(X))
    bkps = algo.predict(pen=pen)[:-1]     # drop the trailing endpoint (== T)
    grid = np.asarray(age_grid)
    return dict(n_stages=len(bkps) + 1,
                boundary_ages=[float(grid[b]) for b in bkps],
                pen=float(pen))


def _stage_windows_at(age_grid, infl_ix):
    """Column indices and durations for two stages split at grid index infl_ix."""
    cols1 = np.arange(0, infl_ix + 1)
    cols2 = np.arange(infl_ix, len(age_grid))
    dur1  = age_grid[infl_ix] - age_grid[0]
    dur2  = age_grid[-1]      - age_grid[infl_ix]
    return cols1, cols2, dur1, dur2


def _stage_windows(age_grid, inflection_age):
    """Column indices and durations for the two developmental stages (see compute_stage_maps)."""
    return _stage_windows_at(age_grid, int(np.argmin(np.abs(age_grid - inflection_age))))


def _integrate_deriv(deriv, age_grid, cols):
    """Trapezoidal integral of a (n_regions, n_grid) derivative over `cols` (net change)."""
    x = age_grid[cols]
    d = deriv[:, cols]
    finite = np.isfinite(x) & np.isfinite(d).any(axis=0)  # drop the edge-NaN derivative column
    return np.trapezoid(d[:, finite], x[finite], axis=1)


def _bootstrap_one(idx, deflection, age, nuisance, age_grid, n_splines, lam, demean,
                   sa_axis_vals, method, stage_windows, annualized,
                   stage_signal, report_signals, boundary_cfg):
    """One bootstrap resample: refit the deflection GAMs (all regions) once, then reduce the
    fitted stack to the requested summaries. Runs in a worker process; BLAS pinned to one
    thread so many workers don't oversubscribe cores.

    stage_windows : precomputed integration windows (fixed-boundary mode), or None.
    stage_signal  : if not None, auto-detect the boundary from this signal per resample and
                    integrate the stage maps over the resample's own window (overrides
                    stage_windows).
    report_signals: signals to additionally detect + report a boundary age for (CI diagnostics).
    boundary_cfg  : kwargs forwarded to detect_stage_boundary.
    lam           : scalar (same for every region) or a per-region array of frozen penalties
                    (e.g. GCV values selected once on the full sample) to reuse here.

    Returns dict: 'sa_curve', 'stage1', 'stage2' (any may be None) and 'boundaries'
    ({signal -> boundary_age}).
    """
    age_b  = age[idx]
    defl_b = deflection[idx]
    nuis_b = nuisance[idx] if nuisance is not None else None
    n_regions = deflection.shape[1]
    n_grid    = len(age_grid)
    preds  = np.full((n_regions, n_grid), np.nan)
    derivs = np.full((n_regions, n_grid), np.nan)
    lam_per_region = np.ndim(lam) > 0   # frozen per-region penalties vs a single scalar

    with (threadpool_limits(1) if threadpool_limits is not None else nullcontext()):
        for i in range(n_regions):
            y = defl_b[:, i].copy()
            if demean:
                y = y - np.nanmean(y)
            lam_i = lam[i] if lam_per_region else lam
            if lam_per_region and not np.isfinite(lam_i):
                lam_i = 0.1     # region unfit on the full sample; harmless fallback
            pred, _, _, deriv, _ = fit_gam(age_b, y, age_grid, n_splines=n_splines, lam=lam_i,
                                           nuisance=nuis_b)
            preds[i]  = pred
            derivs[i] = deriv

    out = {'sa_curve': None, 'stage1': None, 'stage2': None, 'boundaries': {}}
    if sa_axis_vals is not None:
        out['sa_curve'] = compute_deriv_sa_corr(derivs, sa_axis_vals, method=method)

    # Data-driven boundary(ies): the signal used for the stage maps plus any extra CI signals.
    need = set(report_signals or ())
    if stage_signal is not None:
        need.add(stage_signal)
    detected = {sig: detect_stage_boundary(preds, age_grid, signal=sig, **boundary_cfg)
                for sig in need}
    out['boundaries'] = {sig: r['boundary_age'] for sig, r in detected.items()}

    # Stage maps: an auto per-resample window (overrides the fixed one) or the fixed window.
    windows = (_stage_windows_at(age_grid, detected[stage_signal]['boundary_idx'])
               if stage_signal is not None else stage_windows)
    if windows is not None:
        cols1, cols2, dur1, dur2 = windows
        s1 = _integrate_deriv(derivs, age_grid, cols1)
        s2 = _integrate_deriv(derivs, age_grid, cols2)
        if annualized:
            s1, s2 = s1 / dur1, s2 / dur2
        out['stage1'], out['stage2'] = s1, s2
    return out


_STAGE_SIGNALS = ('pred', 'deriv1', 'deriv2')


def bootstrap_gam_maps(deflection, age, age_grid, *, sa_axis_vals=None, inflection_age=None,
                       n_splines=4, lam=0.1, demean=True, nuisance=None, n_boot=200,
                       annualized=True, method='pearson', random_state=None,
                       report_boundary_signals=(), boundary_kw=None,
                       n_jobs=-1, verbose=True):
    """Subject bootstrap of the deflection GAMs, reduced to the summaries fig4/fig5 need.

    Each iteration resamples subjects with replacement and refits the per-region deflection
    GAMs **once**, then reduces the fitted stack to (optionally) the derivative x S-A axis
    correlation curve, the stage-1/stage-2 net-change maps, and a data-driven stage boundary.
    Computing all of these from the same refit removes the duplicated fitting that fig4 (SA-corr
    CI) and fig5 (enrichment stage maps) previously performed separately.

    Iterations run in parallel over subject resamples (joblib). Resample indices are drawn once
    in the main process from ``random_state``, so the result is identical for any ``n_jobs``. The
    'sa_curves'/'stage1'/'stage2' arrays match the previous implementation for a given seed
    whenever a *fixed* inflection_age is used (boundary reporting is purely additive).

    Parameters
    ----------
    deflection : (n_subjects, n_regions) ndarray
    age : (n_subjects,) ndarray
    age_grid : (n_grid,) ndarray
    sa_axis_vals : (n_regions,) ndarray or None
        If given, return the derivative x S-A correlation curve per bootstrap.
    inflection_age : float, {'pred', 'deriv1', 'deriv2'}, or None
        Controls the stage-1/stage-2 net-change maps:
          * float  -> fixed boundary at that age (previous behaviour),
          * signal -> detect the boundary per resample from that signal (data-driven; each
                      resample integrates over its own stage windows),
          * None   -> no stage maps.
    report_boundary_signals : sequence of {'pred', 'deriv1', 'deriv2'}
        Extra signals to detect + report a boundary age for per resample (for boundary CIs),
        beyond the one driving the stage maps. Detection reuses the same refit, so this is cheap.
    boundary_kw : dict or None
        Forwarded to detect_stage_boundary (normalize, min_segment_frac, search_bounds).
        Defaults to dict(normalize='zscore', min_segment_frac=0.10).
    annualized : bool
        Divide each stage's net change by its duration in years (matches compute_stage_maps).
    n_jobs : int
        joblib worker count (-1 = all cores).
    Other parameters mirror fit_gam / fit_series.

    Returns
    -------
    dict with any of:
        'sa_curves' : (n_boot, n_grid)             -- if sa_axis_vals is given
        'stage1', 'stage2' : (n_boot, n_regions)   -- if inflection_age is given
        'boundary_<signal>' : (n_boot,)            -- detected boundary ages per resample, for
                                                      the stage-map signal and each requested
                                                      report_boundary_signals entry
    """
    boundary_cfg = dict(normalize='zscore', min_segment_frac=0.10)
    if boundary_kw:
        boundary_cfg.update(boundary_kw)

    if isinstance(inflection_age, str):
        if inflection_age not in _STAGE_SIGNALS:
            raise ValueError(f"inflection_age signal must be one of {_STAGE_SIGNALS}; "
                             f"got {inflection_age!r}")
        stage_signal, stage_windows, want_stage_maps = inflection_age, None, True
    elif inflection_age is not None:
        stage_signal = None
        stage_windows = _stage_windows(age_grid, inflection_age)
        want_stage_maps = True
    else:
        stage_signal, stage_windows, want_stage_maps = None, None, False

    report_signals = tuple(report_boundary_signals or ())
    bad = set(report_signals) - set(_STAGE_SIGNALS)
    if bad:
        raise ValueError(f"report_boundary_signals must be subset of {_STAGE_SIGNALS}; got {bad}")

    if sa_axis_vals is None and not want_stage_maps and not report_signals:
        raise ValueError("Request at least one of sa_axis_vals, inflection_age, "
                         "or report_boundary_signals.")

    rng     = np.random.default_rng(random_state)
    n_subs  = deflection.shape[0]
    idx_all = [rng.integers(0, n_subs, size=n_subs) for _ in range(n_boot)]

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_bootstrap_one)(
            idx, deflection, age, nuisance, age_grid, n_splines, lam, demean,
            sa_axis_vals, method, stage_windows, annualized,
            stage_signal, report_signals, boundary_cfg)
        for idx in idx_all)

    out = {}
    if sa_axis_vals is not None:
        out['sa_curves'] = np.asarray([r['sa_curve'] for r in results])
    if want_stage_maps:
        out['stage1'] = np.asarray([r['stage1'] for r in results])
        out['stage2'] = np.asarray([r['stage2'] for r in results])
    boundary_sigs = set(report_signals)
    if stage_signal is not None:
        boundary_sigs.add(stage_signal)
    for sig in sorted(boundary_sigs):
        out[f'boundary_{sig}'] = np.asarray([r['boundaries'].get(sig, np.nan) for r in results])
    return out


def bootstrap_deriv_sa_corr(deflection, age, sa_axis_vals, age_grid,
                             n_splines=4, lam=0.1, demean=True, nuisance=None,
                             n_boot=200, method='pearson', random_state=None,
                             verbose=True, n_jobs=-1):
    """Bootstrap CI for the derivative-SA correlation curve.

    Thin backward-compatible wrapper over :func:`bootstrap_gam_maps` (parallel). For a given
    ``random_state`` it returns the same ``(n_boot, n_grid)`` curves as the previous serial
    implementation.
    """
    return bootstrap_gam_maps(
        deflection, age, age_grid, sa_axis_vals=sa_axis_vals, inflection_age=None,
        n_splines=n_splines, lam=lam, demean=demean, nuisance=nuisance, n_boot=n_boot,
        annualized=True, method=method, random_state=random_state,
        n_jobs=n_jobs, verbose=verbose)['sa_curves']


def add_sa_decile_colorbar(fig, ax_ref, cmap, n_deciles,
                            cbar_thickness=0.012, pad=0.010, fontsize=7,
                            orientation='horizontal'):
    """Add an SA-decile colorbar adjacent to ax_ref, aligned to its data extent.

    orientation='horizontal' places a strip below ax_ref spanning its width;
    orientation='vertical' places a strip to the right of ax_ref spanning its height.
    Call after fig.canvas.draw() so that axis positions are finalised.

    Parameters
    ----------
    fig : matplotlib Figure
    ax_ref : matplotlib Axes
        The Axes whose extent the colorbar is aligned to (width if horizontal,
        height if vertical).
    cmap : matplotlib colormap
        Colormap to display (e.g. from create_sydnor_sa_colormap()).
    n_deciles : int
        Number of decile ticks to display.
    cbar_thickness : float
        Figure-fraction thickness of the colorbar strip (height if horizontal,
        width if vertical).
    pad : float
        Figure-fraction gap between ax_ref's tight bbox and the colorbar.
    fontsize : int
        Colorbar label font size.
    orientation : {'horizontal', 'vertical'}
        Placement relative to ax_ref (below or to the right).

    Returns
    -------
    ax_cb : matplotlib Axes
        The new colorbar Axes.
    """
    renderer = fig.canvas.get_renderer()
    fw, fh   = np.array(fig.get_size_inches()) * fig.dpi
    bbox     = ax_ref.get_window_extent(renderer=renderer)
    tight    = ax_ref.get_tightbbox(renderer=renderer)
    if orientation == 'vertical':
        rect = [tight.x1 / fw + pad, bbox.y0 / fh, cbar_thickness, bbox.height / fh]
    else:
        rect = [bbox.x0 / fw, tight.y0 / fh - pad - cbar_thickness,
                bbox.width / fw, cbar_thickness]
    ax_cb = fig.add_axes(rect)
    norm = mcolors.Normalize(vmin=1, vmax=n_deciles)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation=orientation)
    cb.set_label('S-A decile', fontsize=fontsize)
    cb.set_ticks(list(range(1, n_deciles + 1)))
    cb.ax.tick_params(labelsize=fontsize - 1, pad=1)
    return ax_cb


def plot_deriv_sa_corr_with_ci(ax, age_grid, r_vals, r_ci_lo, r_ci_hi,
                                lw=1.8, color='#4C72B0', fontsize=8):
    """Plot pre-computed r(derivative × SA-axis) with bootstrap CI and annotations.

    Shades the CI band darker where it lies entirely above or below zero
    (significant), lighter where it straddles zero. Labels the ages where the CI
    band's significance begins/ends with text annotations.

    Parameters
    ----------
    ax : matplotlib Axes
    age_grid : (n_grid,) ndarray
        Evaluation points (x-axis).
    r_vals : (n_grid,) ndarray
        Observed r curve.
    r_ci_lo : (n_grid,) ndarray
        2.5th bootstrap percentile.
    r_ci_hi : (n_grid,) ndarray
        97.5th bootstrap percentile.
    lw : float
        Line width.
    color : str
        Line and fill colour.
    fontsize : int
        Base font size (annotations use fontsize - 1).
    """
    _sig = (r_ci_lo > 0) | (r_ci_hi < 0)
    ax.fill_between(age_grid, r_ci_lo, r_ci_hi,
                    where=_sig,  color=color, alpha=0.35, interpolate=True)
    ax.fill_between(age_grid, r_ci_lo, r_ci_hi,
                    where=~_sig, color=color, alpha=0.12, interpolate=True)
    ax.plot(age_grid, r_vals, color=color, lw=lw)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)

    for bound_arr, sig_mask, y_pos, y_va in [
        (r_ci_lo, r_ci_lo > 0, 0.99, 'top'),
        (r_ci_hi, r_ci_hi < 0, 0.03, 'bottom'),
    ]:
        _diffs = np.diff(sig_mask.astype(int))
        for idx in np.where(_diffs)[0]:
            a0, a1    = bound_arr[idx], bound_arr[idx + 1]
            cross_age = age_grid[idx] + (-a0 / (a1 - a0)) * (age_grid[idx + 1] - age_grid[idx])
            if _diffs[idx] > 0:
                ha, xpos = 'right', cross_age - 0.15
            else:
                ha, xpos = 'left',  cross_age + 0.15
            ax.text(xpos, y_pos, f'{cross_age:.1f} yr',
                    ha=ha, va=y_va, fontsize=fontsize - 1, color='k',
                    transform=ax.get_xaxis_transform())


def compute_deriv_sa_corr(deriv, sa_axis_vals, method='pearson'):
    """Pearson/Spearman r between per-region SA values and the derivative at each age point.

    Parameters
    ----------
    deriv : (n_regions, n_grid) ndarray
        Per-region temporal derivatives.
    sa_axis_vals : (n_regions,) ndarray
        SA values in parcellation order.
    method : {'pearson', 'spearman'}
        Correlation type.

    Returns
    -------
    r_vals : (n_grid,) ndarray
        Correlation at each age point; NaN where fewer than 3 regions are valid.
    """
    n_grid  = deriv.shape[1]
    r_vals  = np.full(n_grid, np.nan)
    corr_fn = sp.stats.pearsonr if method == 'pearson' else sp.stats.spearmanr
    for t in range(n_grid):
        col  = deriv[:, t]
        mask = ~np.isnan(col)
        if mask.sum() > 2:
            r_vals[t] = corr_fn(sa_axis_vals[mask], col[mask]).statistic
    return r_vals


def compute_stage_maps(gam_smooths, age_grid, inflection_age=17, flat_ratio_thresh=0.5,
                       annualized=False):
    """Net change (or annualized rate of change) in fitted (absolute) INT deflection over two developmental stages.

    Reuses the per-parcel GAM smooths already fit for Figure 4 (no refitting). For each
    parcel, stage 1 = age_grid[0] -> inflection_age and stage 2 = inflection_age -> age_grid[-1].
    Net change is the integral of the fitted first derivative over each window (equivalently,
    the difference of the fitted value at the window endpoints). The deflection metric is
    identical to Figure 4 (absolute deflection; absolute=True in compute_deflections).

    The first grid column of the GAM derivative is non-finite by construction (a finite-
    difference edge effect), so each window's integral is taken over the finite derivative
    samples only; this is numerically identical (r > 0.999) to the endpoint difference, which
    is reported as a cross-check.

    Parameters
    ----------
    gam_smooths : dict with 'pred' and 'deriv', each (n_regions, n_grid) on age_grid
        e.g. results_reg['deflection'].
    age_grid : (n_grid,) ages at which the smooths/derivatives were evaluated.
    inflection_age : float, default 17. Stage boundary in years.
    flat_ratio_thresh : float, default 0.5. Warn if SD(stage1) < thresh * SD(stage2).
    annualized : bool, default False. If True, return the annualized rate of change
        (net change divided by the stage duration in years) instead of the total net change.

    Returns
    -------
    stage1, stage2 : (n_regions,) signed net-change vectors (or annualized rates if annualized=True).
    """
    pred  = np.asarray(gam_smooths['pred'])
    deriv = np.asarray(gam_smooths['deriv'])
    n_grid = deriv.shape[1]
    idx    = int(np.argmin(np.abs(age_grid - inflection_age)))

    def _integrate(cols):
        x = age_grid[cols]
        d = deriv[:, cols]
        finite = np.isfinite(x) & np.isfinite(d).any(axis=0)  # keep cols finite for >=1 region (drops the edge-NaN col); all-NaN regions stay NaN
        return np.trapezoid(d[:, finite], x[finite], axis=1)

    # Integral of the fitted first derivative over each window (signed net change)
    stage1 = _integrate(np.arange(0, idx + 1))
    stage2 = _integrate(np.arange(idx, n_grid))

    # Cross-check: equals fitted-value difference at window endpoints
    endpoint1 = pred[:, idx] - pred[:, 0]
    endpoint2 = pred[:, -1]  - pred[:, idx]
    max_dev = max(np.nanmax(np.abs(stage1 - endpoint1)),
                  np.nanmax(np.abs(stage2 - endpoint2)))

    sd1, sd2 = np.nanstd(stage1), np.nanstd(stage2)
    var1, var2 = np.nanvar(stage1), np.nanvar(stage2)
    print(f'Stage boundary: age_grid[{idx}] = {age_grid[idx]:.2f} yr (target {inflection_age})')
    print(f'Stage 1 ({age_grid[0]:.1f}->{age_grid[idx]:.1f} yr): spatial SD = {sd1:.4f}, spatial VAR = {var1:.4f}')
    print(f'Stage 2 ({age_grid[idx]:.1f}->{age_grid[-1]:.1f} yr): spatial SD = {sd2:.4f}, spatial VAR = {var2:.4f}')
    print(f'Integral vs endpoint max abs deviation = {max_dev:.2e} '
          f'(small edge-truncation expected; should be << SD)')

    if sd2 > 0 and sd1 < flat_ratio_thresh * sd2:
        print(f'WARNING: stage-1 map is near-flat relative to stage-2 '
              f'(SD ratio = {sd1 / sd2:.2f} < {flat_ratio_thresh}). A spatially flat stage-1 '
              f'map cannot be expected to correlate with any spatially structured receptor map.')

    if annualized:
        # Convert total net change to an annualized rate (change per year) over each window
        dur1 = age_grid[idx] - age_grid[0]
        dur2 = age_grid[-1]  - age_grid[idx]
        stage1 = stage1 / dur1
        stage2 = stage2 / dur2
        print(f'Returning annualized rate of change (net change / stage duration): '
              f'stage 1 over {dur1:.1f} yr, stage 2 over {dur2:.1f} yr.')

    return stage1, stage2
