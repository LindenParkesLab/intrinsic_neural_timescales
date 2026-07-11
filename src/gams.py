"""GAM fitting and visualisation utilities for intrinsic neural timescales."""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygam

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; the bootstrap runs without a progress bar
    tqdm = None


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

def fit_gam(age, y, age_grid, n_splines=4, lam=0.1, show_ci=False, nuisance=None):
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
    lam : float
        Smoothing penalty for the age smooth.
    show_ci : bool
        Whether to compute 95% prediction intervals.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates entered as linear terms l() in the GAM. When predicting on
        age_grid, nuisance values are fixed at their within-sample means so the
        curve reflects the age smooth only.

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

    if nuisance is not None:
        nuisance        = np.asarray(nuisance)
        nuisance_masked = nuisance[mask]
        X_fit           = np.column_stack([age[mask], nuisance_masked])
        terms           = pygam.s(0, n_splines=n_splines, lam=lam)
        for i in range(1, nuisance.shape[1] + 1):
            terms = terms + pygam.l(i)
        nuisance_means = nuisance_masked.mean(axis=0)
        X_pred = np.column_stack([
            age_grid,
            np.tile(nuisance_means, (len(age_grid), 1)),
        ])
    else:
        X_fit  = age[mask].reshape(-1, 1)
        terms  = pygam.s(0, n_splines=n_splines, lam=lam)
        X_pred = age_grid.reshape(-1, 1)

    gam   = pygam.LinearGAM(terms)
    gam.fit(X_fit, Y)
    pred  = gam.predict(X_pred)
    dt    = np.diff(age_grid)
    deriv = np.concatenate([[np.nan], np.diff(pred) / dt])

    ci_lo, ci_hi = None, None
    if show_ci:
        ci    = gam.prediction_intervals(X_pred, width=0.95)
        ci_lo = ci[:, 0]
        ci_hi = ci[:, 1]

    return pred, ci_lo, ci_hi, deriv, gam


def fit_series(age, arr, age_grid,
               n_splines=4, lam=0.1, demean=True, show_ci=False, nuisance=None):
    """Fit GAMs for every column (series) in arr.

    Parameters
    ----------
    age : (n_subjects,) ndarray
        Subject ages.
    arr : (n_subjects, n_series) ndarray
        One column per series to fit.
    age_grid : (n_grid,) ndarray
        Evaluation points.
    n_splines, lam, show_ci : see fit_gam.
    demean : bool
        If True, each series is mean-centred (ignoring NaNs) before fitting.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates passed through to fit_gam.

    Returns
    -------
    dict
        Keys 'pred' (n_series, n_grid), 'deriv' (n_series, n_grid),
        'ci_lo'/'ci_hi' (n_series, n_grid) or None, and 'gams' (list of n_series
        fitted pygam.LinearGAM objects, None where a series had too few valid
        subjects).
    """
    n_ser  = arr.shape[1]
    preds  = np.full((n_ser, len(age_grid)), np.nan)
    derivs = np.full((n_ser, len(age_grid)), np.nan)
    ci_los = np.full((n_ser, len(age_grid)), np.nan) if show_ci else None
    ci_his = np.full((n_ser, len(age_grid)), np.nan) if show_ci else None
    gams   = []

    for i in range(n_ser):
        y = arr[:, i].copy()
        if demean:
            y = y - np.nanmean(y)
        pred, ci_lo, ci_hi, deriv, gam = fit_gam(
            age, y, age_grid, n_splines=n_splines, lam=lam,
            show_ci=show_ci, nuisance=nuisance)
        preds[i]  = pred
        derivs[i] = deriv
        if show_ci:
            ci_los[i] = ci_lo
            ci_his[i] = ci_hi
        gams.append(gam)

    return {
        'pred':  preds,
        'deriv': derivs,
        'ci_lo': ci_los,
        'ci_hi': ci_his,
        'gams':  gams,
    }


def fit_all_panels(age, rest, task, deflection, age_grid,
                   n_splines=4, lam=0.1, demean=True, show_ci=False, nuisance=None):
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

    Returns
    -------
    dict
        Keys 'rest', 'task', 'deflection' (each a fit_series result dict) plus
        'age_grid' (the evaluation grid).
    """
    kw      = dict(n_splines=n_splines, lam=lam, demean=demean,
                   show_ci=show_ci, nuisance=nuisance)
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


def bootstrap_deriv_sa_corr(deflection, age, sa_axis_vals, age_grid,
                             n_splines=4, lam=0.1, demean=True, nuisance=None,
                             n_boot=200, method='pearson', random_state=None,
                             verbose=True):
    """Bootstrap CI for the derivative-SA correlation curve.

    Each iteration resamples subjects with replacement, refits the deflection
    GAMs for all regions, and computes the derivative-SA correlation at each age
    point.

    Parameters
    ----------
    deflection : (n_subjects, n_regions) ndarray
        Per-subject deflection values.
    age : (n_subjects,) ndarray
        Subject ages.
    sa_axis_vals : (n_regions,) ndarray
        SA values in parcellation order.
    age_grid : (n_grid,) ndarray
        Evaluation points.
    nuisance : (n_subjects, n_nuisance) ndarray or None
        Covariates passed through to fit_gam.
    n_boot : int
        Number of bootstrap iterations.
    method : {'pearson', 'spearman'}
        Correlation type.
    random_state : int or None
        Seed for the resampling RNG.
    verbose : bool
        Show a tqdm progress bar if tqdm is installed.

    Returns
    -------
    boot_curves : (n_boot, n_grid) ndarray
        Bootstrap distribution of correlation curves.
    """
    rng        = np.random.default_rng(random_state)
    n_subs, n_regions = deflection.shape
    n_grid     = len(age_grid)
    boot_curves = np.full((n_boot, n_grid), np.nan)

    iterations = range(n_boot)
    if verbose and tqdm is not None:
        iterations = tqdm(iterations, desc='Bootstrap')

    for b in iterations:
        idx    = rng.integers(0, n_subs, size=n_subs)
        age_b  = age[idx]
        defl_b = deflection[idx]
        nuis_b = nuisance[idx] if nuisance is not None else None

        derivs = np.full((n_regions, n_grid), np.nan)
        for i in range(n_regions):
            y = defl_b[:, i].copy()
            if demean:
                y -= np.nanmean(y)
            _, _, _, deriv, _ = fit_gam(age_b, y, age_grid,
                                        n_splines=n_splines, lam=lam,
                                        nuisance=nuis_b)
            derivs[i] = deriv

        boot_curves[b] = compute_deriv_sa_corr(derivs, sa_axis_vals, method=method)

    return boot_curves


def add_sa_decile_colorbar(fig, ax_ref, cmap, n_deciles,
                            cbar_height=0.012, pad=0.010, fontsize=7):
    """Add a horizontal SA-decile colorbar below ax_ref, aligned to its data extent.

    Call after fig.canvas.draw() so that axis positions are finalised.

    Parameters
    ----------
    fig : matplotlib Figure
    ax_ref : matplotlib Axes
        The Axes whose x-extent defines the colorbar width.
    cmap : matplotlib colormap
        Colormap to display (e.g. from create_sydnor_sa_colormap()).
    n_deciles : int
        Number of decile ticks to display.
    cbar_height : float
        Figure-fraction height of the colorbar strip.
    pad : float
        Figure-fraction gap between ax_ref's tight bbox and the colorbar.
    fontsize : int
        Colorbar label font size.

    Returns
    -------
    ax_cb : matplotlib Axes
        The new colorbar Axes.
    """
    renderer = fig.canvas.get_renderer()
    fw, fh   = np.array(fig.get_size_inches()) * fig.dpi
    bbox     = ax_ref.get_window_extent(renderer=renderer)
    tight    = ax_ref.get_tightbbox(renderer=renderer)
    ax_cb    = fig.add_axes([
        bbox.x0    / fw,
        tight.y0   / fh - pad - cbar_height,
        bbox.width / fw,
        cbar_height,
    ])
    norm = mcolors.Normalize(vmin=1, vmax=n_deciles)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label('S-A decile', fontsize=fontsize)
    cb.set_ticks(list(range(1, n_deciles + 1)))
    cb.ax.tick_params(labelsize=fontsize - 1, pad=1)
    return ax_cb


def plot_deriv_sa_corr_with_ci(ax, age_grid, r_vals, r_ci_lo, r_ci_hi,
                                lw=1.8, color='#4C72B0', fontsize=8):
    """Plot pre-computed r(derivative × SA-axis) with bootstrap CI and annotations.

    Shades the CI band darker where it lies entirely above or below zero
    (significant), lighter where it straddles zero. Marks zero-crossing ages with
    dotted vertical lines.

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
            ax.axvline(cross_age, color='k', lw=0.75, ls=':', alpha=0.6)
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
