"""S-A axis, two-stage developmental maps, and figure utilities for intrinsic neural timescales.

Generic GAM fitting, change-point detection, and the bootstrap engine live in
``snaplab_tools.gams``; this module keeps only the project-specific pieces (S-A axis correlations,
two-stage net-change maps, and the figure builders) and wires the generic bootstrap engine to the
S-A / stage reductions that fig4 and fig5 need.
"""
import functools

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from snaplab_tools.gams import (
    fit_series, detect_change_point, integrate_derivative, segment_windows_at,
    bootstrap_gam_fits, SIGNALS, _DEFAULT_LAM_GRID,
)


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
# Bootstrap: S-A correlation curve + two-stage net-change maps (over the generic engine)
# =============================================================================

def _reduce_sa_stage(preds, derivs, x_grid, *, sa_axis_vals, method, stage_signal,
                     report_signals, boundary_cfg, stage_windows, annualized):
    """Per-resample reduction for bootstrap_gam_maps: the derivative x S-A correlation curve, the
    detected change-point location(s), and the stage-1/stage-2 net-change maps. Module-level so
    joblib can pickle it (bootstrap_gam_maps wraps it in functools.partial)."""
    out = {}
    if sa_axis_vals is not None:
        out['sa_curve'] = compute_deriv_sa_corr(derivs, sa_axis_vals, method=method)

    need = set(report_signals or ())
    if stage_signal is not None:
        need.add(stage_signal)
    detected = {sig: detect_change_point(preds, x_grid, signal=sig, **boundary_cfg) for sig in need}
    for sig, r in detected.items():
        out[f'boundary_{sig}'] = r['location']

    windows = (segment_windows_at(x_grid, detected[stage_signal]['index'])
               if stage_signal is not None else stage_windows)
    if windows is not None:
        cols1, cols2, dur1, dur2 = windows
        s1 = integrate_derivative(derivs, x_grid, cols1)
        s2 = integrate_derivative(derivs, x_grid, cols2)
        if annualized:
            s1, s2 = s1 / dur1, s2 / dur2
        out['stage1'], out['stage2'] = s1, s2
    return out


def bootstrap_gam_maps(deflection, age, age_grid, *, sa_axis_vals=None, inflection_age=None,
                       n_splines=4, lam=0.1, demean=True, nuisance=None, n_boot=200,
                       annualized=True, method='pearson', random_state=None,
                       report_boundary_signals=(), boundary_kw=None,
                       n_jobs=-1, verbose=True):
    """Subject bootstrap of the deflection GAMs, reduced to the S-A / stage summaries fig4/fig5 need.

    Thin project wrapper over ``snaplab_tools.gams.bootstrap_gam_fits``: it resamples subjects and
    refits every region's deflection GAM once (the generic engine), then reduces each refit to the
    derivative x S-A axis correlation curve, the stage-1/stage-2 net-change maps, and data-driven
    change-point location(s) (via ``_reduce_sa_stage``). Computing all of these from the same refit
    removes the duplicated fitting fig4 (SA-corr CI) and fig5 (enrichment stage maps) once did
    separately.

    Parameters
    ----------
    deflection : (n_subjects, n_regions) ndarray
    age : (n_subjects,) ndarray
    age_grid : (n_grid,) ndarray
    sa_axis_vals : (n_regions,) ndarray or None
        If given, return the derivative x S-A correlation curve per bootstrap ('sa_curves').
    inflection_age : float, one of SIGNALS, or None
        float -> fixed stage boundary; a signal ('pred'/'deriv1'/'deriv2') -> detect the boundary
        per resample from that signal; None -> no stage maps.
    report_boundary_signals : sequence of SIGNALS
        Extra signals to also detect + report a change-point location for per resample (boundary
        CIs), beyond the one driving the stage maps.
    lam : float, 'gcv', or (n_regions,) array
        Per-region smoothing; an array freezes penalties (e.g. GCV values from the full sample).
    boundary_kw : dict or None
        Forwarded to detect_change_point (normalize, min_segment_frac, search_bounds).

    Returns
    -------
    dict with any of:
        'sa_curves' : (n_boot, n_grid)             -- if sa_axis_vals is given
        'stage1', 'stage2' : (n_boot, n_regions)   -- if inflection_age is given
        'boundary_<signal>' : (n_boot,)            -- change-point location per resample
    """
    boundary_cfg = dict(normalize='zscore', min_segment_frac=0.10)
    if boundary_kw:
        boundary_cfg.update(boundary_kw)

    if isinstance(inflection_age, str):
        if inflection_age not in SIGNALS:
            raise ValueError(f"inflection_age signal must be one of {SIGNALS}; got {inflection_age!r}")
        stage_signal, stage_windows, want_stage_maps = inflection_age, None, True
    elif inflection_age is not None:
        stage_signal    = None
        stage_windows   = segment_windows_at(age_grid, int(np.argmin(np.abs(age_grid - inflection_age))))
        want_stage_maps = True
    else:
        stage_signal, stage_windows, want_stage_maps = None, None, False

    report_signals = tuple(report_boundary_signals or ())
    bad = set(report_signals) - set(SIGNALS)
    if bad:
        raise ValueError(f"report_boundary_signals must be subset of {SIGNALS}; got {bad}")

    if sa_axis_vals is None and not want_stage_maps and not report_signals:
        raise ValueError("Request at least one of sa_axis_vals, inflection_age, "
                         "or report_boundary_signals.")

    reduce = functools.partial(
        _reduce_sa_stage, sa_axis_vals=sa_axis_vals, method=method, stage_signal=stage_signal,
        report_signals=report_signals, boundary_cfg=boundary_cfg, stage_windows=stage_windows,
        annualized=annualized)
    stacked = bootstrap_gam_fits(deflection, age, age_grid, reduce=reduce, n_boot=n_boot,
                                 n_splines=n_splines, lam=lam, nuisance=nuisance, demean=demean,
                                 random_state=random_state, n_jobs=n_jobs, verbose=verbose)

    out = {}
    if sa_axis_vals is not None:
        out['sa_curves'] = stacked['sa_curve']
    if want_stage_maps:
        out['stage1'], out['stage2'] = stacked['stage1'], stacked['stage2']
    for k in stacked:
        if k.startswith('boundary_'):
            out[k] = stacked[k]
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
