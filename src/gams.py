"""S-A axis, two-stage developmental maps, and figure utilities for intrinsic neural timescales.

Generic GAM fitting, change-point detection, and the bootstrap engine live in
``snaplab_tools.gams``; this module keeps only the project-specific pieces (S-A axis correlations,
two-stage net-change maps, and the figure builders) and wires the generic bootstrap engine to the
S-A / stage reductions that fig3 and fig4 need.
"""
import functools
import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.timescales import rest_window_token
from snaplab_tools.gams import (
    fit_series, detect_change_point, integrate_derivative, segment_windows_at,
    bootstrap_gam_fits, SIGNALS, _DEFAULT_LAM_GRID,
)


# =============================================================================
# Run identity: the signal vocabulary, the variant tags, and the cache path
#
# fig3 writes the GAM cache and fig4 reads it, so every component of that filename and every name
# in the shared option vocabulary has to agree between the two notebooks. These functions are that
# agreement -- the contract is executable here rather than restated by hand in both notebooks.
# NOTE ``SIGNAL_TYPES`` (rest / task / deflection, the INT signal being modelled) is a different
# vocabulary from ``SIGNALS`` imported above (pred / deriv1 / deriv2, the curve a change point is
# detected on). Both are needed; do not conflate them.
# =============================================================================

SIGNAL_TYPES = ('rest', 'task', 'deflection')

GAM_RECIPES = {
    # 'historical' -> the original fixed-penalty recipe.
    # 'gcv'        -> lam chosen per series by GCV on the full sample, then frozen for bootstraps.
    'historical': dict(n_splines=4, lam=0.1),
    'gcv':        dict(n_splines=5, lam='gcv'),
}


def signal_labels(absolute=True):
    """Axis / title label per signal type. ``absolute`` names the deflection definition."""
    return {'rest': 'Rest INTs', 'task': 'Task INTs',
            'deflection': f'INT Deflections ({"abs." if absolute else "signed"})'}


def signal_units(absolute=True):
    """Short unit per signal type, for '\u0394<unit> / yr' style axis labels."""
    return {'rest': 'INT', 'task': 'INT',
            'deflection': '|\u0394INT|' if absolute else '\u0394INT'}


def gam_variant_tags(fit_recipe, absolute, demean, rest_length='full', rest_match=None):
    """Resolve a run's identity: fitting recipe plus the two data-definition flags.

    Parameters
    ----------
    fit_recipe : {'historical', 'gcv'}
        Key into :data:`GAM_RECIPES`; fixes ``n_splines`` and ``lam``.
    absolute : bool
        Deflection definition: True -> |task - rest|, False -> signed (task - rest).
    demean : bool
        Centre each series on its own mean before fitting.
    rest_length : {'full', 'matched'}
        Rest INTs from full-length runs, or from length-matched windows (``_restlen-<task>``; see
        ``src.timescales.REST_WINDOW_MATCH``). It changes the rest side of every deflection, so it
        is part of the run's identity. ``'full'`` adds no tag, so every existing cache and panel
        name is unchanged.
    rest_match : str or None
        The matched task under ``'matched'`` (e.g. ``'CARIT'``, from
        ``src.timescales.rest_window_match_for``); it names the tag, ``_restlen-CARIT``. Required
        with ``'matched'``, refused with ``'full'``.

    Returns
    -------
    dict
        ``fit_recipe``, ``n_splines``, ``lam``, ``recipe_tag``, ``absolute_tag``, ``demean_tag``,
        ``rest_length``, ``rest_match``, ``rest_tag`` and ``variant_tag`` (= absolute_tag + demean_tag + rest_tag,
        the component both notebooks put in panel filenames).
    """
    if fit_recipe not in GAM_RECIPES:
        raise ValueError(f"fit_recipe must be one of {sorted(GAM_RECIPES)}; got {fit_recipe!r}")
    n_splines = GAM_RECIPES[fit_recipe]['n_splines']
    lam       = GAM_RECIPES[fit_recipe]['lam']
    recipe_tag   = '' if fit_recipe == 'historical' else f'_ns{n_splines}-gcv'
    absolute_tag = '_abs' if absolute else '_signed'
    demean_tag   = '_demean' if demean else '_nodemean'
    if rest_length not in ('full', 'matched'):
        raise ValueError(f"rest_length must be 'full' or 'matched'; got {rest_length!r}")
    if rest_length == 'matched' and not rest_match:
        raise ValueError("rest_length='matched' needs rest_match, the matched task's name (e.g. "
                         "'CARIT'; src.timescales.rest_window_match_for)")
    if rest_length == 'full' and rest_match is not None:
        raise ValueError(f"rest_match={rest_match!r} is only meaningful with rest_length='matched'")
    rest_tag     = '' if rest_length == 'full' else f'_{rest_window_token(rest_match)}'
    return dict(fit_recipe=fit_recipe, n_splines=n_splines, lam=lam,
                recipe_tag=recipe_tag, absolute_tag=absolute_tag, demean_tag=demean_tag,
                rest_length=rest_length, rest_match=rest_match, rest_tag=rest_tag,
                variant_tag=f'{absolute_tag}{demean_tag}{rest_tag}')


def gam_cache_path(gam_out_dir, dataset, n_regions, task_scan, tags):
    """Path of the fig3 GAM export for one variant (``tags`` from :func:`gam_variant_tags`)."""
    return os.path.join(
        gam_out_dir,
        f'{dataset}_{n_regions}_{task_scan}{tags["recipe_tag"]}{tags["absolute_tag"]}'
        f'{tags["demean_tag"]}{tags.get("rest_tag", "")}_regional_gam_results.npy')


# Head-motion column(s) per dataset; None means this dataset gets no nuisance covariates.
FD_COLUMN = {'HCP-D': 'meanFD', 'HCP-YA': 'meanRelRMS_rest',
             'WAND': 'meanFD_rest', 'Penn_LEAD': 'meanFD_rest'}


def build_nuisance(df_behavior, rest, dataset, include_age=False, fd_columns=None):
    """Nuisance covariates: [(age,) sex, head motion..., subject mean resting INT].

    fig3 fits with these and fig4 re-fits with them inside its bootstrap; if the two ever disagree
    fig4 stops reproducing fig3's fit while still looking correct, so both call this. fig2 calls it
    too, with ``include_age=True`` and an explicit ``fd_columns`` pair.

    Parameters
    ----------
    df_behavior : pd.DataFrame
        Must carry 'sex', 'age' when ``include_age``, and every requested motion column.
    rest : (n_subjects, n_regions) ndarray
        Resting INTs; the row mean becomes the last covariate. fig3/fig4 must pass the
        FULL-parcellation matrix: under a region mask the subject mean would be taken over a
        different set of regions in the two notebooks, which is one more reason fig4 refuses a
        masked export. That is a constraint of the shared-cache contract, not of this function --
        fig2 deliberately passes its masked matrix, since it shares no export with anything.
    dataset : str
        Key into :data:`FD_COLUMN`, used only when ``fd_columns`` is None.
    include_age : bool
        Prepend 'age'. False for fig3/fig4, where age is the smooth term and so cannot also be a
        nuisance; True for fig2, where age is an ordinary confound.
    fd_columns : str, sequence of str, or None
        Motion column(s), overriding the :data:`FD_COLUMN` default. fig2 passes rest and its task
        condition separately (e.g. ``['meanRelRMS_rest', 'meanRelRMS_WM']``): the two correlate at
        r ~ 0.82 but the INT deflection is a task-minus-rest difference, so a single averaged
        motion term would average over the very contrast the deflection is built on.

    Returns
    -------
    (n_subjects, k) ndarray, or None when the dataset has no motion column and none was given.

    Raises
    ------
    KeyError
        If any requested column is absent from ``df_behavior`` -- a silent drop here would
        quietly change the model.
    """
    cols = FD_COLUMN.get(dataset) if fd_columns is None else fd_columns
    if cols is None:
        return None
    if isinstance(cols, str):
        cols = [cols]

    names = (['age'] if include_age else []) + ['sex'] + list(cols)
    missing = [c for c in names if c not in df_behavior.columns]
    if missing:
        raise KeyError(
            f"build_nuisance: {missing} not in df_behavior for dataset {dataset!r}. "
            f"HCP-YA motion columns come from src.data.load_hcpya_movement; delete "
            f"data/hcpya_movement_relativerms.csv to rebuild the cache."
        )

    return np.concatenate([df_behavior[names].values.astype(float),
                           np.nanmean(rest, axis=1)[:, np.newaxis]], axis=1)


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
    """Subject bootstrap of the deflection GAMs, reduced to the S-A / stage summaries fig3/fig4 need.

    Thin project wrapper over ``snaplab_tools.gams.bootstrap_gam_fits``: it resamples subjects and
    refits every region's deflection GAM once (the generic engine), then reduces each refit to the
    derivative x S-A axis correlation curve, the stage-1/stage-2 net-change maps, and data-driven
    change-point location(s) (via ``_reduce_sa_stage``). Computing all of these from the same refit
    removes the duplicated fitting fig3 (SA-corr CI) and fig4 (enrichment stage maps) once did
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
        Colormap to display (e.g. register_custom_colormaps()['snap_sydnor_sa']).
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


def plot_r_age(ax, age_grid, r_vals, *, ci=None, sig=None, style='band',
               lw=1.8, color='#4C72B0', fontsize=8, annotate=True, label=None):
    """Draw an r(age) curve and mark where it is significant, in either of the two house idioms.

    fig3's panel E and fig4's coupling traces are the same object -- a correlation against age with
    a significance verdict per age point -- drawn two ways. This is that one drawing routine, so the
    two figures cannot drift apart, and so a curve can carry a band from one test and a significance
    verdict from another (fig3's SIG_METHOD='both': bootstrap band, spatial-null marks).

    Parameters
    ----------
    ax : matplotlib Axes
    age_grid : (n_grid,) ndarray
        Evaluation points (x-axis).
    r_vals : (n_grid,) ndarray
        Observed correlation per age.
    ci : (lo, hi) tuple of (n_grid,) ndarrays, or None
        Confidence band. Required for ``style='band'``; optional otherwise (when given it also
        sharpens the annotated ages, see below).
    sig : (n_grid,) bool ndarray, or None
        Significance per age. Defaults to "the band excludes zero" when ``ci`` is given, which
        reproduces the original bootstrap-only behaviour. Pass it explicitly to mark significance
        from a different test than the one drawing the band.
    style : {'band', 'trace'}
        'band'  -- shade the CI, darker where significant, with the curve solid on top (fig3 E).
        'trace' -- no shading: the curve is faded across the full range and bold where significant
                   (fig4's coupling traces).
    annotate : bool
        Label the ages at which significance starts and stops. With ``ci`` the age is refined by
        interpolating the relevant bound's zero crossing (what fig3 has always reported); without
        it the midpoint of the straddling grid points is used, since nothing finer is defined.
    label : str or None
        Legend label for the curve.

    Returns
    -------
    (n_grid,) bool ndarray
        The significance mask actually used, so callers can report from the same verdict they drew.
    """
    if style not in ('band', 'trace'):
        raise ValueError(f"style must be 'band' or 'trace'; got {style!r}")
    r_vals = np.asarray(r_vals, dtype=float)
    lo, hi = (None, None) if ci is None else (np.asarray(ci[0], float), np.asarray(ci[1], float))
    if style == 'band' and ci is None:
        raise ValueError("style='band' needs ci=(lo, hi); pass style='trace' for a bare curve.")
    if sig is None:
        if ci is None:
            raise ValueError('Pass sig=..., or ci=... to derive it from the band excluding zero.')
        sig = (lo > 0) | (hi < 0)
    sig = np.asarray(sig, dtype=bool)

    if style == 'band':
        ax.fill_between(age_grid, lo, hi, where=sig,  color=color, alpha=0.35, interpolate=True)
        ax.fill_between(age_grid, lo, hi, where=~sig, color=color, alpha=0.12, interpolate=True)
        ax.plot(age_grid, r_vals, color=color, lw=lw, label=label)
    else:
        ax.plot(age_grid, r_vals, color=color, lw=lw * 0.5, alpha=0.22)          # full, faded
        ax.plot(age_grid, np.where(sig, r_vals, np.nan), color=color, lw=lw, label=label)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)

    if annotate:
        for idx in np.where(np.diff(sig.astype(int)))[0]:
            entering = sig[idx + 1]                       # True = significance starts here
            positive = r_vals[idx + (1 if entering else 0)] > 0
            if ci is None:
                cross_age = 0.5 * (age_grid[idx] + age_grid[idx + 1])
            else:
                # The bound that is crossing zero: the lower one for a positive coupling, the
                # upper one for a negative coupling.
                bound  = lo if positive else hi
                a0, a1 = bound[idx], bound[idx + 1]
                frac   = 0.0 if a1 == a0 else -a0 / (a1 - a0)
                frac   = min(max(frac, 0.0), 1.0)         # a foreign sig mask can transition where
                                                          # the bound does not actually cross zero
                cross_age = age_grid[idx] + frac * (age_grid[idx + 1] - age_grid[idx])
            ha, xpos = ('right', cross_age - 0.15) if entering else ('left', cross_age + 0.15)
            y_pos, y_va = (0.99, 'top') if positive else (0.03, 'bottom')
            ax.text(xpos, y_pos, f'{cross_age:.1f} yr', ha=ha, va=y_va,
                    fontsize=fontsize - 1, color='k', transform=ax.get_xaxis_transform())
    return sig


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


# =============================================================================
# Neurobiological map x developmental-rate coupling (Fig-4E generalised)
# =============================================================================

def grid_derivatives(pred, x_grid, order):
    """kth temporal derivative of per-region GAM predictions via repeated central differences.

    Numerical differentiation (``np.gradient``, edge_order=2) applied ``order`` times along the age
    grid -- the same convention fig3 uses for its 2nd derivative, generalised to arbitrary order. The
    pyGAM object is never queried for analytic derivatives, so higher orders come purely from the
    grid; a 3rd derivative of a low-knot penalised spline is the least stable (small and edge-sensitive).

    Parameters
    ----------
    pred : (n_regions, n_grid) ndarray
        Per-region GAM smooths evaluated on ``x_grid``.
    x_grid : (n_grid,) ndarray
        Evaluation points (ages).
    order : int
        Derivative order (>= 0). ``order=0`` returns ``pred`` unchanged.

    Returns
    -------
    (n_regions, n_grid) ndarray
        The ``order``-th derivative at each grid point.
    """
    out = np.asarray(pred, dtype=float)
    for _ in range(int(order)):
        out = np.gradient(out, x_grid, axis=1, edge_order=2)
    return out


def deriv_map_spatial_null_p(deriv, map_vals, map_surrogates, method='pearson'):
    """Observed corr(map x derivative)(age) and a pointwise SA-preserving-null two-sided p per age.

    At each age column the observed statistic is the correlation across regions between
    ``map_vals`` and the regional derivative; the null at that age is the same correlation against
    every spatial-autocorrelation-preserving surrogate in ``map_surrogates``. The two-sided p is
    ``(1 + #{|r_null| >= |r_obs|}) / (n_perms + 1)`` per age -- POINTWISE and uncorrected across the
    (autocorrelated) ages, so it is descriptive "where along age is the coupling spatially specific",
    not a family-wise claim. One matmul builds the whole null.

    Regions are restricted to those finite in the map, across the whole derivative curve, and across
    all surrogates, so the returned ``r_obs`` uses exactly the regions the p-values are built on.

    Parameters
    ----------
    deriv : (n_regions, n_grid) ndarray
        Per-region temporal derivative (a single order).
    map_vals : (n_regions,) ndarray
    map_surrogates : (n_perms, n_regions) ndarray
        SA-preserving surrogates of ``map_vals`` (e.g. BrainSMASH).
    method : {'pearson', 'spearman'}
        Correlation across regions. 'spearman' rank-transforms the map, every surrogate and every
        age column ACROSS REGIONS first, then runs the same Pearson machinery -- which is what
        Spearman is, and keeps the whole null a single matmul. Ranking is done after the finite-
        region restriction, so ranks run over exactly the regions entering the statistic.

    Returns
    -------
    r_obs : (n_grid,) ndarray
        Observed correlation per age (NaN where < 3 valid regions or a flat derivative column).
    p_two : (n_grid,) ndarray
        Pointwise two-sided spatial-null p per age (NaN where r_obs is NaN).
    """
    if method not in ('pearson', 'spearman'):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")
    deriv = np.asarray(deriv, dtype=float)
    m     = np.asarray(map_vals, dtype=float)
    S     = np.asarray(map_surrogates, dtype=float)
    n_grid  = deriv.shape[1]
    n_perms = S.shape[0]
    r_obs = np.full(n_grid, np.nan)
    p_two = np.full(n_grid, np.nan)

    fin = np.isfinite(m) & np.isfinite(deriv).all(axis=1) & np.isfinite(S).all(axis=0)
    nv  = int(fin.sum())
    if nv < 3:
        return r_obs, p_two

    D  = deriv[fin]                                          # (nv, n_grid)
    Sv = S[:, fin]                                           # (n_perms, nv)
    x  = m[fin]
    if method == 'spearman':
        # Rank across regions: down the columns of D, along the rows of Sv (and the map itself).
        # Average ranks for ties, so this equals scipy's spearmanr on every pair.
        D  = sp.stats.rankdata(D, axis=0)
        Sv = sp.stats.rankdata(Sv, axis=1)
        x  = sp.stats.rankdata(x)
    xz = (x - x.mean()) / x.std()
    with np.errstate(invalid='ignore', divide='ignore'):
        Dz = (D - D.mean(axis=0, keepdims=True)) / D.std(axis=0, keepdims=True)
        Sz = (Sv - Sv.mean(axis=1, keepdims=True)) / Sv.std(axis=1, keepdims=True)

    r_obs  = (xz @ Dz) / nv                                  # (n_grid,)
    null_r = (Sz @ Dz) / nv                                  # (n_perms, n_grid)

    good = np.isfinite(r_obs)
    ge   = np.sum(np.abs(null_r[:, good]) >= np.abs(r_obs[good])[None, :], axis=0)
    p_two[good] = (1 + ge) / (n_perms + 1)
    r_obs[~good] = np.nan
    return r_obs, p_two


# =============================================================================
# Regional developmental timing vs the S-A axis (stage-free, bin-free)
# =============================================================================

def regional_change_timing(pred, x_grid, min_total=1e-8):
    """Per-region developmental timing: the magnitude-weighted mean age of the GAM rate of change.

    For each region, ``tau = sum(age * |rate|) / sum(|rate|)`` over the age grid, where the rate is
    the 1st temporal derivative of the region's GAM smooth (via :func:`grid_derivatives`). This is a
    continuous, stage-free, bin-free summary of *when* a region does its changing (weighted by how
    fast it changes): larger ``tau`` = later developer. It replaces the discrete stage-2 net-change
    map / S-A decile enrichment as the regional quantity to correlate against the S-A axis.

    Parameters
    ----------
    pred : (n_regions, n_grid) ndarray
        Per-region GAM smooths evaluated on ``x_grid``.
    x_grid : (n_grid,) ndarray
        Evaluation points (ages).
    min_total : float
        Flatness guard: regions whose total |rate| falls below this (near-flat, or unfit/all-NaN)
        return NaN, since their weighted-mean age is not meaningfully defined.

    Returns
    -------
    tau : (n_regions,) ndarray
        Weighted-mean age of change per region (in the units of ``x_grid``); NaN where undefined.
    """
    d1  = grid_derivatives(pred, x_grid, 1)          # (n_regions, n_grid)
    w   = np.abs(d1)
    tot = np.nansum(w, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        tau = np.nansum(w * np.asarray(x_grid, dtype=float)[None, :], axis=1) / tot
    tau[~np.isfinite(tau) | (tot < min_total)] = np.nan
    return tau


def _corr_stat(a, b, method):
    fn = sp.stats.spearmanr if method == 'spearman' else sp.stats.pearsonr
    return fn(a, b).statistic


def regional_stage_net_change(pred, x_grid, inflection, stage=2, annualized=False):
    """Signed net change in the fitted smooth within one developmental stage, per region.

    ``int_stage rate d(age)``, i.e. the fitted value at the end of the stage minus its value at the
    start. Stage 1 is ``age <= inflection`` and stage 2 is ``age >= inflection`` (the boundary grid
    point belongs to both, so the two stages tile the grid and their net changes sum to the total).

    With ``annualized=True`` the net change is divided by that stage's duration in years, giving the
    mean rate of change per year. The divisor is the same constant for every region, so annualisation
    rescales the map's units without changing its correlation with any other map -- but it does make
    the two stages comparable to each other, since they span different numbers of years.

    Parameters
    ----------
    pred : (n_regions, n_grid) ndarray
    x_grid : (n_grid,) ndarray
    inflection : float
        Stage boundary age (end of stage 1 / start of stage 2).
    stage : {1, 2}
        Which side of ``inflection`` to integrate over.
    annualized : bool
        If True, divide by the stage duration -> change per year.

    Returns
    -------
    (n_regions,) ndarray
    """
    if stage not in (1, 2):
        raise ValueError(f'stage must be 1 or 2, got {stage!r}')
    d1  = grid_derivatives(pred, x_grid, 1)
    x   = np.asarray(x_grid, dtype=float)
    sel = x <= inflection if stage == 1 else x >= inflection
    if sel.sum() < 2:
        return np.full(np.asarray(pred).shape[0], np.nan)
    net = np.trapezoid(d1[:, sel], x[sel], axis=1)
    if annualized:
        dur = x[sel][-1] - x[sel][0]
        net = net / dur if dur > 0 else np.full_like(net, np.nan)
    return net


# Named stage net-change metrics -> (stage, annualized), consumed by :func:`sa_metric_map`.
STAGE_NET_METRICS = {
    'early_net':        (1, False),
    'early_net_annual': (1, True),
    'late_net':         (2, False),
    'late_net_annual':  (2, True),
}


def sa_metric_map(name, pred, x_grid, inflection=None):
    """Dispatch a named regional developmental metric -> (n_regions,) map.

    ``name`` is one of the stage net-change metrics in :data:`STAGE_NET_METRICS` --
    'early_net_annual' / 'late_net_annual' (:func:`regional_stage_net_change` with
    ``annualized=True``; the annualised stage-1 / stage-2 net-change maps, the Figure-5 metrics) and
    their un-annualised 'early_net' / 'late_net' variants -- or 'timing'
    (:func:`regional_change_timing`). All except 'timing' require ``inflection``. Kept module-level
    (not a lambda table) so the bootstrap reduce that calls it stays joblib-picklable.
    """
    if name in STAGE_NET_METRICS:
        if inflection is None:
            raise ValueError(f"inflection is required for the {name!r} metric")
        stage, annualized = STAGE_NET_METRICS[name]
        return regional_stage_net_change(pred, x_grid, inflection,
                                         stage=stage, annualized=annualized)
    if name == 'timing':
        return regional_change_timing(pred, x_grid)
    raise ValueError(f"unknown S-A metric {name!r}")


def _reduce_metric_sa(preds, derivs, x_grid, *, metrics, sa_axis_vals, method, inflection):
    """Per-resample reduction for :func:`bootstrap_metric_sa_corr`: each named metric map and its
    correlation with the S-A axis. Module-level so joblib can pickle it (bind args via partial)."""
    out = {}
    for name in metrics:
        vec = sa_metric_map(name, preds, x_grid, inflection)
        fin = np.isfinite(vec) & np.isfinite(sa_axis_vals)
        r = _corr_stat(vec[fin], sa_axis_vals[fin], method) if fin.sum() >= 3 else np.nan
        out[f'r|{name}']   = float(r)
        out[f'map|{name}'] = vec
    return out


def bootstrap_metric_sa_corr(deflection, age, age_grid, sa_axis_vals, *,
                             metrics=('late_net_annual',), method='spearman', inflection=None,
                             n_splines=4, lam=0.1, demean=True, nuisance=None, n_boot=200,
                             random_state=None, n_jobs=-1, verbose=True):
    """Subject-bootstrap CI for corr(regional developmental metric, S-A axis), for one or more metrics.

    One bootstrap of the regional deflection GAMs serves every metric: each resample refits, then
    recomputes each metric map and correlates it with the S-A axis. Freeze ``lam``/``n_splines``/
    ``random_state`` to the full-sample fit to match the reference bootstrap.

    Returns
    -------
    dict
        Per metric ``name``: ``'r|name'`` (n_boot,) correlation distribution and ``'map|name'``
        (n_boot, n_regions) map distribution. ``inflection`` is required if any stage net-change
        metric (see :data:`STAGE_NET_METRICS`) is requested.
    """
    if any(m in STAGE_NET_METRICS for m in metrics) and inflection is None:
        raise ValueError('inflection is required when a stage net-change metric '
                         f'({sorted(STAGE_NET_METRICS)}) is requested')
    reduce = functools.partial(_reduce_metric_sa, metrics=tuple(metrics),
                               sa_axis_vals=np.asarray(sa_axis_vals, dtype=float),
                               method=method, inflection=inflection)
    return bootstrap_gam_fits(deflection, age, age_grid, reduce=reduce, n_boot=n_boot,
                              n_splines=n_splines, lam=lam, nuisance=nuisance, demean=demean,
                              random_state=random_state, n_jobs=n_jobs, verbose=verbose)


def sa_decile_change(values, values_boot, sa_axis_vals, n_deciles=10, ci=(2.5, 97.5)):
    """Mean regional value within each S-A decile, with a subject-bootstrap CI and a p vs 0 per decile.

    Regions are ranked by ``sa_axis_vals`` and split into ``n_deciles`` equal-count bins (decile 1 =
    lowest S-A / sensorimotor, decile ``n_deciles`` = highest / association). For each decile the
    point estimate is the mean of ``values`` over its regions; the uncertainty comes from the same
    per-region bootstrap distribution ``values_boot`` (e.g. the annualised stage-2 net-change map
    bootstrap), reduced to a decile-mean distribution across resamples. This is a subject-bootstrap
    test of "is the mean change in this decile above/below 0", not a spatial-enrichment null.

    Parameters
    ----------
    values : (n_regions,) ndarray
        Point-estimate regional map (e.g. annualised stage-2 net change).
    values_boot : (n_boot, n_regions) ndarray
        Bootstrap distribution of the same map (subject-resampled GAM refits).
    sa_axis_vals : (n_regions,) ndarray
        S-A axis values used to rank/split regions into deciles.
    n_deciles : int
    ci : (float, float)
        Lower/upper percentiles for the bootstrap CI.

    Returns
    -------
    dict of (n_deciles,) arrays plus a per-region label vector:
        'decile' (1..n_deciles), 'mean', 'ci_lo', 'ci_hi', 'p_boot' (two-sided vs 0), 'sig'
        (CI excludes 0), 'n' (regions per decile), and 'labels' (n_regions,) with each region's
        decile (0 = excluded: NaN S-A axis).
    """
    values = np.asarray(values, dtype=float)
    vb     = np.asarray(values_boot, dtype=float)
    sa     = np.asarray(sa_axis_vals, dtype=float)

    valid   = np.isfinite(sa)
    n_valid = int(valid.sum())
    order   = np.argsort(np.where(valid, sa, np.inf))       # push NaN-S-A regions to the end
    labels  = np.zeros(sa.shape[0], dtype=int)
    edges   = np.linspace(0, n_valid, n_deciles + 1).astype(int)
    for d in range(n_deciles):
        labels[order[edges[d]:edges[d + 1]]] = d + 1

    mean = np.full(n_deciles, np.nan); lo = np.full(n_deciles, np.nan)
    hi   = np.full(n_deciles, np.nan); p  = np.full(n_deciles, np.nan)
    ns   = np.zeros(n_deciles, dtype=int)
    for d in range(n_deciles):
        reg = np.where(labels == d + 1)[0]
        reg = reg[np.isfinite(values[reg])]
        ns[d] = reg.size
        if reg.size == 0:
            continue
        mean[d]      = np.nanmean(values[reg])
        boot_means   = np.nanmean(vb[:, reg], axis=1)        # (n_boot,) decile-mean per resample
        lo[d], hi[d] = np.nanpercentile(boot_means, ci)
        frac = min(np.mean(boot_means <= 0), np.mean(boot_means >= 0))
        p[d] = min(2.0 * frac, 1.0)                          # two-sided bootstrap p vs 0

    sig = (lo > 0) | (hi < 0)
    return dict(decile=np.arange(1, n_deciles + 1), mean=mean, ci_lo=lo, ci_hi=hi,
                p_boot=p, sig=sig, n=ns, labels=labels)
