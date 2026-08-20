import os
import warnings

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from snaplab_tools.plotting.utils import (make_diverging_cmap,
                                          make_sequential_cmap, register_custom_colormaps,
                                          show_colormaps, cvd_min_delta_e)
from snaplab_tools.stats import significance_stars

try:
    import adjustText
except ImportError:  # optional; scatter labels fall back to no repulsion
    adjustText = None


def setup_notebook_style():
    """Apply the project's matplotlib settings. Call once at the top of a figure notebook.

    Six notebooks each carried an identical ``plt.rcParams.update({...})`` block; this is that
    block, once.

    Deliberately *not* built on ``snaplab_tools.plotting.utils.set_plotting_params``: that
    function calls ``sns.set_theme()``, which rewrites rcParams wholesale including the colour
    cycle, and would change the appearance of any panel that relies on matplotlib's default
    colours. The submission-critical part of it -- Type-42 (editable) PDF/PS fonts, and
    ``svg.fonttype='none'`` so SVG text stays text -- is reproduced here instead.
    """
    plt.rcParams.update({
        'font.size':         8,
        'figure.dpi':        300,
        'svg.fonttype':      'none',    # keep SVG text editable in Illustrator
        'pdf.fonttype':      42,        # Type 42 = editable text, not outlines
        'ps.fonttype':       42,
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Roboto'],
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


# ── Figure export convention ────────────────────────────────────────────────────────────────────
# save_panel and its DPI constants now live in snaplab_tools.plotting.utils -- the convention is
# not project-specific. Only the scratch location below is.
#: Root of the figure scratch tree; see :func:`figure_scratch_dir`.
FIGURE_SCRATCH_BASE = '/mnt/lindenmp_nas/lindenmp/work/figure_scratch/intrinsic_neural_timescales'


def annotate_significance_brackets(ax, pairs, labels, y0=None, y_step=None,
                                   tick_frac=0.02, text_pad_frac=0.0,
                                   line_width=0.8, color='black', fontsize=7,
                                   expand_ylim=True):
    """Draw stacked significance brackets between pairs of categorical x-positions.

    Vendored into this project from snaplab_tools (the function was removed there when
    statistics/plotting were consolidated). Each bracket is a flat-topped line spanning two
    x-positions with a label (e.g. ``'***'``, ``'ns'``) centered above it. Brackets are stacked
    bottom-to-top in the order given, so pass them shortest-span-first to avoid overlap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes with a categorical x-axis (integer positions 0..n-1, as
        produced by seaborn violin/box plots).
    pairs : sequence of (float, float)
        (x1, x2) index positions to connect, one per bracket.
    labels : sequence of str
        Text centered above each bracket; same length and order as ``pairs``.
    y0 : float or None
        Data-y of the lowest bracket. Defaults to the current top ylim, which
        already clears any data/annotations drawn so far.
    y_step : float or None
        Vertical gap (data units) between successive brackets. Defaults to 8% of
        the current y-range.
    tick_frac : float, default=0.02
        Length of the downward end-ticks, as a fraction of the y-range.
    text_pad_frac : float, default=0.0
        Gap between a bracket line and its label, as a fraction of the y-range.
    line_width, color, fontsize : styling of the brackets and labels.
    expand_ylim : bool, default=True
        Raise the top ylim so the tallest bracket and its label fit.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if len(pairs) != len(labels):
        raise ValueError("`pairs` and `labels` must have the same length")

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if y0 is None:
        y0 = ymax
    if y_step is None:
        y_step = 0.08 * yr
    tick = tick_frac * yr

    top = y0
    for k, ((x1, x2), label) in enumerate(zip(pairs, labels)):
        y = y0 + k * y_step
        ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
                lw=line_width, color=color, clip_on=False)
        ax.text((x1 + x2) / 2, y + text_pad_frac * yr, label,
                ha='center', va='bottom', color=color, fontsize=fontsize,
                clip_on=False)
        top = y

    if expand_ylim:
        ax.set_ylim(ymin, max(ymax, top + 0.05 * yr))

    return ax



def plot_repeated_measures_violins(df, ax, color_vector=None, colormap='viridis',
                                   point_alpha=1, line_alpha=1, violin_alpha=0.5,
                                   point_size=1, line_width=0.5,
                                   correlation_method=None, add_colorbar=True,
                                   annotate_variance=False, fontsize=8):
    """
    Create violin plots for repeated measures data with connecting lines.
    
    Each row represents one observation measured across multiple conditions.
    Violin plots show the distribution of values for each condition, with
    individual observations connected by lines.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing repeated measures data. Each row is one observation,
        each column is one condition.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    color_vector : array-like, optional
        Vector of values (length = n_rows) for coloring observations.
        If provided, points and lines are colored by this vector.
    colormap : str, default='viridis'
        Matplotlib colormap name for coloring when color_vector is provided.
    point_alpha : float, default=1
        Transparency for individual data points (0-1).
    line_alpha : float, default=1
        Transparency for connecting lines (0-1).
    violin_alpha : float, default=0.5
        Transparency for violin plots (0-1).
    point_size : float, default=1
        Size of individual data points.
    line_width : float, default=0.5
        Width of connecting lines.
    correlation_method : str or None, default=None
        Correlation method for annotations: 'pearson', 'spearman', or None.
        If provided, correlations between color_vector and each condition are annotated.
    add_colorbar : bool, default=True
        Whether to add colorbar when color_vector is provided.
    annotate_variance : bool, default=False
        Whether to annotate standard deviation for each condition.
        If correlation_method is None, only shows SD. Otherwise shows both.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    
    # Prepare data
    df_clean = _drop_index_column(df.copy())
    n_observations = len(df_clean)
    condition_names = df_clean.columns.tolist()
    n_conditions = len(condition_names)
    
    
    # Reshape to long format for seaborn
    df_long = pd.melt(df_clean.reset_index(), 
                      id_vars=['index'], 
                      value_vars=condition_names, 
                      var_name='Condition', 
                      value_name='Value')
    df_long.rename(columns={'index': 'Subject'}, inplace=True)
    
    # Create violin plots
    sns.violinplot(data=df_long, x='Condition', y='Value', color='gray',
                   ax=ax, alpha=violin_alpha, inner=None)
    
    # Prepare colors
    if color_vector is not None:
        if len(color_vector) != n_observations:
            raise ValueError(
                f"Color vector length ({len(color_vector)}) must match "
                f"number of observations ({n_observations})"
            )
        point_colors, cmap = _create_color_mapping(color_vector, colormap)
    else:
        point_colors = ['black'] * n_observations
        cmap = None
    
    # Plot lines and points for each observation
    x_positions = np.arange(n_conditions)
    
    for i in range(n_observations):
        y_values = df_clean.iloc[i, :].values
        
        # Add jitter to x-positions
        x_jitter = np.random.normal(0, 0.05, n_conditions)
        x_jittered = x_positions + x_jitter
        
        # Plot connecting line
        line_color = point_colors[i] if color_vector is not None else 'gray'
        ax.plot(x_jittered, y_values, 
                color=line_color, alpha=line_alpha, linewidth=line_width, zorder=1)
        
        # Plot points
        ax.scatter(x_jittered, y_values, 
                   color=point_colors[i], alpha=point_alpha, 
                   s=point_size, zorder=2)
    
    # Add annotations if requested
    if color_vector is not None and correlation_method is not None:
        y_min, y_max = ax.get_ylim()
        
        for i, condition in enumerate(condition_names):
            condition_values = df_clean[condition].values
            valid_mask = ~df_clean[condition].isnull().values
            
            # Compute correlation
            if correlation_method.lower() == 'pearson':
                corr_value, _ = sp.stats.pearsonr(condition_values[valid_mask], 
                                                   color_vector[valid_mask])
                corr_symbol = 'r'
            elif correlation_method.lower() == 'spearman':
                corr_value, _ = sp.stats.spearmanr(condition_values[valid_mask], 
                                                    color_vector[valid_mask])
                corr_symbol = 'ρ'
            else:
                raise ValueError("correlation_method must be 'pearson' or 'spearman'")
            
            # Skip annotation if correlation is 1.00 (e.g. variable correlated with itself)
            if round(corr_value, 2) == 1.0:
                continue

            # Build annotation text
            annotation_text = f'{corr_symbol}={corr_value:.2f}'
            
            # Add variance if requested
            if annotate_variance:
                sd_value = np.nanstd(condition_values[valid_mask], ddof=1)
                annotation_text += f',\nσ={sd_value:.2f}'
            
            # Position annotation
            annotation_y = _calculate_annotation_height(condition_values, y_min, y_max)
            ax.text(i, annotation_y, annotation_text,
                    ha='center', va='bottom', fontsize=fontsize-2)

        # Adjust y-axis to accommodate annotations
        if annotate_variance:
            ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.25)
        else:
            ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.125)

    elif annotate_variance:
        # Only show variance, no correlation
        y_min, y_max = ax.get_ylim()

        for i, condition in enumerate(condition_names):
            condition_values = df_clean[condition].values
            valid_mask = ~df_clean[condition].isnull().values

            sd_value = np.nanstd(condition_values[valid_mask], ddof=1)
            annotation_text = f'σ={sd_value:.2f}'

            # Position annotation
            annotation_y = _calculate_annotation_height(condition_values, y_min, y_max)
            ax.text(i, annotation_y, annotation_text,
                    ha='center', va='bottom', fontsize=fontsize-2)
        
        # Adjust y-axis to accommodate annotations
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.125)
    
    # Customize plot appearance
    ax.set_xlabel('', fontsize=fontsize)
    # ax.set_ylabel('Value', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=fontsize)
    ax.grid(True, alpha=0.3, axis='y')

    # Add colorbar if requested
    if color_vector is not None and add_colorbar and cmap is not None:
        _add_colorbar(ax, cmap, min(color_vector), max(color_vector), 
                     'Color vector', shrink=0.8)
    
    return ax


def plot_subject_level_violins(df, ax, observation_indices,
                               violin_alpha=0.5, point_alpha=0.5, point_size=1,
                               line_alpha=0.25, line_width=0.5,
                               show_points=True, point_jitter=0.02,
                               show_lines=False, show_violins=True,
                               line_color_by='subject', ylim=None,
                               fit_trends=False, trend_line_width=1, trend_alpha=1,
                               hide_legend=False, sort_by_median=False,
                               add_colorbar=True, show_r2_annotation=False,
                               annotate_variance=False, fontsize=8):
    """
    Create violin plots showing subject-level variation for selected regions.
    
    Each observation index (typically a brain region) gets its own violin per condition.
    Data is expected to have multiple subjects, each with measurements across
    all observation indices (e.g., 400 brain regions).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'subject' column and condition columns.
        Expected structure: N observations per subject, with subjects stacked.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    observation_indices : list or array-like
        List of observation indices to include (e.g., region indices 0-399).
    violin_alpha : float, default=0.5
        Transparency for violin plots (0-1).
    point_alpha : float, default=0.5
        Transparency for individual subject points (0-1).
    point_size : float, default=1
        Size of individual subject points.
    line_alpha : float, default=0.25
        Transparency for connecting lines (0-1).
    line_width : float, default=0.5
        Width of connecting lines.
    show_points : bool, default=True
        Whether to show individual subject points.
    point_jitter : float, default=0.02
        Amount of jitter for point positions.
    show_lines : bool, default=False
        Whether to show connecting lines across conditions for each subject.
    show_violins : bool, default=True
        Whether to show violin plots (distributions).
    line_color_by : str, default='subject'
        How to color connecting lines: 'subject' (gradient across subjects) 
        or 'region' (same color as violins for each region).
    ylim : tuple or None, default=None
        Y-axis limits as (ymin, ymax). If None, uses automatic limits.
    fit_trends : bool, default=False
        Whether to fit polynomial trends to distribution means within each condition.
    trend_line_width : float, default=1
        Width of trend lines.
    trend_alpha : float, default=1
        Transparency of trend lines (0-1).
    hide_legend : bool, default=False
        Whether to hide the legend.
    sort_by_median : bool, default=False
        Whether to sort x-axis (conditions) by median value in descending order.
    add_colorbar : bool, default=True
        Whether to add colorbar when line_color_by='subject'.
    show_r2_annotation : bool, default=False
        Whether to show R² values above each condition (only when fit_trends=True).
    annotate_variance : bool, default=False
        Whether to annotate standard deviation for each condition.
        If show_r2_annotation is False, only shows SD. Otherwise shows both.

        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    trend_results : dict, optional
        Dictionary containing trend fitting results if fit_trends=True.
        
    Notes
    -----
    NaN values are handled as follows:
    - Median calculations use np.nanmedian
    - Mean calculations (for trends) use np.nanmean
    - NaN points are silently excluded from scatter plots
    - NaN values are skipped in connecting lines
    - Violins automatically handle NaN via seaborn
    """
    
    # Prepare data
    df_clean = _drop_index_column(df.copy())
    
    # Ensure observation_indices is a list
    if not isinstance(observation_indices, (list, np.ndarray)):
        observation_indices = [observation_indices]
    
    # Identify subjects and conditions
    subject_ids = df_clean['subject'].unique()
    n_subjects = len(subject_ids)
    condition_names = [col for col in df_clean.columns if col != 'subject']
    
    
    # Create observation index within each subject
    df_clean['obs_index'] = df_clean.groupby('subject').cumcount()
    
    # Filter for selected observations
    df_filtered = df_clean[df_clean['obs_index'].isin(observation_indices)].copy()
    
    
    # Sort conditions by median if requested
    if sort_by_median:
        condition_medians = {
            cond: np.nanmedian(df_filtered[cond].values) 
            for cond in condition_names
        }
        condition_names = sorted(condition_names, 
                                key=lambda x: condition_medians[x], 
                                reverse=True)
        
    
    # Reshape to long format
    df_long = pd.melt(df_filtered, 
                      id_vars=['subject', 'obs_index'], 
                      value_vars=condition_names,
                      var_name='Condition', 
                      value_name='Value')
    
    # Create region labels
    df_long['Region'] = 'Region_' + df_long['obs_index'].astype(str)
    
    # Convert Condition to categorical to maintain sort order
    df_long['Condition'] = pd.Categorical(df_long['Condition'], 
                                          categories=condition_names, 
                                          ordered=True)
    
    # Prepare colors for observations (regions)
    n_observations = len(observation_indices)
    if n_observations == 1:
        observation_colors = ['#FFA500']  # Soft orange
    else:
        sydnor_sa_cmap = register_custom_colormaps()['snap_sydnor_sa']
        observation_colors = [
            sydnor_sa_cmap(i / (n_observations - 1)) 
            for i in range(n_observations)
        ]
    
    # Prepare colors for subjects (for line coloring)
    if line_color_by == 'subject':
        subject_cmap = plt.get_cmap('viridis')
        subject_colors = {
            subj: subject_cmap(i / max(1, n_subjects - 1)) 
            for i, subj in enumerate(subject_ids)
        }
    else:
        subject_colors = None
    
    # Create violin plots
    if show_violins:
        sns.violinplot(data=df_long, x='Condition', y='Value', hue='Region',
                       ax=ax, alpha=violin_alpha, inner=None, 
                       palette=observation_colors)
    
    # Fit trends if requested
    trend_results = {}
    if fit_trends and n_observations >= 3:
        
        trend_color = "#3B3B3B"  # Soft grey
        n_conditions = len(condition_names)
        violin_width = 0.8
        violin_spacing = violin_width / n_observations
        
        # Calculate means for each condition and region
        condition_means = {}
        for cond in condition_names:
            region_means = []
            for obs_idx in observation_indices:
                mask = ((df_long['Condition'] == cond) & 
                       (df_long['obs_index'] == obs_idx))
                region_data = df_long[mask]['Value']
                region_means.append(np.nanmean(region_data))
            condition_means[cond] = np.array(region_means)
        
        # Fit trends for each condition
        for cond_idx, cond in enumerate(condition_names):
            means = condition_means[cond]
            x_data = np.arange(len(observation_indices))
            
            # Fit polynomial trends (linear and quadratic)
            degrees = [1, 2]
            degree_names = ['Linear', 'Quadratic']
            r2_scores = []
            fitted_polys = []
            
            for degree in degrees:
                try:
                    coeffs = np.polyfit(x_data, means, degree)
                    poly_func = np.poly1d(coeffs)
                    
                    # Calculate R²
                    y_pred = poly_func(x_data)
                    ss_res = np.sum((means - y_pred) ** 2)
                    ss_tot = np.sum((means - np.nanmean(means)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    r2_scores.append(r2)
                    fitted_polys.append(poly_func)
                    
                except np.linalg.LinAlgError:
                    r2_scores.append(-np.inf)
                    fitted_polys.append(None)
            
            # Select best fitting trend
            best_idx = np.argmax(r2_scores)
            best_degree = degrees[best_idx]
            best_r2 = r2_scores[best_idx]
            best_poly = fitted_polys[best_idx]
            
            # Store results
            trend_results[cond] = {
                'means': means,
                'x_data': x_data,
                'r2_scores': dict(zip(degree_names, r2_scores)),
                'best_degree': best_degree,
                'best_r2': best_r2,
                'best_poly': best_poly,
                'all_polys': dict(zip(degree_names, fitted_polys))
            }
            
            # Plot best fitting trend
            if best_poly is not None and best_r2 > 0:
                # Create smooth x values for trend line
                x_smooth = np.linspace(0, len(observation_indices) - 1, 100)
                y_smooth = best_poly(x_smooth)
                
                # Convert to plot positions
                x_plot_smooth = []
                for x_val in x_smooth:
                    if n_observations == 1:
                        x_plot_pos = cond_idx
                    else:
                        obs_idx_float = x_val
                        x_offset = (obs_idx_float - (n_observations - 1) / 2) * violin_spacing
                        x_plot_pos = cond_idx + x_offset
                    x_plot_smooth.append(x_plot_pos)
                
                # Plot trend line
                ax.plot(x_plot_smooth, y_smooth, 
                       color=trend_color, linewidth=trend_line_width, 
                       alpha=trend_alpha, linestyle='-', zorder=10)
            
    
    # Add connecting lines if requested
    if show_lines:
        n_conditions = len(condition_names)
        violin_width = 0.8
        violin_spacing = violin_width / n_observations
        
        for subject in subject_ids:
            for obs_idx, obs_val in enumerate(observation_indices):
                # Get data for this subject and observation
                mask = ((df_filtered['subject'] == subject) & 
                       (df_filtered['obs_index'] == obs_val))
                subject_obs_data = df_filtered[mask]
                
                if len(subject_obs_data) > 0:
                    # Get y-values and x-positions across conditions
                    y_values = []
                    x_positions = []
                    
                    for cond_idx, cond in enumerate(condition_names):
                        if cond in subject_obs_data.columns:
                            y_val = subject_obs_data[cond].iloc[0]
                            
                            # Skip NaN values
                            if np.isnan(y_val):
                                continue
                            
                            y_values.append(y_val)
                            
                            # Calculate x position for this observation's violin
                            if n_observations == 1:
                                x_pos = cond_idx
                            else:
                                x_offset = (obs_idx - (n_observations - 1) / 2) * violin_spacing
                                x_pos = cond_idx + x_offset
                            x_positions.append(x_pos)
                    
                    # Draw connecting line
                    if len(y_values) > 1:
                        if line_color_by == 'subject':
                            line_color = subject_colors[subject]
                        else:  # line_color_by == 'region'
                            line_color = observation_colors[obs_idx]
                            
                        ax.plot(x_positions, y_values, 
                               color=line_color, alpha=line_alpha, 
                               linewidth=line_width, zorder=1)
    
    # Add individual subject points if requested
    if show_points:
        violin_width = 0.8
        violin_spacing = violin_width / n_observations
        
        for cond_idx, cond in enumerate(condition_names):
            for obs_idx, obs_val in enumerate(observation_indices):
                # Filter data for this condition and observation
                mask = ((df_long['Condition'] == cond) & 
                       (df_long['obs_index'] == obs_val))
                data_subset = df_long[mask].copy()
                
                # Remove NaN values
                data_subset = data_subset[~data_subset['Value'].isna()]
                
                if len(data_subset) > 0:
                    # Calculate x position
                    if n_observations == 1:
                        x_pos = cond_idx
                    else:
                        x_offset = (obs_idx - (n_observations - 1) / 2) * violin_spacing
                        x_pos = cond_idx + x_offset
                    
                    # Add jitter
                    x_jittered = x_pos + np.random.normal(0, point_jitter, len(data_subset))
                    
                    # Plot points
                    ax.scatter(x_jittered, data_subset['Value'], 
                             color=observation_colors[obs_idx], 
                             alpha=point_alpha, s=point_size, zorder=3)
    
    # Add annotations if requested
    if show_r2_annotation and fit_trends and trend_results:
        y_min, y_max = ax.get_ylim()
        
        for cond_idx, cond in enumerate(condition_names):
            if cond in trend_results:
                r2_value = trend_results[cond]['best_r2']
                
                # Get condition data for annotation positioning and variance calculation
                condition_data = df_long[df_long['Condition'] == cond]['Value'].values
                annotation_y = _calculate_annotation_height(condition_data, 
                                                           y_min, y_max)
                
                # Build annotation text
                annotation_text = f"R²={r2_value:.2f}"
                
                # Add variance if requested
                if annotate_variance:
                    sd_value = np.nanstd(condition_data, ddof=1)
                    annotation_text += f",\nσ={sd_value:.2f}"
                
                # Place annotation
                ax.text(cond_idx, annotation_y, annotation_text,
                       ha='center', va='bottom', fontsize=fontsize-2)

        # Adjust y-axis to accommodate annotations
        if annotate_variance:
            ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.3)
        else:
            ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)

    elif annotate_variance:
        # Only show variance, no R²
        y_min, y_max = ax.get_ylim()

        for cond_idx, cond in enumerate(condition_names):
            # Get condition data
            condition_data = df_long[df_long['Condition'] == cond]['Value'].values

            sd_value = np.nanstd(condition_data, ddof=1)
            annotation_text = f"σ={sd_value:.2f}"

            # Position annotation
            annotation_y = _calculate_annotation_height(condition_data, y_min, y_max)
            ax.text(cond_idx, annotation_y, annotation_text,
                   ha='center', va='bottom', fontsize=fontsize-2)
        
        # Adjust y-axis to accommodate annotations
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
    
    # Customize plot appearance
    ax.set_xlabel('', fontsize=fontsize)
    # ax.set_ylabel('Value', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xticks(range(len(condition_names)))
    ax.set_xticklabels(condition_names, rotation=0, ha='center', fontsize=fontsize)
    ax.grid(True, alpha=0.3, axis='y')

    # Create legend
    legend_handles = []
    legend_labels = []
    
    if n_observations > 1:
        if show_violins:
            handles, labels = ax.get_legend_handles_labels()
            region_handles = [h for h, l in zip(handles, labels) if 'Region_' in l]
            region_labels = [f'Index {idx+1}' for idx in observation_indices]
            legend_handles.extend(region_handles)
            legend_labels.extend(region_labels)
        else:
            region_elements = [
                mpatches.Patch(facecolor=observation_colors[i], label=f'Index {idx}')
                for i, idx in enumerate(observation_indices)
            ]
            legend_handles.extend(region_elements)
            legend_labels.extend([f'Index {idx+1}' for idx in observation_indices])
    
    # Set legend
    if legend_handles and not hide_legend:
        ax.legend(handles=legend_handles, labels=legend_labels,
                 title='Group Index', loc='upper right',
                 bbox_to_anchor=(1, 1), fontsize=fontsize)
    
    # Add colorbar for subject coloring if applicable
    if show_lines and line_color_by == 'subject' and add_colorbar:
        _add_colorbar(ax, subject_cmap, 0, n_subjects - 1, 
                     'Subject Index', shrink=0.8)
        
        # Set colorbar ticks
        cbar = ax.figure.axes[-1]  # Get the colorbar axes
        if n_subjects <= 10:
            cbar.set_ticks(range(n_subjects))
        else:
            tick_indices = np.linspace(0, n_subjects - 1, 5, dtype=int)
            cbar.set_ticks(tick_indices)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if fit_trends:
        return ax, trend_results
    else:
        return ax


# =============================================================================
# Task-regression worked example (supplementary)
# =============================================================================

def _task_block_spans(events, tr, n_timepoints=None):
    """Merge a BIDS events table into contiguous (start, end) spans in timepoint units.

    For shading task blocks behind a plotted time series. Overlapping or abutting trials are
    merged, so a blocked design draws as a handful of wide bands rather than one band per trial,
    and back-to-back trials of different conditions become a single block.

    Kept in this project for now rather than in ``snaplab_tools``: general enough to move up
    later, but it has only one caller so far.

    Parameters
    ----------
    events : pandas.DataFrame
        BIDS events table with ``onset`` and ``duration`` columns, in seconds.
    tr : float
        Repetition time in seconds; converts seconds to timepoints.
    n_timepoints : int or None
        Clip spans to ``[0, n_timepoints - 1]``. ``None`` leaves them unclipped.

    Returns
    -------
    list of (float, float)
        Merged spans in timepoint units, ascending. Empty if `events` is None, empty, or lacks
        the required columns -- shading is decoration, so a malformed table returns nothing
        rather than raising in the middle of drawing a figure.
    """
    if events is None or not len(events):
        return []
    try:
        onsets = np.asarray(events['onset'], dtype=float)
        durations = np.asarray(events['duration'], dtype=float)
    except (KeyError, TypeError, ValueError):
        return []

    merged = []
    for start, end in sorted(zip(onsets / tr, (onsets + durations) / tr)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    if n_timepoints is None:
        return [(float(s), float(e)) for s, e in merged]
    return [(float(max(0.0, s)), float(min(n_timepoints - 1, e))) for s, e in merged if e > 0]


def style_ax(ax, xlabel, ylabel, title, demean=False, fontsize=8,
             title_pad=None, tick_pad=0.25):
    """Label and de-clutter a line panel: labels, title, tick size, top/right spines off.

    The single implementation for the project. Defaults reproduce what ``src.gams.style_ax``
    did, so existing callers are unaffected; pass ``title_pad`` / ``tick_pad`` for tighter
    panels.

    Parameters
    ----------
    ax : matplotlib Axes
    xlabel, ylabel, title : str or None
        Axis labels and title.
    demean : bool
        Draw a dashed zero reference line, for panels showing deviations from a mean.
    fontsize : int
        Base font size for labels, title and ticks.
    title_pad : float or None
        Title padding in points. None uses the rcParam default.
    tick_pad : float
        Tick label padding in points.
    """
    if demean:
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=1)
    ax.set_title(title, fontsize=fontsize, pad=title_pad)
    ax.tick_params(labelsize=fontsize, pad=tick_pad)
    ax.spines[['top', 'right']].set_visible(False)


def plot_task_regression_example(raw_data, residual_data, design_matrix,
                                 acf_raw, acf_resid, acf_design,
                                 int_raw, int_resid, tr,
                                 task_events=None, brain_map=None, brain_map_label='S-A axis rank',
                                 example_region=None, max_lag_seconds=60.0,
                                 heatmap_cmap='snap_blue_red', map_cmap='snap_sydnor_sa',
                                 heatmap_pct=98, normalize_rows=True,
                                 figsize=(7.25, 7.5), fontsize=8):
    """Draw the single-subject worked example of what task regression does to the ACF.

    Five panels sharing one argument: the task design is itself autocorrelated (D), regions
    that track it inherit that autocorrelation (A vs B, C), and because the INT is read off
    the ACF's decay the inherited structure inflates the INT directly (E).

    Panels A, B and C share the scan's time axis with the design-matrix strip above them, so
    a block in the design can be followed down into the data, into the residual, and into a
    single region's timeseries.

    This function only draws -- every estimate is computed by the caller, so the notebook and
    the figure cannot disagree about what was fit.

    Parameters
    ----------
    raw_data : (n_regions, n_timepoints) ndarray
        Timeseries before task regression (detrended + nuisance-regressed).
    residual_data : (n_regions, n_timepoints) ndarray
        Residuals of the task GLM, from ``fit_glm``. May contain all-NaN rows for regions the
        GLM excluded; those regions are dropped from every panel.
    design_matrix : pd.DataFrame
        The nilearn design matrix. ``drift_*`` and ``constant`` columns are dropped before
        drawing, so the strip shows the task regressors only.
    acf_raw, acf_resid : (n_regions, n_lags) ndarray
        ACFs of ``raw_data`` and ``residual_data``.
    acf_design : (n_lags,) or (1, n_lags) ndarray
        ACF of the flattened, HRF-convolved task regressor -- the design's own autocorrelation.
    int_raw, int_resid : (n_regions,) ndarray
        INT per region before and after regression, in seconds.
    tr : float
        Repetition time in seconds.
    task_events : pd.DataFrame or None
        Events table (onset/duration). Used only to shade blocks behind panel C.
    brain_map : (n_regions,) ndarray or None
        Map used to order the heatmap rows and to colour panel E -- the S-A axis in practice.
        Regions are left in acquisition order if None.
    brain_map_label : str
        Colourbar label for ``brain_map``.
    example_region : int or None
        Region index for panel C, in the *original* (unsorted) region order. Defaults to the
        region whose variance is most reduced by the GLM, i.e. the one that tracks the design
        most closely.
    max_lag_seconds : float
        Right-hand limit of the ACF panel.
    heatmap_cmap : str
        Diverging colormap for panels A and B. Registered by ``register_custom_colormaps``.
    map_cmap : str
        Colormap for ``brain_map`` in panel E.
    heatmap_pct : float
        Percentile of the displayed values setting the symmetric colour limits shared by B and C.
    normalize_rows : bool
        Scale every region in panels B and C by *its own raw* mean and SD, so that a few
        high-amplitude parcels cannot set the colour limits for all 400. Both panels use the
        raw scaling, so the variance that regression removes still shows as fading. Display
        only -- no panel's estimates are affected.
    figsize : tuple
    fontsize : int

    Returns
    -------
    fig : matplotlib Figure
    axes : dict
        Keys ``'A'``..``'F'`` -- matching the letters drawn on the panels: A design strip,
        B raw heatmap, C residual heatmap, D example region, E ACFs, F INT scatter -- plus
        the colourbar axes ``'cbar_data'`` and ``'cbar_map'``.
    stats : dict
        Numbers the caption needs: ``example_region``, ``example_var_explained``,
        ``int_raw_mean``, ``int_resid_mean``, ``int_median_change``, ``frac_regions_reduced``,
        ``zero_crossing_raw``, ``zero_crossing_resid``, ``zero_crossing_design``, ``n_valid``.
    """
    register_custom_colormaps()

    raw_data = np.asarray(raw_data, dtype=float)
    residual_data = np.asarray(residual_data, dtype=float)
    acf_raw = np.atleast_2d(np.asarray(acf_raw, dtype=float))
    acf_resid = np.atleast_2d(np.asarray(acf_resid, dtype=float))
    acf_design = np.squeeze(np.asarray(acf_design, dtype=float))
    int_raw = np.asarray(int_raw, dtype=float)
    int_resid = np.asarray(int_resid, dtype=float)

    n_regions, n_timepoints = raw_data.shape

    # fit_glm returns all-NaN rows for regions it excluded (NaN or zero-variance input).
    # Carrying them through would poison the region means in D and open gaps in the heatmaps,
    # so drop them everywhere and report how many survived.
    valid = ~(np.isnan(raw_data).any(axis=1) | np.isnan(residual_data).any(axis=1))
    if not valid.any():
        raise ValueError('no regions survived the task GLM -- nothing to plot')

    # ---- region order for the heatmaps -------------------------------------------------
    if brain_map is not None:
        brain_map = np.asarray(brain_map, dtype=float)
        if brain_map.shape[0] != n_regions:
            raise ValueError(f'brain_map has {brain_map.shape[0]} regions, data has {n_regions}')
        order = np.argsort(np.where(valid, brain_map, np.inf), kind='stable')[:valid.sum()]
    else:
        order = np.flatnonzero(valid)

    # ---- example region ----------------------------------------------------------------
    # Variance explained by the design, per region. The maximum is the region that most
    # closely tracks the task, which is the one worth showing as a timeseries. Computed on
    # valid regions only -- an all-NaN row has no variance to take a ratio of.
    var_explained = np.full(n_regions, np.nan)
    var_explained[valid] = 1.0 - (residual_data[valid].var(axis=1) / raw_data[valid].var(axis=1))
    if example_region is None:
        example_region = int(np.nanargmax(var_explained))
    example_region = int(example_region)

    # ---- shared time axis --------------------------------------------------------------
    time_seconds = np.arange(n_timepoints) * tr
    time_minutes = time_seconds / 60.0

    # ---- figure scaffold ---------------------------------------------------------------
    # Row 4 is an empty spacer: rows 0-3 share the time axis and want to sit tight against
    # each other, but the bottom row needs clearance for panel D's x-label.
    # The design strip grows with the number of regressors so its row labels never collide --
    # HCP-WM has eight, which does not fit in the height a two-regressor design needs.
    dm_cols = [c for c in design_matrix.columns
               if not str(c).startswith('drift_') and c != 'constant']
    design_height = np.clip(0.11 * len(dm_cols), 0.35, 1.10)

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(
        6, 2, figure=fig,
        height_ratios=[design_height, 1.15, 1.15, 0.75, 0.32, 1.30],
        width_ratios=[1, 0.028],
        hspace=0.22, wspace=0.02,
    )

    ax_design = fig.add_subplot(outer[0, 0])
    ax_raw = fig.add_subplot(outer[1, 0], sharex=ax_design)
    ax_resid = fig.add_subplot(outer[2, 0], sharex=ax_design)
    ax_series = fig.add_subplot(outer[3, 0], sharex=ax_design)
    ax_cbar_data = fig.add_subplot(outer[1:3, 1])

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5, 0], wspace=0.34)
    ax_acf = fig.add_subplot(bottom[0, 0])
    ax_int = fig.add_subplot(bottom[0, 1])
    ax_cbar_map = fig.add_subplot(outer[5, 1])

    def _panel_label(ax, letter):
        ax.text(-0.055, 1.0, letter, transform=ax.transAxes, fontsize=fontsize + 2,
                fontweight='bold', ha='right', va='bottom')

    # ---- design matrix strip -----------------------------------------------------------
    task_cols = dm_cols
    dm_task = design_matrix[task_cols].values.T

    # vmin=0 rather than the data minimum: the HRF undershoot dips slightly negative, which
    # would otherwise put the inter-block baseline at mid-grey and make the whole strip muddy.
    # Clipping it to white keeps 'no task' reading as blank.
    ax_design.imshow(dm_task, cmap='Greys', aspect='auto', interpolation='nearest',
                     vmin=0, vmax=np.nanmax(dm_task),
                     extent=(-0.5, n_timepoints - 0.5, len(task_cols) - 0.5, -0.5))
    ax_design.set_yticks(np.arange(len(task_cols)))
    ax_design.set_yticklabels([str(c)[:14] for c in task_cols], fontsize=max(4, fontsize - 3))
    ax_design.set_ylabel('Task\nregressors', fontsize=fontsize, labelpad=1)
    ax_design.tick_params(axis='x', labelbottom=False, length=0)
    ax_design.tick_params(axis='y', length=0, pad=1)
    ax_design.spines[:].set_visible(False)
    _panel_label(ax_design, 'A')
    ax_design.set_title('HRF-convolved task design', fontsize=fontsize, pad=3)

    # ---- A / B heatmaps ----------------------------------------------------------------
    # Parcels differ several-fold in raw signal amplitude, so on a shared absolute colour
    # scale a handful of high-amplitude regions set the limits and everything else washes
    # out. Scale each region by *its own raw* mean and SD and apply that same scaling to the
    # residual: regions become comparable, while the drop in variance that regression causes
    # is still visible as fading rather than being normalised away (which is what
    # standardising each panel separately would do).
    if normalize_rows:
        mu = raw_data.mean(axis=1, keepdims=True)
        sd = raw_data.std(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            raw_disp = (raw_data - mu) / sd
            resid_disp = (residual_data - mu) / sd
        cbar_label = 'Signal (SD of raw)'
    else:
        raw_disp, resid_disp = raw_data, residual_data
        cbar_label = 'Signal (a.u.)'

    vmax = np.nanpercentile(np.abs(raw_disp[valid]), heatmap_pct)
    heat_kw = dict(cmap=heatmap_cmap, vmin=-vmax, vmax=vmax, aspect='auto',
                   interpolation='nearest',
                   extent=(-0.5, n_timepoints - 0.5, len(order) - 0.5, -0.5))
    im = ax_raw.imshow(raw_disp[order], **heat_kw)
    ax_resid.imshow(resid_disp[order], **heat_kw)

    ylabel = 'Region' if brain_map is None else f'Region\n({brain_map_label} order)'
    for ax, title in ((ax_raw, 'Before task regression'),
                      (ax_resid, 'After task regression')):
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=1)
        ax.set_title(title, fontsize=fontsize, pad=3)
        ax.set_yticks([0, len(order) - 1])
        ax.set_yticklabels(['1', str(len(order))], fontsize=fontsize)
        ax.tick_params(axis='x', labelbottom=False, length=0)
        ax.tick_params(axis='y', labelsize=fontsize, pad=1)
        ax.spines[:].set_visible(False)
    _panel_label(ax_raw, 'B')
    _panel_label(ax_resid, 'C')

    cbar = fig.colorbar(im, cax=ax_cbar_data)
    cbar.set_label(cbar_label, fontsize=fontsize, labelpad=2)
    cbar.ax.tick_params(labelsize=max(5, fontsize - 2), pad=1)
    cbar.outline.set_visible(False)

    # ---- C: single region --------------------------------------------------------------
    for start, end in _task_block_spans(task_events, tr, n_timepoints):
        ax_series.axvspan(start, end, color='0.85', lw=0, zorder=0)
    ax_series.plot(np.arange(n_timepoints), raw_data[example_region], lw=0.6,
                   color='#B2182B', label='before', zorder=2)
    ax_series.plot(np.arange(n_timepoints), residual_data[example_region], lw=0.6,
                   color='#2166AC', label='after', zorder=3)
    ax_series.set_ylabel('Signal (a.u.)', fontsize=fontsize, labelpad=1)
    ax_series.set_xlabel('Time (minutes)', fontsize=fontsize, labelpad=1)
    # Locate the region by its position on the brain map rather than by a marker on the
    # heatmaps: 400 rows in a couple of inches puts a single row below one pixel, so a
    # positional arrow cannot actually point at it.
    rank_note = ('' if brain_map is None
                 else f' [{brain_map_label} {brain_map[example_region]:.0f}/{n_regions}]')
    ax_series.set_title(
        f'Region {example_region + 1}{rank_note}: most task-driven '
        f'({100 * var_explained[example_region]:.0f}% of variance explained by the design)',
        fontsize=fontsize, pad=3)
    # Tick on whole (or half) minutes rather than on evenly spaced frames, which would label
    # the axis with arbitrary decimals.
    step = 1.0 if time_minutes[-1] >= 3 else 0.5
    tick_minutes = np.arange(0, time_minutes[-1] + 1e-9, step)
    ax_series.set_xticks(tick_minutes * 60.0 / tr)
    ax_series.set_xticklabels([f'{m:g}' for m in tick_minutes], fontsize=fontsize)
    ax_series.set_xlim(-0.5, n_timepoints - 0.5)
    ax_series.tick_params(labelsize=fontsize, pad=1)
    ax_series.spines[['top', 'right']].set_visible(False)
    # Headroom plus an opaque frame: the block shading spans the full panel height, so a
    # frameless legend would sit directly on top of the traces.
    ax_series.margins(y=0.22)
    leg = ax_series.legend(fontsize=max(5, fontsize - 2), loc='upper right', ncol=2,
                           handlelength=1.2, columnspacing=0.8, borderaxespad=0.1,
                           frameon=True, framealpha=0.9, edgecolor='none')
    leg.get_frame().set_facecolor('white')
    _panel_label(ax_series, 'D')

    # ---- D: ACFs -----------------------------------------------------------------------
    max_lag = int(min(acf_raw.shape[1], acf_resid.shape[1], np.ceil(max_lag_seconds / tr) + 1))
    lag_seconds = np.arange(max_lag) * tr

    def _band(acf, color, label):
        mean = np.nanmean(acf[valid, :max_lag], axis=0)
        sd = np.nanstd(acf[valid, :max_lag], axis=0)
        ax_acf.fill_between(lag_seconds, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)
        ax_acf.plot(lag_seconds, mean, color=color, lw=1.0, label=label, zorder=3)
        return mean

    ax_acf.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4, zorder=1)
    mean_raw = _band(acf_raw, '#B2182B', 'before')
    mean_resid = _band(acf_resid, '#2166AC', 'after')

    design_curve = acf_design[:max_lag]
    ax_acf.plot(lag_seconds[:len(design_curve)], design_curve, color='0.35', lw=1.0, ls=(0, (3, 1.5)),
                label='task design', zorder=2)

    def _first_crossing(curve):
        below = np.flatnonzero(curve < 0)
        return float(below[0] * tr) if below.size else np.nan

    zc_raw, zc_resid = _first_crossing(mean_raw), _first_crossing(mean_resid)
    zc_design = _first_crossing(design_curve)
    for zc, color in ((zc_raw, '#B2182B'), (zc_resid, '#2166AC')):
        if np.isfinite(zc):
            ax_acf.plot([zc], [0], marker='v', color=color, markersize=3, clip_on=False, zorder=4)

    ax_acf.set_xlim(0, lag_seconds[-1])
    style_ax(ax_acf, 'Lag (seconds)', 'Autocorrelation',
             'ACF across regions (mean $\\pm$ SD)', fontsize=fontsize,
             title_pad=3, tick_pad=1)
    ax_acf.legend(fontsize=max(5, fontsize - 2), frameon=False, loc='upper right',
                  handlelength=1.4, borderaxespad=0.1)
    _panel_label(ax_acf, 'E')

    # ---- E: INT before vs after --------------------------------------------------------
    both = valid & np.isfinite(int_raw) & np.isfinite(int_resid)
    colours = brain_map[both] if brain_map is not None else None
    sc = ax_int.scatter(int_raw[both], int_resid[both], c=colours, cmap=map_cmap,
                        s=3, lw=0, alpha=0.85, zorder=3)

    lo = float(np.nanmin([int_raw[both].min(), int_resid[both].min()]))
    hi = float(np.nanmax([int_raw[both].max(), int_resid[both].max()]))
    pad = 0.05 * (hi - lo)
    ax_int.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='k', lw=0.5, ls='--',
                alpha=0.5, zorder=2)
    ax_int.set_xlim(lo - pad, hi + pad)
    ax_int.set_ylim(lo - pad, hi + pad)
    ax_int.set_aspect('equal', adjustable='box')

    delta = int_resid[both] - int_raw[both]
    frac_reduced = float(np.mean(delta < 0))
    ax_int.text(0.04, 0.96,
                f'median $\\Delta$ = {np.median(delta):+.2f} s\n{100 * frac_reduced:.0f}% of regions reduced',
                transform=ax_int.transAxes, fontsize=max(5, fontsize - 2), va='top', ha='left')
    style_ax(ax_int, 'INT before (s)', 'INT after (s)', 'INT per region',
             fontsize=fontsize, title_pad=3, tick_pad=1)
    _panel_label(ax_int, 'F')

    if brain_map is not None:
        cbar_map = fig.colorbar(sc, cax=ax_cbar_map)
        cbar_map.set_label(brain_map_label, fontsize=fontsize, labelpad=2)
        cbar_map.ax.tick_params(labelsize=max(5, fontsize - 2), pad=1)
        cbar_map.outline.set_visible(False)
    else:
        ax_cbar_map.set_visible(False)

    # Keys match the letters drawn on the panels, so a caption edit and a code edit
    # can refer to the same thing.
    axes = {'A': ax_design, 'B': ax_raw, 'C': ax_resid, 'D': ax_series,
            'E': ax_acf, 'F': ax_int,
            'cbar_data': ax_cbar_data, 'cbar_map': ax_cbar_map}
    stats = {
        'example_region': example_region,
        'example_var_explained': float(var_explained[example_region]),
        'int_raw_mean': float(np.nanmean(int_raw[both])),
        'int_resid_mean': float(np.nanmean(int_resid[both])),
        'int_median_change': float(np.median(delta)),
        'frac_regions_reduced': frac_reduced,
        'zero_crossing_raw': zc_raw,
        'zero_crossing_resid': zc_resid,
        'zero_crossing_design': zc_design,
        'n_valid': int(valid.sum()),
    }
    return fig, axes, stats


# ==================== Support Utility Functions ====================
def _drop_index_column(df):
    """Remove first column if it's just an index."""
    if df.columns[0] in ['', 'Unnamed: 0'] or df.iloc[:, 0].dtype in ['int64', 'int32']:
        return df.iloc[:, 1:]
    return df


def _create_color_mapping(color_vector, colormap):
    """Create color mapping from color vector using specified colormap."""
    cmap = plt.get_cmap(colormap)
    unique_values = np.unique(color_vector)
    n_unique = len(unique_values)
    
    if n_unique == 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (n_unique - 1)) for i in range(n_unique)]
    
    color_dict = dict(zip(unique_values, colors))
    point_colors = [color_dict[val] for val in color_vector]
    return point_colors, cmap


def _add_colorbar(ax, cmap, vmin, vmax, label, shrink=0.8):
    """Add colorbar to axes."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
    cbar.set_label(label, fontsize=10)
    return cbar


def _calculate_annotation_height(data_values, y_min, y_max, offset_fraction=0.05):
    """Calculate height for annotations above data."""
    y_range = y_max - y_min
    data_max = np.nanmax(data_values)
    return data_max + y_range * offset_fraction


def figure_scratch_dir(dataset=None, base=FIGURE_SCRATCH_BASE):
    """Resolve (and create) the scratch directory a notebook writes its panels into.

    One definition of where panels land, so the notebooks stop rebuilding the same path from an
    f-string apiece. Pass ``dataset`` for a per-dataset subdirectory; omit it for figures that
    span datasets (fig1, the schematic).

    Parameters
    ----------
    dataset : str or None
        Dataset subdirectory, e.g. 'HCP-YA'. None writes to ``base`` itself.
    base : str
        Root scratch directory.

    Returns
    -------
    str
        The created directory.
    """
    out = os.path.join(base, dataset) if dataset else base
    os.makedirs(out, exist_ok=True)
    return out


