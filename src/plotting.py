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

from snaplab_tools.plotting.utils import create_sydnor_sa_colormap, determine_significance

try:
    import adjustText
except ImportError:  # optional; scatter labels fall back to no repulsion
    adjustText = None


def plot_autocorrelation_analysis(autocorr_raw, autocorr_resid, zero_crossings_raw, zero_crossings_resid, tr,
                                  max_lag=None, n_example_voxels=5, figsize=(12.5, 5), save_path=None, dpi=300):
    """Plot autocorrelation analysis results (mean ACF, example voxels, INT histogram).

    Parameters
    ----------
    autocorr_raw : (n_voxels, n_lags) ndarray
        Autocorrelation before task removal.
    autocorr_resid : (n_voxels, n_lags) ndarray
        Autocorrelation after task removal.
    zero_crossings_raw : (n_voxels,) ndarray
        First zero-crossing indices before removal; may contain NaNs.
    zero_crossings_resid : (n_voxels,) ndarray
        First zero-crossing indices after removal; may contain NaNs.
    tr : float
        Repetition time in seconds.
    max_lag : int or None
        Maximum lag to plot (in timepoints); defaults to the largest zero-crossing.
    n_example_voxels : int
        Number of example voxel ACF curves to show.
    figsize : tuple
        Figure size.
    save_path : str or None
        If given, save the figure to this path.
    dpi : int
        Resolution for the saved figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    
    # Limit the lag range for plotting
    if max_lag is None:
        max_lag = int(np.nanmax((np.nanmax(zero_crossings_resid), np.nanmax(zero_crossings_raw))))
    # max_lag = min(max_lag, autocorr_raw.shape[1])
    lag_times = np.arange(max_lag) * tr  # Convert to seconds
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('fMRI Autocorrelation Analysis Results (using nilearn)', fontsize=16, fontweight='bold')
    
    # Plot 1: Average ACF comparison (before vs after)
    ax1 = axes[0]
    mean_acf_raw = np.nanmean(autocorr_raw[:, :max_lag], axis=0)
    mean_acf_resid = np.nanmean(autocorr_resid[:, :max_lag], axis=0)
    std_acf_raw = np.nanstd(autocorr_raw[:, :max_lag], axis=0)
    std_acf_resid = np.nanstd(autocorr_resid[:, :max_lag], axis=0)
    
    ax1.plot(lag_times, mean_acf_raw, 'b-', linewidth=2, label='Before task removal', alpha=0.8)
    ax1.fill_between(lag_times, mean_acf_raw - std_acf_raw, mean_acf_raw + std_acf_raw, 
                     color='blue', alpha=0.2)
    
    ax1.plot(lag_times, mean_acf_resid, 'r-', linewidth=2, label='After task removal', alpha=0.8)
    ax1.fill_between(lag_times, mean_acf_resid - std_acf_resid, mean_acf_resid + std_acf_resid, 
                     color='red', alpha=0.2)
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Lag (seconds)')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Mean Autocorrelation Function\n(±1 SD across voxels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Example individual voxel ACF curves
    ax2 = axes[1]
    n_voxels = autocorr_raw.shape[0]
    
    # Find valid (non-NaN) voxels for example plotting
    valid_voxels_mask = ~(np.isnan(zero_crossings_raw) | np.isnan(zero_crossings_resid))
    valid_indices = np.where(valid_voxels_mask)[0]
    
    if len(valid_indices) > 0:
        # Select evenly spaced examples from valid voxels
        n_examples = min(n_example_voxels, len(valid_indices))
        example_indices = valid_indices[np.linspace(0, len(valid_indices)-1, n_examples, dtype=int)]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_examples))
        for i, voxel_idx in enumerate(example_indices):
            ax2.plot(lag_times, autocorr_raw[voxel_idx, :max_lag], '--', 
                    color=colors[i], alpha=0.6, linewidth=1)
            ax2.plot(lag_times, autocorr_resid[voxel_idx, :max_lag], '-', 
                    color=colors[i], alpha=0.8, linewidth=2, 
                    label=f'Voxel {voxel_idx}')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No valid voxels', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (seconds)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title(f'Example Individual Voxel ACF\n(dashed=before, solid=after)')
    ax2.grid(True, alpha=0.3)
        
    # Plot 3: Zero-crossing distribution
    ax3 = axes[2]
    
    # Filter out NaN values for histogram
    valid_raw = zero_crossings_raw[~np.isnan(zero_crossings_raw)]
    valid_resid = zero_crossings_resid[~np.isnan(zero_crossings_resid)]
    
    if len(valid_raw) > 0:
        zero_crossings_seconds = valid_raw * tr
        ax3.hist(zero_crossings_seconds, bins=20, alpha=0.7, color='b', edgecolor='black', label='Before')
        ax3.axvline(np.nanmean(zero_crossings_seconds), color='b', linestyle='--', linewidth=2,
                   label=f'Mean (before): {np.nanmean(zero_crossings_seconds):.1f}s')
    
    if len(valid_resid) > 0:
        zero_crossings_seconds = valid_resid * tr
        ax3.hist(zero_crossings_seconds, bins=20, alpha=0.7, color='r', edgecolor='black', label='After')
        ax3.axvline(np.nanmean(zero_crossings_seconds), color='r', linestyle='--', linewidth=2,
                   label=f'Mean (after): {np.nanmean(zero_crossings_seconds):.1f}s')
    
    if len(valid_raw) == 0 and len(valid_resid) == 0:
        ax3.text(0.5, 0.5, 'No valid zero-crossings', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel('INTs (seconds)')
    ax3.set_ylabel('Number of Voxels')
    ax3.set_title('Distribution of INTs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')

    return fig


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
        sydnor_sa_cmap = create_sydnor_sa_colormap()
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


def plot_subject_timeseries_heatmap(raw_data, scan_string,
                                    residual_data=None, design_matrix=None,
                                    figsize=None,
                                    cmap='RdBu_r', vmin=None, vmax=None,
                                    time_units='minutes', tr=None,
                                    title_prefix="Neural Time Series",
                                    save_path=None, dpi=300):
    """Plot a subject's neural timeseries as heatmaps, optionally with task-regression rows.

    With residual_data and design_matrix, produces 5 rows (raw, task-regressed
    residuals, task signal = raw - residual, its variance across regions, and the
    design matrix); otherwise a single raw-timeseries row.

    Parameters
    ----------
    raw_data : (n_regions, n_timepoints) ndarray
        Raw timeseries.
    scan_string : str
        Scan identifier (e.g. 'sub-100307_modality-MRI3T_tr-0.72_...').
    residual_data : (n_regions, n_timepoints) ndarray or None
        Task-regressed residuals. If None, only raw data is plotted.
    design_matrix : ndarray, pd.DataFrame, or None
        Task regressors. If None, only raw data is plotted.
    figsize : tuple or None
        Figure size; auto-determined from the number of rows if None.
    cmap : str
        Colormap for the neural-data heatmaps.
    vmin, vmax : float or None
        Colour limits; 2nd/98th percentiles if None.
    time_units : {'minutes', 'seconds', 'timepoints'}
        Units for the time axis.
    tr : float or None
        Repetition time in seconds; parsed from scan_string if None.
    title_prefix : str
        Title prefix.
    save_path : str or None
        If given, save the figure to this path.
    dpi : int
        Resolution for the saved figure.

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of matplotlib Axes
    """
    
    n_regions, n_timepoints = raw_data.shape
    
    # Extract TR from scan string if not provided
    if tr is None:
        tr_parts = [p for p in scan_string.split('_') if p.startswith('tr-')]
        tr = float(tr_parts[0].split('-')[1]) if tr_parts else 1.0
    
    # Create time axis and labels
    time_multiplier = {'seconds': tr, 'minutes': tr/60, 'timepoints': 1}[time_units]
    time_axis = np.arange(n_timepoints) * time_multiplier
    time_label = f"Time ({time_units})"
    
    # ============== DETERMINE PLOT MODE ==============
    # Check if we have residual data and design matrix
    has_task_analysis = (residual_data is not None and design_matrix is not None)
    
    # ============== GET TASK REGRESSION DATA (if available) ==============
    if has_task_analysis:
        task_signal = raw_data - residual_data
    
    # ============== SETUP PLOTTING ==============
    # Determine color scaling (using nan-aware functions)
    all_neural_data = [raw_data.flatten()]
    if has_task_analysis:
        all_neural_data.extend([residual_data.flatten(), task_signal.flatten()])
    
    combined_data = np.concatenate(all_neural_data)
    if vmin is None:
        vmin = np.nanpercentile(combined_data, 2)
    if vmax is None:
        vmax = np.nanpercentile(combined_data, 98)
    
    # Create coordinate system
    x_coords = np.arange(n_timepoints)
    
    # Setup tick positions
    n_time_ticks = min(8, max(3, n_timepoints//50))
    time_tick_pos = np.linspace(0, n_timepoints-1, n_time_ticks, dtype=int)
    time_tick_labels = [f'{time_axis[i]:.1f}' for i in time_tick_pos]
    
    n_region_ticks = min(10, max(3, n_regions//50))
    region_tick_pos = np.linspace(0, n_regions-1, n_region_ticks, dtype=int)
    region_tick_labels = [f'{i+1}' for i in region_tick_pos]
    
    # Extract subject info for title
    try:
        subject_parts = [p for p in scan_string.split('_') if p.startswith(('sub-', 'scan-'))]
        subtitle = ', '.join(subject_parts)
    except Exception:
        subtitle = scan_string
    
    # ============== CREATE FIGURE ==============
    n_plots = 5 if has_task_analysis else 1
    
    # Auto-determine figure size if not provided
    if figsize is None:
        figsize = (12, 18) if has_task_analysis else (12, 4)
    
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=figsize)
    
    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    
    # Force consistent x-limits for all plots
    for ax in axes:
        ax.set_xlim(0, n_timepoints - 1)
    
    # Helper function for adding colorbars
    def add_colorbar(im, ax, label):
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=11)
        return cbar
    
    # Helper function for setting ticks
    def set_plot_ticks(ax, show_xlabel=False):
        ax.set_xticks(time_tick_pos)
        ax.set_xticklabels(time_tick_labels if show_xlabel else [])
        if show_xlabel:
            ax.set_xlabel(time_label, fontsize=12)
        
        if len(region_tick_pos) > 0:
            ax.set_yticks(region_tick_pos)
            ax.set_yticklabels(region_tick_labels)
    
    # ============== PLOT 1: RAW DATA ==============
    im1 = axes[0].imshow(raw_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                         aspect='auto', interpolation='nearest')
    axes[0].set_title('Raw Time Series', fontsize=14, fontweight='bold', pad=8)
    axes[0].set_ylabel('Brain Region', fontsize=12)
    set_plot_ticks(axes[0], show_xlabel=(n_plots == 1))
    add_colorbar(im1, axes[0], 'Signal Amplitude')
    
    # ============== REMAINING PLOTS (IF TASK ANALYSIS AVAILABLE) ==============
    if has_task_analysis:
        # Plot 2: Residual data
        im2 = axes[1].imshow(residual_data, cmap=cmap, vmin=vmin, vmax=vmax,
                             aspect='auto', interpolation='nearest')
        axes[1].set_title('Task-Regressed Residuals', fontsize=14, fontweight='bold', pad=8)
        axes[1].set_ylabel('Brain Region', fontsize=12)
        set_plot_ticks(axes[1])
        add_colorbar(im2, axes[1], 'Signal Amplitude')
        
        # Plot 3: Task signal
        vmax_ts = np.nanpercentile(np.abs(task_signal), 98)
        im3 = axes[2].imshow(task_signal, cmap='berlin', vmin=-vmax_ts, vmax=vmax_ts,
                             aspect='auto', interpolation='nearest')
        axes[2].set_title('Effect of Task Regression (Raw - Residual)', fontsize=14, fontweight='bold', pad=8)
        axes[2].set_ylabel('Brain Region', fontsize=12)
        set_plot_ticks(axes[2])
        add_colorbar(im3, axes[2], 'Signal Amplitude Delta')
        
        # Plot 4: Variance line plot
        which_stat = 'var'
        if which_stat == 'mean':
            task_variance = np.nanmean(np.abs(task_signal), axis=0)
            stat_label = 'Absolute Mean'
        elif which_stat == 'var':
            task_variance = np.nanvar(task_signal, axis=0)
            stat_label = 'Variance'

        axes[3].plot(x_coords, task_variance, 'b-', linewidth=2, alpha=0.8)
        axes[3].fill_between(x_coords, task_variance, alpha=0.3, color='blue')
        axes[3].set_title(f'{stat_label} of Task-Regression Effect Across Brain Regions', 
                          fontsize=14, fontweight='bold', pad=8)
        axes[3].set_ylabel(f'{stat_label}', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xticks(time_tick_pos)
        axes[3].set_xticklabels([])
        
        # Add colorbar for variance plot (using nan-aware min/max)
        sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=np.nanmin(task_variance), vmax=np.nanmax(task_variance)))
        sm.set_array([])
        add_colorbar(sm, axes[3], 'Dummy Colorbar')
        
        # Plot 5: Design matrix
        if design_matrix is not None:
            # Convert to array and transpose
            if hasattr(design_matrix, 'values'):
                dm_array = design_matrix.values.T
                regressor_names = design_matrix.columns.tolist()
            else:
                dm_array = design_matrix.T
                regressor_names = [f'Regressor {i+1}' for i in range(dm_array.shape[0])]
            
            n_regressors = dm_array.shape[0]
            
            im5 = axes[4].imshow(dm_array, cmap='viridis', aspect='auto', interpolation='nearest')
            axes[4].set_title('Design Matrix (Task Regressors)', fontsize=14, fontweight='bold', pad=8)
            axes[4].set_ylabel('Regressors', fontsize=12)
            
            # Set x-axis ticks to match other plots
            axes[4].set_xticks(time_tick_pos)
            axes[4].set_xticklabels(time_tick_labels)
            axes[4].set_xlabel(time_label, fontsize=12)
            
            # Set y-axis for regressors (not brain regions)
            if n_regressors <= 15:
                axes[4].set_yticks(np.arange(n_regressors))
                truncated_names = [name[:15] + '...' if len(name) > 15 else name 
                                   for name in regressor_names]
                axes[4].set_yticklabels(truncated_names, fontsize=9)
            else:
                n_reg_ticks = min(10, n_regressors//2)
                reg_indices = np.linspace(0, n_regressors-1, n_reg_ticks, dtype=int)
                axes[4].set_yticks(reg_indices)
                axes[4].set_yticklabels([f'Reg {i+1}' for i in reg_indices])
            
            add_colorbar(im5, axes[4], 'Regressor Weight')
        else:
            # No design matrix available
            axes[4].text(0.5, 0.5, 'Design Matrix Not Available', 
                         ha='center', va='center', transform=axes[4].transAxes, 
                         fontsize=14, style='italic')
            axes[4].set_title('Design Matrix (Task Regressors)', fontsize=14, fontweight='bold', pad=8)
            axes[4].set_ylabel('Regressors', fontsize=12)
            set_plot_ticks(axes[4], show_xlabel=True)
    
    # ============== FINALIZE FIGURE ==============
    info_text = f'Regions: {n_regions}, Timepoints: {n_timepoints}, TR: {tr}s'
    title_text = f'{title_prefix} - {subtitle}. {info_text}'
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.95 if has_task_analysis else 0.98)
    
    plt.tight_layout()
    if has_task_analysis:
        plt.subplots_adjust(top=0.89, bottom=0.07)
    else:
        plt.subplots_adjust(top=0.88, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    return fig, tuple(axes)


# =============================================================================
# Lollipop plots (PET / feature correlation summaries)
# =============================================================================

def _draw_lollipop(ax, plot_delta, sig_either, color_map, nonsig_color,
                   linewidth, markersize, sig_alpha, nonsig_alpha,
                   absolute_delta, delta_label, delta_title, xlim, fontsize):
    """Draw a single horizontal lollipop panel onto ax."""
    order = plot_delta.sort_values(ascending=False).index.tolist()
    y_positions = np.arange(len(order))

    for y, feature in zip(y_positions, order):
        dr  = plot_delta[feature]
        sig = sig_either[feature]
        color = color_map[feature] if sig else nonsig_color
        alpha = sig_alpha if sig else nonsig_alpha
        ax.hlines(y, 0, dr, color=color, linewidth=linewidth, alpha=alpha)
        ax.plot(dr, y, 'o', color=color, markersize=markersize, alpha=alpha)

    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_yticks(y_positions)
    stripped = [f.split('_', 1)[1] if '_' in f else f for f in order]
    ax.set_yticklabels(stripped, fontsize=fontsize - 1)
    ax.set_xlabel(delta_label, fontsize=fontsize - 1)
    ax.set_title(delta_title, fontsize=fontsize, fontweight='bold')
    ax.tick_params(axis='x', labelsize=fontsize - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_faceted_lollipops(df_r, df_p, category_colors=None, figsize=None,
                            markersize=5, linewidth=1.2, alpha=1.0,
                            nonsig_alpha=0.75, xlim=None,
                            rank_reference='Rest', show_legend=True,
                            annotate_sig=False, annotation_fontsize=5,
                            annotation_offset=0.02, fontsize=8):
    """Horizontal faceted lollipop plot of correlations, one panel per condition.

    Each panel plots the features sorted by r, coloured by feature category (the
    prefix before the first '_').

    Parameters
    ----------
    df_r : pd.DataFrame
        Correlations (features x conditions).
    df_p : pd.DataFrame
        P-values matching df_r.
    category_colors : dict, list, or None
        Per-category colours; a tab10 mapping is built if None.
    rank_reference : str or None
        Column used to rank the other panels by their Spearman rho against it.
    show_legend : bool
        Draw a category legend below the panels.
    annotate_sig : bool
        Annotate lollipop tips with **/*/ns.
    fontsize : int
        Base font size.

    Returns
    -------
    fig : matplotlib Figure
    axes : list of matplotlib Axes
    """
    features       = df_r.index.tolist()
    all_conditions = df_r.columns.tolist()

    rank_ref_active = rank_reference is not None and rank_reference in all_conditions
    if rank_reference is not None and not rank_ref_active:
        warnings.warn(
            f"rank_reference='{rank_reference}' not found in df_r columns. "
            "Panel ordering skipped.", UserWarning, stacklevel=2)

    if rank_ref_active:
        other = [c for c in all_conditions if c != rank_reference]
        rhos  = {c: sp.stats.spearmanr(df_r[rank_reference], df_r[c])[0] for c in other}
        conditions = [rank_reference] + sorted(other, key=lambda c: rhos[c], reverse=True)
    else:
        conditions = all_conditions

    categories        = [f.split('_')[0] for f in features]
    unique_categories = list(dict.fromkeys(categories))

    if category_colors is None:
        cmap = mpl.colormaps['tab10']
        category_colors = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
    elif isinstance(category_colors, (list, tuple)):
        category_colors = {cat: category_colors[i] for i, cat in enumerate(unique_categories)}

    color_map = {f: category_colors[cat] for f, cat in zip(features, categories)}

    if xlim is None:
        absmax = np.nanmax(np.abs(df_r.values))
        bound  = round(absmax + 0.15, 1)
        xlim   = (-bound, bound)

    n_conditions = len(conditions)
    n_features   = len(features)
    if figsize is None:
        figsize = (2.8 * n_conditions, max(4, n_features * 0.18))

    fig, axes = plt.subplots(1, n_conditions, figsize=figsize, sharey=False, dpi=300)
    if n_conditions == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        r_vals = df_r[condition]
        p_vals = df_p[condition]
        order  = r_vals.sort_values(ascending=False).index.tolist()
        y_pos  = np.arange(len(order))

        for y, feature in zip(y_pos, order):
            r     = r_vals[feature]
            p     = p_vals[feature]
            color = color_map[feature]
            a     = alpha if p < 0.05 else nonsig_alpha
            ax.hlines(y, 0, r, color=color, linewidth=linewidth, alpha=a)
            ax.plot(r, y, 'o', color=color, markersize=markersize, alpha=a)
            if annotate_sig:
                label = determine_significance(p)
                xoff  = annotation_offset if r >= 0 else -annotation_offset
                ax.text(r + xoff, y, label, ha='left' if r >= 0 else 'right',
                        va='center', fontsize=annotation_fontsize, color=color, alpha=a)

        ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.split('_', 1)[1] if '_' in f else f for f in order],
                            fontsize=fontsize - 1)

        if rank_ref_active and condition != rank_reference:
            rho   = sp.stats.spearmanr(df_r[rank_reference], df_r[condition])[0]
            title = f"{condition}\n$\\rho_{{rank}}$ = {rho:.2f} (vs. {rank_reference})"
        else:
            title = condition

        ax.set_title(title, fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('Correlation', fontsize=fontsize - 1)
        ax.tick_params(axis='x', labelsize=fontsize - 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    legend_handles = [mpatches.Patch(color=category_colors[cat], label=cat)
                      for cat in unique_categories]
    if show_legend:
        leg = fig.legend(handles=legend_handles, loc='lower center', ncol=1,
                         fontsize=fontsize - 1, frameon=False, bbox_to_anchor=(0.5, 0))
        fig.canvas.draw()
        leg_height = leg.get_window_extent().height / fig.bbox.height
        fig.tight_layout(rect=[0, leg_height + 0.02, 1, 1])
    else:
        fig.tight_layout()

    return fig, axes


def plot_two_condition_supplement(df_r, df_p, reference, sig_threshold=0.05,
                                   category_colors=None, figsize=None,
                                   markersize=5, linewidth=1.2,
                                   sig_alpha=1.0, nonsig_alpha=0.25,
                                   nonsig_color='#aaaaaa',
                                   annotate_scatter=False, annotation_fontsize=5,
                                   absolute_delta=False, show_lollipop=True,
                                   split_lollipop=False, show_legend=True,
                                   fontsize=8):
    """Scatter + optional lollipop supplement for a two-condition correlation table.

    Left panel: scatter of r(reference) vs r(task). Right panel(s): horizontal
    lollipop of the change in correlation (task - reference), sorted descending.

    Parameters
    ----------
    df_r : pd.DataFrame
        Exactly 2 columns, one of which is `reference`.
    df_p : pd.DataFrame
        Matching p-values.
    reference : str
        Column name of the reference condition.
    show_lollipop : bool
        Show the lollipop panel(s) alongside the scatter.
    split_lollipop : bool
        Split the lollipop by the sign of r(reference).
    show_legend : bool
        Draw a category legend.

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of matplotlib Axes
    """
    if df_r.shape[1] != 2:
        raise ValueError('plot_two_condition_supplement requires exactly 2 conditions.')
    if reference not in df_r.columns:
        raise ValueError(f"Reference '{reference}' not found in df_r columns.")

    task     = [c for c in df_r.columns if c != reference][0]
    features = df_r.index.tolist()

    categories        = [f.split('_')[0] for f in features]
    unique_categories = list(dict.fromkeys(categories))
    if category_colors is None:
        cmap = mpl.colormaps['tab10']
        category_colors = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
    color_map  = {f: category_colors[cat] for f, cat in zip(features, categories)}
    sig_either = (df_p[reference] < sig_threshold) | (df_p[task] < sig_threshold)

    if figsize is None:
        n_features = len(features)
        figsize = (12, max(4, n_features * 0.18)) if show_lollipop else (5, max(4, n_features * 0.18))

    fig = plt.figure(figsize=figsize, dpi=300)

    if not show_lollipop:
        ax_scatter   = fig.add_subplot(1, 1, 1)
        lollipop_axes = []
    elif not split_lollipop:
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.33])
        ax_scatter    = fig.add_subplot(gs[0, 0])
        ax_lollipop   = fig.add_subplot(gs[0, 1])
        lollipop_axes = [ax_lollipop]
    else:
        pos_features = df_r.index[df_r[reference] >= 0].tolist()
        neg_features = df_r.index[df_r[reference] <  0].tolist()
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 1, 1])
        ax_scatter      = fig.add_subplot(gs[0, 0])
        ax_loll_neg     = fig.add_subplot(gs[0, 1])
        ax_loll_pos     = fig.add_subplot(gs[0, 2])
        lollipop_axes   = [ax_loll_neg, ax_loll_pos]

    if show_lollipop:
        delta_r    = df_r[task] - df_r[reference]
        plot_delta = delta_r.abs() if absolute_delta else delta_r
        delta_label = (f'|Δr|  (|{task} − {reference}|)' if absolute_delta
                       else f'Δr  ({task} − {reference})')
        delta_title_base = (f'Absolute change\n(|{task} − {reference}|)' if absolute_delta
                            else f'Change in correlation\n({task} − {reference})')
        lollipop_xlim = (
            -0.05 if absolute_delta else min(-0.1, plot_delta.min() - 0.05),
            max(0.1, plot_delta.max() + 0.05),
        )
        lollipop_subsets = (
            [(plot_delta, delta_title_base)] if not split_lollipop
            else [(plot_delta.loc[neg_features], f'r({reference}) < 0'),
                  (plot_delta.loc[pos_features], f'r({reference}) ≥ 0')]
        )
        for ax_l, (subset_delta, title) in zip(lollipop_axes, lollipop_subsets):
            _draw_lollipop(ax_l, subset_delta, sig_either, color_map, nonsig_color,
                           linewidth, markersize, sig_alpha, nonsig_alpha,
                           absolute_delta, delta_label, title, lollipop_xlim, fontsize)

    absmax = max(np.abs(df_r.values).max(), 0.1)
    bound  = round(absmax + 0.05, 1)
    ax_scatter.set_xlim(-bound, bound)
    ax_scatter.set_ylim(-bound, bound)

    texts, x_coords, y_coords = [], [], []
    for feature in features:
        r_ref  = df_r.loc[feature, reference]
        r_task = df_r.loc[feature, task]
        sig    = sig_either[feature]
        color  = color_map[feature] if sig else nonsig_color
        a      = sig_alpha if sig else nonsig_alpha
        ax_scatter.plot(r_ref, r_task, 'o', color=color, markersize=markersize, alpha=a)
        x_coords.append(r_ref)
        y_coords.append(r_task)
        if annotate_scatter:
            label = '_'.join(feature.split('_')[1:])
            texts.append(ax_scatter.text(r_ref, r_task, label,
                                          fontsize=annotation_fontsize, color=color, alpha=a))

    if annotate_scatter and texts:
        if adjustText is not None:
            adjustText.adjust_text(texts, x=x_coords, y=y_coords, ax=ax_scatter,
                                    expand=(1.5, 2.0), force_text=(0.5, 1.0),
                                    force_points=(0.5, 1.0), ensure_inside_axes=True,
                                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.4, alpha=0.5))

    ax_scatter.plot([-bound, bound], [-bound, bound], color='black', lw=0.7, ls='--', alpha=0.4)
    ax_scatter.axhline(0, color='grey', lw=0.4, alpha=0.3)
    ax_scatter.axvline(0, color='grey', lw=0.4, alpha=0.3)
    ax_scatter.set_xlabel(f'r  ({reference})', fontsize=fontsize - 1)
    ax_scatter.set_ylabel(f'r  ({task})', fontsize=fontsize - 1)
    ax_scatter.set_title(f'{reference} vs. {task}', fontsize=fontsize, fontweight='bold')
    ax_scatter.tick_params(labelsize=fontsize - 1)
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)

    legend_handles = [mpatches.Patch(color=category_colors[cat], label=cat)
                      for cat in unique_categories]
    legend_handles.append(mpatches.Patch(color=nonsig_color, alpha=nonsig_alpha,
                                           label=f'p ≥ {sig_threshold} (both conditions)'))
    fig.tight_layout()

    if show_lollipop and split_lollipop:
        fig.canvas.draw()
        scatter_bbox = ax_scatter.get_position()
        for ax_l in lollipop_axes:
            pos = ax_l.get_position()
            ax_l.set_position([pos.x0, scatter_bbox.y0, pos.width, scatter_bbox.height])

    if show_legend:
        fig.canvas.draw()
        renderer    = fig.canvas.get_renderer()
        fig_h_px    = fig.get_size_inches()[1] * fig.dpi
        all_axes    = [ax_scatter] + list(lollipop_axes)
        min_bot_fig = min(ax.get_tightbbox(renderer).y0 for ax in all_axes) / fig_h_px
        fig.legend(handles=legend_handles, loc='upper center',
                   ncol=len(legend_handles), fontsize=fontsize - 1, frameon=False,
                   bbox_to_anchor=(0.5, min_bot_fig - 0.01))

    if not show_lollipop:
        return fig, (ax_scatter,)
    elif not split_lollipop:
        return fig, (ax_scatter, ax_lollipop)
    else:
        return fig, (ax_scatter, ax_loll_pos, ax_loll_neg)


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
