import os, sys, time, getpass, warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def plot_autocorrelation_analysis(autocorr_raw, autocorr_resid, zero_crossings_raw, zero_crossings_resid, 
                                  design_matrix, events, tr, 
                                  n_example_voxels=5, figsize=(10, 5)):
    """
    Plot autocorrelation analysis results and block design.
    
    Parameters:
    - autocorr_raw: autocorrelation before task removal (n_voxels x n_lags)
    - autocorr_resid: autocorrelation after task removal (n_voxels x n_lags)
    - zero_crossings: first zero-crossing indices (n_voxels,)
    - design_matrix: nilearn DesignMatrix object
    - events: pandas DataFrame with events (onset, duration, trial_type) for nilearn
    - tr: repetition time in seconds
    - max_lag: maximum lag to plot (in timepoints)
    - n_example_voxels: number of example voxel ACF curves to show
    - figsize: figure size tuple
    """
    
    # Limit the lag range for plotting
    max_lag = 25
    max_lag = min(max_lag, autocorr_raw.shape[1])
    lag_times = np.arange(max_lag) * tr  # Convert to seconds
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('fMRI Autocorrelation Analysis Results (using nilearn)', fontsize=16, fontweight='bold')
    
    # Plot 1: Average ACF comparison (before vs after)
    ax1 = axes[0, 0]
    mean_acf_raw = np.mean(autocorr_raw[:, :max_lag], axis=0)
    mean_acf_resid = np.mean(autocorr_resid[:, :max_lag], axis=0)
    std_acf_raw = np.std(autocorr_raw[:, :max_lag], axis=0)
    std_acf_resid = np.std(autocorr_resid[:, :max_lag], axis=0)
    
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
    ax2 = axes[0, 1]
    n_voxels = autocorr_raw.shape[0]
    example_indices = np.linspace(0, n_voxels-1, n_example_voxels, dtype=int)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_example_voxels))
    for i, voxel_idx in enumerate(example_indices):
        ax2.plot(lag_times, autocorr_raw[voxel_idx, :max_lag], '--', 
                color=colors[i], alpha=0.6, linewidth=1)
        ax2.plot(lag_times, autocorr_resid[voxel_idx, :max_lag], '-', 
                color=colors[i], alpha=0.8, linewidth=2, 
                label=f'Voxel {voxel_idx}')
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (seconds)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title(f'Example Individual Voxel ACF\n(dashed=before, solid=after)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Design matrix regressors
    ax3 = axes[1, 0]
    
    # Plot design matrix columns (task regressors and HRF-convolved)
    time_vector = np.arange(design_matrix.shape[0]) * tr
    task_columns = [col for col in design_matrix.columns if not any(x in col.lower() for x in ['drift', 'constant'])]
    
    if len(task_columns) > 0:
        # Show task regressors
        for i, col in enumerate(task_columns[:3]):  # Show first 3 task regressors
            ax3.plot(time_vector, design_matrix[col], label=col, linewidth=2, alpha=0.8)
        
        # Overlay event markers
        for _, event in events.iterrows():
            ax3.axvspan(event['onset'], event['onset'] + event['duration'], 
                       alpha=0.2, color='orange', label='Events' if _ == 0 else "")
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Design Matrix Regressors\n(nilearn HRF convolution)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No task regressors found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Design Matrix')
    
    # Plot 4: Zero-crossing distribution
    ax4 = axes[1, 1]
    zero_crossings_seconds = zero_crossings_raw * tr
    ax4.hist(zero_crossings_seconds, bins=20, alpha=0.7, color='b', edgecolor='black')
    ax4.axvline(np.mean(zero_crossings_seconds), color='b', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(zero_crossings_seconds):.1f}s')
    zero_crossings_seconds = zero_crossings_resid * tr
    ax4.hist(zero_crossings_seconds, bins=20, alpha=0.7, color='r', edgecolor='black')
    ax4.axvline(np.mean(zero_crossings_seconds), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(zero_crossings_seconds):.1f}s')
    ax4.set_xlabel('First Zero-Crossing (seconds)')
    ax4.set_ylabel('Number of Voxels')
    ax4.set_title('Distribution of Zero-Crossings')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_repeated_measures_violins(df, color_vector=None, figsize=(15, 8), 
                                   alpha_points=0.6, alpha_lines=0.3, 
                                   violin_alpha=0.7, colormap='viridis',
                                   line_width=0.5, point_size=2):
    """
    Create violin plots for repeated measures data with connecting lines.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the repeated measures data
    color_vector : array-like, optional
        Vector of integers (length equal to number of rows) for coloring observations
    figsize : tuple, default (15, 8)
        Figure size (width, height)
    alpha_points : float, default 0.6
        Transparency for individual data points
    alpha_lines : float, default 0.3
        Transparency for connecting lines
    violin_alpha : float, default 0.7
        Transparency for violin plots
    colormap : str, default 'viridis'
        Colormap name for coloring points when color_vector is provided
    line_width : float, default 0.5
        Width of connecting lines
    point_size : float, default 2
        Size of individual data points
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Drop the first column if it's just an index
    if df.columns[0] in ['', 'Unnamed: 0'] or df.iloc[:, 0].dtype in ['int64', 'int32']:
        df = df.iloc[:, 1:]
    
    # Get condition names (column names)
    conditions = df.columns.tolist()
    n_conditions = len(conditions)
    n_observations = len(df)
    
    print(f"Data shape: {df.shape}")
    print(f"Conditions: {conditions}")
    
    # Reshape data from wide to long format for seaborn
    df_long = pd.melt(df.reset_index(), id_vars=['index'], 
                      value_vars=conditions, 
                      var_name='Condition', value_name='Value')
    df_long.rename(columns={'index': 'Subject'}, inplace=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plots
    sns.violinplot(data=df_long, x='Condition', y='Value', color='gray',
                   ax=ax, alpha=violin_alpha, inner=None)
    
    # Prepare colors for points if color_vector is provided
    if color_vector is not None:
        if len(color_vector) != n_observations:
            raise ValueError(f"Color vector length ({len(color_vector)}) must match number of observations ({n_observations})")
        
        # Create colormap
        cmap = plt.cm.get_cmap(colormap)
        unique_colors = np.unique(color_vector)
        colors = [cmap(i / (len(unique_colors) - 1)) for i in range(len(unique_colors))]
        color_dict = dict(zip(unique_colors, colors))
        point_colors = [color_dict[val] for val in color_vector]
    else:
        point_colors = ['black'] * n_observations
    
    # Add individual data points and connecting lines
    x_positions = range(n_conditions)
    
    for i in range(n_observations):
        # Get y-values for this subject across all conditions
        y_values = df.iloc[i, :].values
        
        # Add small random jitter to x-positions for better visibility
        jitter = np.random.normal(0, 0.05, n_conditions)
        x_jittered = np.array(x_positions) + jitter
        
        # Plot connecting lines
        ax.plot(x_jittered, y_values, 
               color=point_colors[i] if color_vector is not None else 'gray',
               alpha=alpha_lines, linewidth=line_width, zorder=1)
        
        # Plot individual points
        ax.scatter(x_jittered, y_values, 
                  color=point_colors[i], alpha=alpha_points, 
                  s=point_size, zorder=2)
    
    # Customize the plot
    ax.set_xlabel('Condition', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add colorbar if color_vector is provided
    if color_vector is not None:
        # Create a mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(vmin=min(color_vector), 
                                                   vmax=max(color_vector)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Color vector', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig, ax


def plot_subject_level_violins(df, observation_indices, figsize=(15, 8), 
                               violin_alpha=0.7, point_alpha=0.6, point_size=3,
                               show_points=True, point_jitter=0.02,
                               show_lines=False, line_alpha=0.3, line_width=0.5,
                               show_violins=True, line_color_by='subject', ylim=None):
    """
    Create violin plots showing subject-level variation for selected regions.
    Each region index gets its own violin per condition.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the repeated measures data with 'subject' column
        Expected structure: 400 rows per subject, with subjects stacked
    observation_indices : list or array-like
        List of region indices to include (0-399)
    figsize : tuple, default (15, 8)
        Figure size (width, height)
    violin_alpha : float, default 0.7
        Transparency for violin plots
    point_alpha : float, default 0.6
        Transparency for individual subject points
    point_size : float, default 3
        Size of individual subject points
    show_points : bool, default True
        Whether to show individual subject points
    point_jitter : float, default 0.02
        Amount of jitter for point positions
    show_lines : bool, default False
        Whether to show connecting lines across conditions for each subject
    line_alpha : float, default 0.3
        Transparency for connecting lines
    line_width : float, default 0.5
        Width of connecting lines
    show_violins : bool, default True
        Whether to show violin plots (distributions)
    line_color_by : str, default 'subject'
        How to color the connecting lines: 'subject' (orange-purple gradient across subjects) 
        or 'region' (same color as violins for each region)
    ylim : tuple or None, default None
        Y-axis limits as (ymin, ymax). If None, uses automatic limits
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Make a copy to avoid modifying original data
    df_work = df.copy()
    
    # Drop the first column if it's just an index
    if df_work.columns[0] in ['', 'Unnamed: 0'] or df_work.iloc[:, 0].dtype in ['int64', 'int32']:
        df_work = df_work.iloc[:, 1:]
    
    # Ensure observation_indices is a list
    if not isinstance(observation_indices, (list, np.ndarray)):
        observation_indices = [observation_indices]
    
    # Get unique subjects and conditions
    subjects = df_work['subject'].unique()
    condition_cols = [col for col in df_work.columns if col != 'subject']
    n_subjects = len(subjects)
    
    print(f"Number of subjects: {n_subjects}")
    print(f"Selected regions: {observation_indices}")
    print(f"Conditions: {condition_cols}")
    
    # Create observation index within each subject
    df_work['obs_index'] = df_work.groupby('subject').cumcount()
    
    # Filter for selected observations
    df_filtered = df_work[df_work['obs_index'].isin(observation_indices)].copy()
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Expected rows: {n_subjects * len(observation_indices)}")
    
    # Reshape data from wide to long format
    df_long = pd.melt(df_filtered, 
                      id_vars=['subject', 'obs_index'], 
                      value_vars=condition_cols,
                      var_name='Condition', 
                      value_name='Value')
    
    # Create region labels for better readability
    df_long['Region'] = 'Region_' + df_long['obs_index'].astype(str)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for different regions - soft orange to purple gradient
    n_obs = len(observation_indices)
    if n_obs == 1:
        obs_colors = ['#FFB366']  # Soft orange
    else:
        # Create a custom colormap from soft orange to purple
        from matplotlib.colors import LinearSegmentedColormap
        orange_color = '#FFB366'  # Soft orange
        purple_color = '#9370DB'  # Purple
        
        # Create colormap from orange to purple
        colors = [orange_color, purple_color]
        custom_cmap = LinearSegmentedColormap.from_list('orange_purple', colors, N=256)
        
        # Sample colors evenly across the gradient
        obs_colors = [custom_cmap(i / (n_obs - 1)) for i in range(n_obs)]
    
    # Create subject colors for lines if needed
    if line_color_by == 'subject':
        # Use viridis colormap for subject coloring
        subject_cmap = plt.cm.get_cmap('viridis')
        
        # Create colors for all subjects
        n_subjects = len(subjects)
        subject_colors = {subj: subject_cmap(i / max(1, n_subjects - 1)) 
                         for i, subj in enumerate(subjects)}
    else:
        subject_colors = None
    
    # Create violin plots split by region (optional)
    if show_violins:
        sns.violinplot(data=df_long, x='Condition', y='Value', hue='Region',
                       ax=ax, alpha=violin_alpha, inner=None, 
                       palette=obs_colors)
    
    # Add connecting lines for each subject if requested
    if show_lines:
        # Get x positions for conditions
        n_conditions = len(condition_cols)
        
        # Calculate positions for each observation within each condition
        width = 0.8  # Default violin width in seaborn
        spacing = width / n_obs
        
        # For each subject, draw connecting lines for each region
        for subject in subjects:
            for obs_idx, obs_val in enumerate(observation_indices):
                # Get data for this subject and region across all conditions
                subject_obs_data = df_filtered[(df_filtered['subject'] == subject) & 
                                              (df_filtered['obs_index'] == obs_val)]
                
                if len(subject_obs_data) > 0:
                    # Get y-values for this subject/region across conditions
                    y_values = []
                    x_positions = []
                    
                    for cond_idx, condition in enumerate(condition_cols):
                        if condition in subject_obs_data.columns:
                            y_val = subject_obs_data[condition].iloc[0]
                            y_values.append(y_val)
                            
                            # Calculate x position for this region's violin center
                            if n_obs == 1:
                                x_pos = cond_idx
                            else:
                                x_offset = (obs_idx - (n_obs - 1) / 2) * spacing
                                x_pos = cond_idx + x_offset
                            x_positions.append(x_pos)
                    
                    # Draw connecting line for this subject/region
                    if len(y_values) > 1:  # Only draw if we have multiple points
                        # Choose line color based on line_color_by parameter
                        if line_color_by == 'subject':
                            line_color = subject_colors[subject]
                        else:  # line_color_by == 'region'
                            line_color = obs_colors[obs_idx]
                            
                        ax.plot(x_positions, y_values, 
                               color=line_color, 
                               alpha=line_alpha, 
                               linewidth=line_width, 
                               zorder=1)
    
    # Add individual subject points if requested
    if show_points:
        # Calculate positions for each region within each condition
        # This accounts for the spacing seaborn uses for grouped violins
        width = 0.8  # Default violin width in seaborn
        spacing = width / n_obs
        
        for cond_idx, condition in enumerate(condition_cols):
            for obs_idx, obs_val in enumerate(observation_indices):
                # Filter data for this condition and region
                mask = ((df_long['Condition'] == condition) & 
                       (df_long['obs_index'] == obs_val))
                data_subset = df_long[mask]
                
                if len(data_subset) > 0:
                    # Calculate x position for this region's violin
                    if n_obs == 1:
                        x_pos = cond_idx
                    else:
                        x_offset = (obs_idx - (n_obs - 1) / 2) * spacing
                        x_pos = cond_idx + x_offset
                    
                    # Add jitter
                    x_jittered = x_pos + np.random.normal(0, point_jitter, len(data_subset))
                    
                    # Plot points
                    ax.scatter(x_jittered, data_subset['Value'], 
                             color=obs_colors[obs_idx], 
                             alpha=point_alpha, 
                             s=point_size, 
                             zorder=3)
    
    # Customize the plot
    ax.set_xlabel('Condition', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    
    # Ensure condition labels are always shown
    ax.set_xticks(range(len(condition_cols)))
    ax.set_xticklabels(condition_cols, rotation=45, ha='right')
    
    # Create title with region info
    # obs_str = ', '.join(map(str, observation_indices))
    # title = f'Subject-Level Variation by Region\nRegions: {obs_str} ({n_subjects} subjects)'
    # ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Customize legend
    if len(observation_indices) > 1:
        if show_violins:
            # Use legend from violin plots
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [f'Region {idx}' for idx in observation_indices], 
                     title='Region Index', loc='upper right')
        else:
            # Create custom legend for points/lines when no violins
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=obs_colors[i], label=f'Region {idx}') 
                              for i, idx in enumerate(observation_indices)]
            ax.legend(handles=legend_elements, title='Region Index', loc='upper right')
    else:
        if show_violins and ax.get_legend():
            ax.get_legend().remove()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add colorbar for subject coloring if applicable
    if show_lines and line_color_by == 'subject':
        # Create a mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=subject_cmap, 
                                  norm=plt.Normalize(vmin=0, vmax=n_subjects-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Subject Index', fontsize=10)
        
        # Set colorbar ticks to show meaningful subject indices
        if n_subjects <= 10:
            cbar.set_ticks(range(n_subjects))
        else:
            # For many subjects, show fewer ticks
            tick_indices = np.linspace(0, n_subjects-1, 5, dtype=int)
            cbar.set_ticks(tick_indices)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_correlation(x, y, x_label=None, y_label=None, title=None, 
                    method='pearson', color='#2E86AB', alpha=0.6, 
                    size=20, figsize=(3, 3), show_line=True, 
                    show_confidence=True, show_stats=True,
                    stats_position='upper left', font_size=8,
                    grid=True, grid_alpha=0.3, dpi=300,
                    outlier_threshold=None, highlight_outliers=False, 
                    save_path=None, return_stats=False, ax=None):
    """
    Parameters:
    -----------
    x, y : array-like, pandas.Series, or numpy.ndarray
        Input variables to correlate. If pandas Series, their names will be used 
        as default axis labels unless x_label/y_label are explicitly provided.
    x_label, y_label : str, optional
        Labels for x and y axes. If None and inputs are pandas Series, 
        the Series names will be used automatically.
    title : str, optional
        Plot title
    method : str, default 'pearson'
        Correlation method ('pearson', 'spearman')
    color : str, default '#2E86AB'
        Color for scatter points and regression line
    alpha : float, default 0.6
        Transparency of scatter points
    size : float, default 20
        Size of scatter points
    figsize : tuple, default (3, 3)
        Figure size in inches (only used if ax=None)
    show_line : bool, default True
        Whether to show regression line
    show_confidence : bool, default True
        Whether to show confidence interval around regression line
    show_stats : bool, default True
        Whether to display correlation statistics on plot
    stats_position : str, default 'upper left'
        Position of statistics text box
    font_size : int, default 8
        Base font size for labels and text
    grid : bool, default True
        Whether to show grid
    grid_alpha : float, default 0.3
        Grid transparency
    outlier_threshold : float, optional
        Z-score threshold for outlier detection (e.g., 2.5)
    highlight_outliers : bool, default False
        Whether to highlight outliers in different color
    save_path : str, optional
        Path to save the figure (only used if ax=None)
    return_stats : bool, default False
        Whether to return correlation statistics
    ax : matplotlib.axes.Axes, optional
        Pre-existing axis to plot on. If None, creates new figure and axis.
        
    Returns:
    --------
    fig, ax : matplotlib objects (if ax=None)
        Figure and axis objects
    ax : matplotlib axis object (if ax was provided)
        The provided axis object
    stats_dict : dict, optional
        Dictionary with correlation statistics (if return_stats=True)
    """
    
    # Handle pandas Series input and extract labels
    x_series_name = None
    y_series_name = None
    
    if isinstance(x, pd.Series):
        x_series_name = x.name
        x = x.values
    if isinstance(y, pd.Series):
        y_series_name = y.name
        y = y.values
    
    # Set default labels from pandas Series names if not provided
    if x_label is None and x_series_name is not None:
        x_label = str(x_series_name)
    if y_label is None and y_series_name is not None:
        y_label = str(y_series_name)
    
    # Convert to numpy arrays and handle missing data
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 3:
        raise ValueError("Need at least 3 valid data points for correlation")
    
    # Calculate correlation
    if method.lower() == 'pearson':
        corr_coef, p_value = pearsonr(x_clean, y_clean)
        method_name = "Pearson"
    elif method.lower() == 'spearman':
        corr_coef, p_value = spearmanr(x_clean, y_clean)
        method_name = "Spearman"
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    # Detect outliers if requested
    outlier_mask = np.zeros(len(x_clean), dtype=bool)
    if outlier_threshold is not None:
        z_scores_x = np.abs(stats.zscore(x_clean))
        z_scores_y = np.abs(stats.zscore(y_clean))
        outlier_mask = (z_scores_x > outlier_threshold) | (z_scores_y > outlier_threshold)
    
    # Create figure and axis or use provided axis
    created_new_figure = ax is None
    if created_new_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot scatter points
    if highlight_outliers and np.any(outlier_mask):
        # Plot normal points
        normal_mask = ~outlier_mask
        ax.scatter(x_clean[normal_mask], y_clean[normal_mask], 
                  c=color, alpha=alpha, s=size, edgecolors='white', 
                  linewidth=0.5, label='Data points')
        
        # Plot outliers
        ax.scatter(x_clean[outlier_mask], y_clean[outlier_mask], 
                  c='red', alpha=alpha+0.2, s=size*1.2, edgecolors='darkred', 
                  linewidth=1, marker='^', label='Outliers')
    else:
        ax.scatter(x_clean, y_clean, c=color, alpha=alpha, s=size, 
                  edgecolors='white', linewidth=0.5)
    
    # Add regression line and confidence interval
    if show_line and len(x_clean) > 2:
        # Calculate regression line
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x_clean, y_clean)
        
        # Create line points
        x_line = np.linspace(np.min(x_clean), np.max(x_clean), 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, color=color, linewidth=2.5, alpha=0.8, 
               linestyle='-', label='Regression line')
        
        # Add confidence interval
        if show_confidence:
            # Calculate confidence interval
            def predict_interval(x_new, x_data, y_data, confidence=0.95):
                n = len(x_data)
                x_mean = np.mean(x_data)
                sxx = np.sum((x_data - x_mean) ** 2)
                sxy = np.sum((x_data - x_mean) * (y_data - np.mean(y_data)))
                syy = np.sum((y_data - np.mean(y_data)) ** 2)
                
                s = np.sqrt((syy - sxy**2/sxx) / (n - 2))
                t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
                
                margin = t_val * s * np.sqrt(1/n + (x_new - x_mean)**2/sxx)
                y_pred = slope * x_new + intercept
                
                return y_pred - margin, y_pred + margin
            
            y_lower, y_upper = predict_interval(x_line, x_clean, y_clean)
            ax.fill_between(x_line, y_lower, y_upper, color=color, alpha=0.15, 
                           label='95% Confidence Interval')
    
    # Add statistics text box
    if show_stats:
        # Determine significance stars
        if p_value < 0.001:
            sig_stars = "***"
        elif p_value < 0.01:
            sig_stars = "**"
        elif p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "ns"
        
        # Create statistics text
        stats_text = f"{method_name} r = {corr_coef:.3f}{sig_stars}\n"
        stats_text += f"p = {p_value:.3e}\n"
        stats_text += f"n = {len(x_clean)}"
        
        if np.any(outlier_mask):
            stats_text += f"\nOutliers: {np.sum(outlier_mask)}"
        
        # Position the text box
        position_dict = {
            'upper left': (0.05, 0.95),
            'upper right': (0.95, 0.95),
            'lower left': (0.05, 0.05),
            'lower right': (0.95, 0.05)
        }
        
        text_x, text_y = position_dict.get(stats_position, (0.05, 0.95))
        ha = 'left' if text_x < 0.5 else 'right'
        va = 'top' if text_y > 0.5 else 'bottom'
        
        ax.text(text_x, text_y, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor='gray', alpha=0.8),
               fontsize=font_size-1, ha=ha, va=va, family='monospace')
    
    # Customize appearance
    ax.set_xlabel(x_label or 'X Variable', fontsize=font_size)
    ax.set_ylabel(y_label or 'Y Variable', fontsize=font_size)
    
    if title:
        ax.set_title(title, fontsize=font_size, fontweight='bold', pad=20)
    
    # Grid
    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=font_size,
                  length=6, width=1.2, colors='black')
    
    # Legend if outliers are highlighted
    if highlight_outliers and np.any(outlier_mask):
        ax.legend(fontsize=font_size-1, loc='best', framealpha=0.9)
    
    # Only apply tight_layout and save if we created a new figure
    if created_new_figure:
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
    
    # Prepare return values
    stats_dict = None
    if return_stats:
        stats_dict = {
            'correlation': corr_coef,
            'p_value': p_value,
            'n_samples': len(x_clean),
            'method': method_name,
            'n_outliers': np.sum(outlier_mask) if outlier_threshold else 0,
            'outlier_indices': np.where(valid_mask)[0][outlier_mask] if outlier_threshold else []
        }
    
    # Return appropriate values based on whether we created a new figure
    if created_new_figure:
        if return_stats:
            return fig, ax, stats_dict
        else:
            return fig, ax
    else:
        if return_stats:
            return ax, stats_dict
        else:
            return ax