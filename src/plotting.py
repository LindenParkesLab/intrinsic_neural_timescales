import os, sys, time, getpass
import numpy as np
import pandas as pd
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