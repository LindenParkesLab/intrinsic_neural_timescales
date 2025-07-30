import os, sys, time, getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nilearn.glm.first_level import make_first_level_design_matrix, compute_regressor
from nilearn.glm.first_level import FirstLevelModel

username = getpass.getuser()
if username == 'lindenmp':
    sys.path.extend(['/home/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lindenmp/research_projects/nctpy/src'])
elif username == 'lp756':
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/snaplab_tools'])
    sys.path.extend(['/home/lp756/projects/f_lp756_1/lindenmp/research_projects/nctpy/src'])

from snaplab_tools.derivs import compute_acf, compute_alff, compute_rlfp

def compute_autocorrelation_function(data):
    """
    Compute autocorrelation function for time series data.
    
    Parameters:
    - data: 2D array (voxels x timepoints) or 1D array (single time series)
    
    Returns:
    - autocorr: autocorrelation function(s)
    - zero_crossings: first zero-crossing indices
    """
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    autocorr_list = []
    zero_crossings = []
    
    for voxel_ts in data:
        if np.std(voxel_ts) > 0:  # Avoid division by zero
            # Standardize the time series
            standardized = (voxel_ts - np.mean(voxel_ts)) / np.std(voxel_ts)
            
            # Compute full autocorrelation
            acf = np.correlate(standardized, standardized, mode='full')
            acf = acf[len(acf)//2:]  # Take only non-negative lags
            acf = acf / acf[0]  # Normalize by lag-0 value
            
            autocorr_list.append(acf)
            
            # Find first zero-crossing
            negative_indices = np.where(acf < 0)[0]
            if len(negative_indices) > 0:
                zero_crossings.append(negative_indices[0])
            else:
                zero_crossings.append(len(acf) - 1)  # No zero crossing found
        else:
            # Handle constant time series
            n_timepoints = len(voxel_ts)
            autocorr_list.append(np.ones(n_timepoints))
            zero_crossings.append(n_timepoints - 1)
    
    return np.array(autocorr_list), np.array(zero_crossings)


def compute_timescales(timeseries, which_version='acf', return_mean=True, tr=np.nan):
    n_parcels = timeseries.shape[1]
    n_subs = timeseries.shape[-1]
    print(n_subs, n_parcels)
    timescales = np.zeros((n_parcels, n_subs))
    for i in tqdm(np.arange(n_subs)):
        if which_version == 'acf':
            try:
                autocorr_raw, timescales[:, i] = compute_autocorrelation_function(timeseries[:, :, i].T)
            except:
                timescales[:, i] = np.nan
        elif which_version == 'alff':
            try:
                timescales[:, i] = compute_alff(timeseries[:, :, i].T, low_pass=0.08, high_pass=0.01, TR=tr, sample_mask=None)
            except:
                timescales[:, i] = np.nan
        else:
            for j in np.arange(n_parcels):
                try:
                    timescales[j, i] = compute_rlfp(timeseries[:, j, i], tr=tr, low=0.01, high=0.08)
                    # time_series = signal.detrend(timeseries[:, j, i])
                    # if which_version == 'acf':
                    #     _, timescales[j, i] = compute_acf(timeseries[:, j, i], nlags=20)
                    # elif which_version == 'rlfp':
                    #     timescales[j, i] = compute_rlfp(timeseries[:, j, i], tr=tr, low=0.01, high=0.08)
                except:
                    timescales[j, i] = np.nan
    if return_mean:
        return np.nanmean(timescales, axis=1)
    else:
        return timescales


def get_task_events(ev_dir):
    task_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    conditions = os.listdir(ev_dir)
    for condition in conditions:
        task_events_tmp = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

        try:
            ev_data = np.genfromtxt(os.path.join(ev_dir, condition))

            if ev_data.ndim == 1:
                task_events_tmp.loc[0, 'onset'] = ev_data[0]
                task_events_tmp.loc[0, 'duration'] = ev_data[1]
                task_events_tmp.loc[0, 'trial_type'] = condition.replace('.txt', '')

            elif ev_data.ndim == 2:
                for block in np.arange(ev_data.shape[0]):
                    task_events_tmp.loc[block, 'onset'] = ev_data[block, 0]
                    task_events_tmp.loc[block, 'duration'] = ev_data[block, 1]
                    task_events_tmp.loc[block, 'trial_type'] = condition.replace('.txt', '')
        except:
            pass
        
        task_events = pd.concat((task_events, task_events_tmp), axis=0)

    task_events.reset_index(inplace=True, drop=True)    
    
    return task_events

    
def create_design_matrix_nilearn(events, n_timepoints, tr, hrf_model='glover', drift_model='cosine', 
                                 high_pass=0.01, add_regs=None):
    """
    Create design matrix using nilearn's implementation.
    
    Parameters:
    - events: pandas DataFrame with events (onset, duration, trial_type)
    - n_timepoints: number of timepoints
    - tr: repetition time in seconds
    - hrf_model: HRF model ('glover', 'spm', 'glover + derivative', etc.)
    - drift_model: drift model ('polynomial', 'cosine', None)
    - high_pass: high-pass filter cutoff in Hz
    - add_regs: additional regressors (optional)
    
    Returns:
    - design_matrix: nilearn DesignMatrix object
    """
    
    # Create frame times
    frame_times = np.arange(n_timepoints) * tr
    
    # Use nilearn to create design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=add_regs
    )
    
    return design_matrix

def fit_glm(fmri_data, design_matrix):
    """
    Fit GLM.
    
    Parameters:
    - fmri_data: 2D array (voxels x timepoints)
    - design_matrix: nilearn DesignMatrix object
    
    Returns:
    - residuals: residual time series after GLM fitting
    - predicted: predicted time series
    - betas: regression coefficients
    """
    
    try:
        # Convert design matrix to numpy array
        X = design_matrix.values
        
        # Fit GLM using least squares
        betas, residuals_sum, rank, s = np.linalg.lstsq(X, fmri_data.T, rcond=None)
        
        # Compute predicted and residual signals
        predicted = np.dot(X, betas).T
        residuals = fmri_data - predicted
        
        return residuals, predicted, betas.T
        
    except np.linalg.LinAlgError as e:
        # GLM fitting failed, use original data as residuals
        return fmri_data.copy(), np.zeros_like(fmri_data), np.zeros((fmri_data.shape[0], design_matrix.shape[1]))


def compute_autocorrelation_with_hrf(fmri_data, events, tr, mask=None, hrf_model='glover', drift_model='cosine'):
    """
    Compute autocorrelation of fMRI data using nilearn for block design analysis.
    
    Parameters:
    - fmri_data: 4D numpy array (x,y,z,time) or 2D array (voxels,time)
    - events: pandas DataFrame with events (onset, duration, trial_type) or path to CSV
    - tr: repetition time in seconds
    - mask: binary mask for voxel selection (optional)
    - hrf_model: HRF model for nilearn ('glover', 'spm', 'glover + derivative', etc.)
    - drift_model: drift model for nilearn ('cosine', 'polynomial', None)
    
    Returns:
    - autocorr_raw: autocorrelation before task removal (n_voxels x n_lags)
    - autocorr_resid: autocorrelation after task removal (n_voxels x n_lags)  
    - zero_crossings: first zero-crossing indices (n_voxels,)
    - residuals: residual time series after GLM (n_voxels x n_timepoints)
    - design_matrix: nilearn DesignMatrix object
    """

    # Handle input data shape
    if fmri_data.ndim == 4:
        original_shape = fmri_data.shape
        n_voxels = np.prod(original_shape[:3])
        n_timepoints = original_shape[3]
        fmri_data_2d = fmri_data.reshape((n_voxels, n_timepoints))
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            fmri_data_2d = fmri_data_2d[mask_flat > 0, :]
    elif fmri_data.ndim == 2:
        fmri_data_2d = fmri_data.copy()
        n_timepoints = fmri_data_2d.shape[1]
    else:
        raise ValueError("fMRI data must be 2D (voxels x time) or 4D (x,y,z,time)")

    # Compute raw autocorrelation (before task removal)
    autocorr_raw, zero_crossings_raw = compute_autocorrelation_function(fmri_data_2d)

    # Create design matrix using nilearn
    design_matrix = create_design_matrix_nilearn(events, n_timepoints, tr, hrf_model=hrf_model, drift_model=drift_model)
    
    # Fit GLM using nilearn approach
    residuals, predicted, betas = fit_glm(fmri_data_2d, design_matrix)
    
    # Compute autocorrelation on residuals
    autocorr_resid, zero_crossings_resid = compute_autocorrelation_function(residuals)

    return autocorr_raw, zero_crossings_raw, autocorr_resid, zero_crossings_resid, residuals, design_matrix


