import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import pyfftw
pyfftw.interfaces.cache.enable()  # Cache FFT plans
from pyfftw.interfaces.numpy_fft import rfft, irfft

from src.data import DataLoader

class Timescales(DataLoader):
    def __init__(
        self,
        dataset: str,
        timeseries_data: list,
        modality: str,
        load_behavior: bool = True,
    ):
        # Call the parent class's constructor
        super().__init__(dataset=dataset, load_behavior=load_behavior)
        
        self.modality = modality
        self.tr = self.get_tr(modality=modality)
        self.timeseries_data = timeseries_data
        self.scans = list(timeseries_data.keys())
        
        print(f"\n\tInitialized Timescales for dataset: {dataset}")
        print(f"\tModality: {self.modality}")
        print(f"\tTR: {self.tr}")

    def compute_autocorrelation_function(
        self,
        use_fft: bool = False,
        freq_window: tuple = None,
        filter_order: int = 4
    ):
        autocorr_dict = dict.fromkeys(self.scans)
        
        for scan in tqdm(autocorr_dict.keys()):
            autocorr_dict[scan] = compute_acf(self.timeseries_data[scan],
                                              use_fft=use_fft, freq_window=freq_window,
                                              tr=self.tr, filter_order=filter_order)
            
        self.autocorr = autocorr_dict

     
# ==================== Standalone Functions ====================

def compute_acf(
        data,
        use_fft: bool = False,
        freq_window: tuple = None,
        tr: float = None, 
        filter_order: int = 4
    ):
    """
    Compute autocorrelation function for time series data with optional frequency filtering.
    
    Parameters:
    -----------
    data : array-like
        2D array (n_parcels, n_timepoints) or 1D array (single time series)
    use_fft : bool, default=False
        Whether to use FFT for computation
    freq_window : tuple, optional
        Frequency band (low_freq, high_freq) in Hz for filtering.
        Requires tr to be specified.
        Time-domain filtering is applied before ACF computation.
    tr : float, optional
        Acquisition timing. sampling rate in Hz is 1 / tr
    filter_order : int, default=4
        Order of Butterworth filter for frequency filtering
        
    Returns:
    --------
    autocorr : ndarray
        Autocorrelation function(s)
    """
    
    # Validation
    if freq_window is not None and tr is None:
        raise ValueError("tr must be specified when using freq_window")
    elif freq_window is not None and tr is not None:
        sampling_rate = 1 / tr
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    autocorr_list = []
    
    for timeseries in data:
        if np.std(timeseries) > 0:  # Avoid division by zero
            
            # Apply frequency filtering BEFORE computing autocorrelation
            if freq_window is not None:
                timeseries_filtered = apply_bandpass_filter(timeseries, sampling_rate, freq_window, filter_order)
            else:
                timeseries_filtered = timeseries.copy()
            
            # Standardize the (potentially filtered) time series
            standardized = (timeseries_filtered - np.mean(timeseries_filtered)) / np.std(timeseries_filtered)
            
            if use_fft:
                n = len(standardized)
                # Pad to next power of 2 for optimal FFT performance
                fft_size = 2 ** (int(np.log2(2 * n - 1)) + 1)
                padded = np.zeros(fft_size)
                padded[:n] = standardized
                
                # Real FFT for autocorrelation
                fft_data = pyfftw.interfaces.numpy_fft.rfft(padded)
                acf_full = pyfftw.interfaces.numpy_fft.irfft(fft_data * np.conj(fft_data))
                
                # Take positive lags and normalize
                acf = acf_full[:n]
            else:
                # Compute full autocorrelation
                acf = np.correlate(standardized, standardized, mode='full')
                acf = acf[len(acf)//2:]  # Take only non-negative lags
            
            acf = acf / acf[0]  # Normalize by lag-0 value
            autocorr_list.append(acf)
            
    return np.array(autocorr_list)


def apply_bandpass_filter(
    timeseries,
    sampling_rate: float,
    freq_window: tuple,
    filter_order: int = 4
    ):
    """
    Apply bandpass filter to time series using Butterworth filter.
    
    Parameters:
    -----------
    timeseries : ndarray
        1D time series data
    sampling_rate : float
        Sampling rate in Hz
    freq_window : tuple
        (low_freq, high_freq) in Hz
    filter_order : int
        Order of Butterworth filter
        
    Returns:
    --------
    timeseries_filtered : ndarray
        Filtered time series
    """
    low_freq, high_freq = freq_window
    nyquist = sampling_rate / 2.0
    
    # Normalize frequencies to Nyquist frequency
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low_norm = max(low_norm, 0.001)  # Avoid DC component issues
    high_norm = min(high_norm, 0.999)  # Avoid Nyquist frequency issues
    
    # Design Butterworth bandpass filter
    b, a = sp.signal.butter(filter_order, [low_norm, high_norm], btype='band')
    
    # Apply filter using filtfilt for zero-phase filtering
    timeseries_filtered = sp.signal.filtfilt(b, a, timeseries)
    
    return timeseries_filtered