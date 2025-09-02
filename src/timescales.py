import os
import numpy as np
import scipy as sp
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm

import pyfftw
pyfftw.interfaces.cache.enable()  # Cache FFT plans
from pyfftw.interfaces.numpy_fft import rfft, irfft

from src.data import DataLoader, parse_scan_string, build_scan_string

class Timescales(DataLoader):
    def __init__(
        self,
        dataset: str,
        timeseries_data: dict,
        modality: str,
        load_behavior: bool = True
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
        print(f"\tNumber of scans: {len(self.scans)}")
        print(f"\tOutput directory: {self.output_dir}")

    # ============== Save/Load Infrastructure with Key-Value Naming ==============
    
    def _get_output_filepath(
        self,
        subject_id: str,
        output_type: str,
        scan_name: str,
        modality: str,
        atlas: str,
        suffix: str = '.npy',
        mode: str = 'load'
    ) -> Path:
        """
        Generate standardized filepath for subject-specific outputs using key-value naming.
        
        Args:
            subject_id: Subject identifier
            output_type: Type of output (e.g., 'autocorr', 'timescale_acf')
            scan_name: Name of scan (e.g., 'rfMRIREST1LR')
            modality: Modality string (e.g., 'MRI3T')
            atlas: Atlas name
            suffix: File extension
        
        Returns:
            Path object for the output file
        """
        # Create output subdirectory structure
        output_subdir = self.output_dir / 'timescales_outputs' / self.dataset / output_type
        if mode == 'save':
            output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Build filename using key-value format
        filename = build_scan_string(
            sub=subject_id,
            modality=modality,
            scan=scan_name,
            atlas=atlas,
            stat=output_type
        ) + suffix
        
        return output_subdir / filename

    def _save_subject_output(
        self,
        data: Any,
        subject_id: str,
        output_type: str,
        scan_name: str,
        modality: str,
        atlas: str = None,
        metadata: Dict = None,
        use_pickle: bool = False
    ) -> Path:
        """Save subject-specific output data with key-value naming."""
        suffix = '.pkl' if use_pickle else '.npy'
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            scan_name=scan_name,
            modality=modality,
            atlas=atlas,
            suffix=suffix,
            mode='save'
        )
        
        # Save data
        if use_pickle:
            with open(filepath, 'wb') as f:
                pickle.dump({'data': data, 'metadata': metadata}, f)
        else:
            if metadata is not None:
                # Save as npz with metadata
                np.savez(filepath.with_suffix('.npz'), 
                        data=data, 
                        metadata=metadata)
            else:
                np.save(filepath, data)
        
        return filepath

    def _load_subject_output(
        self,
        subject_id: str,
        output_type: str,
        scan_name: str,
        modality: str,
        atlas: str = None,
        use_pickle: bool = False
    ) -> Optional[Any]:
        """Load previously saved subject-specific output with key-value naming."""
        suffix = '.pkl' if use_pickle else '.npy'
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            scan_name=scan_name,
            modality=modality,
            atlas=atlas,
            suffix=suffix,
            mode='load'
        )
        
        # Check for .npz if .npy doesn't exist
        if not filepath.exists() and suffix == '.npy':
            filepath = filepath.with_suffix('.npz')
        
        if not filepath.exists():
            return None
        
        # Load data
        try:
            if use_pickle:
                with open(filepath, 'rb') as f:
                    loaded = pickle.load(f)
                    return loaded.get('data', loaded)
            elif filepath.suffix == '.npz':
                loaded = np.load(filepath, allow_pickle=True)
                return loaded['data']
            else:
                return np.load(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def _check_output_exists(
        self,
        subject_id: str,
        output_type: str,
        scan_name: str,
        modality: str,
        atlas: str = None
    ) -> bool:
        """Check if a subject output file already exists."""
        for suffix in ['.npy', '.npz', '.pkl']:
            filepath = self._get_output_filepath(
                subject_id=subject_id,
                output_type=output_type,
                scan_name=scan_name,
                modality=modality,
                atlas=atlas,
                suffix=suffix,
                moad='load'
            )
            if filepath.exists():
                return True
        return False

    def _parse_scan_key(self, scan_key: str) -> Dict[str, str]:
        """
        Parse scan key using key-value format.
        
        Args:
            scan_key: String like "sub-100307_modality-MRI3T_scan-rfMRIREST1LR_atlas-Schaefer4007"
        
        Returns:
            Dictionary with parsed values
        """
        return parse_scan_string(scan_key)


    # ============== Computation Methods ==============
    
    def compute_autocorrelation_function(
        self,
        use_fft: bool = False,
        freq_window: tuple = None,
        filter_order: int = 4,
        force_recompute: bool = False,
        save_outputs: bool = True,
        verbose: bool = True
    ):
        """
        Compute autocorrelation function for all scans with caching.
        
        """
        autocorr_dict = dict.fromkeys(self.scans)
        
        # Track what was loaded vs computed
        n_loaded = 0
        n_computed = 0
        n_failed = 0
        if freq_window is None:
            output_type = 'autocorr'
        else:
            output_type = f'autocorr_freq-{freq_window[0]}hz-{freq_window[1]}hz'

        # Process each scan
        for scan_key in tqdm(self.scans, desc="Computing autocorrelation"):
            # Parse scan key using key-value format
            kv_dict = self._parse_scan_key(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality_str = kv_dict.get('modality', '')
            scan_name = kv_dict.get('scan', '')
            atlas = kv_dict.get('atlas', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_acf = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    scan_name=scan_name,
                    modality=modality_str,
                    atlas=atlas
                )
                
                if loaded_acf is not None:
                    autocorr_dict[scan_key] = loaded_acf
                    n_loaded += 1
                    continue
            
            # Compute autocorrelation
            try:
                acf_result = compute_acf(
                    self.timeseries_data[scan_key],
                    use_fft=use_fft,
                    freq_window=freq_window,
                    tr=self.tr,
                    filter_order=filter_order
                )
                
                autocorr_dict[scan_key] = acf_result
                n_computed += 1
                
                # Save output if requested
                if save_outputs:
                    self._save_subject_output(
                        data=acf_result,
                        subject_id=subject_id,
                        output_type=output_type,
                        scan_name=scan_name,
                        modality=modality_str,
                        atlas=atlas,
                        metadata={
                            'use_fft': use_fft,
                            'freq_window': freq_window,
                            'filter_order': filter_order,
                            'tr': self.tr,
                            'modality': self.modality,
                            'dataset': self.dataset
                        }
                    )
            
            except Exception as e:
                print(f"Warning: Failed to compute autocorrelation for {scan_key}: {e}")
                autocorr_dict[scan_key] = None
                n_failed += 1
        
        # Store results
        self.autocorr_dict = autocorr_dict
        
        if verbose:
            print(f"\nAutocorrelation computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed: {n_computed}")
            if n_failed > 0:
                print(f"  Failed: {n_failed}")
            print(f"  Total: {len(self.scans)}")

    def compute_timescale_measures(
        self,
        method: str = 'zero_crossing',
        force_recompute: bool = False,
        save_outputs: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute timescale measures from autocorrelation functions.
        
        Args:
            method: Method to use ('zero_crossing', 'auc', 'decay_time')
            force_recompute: Force recomputation even if saved
            save_outputs: Save computed outputs
            verbose: Print progress
        
        Returns:
            Dictionary of timescale measures for each scan
        """
        if not hasattr(self, 'autocorr_dict') or self.autocorr_dict is None:
            raise ValueError("Must compute autocorrelation first")
        
        timescales_dict = {}
        n_loaded = 0
        n_computed = 0
        output_type = f'timescale_{method}'
        
        for scan_key in tqdm(self.scans, desc=f"Computing {method} timescales"):
            # Parse scan key
            kv_dict = self._parse_scan_key(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality_str = kv_dict.get('modality', '')
            scan_name = kv_dict.get('scan', '')
            atlas = kv_dict.get('atlas', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_ts = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    scan_name=scan_name,
                    modality=modality_str,
                    atlas=atlas
                )
                
                if loaded_ts is not None:
                    timescales_dict[scan_key] = loaded_ts
                    n_loaded += 1
                    continue
            
            # Compute timescale measure
            if self.autocorr_dict.get(scan_key) is None:
                timescales_dict[scan_key] = None
                continue
                
            acf = self.autocorr_dict[scan_key]
            
            try:
                if method == 'zero_crossing':
                    # Find first zero crossing for each parcel
                    timescale = self._compute_zero_crossing(acf)
                elif method == 'auc_zero_crossing':
                    # Area under curve
                    timescale = self._compute_auc(acf, max_lag=None)
                elif method == 'auc_100':
                    # Area under curve
                    timescale = self._compute_auc(acf, max_lag=100)
                elif method == 'decay_time_1e':
                    # Time to decay to 1/e
                    threshold = 1 / np.e
                    timescale = self._compute_decay_time(acf, threshold)
                elif method == 'decay_time_half':
                    # Time to decay to 1/e
                    threshold = 0.5
                    timescale = self._compute_decay_time(acf, threshold)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                timescales_dict[scan_key] = timescale
                n_computed += 1
                
                # Save output
                if save_outputs:
                    self._save_subject_output(
                        data=timescale,
                        subject_id=subject_id,
                        output_type=output_type,
                        scan_name=scan_name,
                        modality=modality_str,
                        atlas=atlas,
                        metadata={'method': method, 'tr': self.tr}
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to compute {method} for {scan_key}: {e}")
                timescales_dict[scan_key] = None

        self.timescales_dict = timescales_dict
        
        if verbose:
            print(f"\n{method} timescale computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed: {n_computed}")
            print(f"  Total: {len(self.scans)}")

    # ============== Timescale Computation Methods ==============
    
    def _compute_zero_crossing(self, acf: np.ndarray) -> np.ndarray:
        """Find first zero crossing for each parcel."""
        n_parcels = acf.shape[0]
        zero_crossings = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            negative_indices = np.where(acf[i] < 0)[0]
            if len(negative_indices) > 0:
                zero_crossings[i] = negative_indices[0]
            else:
                zero_crossings[i] = np.nan
        
        return zero_crossings
    
    def _compute_auc(self, acf: np.ndarray, max_lag: int = None) -> np.ndarray:
        """Compute area under curve up to some max lag (or zero crossing)."""
        n_parcels = acf.shape[0]
        auc = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            if max_lag is None:
                max_lag_i = np.where(acf[i] < 0)[0][0]
            else:
                max_lag_i = max_lag

            try:
                auc[i] = np.trapz(acf[i, :max_lag_i], dx=self.tr)
            except:
                auc[i] = np.nan
        
        return auc
    
    def _compute_decay_time(self, acf: np.ndarray, threshold: float = 1.0 / np.e) -> np.ndarray:
        """Find time when ACF decays to 1/e."""
        n_parcels = acf.shape[0]
        decay_times = np.zeros(n_parcels)
        # threshold = 1.0 / np.e
        
        for i in range(n_parcels):
            # Find where ACF drops below threshold
            below_threshold = np.where(acf[i] < threshold)[0]
            if len(below_threshold) > 0:
                decay_times[i] = below_threshold[0] * self.tr
            else:
                decay_times[i] = np.nan
        
        return decay_times

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