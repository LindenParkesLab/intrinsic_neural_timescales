import os, glob, pickle, pyfftw
import numpy as np
import scipy as sp
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
from nilearn.glm.first_level import make_first_level_design_matrix
pyfftw.interfaces.cache.enable()  # Cache FFT plans
from pyfftw.interfaces.numpy_fft import rfft, irfft

from src.data import DataLoader, parse_scan_string, build_scan_string

class Timescales(DataLoader):
    def __init__(
        self,
        dataset: str,
        timeseries_data: dict,
        load_behavior: bool = True,
        verbose: bool = True
    ):
        # Call the parent class's constructor
        super().__init__(dataset=dataset, load_behavior=load_behavior, verbose=verbose)
        
        self.timeseries_data = timeseries_data
        if timeseries_data is not None:
            self.scans = list(timeseries_data.keys())
        
        if verbose:
            print(f"\n\tInitialized Timescales for dataset: {dataset}")
            if timeseries_data is not None:
                print(f"\tNumber of scans: {len(self.scans)}")
            print(f"\tOutput directory: {self.output_dir}")

    # ============== Save/Load Infrastructure ==============
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

    def _parse_scan_string(self, scan_key: str) -> Dict[str, str]:
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
        use_residuals: bool = False,
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
        if use_residuals:
            output_type = 'residuals_' + output_type

        # Process each scan
        for scan_key in tqdm(self.scans, desc="Computing autocorrelation"):
            # Parse scan key using key-value format
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality = kv_dict.get('modality', '')
            condition = kv_dict.get('condition', '')
            scan_name = kv_dict.get('scan', '')
            atlas = kv_dict.get('atlas', '')
            tr = kv_dict.get('tr', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_acf = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    scan_name=scan_name,
                    modality=modality,
                    atlas=atlas
                )
                
                if loaded_acf is not None:
                    autocorr_dict[scan_key] = loaded_acf
                    n_loaded += 1
                    continue
            
            # Compute autocorrelation
            try:
                if use_residuals and condition == 'task':
                    # print('Using residuals...')
                    timeseries_scan = self.timeseries_residuals[scan_key]
                else:
                    timeseries_scan = self.timeseries_data[scan_key]

                acf_result = compute_acf(
                    timeseries_scan,
                    use_fft=use_fft,
                    freq_window=freq_window,
                    tr=tr,
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
                        modality=modality,
                        atlas=atlas,
                        metadata={
                            'use_fft': use_fft,
                            'freq_window': freq_window,
                            'filter_order': filter_order,
                            'tr': tr,
                            'modality': modality,
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
        suffix: str = None,
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
        if suffix:
            output_type = f'{output_type}_{suffix}'
        
        for scan_key in tqdm(self.scans, desc=f"Computing {method} timescales"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality = kv_dict.get('modality', '')
            scan_name = kv_dict.get('scan', '')
            atlas = kv_dict.get('atlas', '')
            tr = kv_dict.get('tr', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_timescales = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    scan_name=scan_name,
                    modality=modality,
                    atlas=atlas
                )
                
                if loaded_timescales is not None:
                    timescales_dict[scan_key] = loaded_timescales
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
                    timescale = self._compute_zero_crossing(acf) * tr
                elif method == 'auc_zero_crossing':
                    # Area under curve
                    timescale = self._compute_auc(acf, tr, max_lag=None)
                elif method == 'auc_100':
                    # Area under curve
                    timescale = self._compute_auc(acf, tr, max_lag=100)
                elif method == 'decay_time_1e':
                    # Time to decay to 1/e
                    threshold = 1 / np.e
                    timescale = self._compute_decay_time(acf, tr, threshold)
                elif method == 'decay_time_half':
                    # Time to decay to 1/e
                    threshold = 0.5
                    timescale = self._compute_decay_time(acf, tr, threshold)
                elif method == 'exponential_tau':
                    fs = 1 / tr
                    timescale, _ = self._compute_tau_from_acf(acf, fs)
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
                        modality=modality,
                        atlas=atlas,
                        metadata={'method': method, 'tr': tr}
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
    
    def _compute_auc(self, acf: np.ndarray, tr: float, max_lag: int = None) -> np.ndarray:
        """Compute area under curve up to some max lag (or zero crossing)."""
        n_parcels = acf.shape[0]
        auc = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            if max_lag is None:
                max_lag_i = np.where(acf[i] < 0)[0][0]
            else:
                max_lag_i = max_lag

            try:
                auc[i] = np.trapz(acf[i, :max_lag_i], dx=tr)
            except:
                auc[i] = np.nan
        
        return auc
    
    def _compute_decay_time(self, acf: np.ndarray, tr: float, threshold: float = 1.0 / np.e) -> np.ndarray:
        """Find time when ACF decays to 1/e."""
        n_parcels = acf.shape[0]
        decay_times = np.zeros(n_parcels)
        # threshold = 1.0 / np.e
        
        for i in range(n_parcels):
            # Find where ACF drops below threshold
            below_threshold = np.where(acf[i] < threshold)[0]
            if len(below_threshold) > 0:
                decay_times[i] = below_threshold[0] * tr
            else:
                decay_times[i] = np.nan
        
        return decay_times

    def _compute_tau_from_acf(self, acf: np.ndarray, fs: float, max_lag_ms=None):
        """
        Compute neuronal timescale (tau) from pre-computed autocorrelation function.
        
        Based on Gao et al. 2020, Figure 1C methodology.
        
        Parameters:
        -----------
        acf : array-like
            Pre-computed autocorrelation function. Should be normalized (ACF[0] = 1).
        fs : float
            Sampling frequency in Hz
        max_lag_ms : float, optional
            Maximum lag to consider in milliseconds. If None, uses full ACF.
            
        Returns:
        --------
        tau : float or array
            Timescale(s) in seconds
        fit_info : dict
            Diagnostic information about the fit
        """
        # Ensure input is 2D (n_series, n_lags)
        n_parcels, n_lags = acf.shape
        
        # Create lag times in seconds
        lags = np.arange(n_lags) / fs
        
        # Limit to max_lag_ms if specified
        if max_lag_ms is not None:
            max_lag_samples = int(max_lag_ms * fs / 1000)
            max_lag_samples = min(max_lag_samples, n_lags)
            acf = acf[:, :max_lag_samples]
            lags = lags[:max_lag_samples]
            n_lags = max_lag_samples
        
        taus = np.zeros(n_parcels)
        fit_infos = []
        
        for i in range(n_parcels):
            acf_single = acf[i, :]
            
            tau, fit_info = fit_exponential_decay(lags, acf_single)
            taus[i] = tau
            fit_infos.append(fit_info)
        
        return taus, fit_infos

    # ==================== Deconfounding Functions ====================
    def regress_task_structure(self):
        timeseries_residuals = dict().fromkeys(self.timeseries_data)
        task_events = dict().fromkeys(self.timeseries_data)
        design_matrices = dict().fromkeys(self.timeseries_data)
        
        for scan_key in tqdm(self.scans, desc=f"Regressing task structure out of timeseries"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            condition = kv_dict.get('condition', '')
            modality = kv_dict.get('modality', '')
            tr = kv_dict.get('tr', '')
            
            if condition == 'task':
                try:
                    timeseries = self.timeseries_data[scan_key]
                    n_timepoints = timeseries.shape[1]

                    if modality == 'MEG':
                        hrf_model = None
                        drift_model = None
                    elif modality == 'MRI3T' or modality == 'MRI7T':
                        hrf_model = 'glover'
                        drift_model = 'cosine'

                    # Create design matrix using nilearn
                    task_events[scan_key] = self.get_task_events(scan_key)
                    design_matrices[scan_key] = create_design_matrix_nilearn(task_events[scan_key], n_timepoints, tr, 
                                                                 hrf_model=hrf_model, drift_model=drift_model
                                                                 )
                    
                    # Fit GLM using nilearn approach
                    timeseries_residuals[scan_key], predicted, betas = fit_glm(timeseries, design_matrices[scan_key])
                except:
                    timeseries_residuals[scan_key] = None
                    task_events[scan_key] = None
                    design_matrices[scan_key] = None
            else:
                timeseries_residuals[scan_key] = None
                task_events[scan_key] = None
                design_matrices[scan_key] = None

        self.timeseries_residuals = timeseries_residuals
        self.task_events = task_events
        self.design_matrices = design_matrices
    
    # ==================== Analysis Functions ====================        
    def match_registry_to_behavior(self):
        if not hasattr(self, 'scan_registry') or self.scan_registry is None:
            self.scan_registry = self.list_registered_scans()

        df_behavior_matched = pd.DataFrame(columns=self.df_behavior.columns)
        df_behavior_matched.index.name = 'subject_id'
        
        for subject_id in self.scan_registry['subject_id'].unique():
            subject_id = str(subject_id)
            if subject_id in self.df_behavior.index and subject_id not in df_behavior_matched.index:
                df_behavior_matched.loc[subject_id] = self.df_behavior.loc[subject_id]
            elif subject_id in self.df_behavior.index and subject_id in df_behavior_matched.index:
                pass
            
        self.df_behavior_matched = df_behavior_matched
        
    def get_precomputed_outputs(
        self,
        output_type: str = 'zero_crossing'
    ) -> Dict[str, Any]:
        """
        Get pre-computed measures.
        
        Args:
            method: Method to use ('zero_crossing', 'auc', 'decay_time')
        
        Returns:
            Dictionary of measures for each scan
        """
        if not hasattr(self, 'scan_registry') or self.scan_registry is None:
            self.scan_registry = self.list_registered_scans()

        data_dict = {}
        
        for scan_key in tqdm(self.scan_registry['scan_string'], desc=f"Loading {output_type}"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality = kv_dict.get('modality', '')
            scan_name = kv_dict.get('scan', '')
            atlas = kv_dict.get('atlas', '')
            
            # Check for existing output
            loaded_data = self._load_subject_output(
                subject_id=subject_id,
                output_type=output_type,
                scan_name=scan_name,
                modality=modality,
                atlas=atlas
            )
            
            data_dict[scan_key] = loaded_data
            
        self.data_dict = data_dict

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

def create_design_matrix_nilearn(
    events,
    n_timepoints,
    tr,
    hrf_model='glover',
    drift_model='cosine',
    high_pass=0.01,
    add_regs=None
    ):
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

def fit_glm(data, design_matrix):
    """
    Fit GLM.
    
    Parameters:
    - data: 2D array (n_parcels, n_timepoints)
    - design_matrix: nilearn DesignMatrix object
    
    Returns:
    - residuals: residual time series after GLM fitting
    - predicted: predicted time series
    - betas: regression coefficients
    """
    
    # Convert design matrix to numpy array
    X = design_matrix.values
    
    # Fit GLM using least squares
    betas, residuals_sum, rank, s = np.linalg.lstsq(X, data.T, rcond=None)
    
    # Compute predicted and residual signals
    predicted = np.dot(X, betas).T
    residuals = data - predicted
    
    return residuals, predicted, betas.T

def fit_exponential_decay(lags, acf):
    """
    Fit exponential decay function to ACF: ACF(t) = exp(-t/tau)
    
    This is done by linear regression in log space: log(ACF) = -t/tau + const
    Following Gao et al. 2020 methodology.
    """
    
    # Find first zero crossing to limit fit range (positive ACF only)
    negative_idx = np.where(acf <= 0)[0]
    if len(negative_idx) > 0:
        end_idx = negative_idx[0]
    else:
        end_idx = len(acf)
    
    # Exclude lag 0 and fit only positive values
    start_idx = 1  # Skip lag 0 (ACF=1)
    lags_fit = lags[start_idx:end_idx]
    acf_fit = acf[start_idx:end_idx]
    
    # Remove any remaining non-positive values
    valid_idx = acf_fit > 0
    lags_fit = lags_fit[valid_idx]
    acf_fit = acf_fit[valid_idx]
    
    try:
        # Linear fit in log space: log(ACF) = -t/tau + const
        # So slope = -1/tau, therefore tau = -1/slope
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(lags_fit, np.log(acf_fit))
        
        if slope >= 0:
            # Non-decaying or increasing ACF - invalid
            tau = np.nan
            error = 'positive_slope'
        else:
            tau = -1 / slope
            error = None
        
        fit_info = {
            'tau': tau,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'n_points': len(lags_fit),
            'fit_range': [lags_fit[0], lags_fit[-1]],
            'lags_fit': lags_fit,
            'acf_fit': acf_fit,
            'log_acf_fit': np.log(acf_fit),
            'error': error
        }
    except (ValueError, RuntimeWarning) as e:
        # Fitting failed (e.g., log of negative numbers)
        tau = np.nan
        fit_info = {
            'tau': np.nan,
            'r_squared': np.nan,
            'n_points': len(lags_fit),
            'fit_range': [lags_fit[0] if len(lags_fit) > 0 else np.nan, 
                         lags_fit[-1] if len(lags_fit) > 0 else np.nan],
            'error': f'fitting_failed: {str(e)}'
        }
    
    return tau, fit_info 