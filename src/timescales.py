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
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
from collections import defaultdict
from joblib import Parallel, delayed

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
        task: str,
        modality: str,
        seg: str,
        suffix: str = '.npy',
        mode: str = 'load'
    ) -> Path:
        """
        Generate standardized filepath for subject-specific outputs using key-value naming.
        
        Args:
            subject_id: Subject identifier
            output_type: Type of output (e.g., 'autocorr', 'timescale_acf')
            task: Name of scan (e.g., 'rfMRIREST1LR')
            modality: Modality string (e.g., 'MRI3T')
            seg: Atlas name
            suffix: File extension
        
        Returns:
            Path object for the output file
        """
        # Create output subdirectory structure
        output_subdir = self.output_dir / 'data' / 'timescales_outputs' / self.dataset / output_type
        if mode == 'save':
            output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Build filename using key-value format
        filename = build_scan_string(
            sub=subject_id,
            modality=modality,
            task=task,
            seg=seg,
            stat=output_type
        ) + suffix
        
        return output_subdir / filename

    def _save_subject_output(
        self,
        data: Any,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None,
        metadata: Dict = None,
    ) -> Path:
        """Save subject-specific output data with key-value naming."""
        if data.ndim == 1:
            suffix = '.txt'
        elif data.ndim > 1:
            suffix = '.npy'
        
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            task=task,
            modality=modality,
            seg=seg,
            suffix=suffix,
            mode='save'
        )
        
        # Save data
        if suffix == '.npy' and metadata is not None:
            np.savez(filepath.with_suffix('.npz'), 
                     data=data, metadata=metadata)
        elif suffix == '.npy' and metadata is None:
            np.save(filepath, data)
        elif suffix == '.txt':
            np.savetxt(filepath, data)
        
        return filepath

    def _load_subject_output(
        self,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None,
    ) -> Optional[Any]:
        """Load previously saved subject-specific output with key-value naming."""
        suffix = '.npy'
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            task=task,
            modality=modality,
            seg=seg,
            suffix=suffix,
            mode='load'
        )
        
        # Check for .npz if .npy doesn't exist
        if not filepath.exists() and suffix == '.npy':
            filepath = filepath.with_suffix('.npz')
            # if that doesn't find the file, try .txt
            if not filepath.exists():
                filepath = filepath.with_suffix('.txt')
        
        if not filepath.exists():
            return None
        
        # Load data
        try:
            if filepath.suffix == '.npz':
                loaded = np.load(filepath, allow_pickle=True)
                return loaded['data']
            elif filepath.suffix == '.npy':
                return np.load(filepath)
            elif filepath.suffix == '.txt':
                return np.loadtxt(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def _check_output_exists(
        self,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None
    ) -> bool:
        """Check if a subject output file already exists."""
        for suffix in ['.npy', '.npz', '.pkl']:
            filepath = self._get_output_filepath(
                subject_id=subject_id,
                output_type=output_type,
                task=task,
                modality=modality,
                seg=seg,
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
        force_recompute: bool = False,
        save_outputs: bool = True,
        suffix: str = None,
        use_residuals: bool = False,
        lowpass_cutoff: float = None,
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
        if lowpass_cutoff is None:
            output_type = 'autocorr'
        else:
            output_type = f'autocorr_lp-{lowpass_cutoff}hz'
        if suffix:
            output_type = f'{output_type}_{suffix}'
        if use_residuals:
            output_type = output_type + '_resids'

        # Process each scan
        for scan_key in tqdm(self.scans, desc="Computing autocorrelation"):
            # Parse scan key using key-value format
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality = kv_dict.get('modality', '')
            condition = kv_dict.get('condition', '')
            task = kv_dict.get('task', '')
            seg = kv_dict.get('seg', '')
            tr = kv_dict.get('tr', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_acf = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    task=task,
                    modality=modality,
                    seg=seg
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

                if lowpass_cutoff is not None:
                    sampling_freq = 1 / tr
                    timeseries_scan = apply_lowpass_filter(timeseries_scan, sampling_freq, lowpass_cutoff)
                acf_result = compute_acf(timeseries_scan, use_fft=use_fft)

                autocorr_dict[scan_key] = acf_result
                n_computed += 1
                
                # Save output if requested
                if save_outputs:
                    self._save_subject_output(
                        data=acf_result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        metadata={
                            'use_fft': use_fft,
                            'lowpass_cutoff': lowpass_cutoff,
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
        use_residuals: bool = False,
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
        if method == 'exponential_tau':
            tau_fit_infos = {}
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
            condition = kv_dict.get('condition', '')
            task = kv_dict.get('task', '')
            seg = kv_dict.get('seg', '')
            tr = kv_dict.get('tr', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_timescales = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    task=task,
                    modality=modality,
                    seg=seg
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
                    timescales = self._compute_zero_crossing(acf) * tr
                elif method == 'auc_zero_crossing':
                    # Area under curve
                    timescales = self._compute_auc(acf, tr, max_lag=None)
                elif method == 'decay_time_1e':
                    # Time to decay to 1/e
                    threshold = 1 / np.e
                    timescales = self._compute_decay_time(acf, tr, threshold)
                elif method == 'decay_time_half':
                    # Time to decay to 1/e
                    threshold = 0.5
                    timescales = self._compute_decay_time(acf, tr, threshold)
                elif method == 'tau':
                    # Compute tau for each parcel
                    timescales = self._compute_tau_from_acf(acf, tr, version='loglin')
                elif method == 'tau_murray':
                    # Compute tau for each parcel using Murray et al. 2014
                    timescales, _ = self._compute_tau_from_acf(acf, tr, version='murray')
                elif method == 'B_murray':
                    # Compute B for each parcel using Murray et al. 2014
                    _, timescales = self._compute_tau_from_acf(acf, tr, version='murray')
                elif method == 'tau_murray_par':
                    # Compute tau for each parcel using Murray et al. 2014
                    timescales, _ = self._compute_tau_murray_parallel(acf, tr, int(os.cpu_count() / 2))
                elif method == 'B_murray_par':
                    # Compute B for each parcel using Murray et al. 2014
                    _, timescales = self._compute_tau_murray_parallel(acf, tr, int(os.cpu_count() / 2))
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                timescales_dict[scan_key] = timescales
                n_computed += 1
                
                # Save output
                if save_outputs:
                    self._save_subject_output(
                        data=timescales,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        metadata={'method': method, 'tr': tr}
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to compute {method} for {scan_key}: {e}")
                timescales_dict[scan_key] = None

        self.timescales_dict = timescales_dict
        if method == 'exponential_tau':
            self.tau_fit_infos = tau_fit_infos

        if verbose:
            print(f"\n{method} timescale computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed: {n_computed}")
            print(f"  Total: {len(self.scans)}")

    def compute_timeseries_measures(
        self,
        method: str = 'tsnr',
        force_recompute: bool = False,
        save_outputs: bool = True,
        use_residuals: bool = False,
        verbose: bool = True
    ):
        """
        Compute temporal signal-to-noise ratio (tSNR) for all scans with caching.
        
        Parameters:
        -----------
        force_recompute : bool, default=False
            If True, recompute even if cached results exist
        save_outputs : bool, default=True
            If True, save computed results to disk
        verbose : bool, default=True
            If True, print progress information
        """
        data_dict = dict.fromkeys(self.scans)
        
        # Track what was loaded vs computed
        n_loaded = 0
        n_computed = 0
        n_failed = 0
        output_type = f'metric_{method}'
        if use_residuals:
            output_type = output_type + '_resids'

        # Process each scan
        for scan_key in tqdm(self.scans, desc=f"Computing {method}"):
            # Parse scan key using key-value format
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality = kv_dict.get('modality', '')
            condition = kv_dict.get('condition', '')
            task = kv_dict.get('task', '')
            seg = kv_dict.get('seg', '')
            tr = kv_dict.get('tr', '')
            
            # Check for existing output
            if not force_recompute:
                loaded_metric = self._load_subject_output(
                    subject_id=subject_id,
                    output_type=output_type,
                    task=task,
                    modality=modality,
                    seg=seg
                )
                
                if loaded_metric is not None:
                    data_dict[scan_key] = loaded_metric
                    n_loaded += 1
                    continue
            
            # Compute timeseries measure
            try:
                if use_residuals and condition == 'task':
                    # print('Using residuals...')
                    timeseries_scan = self.timeseries_residuals[scan_key]
                else:
                    timeseries_scan = self.timeseries_data[scan_key]

                if method == 'tsnr':
                    timeseries_result = calculate_tsnr(timeseries_scan)
                elif method == 'var':
                    timeseries_result = np.var(timeseries_scan, axis=1)
                
                data_dict[scan_key] = timeseries_result
                n_computed += 1
                
                # Save output if requested
                if save_outputs:
                    self._save_subject_output(
                        data=timeseries_result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        metadata={
                            'tr': tr,
                            'modality': modality,
                            'dataset': self.dataset
                        }
                    )
            
            except Exception as e:
                print(f"Warning: Failed to compute tSNR for {scan_key}: {e}")
                data_dict[scan_key] = None
                n_failed += 1
        
        # Store results
        self.data_dict = data_dict
        
        if verbose:
            print(f"\n{method} computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed: {n_computed}")
            if n_failed > 0:
                print(f"  Failed: {n_failed}")
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
            try:
                if max_lag is None:
                    max_lag_i = np.where(acf[i] < 0)[0][0]
                else:
                    max_lag_i = max_lag

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

    def _compute_tau_from_acf(self, acf: np.ndarray, tr: float, version: str):
        """Compute tau for each parcel using Murray et al. 2014."""
        n_parcels = acf.shape[0]
        n_lags = acf.shape[1]
        tau = np.zeros(n_parcels)
        B = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            try:
                if version == 'loglin':
                    tau[i], A = fit_tau(acf[i], delta=tr)
                elif version == 'murray':
                    # tau[i], B[i], A = fit_tau_murray(acf[i], delta=tr)
                    if n_lags > 10000:
                        negative_indices = np.where(acf[i] < 0)[0]
                        if len(negative_indices) > 0:
                            time_window = negative_indices[0] + (negative_indices[0] * 0.5)
                            max_lag = get_acf_time_index(time_window=time_window, tr=tr)
                            tau[i], A, B[i] = fit_tau_murray(acf[i, :max_lag], delta=tr)
                        else:
                            tau[i], A, B[i] = fit_tau_murray(acf[i], delta=tr)
                    else:
                        tau[i], A, B[i] = fit_tau_murray(acf[i], delta=tr)
            except:
                tau[i] = np.nan
                B[i] = np.nan

        if version == 'murray':
            return tau, B
        else:
            return tau

    def _compute_tau_murray_parallel(self, acf: np.ndarray, tr: float, n_jobs=-1):
        """
        Parallel CPU version using scipy.
        
        Parameters
        ----------
        acf : np.ndarray
            Shape (n_regions, n_lags)
        tr : float
            Repetition time
        n_jobs : int
            Number of parallel jobs (-1 = all CPUs)
        
        Returns
        -------
        tau : np.ndarray
            Shape (n_regions,)
        B : np.ndarray
            Shape (n_regions,)
        """
        
        def fit_single_region(acf_single):
            try:
                tau, A, B = fit_tau_murray(acf_single, delta=tr)
                return tau, B
            except Exception as e:
                return np.nan, np.nan
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_single_region)(acf[i]) for i in range(len(acf))
        )
        
        tau = np.array([r[0] for r in results])
        B = np.array([r[1] for r in results])
        
        return tau, B

    # ==================== Deconfounding Functions ====================
    def regress_task_structure(self):
        timeseries_residuals = dict().fromkeys(self.timeseries_data)
        task_events = dict().fromkeys(self.timeseries_data)
        design_matrices = dict().fromkeys(self.timeseries_data)
        task_events_acf = dict().fromkeys(self.timeseries_data)
        
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
                    tsk_evnts = self.get_task_events(scan_key)
                    # replace all trials with a single trial types
                    # tsk_evnts['trial_type'] = 'trial'
                    design_matrix = create_design_matrix_nilearn(tsk_evnts, n_timepoints, tr, 
                                                                 hrf_model=hrf_model, drift_model=drift_model
                                                                 )
                    # Fit GLM using nilearn approach
                    timeseries_residuals[scan_key], predicted, betas = fit_glm(timeseries, design_matrix)
                    task_events[scan_key] = tsk_evnts
                    design_matrices[scan_key] = design_matrix
                    
                    # get autocorrelation of task events
                    tsk_evnts_flat = tsk_evnts.copy()
                    tsk_evnts_flat['trial_type'] = 'trial'
                    design_matrix_flat = create_design_matrix_nilearn(tsk_evnts_flat, n_timepoints, tr, 
                                                                      hrf_model=hrf_model, drift_model=drift_model
                                                                      )
                    task_events_acf[scan_key] = compute_acf(design_matrix_flat['trial'].values)
                except:
                    timeseries_residuals[scan_key] = None
                    task_events[scan_key] = None
                    design_matrices[scan_key] = None
                    task_events_acf[scan_key] = None
            else:
                timeseries_residuals[scan_key] = None
                task_events[scan_key] = None
                design_matrices[scan_key] = None
                task_events_acf[scan_key] = None

        self.timeseries_residuals = timeseries_residuals
        self.task_events = task_events
        self.design_matrices = design_matrices
        self.task_events_acf = task_events_acf
    
    # ==================== Analysis Functions ====================        
    def match_registry_to_behavior(self):
        if not hasattr(self, 'scan_registry') or self.scan_registry is None:
            self.scan_registry = self.list_registered_scans()

        df_behavior_matched = pd.DataFrame(columns=self.df_behavior.columns)
        df_behavior_matched.index.name = 'subject_id'
        
        for subject_id in self.scan_registry['sub'].unique():
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
            task = kv_dict.get('task', '')
            seg = kv_dict.get('seg', '')
            
            # Check for existing output
            loaded_data = self._load_subject_output(
                subject_id=subject_id,
                output_type=output_type,
                task=task,
                modality=modality,
                seg=seg
            )
            
            data_dict[scan_key] = loaded_data
            
        self.data_dict = data_dict

# ==================== Standalone Functions ====================
def compute_acf(
        data,
        use_fft: bool = False,
    ):
    """
    Compute autocorrelation function for time series data with optional frequency filtering.
    
    Parameters:
    -----------
    data : array-like
        2D array (n_parcels, n_timepoints) or 1D array (single time series)
    use_fft : bool, default=False
        Whether to use FFT for computation
        
    Returns:
    --------
    autocorr : ndarray
        Autocorrelation function(s)
    """
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    autocorr_list = []
    
    for timeseries in data:
        if np.std(timeseries) > 0:  # Avoid division by zero
            # Standardize the (potentially filtered) time series
            standardized = (timeseries - np.mean(timeseries)) / np.std(timeseries)
            n = len(standardized)
            
            if use_fft:
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
        else:
            acf = np.zeros(n)
            acf[:] = np.nan
            autocorr_list.append(acf)
            
    return np.array(autocorr_list)

def fit_tau(acf, delta):
    """
    Fast tau estimation without offset (log-linear fit).
    """
    k = np.arange(1, len(acf))  # Skip k=0 to avoid log(acf[0])=log(1)=0
    
    # Only use positive ACF values
    mask = acf[k] > 0
    k_valid = k[mask]
    acf_valid = acf[k][mask]
    
    # Linear fit: log(ACF) = log(A) - k*delta/tau
    log_acf = np.log(acf_valid)
    slope, intercept = np.polyfit(k_valid, log_acf, 1)
    
    tau = -delta / slope
    A = np.exp(intercept)
    
    return tau, A

def fit_tau_murray(acf, delta):
    """
    Fit intrinsic timescale (tau) using Murray et al. 2014 approach.
    
    Fits: R(kΔ) = A[exp(-kΔ/τ) + B]
    which expands to: R(kΔ) = A·exp(-kΔ/τ) + A·B
    
    Following Murray et al. 2014, fitting starts at the time lag with maximum 
    decrease to avoid refractoriness/negative adaptation at short lags.
    
    Parameters
    ----------
    acf : np.ndarray
        Normalized autocorrelation function (ACF[0] should be ~1)
    delta : float
        Time bin width (e.g., 0.05 for 50ms bins, or TR for fMRI)
    
    Returns
    -------
    tau : float
        Intrinsic timescale
    A : float
        Amplitude parameter
    B : float
        Offset scaling factor (additive offset in model is A·B)
    
    Raises
    ------
    RuntimeError
        If curve fitting fails to converge
    ValueError
        If ACF is too short or invalid
    """
    if len(acf) < 2:
        raise ValueError("ACF must have at least 2 points")
    
    def murray_model(k, A, tau, B):
        return A * (np.exp(-k * delta / tau) + B)
    
    # Find time lag with maximum decrease (Murray et al. 2014)
    # Maximum decrease = most negative gradient
    acf_diff = np.diff(acf)
    start_idx = np.argmin(acf_diff)  # Index with steepest negative slope
    
    # Subset ACF from start_idx onward
    acf_subset = acf[start_idx:]
    k = np.arange(start_idx, start_idx + len(acf_subset))
    
    # Initialize parameters
    # Tau: find where ACF drops to ~0.4 in the subset
    tau_init_idx = np.where(acf_subset < 0.4)[0]
    if len(tau_init_idx) > 0:
        tau_init = tau_init_idx[0] * delta
    else:
        tau_init = len(acf_subset) * delta / 4
    
    # Ensure tau_init is reasonable
    tau_init = max(tau_init, delta)  # At least one time bin
    
    # A: approximate from ACF at start (assuming small B)
    A_init = acf_subset[0] / (1 + 0.1)
    A_init = max(A_init, 0.1)  # Ensure positive
    
    # B: small positive value
    B_init = 0.1
    
    p0 = [A_init, tau_init, B_init]
    bounds = ([0, delta/10, -0.5], [2, np.inf, 0.5])
    
    try:
        popt, _ = curve_fit(
            murray_model, k, acf_subset, 
            p0=p0, bounds=bounds, maxfev=5000
        )
        A, tau, B = popt
        return tau, A, B
        
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to fit Murray model: {str(e)}. "
            f"ACF range: [{acf.min():.3f}, {acf.max():.3f}], "
            f"Start index: {start_idx}/{len(acf)}, "
            f"Initial params: A={A_init:.3f}, tau={tau_init:.3f}, B={B_init:.3f}"
        )

def get_acf_time_index(time_window=25, tr=None):
    """
    Returns the slice endpoint for an ACF array to cover a specified time window.
    
    Parameters
    ----------
    time_window : float, optional
        Length of time to return in seconds (default: 25)
    tr : float
        Repetition time in seconds
        
    Returns
    -------
    int
        Slice endpoint (use as acf[:endpoint] to get lags 0 to n)
    """
    if tr is None:
        raise ValueError("TR must be provided")
    
    max_lag = int(time_window / tr)
    return max_lag + 1

def apply_lowpass_filter(data, sampling_freq, lowpass_cutoff):
    """
    Apply progressive lowpass filtering to MEG data and compute autocorrelation functions.
    
    Parameters
    ----------
    data : ndarray
        Data with shape (n_regions, n_timepoints)
    sampling_freq : float
        Sampling frequency in Hz (e.g., 500)
    lowpass_cutoff : float
        Frequency cutoff in Hz for lowpass filtering (e.g., 50, 30, 10, 5, 1, 0.5)
        Filter will remove faster fluctuations

    """
    n_regions, n_timepoints = data.shape

    # Design lowpass filter (Butterworth, 4th order)
    nyquist = sampling_freq / 2
    normalized_cutoff = lowpass_cutoff / nyquist
    
    # Ensure lowpass_cutoff is below Nyquist frequency
    if normalized_cutoff >= 1.0:
        print(f"Warning: Cutoff {lowpass_cutoff} Hz is >= Nyquist frequency {nyquist} Hz.")
        
    b, a = sp.signal.butter(4, normalized_cutoff, btype='low')
    
    # Filter each region
    filtered_data = np.zeros_like(data)
    for region_idx in range(n_regions):
        # Use filtfilt for zero-phase filtering
        filtered_data[region_idx, :] = sp.signal.filtfilt(b, a, data[region_idx, :])
            
    return filtered_data

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
    Fit GLM with masking for NaN and zero-variance regions.
    
    Parameters:
    - data: 2D array (n_parcels, n_timepoints)
    - design_matrix: nilearn DesignMatrix object
    
    Returns:
    - residuals: residual time series after GLM fitting
    - predicted: predicted time series
    - betas: regression coefficients
    - valid_mask: boolean mask indicating which regions were included in fitting
    """
    
    # Convert design matrix to numpy array
    X = design_matrix.values
    n_parcels, n_timepoints = data.shape
    
    # Initialize outputs
    residuals = np.full_like(data, np.nan)
    predicted = np.full_like(data, np.nan)
    betas = np.full((n_parcels, X.shape[1]), np.nan)
    
    # Create mask for valid regions (no NaNs and non-zero variance)
    has_nan = np.any(np.isnan(data), axis=1)
    has_zero_variance = np.var(data, axis=1) == 0
    valid_mask = ~(has_nan | has_zero_variance)
    
    if np.sum(valid_mask) == 0:
        # No valid regions - return all NaNs
        return residuals, predicted, betas, valid_mask
    
    # Fit GLM only for valid regions
    data_valid = data[valid_mask, :]
    
    # Fit GLM using least squares
    betas_valid, residuals_sum, rank, s = np.linalg.lstsq(X, data_valid.T, rcond=None)
    
    # Compute predicted and residual signals for valid regions
    predicted_valid = np.dot(X, betas_valid).T
    residuals_valid = data_valid - predicted_valid
    
    # Place results back into full arrays
    residuals[valid_mask, :] = residuals_valid
    predicted[valid_mask, :] = predicted_valid
    betas[valid_mask, :] = betas_valid.T
    
    return residuals, predicted, betas

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
    
    # lags_fit = lags[1:]
    # acf_fit = acf[1:]
    
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

def calculate_tsnr(timeseries_matrix):
    """
    Calculate temporal signal-to-noise ratio (tSNR) for ROI timeseries data.
    
    Parameters
    ----------
    timeseries_matrix : numpy.ndarray
        2D array with shape (n_rois, n_timepoints) where each row represents
        an ROI and each column represents a timepoint.
    
    Returns
    -------
    tsnr : numpy.ndarray
        1D array with shape (n_rois,) containing tSNR values for each ROI.
        tSNR = mean(signal) / std(signal) across time for each ROI.
    
    Notes
    -----
    - Higher tSNR values indicate better signal quality
    - ROIs with zero standard deviation will result in inf values
    - Consider masking out ROIs with very low signal or high motion
    """
    # Calculate mean and standard deviation across time (axis=1)
    mean_signal = np.mean(timeseries_matrix, axis=1)
    std_signal = np.std(timeseries_matrix, axis=1, ddof=1)  # ddof=1 for sample std
    
    # Calculate tSNR, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr = mean_signal / std_signal
    
    return tsnr

# ==================== Standalone Utility Functions (for use in notebooks) ====================
def compute_deflections(rest_timescales, task_timescales, nuisance_regression=False):
    rest_timescales = rest_timescales.reshape(-1, 1)
    task_timescales = task_timescales.reshape(-1, 1)

    int_deflections = task_timescales - rest_timescales
    
    if nuisance_regression:
        deflections_mean = np.mean(int_deflections)
        
        X = np.concatenate((rest_timescales, np.ones((rest_timescales.shape[0], 1))), axis=1)
        beta = np.dot(np.linalg.pinv(X), int_deflections)
        predicted = np.dot(X, beta)
        int_deflections = int_deflections - predicted
        # print(int_deflections.shape, predicted.shape)
        int_deflections += deflections_mean
    
    return int_deflections[:, 0]

def filter_data_dict(data_dict, data_dict_residuals=None, remove_subject_level=True):
    """
    Filter data dictionary by removing scans with missing or problematic data.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of scan data
    data_dict_residuals : dict, optional
        Dictionary of residualized data to replace task scans
    remove_subject_level : bool, default=True
        If True, remove all scans for a subject if any scan has issues.
        If False, only remove the specific problematic scan.
    
    Returns
    -------
    data_dict : dict
        Filtered data dictionary
    data_dict_residuals : dict, optional
        Only returned if data_dict_residuals was provided
    """
    
    remove_these_keys = []
    
    # Check for None values
    for scan_key in data_dict.keys():
        if data_dict[scan_key] is None:
            if remove_subject_level:
                sub_key = scan_key.split('_')[0]
                remove_these_keys.append([k for k in data_dict.keys() if sub_key in k])
            else:
                remove_these_keys.append([scan_key])
    
    # Optionally replace task data with residualized data
    if data_dict_residuals is not None:
        for scan_key in data_dict.keys():
            if 'condition-task' in scan_key:
                try:
                    data_dict[scan_key] = data_dict_residuals[scan_key].copy()
                except:
                    if remove_subject_level:
                        sub_key = scan_key.split('_')[0]
                        remove_these_keys.append([k for k in data_dict.keys() if sub_key in k])
                    else:
                        remove_these_keys.append([scan_key])
    
    # Flatten list and remove duplicates
    remove_these_keys = list(set([item for sublist in remove_these_keys for item in sublist]))
    
    # Remove marked keys
    for remove_key in remove_these_keys:
        if remove_key in data_dict:
            del data_dict[remove_key]
    
    if data_dict_residuals is None:
        return data_dict
    else:
        return data_dict, data_dict_residuals

def create_ints_dataframe_custom_grouping(data_dict, n_regions=400,
                                          average_3t_rest=False, average_7t_rest=False, average_all_rest=False, 
                                          average_7t_movies=False,
                                          average_3t_task_pairs=False, average_3t_tasks=False, 
                                          average_meg_rest=False, average_meg_task=False,
                                          sort_by_median=False):
    """
    Create a dataframe with brain regions as indices and unique scans as columns.
    Groups scans by scan type only (ignoring subject, modality, condition, atlas).
    Values represent INTs averaged over subjects for each scan type.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are scan strings like 
        'sub-100307_modality-MEG_condition-rest_scan-3Restin_atlas-Schaefer4007'
        and values are vectors of INTs (one per brain region)
    n_regions : int, default 400
        Number of brain regions
    average_3t_rest : bool, default False
        If True, average 3T rest scans (rfMRIREST1LR, rfMRIREST1RL, etc.) into "REST_3T"
        Cannot be True if average_all_rest is True
    average_3t_task_pairs : bool, default False
        If True, average 3T task LR/RL pairs (e.g., tfMRIWMLR + tfMRIWMRL)
        Cannot be True if average_3t_tasks is True
    average_3t_tasks : bool, default False
        If True, average ALL 3T task scans (all tfMRI scans except 7T movies) into "Tasks"
        Cannot be True if average_3t_task_pairs is True
    average_7t_rest : bool, default False
        If True, average 7T rest scans (rfMRIREST17TPA, rfMRIREST27TAP, etc.) into "REST_7T"
        Cannot be True if average_all_rest is True
    average_7t_movies : bool, default False
        If True, average 7T movie scans (tfMRIMOVIE17TAP, tfMRIMOVIE27TPA, etc.) into "Movies"
    average_all_rest : bool, default False
        If True, average ALL rest scans (both 3T and 7T) into "Rest"
        Cannot be True if average_3t_rest or average_7t_rest is True
    sort_by_median : bool, default False
        If True, sort columns by their median INT values in descending order
    
    Returns:
    --------
    df_ints : pd.DataFrame
        DataFrame with brain regions as index and unique scan types as columns.
        Values are INTs averaged across subjects for each scan type.
    """
    
    # Check for mutually exclusive options
    if average_3t_task_pairs and average_3t_tasks:
        raise ValueError("average_3t_task_pairs and average_3t_tasks cannot both be True")
    
    if average_all_rest and (average_3t_rest or average_7t_rest):
        raise ValueError("average_all_rest cannot be True if average_3t_rest or average_7t_rest is True")
    
    def extract_scan_name(scan_key):
        """Extract just the scan part from a full scan key."""
        parts = scan_key.split('_')
        for part in parts:
            if part.startswith('task-'):
                return part
        return None
    
    # Group scans by scan type only
    scan_groups = defaultdict(list)
    
    for scan_key, ints_vector in data_dict.items():
        scan_name = extract_scan_name(scan_key)
        if scan_name:
            scan_groups[scan_name].append(ints_vector)
    
    # Check that all vectors have the expected length
    for scan_key, ints_vector in data_dict.items():
        if len(ints_vector) != n_regions:
            raise ValueError(f"Expected {n_regions} regions, but scan {scan_key} has {len(ints_vector)} values")
    
    # Calculate mean INTs for each scan type across subjects
    averaged_data = {}
    
    for scan_name, vectors_list in scan_groups.items():
        if len(vectors_list) > 0:
            stacked_vectors = np.stack(vectors_list, axis=0)
            mean_vector = np.nanmean(stacked_vectors, axis=0)
            # mean_vector = np.nanmedian(stacked_vectors, axis=0)
            averaged_data[scan_name] = mean_vector
            
            print(f"Scan: {scan_name}")
            print(f"  Number of subjects: {len(vectors_list)}")
            print(f"  Mean INT range: [{mean_vector.min():.3f}, {mean_vector.max():.3f}]")
    
    # Apply optional scan averaging
    if average_7t_rest or average_7t_movies or average_3t_rest or average_3t_task_pairs or average_3t_tasks or average_all_rest or average_meg_rest or average_meg_task:
        print(f"\nApplying optional scan averaging...")

    # Average ALL rest scans (3T and 7T)
    if average_all_rest:
        all_rest_scans = ['task-rfMRIREST1LR', 'task-rfMRIREST1RL', 'task-rfMRIREST2LR', 'task-rfMRIREST2RL',
                          'task-rfMRIREST1AP', 'task-rfMRIREST1PA', 'task-rfMRIREST2AP', 'task-rfMRIREST2PA',
                          'task-rfMRIREST17TPA', 'task-rfMRIREST27TAP', 'task-rfMRIREST37TPA', 'task-rfMRIREST47TAP']
        all_rest_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in all_rest_scans:
                all_rest_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if all_rest_data:
            all_rest_mean = np.nanmean(np.stack(all_rest_data, axis=0), axis=0)
            averaged_data['Rest'] = all_rest_mean
            print(f"  Averaged {len(all_rest_data)} rest scans (3T and 7T) into: Rest")

    # Average 3T rest scans  
    if average_3t_rest:
        rest_3t_scans = ['task-rfMRIREST1LR', 'task-rfMRIREST1RL', 'task-rfMRIREST2LR', 'task-rfMRIREST2RL',
                         'task-rfMRIREST1AP', 'task-rfMRIREST1PA', 'task-rfMRIREST2AP', 'task-rfMRIREST2PA']
        rest_3t_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in rest_3t_scans:
                rest_3t_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if rest_3t_data:
            rest_3t_mean = np.nanmean(np.stack(rest_3t_data, axis=0), axis=0)
            averaged_data['Rest-3T'] = rest_3t_mean
            print(f"  Averaged {len(rest_3t_data)} 3T rest scans into: scan-REST3T")
    
    # Average ALL 3T task scans (excluding 7T movies)
    if average_3t_tasks:
        task_3t_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name.startswith('task-tfMRI') and 'MOVIE' not in scan_name:
                task_3t_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if task_3t_data:
            task_3t_mean = np.nanmean(np.stack(task_3t_data, axis=0), axis=0)
            averaged_data['Tasks'] = task_3t_mean
            print(f"  Averaged {len(task_3t_data)} 3T task scans into: Tasks")
    
    # Average 3T task LR/RL pairs
    if average_3t_task_pairs:
        task_pairs = {}
        
        for scan_name in list(averaged_data.keys()):
            if scan_name.startswith('task-tfMRI') and (scan_name.endswith('LR') or scan_name.endswith('RL') or scan_name.endswith('AP') or scan_name.endswith('PA')):
                # Get base task name (remove LR/RL/AP/PA)
                base_task = scan_name[:-2]  # Remove last 2 characters
                
                if base_task not in task_pairs:
                    task_pairs[base_task] = []
                task_pairs[base_task].append((scan_name, averaged_data[scan_name]))
        
        # Average pairs and remove originals
        for base_task, scan_list in task_pairs.items():
            if len(scan_list) == 2:  # We have both LR and RL (or AP and PA)
                vectors = [item[1] for item in scan_list]
                pair_mean = np.nanmean(np.stack(vectors, axis=0), axis=0)
                
                # Remove original scans
                for scan_name, _ in scan_list:
                    del averaged_data[scan_name]
                
                # Create new scan name
                new_scan_name = base_task  # base_task already includes 'task-'
                averaged_data[new_scan_name] = pair_mean
                print(f"  Averaged LR/RL pair for {base_task} into: {new_scan_name}")

    # Average 7T rest scans
    if average_7t_rest:
        rest_7t_scans = ['task-rfMRIREST17TPA', 'task-rfMRIREST27TAP', 'task-rfMRIREST37TPA', 'task-rfMRIREST47TAP']
        rest_7t_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in rest_7t_scans:
                rest_7t_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if rest_7t_data:
            rest_7t_mean = np.nanmean(np.stack(rest_7t_data, axis=0), axis=0)
            averaged_data['Rest-7T'] = rest_7t_mean
            print(f"  Averaged {len(rest_7t_data)} 7T rest scans into: scan-REST7T")
    
    # Average 7T movie scans
    if average_7t_movies:
        movie_7t_scans = ['task-tfMRIMOVIE17TAP', 'task-tfMRIMOVIE27TPA', 'task-tfMRIMOVIE37TPA', 'task-tfMRIMOVIE47TAP']
        movie_7t_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in movie_7t_scans:
                movie_7t_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if movie_7t_data:
            movie_7t_mean = np.nanmean(np.stack(movie_7t_data, axis=0), axis=0)
            averaged_data['Movies'] = movie_7t_mean
            print(f"  Averaged {len(movie_7t_data)} 7T movie scans into: Movies")
            
    # Average MEG rest scans
    if average_meg_rest:
        rest_meg_scans = ['task-3Restin', 'task-4Restin', 'task-5Restin']
        rest_meg_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in rest_meg_scans:
                rest_meg_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if rest_meg_data:
            rest_meg_mean = np.nanmean(np.stack(rest_meg_data, axis=0), axis=0)
            averaged_data['Rest-MEG'] = rest_meg_mean
            print(f"  Averaged {len(rest_meg_data)} MEG rest scans into: scan-RESTMEG")

    # Average MEG task scans
    if average_meg_task:
        task_meg_scans = ['task-6Wrkmem', 'task-7Wrkmem']
        task_meg_data = []
        
        for scan_name in list(averaged_data.keys()):
            if scan_name in task_meg_scans:
                task_meg_data.append(averaged_data[scan_name])
                del averaged_data[scan_name]
        
        if task_meg_data:
            task_meg_mean = np.nanmean(np.stack(task_meg_data, axis=0), axis=0)
            averaged_data['Task-MEG'] = task_meg_mean
            print(f"  Averaged {len(task_meg_data)} MEG task scans into: scan-TASKMEG")

    # Create DataFrame
    df_ints = pd.DataFrame(averaged_data)
    df_ints.index.name = 'brain_region'
    df_ints.index = range(n_regions)
    df_ints.columns = df_ints.columns.str.replace('task-', '', regex=False)
    
    # Sort columns by median if requested
    if sort_by_median:
        column_summary_stat = df_ints.median(axis=0)
        sorted_columns = column_summary_stat.sort_values(ascending=False).index
        df_ints = df_ints[sorted_columns]
        print(f"Sorted columns by median INT values (descending order)")
    
    print(f"\nCreated DataFrame with shape: {df_ints.shape}")
    print(f"Brain regions (rows): {df_ints.shape[0]}")
    print(f"Unique scan types (columns): {df_ints.shape[1]}")
    print(f"Final scan types: {list(df_ints.columns)}")
    
    return df_ints

def sort_and_group_timescales(timescales, brain_map, ntile=10):
    n_nodes = len(brain_map)
    brain_map_sort_idx = np.argsort(brain_map)
    
    timescales_sorted = timescales[:, brain_map_sort_idx]
    timescales_sorted_dec = np.zeros((timescales_sorted.shape[0], ntile))

    step_size = int(n_nodes / ntile)
    counter = 0
    for i in np.arange(0, n_nodes, step_size):
        timescales_sorted_dec[:, counter] = timescales_sorted[:, i:i+step_size].mean(axis=1)
        counter += 1
        
    return timescales_sorted_dec

def convert_data_dict(data_dict, scan_registry, brain_map=None, verbose=True):
    data_dict_converted = dict().fromkeys(scan_registry['task'].unique())

    for task in data_dict_converted.keys():
        task_filter = scan_registry['task'] == task
        sub_list = list(scan_registry.loc[task_filter, 'scan_string'])
        extracted_data = {key: data_dict[key] for key in sub_list if key in data_dict}
        extracted_data = np.array(list(extracted_data.values()))

        if brain_map is not None:
            extracted_data = sort_and_group_timescales(extracted_data, brain_map)

        data_dict_converted[task] = extracted_data
        
        if verbose:
            print(task, extracted_data.shape)
            
    return data_dict_converted

def average_data_dict_converted(data_dict_converted, average_7t_rest=False, average_3t_rest=False, 
                                average_3t_task_pairs=False, average_3t_tasks=False,
                                average_7t_movies=False, average_all_rest=False,
                                average_meg_rest=False, average_meg_task=False):
    """
    Apply scan averaging to a dictionary of scan data organized by brain systems.
    
    Parameters:
    -----------
    data_dict_converted : dict
        Dictionary where keys are scan names (e.g., 'rfMRIREST1LR', 'tfMRIWMLR')
        and values are arrays of shape (n_subjects, 10) where 10 corresponds to brain systems
    average_7t_rest : bool, default False
        If True, average 7T rest scans (rfMRIREST17TPA, rfMRIREST27TAP, etc.) into "REST_7T"
        Cannot be True if average_all_rest is True
    average_3t_rest : bool, default False
        If True, average 3T rest scans (rfMRIREST1LR, rfMRIREST1RL, etc.) into "REST_3T"
        Cannot be True if average_all_rest is True
    average_3t_task_pairs : bool, default False
        If True, average 3T task LR/RL pairs (e.g., tfMRIWMLR + tfMRIWMRL)
        Cannot be True if average_3t_tasks is True
    average_3t_tasks : bool, default False
        If True, average ALL 3T task scans (all tfMRI scans except 7T movies) into "Tasks"
        Cannot be True if average_3t_task_pairs is True
    average_7t_movies : bool, default False
        If True, average 7T movie scans (tfMRIMOVIE17TAP, tfMRIMOVIE27TPA, etc.) into "MOVIE"
    average_all_rest : bool, default False
        If True, average ALL rest scans (both 3T and 7T) into "Rest"
        Cannot be True if average_3t_rest or average_7t_rest is True
    average_meg_rest : bool, default False
        If True, average MEG rest scans into "RESTMEG"
    average_meg_task : bool, default False
        If True, average MEG task scans into "TASKMEG"
    
    Returns:
    --------
    averaged_dict : dict
        Dictionary with averaged scan data, same structure as input but with averaged scans
    """
    
    # Check for mutually exclusive options
    if average_3t_task_pairs and average_3t_tasks:
        raise ValueError("average_3t_task_pairs and average_3t_tasks cannot both be True")
    
    if average_all_rest and (average_3t_rest or average_7t_rest):
        raise ValueError("average_all_rest cannot be True if average_3t_rest or average_7t_rest is True")
    
    # Create a copy to avoid modifying the original
    averaged_dict = data_dict_converted.copy()
    
    print("Applying scan averaging to centiles dictionary...")
    
    # 1. Average ALL rest scans (3T and 7T)
    if average_all_rest:
        all_rest_scans = ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL',
                          'rfMRIREST1AP', 'rfMRIREST1PA', 'rfMRIREST2AP', 'rfMRIREST2PA',
                          'rfMRIREST17TPA', 'rfMRIREST27TAP', 'rfMRIREST37TPA', 'rfMRIREST47TAP']
        all_rest_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in all_rest_scans:
                all_rest_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if all_rest_data:
            all_rest_mean = np.nanmean(np.stack(all_rest_data, axis=0), axis=0)
            averaged_dict['Rest'] = all_rest_mean
            print(f"  Averaged {len(all_rest_data)} rest scans (3T and 7T) into: Rest")
            print(f"    Shape: {all_rest_mean.shape}")
    
    # 2. Average 7T rest scans
    if average_7t_rest:
        rest_7t_scans = ['rfMRIREST17TPA', 'rfMRIREST27TAP', 'rfMRIREST37TPA', 'rfMRIREST47TAP']
        rest_7t_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in rest_7t_scans:
                rest_7t_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if rest_7t_data:
            # Average across the different 7T rest scans (axis=0 combines scans)
            rest_7t_mean = np.nanmean(np.stack(rest_7t_data, axis=0), axis=0)
            averaged_dict['Rest-7T'] = rest_7t_mean
            print(f"  Averaged {len(rest_7t_data)} 7T rest scans into: REST7T")
            print(f"    Shape: {rest_7t_mean.shape}")
    
    # 3. Average 3T rest scans  
    if average_3t_rest:
        rest_3t_scans = ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL',
                         'rfMRIREST1AP', 'rfMRIREST1PA', 'rfMRIREST2AP', 'rfMRIREST2PA']
        rest_3t_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in rest_3t_scans:
                rest_3t_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if rest_3t_data:
            rest_3t_mean = np.nanmean(np.stack(rest_3t_data, axis=0), axis=0)
            averaged_dict['Rest-3T'] = rest_3t_mean
            print(f"  Averaged {len(rest_3t_data)} 3T rest scans into: REST3T")
            print(f"    Shape: {rest_3t_mean.shape}")
    
    # 4. Average ALL 3T task scans (excluding 7T movies)
    if average_3t_tasks:
        task_3t_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name.startswith('tfMRI') and 'MOVIE' not in scan_name:
                task_3t_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if task_3t_data:
            task_3t_mean = np.nanmean(np.stack(task_3t_data, axis=0), axis=0)
            averaged_dict['Tasks'] = task_3t_mean
            print(f"  Averaged {len(task_3t_data)} 3T task scans into: Tasks")
            print(f"    Shape: {task_3t_mean.shape}")
    
    # 5. Average 3T task LR/RL pairs
    if average_3t_task_pairs:
        task_pairs = {}
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name.startswith('tfMRI') and (scan_name.endswith('LR') or scan_name.endswith('RL') or scan_name.endswith('AP') or scan_name.endswith('PA')):
                # Get base task name (remove LR/RL/AP/PA)
                base_task = scan_name[:-2]  # Remove last 2 characters
                
                if base_task not in task_pairs:
                    task_pairs[base_task] = []
                task_pairs[base_task].append((scan_name, averaged_dict[scan_name]))
        
        # Average pairs and remove originals
        for base_task, scan_list in task_pairs.items():
            if len(scan_list) == 2:  # We have both LR and RL (or AP and PA)
                vectors = [item[1] for item in scan_list]
                pair_mean = np.nanmean(np.stack(vectors, axis=0), axis=0)
                
                # Remove original scans
                for scan_name, _ in scan_list:
                    del averaged_dict[scan_name]
                
                # Create new scan name (remove tfMRI prefix for cleaner name)
                new_scan_name = base_task.replace('tfMRI', '')  # e.g., 'tfMRIWM' -> 'WM'
                averaged_dict[new_scan_name] = pair_mean
                print(f"  Averaged LR/RL pair for {base_task} into: {new_scan_name}")
                print(f"    Shape: {pair_mean.shape}")
    
    # 6. Average 7T movie scans
    if average_7t_movies:
        movie_7t_scans = ['tfMRIMOVIE17TAP', 'tfMRIMOVIE27TPA', 'tfMRIMOVIE37TPA', 'tfMRIMOVIE47TAP']
        movie_7t_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in movie_7t_scans:
                movie_7t_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if movie_7t_data:
            movie_7t_mean = np.nanmean(np.stack(movie_7t_data, axis=0), axis=0)
            averaged_dict['Movies'] = movie_7t_mean
            print(f"  Averaged {len(movie_7t_data)} 7T movie scans into: Movies")
            print(f"    Shape: {movie_7t_mean.shape}")
    
    # 7. Average MEG rest scans
    if average_meg_rest:
        rest_meg_scans = ['3Restin', '4Restin', '5Restin']
        rest_meg_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in rest_meg_scans:
                rest_meg_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if rest_meg_data:
            rest_meg_mean = np.nanmean(np.stack(rest_meg_data, axis=0), axis=0)
            averaged_dict['Rest-MEG'] = rest_meg_mean
            print(f"  Averaged {len(rest_meg_data)} MEG rest scans into: RESTMEG")
            print(f"    Shape: {rest_meg_mean.shape}")
    
    # 8. Average MEG task scans
    if average_meg_task:
        task_meg_scans = ['6Wrkmem', '7Wrkmem']
        task_meg_data = []
        
        for scan_name in list(averaged_dict.keys()):
            if scan_name in task_meg_scans:
                task_meg_data.append(averaged_dict[scan_name])
                del averaged_dict[scan_name]
        
        if task_meg_data:
            task_meg_mean = np.nanmean(np.stack(task_meg_data, axis=0), axis=0)
            averaged_dict['Task-MEG'] = task_meg_mean
            print(f"  Averaged {len(task_meg_data)} MEG task scans into: TASKMEG")
            print(f"    Shape: {task_meg_mean.shape}")
    
    # Print summary
    print(f"\nFinal dictionary contains {len(averaged_dict)} scan types:")
    for scan_name, data in averaged_dict.items():
        print(f"  {scan_name}: {data.shape}")
    
    # Print summary of averaging operations
    averaging_ops = []
    if average_all_rest: averaging_ops.append("all rest")
    if average_7t_rest: averaging_ops.append("7T rest")
    if average_3t_rest: averaging_ops.append("3T rest")
    if average_3t_tasks: averaging_ops.append("3T all tasks")
    if average_3t_task_pairs: averaging_ops.append("3T task pairs")
    if average_7t_movies: averaging_ops.append("7T movies")
    if average_meg_rest: averaging_ops.append("MEG rest")
    if average_meg_task: averaging_ops.append("MEG task")
    
    if averaging_ops:
        print(f"\nApplied averaging for: {', '.join(averaging_ops)}")
    else:
        print("\nNo scan averaging applied")
    
    return averaged_dict
