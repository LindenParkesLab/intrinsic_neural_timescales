import os
import numpy as np
import scipy as sp
import pandas as pd
import pyfftw
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy.optimize import curve_fit, OptimizeWarning
from collections import defaultdict
from joblib import Parallel, delayed
import statsmodels.api as sm

pyfftw.interfaces.cache.enable()  # cache FFT plans

from src.data import DataLoader, ScanInfo, parse_scan_string, build_scan_string
from snaplab_tools.signal import apply_frequency_filter
from snaplab_tools.stats import partial_pearsonr
import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timescale method registry
# Each entry is a callable: (acf: np.ndarray, tr: float, **kwargs) -> np.ndarray
# Add new methods here without touching Timescales.compute_timescale_measures.
# ---------------------------------------------------------------------------

def _ts_method_zero_crossing(acf, tr, inst, **kw):
    return inst._compute_zero_crossing(acf) * tr

def _ts_method_zero_crossing_robust_nc2(acf, tr, inst, **kw):
    return inst._compute_zero_crossing_robust(acf, n_crossings=2) * tr

def _ts_method_zero_crossing_robust_nc3(acf, tr, inst, **kw):
    return inst._compute_zero_crossing_robust(acf, n_crossings=3) * tr

def _ts_method_auc_zero_crossing(acf, tr, inst, **kw):
    return inst._compute_auc(acf, threshold=0.0) * tr

def _ts_method_auc_half_crossing(acf, tr, inst, **kw):
    return inst._compute_auc(acf, threshold=0.5) * tr

def _ts_method_decay_time_half(acf, tr, inst, **kw):
    return inst._compute_decay_time(acf, tr=tr, threshold=0.5)

def _ts_method_lag_one(acf, tr, inst, **kw):
    return acf[:, 1]

def _ts_method_tau(acf, tr, inst, **kw):
    return inst._compute_tau_from_acf(acf, tr=tr, version='loglin')

def _ts_method_tau_howell(acf, tr, inst, **kw):
    return inst._compute_tau_from_acf(acf, tr=tr, version='howell')

def _ts_method_tau_murray_par(acf, tr, inst, **kw):
    tau, _ = inst._compute_tau_murray_parallel(acf, tr=tr, n_jobs=-1)
    return tau

def _ts_method_B_murray_par(acf, tr, inst, **kw):
    _, B = inst._compute_tau_murray_parallel(acf, tr=tr, n_jobs=-1)
    return B

def _ts_method_B_murray_par_lag3mins(acf, tr, inst, **kw):
    n_tp = int(3 * 60 / tr)
    _, B = inst._compute_tau_murray_parallel(acf[:, :n_tp], tr=tr, n_jobs=int(os.cpu_count() / 2))
    return B

def _ts_method_B_murray_par_lag4mins(acf, tr, inst, **kw):
    n_tp = int(4 * 60 / tr)
    _, B = inst._compute_tau_murray_parallel(acf[:, :n_tp], tr=tr, n_jobs=int(os.cpu_count() / 2))
    return B

def _ts_method_B_murray_par_lag5mins(acf, tr, inst, **kw):
    n_tp = int(5 * 60 / tr)
    _, B = inst._compute_tau_murray_parallel(acf[:, :n_tp], tr=tr, n_jobs=int(os.cpu_count() / 2))
    return B

TIMESCALE_METHODS = {
    'zero_crossing':              _ts_method_zero_crossing,
    'zero_crossing_robust_nc2':   _ts_method_zero_crossing_robust_nc2,
    'zero_crossing_robust_nc3':   _ts_method_zero_crossing_robust_nc3,
    'auc_zero_crossing':          _ts_method_auc_zero_crossing,
    'auc_half_crossing':          _ts_method_auc_half_crossing,
    'decay_time_half':            _ts_method_decay_time_half,
    'lag_one':                    _ts_method_lag_one,
    'tau':                        _ts_method_tau,
    'tau_howell':                 _ts_method_tau_howell,
    'tau_murray_par':             _ts_method_tau_murray_par,
    'B_murray_par':               _ts_method_B_murray_par,
    'B_murray_par_lag3mins':      _ts_method_B_murray_par_lag3mins,
    'B_murray_par_lag4mins':      _ts_method_B_murray_par_lag4mins,
    'B_murray_par_lag5mins':      _ts_method_B_murray_par_lag5mins,
}


class Timescales(DataLoader):
    def __init__(
        self,
        dataset: str,
        timeseries_data: dict,
        load_behavior: bool = True,
        verbose: bool = True
    ):
        """Initialise Timescales for a dataset and its per-scan timeseries.

        Parameters
        ----------
        dataset : str
            Dataset name (passed to DataLoader).
        timeseries_data : dict
            Mapping scan_key -> (n_parcels, n_timepoints) array. May be None.
        load_behavior : bool
            Whether DataLoader should load behavioral data on init.
        verbose : bool
            Print an initialisation summary.
        """
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
        space: str = None,
        session: str = None,
        run: str = None,
        suffix: str = '.npy',
        mode: str = 'load'
    ) -> Path:
        """Generate a standardized filepath for subject-specific outputs (key-value naming).

        The filename includes every field that uniquely identifies a scan so that
        outputs for different sessions, spaces, or runs never collide.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        output_type : str
            Type of output (e.g. 'autocorr', 'timescale_acf').
        task : str
            Task/run label (e.g. 'rfMRIREST1LR').
        modality : str
            Modality string (e.g. 'MRI3T').
        seg : str
            Atlas/parcellation name.
        space : str
            Template space (e.g. 'fsLR', 'MNI152NLin2009cAsym').
        session : str
            Session label (e.g. '1', '01'); required for longitudinal data.
        run : str
            Run label (e.g. '01'); included when present.
        suffix : str
            File extension.
        mode : {'load', 'save'}
            'save' creates parent dirs; 'load' does not.

        Returns
        -------
        Path
            Path to the output file.
        """
        # Create output subdirectory structure
        output_subdir = self.output_dir / 'data' / 'timescales_outputs' / self.dataset / output_type
        if mode == 'save':
            output_subdir.mkdir(parents=True, exist_ok=True)

        # Build filename — include all fields that uniquely identify a scan
        filename = build_scan_string(
            sub=subject_id,
            ses=session,
            modality=modality,
            task=task,
            space=space,
            seg=seg,
            run=run,
            stat=output_type,
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
        space: str = None,
        session: str = None,
        run: str = None,
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
            space=space,
            session=session,
            run=run,
            suffix=suffix,
            mode='save',
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
        space: str = None,
        session: str = None,
        run: str = None,
    ) -> Optional[Any]:
        """Load previously saved subject-specific output with key-value naming."""
        suffix = '.npy'
        filepath = self._get_output_filepath(
            subject_id=subject_id,
            output_type=output_type,
            task=task,
            modality=modality,
            seg=seg,
            space=space,
            session=session,
            run=run,
            suffix=suffix,
            mode='load',
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
            logger.warning("Could not load %s: %s", filepath, e)
            return None

    def _check_output_exists(
        self,
        subject_id: str,
        output_type: str,
        task: str,
        modality: str,
        seg: str = None,
        space: str = None,
        session: str = None,
        run: str = None,
    ) -> bool:
        """Check if a subject output file already exists."""
        for suffix in ['.npy', '.npz', '.pkl']:
            filepath = self._get_output_filepath(
                subject_id=subject_id,
                output_type=output_type,
                task=task,
                modality=modality,
                seg=seg,
                space=space,
                session=session,
                run=run,
                suffix=suffix,
                mode='load',
            )
            if filepath.exists():
                return True
        return False

    def _parse_scan_string(self, scan_key: str) -> Dict[str, str]:
        """Parse a scan key using key-value format.

        Parameters
        ----------
        scan_key : str
            Scan-key string, e.g.
            "sub-100307_modality-MRI3T_task-rfMRIREST1LR_seg-Schaefer4007".

        Returns
        -------
        dict
            Parsed key-value pairs (see parse_scan_string).
        """
        return parse_scan_string(scan_key)

    # ============== Computation Methods ==============
    def _run_for_all_scans(
        self,
        compute_fn,
        output_type: str,
        use_residuals: bool = False,
        force_recompute: bool = False,
        save_outputs: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Shared iteration loop for all per-scan compute methods.

        For each scan in self.scans:
          1. Try to load from cache (unless force_recompute).
          2. If not cached, call compute_fn(timeseries_scan, kv_dict) → result.
          3. Optionally save the result.

        Parameters
        ----------
        compute_fn : callable
            Signature: (timeseries_scan: np.ndarray, kv_dict: dict) -> np.ndarray
            Should raise on failure; exceptions are caught here and logged.
        output_type : str
            Used for cache file naming and logging.
        use_residuals : bool
            If True and condition == 'task', use self.timeseries_residuals instead
            of self.timeseries_data.
        force_recompute : bool
            Skip cache lookup if True.
        save_outputs : bool
            Persist results to disk if True.
        verbose : bool
            Print summary counts when True.

        Returns
        -------
        dict mapping scan_key -> result array (or None on failure)
        """
        results = dict.fromkeys(self.scans)
        n_loaded = n_computed = n_failed = 0

        for scan_key in tqdm(self.scans, desc=f"Computing {output_type}"):
            kv_dict = self._parse_scan_string(scan_key)
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            condition  = kv_dict.get('condition', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')
            tr         = kv_dict.get('tr', '')

            if not force_recompute:
                cached = self._load_subject_output(
                    subject_id=subject_id, output_type=output_type,
                    task=task, modality=modality, seg=seg,
                    space=space, session=session, run=run,
                )
                if cached is not None:
                    results[scan_key] = cached
                    n_loaded += 1
                    continue

            try:
                if use_residuals and condition == 'task':
                    timeseries_scan = self.timeseries_residuals[scan_key]
                else:
                    timeseries_scan = self.timeseries_data[scan_key]

                result = compute_fn(timeseries_scan, kv_dict)
                results[scan_key] = result
                n_computed += 1

                if save_outputs:
                    self._save_subject_output(
                        data=result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        space=space,
                        session=session,
                        run=run,
                        metadata={'output_type': output_type, 'tr': tr, 'modality': modality,
                                  'dataset': self.dataset},
                    )
            except Exception as e:
                logger.warning("Failed to compute %s for %s: %s", output_type, scan_key, e)
                results[scan_key] = None
                n_failed += 1

        if verbose:
            print(f"\n{output_type} computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed:    {n_computed}")
            if n_failed:
                print(f"  Failed:            {n_failed}")
            print(f"  Total:             {len(self.scans)}")

        return results

    def compute_autocorrelation_function(
        self,
        use_fft: bool = False,
        force_recompute: bool = False,
        save_outputs: bool = True,
        suffix: str = None,
        use_residuals: bool = False,
        lowpass: float = None,
        highpass: float = None,
        verbose: bool = True,
    ):
        """Compute (or load) the ACF for every scan into self.autocorr_dict.

        Runs compute_acf per scan via _run_for_all_scans, caching results to disk.
        lowpass/highpass affect only the output_type name (filtering is applied at
        load time, not here).

        Parameters
        ----------
        use_fft : bool
            Use the FFT-based ACF path in compute_acf.
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        suffix : str or None
            Extra tag appended to output_type.
        use_residuals : bool
            Use self.timeseries_residuals for task scans instead of raw timeseries.
        lowpass, highpass : float or None
            Used only for output naming (see above).
        verbose : bool
            Print summary counts.
        """
        output_type = 'autocorr'
        if highpass is not None and lowpass is not None:
            output_type = f'autocorr_bp-{highpass}-{lowpass}hz'
        elif highpass is not None:
            output_type = f'autocorr_hp-{highpass}hz'
        elif lowpass is not None:
            output_type = f'autocorr_lp-{lowpass}hz'
        if suffix:
            output_type = f'{output_type}_{suffix}'
        if use_residuals:
            output_type = f'{output_type}_resids'

        # lowpass/highpass are used for output naming only; filtering is applied at load time via load_batch
        def _compute(timeseries_scan, kv_dict):
            return compute_acf(timeseries_scan, use_fft=use_fft)

        self.autocorr_dict = self._run_for_all_scans(
            compute_fn=_compute,
            output_type=output_type,
            use_residuals=use_residuals,
            force_recompute=force_recompute,
            save_outputs=save_outputs,
            verbose=verbose,
        )

    def compute_timescale_measures(
        self,
        method: str = 'zero_crossing',
        force_recompute: bool = False,
        save_outputs: bool = True,
        suffix: str = None,
        use_residuals: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Compute timescale measures from the autocorrelation functions.

        Applies the chosen method from TIMESCALE_METHODS to each scan's ACF in
        self.autocorr_dict, storing results in self.timescales_dict (with disk
        caching). Requires compute_autocorrelation_function to have been run.

        Parameters
        ----------
        method : str
            Key into TIMESCALE_METHODS (e.g. 'zero_crossing', 'tau').
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        suffix : str or None
            Extra tag appended to output_type.
        use_residuals : bool
            Accepted for API symmetry; ignored here (residual handling is upstream
            in compute_autocorrelation_function).
        verbose : bool
            Print summary counts.
        """
        if not hasattr(self, 'autocorr_dict') or self.autocorr_dict is None:
            raise ValueError("Must compute autocorrelation first")

        if method not in TIMESCALE_METHODS:
            raise ValueError(
                f"Unknown timescale method '{method}'. "
                f"Available: {list(TIMESCALE_METHODS)}"
            )
        method_fn = TIMESCALE_METHODS[method]

        output_type = f'timescale_{method}'
        if suffix:
            output_type = f'{output_type}_{suffix}'

        # compute_timescale_measures keeps its own loop (rather than using
        # _run_for_all_scans) because it needs to look up autocorr_dict by
        # scan_key rather than using timeseries_data/timeseries_residuals.
        results = dict.fromkeys(self.scans)
        n_loaded = n_computed = n_failed = 0

        for scan_key in tqdm(self.scans, desc=f"Computing {method} timescales"):
            kv_dict = self._parse_scan_string(scan_key)
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')
            tr         = kv_dict.get('tr', '')

            if not force_recompute:
                cached = self._load_subject_output(
                    subject_id=subject_id, output_type=output_type,
                    task=task, modality=modality, seg=seg,
                    space=space, session=session, run=run,
                )
                if cached is not None:
                    results[scan_key] = cached
                    n_loaded += 1
                    continue

            acf = self.autocorr_dict.get(scan_key)
            if acf is None:
                results[scan_key] = None
                continue

            try:
                tr_float = float(tr) if tr else 1.0
                result = method_fn(acf, tr_float, inst=self)
                results[scan_key] = result
                n_computed += 1

                if save_outputs:
                    self._save_subject_output(
                        data=result,
                        subject_id=subject_id,
                        output_type=output_type,
                        task=task,
                        modality=modality,
                        seg=seg,
                        space=space,
                        session=session,
                        run=run,
                        metadata={'method': method, 'tr': tr},
                    )
            except Exception as e:
                logger.warning("Failed to compute %s for %s: %s", method, scan_key, e)
                results[scan_key] = None
                n_failed += 1

        self.timescales_dict = results

        if verbose:
            print(f"\n{method} timescale computation complete:")
            print(f"  Loaded from cache: {n_loaded}")
            print(f"  Newly computed:    {n_computed}")
            if n_failed:
                print(f"  Failed:            {n_failed}")
            print(f"  Total:             {len(self.scans)}")

    def compute_timeseries_measures(
        self,
        method: str = 'tsnr',
        force_recompute: bool = False,
        save_outputs: bool = True,
        use_residuals: bool = False,
        verbose: bool = True,
    ):
        """Compute a per-scan timeseries metric into self.data_dict.

        Supports 'tsnr' (temporal SNR) and 'var' (temporal variance), applied per
        scan via _run_for_all_scans with disk caching.

        Parameters
        ----------
        method : {'tsnr', 'var'}
            Metric to compute.
        force_recompute : bool
            Ignore cached files and recompute.
        save_outputs : bool
            Persist results to disk.
        use_residuals : bool
            Use self.timeseries_residuals for task scans instead of raw timeseries.
        verbose : bool
            Print summary counts.
        """
        output_type = f'metric_{method}'
        if use_residuals:
            output_type = f'{output_type}_resids'

        _METRIC_FNS = {
            'tsnr': calculate_tsnr,
            'var':  lambda ts: np.var(ts, axis=1),
        }
        if method not in _METRIC_FNS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(_METRIC_FNS)}")
        metric_fn = _METRIC_FNS[method]

        def _compute(timeseries_scan, kv_dict):
            return metric_fn(timeseries_scan)

        self.data_dict = self._run_for_all_scans(
            compute_fn=_compute,
            output_type=output_type,
            use_residuals=use_residuals,
            force_recompute=force_recompute,
            save_outputs=save_outputs,
            verbose=verbose,
        )

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

    def _compute_zero_crossing_robust(self, acf: np.ndarray,
                                      n_crossings: int = 3,
                                      max_gap: int = 20,
                                      aggregation: str = 'mean') -> np.ndarray:
        """Find first N zero crossings for each parcel and aggregate.
        
        Only includes crossings that are close together (quick oscillations).
        Stops including crossings once a gap larger than max_gap is encountered.
        
        Parameters
        ----------
        acf : np.ndarray
            Autocorrelation function, shape (n_parcels, n_lags)
        n_crossings : int, default=3
            Maximum number of early zero crossings to sample
        max_gap : int, default=20
            Maximum allowed gap (in lags) between consecutive crossings.
            Stops including crossings once gap exceeds this threshold.
        aggregation : str, default='mean'
            Aggregation method: 'mean' or 'median'

        Returns
        -------
        np.ndarray
            Aggregated zero crossing lag for each parcel, shape (n_parcels,)
        """
        n_parcels = acf.shape[0]
        zero_crossings_agg = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            # Find all negative indices
            negative_indices = np.where(acf[i] < 0)[0]
            
            if len(negative_indices) == 0:
                zero_crossings_agg[i] = np.nan
            else:
                # Select crossings that are close together
                selected_crossings = [negative_indices[0]]  # Always include first crossing
                
                for j in range(1, min(len(negative_indices), n_crossings)):
                    gap = negative_indices[j] - negative_indices[j-1]
                    if gap <= max_gap:
                        selected_crossings.append(negative_indices[j])
                    else:
                        # Gap too large, stop collecting
                        break
                
                # Aggregate
                if aggregation == 'mean':
                    zero_crossings_agg[i] = np.mean(selected_crossings)
                elif aggregation == 'median':
                    zero_crossings_agg[i] = np.median(selected_crossings)
                else:
                    raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")
        
        return zero_crossings_agg
    
    def _compute_auc(self, acf: np.ndarray, max_lag: int = None, threshold: float = 0) -> np.ndarray:
        """Compute area under curve up to some max lag (or zero crossing)."""
        n_parcels = acf.shape[0]
        auc = np.zeros(n_parcels)
        
        for i in range(n_parcels):
            try:
                if max_lag is not None:
                    max_lag_i = max_lag
                else:
                    max_lag_i = np.where(acf[i] < threshold)[0][0]

                # auc[i] = np.trapezoid(acf[i, :max_lag_i], dx=tr)
                auc[i] = np.sum(acf[i, :max_lag_i])
            except Exception:
                auc[i] = np.nan
        
        return auc
    
    def _compute_decay_time(self, acf: np.ndarray, tr: float, threshold: float = 1.0 / np.e) -> np.ndarray:
        """Find time when ACF decays to 1/e."""
        n_parcels = acf.shape[0]
        decay_times = np.zeros(n_parcels)
        
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
                elif version == 'howell':
                    results = fit_acf_exp(acf[i], tr=tr,
                                          start_lag=True, exclude_negative=False, exclude_inf=False,
                                          n_over_10_cap=False, max_fit_lag=None,
                                          a_const=True, b_const=True, maxfev=20000)
                    tau[i] = results['tau']
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
            except Exception:
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
        """Regress task-evoked structure out of every task scan's timeseries.

        For each task scan, builds a nilearn design matrix from its events
        (HRF-convolved for fMRI; raw for MEG), fits a per-region GLM, and keeps the
        residuals. Also stores the events, design matrices, per-condition beta
        contrast maps, and the ACF of the flattened task regressor. Rest scans and
        any failures are stored as None.

        Sets self.timeseries_residuals, self.task_events, self.design_matrices,
        self.task_events_acf, and self.task_contrasts (all keyed by scan_key).
        """
        timeseries_residuals = dict().fromkeys(self.timeseries_data)
        task_events = dict().fromkeys(self.timeseries_data)
        design_matrices = dict().fromkeys(self.timeseries_data)
        task_events_acf = dict().fromkeys(self.timeseries_data)
        task_contrasts = dict().fromkeys(self.timeseries_data)

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

                    # Reconstruct ScanInfo so get_task_events receives the correct type.
                    scan_info = ScanInfo(
                        subject_id=kv_dict.get('sub', ''),
                        task=kv_dict.get('task', ''),
                        space=kv_dict.get('space', ''),
                        seg=kv_dict.get('seg', ''),
                        modality=modality,
                        tr=float(tr) if tr else 0.0,
                        session=kv_dict.get('ses'),
                        run=kv_dict.get('run'),
                    )
                    # Create design matrix using nilearn
                    tsk_evnts = self.get_task_events(scan_info)
                    # replace all trials with a single trial types
                    # tsk_evnts['trial_type'] = 'trial'
                    design_matrix = create_design_matrix_nilearn(tsk_evnts, n_timepoints, tr,
                                                                 hrf_model=hrf_model, drift_model=drift_model
                                                                 )
                    # Fit GLM using nilearn approach
                    timeseries_residuals[scan_key], predicted, betas = fit_glm(timeseries, design_matrix)
                    task_events[scan_key] = tsk_evnts
                    design_matrices[scan_key] = design_matrix

                    # Extract task-based contrasts (beta maps) for each condition.
                    # Nuisance columns (drift_*, constant) are excluded; each remaining
                    # column's beta vector is a parcel-wise contrast map.
                    dm_columns = design_matrix.columns.tolist()
                    task_cols = [col for col in dm_columns
                                 if not col.startswith('drift_') and col != 'constant']
                    task_contrasts[scan_key] = {
                        col: betas[:, dm_columns.index(col)] for col in task_cols
                    }

                    # get autocorrelation of task events
                    tsk_evnts_flat = tsk_evnts.copy()
                    tsk_evnts_flat['trial_type'] = 'trial'
                    design_matrix_flat = create_design_matrix_nilearn(tsk_evnts_flat, n_timepoints, tr,
                                                                      hrf_model=hrf_model, drift_model=drift_model
                                                                      )
                    task_events_acf[scan_key] = compute_acf(design_matrix_flat['trial'].values)
                except Exception:
                    timeseries_residuals[scan_key] = None
                    task_events[scan_key] = None
                    design_matrices[scan_key] = None
                    task_events_acf[scan_key] = None
                    task_contrasts[scan_key] = None
            else:
                timeseries_residuals[scan_key] = None
                task_events[scan_key] = None
                design_matrices[scan_key] = None
                task_events_acf[scan_key] = None
                task_contrasts[scan_key] = None

        self.timeseries_residuals = timeseries_residuals
        self.task_events = task_events
        self.design_matrices = design_matrices
        self.task_events_acf = task_events_acf
        self.task_contrasts = task_contrasts
    
    # ==================== Analysis Functions ====================        
    def match_registry_to_behavior(self):
        """Subset behavioral data to the subjects present in the scan registry.

        Builds self.df_behavior_matched: one row per unique registry subject that
        also appears in self.df_behavior, preserving registry order.
        """
        df_behavior_matched = pd.DataFrame(columns=self.df_behavior.columns)
        df_behavior_matched.index.name = 'subject_id'
        
        for subject_id in list(dict.fromkeys(scan.subject_id for scan in self.scan_registry)):
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
        """Load previously saved per-scan outputs into self.data_dict.

        Iterates the scan registry and loads each scan's saved output_type file
        (if present) via _load_subject_output; scans with no saved file map to None.

        Parameters
        ----------
        output_type : str
            Output type to load (e.g. 'zero_crossing', 'timescale_tau').
        """
        data_dict = {}

        for scan_key in tqdm([str(scan) for scan in self.scan_registry], desc=f"Loading {output_type}"):
            # Parse scan key
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            modality   = kv_dict.get('modality', '')
            task       = kv_dict.get('task', '')
            seg        = kv_dict.get('seg', '')
            space      = kv_dict.get('space')
            session    = kv_dict.get('ses')
            run        = kv_dict.get('run')

            # Check for existing output
            loaded_data = self._load_subject_output(
                subject_id=subject_id,
                output_type=output_type,
                task=task,
                modality=modality,
                seg=seg,
                space=space,
                session=session,
                run=run,
            )
            
            data_dict[scan_key] = loaded_data
            
        self.data_dict = data_dict

    # ==================== Utility Functions ====================
    def get_min_scan_length_per_subject(self):
        """Find the shortest scan length (in frames) for each subject.

        Examines all scans in timeseries_data, extracts each scan's number of
        timepoints, and takes the per-subject minimum. Useful for choosing a
        truncation length for analyses requiring equal-length timeseries. Results
        are stored on the instance:

        - self.scan_lengths_df : per-scan DataFrame with columns
          ['scan_key', 'subject_id', 'task', 'modality', 'condition', 'n_frames'].
        - self.min_scan_lengths : Series indexed by subject_id, minimum n_frames.

        Raises
        ------
        ValueError
            If timeseries_data is empty or no scan has a determinable length.
        """
        if len(self.timeseries_data) == 0:
            raise ValueError("timeseries_data is empty.")
        
        # Collect scan length information
        scan_info = []
        
        for scan_key, timeseries in self.timeseries_data.items():
            # Parse scan key to extract metadata
            kv_dict = self._parse_scan_string(scan_key)
            
            subject_id = kv_dict.get('sub', '')
            task = kv_dict.get('task', '')
            modality = kv_dict.get('modality', '')
            condition = kv_dict.get('condition', '')
            
            # Determine number of frames
            if timeseries is None:
                n_frames = None
            else:
                try:
                    if timeseries.ndim == 2:
                        # Standard format: (n_regions, n_timepoints)
                        n_frames = timeseries.shape[1]
                    elif timeseries.ndim == 1:
                        # Single timeseries: (n_timepoints,)
                        n_frames = timeseries.shape[0]
                    else:
                        # Unexpected dimensionality
                        n_frames = None
                except Exception:
                    n_frames = None
            
            scan_info.append({
                'scan_key': scan_key,
                'subject_id': subject_id,
                'task': task,
                'modality': modality,
                'condition': condition,
                'n_frames': n_frames
            })
        
        # Create DataFrame
        self.scan_lengths_df = pd.DataFrame(scan_info)
        
        # Remove scans with None n_frames for aggregation
        valid_scans = self.scan_lengths_df[self.scan_lengths_df['n_frames'].notna()].copy()
        
        if len(valid_scans) == 0:
            raise ValueError("No valid scans found with determinable lengths.")
        
        # Find minimum scan length per subject
        self.min_scan_lengths = valid_scans.groupby('subject_id')['n_frames'].min()
        self.min_scan_lengths.name = 'min_n_frames'

# ==================== Standalone Functions ====================
def compute_acf(
        data,
        use_fft: bool = False,
    ):
    """Compute the autocorrelation function for timeseries data.

    Each series is mean/std standardized before the ACF is computed, then
    normalized by its lag-0 value. Series with zero variance return all-NaN.

    Parameters
    ----------
    data : ndarray
        2D array (n_parcels, n_timepoints), or 1D array (single series).
    use_fft : bool
        Use an FFT-based computation instead of np.correlate.

    Returns
    -------
    ndarray
        Autocorrelation functions, shape (n_parcels, n_lags).
    """
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = data.shape[1]
    
    autocorr_list = []
    
    for timeseries in data:
        if np.std(timeseries) > 0:  # Avoid division by zero
            # Standardize the (potentially filtered) time series
            standardized = (timeseries - np.mean(timeseries)) / np.std(timeseries)
            
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




def create_design_matrix_nilearn(
    events,
    n_timepoints,
    tr,
    hrf_model='glover',
    drift_model='cosine',
    high_pass=0.01,
    add_regs=None
    ):
    """Create a first-level design matrix using nilearn.

    Parameters
    ----------
    events : pd.DataFrame
        Events with columns onset, duration, trial_type.
    n_timepoints : int
        Number of timepoints (frames).
    tr : float
        Repetition time in seconds.
    hrf_model : str
        HRF model ('glover', 'spm', 'glover + derivative', ...).
    drift_model : str or None
        Drift model ('polynomial', 'cosine', or None).
    high_pass : float
        High-pass filter cutoff in Hz.
    add_regs : ndarray or None
        Additional regressors.

    Returns
    -------
    pd.DataFrame
        nilearn design matrix (task regressors, drift terms, constant).
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
    """Fit a GLM per region, masking NaN and zero-variance regions.

    Regions with any NaN or zero temporal variance are excluded from the fit and
    returned as NaN in all three outputs.

    Parameters
    ----------
    data : ndarray
        2D array (n_parcels, n_timepoints).
    design_matrix : pd.DataFrame
        nilearn design matrix.

    Returns
    -------
    residuals : ndarray
        Residual timeseries after GLM fitting (NaN for excluded regions).
    predicted : ndarray
        Predicted timeseries (NaN for excluded regions).
    betas : ndarray
        Regression coefficients, shape (n_parcels, n_regressors).
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
        return residuals, predicted, betas
    
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
    """Fit an exponential decay ACF(t) = exp(-t / tau) by log-linear regression.

    Fits log(ACF) = -t / tau + const over positive ACF values up to the first
    zero crossing (lag 0 excluded), following Gao et al. 2020.

    Parameters
    ----------
    lags : ndarray
        Lag times (seconds), same length as acf.
    acf : ndarray
        Autocorrelation values (acf[0] ~ 1).

    Returns
    -------
    tau : float
        Estimated timescale, or NaN if the fit failed or the slope was non-negative.
    fit_info : dict
        Fit diagnostics (slope, intercept, r_squared, n_points, error, ...).
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

def fit_acf_exp(acf_vals, tr,
                a_const=False, b_const=False,
                exclude_negative=True, exclude_inf=True,
                max_fit_lag=None, n_over_10_cap=False, start_lag=False,
                p0=(2.0, 1.0, 0.0),
                bounds=None, maxfev=20000):
    """
    Murray-style intrinsic timescale fit.

    Model (Murray form):
        R(kΔ) = A * (exp(-kΔ / tau) + B)

    where:
        - Δ is TR in seconds
        - k is the absolute lag index: k = start_idx, start_idx+1, ...
        - B is Murray's B (inside parentheses); additive offset is A*B

    Parameters
    ----------
    acf_vals : array-like
        1D array of ACF values, where acf_vals[0] should be ~1 (normalized).
    tr : float
        Repetition time in seconds (Δ in the model equation).
    a_const : bool, default=False
        If True, fix A = 1 and do not fit it.
    b_const : bool, default=False
        If True, fix B = 0 and do not fit it.
    exclude_negative : bool, default=True
        If True, truncate at first lag with acf <= 0.
    exclude_inf : bool, default=True
        If True, exclude non-finite acf values (nan/inf).
    max_fit_lag : int, optional
        If provided, fit only up to this absolute lag index (inclusive).
        E.g., max_fit_lag=20 means fit through lag 20 regardless of start_lag.
    n_over_10_cap : bool, default=True
        If True, additionally cap fit to N//10 lags (stability heuristic),
        where N = len(acf_vals) - 1.
    start_lag : bool or int, default=False
        Starting lag index for fit:
        - False or None: start at lag 1
        - True: start_idx = argmin(diff(acf)) (Murray heuristic)
        - int: explicit start lag index
    p0 : tuple, default=(2.0, 1.0, 0.0)
        Initial guesses in order (tau, A, B). Ignored for fixed parameters.
    bounds : tuple, optional
        Bounds for parameters passed to curve_fit. If None, reasonable defaults used.
    maxfev : int, default=20000
        Max function evaluations for curve_fit.

    Returns
    -------
    result : dict
        {
          "tau": float,
          "A": float,
          "B": float,
          "success": bool,
          "start_idx": int,
          "n_points": int,
          "lags_used": np.ndarray,
          "acf_used": np.ndarray
        }
    """
    # Input validation
    acf = np.asarray(acf_vals, dtype=float)
    if acf.ndim != 1 or acf.size < 2:
        return {
            "tau": np.nan, "A": np.nan, "B": np.nan, "success": False,
            "start_idx": None, "n_points": 0, 
            "lags_used": np.array([]), "acf_used": np.array([])
        }
    
    tr = float(tr)
    if tr <= 0:
        raise ValueError(f"tr must be positive, got {tr}")

    # Determine start index
    if start_lag is True:
        start_idx = int(np.argmin(np.diff(acf)))
    elif start_lag is False or start_lag is None:
        start_idx = 1
    else:
        start_idx = max(0, int(start_lag))

    if start_idx >= acf.size:
        return {
            "tau": np.nan, "A": np.nan, "B": np.nan, "success": False,
            "start_idx": start_idx, "n_points": 0,
            "lags_used": np.array([]), "acf_used": np.array([])
        }

    # Extract subset starting from start_idx with absolute lag indices
    acf_subset = acf[start_idx:].copy()
    k = np.arange(start_idx, start_idx + acf_subset.size, dtype=float)

    # Clean: exclude negative values
    if exclude_negative:
        bad = np.where(acf_subset <= 0)[0]
        if bad.size > 0:
            cut = int(bad[0])
            if cut == 0:
                return {
                    "tau": np.nan, "A": np.nan, "B": np.nan, "success": False,
                    "start_idx": start_idx, "n_points": 0,
                    "lags_used": np.array([]), "acf_used": np.array([])
                }
            acf_subset = acf_subset[:cut]
            k = k[:cut]

    # Clean: exclude non-finite values
    if exclude_inf:
        m = np.isfinite(acf_subset)
        acf_subset = acf_subset[m]
        k = k[m]

    if acf_subset.size < 3:
        return {
            "tau": np.nan, "A": np.nan, "B": np.nan, "success": False,
            "start_idx": start_idx, "n_points": int(acf_subset.size),
            "lags_used": k, "acf_used": acf_subset
        }

    # Apply fitting caps
    cap_points = None

    # Cap based on absolute max_fit_lag
    if max_fit_lag is not None:
        max_fit_lag = int(max_fit_lag)
        if max_fit_lag < start_idx:
            # max_fit_lag is before our start point - no data to fit
            return {
                "tau": np.nan, "A": np.nan, "B": np.nan, "success": False,
                "start_idx": start_idx, "n_points": 0,
                "lags_used": np.array([]), "acf_used": np.array([])
            }
        # Convert absolute lag to number of points from start_idx
        cap_points = max(2, max_fit_lag - start_idx + 1)

    # Cap based on N//10 heuristic (total ACF length)
    if n_over_10_cap:
        N = acf.size - 1
        cap_n10 = max(2, (N // 10) + 1)
        cap_points = cap_n10 if cap_points is None else min(cap_points, cap_n10)

    # Apply the cap
    if cap_points is not None and acf_subset.size > cap_points:
        acf_subset = acf_subset[:cap_points]
        k = k[:cap_points]

    # Prepare bounds
    if bounds is None:
        tau_lo = tr / 10.0
        if (not a_const) and (not b_const):
            bounds_use = ([0.0, tau_lo, -0.5], [2.0, np.inf, 0.5])  # (A, tau, B)
        elif a_const and (not b_const):
            bounds_use = ([tau_lo, -0.5], [np.inf, 0.5])            # (tau, B)
        elif (not a_const) and b_const:
            bounds_use = ([0.0, tau_lo], [2.0, np.inf])             # (A, tau)
        else:
            bounds_use = ([tau_lo], [np.inf])                       # (tau)
    else:
        bounds_use = bounds

    # Fit the model
    tau0, A0, B0 = p0

    try:
        if (not a_const) and (not b_const):
            def model(k, A, tau, B):
                return A * (np.exp(-k * tr / tau) + B)
            p0_use = [A0, tau0, B0]
            popt, _ = curve_fit(model, k, acf_subset, p0=p0_use, 
                               bounds=bounds_use, maxfev=maxfev)
            A_hat, tau_hat, B_hat = popt

        elif a_const and (not b_const):
            def model(k, tau, B):
                return (np.exp(-k * tr / tau) + B)
            p0_use = [tau0, B0]
            popt, _ = curve_fit(model, k, acf_subset, p0=p0_use, 
                               bounds=bounds_use, maxfev=maxfev)
            tau_hat, B_hat = popt
            A_hat = 1.0

        elif (not a_const) and b_const:
            def model(k, A, tau):
                return A * np.exp(-k * tr / tau)
            p0_use = [A0, tau0]
            popt, _ = curve_fit(model, k, acf_subset, p0=p0_use, 
                               bounds=bounds_use, maxfev=maxfev)
            A_hat, tau_hat = popt
            B_hat = 0.0

        else:  # Both A and B constant
            def model(k, tau):
                return np.exp(-k * tr / tau)
            p0_use = [tau0]
            popt, _ = curve_fit(model, k, acf_subset, p0=p0_use, 
                               bounds=bounds_use, maxfev=maxfev)
            tau_hat = popt[0]
            A_hat, B_hat = 1.0, 0.0

        return {
            "tau": float(tau_hat),
            "A": float(A_hat),
            "B": float(B_hat),
            "success": True,
            "start_idx": start_idx,
            "n_points": int(acf_subset.size),
            "lags_used": k,
            "acf_used": acf_subset
        }

    except (RuntimeError, ValueError, OptimizeWarning):
        return {
            "tau": np.nan,
            "A": (1.0 if a_const else np.nan),
            "B": (0.0 if b_const else np.nan),
            "success": False,
            "start_idx": start_idx,
            "n_points": int(acf_subset.size),
            "lags_used": k,
            "acf_used": acf_subset
        }

# ==================== Standalone Utility Functions (for use in notebooks) ====================
def compute_deflections(rest_timescales, task_timescales, absolute=False):
    """Compute task-vs-rest timescale deflections (task minus rest).

    Elementwise, so it works on any pair of matching-shaped arrays — typically 1-D
    (n_subjects,) for a single region, or 2-D (n_subjects, n_regions).

    Parameters
    ----------
    rest_timescales, task_timescales : array-like
        Matching-shaped timescale values. Series/DataFrame are converted to arrays.
    absolute : bool
        Return |task - rest| rather than the signed difference. Note that task
        shortens INT in ~95% of subject x region cells (HCP-YA WM), so the signed
        deflection is near-unidirectional and abs() is close to a sign flip; the
        signed value is directly interpretable as "amount of INT shortening".

    Returns
    -------
    ndarray
        Deflections, same shape as the inputs.

    Notes
    -----
    This function deliberately has no rest-INT control. Regressing rest out of
    (task - rest) is algebraically identical to residualising task on rest — the
    subtraction cancels exactly — which silently turns a deflection into a residual
    and destroys the S-A structure that fig2 tests. Control for rest INT at the
    analysis step instead (see compute_regional_corrs's `covariate`).
    """
    if isinstance(rest_timescales, (pd.Series, pd.DataFrame)):
        rest_timescales = rest_timescales.to_numpy()
    if isinstance(task_timescales, (pd.Series, pd.DataFrame)):
        task_timescales = task_timescales.to_numpy()

    rest_timescales = np.asarray(rest_timescales, dtype=float)
    task_timescales = np.asarray(task_timescales, dtype=float)

    if rest_timescales.shape != task_timescales.shape:
        raise ValueError(
            f"rest_timescales and task_timescales must have the same shape; "
            f"got {rest_timescales.shape} and {task_timescales.shape}."
        )

    int_deflections = task_timescales - rest_timescales
    return np.abs(int_deflections) if absolute else int_deflections


def compute_deflections_glm(rest_timescales, task_timescales, features, standardize=False):
    """Regress task timescales on each feature plus rest timescales (per feature).

    For each column of features, fits OLS task ~ const + feature + rest and reports
    the feature coefficient and stats. Also returns partial residuals of task and
    feature after removing rest (for residual scatter plots).

    Parameters
    ----------
    rest_timescales, task_timescales : array-like
        Per-region timescale vectors (n_regions,). Series are converted to arrays.
    features : pd.Series, pd.DataFrame, or ndarray
        One or more region-wise predictors; each column is modelled separately.
    standardize : bool
        Z-score rest, task, and each feature before fitting.

    Returns
    -------
    dict
        feature_name -> dict of model, coef, pvalue, tstat, conf_int, r_squared,
        r_squared_adj, task_partial_resid, feature_partial_resid, valid_mask.
    """
    if isinstance(rest_timescales, pd.Series):
        rest_timescales = rest_timescales.to_numpy()
    if isinstance(task_timescales, pd.Series):
        task_timescales = task_timescales.to_numpy()

    rest_timescales = np.asarray(rest_timescales, dtype=float)
    task_timescales = np.asarray(task_timescales, dtype=float)

    if standardize:
        rest_timescales = (rest_timescales - np.nanmean(rest_timescales)) / np.nanstd(rest_timescales)
        task_timescales = (task_timescales - np.nanmean(task_timescales)) / np.nanstd(task_timescales)

    n_regions = len(rest_timescales)

    if isinstance(features, pd.Series):
        features = features.to_frame()
    elif isinstance(features, np.ndarray):
        if features.ndim == 1:
            features = pd.DataFrame(features, columns=['feature_0'])
        else:
            features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])

    results = {}

    for feature_name in features.columns:
        feature_vals = features[feature_name].to_numpy().astype(float)

        if standardize:
            feature_vals = (feature_vals - np.nanmean(feature_vals)) / np.nanstd(feature_vals)

        valid_mask = ~(np.isnan(rest_timescales) | np.isnan(task_timescales) | np.isnan(feature_vals))

        rest_valid = rest_timescales[valid_mask]
        task_valid = task_timescales[valid_mask]
        feature_valid = feature_vals[valid_mask]

        X = sm.add_constant(np.column_stack((feature_valid, rest_valid)))
        y = task_valid

        model = sm.OLS(y, X).fit()

        X_rest = sm.add_constant(rest_valid)
        task_resid_valid = sm.OLS(task_valid, X_rest).fit().resid
        feature_resid_valid = sm.OLS(feature_valid, X_rest).fit().resid

        task_resid = np.full(n_regions, np.nan)
        feature_resid = np.full(n_regions, np.nan)
        task_resid[valid_mask] = task_resid_valid
        feature_resid[valid_mask] = feature_resid_valid

        results[feature_name] = {
            'model':                 model,
            'coef':                  model.params[1],
            'pvalue':                model.pvalues[1],
            'tstat':                 model.tvalues[1],
            'conf_int':              model.conf_int()[1],
            'r_squared':             model.rsquared,
            'r_squared_adj':         model.rsquared_adj,
            'task_partial_resid':    task_resid,
            'feature_partial_resid': feature_resid,
            'valid_mask':            valid_mask,
        }

    return results


def compute_regional_corrs(rest_data, task_data, behavior_data,
                           covariate=None, absolute=False, verbose=False, return_pval=False):
    """Correlate rest, task, and deflection timescales with a behavior, per region.

    For each region, correlates the behavior with rest, task, and the task-vs-rest
    deflection across subjects. Subjects with missing behavior are dropped first;
    regions with < 3 valid subjects are skipped. All three correlations for a region
    use the same subject set, so they are directly comparable.

    Parameters
    ----------
    rest_data, task_data : ndarray
        (n_subjects, n_regions) timescale matrices.
    behavior_data : ndarray
        (n_subjects,) behavioral values.
    covariate : ndarray or None
        (n_subjects,) subject-level covariate. When given, every correlation becomes a
        partial correlation controlling for it (df = n - 3).

        Intended use is each subject's MEAN rest INT, np.nanmean(rest_data, axis=1) —
        one scalar per subject, constant across regions, so it removes global INT
        differences without reshaping the regional map. This matches the nuisance
        covariate used by fig4's GAMs.

        Do NOT pass a region-specific rest vector. rest_data[:, r] carries the S-A
        gradient, so regressing it out strips the S-A structure these correlations are
        used to test — empirically it collapses fig2's S-A coupling from r ~ +0.56 to
        r ~ -0.18 (see the diagnostics cell in timescales_analysis_fig2.ipynb).
    absolute : bool
        Use |task - rest| instead of the signed deflection (see compute_deflections).
    verbose : bool
        Print the valid-subject count.
    return_pval : bool
        Also return the three p-value arrays.

    Returns
    -------
    tuple of ndarray
        (rest_corr, task_corr, deflection_corr), each (n_regions,); plus the
        matching p-value arrays when return_pval is True.
    """
    n_regions = rest_data.shape[1]
    rest_correlations       = np.full(n_regions, np.nan)
    task_correlations       = np.full(n_regions, np.nan)
    deflection_correlations = np.full(n_regions, np.nan)
    rest_pvalues            = np.full(n_regions, np.nan)
    task_pvalues            = np.full(n_regions, np.nan)
    deflection_pvalues      = np.full(n_regions, np.nan)

    behavior_data = np.asarray(behavior_data, dtype=float)

    beh_valid      = ~np.isnan(behavior_data)
    behavior_clean = behavior_data[beh_valid]
    rest_clean     = rest_data[beh_valid, :]
    task_clean     = task_data[beh_valid, :]

    covariate_clean = None
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float)
        if covariate.shape != behavior_data.shape:
            raise ValueError(
                f"covariate must be one value per subject, matching behavior_data "
                f"{behavior_data.shape}; got {covariate.shape}. Pass a subject-level "
                f"scalar such as np.nanmean(rest_data, axis=1), not a regional vector."
            )
        covariate_clean = covariate[beh_valid]

    if verbose:
        print(f"  Valid subjects: {np.sum(beh_valid)}/{len(behavior_data)}")

    deflections = compute_deflections(rest_clean, task_clean, absolute=absolute)

    for region_idx in range(n_regions):
        rest_region = rest_clean[:, region_idx]
        task_region = task_clean[:, region_idx]

        # Joint mask so rest / task / deflection all describe the same subject set.
        # (isnan(deflection) is exactly isnan(rest) | isnan(task).)
        valid = ~(np.isnan(rest_region) | np.isnan(task_region))
        if valid.sum() < 3:
            continue

        behavior_v  = behavior_clean[valid]
        covariate_v = None if covariate_clean is None else covariate_clean[valid]

        rest_correlations[region_idx], rest_pvalues[region_idx], _ = partial_pearsonr(
            rest_region[valid], behavior_v, covariate_v)
        task_correlations[region_idx], task_pvalues[region_idx], _ = partial_pearsonr(
            task_region[valid], behavior_v, covariate_v)
        deflection_correlations[region_idx], deflection_pvalues[region_idx], _ = partial_pearsonr(
            deflections[valid, region_idx], behavior_v, covariate_v)

    if return_pval:
        return (rest_correlations, task_correlations, deflection_correlations,
                rest_pvalues, task_pvalues, deflection_pvalues)
    else:
        return (rest_correlations, task_correlations, deflection_correlations)


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
                except Exception:
                    if remove_subject_level:
                        sub_key = scan_key.split('_')[0]
                        remove_these_keys.append([k for k in data_dict.keys() if sub_key in k])
                    else:
                        remove_these_keys.append([scan_key])
    
    # Flatten list and remove duplicates
    remove_these_keys = list(dict.fromkeys([item for sublist in remove_these_keys for item in sublist]))
    
    # Remove marked keys
    for remove_key in remove_these_keys:
        if remove_key in data_dict:
            del data_dict[remove_key]
    
    if data_dict_residuals is None:
        return data_dict
    else:
        return data_dict, data_dict_residuals

def _apply_scan_groupings(
    data_dict: dict,
    average_3t_rest: bool = False,
    average_7t_rest: bool = False,
    average_all_rest: bool = False,
    average_7t_movies: bool = False,
    average_3t_task_pairs: bool = False,
    average_3t_tasks: bool = False,
    average_meg_rest: bool = False,
    average_meg_task: bool = False,
) -> dict:
    """
    Apply scan-grouping/averaging logic to a data_dict.

    All keys in data_dict are expected to be scan-key strings; values are
    1-D arrays of shape (n_regions,).  Returns a new dict with the same
    structure after the requested averaging operations are applied.

    Raises ValueError on mutually-exclusive flag combinations.
    """
    if average_3t_rest and average_all_rest:
        raise ValueError("average_3t_rest and average_all_rest are mutually exclusive")
    if average_7t_rest and average_all_rest:
        raise ValueError("average_7t_rest and average_all_rest are mutually exclusive")
    if average_3t_task_pairs and average_3t_tasks:
        raise ValueError("average_3t_task_pairs and average_3t_tasks are mutually exclusive")

    out = dict(data_dict)  # shallow copy — we replace keys, not mutate values

    def _avg_group(keys_to_merge, new_key):
        """Average a list of keys into new_key; remove originals."""
        arrays = [out[k] for k in keys_to_merge if k in out and out[k] is not None]
        if not arrays:
            return
        out[new_key] = np.nanmean(np.stack(arrays, axis=0), axis=0)
        for k in keys_to_merge:
            out.pop(k, None)

    def _keys_matching(patterns):
        return [k for k in list(out) if any(p in k for p in patterns)]

    # HCP-YA 3T rest (LR/RL phase-encoding) and HCP-D 3T rest (AP/PA phase-encoding)
    _3t_rest_tasks  = ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL',   # HCP-YA
                       'rfMRIREST1PA', 'rfMRIREST2PA']   # HCP-D
                    #    'rfMRIREST1AP', 'rfMRIREST1PA', 'rfMRIREST2AP', 'rfMRIREST2PA']   # HCP-D
                    
    _7t_rest_tasks  = ['rfMRIREST17TPA', 'rfMRIREST27TAP', 'rfMRIREST37TPA', 'rfMRIREST47TAP']
    _7t_movie_tasks = ['tfMRIMOVIE17TAP', 'tfMRIMOVIE27TPA', 'tfMRIMOVIE37TPA', 'tfMRIMOVIE47TAP']
    _3t_task_pairs  = [
        # HCP-YA — LR/RL counterbalanced pairs
        (['tfMRIWMLR',          'tfMRIWMRL'],          'task-WM'),
        (['tfMRIEMOTIONLR',     'tfMRIEMOTIONRL'],     'task-EMOTION'),
        (['tfMRIGAMBLINGLR',    'tfMRIGAMBLINGRL'],    'task-GAMBLING'),
        (['tfMRILANGUAGELR',    'tfMRILANGUAGERL'],    'task-LANGUAGE'),
        (['tfMRIRELATIONALLR',  'tfMRIRELATIONALRL'],  'task-RELATIONAL'),
        (['tfMRISOCIALLR',      'tfMRISOCIALRL'],      'task-SOCIAL'),
        # HCP-D — single PA direction (no LR/RL counterpart)
        (['tfMRICARITPA'],                              'task-CARIT'),
        (['tfMRIEMOTIONPA'],                            'task-EMOTION'),
        (['tfMRIGUESSINGPA'],                           'task-GUESSING'),
    ]
    _meg_rest_tasks = ['3Restin', '4Restin', '5Restin']
    _meg_task_tasks = ['6Wrkmem', '7Wrkmem']

    if average_3t_rest:
        _avg_group(_keys_matching(_3t_rest_tasks), 'task-REST_3T')
    if average_7t_rest:
        _avg_group(_keys_matching(_7t_rest_tasks), 'task-REST_7T')
    if average_all_rest:
        _avg_group(_keys_matching(_3t_rest_tasks + _7t_rest_tasks), 'task-Rest')
    if average_7t_movies:
        _avg_group(_keys_matching(_7t_movie_tasks), 'task-Movies')
    if average_3t_task_pairs:
        for task_names, new_key in _3t_task_pairs:
            _avg_group(_keys_matching(task_names), new_key)
    if average_3t_tasks:
        all_3t_task_names = [t for pair, _ in _3t_task_pairs for t in pair]
        _avg_group(_keys_matching(all_3t_task_names), 'task-Tasks')
    if average_meg_rest:
        _avg_group(_keys_matching(_meg_rest_tasks), 'task-MEG_Rest')
    if average_meg_task:
        _avg_group(_keys_matching(_meg_task_tasks), 'task-MEG_Task')

    return out


def create_ints_dataframe_custom_grouping(
    data_dict,
    n_regions=400,
    average_3t_rest=False,
    average_7t_rest=False,
    average_all_rest=False,
    average_7t_movies=False,
    average_3t_task_pairs=False,
    average_3t_tasks=False,
    average_meg_rest=False,
    average_meg_task=False,
    sort_by_median=False,
):
    """
    Create a DataFrame (regions × scan-types) of mean INTs.

    Parameters
    ----------
    data_dict : dict
        scan_key → 1-D array of shape (n_regions,)
    n_regions : int
        Number of brain regions (used only to initialise the output DataFrame)
    average_* : bool
        Averaging flags — see _apply_scan_groupings for semantics.
    sort_by_median : bool
        Sort columns by descending median across regions.
    """
    def _extract_scan_name(scan_key):
        """Return the task-entity value from a scan key string."""
        for part in scan_key.split('_'):
            if part.startswith('task-'):
                return part[len('task-'):]
        return scan_key

    # Group by scan type (strip per-subject info), averaging across subjects
    by_scan_type = defaultdict(list)
    for scan_key, arr in data_dict.items():
        if arr is not None:
            by_scan_type[_extract_scan_name(scan_key)].append(arr)

    scan_type_means = {
        name: np.nanmean(np.stack(arrs, axis=0), axis=0)
        for name, arrs in by_scan_type.items()
    }

    # Apply grouping/averaging
    scan_type_means = _apply_scan_groupings(
        scan_type_means,
        average_3t_rest=average_3t_rest,
        average_7t_rest=average_7t_rest,
        average_all_rest=average_all_rest,
        average_7t_movies=average_7t_movies,
        average_3t_task_pairs=average_3t_task_pairs,
        average_3t_tasks=average_3t_tasks,
        average_meg_rest=average_meg_rest,
        average_meg_task=average_meg_task,
    )

    df = pd.DataFrame(scan_type_means, index=range(n_regions))

    if sort_by_median:
        order = df.median(axis=0).sort_values(ascending=False).index
        df = df[order]

    return df

def sort_and_group_timescales(timescales, brain_map, ntile=10):
    """Sort regions by a brain map and average them into equal-sized ntiles.

    Reorders regions by ascending brain_map value, then averages consecutive
    equal-sized blocks into `ntile` bins (nanmean across regions in each bin).

    Parameters
    ----------
    timescales : ndarray
        (n_subjects, n_regions) values.
    brain_map : ndarray
        (n_regions,) values used to sort regions (e.g. the S-A axis).
    ntile : int
        Number of equal-sized bins.

    Returns
    -------
    ndarray
        (n_subjects, ntile) block-averaged values.
    """
    n_nodes = len(brain_map)
    brain_map_sort_idx = np.argsort(brain_map)
    
    timescales_sorted = timescales[:, brain_map_sort_idx]
    timescales_sorted_dec = np.zeros((timescales_sorted.shape[0], ntile))

    step_size = int(n_nodes / ntile)
    counter = 0
    for i in np.arange(0, n_nodes, step_size):
        # timescales_sorted_dec[:, counter] = timescales_sorted[:, i:i+step_size].mean(axis=1)
        timescales_sorted_dec[:, counter] = np.nanmean(timescales_sorted[:, i:i+step_size], axis=1)
        counter += 1
        
    return timescales_sorted_dec

def convert_data_dict(data_dict, scan_registry, brain_map=None, verbose=True):
    """Regroup a flat scan_key -> array dict into a task -> stacked-array dict.

    For each unique task in the registry, stacks that task's per-subject arrays.
    If brain_map is given, each task's stack is additionally sorted and binned via
    sort_and_group_timescales.

    Parameters
    ----------
    data_dict : dict
        scan_key -> 1-D region array.
    scan_registry : list of ScanInfo
        Registry used to enumerate tasks and their scan keys.
    brain_map : ndarray or None
        If given, apply sort_and_group_timescales per task.
    verbose : bool
        Print each task's output shape.

    Returns
    -------
    dict
        task -> (n_subjects, n_regions) array, or (n_subjects, ntile) if brain_map
        was provided.
    """
    unique_tasks = list(dict.fromkeys(scan.task for scan in scan_registry))
    data_dict_converted = dict.fromkeys(unique_tasks)

    for task in data_dict_converted.keys():
        sub_list = [str(scan) for scan in scan_registry if scan.task == task]
        extracted_data = {key: data_dict[key] for key in sub_list if key in data_dict}
        extracted_data = np.array(list(extracted_data.values()))

        if brain_map is not None:
            extracted_data = sort_and_group_timescales(extracted_data, brain_map)

        data_dict_converted[task] = extracted_data
        
        if verbose:
            print(task, extracted_data.shape)
            
    return data_dict_converted

def average_data_dict_converted(
    data_dict_converted,
    average_7t_rest=False,
    average_3t_rest=False,
    average_all_rest=False,
    average_7t_movies=False,
    average_3t_task_pairs=False,
    average_3t_tasks=False,
    average_meg_rest=False,
    average_meg_task=False,
):
    """Average scan groups within a task-keyed dict produced by convert_data_dict."""
    # data_dict_converted is keyed by task name; values are (n_subjects, n_regions) arrays.
    # _apply_scan_groupings._keys_matching uses substring matching, so it works whether
    # the key is a full scan string or just a bare task name like 'rfMRIREST1LR'.
    flat = {task: arr for task, arr in data_dict_converted.items() if arr is not None}

    return _apply_scan_groupings(
        flat,
        average_3t_rest=average_3t_rest,
        average_7t_rest=average_7t_rest,
        average_all_rest=average_all_rest,
        average_7t_movies=average_7t_movies,
        average_3t_task_pairs=average_3t_task_pairs,
        average_3t_tasks=average_3t_tasks,
        average_meg_rest=average_meg_rest,
        average_meg_task=average_meg_task,
    )
