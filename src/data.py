import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression

# Enhanced ScanInfo dataclass that matches HCP conventions
@dataclass
class HCPScanInfo:
    """Container for HCP scan metadata matching exact naming conventions."""
    subject_id: str
    condition: str  # 'rest', 'task'
    atlas: str  # e.g., 'Schaefer4007'
    modality: str  # 'MRI3T', 'MRI7T', 'MEG'
    
    # Modality-specific fields
    # For 3T/7T fMRI:
    task_name: Optional[str] = None  # 'WM', 'GAMBLING', 'EMOTION', etc.
    phase_encoding: Optional[str] = None  # 'LR', 'RL' for 3T; 'PA', 'AP' for 7T
    scan_number: Optional[int] = None  # 1-2 for 3T REST, 1-4 for 7T REST/MOVIE
    
    # For MEG:
    meg_scan_number: Optional[int] = None  # 3-5 for REST, 6-7 for WM
    meg_task_name: Optional[str] = None  # 'Restin' or 'Wrkmem'
    
    def get_task_string(self) -> str:
        """Get the exact task string as it appears in HCP filenames."""
        
        if self.modality == 'MRI3T':
            if self.condition == 'rest':
                # 3T REST: rfMRIREST1LR, rfMRIREST2RL, etc.
                return f"rfMRIREST{self.scan_number}{self.phase_encoding}"
            elif self.condition == 'task':
                # 3T Task: tfMRIEMOTIONLR, tfMRIWMRL, etc. (no underscore!)
                return f"tfMRI{self.task_name}{self.phase_encoding}"
                
        elif self.modality == 'MRI7T':
            if self.condition == 'rest':
                # 7T REST: rfMRIREST17TPA, rfMRIREST27TAP, etc.
                return f"rfMRIREST{self.scan_number}7T{self.phase_encoding}"
            elif self.condition == 'task':
                # 7T Movie: tfMRIMOVIE17TAP, tfMRIMOVIE27TPA, etc.
                return f"tfMRIMOVIE{self.scan_number}7T{self.phase_encoding}"
                
        elif self.modality == 'MEG':
            # MEG: 3-Restin, 6-Wrkmem, etc.
            return f"{self.meg_scan_number}{self.meg_task_name}"
        
        raise ValueError(f"Cannot construct task string for {self}")
    
    def __str__(self):
        """Return key-value formatted string representation."""
        task_str = self.get_task_string()
        return f"sub-{self.subject_id}_modality-{self.modality}_condition-{self.condition}_scan-{task_str}_atlas-{self.atlas}"


def parse_scan_string(scan_string: str) -> Dict[str, str]:
    """
    Parse a key-value formatted scan string into a dictionary.
    
    Args:
        scan_string: String like "sub-100307_modality-MRI3T_scan-rfMRIREST1LR_atlas-Schaefer4007"
    
    Returns:
        Dictionary with keys: sub, modality, scan, atlas, etc.
    """
    parts = scan_string.split('_')
    kv_dict = {}
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)  # Split only on first hyphen
            kv_dict[key] = value
    return kv_dict


def build_scan_string(**kwargs) -> str:
    """
    Build a scan string from key-value pairs.
    
    Args:
        **kwargs: Key-value pairs (e.g., sub='100307', modality='MRI3T', etc.)
    
    Returns:
        Formatted scan string
    """
    parts = []
    # Ensure consistent ordering
    key_order = ['sub', 'modality', 'scan', 'atlas', 'stat']
    
    for key in key_order:
        if key in kwargs and kwargs[key] is not None:
            parts.append(f"{key}-{kwargs[key]}")
    
    # Add any additional keys not in the standard order
    for key, value in kwargs.items():
        if key not in key_order and value is not None:
            parts.append(f"{key}-{value}")
    
    return '_'.join(parts)

class DataLoader:
    """
    Class for generating and analyzing neural timescales from preprocessed timeseries data.
    
    Attributes:
        dataset (str): Name of the dataset (e.g., 'HCP-YA')
        base_dir (Path): Base directory for all data
        output_dir (Path): Directory for saving outputs
        df_behavior (pd.DataFrame): Subject-level behavioral/demographic data
        modality_params (dict): TR and other parameters for each modality
        scan_registry (list): List of ScanInfo objects to track available scans
    """
    
    # MEG sampling rate in Hz
    MEG_SAMPLING_RATE = 508.63
    
    # Default TR values for common modalities (in seconds)
    DEFAULT_TR = {
        'MRI3T': 0.72,
        'MRI7T': 1.0,
        'MEG': 1 / MEG_SAMPLING_RATE
    }
    
    # Map dataset names to base directories
    BASE_PATHS = {
        'HCP-YA': '/mnt/storage_ssd_raid/research_data/HCP_YA',
    }
    
    # Map dataset names to base directories
    RAWDATA_PATHS = {
        'HCP-YA': os.path.join(BASE_PATHS['HCP-YA'], 'HCP_1200'),
    }
    
    # Map dataset names to base directories
    DERIVDATA_PATHS = {
        'HCP-YA': os.path.join(BASE_PATHS['HCP-YA'], 'derivatives'),
    }
    
    # Subdirectory structure for different data types
    SUBDIR_MAP = {
        'HCP-YA': {
            'MRI3T': {'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'fmri_parcellated'),},
            'MRI7T': {'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], '7T_fmri_parcellated'),},
            'MEG': {'meg': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'meg_parcellated'),}
        }
    }
    
    def __init__(
        self,
        dataset: str,
        base_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        load_behavior: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the Timescales class for a specific dataset.
        
        Args:
            dataset: Name of the dataset (e.g., 'HCP-YA')
            base_dir: Override the default base directory for this dataset
            output_dir: Directory for saving outputs (default: ~/home/intrinsic_neural_timescales)
            load_behavior: Whether to load the behavioral CSV on initialization
        """
        self.dataset = dataset
        
        # Set up base directory
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif dataset in self.BASE_PATHS:
            self.base_dir = Path(self.BASE_PATHS[dataset])
        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Please provide base_dir.")
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir}")
        
        self.rawdata_dir = self.RAWDATA_PATHS[dataset]
        self.derivdata_dir = self.DERIVDATA_PATHS[dataset]
        
        # Set up output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir).expanduser()
        else:
            self.output_dir = Path('/mnt/storage_ssd_raid/research_projects/intrinsic_neural_timescales')
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up modality parameters (TR values)
        self.modality_params = self.DEFAULT_TR.copy()
        
        # Initialize scan registry
        self.scan_registry = []
        
        # Load behavioral data if requested
        self.df_behavior = None
        if load_behavior:
            self._load_behavioral_data()
        
        # Initialize data cache (will store loaded timeseries)
        self._data_cache = {}
        
        if verbose:
            print(f"Initialized DataLoader for dataset: {dataset}")
            print(f"Base directory: {self.base_dir}")
            print(f"Raw data directory: {self.rawdata_dir}")
            print(f"Derivatives data directory: {self.derivdata_dir}")
            print(f"Output directory: {self.output_dir}")
            if self.df_behavior is not None:
                print(f"Loaded behavioral data: {len(self.df_behavior)} subjects")
    
    def _load_behavioral_data(self):
        """Load the behavioral CSV file for this dataset."""

        try:
            if self.dataset == 'HCP-YA':
                behavior_path_unrestricted = self.base_dir / 'unrestricted_lindenmp_9_26_2022_16_52_14.csv'
                behavior_path_restricted = self.base_dir / 'RESTRICTED_lindenmp_9_26_2022_16_42_3.csv'

                df_behavior_unrestricted = pd.read_csv(behavior_path_unrestricted, index_col=0)
                df_behavior_restricted = pd.read_csv(behavior_path_restricted, index_col=0)
                df_behavior = pd.merge(df_behavior_unrestricted, df_behavior_restricted, left_index=True, right_index=True)
                df_behavior['age'] = df_behavior['Age_in_Yrs']
                df_behavior['sex'] = df_behavior['Gender'] == 'M'
                df_behavior['sex'] = df_behavior['sex'].astype(int)
                df_behavior.index = df_behavior.index.astype(str)
                df_behavior.index.name = 'subject_id'
                
                # Apply replacements to column names
                words_to_convert = ['Emotion', 'Gambling', 'Language', 'Relational', 'Social']
                new_columns = df_behavior.columns.to_series()
                for word in words_to_convert:
                    new_columns = new_columns.str.replace(word, word.upper(), regex=False)
                df_behavior.columns = new_columns
            else:
                pass
            self.df_behavior = df_behavior
        except:
            print(f"Warning: Could not find df_behavior.csv at expected locations")

    def get_tr(self, modality: str) -> float:
        """Get the TR for a given modality."""
        if modality not in self.modality_params:
            raise ValueError(f"Unknown modality: {modality}")
        return self.modality_params[modality]

    # ==================== Registration Methods ====================
    def register_3t_rest_scans(
        self,
        subject_ids: List[str],
        atlas: str = 'Schaefer4007',
        sessions: List[int] = [1, 2],
        phase_encodings: List[str] = ['LR', 'RL']
    ) -> List[HCPScanInfo]:
        """
        Register 3T REST scans for given subjects.
        
        Args:
            subject_ids: List of subject IDs
            atlas: Atlas name (default 'Schaefer4007')
            sessions: REST sessions to include (1 and/or 2)
            phase_encodings: Phase encodings to include ('LR' and/or 'RL')
        
        Returns:
            List of registered HCPScanInfo objects
        """
        registered = []
        for subject_id in subject_ids:
            for session in sessions:
                for pe in phase_encodings:
                    scan = HCPScanInfo(
                        subject_id=str(subject_id),
                        condition='rest',
                        atlas=atlas,
                        modality='MRI3T',
                        scan_number=session,
                        phase_encoding=pe
                    )
                    
                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
        return registered
    
    def register_3t_task_scans(
        self,
        subject_ids: List[str],
        task_names: Optional[List[str]] = None,
        atlas: str = 'Schaefer4007',
        phase_encodings: List[str] = ['LR', 'RL']
    ) -> List[HCPScanInfo]:
        """
        Register 3T task scans for given subjects.
        
        Args:
            subject_ids: List of subject IDs
            task_names: Task names (default: all HCP tasks)
            atlas: Atlas name
            phase_encodings: Phase encodings to include
        
        Returns:
            List of registered HCPScanInfo objects
        """
        if task_names is None:
            # All HCP-YA 3T tasks
            task_names = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 
                         'RELATIONAL', 'SOCIAL', 'WM']
        
        registered = []
        for subject_id in subject_ids:
            for task_name in task_names:
                for pe in phase_encodings:
                    scan = HCPScanInfo(
                        subject_id=str(subject_id),
                        condition='task',
                        atlas=atlas,
                        modality='MRI3T',
                        task_name=task_name,
                        phase_encoding=pe
                    )
                    
                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
        return registered
    
    def register_7t_rest_scans(
        self,
        subject_ids: List[str],
        atlas: str = 'Schaefer4007',
        sessions: List[int] = [1, 2, 3, 4]
    ) -> List[HCPScanInfo]:
        """
        Register 7T REST scans for given subjects.
        Note: 7T REST uses alternating PA/AP phase encoding.
        
        Args:
            subject_ids: List of subject IDs
            atlas: Atlas name
            sessions: REST sessions to include (1-4)
        
        Returns:
            List of registered HCPScanInfo objects
        """
        # 7T REST phase encoding pattern: 1=PA, 2=AP, 3=PA, 4=AP
        pe_pattern = {1: 'PA', 2: 'AP', 3: 'PA', 4: 'AP'}
        
        registered = []
        for subject_id in subject_ids:
            for session in sessions:
                scan = HCPScanInfo(
                    subject_id=str(subject_id),
                    condition='rest',
                    atlas=atlas,
                    modality='MRI7T',
                    scan_number=session,
                    phase_encoding=pe_pattern[session]
                )
                
                if scan not in self.scan_registry:
                    self.scan_registry.append(scan)
                registered.append(scan)
        return registered
    
    def register_7t_movie_scans(
        self,
        subject_ids: List[str],
        atlas: str = 'Schaefer4007',
        movies: List[int] = [1, 2, 3, 4]
    ) -> List[HCPScanInfo]:
        """
        Register 7T movie scans for given subjects.
        Note: 7T movies use alternating AP/PA phase encoding.
        
        Args:
            subject_ids: List of subject IDs
            atlas: Atlas name
            movies: Movie numbers to include (1-4)
        
        Returns:
            List of registered HCPScanInfo objects
        """
        # 7T movie phase encoding pattern: 1=AP, 2=PA, 3=PA, 4=AP
        pe_pattern = {1: 'AP', 2: 'PA', 3: 'PA', 4: 'AP'}
        
        registered = []
        for subject_id in subject_ids:
            for movie_num in movies:
                scan = HCPScanInfo(
                    subject_id=str(subject_id),
                    condition='task',
                    atlas=atlas,
                    modality='MRI7T',
                    scan_number=movie_num,
                    phase_encoding=pe_pattern[movie_num]
                )
                
                if scan not in self.scan_registry:
                    self.scan_registry.append(scan)
                registered.append(scan)
        return registered
    
    def register_meg_scans(
        self,
        subject_ids: List[str],
        atlas: str = 'Schaefer4007',
        rest_runs: List[int] = None, # [3, 4, 5],
        wm_runs: List[int] = None # [6, 7]
    ) -> List[HCPScanInfo]:
        """
        Register MEG scans for given subjects.
        
        Args:
            subject_ids: List of subject IDs
            atlas: Atlas name
            rest_runs: REST run numbers (typically 3, 4, 5)
            wm_runs: Working memory run numbers (typically 6, 7)
        
        Returns:
            List of registered HCPScanInfo objects
        """
        registered = []
        
        for subject_id in subject_ids:
            # Register REST runs
            if rest_runs is not None:
                for run_num in rest_runs:
                    scan = HCPScanInfo(
                        subject_id=str(subject_id),
                        condition='rest',
                        atlas=atlas,
                        modality='MEG',
                        meg_scan_number=run_num,
                        meg_task_name='Restin'
                    )
                    
                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
            
            # Register WM runs
            if wm_runs is not None:
                for run_num in wm_runs:
                    scan = HCPScanInfo(
                        subject_id=str(subject_id),
                        condition='task',
                        atlas=atlas,
                        modality='MEG',
                        meg_scan_number=run_num,
                        meg_task_name='Wrkmem'
                    )

                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
        
        return registered
    
    def validate_scan_registry(self, verbose: bool = True) -> Dict[str, List]:
        """
        Check which scans in the registry exist and automatically remove non-existent ones.
        
        Args:
            verbose: Print detailed information
            
        Returns:
            Dictionary with 'existing' and 'missing' scan lists
        """
        if not self.scan_registry:
            if verbose:
                print("Scan registry is empty.")
            return {'existing': [], 'missing': [], 'removed': []}
        
        if verbose:
            print(f"Validating {len(self.scan_registry)} scans in registry...")
        
        existing_scans = []
        missing_scans = []
        
        # Check each scan in the registry
        for scan in self.scan_registry:
            try:
                filepath = self.get_scan_filepath(scan)
                if filepath.exists():
                    existing_scans.append(scan)
                else:
                    missing_scans.append((scan, filepath))
                    if verbose:
                        print(f"  Missing: {scan} -> {filepath}")
            except Exception as e:
                missing_scans.append((scan, f"Error: {e}"))
                if verbose:
                    print(f"  Error checking {scan}: {e}")
        
        # Automatically remove all missing scans
        for scan, _ in missing_scans:
            self.scan_registry.remove(scan)
        
        # Summary
        if verbose:
            print(f"\nValidation Summary:")
            print(f"  Initially: {len(self.scan_registry) + len(missing_scans)} scans")
            print(f"  Existing: {len(existing_scans)}")
            print(f"  Removed:  {len(missing_scans)}")
            print(f"  Final registry: {len(self.scan_registry)} scans")
        
        return {
            'existing': existing_scans,
            'missing': missing_scans
        }

    # ==================== Loading Methods ====================
    def get_scan_filepath(self, scan: HCPScanInfo) -> Path:
        """
        Get the exact filepath for an HCP scan.
        
        Args:
            scan: HCPScanInfo object
        
        Returns:
            Path to the timeseries file (without checking existence)
        """
        # Base paths from configuration
        if scan.modality == 'MRI3T':
            base_path = Path(self.SUBDIR_MAP[self.dataset]['MRI3T']['fmri'])
        elif scan.modality == 'MRI7T':
            base_path = Path(self.SUBDIR_MAP[self.dataset]['MRI7T']['fmri'])
        elif scan.modality == 'MEG':
            base_path = Path(self.SUBDIR_MAP[self.dataset]['MEG']['meg'])
        else:
            raise ValueError(f"Unknown modality: {scan.modality}")
        
        # Subject directory
        subject_dir = base_path / str(scan.subject_id)
        
        # Construct filename
        task_str = scan.get_task_string()
        filename_base = f"{scan.subject_id}_task-{task_str}_atlas-{scan.atlas}_stat-mean_timeseries"
        
        # Extension depends on modality
        if scan.modality == 'MEG':
            filename = f"{filename_base}.ptseries.npy"
        else:
            filename = f"{filename_base}.ptseries.nii"
        
        return subject_dir / filename
    
    def load_scan_timeseries(
        self,
        scan: HCPScanInfo,
        cache: bool = True,
        nuisance_regression: bool = True,
        verbose: bool = False,
        dry_load: bool = False
    ) -> np.ndarray:
        """
        Load timeseries for a single HCP scan.
        
        Args:
            scan: HCPScanInfo object
            cache: Whether to cache the data
            verbose: Print loading details
        
        Returns:
            Timeseries array of shape (n_parcels, n_timepoints)
        """
        # Check cache
        cache_key = str(scan)
        if cache and cache_key in self._data_cache:
            if verbose:
                print(f"  Using cached: {cache_key}")
            return self._data_cache[cache_key]
        
        # Get filepath
        filepath = self.get_scan_filepath(scan)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if verbose:
            print(f"  Loading: {filepath}")
        
        # Load based on file type
        if filepath.suffix == '.npy':
            timeseries = np.load(filepath)
        elif '.nii' in filepath.suffixes:
            img = nib.load(str(filepath))
            timeseries = img.get_fdata()
        else:
            raise ValueError(f"Unknown file type: {filepath}")
        
        # Convert to float32 for memory efficiency
        timeseries = timeseries.astype(np.float32)
        
        # Run nuisance regression
        if nuisance_regression:
            global_signal = np.mean(timeseries, axis=1).reshape(-1, 1)
            global_signal_diff = np.append(0, np.diff(global_signal, axis=0)).reshape(-1, 1)
            nuis = np.concatenate((global_signal, global_signal_diff), axis=1)
            nuis = np.concatenate((nuis, np.square(nuis)), axis=1)

            regr = LinearRegression()
            regr.fit(nuis, timeseries)
            y_pred = regr.predict(nuis)
            timeseries = timeseries - y_pred
        
        # z score
        timeseries = sp.stats.zscore(timeseries, axis=0)
        
        # Transpose data to (n_parcels, n_timepoints)
        timeseries = timeseries.T
        
        if verbose:
            print(f"    Shape: {timeseries.shape}")
        
        # Cache if requested
        if cache:
            self._data_cache[cache_key] = timeseries
        
        if dry_load:
            return None
        else:
            return timeseries
    
    def load_batch(
        self,
        subjects: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        atlas: Optional[str] = None,
        cache: bool = False,
        nuisance_regression: bool = True,
        verbose: bool = True,
        dry_load: bool = False
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Load multiple scans with filtering.
        
        Args:
            subjects: Filter to specific subjects
            modalities: Filter to specific modalities
            conditions: Filter to specific conditions
            atlas: Filter to specific atlas
            cache: Whether to cache data
            nuisance_regression: Whether to apply nuisance regression
            verbose: Print progress
            dry_load: If True, return scan keys with None values instead of loading data
        
        Returns:
            Dictionary mapping scan identifiers to timeseries (or None if dry_load=True)
        """
        # Filter scans
        scans_to_load = []
        for scan in self.scan_registry:
            if subjects and scan.subject_id not in [str(s) for s in subjects]:
                continue
            if modalities and scan.modality not in modalities:
                continue
            if conditions and scan.condition not in conditions:
                continue
            if atlas and scan.atlas != atlas:
                continue
            scans_to_load.append(scan)
        
        if not scans_to_load:
            raise ValueError("No scans match the filters")
        
        if verbose:
            print(f"{'Dry loading' if dry_load else 'Loading'} {len(scans_to_load)} scans...")
        
        # Load scans or return None for dry load
        loaded = {}
        failed = []
        
        for i, scan in enumerate(scans_to_load):
            try:
                if verbose and i % 10 == 0:
                    print(f"  [{i+1}/{len(scans_to_load)}] {scan.subject_id} {scan.get_task_string()}")
                
                loaded[str(scan)] = self.load_scan_timeseries(
                    scan, 
                    cache=cache, 
                    verbose=False, 
                    nuisance_regression=nuisance_regression,
                    dry_load=dry_load
                )
                
            except Exception as e:
                failed.append((scan, str(e)))
                if verbose and len(failed) <= 5:
                    print(f"    Failed: {e}")
        
        # Summary
        if verbose:
            print(f"\nLoaded: {len(loaded)}/{len(scans_to_load)} scans")
            if failed:
                print(f"Failed: {len(failed)} scans")
                for scan, error in failed[:3]:
                    print(f"  {scan.subject_id}: {error[:100]}")
        
        return loaded
    
    # ==================== Utility Methods ====================
    def clear_cache(self):
        """Clear the data cache to free memory."""
        self._data_cache.clear()
        print(f"Cleared data cache")
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache."""
        info = {
            'n_cached': len(self._data_cache),
            'cached_scans': list(self._data_cache.keys()),
            'memory_mb': sum(
                arr.nbytes / 1024 / 1024 
                for arr in self._data_cache.values()
            )
        }
        return info
    
    def list_registered_scans(self) -> pd.DataFrame:
        """Return a DataFrame of all registered scans with key-value naming."""
        if not self.scan_registry:
            return pd.DataFrame()
        
        scan_data = []
        for scan in self.scan_registry:
            # Parse the key-value string
            scan_str = str(scan)
            kv_dict = parse_scan_string(scan_str)
            
            scan_data.append({
                'scan_string': scan_str,
                'subject_id': kv_dict.get('sub', ''),
                'modality': kv_dict.get('modality', ''),
                'scan': kv_dict.get('scan', ''),
                'atlas': kv_dict.get('atlas', ''),
                'condition': scan.condition,
                'tr': self.get_tr(scan.modality)
            })
        
        return pd.DataFrame(scan_data)
    
    def export_scan_registry(self, file_name: str = 'scan_registry.csv'):
        """Export scan registry as a DataFrame."""

        scan_registry = self.list_registered_scans()
        scan_registry.to_csv(os.path.join(self.output_dir, file_name))

    def load_scan_registry(self, file_name: str = 'scan_registry.csv'):
        """ Load a pre-save scan registry"""
        try:
            self.scan_registry = pd.read_csv(os.path.join(self.output_dir, file_name), index_col=0)
        except:
            print(f'{file_name} not found!')

    def summary(self):
        """Print a summary of the current state."""
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset}")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Registered scans: {len(self.scan_registry)}")
        
        if self.scan_registry:
            df_scans = self.list_registered_scans()
            print(f"\nScans by modality:")
            print(df_scans['modality'].value_counts())
            print(f"\nScans by condition:")
            print(df_scans['condition'].value_counts())
        
        if self.df_behavior is not None:
            print(f"\nBehavioral data: {self.df_behavior.shape}")
        
        cache_info = self.get_cache_info()
        if cache_info['n_cached'] > 0:
            print(f"\nCache: {cache_info['n_cached']} scans, {cache_info['memory_mb']:.1f} MB")
        
        print(f"{'='*50}\n")
    
    def __repr__(self):
        return (f"Timescales(dataset='{self.dataset}', "
                f"n_registered_scans={len(self.scan_registry)})")