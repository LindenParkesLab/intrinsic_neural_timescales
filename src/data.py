import os, glob, re
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
class ScanInfo:
    """Container for preprocessed scan metadata matching BIDS naming conventions.
    
    Example filename:
        sub-100206_ses-01_task-rfMRIREST1LR_space-fsLR_seg-Schaefer4007_stat-mean_timeseries.ptseries.nii
        sub-100206_ses-01_task-tfMRIWMRL_space-fsLR_seg-Schaefer4007_stat-mean_timeseries.ptseries.nii
        sub-00395_ses-03_task-reversallearning_space-MNI152NLin2009cAsym_seg-Schaefer4007_stat-mean_timeseries.tsv
    """
    subject_id: str
    session: str  # e.g., '01', '02', '03'
    task: str  # e.g., 'rest', 'reversallearning', 'nback'
    space: str  # e.g., 'MNI152NLin2009cAsym', 'fsLR'
    seg: str  # segmentation, e.g., 'Schaefer4007', 'Glasser360'
    modality: str
    tr: float
    
    # Optional fields
    stat: Optional[str] = 'mean'  # Usually 'mean' for timeseries
    run: Optional[str] = None  # e.g., '01', '02' for multiple runs of same task
    desc: Optional[str] = None  # Optional description field
    den: Optional[str] = None  # Density for surface data, e.g., '91k', '32k'
    
    def __post_init__(self):
        if 'rest' in self.task.lower():
            self.condition = 'rest'
        else:
            self.condition = 'task'
    
    def __str__(self):
        """Get string version of scan"""
        entities = [
            f"sub-{self.subject_id}",
            f"ses-{self.session}",
            f"task-{self.task}",
            f"space-{self.space}",
            f"seg-{self.seg}"
        ]
        
        entities.append(f"modality-{self.modality}")
        entities.append(f"tr-{self.tr}")
        entities.append(f"condition-{self.condition}")
        
        if self.run is not None:
            entities.append(f"run-{self.run}")
        
        if self.den is not None:
            entities.append(f"den-{self.den}")
        
        if self.desc is not None:
            entities.append(f"desc-{self.desc}")
        
        return "_".join(entities)
        
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
    HCPYA_MEG_SAMPLING_RATE = 508.63
    
    # Default TR values for common modalities (in seconds)
    DEFAULT_TR = {
        'HCP-YA': {
            'MRI3T': 0.72,
            'MRI7T': 1.0,
            'MEG': 1 / HCPYA_MEG_SAMPLING_RATE
        },
        'HCP-D': {
            'MRI3T': 0.80            
        },
        'WAND': {
            'MRI3T': 2.0            
        },
    }
    
    # Map dataset names to base directories
    BASE_PATHS = {
        'HCP-YA': '/mnt/storage_ssd_raid/research_data/HCP_YA',
        'HCP-D': '/mnt/storage_ssd_raid/research_data/HCP_D',
        'WAND': '/mnt/storage_ssd_raid/research_data/WAND',
    }
    
    # Map dataset names to base directories
    RAWDATA_PATHS = {
        'HCP-YA': os.path.join(BASE_PATHS['HCP-YA'], 'HCP_1200'),
        'HCP-D': os.path.join(BASE_PATHS['HCP-D'], 'processed'),
        'WAND': os.path.join(BASE_PATHS['WAND'], 'bids'),
    }
    
    # Map dataset names to base directories
    DERIVDATA_PATHS = {
        'HCP-YA': os.path.join(BASE_PATHS['HCP-YA'], 'derivatives'),
        'HCP-D': os.path.join(BASE_PATHS['HCP-D'], 'derivatives'),
        'WAND': os.path.join(BASE_PATHS['WAND'], 'derivatives'),
    }
    
    # Subdirectory structure for different data types
    SUBDIR_MAP = {
        'HCP-YA': {
            'MRI3T': {'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'parcellated_data'),},
            'MRI7T': {'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'parcellated_data_7T'),},
            'MEG': {'meg': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'meg_parcellated'),}
        },
        'HCP-D': {
            'MRI3T': {'fmri': os.path.join(DERIVDATA_PATHS['HCP-D'], 'parcellated_data'),}
        },
        'WAND': {
            'MRI3T': {'fmri': os.path.join(DERIVDATA_PATHS['WAND'], 'xcpd'),}
        },
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
            self.output_dir = Path('/media/lindenmp/storage_ssd_1/research_projects/intrinsic_neural_timescales')
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up modality parameters (TR values)
        self.modality_params = self.DEFAULT_TR[dataset].copy()
        
        # Initialize scan registry
        self.scan_registry = []
        
        # Load behavioral data if requested
        self.df_behavior = None
        if load_behavior:
            self.load_behavioral_data()
        
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
                
    def get_tr(self, modality: str) -> float:
        """Get the TR for a given modality."""
        if modality not in self.modality_params:
            raise ValueError(f"Unknown modality: {modality}")
        return self.modality_params[modality]

    # ==================== Registration Methods ====================
    def register_scans(
        self,
        subject_ids: List[str],
        sessions: List[str],
        tasks: List[str],
        space: str,
        seg: str,
        modality: str
    ) -> List['ScanInfo']:
        """
        Register XCP-D preprocessed scans for given subjects.
        
        Args:
            subject_ids: List of subject IDs (e.g., ['00395', '00396'])
        
        Returns:
            List of registered ScanInfo objects
        """
        tr = self.get_tr(modality=modality)
        registered = []
        for subject_id in subject_ids:
            for session in sessions:
                for task in tasks:
                    scan = ScanInfo(
                        subject_id=str(subject_id),
                        session=session,
                        task=task,
                        space=space,
                        seg=seg,
                        modality=modality,
                        tr=tr
                    )
                    if scan not in self.scan_registry:
                        self.scan_registry.append(scan)
                    registered.append(scan)
        
        return registered

    def validate_scan_registry(
        self, 
        verbose: bool = True,
        remove_incomplete_subjects: bool = False
    ) -> Dict[str, List]:
        """
        Check which scans in the registry exist and automatically remove non-existent ones.
        Optionally remove all scans for subjects with any missing data.
        
        Args:
            verbose: Print detailed information
            remove_incomplete_subjects: If True, remove all scans for any subject 
                                    that had missing scans
            
        Returns:
            Dictionary with keys:
                - 'existing': List of scans that exist
                - 'missing': List of tuples (scan, filepath/error) for missing scans
                - 'removed_subjects': List of subject IDs removed due to incomplete data
        """
        if not self.scan_registry:
            if verbose:
                print("Scan registry is empty.")
            return {'existing': [], 'missing': [], 'removed_subjects': []}
        
        initial_count = len(self.scan_registry)
        
        if verbose:
            print(f"Validating {initial_count} scans in registry...")

        existing_scans = []
        missing_scans = []
        
        # Check each scan in the registry
        for scan in self.scan_registry:
            try:
                filepath = self.get_scan_filepath(scan)
                task_events = self.get_task_events(scan)
                
                # if filepath.exists() and scan.condition == 'task' and task_events is not None:
                #     existing_scans.append(scan)
                # elif filepath.exists() and scan.condition == 'rest' and task_events is None:
                #     existing_scans.append(scan)
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
        
        removed_subjects = []
        
        # Optionally remove all scans for subjects with any missing data
        if remove_incomplete_subjects and missing_scans:
            if verbose:
                print(f"\nRemoving all scans for subjects with incomplete data...")
            
            # Identify subjects that had missing scans
            subjects_with_missing = set(scan.subject_id for scan, _ in missing_scans)
            
            # Remove all remaining scans for these subjects
            scans_to_remove = [
                scan for scan in self.scan_registry 
                if scan.subject_id in subjects_with_missing
            ]
            
            for scan in scans_to_remove:
                self.scan_registry.remove(scan)
                if verbose:
                    print(f"  Removed: {scan} (subject {scan.subject_id} incomplete)")
            
            removed_subjects = sorted(list(subjects_with_missing))
            
            if verbose:
                print(f"\nRemoved {len(scans_to_remove)} additional scans from "
                    f"{len(removed_subjects)} subjects with incomplete data")
        
        # Summary
        if verbose:
            print(f"\nValidation Summary:")
            print(f"  Initially: {initial_count} scans")
            print(f"  Existing: {len(existing_scans)}")
            print(f"  Missing/Removed:  {len(missing_scans)} scans")
            if remove_incomplete_subjects:
                print(f"  Incomplete subjects removed: {len(removed_subjects)} subjects")
                print(f"    Additional scans removed: {len(existing_scans) - len(self.scan_registry)} scans")
            print(f"  Final registry: {len(self.scan_registry)} scans")
        
        return {
            'existing': existing_scans,
            'missing': missing_scans,
            'removed_subjects': removed_subjects
        }

    # ==================== Loading Methods ====================
    def load_behavioral_data(self):
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

    def get_task_events(self, scan):
        # Parse scan key
        kv_dict = parse_scan_string(str(scan))
        # print(kv_dict)
        
        subject_id = kv_dict.get('sub', '')
        task = kv_dict.get('task', '')
        modality = kv_dict.get('modality', '')
        
        if self.dataset == 'HCP-D':
            subject_id = split_hcpd_string(subject_id)
        
        try:
            if modality == 'MEG':
                subdir_str = f'{task[:1]}-{task[1:]}'
                task_events_file_path = glob.glob(os.path.join(self.rawdata_dir, subject_id, 'unprocessed', 'MEG', subdir_str, 'EPRIME', f'{subject_id}_*_{task[1:]}_*.xlsx'))[0]
                events = get_task_events_meg(task_events_file_path=task_events_file_path, verbose=False)
                task_events = events[['onset', 'duration', 'trial_type']]
            elif modality == 'MRI3T':
                results_dir = os.path.join(self.rawdata_dir, subject_id, 'MNINonLinear', 'Results')
                contents = os.listdir(results_dir)
                contents_joined = [''.join(content.split('_')) for content in contents]
                idx = contents_joined.index(task)
                ev_dir = os.path.join(results_dir, contents[idx], 'EVs')
                task_events = get_task_events_fmri(ev_dir)
            elif modality == 'MRI7T':
                movie_metadata_dir = os.path.join(self.output_dir, 'data', 'movie_data')
                
                if 'MOVIE1' in task:
                    timing_file = '7T_MOVIE1_CC1_v2_Video_Timing.csv'
                elif 'MOVIE2' in task:
                    timing_file = '7T_MOVIE2_HO1_v2_Video_Timing.csv'
                elif 'MOVIE3' in task:
                    timing_file = '7T_MOVIE3_CC2_v2_Video_Timing.csv'
                elif 'MOVIE4' in task:
                    timing_file = '7T_MOVIE4_HO2_v2_Video_Timing.csv'
                
                block_design_file = os.path.join(movie_metadata_dir, timing_file)
                block_design_data = pd.read_csv(block_design_file)
                block_design_data = block_design_data.loc[block_design_data['movie.block'] != 0.0]

                # Create task_events DataFrame for nilearn (task blocks only by default)
                task_events = pd.DataFrame({
                    'onset': block_design_data['block_start_time'],
                    'duration': block_design_data['block_duration'], 
                    'trial_type': 'movie'
                }).dropna().sort_values('onset').reset_index(drop=True)
        
            return task_events
        except:
            return None
    
    def get_scan_filepath(self, scan: ScanInfo) -> Path:
        """
        Get the exact filepath for an HCP scan.
        
        Args:
            scan: ScanInfo object
        
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
        subject_dir = base_path / f'sub-{scan.subject_id}' / f'ses-{scan.session}' / 'func'
        
        # Construct filename
        filename_base = f"sub-{scan.subject_id}_ses-{scan.session}_task-{scan.task}_space-{scan.space}_seg-{scan.seg}_stat-mean_timeseries"
        
        # Extension depends on modality
        if self.dataset == 'HCP-YA' and scan.modality == 'MEG':
            filename = f"{filename_base}.ptseries.npy"
        elif self.dataset == 'HCP-YA' and scan.modality != 'MEG':
            filename = f"{filename_base}.ptseries.nii"
        elif self.dataset == 'HCP-D':
            filename = f"{filename_base}.ptseries.nii"
        else:
            filename = f"{filename_base}.tsv"
            
        return subject_dir / filename
    
    def load_scan_timeseries(
        self,
        scan: ScanInfo,
        cache: bool = True,
        nuisance_regression: bool = True,
        standardize: bool = False,
        verbose: bool = False,
        dry_load: bool = False
    ) -> np.ndarray:
        """
        Load timeseries for a single HCP scan.
        
        Args:
            scan: ScanInfo object
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
        elif filepath.suffix == '.tsv':
            df = pd.read_csv(filepath, sep='\t')
            timeseries = df.values  # shape: (n_timepoints, n_parcels)
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
        if standardize:
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
        seg: Optional[str] = None,
        cache: bool = False,
        nuisance_regression: bool = True,
        standardize: bool = False,
        verbose: bool = True,
        dry_load: bool = False
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Load multiple scans with filtering.
        
        Args:
            subjects: Filter to specific subjects
            modalities: Filter to specific modalities
            conditions: Filter to specific conditions
            seg: Filter to specific seg
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
            if seg and scan.seg != seg:
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
                    print(f"  [{i+1}/{len(scans_to_load)}] {scan.subject_id} {scan.task}")
                
                loaded[str(scan)] = self.load_scan_timeseries(
                    scan, 
                    cache=cache, 
                    verbose=False, 
                    nuisance_regression=nuisance_regression,
                    standardize=standardize,
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
            kv_dict = {'scan_string': scan_str, **kv_dict}
            scan_data.append(kv_dict)
        
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
        
# ==================== Standalone Functions ====================
def parse_scan_string(scan_string: str) -> Dict[str, str]:
    """
    Parse a key-value formatted scan string into a dictionary.
    
    Args:
        scan_string: String like "sub-100307_modality-MRI3T_task-rfMRIREST1LR_atlas-Schaefer4007"
    
    Returns:
        Dictionary with keys: sub, modality, task, seg, etc.
    """
    parts = scan_string.split('_')
    kv_dict = {}
    for part in parts:
        if '-' in part:
            key, value = part.split('-', 1)  # Split only on first hyphen
            kv_dict[key] = value
            
    if 'tr' in kv_dict:
        kv_dict['tr'] = float(kv_dict['tr'])
        
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
    key_order = ['sub', 'modality', 'task', 'seg', 'stat']
    
    for key in key_order:
        if key in kwargs and kwargs[key] is not None:
            parts.append(f"{key}-{kwargs[key]}")
    
    # Add any additional keys not in the standard order
    for key, value in kwargs.items():
        if key not in key_order and value is not None:
            parts.append(f"{key}-{value}")
    
    return '_'.join(parts)

def get_task_events_fmri(ev_dir):
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

def get_task_events_meg(task_events_file_path, 
                        onset_column='Stim.OnsetTime',
                        duration_ms=2000,
                        verbose=True
                        ):
    """
    Extract trial onsets and durations from MEG working memory task Excel file.
    
    Parameters:
    -----------
    task_events_file_path : str
        Path to the MEG working memory Excel file (e.g., '100307_MEG_Wrkmem_run1.xlsx')
    onset_column : str, default 'Stim.OnsetTime'
        Column to use for trial onset timing. Options:
        - 'Stim.OnsetTime': When stimulus appears (recommended)
        - 'CueTarget.OnsetTime': When cue/target appears (sparse data)
        - 'SyncSlide.OnsetTime': Synchronization events
        - 'GetReady.OnsetTime': Get ready cue onset
    duration_ms : float, default 2000
        Duration of each trial in milliseconds
    
    Returns:
    --------
    events_df : pd.DataFrame
        DataFrame with columns ['onset', 'duration', 'trial_type', 'block_number'] in seconds
        where trial_type is '0back'/'2back' and block_number is the block index
        suitable for use with nilearn GLM functions
    """
    
    # Read Excel file
    try:
        data = pd.read_excel(task_events_file_path, sheet_name=0)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")
    
    # Check if onset column exists and has sufficient data
    if onset_column not in data.columns:
        available_cols = [col for col in data.columns if 'OnsetTime' in col]
        raise ValueError(f"Column '{onset_column}' not found. Available timing columns: {available_cols}")
    
    timing_data_count = data[onset_column].notna().sum()
    if verbose:
        print(f"Timing column '{onset_column}' has {timing_data_count}/{len(data)} non-null values")
    
    # Remove rows where onset time is missing or where BlockType is missing (non-task trials)
    valid_trials = data[onset_column].notna() & data['BlockType'].notna()
    data_clean = data[valid_trials].copy()
    
    if len(data_clean) == 0:
        raise ValueError(f"No valid trials found with both {onset_column} and BlockType data")
    
    if verbose:
        print(f"Found {len(data_clean)} valid trials (removed {len(data) - len(data_clean)} non-task trials)")
    
    # Extract onsets and convert from milliseconds to seconds
    onsets = data_clean[onset_column].values
    onsets = onsets / 1000.0  # Convert ms to seconds
    
    # Set durations (convert from ms to seconds)
    duration_sec = duration_ms / 1000.0
    durations = np.full(len(onsets), duration_sec)
    
    # Determine trial types (0back/2back from BlockType) and block numbers
    trial_types = determine_trial_types_fixed(data_clean)
    block_numbers = extract_block_numbers(data_clean)
    
    # Create events DataFrame
    events_df = pd.DataFrame({
        'onset': onsets,
        'duration': durations,
        'trial_type': trial_types,
        'block_number': block_numbers
    })
    
    # Sort by onset time
    events_df = events_df.sort_values('onset').reset_index(drop=True)
    
    # Print summary
    if verbose:
        print(f"\nTrial extraction summary:")
        print(f"- Total trials: {len(events_df)}")
        print(f"- Trial duration: {duration_sec:.3f} seconds")
        print(f"- Onset range: {events_df['onset'].min():.1f} to {events_df['onset'].max():.1f} seconds")
        print(f"- Trial types: {events_df['trial_type'].value_counts().to_dict()}")
        print(f"- Block numbers: {sorted(events_df['block_number'].unique())}")
    
    return events_df

def determine_trial_types_fixed(data):
    """Extract trial types (0back/2back) from BlockType column."""
    
    block_types = data['BlockType'].values
    trial_types = []
    for block_type in block_types:
        if pd.isna(block_type):
            trial_types.append('unknown')
        elif '0-Back' in str(block_type) or '0Back' in str(block_type):
            trial_types.append('0back')
        elif '2-Back' in str(block_type) or '2Back' in str(block_type):
            trial_types.append('2back')
        else:
            trial_types.append('unknown')
    return trial_types

def extract_block_numbers(data):
    """Extract block numbers from BlockNumber column."""
    
    block_numbers = data['BlockNumber'].values
    return [int(num) if not pd.isna(num) else -1 for num in block_numbers]

def split_hcpd_string(s):
    """Split string by inserting underscores before V and M."""
    return re.sub(r'([VM])', r'_\1', s)