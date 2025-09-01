import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ScanInfo:
    """Container for scan metadata."""
    subject_id: str
    condition: str  # 'rest', 'task', 'movie'
    scan_number: int
    atlas: str
    modality: str  # 'MRI_3T', 'MRI_7T', 'MEG'
    
    def __str__(self):
        return f"{self.subject_id}_{self.condition}_{self.scan_number}_{self.atlas}_{self.modality}"


class Timescales:
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
        'MRI_3T': 0.72,
        'MRI_7T': 1.0,
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
        'MRI_3T': {
            'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'fmri_parcellated'),
        },
        'MRI_7T': {
            'fmri': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'fmri_7T_parcellated'),
        },
        'MEG': {
            'meg': os.path.join(DERIVDATA_PATHS['HCP-YA'], 'meg_parcellated'),
        }
    }
    
    def __init__(
        self,
        dataset: str,
        base_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        load_behavior: bool = True
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
            self.output_dir = Path('~/intrinsic_neural_timescales').expanduser()
        
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
        
        print(f"Initialized Timescales for dataset: {dataset}")
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
    
    def register_scan(
        self,
        subject_id: str,
        condition: str,
        scan_number: int,
        atlas: str,
        modality: str
    ) -> ScanInfo:
        """
        Register a scan that will be analyzed.
        
        Args:
            subject_id: Subject identifier
            condition: Experimental condition ('rest', 'task', 'movie')
            scan_number: Scan number (1, 2, 3, etc.)
            atlas: Atlas name (e.g., 'Schaefer400')
            modality: Imaging modality ('MRI_3T', 'MRI_7T', 'MEG')
        
        Returns:
            ScanInfo object that was registered
        """
        scan = ScanInfo(
            subject_id=subject_id,
            condition=condition,
            scan_number=scan_number,
            atlas=atlas,
            modality=modality
        )
        self.scan_registry.append(scan)
        return scan
    
    def register_scans_from_list(
        self,
        scan_list: List[Dict]
    ) -> List[ScanInfo]:
        """
        Register multiple scans from a list of dictionaries.
        
        Args:
            scan_list: List of dictionaries with scan parameters
        
        Returns:
            List of registered ScanInfo objects
        """
        registered = []
        for scan_params in scan_list:
            scan = self.register_scan(**scan_params)
            registered.append(scan)
        return registered
    
    def get_data_path(
        self,
        scan: ScanInfo,
        data_type: str = 'fmri'
    ) -> Path:
        """
        Construct the path to data for a given scan.
        
        Args:
            scan: ScanInfo object
            data_type: Type of data ('fmri', 'meg', etc.)
        
        Returns:
            Path to the data directory/file
        """
        if scan.modality not in self.SUBDIR_MAP:
            raise ValueError(f"Unknown modality: {scan.modality}")
        
        if data_type not in self.SUBDIR_MAP[scan.modality]:
            raise ValueError(f"Unknown data type '{data_type}' for modality '{scan.modality}'")
        
        subdir = self.SUBDIR_MAP[scan.modality][data_type]
        return self.base_dir / subdir
    
    def get_tr(self, modality: str) -> float:
        """Get the TR for a given modality."""
        if modality not in self.modality_params:
            raise ValueError(f"Unknown modality: {modality}")
        return self.modality_params[modality]
    
    def list_registered_scans(self) -> pd.DataFrame:
        """Return a DataFrame of all registered scans."""
        if not self.scan_registry:
            return pd.DataFrame()
        
        scan_data = []
        for scan in self.scan_registry:
            scan_data.append({
                'subject_id': scan.subject_id,
                'condition': scan.condition,
                'scan_number': scan.scan_number,
                'atlas': scan.atlas,
                'modality': scan.modality,
                'tr': self.get_tr(scan.modality)
            })
        
        return pd.DataFrame(scan_data)
    
    def __repr__(self):
        return (f"Timescales(dataset='{self.dataset}', "
                f"n_registered_scans={len(self.scan_registry)})")
    
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
        
        print(f"{'='*50}\n")