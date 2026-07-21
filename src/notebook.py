import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from scipy.spatial import distance

from src.timescales import Timescales
from src.timescales import filter_data_dict
from snaplab_tools.plotting.utils import create_sydnor_sa_colormap

from snaplab_tools.topology import threshold_adjacency_consistency, volume_normalize_adjacency
from snaplab_tools.utils import normalize_x
from snaplab_tools.decomposition import pca_with_nan_handling


# ==============================================================================
# Dataset processing parameters
# Consumed by timescales_process.ipynb and other processing notebooks.
# Each entry specifies what to pass to DataLoader.register_scans(), plus
# processing flags (nuisance_regression, task_regression, etc.).
# Modify here to change how a dataset is processed across all notebooks.
# ==============================================================================

DATASET_PARAMS = {
    'HCP-YA': {
        'modalities':  ['MRI3T'],  # also supports 'MRI7T', 'MEG'
        # 'modalities':  ['MRI3T', 'MRI7T'],  # also supports 'MRI7T', 'MEG'
        # 'modalities':  ['MEG'],  # also supports 'MRI7T', 'MEG'
        'seg':         'Schaefer4007',
        'space':       'fsLR',
        'sessions':    ['01'],
        'tasks': {
            'MRI3T': ['rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL',
                      'tfMRIWMLR', 'tfMRIWMRL',
                      'tfMRIEMOTIONLR', 'tfMRIEMOTIONRL',
                      'tfMRIGAMBLINGLR', 'tfMRIGAMBLINGRL',
                      'tfMRILANGUAGELR', 'tfMRILANGUAGERL',
                      'tfMRIRELATIONALLR', 'tfMRIRELATIONALRL',
                      'tfMRISOCIALLR', 'tfMRISOCIALRL'],
            'MRI7T': ['rfMRIREST17TPA', 'rfMRIREST27TAP', 'rfMRIREST37TPA', 'rfMRIREST47TAP',
                      'tfMRIMOVIE17TAP', 'tfMRIMOVIE27TPA', 'tfMRIMOVIE37TPA', 'tfMRIMOVIE47TAP'],
            'MEG':   ['3Restin', '4Restin', '5Restin', '6Wrkmem', '7Wrkmem'],
        },
        'subject_ids_file': 'subject_ids.txt',       # or 'subject_ids_subsample.txt'
        'detrend':             True,
        'nuisance_regression': True,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'HCP-D': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'fsLR',
        'sessions':    ['01'],
        'tasks': {
            'MRI3T': ['rfMRIREST1PA', 'rfMRIREST2PA',
                      'tfMRICARITPA', 'tfMRIEMOTIONPA', 'tfMRIGUESSINGPA'],
        },
        'subject_ids_file': 'subject_ids_meanFD.txt',
        'detrend':             True,
        'nuisance_regression': True,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'WAND': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    ['03'],
        'tasks': {
            'MRI3T': ['rest', 'reversallearning'],
        },
        'subject_ids_file': 'subject_ids_bids.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     False,
        'remove_incomplete_subjects': True,
    },
    'PNC': {
        'modalities':  ['MRI3T'],
        'seg':         '4S456Parcels',
        'space':       'fsLR',
        'sessions':    ['PNC1'],
        'tasks': {
            'MRI3T': ['rest', 'frac2back'],
        },
        'subject_ids_file': 'subject_ids_lowmotion.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'UCLA': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    [None],
        'tasks': {
            'MRI3T': ['rest', 'bart', 'stopsignal', 'scap'],
        },
        'subject_ids_file': 'subject_ids_not1_hc.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'NKI': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer2018p400n17',
        'space':       'MNI152NLin6ASym',
        'sessions':    ['BAS1'],
        'tasks': {
            'MRI3T': ['rest', 'CHECKERBOARD'],
        },
        'subject_ids_file': 'subject_ids.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'Penn_LEAD': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    ['1', '2'],  # add '3' for subjects with a third visit
        'tasks': {
            'MRI3T': ['rest', 'nback'],
        },
        'subject_ids_file': 'subject_ids.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'HBN': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer2018p400n17',
        'space':       'MNI152NLin6ASym',
        'sessions':    ['HBNsiteCBIC', 'HBNsiteRU', 'HBNsiteCUNY'],
        'tasks': {
            'MRI3T': ['rest'],
        },
        'subject_ids_file': 'subject_ids.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     False,
        'remove_incomplete_subjects': True,
    },
}


def get_dataset_params(dataset, seg=None):
    """Return the processing parameters for ``dataset``, optionally overriding ``seg``.

    Parameters
    ----------
    dataset : str
        Key into DATASET_PARAMS (e.g. 'HCP-YA', 'HCP-D').
    seg : str, optional
        Parcellation/atlas name (e.g. 'Schaefer4007', 'Schaefer10007'). If provided,
        it overrides the dataset-specific default; if None, the default defined in
        DATASET_PARAMS is used. This controls the ``seg-`` token written into the
        scan registry, and therefore which parcellation resolution the precomputed
        timescale outputs are loaded at.

    Returns
    -------
    dict
        A shallow copy of ``DATASET_PARAMS[dataset]`` with ``seg`` applied. The copy
        prevents callers from mutating the module-level defaults.
    """
    if dataset not in DATASET_PARAMS:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Available: {sorted(DATASET_PARAMS)}.")
    params = dict(DATASET_PARAMS[dataset])
    if seg is not None:
        params['seg'] = seg
    return params


# ==============================================================================
# Private helpers for load_notebook_data
# ==============================================================================

def _setup_processor(dataset):
    """Create a metadata-only Timescales processor and ensure its results dir exists.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    tuple
        (Timescales processor, results_dir path).
    """
    timescales_processor = Timescales(dataset=dataset, timeseries_data=None)
    results_dir = os.path.join(timescales_processor.output_dir, 'results', dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return timescales_processor, results_dir


def _load_timescale_outputs(processor, which_modality, use_7t, output_type, use_resids, filter_str):
    """Load precomputed timescale (and tSNR) outputs for a dataset into dicts.

    Loads the appropriate scan registry (multi/MEG variants for HCP-YA), then the
    tSNR and timescale outputs, filtering out subjects with missing scans. When
    use_resids is True, task residual timescales are loaded and substituted in.

    Parameters
    ----------
    processor : Timescales
        Processor whose registry/outputs are loaded.
    which_modality : {'FMRI', 'MEG'}
        Selects the HCP-YA registry variant.
    use_7t : bool
        Use the HCP-YA multi-session (incl. 7T) registry.
    output_type : str
        Timescale output type to load.
    use_resids : bool
        Also load residual timescales and merge them in.
    filter_str : str or None
        If given, overrides output_type to a filtered zero-crossing variant.

    Returns
    -------
    tuple
        (data_dict, data_dict_tsnr, data_dict_residuals); the last is None unless
        use_resids.
    """
    dataset = processor.dataset

    if dataset == 'HCP-YA' and which_modality == 'FMRI' and use_7t:
        processor.load_scan_registry(f'scan_registry_{dataset}_multi.csv')
    elif dataset == 'HCP-YA' and which_modality == 'MEG':
        processor.load_scan_registry(f'scan_registry_{dataset}_MEG.csv')
    else:
        processor.load_scan_registry(f'scan_registry_{dataset}.csv')

    if dataset == 'HCP-YA':
        processor.match_registry_to_behavior()

    processor.get_precomputed_outputs(output_type='metric_tsnr')
    data_dict_tsnr = filter_data_dict(processor.data_dict.copy(), remove_subject_level=True)

    if filter_str is not None:
        output_type = 'timescale_zero_crossing' + filter_str
    output_type_resid = output_type + '_resids'

    processor.get_precomputed_outputs(output_type=output_type)
    data_dict = processor.data_dict.copy()

    if use_resids:
        processor.get_precomputed_outputs(output_type=output_type_resid)
        data_dict_residuals = processor.data_dict.copy()
        data_dict, data_dict_residuals = filter_data_dict(data_dict, data_dict_residuals)
    else:
        data_dict = filter_data_dict(data_dict, remove_subject_level=True)
        data_dict_residuals = None

    return data_dict, data_dict_tsnr, data_dict_residuals


def _record_connectivity_subject(i, adjacency, voxel_counts_file, adjacency_matrices, strength, missing_label):
    """Volume-normalize one subject's adjacency and record it, or mark the subject missing.

    Shared tail for the HCP-YA and HCP-D connectivity loaders so both apply the same guard
    (adjacency present AND its region-sizes file present), the same volume normalization, and
    the same missing-subject handling: the adjacency slice and the strength column are both
    set to NaN. Mutates adjacency_matrices and strength in place at index i.
    """
    if adjacency is not None and os.path.exists(voxel_counts_file):
        voxel_counts = np.loadtxt(voxel_counts_file)
        adjacency_norm = volume_normalize_adjacency(adjacency, voxel_counts)
        strength[:, i] = np.log(adjacency_norm.sum(axis=0))
        adjacency_matrices[:, :, i] = adjacency
    else:
        adjacency_matrices[:, :, i] = np.nan
        strength[:, i] = np.nan
        print(f'\t{missing_label} not found')


def _load_hcpya_connectivity(derivdata_dir, subject_ids, n_regions):
    """Load per-subject HCP-YA structural connectomes and node strengths.

    Reads each subject's MRtrix streamline-count matrix, volume-normalizes it, and
    records the log node strength. Missing subjects become NaN columns/slices.

    Parameters
    ----------
    derivdata_dir : str or Path
        Derivatives directory.
    subject_ids : sequence of str
        Subjects to load, defining the output column order.
    n_regions : int
        Parcellation resolution (Schaefer{n_regions}7).

    Returns
    -------
    adjacency_matrices : ndarray
        (n_regions, n_regions, n_subjects) per-subject connectomes.
    adjacency_mean : ndarray
        (n_regions, n_regions) consistency-thresholded group mean.
    strength : ndarray
        (n_regions, n_subjects) log node strengths.
    """
    print('Loading HCP-YA connectivity data...')
    n_subs = len(subject_ids)
    adjacency_matrices = np.full((n_regions, n_regions, n_subs), np.nan)
    strength = np.full((n_regions, n_subs), np.nan)

    for i in np.arange(n_subs):
        sub = subject_ids[i]
        adjacency_file = os.path.join(derivdata_dir, 'mrtrix_10m', f'sub-{sub}',
                                      f'sub-{sub}_atlas-Schaefer{n_regions}7_space-T1w_stat-streamlinecount_relmat.csv')
        voxel_counts_file = os.path.join(derivdata_dir, 'parcellations', f'sub-{sub}',
                                         f'sub-{sub}_atlas-Schaefer{n_regions}7_space-T1w_res-1_regionsizes.txt')

        adjacency = np.loadtxt(adjacency_file, delimiter=',', dtype=float) if os.path.exists(adjacency_file) else None
        _record_connectivity_subject(i, adjacency, voxel_counts_file, adjacency_matrices, strength, f'sub-{sub}')

    adjacency_mean = threshold_adjacency_consistency(adjacency_matrices)
    return adjacency_matrices, adjacency_mean, strength


def _load_hcpd_connectivity(derivdata_dir, subject_ids, n_regions):
    """Load per-subject HCP-D structural connectomes and node strengths.

    Like _load_hcpya_connectivity, but averages the AP and PA phase-encoding
    acquisitions where both exist before volume-normalizing.

    Parameters
    ----------
    derivdata_dir : str or Path
        Derivatives directory.
    subject_ids : sequence of str
        Subjects to load, defining the output column order.
    n_regions : int
        Parcellation resolution (Schaefer{n_regions}7).

    Returns
    -------
    adjacency_matrices : ndarray
        (n_regions, n_regions, n_subjects) per-subject connectomes.
    adjacency_mean : ndarray
        (n_regions, n_regions) consistency-thresholded group mean.
    strength : ndarray
        (n_regions, n_subjects) log node strengths.
    """
    print('Loading HCP-D connectivity data...')
    n_subs = len(subject_ids)
    adjacency_matrices = np.full((n_regions, n_regions, n_subs), np.nan)
    strength = np.full((n_regions, n_subs), np.nan)

    voxel_counts_file_tpl = os.path.join(
        derivdata_dir, 'parcellations', 'sub-{sub}', 'ses-01',
        f'sub-{{sub}}_ses-01_atlas-Schaefer{n_regions}7_space-T1w_res-1_regionsizes.txt'
    )

    for i in np.arange(n_subs):
        sub = subject_ids[i]

        # Average AP and PA phase-encoding acquisitions where both exist
        adj_sum = np.zeros((n_regions, n_regions))
        n_found = 0
        for ses_dir in ['ses-01AP', 'ses-01PA']:
            adjacency_file = os.path.join(
                derivdata_dir, 'mrtrix_10m', f'sub-{sub}', ses_dir,
                f'sub-{sub}_ses-01_atlas-Schaefer{n_regions}7_stat-streamlinecount_relmat.csv'
            )
            if os.path.exists(adjacency_file):
                adj_sum += np.loadtxt(adjacency_file, delimiter=',', dtype=float)
                n_found += 1

        adjacency = adj_sum / n_found if n_found > 0 else None
        voxel_counts_file = voxel_counts_file_tpl.format(sub=sub)
        _record_connectivity_subject(i, adjacency, voxel_counts_file, adjacency_matrices, strength, f'sub-{sub}')

    adjacency_mean = threshold_adjacency_consistency(adjacency_matrices)
    return adjacency_matrices, adjacency_mean, strength


def _load_myelin(derivdata_dir, subject_ids, n_regions, label='myelin'):
    """Per-subject Schaefer 7-network T1w/T2w myelin, aligned to subject_ids.

    Loads one parcellated pscalar per subject from the parcellated_data layout, which is
    identical for HCP-YA and HCP-D (only the subject-ID format differs, and that is carried
    in subject_ids). Each map is squeezed to (n_regions,) on load. Subjects without an anat
    file get an all-NaN column. The parcellation resolution is selected by n_regions
    (seg-Schaefer{n_regions}7).

    Returns
    -------
    myelin : ndarray, shape (n_regions, n_subjects)
        Per-subject myelin, column i aligned to subject_ids[i].
    myelin_mean : ndarray, shape (n_regions,)
        Per-region mean over subjects (NaN-safe).
    """
    print(f'Loading {label} data...')
    n_subs = len(subject_ids)
    myelin = np.full((n_regions, n_subs), np.nan)

    for i in np.arange(n_subs):
        myelin_file = os.path.join(derivdata_dir, 'parcellated_data', f'sub-{subject_ids[i]}', 'ses-01', 'anat',
                                   f'sub-{subject_ids[i]}_ses-01_space-fsLR_seg-Schaefer{n_regions}7_stat-mean_myelin.pscalar.nii')
        if os.path.exists(myelin_file):
            img = nib.load(str(myelin_file))
            myelin[:, i] = np.asarray(img.get_fdata()).squeeze()
        else:
            print(f'\tsub-{subject_ids[i]} not found')

    return myelin, np.nanmean(myelin, axis=1)


def _load_hcpya_myelin(derivdata_dir, subject_ids, n_regions):
    """Load HCP-YA per-subject T1w/T2w myelin (see _load_myelin)."""
    return _load_myelin(derivdata_dir, subject_ids, n_regions, label='HCP-YA myelin')


def _load_hcpd_myelin(derivdata_dir, subject_ids, n_regions):
    """Load HCP-D per-subject T1w/T2w myelin (see _load_myelin)."""
    return _load_myelin(derivdata_dir, subject_ids, n_regions, label='HCP-D myelin')


def _sync_subject_level_data(adjacency_matrices, strength, myelin, subject_ids):
    """Restrict the per-subject arrays to subjects present in BOTH connectivity and myelin.

    A subject is present in a modality when its slice has any finite value (the loaders fill
    missing subjects with NaN). The three per-subject arrays and subject_ids are filtered to
    the intersection along the subject axis so a single subject_ids vector indexes all of
    them. Group-level estimates (adjacency_mean, df_neuro region maps) are computed upstream
    over each modality's full sample and are intentionally left untouched.

    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_regions, n_regions, n_subjects)
    strength : ndarray, shape (n_regions, n_subjects)
    myelin : ndarray, shape (n_regions, n_subjects)
    subject_ids : sequence of str, length n_subjects

    Returns
    -------
    adjacency_matrices, strength, myelin : ndarray
        Filtered along the subject axis to the connectivity∩myelin set.
    subject_ids : ndarray
        Subject IDs aligned to the columns/slices of the returned arrays.
    """
    subject_ids = np.asarray(subject_ids)

    has_conn = np.isfinite(adjacency_matrices).any(axis=(0, 1))
    has_myelin = np.isfinite(myelin).any(axis=0)
    keep = has_conn & has_myelin

    n_dropped = int((~keep).sum())
    if n_dropped:
        conn_only = subject_ids[has_conn & ~has_myelin]
        myelin_only = subject_ids[has_myelin & ~has_conn]
        neither = subject_ids[~has_conn & ~has_myelin]
        print(f'\tSyncing connectivity and myelin: keeping {int(keep.sum())}/{len(keep)} '
              f'subjects ({n_dropped} dropped).')
        if len(conn_only):
            print(f'\t\tconnectivity but no myelin ({len(conn_only)}): {list(conn_only)}')
        if len(myelin_only):
            print(f'\t\tmyelin but no connectivity ({len(myelin_only)}): {list(myelin_only)}')
        if len(neither):
            print(f'\t\tmissing both ({len(neither)}): {list(neither)}')

    return adjacency_matrices[:, :, keep], strength[:, keep], myelin[:, keep], subject_ids[keep]


# ==============================================================================
# Private helpers for load_supporting_data
# ==============================================================================

def _load_axes_and_parcellation(in_dir, n_regions):
    """Load the S-A / S-F / micro axes and Schaefer centroids (the region-space core).

    These define the region space, so any file missing at the requested resolution
    is fatal. Axes are returned as rank values (the S-F axis is the negated cyto
    map, ranked). Also builds the Euclidean centroid distance matrix and Yeo systems.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution (7-network Schaefer).

    Returns
    -------
    tuple
        (parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems,
        n_systems, sa_axis, sf_axis, micro_axis).

    Raises
    ------
    FileNotFoundError
        If any core axis/centroid file is missing.
    ValueError
        If the centroid file's region count does not match n_regions.
    """
    # Core resources: the SA/SF/micro axes and the parcellation centroids define the region
    # space itself, so a file missing at the requested resolution is fatal (cannot be skipped).
    sa_file = os.path.join(in_dir, f'schaefer{n_regions}-7_sa-axis.npy')
    cyto_file = os.path.join(in_dir, f'schaefer{n_regions}-7_cyto.npy')
    micro_file = os.path.join(in_dir, f'schaefer{n_regions}-7_micro.npy')
    parc_file = os.path.join(in_dir, f'Schaefer2018_{n_regions}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')

    missing = [p for p in (sa_file, cyto_file, micro_file, parc_file) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Core parcellation/axis files missing for n_regions={n_regions} (7-network Schaefer). "
            f"These define the region space and cannot be skipped:\n  " + "\n  ".join(missing) +
            f"\nCheck {in_dir} for available resolutions (e.g. 'schaefer<N>-7_sa-axis.npy' and "
            f"'Schaefer2018_<N>Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')."
        )

    sa_axis = sp.stats.rankdata(np.load(sa_file))
    sf_axis = sp.stats.rankdata(np.load(cyto_file) * -1)
    micro_axis = sp.stats.rankdata(np.load(micro_file))

    parc = pd.read_csv(parc_file, header=0, names=['ROI_Label', 'ROI_Name', 'R', 'A', 'S'], index_col=0)
    if parc.shape[0] != n_regions:
        raise ValueError(
            f"Parcellation file {parc_file} has {parc.shape[0]} regions but n_regions={n_regions} was requested."
        )

    distance_matrix = distance.squareform(distance.pdist(parc[['R', 'A', 'S']], 'euclidean'))
    distance_matrix_norm = normalize_x(distance_matrix)

    # ROI names look like '7Networks_LH_Vis_1'; the Yeo system is the token after the
    # hemisphere prefix. .str.strip() tolerates the space-padding present in some centroid
    # files (e.g. the 400-region file) but absent in others.
    yeo_systems = parc['ROI_Name'].str.strip()
    yeo_systems = yeo_systems.str.replace('7Networks_LH_', '', regex=False)
    yeo_systems = yeo_systems.str.replace('7Networks_RH_', '', regex=False)
    yeo_systems = [s.split('_')[0] for s in yeo_systems]
    n_systems = len(np.unique(yeo_systems))

    return parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems, n_systems, sa_axis, sf_axis, micro_axis


def _load_gene_expression(in_dir, n_regions, n_components=2):
    """Load AHBA gene-expression maps for a curated panel plus PCA components.

    Optional annotation family: warns and returns (None, None, None) if the
    expression file is absent at this resolution.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.
    n_components : int
        Number of PCA components to add.

    Returns
    -------
    df_gene : pd.DataFrame or None
        Per-gene columns plus PCA_Gene-PC{i} columns.
    gene_expression_nanmask : ndarray or None
        Boolean mask of regions with any non-NaN expression.
    pca_results_genes : dict or None
        Output of pca_with_nan_handling on the brain-gene panel.
    """
    # Optional annotation map: warn and skip if unavailable at this resolution rather than failing.
    gene_file = os.path.join(in_dir, f'expression_schaefer{n_regions}-7.csv')
    if not os.path.exists(gene_file):
        print(f"\tWARNING: gene expression not available at n_regions={n_regions} "
              f"({os.path.basename(gene_file)}); skipping gene maps.")
        return None, None, None

    gene_expression = pd.read_csv(gene_file, index_col=0)
    gao_brain_genes = np.loadtxt(os.path.join(in_dir, 'gao_brain_genes.txt'), dtype=str)
    existing_columns = [col for col in gao_brain_genes if col in set(gene_expression.columns)]
    gene_expression_brain = gene_expression[existing_columns]
    gene_expression_nanmask = np.isnan(gene_expression.values).sum(axis=1) != gene_expression.shape[1]

    # ---- Existing curated panel (cell-type / channel / receptor; tested for stage 1) ----
    genes_original = ['GRIN2B', 'GRIN2A', 'GABRA3', 'GABRA1', 'PVALB', 'PRR5', 'KCNA3', 'SCN1A']

    # ---- New panel 1: recurrent / glutamatergic synaptic strength (lead hypothesis) ----
    genes_recurrent_excitation = [
        'GRIN1', 'GRIN2A', 'GRIN2B',            # NMDA receptor subunits
        'GRIA1', 'GRIA2', 'GRIA3', 'GRIA4',     # AMPA receptor subunits
        'GRM5',                                 # metabotropic glutamate receptor 5
        'CAMK2A', 'CAMK2B',                     # postsynaptic Ca2+/calmodulin kinase
        'DLG4',                                 # PSD-95
        'SYNGAP1', 'SHANK3', 'HOMER1',          # postsynaptic scaffolding / signaling
    ]

    # ---- New panel 2: presynaptic / calcium-regulated exocytosis (Xia/Dong-implicated axis) ----
    genes_presynaptic_exocytosis = [
        'SYT1',                                 # synaptotagmin-1 (Ca2+ sensor)
        'STX1A', 'STXBP1', 'SNAP25', 'VAMP2',   # SNARE complex
        'RIMS1',                                # active-zone release machinery
        'CACNA1A', 'CACNA1B',                   # presynaptic Ca2+ channels (P/Q, N-type)
        'SV2A',                                 # synaptic vesicle glycoprotein 2A
        'SYN1', 'SYP',                          # synapsin-1, synaptophysin
        'CPLX1', 'CPLX2',                       # complexins
    ]

    # ---- Combined (deduplicated, order preserved) ----
    gene_list = list(dict.fromkeys(
        genes_original + genes_recurrent_excitation + genes_presynaptic_exocytosis
    ))

    df_gene = pd.DataFrame(index=np.arange(n_regions))
    for gene in gene_list:
        df_gene[f'Gene_{gene}'] = gene_expression[gene].values

    pca_results_genes = pca_with_nan_handling(gene_expression_brain.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_gene[f'PCA_Gene-PC{i+1}'] = pca_results_genes['scores'][:, i]

    return df_gene, gene_expression_nanmask, pca_results_genes


def _load_pet_maps(in_dir, n_regions, n_components=2):
    """Load Hansen PET receptor maps plus PCA components.

    Optional annotation family: warns and returns (None, None, None) if the
    receptor file is absent at this resolution.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.
    n_components : int
        Number of PCA components to add.

    Returns
    -------
    df_neuro_pet : pd.DataFrame or None
        PET_ columns plus PCA_PET-PC{i} columns.
    pet_categories : pd.DataFrame or None
        Receptor-to-category table.
    pca_results_pet : dict or None
        Output of pca_with_nan_handling on the PET maps.
    """
    # Optional annotation map: warn and skip if unavailable at this resolution rather than failing.
    pet_file = os.path.join(in_dir, 'hansen_receptors', 'results', f'receptor_data_scale{n_regions:03d}.csv')
    if not os.path.exists(pet_file):
        print(f"\tWARNING: PET receptor maps not available at n_regions={n_regions} "
              f"({os.path.basename(pet_file)}); skipping PET maps.")
        return None, None, None

    df_neuro_pet = pd.read_csv(pet_file, index_col=None, header=None)
    pet_names = np.load(os.path.join(in_dir, 'hansen_receptors', 'data', 'receptor_names_pet.npy'))
    pet_names = ['PET_' + s for s in pet_names]
    pet_categories = pd.read_csv(os.path.join(in_dir, 'pet_categories_custom.csv'), index_col=0, header=0)
    df_neuro_pet.columns = pet_names

    pca_results_pet = pca_with_nan_handling(df_neuro_pet.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_neuro_pet[f'PCA_PET-PC{i+1}'] = pca_results_pet['scores'][:, i]

    return df_neuro_pet, pet_categories, pca_results_pet


def _load_cell_type_maps(in_dir, n_regions, n_components=2):
    """Load Jorstad cell-type expression maps plus PCA components.

    Optional annotation family: warns and returns (None, None) if the cell-type
    file is absent at this resolution.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.
    n_components : int
        Number of PCA components to add.

    Returns
    -------
    df_neuro_cell : pd.DataFrame or None
        Cell_ columns plus PCA_Cell-PC{i} columns.
    pca_results_cell : dict or None
        Output of pca_with_nan_handling on the cell-type maps.
    """
    # Optional annotation map: warn and skip if unavailable at this resolution rather than failing.
    cell_file = os.path.join(in_dir, 'human-cellular-func-con', 'cell_maps', 'Schaefer_atalas', 'Jorstad',
                             f'schaeffer_Jorstad_{n_regions}_7Net_expr_mat_new_NormZscore0.3.csv')
    if not os.path.exists(cell_file):
        print(f"\tWARNING: cell-type maps not available at n_regions={n_regions} "
              f"({os.path.basename(cell_file)}); skipping cell maps.")
        return None, None

    df_neuro_cell = pd.read_csv(cell_file, header=0, index_col=-1).transpose()
    df_neuro_cell = df_neuro_cell.reset_index(drop=True)
    cell_names = ['Cell_' + s.replace(' ', '').replace('/', '') for s in list(df_neuro_cell.columns)]
    df_neuro_cell.columns = cell_names

    pca_results_cell = pca_with_nan_handling(df_neuro_cell.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_neuro_cell[f'PCA_Cell-PC{i+1}'] = pca_results_cell['scores'][:, i]

    return df_neuro_cell, pca_results_cell


# Baum et al. precomputed T1w/T2w myelin effect maps. Each metric maps to a pscalar filename
# template (parameterized by parcellation) and a human-readable label. The dict keys are the
# canonical metric names used everywhere downstream (df_baum columns, BrainSMASH null-cache
# filenames), so they must match the null files precomputed on disk (e.g. 'baum-aroc_...').
BAUM_MYELIN_MAPS = {
    'baum-aroc':   ('Figure4_mean_posterior_annualized_roc_T1wT2w_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
                    'Myelin AROC (Baum)'),
    'baum-aoms':   ('Figure6_median_posterior_age_of_max_slope_T1wT2w_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
                    'Myelin AOMS (Baum)'),
    'baum-deriv2': ('Figure5_mean_posterior_deriv2_T1wT2w_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
                    'Myelin Deriv2 (Baum)'),
    # 'baum-young':  ('Figure2_mean_young_T1wT2w_8to10_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
    #                 'Myelin Young (Baum)'),
    # 'baum-mid':    ('Figure2_mean_mid_T1wT2w_14to16_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
    #                 'Myelin Mid (Baum)'),
    # 'baum-old':    ('Figure2_mean_old_T1wT2w_19to21_space-fsLR_seg-Schaefer{seg}.pscalar.nii',
    #                 'Myelin Old (Baum)'),
}


def _load_baum_myelin_maps(in_dir, n_regions):
    """Load the Baum et al. precomputed T1w/T2w myelin effect maps for a given parcellation.

    Optional annotation family: unlike gene/PET/cell maps (Schaefer-400 only), the Baum maps are
    provided at several Schaefer resolutions (100/200/400/800/1000, all 7-network). The seg suffix
    is 'Schaefer{n_regions}7' (e.g. n_regions=400 -> 'Schaefer4007'). Warn and skip (return None)
    if the baum_data directory is absent or no map exists at this resolution; skip individually any
    metric whose file is missing.

    Returns
    -------
    df_baum : pd.DataFrame (n_regions x n_available_maps) or None
        Columns are the canonical metric names (BAUM_MYELIN_MAPS keys, e.g. 'baum-aroc'); these
        must match the BrainSMASH null-cache filenames on disk. None if no maps were loaded.
    baum_labels : dict or None
        Metric name -> human-readable label, restricted to the maps actually loaded.
    """
    baum_dir = os.path.join(in_dir, 'baum_data')
    if not os.path.isdir(baum_dir):
        print(f"\tWARNING: Baum myelin maps directory not found ({baum_dir}); skipping Baum maps.")
        return None, None

    seg = f'{n_regions}7'
    df_baum = pd.DataFrame(index=np.arange(n_regions))
    baum_labels = {}
    for metric, (template, label) in BAUM_MYELIN_MAPS.items():
        map_file = os.path.join(baum_dir, template.format(seg=seg))
        if not os.path.exists(map_file):
            print(f"\tWARNING: Baum map '{metric}' not available at n_regions={n_regions} "
                  f"({os.path.basename(map_file)}); skipping this map.")
            continue
        data = nib.load(map_file).get_fdata()[0]
        if data.shape[0] != n_regions:
            raise ValueError(
                f"Baum map {map_file} has {data.shape[0]} regions but n_regions={n_regions} "
                f"was requested.")
        df_baum[metric] = data
        baum_labels[metric] = label

    if df_baum.shape[1] == 0:
        print(f"\tWARNING: no Baum myelin maps found at n_regions={n_regions}; skipping Baum maps.")
        return None, None

    print(f'Baum myelin maps: {df_baum.shape[1]} maps over {df_baum.shape[0]} parcels '
          f'({", ".join(df_baum.columns)}).')
    return df_baum, baum_labels


# ==============================================================================
# Public API
# ==============================================================================

def load_notebook_data(dataset='HCP-YA', which_modality='FMRI',
                       use_resids=True, use_7t=False,
                       output_type='timescale_zero_crossing_robust_nc3', filter_str=None, # '_lp-0.5hz'
                       load_subject_data=True, n_regions=400):
    """Load all data a notebook needs for a dataset into a single dict.

    Sets up the Timescales processor, loads supporting region maps (axes, gene /
    PET / cell / Baum annotations), the timescale and tSNR data dicts, and — for
    HCP-YA and HCP-D with load_subject_data — per-subject structural connectivity
    and myelin (synced to the connectivity-and-myelin subject set, with group maps
    added to supporting_data['df_neuro']).

    Parameters
    ----------
    dataset : str
        Dataset name (key into DATASET_PARAMS).
    which_modality : {'FMRI', 'MEG'}
        Selects the HCP-YA registry variant.
    use_resids : bool
        Load residual (task-regressed) timescales.
    use_7t : bool
        Use the HCP-YA multi-session (incl. 7T) registry.
    output_type : str
        Timescale output type to load.
    filter_str : str or None
        Optional frequency-filter tag (see _load_timescale_outputs).
    load_subject_data : bool
        Load per-subject connectivity/myelin for HCP-YA and HCP-D.
    n_regions : int
        Parcellation resolution.

    Returns
    -------
    dict
        Always includes timescales_processor, results_dir, sydnor_sa_cmap,
        data_dict, data_dict_tsnr, supporting_data; for HCP-YA/HCP-D with
        load_subject_data, also adjacency_matrices, adjacency_mean, strength,
        myelin, subject_ids.
    """

    print('Loading data for notebooks...')

    timescales_processor, results_dir = _setup_processor(dataset)
    derivdata_dir = timescales_processor.derivdata_dir

    sydnor_sa_cmap = create_sydnor_sa_colormap()
    supporting_data = load_supporting_data(
        os.path.join(timescales_processor.output_dir, 'data', 'supporting_data'), n_regions=n_regions)
    gene_expression_nanmask = supporting_data['gene_expression_nanmask']

    data_dict, data_dict_tsnr, _ = _load_timescale_outputs(
        timescales_processor, which_modality, use_7t, output_type, use_resids, filter_str)

    subject_ids = list(dict.fromkeys(scan.subject_id for scan in timescales_processor.scan_registry))
    n_subs = len(subject_ids)
    print(f'Number of subjects: {n_subs}')

    notebook_data = {
        'timescales_processor': timescales_processor,
        'results_dir': results_dir,
        'sydnor_sa_cmap': sydnor_sa_cmap,
        'data_dict': data_dict,
        'data_dict_tsnr': data_dict_tsnr,
        'supporting_data': supporting_data,
    }

    if dataset == 'HCP-YA' and load_subject_data:
        adjacency_matrices, adjacency_mean, strength = _load_hcpya_connectivity(derivdata_dir, subject_ids, n_regions)
        myelin, myelin_mean = _load_hcpya_myelin(derivdata_dir, subject_ids, n_regions)

        # Group-level region maps use each modality's full available sample. These are
        # region-indexed, so they are independent of the per-subject sync below.
        supporting_data['df_neuro']['HCPYA_Strength'] = np.log(adjacency_mean.sum(axis=0))
        supporting_data['df_neuro']['HCPYA_Degree'] = np.sum(adjacency_mean > 0, axis=0)
        supporting_data['df_neuro']['HCPYA_Myelin'] = myelin_mean

        # Orient gene PC1 to correlate positively with myelin. Skipped if gene maps were
        # unavailable at this resolution (gene_expression_nanmask is None).
        if gene_expression_nanmask is not None and 'PCA_Gene-PC1' in supporting_data['df_neuro'].columns:
            myelin_gene_corr = sp.stats.spearmanr(
                supporting_data['df_neuro'].loc[gene_expression_nanmask, 'PCA_Gene-PC1'],
                supporting_data['df_neuro'].loc[gene_expression_nanmask, 'HCPYA_Myelin'])[0]
            if myelin_gene_corr < 0:
                supporting_data['df_neuro'].loc[:, 'PCA_Gene-PC1'] *= -1

        # Sync per-subject arrays to the connectivity∩myelin subject set so a single
        # subject_ids vector indexes adjacency_matrices, strength, and myelin alike.
        adjacency_matrices, strength, myelin, subject_ids = _sync_subject_level_data(
            adjacency_matrices, strength, myelin, subject_ids)
        notebook_data.update({
            'adjacency_matrices': adjacency_matrices,
            'adjacency_mean': adjacency_mean,
            'strength': strength,
            'myelin': myelin,
            'subject_ids': subject_ids,
        })

    elif dataset == 'HCP-D' and load_subject_data:
        adjacency_matrices, adjacency_mean, strength = _load_hcpd_connectivity(derivdata_dir, subject_ids, n_regions)
        # Per-subject T1w/T2w myelin, aligned to the registry subject order (HCD...V1MR).
        myelin, myelin_mean = _load_hcpd_myelin(derivdata_dir, subject_ids, n_regions)

        # Group-level region maps use each modality's full available sample. These are
        # region-indexed, so they are independent of the per-subject sync below.
        supporting_data['df_neuro']['HCPD_Strength'] = np.log(adjacency_mean.sum(axis=0))
        supporting_data['df_neuro']['HCPD_Degree'] = np.sum(adjacency_mean > 0, axis=0)
        supporting_data['df_neuro']['HCPD_Myelin'] = myelin_mean

        # Sync per-subject arrays to the connectivity∩myelin subject set so a single
        # subject_ids vector indexes adjacency_matrices, strength, and myelin alike. The
        # filtered subject_ids are exposed so downstream code can intersect with df_behavior.
        adjacency_matrices, strength, myelin, subject_ids = _sync_subject_level_data(
            adjacency_matrices, strength, myelin, subject_ids)
        notebook_data.update({
            'adjacency_matrices': adjacency_matrices,
            'adjacency_mean': adjacency_mean,
            'strength': strength,
            'myelin': myelin,
            'subject_ids': subject_ids,
        })

    return notebook_data


def data_dict_to_dataframe(data_dict_centiles):
    """
    Reshape a centile data dict into a long-format DataFrame for plotting.

    Parameters
    ----------
    data_dict_centiles : dict
        Keys are condition names, values are (n_subjects, n_deciles) arrays.

    Returns
    -------
    pd.DataFrame with a 'subject' column and one column per condition.
    """
    df_plot = pd.DataFrame()
    for c, condition in enumerate(data_dict_centiles.keys()):
        df_tmp_1 = pd.DataFrame()
        n_subs_tmp = data_dict_centiles[condition].shape[0]
        n_dec_tmp = data_dict_centiles[condition].shape[1]
        for i in np.arange(n_subs_tmp):
            df_tmp_2 = pd.DataFrame()
            df_tmp_2['subject'] = np.zeros(n_dec_tmp) + i
            df_tmp_2[condition] = data_dict_centiles[condition][i, :]
            df_tmp_1 = pd.concat((df_tmp_1, df_tmp_2), axis=0)
        if c == 0:
            df_plot['subject'] = df_tmp_1['subject']
        df_plot[condition] = df_tmp_1[condition]
    return df_plot


def average_pet_markers(df_neuro, pet_categories, pet_category='neurotransmitter_function'):
    """Average PET receptor maps within each functional category.

    Parameters
    ----------
    df_neuro : pd.DataFrame
        Regions x PET features (z-scored).
    pet_categories : pd.DataFrame
        Indexed by PET feature name; must contain the pet_category column.
    pet_category : str
        Column in pet_categories to group by.

    Returns
    -------
    pd.DataFrame
        One column per unique category, named 'Mean_<category>'.
    """
    unique_markers = pet_categories[pet_category].unique()
    df_out = pd.DataFrame()
    for marker in unique_markers:
        mask = pet_categories[pet_category] == marker
        df_out[f'Mean_{marker}'] = np.nanmean(
            df_neuro.loc[:, pet_categories.index].iloc[:, mask.values], axis=1)
    return df_out


def load_supporting_data(in_dir, n_regions=400):
    """Load all region-level supporting maps into a single dict.

    Combines the region-space core (axes, centroids, distance matrix, Yeo systems)
    with the optional annotation families (gene, PET, cell-type, Baum myelin), each
    of which is None when unavailable at the requested resolution. Gene / PET / cell
    maps are concatenated into df_neuro alongside the S-A and S-F axes.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.

    Returns
    -------
    dict
        parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems,
        n_systems, gene_expression_nanmask, pet_categories, df_neuro, df_baum,
        baum_labels, and the three pca_results_* entries.
    """
    parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems, n_systems, sa_axis, sf_axis, micro_axis = \
        _load_axes_and_parcellation(in_dir, n_regions)

    df_gene, gene_expression_nanmask, pca_results_genes = _load_gene_expression(in_dir, n_regions)
    df_neuro_pet, pet_categories, pca_results_pet = _load_pet_maps(in_dir, n_regions)
    df_neuro_cell, pca_results_cell = _load_cell_type_maps(in_dir, n_regions)
    df_baum, baum_labels = _load_baum_myelin_maps(in_dir, n_regions)

    df_neuro = pd.DataFrame(index=np.arange(n_regions))
    df_neuro['SA_Axis'] = sa_axis
    df_neuro['SF_Axis'] = sf_axis
    # Optional annotation frames are None when unavailable at this resolution (see _load_* warnings).
    df_neuro = pd.concat([df_neuro] + [f for f in (df_gene, df_neuro_pet, df_neuro_cell) if f is not None], axis=1)

    return {
        'parc': parc,
        'n_regions': n_regions,
        'distance_matrix': distance_matrix,
        'distance_matrix_norm': distance_matrix_norm,
        'yeo_systems': yeo_systems,
        'n_systems': n_systems,
        'gene_expression_nanmask': gene_expression_nanmask,
        'pet_categories': pet_categories,
        'df_neuro': df_neuro,
        'df_baum': df_baum,
        'baum_labels': baum_labels,
        'pca_results_genes': pca_results_genes,
        'pca_results_pet': pca_results_pet,
        'pca_results_cell': pca_results_cell,
    }
