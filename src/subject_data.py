"""Per-subject structural data: streamline connectivity and T1w/T2w myelin.

Loaded per subject and synced to a common subject set, unlike the region-level annotation maps in
:mod:`src.supporting_data`, which are one vector per parcel and shared by every subject.
"""

import os

import numpy as np
import nibabel as nib

from snaplab_tools.topology import threshold_adjacency_consistency, volume_normalize_adjacency


def region_sizes_file(dataset, derivdata_dir, sub, n_regions):
    """Path of one subject's T1w-space parcel voxel counts, which volume normalisation divides by.

    The one definition of this path: the connectivity loaders below and :mod:`src.nct` both read it.
    """
    if dataset == 'HCP-YA':
        return os.path.join(derivdata_dir, 'parcellations', f'sub-{sub}',
                            f'sub-{sub}_atlas-Schaefer{n_regions}7_space-T1w_res-1_regionsizes.txt')
    if dataset == 'HCP-D':
        return os.path.join(derivdata_dir, 'parcellations', f'sub-{sub}', 'ses-01',
                            f'sub-{sub}_ses-01_atlas-Schaefer{n_regions}7_space-T1w_res-1_regionsizes.txt')
    raise ValueError(f'no region-size files for {dataset}; only HCP-YA and HCP-D have connectomes')


def load_region_sizes(dataset, derivdata_dir, subject_ids, n_regions):
    """Per-subject parcel voxel counts, ``(n_regions, n_subjects)`` in ``subject_ids`` order.

    A subject without a region-size file gets an all-NaN column, as the connectivity loaders treat it.
    """
    sizes = np.full((n_regions, len(subject_ids)), np.nan)
    for i, sub in enumerate(subject_ids):
        path = region_sizes_file(dataset, derivdata_dir, sub, n_regions)
        if os.path.exists(path):
            sizes[:, i] = np.loadtxt(path)
    return sizes


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
        voxel_counts_file = region_sizes_file('HCP-YA', derivdata_dir, sub, n_regions)

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
        voxel_counts_file = region_sizes_file('HCP-D', derivdata_dir, sub, n_regions)
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

