"""Region-level supporting maps: the region-space core plus optional annotation families.

Everything here is indexed by parcel and shared across subjects -- the S-A / S-F axes, Schaefer
geometry, gene expression and cell-type maps. Per-subject structural data lives in
:mod:`src.subject_data`.
"""

import os

import numpy as np
import scipy as sp
import pandas as pd
from scipy.spatial import distance

from snaplab_tools.utils import normalize_x


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


def gene_panel_file(in_dir, gene_panel):
    """Path to a brain-gene list, validated against what is actually on disk.

    A panel tag resolves to ``<in_dir>/<gene_panel>_brain_genes.txt``, so adding a new list needs
    no code change -- drop the file in and pass its prefix. An unknown tag raises with the
    available panels listed, rather than silently falling back to a default.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    gene_panel : str
        Panel tag, e.g. 'gao' or 'dear'.

    Returns
    -------
    str
        Absolute path to the panel file.
    """
    path = os.path.join(in_dir, f'{gene_panel}_brain_genes.txt')
    if not os.path.exists(path):
        suffix = '_brain_genes.txt'
        available = sorted(f[:-len(suffix)] for f in os.listdir(in_dir) if f.endswith(suffix))
        raise FileNotFoundError(
            f"gene_panel={gene_panel!r} -> {os.path.basename(path)} not found in {in_dir}. "
            f"Available panels: {available}")
    return path


def _load_gene_expression(in_dir, n_regions, gene_panel='gao'):
    """Load AHBA gene-expression maps: a curated panel plus the full brain-gene panel.

    No PCA is run here -- the brain-gene panel is returned as-is so the analysis notebook that
    needs components can fit them itself (and re-fit them, e.g. to bootstrap the loadings).

    Optional annotation family: warns and returns (None, None, None) if the
    expression file is absent at this resolution.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.
    gene_panel : str
        Which brain-gene list defines ``gene_expression_brain`` -- the matrix a gene PCA is fit on.
        Resolves via :func:`gene_panel_file`; 'gao' (~2.4k brain-expressed genes) and 'dear'
        (~8k) ship with the project. NOTE this does NOT affect ``df_gene``: the curated Gene_*
        maps below are a fixed hypothesis-driven list, identical under every panel.

    Returns
    -------
    df_gene : pd.DataFrame or None
        Gene_{symbol} columns for the curated panel.
    gene_expression_nanmask : ndarray or None
        Boolean mask of regions with any non-NaN expression.
    gene_expression_brain : pd.DataFrame or None
        (n_regions x n_genes) expression over the selected brain-gene list -- the matrix a gene PCA
        should be fit on.
    """
    # Validate the panel tag first, so a typo fails loudly at every resolution rather than only
    # where the expression matrix happens to exist.
    panel_file = gene_panel_file(in_dir, gene_panel)

    # Optional annotation map: warn and skip if unavailable at this resolution rather than failing.
    gene_file = os.path.join(in_dir, f'expression_schaefer{n_regions}-7.csv')
    if not os.path.exists(gene_file):
        print(f"\tWARNING: gene expression not available at n_regions={n_regions} "
              f"({os.path.basename(gene_file)}); skipping gene maps.")
        return None, None, None

    gene_expression = pd.read_csv(gene_file, index_col=0)
    brain_genes = np.loadtxt(panel_file, dtype=str)
    existing_columns = [col for col in brain_genes if col in set(gene_expression.columns)]
    gene_expression_brain = gene_expression[existing_columns]
    gene_expression_nanmask = np.isnan(gene_expression.values).sum(axis=1) != gene_expression.shape[1]
    print(f"\tBrain-gene panel '{gene_panel}': {len(existing_columns)}/{len(brain_genes)} symbols "
          f'present in the expression matrix -> PCA input {gene_expression_brain.shape}.')

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

    return df_gene, gene_expression_nanmask, gene_expression_brain


def _load_cell_type_maps(in_dir, n_regions):
    """Load Jorstad cell-type expression maps.

    No PCA is run here -- see :func:`_load_gene_expression`.

    Optional annotation family: warns and returns None if the cell-type
    file is absent at this resolution.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.

    Returns
    -------
    df_neuro_cell : pd.DataFrame or None
        Cell_{type} columns -- also the matrix a cell-type PCA should be fit on.
    """
    # Optional annotation map: warn and skip if unavailable at this resolution rather than failing.
    cell_file = os.path.join(in_dir, 'human-cellular-func-con', 'cell_maps', 'Schaefer_atalas', 'Jorstad',
                             f'schaeffer_Jorstad_{n_regions}_7Net_expr_mat_new_NormZscore0.3.csv')
    if not os.path.exists(cell_file):
        print(f"\tWARNING: cell-type maps not available at n_regions={n_regions} "
              f"({os.path.basename(cell_file)}); skipping cell maps.")
        return None

    df_neuro_cell = pd.read_csv(cell_file, header=0, index_col=-1).transpose()
    df_neuro_cell = df_neuro_cell.reset_index(drop=True)
    cell_names = ['Cell_' + s.replace(' ', '').replace('/', '') for s in list(df_neuro_cell.columns)]
    df_neuro_cell.columns = cell_names

    return df_neuro_cell


def load_supporting_data(in_dir, n_regions=400, gene_panel='gao'):
    """Load all region-level supporting maps into a single dict.

    Combines the region-space core (axes, centroids, distance matrix, Yeo systems)
    with the optional annotation families (gene, cell-type), each of which is None
    when unavailable at the requested resolution. Gene / cell maps are concatenated
    into df_neuro alongside the S-A and S-F axes.

    Parameters
    ----------
    in_dir : str
        Supporting-data directory.
    n_regions : int
        Parcellation resolution.
    gene_panel : str
        Brain-gene list backing gene_expression_brain ('gao', 'dear', ...). Echoed back as
        'gene_panel' so callers can tag outputs that depend on it -- anything derived from the
        gene PCA. df_neuro's curated Gene_* maps do not depend on it.

    Returns
    -------
    dict
        parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems,
        n_systems, gene_expression_nanmask, df_neuro,
        gene_expression_brain (the full brain-gene panel, for callers that want to
        fit their own components), and gene_panel.
    """
    parc, n_regions, distance_matrix, distance_matrix_norm, yeo_systems, n_systems, sa_axis, sf_axis, micro_axis = \
        _load_axes_and_parcellation(in_dir, n_regions)

    df_gene, gene_expression_nanmask, gene_expression_brain = _load_gene_expression(
        in_dir, n_regions, gene_panel=gene_panel)
    df_neuro_cell = _load_cell_type_maps(in_dir, n_regions)

    df_neuro = pd.DataFrame(index=np.arange(n_regions))
    df_neuro['SA_Axis'] = sa_axis
    df_neuro['SF_Axis'] = sf_axis
    # Optional annotation frames are None when unavailable at this resolution (see _load_* warnings).
    df_neuro = pd.concat([df_neuro] + [f for f in (df_gene, df_neuro_cell) if f is not None], axis=1)

    return {
        'parc': parc,
        'n_regions': n_regions,
        'distance_matrix': distance_matrix,
        'distance_matrix_norm': distance_matrix_norm,
        'yeo_systems': yeo_systems,
        'n_systems': n_systems,
        'gene_expression_nanmask': gene_expression_nanmask,
        'df_neuro': df_neuro,
        'gene_expression_brain': gene_expression_brain,
        'gene_panel': gene_panel,
    }

