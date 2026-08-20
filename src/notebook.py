"""Assemble everything a figure notebook needs into one dict.

The orchestrator: it wires the :class:`~src.timescales.Timescales` processor together with the
region-level maps from :mod:`src.supporting_data` and the per-subject structural data from
:mod:`src.subject_data`. Dataset processing parameters live in :mod:`src.config`.
"""

import os

import numpy as np
import pandas as pd

from src.timescales import Timescales, filter_data_dict
from src.supporting_data import load_supporting_data
from src.subject_data import (_load_hcpya_connectivity, _load_hcpd_connectivity,
                              _load_hcpya_myelin, _load_hcpd_myelin, _sync_subject_level_data)
from snaplab_tools.plotting.utils import register_custom_colormaps


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


def load_notebook_data(dataset='HCP-YA', which_modality='FMRI',
                       use_resids=True, use_7t=False,
                       output_type='timescale_zero_crossing_robust_nc3', filter_str=None, # '_lp-0.5hz'
                       load_subject_data=True, n_regions=400, gene_panel='gao'):
    """Load all data a notebook needs for a dataset into a single dict.

    Sets up the Timescales processor, loads supporting region maps (axes, gene /
    cell annotations), the timescale and tSNR data dicts, and — for
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
    gene_panel : str
        Brain-gene list the gene PCA is fit on ('gao', 'dear', ...). Surfaces as
        supporting_data['gene_panel']; tag anything derived from the gene PCA with it.

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

    sydnor_sa_cmap = register_custom_colormaps()['snap_sydnor_sa']
    supporting_data = load_supporting_data(
        os.path.join(timescales_processor.output_dir, 'data', 'supporting_data'),
        n_regions=n_regions, gene_panel=gene_panel)

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

