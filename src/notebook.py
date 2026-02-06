import os, sys
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from tqdm import tqdm

sys.path.extend([r'/mnt/storage_ssd_raid/research_projects/snaplab_tools'])
sys.path.extend([r'/mnt/storage_ssd_raid/research_projects/nctpy/src'])

from src.timescales import Timescales
from src.timescales import filter_data_dict
from src.data import load_supporting_data
from src.plotting import create_sydnor_sa_colormap
from snaplab_tools.topology import threshold_adjacency_consistency, volume_normalize_adjacency
from nctpy.utils import matrix_normalization


def load_notebook_data(dataset='HCP-YA', which_modality='FMRI', 
                       use_resids=True, use_7t=False, 
                       output_type='timescale_zero_crossing_robust_nc3', meg_freq='_lp-0.5hz'):

    # setup timescales process
    timescales_processor = Timescales(dataset=dataset, timeseries_data=None)
    derivdata_dir = timescales_processor.derivdata_dir
    output_dir = timescales_processor.output_dir
    results_dir = os.path.join(output_dir, 'results', dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # extra supporting stuff
    sydnor_sa_cmap = create_sydnor_sa_colormap()
    supporting_data = load_supporting_data(os.path.join(timescales_processor.output_dir, 'data', 'supporting_data'))
    gene_expression_nanmask = supporting_data['gene_expression_nanmask']
    n_regions = supporting_data['n_regions']

    ########################################################################################################################################
    # load timescale outputs
    ########################################################################################################################################
    if dataset == 'HCP-YA' and which_modality == 'FMRI' and use_7t:
        timescales_processor.load_scan_registry(f'scan_registry_{dataset}_7T.csv')
    elif dataset == 'HCP-YA' and which_modality == 'FMRI' and not use_7t:
        timescales_processor.load_scan_registry(f'scan_registry_{dataset}.csv')
    elif dataset == 'HCP-YA' and which_modality == 'MEG':
        timescales_processor.load_scan_registry(f'scan_registry_{dataset}_MEG.csv')
    else:
        timescales_processor.load_scan_registry(f'scan_registry_{dataset}.csv')

    if dataset == 'HCP-YA':
        timescales_processor.match_registry_to_behavior()
        timescales_processor.df_behavior_matched.shape

    # Subject-level INTs
    if which_modality == 'FMRI':
        pass
        # output_type = 'timescale_zero_crossing'
        # output_type = 'timescale_zero_crossing_robust_nc2'
        # output_type = 'timescale_zero_crossing_robust_nc3'
        # output_type = 'timescale_zero_crossing_robust_nc5'
        # output_type = 'timescale_zero_crossing_robust_nc10'
        # output_type = 'timescale_auc_zero_crossing'
        # output_type = 'timescale_auc_half_crossing'
        # output_type = 'timescale_decay_time_half'
        # output_type = 'timescale_tau_howell'
        # output_type = 'timescale_tau_murray_par'
        # output_type = 'timescale_B_murray_par'

        # output_type = 'timescale_zero_crossing_0bk'
        # output_type = 'timescale_zero_crossing_2bk'
    elif which_modality == 'MEG':
        # meg_freq = '_lp-30hz'
        # meg_freq = '_lp-10hz'
        # meg_freq = '_lp-5hz'
        # meg_freq = '_lp-1hz'
        # meg_freq = '_lp-0.5hz'
        output_type = 'timescale_zero_crossing' + meg_freq
    output_type_resid = output_type + '_resids'
    print(output_type)

    timescales_processor.get_precomputed_outputs(output_type=output_type)
    data_dict = timescales_processor.data_dict.copy()
    print(len(data_dict))

    if use_resids:
        timescales_processor.get_precomputed_outputs(output_type=output_type_resid)
        data_dict_residuals = timescales_processor.data_dict.copy()
        data_dict, data_dict_residuals = filter_data_dict(data_dict, data_dict_residuals)
    else:
        data_dict = filter_data_dict(data_dict, remove_subject_level=True)
    print(len(data_dict))

    ########################################################################################################################################
    # load subject level neuro data
    ########################################################################################################################################
    subject_ids = timescales_processor.scan_registry['sub'].unique()
    n_subs = len(subject_ids)
    print(f'Number of subjects: {n_subs}')

    if dataset == 'HCP-YA':
        # connectivity
        print('Loading connectivity data...')
        adjacency_matrices = np.zeros((n_regions, n_regions, n_subs))

        for i in np.arange(n_subs):
            adjacency_file = os.path.join(derivdata_dir, 'mrtrix_10m', f'sub-{subject_ids[i]}', 
                                          f'sub-{subject_ids[i]}_atlas-Schaefer4007_space-T1w_stat-streamlinecount_relmat.csv')
            voxel_counts_file = os.path.join(derivdata_dir, 'parcellations', f'sub-{subject_ids[i]}',
                                             f'sub-{subject_ids[i]}_atlas-Schaefer4007_space-T1w_res-1_regionsizes.txt')
            if os.path.exists(adjacency_file):
                adjacency = np.loadtxt(adjacency_file, delimiter=',', dtype=float)
                voxel_counts = np.loadtxt(voxel_counts_file)
                adjacency_norm = volume_normalize_adjacency(adjacency, voxel_counts)
                adjacency_matrices[:, :, i] = adjacency_norm
                # adjacency_matrices[:, :, i] = adjacency
            else:
                adjacency_matrices[:, :, i] = np.nan
                print(f'\tsub-{subject_ids[i]} not found')
                    
        adjacency_mean = threshold_adjacency_consistency(adjacency_matrices)
        # adjacency_mean = normalize_x(adjacency_mean)
        # adjacency_mean = np.multiply(adjacency_mean, distance_matrix_norm) # scale by distance. This biased toward long-range connectivity
        adjacency_mean = adjacency_mean / adjacency_mean.max()

        supporting_data['dict_neuro']['Strength'] = np.log(adjacency_mean.sum(axis=0))
        # supporting_data['dict_neuro']['SC_Mean'] = np.log(adjacency_mean.mean(axis=0))
        supporting_data['dict_neuro']['SC_Degree'] = np.sum(adjacency_mean > 0, axis=0)
        adjacency_norm = matrix_normalization(adjacency_mean, system='continuous')
        eigenvalues, eigenvectors = np.linalg.eig(adjacency_norm)
        supporting_data['dict_neuro']['SC_Eigenvalues'] = eigenvalues
        # supporting_data['dict_neuro']['eigenvalues'] = winsorize(eigenvalues, 0, 95)

        # myelin
        print('Loading myelin data...')
        myelin = np.zeros((n_regions, n_subs))

        for i in np.arange(n_subs):
            myelin_file = os.path.join(derivdata_dir, 'parcellated_data', f'sub-{subject_ids[i]}', 'ses-01', 'anat',
                                    f'sub-{subject_ids[i]}_ses-01_space-fsLR_seg-Schaefer4007_stat-mean_myelin.pscalar.nii')
            if os.path.exists(myelin_file):
                img = nib.load(str(myelin_file))
                myelin[:, i] = img.get_fdata()
            else:
                myelin[:, i] = np.nan
                print(f'\tsub-{subject_ids[i]} not found')
                
        myelin_mean = np.nanmean(myelin, axis=1)
        supporting_data['dict_neuro']['Myelin'] = myelin_mean
        
        myelin_gene_corr = sp.stats.spearmanr(supporting_data['dict_neuro']['Gene (PC1)'][gene_expression_nanmask], supporting_data['dict_neuro']['Myelin'][gene_expression_nanmask])[0]
        if myelin_gene_corr < 0:
            supporting_data['dict_neuro']['Gene (PC1)'] = supporting_data['dict_neuro']['Gene (PC1)'] * -1
        else:
            pass

    ########################################################################################################################################
    # return
    ########################################################################################################################################
    notebook_data = {
        'timescales_processor': timescales_processor,
        'results_dir': results_dir,
        'sydnor_sa_cmap': sydnor_sa_cmap,
        'data_dict': data_dict,
        'supporting_data': supporting_data
    }

    try:
        notebook_data['adjacency_matrices'] = adjacency_matrices
    except:
        pass
    
    try:
        notebook_data['myelin'] = myelin
    except:
        pass
    
    return notebook_data