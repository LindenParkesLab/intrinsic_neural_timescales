import os, sys
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.spatial import distance

sys.path.extend([r'/mnt/storage_ssd_raid/research_projects/snaplab_tools'])
sys.path.extend([r'/mnt/storage_ssd_raid/research_projects/nctpy/src'])

from src.timescales import Timescales
from src.timescales import filter_data_dict
from src.plotting import create_sydnor_sa_colormap

from snaplab_tools.topology import threshold_adjacency_consistency, volume_normalize_adjacency
from snaplab_tools.utils import normalize_x
from snaplab_tools.decomposition import pca_with_nan_handling
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


# ==================== Standalone Utility Functions (for use in notebooks) ====================
def load_supporting_data(in_dir):
    
    sa_axis_file = os.path.join(in_dir, 'schaefer400-7_sa-axis.npy')
    sa_axis = np.load(sa_axis_file)
    sa_axis = sp.stats.rankdata(sa_axis)

    sf_axis_file = os.path.join(in_dir, 'schaefer400-7_cyto.npy')
    sf_axis = np.load(sf_axis_file)
    # sf_axis = sp.stats.rankdata(sf_axis)
    
    micro_file = os.path.join(in_dir, 'schaefer400-7_micro.npy')
    micro_axis = np.load(micro_file)
    # micro_axis = sp.stats.rankdata(micro_axis)

    # Load parcellation data
    parc_file = os.path.join(in_dir, 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    parc = pd.read_csv(parc_file, header=0, names=['ROI_Label', 'ROI_Name', 'R', 'A', 'S'], index_col=0)
    n_regions = parc.shape[0]

    distance_matrix = distance.pdist(
        parc[['R', 'A', 'S']], "euclidean"
    )  # get euclidean distances between nodes
    distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
    distance_matrix_norm = normalize_x(distance_matrix)
    # distance_matrix_norm = np.abs(normalize_x(distance_matrix) - 1)

    # Extract Yeo systems
    yeo_systems = parc['ROI_Name']
    yeo_systems = yeo_systems.str.replace('7Networks_LH_', '')
    yeo_systems = yeo_systems.str.replace('7Networks_RH_', '')
    yeo_systems = [i.split('_')[0] for i in yeo_systems]
    yeo_systems = [i.split(' ')[1] for i in yeo_systems]
    n_systems = len(np.unique(yeo_systems))

    # Load Allen Human Brain Atlas (AHBA) gene expression
    gene_expression = pd.read_csv(os.path.join(in_dir, 'expression_schaefer400-7.csv'), index_col=0)
    # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # gene_expression = pd.DataFrame(data=imputer.fit_transform(gene_expression.values), columns=gene_expression.columns, index=gene_expression.index)
    gao_brain_genes = np.loadtxt(os.path.join(in_dir, 'gao_brain_genes.txt'), dtype=str)
    existing_columns = list(set(gao_brain_genes) & set(gene_expression.columns))
    gene_expression_brain = gene_expression[existing_columns]
    gene_expression_nanmask = np.isnan(gene_expression.values).sum(axis=1) != gene_expression.shape[1]

    # Cell Types
    Jorstad_schaefer400 = pd.read_csv(os.path.join(in_dir, 'cell_maps_Zhang2024', 'schaeffer_Jorstad_FullCombinedFullCombined_400_7Net_expr_mat_new_NormZscore0.3.csv'), header=0, index_col=-1)
    Jorstad_schaefer400 = Jorstad_schaefer400.transpose()
    # Jorstad_schaefer400.replace(0, 0.0001, inplace=True)

    # combine into dictionary
    dict_neuro = dict()
    dict_neuro['SA_Axis'] = sa_axis
    dict_neuro['SF_Axis'] = sf_axis
    dict_neuro['Micro_Axis'] = micro_axis

    # PCA of brain-related genes
    results = pca_with_nan_handling(gene_expression_brain.values, n_components=2, standardize=True, impute=True)
    # print(results.keys())
    # print(results['explained_variance_ratio'])

    # df_neuro['Gene (PC1)'] = sp.stats.zscore(results['scores'][:, 0])
    # df_neuro['Gene (PC1)'] = results['scores'][:, 0]
    dict_neuro['Gene (PC1)'] = results['scores'][:, 0]

    # Create data frame for gene maps
    gene_list = ['GRIA1', 'GRIN1', 'GRIN2A', 'GRIN2B',
                'GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5',
                'PVALB', 'SST', 'PRR5', 'KCNA3']
    # gene_list = ['GRIN2B', 'GRIN2A', 'GABRA3', 'GABRA1', 'PVALB', 'PRR5', 'KCNA3']
    # gene_list = [ampa + nmda + gaba + ['PVALB', 'PRR5', 'KCNA3']][0]
    for gene in gene_list:
        dict_neuro[f'Gene_{gene}'] = gene_expression[gene].values

    ampa = ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4']
    nmda = ['GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2C']
    gaba = ['GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5',
            'GABRB1', 'GABRB2', 'GABRB3', 'GABRG1', 'GABRG2', 'GABRG3']
    # excitation = gene_expression.loc[:, nmda].mean(axis=1).values
    excitation = gene_expression.loc[:, ampa + nmda].mean(axis=1).values
    inhibition = gene_expression.loc[:, gaba].mean(axis=1).values
    # inhibition = gene_expression.loc[:, 'PVALB'].values
    ei_gene = (normalize_x(excitation)+1) / (normalize_x(inhibition)+1)
    dict_neuro['Gene_EI_ratio'] = ei_gene

    supporting_data = {
        # 'sa_axis': sa_axis,
        # 'sf_axis': sf_axis,
        # 'micro_axis': micro_axis,
        'parc': parc,
        'n_regions': n_regions,
        'distance_matrix': distance_matrix,
        'distance_matrix_norm': distance_matrix_norm,
        'yeo_systems': yeo_systems,
        'n_systems': n_systems,
        'dict_neuro': dict_neuro,
        'gene_expression_nanmask': gene_expression_nanmask
        # 'gene_expression': gene_expression,
        # 'gene_expression_brain': gene_expression_brain,
        # 'Jorstad_schaefer400': Jorstad_schaefer400,
    }
    
    return supporting_data