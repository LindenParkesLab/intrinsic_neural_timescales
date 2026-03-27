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


# ==============================================================================
# Dataset processing parameters
# Consumed by timescales_process.ipynb and other processing notebooks.
# Each entry specifies what to pass to DataLoader.register_scans(), plus
# processing flags (nuisance_regression, task_regression, etc.).
# Modify here to change how a dataset is processed across all notebooks.
# ==============================================================================

DATASET_PARAMS = {
    'HCP-YA': {
        # 'modalities':  ['MRI3T'],  # also supports 'MRI7T', 'MEG'
        'modalities':  ['MRI3T', 'MRI7T'],  # also supports 'MRI7T', 'MEG'
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
        'nuisance_regression': False,
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
        'subject_ids_file': 'subject_ids.txt',
        'nuisance_regression': False,
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
        'nuisance_regression': False,
        'task_regression':     False,
        'remove_incomplete_subjects': False,
    },
    'PNC': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer2018p400n17',
        'space':       'MNI152NLin6ASym',
        'sessions':    ['PNC1'],
        'tasks': {
            'MRI3T': ['rest', 'frac2back'],
        },
        'subject_ids_file': 'subject_ids_lowmotion.txt',
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
            'MRI3T': ['rest', 'bart', 'stopsignal'],
        },
        'subject_ids_file': 'subject_ids_not1_hc.txt',
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
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'Penn_LEAD': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    ['1'],
        'tasks': {
            'MRI3T': ['rest', 'nback'],
        },
        'subject_ids_file': 'subject_ids.txt',
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'HBN': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer2018p400n17',
        'space':       'MNI152NLin6ASym',
        'sessions':    ['HBNsiteCBIC'],   # also 'HBNsiteRU'
        'tasks': {
            'MRI3T': ['rest'],
        },
        'subject_ids_file': 'subject_ids_HBNsiteCBIC.txt',
        'nuisance_regression': False,
        'task_regression':     False,
        'remove_incomplete_subjects': False,
    },
}


def load_notebook_data(dataset='HCP-YA', which_modality='FMRI',
                       use_resids=True, use_7t=False, 
                       output_type='timescale_zero_crossing_robust_nc3', meg_freq='_lp-0.5hz',
                       load_subject_data=True):
    
    print('Loading data for notebooks...')

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

    # load tsnr
    timescales_processor.get_precomputed_outputs(output_type='metric_tsnr')
    data_dict_tsnr = timescales_processor.data_dict.copy()
    data_dict_tsnr = filter_data_dict(data_dict_tsnr, remove_subject_level=True)
    print(len(data_dict_tsnr))

    # Subject-level INTs
    if which_modality == 'MEG':
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
    subject_ids = list(dict.fromkeys(scan.subject_id for scan in timescales_processor.scan_registry))
    n_subs = len(subject_ids)
    print(f'Number of subjects: {n_subs}')

    if dataset == 'HCP-YA' and load_subject_data:
        # connectivity
        print('Loading connectivity data...')
        adjacency_matrices = np.zeros((n_regions, n_regions, n_subs))
        strength = np.zeros((n_regions, n_subs))

        for i in np.arange(n_subs):
            adjacency_file = os.path.join(derivdata_dir, 'mrtrix_10m', f'sub-{subject_ids[i]}', 
                                          f'sub-{subject_ids[i]}_atlas-Schaefer4007_space-T1w_stat-streamlinecount_relmat.csv')
            voxel_counts_file = os.path.join(derivdata_dir, 'parcellations', f'sub-{subject_ids[i]}',
                                             f'sub-{subject_ids[i]}_atlas-Schaefer4007_space-T1w_res-1_regionsizes.txt')
            if os.path.exists(adjacency_file):
                adjacency = np.loadtxt(adjacency_file, delimiter=',', dtype=float)
                voxel_counts = np.loadtxt(voxel_counts_file)
                adjacency_norm = volume_normalize_adjacency(adjacency, voxel_counts)
                strength[:, i] = np.log(adjacency_norm.sum(axis=0))
                # adjacency_matrices[:, :, i] = adjacency_norm
                adjacency_matrices[:, :, i] = adjacency
            else:
                adjacency_matrices[:, :, i] = np.nan
                print(f'\tsub-{subject_ids[i]} not found')
                    
        adjacency_mean = threshold_adjacency_consistency(adjacency_matrices)
        # adjacency_mean = normalize_x(adjacency_mean)
        # adjacency_mean = np.multiply(adjacency_mean, distance_matrix_norm) # scale by distance. This biased toward long-range connectivity
        # adjacency_mean = adjacency_mean / adjacency_mean.max()

        supporting_data['df_neuro'].loc[:,'HCPYA_Strength'] = np.log(adjacency_mean.sum(axis=0))
        # supporting_data['df_neuro'].loc[:,'SC_Mean'] = np.log(adjacency_mean.mean(axis=0))
        supporting_data['df_neuro'].loc[:,'HCPYA_Degree'] = np.sum(adjacency_mean > 0, axis=0)
        adjacency_norm = matrix_normalization(adjacency_mean, system='continuous')
        # eigenvalues, eigenvectors = np.linalg.eig(adjacency_norm)
        # supporting_data['df_neuro'].loc[:,'SC_Eigenvalues'] = eigenvalues
        # supporting_data['df_neuro'].loc[:,'eigenvalues'] = winsorize(eigenvalues, 0, 95)

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
        supporting_data['df_neuro'].loc[:,'HCPYA_Myelin'] = myelin_mean
        
        myelin_gene_corr = sp.stats.spearmanr(supporting_data['df_neuro'].loc[gene_expression_nanmask,'PCA_Gene-PC1'],
                                              supporting_data['df_neuro'].loc[gene_expression_nanmask,'HCPYA_Myelin'])[0]
        if myelin_gene_corr < 0:
            supporting_data['df_neuro'].loc[:,'PCA_Gene-PC1'] = supporting_data['df_neuro'].loc[:,'PCA_Gene-PC1'] * -1
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
        'data_dict_tsnr': data_dict_tsnr,
        'supporting_data': supporting_data
    }

    try:
        notebook_data['adjacency_matrices'] = adjacency_matrices
        notebook_data['adjacency_mean'] = adjacency_mean
        notebook_data['strength'] = strength
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
    sf_axis = sp.stats.rankdata(sf_axis * -1)
    
    micro_file = os.path.join(in_dir, 'schaefer400-7_micro.npy')
    micro_axis = np.load(micro_file)
    micro_axis = sp.stats.rankdata(micro_axis)

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

    # setup neuro dataframe
    n_components = 2
    df_neuro = pd.DataFrame(index=np.arange(n_regions))
    df_neuro.loc[:,'SA_Axis'] = sa_axis
    df_neuro.loc[:,'SF_Axis'] = sf_axis
    # df_neuro.loc[:,'Micro_Axis'] = micro_axis

    # 1) HBA gene maps
    # gene_list = ['GRIA1', 'GRIN1', 'GRIN2A', 'GRIN2B',
    #             'GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5',
    #             'PVALB', 'SST', 'PRR5', 'KCNA3']
    gene_list = ['GRIN2B', 'GRIN2A', 'GABRA3', 'GABRA1', 'PVALB', 'PRR5', 'KCNA3', 'SCN1A'] # selection from lit review
    # gene_list = [ampa + nmda + gaba + ['PVALB', 'PRR5', 'KCNA3']][0]
    for gene in gene_list:
        df_neuro.loc[:,f'Gene_{gene}'] = gene_expression[gene].values
    pca_results_genes = pca_with_nan_handling(gene_expression_brain.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_neuro.loc[:,f'PCA_Gene-PC{i+1}'] = pca_results_genes['scores'][:, i]

    # ampa = ['GRIA1', 'GRIA2', 'GRIA3', 'GRIA4']
    # nmda = ['GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2C']
    # gaba = ['GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5',
    #         'GABRB1', 'GABRB2', 'GABRB3', 'GABRG1', 'GABRG2', 'GABRG3']
    # excitation = gene_expression.loc[:, nmda].mean(axis=1).values
    # excitation = np.nanmean(gene_expression.loc[:, ampa + nmda].values, axis=1)
    # inhibition = np.nanmean(gene_expression.loc[:, gaba].values, axis=1)
    # inhibition = gene_expression.loc[:, 'PVALB'].values
    # ei_gene = (normalize_x(excitation)+1) / (normalize_x(inhibition)+1)
    # df_neuro.loc[:,'Gene_EI_ratio'] = ei_gene

    # 2) Hansen PET maps
    df_neuro_pet = pd.read_csv(os.path.join(in_dir, 'hansen_receptors', 'results', 'receptor_data_scale400.csv'),
                               index_col=None, header=None)
    pet_names = np.load(os.path.join(in_dir, 'hansen_receptors', 'data', 'receptor_names_pet.npy'))
    pet_names = ['PET_' + s for s in pet_names]
    pet_categories = pd.read_csv(os.path.join(in_dir, 'pet_categories_custom.csv'), index_col=0, header=0)
    df_neuro_pet.columns = pet_names
    pca_results_pet = pca_with_nan_handling(df_neuro_pet.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_neuro_pet.loc[:,f'PCA_PET-PC{i+1}'] = pca_results_pet['scores'][:, i]
    df_neuro = pd.concat((df_neuro, df_neuro_pet), axis=1)

    # 3) Zhang cell-type maps
    df_neuro_cell = pd.read_csv(os.path.join(in_dir, 'human-cellular-func-con', 'cell_maps', 'Schaefer_atalas', 'Jorstad',
                                       'schaeffer_Jorstad_400_7Net_expr_mat_new_NormZscore0.3.csv'),
                          header=0, index_col=-1).transpose()
    df_neuro_cell = df_neuro_cell.reset_index(drop=True)
    cell_names = ['Cell_' + s for s in list(df_neuro_cell.columns)]
    df_neuro_cell.columns = cell_names
    pca_results_cell = pca_with_nan_handling(df_neuro_cell.values, n_components=n_components, standardize=True, impute=True)
    for i in np.arange(n_components):
        df_neuro_cell.loc[:,f'PCA_Cell-PC{i+1}'] = pca_results_cell['scores'][:, i]
    # df_neuro = pd.concat((df_neuro, df_neuro_cell), axis=1)

    supporting_data = {
        'parc': parc,
        'n_regions': n_regions,
        'distance_matrix': distance_matrix,
        'distance_matrix_norm': distance_matrix_norm,
        'yeo_systems': yeo_systems,
        'n_systems': n_systems,
        'gene_expression_nanmask': gene_expression_nanmask,
        'pet_categories': pet_categories,
        'df_neuro': df_neuro,
        'pca_results_genes': pca_results_genes,
        'pca_results_pet': pca_results_pet,
        'pca_results_cell': pca_results_cell,
    }
    
    return supporting_data