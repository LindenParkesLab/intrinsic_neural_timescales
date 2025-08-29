# import
import os, sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

def process_data(data_dict, combine_rest=False, combine_movies=False, which_data='HCP-YA'):
    data_dict_processed = dict()
    conditions = list(data_dict.keys())
    # print(conditions)

    if combine_rest is True:
        rest_list = [s for s, s in enumerate(conditions) if 'REST' in s]
        data_dict_processed['REST'] = np.zeros(data_dict[rest_list[0]].shape)
        for condition in rest_list:
            data_dict_processed['REST'] += data_dict[condition].copy()
        data_dict_processed['REST'] = np.divide(data_dict_processed['REST'], len(rest_list))
    else:
        if which_data == 'HCP-YA':
            # process 7T rest
            rest_7T_list = [s for s, s in enumerate(conditions) if 'REST' in s and '7T' in s]
            data_dict_processed['REST_7T'] = np.zeros(data_dict[rest_7T_list[0]].shape)
            for condition in rest_7T_list:
                data_dict_processed['REST_7T'] += data_dict[condition].copy()
            data_dict_processed['REST_7T'] = np.divide(data_dict_processed['REST_7T'], len(rest_7T_list))
        
        # process 3T rest
        rest_3T_list = [s for s, s in enumerate(conditions) if 'REST' in s and '7T' not in s]
        data_dict_processed['REST_3T'] = np.zeros(data_dict[rest_3T_list[0]].shape)
        for condition in rest_3T_list:
            data_dict_processed['REST_3T'] += data_dict[condition].copy()
        data_dict_processed['REST_3T'] = np.divide(data_dict_processed['REST_3T'], len(rest_3T_list))

    # process movies
    try:
        movie_list = [s for s, s in enumerate(conditions) if 'tfMRI' in s and '7T' in s]
        if combine_movies is True:
            data_dict_processed['MOVIES'] = np.zeros(data_dict[movie_list[0]].shape)
            for condition in movie_list:
                data_dict_processed['MOVIES'] += data_dict[condition].copy()
            data_dict_processed['MOVIES'] = np.divide(data_dict_processed['MOVIES'], len(movie_list))
        else:
            for condition in movie_list:
                data_dict_processed[condition] = data_dict[condition].copy()
    except:
        pass

    # process tasks
    if which_data == 'HCP-YA':
        task_list_lr = [s for s, s in enumerate(conditions) if 'tfMRI' in s and 'LR' in s and '7T' not in s]
        task_list_rl = [s for s, s in enumerate(conditions) if 'tfMRI' in s and 'RL' in s and '7T' not in s]
        for condition in task_list_lr:
            condition_stem = '{0}_{1}'.format(condition.split('_')[0], condition.split('_')[1])
            data_dict_processed[condition_stem] = data_dict[condition].copy()
            condition_other = '{0}_{1}_RL'.format(condition.split('_')[0], condition.split('_')[1])
            # print(condition, condition_stem, condition_other)
            if condition_other in task_list_rl:
                data_dict_processed[condition_stem] += data_dict[condition_other].copy()
                data_dict_processed[condition_stem] = data_dict_processed[condition_stem] / 2
    elif which_data == 'HCP-D':
        task_list = [s for s, s in enumerate(conditions) if 'tfMRI' in s and 'PA' in s]
        for condition in task_list:
            data_dict_processed[condition] = data_dict[condition].copy()

    new_conditions = list(data_dict_processed.keys())
    print(new_conditions)

    return data_dict_processed

def sort_and_group_timescales(timescales, brain_map, ntile=10):
    n_nodes = len(brain_map)
    brain_map_sort_idx = np.argsort(brain_map)
    
    timescales_sorted = timescales[:, brain_map_sort_idx]
    timescales_sorted_dec = np.zeros((timescales_sorted.shape[0], ntile))

    step_size = int(n_nodes / ntile)
    counter = 0
    for i in np.arange(0, n_nodes, step_size):
        timescales_sorted_dec[:, counter] = timescales_sorted[:, i:i+step_size].mean(axis=1)
        counter += 1
        
    return timescales_sorted_dec

def rename_task_columns_to_uppercase(df):
    """
    Alternative version using pandas string methods for column renaming.
    """
    words_to_convert = ['Emotion', 'Gambling', 'Language', 'Relational', 'Social']
    
    df_renamed = df.copy()
    
    # Apply replacements to column names
    new_columns = df.columns.to_series()
    for word in words_to_convert:
        new_columns = new_columns.str.replace(word, word.upper(), regex=False)
    
    df_renamed.columns = new_columns
    
    return df_renamed

def load_data_for_analysis(which_data='HCP-YA', atlas='Schaefer4007', 
                           indir='/home/lindenmp/research_projects/intrinsic_neural_timescales/data',
                           outdir=None, include_movies=True):
    """
    Load common fMRI data needed for both timescales analysis notebooks.
    
    This function loads all the common data structures used in both 
    results_fmri_timescales.ipynb and results_fmri_timescales_behavior.ipynb:
    - SA axis data
    - Subject IDs
    - Behavioral data 
    - Rest timescales
    - Task timescales
    - Movie timescales (if available)
    - Parcellation data
    - Yeo systems information
    
    Parameters:
    -----------
    which_data : str, default 'HCP-YA'
        Dataset to load ('HCP-YA' or 'HCP-D')
    atlas : str, default 'Schaefer4007'
        Atlas to use ('Schaefer4007', 'Schaefer2007', 'Schaefer1007')
    indir : str
        Input directory containing the raw data files
    outdir : str, optional
        Output directory. If None, constructed from which_data
    include_movies : bool, default True
        Whether to load movie timescales (only for HCP-YA)
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing all loaded data with keys:
        - 'sa_axis': SA axis values
        - 'sf_axis': SF axis values
        - 'subject_ids': Array of subject IDs
        - 'n_subs': Number of subjects
        - 'df_behavior': Behavioral dataframe with renamed columns
        - 'rest_timescales': Dict of rest timescales
        - 'task_timescales': Dict of task timescales  
        - 'movie_timescales': Dict of movie timescales (if available)
        - 'parc': Parcellation dataframe
        - 'yeo_systems': List of Yeo system labels
        - 'config': Configuration dict with TR values, scan names, etc.
    """
    
    if outdir is None:
        outdir = f'/home/lindenmp/research_projects/intrinsic_neural_timescales/results/{which_data}'
    
    # Set up configuration based on dataset and atlas
    config = {}
    
    # Atlas configuration
    if atlas == 'Schaefer4007':
        config['n_parcels'] = 400
        parc_file = os.path.join(indir, 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    elif atlas == 'Schaefer2007':
        config['n_parcels'] = 200
        parc_file = os.path.join(indir, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    elif atlas == 'Schaefer1007':
        config['n_parcels'] = 100
        parc_file = os.path.join(indir, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    else:
        raise ValueError(f"Unknown atlas: {atlas}")
    
    # Dataset-specific configuration
    if which_data == 'HCP-YA':
        config['tr_3T'] = 0.720
        config['tr_7T'] = 1.0
        if include_movies:
            config['rest_scans'] = ['rfMRI_REST1_7T_PA', 'rfMRI_REST2_7T_AP', 'rfMRI_REST3_7T_PA', 'rfMRI_REST4_7T_AP', 
                                    'rfMRI_REST1_LR', 'rfMRI_REST2_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_RL']
            config['movies'] = ['tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP']
        else:
            config['rest_scans'] = ['rfMRI_REST1_LR', 'rfMRI_REST2_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_RL']
            config['movies'] = []
        config['tasks'] = ['tfMRI_WM_LR', 'tfMRI_EMOTION_LR', 'tfMRI_GAMBLING_LR', 'tfMRI_LANGUAGE_LR', 
                          'tfMRI_RELATIONAL_LR', 'tfMRI_SOCIAL_LR', 'tfMRI_WM_RL', 'tfMRI_EMOTION_RL', 
                          'tfMRI_GAMBLING_RL', 'tfMRI_LANGUAGE_RL', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_RL']
    elif which_data == 'HCP-D':
        config['tr_3T'] = 0.800
        config['rest_scans'] = ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']
        config['tasks'] = ['tfMRI_CARIT_PA']
        config['movies'] = []
    else:
        raise ValueError(f"Unknown dataset: {which_data}")
    
    print(f"Loading common fMRI data for {which_data} with {atlas} atlas...")
    
    # Load SA axis
    sa_axis_file = os.path.join(indir, f'schaefer{config["n_parcels"]}-7_sa-axis.npy')
    sa_axis = np.load(sa_axis_file)
    sa_axis = sp.stats.rankdata(sa_axis)
    sa_axis_sort_idx = np.argsort(sa_axis)
    print(f"Loaded SA axis with shape: {sa_axis.shape}")
    
    # Load SF axis
    sf_axis_file = os.path.join(indir, f'schaefer{config["n_parcels"]}-7_cyto.npy')
    sf_axis = np.load(sf_axis_file) * -1  # Invert SF axis
    sf_axis = sp.stats.rankdata(sf_axis)
    sf_axis_sort_idx = np.argsort(sf_axis)
    print(f"Loaded SF axis with shape: {sf_axis.shape}")
    
    # Load subject IDs and handle HCP-YA 3T/7T matching
    if which_data == 'HCP-YA' and include_movies:
        subject_ids = np.loadtxt(os.path.join(indir, f'{which_data}_{atlas}_subjids.txt'), dtype=str)
        subject_ids_7t = np.loadtxt(os.path.join(indir, f'{which_data}-7T_{atlas}_subjids.txt'), dtype=str)

        # Find subjects that exist in both 3T and 7T datasets
        found_indices_3t = []
        for index, item in enumerate(subject_ids):
            if item in subject_ids_7t:
                found_indices_3t.append(index)
                
        found_indices_7t = []
        for index, item in enumerate(subject_ids_7t):
            if item in subject_ids:
                found_indices_7t.append(index)
                
        subject_ids = subject_ids[found_indices_3t]
        n_subs = len(found_indices_3t)
        
        print(f"Found {len(found_indices_3t)} subjects with both 3T and 7T data")
        config['found_indices_3t'] = found_indices_3t
        config['found_indices_7t'] = found_indices_7t
    else:
        subject_ids = np.loadtxt(os.path.join(outdir, f'{which_data}_{atlas}_subjids.txt'), dtype=str)
        n_subs = len(subject_ids)
        print(f"Loaded {n_subs} subjects")
    
    # Load behavioral data
    if which_data == 'HCP-YA':
        df_behavior = pd.read_csv('/mnt/storage_ssd_raid/research_data/HCP_YA/unrestricted_lindenmp_9_26_2022_16_52_14.csv', index_col=0)
        df_behavior = df_behavior.loc[subject_ids.astype(int)]
        df_behavior_restricted = pd.read_csv('/mnt/storage_ssd_raid/research_data/HCP_YA/RESTRICTED_lindenmp_9_26_2022_16_42_3.csv', index_col=0)
        df_behavior_restricted = df_behavior_restricted.loc[subject_ids.astype(int)]
        df_behavior['age'] = df_behavior_restricted['Age_in_Yrs']
        df_behavior['sex'] = df_behavior['Gender'] == 'M'
        df_behavior['sex'] = df_behavior['sex'].astype(int)
        df_behavior = rename_task_columns_to_uppercase(df_behavior)
        print(f"Loaded behavioral data with shape: {df_behavior.shape}")
    else:
        df_behavior = None
        print("Behavioral data loading not implemented for HCP-D")
    
    # Load rest timescales
    rest_timescales = dict()
    file_suffix = 'timescales_acf-lag.npy'
    # file_suffix = 'timescales_acf-auc.npy'
    for which_scan in config['rest_scans']:
        print(f"Loading {which_scan}...")
        if which_data == 'HCP-YA' and include_movies:
            try:
                timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_3T']
                timescales = timescales[found_indices_3t]
            except:
                timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}-7T_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_7T']
                timescales = timescales[found_indices_7t]
        else:
            timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_3T']
        
        print(f"  Shape: {timescales.shape}")
        rest_timescales[which_scan] = timescales
    
    # Load task timescales
    task_timescales = dict()
    for which_scan in config['tasks']:
        print(f"Loading {which_scan}...")
        timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_3T']
        if which_data == 'HCP-YA' and include_movies:
            timescales = timescales[found_indices_3t]
        print(f"  Shape: {timescales.shape}")
        task_timescales[which_scan] = timescales

    # Load movie timescales (HCP-YA only)
    movie_timescales = dict()
    if which_data == 'HCP-YA' and include_movies:
        for which_scan in config['movies']:
            print(f"Loading {which_scan}...")
            timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}-7T_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_7T']
            timescales = timescales[found_indices_7t]
            print(f"  Shape: {timescales.shape}")
            movie_timescales[which_scan] = timescales
            
    # Load myelin map
    myelin_file = os.path.join(indir, f'{which_data.replace("_", "")}_{atlas}_myelin.npy')
    myelin_maps = np.load(myelin_file)
    
    # Load parcellation data
    parc = pd.read_csv(parc_file, header=0, names=['ROI_Label', 'ROI_Name', 'R', 'A', 'S'])
    
    # Extract Yeo systems
    yeo_systems = parc['ROI_Name']
    yeo_systems = yeo_systems.str.replace('7Networks_LH_', '')
    yeo_systems = yeo_systems.str.replace('7Networks_RH_', '')
    yeo_systems = [i.split('_')[0] for i in yeo_systems]
    yeo_systems = [i.split(' ')[1] for i in yeo_systems]
    
    print(f"Extracted {len(set(yeo_systems))} unique Yeo systems: {set(yeo_systems)}")
    
    # Create return dictionary
    data_dict = {
        'sa_axis': sa_axis,
        'sa_axis_sort_idx': sa_axis_sort_idx,
        'sf_axis': sf_axis,
        'sf_axis_sort_idx': sf_axis_sort_idx,
        'subject_ids': subject_ids,
        'n_subs': n_subs,
        'df_behavior': df_behavior,
        'rest_timescales': rest_timescales,
        'task_timescales': task_timescales,
        'movie_timescales': movie_timescales,
        'myelin_maps': myelin_maps,
        'parc': parc,
        'yeo_systems': yeo_systems,
        'config': config
    }
    
    print(f"\nSuccessfully loaded common fMRI data:")
    print(f"  - SA axis: {sa_axis.shape}")
    print(f"  - SF axis: {sf_axis.shape}")
    print(f"  - Subjects: {n_subs}")
    print(f"  - Rest conditions: {len(rest_timescales)}")
    print(f"  - Task conditions: {len(task_timescales)}")
    print(f"  - Movie conditions: {len(movie_timescales)}")
    print(f"  - Brain regions: {config['n_parcels']}")
    if df_behavior is not None:
        print(f"  - Behavioral measures: {df_behavior.shape[1]}")
    
    return data_dict
