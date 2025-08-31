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
                           outdir=None, include_3T=True, include_7T=False, include_meg=False):
    """
    Load common fMRI and MEG data needed for timescales analysis notebooks.
    
    This function loads all the common data structures used in timescales analysis:
    - SA axis data
    - Subject IDs
    - Behavioral data 
    - Rest timescales
    - Task timescales
    - Movie timescales (if 7T data included)
    - MEG timescales (if MEG data included)
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
    include_3T : bool, default True
        Whether to load 3T fMRI timescales
    include_7T : bool, default False
        Whether to load 7T fMRI timescales (including movie data for HCP-YA)
    include_meg : bool, default False
        Whether to load MEG timescales (only available for HCP-YA)
        
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
        - 'movie_timescales': Dict of movie timescales (if 7T available)
        - 'meg_timescales': Dict of MEG timescales (if MEG requested)
        - 'parc': Parcellation dataframe
        - 'yeo_systems': List of Yeo system labels
        - 'config': Configuration dict with TR values, scan names, etc.
    """
    
    # Validate input parameters
    if not any([include_3T, include_7T, include_meg]):
        raise ValueError("At least one of include_3T, include_7T, or include_meg must be True")
    
    if include_meg and which_data != 'HCP-YA':
        print("Warning: MEG data not available for HCP-D, setting include_meg=False")
        include_meg = False
    
    if include_7T and which_data == 'HCP-D':
        print("Warning: 7T data not available for HCP-D, setting include_7T=False")
        include_7T = False
    
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
        meg_sampling_rate = 508.63
        config['tr_meg'] = 1 / meg_sampling_rate
        
        # Configure scans based on requested modalities
        config['rest_scans'] = []
        config['movies'] = []
        
        if include_3T:
            config['rest_scans'].extend(['rfMRI_REST1_LR', 'rfMRI_REST2_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_RL'])
        
        if include_7T:
            config['rest_scans'].extend(['rfMRI_REST1_7T_PA', 'rfMRI_REST2_7T_AP', 'rfMRI_REST3_7T_PA', 'rfMRI_REST4_7T_AP'])
            config['movies'] = ['tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP']
        
        # Task scans are only available for 3T
        if include_3T:
            config['tasks'] = ['tfMRI_WM_LR', 'tfMRI_EMOTION_LR', 'tfMRI_GAMBLING_LR', 'tfMRI_LANGUAGE_LR', 
                               'tfMRI_RELATIONAL_LR', 'tfMRI_SOCIAL_LR', 'tfMRI_WM_RL', 'tfMRI_EMOTION_RL', 
                               'tfMRI_GAMBLING_RL', 'tfMRI_LANGUAGE_RL', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_RL']
        else:
            config['tasks'] = []
        
        if include_meg:
            config['meg_scans'] = ['3-Restin', '4-Restin', '5-Restin', '6-Wrkmem', '7-Wrkmem']
            
    elif which_data == 'HCP-D':
        config['tr_3T'] = 0.800
        if include_3T:
            config['rest_scans'] = ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']
            config['tasks'] = ['tfMRI_CARIT_PA']
        else:
            config['rest_scans'] = []
            config['tasks'] = []
        config['movies'] = []
        # MEG already handled above with warning
            
    else:
        raise ValueError(f"Unknown dataset: {which_data}")
    
    print(f"Loading data for {which_data} with {atlas} atlas...")
    modalities = []
    if include_3T:
        modalities.append("3T")
    if include_7T:
        modalities.append("7T")
    if include_meg:
        modalities.append("MEG")
    print(f"Requested modalities: {', '.join(modalities)}")
    
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
    
    # Load subject IDs and handle cross-modality matching
    if which_data == 'HCP-YA' and sum([include_3T, include_7T, include_meg]) > 1:
        # Load all subject ID sets for requested modalities
        subject_sets = []
        subject_labels = []
        subject_ids_3t = None
        subject_ids_7t = None
        subject_ids_meg = None
        
        if include_3T:
            subject_ids_3t = np.loadtxt(os.path.join(outdir, f'{which_data}_{atlas}_subjids.txt'), dtype=str)
            subject_sets.append(set(subject_ids_3t))
            subject_labels.append('3T')
        
        if include_7T:
            subject_ids_7t = np.loadtxt(os.path.join(indir, f'{which_data}-7T_{atlas}_subjids.txt'), dtype=str)
            subject_sets.append(set(subject_ids_7t))
            subject_labels.append('7T')
        
        if include_meg:
            subject_ids_meg = np.loadtxt(os.path.join(outdir, f'{which_data}-MEG_{atlas}_subjids.txt'), dtype=str)
            subject_sets.append(set(subject_ids_meg))
            subject_labels.append('MEG')
        
        # Find intersection of all requested modalities
        common_subjects = set.intersection(*subject_sets)
        subject_ids = np.array(list(common_subjects))
        
        # Create indices for each modality
        found_indices_3t = None
        found_indices_7t = None
        found_indices_meg = None
        
        if include_3T and subject_ids_3t is not None:
            found_indices_3t = [i for i, subj in enumerate(subject_ids_3t) if subj in common_subjects]
        
        if include_7T and subject_ids_7t is not None:
            found_indices_7t = [i for i, subj in enumerate(subject_ids_7t) if subj in common_subjects]
        
        if include_meg and subject_ids_meg is not None:
            found_indices_meg = [i for i, subj in enumerate(subject_ids_meg) if subj in common_subjects]
        
        n_subs = len(subject_ids)
        modality_str = ' + '.join(subject_labels)
        print(f"Found {n_subs} subjects with {modality_str} data")
        
    else:
        # Single modality or HCP-D case
        if include_3T:
            subject_ids = np.loadtxt(os.path.join(outdir, f'{which_data}_{atlas}_subjids.txt'), dtype=str)
        elif include_7T:
            subject_ids = np.loadtxt(os.path.join(indir, f'{which_data}-7T_{atlas}_subjids.txt'), dtype=str)
        elif include_meg:
            subject_ids = np.loadtxt(os.path.join(outdir, f'{which_data}-MEG_{atlas}_subjids.txt'), dtype=str)
        
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
    
    file_suffix = 'timescales_acf-lag.npy'
    
    # Load rest timescales
    rest_timescales = dict()
    for which_scan in config['rest_scans']:
        print(f"Loading {which_scan}...")
        
        # Determine which modality this scan belongs to and load accordingly
        if '7T' in which_scan:
            # 7T scan
            timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}-7T_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_7T']
            if sum([include_3T, include_7T, include_meg]) > 1 and found_indices_7t is not None:
                timescales = timescales[found_indices_7t]
        else:
            # 3T scan
            timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_3T']
            if sum([include_3T, include_7T, include_meg]) > 1 and found_indices_3t is not None:
                timescales = timescales[found_indices_3t]
        
        print(f"  Shape: {timescales.shape}")
        rest_timescales[which_scan] = timescales
    
    # Load task timescales (3T only)
    task_timescales = dict()
    for which_scan in config['tasks']:
        print(f"Loading {which_scan}...")
        timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_3T']
        if sum([include_3T, include_7T, include_meg]) > 1 and found_indices_3t is not None:
            timescales = timescales[found_indices_3t]
        print(f"  Shape: {timescales.shape}")
        task_timescales[which_scan] = timescales

    # Load movie timescales (7T only)
    movie_timescales = dict()
    if include_7T:
        for which_scan in config['movies']:
            print(f"Loading {which_scan}...")
            timescales = np.load(os.path.join(outdir, f'{which_data.replace("_", "")}-7T_{atlas}_{which_scan.replace("_", "")}_{file_suffix}')) * config['tr_7T']
            if sum([include_3T, include_7T, include_meg]) > 1 and found_indices_7t is not None:
                timescales = timescales[found_indices_7t]
            print(f"  Shape: {timescales.shape}")
            movie_timescales[which_scan] = timescales
    
    # Load MEG timescales
    meg_timescales = dict()
    if include_meg:
        for which_scan in config['meg_scans']:
            print(f"Loading MEG {which_scan}...")
            timescales = np.load(os.path.join(outdir, f'{which_data}-MEG_{atlas}_{which_scan}_{file_suffix}')) * config['tr_meg']
            if sum([include_3T, include_7T, include_meg]) > 1 and found_indices_meg is not None:
                timescales = timescales[found_indices_meg]
            print(f"  Shape: {timescales.shape}")
            meg_timescales[which_scan] = timescales

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
        'meg_timescales': meg_timescales,
        'parc': parc,
        'yeo_systems': yeo_systems,
        'config': config
    }
    
    print(f"\nSuccessfully loaded data:")
    print(f"  - SA axis: {sa_axis.shape}")
    print(f"  - SF axis: {sf_axis.shape}")
    print(f"  - Subjects: {n_subs}")
    print(f"  - Rest conditions: {len(rest_timescales)}")
    if include_3T:
        print(f"  - Task conditions: {len(task_timescales)}")
    if include_7T:
        print(f"  - Movie conditions: {len(movie_timescales)}")
    if include_meg:
        print(f"  - MEG conditions: {len(meg_timescales)}")
    print(f"  - Brain regions: {config['n_parcels']}")
    if df_behavior is not None:
        print(f"  - Behavioral measures: {df_behavior.shape[1]}")
    
    return data_dict
