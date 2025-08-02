# import
import os, sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

def process_data(data_dict, combine_rest=False, combine_movies=False):
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
