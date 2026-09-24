"""Dataset processing parameters.

Consumed by ``timescales_process.ipynb`` and anything else that needs to know how a dataset is
loaded and processed. Kept free of imports so it can be read as pure configuration.
"""

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
        'subject_ids_file': 'subject_ids_lowmotion.txt',
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
        'subject_ids_file': 'subject_ids_lowmotion.txt',
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
        # Longitudinal: 132 subjects at session 1, 73 at session 2, 20 at session 3.
        'sessions':    ['1', '2', '3'],
        'tasks': {
            'MRI3T': ['rest', 'nback'],
        },
        'subject_ids_file': 'subject_ids.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
        # Completeness is judged per (subject, session): a session missing rest or nback is
        # dropped, the subject's other sessions are kept. Per subject, every subject not scanned
        # at every timepoint would be discarded.
        'incomplete_unit':     'session',
    },
    'AOMIC_PIOP2': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    [None],
        'tasks': {
            'MRI3T': ['restingstate', 'stopsignal', 'workingmemory', 'emomatching'],
        },
        'subject_ids_file': 'subject_ids_lowmotion.txt',
        'detrend':             False,
        'nuisance_regression': False,
        'task_regression':     True,
        'remove_incomplete_subjects': True,
    },
    'AOMIC_PIOP1': {
        'modalities':  ['MRI3T'],
        'seg':         'Schaefer4007',
        'space':       'MNI152NLin2009cAsym',
        'sessions':    [None],
        'tasks': {
            # restingstate and faces are multiband at TR 0.75 s; the other four sequential at
            # TR 2.0 s. The per-task TR lives in the YAML's task_params.
            'MRI3T': ['restingstate', 'faces', 'anticipation', 'emomatching', 'gstroop',
                      'workingmemory'],
        },
        'subject_ids_file': 'subject_ids_lowmotion.txt',
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

