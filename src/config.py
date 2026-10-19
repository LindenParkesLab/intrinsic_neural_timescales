"""Dataset processing parameters.

Consumed by ``timescales_process.ipynb`` and anything else that needs to know how a dataset is
loaded and processed. Kept free of imports so it can be read as pure configuration.
"""

DATASET_PARAMS = {
    'HCP-YA': {
        'modalities':  ['MRI3T'],
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
        },
        'subject_ids_file': 'subject_ids.txt',
        'rest_variant':        'clean',
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
        # Here, a session missing rest or nback is dropped, the subject's other sessions are kept. 
        'incomplete_unit':     'session',
    },
}


def get_dataset_params(dataset, seg=None, rest_variant=None, modalities=None):
    """Return the processing parameters for ``dataset``, optionally overriding ``seg``, ``rest_variant``
    and ``modalities``.

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
    rest_variant : str, optional
        Source of the rest runs (e.g. 'noFIX'). If provided, it overrides the dataset's
        default; if None, the default is kept, and a dataset without the key reads
        'clean'. A variant the dataset's YAML does not list raises at registration.
    modalities : list of str, optional
        Modalities to register (e.g. ``['MRI3T', 'MRI7T']`` for HCP-YA's multi registry). Each must
        have a ``tasks`` entry. If None, the default in DATASET_PARAMS is used.

    Returns
    -------
    dict
        A shallow copy of ``DATASET_PARAMS[dataset]`` with the overrides applied; ``rest_variant``
        is always present. The copy prevents callers from mutating the module-level defaults.
    """
    if dataset not in DATASET_PARAMS:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Available: {sorted(DATASET_PARAMS)}.")
    params = dict(DATASET_PARAMS[dataset])
    if seg is not None:
        params['seg'] = seg
    params.setdefault('rest_variant', 'clean')
    if rest_variant is not None:
        params['rest_variant'] = rest_variant
    if modalities is not None:
        unknown = [m for m in modalities if m not in params['tasks']]
        if unknown:
            raise ValueError(f"{dataset} has no tasks for modalities {unknown}; it has {sorted(params['tasks'])}.")
        params['modalities'] = list(modalities)
    return params

