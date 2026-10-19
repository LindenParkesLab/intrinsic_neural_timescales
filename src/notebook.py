"""Assemble everything a figure notebook needs into one dict.

"""

import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from snaplab_tools import stats as snap_stats

from src.data import REST_VARIANT_DEFAULT, load_dataset_config
from src.timescales import (INT_METHOD, Timescales, average_data_dict_converted, condition_runs, convert_data_dict,
                            create_ints_dataframe_custom_grouping, filter_data_dict, parse_scan_string,
                            rest_control_output_type, rest_variant_output_type, rest_window_covered,
                            rest_window_matches, rest_window_output_type, rest_window_token)
from src.supporting_data import load_myelin_map, load_supporting_data
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


def _resolve_rest_match(dataset, rest_match):
    """Which matched task's windows to load. ``None`` picks the dataset's only match (HCP-YA: WM)
    and raises when it has several (HCP-D), so a notebook cannot load CARIT-length rest for an
    EMOTION analysis by default."""
    matches = rest_window_matches(dataset)
    if not matches:
        raise ValueError(
            f"rest_length='matched', but {dataset} has no length-matched rest (see "
            f"src.timescales.REST_WINDOW_MATCH). Keeping full-length rest silently would mix the "
            f"two, so this raises instead.")
    if rest_match is None:
        if len(matches) > 1:
            raise ValueError(
                f"{dataset}'s rest is length-matched to several tasks {list(matches)}; pass rest_match "
                "(src.timescales.rest_window_match_for(dataset, task) gives it for the task analysed).")
        return matches[0]
    if rest_match not in matches:
        raise ValueError(f'{dataset} has no rest windows matched to {rest_match!r}; it has {list(matches)}')
    return rest_match


def _substitute_rest_windows(processor, data_dict, output_type, rest_match):
    """Swap each covered rest scan's full-length INT for its window matched to ``rest_match``.

    Done before ``filter_data_dict``: a covered rest scan whose window is missing becomes None and
    is dropped by that function's existing check, exactly as a missing residual drops a subject.
    Uncovered rest scans (e.g. HCP-YA 7T) keep their full-length INT, and the count says so.
    """
    dataset = processor.dataset
    rest_keys = [k for k in data_dict if parse_scan_string(k).get('condition') == 'rest']
    covered = [k for k in rest_keys
               if rest_window_covered(dataset, parse_scan_string(k).get('modality', ''), rest_match)]
    if not covered:
        raise ValueError(
            f"rest_length='matched', but no {dataset} rest scan has windows matched to "
            f"{rest_match!r} (see src.timescales.REST_WINDOW_MATCH). Keeping full-length rest "
            f"silently would mix the two, so this raises instead.")
    processor.get_precomputed_outputs(output_type=rest_window_output_type(output_type, rest_match))
    windows = processor.data_dict
    missing = [k for k in covered if windows.get(k) is None]
    for k in covered:
        data_dict[k] = windows.get(k)
    msg = (f"  rest_length='matched' ({rest_window_token(rest_match)}): "
           f"{len(covered) - len(missing)} rest scans use length-matched windows")
    if missing:
        msg += f"; {len(missing)} have no window (their subjects are dropped)"
    if len(rest_keys) > len(covered):
        msg += f"; {len(rest_keys) - len(covered)} uncovered rest scans keep full length"
    print(msg)
    return data_dict


def _load_timescale_outputs(processor, use_7t, output_type, use_resids, rest_length='full', rest_match=None):
    """Load precomputed timescale (and tSNR) outputs for a dataset into dicts.

    Loads the scan registry (HCP-YA's 3T + 7T multi registry when use_7t), then the
    tSNR and timescale outputs, filtering out subjects with missing scans. When
    use_resids is True, task residual timescales are loaded and substituted in.

    Parameters
    ----------
    processor : Timescales
        Processor whose registry/outputs are loaded.
    use_7t : bool
        Use the HCP-YA multi-session (incl. 7T) registry.
    output_type : str
        Timescale output type to load.
    use_resids : bool
        Also load residual timescales and merge them in.
    rest_length : {'full', 'matched'}
        See :func:`load_notebook_data`.
    rest_match : str or None
        See :func:`load_notebook_data`.

    Returns
    -------
    tuple
        (data_dict, data_dict_tsnr, data_dict_residuals); the last is None unless
        use_resids.
    """
    dataset = processor.dataset

    if dataset == 'HCP-YA' and use_7t:
        processor.load_scan_registry(f'scan_registry_{dataset}_multi.csv')
    else:
        processor.load_scan_registry(f'scan_registry_{dataset}.csv')

    if dataset == 'HCP-YA':
        processor.match_registry_to_behavior()

    processor.get_precomputed_outputs(output_type='metric_tsnr')
    data_dict_tsnr = filter_data_dict(processor.data_dict.copy(), remove_subject_level=True)

    output_type_resid = output_type + '_resids'

    processor.get_precomputed_outputs(output_type=output_type)
    data_dict = processor.data_dict.copy()

    if rest_length == 'matched':
        data_dict = _substitute_rest_windows(processor, data_dict, output_type,
                                             _resolve_rest_match(dataset, rest_match))
    elif rest_length != 'full':
        raise ValueError(f"rest_length must be 'full' or 'matched'; got {rest_length!r}")
    elif rest_match is not None:
        raise ValueError(f"rest_match={rest_match!r} is only meaningful with rest_length='matched'")

    if use_resids:
        processor.get_precomputed_outputs(output_type=output_type_resid)
        data_dict_residuals = processor.data_dict.copy()
        data_dict, data_dict_residuals = filter_data_dict(data_dict, data_dict_residuals)
    else:
        data_dict = filter_data_dict(data_dict, remove_subject_level=True)
        data_dict_residuals = None

    return data_dict, data_dict_tsnr, data_dict_residuals


def load_notebook_data(dataset='HCP-YA', use_resids=True, use_7t=False,
                       output_type='timescale_auc_zero_crossing',
                       load_myelin=False, n_regions=400, gene_panel='gao',
                       rest_length='full', rest_match=None):
    """Load all data a notebook needs for a dataset into a single dict.

    Sets up the Timescales processor, loads supporting region maps (axes, gene /
    cell annotations) and the timescale and tSNR data dicts. For HCP-YA and HCP-D with
    ``load_myelin``, the dataset's own group-mean T1w/T2w map over its registry subjects is
    added to ``supporting_data['df_neuro']`` as ``'<DATASET>_Myelin'`` (e.g. ``'HCPD_Myelin'``).

    Parameters
    ----------
    dataset : str
        Dataset name (key into DATASET_PARAMS).
    use_resids : bool
        Load residual (task-regressed) timescales.
    use_7t : bool
        Use the HCP-YA multi-session (incl. 7T) registry.
    output_type : str
        Timescale output type to load, e.g. ``'timescale_auc_zero_crossing'``
    load_myelin : bool
        Add the group-mean myelin map (HCP-YA and HCP-D only).
    n_regions : int
        Parcellation resolution.
    gene_panel : str
        Brain-gene list the gene PCA is fit on ('gao', 'dear', ...). Surfaces as
        supporting_data['gene_panel']; tag anything derived from the gene PCA with it.
    rest_length : {'full', 'matched'}
        Rest INTs from full-length runs, or from length-matched windows (``_restlen-<task>``) for
        the dataset x modality pairs in ``src.timescales.REST_WINDOW_MATCH``. Raises for a dataset
        with no coverage rather than silently keeping full-length rest.
    rest_match : str or None
        Which matched task's windows to load under ``'matched'`` (e.g. ``'CARIT'``; get it with
        ``src.timescales.rest_window_match_for(dataset, task)``). ``None`` is resolved only for a
        dataset with a single match (HCP-YA: WM) and raises for one with several (HCP-D).

    Returns
    -------
    dict
        timescales_processor, results_dir, sydnor_sa_cmap, data_dict, data_dict_tsnr and
        supporting_data.
    """

    print('Loading data for notebooks...')

    timescales_processor, results_dir = _setup_processor(dataset)

    sydnor_sa_cmap = register_custom_colormaps()['snap_sydnor_sa']
    supporting_data = load_supporting_data(
        os.path.join(timescales_processor.output_dir, 'data', 'supporting_data'),
        n_regions=n_regions, gene_panel=gene_panel)

    data_dict, data_dict_tsnr, _ = _load_timescale_outputs(
        timescales_processor, use_7t, output_type, use_resids,
        rest_length=rest_length, rest_match=rest_match)

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

    if load_myelin:
        if dataset not in ('HCP-YA', 'HCP-D'):
            raise ValueError(f'load_myelin: only HCP-YA and HCP-D have T1w/T2w maps; got {dataset!r}')
        column = f"{dataset.replace('-', '')}_Myelin"
        supporting_data['df_neuro'][column] = load_myelin_map(
            timescales_processor.derivdata_dir, subject_ids, n_regions)

    return notebook_data


# ==============================================================================
# Matched rest controls: a task INT against rest put through the task's processing
# ==============================================================================
# A task INT is normally contrasted with the dataset's own rest, full length or cut to the task's run
# length -- both are read by load_notebook_data (rest_length='full' | 'matched'). The alternative is each
# task run's matched rest control (restctrl), read here.

#: Rest sources a control can be cut from: the default, or a variant the dataset's YAML lists for some
#: modality (HCP-YA 3T: 'noFIX', rest without ICA-FIX, denoised as the task is).
REST_SOURCES = (REST_VARIANT_DEFAULT, 'noFIX')


def control_output_type(dataset, modality, rest_source=REST_VARIANT_DEFAULT, filter=None):
    """The output_type a modality's matched rest controls are read from, e.g.
    ``timescale_auc_zero_crossing_noFIX_restctrl``. A modality whose YAML does not list ``rest_source``
    (HCP-YA 7T) reads the default source, as its task runs were denoised."""
    if rest_source not in REST_SOURCES:
        raise ValueError(f'rest_source must be one of {list(REST_SOURCES)}; got {rest_source!r}')
    listed = load_dataset_config(dataset).modalities[modality].rest_variants
    base = rest_variant_output_type(f'timescale_{INT_METHOD}', rest_source if rest_source in listed
                                    else REST_VARIANT_DEFAULT)
    return rest_control_output_type(f'{base}_{filter}' if filter else base, 'full')


def control_tag(rest_source=REST_VARIANT_DEFAULT):
    """File-name tag of a figure drawn against matched controls: ``_cmp-restctrl``, or
    ``_rest-<source>_cmp-restctrl`` for a non-default rest source."""
    return ('' if rest_source == REST_VARIANT_DEFAULT else f'_rest-{rest_source}') + '_cmp-restctrl'


def load_rest_controls(processor, data_dict, conditions, rest_source=REST_VARIANT_DEFAULT, filter=None,
                       verbose=True):
    """Each condition's matched rest controls, keyed like the task scans of ``data_dict``.

    Parameters
    ----------
    processor : Timescales
        With the registry the notebook loaded (and pruned).
    data_dict : dict
        The notebook's scan_key -> INT dict.
    conditions : sequence of str
        Task conditions as the notebook's grouping names them (e.g. ``'task-WM'``, ``'tfMRICARITPA'``).
    rest_source, filter
        Which controls: see :func:`control_output_type`.

    Returns
    -------
    data_dict : dict
        The input without every subject that lacks a control for any condition -- the rule
        ``filter_data_dict`` applies to a missing residual.
    controls : dict
        scan_key -> control INT for each task run of ``conditions``, pruned to the same subjects. Group it
        as the task scans are grouped to get each condition's control column.
    dropped : dict
        condition -> sorted subjects that lacked a control.

    Raises
    ------
    FileNotFoundError
        When a modality has no control cached at all (not built, or the wrong ``rest_source`` / ``filter``).
    """
    parsed = {k: parse_scan_string(k) for k in data_dict}
    task_runs = sorted({kv['task'] for kv in parsed.values() if kv.get('condition') == 'task'})
    cache, controls, dropped = {}, {}, {}
    for condition in conditions:
        runs = condition_runs(condition, task_runs)
        missing = set()
        for k, kv in parsed.items():
            if kv.get('task') not in runs:
                continue
            output_type = control_output_type(processor.dataset, kv['modality'], rest_source, filter)
            if output_type not in cache:
                processor.get_precomputed_outputs(output_type=output_type)
                cache[output_type] = dict(processor.data_dict)
            controls[k] = cache[output_type].get(k)
            if controls[k] is None:
                missing.add(kv['sub'])
        dropped[condition] = sorted(missing)
        if len(missing) == len({parsed[k]['sub'] for k in parsed if parsed[k].get('task') in runs}):
            raise FileNotFoundError(f'{processor.dataset} {condition}: no matched rest control is cached '
                                    f'(rest_source={rest_source!r}, filter={filter!r}); build them in '
                                    'timescales_process.ipynb')
    gone = set().union(*dropped.values())
    data_dict = {k: v for k, v in data_dict.items() if parsed[k]['sub'] not in gone}
    controls = {k: v for k, v in controls.items() if parsed[k]['sub'] not in gone}
    if verbose:
        print(f'  matched rest controls ({rest_source}): '
              + '; '.join(f'{c}: {len(s)} subjects dropped' for c, s in dropped.items()))
    return data_dict, controls, dropped


def reference_key(condition):
    """Column key of a condition's own control column."""
    return f'ref:{condition}'


def reference_layout(custom_order, controls=False):
    """Column order and each condition's reference.

    Without controls the order is ``custom_order`` and every condition's reference is its first column, the
    shared rest. With controls each condition is preceded by its own control column, ``reference_key``.
    """
    rest, conditions = custom_order[0], list(custom_order[1:])
    if not controls:
        return list(custom_order), {c: rest for c in conditions}
    return ([k for c in conditions for k in (reference_key(c), c)],
            {c: reference_key(c) for c in conditions})


def ttest_vs_references(df, reference_of, alternative='two-sided', correction='fdr_bh'):
    """Paired t-tests of each condition column against its own reference column, corrected across conditions.

    With one shared reference this is exactly ``paired_ttest_vs_reference(df, reference=<it>)``. With
    per-condition references each pair is tested alone and the p-values are corrected together, the
    same correction over the same number of comparisons.
    """
    references = set(reference_of.values())
    if len(references) == 1:
        return snap_stats.paired_ttest_vs_reference(df, reference=references.pop(), alternative=alternative,
                                                    correction=correction)
    out = pd.concat([snap_stats.paired_ttest_vs_reference(df[[ref, c]], reference=ref, alternative=alternative,
                                                          correction=None)
                     for c, ref in reference_of.items()])
    valid = out['p'].notna()
    out['p_corr'] = out['p']
    if correction is not None and valid.any():
        out.loc[valid, 'p_corr'] = multipletests(out.loc[valid, 'p'].to_numpy(), method=correction)[1]
    out['sig'] = out['p_corr'].apply(snap_stats.significance_stars)
    return out


# ==============================================================================
# Figure 1 (fig1_super and sfig1): configuration, loading and the coupling test
# ==============================================================================

#: Scan-grouping flags a Figure 1 dataset config may set (src.timescales._apply_scan_groupings).
FIG1_GROUP_FLAGS = [
    'average_3t_rest', 'average_7t_rest', 'average_all_rest', 'average_7t_movies',
    'average_3t_task_pairs', 'average_3t_tasks',
]

#: Display labels of Figure 1's condition keys.
FIG1_LABELS = {
    'rest':             'Rest',
    'task-Rest':        'Rest',
    'task-REST_3T':     'Rest',
    'task-REST_7T':     'Rest (7T)',
    'reversallearning': 'Reversal\nLearning',
    'nback':            'Working\nMemory',
    'task-Movies':      'Movies',
    'task-Tasks':       'Tasks',
    'task-WM':          'Working\nMemory',
    'tfMRIEMOTIONPA':   'Emotion',
    'tfMRICARITPA':     'Carit',
    'tfMRIGUESSINGPA':  'Guessing',
}

#: Y-axis label of Figure 1's rows: '{}' is 'mean' (row A) or 'subjects' (row B).
FIG1_AXIS_LABEL = 'INTs (s, {})'

# Per-dataset loading configuration of Figure 1.
FIG1_DATASET_CONFIGS = {
    'HCP-YA': {
        'use_7t':         True,
        'use_resids':     True,
        'retain_scans':   None,
        'average_3t_rest':       False,
        'average_7t_rest':       False,
        'average_all_rest':      True,
        'average_7t_movies':     True,
        'average_3t_task_pairs': True,
        'average_3t_tasks':      False,
        'custom_order':   ['task-Rest', 'task-Movies', 'task-WM'],
        'title':          'HCP-YA',
    },
    'WAND': {
        'use_7t':         False,
        'use_resids':     False,
        'retain_scans':   None,
        'average_3t_rest':       True,
        'average_7t_rest':       True,
        'average_all_rest':      False,
        'average_7t_movies':     False,
        'average_3t_task_pairs': True,
        'average_3t_tasks':      False,
        'custom_order':   ['rest', 'reversallearning'],
        'title':          'WAND',
    },
    'HCP-D': {
        'use_7t':         False,
        'use_resids':     True,
        'filter':         'bp-0.01-0.08hz',
        'retain_scans':   None,
        'average_3t_rest':       True,
        'average_7t_rest':       False,
        'average_all_rest':      False,
        'average_7t_movies':     False,
        'average_3t_task_pairs': False,
        'average_3t_tasks':      False,
        'custom_order':   ['task-REST_3T', 'tfMRIEMOTIONPA', 'tfMRICARITPA', 'tfMRIGUESSINGPA'],
        'title':          'HCP-D',
    },
    'Penn_LEAD': {
        'use_7t':         False,
        'use_resids':     True,
        'retain_scans':   None,
        'average_3t_rest':       False,
        'average_7t_rest':       False,
        'average_all_rest':      False,
        'average_7t_movies':     False,
        'average_3t_task_pairs': False,
        'average_3t_tasks':      False,
        'custom_order':   ['rest', 'nback'],
        'title':          'Penn LEAD',
    },
}


def load_condition_data(dataset, cfg, penn_session1_only=True, verbose=True):
    """Load one Figure 1 dataset: its INTs at ``cfg``'s filter, pruned as the figure needs.

    Penn LEAD keeps session 1 only when ``penn_session1_only``; ``cfg['retain_scans']`` keeps the scans whose
    key contains one of its entries. Both are pruned from the data dict and the processor's registry, so
    everything downstream sees the same scans.

    Returns
    -------
    dict
        ``processor``, ``data_dict``, ``df_neuro`` and ``output_type`` (``timescale_auc_zero_crossing``, plus
        ``cfg['filter']`` when set).
    """
    output_type = f'timescale_{INT_METHOD}' + (f"_{cfg['filter']}" if cfg.get('filter') else '')
    notebook_data = load_notebook_data(
        dataset=dataset,
        use_7t=cfg['use_7t'],
        output_type=output_type,
        use_resids=cfg['use_resids'],
    )
    processor = notebook_data['timescales_processor']
    data_dict = notebook_data['data_dict']
    df_neuro = notebook_data['supporting_data']['df_neuro']

    if penn_session1_only and dataset == 'Penn_LEAD':
        keep_scans = [s for s in processor.scan_registry if str(s.session) == '1']
        keep_keys = {str(s) for s in keep_scans}
        data_dict = {k: v for k, v in data_dict.items() if k in keep_keys}
        processor.scan_registry = keep_scans
        if verbose:
            print(f'  Penn LEAD: session 1 only -> {len(keep_scans)} scans, '
                  f'{len({s.subject_id for s in keep_scans})} subjects')

    retain_scans = cfg.get('retain_scans')
    if retain_scans:
        keep_scans = [s for s in processor.scan_registry if any(sub in str(s) for sub in retain_scans)]
        keep_keys = {str(s) for s in keep_scans}
        data_dict = {k: v for k, v in data_dict.items() if k in keep_keys}
        processor.scan_registry = keep_scans
        if verbose:
            print(f'  retain_scans {retain_scans} -> {len(keep_scans)} scans, '
                  f'{len({s.subject_id for s in keep_scans})} subjects')
    return dict(processor=processor, data_dict=data_dict, df_neuro=df_neuro, output_type=output_type)


def group_conditions(processor, data_dict, df_neuro, cfg, brain_map='SA_Axis', order=None, controls=None):
    """Figure 1's three groupings of one dataset, with ``cfg``'s scan-grouping flags.

    Parameters
    ----------
    order : list or None
        Column order; ``cfg['custom_order']`` when None. Columns absent from it are dropped.
    controls : dict or None
        Matched rest controls from :func:`load_rest_controls`. When given, each condition's control column
        (``reference_key(condition)``) is added, grouped exactly as the condition's own column is.

    Returns
    -------
    dict
        ``df_ints`` (regions x columns, group means), ``data_dict_centiles`` (column -> subjects x S-A
        deciles), ``data_dict_converted`` (column -> subjects x regions) and ``subject_ids``.
    """
    group_kwargs = {flag: cfg[flag] for flag in FIG1_GROUP_FLAGS if flag in cfg}
    order = cfg['custom_order'] if order is None else order
    registry = processor.scan_registry

    def ints(d):
        return create_ints_dataframe_custom_grouping(d, statistic='mean', **group_kwargs)

    def centiles(d):
        return average_data_dict_converted(
            convert_data_dict(d, registry, brain_map=df_neuro[brain_map], verbose=False), **group_kwargs)

    def converted(d):
        return average_data_dict_converted(convert_data_dict(d, registry, verbose=False), **group_kwargs)

    # sort_by_median only matters when there is no custom_order (custom_order reindexes below).
    df_ints = create_ints_dataframe_custom_grouping(
        data_dict, sort_by_median=not bool(cfg['custom_order']), statistic='mean', **group_kwargs)
    data_dict_centiles = centiles(data_dict)
    data_dict_converted = converted(data_dict)
    if controls is not None:
        ctrl_ints, ctrl_centiles, ctrl_converted = ints(controls), centiles(controls), converted(controls)
        for condition in cfg['custom_order'][1:]:
            df_ints[reference_key(condition)] = ctrl_ints[condition]
            data_dict_centiles[reference_key(condition)] = ctrl_centiles[condition]
            data_dict_converted[reference_key(condition)] = ctrl_converted[condition]
    if order:
        df_ints = df_ints[order]
        data_dict_centiles = {k: data_dict_centiles[k] for k in order if k in data_dict_centiles}
        data_dict_converted = {k: data_dict_converted[k] for k in order if k in data_dict_converted}
    subject_ids = list(dict.fromkeys(scan.subject_id for scan in registry))
    return dict(df_ints=df_ints, data_dict_centiles=data_dict_centiles,
                data_dict_converted=data_dict_converted, subject_ids=subject_ids)


def coupling_tests(data_dict_converted, reference_of, sa_axis, method='spearman'):
    """Each subject's coupling of their regional INT map with the S-A axis, and the decoupling test.

    Parameters
    ----------
    data_dict_converted : dict
        Column -> (n_subjects, n_regions) INTs.
    reference_of : dict
        Condition -> the column it is tested against.
    sa_axis : (n_regions,) ndarray

    Returns
    -------
    coupling : dict
        Column -> (n_subjects,) per-subject coupling (``subject_wise_coupling``).
    rows : list of dict
        One per condition: ``condition``, ``reference``, ``n``, ``rho_rest`` (mean coupling of the reference),
        ``rho_condition``, ``t``, ``p`` -- ``paired_coupling_test`` of the reference against the condition
        (Fisher-z, paired t, its default one-sided alternative). Uncorrected: the caller chooses the FDR family.
    """
    coupling = {c: snap_stats.subject_wise_coupling(v, sa_axis, method=method)
                for c, v in data_dict_converted.items()}
    rows = []
    for cond, ref in reference_of.items():
        rr, rc, t, p, n = snap_stats.paired_coupling_test(coupling[ref], coupling[cond])
        rows.append(dict(condition=cond, reference=ref, n=n, rho_rest=rr, rho_condition=rc, t=t, p=p))
    return coupling, rows


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

