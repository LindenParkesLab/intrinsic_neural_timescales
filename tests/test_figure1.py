"""Figure 1's shared code (fig1_super and sfig1): src.notebook grouping and tests, src.plotting panels."""
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from snaplab_tools.plotting.utils import register_custom_colormaps
from snaplab_tools.stats import paired_coupling_test, subject_wise_coupling

from src.data import ScanInfo
from src.notebook import (FIG1_DATASET_CONFIGS, FIG1_GROUP_FLAGS, FIG1_LABELS, coupling_tests, group_conditions,
                          reference_key)
from src.plotting import draw_coupling_boxes, draw_coupling_scatter, draw_int_rows

register_custom_colormaps()      # 'snap_sydnor_sa', which the notebooks get via load_notebook_data
N_REGIONS = 400    # create_ints_dataframe_custom_grouping builds a 400-region frame


def _registry():
    return [ScanInfo(subject_id=s, task=t, space='fsLR', seg='Schaefer4007', modality='MRI3T', tr=0.72, session='01')
            for s in ('1', '2', '3') for t in ('rfMRIREST1LR', 'tfMRIWMLR', 'tfMRIWMRL')]


def _data(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return {str(s): scale * rng.uniform(1, 3, N_REGIONS) for s in _registry()}


CFG = {'average_3t_task_pairs': True, 'custom_order': ['rfMRIREST1LR', 'task-WM']}
PROC = type('P', (), {'scan_registry': _registry()})()
DF_NEURO = pd.DataFrame({'SA_Axis': np.arange(1, N_REGIONS + 1)})


def test_the_configs_are_constants_a_notebook_copies():
    cfg = copy.deepcopy(FIG1_DATASET_CONFIGS)
    cfg['HCP-D']['filter'] = None
    cfg['HCP-YA']['custom_order'].append('task-Tasks')
    assert FIG1_DATASET_CONFIGS['HCP-YA']['custom_order'] == ['task-Rest', 'task-Movies', 'task-WM']
    assert all(set(FIG1_GROUP_FLAGS) <= set(c) for c in FIG1_DATASET_CONFIGS.values())
    assert FIG1_DATASET_CONFIGS['HCP-D']['filter'] == 'bp-0.01-0.08hz' and FIG1_LABELS['task-WM'] == 'Working\nMemory'


def test_group_conditions_orders_columns_and_adds_references_grouped_as_their_condition():
    data = _data()
    controls = {k: v + 1 for k, v in _data(1).items() if 'tfMRIWM' in k}
    out = group_conditions(PROC, data, DF_NEURO, CFG)
    assert list(out['df_ints'].columns) == ['rfMRIREST1LR', 'task-WM'] and out['subject_ids'] == ['1', '2', '3']
    wm = np.mean([[data[str(s)] for s in _registry() if s.task == t] for t in ('tfMRIWMLR', 'tfMRIWMRL')], axis=0)
    np.testing.assert_allclose(out['data_dict_converted']['task-WM'], wm)
    ref = reference_key('task-WM')
    paired = group_conditions(PROC, data, DF_NEURO, CFG, order=[ref, 'task-WM'],
                              controls=controls)
    assert list(paired['df_ints'].columns) == [ref, 'task-WM']
    ctrl = np.mean([[controls[str(s)] for s in _registry() if s.task == t] for t in ('tfMRIWMLR', 'tfMRIWMRL')], axis=0)
    np.testing.assert_allclose(paired['data_dict_converted'][ref], ctrl)
    assert paired['data_dict_centiles'][ref].shape == (3, 10)


def test_coupling_tests_pair_each_condition_with_its_reference():
    conv = group_conditions(PROC, _data(), DF_NEURO, CFG)['data_dict_converted']
    sa = DF_NEURO['SA_Axis'].to_numpy()
    coupling, rows = coupling_tests(conv, {'task-WM': 'rfMRIREST1LR'}, sa)
    np.testing.assert_array_equal(coupling['task-WM'], subject_wise_coupling(conv['task-WM'], sa, method='spearman'))
    expected = paired_coupling_test(coupling['rfMRIREST1LR'], coupling['task-WM'])
    assert (rows[0]['rho_rest'], rows[0]['rho_condition'], rows[0]['t'], rows[0]['p'], rows[0]['n']) == expected
    assert rows[0]['reference'] == 'rfMRIREST1LR'


def test_the_panels_draw():
    g = group_conditions(PROC, _data(), DF_NEURO, CFG)
    fig, axes = plt.subplots(2, 2)
    tt = draw_int_rows(axes[0, 0], axes[1, 0], g['df_ints'], g['data_dict_centiles'], {'task-WM': 'rfMRIREST1LR'},
                       DF_NEURO['SA_Axis'], {'task-WM': 'WM'}, 'X', ylabels=('a', 'b'))
    assert list(tt.index) == ['WM']
    coupling, _ = coupling_tests(g['data_dict_converted'], {'task-WM': 'rfMRIREST1LR'}, DF_NEURO['SA_Axis'].to_numpy())
    draw_coupling_boxes(axes[0, 1], [dict(title='X', columns=['rfMRIREST1LR', 'task-WM'],
                                          reference_of={'task-WM': 'rfMRIREST1LR'}, coupling=coupling)],
                        {('X', 'task-WM'): '*'}, {}, ('r', 'b'), ('Rest', 'Task'))
    assert len(axes[0, 1].patches) >= 2
    draw_coupling_scatter(axes[1, 1], coupling['rfMRIREST1LR'], coupling['task-WM'], 'Rest', 'WM', ('r', 'b'), 'X')
    assert axes[1, 1].get_title() == 'X'
    plt.close(fig)
