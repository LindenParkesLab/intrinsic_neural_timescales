"""Smoke tests for the region-level supporting maps and the dataset config.

These cover the contract every figure notebook depends on -- that load_supporting_data returns
the documented keys, and that df_neuro is region-indexed at the requested resolution. They read
real files from data/supporting_data and skip if it is absent, so they are safe on a machine
without the data mounted.
"""
import os

import numpy as np
import pytest

from src.config import DATASET_PARAMS, get_dataset_params
from src.supporting_data import load_supporting_data

SUPPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'supporting_data')
pytestmark = pytest.mark.skipif(not os.path.isdir(SUPPORT_DIR),
                                reason='data/supporting_data not available')

DOCUMENTED_KEYS = {
    'parc', 'n_regions', 'distance_matrix', 'distance_matrix_norm', 'yeo_systems',
    'n_systems', 'gene_expression_nanmask', 'df_neuro', 'gene_expression_brain', 'gene_panel',
}


@pytest.fixture(scope='module')
def supporting():
    return load_supporting_data(SUPPORT_DIR, n_regions=400)


class TestLoadSupportingData:
    def test_returns_documented_keys(self, supporting):
        assert set(supporting) == DOCUMENTED_KEYS

    def test_region_indexed_at_requested_resolution(self, supporting):
        assert supporting['n_regions'] == 400
        assert len(supporting['df_neuro']) == 400
        assert supporting['distance_matrix'].shape == (400, 400)

    def test_core_axes_present_and_ranked(self, supporting):
        # SA_Axis is stored as ranks 1..n_regions -- several analyses rely on that.
        sa = supporting['df_neuro']['SA_Axis'].values
        assert np.nanmin(sa) == 1 and np.nanmax(sa) == 400
        assert 'SF_Axis' in supporting['df_neuro'].columns

    def test_no_pet_columns_remain(self, supporting):
        # PET was removed from the project; a stray PET_* column means it crept back in.
        assert [c for c in supporting['df_neuro'].columns if c.startswith('PET')] == []

    def test_gene_panel_is_echoed_back(self):
        out = load_supporting_data(SUPPORT_DIR, n_regions=400, gene_panel='gao')
        assert out['gene_panel'] == 'gao'

    def test_distance_matrix_is_symmetric_zero_diagonal(self, supporting):
        d = supporting['distance_matrix']
        assert np.allclose(d, d.T, equal_nan=True)
        assert np.allclose(np.diag(d), 0)


class TestDatasetConfig:
    def test_every_dataset_has_the_required_processing_flags(self):
        required = {'modalities', 'seg', 'space', 'sessions', 'tasks',
                    'subject_ids_file', 'detrend', 'nuisance_regression', 'task_regression'}
        for name, cfg in DATASET_PARAMS.items():
            assert required <= set(cfg), f'{name} missing {required - set(cfg)}'

    def test_tasks_are_declared_for_every_modality(self):
        for name, cfg in DATASET_PARAMS.items():
            for modality in cfg['modalities']:
                assert modality in cfg['tasks'], f'{name}: no tasks for {modality}'

    def test_get_dataset_params_seg_override(self):
        default = get_dataset_params('HCP-YA')
        override = get_dataset_params('HCP-YA', seg='Schaefer10007')
        assert default['seg'] == 'Schaefer4007'
        assert override['seg'] == 'Schaefer10007'

    def test_get_dataset_params_does_not_mutate_the_module_constant(self):
        # get_dataset_params hands back a config the caller may edit; DATASET_PARAMS must not
        # change underneath other datasets.
        before = DATASET_PARAMS['HCP-YA']['seg']
        get_dataset_params('HCP-YA', seg='Schaefer10007')
        assert DATASET_PARAMS['HCP-YA']['seg'] == before
