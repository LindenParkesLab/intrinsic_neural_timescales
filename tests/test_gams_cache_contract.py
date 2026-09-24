"""The fig3 -> fig4 cache contract.

fig3 writes one GAM export per (fit_recipe x absolute x demean) variant and fig4 reads it back by
resolving the identical path from the identical inputs. Nothing checks that agreement at runtime
beyond the file being found, so a silent change to a tag -- a renamed flag, a reordered component,
a lost separator -- would make fig4 either miss the file or, worse, read the wrong variant. These
tests pin the mapping.
"""
import itertools
import os

import pytest

from src.gams import GAM_RECIPES, gam_variant_tags, gam_cache_path

RECIPES = sorted(GAM_RECIPES)
FLAGS = list(itertools.product(RECIPES, [True, False], [True, False]))


class TestVariantTags:
    def test_rejects_unknown_recipe(self):
        with pytest.raises(ValueError):
            gam_variant_tags('not-a-recipe', absolute=True, demean=True)

    @pytest.mark.parametrize('recipe', RECIPES)
    def test_carries_recipe_fit_parameters(self, recipe):
        tags = gam_variant_tags(recipe, absolute=True, demean=True)
        assert tags['fit_recipe'] == recipe
        assert tags['n_splines'] == GAM_RECIPES[recipe]['n_splines']
        assert tags['lam'] == GAM_RECIPES[recipe]['lam']

    @pytest.mark.parametrize('absolute', [True, False])
    @pytest.mark.parametrize('demean', [True, False])
    def test_tag_values_are_the_documented_ones(self, absolute, demean):
        tags = gam_variant_tags('historical', absolute, demean)
        assert tags['absolute_tag'] == ('_abs' if absolute else '_signed')
        assert tags['demean_tag'] == ('_demean' if demean else '_nodemean')

    def test_historical_recipe_has_no_tag(self):
        # The historical recipe is the unmarked default; only 'gcv' adds a token.
        assert gam_variant_tags('historical', True, True)['recipe_tag'] == ''
        assert gam_variant_tags('gcv', True, True)['recipe_tag'] != ''

    @pytest.mark.parametrize('recipe,absolute,demean', FLAGS)
    def test_variant_tag_is_absolute_then_demean(self, recipe, absolute, demean):
        # Both notebooks build panel filenames as ..._{variant_tag}; the order is part of the
        # contract, not an implementation detail.
        tags = gam_variant_tags(recipe, absolute, demean)
        assert tags['variant_tag'] == tags['absolute_tag'] + tags['demean_tag']


class TestCachePath:
    @pytest.mark.parametrize('recipe,absolute,demean', FLAGS)
    def test_every_variant_gets_a_distinct_file(self, recipe, absolute, demean):
        paths = {gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA',
                                gam_variant_tags(r, a, d))
                 for r, a, d in FLAGS}
        assert len(paths) == len(FLAGS), 'two variants collide on one cache file'

    def test_path_components_all_appear(self):
        tags = gam_variant_tags('gcv', absolute=False, demean=False)
        p = gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA', tags)
        assert os.path.dirname(p) == '/out'
        name = os.path.basename(p)
        for part in ('HCP-D', '400', 'tfMRICARITPA', tags['recipe_tag'],
                     tags['absolute_tag'], tags['demean_tag']):
            assert part in name, f'{part!r} missing from {name!r}'
        assert name.endswith('_regional_gam_results.npy')

    def test_is_deterministic(self):
        tags = gam_variant_tags('historical', True, True)
        args = ('/out', 'HCP-D', 400, 'tfMRICARITPA', tags)
        assert gam_cache_path(*args) == gam_cache_path(*args)

    def test_dataset_and_resolution_separate_files(self):
        tags = gam_variant_tags('historical', True, True)
        a = gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA', tags)
        b = gam_cache_path('/out', 'PNC', 400, 'tfMRICARITPA', tags)
        c = gam_cache_path('/out', 'HCP-D', 200, 'tfMRICARITPA', tags)
        assert len({a, b, c}) == 3


class TestRestLength:
    """Length-matched rest changes the rest side of every deflection, so it is part of the run's
    identity -- and so is which task the rest is matched to. 'full' must leave every existing cache
    and panel name untouched."""

    def test_full_is_the_default_and_changes_nothing(self):
        tags = gam_variant_tags('gcv', True, True)
        assert tags['rest_length'] == 'full' and tags['rest_match'] is None
        assert tags['rest_tag'] == '' and tags['variant_tag'] == '_abs_demean'
        assert (gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA', tags)
                == '/out/HCP-D_400_tfMRICARITPA_ns5-gcv_abs_demean_regional_gam_results.npy')

    def test_matched_tags_the_path_and_the_panel_names_with_its_task(self):
        tags = gam_variant_tags('gcv', True, True, rest_length='matched', rest_match='CARIT')
        assert tags['rest_tag'] == '_restlen-CARIT'
        assert tags['variant_tag'] == '_abs_demean_restlen-CARIT'
        assert gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA', tags).endswith(
            '_abs_demean_restlen-CARIT_regional_gam_results.npy')

    @pytest.mark.parametrize('recipe,absolute,demean', FLAGS)
    def test_full_and_each_match_never_share_a_file(self, recipe, absolute, demean):
        modes = [('full', None)] + [('matched', m) for m in ('CARIT', 'EMOTION', 'GUESSING')]
        paths = {gam_cache_path('/out', 'HCP-D', 400, 'tfMRICARITPA',
                                gam_variant_tags(r, a, d, rest_length=rl, rest_match=rm))
                 for r, a, d in FLAGS for rl, rm in modes}
        assert len(paths) == len(modes) * len(FLAGS)

    def test_rejects_unknown_rest_length(self):
        with pytest.raises(ValueError, match='rest_length'):
            gam_variant_tags('gcv', True, True, rest_length='half')

    def test_matched_needs_its_task(self):
        with pytest.raises(ValueError, match='rest_match'):
            gam_variant_tags('gcv', True, True, rest_length='matched')

    def test_full_refuses_a_task(self):
        with pytest.raises(ValueError, match='rest_match'):
            gam_variant_tags('gcv', True, True, rest_match='CARIT')
