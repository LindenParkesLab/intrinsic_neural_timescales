"""The group-level INT table: ``create_ints_dataframe_custom_grouping`` and its statistic.

The table is built in TWO aggregation steps -- subjects collapse to one value per region, then
runs collapse into a condition (``_apply_scan_groupings``) -- and ``statistic`` drives both, so a
'median' table is never a median of means. ``'mean'`` is the default and the historical behaviour;
every existing caller passes no statistic and must be unaffected.

Stubs only: no data mount, no NAS reads.
"""
import inspect

import numpy as np
import pytest

from src.timescales import _apply_scan_groupings, create_ints_dataframe_custom_grouping

N_REGIONS = 2


def _key(sub, task):
    return f'sub-{sub}_task-{task}_space-fsLR_seg-Schaefer4007_condition-rest'


def _skewed_subjects(task='rfMRIREST1LR'):
    """Three subjects whose mean (4) and median (2) differ -- one long-INT outlier."""
    return {_key('01', task): np.array([1.0, 10.0]),
            _key('02', task): np.array([2.0, 20.0]),
            _key('03', task): np.array([9.0, 90.0])}


class TestDefaultIsUnchanged:
    def test_both_functions_default_to_mean(self):
        for fn in (create_ints_dataframe_custom_grouping, _apply_scan_groupings):
            assert inspect.signature(fn).parameters['statistic'].default == 'mean'

    def test_the_default_is_the_historical_mean(self):
        df = create_ints_dataframe_custom_grouping(_skewed_subjects(), n_regions=N_REGIONS)
        np.testing.assert_allclose(df['rfMRIREST1LR'].values, [4.0, 40.0])

    def test_passing_mean_explicitly_matches_the_default(self):
        data = _skewed_subjects()
        np.testing.assert_array_equal(
            create_ints_dataframe_custom_grouping(data, n_regions=N_REGIONS).values,
            create_ints_dataframe_custom_grouping(data, n_regions=N_REGIONS, statistic='mean').values)


class TestSubjectStatistic:
    def test_median_collapses_subjects_robustly(self):
        df = create_ints_dataframe_custom_grouping(_skewed_subjects(), n_regions=N_REGIONS,
                                                   statistic='median')
        np.testing.assert_allclose(df['rfMRIREST1LR'].values, [2.0, 20.0])

    def test_nans_are_ignored_not_propagated(self):
        data = _skewed_subjects()
        data[_key('04', 'rfMRIREST1LR')] = np.array([np.nan, np.nan])
        df = create_ints_dataframe_custom_grouping(data, n_regions=N_REGIONS, statistic='median')
        assert np.isfinite(df['rfMRIREST1LR'].values).all()


class TestRunGroupingUsesTheSameStatistic:
    """The second aggregation. If it stayed on the mean, a 'median' table would be a median of
    subjects followed by a mean of runs -- a mixture, and not what the argument says."""

    def _four_rest_runs(self):
        # One subject per run; run values 1, 2, 3, 100 -> mean 26.5, median 2.5.
        data = {}
        for run, value in zip(('rfMRIREST1LR', 'rfMRIREST1RL', 'rfMRIREST2LR', 'rfMRIREST2RL'),
                              (1.0, 2.0, 3.0, 100.0)):
            data[_key('01', run)] = np.array([value, value])
        return data

    def test_mean_over_four_rest_runs(self):
        df = create_ints_dataframe_custom_grouping(self._four_rest_runs(), n_regions=N_REGIONS,
                                                   average_3t_rest=True)
        np.testing.assert_allclose(df['task-REST_3T'].values, [26.5, 26.5])

    def test_median_over_four_rest_runs(self):
        df = create_ints_dataframe_custom_grouping(self._four_rest_runs(), n_regions=N_REGIONS,
                                                   average_3t_rest=True, statistic='median')
        np.testing.assert_allclose(df['task-REST_3T'].values, [2.5, 2.5])

    def test_a_two_run_pair_is_identical_either_way(self):
        """The median of two values IS their mean, so LR/RL pairs cannot distinguish the two --
        which is why this only matters for rest and average_3t_tasks."""
        data = {_key('01', 'tfMRIWMLR'): np.array([1.0, 1.0]),
                _key('01', 'tfMRIWMRL'): np.array([5.0, 5.0])}
        kw = dict(n_regions=N_REGIONS, average_3t_task_pairs=True)
        np.testing.assert_array_equal(
            create_ints_dataframe_custom_grouping(data, **kw).values,
            create_ints_dataframe_custom_grouping(data, **kw, statistic='median').values)

    def test_apply_scan_groupings_honours_it_directly(self):
        groups = {'rfMRIREST1LR': np.array([1.0]), 'rfMRIREST1RL': np.array([2.0]),
                  'rfMRIREST2LR': np.array([3.0]), 'rfMRIREST2RL': np.array([100.0])}
        out = _apply_scan_groupings(dict(groups), average_3t_rest=True, statistic='median')
        np.testing.assert_allclose(out['task-REST_3T'], [2.5])


class TestGuards:
    @pytest.mark.parametrize('bad', ['Median', 'nanmedian', 'mode', None])
    def test_an_unknown_statistic_raises_in_both_functions(self, bad):
        with pytest.raises(ValueError, match='statistic'):
            create_ints_dataframe_custom_grouping(_skewed_subjects(), n_regions=N_REGIONS,
                                                  statistic=bad)
        with pytest.raises(ValueError, match='statistic'):
            _apply_scan_groupings({'rfMRIREST1LR': np.array([1.0])}, statistic=bad)

    def test_sort_by_median_is_a_sort_key_not_the_statistic(self):
        """Column ORDER is by median regardless; the VALUES follow `statistic`."""
        data = {_key('01', 'rfMRIREST1LR'): np.array([1.0, 1.0]),
                _key('02', 'rfMRIREST1LR'): np.array([1.0, 7.0]),
                _key('01', 'tfMRIWMLR'): np.array([5.0, 5.0]),
                _key('02', 'tfMRIWMLR'): np.array([5.0, 5.0])}
        df = create_ints_dataframe_custom_grouping(data, n_regions=N_REGIONS, sort_by_median=True,
                                                   statistic='median')
        assert list(df.columns) == ['tfMRIWMLR', 'rfMRIREST1LR']      # 5.0 > 1.0/4.0
        np.testing.assert_allclose(df['rfMRIREST1LR'].values, [1.0, 4.0])
