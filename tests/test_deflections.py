"""
Tests for the deflection layer (src/timescales.py): compute_deflections and
compute_regional_corrs.

No data mounts required — everything here runs on synthetic arrays.

The partial-correlation primitive these build on (partial_pearsonr) is generic, so it lives
in snaplab_tools.stats and is tested there; only its wiring into compute_regional_corrs is
tested here.

These functions previously carried a `nuisance_regression` option that regressed rest INT
out of (task - rest). That is algebraically identical to residualising task on rest (the
subtraction cancels exactly), which silently redefined the deflection and collapsed fig2's
S-A coupling. The option is gone; test_no_rest_control_parameters guards against it coming
back, and TestNuisanceRegressionEquivalence documents the algebra that made it a trap.
"""

import inspect

import numpy as np
import pytest
import scipy as sp

from snaplab_tools.stats import partial_pearsonr
from src.timescales import compute_deflections, compute_regional_corrs


@pytest.fixture
def rng():
    return np.random.default_rng(20260716)


class TestComputeDeflections:
    def test_signed_is_plain_difference(self, rng):
        rest, task = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
        np.testing.assert_allclose(compute_deflections(rest, task), task - rest)

    def test_absolute_is_magnitude(self, rng):
        rest, task = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
        np.testing.assert_allclose(
            compute_deflections(rest, task, absolute=True), np.abs(task - rest))

    def test_elementwise_so_1d_matches_2d_column(self, rng):
        """The whole point of dropping matrix mode: per-region == per-matrix, always."""
        rest, task = rng.normal(size=(40, 5)), rng.normal(size=(40, 5))
        matrix = compute_deflections(rest, task, absolute=True)
        for r in range(5):
            vector = compute_deflections(rest[:, r], task[:, r], absolute=True)
            np.testing.assert_allclose(matrix[:, r], vector)

    def test_shape_preserved(self, rng):
        for shape in [(10,), (10, 3), (4, 5, 6)]:
            out = compute_deflections(rng.normal(size=shape), rng.normal(size=shape))
            assert out.shape == shape

    def test_nan_propagates(self):
        rest = np.array([1.0, np.nan, 3.0])
        task = np.array([2.0, 5.0, np.nan])
        out = compute_deflections(rest, task)
        assert np.isnan(out[1]) and np.isnan(out[2])
        assert out[0] == pytest.approx(1.0)

    def test_does_not_mutate_inputs(self, rng):
        rest, task = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        rest_before, task_before = rest.copy(), task.copy()
        compute_deflections(rest, task, absolute=True)
        np.testing.assert_array_equal(rest, rest_before)
        np.testing.assert_array_equal(task, task_before)

    def test_accepts_pandas(self):
        pd = pytest.importorskip("pandas")
        rest, task = pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0, 6.0])
        np.testing.assert_allclose(compute_deflections(rest, task), [1.0, 2.0, 3.0])

    def test_shape_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="same shape"):
            compute_deflections(rng.normal(size=(10, 3)), rng.normal(size=(10, 4)))

    def test_no_rest_control_parameters(self):
        """Regression guard: these options must not come back (see module docstring)."""
        params = inspect.signature(compute_deflections).parameters
        assert 'nuisance_regression' not in params
        assert 'standardize' not in params
        assert set(params) == {'rest_timescales', 'task_timescales', 'absolute'}


class TestNuisanceRegressionEquivalence:
    """Documents why regressing rest out of (task - rest) was a trap."""

    def test_residualising_the_difference_equals_residualising_task(self, rng):
        rest = rng.normal(1.0, 0.3, 300)
        task = 0.6 * rest + rng.normal(0, 0.2, 300)

        def _resid(y, x):
            X = np.column_stack([x, np.ones(len(x))])
            return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        # resid((task - rest) ~ rest) == resid(task ~ rest), exactly.
        np.testing.assert_allclose(_resid(task - rest, rest), _resid(task, rest), atol=1e-10)


class TestComputeRegionalCorrs:
    def test_shapes_and_pval_return(self, rng):
        rest, task = rng.normal(size=(60, 7)), rng.normal(size=(60, 7))
        beh = rng.normal(size=60)
        out3 = compute_regional_corrs(rest, task, beh)
        assert len(out3) == 3 and all(o.shape == (7,) for o in out3)
        out6 = compute_regional_corrs(rest, task, beh, return_pval=True)
        assert len(out6) == 6 and all(o.shape == (7,) for o in out6)

    def test_matches_manual_pearson_without_covariate(self, rng):
        rest, task = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
        beh = rng.normal(size=50)
        rest_c, task_c, defl_c = compute_regional_corrs(rest, task, beh, absolute=True)
        for r in range(4):
            assert rest_c[r] == pytest.approx(sp.stats.pearsonr(rest[:, r], beh)[0])
            assert task_c[r] == pytest.approx(sp.stats.pearsonr(task[:, r], beh)[0])
            manual = sp.stats.pearsonr(np.abs(task[:, r] - rest[:, r]), beh)[0]
            assert defl_c[r] == pytest.approx(manual)

    def test_covariate_matches_partial_pearsonr(self, rng):
        rest, task = rng.normal(size=(80, 5)), rng.normal(size=(80, 5))
        beh = rng.normal(size=80)
        cov = np.nanmean(rest, axis=1)
        _, _, defl_c = compute_regional_corrs(rest, task, beh, covariate=cov)
        for r in range(5):
            expected, _, _ = partial_pearsonr(task[:, r] - rest[:, r], beh, cov)
            assert defl_c[r] == pytest.approx(expected)

    def test_multi_covariate_matches_partial_pearsonr(self, rng):
        """(n, k) covariates: fig2 controls rest INT, age, sex and head motion together."""
        rest, task = rng.normal(size=(90, 5)), rng.normal(size=(90, 5))
        beh = rng.normal(size=90)
        cov = np.column_stack([np.nanmean(rest, axis=1),          # subject mean rest INT
                               rng.uniform(22, 37, size=90),      # age
                               rng.integers(0, 2, size=90),       # sex
                               rng.uniform(0.03, 0.3, size=90)])  # head motion
        _, _, defl_c = compute_regional_corrs(rest, task, beh, covariate=cov)
        for r in range(5):
            expected, _, _ = partial_pearsonr(task[:, r] - rest[:, r], beh, cov)
            assert defl_c[r] == pytest.approx(expected)

    def test_single_column_2d_matches_1d_covariate(self, rng):
        """(n, 1) and (n,) are the same covariate and must give the same answer."""
        rest, task = rng.normal(size=(60, 4)), rng.normal(size=(60, 4))
        beh = rng.normal(size=60)
        cov = np.nanmean(rest, axis=1)
        _, _, flat = compute_regional_corrs(rest, task, beh, covariate=cov)
        _, _, col = compute_regional_corrs(rest, task, beh, covariate=cov[:, None])
        np.testing.assert_allclose(flat, col, rtol=0, atol=1e-12)

    def test_covariate_wrong_length_raises(self, rng):
        rest, task = rng.normal(size=(60, 4)), rng.normal(size=(60, 4))
        beh = rng.normal(size=60)
        with pytest.raises(ValueError, match='n_subjects'):
            compute_regional_corrs(rest, task, beh, covariate=rng.normal(size=(59, 2)))

    def test_absolute_flag_flips_sign_when_unidirectional(self, rng):
        """When task < rest everywhere, |d| == -d, so correlations must mirror exactly."""
        rest = rng.uniform(2.0, 4.0, size=(70, 6))
        task = rest - rng.uniform(0.5, 1.5, size=(70, 6))   # task always shorter
        beh = rng.normal(size=70)
        _, _, signed = compute_regional_corrs(rest, task, beh, absolute=False)
        _, _, absol = compute_regional_corrs(rest, task, beh, absolute=True)
        np.testing.assert_allclose(signed, -absol, atol=1e-12)

    def test_subject_level_covariate_preserves_regional_pattern(self, rng):
        """The design property that motivates the subject-mean-rest covariate.

        A covariate that is constant across regions removes a subject-level confound
        without reshaping the regional correlation map — which is why fig2 can control
        for rest INT this way without destroying the S-A coupling it tests.

        Ground truth here is `region_w`, a region-varying coupling standing in for the
        S-A axis. Both the controlled and uncontrolled maps must recover it.
        """
        n_sub, n_reg = 400, 40
        region_w   = np.linspace(-1, 1, n_reg)   # region-varying coupling ('S-A axis')
        global_int = rng.normal(0, 1, n_sub)     # subject-level confound (global INT)
        signal     = rng.normal(0, 1, n_sub)     # true behavioural driver

        beh  = signal + 0.8 * global_int + rng.normal(0, 0.5, n_sub)
        defl = (region_w[None, :] * signal[:, None]      # region-varying true effect
                + 0.5 * global_int[:, None]              # confound leaks in globally
                + rng.normal(0, 0.3, (n_sub, n_reg)))
        rest = global_int[:, None] + rng.normal(0, 0.4, (n_sub, n_reg))
        task = rest + defl

        cov = np.nanmean(rest, axis=1)           # ~ global_int, one scalar per subject
        _, _, no_ctrl   = compute_regional_corrs(rest, task, beh)
        _, _, subj_ctrl = compute_regional_corrs(rest, task, beh, covariate=cov)

        assert sp.stats.pearsonr(no_ctrl, region_w)[0] > 0.9
        assert sp.stats.pearsonr(subj_ctrl, region_w)[0] > 0.9
        assert sp.stats.pearsonr(no_ctrl, subj_ctrl)[0] > 0.9

    def test_missing_behavior_subjects_dropped(self, rng):
        rest, task = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
        beh = rng.normal(size=50)
        beh[:10] = np.nan
        _, _, defl_c = compute_regional_corrs(rest, task, beh, absolute=True)
        for r in range(3):
            manual = sp.stats.pearsonr(
                np.abs(task[10:, r] - rest[10:, r]), beh[10:])[0]
            assert defl_c[r] == pytest.approx(manual)

    def test_region_with_too_few_subjects_is_nan(self, rng):
        rest, task = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
        beh = rng.normal(size=30)
        rest[2:, 1] = np.nan     # region 1 keeps only 2 valid subjects
        rest_c, task_c, defl_c = compute_regional_corrs(rest, task, beh)
        assert np.isnan(rest_c[1]) and np.isnan(task_c[1]) and np.isnan(defl_c[1])
        assert np.isfinite(defl_c[0]) and np.isfinite(defl_c[2])

    def test_regional_covariate_is_rejected(self, rng):
        """Guard the documented footgun: covariates must be subject-level, not regional.

        Since (n, k) covariates are now allowed, a whole regional matrix is no longer caught by
        the length check -- it is caught by its column count matching n_regions.
        """
        rest, task = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
        beh = rng.normal(size=40)
        with pytest.raises(ValueError, match="one column per region"):
            compute_regional_corrs(rest, task, beh, covariate=rest)

    def test_no_deflection_method_parameter(self):
        """Regression guard: 'simple'/'matrix'/'glm' must not come back."""
        params = inspect.signature(compute_regional_corrs).parameters
        assert 'deflection_method' not in params
        assert 'nuisance_regression' not in params
        assert 'standardize' not in params
        assert 'covariate' in params
