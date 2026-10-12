"""Longitudinal deflections (``src.longitudinal``): assembly, age split, change, and map statistics.

What these pin, each against a known answer: a missing scan drops its session, not its subject; the
deflection is task minus the matched control, signed; the within/between age split; annualised
change is the per-subject OLS slope (a difference quotient for two sessions); the per-region OLS
and the fast pattern-grouped intercept agree, including with missing parcels and bootstrap weights;
the ICC is consistency, so a shift shared by everyone does not lower it; and the mixed model
recovers a planted within-person slope and tells it apart from a different between-person one.

Stubs only: no data mount, no NAS reads.
"""
import numpy as np
import pandas as pd
import pytest

from src.longitudinal import (DEFLECTIONS, add_age_components, annualized_change, assemble_sessions,
                              bin_means, bootstrap_intercept_corr, coverage_mask, design_matrix,
                              intercept_map, map_corr, mixed_model_maps, regress_maps, sa_deciles,
                              sa_gradient_model, two_session_icc, fit_age_gam,
                              cluster_bootstrap_age_gam)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _outputs(present, n_regions=3):
    """Signal tables where every (sub, ses) in ``present`` has distinct, recognisable values."""
    out = {k: {} for k in ('rest', 'task', 'control', 'control_drift')}
    for i, (sub, ses) in enumerate(present):
        base = float(i + 1)
        out['rest'][(sub, ses, 'rest')] = np.full(n_regions, base)
        out['task'][(sub, ses, 'nback')] = np.full(n_regions, 10 * base)
        out['control'][(sub, ses, 'nback')] = np.full(n_regions, 2 * base)
        out['control_drift'][(sub, ses, 'nback')] = np.full(n_regions, 3 * base)
    registry = [(s, ses, t) for s, ses in present for t in ('rest', 'nback')]
    return out, registry


class TestAssembly:
    def test_a_missing_scan_drops_its_session_not_its_subject(self):
        out, registry = _outputs([('A', '1'), ('A', '2'), ('B', '1')])
        out['control'][('A', '2', 'nback')] = None
        data = assemble_sessions(out, registry)
        assert list(zip(data.df['sub'], data.df['ses'])) == [('A', '1'), ('B', '1')]
        assert all(v.shape == (2, 3) for v in data.arrays.values())

    def test_deflections_are_task_minus_each_reference(self):
        out, registry = _outputs([('A', '1')])
        a = assemble_sessions(out, registry).arrays
        assert a['deflection'][0, 0] == 10 - 2          # task - matched control, signed
        assert a['deflection_drift'][0, 0] == 10 - 3
        assert a['deflection_raw'][0, 0] == 10 - 1
        assert set(DEFLECTIONS) <= set(a)


class TestAgeSplit:
    def test_within_sums_to_zero_and_between_is_centred(self):
        df = pd.DataFrame({'sub': ['A', 'A', 'B', 'C', 'C', 'C'],
                           'age': [10.0, 12.0, 15.0, 11.0, 12.0, 13.0]})
        out = add_age_components(df)
        assert np.allclose(out.groupby('sub')['age_within'].sum(), 0)
        assert out.loc[out['sub'] == 'B', 'age_within'].item() == 0      # single session
        # centred on the mean of SUBJECT means (11, 15, 12 -> 38/3), not of sessions
        assert np.isclose(out.groupby('sub')['age_between'].first().mean(), 0)
        assert out['n_sessions'].tolist() == [2, 2, 1, 3, 3, 3]


class TestAnnualizedChange:
    def test_two_sessions_is_the_difference_quotient_and_three_the_ols_slope(self):
        df = pd.DataFrame({'sub': ['A', 'A', 'B', 'C', 'C', 'C'],
                           'age': [10.0, 12.0, 14.0, 10.0, 11.0, 13.0],
                           'fd_rest': [0.2, 0.1, 0.3, 0.1, 0.1, 0.1],
                           'fd_task': [0.3, 0.3, 0.3, 0.2, 0.2, 0.2]})
        values = np.array([[1.0], [2.0], [9.0], [0.0], [0.5], [1.5]])     # C: slope 0.5 exactly
        change, df_sub = annualized_change(values, df)
        assert df_sub['sub'].tolist() == ['A', 'C']                       # B has one session
        assert change[0, 0] == pytest.approx((2 - 1) / (12 - 10))
        assert change[1, 0] == pytest.approx(0.5)
        assert df_sub.loc[0, 'delta_fd_rest'] == pytest.approx((0.1 - 0.2) / 2)
        assert df_sub.loc[0, 'span'] == 2 and df_sub.loc[0, 'age_first'] == 10


class TestMapRegression:
    def test_recovers_known_coefficients(self, rng):
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        Y = X @ np.array([[1.0, -2.0], [0.5, 3.0]]) + 0.01 * rng.standard_normal((n, 2))
        fit = regress_maps(Y, X, ['intercept', 'x'])
        assert np.allclose(fit['beta'].to_numpy(), [[1.0, 0.5], [-2.0, 3.0]], atol=0.01)
        assert (fit['p']['x'] < 1e-10).all()

    def test_missing_values_are_dropped_per_region(self, rng):
        n = 50
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        Y = rng.standard_normal((n, 3))
        Y[:5, 1] = np.nan
        fit = regress_maps(Y, X)
        assert fit['n'].tolist() == [50, 45, 50]
        ref = np.linalg.lstsq(X[5:], Y[5:, 1], rcond=None)[0]
        assert np.allclose(fit['beta'].iloc[1].to_numpy(), ref)

    def test_design_matrix_centres_only_what_it_is_told_to(self):
        df = pd.DataFrame({'age': [10.0, 20.0], 'dfd': [0.5, 1.5]})
        X, names = design_matrix(df, centred=('age',), raw=('dfd',))
        assert names == ['intercept', 'age', 'dfd']
        assert X[:, 1].tolist() == [-5.0, 5.0] and X[:, 2].tolist() == [0.5, 1.5]


class TestInterceptMap:
    def test_matches_per_region_ols_with_missing_parcels(self, rng):
        n, R = 40, 30
        X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
        Y = rng.standard_normal((n, R))
        Y[rng.random((n, R)) < 0.1] = np.nan
        fast = intercept_map(Y, X)
        slow = regress_maps(Y, X)['beta'].iloc[:, 0].to_numpy()
        assert np.allclose(fast, slow)

    def test_weights_are_row_duplication(self, rng):
        """A bootstrap count of 2 must mean exactly 'this subject appears twice'."""
        n, R = 20, 5
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        Y = rng.standard_normal((n, R))
        Y[3, 2] = np.nan
        w = rng.integers(0, 3, n)
        idx = np.repeat(np.arange(n), w)
        assert np.allclose(intercept_map(Y, X, weights=w), intercept_map(Y[idx], X[idx]))

    def test_bootstrap_centres_on_the_observed_correlation(self, rng):
        n, R = 60, 100
        target = rng.standard_normal(R)
        Y = target[None, :] + 0.5 * rng.standard_normal((n, R))
        X = np.ones((n, 1))
        observed = map_corr(intercept_map(Y, X), target)
        rs = bootstrap_intercept_corr(Y, X, target, n_boot=200, seed=1)
        assert abs(np.median(rs) - observed) < 0.05 and rs.std() > 0


class TestIcc:
    def _two_sessions(self, a, b):
        n = len(a)
        df = pd.DataFrame({'sub': [f's{i}' for i in range(n)] * 2, 'ses': ['1'] * n + ['2'] * n})
        return np.concatenate([a, b])[:, None], df

    def test_a_shared_shift_does_not_lower_consistency(self, rng):
        a = rng.standard_normal(30)
        icc, n = two_session_icc(*self._two_sessions(a, a + 5.0))
        assert icc[0] == pytest.approx(1.0) and n[0] == 30

    def test_noise_lowers_it(self, rng):
        a = rng.standard_normal(200)
        icc, _ = two_session_icc(*self._two_sessions(a, a + rng.standard_normal(200)))
        assert 0.3 < icc[0] < 0.7                                         # true ICC is 0.5

    def test_incomplete_pairs_are_dropped(self, rng):
        a, b = rng.standard_normal(20), rng.standard_normal(20)
        b[:4] = np.nan
        _, n = two_session_icc(*self._two_sessions(a, b))
        assert n[0] == 16


class TestMixedModel:
    def test_recovers_a_within_slope_distinct_from_the_between_slope(self, rng):
        rows, y = [], []
        for s in range(80):
            age0 = rng.uniform(9, 16)
            u = rng.standard_normal()
            for t in (0.0, 2.0):
                age = age0 + t
                rows.append({'sub': f's{s}', 'age': age, 'sex': s % 2,
                             'fd_rest': 0.2, 'fd_task': 0.2})
                # within-person slope -0.10/yr; between-person (cohort) slope +0.10/yr
                y.append(0.10 * age0 - 0.10 * t + u + 0.05 * rng.standard_normal())
        df = add_age_components(pd.DataFrame(rows))
        df['fd_rest'] += 0.01 * rng.standard_normal(len(df))            # avoid a singular design
        df['fd_task'] += 0.01 * rng.standard_normal(len(df))
        res = mixed_model_maps(np.array(y)[:, None], df)
        assert res.loc[0, 'beta_age_within'] == pytest.approx(-0.10, abs=0.02)
        assert res.loc[0, 'beta_age_between'] > 0
        assert res.loc[0, 'p_diff'] < 0.01


class TestUnitCovariate:
    """The matched-control INT differs by region, so each region's model gets its own column."""

    def test_regress_maps_appends_the_units_own_column(self, rng):
        n, R = 60, 4
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        C = rng.standard_normal((n, R))
        Y = 0.5 + 2.0 * X[:, [1]] + 0.8 * C + 0.01 * rng.standard_normal((n, R))
        fit = regress_maps(Y, X, ['intercept', 'x'], unit_covariate=C, unit_covariate_name='control')
        assert list(fit['beta'].columns) == ['intercept', 'x', 'control']
        assert np.allclose(fit['beta']['control'], 0.8, atol=0.01)
        assert np.allclose(fit['beta']['x'], 2.0, atol=0.01)

    def test_intercept_map_agrees_with_regress_maps_and_weights_duplicate(self, rng):
        n, R = 30, 6
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        C = rng.standard_normal((n, R))
        Y = rng.standard_normal((n, R))
        Y[2, 3] = np.nan
        slow = regress_maps(Y, X, unit_covariate=C)['beta'].iloc[:, 0].to_numpy()
        assert np.allclose(intercept_map(Y, X, unit_covariate=C), slow)
        w = rng.integers(0, 3, n)
        idx = np.repeat(np.arange(n), w)
        assert np.allclose(intercept_map(Y, X, weights=w, unit_covariate=C),
                           intercept_map(Y[idx], X[idx], unit_covariate=C[idx]), equal_nan=True)

    def test_mixed_model_takes_a_per_region_covariate(self, rng):
        rows, y, c = [], [], []
        for s in range(60):
            age0, u = rng.uniform(9, 16), rng.standard_normal()
            for t in (0.0, 2.0):
                ctrl = rng.standard_normal(2)
                rows.append({'sub': f's{s}', 'age': age0 + t, 'sex': s % 2,
                             'fd_rest': 0.2 + 0.01 * rng.standard_normal(),
                             'fd_task': 0.2 + 0.01 * rng.standard_normal()})
                c.append(ctrl)
                y.append(0.7 * ctrl - 0.1 * t + u + 0.05 * rng.standard_normal(2))
        df = add_age_components(pd.DataFrame(rows))
        res = mixed_model_maps(np.array(y), df, unit_covariates={'control': np.array(c)})
        assert np.allclose(res['beta_control'], 0.7, atol=0.03)
        assert np.allclose(res['beta_age_within'], -0.1, atol=0.03)


class TestSaBins:
    def test_deciles_run_from_one_to_ten_in_axis_order(self):
        d = sa_deciles(np.arange(100.0))
        assert d.min() == 1 and d.max() == 10 and np.all(np.diff(d) >= 0)
        assert np.bincount(d)[1:].tolist() == [10] * 10

    def test_coverage_and_bin_means_skip_what_they_should(self):
        v = np.array([[1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, np.nan], [1.0, 2.0, np.nan, 4.0]])
        keep = coverage_mask(v, 0.6)
        assert keep.tolist() == [True, True, False, True]
        m = bin_means(v, np.array([1, 1, 2, 2]), keep)
        assert m[:, 0].tolist() == [1.5, 1.5, 1.5]
        assert m[0, 1] == 4.0 and np.isnan(m[1, 1])      # its only kept parcel is missing there

    def test_gradient_model_recovers_a_planted_axis_gradient(self, rng):
        """Within-person age effect of -0.10/yr at the sensorimotor end, +0.10/yr at the
        association end: the age_within:sa_pos interaction must come out near +0.20."""
        n_bins, rows, task, ctrl = 10, [], [], []
        pos = np.arange(n_bins) / (n_bins - 1) - 0.5
        for s in range(60):
            age0, u = rng.uniform(9, 16), rng.standard_normal()
            for t in (0.0, 2.0):
                rows.append({'sub': f's{s}', 'ses': str(int(t)), 'age': age0 + t, 'sex': s % 2,
                             'fd_rest': 0.2 + 0.01 * rng.standard_normal(),
                             'fd_task': 0.2 + 0.01 * rng.standard_normal()})
                c = rng.standard_normal(n_bins)
                ctrl.append(c)
                task.append(0.5 * c + (0.2 * pos) * t + u + 0.05 * rng.standard_normal(n_bins))
        df = add_age_components(pd.DataFrame(rows))
        _, summary = sa_gradient_model(np.array(task), np.array(ctrl), df)
        assert summary.loc['age_within:sa_pos', 'beta'] == pytest.approx(0.20, abs=0.03)
        assert summary.loc['age_within:sa_pos', 'p'] < 1e-6
        assert summary.loc['control', 'beta'] == pytest.approx(0.5, abs=0.03)


def _curve_data(rng, n_sub=80):
    """Two sessions per subject; y = 0.02 (age - 13)^2 + 0.5 nuisance + subject intercept + noise."""
    ages, nus, ys, subs = [], [], [], []
    for s in range(n_sub):
        a0, u = rng.uniform(9, 16), 0.3 * rng.standard_normal()
        for t in (0.0, 2.0):
            nu = rng.standard_normal()
            ages.append(a0 + t); nus.append(nu); subs.append(f's{s}')
            ys.append(0.02 * (a0 + t - 13) ** 2 + 0.5 * nu + u + 0.05 * rng.standard_normal())
    return np.array(ages), np.array(ys), np.array(nus)[:, None], np.array(subs)


class TestAgeGam:
    def test_recovers_the_age_curve_through_a_nuisance_term(self, rng):
        age, y, nu, _ = _curve_data(rng)
        grid = np.linspace(10, 17, 15)
        fit = fit_age_gam(age, y, grid, nuisance=nu, nuisance_at=[0.0])
        truth = 0.02 * (grid - 13) ** 2
        # level is shifted by the mean subject intercept; the SHAPE must match
        assert np.corrcoef(fit['pred'], truth)[0, 1] > 0.95
        assert fit['n'] == len(age) and fit['edof'] > 1

    def test_nuisance_is_held_where_it_is_told(self, rng):
        age, y, nu, _ = _curve_data(rng)
        grid = np.linspace(10, 17, 5)
        at0 = fit_age_gam(age, y, grid, nu, [0.0], lam=1.0)
        at1 = fit_age_gam(age, y, grid, nu, [1.0], lam=1.0)
        assert np.allclose(at1['pred'] - at0['pred'], 0.5, atol=0.05)   # the nuisance slope
        # adjusted points remove each observation's nuisance deviation from the held value
        assert np.allclose(at1['adjusted'] - at0['adjusted'], at1['pred'][0] - at0['pred'][0])

    def test_cluster_bootstrap_is_reproducible_and_covers_the_shape(self, rng):
        age, y, nu, subs = _curve_data(rng)
        grid = np.linspace(10, 17, 8)
        a = cluster_bootstrap_age_gam(age, y, subs, grid, nu, [0.0], lam=1.0, n_boot=60, seed=3)
        b = cluster_bootstrap_age_gam(age, y, subs, grid, nu, [0.0], lam=1.0, n_boot=60, seed=3)
        assert a.shape == (60, 8) and np.allclose(a, b, equal_nan=True)
        full = fit_age_gam(age, y, grid, nu, [0.0], lam=1.0)['pred']
        lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=0)
        assert np.all((full >= lo) & (full <= hi))


    @pytest.mark.parametrize('lam', ['gcv', 1.0])
    def test_matches_the_library_fit_exactly(self, rng, lam):
        """Penn LEAD's GAMs must behave like fig3's: same terms, same penalty handling. At the
        sample's own nuisance means the curve is snaplab_tools.gams.fit_gam's, bit for bit."""
        from snaplab_tools.gams import fit_gam
        age, y, nu, _ = _curve_data(rng)
        grid = np.linspace(10, 17, 12)
        ours = fit_age_gam(age, y, grid, nu, None, n_splines=5, lam=lam)
        pred, *_ = fit_gam(age, y, grid, n_splines=5, lam=lam, nuisance=nu)
        assert np.allclose(ours['pred'], pred, rtol=0, atol=1e-12)
