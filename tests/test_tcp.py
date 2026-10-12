"""src.tcp: the reliability helpers (unit, no disk) and the subject assembly (integration)."""
import numpy as np
import pandas as pd
import pytest

from src.longitudinal import two_session_icc
from src.tcp import identification, paired_icc, spearman_brown
from tests.conftest import integration_tcp as integration


class TestPairedIcc:
    def test_is_two_session_icc(self):
        """The same statistic Penn LEAD reported, not a reimplementation."""
        rng = np.random.default_rng(0)
        true = rng.normal(size=(50, 4))
        a, b = true + rng.normal(size=(50, 4)), true + rng.normal(size=(50, 4))
        df = pd.DataFrame({'sub': list(range(50)) * 2, 'ses': ['1'] * 50 + ['2'] * 50})
        np.testing.assert_allclose(paired_icc(a, b)[0], two_session_icc(np.vstack([a, b]), df)[0])

    def test_identical_is_one_and_one_dimensional_input_works(self):
        x = np.random.default_rng(1).normal(size=30)
        assert paired_icc(x, x)[0][0] == pytest.approx(1.0)

    def test_bootstrap_duplicates_count_as_separate_rows(self):
        """Resampled subjects must not collapse onto one id, or the bootstrap shrinks n."""
        x = np.random.default_rng(2).normal(size=(20, 1))
        i = np.r_[np.arange(20), np.arange(20)]
        assert paired_icc(x[i], x[i])[1][0] == 40


def test_spearman_brown():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(0.0) == 0.0


class TestIdentification:
    def test_perfect_and_chance(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(40, 100))
        assert identification(a, a)['accuracy'] == 1.0
        out = identification(a, rng.normal(size=(40, 100)))
        assert out['chance'] == pytest.approx(1 / 40) and out['accuracy'] < 0.2

    def test_shared_group_map_does_not_identify(self):
        """Every subject = the same group map + noise: centring removes the map, so no one is found."""
        rng = np.random.default_rng(4)
        group = rng.normal(size=100) * 10
        a = group + rng.normal(size=(40, 100))
        b = group + rng.normal(size=(40, 100))
        assert identification(a, b)['accuracy'] < 0.2


@integration
class TestLoad:
    @pytest.fixture(scope='class')
    def data(self):
        from src.tcp import load_tcp
        return load_tcp()

    def test_cohort_and_pairing(self, data):
        assert len(data.df) == 206 and data.df['sub'].is_unique
        assert data.pairing == {'stroopAP': 'rest1AP', 'stroopPA': 'rest1PA', 'hammerAP': 'rest1AP'}

    def test_deflections_are_built_from_their_own_run(self, data):
        A = data.arrays
        np.testing.assert_array_equal(A['deflection_stroopPA'], A['task_stroopPA'] - A['control_stroopPA'])
        np.testing.assert_array_equal(A['deflection_raw_stroopPA'], A['task_stroopPA'] - A['rest_rest1PA'])

    def test_run_average_needs_both_runs(self, data):
        A = data.arrays
        one_missing = np.isnan(A['task_stroopAP']) ^ np.isnan(A['task_stroopPA'])
        assert one_missing.any() and np.isnan(A['task_stroop'][one_missing]).all()


class TestDxClass:
    @pytest.mark.parametrize('text,label', [
        ('MDD (w/ mild anxious distress, melancholic features)', 'mood'), ('BPII', 'mood'),
        ('cyclothymia', 'mood'), ('PMDD', 'mood'), ('SZ (ruleout SZA)', 'psychotic'),
        ('SZ or SZA', 'psychotic'), ('social anxiety', 'anxiety'), ('OCD', 'anxiety'),
        ('PTSD', 'anxiety'), ('mild ADHD', 'other'), ('present polysub use (ER)', 'other'),
        ('binge eating disorder', 'other'), (' past GAD', 'past only'),
        ('Past agoraphobia with panic disorder', 'past only'), ('past MDD (due to SUD)', 'past only')])
    def test_rules(self, text, label):
        from src.tcp import dx_class
        assert dx_class(text) == label

    def test_missing_and_unknown(self):
        from src.tcp import dx_class
        assert dx_class(None) is None and dx_class(pd.NA) is None
        with pytest.raises(ValueError):
            dx_class('something new')
