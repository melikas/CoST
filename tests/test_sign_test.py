import numpy as np

from tasks.sign_test import sign_p


def test_all_ties_is_no_evidence():
    """The bug this module exists for: counting ties as losses made an arm compared with
    itself the most significant result the test can return."""
    assert sign_p(np.zeros(24)) == 1.0


def test_ties_are_dropped_not_counted():
    #  6 positive, 0 negative, 18 tied -> the test sees 6 of 6
    d = np.array([1.0] * 6 + [0.0] * 18)
    assert np.isclose(sign_p(d), 2 / 2 ** 6)


def test_nans_are_dropped():
    assert np.isclose(sign_p(np.array([1.0, 1.0, np.nan, 1.0])), 2 / 2 ** 3)


def test_symmetric_split_is_not_significant():
    assert sign_p(np.array([1.0] * 12 + [-1.0] * 12)) == 1.0


def test_unanimous_is_significant():
    assert sign_p(np.ones(24)) < 1e-6


def test_matches_the_textbook_value():
    # 19 of 24 positive, two-sided: 2 * P(X <= 5), X ~ Binomial(24, 0.5)
    from math import comb
    want = 2 * sum(comb(24, i) for i in range(6)) / 2 ** 24
    assert np.isclose(sign_p(np.array([1.0] * 19 + [-1.0] * 5)), want)
