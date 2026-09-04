"""The two-sided sign test, in one place.

It existed twice, and both copies counted TIES as losses. That is not a detail: an arm
compared against itself -- `trend seg 1 + spec` IS the production readout, reassembled from
parts -- differs by exactly zero in every pair, and a tie-as-loss rule reports that as
k=0 of n, which is the most significant result the test can return. The correct rule
discards ties and tests the rest, so an all-tie comparison returns p=1.
"""
from math import comb

import numpy as np


def sign_p(d):
    """Two-sided sign test on paired differences `d`. NaNs and exact ties are dropped.

    Returns 1.0 when nothing survives, because no evidence is not strong evidence.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = int((d > 0).sum())
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def sign_summary(d):
    """(wins, comparable pairs, p) -- the three numbers a row of a paired table shows.

    All three drop NaNs and ties together, so the count and the p-value describe the same
    set of pairs. Printing `0/24` beside p=1.0 because the count kept 24 ties while the test
    dropped them reads as a unanimous loss, which is the opposite of what happened.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    return int((d > 0).sum()), len(d), sign_p(d)
