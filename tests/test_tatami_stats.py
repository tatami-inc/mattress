import numpy as np
from mattress import initialize, InitializedMatrix
import delayedarray as da
import scipy
import random

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_simple_stats():
    y = np.random.rand(1000, 100)
    ptr = initialize(y)

    assert np.allclose(ptr.sum(True), y.sum(axis=1))
    assert np.allclose(ptr.sum(False), y.sum(axis=0))

    m, v = ptr.variance(True) 
    assert np.allclose(m, y.mean(axis=1))
    assert np.allclose(v, y.var(axis=1, ddof=1))

    m, v = ptr.variance(False) 
    assert np.allclose(m, y.mean(axis=0))
    assert np.allclose(v, y.var(axis=0, ddof=1))

    assert ptr.median(True).shape == (1000,)
    assert ptr.median(False).shape == (100,)

    mn, mx = ptr.range(True)
    assert (mn == y.min(axis=1)).all()
    assert (mx == y.max(axis=1)).all()

    mn, mx = ptr.range(False)
    assert (mn == y.min(axis=0)).all()
    assert (mx == y.max(axis=0)).all()


def test_nan_counts():
    y = np.random.rand(20, 10)
    y[0, 0] = np.nan
    y[1, 2] = np.nan
    y[4, 3] = np.nan
    y[2, 3] = np.nan

    ptr = initialize(y)
    rnan = ptr.nan_count(True)
    assert rnan[0] == 1

    cnan = ptr.nan_count(False)
    assert cnan[3] == 2


def _validate_group_stat(res, levels, mat, row, grouping, fun):
    unique = {}
    for l in levels:
        unique[l] = []
    for i, g in enumerate(grouping):
        unique[g].append(i)
    for i, l in enumerate(levels):
        idx = np.array(unique[l])
        if row:
            assert np.allclose(res[:,i], fun(mat[:,idx]))
        else:
            assert np.allclose(res[i,:], fun(mat[idx,:]))


def test_group_sum():
    y = np.random.rand(20, 10)
    ptr = initialize(y)

    cgrouping = random.choices("ABCD", k=y.shape[1])
    rgrouping = []
    for i in range(y.shape[0]):
        rgrouping.append(i % 3)

    rsum, rlev = ptr.group_sum(True, cgrouping)
    _validate_group_stat(rsum, rlev, y, True, cgrouping, lambda submat : submat.sum(axis=1))

    csum, clev = ptr.group_sum(False, rgrouping)
    _validate_group_stat(csum, clev, y, False, rgrouping, lambda submat : submat.sum(axis=0))


def test_group_variance():
    y = np.random.rand(20, 10)
    ptr = initialize(y)

    cgrouping = list("ABCABCABCA") # ensure at least 2 observations for each group to avoid handling NaNs in comparisons.
    rgrouping = []
    for i in range(y.shape[0]):
        rgrouping.append(i % 3)

    rmean, rvar, rlev = ptr.group_variance(True, cgrouping)
    _validate_group_stat(rmean, rlev, y, True, cgrouping, lambda submat : submat.mean(axis=1))
    _validate_group_stat(rvar, rlev, y, True, cgrouping, lambda submat : submat.var(axis=1, ddof=1))

    cmean, cvar, clev = ptr.group_variance(False, rgrouping)
    _validate_group_stat(cmean, clev, y, False, rgrouping, lambda submat : submat.mean(axis=0))
    _validate_group_stat(cvar, clev, y, False, rgrouping, lambda submat : submat.var(axis=0, ddof=1))


def test_group_median():
    y = np.random.rand(20, 10)
    ptr = initialize(y)

    cgrouping = random.choices("ABCD", k=y.shape[1])
    rgrouping = []
    for i in range(y.shape[0]):
        rgrouping.append(i % 3)

    rmed, rlev = ptr.group_median(True, cgrouping)
    _validate_group_stat(rmed, rlev, y, True, cgrouping, lambda submat : initialize(submat).median(True))

    cmed, clev = ptr.group_median(False, rgrouping)
    _validate_group_stat(cmed, clev, y, False, rgrouping, lambda submat : initialize(submat).median(False))
