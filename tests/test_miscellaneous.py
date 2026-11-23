import numpy as np
from mattress import initialize, InitializedMatrix
import delayedarray as da
import scipy
import random

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_pointer_rewrap():
    y = np.random.rand(1000, 100)
    ptr = initialize(y)
    reptr = initialize(ptr)
    assert ptr.ptr == reptr.ptr


def test_simple_properties():
    y = np.random.rand(1000, 100)
    ptr = initialize(y)

    assert ptr.shape == (1000, 100)
    assert ptr.nrow() == 1000
    assert ptr.ncol() == 100
    assert ptr.dtype == np.float64
    assert not ptr.is_sparse()
    assert ptr.prefer_rows()

    assert (ptr.get(True, 0) == y[0,:]).all()
    assert (ptr.get(True, 999) == y[999,:]).all()
    assert (ptr.get(False, 0) == y[:,0]).all()
    assert (ptr.get(False, 99) == y[:,99]).all()


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


def test_matrix_vector_product():
    y = np.random.rand(20, 10)
    ptr = initialize(y)

    v = np.random.rand(10)
    assert np.allclose(np.matvec(y, v), ptr.multiply(v, right = True))
    v = np.random.rand(20)
    assert np.allclose(np.matvec(y.T, v).T, ptr.multiply(v, right = False))


def test_matrix_columns_product_dense():
    y = np.random.rand(20, 10)
    assert y.flags["C_CONTIGUOUS"]
    ptr = initialize(y)
    ptrf = initialize(np.asfortranarray(y))

    z = np.random.rand(10, 15)
    zf = np.asfortranarray(z)
    expected = np.matmul(y, z)
    assert np.allclose(expected, ptr.multiply(z, right = True))
    assert np.allclose(expected, ptr.multiply(zf, right = True))
    assert np.allclose(expected, ptrf.multiply(z, right = True))
    assert np.allclose(expected, ptrf.multiply(zf, right = True))

    z = np.random.rand(15, 20)
    zf = np.asfortranarray(z)
    expected = np.matmul(z, y)
    assert np.allclose(expected, ptr.multiply(z, right = False))
    assert np.allclose(expected, ptr.multiply(zf, right = False))
    assert np.allclose(expected, ptrf.multiply(z, right = False))
    assert np.allclose(expected, ptrf.multiply(zf, right = False))


def test_matrix_columns_product_sparse():
    y = scipy.sparse.random(200, 500, 0.1).tocsr()
    ptr = initialize(y)
    ptrf = initialize(y.tocsc())

    z = np.random.rand(500, 15)
    zf = np.asfortranarray(z)
    expected = np.matmul(y.toarray(), z)
    assert np.allclose(expected, ptr.multiply(z, right = True))
    assert np.allclose(expected, ptr.multiply(zf, right = True))
    assert np.allclose(expected, ptrf.multiply(z, right = True))
    assert np.allclose(expected, ptrf.multiply(zf, right = True))

    z = np.random.rand(15, 200)
    zf = np.asfortranarray(z)
    expected = np.matmul(z, y.toarray())
    assert np.allclose(expected, ptr.multiply(z, right = False))
    assert np.allclose(expected, ptr.multiply(zf, right = False))
    assert np.allclose(expected, ptrf.multiply(z, right = False))
    assert np.allclose(expected, ptrf.multiply(zf, right = False))


def test_matrix_matrix_product():
    y1 = np.random.rand(20, 10)
    ptr1 = initialize(y1)

    y2 = np.random.rand(10, 15)
    ptr2 = initialize(y2)
    expected = np.matmul(y1, y2)
    assert np.allclose(expected, ptr1.multiply(ptr2))

    y2 = np.random.rand(15, 20)
    ptr2 = initialize(y2)
    expected = np.matmul(y2, y1)
    assert np.allclose(expected, ptr1.multiply(ptr2, right = False))


def test_DelayedArray_rewrap_dense():
    y = np.random.rand(1000, 100)
    ptr = initialize(y)

    da2 = da.DelayedArray(ptr)
    assert isinstance(da2.seed, InitializedMatrix)
    assert (np.array(da2) == y).all()

    # These all indirectly call extract_dense_array().
    sub = da2[10:50, 0:100:2]
    assert (np.array(sub) == y[10:50, 0:100:2]).all()

    sub = da2[10:50, :]
    assert (np.array(sub) == y[10:50, :]).all()

    sub = da2[:, 10:50]
    assert (np.array(sub) == y[:, 10:50]).all()


def test_DelayedArray_rewrap_sparse_csc():
    y = scipy.sparse.random(200, 500, 0.1).tocsc()
    ptr = initialize(y)

    da2 = da.DelayedArray(ptr)
    assert isinstance(da2.seed, InitializedMatrix)
    assert da.is_sparse(da2)

    full = da.to_sparse_array(da2)
    assert isinstance(full, da.SparseNdarray)
    assert (np.array(full) == y.toarray()).all()

    # These all indirectly call InitializedMatrix's extract_sparse_array().
    sub = da.extract_sparse_array(da2, (range(10, 50), range(0, 100, 2)))
    assert (np.array(sub) == y[10:50, 0:100:2].toarray()).all()

    sub = da.extract_sparse_array(da2, (range(10, 50), range(y.shape[1])))
    assert (np.array(sub) == y[10:50, :].toarray()).all()

    sub = da.extract_sparse_array(da2, (range(y.shape[0]), range(10, 50)))
    assert (np.array(sub) == y[:, 10:50].toarray()).all()


def test_DelayedArray_rewrap_sparse_csr():
    y = scipy.sparse.random(200, 500, 0.1).tocsr()
    ptr = initialize(y)

    da2 = da.DelayedArray(ptr)
    assert isinstance(da2.seed, InitializedMatrix)
    assert da.is_sparse(da2)

    full = da.to_sparse_array(da2)
    assert isinstance(full, da.SparseNdarray)
    assert (np.array(full) == y.toarray()).all()

    # These all indirectly call InitializedMatrix's extract_sparse_array().
    sub = da.extract_sparse_array(da2, (range(10, 50), range(0, 100, 2)))
    assert (np.array(sub) == y[10:50, 0:100:2].toarray()).all()

    sub = da.extract_sparse_array(da2, (range(10, 50), range(y.shape[1])))
    assert (np.array(sub) == y[10:50, :].toarray()).all()

    sub = da.extract_sparse_array(da2, (range(y.shape[0]), range(10, 50)))
    assert (np.array(sub) == y[:, 10:50].toarray()).all()


def test_array_conversion():
    y = np.random.rand(1000, 100) * 10
    ptr = initialize(y)

    out = ptr.__array__()
    assert (out == y).all()

    out = ptr.__array__(dtype=np.int32)
    assert (out == y.astype(dtype=np.int32)).all()

    out = ptr.__DelayedArray_dask__()
    assert (out == y).all()


def test_includes():
    import os
    import mattress
    path = mattress.includes()
    assert isinstance(path, str)
    assert os.path.exists(os.path.join(path, "mattress.h"))
