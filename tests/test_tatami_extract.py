import numpy as np
from mattress import initialize, InitializedMatrix
import delayedarray as da
import scipy
import random

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_DelayedArray_extract_dense():
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


def test_DelayedArray_extract_sparse_csc():
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


def test_DelayedArray_extract_sparse_csr():
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
