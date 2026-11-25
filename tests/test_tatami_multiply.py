import numpy as np
from mattress import initialize, InitializedMatrix
import delayedarray as da
import scipy
import random

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


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
