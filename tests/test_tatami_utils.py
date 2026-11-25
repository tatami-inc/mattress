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
