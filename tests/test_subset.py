import numpy as np
import delayedarray as da
from mattress import tatamize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_delayed_unary_isometric_subset():
    y = np.random.rand(1000, 100)
    x = da.DelayedArray(y)

    sub = x[:,1:20:2]
    assert isinstance(sub.seed, da.Subset)
    ptr = tatamize(sub)
    assert all(ptr.row(0) == y[0,1:20:2])
    assert all(ptr.column(1) == y[:,3])

    sub = x[5:200:5,:]
    assert isinstance(sub.seed, da.Subset)
    ptr = tatamize(sub)
    assert all(ptr.row(0) == y[5,:])
    assert all(ptr.column(1) == y[5:200:5,1])

    sub = x[5:200:5,10:90:10]
    assert isinstance(sub.seed, da.Subset)
    ptr = tatamize(sub)
    assert all(ptr.row(0) == y[5,10:90:10])
    assert all(ptr.column(1) == y[5:200:5,20])
