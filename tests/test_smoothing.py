import numpy as np
import pytest

from stdpipe.smoothing import ApproxLoessRegressor


def test_loess_clamps_k_to_n():
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.zeros(5, dtype=float)

    reg = ApproxLoessRegressor(k=10, robust_iters=0)
    reg.fit(X, y)
    yhat = reg.predict(X)

    assert yhat.shape == (5,)


def test_loess_rejects_negative_sample_weight():
    X = np.arange(3, dtype=float).reshape(-1, 1)
    y = np.zeros(3, dtype=float)
    w = np.array([1.0, -1.0, 1.0])

    reg = ApproxLoessRegressor(k=3)
    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        reg.fit(X, y, sample_weight=w)


def test_loess_robust_reduces_outlier_influence():
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 0.0, 0.0, 0.0, 100.0])

    reg_plain = ApproxLoessRegressor(k=5, robust_iters=0)
    yhat_plain = reg_plain.fit(X, y).predict(X)

    reg_robust = ApproxLoessRegressor(k=5, robust_iters=1)
    yhat_robust = reg_robust.fit(X, y).predict(X)

    assert yhat_robust[-1] < yhat_plain[-1]
    assert yhat_robust[-1] < 0.95 * yhat_plain[-1]
