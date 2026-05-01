import numpy as np
import pytest

from stdpipe.smoothing import ApproxLoessRegressor, fit_vector_field_2d


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


def _smooth_field_samples(N=2000, W=1024, H=1024, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, W, N); y = rng.uniform(0, H, N)
    xn = (x - W / 2) / (W / 2)
    yn = (y - H / 2) / (H / 2)
    dx_t = 0.4 * xn + 0.2 * yn ** 2
    dy_t = 0.3 * yn - 0.25 * xn * yn
    dx = dx_t + rng.normal(0, noise, N)
    dy = dy_t + rng.normal(0, noise, N)
    return x, y, dx, dy, dx_t, dy_t, (H, W)


@pytest.mark.parametrize("backend", ["loess", "grid"])
def test_fit_vector_field_2d_recovers_smooth_truth(backend):
    x, y, dx, dy, dx_t, dy_t, image_shape = _smooth_field_samples()
    if backend == "loess":
        pred = fit_vector_field_2d(
            x, y, dx, dy, backend="loess",
            scales=(image_shape[1] / 12.0, image_shape[0] / 8.0),
            k=120, robust_iters=0,
        )
    else:
        pred = fit_vector_field_2d(
            x, y, dx, dy, backend="grid",
            image_shape=image_shape, grid_shape=(16, 12),
            min_per_cell=4, smooth_sigma=1.0,
        )
    dx_p, dy_p = pred(x, y)
    # Smoothed prediction should sit close to the noiseless truth.
    assert np.std(dx_p - dx_t) < 0.04
    assert np.std(dy_p - dy_t) < 0.04


def test_fit_vector_field_2d_scalar_field():
    x, y, dx, _, _, _, image_shape = _smooth_field_samples()
    pred = fit_vector_field_2d(
        x, y, dx, backend="grid",
        image_shape=image_shape, grid_shape=(12, 12),
    )
    out = pred(x[:5], y[:5])
    # Scalar field returns a single ndarray, not a tuple.
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)


def test_fit_vector_field_2d_grid_matches_reference():
    """The grid backend should reproduce the reference implementation
    derived from test_3_forced_catalog_phase.fit_residual_grid bit-for-bit."""
    from scipy.interpolate import RegularGridInterpolator
    from stdpipe.smoothing import _fit_grid_one
    x, y, dx, dy, _, _, image_shape = _smooth_field_samples()
    nx, ny = 16, 12
    x_edges = np.linspace(0.0, image_shape[1], nx + 1)
    y_edges = np.linspace(0.0, image_shape[0], ny + 1)
    interp_dx, _, _, _ = _fit_grid_one(
        x, y, dx, x_edges, y_edges, min_per_cell=4, smooth_sigma=1.0,
    )
    pred = fit_vector_field_2d(
        x, y, dx, dy, backend="grid",
        image_shape=image_shape, grid_shape=(nx, ny),
        min_per_cell=4, smooth_sigma=1.0,
    )
    pred_dx, _ = pred(x[:50], y[:50])
    ref_dx = interp_dx(np.c_[y[:50], x[:50]])
    np.testing.assert_allclose(pred_dx, ref_dx, atol=0, rtol=0)


def test_fit_vector_field_2d_unknown_backend_raises():
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="unknown backend"):
        fit_vector_field_2d(x, x, x, backend="bogus")


def test_fit_vector_field_2d_grid_rejects_loess_kwargs():
    x = np.linspace(0, 100, 100)
    with pytest.raises(TypeError, match="unexpected kwargs"):
        fit_vector_field_2d(
            x, x, x, backend="grid",
            image_shape=(100, 100), robust_iters=2,
        )
