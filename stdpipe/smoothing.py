from typing import Callable, Optional, Tuple, Union

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def _tukey_bisquare(u: np.ndarray) -> np.ndarray:
    """
    Tukey bisquare weights for standardized residuals u = r / (c * s).
    Returns weights in [0,1].
    """
    w = np.zeros_like(u, dtype=float)
    m = np.abs(u) < 1.0
    t = 1.0 - u[m] ** 2
    w[m] = t**2
    return w


def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate via MAD (consistent for normal)."""
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12


@dataclass
class ApproxLoessRegressor:
    """
    Approximate LOESS/LOWESS via kNN local linear regression in D dimensions.

    Supports:
    - adaptive bandwidth (h = dist to k-th neighbor)
    - Gaussian kernel
    - robust IRLS using Tukey bisquare

    Typical use: model smooth trend y = f(x, y, mag) and subtract.
    """

    k: int = 300
    scales: tuple[float, ...] | None = None  # per-dimension scaling for distance metric
    kernel: str = "gaussian"  # currently only gaussian
    robust_iters: int = 2
    robust_c: float = 4.685  # Tukey tuning constant
    min_bandwidth: float = 1e-6
    ridge: float = 1e-10  # tiny Tikhonov for numerical stability
    leaf_size: int = 40
    n_jobs: int = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """
        Fit stores training set and builds NN index.
        X: (N, D)
        y: (N,)
        sample_weight: optional base weights (e.g., inverse variance). Robust weights are applied on top.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be (N, D)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have length N")

        D = X.shape[1]
        if self.scales is None:
            self.scales = np.ones(D, dtype=float)
        elif len(self.scales) != D:
            raise ValueError(f"scales must have length D={D}")

        self.X_ = X
        self.y_ = y
        self.base_w_ = (
            np.ones_like(y)
            if sample_weight is None
            else np.asarray(sample_weight, float).reshape(-1)
        )
        if self.base_w_.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have length N")
        if np.any(self.base_w_ < 0):
            raise ValueError("sample_weight must be non-negative")

        self.scales_ = np.asarray(self.scales, float)
        self.Xs_ = self.X_ / self.scales_  # scaled for distance computations
        self.n_samples_ = self.X_.shape[0]
        self.k_ = min(self.k, self.n_samples_)

        self.nn_ = NearestNeighbors(
            n_neighbors=self.k_,
            algorithm="auto",
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(self.Xs_)
        return self

    def predict(self, Xq: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """
        Predict yhat for query points Xq using local linear LOESS.
        """
        Xq = np.asarray(Xq, float)
        if Xq.ndim != 2:
            raise ValueError("Xq must be (M, D)")
        if Xq.shape[1] != self.X_.shape[1]:
            raise ValueError("Xq must have same D as training X")

        M, D = Xq.shape
        Xqs = Xq / self.scales_

        yhat = np.empty(M, float)

        # Robust IRLS: we compute robust weights based on residuals at training points
        # by LOESS self-prediction. This is optional but very useful to downweight artifacts
        # in the training set.
        robust_w_train = np.ones_like(self.y_, float)

        for it in range(max(1, self.robust_iters + 1)):
            # For stability: during IRLS, we predict at TRAIN points to update residual weights.
            # Final iteration uses those weights to predict at query points.
            if it < self.robust_iters:
                yfit_train = self._predict_core(
                    self.X_, self.Xs_, robust_w_train, chunk=chunk, exclude_self=True
                )
                resid = self.y_ - yfit_train
                s = _mad_sigma(resid)
                u = resid / (self.robust_c * s)
                robust_w_train = _tukey_bisquare(u)
            else:
                yhat[:] = self._predict_core(Xq, Xqs, robust_w_train, chunk=chunk)

        return yhat

    def _predict_core(
        self,
        Xq: np.ndarray,
        Xqs: np.ndarray,
        robust_w_train: np.ndarray,
        chunk: int = 4096,
        exclude_self: bool = False,
    ) -> np.ndarray:
        """
        Core prediction with fixed robust weights for training points.
        Uses local linear regression with weights = kernel(distance/h) * base_w * robust_w.
        If exclude_self is True, the nearest self-neighbor is removed where possible.
        """
        M, D = Xq.shape
        out = np.empty(M, float)

        # Query neighbors
        for start in range(0, M, chunk):
            end = min(M, start + chunk)
            k_base = self.k_
            k_query = k_base + 1 if exclude_self and self.n_samples_ > k_base else k_base
            dists, idxs = self.nn_.kneighbors(
                Xqs[start:end], n_neighbors=k_query, return_distance=True
            )  # (m, k_query)
            if exclude_self and k_query > k_base:
                m = end - start
                dists_k = np.empty((m, k_base), float)
                idxs_k = np.empty((m, k_base), int)
                for i in range(m):
                    if dists[i, 0] == 0.0:
                        dists_k[i] = dists[i, 1 : k_base + 1]
                        idxs_k[i] = idxs[i, 1 : k_base + 1]
                    else:
                        dists_k[i] = dists[i, 0:k_base]
                        idxs_k[i] = idxs[i, 0:k_base]
                dists, idxs = dists_k, idxs_k
            # adaptive bandwidth per query point: h = distance to furthest neighbor
            h = np.maximum(dists[:, -1], self.min_bandwidth)  # (m,)
            # kernel weights
            if self.kernel != "gaussian":
                raise NotImplementedError("Only gaussian kernel is implemented")

            # Gaussian kernel with adaptive bandwidth:
            # w_kernel = exp(-0.5*(d/h)^2)
            z = dists / h[:, None]
            w_kernel = np.exp(-0.5 * z * z)
            if exclude_self and k_query == k_base:
                w_kernel = np.where(dists == 0.0, 0.0, w_kernel)

            # Pull neighbor values
            Yn = self.y_[idxs]  # (m, k)
            # Combine weights: base * robust * kernel
            w = w_kernel * (self.base_w_[idxs] * robust_w_train[idxs])  # (m, k)

            # Local linear regression around each query:
            # y ≈ b0 + b1*(x-xq) + b2*(y-yq) + b3*(mag-magq) ...
            # Design matrix per query: A = [1, dX] with shape (k, 1+D)
            Xn = self.X_[idxs]  # (m, k, D)
            dX = Xn - Xq[start:end, None, :]  # (m, k, D)

            # Build weighted normal equations per query:
            # Beta = (A^T W A + ridge*I)^(-1) (A^T W y)
            # where A = [1, dX]
            m = end - start
            P = 1 + D
            # A: (m, k, P)
            A = np.empty((m, k_base, P), float)
            A[:, :, 0] = 1.0
            A[:, :, 1:] = dX

            # Compute ATA and ATy with vectorized einsum
            # Apply weights by multiplying rows of A and y by sqrt(w)
            sw = np.sqrt(np.maximum(w, 0.0))
            Aw = A * sw[:, :, None]  # (m, k, P)
            yw = Yn * sw  # (m, k)

            ATA = np.einsum("mkp,mkq->mpq", Aw, Aw)  # (m, P, P)
            ATy = np.einsum("mkp,mk->mp", Aw, yw)  # (m, P)

            # Ridge for numerical stability (especially in sparse regions)
            ATA[:, range(P), range(P)] += self.ridge

            # Solve per-query small linear system
            # We need b0 only (intercept) because dX=0 at query point.
            # Use np.linalg.solve in a loop over m (P is small: 4 when D=3).
            b0 = np.empty(m, float)
            for i in range(m):
                beta = np.linalg.solve(ATA[i], ATy[i])
                b0[i] = beta[0]
            out[start:end] = b0

        return out


def _fit_loess_field_2d(x, y, dx, dy, scales, k, **kwargs):
    pos = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    model_dx = ApproxLoessRegressor(k=k, scales=scales, **kwargs)
    model_dx.fit(pos, np.asarray(dx, float))
    if dy is None:
        def predict(xq, yq):
            q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
            return model_dx.predict(q)
        return predict
    model_dy = ApproxLoessRegressor(k=k, scales=scales, **kwargs)
    model_dy.fit(pos, np.asarray(dy, float))
    def predict(xq, yq):
        q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
        return model_dx.predict(q), model_dy.predict(q)
    return predict


def _fill_grid_nearest(values, valid, x_centers, y_centers):
    filled = np.array(values, copy=True)
    if np.all(valid):
        return filled
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    pts_valid = np.c_[xx[valid], yy[valid]]
    pts_missing = np.c_[xx[~valid], yy[~valid]]
    tree = cKDTree(pts_valid)
    _, idx = tree.query(pts_missing, k=1)
    filled[~valid] = values[valid][idx]
    return filled


def _smooth_grid(values, weights, sigma):
    if sigma <= 0:
        return values
    num = gaussian_filter(values * weights, sigma=sigma, mode="nearest")
    den = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.array(values, copy=True)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _fit_grid_one(x, y, vals, x_edges, y_edges, min_per_cell, smooth_sigma):
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    grid = np.full((ny, nx), np.nan, dtype=float)
    counts = np.zeros((ny, nx), dtype=float)
    threshold = max(1, int(min_per_cell))
    valid = None
    while True:
        grid.fill(np.nan)
        counts.fill(0.0)
        # Group sample indices by 1-D bin id, then per-bin nanmedian
        bin_id = iy * nx + ix
        order = np.argsort(bin_id, kind="stable")
        bin_sorted = bin_id[order]
        vals_sorted = vals[order]
        starts = np.searchsorted(bin_sorted, np.arange(ny * nx))
        ends = np.searchsorted(bin_sorted, np.arange(ny * nx) + 1)
        for b, (s, e) in enumerate(zip(starts, ends)):
            if e - s >= threshold:
                jy, jx = divmod(b, nx)
                grid[jy, jx] = np.nanmedian(vals_sorted[s:e])
                counts[jy, jx] = e - s
        valid = np.isfinite(grid) & (counts > 0)
        if np.any(valid) or threshold == 1:
            break
        threshold = max(1, threshold // 2)

    if not np.any(valid):
        raise RuntimeError("No grid cells have any samples")

    grid = _fill_grid_nearest(grid, valid, x_centers, y_centers)
    weights = np.where(valid, counts, 0.0)
    grid = _smooth_grid(grid, weights, smooth_sigma)
    interp = RegularGridInterpolator(
        (y_centers, x_centers), grid,
        bounds_error=False, fill_value=None,
    )
    return interp, grid, counts, threshold


def _fit_grid_field_2d(
    x, y, dx, dy, image_shape, grid_shape, min_per_cell, smooth_sigma,
):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = np.asarray(dx, float)
    if image_shape is None:
        H = float(np.max(y)) - float(np.min(y))
        W = float(np.max(x)) - float(np.min(x))
        x0, y0 = float(np.min(x)), float(np.min(y))
        x_edges = np.linspace(x0, x0 + W, grid_shape[0] + 1)
        y_edges = np.linspace(y0, y0 + H, grid_shape[1] + 1)
    else:
        x_edges = np.linspace(0.0, image_shape[1], grid_shape[0] + 1)
        y_edges = np.linspace(0.0, image_shape[0], grid_shape[1] + 1)

    interp_dx, _, _, _ = _fit_grid_one(
        x, y, dx, x_edges, y_edges, min_per_cell, smooth_sigma,
    )
    if dy is None:
        def predict(xq, yq):
            pts = np.c_[np.asarray(yq, float), np.asarray(xq, float)]
            return np.asarray(interp_dx(pts), float)
        return predict
    interp_dy, _, _, _ = _fit_grid_one(
        x, y, np.asarray(dy, float), x_edges, y_edges, min_per_cell, smooth_sigma,
    )
    def predict(xq, yq):
        pts = np.c_[np.asarray(yq, float), np.asarray(xq, float)]
        return (np.asarray(interp_dx(pts), float),
                np.asarray(interp_dy(pts), float))
    return predict


def fit_vector_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: Optional[np.ndarray] = None,
    *,
    backend: str = "loess",
    scales: Optional[Tuple[float, float]] = None,
    k: int = 200,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_shape: Tuple[int, int] = (12, 8),
    min_per_cell: int = 6,
    smooth_sigma: float = 1.0,
    **kwargs,
) -> Callable[[np.ndarray, np.ndarray],
              Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Fit a smooth scalar or vector 2-D field to scattered samples.

    Reconstructs a 2-D field (e.g. astrometric (dx, dy) residuals) from
    per-source positions and per-source measurements and returns a callable
    that evaluates the smoothed field at arbitrary positions.

    Two backends are available with the same return interface:

    * ``backend="loess"`` (default) wraps :class:`ApproxLoessRegressor`.
      High-quality local-linear smoothing with adaptive bandwidth and
      optional robust IRLS. Best when prediction is needed at modest
      numbers of points (a few times the fit size).
    * ``backend="grid"`` bins the samples on a regular ``grid_shape`` grid,
      takes per-cell medians, fills empty cells from the nearest filled
      cell, optionally Gaussian-smooths in cell units, and returns a
      bilinear interpolator. ~600–1000× faster at prediction than LOESS,
      at the cost of cell-scale resolution and blockier output.

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Sample positions.
    dx : array-like, shape (N,)
        Sample values, or first component of a vector field if ``dy`` is
        also provided.
    dy : array-like, shape (N,), optional
        Second component of a vector field. When given, the returned
        ``predict`` callable evaluates both components at once.
    backend : {"loess", "grid"}
        Smoothing backend.
    scales : (sx, sy) tuple, optional
        LOESS only. Per-axis distance scaling forwarded to
        :class:`ApproxLoessRegressor`. Defaults to ``(1.0, 1.0)``.
    k : int
        LOESS only. Neighbour count for each local linear fit.
    image_shape : (H, W) tuple, optional
        Grid only. Image shape used to lay out the grid edges. If omitted,
        the bounding box of the input ``(x, y)`` is used.
    grid_shape : (nx, ny) tuple
        Grid only. Number of cells in x and y.
    min_per_cell : int
        Grid only. Minimum sample count per cell required for a valid
        median; cells below the threshold are filled from the nearest
        valid neighbour. Threshold is halved automatically until at least
        one cell is valid.
    smooth_sigma : float
        Grid only. Gaussian smoothing sigma in cell units applied to the
        gridded medians (count-weighted).
    **kwargs :
        LOESS only. Additional keyword arguments forwarded to
        :class:`ApproxLoessRegressor` (``robust_iters``, ``robust_c``, ...).

    Returns
    -------
    predict : callable
        ``predict(xq, yq)`` returns a single ``ndarray`` for a scalar
        field, or a ``(dx_pred, dy_pred)`` tuple for a vector field.
    """
    if backend == "loess":
        if scales is None:
            scales = (1.0, 1.0)
        return _fit_loess_field_2d(x, y, dx, dy, scales, k, **kwargs)
    elif backend == "grid":
        if kwargs:
            raise TypeError(
                f"unexpected kwargs for grid backend: {sorted(kwargs)}"
            )
        return _fit_grid_field_2d(
            x, y, dx, dy, image_shape, grid_shape, min_per_cell, smooth_sigma,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'loess' or 'grid'")
