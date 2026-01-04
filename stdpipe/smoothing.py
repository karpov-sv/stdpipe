import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors


def _tukey_bisquare(u: np.ndarray) -> np.ndarray:
    """
    Tukey bisquare weights for standardized residuals u = r / (c * s).
    Returns weights in [0,1].
    """
    w = np.zeros_like(u, dtype=float)
    m = np.abs(u) < 1.0
    t = 1.0 - u[m] ** 2
    w[m] = t ** 2
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
    scales: tuple[float, ...] | None = None # per-dimension scaling for distance metric
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
        self.base_w_ = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
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
                        dists_k[i] = dists[i, 1:k_base + 1]
                        idxs_k[i] = idxs[i, 1:k_base + 1]
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
            Aw = A * sw[:, :, None]              # (m, k, P)
            yw = Yn * sw                          # (m, k)

            ATA = np.einsum("mkp,mkq->mpq", Aw, Aw)  # (m, P, P)
            ATy = np.einsum("mkp,mk->mp", Aw, yw)    # (m, P)

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
