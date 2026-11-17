"""
Background estimation routines for astronomical images.

This module provides various methods for estimating and subtracting background
from astronomical images, including grid-based methods (SEP, photutils) and
non-grid methods (percentile filtering, morphological opening).
"""

import numpy as np

from astropy.stats import mad_std


def get_background_percentile(
    image, mask=None, size=51, percentile=25.0, maxiters=3, sigma=3.0, use_mode=False
):
    """
    Background estimation via percentile filtering (non-grid method).

    Uses a sliding window to compute low percentile value at each pixel.
    Stars (positive outliers) are naturally above the percentile and thus rejected.

    This is a truly non-grid method: operates on all pixels with sliding window,
    not limited by grid cell resolution.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels to exclude)
    size : int
        Filter kernel size (should be larger than typical star size, e.g., 51-101 pixels).
        Larger values are more robust but may miss fine background structure.
    percentile : float
        Percentile value to use (0-100). Typical values:
        - 25: 25th percentile (default, good for moderate star density)
        - 10: More aggressive star rejection (for crowded fields)
        - 50: Median (less star rejection but more robust to noise)
    maxiters : int
        Number of sigma-clipping iterations to refine percentile estimate
    sigma : float
        Sigma threshold for clipping
    use_mode : bool
        If True, estimate mode instead of percentile (experimental)

    Returns
    -------
    background : ndarray
        Estimated background map

    Notes
    -----
    - Computationally intensive for large kernels (scales as O(N * kernel^2))
    - Consider using smaller kernels (31-51) for large images
    - For very crowded fields, use lower percentiles (10-15)
    - Handles arbitrary background shapes (not limited to polynomial forms)
    """
    from scipy.ndimage import percentile_filter, median_filter

    # Work with a copy to avoid modifying input
    img = image.copy()

    # Apply mask by replacing with NaN (will be ignored in percentile computation)
    has_nan = False
    if mask is not None:
        img[mask] = np.nan
        has_nan = True

    # Fix #3: Define masked_percentile closure once before the loop
    def masked_percentile(values):
        valid = values[~np.isnan(values)]
        if len(valid) < 3:
            return np.nan
        return np.percentile(valid, percentile)

    # Iterative sigma-clipping approach
    for iteration in range(maxiters):
        if use_mode:
            # Mode estimation via histogram (in sliding window)
            # This is expensive, so use median as approximation
            back = median_filter(img, size=size)
        else:
            # Fix #2 & #4: Use fast percentile_filter with NaN-filling approach
            # instead of slow generic_filter with Python callback
            if has_nan:
                # Two-pass approach: fill NaN with local median, then run fast filter
                # Pass 1: Fill NaN pixels with local median
                img_filled = img.copy()
                nan_mask = np.isnan(img_filled)
                if np.any(nan_mask):
                    # Use median filter to get fill values
                    # Temporarily replace NaN with image median for the filter
                    fill_val = np.nanmedian(img_filled)
                    img_filled[nan_mask] = fill_val
                    local_med = median_filter(img_filled, size=min(size, 15))
                    img_filled[nan_mask] = local_med[nan_mask]

                # Pass 2: Run fast C-based percentile_filter on the filled image
                back = percentile_filter(img_filled, percentile=percentile, size=size)
            else:
                back = percentile_filter(img, percentile=percentile, size=size)

        # Sigma-clip for next iteration
        if iteration < maxiters - 1:
            residual = img - back
            # Estimate RMS from negative residuals (not contaminated by stars)
            neg_residuals = residual[residual < 0]
            if len(neg_residuals) > 100:
                rms = np.std(neg_residuals) * np.sqrt(2)  # Correct for one-sided
            else:
                rms = mad_std(residual[np.isfinite(residual)])

            # Clip positive outliers (stars)
            outliers = residual > sigma * rms
            img[outliers] = np.nan
            if np.any(outliers):
                has_nan = True

    # Fill any remaining NaN values
    if np.any(np.isnan(back)):
        valid = np.isfinite(back)
        if np.sum(valid) > 0:
            # Use median of valid pixels
            fill_value = np.median(back[valid])
            back[~valid] = fill_value

    return back


def get_background_morphology(
    image, mask=None, size=25, iterations=1, smooth=True, smooth_size=None
):
    """
    Background estimation via morphological opening (non-grid method).

    Performs morphological opening (erosion followed by dilation) which removes
    structures smaller than the structuring element (stars). Fast and simple.

    This is a truly non-grid method: operates on all pixels with sliding window.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels to exclude)
    size : int
        Size of structuring element (should be larger than typical star size).
        Typical values: 15-35 pixels depending on seeing.
    iterations : int
        Number of opening iterations (default: 1, more iterations = more aggressive)
    smooth : bool
        If True, apply additional smoothing to reduce kernel-sized artifacts.
        This significantly improves the smoothness of the result.
    smooth_size : int, optional
        Size of smoothing kernel in pixels. If None, automatically set based on
        the structuring element size. Use larger values for smoother results.

    Returns
    -------
    background : ndarray
        Estimated background map

    Notes
    -----
    - Very fast (much faster than percentile filtering)
    - Simple and robust
    - Can create kernel-sized artifacts without smoothing
    - May suppress legitimate background features similar in size to stars
    - Best for images with clear point sources on smooth backgrounds
    - Use smooth=True (default) to eliminate spotty kernel artifacts
    """
    from scipy.ndimage import grey_opening, median_filter, gaussian_filter

    # Work with a copy
    img = image.copy()

    # Handle mask by replacing with local median
    if mask is not None:
        # Fill masked regions with local median to avoid artifacts
        if np.any(mask):
            # Simple approach: use median filter to fill masked regions
            img_filled = median_filter(img, size=5)
            img[mask] = img_filled[mask]

    # Morphological opening with circular structuring element
    # Create circular structuring element
    y, x = np.ogrid[-size // 2 : size // 2 + 1, -size // 2 : size // 2 + 1]
    structure = (x * x + y * y) <= (size / 2) ** 2

    # Apply opening
    back = grey_opening(img, structure=structure)

    # Additional iterations if requested
    for i in range(iterations - 1):
        back = grey_opening(back, structure=structure)

    # Apply smoothing to reduce kernel-sized artifacts
    # Fix #6: smooth_size consistently means filter size in pixels for both paths
    if smooth:
        if mask is not None and np.any(mask):
            # Use median filter for masked images (more robust to artifacts)
            if smooth_size is None:
                filter_size = max(5, size // 5)
            else:
                filter_size = int(smooth_size)
            back = median_filter(back, size=filter_size)
        else:
            # Use Gaussian smoothing for clean images (smoother results)
            if smooth_size is None:
                # Default: adaptive smoothing based on kernel size
                sigma_val = max(2.0, size / 5.0)
            else:
                # Convert filter size to sigma (size ~ 3*sigma is a reasonable heuristic)
                sigma_val = smooth_size / 3.0
            back = gaussian_filter(back, sigma=sigma_val, mode='reflect')

    return back


def estimate_background_rms_percentile(image, mask=None, size=51):
    """
    Estimate background RMS using percentile range.

    Uses interquartile range (IQR) as robust RMS estimator.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels)
    size : int
        Kernel size for local percentile estimation

    Returns
    -------
    rms : ndarray
        Background RMS map
    """
    from scipy.ndimage import percentile_filter, median_filter

    # Fix #4: Replace slow generic_filter with fast two-pass approach
    if mask is not None:
        img = image.copy()
        img[mask] = np.nan

        # Two-pass: fill NaN with local median, then run fast percentile_filter
        nan_mask = np.isnan(img)
        if np.any(nan_mask):
            fill_val = np.nanmedian(img)
            img_filled = img.copy()
            img_filled[nan_mask] = fill_val
            local_med = median_filter(img_filled, size=min(size, 15))
            img_filled[nan_mask] = local_med[nan_mask]
        else:
            img_filled = img

        # Fix #9: Always use p25/p75 for IQR regardless of caller's percentile
        p25 = percentile_filter(img_filled, percentile=25, size=size)
        p75 = percentile_filter(img_filled, percentile=75, size=size)
    else:
        p25 = percentile_filter(image, percentile=25, size=size)
        p75 = percentile_filter(image, percentile=75, size=size)

    # IQR / 1.349 ≈ sigma for Gaussian
    rms = (p75 - p25) / 1.349

    return rms


def estimate_background_rms_local(image, background, mask=None, size=51):
    """
    Estimate background RMS from residuals.

    Computes local standard deviation of (image - background).

    Parameters
    ----------
    image : ndarray
        Input image
    background : ndarray
        Estimated background
    mask : ndarray, optional
        Boolean mask (True = masked pixels)
    size : int
        Kernel size for local RMS estimation

    Returns
    -------
    rms : ndarray
        Background RMS map
    """
    from scipy.ndimage import uniform_filter

    residual = image - background

    if mask is not None:
        residual = residual.copy()
        residual[mask] = np.nan

    # Fix #4: Replace slow generic_filter with fast uniform_filter-based local std
    # For masked data, fill NaN first then use fast filters
    has_nan = np.any(np.isnan(residual))

    if has_nan:
        nan_mask = np.isnan(residual)
        res_filled = residual.copy()
        fill_val = np.nanmedian(res_filled)
        res_filled[nan_mask] = fill_val

        # Local mean and mean of squares via uniform_filter
        local_mean = uniform_filter(res_filled, size=size, mode='reflect')
        local_mean_sq = uniform_filter(res_filled**2, size=size, mode='reflect')
    else:
        local_mean = uniform_filter(residual, size=size, mode='reflect')
        local_mean_sq = uniform_filter(residual**2, size=size, mode='reflect')

    # Var = E[X^2] - E[X]^2
    local_var = local_mean_sq - local_mean**2
    # Clamp negative values from numerical precision
    local_var = np.maximum(local_var, 0.0)
    rms = np.sqrt(local_var)

    # Fill NaN values
    if np.any(np.isnan(rms)):
        valid = np.isfinite(rms)
        if np.sum(valid) > 0:
            fill_value = np.median(rms[valid])
            rms[~valid] = fill_value

    return rms


def get_background_polynomial(
    image,
    mask=None,
    order=1,
    sigma=3.0,
    maxiters=5,
    directional=False,
    direction=None,
):
    """Estimate background as a low-order 2D polynomial.

    Fits a smooth polynomial surface to the image using iterative
    sigma-clipping to reject stars and other outliers.  By construction,
    low-order polynomials cannot represent localised features such as
    nebulae, so only the smooth gradient component is captured.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    mask : 2D bool ndarray, optional
        Pixels to exclude (True = masked).
    order : int
        Polynomial order: 1 = plane, 2 = quadratic, 3 = cubic, etc.
        Default 1.
    sigma : float
        Sigma threshold for iterative outlier clipping.  Default 3.0.
    maxiters : int
        Maximum number of sigma-clipping iterations.  Default 5.
    directional : bool or float
        Controls directional constraint for higher-order (≥ 2) terms.

        - ``False`` (default) — use all 2D polynomial terms.
        - ``True`` — auto-detect the gradient direction from a linear
          fit, then restrict order ≥ 2 terms to powers of the
          directional coordinate *t* = *x* cos θ + *y* sin θ.
          This prevents high-order polynomials from fitting localised
          features while still capturing curvature along the gradient.
        - A ``float`` value — same as ``True`` but uses the given angle
          (degrees, 0 = x-axis, 90 = y-axis) instead of auto-detecting.

        For order ≤ 1 this parameter has no effect (a plane already
        has only 3 coefficients).
    direction : float or None
        *Deprecated* synonym for ``directional=<angle>``.  If both are
        given, *directional* takes precedence.

    Returns
    -------
    background : 2D ndarray
        Fitted polynomial background, same shape as *image*.
    rms : float
        Scalar RMS of the residuals (clipped pixels excluded).

    Notes
    -----
    **Full 2D mode** (``directional=False``):

    Number of polynomial coefficients = (order+1)(order+2)/2:
    order 1 → 3, order 2 → 6, order 3 → 10, order 4 → 15.

    **Directional mode** (``directional=True`` or angle):

    Number of coefficients = order + 2 (full linear + higher-order
    directional terms): order 2 → 4, order 3 → 5, order 4 → 6.

    Coordinates are normalised to [-1, 1] internally for numerical
    stability.  The fit uses random sub-sampling (100 000 points) for
    images larger than ~300×300 to keep memory and CPU bounded.

    Examples
    --------
    Remove a linear gradient:

    >>> bg, rms = get_background_polynomial(image, order=1)

    Fit a cubic curve along the gradient direction (5 parameters):

    >>> bg, rms = get_background_polynomial(image, order=3, directional=True)

    Specify the gradient direction explicitly (e.g. 45°):

    >>> bg, rms = get_background_polynomial(image, order=3, directional=45.0)
    """
    ny, nx = image.shape

    # --- resolve directional / direction arguments -----------------------
    if isinstance(directional, (int, float)) and not isinstance(directional, bool):
        # directional=45.0  →  direction angle in degrees
        direction_deg = float(directional)
        directional = True
    elif directional and direction is not None:
        direction_deg = float(direction)
    elif direction is not None:
        direction_deg = float(direction)
        directional = True
    else:
        direction_deg = None  # auto-detect later if directional=True

    # Directional constraint only matters for order >= 2
    if order < 2:
        directional = False

    # --- build a fixed-size working sample -------------------------------
    # All fitting and sigma-clipping operates on this subsample only;
    # the full-grid evaluation is a single pass at the end.
    MAX_SAMPLE = 100_000

    data_flat = image.ravel()
    valid_idx = np.where(np.isfinite(data_flat))[0]
    if mask is not None:
        valid_idx = valid_idx[~mask.ravel()[valid_idx]]

    if len(valid_idx) == 0:
        return np.full_like(image, np.nan, dtype=np.float64), np.nan

    # Subsample once — reused for every clipping iteration
    if len(valid_idx) > MAX_SAMPLE:
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(valid_idx, MAX_SAMPLE, replace=False)
    else:
        sample_idx = valid_idx

    # Coordinates and data for the working sample (normalised to [-1, 1])
    x_scale = max(nx / 2, 1)
    y_scale = max(ny / 2, 1)
    sample_y, sample_x = np.divmod(sample_idx, nx)
    xs = (sample_x.astype(np.float64) - nx / 2) / x_scale
    ys = (sample_y.astype(np.float64) - ny / 2) / y_scale
    ds = data_flat[sample_idx].astype(np.float64)

    # --- helpers ---------------------------------------------------------
    def _poly_terms(ord_):
        """Return list of (p, q) exponent pairs for full 2D polynomial."""
        return [(p, q) for p in range(ord_ + 1) for q in range(ord_ + 1 - p)]

    terms = _poly_terms(order)

    def _build_basis(x, y, theta_rad):
        """Build design matrix for the current model."""
        if directional:
            t = x * np.cos(theta_rad) + y * np.sin(theta_rad)
            cols = [np.ones_like(x), x, y]
            tk = t.copy()
            for _ in range(2, order + 1):
                tk = tk * t
                cols.append(tk)
            return np.column_stack(cols)
        cols = []
        for p, q in terms:
            col = np.ones_like(x)
            if p:
                col = col * x**p
            if q:
                col = col * y**q
            cols.append(col)
        return np.column_stack(cols)

    def _eval_2d(coeffs, theta_rad):
        """Evaluate polynomial on the full pixel grid via broadcasting."""
        x_1d = (np.arange(nx, dtype=np.float64) - nx / 2) / x_scale
        y_1d = (np.arange(ny, dtype=np.float64) - ny / 2) / y_scale
        bg = np.full((ny, nx), coeffs[0], dtype=np.float64)
        if directional:
            bg += coeffs[1] * x_1d[np.newaxis, :]
            bg += coeffs[2] * y_1d[:, np.newaxis]
            t = np.cos(theta_rad) * x_1d[np.newaxis, :] + np.sin(theta_rad) * y_1d[:, np.newaxis]
            tk = t.copy()
            for k in range(2, order + 1):
                tk = tk * t
                bg += coeffs[k + 1] * tk
        else:
            idx = 0
            for p, q in terms:
                if idx == 0:
                    idx += 1
                    continue  # constant already handled
                term = np.ones((ny, nx), dtype=np.float64)
                if p:
                    term *= x_1d[np.newaxis, :] ** p
                if q:
                    term *= y_1d[:, np.newaxis] ** q
                bg += coeffs[idx] * term
                idx += 1
        return bg

    # --- auto-detect gradient direction ----------------------------------
    theta_rad = 0.0
    if directional and direction_deg is None:
        A = np.column_stack([np.ones_like(xs), xs, ys])
        c, _, _, _ = np.linalg.lstsq(A, ds, rcond=None)
        theta_rad = np.arctan2(c[2], c[1])
    elif direction_deg is not None:
        theta_rad = np.radians(direction_deg)

    # --- iterative sigma-clipped polynomial fit --------------------------
    # All iterations work on the fixed subsample (xs, ys, ds).
    # Note: some BLAS implementations (e.g. Apple Accelerate) raise
    # spurious divide-by-zero warnings during matmul/lstsq even when
    # all inputs and outputs are finite.
    good = np.ones(len(xs), dtype=bool)
    coeffs = None
    rms_val = np.nan

    n_params = (order + 2) if directional else len(terms)

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for iteration in range(maxiters + 1):
            n_good = int(np.sum(good))
            if n_good < n_params + 1:
                break

            A = _build_basis(xs[good], ys[good], theta_rad)
            coeffs, _, _, _ = np.linalg.lstsq(A, ds[good], rcond=None)

            if iteration == maxiters:
                break

            # Evaluate model on the subsample for clipping
            model = _build_basis(xs, ys, theta_rad) @ coeffs
            res = ds - model

            # Robust RMS via MAD
            med_res = np.median(res)
            rms_val = 1.4826 * np.median(np.abs(res - med_res))
            if rms_val <= 0:
                break

            good = np.abs(res - med_res) < sigma * rms_val

    # --- final evaluation on the full grid (single pass) -----------------
    if coeffs is None:
        return np.full_like(image, np.nan, dtype=np.float64), np.nan

    background = _eval_2d(coeffs, theta_rad)

    # Final RMS from clipped subsample residuals
    if np.any(good) and np.all(np.isfinite(coeffs)):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            model_good = _build_basis(xs[good], ys[good], theta_rad) @ coeffs
        rms_val = float(mad_std(ds[good] - model_good))
    else:
        rms_val = np.nan

    return background, rms_val


def get_background_gp(
    image,
    mask=None,
    max_points=3000,
    grid_step=None,
    clip_sigma=3.5,
    n_clip_iter=2,
    length_scale=64.0,
    length_scale_bounds=(8.0, 512.0),
    matern_nu=1.5,
    white_noise=1.0,
    white_noise_bounds=(1e-3, 1e3),
    normalize_y=True,
    n_restarts_optimizer=1,
    random_state=0,
    get_uncertainty=False,
    chunk_rows=256,
):
    """
    Background estimation via Gaussian Process regression (non-grid method).

    Uses a Gaussian Process with a Matern kernel to model the background as a
    smooth function of spatial coordinates (x, y). This method is very flexible
    and can handle complex background variations including gradients, vignetting,
    and other large-scale structures.

    Workflow:
      1. Select candidate background pixels (masked + sigma clipping to remove stars)
      2. Subsample them randomly or on a grid (to keep computation tractable)
      3. Fit a GP to the (x, y) -> background mapping
      4. Predict background map everywhere (with optional uncertainty)

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels to exclude)
    max_points : int
        Maximum number of training points for the GP. More points = better fit
        but much slower (O(N^3) cost). Typical values: 1000-5000.
    grid_step : int, optional
        If set, sample every N pixels on a grid (after masking). If None,
        random sampling is used.
    clip_sigma : float
        Sigma threshold for iterative clipping of outliers (stars) in intensity space
    n_clip_iter : int
        Number of sigma-clipping iterations to remove stars
    length_scale : float
        Initial guess for GP length scale in pixels. Should be larger than
        typical star size. Typical values: 32-128 pixels.
    length_scale_bounds : tuple
        Bounds for length scale optimization (min, max)
    matern_nu : float
        Smoothness parameter for Matern kernel. Typical values:
        - 0.5: Exponential kernel (rough)
        - 1.5: Once differentiable (default, good balance)
        - 2.5: Twice differentiable (very smooth)
    white_noise : float
        Initial guess for white noise level (in image units, e.g., ADU)
    white_noise_bounds : tuple
        Bounds for white noise optimization (min, max)
    normalize_y : bool
        If True, normalize target values during GP training (recommended)
    n_restarts_optimizer : int
        Number of random restarts for hyperparameter optimization (more = better
        but slower)
    random_state : int
        Random seed for reproducibility
    get_uncertainty : bool
        If True, return uncertainty map. If False, return None for RMS.
    chunk_rows : int
        Process prediction in chunks of this many rows to reduce memory usage

    Returns
    -------
    background : ndarray
        Estimated background map
    uncertainty : ndarray or None
        If get_uncertainty=True: 2D array of GP posterior standard deviation
        If get_uncertainty=False: None

    Notes
    -----
    - Requires scikit-learn to be installed
    - Computationally expensive for large images (use smaller max_points)
    - Very flexible: can handle arbitrary background shapes
    - Length scale should be >> PSF size (otherwise may fit stars)
    - For very large images, consider using grid_step to reduce training points

    Examples
    --------
    Basic usage with default parameters:

    >>> bg = get_background_gp(image, mask=mask)

    Custom length scale for wide-field imaging:

    >>> bg = get_background_gp(image, mask=mask, length_scale=128.0)

    Get uncertainty map:

    >>> bg, bg_std = get_background_gp(image, mask=mask, get_uncertainty=True)

    Faster computation with grid sampling:

    >>> bg = get_background_gp(image, mask=mask, max_points=1000, grid_step=10)
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError:
        raise ImportError(
            "Gaussian Process background estimation requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    # Prepare image and mask
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be 2D")
    ny, nx = img.shape

    if mask is None:
        mask = np.zeros(img.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {img.shape}")

    # Step 1: Select candidate background pixels via sigma clipping
    valid = ~mask & np.isfinite(img)
    y_vals = img[valid]

    if y_vals.size < 50:
        raise RuntimeError("Too few valid pixels to estimate background")

    # Iterative sigma clipping to remove stars
    # Fix #7: Use mad_std directly instead of inner robust_sigma function
    keep = np.ones(y_vals.shape, dtype=bool)
    for _ in range(max(0, int(n_clip_iter))):
        yy = y_vals[keep]
        if yy.size < 50:
            break
        med = np.median(yy)
        sig = mad_std(yy, ignore_nan=True)
        if not np.isfinite(sig) or sig <= 0:
            break
        lo = med - clip_sigma * sig
        hi = med + clip_sigma * sig
        keep = keep & (y_vals >= lo) & (y_vals <= hi)

    # Get coordinates for kept pixels
    ys_idx, xs_idx = np.nonzero(valid)
    xs_idx = xs_idx[keep]
    ys_idx = ys_idx[keep]
    y_vals = y_vals[keep]

    # Step 2: Subsample training points
    rng = np.random.default_rng(random_state)

    if grid_step is not None and grid_step > 1:
        # Grid-based subsampling
        gx = xs_idx // grid_step
        gy = ys_idx // grid_step
        # Unique grid cells
        key = gx.astype(np.int64) + (gy.astype(np.int64) << 32)
        _, idx = np.unique(key, return_index=True)
        xs_sub, ys_sub, y_sub = xs_idx[idx], ys_idx[idx], y_vals[idx]
    else:
        xs_sub, ys_sub, y_sub = xs_idx, ys_idx, y_vals

    # Further random subsampling if needed
    n_candidates = y_sub.size
    if n_candidates > max_points:
        idx = rng.choice(n_candidates, size=max_points, replace=False)
        xs_sub, ys_sub, y_sub = xs_sub[idx], ys_sub[idx], y_sub[idx]

    # Prepare training data
    X_train = np.column_stack([xs_sub, ys_sub]).astype(float)
    y_train = y_sub.astype(float)

    # Step 3: Build and fit GP
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        nu=matern_nu,
    ) + WhiteKernel(
        noise_level=white_noise,
        noise_level_bounds=white_noise_bounds,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(X_train, y_train)

    # Step 4: Predict full background map in chunks
    background = np.empty((ny, nx), dtype=float)
    if get_uncertainty:
        uncertainty = np.empty((ny, nx), dtype=float)
    else:
        uncertainty = None

    x_coords = np.arange(nx, dtype=float)

    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        rows = np.arange(y0, y1, dtype=float)

        # Create (x, y) coordinate pairs for this chunk
        X_pred = np.column_stack(
            [
                np.tile(x_coords, y1 - y0),
                np.repeat(rows, nx),
            ]
        )

        if get_uncertainty:
            pred, pred_std = gp.predict(X_pred, return_std=True)
            background[y0:y1, :] = pred.reshape((y1 - y0, nx))
            uncertainty[y0:y1, :] = pred_std.reshape((y1 - y0, nx))
        else:
            pred = gp.predict(X_pred, return_std=False)
            background[y0:y1, :] = pred.reshape((y1 - y0, nx))

    # Fix #8: Always return (background, uncertainty_or_None)
    # When get_uncertainty=False, return None so caller can compute RMS if needed
    return background, uncertainty


def get_background(image, mask=None, method='sep', size=128, get_rms=False, **kwargs):
    """
    Estimate background using various methods.

    This is the main interface for background estimation, supporting multiple
    algorithms optimized for different scenarios.

    Parameters
    ----------
    image : ndarray
        Input image
    mask : ndarray, optional
        Boolean mask (True = masked pixels)
    method : str
        Background estimation method:

        - 'sep': SEP grid-based (default) - Fast, accurate for flat/linear backgrounds
        - 'photutils': photutils grid-based - Similar to SEP
        - 'polynomial': Low-order polynomial fit - Captures smooth gradients, preserves nebulae
        - 'percentile': Percentile filtering (non-grid, robust to stars) - Slow but very robust
        - 'morphology': Morphological opening (non-grid, fast) - Good for moderate gradients
        - 'gp': Gaussian Process regression (non-grid, very flexible) - Best for complex backgrounds
    size : int
        Grid size for sep/photutils, or kernel size for percentile/morphology.
        For 'gp' method, this parameter is used as the initial length_scale.
        Typical values:

        - SEP/photutils: 64-128 pixels
        - Percentile: 31-51 pixels (smaller for gradients)
        - Morphology: 15-35 pixels (should be > star size)
        - GP: 32-128 pixels (length scale, should be >> PSF size)
    get_rms : bool
        If True, return (background, background_rms). For GP method with get_rms=True,
        returns the full uncertainty map instead of a scalar RMS.
    **kwargs : dict
        Additional parameters passed to the method:

        - For polynomial: order, sigma, maxiters, directional, direction
          (see get_background_polynomial for full list)
        - For percentile: percentile, maxiters, sigma
        - For morphology: iterations, smooth, smooth_size
        - For gp: max_points, grid_step, clip_sigma, n_clip_iter, matern_nu, etc.
          (see get_background_gp for full list)

    Returns
    -------
    background : ndarray
        Estimated background
    background_rms : ndarray or float, optional
        Background RMS (if get_rms=True). For GP method, this is a 2D uncertainty map.

    Examples
    --------
    Default grid-based background estimation:

    >>> bg = get_background(image, method='sep', size=128)

    Non-grid morphological method for moderate gradients:

    >>> bg = get_background(image, method='morphology', size=25)

    Morphology with extra smoothing to eliminate kernel artifacts:

    >>> bg = get_background(image, method='morphology', size=25, smooth_size=15)

    Percentile method for very crowded fields:

    >>> bg = get_background(image, method='percentile', size=31, percentile=15.0)

    Gaussian Process for complex backgrounds with vignetting:

    >>> bg = get_background(image, method='gp', size=64, max_points=2000)

    Get both background and RMS:

    >>> bg, bg_rms = get_background(image, method='sep', get_rms=True)

    Get GP background with uncertainty map:

    >>> bg, bg_std = get_background(image, method='gp', get_rms=True)

    Polynomial fit for gradient removal (preserves nebulae):

    >>> bg = get_background(image, method='polynomial', order=2)

    Directional gradient with cubic curvature:

    >>> bg = get_background(image, method='polynomial', order=3, directional=True)

    See Also
    --------
    get_background_polynomial : Polynomial fit details
    get_background_percentile : Percentile filtering details
    get_background_morphology : Morphological opening details
    get_background_gp : Gaussian Process regression details

    Notes
    -----
    **Method Selection Guide**:

    - **Flat backgrounds**: Use 'sep' (default) - fastest and most accurate
    - **Linear gradients**: Use 'sep' with size=64-128
    - **Smooth gradients with nebulae**: Use 'polynomial' — only captures the
      gradient, preserves localised background features by construction
    - **Directional gradients**: Use 'polynomial' with ``directional=True`` —
      higher-order terms confined to the gradient direction
    - **Quadratic gradients**: Use 'morphology' or local gradient fitting (bkg_order)
    - **Complex backgrounds** (vignetting, strong curvature): Use 'gp'
    - **Very crowded fields**: Use 'percentile' with percentile=10-15
    - **Large images**: Avoid 'percentile' (very slow), use 'sep' or 'morphology'

    **Performance**:

    - SEP: ~1-3 ms (512x512 image)
    - Morphology: ~300 ms (512x512 image)
    - Percentile: ~2-5 seconds (512x512 image)
    - GP: ~1-10 seconds (512x512 image, depends on max_points)

    For complex backgrounds (strong curvature, vignetting), consider using
    local gradient fitting instead (bkg_order parameter in measure_objects),
    or the GP method for maximum flexibility.
    """
    if method == 'sep':
        # Fix #1: Lazy import
        try:
            import sep_x as sep
        except ImportError:
            import sep

        # Fix #5: Avoid unnecessary copy if already float64 and contiguous
        image = np.ascontiguousarray(image, dtype=np.double)
        bg = sep.Background(image, mask=mask, bw=size, bh=size, **kwargs)

        back, backrms = bg.back(), bg.rms()
    elif method == 'photutils':
        # Fix #1: Lazy import
        import photutils.background

        bg = photutils.background.Background2D(image, size, mask=mask, **kwargs)
        back, backrms = bg.background, bg.background_rms
    elif method == 'percentile':
        back = get_background_percentile(image, mask=mask, size=size, **kwargs)
        # Estimate RMS from percentile range if needed
        # Fix #9: Don't pass **kwargs to RMS estimator (percentile param would
        # incorrectly override the fixed p25/p75 used for IQR)
        if get_rms:
            backrms = estimate_background_rms_percentile(image, mask=mask, size=size)
        else:
            backrms = None
    elif method == 'morphology':
        back = get_background_morphology(image, mask=mask, size=size, **kwargs)
        # Estimate RMS from local statistics if needed
        if get_rms:
            backrms = estimate_background_rms_local(image, back, mask=mask, size=size)
        else:
            backrms = None
    elif method == 'gp':
        # Use size as length_scale if not explicitly provided
        if 'length_scale' not in kwargs:
            kwargs['length_scale'] = float(size)
        # Set get_uncertainty based on get_rms
        kwargs['get_uncertainty'] = get_rms
        back, gp_uncertainty = get_background_gp(image, mask=mask, **kwargs)
        # Fix #8: When get_uncertainty=False, gp_uncertainty is None.
        # Compute RMS via estimate_background_rms_local if get_rms is requested.
        if get_rms:
            if gp_uncertainty is not None:
                backrms = gp_uncertainty
            else:
                backrms = estimate_background_rms_local(image, back, mask=mask, size=size)
        else:
            backrms = None
    elif method == 'polynomial':
        back, poly_rms = get_background_polynomial(image, mask=mask, **kwargs)
        backrms = np.full_like(back, poly_rms) if get_rms else None
    else:
        raise ValueError(f"Unknown background method: {method}")

    if get_rms:
        return back, backrms
    else:
        return back


def _noise_excess_kernel(radius, profile=None):
    """Weights of a sum in a circular aperture of given radius (exact pixel
    overlap), optionally multiplied by a centered profile stamp. Returns
    pixel offsets ``(dy, dx)`` of the non-zero weights, the weights, and the
    half-size of the kernel."""
    from photutils.aperture import CircularAperture

    half = int(np.ceil(radius))
    size = 2 * half + 1
    weights = CircularAperture((half, half), r=radius).to_mask(method='exact').to_image((size, size))

    if profile is not None:
        # Crop or zero-pad the stamp to the kernel size, keeping its center pixel
        ph, pw = profile.shape
        stamp = np.zeros((size, size))
        cy, cx = ph // 2, pw // 2
        y0, y1 = max(cy - half, 0), min(cy + half + 1, ph)
        x0, x1 = max(cx - half, 0), min(cx + half + 1, pw)
        stamp[y0 - (cy - half) : y1 - (cy - half), x0 - (cx - half) : x1 - (cx - half)] = profile[
            y0:y1, x0:x1
        ]
        weights = weights * stamp

    dy, dx = np.nonzero(weights)
    return dy - half, dx - half, weights[dy, dx], half


def _noise_excess_measured(shape, bad, measure, aper, fwhm, spacing, nsamples, seed, get_stats, log):
    """Noise excess measured with the photometry routine *measure* on a grid of empty positions."""
    from astropy.table import Table

    if np.ndim(aper) != 0:
        raise ValueError('Only scalar aper is supported with measure')

    fwhm_s = float(fwhm) if fwhm is not None else None
    radius_pix = float(aper) * fwhm_s if fwhm_s is not None else float(aper)
    dy, dx, _, half = _noise_excess_kernel(radius_pix)

    if spacing is None:
        spacing = 3 * max(radius_pix, fwhm_s or 0.0) + 1

    # Regular grid with random offset, so that the positions are farther from each other
    # than the grouping radius, with sub-pixel jitter to sample all pixel phases
    rng = np.random.default_rng(seed)
    ny, nx = shape
    gx = np.arange(half + 1 + rng.uniform(0, spacing), nx - half - 2, spacing)
    gy = np.arange(half + 1 + rng.uniform(0, spacing), ny - half - 2, spacing)
    x, y = [_.ravel() for _ in np.meshgrid(gx, gy)]
    x = x + rng.uniform(-0.5, 0.5, x.size)
    y = y + rng.uniform(-0.5, 0.5, y.size)

    # Aperture around the nearest pixel must not touch masked pixels
    if x.size:
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        good = ~np.any(bad[yi[:, None] + dy, xi[:, None] + dx], axis=1)
        x, y = x[good], y[good]
    idx = rng.permutation(x.size)[:nsamples]
    x, y = x[idx], y[idx]

    z = np.zeros(0)
    if x.size >= 50:
        res = measure(Table({'x': x, 'y': y}))
        flux = np.asarray(res['flux'], dtype=float)
        fluxerr = np.asarray(res['fluxerr'], dtype=float)
        good = np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
        z = flux[good] / fluxerr[good]

    if z.size < 50:
        log(
            'Only %d unmasked positions of radius %.1f px properly measured, cannot estimate noise excess'
            % (z.size, radius_pix)
        )
        excess, stats = np.nan, {'nsamples': int(z.size), 'excess_err': np.nan, 'median': np.nan}
    else:
        excess = mad_std(z)
        stats = {
            'nsamples': int(z.size),
            'excess_err': 1.17 * excess / np.sqrt(z.size),
            'median': np.median(z),
        }
        log(
            'Noise excess measured by photometry with radius %.1f px: %.3f +/- %.3f over %d positions, median %.2f'
            % (radius_pix, excess, stats['excess_err'], z.size, stats['median'])
        )

    if get_stats:
        return excess, stats
    else:
        return excess


def estimate_noise_excess(
    image,
    err=None,
    mask=None,
    bg=None,
    aper=3,
    fwhm=None,
    psf=None,
    optimal=False,
    measure=None,
    spacing=None,
    nsamples=5000,
    bg_size=128,
    seed=0,
    get_stats=False,
    verbose=False,
):
    """Excess of the actual noise of flux measurements over the one predicted by a noise map.

    Measures the weighted sums of the background-subtracted image in apertures
    placed at random positions free of masked pixels, normalizes each by the
    noise predicted for it from the per-pixel noise map the same way as
    :func:`stdpipe.photometry_measure.measure_objects` does, and returns the
    robust standard deviation of these normalized sums - i.e. the factor the
    flux errors of the photometry should be multiplied by. It is larger than 1
    if the pixel noise is correlated (resampled or stacked images, difference
    images, convolved templates), if the background has structure on the
    aperture scale that the background model does not follow, or if the noise
    map underestimates the noise in general.

    The noise of a sum is predicted as ``sqrt(sum(w**2 * err**2))``, with
    ``w`` the pixel overlap fraction for aperture sums (as photutils >= 3.1
    and sep-x >= 1.5.4 propagate the errors) or the weights for optimal
    extraction, so the excess is 1 for uncorrelated noise. Older photutils
    versions use ``sum(w * err**2)`` for aperture sums, which overestimates the
    noise of the pixels on the aperture edge, so that
    :func:`stdpipe.photometry_measure.measure_objects` reports flux errors
    slightly too large (by ~3% for a radius of 5 pixels) with them.

    These sums use the global background only. If the photometry does more -
    e.g. subtracts a local background from an annulus, which both adds the
    noise of its estimate (large for the ``'mode'`` estimator) and removes the
    background structure on larger scales, fits the PSF, or groups the
    sources - pass the photometry routine itself as *measure*. It is then run
    on a grid of empty positions, and the excess is the robust standard
    deviation of its ``flux / fluxerr``, which accounts for everything it does
    by construction.

    The excess depends on the aperture: correlation over a few pixels does
    not change the noise of single pixels, but does increase the noise of
    their sums. It should therefore be measured with the same aperture (and
    weighting) as the photometry it is applied to, and the noise maps passed
    to :func:`stdpipe.photometry_measure.measure_objects` (``err``,
    ``bg_err``) multiplied by it.

    The positions where the aperture touches a masked pixel are rejected, so
    *mask* should cover all the objects in the image - e.g. the dilated
    segmentation map of the detection, or the mask of the difference image
    (plus the residuals of bright stars) for difference imaging. The robust
    estimator tolerates a small fraction of contaminated apertures from faint
    undetected sources.

    Parameters
    ----------
    image : ndarray
        Input image.
    err : ndarray, float or None
        Per-pixel noise map (or a single value) predicted for the image, in
        image units. If None, the RMS map of the background estimated with
        :func:`get_background` is used.
    mask : ndarray of bool or None
        Mask of pixels to exclude (True = masked). Non-finite pixels of the
        image and of *err*, and pixels with non-positive *err*, are always
        excluded.
    bg : ndarray, float or None
        Background to subtract. If None, it is estimated with
        :func:`get_background` (SEP method, grid size *bg_size*).
    aper : float or sequence of float
        Aperture radius, in pixels, or in units of FWHM if *fwhm* is set (as
        in :func:`stdpipe.photometry_measure.measure_objects`). For a
        sequence, the excess is computed for every radius.
    fwhm : float or callable or None
        FWHM in pixels. Sets the unit of *aper*, and the width of the Gaussian
        weights for ``optimal=True`` when *psf* is not given. A callable (e.g.
        :class:`stdpipe.photometry.FWHMMap`) is reduced to its scalar value.
    psf : dict or None
        PSF model (PSFEx or :func:`stdpipe.psf.create_psf_model` structure),
        evaluated at the image center, for the weights of ``optimal=True``.
    optimal : bool
        If True, the sums are weighted with the PSF (or a Gaussian with
        *fwhm*) inside the aperture, as in optimal extraction and, for
        constant noise, PSF fitting. Otherwise plain aperture sums are used.
    measure : callable or None
        Photometry routine to measure the noise with, called once as
        ``measure(obj)`` with a table of ``x``, ``y`` empty positions, and
        returning a table with ``flux`` and ``fluxerr`` columns for them, e.g.
        ``lambda obj: measure_objects(obj, image, aper=1.0, fwhm=fwhm,
        bkgann=[5, 7], bg_err=rms, sn=0)``. It should use the noise maps the
        excess is to be applied to, and no centroiding. If set, *err*, *bg*,
        *psf*, *optimal* and *bg_size* are not used, *aper* (scalar) and
        *fwhm* only define the aperture that must not touch masked pixels.
    spacing : float or None
        Spacing of the grid of positions for *measure*, in pixels. It should
        be larger than the grouping radius of the photometry, so that the
        positions are measured independently. Default is
        ``3 * max(aperture radius, fwhm) + 1``.
    nsamples : int
        Number of apertures to measure. The relative uncertainty of the
        result is about ``1.2 / sqrt(N)`` (the efficiency of the median
        absolute deviation), i.e. ~1.6% for the default, where ``N`` is
        *nsamples* or, if smaller, the number of disjoint apertures fitting
        into the unmasked area.
    bg_size : int
        Background grid size, used if *bg* or *err* is None.
    seed : int or None
        Seed of the random generator placing the apertures.
    get_stats : bool
        If True, also return a dictionary with diagnostics: the number of
        apertures used (``nsamples``), the uncertainty of the excess
        (``excess_err``), and the median of the normalized sums
        (``median``, ~0 unless the background is biased).
    verbose : bool or callable
        Whether to show verbose messages. May be boolean or a print-like
        function.

    Returns
    -------
    excess : float or ndarray
        Ratio of the measured to the predicted noise of the sums, an array
        for a sequence of *aper*. NaN if fewer than 50 apertures fit into the
        unmasked area (or are properly measured by *measure*).
    stats : dict or list of dict
        Diagnostics, only returned if ``get_stats=True``.

    Examples
    --------
    Correct the noise of a resampled image before photometry:

    >>> excess = estimate_noise_excess(image, rms, mask=segm > 0, aper=1.0, fwhm=fwhm)
    >>> obj = measure_objects(obj, image, aper=1.0, fwhm=fwhm, bg_err=excess * rms, gain=gain)

    The same, with local background annulus, measured with the photometry itself:

    >>> kw = dict(aper=1.0, fwhm=fwhm, bkgann=[5, 7], sn=0)
    >>> excess = estimate_noise_excess(image, mask=segm > 0, aper=1.0, fwhm=fwhm,
    ...              measure=lambda o: measure_objects(o, image, bg_err=rms, **kw))
    >>> obj = measure_objects(obj, image, bg_err=excess * rms, gain=gain, **kw)
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    image = np.asarray(image, dtype=np.double)
    bad = ~np.isfinite(image)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)

    if measure is not None:
        return _noise_excess_measured(
            image.shape, bad, measure, aper, fwhm, spacing, nsamples, seed, get_stats, log
        )

    if bg is None or err is None:
        image0 = np.where(bad, 0.0, image)
        bg_est, rms_est = get_background(image0, mask=bad, method='sep', size=bg_size, get_rms=True)
    if bg is None:
        bg = bg_est
    if err is None:
        err = rms_est

    resid = image - bg
    resid = np.broadcast_to(resid, image.shape)
    var = np.broadcast_to(np.asarray(err, dtype=np.double) ** 2, image.shape)
    bad = bad | ~np.isfinite(resid) | ~np.isfinite(var) | ~(var > 0)

    fwhm_s = float(fwhm) if fwhm is not None else None

    profile = None
    if optimal:
        if psf is not None:
            from . import psf as _psf

            profile = _psf.get_psf_stamp(
                psf, x=image.shape[1] / 2, y=image.shape[0] / 2, dx=0, dy=0, normalize=True
            )
        elif fwhm_s is not None:
            sigma = fwhm_s / 2.35482
            half = int(np.ceil(5 * sigma))
            yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
            profile = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        else:
            raise ValueError('optimal=True requires psf or fwhm')

    scalar = np.ndim(aper) == 0
    apers = np.atleast_1d(np.asarray(aper, dtype=np.double))

    rng = np.random.default_rng(seed)
    ny, nx = image.shape

    excesses, stats = [], []
    for radius in apers:
        radius_pix = radius * fwhm_s if fwhm_s is not None else radius
        dy, dx, w, half = _noise_excess_kernel(radius_pix, profile)

        # Random aperture centers not touching masked pixels or the edges,
        # drawn in batches until enough are accepted
        ys, xs = [], []
        naccepted, ntried = 0, 0
        if ny > 2 * half and nx > 2 * half:
            while naccepted < nsamples and ntried < 50 * nsamples:
                batch = max(2 * (nsamples - naccepted), 100)
                y = rng.integers(half, ny - half, batch)
                x = rng.integers(half, nx - half, batch)
                good = ~np.any(bad[y[:, None] + dy, x[:, None] + dx], axis=1)
                ys.append(y[good])
                xs.append(x[good])
                naccepted += np.sum(good)
                ntried += batch

        if naccepted < 50:
            log(
                'Only %d unmasked apertures of radius %.1f px found, cannot estimate noise excess'
                % (naccepted, radius_pix)
            )
            excesses.append(np.nan)
            stats.append({'nsamples': int(naccepted), 'excess_err': np.nan, 'median': np.nan})
            continue

        y = np.concatenate(ys)[:nsamples]
        x = np.concatenate(xs)[:nsamples]
        yi, xi = y[:, None] + dy, x[:, None] + dx

        flux = np.sum(w * resid[yi, xi], axis=1)
        # Variance of the weighted sum, as photutils and SEP propagate it
        fluxerr = np.sqrt(np.sum(w**2 * var[yi, xi], axis=1))
        z = flux / fluxerr

        excess = mad_std(z)
        # Overlapping apertures are not independent: the effective number of
        # samples is at most the number of disjoint apertures in the
        # unmasked area
        nindep = naccepted / ntried * (ny - 2 * half) * (nx - 2 * half) / len(w)
        excess_err = 1.17 * excess / np.sqrt(min(len(z), nindep))
        median = np.median(z)

        log(
            'Noise excess for %s radius %.1f px: %.3f +/- %.3f over %d apertures (%.0f%% accepted), median %.2f'
            % (
                'optimal' if optimal else 'aperture',
                radius_pix,
                excess,
                excess_err,
                len(z),
                100 * naccepted / ntried,
                median,
            )
        )

        excesses.append(excess)
        stats.append({'nsamples': int(len(z)), 'excess_err': excess_err, 'median': median})

    if scalar:
        excesses, stats = excesses[0], stats[0]
    else:
        excesses = np.array(excesses)

    if get_stats:
        return excesses, stats
    else:
        return excesses
