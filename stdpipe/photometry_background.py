"""
Background estimation routines for astronomical images.

This module provides various methods for estimating and subtracting background
from astronomical images, including grid-based methods (SEP, photutils) and
non-grid methods (percentile filtering, morphological opening).
"""

import numpy as np

from astropy.stats import mad_std

import sep
import photutils
import photutils.background


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
    from scipy.ndimage import percentile_filter, median_filter, generic_filter

    # Work with a copy to avoid modifying input
    img = image.copy()

    # Apply mask by replacing with NaN (will be ignored in percentile computation)
    if mask is not None:
        img[mask] = np.nan

    # Iterative sigma-clipping approach
    for iteration in range(maxiters):
        if use_mode:
            # Mode estimation via histogram (in sliding window)
            # This is expensive, so use median as approximation
            back = median_filter(img, size=size)
        else:
            # Percentile filter
            # Note: percentile_filter doesn't handle NaN, so we need masked array approach
            if mask is not None:
                # Use generic_filter with custom function for masked arrays
                def masked_percentile(values):
                    valid = values[~np.isnan(values)]
                    if len(valid) < 3:
                        return np.nan
                    return np.percentile(valid, percentile)

                back = generic_filter(img, masked_percentile, size=size)
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

    # Fill any remaining NaN values
    if np.any(np.isnan(back)):
        valid = np.isfinite(back)
        if np.sum(valid) > 0:
            # Use median of valid pixels
            fill_value = np.median(back[valid])
            back[~valid] = fill_value

    return back


def get_background_morphology(image, mask=None, size=25, iterations=1, smooth=True, smooth_size=None):
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
        Size of smoothing kernel. If None, automatically set to size//3 for
        Gaussian smoothing (good default). Use larger values for smoother results.

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
    y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
    structure = (x*x + y*y) <= (size/2)**2

    # Apply opening
    back = grey_opening(img, structure=structure)

    # Additional iterations if requested
    for i in range(iterations - 1):
        back = grey_opening(back, structure=structure)

    # Apply smoothing to reduce kernel-sized artifacts
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
                # Use size/5 for good balance of smoothness and locality
                sigma = max(2.0, size / 5.0)
            else:
                sigma = smooth_size / 2.355  # Convert FWHM to sigma
            back = gaussian_filter(back, sigma=sigma, mode='reflect')

    return back


def estimate_background_rms_percentile(image, mask=None, size=51, **kwargs):
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
    **kwargs : dict
        Additional parameters (ignored, for compatibility)

    Returns
    -------
    rms : ndarray
        Background RMS map
    """
    from scipy.ndimage import percentile_filter

    # Compute 25th and 75th percentiles
    if mask is not None:
        img = image.copy()
        img[mask] = np.nan

        def masked_percentile_25(values):
            valid = values[~np.isnan(values)]
            return np.percentile(valid, 25) if len(valid) > 3 else np.nan

        def masked_percentile_75(values):
            valid = values[~np.isnan(values)]
            return np.percentile(valid, 75) if len(valid) > 3 else np.nan

        from scipy.ndimage import generic_filter
        p25 = generic_filter(img, masked_percentile_25, size=size)
        p75 = generic_filter(img, masked_percentile_75, size=size)
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
    from scipy.ndimage import generic_filter

    residual = image - background

    if mask is not None:
        residual[mask] = np.nan

    # Local standard deviation
    def local_std(values):
        valid = values[~np.isnan(values)]
        if len(valid) < 3:
            return np.nan
        # Use MAD for robustness
        return mad_std(valid)

    rms = generic_filter(residual, local_std, size=size)

    # Fill NaN values
    if np.any(np.isnan(rms)):
        valid = np.isfinite(rms)
        if np.sum(valid) > 0:
            fill_value = np.median(rms[valid])
            rms[~valid] = fill_value

    return rms


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
        If True, return uncertainty map. If False, return scalar RMS.
    chunk_rows : int
        Process prediction in chunks of this many rows to reduce memory usage

    Returns
    -------
    background : ndarray
        Estimated background map
    uncertainty : ndarray or float
        If get_uncertainty=True: 2D array of GP posterior standard deviation
        If get_uncertainty=False: Scalar robust RMS of residuals

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

    # Helper function for robust sigma estimation
    def robust_sigma(x):
        """Robust sigma estimate via MAD."""
        x_valid = x[np.isfinite(x)]
        if x_valid.size < 5:
            return np.nan
        return mad_std(x_valid)

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
    keep = np.ones(y_vals.shape, dtype=bool)
    for _ in range(max(0, int(n_clip_iter))):
        yy = y_vals[keep]
        if yy.size < 50:
            break
        med = np.median(yy)
        sig = robust_sigma(yy)
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
        X_pred = np.column_stack([
            np.tile(x_coords, y1 - y0),
            np.repeat(rows, nx),
        ])

        if get_uncertainty:
            pred, pred_std = gp.predict(X_pred, return_std=True)
            background[y0:y1, :] = pred.reshape((y1 - y0, nx))
            uncertainty[y0:y1, :] = pred_std.reshape((y1 - y0, nx))
        else:
            pred = gp.predict(X_pred, return_std=False)
            background[y0:y1, :] = pred.reshape((y1 - y0, nx))

    # Return results
    if get_uncertainty:
        return background, uncertainty
    else:
        # Return scalar RMS based on training set residuals
        y_pred_train = gp.predict(X_train, return_std=False)
        resid = y_train - y_pred_train
        rms = robust_sigma(resid)
        return background, float(rms)


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

    See Also
    --------
    get_background_percentile : Percentile filtering details
    get_background_morphology : Morphological opening details
    get_background_gp : Gaussian Process regression details

    Notes
    -----
    **Method Selection Guide**:

    - **Flat backgrounds**: Use 'sep' (default) - fastest and most accurate
    - **Linear gradients**: Use 'sep' with size=64-128
    - **Quadratic gradients**: Use 'morphology' or local gradient fitting (bkg_order)
    - **Complex backgrounds** (vignetting, strong curvature): Use 'gp'
    - **Very crowded fields**: Use 'percentile' with percentile=10-15
    - **Large images**: Avoid 'percentile' (very slow), use 'sep' or 'morphology'

    **Performance**:

    - SEP: ~1-3 ms (512×512 image)
    - Morphology: ~300 ms (512×512 image)
    - Percentile: ~2-5 seconds (512×512 image)
    - GP: ~1-10 seconds (512×512 image, depends on max_points)

    For complex backgrounds (strong curvature, vignetting), consider using
    local gradient fitting instead (bkg_order parameter in measure_objects),
    or the GP method for maximum flexibility.
    """
    if method == 'sep':
        # Ensure correct byte order for SEP
        image = image.astype(np.double)
        bg = sep.Background(image, mask=mask, bw=size, bh=size, **kwargs)

        back, backrms = bg.back(), bg.rms()
    elif method == 'photutils':
        bg = photutils.background.Background2D(image, size, mask=mask, **kwargs)
        back, backrms = bg.background, bg.background_rms
    elif method == 'percentile':
        back = get_background_percentile(image, mask=mask, size=size, **kwargs)
        # Estimate RMS from percentile range if needed
        if get_rms:
            backrms = estimate_background_rms_percentile(image, mask=mask, size=size, **kwargs)
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
        back, backrms = get_background_gp(image, mask=mask, **kwargs)
    else:
        raise ValueError(f"Unknown background method: {method}")

    if get_rms:
        return back, backrms
    else:
        return back
