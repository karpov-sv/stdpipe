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


def get_background_morphology(image, mask=None, size=25, iterations=1, smooth=True):
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
        If True, apply additional median smoothing to reduce artifacts

    Returns
    -------
    background : ndarray
        Estimated background map

    Notes
    -----
    - Very fast (much faster than percentile filtering)
    - Simple and robust
    - Can create edge artifacts
    - May suppress legitimate background features similar in size to stars
    - Best for images with clear point sources on smooth backgrounds
    """
    from scipy.ndimage import grey_opening, median_filter

    # Work with a copy
    img = image.copy()

    # Handle mask by replacing with local median
    if mask is not None:
        # Fill masked regions with local median to avoid artifacts
        if np.any(mask):
            from scipy.ndimage import distance_transform_edt
            from scipy.interpolate import NearestNDInterpolator

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

    # Optional smoothing to reduce artifacts
    if smooth:
        back = median_filter(back, size=5)

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
    size : int
        Grid size for sep/photutils, or kernel size for percentile/morphology.
        Typical values:

        - SEP/photutils: 64-128 pixels
        - Percentile: 31-51 pixels (smaller for gradients)
        - Morphology: 15-35 pixels (should be > star size)
    get_rms : bool
        If True, return (background, background_rms)
    **kwargs : dict
        Additional parameters passed to the method:

        - For percentile: percentile, maxiters, sigma
        - For morphology: iterations, smooth

    Returns
    -------
    background : ndarray
        Estimated background
    background_rms : ndarray, optional
        Background RMS (if get_rms=True)

    Examples
    --------
    Default grid-based background estimation:

    >>> bg = get_background(image, method='sep', size=128)

    Non-grid morphological method for moderate gradients:

    >>> bg = get_background(image, method='morphology', size=25)

    Percentile method for very crowded fields:

    >>> bg = get_background(image, method='percentile', size=31, percentile=15.0)

    Get both background and RMS:

    >>> bg, bg_rms = get_background(image, method='sep', get_rms=True)

    See Also
    --------
    get_background_percentile : Percentile filtering details
    get_background_morphology : Morphological opening details

    Notes
    -----
    **Method Selection Guide**:

    - **Flat backgrounds**: Use 'sep' (default) - fastest and most accurate
    - **Linear gradients**: Use 'sep' with size=64-128
    - **Quadratic gradients**: Use 'morphology' or local gradient fitting (bkg_order)
    - **Very crowded fields**: Use 'percentile' with percentile=10-15
    - **Large images**: Avoid 'percentile' (very slow), use 'sep' or 'morphology'

    **Performance**:

    - SEP: ~1-3 ms (512×512 image)
    - Morphology: ~300 ms (512×512 image)
    - Percentile: ~2-5 seconds (512×512 image)

    For complex backgrounds (strong curvature, vignetting), consider using
    local gradient fitting instead (bkg_order parameter in measure_objects).
    """
    if method == 'sep':
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
    else:
        raise ValueError(f"Unknown background method: {method}")

    if get_rms:
        return back, backrms
    else:
        return back
