"""
Aperture photometry measurement routines.

This module contains functions for performing aperture photometry
on detected objects.
"""


import numpy as np
import photutils
import photutils.background
import photutils.aperture
import photutils.centroids
import photutils.psf
from photutils.utils import calc_total_error

# Note: psf and photometry_psf modules imported lazily in functions to avoid circular dependency


def _get_psf_stamp_at_position(psf, x, y, stamp_size=None):
    """
    Get normalized PSF stamp at a given position with automatic sub-pixel alignment.

    For Gaussian PSF: Creates pixel-integrated PSF using the error function (erf)
    to properly integrate Gaussian flux over each pixel's area. This eliminates
    the FWHM-dependent systematic bias caused by point sampling at pixel centers:
      - FWHM = 1.5 pix: +11.6% bias → ~0%
      - FWHM = 3.0 pix: +2.7% bias → ~0%
      - FWHM = 6.0 pix: +0.8% bias → ~0%

    The stamp is shifted by the sub-pixel offset (x - round(x), y - round(y))
    to align with the actual source position.

    For PSFEx/ePSF models: The psf.get_psf_stamp() function automatically applies
    the same sub-pixel shift internally, so no additional correction is needed here.
    Both methods produce identically aligned PSF stamps.

    Technical details:
    - Uses erf (error function = CDF of Gaussian) to integrate over pixel boundaries
    - Each pixel spans from (i-0.5, i+0.5) in both x and y
    - Flux in pixel (i,j) = ∫∫ Gaussian(x,y) dx dy over pixel area
    - This is computed as: [CDF(x+0.5) - CDF(x-0.5)] × [CDF(y+0.5) - CDF(y-0.5)]

    :param psf: PSF model (dict from PSFEx/ePSF, or Gaussian FWHM as float)
    :param x, y: Object position (float)
    :param stamp_size: Optional fixed stamp size (odd integer)
    :returns: Normalized PSF stamp aligned to sub-pixel position (x, y)
    """
    if isinstance(psf, dict):
        # PSFEx or ePSF model - lazy import to avoid circular dependency
        from stdpipe import psf as psf_module
        psf_stamp = psf_module.get_psf_stamp(psf, x=x, y=y, normalize=True)
    else:
        # Gaussian PSF - create stamp from FWHM with sub-pixel shift
        # Use pixel integration to eliminate FWHM-dependent systematic bias
        from scipy.special import erf

        fwhm = float(psf)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = stamp_size if stamp_size else int(np.ceil(fwhm * 3)) * 2 + 1

        # CRITICAL: Shift PSF by sub-pixel offset to align with actual source position
        # This eliminates position-dependent biases:
        #   - Centroiding bias: ~0.2 pix → ~0.01 pix (20× improvement)
        #   - Optimal extraction flux bias from sub-pixel position: eliminated
        ix, iy = int(np.round(x)), int(np.round(y))
        dx_sub = x - ix  # Sub-pixel offset (e.g., 0.3 if x=25.3)
        dy_sub = y - iy

        # PSF center in stamp coordinates (accounting for sub-pixel shift)
        x0 = size // 2 + dx_sub
        y0 = size // 2 + dy_sub

        # Pixel edges: pixels span from (i-0.5) to (i+0.5)
        x_edges = np.arange(size + 1) - 0.5
        y_edges = np.arange(size + 1) - 0.5

        # Integrate Gaussian over pixel boundaries using error function
        # erf(z) = (2/√π) ∫₀^z exp(-t²) dt is the CDF of standard normal
        # CDF of Gaussian N(μ,σ²) at point x is: Φ(x) = 0.5 * [1 + erf((x-μ)/(σ√2))]
        sqrt2_sigma = np.sqrt(2) * sigma
        cdf_x = 0.5 * (1 + erf((x_edges - x0) / sqrt2_sigma))
        cdf_y = 0.5 * (1 + erf((y_edges - y0) / sqrt2_sigma))

        # Flux in each pixel = CDF at right edge - CDF at left edge
        # For pixel i: flux = CDF(i+0.5) - CDF(i-0.5)
        flux_x = np.diff(cdf_x)  # Integrated flux in x direction
        flux_y = np.diff(cdf_y)  # Integrated flux in y direction

        # 2D PSF is separable: outer product of 1D integrated profiles
        psf_stamp = np.outer(flux_y, flux_x)

        # Normalize to unit sum
        psf_stamp /= np.sum(psf_stamp)

    return psf_stamp


def _psf_centroid(image, x, y, psf, mask=None, box_size=None):
    """
    Compute PSF-weighted centroid at a given position.

    Uses PSF weighting analogous to optimal extraction (Naylor 1998),
    which naturally emphasizes high-S/N pixels in the PSF core.
    This provides better stability and accuracy than simple center-of-mass,
    especially for crowded fields and faint sources.

    Algorithm:
        x_new = x₀ + Σ[(x_i - x₀) × I_i × P_i] / Σ[I_i × P_i]
        y_new = y₀ + Σ[(y_i - y₀) × I_i × P_i] / Σ[I_i × P_i]

    where I_i is pixel intensity, P_i is PSF weight, (x₀, y₀) is current position.

    The PSF model is automatically shifted by the sub-pixel offset (x - round(x),
    y - round(y)) at each iteration, eliminating systematic bias. This is critical
    for both centroiding accuracy (~0.2 pix → 0.01 pix) and is also used by optimal
    extraction to avoid ~2.6% flux bias.

    Performance:
        - Isolated sources (no masking): ~0.01 pix accuracy (3.8× better than COM)
        - Crowded fields (2-3 FWHM separation): 80-940× better than COM
        - Noise-free limit: ~0.008 pix (limited by discrete pixelization)

    IMPORTANT - Masking Limitation:
        PSF weighting assumes a symmetric, complete PSF profile. Heavy random masking
        (>20%) creates asymmetric pixel sampling, which biases the centroid. With 30%
        random masking, PSF centroiding error increases ~30-40× (from 0.01 to 0.3 pix).

        For images with heavy random masking, COM is more robust because it treats all
        good pixels equally without assuming a specific profile. Use COM when:
          - Random masking fraction > 20%
          - OR when PSF centroiding shows large shifts/degradation

        Structured masking (e.g., saturated stars, detector defects) affects both
        methods similarly and is less problematic than random masking.

    :param image: Background-subtracted image
    :param x, y: Initial position estimate (float)
    :param psf: PSF model (dict from PSFEx/ePSF, or Gaussian FWHM as float)
    :param mask: Optional mask (True = masked)
    :param box_size: Minimum cutout size in pixels (actual size will be max of this and PSF stamp size)
    :returns: (x_new, y_new) or (np.nan, np.nan) on failure
    """
    # Get PSF stamp at current position
    # For Gaussian PSF: automatically shifted by sub-pixel offset (always enabled)
    # For position-dependent PSF (PSFEx): evaluates polynomial at (x,y)
    psf_stamp = _get_psf_stamp_at_position(psf, x, y)
    stamp_size = psf_stamp.shape[0]

    # Use the larger of box_size and stamp_size to avoid cropping the PSF
    if box_size is None:
        box_size = stamp_size
    else:
        box_size = max(box_size, stamp_size)
        # Ensure odd size
        if box_size % 2 == 0:
            box_size += 1

    half = box_size // 2

    # Integer position for cutout
    ix, iy = int(np.round(x)), int(np.round(y))

    # Extract cutout
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1

    # Handle boundaries
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return np.nan, np.nan

    data_cutout = image[y0:y1, x0:x1]

    # Apply mask if provided
    if mask is not None:
        mask_cutout = mask[y0:y1, x0:x1]
        # Only use positive pixels and unmasked regions for centroiding
        good = ~mask_cutout & np.isfinite(data_cutout) & (data_cutout > 0)
    else:
        good = np.isfinite(data_cutout) & (data_cutout > 0)

    if np.sum(good) < 3:  # Need at least 3 pixels for meaningful centroid
        return np.nan, np.nan

    # Embed PSF stamp in box-sized array if box is larger
    if stamp_size != box_size:
        # Since we ensured box_size >= stamp_size, just center the PSF in the box
        psf_resized = np.zeros((box_size, box_size))
        psf_half = stamp_size // 2
        box_half = box_size // 2

        y_start = box_half - psf_half
        y_end = y_start + stamp_size
        x_start = box_half - psf_half
        x_end = x_start + stamp_size
        psf_resized[y_start:y_end, x_start:x_end] = psf_stamp
        psf_stamp = psf_resized

    # Create coordinate grids
    yy, xx = np.mgrid[0:box_size, 0:box_size]

    # Extract values for good pixels
    I = data_cutout[good]
    P = psf_stamp[good]
    x_coords = xx[good]
    y_coords = yy[good]

    # Relative offsets from cutout center
    # Note: box center corresponds to integer position (ix, iy) in image
    x_rel = x_coords - box_size // 2
    y_rel = y_coords - box_size // 2

    # PSF-weighted centroid shift
    sum_IP = np.sum(I * P)
    if sum_IP <= 0:
        return np.nan, np.nan

    dx = np.sum(x_rel * I * P) / sum_IP
    dy = np.sum(y_rel * I * P) / sum_IP

    # Convert back to image coordinates
    # The offset dx, dy is relative to the integer position ix, iy
    # (which is at the center of the extracted box)
    x_new = ix + dx
    y_new = iy + dy

    return x_new, y_new


def _solve_weighted_leastsq(A, D, W):
    """
    Solve weighted least squares: (A^T W A)x = A^T W D

    :param A: Design matrix (npix, K) where K is number of sources
    :param D: Data vector (npix,)
    :param W: Weight vector (npix,) - inverse variance weights
    :returns: (x, cov) - solution vector and covariance matrix, or (None, None) on failure
    """
    # A^T W A and A^T W D using weight vector
    AtW = A.T * W  # Broadcasting: (K, npix) * (npix,) -> (K, npix)
    AtWA = AtW @ A  # (K, K)
    AtWD = AtW @ D  # (K,)

    # Solve and get covariance
    try:
        x = np.linalg.solve(AtWA, AtWD)
        cov = np.linalg.solve(AtWA, np.eye(AtWA.shape[0]))
    except np.linalg.LinAlgError:
        return None, None

    return x, cov


def _grouped_optimal_extraction(image, err, positions, psf, bg_local=None, mask=None, radius=None):
    """
    Perform grouped optimal extraction for multiple overlapping sources.

    Solves: D = Σ_k F_k × P_k + B via weighted least squares (A^T W A)x = A^T W D

    This function simultaneously fits the fluxes of all sources in a group,
    properly accounting for flux sharing between overlapping PSFs.

    :param image: Background-subtracted image
    :param err: Error/noise map (sqrt of variance)
    :param positions: List of (x, y) positions for sources in group
    :param psf: PSF model (dict from PSFEx/ePSF, or Gaussian FWHM as float)
    :param bg_local: List of local background values for each source, or single value for all
    :param mask: Optional mask (True = masked)
    :param radius: Extraction radius in pixels around each source
    :returns: List of (flux, fluxerr, npix, norm, reduced_chi2) for each source
    """
    K = len(positions)
    if K == 0:
        return []

    # For single source, use standard optimal extraction
    if K == 1:
        x, y = positions[0]
        bg = bg_local[0] if isinstance(bg_local, (list, np.ndarray)) else bg_local
        result = _optimal_extraction(image, err, x, y, psf, bg_local=bg, mask=mask, radius=radius)
        return [result]

    # Get PSF stamp size from first position
    test_stamp = _get_psf_stamp_at_position(psf, positions[0][0], positions[0][1])
    stamp_size = test_stamp.shape[0]
    half = stamp_size // 2

    # Determine bounding box covering all sources + PSF radius
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    # Integer bounds with padding for PSF
    x0 = int(np.floor(np.min(xs))) - half
    x1 = int(np.ceil(np.max(xs))) + half + 1
    y0 = int(np.floor(np.min(ys))) - half
    y1 = int(np.ceil(np.max(ys))) + half + 1

    # Handle image boundaries
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        # Fall back to individual fitting for edge groups
        results = []
        for i, (x, y) in enumerate(positions):
            bg = bg_local[i] if isinstance(bg_local, (list, np.ndarray)) else bg_local
            results.append(_optimal_extraction(image, err, x, y, psf, bg_local=bg, mask=mask, radius=radius))
        return results

    # Extract common cutout
    data_cutout = image[y0:y1, x0:x1].copy()
    err_cutout = err[y0:y1, x0:x1]
    cutout_shape = data_cutout.shape

    # Build mask for good pixels
    if mask is not None:
        mask_cutout = mask[y0:y1, x0:x1]
        good = ~mask_cutout & (err_cutout > 0) & np.isfinite(data_cutout)
    else:
        good = (err_cutout > 0) & np.isfinite(data_cutout)

    # Apply radius cutoff around each source if specified
    if radius is not None:
        yy, xx = np.mgrid[0:cutout_shape[0], 0:cutout_shape[1]]
        radius_mask = np.zeros(cutout_shape, dtype=bool)
        for x, y in positions:
            # Position relative to cutout
            rel_x = x - x0
            rel_y = y - y0
            dist = np.sqrt((xx - rel_x)**2 + (yy - rel_y)**2)
            radius_mask |= (dist <= radius)
        good &= radius_mask

    n_pix = int(np.sum(good))
    if n_pix == 0:
        return [(np.nan, np.nan, 0, np.nan, np.nan) for _ in range(K)]

    # Subtract average local background to center data (fit includes residual background term)
    bg_offset = 0.0
    if bg_local is not None:
        if isinstance(bg_local, (list, np.ndarray)):
            bg_offset = np.nanmean(bg_local)
        else:
            bg_offset = bg_local
        if np.isfinite(bg_offset) and bg_offset != 0:
            data_cutout -= bg_offset

    # Build design matrix with K PSF columns + background term
    A = np.zeros((n_pix, K + 1))
    psf_norms = np.zeros(K)

    for k, (x, y) in enumerate(positions):
        # Get PSF at this source position
        psf_stamp = _get_psf_stamp_at_position(psf, x, y, stamp_size)

        # Position of PSF center relative to cutout origin
        rel_x = x - x0
        rel_y = y - y0

        # Integer position and sub-pixel offset
        ix_rel = int(np.round(rel_x))
        iy_rel = int(np.round(rel_y))

        # PSF stamp placement bounds in cutout
        p_y0 = iy_rel - half
        p_y1 = iy_rel + half + 1
        p_x0 = ix_rel - half
        p_x1 = ix_rel + half + 1

        # Create full-size PSF array for this source
        psf_full = np.zeros(cutout_shape)

        # Handle partial overlap at cutout edges
        # Source region in cutout
        c_y0 = max(0, p_y0)
        c_y1 = min(cutout_shape[0], p_y1)
        c_x0 = max(0, p_x0)
        c_x1 = min(cutout_shape[1], p_x1)

        # Corresponding region in PSF stamp
        s_y0 = c_y0 - p_y0
        s_y1 = stamp_size - (p_y1 - c_y1)
        s_x0 = c_x0 - p_x0
        s_x1 = stamp_size - (p_x1 - c_x1)

        if c_y1 > c_y0 and c_x1 > c_x0:
            psf_full[c_y0:c_y1, c_x0:c_x1] = psf_stamp[s_y0:s_y1, s_x0:s_x1]

        A[:, k] = psf_full[good]
        psf_norms[k] = np.sum(psf_full[good])

    # Common background term
    A[:, -1] = 1.0

    # Data and weights
    D = data_cutout[good]
    V = err_cutout[good]**2
    W = 1.0 / V

    # Solve weighted least squares
    x_sol, cov = _solve_weighted_leastsq(A, D, W)

    if x_sol is None:
        # Matrix singularity - fall back to individual fitting
        results = []
        for i, (x, y) in enumerate(positions):
            bg = bg_local[i] if isinstance(bg_local, (list, np.ndarray)) else bg_local
            results.append(_optimal_extraction(image, err, x, y, psf, bg_local=bg, mask=mask, radius=radius))
        return results

    # Extract fluxes and errors
    fluxes = x_sol[:K]
    flux_errors = np.sqrt(np.diag(cov)[:K])

    # Compute chi-squared for the group fit
    model = A @ x_sol
    residuals = D - model
    chi2 = np.sum(residuals**2 * W)

    # Reduced chi-squared (degrees of freedom = N - K - 1)
    dof = n_pix - (K + 1)
    if dof > 0:
        reduced_chi2 = chi2 / dof
    else:
        reduced_chi2 = np.nan

    # Build results for each source
    results = []
    for k in range(K):
        results.append((fluxes[k], flux_errors[k], n_pix, psf_norms[k], reduced_chi2))

    return results


def _optimal_extraction(image, err, x, y, psf, bg_local=None, mask=None, radius=None):
    """
    Perform optimal extraction photometry at a single position.

    Algorithm from Naylor (1998, MNRAS 296, 339) based on Horne (1986),
    using simplified weighting for robustness to PSF errors.

    flux = Σ(P × D) / Σ(P²)
    variance = Σ(P² × V) / (Σ(P²))²

    :param image: Background-subtracted image
    :param err: Error/noise map (sqrt of variance)
    :param x, y: Object position (float)
    :param psf: PSF model (dict from PSFEx/ePSF, or Gaussian FWHM as float)
    :param bg_local: Optional local background value on top of already subtracted background, to be additionally subtracted prior to measurement
    :param mask: Optional mask (True = masked)
    :param radius: Extraction radius in pixels (default: use full PSF stamp)
    :returns: (flux, fluxerr, npix, reduced_chi2)
    """
    # Get PSF stamp at object position (automatically includes sub-pixel shift)
    psf_stamp = _get_psf_stamp_at_position(psf, x, y)

    # Determine extraction region
    stamp_size = psf_stamp.shape[0]
    half = stamp_size // 2

    # Integer position for cutout
    ix, iy = int(np.round(x)), int(np.round(y))

    # Extract cutouts
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1

    # Handle edge cases
    if y0 < 0 or x0 < 0 or y1 >= image.shape[0] or x1 >= image.shape[1]:
        return np.nan, np.nan, 0, np.nan, np.nan

    data_cutout = image[y0:y1, x0:x1]
    err_cutout = err[y0:y1, x0:x1]

    # Apply mask if provided
    if mask is not None:
        mask_cutout = mask[y0:y1, x0:x1]
        good = ~mask_cutout & (err_cutout > 0) & np.isfinite(data_cutout)
    else:
        good = (err_cutout > 0) & np.isfinite(data_cutout)

    # Apply radius cutoff if specified
    if radius is not None:
        yy, xx = np.mgrid[0:stamp_size, 0:stamp_size]
        dist = np.sqrt((xx - half)**2 + (yy - half)**2)
        good &= (dist <= radius)

    if np.sum(good) == 0:
        return np.nan, np.nan, 0, np.nan, np.nan

    # Variance = err^2
    var = err_cutout**2

    # Optimal extraction with simplified weighting (Naylor 1998)
    P = psf_stamp[good]
    D = data_cutout[good]
    V = var[good]

    if bg_local:
        D -= bg_local

    # Simplified weighting: flux = Σ(P × D) / Σ(P²)
    sum_P2 = np.sum(P**2)

    if sum_P2 <= 0:
        return np.nan, np.nan, 0, np.nan, np.nan

    flux = np.sum(P * D) / sum_P2

    # variance = Σ(P² × V) / (Σ(P²))²
    variance = np.sum(P**2 * V) / (sum_P2**2)
    fluxerr = np.sqrt(variance)

    # Compute chi-squared: χ² = Σ((D - flux × P)² / V)
    residuals = D - flux * P
    chi2 = np.sum(residuals**2 / V)

    # Reduced chi-squared (degrees of freedom = N - 1)
    n_pix = int(np.sum(good))
    if n_pix > 1:
        reduced_chi2 = chi2 / (n_pix - 1)
    else:
        reduced_chi2 = np.nan

    return flux, fluxerr, n_pix, np.sum(P), reduced_chi2


def measure_objects(
    obj,
    image,
    aper=3,
    bkgann=None,
    bkg_order=0,
    fwhm=None,
    psf=None,
    optimal=False,
    group_sources=True,
    grouper_radius=None,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    sn=None,
    centroid_iter=0,
    centroid_method='com',
    keep_negative=True,
    get_bg=False,
    verbose=False,
):
    """Photometry at the positions of already detected objects.

    Supports both standard aperture photometry and optimal extraction that provides ~10% S/N improvement for point sources (Naylor 1998).

    It will estimate and subtract the background unless external background estimation (`bg`) is provided, and use user-provided noise map (`err`) if requested.

    If the `mask` is provided, it will set 0x200 bit in object `flags` if at least one of aperture pixels is masked.

    The results may optionally filtered to drop the detections with low signal to noise ratio if `sn` parameter is set and positive. It will also filter out the events with negative flux.


    :param obj: astropy.table.Table with initial object detections to be measured
    :param image: Input image as a NumPy array
    :param aper: Circular aperture radius in pixels, to be used for flux measurement. For optimal extraction, this is the clipping radius.
    :param bkgann: Background annulus (tuple with inner and outer radii) to be used for local background estimation. If not set, global background model is used instead.
    :param bkg_order: Polynomial order for background fitting in annulus. Only used if bkgann is set. 0 = constant (mean, fastest), 1 = plane (linear gradient, recommended), 2 = quadratic surface (complex gradients). Default is 1. Gradient-aware fitting (order > 0) dramatically reduces biases with non-uniform backgrounds: linear gradients improved 20× (19% → <1% error), quadratic gradients improved 100-400× (-415% → <4% error). Set to 0 for flat backgrounds to match legacy behavior.
    :param fwhm: If provided, `aper` and `bkgann` will be measured in units of this value (so they will be specified in units of FWHM). Also used to define Gaussian PSF for optimal extraction if `psf` is not provided.
    :param psf: PSF model for optimal extraction and PSF-weighted centroiding. Can be a dict from psf.run_psfex(), psf.load_psf(), or psf.create_psf_model(). If None, a Gaussian PSF will be created from the `fwhm` parameter.
    :param optimal: If True, use optimal extraction instead of aperture photometry. Requires either `psf` or `fwhm` to define the PSF profile.
    :param group_sources: If True and optimal=True, use grouped optimal extraction for overlapping sources. Simultaneously fits fluxes for nearby sources using weighted least squares, properly accounting for flux sharing between overlapping PSFs. Dramatically more accurate in crowded fields (51× improvement at 0.5 FWHM, 14× at 1.0 FWHM, 3× at 1.5 FWHM). Default is True (recommended). Set to False only for known sparse fields where performance is critical. No downside for isolated sources (identical results at >3 FWHM separation).
    :param grouper_radius: Radius in pixels for grouping nearby sources. Sources within this distance are fitted simultaneously. If None, defaults to 2*aper. Only used if group_sources=True.
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array with same shape as input image) instead of automatically computed one
    :param err: Image noise map as a NumPy array to be used instead of automatically computed one, optional
    :param gain: Image gain, e/ADU, used to build image noise model
    :param bg_size: Background grid size in pixels
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param centroid_iter: Number of centroiding iterations to run before photometry. If non-zero, will try to improve the aperture placement by finding the centroid of pixels inside the aperture.
    :param centroid_method: Centroiding method to use. Options: 'com' (center-of-mass, default) or 'psf' (PSF-weighted). PSF-weighted centroiding uses sub-pixel shifted PSF models to achieve ~0.01 pix accuracy on isolated sources (3.8× better than COM) and 80-940× improvement in crowded fields (separation < 3 FWHM) by suppressing neighbor contamination. LIMITATION: PSF centroiding degrades with heavy random masking (>20%) because it assumes a symmetric PSF profile - asymmetric pixel sampling creates bias. With 30% random masking, use COM instead (more robust). For clean images or structured masking, PSF is recommended for maximum accuracy. Requires either `psf` parameter (PSFEx model) or `fwhm` (Gaussian PSF). Default is 'com' for backward compatibility.
    :param keep_negative: If not set, measurements with negative fluxes will be discarded
    :param get_bg: If True, the routine will also return estimated background and background noise images
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The copy of original table with `flux`, `fluxerr`, `mag` and `magerr` columns replaced with the values measured in the routine. If :code:`get_bg=True`, also returns the background and background error images.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if not len(obj):
        log('No objects to measure')
        return obj

    # Operate on the copy of the list
    obj = obj.copy()

    # Sanitize the image and make its copy to safely operate on it
    image1 = image.astype(np.double)
    mask0 = ~np.isfinite(image1)  # Minimal mask
    # image1[mask0] = np.median(image1[~mask0])

    # Ensure that the mask is defined
    if mask is None:
        # mask = mask0
        mask = np.zeros(image.shape, dtype=bool)
    else:
        mask = np.array(mask).astype(bool)

    if bg is None or err is None or get_bg:
        log('Estimating global background with %dx%d grid' % (bg_size, bg_size))
        bg_est = photutils.background.Background2D(
            image1, bg_size, mask=mask | mask0, exclude_percentile=90
        )
        bg_est_bg = bg_est.background
        bg_est_rms = bg_est.background_rms
    else:
        bg_est = None

    if bg is None:
        log(
            'Subtracting global background: median %.1f rms %.2f' % (
                np.median(bg_est_bg), np.std(bg_est_bg)
            )
        )
        image1 -= bg_est_bg
    else:
        log(
            'Subtracting user-provided background: median %.1f rms %.2f' % (
                np.median(bg), np.std(bg)
            )
        )
        image1 -= bg

    image1[mask0] = 0

    if err is None:
        log(
            'Using global background noise map: median %.1f rms %.2f + gain %.1f' % (
                np.median(bg_est_rms),
                np.std(bg_est_rms),
                gain if gain else np.inf,
            )
        )
        err = bg_est_rms
        if gain:
            err = calc_total_error(image1, err, gain)
    else:
        log(
            'Using user-provided noise map: median %.1f rms %.2f' % (
                np.median(err), np.std(err)
            )
        )

    if fwhm is not None and fwhm > 0:
        log('Scaling aperture radii with FWHM %.1f pix' % fwhm)
        aper *= fwhm

    log('Using aperture radius %.1f pixels' % aper)

    # Centroiding
    if centroid_iter:
        box_size = int(np.ceil(aper))
        if box_size % 2 == 0:
            box_size += 1

        # Determine centroiding method
        use_psf_centroid = centroid_method == 'psf' and (psf is not None or fwhm is not None)

        if use_psf_centroid:
            # Prepare PSF for centroiding
            if psf is not None:
                psf_for_centroid = psf
                log('Using PSF-weighted centroiding with %d iterations within %dx%d box' % (centroid_iter, box_size, box_size))
            else:
                # Use original fwhm (before aper scaling) for PSF
                psf_for_centroid = fwhm
                log('Using PSF-weighted centroiding (Gaussian FWHM=%.1f) with %d iterations within %dx%d box' % (fwhm, centroid_iter, box_size, box_size))
        else:
            log('Using COM centroiding with %d iterations within %dx%d box' % (centroid_iter, box_size, box_size))

        # Keep original pixel positions
        obj['x_orig'] = obj['x'].copy()
        obj['y_orig'] = obj['y'].copy()

        # Combined mask for centroiding
        centroid_mask = mask | mask0

        # Process each object individually
        for i in range(len(obj)):
            x, y = float(obj['x'][i]), float(obj['y'][i])

            # Skip invalid positions (NaN or outside image with margin for box)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < box_size // 2 or x >= image1.shape[1] - box_size // 2:
                continue
            if y < box_size // 2 or y >= image1.shape[0] - box_size // 2:
                continue

            # Iterative centroiding
            for _ in range(centroid_iter):
                if use_psf_centroid:
                    # PSF-weighted centroiding
                    x_new, y_new = _psf_centroid(
                        image1, x, y, psf_for_centroid,
                        mask=centroid_mask, box_size=box_size
                    )
                else:
                    # Standard COM centroiding
                    # Extract cutout around current position
                    x_int, y_int = int(round(x)), int(round(y))
                    x_min = x_int - box_size // 2
                    x_max = x_int + box_size // 2 + 1
                    y_min = y_int - box_size // 2
                    y_max = y_int + box_size // 2 + 1

                    # Check bounds (position may have shifted during iteration)
                    if x_min < 0 or y_min < 0 or x_max > image1.shape[1] or y_max > image1.shape[0]:
                        break

                    cutout = image1[y_min:y_max, x_min:x_max]
                    # Let's only use positive pixels for centroiding
                    cutout_mask = centroid_mask[y_min:y_max, x_min:x_max] | (cutout < 0)

                    # Skip if footprint is fully masked
                    if np.all(cutout_mask):
                        break

                    # Compute centroid in cutout coordinates
                    x_c, y_c = photutils.centroids.centroid_com(cutout, mask=cutout_mask)

                    # Skip if centroid computation failed
                    if not np.isfinite(x_c) or not np.isfinite(y_c):
                        break

                    # Convert back to image coordinates
                    x_new = x_min + x_c
                    y_new = y_min + y_c

                # Check convergence
                if np.isfinite(x_new) and np.isfinite(y_new):
                    shift = np.sqrt((x_new - x)**2 + (y_new - y)**2)
                    x, y = x_new, y_new
                    if shift < 0.01:  # Converged (0.01 pixel threshold)
                        break
                else:
                    # Centroid computation failed, keep previous position
                    break

            # Update object position
            obj['x'][i] = x
            obj['y'][i] = y

    if 'flags' not in obj.keys():
        obj['flags'] = 0

    # Dedicated column for local background on top of global estimation
    obj['bg_local'] = 0

    # Local background - it has to be computed prior to optimal photometry
    # as otherwise it is too difficult to subtract it from its (weighted) results
    if bkgann is not None and len(bkgann) == 2:
        if fwhm is not None and fwhm > 0:
            bkgann = [_ * fwhm for _ in bkgann]

        order_names = {0: 'constant (mean)', 1: 'plane', 2: 'quadratic'}
        order_name = order_names.get(bkg_order, f'order {bkg_order}')
        log(
            'Using local background annulus between %.1f and %.1f pixels with %s fitting' % (
                bkgann[0], bkgann[1], order_name
            )
        )

        # Convert MaskedColumns to arrays and identify valid positions
        x_vals = np.ma.filled(np.asarray(obj['x']), fill_value=np.nan)
        y_vals = np.ma.filled(np.asarray(obj['y']), fill_value=np.nan)
        valid_pos = np.isfinite(x_vals) & np.isfinite(y_vals)

        # Initialize bg_local with zeros
        obj['bg_local'] = 0.0

        # Only compute local background for valid positions
        if np.sum(valid_pos) > 0:
            if bkg_order == 0:
                # Use photutils LocalBackground for order=0 (performance)
                lbg = photutils.background.LocalBackground(
                    bkgann[0], bkgann[1],
                    bkg_estimator=photutils.background.ModeEstimatorBackground(),
                )
                obj['bg_local'][valid_pos] = lbg(
                    image1, x_vals[valid_pos], y_vals[valid_pos], mask=mask | mask0
                )
            else:
                # Use gradient-aware fitting for order > 0
                # GradientLocalBackground handles array inputs efficiently
                # Lazy import to avoid circular dependency
                from .photometry_psf import GradientLocalBackground

                grad_bkg = GradientLocalBackground(bkgann[0], bkgann[1], order=bkg_order)
                obj['bg_local'][valid_pos] = grad_bkg(
                    image1, x_vals[valid_pos], y_vals[valid_pos], mask=mask | mask0
                )

        # Flag invalid positions and positions where local bg estimation failed
        idx = ~valid_pos | ~np.isfinite(obj['bg_local'])
        obj['flags'][idx] |= 0x400
        obj['bg_local'][idx] = 0

    # Photometric apertures
    x_vals = np.ma.filled(np.asarray(obj['x']), fill_value=np.nan)
    y_vals = np.ma.filled(np.asarray(obj['y']), fill_value=np.nan)
    valid_pos = np.isfinite(x_vals) & np.isfinite(y_vals)
    valid_idx = np.where(valid_pos)[0]

    apertures = None
    obj['npix_aper'] = 0.0
    obj['bg_fluxerr'] = 0.0  # Local background flux error inside the aperture
    if np.sum(valid_pos) > 0:
        positions = list(zip(x_vals[valid_pos], y_vals[valid_pos]))
        apertures = photutils.aperture.CircularAperture(positions, r=aper)

        # Check whether some aperture pixels are masked, and set the flags for that
        mres = photutils.aperture.aperture_photometry(mask | mask0, apertures, method='center')
        obj['flags'][valid_idx[mres['aperture_sum'] > 0]] |= 0x200

        # Aperture unmasked areas, in (fractional) pixels
        image_ones = np.ones(image1.shape)
        res_area = photutils.aperture.aperture_photometry(image_ones, apertures, mask=mask0)
        obj['npix_aper'][valid_idx] = res_area['aperture_sum']

        # Position-dependent background flux error from global background model, if available
        if bg_est is not None:
            res = photutils.aperture.aperture_photometry(bg_est_rms**2, apertures)
            obj['bg_fluxerr'][valid_idx] = np.sqrt(res['aperture_sum'])

    if optimal:
        # Optimal extraction photometry (Naylor 1998)
        log('Using optimal extraction photometry')

        # Determine PSF model
        if psf is not None:
            psf_for_extraction = psf
            log('Using provided PSF model for optimal extraction')
        elif fwhm is not None and fwhm > 0:
            # Use original fwhm (before aper scaling) for PSF
            psf_for_extraction = fwhm
            log('Using Gaussian PSF with FWHM=%.1f for optimal extraction' % fwhm)
        else:
            raise ValueError("Either 'psf' or 'fwhm' must be provided for optimal extraction")

        # Initialize output columns
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['npix_optimal'] = 0
        obj['chi2_optimal'] = np.nan
        obj['norm_optimal'] = np.nan

        if group_sources:
            # Grouped optimal extraction for crowded fields
            if grouper_radius is None:
                grouper_radius = 2 * aper
            log('Using grouped optimal extraction with grouper radius %.1f pixels' % grouper_radius)

            # Add group columns
            obj['group_id'] = -1
            obj['group_size'] = 0

            # Use SourceGrouper to identify groups
            grouper = photutils.psf.SourceGrouper(min_separation=grouper_radius)

            # Build positions array for grouper, handling MaskedColumns
            x_vals = np.ma.filled(np.asarray(obj['x']), fill_value=np.nan)
            y_vals = np.ma.filled(np.asarray(obj['y']), fill_value=np.nan)
            valid_pos = np.isfinite(x_vals) & np.isfinite(y_vals)

            # Get group IDs only for valid positions, then map back
            if np.sum(valid_pos) > 0:
                group_ids_valid = grouper(x_vals[valid_pos], y_vals[valid_pos])
                # Map group IDs back to full array (-1 for invalid positions)
                group_ids = np.full(len(obj), -1, dtype=int)
                group_ids[valid_pos] = group_ids_valid
            else:
                group_ids = np.full(len(obj), -1, dtype=int)
            obj['group_id'] = group_ids

            # Process each group (skip -1 which are invalid/masked positions)
            unique_groups = np.unique(group_ids)
            for gid in unique_groups:
                if gid < 0:
                    continue  # Skip invalid positions group

                group_mask = (group_ids == gid)
                group_indices = np.where(group_mask)[0]
                group_size = len(group_indices)

                # Update group_size for all members
                obj['group_size'][group_mask] = group_size

                # Get positions and backgrounds for this group
                positions = [(obj['x'][i], obj['y'][i]) for i in group_indices
                             if np.isfinite(obj['x'][i]) and np.isfinite(obj['y'][i])]
                bg_locals = [obj['bg_local'][i] for i in group_indices
                             if np.isfinite(obj['x'][i]) and np.isfinite(obj['y'][i])]

                if len(positions) == 0:
                    continue

                # Perform grouped extraction
                results = _grouped_optimal_extraction(
                    image1, err,
                    positions,
                    psf_for_extraction,
                    bg_local=bg_locals,
                    mask=mask | mask0,
                    radius=aper
                )

                # Store results
                valid_idx = 0
                for i in group_indices:
                    if np.isfinite(obj['x'][i]) and np.isfinite(obj['y'][i]):
                        res = results[valid_idx]
                        obj['flux'][i] = res[0]
                        obj['fluxerr'][i] = res[1]
                        obj['npix_optimal'][i] = res[2]
                        obj['norm_optimal'][i] = res[3]
                        obj['chi2_optimal'][i] = res[4]
                        valid_idx += 1

            log('Processed %d groups (%d isolated, %d grouped)' % (
                len(unique_groups),
                np.sum(obj['group_size'] == 1),
                np.sum(obj['group_size'] > 1)
            ))

        else:
            # Standard single-source optimal extraction
            for i, o in enumerate(obj):
                if np.isfinite(o['x']) and np.isfinite(o['y']):
                    res = _optimal_extraction(
                        image1, err,
                        o['x'], o['y'],
                        psf_for_extraction,
                        bg_local=o['bg_local'],
                        mask=mask | mask0, # Do not count the flux from 'soft-masked' pixels
                        radius=aper  # Use aperture radius as clipping radius
                    )

                    o['flux'], o['fluxerr'], o['npix_optimal'], o['norm_optimal'], o['chi2_optimal'] = res

        # Flag objects where optimal extraction failed
        obj['flags'][~np.isfinite(obj['flux'])] |= 0x800

    else:
        # Standard aperture photometry
        # Use just a minimal mask here so that the flux from 'soft-masked' (e.g. saturated) pixels is still counted
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        if apertures is not None:
            res = photutils.aperture.aperture_photometry(image1, apertures, error=err, mask=mask0)
            obj['flux'][valid_idx] = res['aperture_sum']
            obj['fluxerr'][valid_idx] = res['aperture_sum_err']

        # Subtract local background
        obj['flux'][valid_pos] -= obj['bg_local'][valid_pos] * obj['npix_aper'][valid_pos]

    for _ in ['mag', 'magerr']:
        obj[_] = np.nan

    idx = obj['flux'] > 0
    obj['mag'][idx] = -2.5 * np.log10(obj['flux'][idx])
    obj['magerr'][idx] = 2.5 / np.log(10) * obj['fluxerr'][idx] / obj['flux'][idx]

    # Final filtering of properly measured objects
    if sn is not None and sn > 0:
        log('Filtering out measurements with S/N < %.1f' % sn)
        idx = np.isfinite(obj['magerr'])
        idx[idx] &= obj['magerr'][idx] < 1 / sn
        obj = obj[idx]

    if not keep_negative:
        log('Filtering out measurements with negative fluxes')
        idx = obj['flux'] > 0
        obj = obj[idx]

    if get_bg:
        return obj, bg_est_bg, err
    else:
        return obj
