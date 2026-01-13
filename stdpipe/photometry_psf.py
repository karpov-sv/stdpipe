"""
Routines for PSF photometry using photutils.

This module provides PSF fitting photometry as an alternative to aperture
photometry, which is more accurate for point sources especially in crowded
fields or when PSF wings are significant.
"""


import numpy as np
from astropy.table import Table
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats

import photutils
import photutils.background
import photutils.psf
from photutils.utils import calc_total_error

from . import photometry as phot
from . import psf as psf_module

# Re-export for backward compatibility
from .psf import create_psf_model


def _odd_int(value, min_value=1):
    value = int(np.round(value))
    if value < min_value:
        value = min_value
    if value % 2 == 0:
        value += 1
    return value


def _compute_oversampling(psf_sampling):
    if psf_sampling is None or not np.isfinite(psf_sampling) or psf_sampling <= 0:
        return 1
    if psf_sampling >= 1.0:
        return 1
    return max(1, int(np.rint(1.0 / psf_sampling)))


def _compute_native_psf_size(psf_height, psf_sampling):
    if psf_sampling is None or not np.isfinite(psf_sampling) or psf_sampling <= 0:
        size = psf_height
    elif psf_sampling >= 1.0:
        size = psf_height
    else:
        size = psf_height * psf_sampling
    return _odd_int(size)


def _scale_psf_image_for_photutils(psf_image, oversampling):
    if oversampling is None:
        return psf_image
    factor = float(oversampling) ** 2
    if factor <= 1.0:
        return psf_image
    return psf_image * factor


def _build_circular_fit_mask(shape, x_positions, y_positions, radius):
    mask_fit = np.ones(shape, dtype=bool)
    r2 = radius ** 2
    for x, y in zip(x_positions, y_positions):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x0 = int(np.floor(x - radius))
        x1 = int(np.ceil(x + radius)) + 1
        y0 = int(np.floor(y - radius))
        y1 = int(np.ceil(y + radius)) + 1
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(shape[1], x1)
        y1 = min(shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= r2
        sub = mask_fit[y0:y1, x0:x1]
        sub[inside] = False
    return mask_fit


class GradientLocalBackground(photutils.background.LocalBackground):
    """
    Local background estimator using gradient fitting with sigma-clipping.

    Inherits from photutils.background.LocalBackground but overrides the estimation
    method to fit polynomial gradients instead of taking mean/median.

    Instead of taking mean/median of annulus (assumes flat background),
    fits a polynomial model to the annulus and evaluates at source position.
    Includes sigma-clipping to reject outliers (contaminating sources).

    This dramatically reduces biases with background gradients:
    - Linear gradients: ~20× improvement (19% → <1% error)
    - Quadratic gradients: ~100-400× improvement (-415% → <5% error)
    - Sigma-clipping provides robustness in crowded fields

    Parameters
    ----------
    inner_radius : float
        Inner radius of annulus in pixels
    outer_radius : float
        Outer radius of annulus in pixels
    order : int, optional
        Polynomial order:
        0 = constant (mean, equivalent to standard LocalBackground)
        1 = plane (linear gradient, recommended)
        2 = quadratic surface (complex gradients)
        Default is 1.
    sigma : float, optional
        Sigma threshold for sigma-clipping outliers. Default is 3.0.
        Higher values are more permissive, lower values reject more outliers.
    maxiters : int, optional
        Maximum number of sigma-clipping iterations. Default is 3.
    """

    def __init__(self, inner_radius, outer_radius, order=1, sigma=3.0, maxiters=3):
        # Initialize parent class with dummy bkg_estimator (we'll override __call__)
        super().__init__(inner_radius, outer_radius, bkg_estimator=None)
        self.order = order
        self.sigma = sigma
        self.maxiters = maxiters

    def __call__(self, data, x, y, mask=None):
        """
        Estimate local background at position(s) (x, y).

        Parameters
        ----------
        data : 2D ndarray
            Image data
        x, y : float or array-like
            Source position(s)
        mask : 2D bool ndarray, optional
            Mask (True = masked)

        Returns
        -------
        bg : float or ndarray
            Background value(s) at source position(s)
        """
        # Handle scalar vs array input
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        scalar_input = (len(x) == 1)

        if mask is None:
            mask = np.zeros_like(data, dtype=bool)

        bg_values = np.zeros(len(x))

        size_y, size_x = data.shape
        yy, xx = np.mgrid[:size_y, :size_x]

        for i, (xi, yi) in enumerate(zip(x, y)):
            # Distance from source
            dist = np.sqrt((xx - xi)**2 + (yy - yi)**2)

            # Annulus mask
            annulus_mask = (dist >= self.inner_radius) & (dist <= self.outer_radius) & ~mask

            if not np.any(annulus_mask):
                # Fallback: return median of unmasked data
                bg_values[i] = np.median(data[~mask]) if np.any(~mask) else 0.0
                continue

            # Get annulus data
            x_annulus = xx[annulus_mask].ravel()
            y_annulus = yy[annulus_mask].ravel()
            z_annulus = data[annulus_mask].ravel()

            # Remove NaN/Inf
            valid = np.isfinite(z_annulus)
            x_annulus = x_annulus[valid]
            y_annulus = y_annulus[valid]
            z_annulus = z_annulus[valid]

            if len(z_annulus) < max(10, (self.order + 1) * (self.order + 2) // 2):
                # Not enough points, fallback to mean
                bg_values[i] = np.mean(z_annulus) if len(z_annulus) > 0 else 0.0
                continue

            # Sigma-clipping to reject outliers (contaminating sources)
            # Iteratively fit, compute residuals, reject outliers, refit
            good_mask = np.ones(len(z_annulus), dtype=bool)

            for iteration in range(self.maxiters):
                x_good = x_annulus[good_mask]
                y_good = y_annulus[good_mask]
                z_good = z_annulus[good_mask]

                if len(z_good) < max(10, (self.order + 1) * (self.order + 2) // 2):
                    # Too many rejections, stop
                    break

                # Fit current good points
                if self.order == 0:
                    # Constant (mean)
                    bg_fit = np.mean(z_good)
                    residuals = z_good - bg_fit

                elif self.order == 1:
                    # Plane: z = a + b*(x-x0) + c*(y-y0)
                    dx = x_good - xi
                    dy = y_good - yi
                    A = np.column_stack([np.ones_like(x_good), dx, dy])

                    try:
                        coeffs = np.linalg.lstsq(A, z_good, rcond=None)[0]
                        residuals = z_good - (coeffs[0] + coeffs[1] * dx + coeffs[2] * dy)
                    except np.linalg.LinAlgError:
                        bg_fit = np.mean(z_good)
                        residuals = z_good - bg_fit

                elif self.order == 2:
                    # Quadratic: z = a + b*dx + c*dy + d*dx^2 + e*dy^2 + f*dx*dy
                    dx = x_good - xi
                    dy = y_good - yi
                    A = np.column_stack([
                        np.ones_like(x_good),
                        dx, dy,
                        dx**2, dy**2,
                        dx * dy
                    ])

                    try:
                        coeffs = np.linalg.lstsq(A, z_good, rcond=None)[0]
                        residuals = z_good - (
                            coeffs[0] + coeffs[1] * dx + coeffs[2] * dy +
                            coeffs[3] * dx**2 + coeffs[4] * dy**2 + coeffs[5] * dx * dy
                        )
                    except np.linalg.LinAlgError:
                        bg_fit = np.mean(z_good)
                        residuals = z_good - bg_fit
                else:
                    raise ValueError(f"order={self.order} not supported. Use 0, 1, or 2.")

                # Compute sigma from residuals
                sigma_residuals = np.std(residuals)

                if sigma_residuals == 0:
                    # Perfect fit or constant values, stop
                    break

                # Find outliers in the FULL dataset (not just current good points)
                # Compute residuals for all points using current fit
                if self.order == 0:
                    all_residuals = z_annulus[good_mask] - bg_fit
                elif self.order == 1:
                    dx_all = x_annulus[good_mask] - xi
                    dy_all = y_annulus[good_mask] - yi
                    all_residuals = z_annulus[good_mask] - (coeffs[0] + coeffs[1] * dx_all + coeffs[2] * dy_all)
                elif self.order == 2:
                    dx_all = x_annulus[good_mask] - xi
                    dy_all = y_annulus[good_mask] - yi
                    all_residuals = z_annulus[good_mask] - (
                        coeffs[0] + coeffs[1] * dx_all + coeffs[2] * dy_all +
                        coeffs[3] * dx_all**2 + coeffs[4] * dy_all**2 + coeffs[5] * dx_all * dy_all
                    )

                # Reject outliers beyond sigma threshold
                outliers = np.abs(all_residuals) > self.sigma * sigma_residuals

                if not np.any(outliers):
                    # No more outliers, converged
                    break

                # Update mask - create new mask relative to original good_mask
                good_indices = np.where(good_mask)[0]
                good_mask[good_indices[outliers]] = False

            # Final fit with cleaned data
            x_final = x_annulus[good_mask]
            y_final = y_annulus[good_mask]
            z_final = z_annulus[good_mask]

            if len(z_final) < max(10, (self.order + 1) * (self.order + 2) // 2):
                # Sigma-clipping rejected too many points, use all data
                x_final = x_annulus
                y_final = y_annulus
                z_final = z_annulus

            # Fit gradient with cleaned data
            if self.order == 0:
                # Constant (mean)
                bg_values[i] = np.mean(z_final)

            elif self.order == 1:
                # Plane: z = a + b*(x-x0) + c*(y-y0)
                dx = x_final - xi
                dy = y_final - yi
                A = np.column_stack([np.ones_like(x_final), dx, dy])

                try:
                    coeffs = np.linalg.lstsq(A, z_final, rcond=None)[0]
                    bg_values[i] = coeffs[0]  # Value at source position
                except np.linalg.LinAlgError:
                    bg_values[i] = np.mean(z_final)

            elif self.order == 2:
                # Quadratic: z = a + b*dx + c*dy + d*dx^2 + e*dy^2 + f*dx*dy
                dx = x_final - xi
                dy = y_final - yi
                A = np.column_stack([
                    np.ones_like(x_final),
                    dx, dy,
                    dx**2, dy**2,
                    dx * dy
                ])

                try:
                    coeffs = np.linalg.lstsq(A, z_final, rcond=None)[0]
                    bg_values[i] = coeffs[0]  # Value at source position
                except np.linalg.LinAlgError:
                    bg_values[i] = np.mean(z_final)

            else:
                raise ValueError(f"order={self.order} not supported. Use 0, 1, or 2.")

        return bg_values[0] if scalar_input else bg_values

    def __repr__(self):
        return (f"GradientLocalBackground(inner_radius={self.inner_radius}, "
                f"outer_radius={self.outer_radius}, order={self.order}, "
                f"sigma={self.sigma}, maxiters={self.maxiters})")


def measure_objects_psf(
    obj,
    image,
    psf=None,
    psf_size=None,
    fwhm=None,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    bkgann=None,
    bkg_order=1,
    sn=None,
    fit_shape='circular',
    fit_size=None,
    maxiters=3,
    recentroid=True,
    keep_negative=True,
    get_bg=False,
    use_position_dependent_psf=False,
    group_sources=True,
    grouper_radius=None,
    verbose=False,
):
    """PSF photometry at the positions of already detected objects using photutils.

    Performs PSF fitting photometry which is more accurate than aperture photometry,
    especially for point sources in crowded fields or when accurate flux measurement
    of PSF wings is important.

    This function will estimate and subtract the background unless external background
    estimation (`bg`) is provided, and use user-provided noise map (`err`) if requested.

    If a PSF model is not provided, a simple Gaussian PSF will be constructed based on
    the `fwhm` parameter or estimated from the data.

    :param obj: astropy.table.Table with initial object detections to be measured. Must have 'x' and 'y' columns.
    :param image: Input image as a NumPy array
    :param psf: PSF model to use. Can be:
        - photutils PSF model (e.g., IntegratedGaussianPRF, FittableImageModel)
        - PSFEx PSF structure from :func:`stdpipe.psf.run_psfex`
        - None (will create Gaussian PSF based on fwhm)
    :param psf_size: Size of the PSF model in pixels. If None, will be estimated from PSF or set to 5*fwhm
    :param fwhm: Full width at half maximum in pixels. Used if PSF model is not provided, or to estimate psf_size. If None, will be estimated from obj['fwhm'] if available.
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array with same shape as input image) instead of automatically computed one
    :param err: Image noise map as a NumPy array to be used instead of automatically computed one, optional
    :param gain: Image gain, e/ADU, used to build image noise model
    :param bg_size: Background grid size in pixels
    :param bkgann: Background annulus for local background estimation, [inner_radius, outer_radius] in pixels. If None, no local background subtraction is performed (relies only on global Background2D subtraction). If set, uses gradient-aware local background fitting to handle non-uniform backgrounds. Note: radii are NOT scaled by FWHM (unlike measure_objects). Typical values: [8, 12] pixels.
    :param bkg_order: Polynomial order for gradient-aware background fitting in annulus. Only used if bkgann is set. 0 = constant (mean), 1 = plane (linear gradient, recommended), 2 = quadratic surface (complex gradients). Default is 1. Gradient-aware fitting dramatically reduces biases with non-uniform backgrounds: linear gradients improved 20× (19% → <1% error), quadratic gradients improved 100-400× (-415% → <5% error).
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param fit_shape: Shape of fitting region. Options: 'circular' (default), 'square'. Determines the aperture used for PSF fitting.
    :param fit_size: Size of fitting region in pixels. If None, defaults to psf_size.
    :param maxiters: Maximum number of iterations for PSF fitting
    :param recentroid: If True, allow PSF position to vary during fitting (recommended)
    :param keep_negative: If not set, measurements with negative fluxes will be discarded
    :param get_bg: If True, the routine will also return estimated background and background noise images
    :param use_position_dependent_psf: If True and PSF is a PSFEx model, use polynomial evaluation for position-dependent PSF (evaluates PSF at each source position)
    :param group_sources: If True, use grouped PSF fitting for overlapping sources. Simultaneously fits fluxes for nearby sources, properly accounting for flux sharing between overlapping PSFs. Dramatically more accurate in crowded fields (51× improvement at 0.5 FWHM, 14× at 1.0 FWHM, 3× at 1.5 FWHM). Default is True (recommended). Set to False only for known sparse fields where performance is critical. No downside for isolated sources (identical results at >3 FWHM separation).
    :param grouper_radius: Radius in pixels for grouping nearby sources. If None, defaults to 2*psf_size. Only used if group_sources=True
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The copy of original table with `flux`, `fluxerr`, `mag`, `magerr`, `x_psf`, `y_psf` columns from PSF fitting. Also includes quality of fit columns: `qfit_psf` (fit quality, 0=good), `cfit_psf` (central pixel fit quality), `flags_psf` (photutils fit flags), `npix_psf` (number of unmasked pixels used in fit), and `reduced_chi2_psf` (reduced chi-squared, available in photutils >= 2.3.0). If :code:`get_bg=True`, also returns the background and background error images.

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

    # Operate on the copy of the table
    obj = obj.copy()

    # Sanitize the image and make its copy to safely operate on it
    image1 = image.astype(np.double)
    mask0 = ~np.isfinite(image1)  # Minimal mask

    # Ensure that the mask is defined
    if mask is None:
        mask = mask0
    else:
        mask = mask.astype(bool)

    # Background estimation
    if bg is None or err is None or get_bg:
        log('Estimating global background with %dx%d mesh' % (bg_size, bg_size))
        bg_est = photutils.background.Background2D(
            image1, bg_size, mask=mask | mask0, exclude_percentile=90
        )
        bg_est_bg = bg_est.background
        bg_est_rms = bg_est.background_rms
    else:
        bg_est = None

    if bg is None:
        log(
            'Subtracting global background: median %.1f rms %.2f'
            % (np.median(bg_est_bg), np.std(bg_est_bg))
        )
        image1 -= bg_est_bg
    else:
        log(
            'Subtracting user-provided background: median %.1f rms %.2f'
            % (np.median(bg), np.std(bg))
        )
        image1 -= bg

    image1[mask0] = 0

    # Error map
    if err is None:
        log(
            'Using global background noise map: median %.1f rms %.2f + gain %.1f'
            % (
                np.median(bg_est_rms),
                np.std(bg_est_rms),
                gain if gain else np.inf,
            )
        )
        err = bg_est_rms
        if gain:
            err = calc_total_error(image1, err, gain)
    else:
        log('Using user-provided noise map: median %.1f rms %.2f' % (np.median(err), np.std(err)))

    # Estimate FWHM if not provided
    if fwhm is None:
        if 'fwhm' in obj.colnames:
            # Use median FWHM from detections
            fwhm_vals = obj['fwhm'][np.isfinite(obj['fwhm'])]
            if len(fwhm_vals) > 0:
                fwhm = np.median(fwhm_vals)
                log('Using median FWHM from detections: %.2f pixels' % fwhm)
            else:
                fwhm = 3.0
                log('No valid FWHM values in detections, using default: %.2f pixels' % fwhm)
        else:
            fwhm = 3.0
            log('FWHM not provided and not in object table, using default: %.2f pixels' % fwhm)

    # Create or process PSF model
    psf_is_position_dependent = False  # Track if PSF varies with position

    if psf is None:
        # Create a simple Gaussian PSF
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        log('Creating Gaussian PSF model with sigma=%.2f pixels (FWHM=%.2f)' % (sigma, fwhm))

        # Use CircularGaussianSigmaPRF (replaces deprecated IntegratedGaussianPRF)
        psf_model = photutils.psf.CircularGaussianSigmaPRF(sigma=sigma)

        if psf_size is None:
            psf_size = _odd_int(5 * fwhm)

    elif isinstance(psf, dict) and 'data' in psf and 'sampling' in psf:
        # PSFEx-like dict structure (from run_psfex, load_psf, or create_psf_model)
        psf_data = psf['data']
        psf_sampling = psf['sampling']
        psf_degree = psf.get('degree', 0)

        if use_position_dependent_psf and psf_degree > 0:
            log('Using position-dependent PSFEx PSF model (degree=%d)' % psf_degree)
            # Store the PSFEx model for later use
            # We'll handle position-dependent photometry specially
            psf_model = psf  # Keep original PSFEx dict
            psf_is_position_dependent = True
            if psf_size is None:
                psf_size = _compute_native_psf_size(psf['height'], psf_sampling)
        else:
            log('Using PSFEx/ePSF PSF model (constant across field)')
            # Get PSF stamp at center position (0,0 works for degree=0)
            psf_image = psf_module.get_supersampled_psf_stamp(psf, x=0, y=0, normalize=True)

            # Handle oversampling if needed
            oversampling = _compute_oversampling(psf_sampling)
            psf_image = _scale_psf_image_for_photutils(psf_image, oversampling)
            psf_model = photutils.psf.ImagePSF(psf_image, oversampling=oversampling)
            psf_is_position_dependent = False

            if psf_size is None:
                psf_size = _compute_native_psf_size(psf_image.shape[0], psf_sampling)

    elif isinstance(psf, (photutils.psf.ImagePSF, photutils.psf.FittableImageModel)):
        # Already a photutils PSF model (ImagePSF or legacy FittableImageModel)
        log('Using provided photutils ImagePSF model')
        psf_model = psf
        if psf_size is None:
            psf_size = _odd_int(psf.data.shape[0])

    elif hasattr(psf, 'fwhm'):
        # Photutils ePSF or similar
        log('Using provided photutils PSF model with FWHM')
        psf_model = psf
        if psf_size is None:
            psf_size = _odd_int(psf.data.shape[0]) if hasattr(psf, 'data') else _odd_int(5 * psf.fwhm)

    else:
        # Assume it's a photutils PSF model
        log('Using provided PSF model')
        psf_model = psf
        if psf_size is None:
            psf_size = _odd_int(5 * fwhm)

    log('Using PSF size: %d pixels' % psf_size)

    # Fitting region size
    if fit_size is None:
        fit_size = psf_size
    fit_size = _odd_int(fit_size)
    log('Using fitting region size: %d pixels' % fit_size)

    # Prepare initial positions table
    # Convert MaskedColumns to regular arrays, replacing masked values with NaN
    init_params = Table()
    init_params['x'] = np.ma.filled(np.asarray(obj['x']), fill_value=np.nan)
    init_params['y'] = np.ma.filled(np.asarray(obj['y']), fill_value=np.nan)

    # Track which positions are valid (not NaN/masked)
    valid_pos = np.isfinite(init_params['x']) & np.isfinite(init_params['y'])

    # Add initial flux guesses if available
    if 'flux' in obj.colnames:
        init_params['flux'] = np.ma.filled(np.asarray(obj['flux']), fill_value=np.nan)
    else:
        # Estimate initial flux from image at positions
        init_params['flux'] = 1000.0  # Default initial guess

    if fit_shape not in ['circular', 'square']:
        raise ValueError("fit_shape must be 'circular' or 'square'")

    fit_mask = None
    if fit_shape == 'circular' and np.any(valid_pos):
        fit_mask = _build_circular_fit_mask(
            image1.shape,
            init_params['x'][valid_pos],
            init_params['y'][valid_pos],
            radius=fit_size / 2,
        )

    # Import fitting class
    from astropy.modeling.fitting import LevMarLSQFitter

    # Configure grouping if requested
    grouper = None
    if group_sources:
        if grouper_radius is None:
            grouper_radius = 2 * psf_size
        log('Using grouped PSF fitting with grouper radius %.1f pixels' % grouper_radius)
        grouper = photutils.psf.SourceGrouper(min_separation=grouper_radius)

    # Check for invalid positions (from masked columns)
    n_invalid = np.sum(~valid_pos)
    if n_invalid > 0:
        log('Found %d objects with invalid (masked/NaN) positions, will be skipped' % n_invalid)

    mask_for_fit = mask | mask0
    if fit_mask is not None:
        mask_for_fit = mask_for_fit | fit_mask

    xy_bounds = None if recentroid else 1e-6

    # Perform PSF photometry
    log('Performing PSF photometry on %d objects (%d valid)' % (len(obj), np.sum(valid_pos)))
    log('Settings: %d iterations, recentroid=%s, grouped=%s, position_dependent=%s' % (maxiters, recentroid, group_sources, psf_is_position_dependent))

    # Handle position-dependent PSF separately
    if psf_is_position_dependent:
        log('Performing position-dependent PSF photometry (iterative mode)')
        # Initialize output columns
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['x_psf'] = obj['x']
        obj['y_psf'] = obj['y']
        obj['qfit_psf'] = np.nan
        obj['cfit_psf'] = np.nan
        obj['flags_psf'] = 0
        obj['npix_psf'] = 0
        obj['reduced_chi2_psf'] = np.nan
        if 'flags' not in obj.keys():
            obj['flags'] = 0

        # Get sampling (psf_model is always dict at this point)
        psf_sampling = psf_model['sampling']
        oversampling = _compute_oversampling(psf_sampling)

        # Process objects individually or in small groups
        # For each object, evaluate PSF at its position
        for i in range(len(obj)):
            # Skip invalid positions (masked/NaN)
            if not valid_pos[i]:
                obj['flux'][i] = np.nan
                obj['fluxerr'][i] = np.nan
                obj['flags'][i] |= 0x1000
                continue

            try:
                # Get object position
                x_pos = float(init_params['x'][i])
                y_pos = float(init_params['y'][i])

                # Evaluate PSF at this position using dict-based PSF
                psf_image = psf_module.get_supersampled_psf_stamp(
                    psf_model, x=x_pos, y=y_pos, normalize=True
                )
                psf_image = _scale_psf_image_for_photutils(psf_image, oversampling)

                # Create photutils PSF model for this position
                psf_at_pos = photutils.psf.ImagePSF(psf_image, oversampling=oversampling)

                # Set up local background estimator if requested
                localbkg_estimator = None
                if bkgann is not None and len(bkgann) == 2:
                    localbkg_estimator = GradientLocalBackground(bkgann[0], bkgann[1], order=bkg_order)

                # Set up photometry for this object
                phot_single = photutils.psf.PSFPhotometry(
                    psf_model=psf_at_pos,
                    fit_shape=fit_size,
                    finder=None,
                    grouper=grouper,
                    fitter=LevMarLSQFitter(),
                    fitter_maxiters=maxiters,
                    xy_bounds=xy_bounds,
                    aperture_radius=fit_size / 2,
                    localbkg_estimator=localbkg_estimator
                )

                # Measure this object
                init_single = Table()
                init_single['x'] = [obj['x'][i]]
                init_single['y'] = [obj['y'][i]]
                if 'flux' in init_params.colnames:
                    init_single['flux'] = [init_params['flux'][i]]
                else:
                    init_single['flux'] = [1000.0]

                result_single = phot_single(
                    image1,
                    mask=mask_for_fit,
                    error=err,
                    init_params=init_single
                )

                # Extract results
                obj['flux'][i] = result_single['flux_fit'][0]
                obj['fluxerr'][i] = result_single['flux_err'][0]
                obj['x_psf'][i] = result_single['x_fit'][0]
                obj['y_psf'][i] = result_single['y_fit'][0]

                # Extract quality of fit columns if available
                if 'qfit' in result_single.colnames:
                    obj['qfit_psf'][i] = result_single['qfit'][0]
                if 'cfit' in result_single.colnames:
                    obj['cfit_psf'][i] = result_single['cfit'][0]
                if 'flags' in result_single.colnames:
                    obj['flags_psf'][i] = result_single['flags'][0]
                if 'npixfit' in result_single.colnames:
                    obj['npix_psf'][i] = result_single['npixfit'][0]
                if 'reduced_chi2' in result_single.colnames:
                    obj['reduced_chi2_psf'][i] = result_single['reduced_chi2'][0]

                # Flag if fit failed
                if not np.isfinite(obj['flux'][i]):
                    obj['flags'][i] |= 0x1000

                # Flag if position moved significantly
                if recentroid:
                    if np.sqrt((obj['x_psf'][i] - obj['x'][i])**2 + (obj['y_psf'][i] - obj['y'][i])**2) > 1.0:
                        obj['flags'][i] |= 0x2000

            except Exception as e:
                log('PSF photometry failed for object %d: %s' % (i, str(e)))
                obj['flux'][i] = np.nan
                obj['fluxerr'][i] = np.nan
                obj['flags'][i] |= 0x1000

    else:
        # Standard (non-position-dependent) PSF photometry
        # Initialize output columns with NaN (for invalid positions)
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['x_psf'] = np.ma.filled(np.asarray(obj['x']), fill_value=np.nan)
        obj['y_psf'] = np.ma.filled(np.asarray(obj['y']), fill_value=np.nan)
        obj['qfit_psf'] = np.nan
        obj['cfit_psf'] = np.nan
        obj['flags_psf'] = 0
        obj['npix_psf'] = 0
        obj['reduced_chi2_psf'] = np.nan
        if 'flags' not in obj.keys():
            obj['flags'] = 0

        # Mark invalid positions as failed
        obj['flags'][~valid_pos] |= 0x1000

        # Only proceed if there are valid positions
        if np.sum(valid_pos) > 0:
            try:
                # Filter init_params to valid positions only
                init_params_valid = init_params[valid_pos]

                # Set up local background estimator if requested
                localbkg_estimator = None
                if bkgann is not None and len(bkgann) == 2:
                    inner_rad = bkgann[0]
                    outer_rad = bkgann[1]

                    order_names = {0: 'constant (mean)', 1: 'plane', 2: 'quadratic'}
                    order_name = order_names.get(bkg_order, f'order-{bkg_order}')
                    log('Using local background annulus %.1f-%.1f pixels with %s fitting' % (inner_rad, outer_rad, order_name))

                    # Create gradient-aware local background estimator
                    localbkg_estimator = GradientLocalBackground(inner_rad, outer_rad, order=bkg_order)

                # Set up photometry object
                phot_obj = photutils.psf.PSFPhotometry(
                    psf_model=psf_model,
                    fit_shape=fit_size,
                    finder=None,  # We already have positions
                    grouper=grouper,  # Group nearby sources if requested
                    fitter=LevMarLSQFitter(),  # Levenberg-Marquardt fitter from astropy
                    fitter_maxiters=maxiters,
                    xy_bounds=xy_bounds,
                    aperture_radius=fit_size / 2,
                    localbkg_estimator=localbkg_estimator
                )

                # Do the photometry - photutils 2.x API
                result = phot_obj(
                    image1,
                    mask=mask_for_fit,
                    error=err,
                    init_params=init_params_valid
                )

                # Map results back to full array
                obj['flux'][valid_pos] = result['flux_fit']
                obj['fluxerr'][valid_pos] = result['flux_err']
                obj['x_psf'][valid_pos] = result['x_fit']
                obj['y_psf'][valid_pos] = result['y_fit']

                # Extract quality of fit columns if available
                if 'qfit' in result.colnames:
                    obj['qfit_psf'][valid_pos] = result['qfit']
                if 'cfit' in result.colnames:
                    obj['cfit_psf'][valid_pos] = result['cfit']
                if 'flags' in result.colnames:
                    obj['flags_psf'][valid_pos] = result['flags']
                if 'npixfit' in result.colnames:
                    obj['npix_psf'][valid_pos] = result['npixfit']
                if 'reduced_chi2' in result.colnames:
                    # Available in photutils >= 2.3.0
                    obj['reduced_chi2_psf'][valid_pos] = result['reduced_chi2']

                # Flag objects where fit failed (NaN values)
                bad_idx = valid_pos & ~np.isfinite(obj['flux'])
                obj['flags'][bad_idx] |= 0x1000  # PSF fit failed

                # Flag objects where position moved significantly (>1 pixel)
                if recentroid:
                    moved_idx = valid_pos & (np.sqrt((obj['x_psf'] - init_params['x'])**2 +
                                                      (obj['y_psf'] - init_params['y'])**2) > 1.0)
                    obj['flags'][moved_idx] |= 0x2000  # Large centroid shift

            except Exception as e:
                log('PSF photometry failed: %s' % str(e))
                log('Falling back to NaN values')
                obj['flags'][valid_pos] |= 0x1000

    # Compute magnitudes
    idx = obj['flux'] > 0
    for _ in ['mag', 'magerr']:
        if _ not in obj.keys():
            obj[_] = np.nan

    obj['mag'][idx] = -2.5 * np.log10(obj['flux'][idx])
    obj['mag'][~idx] = np.nan

    obj['magerr'][idx] = 2.5 / np.log(10) * obj['fluxerr'][idx] / obj['flux'][idx]
    obj['magerr'][~idx] = np.nan

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

    log('PSF photometry complete: %d objects measured' % len(obj))

    if get_bg:
        return obj, bg_est_bg, err
    else:
        return obj
