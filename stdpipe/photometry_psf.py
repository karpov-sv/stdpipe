"""
Routines for PSF photometry using photutils.

This module provides PSF fitting photometry as an alternative to aperture
photometry, which is more accurate for point sources especially in crowded
fields or when PSF wings are significant.
"""

import numpy as np
from astropy.table import Table
from astropy.utils import minversion
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats

import photutils
import photutils.background
import photutils.psf
from photutils.utils import calc_total_error

from . import photometry as phot
from . import psf as psf_module
from .photometry_measure import _is_callable_fwhm, _fwhm_median

# Re-export for backward compatibility
from .psf import create_psf_model

# photutils PSFPhotometry flag for possible non-convergence (maxfev reached).
# Bit 1 means only that the fit region was smaller than fit_shape, due to
# masked pixels or image edges, and does not invalidate the fit.
_PHOTUTILS_FLAG_NONCONVERGED = 8

# photutils 3.0 renamed the local background arguments and the ``npixfit``
# result column; old names still work there, with deprecation warnings,
# until 4.0.
if minversion(photutils, '3.0'):
    _LOCALBKG_ARG = 'local_bkg_estimator'
    _INCLUDE_LOCALBKG_ARG = 'include_local_bkg'
    _NPIXFIT_COL = 'n_pixels_fit'
else:
    _LOCALBKG_ARG = 'localbkg_estimator'
    _INCLUDE_LOCALBKG_ARG = 'include_localbkg'
    _NPIXFIT_COL = 'npixfit'


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


def _compute_psf_quality_columns(
    phot_obj,
    psf_model,
    image1,
    err,
    mask_combined,
    x_fit,
    y_fit,
    flux_fit,
    fit_size,
    fwhm,
    log,
):
    """Build per-source residual stamps and compute crowdsource-style quality
    metrics (``qf``, ``fracflux``, ``spread_model``, ``dspread_model``) from the
    output of a successful :class:`photutils.psf.PSFPhotometry` run.

    Returns a dict of length-``N`` arrays, or ``None`` if anything fails.
    """
    from . import photometry_quality as pq

    N = len(x_fit)
    if N == 0:
        return None

    H, W = image1.shape
    half = int(fit_size) // 2
    sz = 2 * half + 1  # ensure odd

    # NB: the model image excludes the fitted local background, while the data
    # stamps below still contain it when ``bkgann`` is used — any residual
    # local background therefore leaks into ``fracflux`` / ``spread_model``.
    try:
        full_model = phot_obj.make_model_image(
            image1.shape, psf_shape=(sz, sz), **{_INCLUDE_LOCALBKG_ARG: False}
        )
    except Exception as e:
        log('Skipping PSF quality stats (model image build failed: %s)' % e)
        return None

    weight_full = np.zeros_like(image1, dtype='f4')
    if err is not None:
        finite = np.isfinite(err) & (err > 0)
        weight_full[finite] = 1.0 / err[finite]
    if mask_combined is not None:
        weight_full[mask_combined.astype(bool)] = 0.0

    impsf_stack = np.zeros((N, sz, sz), dtype='f4')
    im_stack = np.zeros((N, sz, sz), dtype='f4')
    psf_stack = np.zeros((N, sz, sz), dtype='f4')
    weight_stack = np.zeros((N, sz, sz), dtype='f4')

    x_fit = np.asarray(x_fit, dtype='f4')
    y_fit = np.asarray(y_fit, dtype='f4')
    flux_fit = np.asarray(flux_fit, dtype='f4')
    valid = np.isfinite(x_fit) & np.isfinite(y_fit) & np.isfinite(flux_fit)
    filled = np.zeros(N, dtype=bool)

    for i in range(N):
        if not valid[i]:
            continue

        cx = int(np.round(x_fit[i]))
        cy = int(np.round(y_fit[i]))
        x0, x1 = cx - half, cx + half + 1
        y0, y1 = cy - half, cy + half + 1

        sx0, sx1 = max(0, x0), min(W, x1)
        sy0, sy1 = max(0, y0), min(H, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        lx0, lx1 = sx0 - x0, sx1 - x0
        ly0, ly1 = sy0 - y0, sy1 - y0

        yy = np.arange(sy0, sy1, dtype='f4').reshape(-1, 1)
        xx = np.arange(sx0, sx1, dtype='f4').reshape(1, -1)

        try:
            m = psf_model.copy()
            if hasattr(m, 'x_0'):
                m.x_0 = float(x_fit[i])
            if hasattr(m, 'y_0'):
                m.y_0 = float(y_fit[i])
            if hasattr(m, 'flux'):
                m.flux = 1.0
            psf_local = np.asarray(m(xx, yy), dtype='f4')
        except Exception:
            continue

        single_local = psf_local * float(flux_fit[i])
        data_local = image1[sy0:sy1, sx0:sx1].astype('f4')
        model_local = full_model[sy0:sy1, sx0:sx1].astype('f4')

        psf_stack[i, ly0:ly1, lx0:lx1] = psf_local
        im_stack[i, ly0:ly1, lx0:lx1] = data_local
        weight_stack[i, ly0:ly1, lx0:lx1] = weight_full[sy0:sy1, sx0:sx1]
        # neighbour-subtracted = data - (full_model - this_source) = data - full_model + this_source
        impsf_stack[i, ly0:ly1, lx0:lx1] = data_local - model_local + single_local
        filled[i] = True

    # Per-source FWHM (FWHMMap callable, scalar, or None)
    if _is_callable_fwhm(fwhm):
        fwhm_arr = np.asarray(fwhm(x_fit, y_fit), dtype='f4')
    elif fwhm is not None:
        fwhm_arr = float(fwhm)
    else:
        fwhm_arr = None

    quality = pq.compute_psf_quality(
        impsf_stack, im_stack, psf_stack, weight_stack, fwhm=fwhm_arr
    )

    # Sources without a stamp (failed fit, off-image, model evaluation error)
    # would otherwise get zeros from their empty stamps — and spread_model == 0
    # reads as a clean star. Report NaN for them instead.
    for key in quality:
        quality[key] = np.asarray(quality[key], dtype='f4')
        quality[key][~filled] = np.nan

    return quality


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
        scalar_input = len(x) == 1

        if mask is None:
            mask = np.zeros_like(data, dtype=bool)

        bg_values = np.zeros(len(x))

        size_y, size_x = data.shape

        for i, (xi, yi) in enumerate(zip(x, y)):
            # Define bounding box for annulus region (OPTIMIZATION: avoid full-image arrays)
            r_outer = self.outer_radius
            x0 = max(0, int(xi - r_outer) - 1)
            x1 = min(size_x, int(xi + r_outer) + 2)
            y0 = max(0, int(yi - r_outer) - 1)
            y1 = min(size_y, int(yi + r_outer) + 2)

            # Create coordinate grids ONLY for bounding box (not full image)
            yy, xx = np.mgrid[y0:y1, x0:x1]

            # Distance from source (only compute for bounding box pixels)
            dist = np.sqrt((xx - xi) ** 2 + (yy - yi) ** 2)

            # Annulus mask
            annulus_mask = (dist >= self.inner_radius) & (dist <= self.outer_radius)
            if mask is not None:
                annulus_mask &= ~mask[y0:y1, x0:x1]

            if not np.any(annulus_mask):
                # Fallback: return median of unmasked data
                bg_values[i] = np.median(data[~mask]) if np.any(~mask) else 0.0
                continue

            # Get annulus data (extract from bounding box)
            x_annulus = xx[annulus_mask].ravel()
            y_annulus = yy[annulus_mask].ravel()
            data_bbox = data[y0:y1, x0:x1]
            z_annulus = data_bbox[annulus_mask].ravel()

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

                # Fit current good points. ``coeffs`` stays None when the
                # polynomial fit was not performed or failed, so the outlier
                # rejection below falls back to the constant ``bg_fit``.
                coeffs = None

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
                    A = np.column_stack([np.ones_like(x_good), dx, dy, dx**2, dy**2, dx * dy])

                    try:
                        coeffs = np.linalg.lstsq(A, z_good, rcond=None)[0]
                        residuals = z_good - (
                            coeffs[0]
                            + coeffs[1] * dx
                            + coeffs[2] * dy
                            + coeffs[3] * dx**2
                            + coeffs[4] * dy**2
                            + coeffs[5] * dx * dy
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

                # Compute residuals for the current good points using the fit
                # (falls back to the constant when the polynomial fit failed)
                if self.order == 0 or coeffs is None:
                    all_residuals = z_annulus[good_mask] - bg_fit
                elif self.order == 1:
                    dx_all = x_annulus[good_mask] - xi
                    dy_all = y_annulus[good_mask] - yi
                    all_residuals = z_annulus[good_mask] - (
                        coeffs[0] + coeffs[1] * dx_all + coeffs[2] * dy_all
                    )
                elif self.order == 2:
                    dx_all = x_annulus[good_mask] - xi
                    dy_all = y_annulus[good_mask] - yi
                    all_residuals = z_annulus[good_mask] - (
                        coeffs[0]
                        + coeffs[1] * dx_all
                        + coeffs[2] * dy_all
                        + coeffs[3] * dx_all**2
                        + coeffs[4] * dy_all**2
                        + coeffs[5] * dx_all * dy_all
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
                A = np.column_stack([np.ones_like(x_final), dx, dy, dx**2, dy**2, dx * dy])

                try:
                    coeffs = np.linalg.lstsq(A, z_final, rcond=None)[0]
                    bg_values[i] = coeffs[0]  # Value at source position
                except np.linalg.LinAlgError:
                    bg_values[i] = np.mean(z_final)

            else:
                raise ValueError(f"order={self.order} not supported. Use 0, 1, or 2.")

        return bg_values[0] if scalar_input else bg_values

    def __repr__(self):
        return (
            f"GradientLocalBackground(inner_radius={self.inner_radius}, "
            f"outer_radius={self.outer_radius}, order={self.order}, "
            f"sigma={self.sigma}, maxiters={self.maxiters})"
        )


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
    maxiters=100,
    recentroid=True,
    keep_negative=True,
    get_bg=False,
    use_position_dependent_psf=False,
    group_sources=True,
    grouper_radius=None,
    compute_quality=True,
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

    Parameters
    ----------
    obj : `~astropy.table.Table`
        Table with initial object detections to be measured. Must have 'x' and
        'y' columns.
    image : `~numpy.ndarray`
        Input image as a 2D NumPy array.
    psf : photutils PSF model, dict, or None, optional
        PSF model to use. Can be a photutils PSF model (e.g.,
        IntegratedGaussianPRF, FittableImageModel), a PSFEx PSF structure from
        :func:`stdpipe.psf.run_psfex`, or None (will create Gaussian PSF based
        on fwhm).
    psf_size : int or None, optional
        Size of the PSF model in pixels. If None, will be estimated from PSF or
        set to 5*fwhm.
    fwhm : float, callable or None, optional
        Full width at half maximum in pixels. Used if PSF model is not provided,
        or to estimate psf_size. If None, will be estimated from obj['fwhm'] if
        available. A position-dependent callable (e.g.
        :class:`stdpipe.photometry.FWHMMap`) is accepted: its scalar summary
        (median) is used for PSF model construction and sizing, while the
        per-source values are used for the PSF quality metrics.
    mask : `~numpy.ndarray` or None, optional
        Image mask as a boolean array (True values will be masked).
    bg : `~numpy.ndarray` or None, optional
        If provided, use this background (same shape as input image) instead of
        automatically computed one.
    err : `~numpy.ndarray` or None, optional
        Image noise map to be used instead of automatically computed one.
    gain : float or None, optional
        Image gain in e-/ADU, used to build image noise model.
    bg_size : int, optional
        Background grid size in pixels.
    bkgann : list of float or None, optional
        Background annulus for local background estimation, [inner_radius,
        outer_radius] in pixels. If None, no local background subtraction is
        performed (relies only on global Background2D subtraction). If set, uses
        gradient-aware local background fitting to handle non-uniform
        backgrounds. Note: radii are NOT scaled by FWHM (unlike
        measure_objects).
    bkg_order : int, optional
        Polynomial order for local background fitting. 0 = constant (mean),
        1 = plane (linear gradient, recommended), 2 = quadratic surface. Only
        used if bkgann is set.
    sn : float or None, optional
        Minimal S/N ratio for the object to be considered good. If set, all
        measurements with magnitude errors exceeding 1/sn will be discarded.
    fit_shape : str, optional
        Accepted for API compatibility ('circular' or 'square'). photutils
        PSFPhotometry always fits a square region of ``fit_size`` pixels, so
        both values currently behave identically.
    fit_size : int or None, optional
        Size of fitting region in pixels. If None, defaults to psf_size.
    maxiters : int, optional
        Maximum number of model evaluations of the least-squares fitter (it
        is passed to scipy as ``maxfev``), per source or group. Each fitting
        step needs several evaluations to estimate the Jacobian, so too
        small values stop the fit before convergence.
    recentroid : bool, optional
        If True, allow PSF position to vary during fitting (recommended).
    keep_negative : bool, optional
        If False, measurements with negative fluxes will be discarded.
    get_bg : bool, optional
        If True, the routine will also return estimated background and
        background noise images.
    use_position_dependent_psf : bool, optional
        If True and PSF is a PSFEx model, use polynomial evaluation for
        position-dependent PSF (evaluates PSF at each source position).
    group_sources : bool, optional
        If True, use grouped PSF fitting for overlapping sources. Fits nearby
        sources simultaneously for better accuracy in crowded fields.
    grouper_radius : float or None, optional
        Radius in pixels for grouping nearby sources. If None, defaults to
        ``max(2.5*fwhm, fit_size/2)``. Only used if group_sources is True.
    compute_quality : bool, optional
        If True (default), compute crowdsource-style per-source quality
        metrics: ``qf`` (PSF quality factor), ``fracflux`` (fraction of
        stamp flux explained by this source after subtracting the fitted
        models of overlapping neighbours), ``spread_model`` and
        ``dspread_model`` (SExtractor-like star/galaxy classifier and its
        uncertainty). Adds a small overhead per source. Not implemented for
        ``use_position_dependent_psf=True`` — the columns are then NaN.
    verbose : bool or callable, optional
        Whether to show verbose messages during the run. May be either boolean,
        or a ``print``-like function.

    Returns
    -------
    result : `~astropy.table.Table` or tuple
        Copy of original table with ``flux``, ``fluxerr``, ``mag``, ``magerr``
        columns from PSF fitting. With ``recentroid=True``, fitted positions
        of successful fits replace ``x``, ``y``, and the input ones are kept in
        ``x_orig``, ``y_orig``. Also includes quality of
        fit columns: ``qfit_psf`` (fit quality, 0=good), ``cfit_psf`` (central
        pixel fit quality), ``flags_psf`` (photutils fit flags), ``npix_psf``
        (number of unmasked pixels used in fit), and ``reduced_chi2_psf``
        (reduced chi-squared, available in photutils >= 2.3.0). If `get_bg` is
        True, returns a tuple of (table, background, background_error).

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if not len(obj):
        log('No objects to measure')
        return obj

    # Operate on the copy of the table
    obj = obj.copy()

    from .photometry_measure import (
        _prepare_image_and_mask,
        _extract_valid_positions,
        _compute_magnitudes_and_filter,
        _store_fitted_positions,
    )

    image1, mask0, mask = _prepare_image_and_mask(image, mask)

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

    # Scalar summary of the FWHM for PSF model construction and stamp
    # sizing; a position-dependent callable (e.g. FWHMMap) stays in ``fwhm``
    # for the per-source quality metrics.
    if _is_callable_fwhm(fwhm):
        fwhm_scalar = _fwhm_median(fwhm, image.shape)
    else:
        fwhm_scalar = fwhm

    # Create or process PSF model
    psf_is_position_dependent = False  # Track if PSF varies with position

    if psf is None:
        # Create a simple Gaussian PSF
        sigma = fwhm_scalar / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        log(
            'Creating Gaussian PSF model with sigma=%.2f pixels (FWHM=%.2f)'
            % (sigma, fwhm_scalar)
        )

        # Use CircularGaussianSigmaPRF (replaces deprecated IntegratedGaussianPRF)
        psf_model = photutils.psf.CircularGaussianSigmaPRF(sigma=sigma)

        if psf_size is None:
            psf_size = _odd_int(5 * fwhm_scalar)

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
            if psf_degree > 0:
                log(
                    'Using spatially varying PSFEx model as constant PSF '
                    '(evaluated at the polynomial zero-point; '
                    'set use_position_dependent_psf=True for per-source evaluation)'
                )
            else:
                log('Using PSFEx/ePSF PSF model (constant across field)')
            # Evaluate at the polynomial zero-point (~field centre); (0, 0)
            # would extrapolate a varying model to the image corner
            psf_image, psf_origin = psf_module._get_sampled_psf_stamp(
                psf, x=psf.get('x0', 0), y=psf.get('y0', 0)
            )

            # Handle oversampling if needed
            oversampling = _compute_oversampling(psf_sampling)
            psf_image = _scale_psf_image_for_photutils(psf_image, oversampling)
            psf_model = photutils.psf.ImagePSF(
                psf_image, oversampling=oversampling, origin=psf_origin
            )
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
            psf_size = (
                _odd_int(psf.data.shape[0]) if hasattr(psf, 'data') else _odd_int(5 * psf.fwhm)
            )

    else:
        # Assume it's a photutils PSF model
        log('Using provided PSF model')
        psf_model = psf
        if psf_size is None:
            psf_size = _odd_int(5 * fwhm_scalar)

    log('Using PSF size: %d pixels' % psf_size)

    # Fitting region size
    if fit_size is None:
        fit_size = psf_size
    fit_size = _odd_int(fit_size)
    log('Using fitting region size: %d pixels' % fit_size)

    # Prepare initial positions table
    x_vals, y_vals, valid_pos = _extract_valid_positions(obj)
    init_params = Table()
    init_params['x'] = x_vals
    init_params['y'] = y_vals

    # Add initial flux guesses if available
    if 'flux' in obj.colnames:
        flux0 = np.ma.filled(np.ma.asarray(obj['flux'], dtype=float), fill_value=np.nan)
        # A non-finite initial flux poisons the fit of the source (and, in
        # grouped mode, of its whole group) — fall back to a finite guess
        bad0 = ~np.isfinite(flux0)
        if np.any(bad0):
            finite0 = flux0[~bad0]
            flux0[bad0] = np.median(finite0) if len(finite0) else 1000.0
        init_params['flux'] = flux0
    else:
        # Estimate initial flux from image at positions
        init_params['flux'] = 1000.0  # Default initial guess

    if fit_shape not in ['circular', 'square']:
        raise ValueError("fit_shape must be 'circular' or 'square'")

    # Import fitting class
    from astropy.modeling.fitting import LevMarLSQFitter

    # Configure grouping if requested
    grouper = None
    if group_sources:
        if grouper_radius is None:
            # Sources interact when their fitting boxes overlap appreciably;
            # ~2.5 FWHM (or half the fit box) covers that while avoiding the
            # huge chained groups a stamp-sized radius produces in dense fields
            grouper_radius = max(2.5 * fwhm_scalar, 0.5 * fit_size)
        log('Using grouped PSF fitting with grouper radius %.1f pixels' % grouper_radius)
        grouper = photutils.psf.SourceGrouper(min_separation=grouper_radius)

    # Check for invalid positions (from masked columns)
    n_invalid = np.sum(~valid_pos)
    if n_invalid > 0:
        log('Found %d objects with invalid (masked/NaN) positions, will be skipped' % n_invalid)

    mask_for_fit = mask | mask0

    xy_bounds = None if recentroid else 1e-6

    # Perform PSF photometry
    log('Performing PSF photometry on %d objects (%d valid)' % (len(obj), np.sum(valid_pos)))
    log(
        'Settings: %d max evaluations, recentroid=%s, grouped=%s, position_dependent=%s'
        % (maxiters, recentroid, group_sources, psf_is_position_dependent)
    )

    # Handle position-dependent PSF separately
    if psf_is_position_dependent:
        log('Performing position-dependent PSF photometry (per-group PSF evaluation)')
        # Initialize output columns
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['x_psf'] = x_vals
        obj['y_psf'] = y_vals
        obj['qfit_psf'] = np.nan
        obj['cfit_psf'] = np.nan
        obj['flags_psf'] = 0
        obj['npix_psf'] = 0
        obj['reduced_chi2_psf'] = np.nan
        if compute_quality:
            # Quality metrics are not implemented for the per-source iterative
            # mode (no joint model image to subtract neighbours from) — keep
            # the output schema consistent with the standard path via NaNs.
            log('PSF quality metrics are not computed in position-dependent mode')
            obj['qf'] = np.nan
            obj['fracflux'] = np.nan
            obj['spread_model'] = np.nan
            obj['dspread_model'] = np.nan
        if 'flags' not in obj.keys():
            obj['flags'] = 0

        # Mark invalid positions (masked/NaN) as failed
        obj['flags'][~valid_pos] |= 0x1000

        # Get sampling (psf_model is always dict at this point)
        psf_sampling = psf_model['sampling']
        oversampling = _compute_oversampling(psf_sampling)

        # Set up local background estimator if requested (shared by all groups)
        localbkg_estimator = None
        if bkgann is not None and len(bkgann) == 2:
            localbkg_estimator = GradientLocalBackground(bkgann[0], bkgann[1], order=bkg_order)

        # Group nearby sources and fit each group jointly with a PSF
        # evaluated once at the group position; fitting sources one at a
        # time would leave neighbour flux unmodelled exactly in the crowded
        # wide-field cases where a varying PSF matters most
        valid_idx = np.where(valid_pos)[0]
        xv = np.asarray(init_params['x'], dtype=float)[valid_idx]
        yv = np.asarray(init_params['y'], dtype=float)[valid_idx]
        fv = np.asarray(init_params['flux'], dtype=float)[valid_idx]

        if grouper is not None and len(valid_idx) > 1:
            group_ids = np.asarray(grouper(xv, yv))
        else:
            group_ids = np.arange(len(valid_idx))

        for gid in np.unique(group_ids):
            in_group = group_ids == gid
            sel = valid_idx[in_group]

            try:
                # Evaluate PSF at the group mean position (the PSF varies
                # smoothly on the scale of a group)
                psf_image, psf_origin = psf_module._get_sampled_psf_stamp(
                    psf_model,
                    x=float(np.mean(xv[in_group])),
                    y=float(np.mean(yv[in_group])),
                )
                psf_image = _scale_psf_image_for_photutils(psf_image, oversampling)

                # Create photutils PSF model for this group position
                psf_at_pos = photutils.psf.ImagePSF(
                    psf_image, oversampling=oversampling, origin=psf_origin
                )

                # Set up photometry for this group
                phot_group = photutils.psf.PSFPhotometry(
                    psf_model=psf_at_pos,
                    fit_shape=fit_size,
                    finder=None,
                    grouper=grouper,
                    fitter=LevMarLSQFitter(),
                    fitter_maxiters=maxiters,
                    xy_bounds=xy_bounds,
                    aperture_radius=fit_size / 2,
                    **{_LOCALBKG_ARG: localbkg_estimator},
                )

                # Measure this group
                init_group = Table()
                init_group['x'] = xv[in_group]
                init_group['y'] = yv[in_group]
                init_group['flux'] = fv[in_group]

                result_group = phot_group(
                    image1, mask=mask_for_fit, error=err, init_params=init_group
                )

                # Extract results (result rows follow init_params order)
                for row, i in enumerate(sel):
                    obj['flux'][i] = result_group['flux_fit'][row]
                    obj['fluxerr'][i] = result_group['flux_err'][row]
                    obj['x_psf'][i] = result_group['x_fit'][row]
                    obj['y_psf'][i] = result_group['y_fit'][row]

                    # Extract quality of fit columns if available
                    if 'qfit' in result_group.colnames:
                        obj['qfit_psf'][i] = result_group['qfit'][row]
                    if 'cfit' in result_group.colnames:
                        obj['cfit_psf'][i] = result_group['cfit'][row]
                    if 'flags' in result_group.colnames:
                        obj['flags_psf'][i] = result_group['flags'][row]
                    if _NPIXFIT_COL in result_group.colnames:
                        obj['npix_psf'][i] = result_group[_NPIXFIT_COL][row]
                    if 'reduced_chi2' in result_group.colnames:
                        obj['reduced_chi2_psf'][i] = result_group['reduced_chi2'][row]

                    # Flag if fit failed
                    if not np.isfinite(obj['flux'][i]):
                        obj['flags'][i] |= 0x1000
                    # Also flag if fit didn't converge or returned input unchanged
                    elif 'flags' in result_group.colnames:
                        # Possible non-convergence reported by photutils
                        unconverged = (
                            result_group['flags'][row] & _PHOTUTILS_FLAG_NONCONVERGED
                        ) != 0

                        # Fit returned the input unchanged without reporting it
                        unchanged = (
                            obj['flux'][i] == init_group['flux'][row]
                            and obj['x_psf'][i] == init_group['x'][row]
                            and obj['y_psf'][i] == init_group['y'][row]
                        )

                        if unconverged or unchanged:
                            log(
                                'Warning: PSF fit did not converge or returned unchanged parameters for object %d, setting flux to NaN'
                                % i
                            )
                            obj['flux'][i] = np.nan
                            obj['fluxerr'][i] = np.nan
                            obj['flags'][i] |= 0x1000

                    # Flag if position moved significantly
                    if recentroid:
                        if (
                            np.sqrt(
                                (obj['x_psf'][i] - obj['x'][i]) ** 2
                                + (obj['y_psf'][i] - obj['y'][i]) ** 2
                            )
                            > 1.0
                        ):
                            obj['flags'][i] |= 0x2000

            except Exception as e:
                log('PSF photometry failed for group of %d objects: %s' % (len(sel), str(e)))
                for i in sel:
                    obj['flux'][i] = np.nan
                    obj['fluxerr'][i] = np.nan
                    obj['flags'][i] |= 0x1000

    else:
        # Standard (non-position-dependent) PSF photometry
        # Initialize output columns with NaN (for invalid positions)
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['x_psf'] = x_vals
        obj['y_psf'] = y_vals
        obj['qfit_psf'] = np.nan
        obj['cfit_psf'] = np.nan
        obj['flags_psf'] = 0
        obj['npix_psf'] = 0
        obj['reduced_chi2_psf'] = np.nan
        if compute_quality:
            obj['qf'] = np.nan
            obj['fracflux'] = np.nan
            obj['spread_model'] = np.nan
            obj['dspread_model'] = np.nan
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
                    log(
                        'Using local background annulus %.1f-%.1f pixels with %s fitting'
                        % (inner_rad, outer_rad, order_name)
                    )

                    # Create gradient-aware local background estimator
                    localbkg_estimator = GradientLocalBackground(
                        inner_rad, outer_rad, order=bkg_order
                    )

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
                    **{_LOCALBKG_ARG: localbkg_estimator},
                )

                # Do the photometry - photutils 2.x API
                result = phot_obj(
                    image1, mask=mask_for_fit, error=err, init_params=init_params_valid
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
                if _NPIXFIT_COL in result.colnames:
                    obj['npix_psf'][valid_pos] = result[_NPIXFIT_COL]
                if 'reduced_chi2' in result.colnames:
                    # Available in photutils >= 2.3.0
                    obj['reduced_chi2_psf'][valid_pos] = result['reduced_chi2']

                # Flag objects where fit failed (NaN values)
                bad_idx = valid_pos & ~np.isfinite(obj['flux'])
                obj['flags'][bad_idx] |= 0x1000  # PSF fit failed

                # Flag objects where position moved significantly (>1 pixel)
                if recentroid:
                    moved_idx = valid_pos & (
                        np.sqrt(
                            (obj['x_psf'] - init_params['x']) ** 2
                            + (obj['y_psf'] - init_params['y']) ** 2
                        )
                        > 1.0
                    )
                    obj['flags'][moved_idx] |= 0x2000  # Large centroid shift

                if compute_quality:
                    quality = _compute_psf_quality_columns(
                        phot_obj,
                        psf_model,
                        image1,
                        err,
                        mask_for_fit,
                        np.asarray(obj['x_psf'])[valid_pos],
                        np.asarray(obj['y_psf'])[valid_pos],
                        np.asarray(obj['flux'])[valid_pos],
                        fit_size,
                        fwhm,
                        log,
                    )
                    if quality is not None:
                        obj['qf'][valid_pos] = quality['qf']
                        obj['fracflux'][valid_pos] = quality['fracflux']
                        obj['spread_model'][valid_pos] = quality['spread_model']
                        obj['dspread_model'][valid_pos] = quality['dspread_model']

                # Flag fits that didn't converge, or returned the initial guess unchanged
                if 'flags_psf' in obj.colnames and 'flux' in init_params.colnames:
                    unconverged = valid_pos & (
                        (obj['flags_psf'] & _PHOTUTILS_FLAG_NONCONVERGED) != 0
                    )

                    # photutils may return input unchanged without reporting it
                    unchanged = (
                        valid_pos
                        & (obj['flux'] == init_params['flux'])
                        & (obj['x_psf'] == init_params['x'])
                        & (obj['y_psf'] == init_params['y'])
                    )

                    # Combine both conditions
                    failed = unconverged | unchanged

                    if np.sum(failed) > 0:
                        log(
                            'Warning: %d PSF fits failed or returned unchanged parameters, setting flux to NaN'
                            % np.sum(failed)
                        )
                        obj['flux'][failed] = np.nan
                        obj['fluxerr'][failed] = np.nan
                        obj['flags'][failed] |= 0x1000  # PSF fit failed

            except Exception as e:
                log('PSF photometry failed: %s' % str(e))
                log('Falling back to NaN values')
                obj['flags'][valid_pos] |= 0x1000

    # Fitted positions replace the input ones (kept in x_orig/y_orig), except
    # for failed fits
    if recentroid:
        fitted = np.isfinite(np.asarray(obj['flux'], dtype=float))
        _store_fitted_positions(
            obj,
            np.where(fitted, obj['x_psf'], np.nan),
            np.where(fitted, obj['y_psf'], np.nan),
        )
    obj.remove_columns(['x_psf', 'y_psf'])

    obj = _compute_magnitudes_and_filter(obj, sn, keep_negative, log)

    log('PSF photometry complete: %d objects measured' % len(obj))

    if get_bg:
        # Return the background that was actually subtracted
        return obj, (bg if bg is not None else bg_est_bg), err
    else:
        return obj
