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
    sn=None,
    fit_shape='circular',
    fit_size=None,
    maxiters=3,
    recentroid=True,
    keep_negative=True,
    get_bg=False,
    use_position_dependent_psf=False,
    group_sources=False,
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
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param fit_shape: Shape of fitting region. Options: 'circular' (default), 'square'. Determines the aperture used for PSF fitting.
    :param fit_size: Size of fitting region in pixels. If None, defaults to psf_size.
    :param maxiters: Maximum number of iterations for PSF fitting
    :param recentroid: If True, allow PSF position to vary during fitting (recommended)
    :param keep_negative: If not set, measurements with negative fluxes will be discarded
    :param get_bg: If True, the routine will also return estimated background and background noise images
    :param use_position_dependent_psf: If True and PSF is a PSFEx model, use polynomial evaluation for position-dependent PSF (evaluates PSF at each source position)
    :param group_sources: If True, use grouped PSF fitting for overlapping sources (slower but more accurate in crowded fields)
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
            psf_size = int(np.ceil(5 * fwhm))
            if psf_size % 2 == 0:
                psf_size += 1  # Make odd

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
                psf_size = psf['height']
        else:
            log('Using PSFEx/ePSF PSF model (constant across field)')
            # Get PSF stamp at center position (0,0 works for degree=0)
            psf_image = psf_module.get_supersampled_psf_stamp(psf, x=0, y=0, normalize=True)

            # Handle oversampling if needed
            oversampling = int(1 / psf_sampling) if psf_sampling < 1.0 else 1
            psf_model = photutils.psf.ImagePSF(psf_image, oversampling=oversampling)
            psf_is_position_dependent = False

            if psf_size is None:
                psf_size = psf_image.shape[0]

    elif isinstance(psf, (photutils.psf.ImagePSF, photutils.psf.FittableImageModel)):
        # Already a photutils PSF model (ImagePSF or legacy FittableImageModel)
        log('Using provided photutils ImagePSF model')
        psf_model = psf
        if psf_size is None:
            psf_size = psf.data.shape[0]

    elif hasattr(psf, 'fwhm'):
        # Photutils ePSF or similar
        log('Using provided photutils PSF model with FWHM')
        psf_model = psf
        if psf_size is None:
            psf_size = psf.data.shape[0] if hasattr(psf, 'data') else int(np.ceil(5 * psf.fwhm))

    else:
        # Assume it's a photutils PSF model
        log('Using provided PSF model')
        psf_model = psf
        if psf_size is None:
            psf_size = int(np.ceil(5 * fwhm))
            if psf_size % 2 == 0:
                psf_size += 1

    log('Using PSF size: %d pixels' % psf_size)

    # Fitting region size
    if fit_size is None:
        fit_size = psf_size
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

    # Create apertures for fitting regions using safe positions
    # Replace NaN with dummy positions (will be filtered out later)
    x_safe = np.where(valid_pos, init_params['x'], 0.0)
    y_safe = np.where(valid_pos, init_params['y'], 0.0)

    if fit_shape == 'circular':
        fit_apertures = photutils.aperture.CircularAperture(
            list(zip(x_safe, y_safe)),
            r=fit_size / 2
        )
    else:  # 'square'
        fit_apertures = photutils.aperture.RectangularAperture(
            list(zip(x_safe, y_safe)),
            w=fit_size,
            h=fit_size,
            theta=0
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
        oversampling = int(1 / psf_sampling) if psf_sampling < 1.0 else 1

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

                # Create photutils PSF model for this position
                psf_at_pos = photutils.psf.ImagePSF(psf_image, oversampling=oversampling)

                # Set up photometry for this object
                phot_single = photutils.psf.PSFPhotometry(
                    psf_model=psf_at_pos,
                    fit_shape=fit_size,
                    finder=None,
                    grouper=grouper,
                    fitter=LevMarLSQFitter(),
                    aperture_radius=fit_size / 2
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
                    mask=mask,
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

                # Set up photometry object
                phot_obj = photutils.psf.PSFPhotometry(
                    psf_model=psf_model,
                    fit_shape=fit_size,
                    finder=None,  # We already have positions
                    grouper=grouper,  # Group nearby sources if requested
                    fitter=LevMarLSQFitter(),  # Levenberg-Marquardt fitter from astropy
                    aperture_radius=fit_size / 2
                )

                # Do the photometry - photutils 2.x API
                result = phot_obj(
                    image1,
                    mask=mask,
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
