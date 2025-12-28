"""
Aperture photometry measurement routines.

This module contains functions for performing aperture photometry
on detected objects.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import photutils
import photutils.background
import photutils.aperture
import photutils.centroids
from photutils.utils import calc_total_error

from stdpipe import psf as psf_module


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
    # Get PSF stamp at object position
    if isinstance(psf, dict):
        # PSFEx or ePSF model
        psf_stamp = psf_module.get_psf_stamp(psf, x=x, y=y, normalize=True)
    else:
        # Gaussian PSF - create stamp from FWHM
        fwhm = float(psf)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = int(np.ceil(fwhm * 3)) * 2 + 1
        yy, xx = np.mgrid[0:size, 0:size]
        psf_stamp = np.exp(-((xx - size//2)**2 + (yy - size//2)**2) / (2 * sigma**2))
        psf_stamp /= np.sum(psf_stamp)

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
    fwhm=None,
    psf=None,
    optimal=False,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    sn=None,
    centroid_iter=0,
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
    :param fwhm: If provided, `aper` and `bkgann` will be measured in units of this value (so they will be specified in units of FWHM). Also used to define Gaussian PSF for optimal extraction if `psf` is not provided.
    :param psf: PSF model for optimal extraction. Can be a dict from psf.run_psfex(), psf.load_psf(), or psf.create_psf_model(). If None, a Gaussian PSF will be created from the `fwhm` parameter.
    :param optimal: If True, use optimal extraction instead of aperture photometry. Requires either `psf` or `fwhm` to define the PSF profile.
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array with same shape as input image) instead of automatically computed one
    :param err: Image noise map as a NumPy array to be used instead of automatically computed one, optional
    :param gain: Image gain, e/ADU, used to build image noise model
    :param bg_size: Background grid size in pixels
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param centroid_iter: Number of centroiding iterations to run before photometry. If non-zero, will try to improve the aperture placement by finding the centroid of pixels inside the aperture.
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
        mask = mask0
    else:
        mask = mask.astype(bool)

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

    if centroid_iter:
        box_size = int(np.ceil(aper))
        if box_size % 2 == 0:
            box_size += 1
        log('Using centroiding routine with %d iterations within %dx%d box' % (centroid_iter, box_size, box_size))
        # Keep original pixel positions
        obj['x_orig'] = obj['x']
        obj['y_orig'] = obj['y']

        for iter in range(centroid_iter):
            obj['x'],obj['y'] = photutils.centroids.centroid_sources(
                image1,
                obj['x'],
                obj['y'],
                box_size=box_size,
                mask=mask
            )

    if 'flags' not in obj.keys():
        obj['flags'] = 0

    # Dedicated column for local background on top of global estimation
    obj['bg_local'] = 0

    # Local background - it has to be computed prior to optimal photometry
    # as otherwise it is too difficult to subtract it from its (weighted) results
    if bkgann is not None and len(bkgann) == 2:
        if fwhm is not None and fwhm > 0:
            bkgann = [_ * fwhm for _ in bkgann]
        log(
            'Using local background annulus between %.1f and %.1f pixels' % (
                bkgann[0], bkgann[1]
            )
        )

        # Local background
        lbg = photutils.background.LocalBackground(
            bkgann[0], bkgann[1],
            bkg_estimator=photutils.background.ModeEstimatorBackground(),
        )

        obj['bg_local'] = lbg(image1, obj['x'], obj['y'], mask=mask)

        # Sanitize and flag the values where local bg estimation failed
        idx = ~np.isfinite(obj['bg_local'])
        obj['bg_local'][idx] = 0
        obj['flags'][idx] |= 0x400

    # Photometric apertures
    # FIXME: is there any better way to exclude some positions from photometry?..
    positions = [(_['x'], _['y']) if np.isfinite(_['x']) and np.isfinite(_['y']) else (-1000, -1000) for _ in obj]
    apertures = photutils.aperture.CircularAperture(positions, r=aper)

    # Check whether some aperture pixels are masked, and set the flags for that
    mres = photutils.aperture.aperture_photometry(mask | mask0, apertures, method='center')
    obj['flags'][mres['aperture_sum'] > 0] |= 0x200

    # Aperture unmasked areas, in (fractional) pixels
    image_ones = np.ones(image1.shape)
    res_area = photutils.aperture.aperture_photometry(image_ones, apertures, mask=mask0)
    obj['npix_aper'] = res_area['aperture_sum']

    # Position-dependent background flux error from global background model, if available
    obj['bg_fluxerr'] = 0.0  # Local background flux error inside the aperture
    if bg_est is not None:
        res = photutils.aperture.aperture_photometry(bg_est_rms**2, apertures)
        obj['bg_fluxerr'] = np.sqrt(res['aperture_sum'])

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

        # Perform optimal extraction for each object
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['npix_optimal'] = 0
        obj['chi2_optimal'] = np.nan
        obj['norm_optimal'] = np.nan

        for i, o in enumerate(obj):
            if np.isfinite(o['x']) and np.isfinite(o['y']):
                res = _optimal_extraction(
                    image1, err,
                    o['x'], o['y'],
                    psf_for_extraction,
                    bg_local=o['bg_local'],
                    mask=mask, # Do not count the flux from 'soft-masked' pixels
                    radius=aper  # Use aperture radius as clipping radius
                )

                o['flux'], o['fluxerr'], o['npix_optimal'], o['norm_optimal'], o['chi2_optimal'] = res

        # Flag objects where optimal extraction failed
        obj['flags'][~np.isfinite(obj['flux'])] |= 0x800

    else:
        # Standard aperture photometry
        # Use just a minimal mask here so that the flux from 'soft-masked' (e.g. saturated) pixels is still counted
        res = photutils.aperture.aperture_photometry(image1, apertures, error=err, mask=mask0)

        obj['flux'] = res['aperture_sum']
        obj['fluxerr'] = res['aperture_sum_err']

        # Subtract local background
        obj['flux'] -= obj['bg_local'] * obj['npix_aper']

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
