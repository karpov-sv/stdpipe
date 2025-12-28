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


def measure_objects(
    obj,
    image,
    aper=3,
    bkgann=None,
    fwhm=None,
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
    """Aperture photometry at the positions of already detected objects.

    It will estimate and subtract the background unless external background estimation (`bg`) is provided, and use user-provided noise map (`err`) if requested.

    If the `mask` is provided, it will set 0x200 bit in object `flags` if at least one of aperture pixels is masked.

    The results may optionally filtered to drop the detections with low signal to noise ratio if `sn` parameter is set and positive. It will also filter out the events with negative flux.


    :param obj: astropy.table.Table with initial object detections to be measured
    :param image: Input image as a NumPy array
    :param aper: Circular aperture radius in pixels, to be used for flux measurement
    :param bkgann: Background annulus (tuple with inner and outer radii) to be used for local background estimation. If not set, global background model is used instead.
    :param fwhm: If provided, `aper` and `bkgann` will be measured in units of this value (so they will be specified in units of FWHM)
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

    # FIXME: is there any better way to exclude some positions from photometry?..
    positions = [(_['x'], _['y']) if np.isfinite(_['x']) and np.isfinite(_['y']) else (-1000, -1000) for _ in obj]
    apertures = photutils.aperture.CircularAperture(positions, r=aper)
    # Use just a minimal mask here so that the flux from 'soft-masked' (e.g. saturated) pixels is still counted
    res = photutils.aperture.aperture_photometry(image1, apertures, error=err, mask=mask0)

    obj['flux'] = res['aperture_sum']
    obj['fluxerr'] = res['aperture_sum_err']

    if 'flags' not in obj.keys():
        obj['flags'] = 0

    # Check whether some aperture pixels are masked, and set the flags for that
    mres = photutils.aperture.aperture_photometry(mask | mask0, apertures, method='center')
    obj['flags'][mres['aperture_sum'] > 0] |= 0x200

    # Position-dependent background flux error from global background model, if available
    obj['bg_fluxerr'] = 0.0  # Local background flux error inside the aperture
    if bg_est is not None:
        res = photutils.aperture.aperture_photometry(bg_est_rms**2, apertures)
        obj['bg_fluxerr'] = np.sqrt(res['aperture_sum'])

    # Local background
    if bkgann is not None and len(bkgann) == 2:
        if fwhm is not None and fwhm > 0:
            bkgann = [_ * fwhm for _ in bkgann]
        log(
            'Using local background annulus between %.1f and %.1f pixels' % (
                bkgann[0], bkgann[1]
            )
        )

        # Aperture areas
        image_ones = np.ones_like(image1)
        res_area = photutils.aperture.aperture_photometry(image_ones, apertures, mask=mask0)

        # Local background
        lbg = photutils.background.LocalBackground(
            bkgann[0], bkgann[1],
            bkg_estimator=photutils.background.ModeEstimatorBackground(),
        )

        # Dedicated column for local background on top of global estimation
        obj['bg_local'] = lbg(image1, obj['x'], obj['y'], mask=mask)

        # Sanitize and flag the values where local bg estimation failed
        idx = ~np.isfinite(obj['bg_local'])
        obj['bg_local'][idx] = 0
        obj['flags'][idx] |= 0x400

        obj['flux'] -= obj['bg_local'] * res_area['aperture_sum']

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

    if get_bg:
        return obj, bg_est_bg, err
    else:
        return obj
