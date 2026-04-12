"""
Module containing higher-level pipeline building blocks, wrapping together lower-level
functionality of STDPipe modules.
"""

import os
import numpy as np
import itertools
from copy import deepcopy

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

import sep
import astroscrappy

from . import photometry
from . import astrometry
from . import catalogs
from . import psf
from . import utils
from . import cutouts


def make_mask(
    image,
    header=None,
    saturation=None,
    external_mask=None,
    mask_cosmics=False,
    gain=None,
    verbose=True,
):
    """Make a basic mask for an image.

    The mask is a boolean bitmap with ``True`` values marking regions to be
    excluded from further processing.  The following regions are masked:

    - pixels with undefined or non-finite (inf or nan) values
    - regions outside the usable area defined by the ``DATASEC`` or ``TRIMSEC``
      header keyword
    - pixels above the saturation limit, if provided
    - pixels set in the external mask, if provided
    - cosmic rays, if requested

    If ``saturation=True``, the saturation level is estimated from the image as
    ``median + 0.95 * (max - median)``.

    Parameters
    ----------
    image : ndarray
        2D image array to mask.
    header : astropy.io.fits.Header, optional
        FITS header; used to read ``DATASEC`` / ``TRIMSEC`` keywords.
    saturation : float or bool, optional
        Saturation level in ADU.  If ``True``, estimated automatically from
        the image.
    external_mask : ndarray, optional
        Boolean mask to OR with the created mask.
    mask_cosmics : bool, optional
        If True, detect and mask cosmic rays using the LACosmic algorithm.
    gain : float, optional
        Detector gain in e-/ADU, used for cosmic-ray masking.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    ndarray of bool
        Boolean mask with ``True`` marking excluded pixels.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    mask = ~np.isfinite(image)

    if header is not None:
        # DATASEC or TRIMSEC
        for kw in ['TRIMSEC', 'DATASEC']:
            if kw in header:
                x1, x2, y1, y2 = utils.parse_det(header.get(kw))

                log('Masking the region outside of %s = %s' % (kw, header.get(kw)))
                mask1 = np.ones_like(mask)
                mask1[y1 : y2 + 1, x1 : x2 + 1] = False
                mask |= mask1

    if saturation is not None:
        if type(saturation) is bool and saturation == True:
            saturation = 0.05 * np.nanmedian(image) + 0.95 * np.nanmax(image)  # med + 0.95(max-med)
        log('Masking pixels above saturation level %.1f ADU' % saturation)
        mask |= image >= saturation

    if external_mask is not None:
        external_mask = external_mask.astype(bool)
        log('Applying external mask with %d masked pixels' % np.sum(external_mask))
        mask |= external_mask

    if mask_cosmics:
        # We will use custom noise model for astroscrappy as we do not know whether
        # the image is background-subtracted already, or how it was flatfielded
        bg = sep.Background(image, mask=mask)
        rms = bg.rms()
        var = rms**2
        if gain:
            var += np.abs(image - bg.back()) / gain
        cmask, cimage = astroscrappy.detect_cosmics(
            image,
            mask,
            verbose=verbose,
            invar=var.astype(np.float32),
            gain=gain if gain else 1.0,
            satlevel=saturation if saturation else np.max(image),
            cleantype='medmask',
        )
        log(
            'Done masking cosmics, %d (%.1f%%) pixels masked'
            % (np.sum(cmask), 100 * np.sum(cmask) / cmask.shape[0] / cmask.shape[1])
        )
        mask |= cmask

    log(
        '%d (%.1f%%) pixels masked in total'
        % (np.sum(mask), 100 * np.sum(mask) / mask.shape[0] / mask.shape[1])
    )

    return mask


def refine_astrometry(
    obj,
    cat,
    sr=10 / 3600,
    wcs=None,
    order=0,
    cat_col_mag='V',
    cat_col_mag_err=None,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    cat_col_ra_err='e_RAJ2000',
    cat_col_dec_err='e_DEJ2000',
    n_iter=3,
    use_photometry=True,
    min_matches=5,
    method='quadhash',
    update=True,
    verbose=False,
    **kwargs,
):
    """Higher-level astrometric refinement routine.

    Dispatches to quad-hash, SCAMP, astropy, or astrometrynet fitting
    depending on ``method``.

    Parameters
    ----------
    obj : astropy.table.Table
        List of objects on the frame, must contain at least ``x``, ``y``, and ``flux`` columns.
    cat : astropy.table.Table or str
        Reference astrometric catalogue.
    sr : float, optional
        Matching radius in degrees.
    wcs : astropy.wcs.WCS, optional
        Initial WCS solution.
    order : int, optional
        Polynomial order for SIP or PV distortion solution. Default 0 for TAN;
        2 is recommended for ``quadhash``.
    cat_col_mag : str, optional
        Catalogue column name for magnitude.
    cat_col_mag_err : str, optional
        Catalogue column name for magnitude error.
    cat_col_ra : str, optional
        Catalogue column name for Right Ascension.
    cat_col_dec : str, optional
        Catalogue column name for Declination.
    cat_col_ra_err : str, optional
        Catalogue column name for Right Ascension error.
    cat_col_dec_err : str, optional
        Catalogue column name for Declination error.
    n_iter : int, optional
        Number of iterations for Python-based matching.
    use_photometry : bool, optional
        Use photometry-assisted matching in Python-based methods.
    min_matches : int, optional
        Minimum number of good matches required.
    method : str, optional
        Fitting method. One of ``'quadhash'`` (default, 2–7× more accurate),
        ``'scamp'``, ``'astropy'``, or ``'astrometrynet'``.
    update : bool, optional
        If True, update ``ra`` and ``dec`` columns in ``obj`` in-place.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    **kwargs
        All other keyword arguments are passed to the respective refinement function.

    Returns
    -------
    astropy.wcs.WCS or None
        Refined astrometric solution, or None on failure.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    log(
        'Astrometric refinement using %.1f arcsec radius, %s matching and %s WCS fitting'
        % (sr * 3600, 'photometric' if use_photometry else 'simple positional', method)
    )
    if type(cat) == str:
        log('Using %d objects and catalogue %s' % (len(obj), cat))
    else:
        log('Using %d objects and %d catalogue stars' % (len(obj), len(cat)))

    if wcs is not None:
        obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    if method == 'quadhash':
        # Quad-hash pattern matching (default, hopefully more accurate than SCAMP)

        return astrometry.refine_wcs_quadhash(
            obj,
            cat,
            wcs=wcs,
            sr=sr,
            order=order,
            cat_col_ra=cat_col_ra,
            cat_col_dec=cat_col_dec,
            cat_col_mag=cat_col_mag,
            update=update,
            verbose=verbose,
            **kwargs,
        )

    elif method == 'scamp':
        # Fall-through to SCAMP-specific variant
        return astrometry.refine_wcs_scamp(
            obj,
            cat,
            sr=sr,
            wcs=wcs,
            order=order,
            cat_col_mag=cat_col_mag,
            cat_col_mag_err=cat_col_mag_err,
            cat_col_ra=cat_col_ra,
            cat_col_dec=cat_col_dec,
            cat_col_ra_err=cat_col_ra_err,
            cat_col_dec_err=cat_col_dec_err,
            update=update,
            verbose=verbose,
            **kwargs,
        )

    for iter in range(n_iter):
        if use_photometry:
            # Matching involving photometric information
            cat_magerr = cat[cat_col_mag_err] if cat_col_mag_err is not None else None
            m = photometry.match(
                obj['ra'],
                obj['dec'],
                obj['mag'],
                obj['magerr'],
                obj['flags'],
                cat[cat_col_ra],
                cat[cat_col_dec],
                cat[cat_col_mag],
                cat_magerr=cat_magerr,
                sr=sr,
                verbose=verbose,
            )
            if m is None or not m:
                log('Photometric match failed, cannot refine WCS')
                return None
            elif np.sum(m['idx']) < min_matches:
                log('Too few (%d) good photometric matches, cannot refine WCS' % np.sum(m['idx']))
                return None
            else:
                log(
                    'Iteration %d: %d matches, %.1f arcsec rms'
                    % (iter, np.sum(m['idx']), np.std(3600 * m['dist'][m['idx']]))
                )

            wcs = astrometry.refine_wcs_simple(
                obj[m['oidx']][m['idx']],
                cat[m['cidx']][m['idx']],
                order=order,
                match=False,
                method=method,
                verbose=verbose,
            )
        else:
            # Simple positional matching
            wcs = astrometry.refine_wcs_simple(
                obj, cat, order=order, sr=sr, match=True, method=method, verbose=verbose, **kwargs
            )

        if update:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    return wcs


def filter_transient_candidates(
    obj,
    sr=None,
    pixscale=None,
    fwhm=None,
    time=None,
    obj_col_ra='ra',
    obj_col_dec='dec',
    cat=None,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    vizier=[],
    skybot=True,
    ned=False,
    flagged=True,
    flagmask=0x7F00,
    col_id=None,
    vizier_checker_fn=None,
    get_candidates=True,
    remove=True,
    verbose=False,
):
    """Higher-level transient candidate filtering routine.

    Optionally filters out the following classes of objects:

    - flagged ones (``obj['flags'] != 0``)
    - positionally coincident with stars from a provided reference catalogue
    - positionally coincident with stars from Vizier catalogues
    - positionally and temporally coincident with Solar System objects from SkyBoT
    - positionally and temporally coincident with NED objects

    The original object list is never modified; a filtered or annotated copy is returned.

    If ``get_candidates=False``, returns only a boolean mask of objects surviving all filters.
    If ``get_candidates=True`` and ``remove=False``, all objects are returned but annotated
    with ``candidate_*`` columns indicating which filters matched each object, plus a
    ``candidate_good`` column that is ``True`` for objects surviving all filters.

    Parameters
    ----------
    obj : astropy.table.Table
        Input object list.
    sr : float, optional
        Matching radius in degrees. Defaults to half FWHM (if ``pixscale`` is set) or 1 arcsec.
    pixscale : float, optional
        Pixel scale in degrees/pixel. Used to compute the default matching radius.
    fwhm : float, optional
        FWHM in pixels. Estimated from unflagged objects if not provided.
    time : astropy.time.Time or datetime.datetime, optional
        Observation time; required for SkyBoT cross-matching.
    obj_col_ra : str, optional
        Column name for object Right Ascension.
    obj_col_dec : str, optional
        Column name for object Declination.
    cat : astropy.table.Table, optional
        Reference catalogue for spatial cross-matching.
    cat_col_ra : str, optional
        Column name for catalogue Right Ascension.
    cat_col_dec : str, optional
        Column name for catalogue Declination.
    vizier : list of str, optional
        Vizier catalogue identifiers (or short names) to cross-match against.
    skybot : bool, optional
        If True, cross-match with SkyBoT Solar System object positions.
    ned : bool, optional
        If True, cross-match with NED database entries.
    flagged : bool, optional
        If True, filter out objects where ``(obj['flags'] & flagmask) != 0``.
    flagmask : int, optional
        Bitmask applied to ``obj['flags']`` for flag filtering.
    col_id : str, optional
        Column name for a unique object identifier. A ``stdpipe_id`` column is
        created automatically if not specified.
    vizier_checker_fn : callable, optional
        Function ``fn(obj, xcat, catname) -> bool array`` to apply additional
        conditions on Vizier cross-matches before they are considered true matches.
    get_candidates : bool, optional
        If True (default), return the filtered/annotated object list.
        If False, return only a boolean mask over the original list.
    remove : bool, optional
        If True (default), remove filtered entries from the returned list.
        If False, keep all objects and add ``candidate_*`` annotation columns.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.table.Table or ndarray of bool
        Filtered/annotated copy of the object list, or (if ``get_candidates=False``)
        a boolean mask of the same length as the input.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if fwhm is None:
        idx = obj['flags'] == 0
        idx &= obj['magerr'] < 0.05  # S/N > 20
        fwhm = np.median(obj['fwhm'][idx])

    if sr is None:
        if pixscale is not None:
            # Matching radius of half FWHM
            sr = np.median(fwhm * pixscale) / 2
        else:
            # Fallback value of 1 arcsec, should be sensible for most catalogues
            sr = 1 / 3600

    if col_id is None:
        col_id = 'stdpipe_id'

    if col_id not in obj.keys():
        obj_in = obj
        obj = obj.copy()
        obj[col_id] = np.arange(len(obj))
    else:
        obj_in = obj

    if remove == False:
        # We are asked to just mark the matched candidates, not remove from the result
        # So let's create a copy of objects list where we may freely add extra columns without touching the original
        obj_in = obj_in.copy()

    log(
        'Candidate filtering routine started with %d initial candidates and %.1f arcsec matching radius'
        % (len(obj), sr * 3600)
    )
    cand_idx = np.ones(len(obj), dtype=bool)

    # Object flags
    if flagged:
        # Filter out flagged objects (saturated, cosmics, blends, etc)
        idx = (obj['flags'] & flagmask) == 0

        if remove == False:
            obj_in['candidate_flagged'] = ~idx

        cand_idx &= idx
        log(np.sum(cand_idx), 'of them are unflagged')

    # Reference catalogue
    if cat is not None and remove == False:
        obj_in['candidate_refcat'] = False
    if cat is not None and np.any(cand_idx):
        m = astrometry.spherical_match(
            obj[obj_col_ra], obj[obj_col_dec], cat[cat_col_ra], cat[cat_col_dec], sr
        )
        cand_idx[m[0]] = False

        if remove == False:
            obj_in['candidate_refcat'] = False
            obj_in['candidate_refcat'][m[0]] = True

        log(np.sum(cand_idx), 'of them are not matched with reference catalogue')

    # Vizier catalogues
    for catname in vizier or []:
        if remove == False:
            if 'candidate_vizier_' + catname not in obj_in.keys():
                obj_in['candidate_vizier_' + catname] = False

        if not np.any(cand_idx):
            break

        xcat = catalogs.xmatch_objects(
            obj[cand_idx][[col_id, obj_col_ra, obj_col_dec]],
            catname,
            sr,
            col_ra=obj_col_ra,
            col_dec=obj_col_dec,
        )
        if xcat is not None and len(xcat):
            if callable(vizier_checker_fn):
                # Pass matched results through user-supplied checker
                xobj = obj[[np.where(obj[col_id] == _)[0][0] for _ in xcat[col_id]]]
                xidx = vizier_checker_fn(xobj, xcat, catname)
                xcat = xcat[xidx]

            cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

            if remove == False:
                obj_in['candidate_vizier_' + catname][np.in1d(obj[col_id], xcat[col_id])] = True

        log(
            np.sum(cand_idx),
            'remains after matching with',
            catalogs.catalogs.get(catname, {'name': catname})['name'],
        )

    # SkyBoT
    if skybot and np.any(cand_idx):
        if time is None and 'time' in obj.keys():
            time = obj['time']

        if remove == False:
            if 'candidate_skybot' not in obj_in.keys():
                obj_in['candidate_skybot'] = False

        if time is not None:
            xcat = catalogs.xmatch_skybot(
                obj[cand_idx], time=time, col_ra=obj_col_ra, col_dec=obj_col_dec, col_id=col_id
            )
            if xcat is not None and len(xcat):
                cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

                if remove == False:
                    obj_in['candidate_skybot'][np.in1d(obj[col_id], xcat[col_id])] = True

            log(np.sum(cand_idx), 'remains after matching with SkyBot')

    # NED
    if ned and np.any(cand_idx):
        if remove == False:
            if 'candidate_ned' not in obj_in.keys():
                obj_in['candidate_ned'] = False

        xcat = catalogs.xmatch_ned(
            obj[cand_idx],
            sr,
            col_id=col_id,
            col_ra=obj_col_ra,
            col_dec=obj_col_dec,
        )
        if xcat is not None and len(xcat):
            cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

            if remove == False:
                obj_in['candidate_ned'][np.in1d(obj[col_id], xcat[col_id])] = True

        log(np.sum(cand_idx), 'remains after matching with NED')

    if remove == False:
        obj_in['candidate_good'] = False
        obj_in['candidate_good'][cand_idx] = True

    log('%d candidates remaining after filtering' % len(obj[cand_idx]))

    if get_candidates:
        if remove:
            # Return filtered list
            return obj_in[cand_idx].copy()
        else:
            # Return full list with additional columns
            return obj_in
    else:
        # Return just indices
        return cand_idx


def calibrate_photometry(
    obj,
    cat,
    sr=None,
    pixscale=None,
    order=0,
    bg_order=None,
    obj_col_mag='mag',
    obj_col_mag_err='magerr',
    obj_col_ra='ra',
    obj_col_dec='dec',
    obj_col_flags='flags',
    obj_col_x='x',
    obj_col_y='y',
    cat_col_mag='R',
    cat_col_mag_err=None,
    cat_col_mag1=None,
    cat_col_mag2=None,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    update=True,
    verbose=False,
    **kwargs,
):
    """Higher-level photometric calibration routine.

    Wraps :func:`stdpipe.photometry.match` with convenient defaults for typical
    tabular data.

    Parameters
    ----------
    obj : astropy.table.Table
        Table of detected objects.
    cat : astropy.table.Table
        Reference photometric catalogue.
    sr : float, optional
        Matching radius in degrees. Defaults to half the median FWHM in sky
        units if ``pixscale`` is provided, otherwise 1 arcsec.
    pixscale : float, optional
        Pixel scale in degrees/pixel; used to compute the default matching radius.
    order : int, optional
        Spatial polynomial order for the zero-point model (0 = constant).
    bg_order : int or None, optional
        Spatial polynomial order for an additive flux background term.
        None disables this term.
    obj_col_mag : str, optional
        Column name for object instrumental magnitude.
    obj_col_mag_err : str, optional
        Column name for object magnitude error.
    obj_col_ra : str, optional
        Column name for object Right Ascension.
    obj_col_dec : str, optional
        Column name for object Declination.
    obj_col_flags : str, optional
        Column name for object flags.
    obj_col_x : str, optional
        Column name for object x coordinate.
    obj_col_y : str, optional
        Column name for object y coordinate.
    cat_col_mag : str, optional
        Column name for catalogue magnitude.
    cat_col_mag_err : str, optional
        Column name for catalogue magnitude error.
    cat_col_mag1 : str, optional
        First catalogue magnitude column for the color term (e.g. ``'B'``).
    cat_col_mag2 : str, optional
        Second catalogue magnitude column for the color term (e.g. ``'R'``).
    cat_col_ra : str, optional
        Column name for catalogue Right Ascension.
    cat_col_dec : str, optional
        Column name for catalogue Declination.
    update : bool, optional
        If True, add ``mag_calib`` and ``mag_calib_err`` columns to ``obj``
        in-place with the calibrated magnitude and its error.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    **kwargs
        Additional keyword arguments passed to :func:`stdpipe.photometry.match`.

    Returns
    -------
    dict or None
        Dictionary with photometric results as returned by
        :func:`stdpipe.photometry.match`, or None on failure.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if sr is None:
        if pixscale is not None:
            # Matching radius of half FWHM
            sr = np.median(obj['fwhm'] * pixscale) / 2
        else:
            # Fallback value of 1 arcsec, should be sensible for most catalogues
            sr = 1 / 3600

    log(
        'Performing photometric calibration of %d objects vs %d catalogue stars'
        % (len(obj), len(cat))
    )
    log(
        'Using %.1f arcsec matching radius, %s magnitude and spatial order %d'
        % (sr * 3600, cat_col_mag, order)
    )
    if cat_col_mag1 and cat_col_mag2:
        log('Using (%s - %s) color for color term' % (cat_col_mag1, cat_col_mag2))

    if cat_col_mag1 and cat_col_mag2:
        color = cat[cat_col_mag1] - cat[cat_col_mag2]
    else:
        color = None

    if cat_col_mag_err:
        cat_magerr = cat[cat_col_mag_err]
    else:
        cat_magerr = None

    m = photometry.match(
        obj[obj_col_ra],
        obj[obj_col_dec],
        obj[obj_col_mag],
        obj[obj_col_mag_err],
        obj[obj_col_flags],
        cat[cat_col_ra],
        cat[cat_col_dec],
        cat[cat_col_mag],
        cat_magerr=cat_magerr,
        sr=sr,
        cat_color=color,
        obj_x=obj[obj_col_x] if obj_col_x else None,
        obj_y=obj[obj_col_y] if obj_col_y else None,
        spatial_order=order,
        bg_order=bg_order,
        verbose=verbose,
        **kwargs,
    )

    if m:
        log('Photometric calibration finished successfully.')
        # if m['color_term']:
        #     log('Color term is %.2f' % m['color_term'])

        m['cat_col_mag'] = cat_col_mag
        if cat_col_mag1 and cat_col_mag2:
            m['cat_col_mag1'] = cat_col_mag1
            m['cat_col_mag2'] = cat_col_mag2

        if update:
            obj['mag_calib'] = obj[obj_col_mag] + m['zero_fn'](
                obj[obj_col_x], obj[obj_col_y], obj[obj_col_mag]
            )
            obj['mag_calib_err'] = np.hypot(
                obj[obj_col_mag_err],
                m['zero_fn'](obj[obj_col_x], obj[obj_col_y], obj[obj_col_mag], get_err=True),
            )
    else:
        log('Photometric calibration failed')

    return m


def make_random_stars(
    width=None,
    height=None,
    shape=None,
    nstars=100,
    minflux=1,
    maxflux=100000,
    edge=0,
    wcs=None,
    verbose=False,
):
    """Generate a table of random stars.

    Coordinates are distributed uniformly with
    ``edge <= x < width - edge`` and ``edge <= y < height - edge``.
    Fluxes are log-uniform between ``minflux`` and ``maxflux``.

    Parameters
    ----------
    width : int, optional
        Width of the image in pixels.
    height : int, optional
        Height of the image in pixels.
    shape : tuple of int, optional
        Image shape ``(height, width)``; used instead of ``width`` and ``height`` if set.
    nstars : int, optional
        Number of artificial stars to generate.
    minflux : float, optional
        Minimum star flux in ADU.
    maxflux : float, optional
        Maximum star flux in ADU.
    edge : int, optional
        Minimum distance to image edges for star placement.
    wcs : astropy.wcs.WCS, optional
        If provided, sky coordinates ``ra`` and ``dec`` are added to the catalogue.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.table.Table
        Catalogue with columns ``x``, ``y``, ``flux``, ``mag``, and optionally
        ``ra``, ``dec`` if ``wcs`` is provided.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if (width is None or height is None) and shape is not None:
        height, width = shape

    cat = {
        'x': np.random.uniform(edge, width - 1 - edge, nstars),
        'y': np.random.uniform(edge, height - edge, nstars),
        'flux': 10 ** np.random.uniform(np.log10(minflux), np.log10(maxflux), nstars),
    }
    cat = Table(cat)

    if wcs is not None and wcs.celestial:
        cat['ra'], cat['dec'] = wcs.all_pix2world(cat['x'], cat['y'], 0)
    else:
        cat['ra'], cat['dec'] = np.nan, np.nan

    cat['mag'] = -2.5 * np.log10(cat['flux'])

    return cat


def place_random_stars(
    image,
    psf_model,
    nstars=100,
    minflux=1,
    maxflux=100000,
    gain=None,
    saturation=65535,
    edge=0,
    wcs=None,
    verbose=False,
):
    """Randomly place artificial stars into an image in-place.

    Stars are added on top of the existing image content. Poissonian noise is
    applied according to ``gain``, and pixel values are clipped at ``saturation``.
    Coordinates are uniformly distributed; fluxes are log-uniform.

    Parameters
    ----------
    image : ndarray
        2D image array; modified in-place.
    psf_model : dict
        PSF model structure as returned by :func:`stdpipe.psf.run_psfex`.
    nstars : int, optional
        Number of artificial stars to inject.
    minflux : float, optional
        Minimum star flux in ADU.
    maxflux : float, optional
        Maximum star flux in ADU.
    gain : float, optional
        Detector gain in e-/ADU; used to add Poissonian noise to each injected star.
    saturation : float, optional
        Saturation level in ADU; injected pixels above this are clipped.
    edge : int, optional
        Minimum distance to image edges for star placement.
    wcs : astropy.wcs.WCS, optional
        If provided, sky coordinates ``ra`` and ``dec`` are added to the catalogue.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.table.Table
        Catalogue of injected stars as returned by :func:`make_random_stars`,
        with columns ``x``, ``y``, ``flux``, ``mag``, and optionally ``ra``, ``dec``.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    cat = make_random_stars(
        shape=image.shape,
        nstars=nstars,
        minflux=minflux,
        maxflux=maxflux,
        edge=edge,
        wcs=wcs,
        verbose=verbose,
    )

    for _ in cat:
        psf.place_psf_stamp(image, psf_model, _['x'], _['y'], flux=_['flux'], gain=gain)

    if saturation is not None:
        image[image > saturation] = saturation

    return cat


def split_sub_fn(x1, y1, dx1, dy1, *args, get_origin=False, **kwargs):
    """ """
    result = []

    if get_origin:
        result += [x1, y1]

    # Let's find WCS as we may need it later for converting sky coordinates
    wcs = None
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, WCS):
            wcs = arg
            break

    def handle_arg_fn(arg):
        if isinstance(arg, np.ndarray):
            # Image
            return cutouts.crop_image(arg, x1, y1, dx1, dy1)

        if isinstance(arg, fits.Header):
            # FITS header
            subheader = arg.copy()
            subheader['NAXIS1'] = dx1
            subheader['NAXIS2'] = dy1

            # Adjust the WCS keywords if present
            if 'CRPIX1' in subheader and 'CRPIX2' in subheader:
                subheader['CRPIX1'] -= x1
                subheader['CRPIX2'] -= y1

            # Crop position inside original frame
            subheader['CROP_X1'] = x1
            subheader['CROP_X2'] = x1 + dx1
            subheader['CROP_Y1'] = y1
            subheader['CROP_Y2'] = y1 + dy1

            return subheader

        if isinstance(arg, WCS):
            # WCS
            wcs1 = arg.deepcopy()
            # FIXME: is there any more 'official' way of shifting the WCS?
            wcs1.wcs.crpix[0] -= x1
            wcs1.wcs.crpix[1] -= y1

            wcs1 = WCS(wcs1.to_header(relax=True))
            return wcs1

        if isinstance(arg, Table):
            # Table
            table = arg.copy()

            x, y = None, None

            if 'x' in table.colnames and 'y' in table.colnames:
                x, y = table['x'].copy(), table['y'].copy()
                table['x'] -= x1
                table['y'] -= y1
            elif 'ra' in table.colnames and 'dec' in table.colnames and wcs is not None:
                x, y = wcs.all_world2pix(table['ra'], table['dec'], 0)
            elif 'RAJ2000' in table.colnames and 'DEJ2000' in table.colnames and wcs is not None:
                x, y = wcs.all_world2pix(table['RAJ2000'], table['DEJ2000'], 0)

            if x is not None:
                idx = (x > x1) & (x < x1 + dx1)
                idx &= (y > y1) & (y < y1 + dy1)
            else:
                idx = np.ones(len(table), dtype=bool)

            return table[idx]

        if isinstance(arg, dict) and 'x0' in arg and 'y0' in arg:
            # PSF structure, but also maybe some other dict-based objects
            res = deepcopy(arg)
            res['x0'] -= x1
            res['y0'] -= y1

            return res

        return None

    for arg in args + tuple(kwargs.values()):
        result.append(handle_arg_fn(arg))

    return result


def split_image(
    image,
    *args,
    nx=1,
    ny=None,
    overlap=0,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    get_index=False,
    get_origin=False,
    verbose=False,
    **kwargs,
):
    """Generator that splits an image into ``nx × ny`` sub-image blocks.

    Also optionally yields sub-setted copies of any additional images, FITS
    headers, WCS solutions, PSF models, catalogues, or object lists passed as
    positional or keyword arguments.  FITS headers and WCS solutions are
    adjusted to reflect the sub-image astrometry.  Tables are filtered to rows
    whose ``x``/``y``, ``ra``/``dec``, or ``RAJ2000``/``DEJ2000`` coordinates
    fall inside the sub-image.

    Sub-images may optionally overlap by ``overlap`` pixels in every direction,
    so that each part of the original image appears far from an edge in at least
    one block.

    Parameters
    ----------
    image : ndarray
        2D image to split.
    *args
        Additional images, headers, WCS objects, or tables to crop alongside ``image``.
    nx : int, optional
        Number of sub-images along the x axis.
    ny : int, optional
        Number of sub-images along the y axis. Defaults to ``nx``.
    overlap : int, optional
        Number of pixels by which adjacent blocks overlap.
    xmin, xmax : int, optional
        Horizontal extent of the region to split (defaults to full image width).
    ymin, ymax : int, optional
        Vertical extent of the region to split (defaults to full image height).
    get_index : bool, optional
        If True, prepend the sub-image index (0-based) to each yielded list.
    get_origin : bool, optional
        If True, include the sub-image origin ``(x1, y1)`` in each yielded list.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    **kwargs
        Additional images, headers, WCS objects, or tables to crop.

    Yields
    ------
    list
        Each element is a list built from (in order):

        - sub-image index, if ``get_index=True``
        - origin ``x1``, ``y1``, if ``get_origin=True``
        - cropped image
        - cropped versions of each ``*args`` and ``**kwargs`` entry
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if not ny:
        ny = nx

    # Sub-region to split
    if xmin is None:
        xmin = 0
    else:
        xmin = int(max(0, xmin))

    if xmax is None:
        xmax = image.shape[1]
    else:
        xmax = int(min(image.shape[1], xmax))

    if ymin is None:
        ymin = 0
    else:
        ymin = int(max(0, ymin))

    if ymax is None:
        ymax = image.shape[0]
    else:
        ymax = int(min(image.shape[0], ymax))

    width = xmax - xmin
    height = ymax - ymin

    dx, dy = int(np.floor(width / nx)), int(np.floor(height / ny))

    log(
        'Will split the image (%dx%d) into %dx%d pieces with %dx%d pixels size and %d pix overlap'
        % (width, height, nx, ny, dx, dy, overlap)
    )

    for i, (x0, y0) in enumerate(
        itertools.product(range(xmin, xmax - dx + 1, dx), range(ymin, ymax - dy + 1, dy))
    ):
        # Make some overlap
        x1 = max(0, x0 - overlap)
        y1 = max(0, y0 - overlap)
        dx1 = min(x0 - x1 + dx + overlap, xmax - x1)
        dy1 = min(y0 - y1 + dy + overlap, ymax - y1)

        result = split_sub_fn(x1, y1, dx1, dy1, image, *args, get_origin=get_origin, **kwargs)

        if get_index:
            result = [i] + result

        log('Block %d: %d %d - %d %d' % (i, x1, y1, x1 + dx1, y1 + dy1))

        yield result if len(result) > 1 else result[0]


def get_subimage_centered(
    image,
    *args,
    x0=None,
    y0=None,
    width=None,
    height=None,
    get_origin=False,
    verbose=False,
    **kwargs,
):
    """Crop a sub-image centered at a given pixel position.

    Also optionally crops any additional images, masks, headers, WCS solutions,
    PSF models, object lists, etc. passed as arguments.  Behaviour is identical
    to :func:`split_image` for a single block.

    Unlike :func:`stdpipe.utils.crop_image_centered`, this function accepts
    explicit output ``width`` and ``height``.  If the requested region extends
    beyond the image boundary, the returned sub-image will be smaller (no
    padding is applied).

    Parameters
    ----------
    image : ndarray
        2D image to crop.
    *args
        Additional images, headers, WCS objects, or tables to crop alongside ``image``.
    x0 : int or float
        Pixel x coordinate of the desired center in the original image.
    y0 : int or float
        Pixel y coordinate of the desired center in the original image.
    width : int
        Pixel width of the sub-image.
    height : int, optional
        Pixel height of the sub-image. Defaults to ``width``.
    get_origin : bool, optional
        If True, include the sub-image origin ``(x1, y1)`` in the returned list.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    **kwargs
        Additional images, headers, WCS objects, or tables to crop.

    Returns
    -------
    list
        List containing:

        - origin ``x1``, ``y1``, if ``get_origin=True``
        - cropped image
        - cropped versions of each ``*args`` and ``**kwargs`` entry
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if not height:
        height = width

    x1 = int(np.floor(x0 - 0.5 * width))
    y1 = int(np.floor(y0 - 0.5 * height))

    x2 = x1 + width - 1
    y2 = y1 + height - 1

    x1, x2 = max(0, x1), min(x2, image.shape[1] - 1)
    y1, y2 = max(0, y1), min(y2, image.shape[0] - 1)

    log(
        'Will crop the %d %d - %d %d sub-image from original (%dx%d) image centered at %d %d'
        % (x1, y1, x2, y2, image.shape[1], image.shape[0], x0, y0)
    )

    result = split_sub_fn(
        x1, y1, x2 - x1 + 1, y2 - y1 + 1, image, *args, get_origin=get_origin, **kwargs
    )

    return result


def get_detection_limit(obj, sn=5, method='sn', verbose=True):
    """Estimate the detection limit magnitude.

    The object table must contain calibrated magnitudes and errors in
    ``mag_calib`` and ``mag_calib_err`` columns.

    Parameters
    ----------
    obj : astropy.table.Table
        Table with calibrated objects.
    sn : float, optional
        S/N threshold defining the detection limit.
    method : str, optional
        Estimation method. Currently only ``'sn'`` (S/N vs magnitude
        extrapolation) is implemented.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    float or None
        Magnitude corresponding to the detection limit at the given S/N level,
        or None if estimation fails.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if method == 'sn':
        log('Estimating detection limit using S/N vs magnitude method')
        mag0 = photometry.get_detection_limit_sn(
            obj['mag_calib'], 1 / obj['magerr'], sn=sn, verbose=verbose
        )
    elif method == 'bg':
        log('Estimating detection limit using background noise method')
        raise RuntimeError('Not implemented')

    if mag0 is not None:
        log('Detection limit at S/N=%g level is %.2f' % (sn, mag0))
    else:
        log('Error estimating the detection limit!')

    return mag0
