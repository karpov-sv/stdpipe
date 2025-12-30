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
    verbose=True
):
    """
    Make basic mask for the image. The mask is a boolean bitmap with True values marking the regions
    that should be excluded from following processing, thus "masked".

    The routine masks the following regions of the image:
    - pixels with undefined or non-finite (inf or nan) values
    - regions outside of usable area defined by header DATASEC or TRIMSEC keyword
    - pixels with values above the saturation limit, if provided
    - pixels masked in external mask, if provided
    - cosmic rays, if requested

    If :code:`saturation=True`, saturation level will be estimated from the image as :code:`median + 0.95(max - median)`

    :param image: Image to be masked
    :param header: FITS header, optional
    :param saturation: Saturation level. If set to `True`, will be estimated from the image itself. Optional
    :param external_mask: External mask, to be ORed with the created mask. Optional
    :param mask_cosmics: If set, cosmic rays will be masked using LACosmic algorithm.
    :param gain: Gain value to be used for cosmic rays masking
    :returns: Boolean bitmap ("mask") with True values marking the regions that should be excluded from following processing

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
            saturation = 0.05*np.nanmedian(image) + 0.95*np.nanmax(image) # med + 0.95(max-med)
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
            var += np.abs(image - bg.back())/gain
        cmask, cimage = astroscrappy.detect_cosmics(
            image, mask,
            verbose=verbose,
            invar=var.astype(np.float32),
            gain=gain if gain else 1.0,
            satlevel=saturation if saturation else np.max(image),
            cleantype='medmask')
        log('Done masking cosmics, %d (%.1f%%) pixels masked' % (
            np.sum(cmask),
            100*np.sum(cmask)/cmask.shape[0]/cmask.shape[1]
        ))
        mask |= cmask

    log('%d (%.1f%%) pixels masked in total' % (
        np.sum(mask),
        100*np.sum(mask)/mask.shape[0]/mask.shape[1]
    ))

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
    method='astropy',
    update=True,
    verbose=False,
    **kwargs
):

    """Higher-level astrometric refinement routine that may use either SCAMP or pure Python based methods.

    :param obj: List of objects on the frame that should contain at least `x`, `y` and `flux` columns.
    :param cat: Reference astrometric catalogue
    :param sr: Matching radius in degrees
    :param wcs: Initial WCS
    :param order: Polynomial order for SIP or PV distortion solution
    :param cat_col_mag: Catalogue column name for the magnitude in closest band
    :param cat_col_mag_err: Catalogue column name for the magnitude error
    :param cat_col_ra: Catalogue column name for Right Ascension
    :param cat_col_dec: Catalogue column name for Declination
    :param cat_col_ra_err: Catalogue column name for Right Ascension error
    :param cat_col_dec_err: Catalogue column name for Declination error
    :param n_iter: Number of iterations for Python-based matching
    :param use_photometry: Use photometry-assisted method in Python-based matching
    :param min_matches: Minimal number of good matches in Python-based matching
    :param method: May be either 'scamp' or 'astropy' or 'astrometrynet'
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param \\**kwargs: All other parameters will be directly passed to :func:`~stdpipe.astrometry.refine_wcs_scamp`
    :returns: Refined astrometric solution

    """
    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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

    if method == 'scamp':
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
            **kwargs
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
                log(
                    'Too few (%d) good photometric matches, cannot refine WCS'
                    % np.sum(m['idx'])
                )
                return None
            else:
                log(
                    'Iteration %d: %d matches, %.1f arcsec rms'
                    % (iter, np.sum(m['idx']), np.std(3600 * m['dist'][m['idx']]))
                )

            wcs = astrometry.refine_wcs(
                obj[m['oidx']][m['idx']],
                cat[m['cidx']][m['idx']],
                order=order,
                match=False,
                method=method,
                verbose=verbose,
            )
        else:
            # Simple positional matching
            wcs = astrometry.refine_wcs(
                obj,
                cat,
                order=order,
                sr=sr,
                match=True,
                method=method,
                verbose=verbose,
                **kwargs
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

    It optionally filters out the following classes of objects:

       - flagged ones, i.e. with :code:`obj['flags'] != 0`
       - positionally coincident with stars from provided cataloge table (if :code:`cat != None`)
       - positionally coincident with stars from Vizier catalogues provided as a list of names (if `vizier` is non-empty)
       - positionally and temporally coincident with Solar system objects from SkyBoT service (if :code:`skybot = True`)
       - positionally and temporally coincident with NED objects (if :code:`ned = True`).

    If :code:`get_candidates = False`, it returns only the indices of "good" objects, else returning filtered object list.
    If :code:`get_candidates = True` and :code:`remove = False`, it does not remove the objects but just mark them in added columns corresponding to various filters applied.

    The routine will not modify the original list, but return its filtered / modified copy instead.

    :param obj: Input object list
    :param obj_col_ra: Column name for object Right Ascension
    :param obj_col_dec: Column name for object Declination
    :param sr: Matching radius in degrees
    :param pixscale: Pixel scale. If provided, and `sr` is not specified, the latter will be set to half FWHM
    :param fwhm: FWHM value in pixels. If not set, will be estimated as a median of FWHMs of all unflagged objects
    :param time: Time corresponding to the object observations, as :class:`astropy.time.Time` or :class:`datetime.datetime` object.
    :param cat: Input reference catalogue, as :class:`astropy.table.Table` or similar object. Optional
    :param cat_col_ra: Column name for catalogue Right Ascension
    :param cat_col_dec: Column name for catalogue Declination
    :param vizier: List of Vizier catalogue identifiers, or their short names, to cross-match the objects with.
    :param skybot: Whether to cross-match the objects with the positions of known minor planets at the time of observations (specified through `time` parameter)
    :param ned: Whether to cross-match the positions of objects with NED database entries
    :param flagged: Whether to filter out flagged objects, keeping only the ones with :code:`(obj['flags'] & flagmask) == 0`
    :param flagmask: The mask to be used for filtering of flagged objects. Will be ANDed with object flags, so should have a bit set for every relevant mask bit
    :param col_id: Column name for some unique identifier that should not appear twice in the object list. If not specified, new column `stdpipe_id` with unique integer identifiers will be created automatically
    :param vizier_checker_fn: Function to check whether the cross-matched Vizier catalogue entries satisfy some additional conditions to be considered true matches. Signature should be :code:`fn(obj, xcat, catname)`, with passed `obj` having the same shape and order as `xcat` returned by cross-match, and should return boolean array of the same shape. Optional
    :param get_candidates: Whether to return the list of candidates, or (if :code:`get_candidates=False`) just a boolean mask of objects surviving the filters
    :param remove: Whether to remove the filtered entries from the returned list. If not, a number of additional columns (all having names `candidate_*`) will be added for every filter used, with `True` set if the filter matches the object. Finally, the boolean column `candidate_good` will have `True` for the objects surviving all the filters
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Either the copy of the list with "good" candidates, or (if :code:`get_candidates=False`) just a boolean mask of objects surviving the filters, with the same size as original objects list.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if fwhm is None:
        idx = obj['flags'] == 0
        idx &= obj['magerr'] < 0.05 # S/N > 20
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
            obj[obj_col_ra],
            obj[obj_col_dec],
            cat[cat_col_ra],
            cat[cat_col_dec], sr
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
                obj_in['candidate_vizier_' + catname][
                    np.in1d(obj[col_id], xcat[col_id])
                ] = True

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
                obj[cand_idx],
                time=time,
                col_ra=obj_col_ra,
                col_dec=obj_col_dec,
                col_id=col_id
            )
            if xcat is not None and len(xcat):
                cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

                if remove == False:
                    obj_in['candidate_skybot'][
                        np.in1d(obj[col_id], xcat[col_id])
                    ] = True

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
    **kwargs
):

    """Higher-level photometric calibration routine.

    It wraps :func:`stdpipe.photometry.match` routine with some convenient defaults so that it is easier to use with typical tabular data.

    :param obj: Table of detected objects
    :param cat: Reference photometric catalogue
    :param sr: Matching radius in degrees, optional
    :param pixscale: Pixel scale, degrees per pixel. If specified, and `sr` is not set, then median value of half of FWHM, multiplied by pixel scale, is used as a matching radius.
    :param order: Order of zero point spatial polynomial (0 for constant).
    :param bg_order: Order of additive flux term spatial polynomial (None to disable this term in the model)
    :param obj_col_mag: Column name for object instrumental magnitude
    :param obj_col_mag_err: Column name for object magnitude error
    :param obj_col_ra: Column name for object Right Ascension
    :param obj_col_dec: Column name for object Declination
    :param obj_col_flags: Column name for object flags
    :param obj_col_x: Column name for object x coordinate
    :param obj_col_y: Column name for object y coordinate
    :param cat_col_mag: Column name for catalogue magnitude
    :param cat_col_mag_err: Column name for catalogue magnitude error
    :param cat_col_mag1: Column name for the first catalogue magnitude defining the stellar color
    :param cat_col_mag2: Column name for the second catalogue magnitude defining the stellar color
    :param cat_col_ra: Column name for catalogue Right Ascension
    :param cat_col_dec: Column name for catalogue Declination
    :param update: If True, `mag_calib` and `mag_calib_err` columns with calibrated magnitude (without color term) and its error will be added to the object table
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :param \\**kwargs: The rest of keyword arguments will be directly passed to :func:`stdpipe.photometry.match`.
    :returns: The dictionary with photometric results, as returned by :func:`stdpipe.photometry.match`.

    """
    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
        **kwargs
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

    Coordinates are distributed uniformly with :code:`edge <= x < width-edge` and :code:`edge <= y < height-edge`.

    Fluxes are log-uniform between user-provided min and max values.

    Returns the catalogue of generated stars, with at least `x`, `y` and `flux` fields set.
    If `wcs` is set, the returned catalogue will also contain `ra` and `dec` fields with
    sky coordinates of the stars.

    :param width: Width of the image containing generated stars
    :param height: Height of the image containing generated stars
    :param shape: Shape  of the image containing generated stars, to be used instead of `width` and `height` if set
    :param nstars: Number of artificial stars to inject into the image
    :param minflux: Minimal flux of arttificial stars, in ADU units
    :param maxflux: Maximal flux of arttificial stars, in ADU units
    :param edge: Minimal distance to image edges for artificial stars to be placed. Optional
    :param wcs: WCS as :class:`astropy.wcs.WCS` to be used to derive sky coordinates of injected stars. Optional
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The catalogue of injected stars, containing the fluxes, image coordinates and (if `wcs` is set) sky coordinates of injected stars.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
    """Randomly place artificial stars into the image.

    The stars will be placed on top of the existing content of the image, and the Poissonian
    noise will be applied to these stars according to the specified `gain` value. Also, the saturation
    level will be applied to the resulting image according to the `saturation` value.

    Coordinates of the injected stars are distributed uniformly.
    Fluxes are log-uniform between user-provided `minflux` and `maxflux` values in ADU units.

    Returns the catalogue of generated stars, with at least `x`, `y` and `flux` fields set,
    as returned by :func:`stdpipe.pipeline.make_random_stars`

    If `wcs` is set, the returned catalogue will also contain `ra` and `dec` fields with
    sky coordinates of the stars

    :param image: Image where artificial stars will be injected
    :param psf_model: PSF model structure as returned by :func:`stdpipe.psf.run_psfex`
    :param nstars: Number of artificial stars to inject into the image
    :param minflux: Minimal flux of arttificial stars, in ADU units
    :param maxflux: Maximal flux of arttificial stars, in ADU units
    :param gain: Image gain value. If set, will be used to apply Poissonian noise to the source
    :param saturation: Saturation level in ADU units to be applied to the image
    :param edge: Minimal distance to image edges for artificial stars to be placed
    :param wcs: WCS as :class:`astropy.wcs.WCS` to be used to derive sky coordinates of injected stars
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :returns: The catalogue of injected stars, containing the fluxes, image coordinates and (if `wcs` is set) sky coordinates of injected stars.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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


def split_sub_fn(
    x1,
    y1,
    dx1,
    dy1,
    *args,
    get_origin=False,
    **kwargs
):
    """
    """
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
    **kwargs
):
    """Generator function to split the image into several (`nx` x `ny`)
    blocks, while also optionally providing the subsets of images, FITS
    headers, WCS solutions, PSFs, catalogues or object lists for the
    sub-blocks. FITS headers and WCS solutions will be adjusted to properly
    reflect the astrometry in the sub-image. Tables will be sub-setted to only
    include the rows that are inside the sub-image, according to their `x` and
    `y`, or `ra` and `dec`, or `RAJ2000` and `DEJ2000` columns.

    The blocks may optionally be extended by 'overlap' pixels in all
    directions, so that at least in some sub-images every part of original
    image is far from the edge. This parameter may be used e.g. in conjunction
    with `edge` parameter of :func:`stdpipe.photometry.get_objects_sextractor`
    to avoid detecting the same object twice.

    :param image: Image to split
    :param \\*args: Set of additional images, headers, WCS solutions, or tables to split
    :param nx: Number of sub-images in `x` direction
    :param ny: Number of sub-images in `y` direction
    :param overlap: If set, defines how much sub-images will overlap, in pixels.
    :param xmin: If set, defines the image sub-region for splitting, otherwise 0
    :param xmax: If set, defines the image sub-region for splitting, otherwise image.shape[1]
    :param ymin: If set, defines the image sub-region for splitting, otherwise 0
    :param ymax: If set, defines the image sub-region for splitting, otherwise image.shape[0]
    :param get_index: If set, also returns the number of current sub-image, starting from zero
    :param get_origin: If set, also return the sub-image origin pixel coordinates
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :param \\**kwargs: Set of images, headers, WCS solutions, or tables to split
    :returns: Every concesutive call to the generator will return the list of cropped objects corresponding to the next sub-image, as well as some sub-image metadata.

    The returned list is constructed from the following elements:

    - Index of current sub-image, if :code:`get_index=True`
    - `x` and `y` coordinates of current sub-image origin inside the original image
    - Cropped image
    - Cropped additional images, headers, WCS objects or tables in the order of their appearance in the arguments

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if not ny:
        ny = nx

    # Sub-region to split
    if xmin is None:
        xmin = 0
    else:
        xmin = max(0, xmin)

    if xmax is None:
        xmax = image.shape[1]
    else:
        xmax = min(image.shape[1], xmax)

    if ymin is None:
        ymin = 0
    else:
        ymin = max(0, ymin)

    if ymax is None:
        ymax = image.shape[0]
    else:
        ymax = min(image.shape[0], ymax)

    width = xmax - xmin
    height = ymax - ymin

    dx, dy = int(np.floor(width / nx)), int(np.floor(height / ny))

    log(
        'Will split the image (%dx%d) into %dx%d pieces with %dx%d pixels size and %d pix overlap'
        % (width, height, nx, ny, dx, dy, overlap)
    )

    for i, (x0, y0) in enumerate(
        itertools.product(
            range(xmin, xmax - dx + 1, dx), range(ymin, ymax - dy + 1, dy)
        )
    ):
        # Make some overlap
        x1 = max(0, x0 - overlap)
        y1 = max(0, y0 - overlap)
        dx1 = min(x0 - x1 + dx + overlap, xmax - x1)
        dy1 = min(y0 - y1 + dy + overlap, ymax - y1)

        result = split_sub_fn(
            x1,
            y1,
            dx1,
            dy1,
            image,
            *args,
            get_origin=get_origin,
            **kwargs
        )

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
    **kwargs
):
    """Convenience function for getting the cropped sub-image centered at a
    given pixel position, while also optionally providing the mask, header,
    wcs, psf, object list etc for it.  Its behaviour and arguments are mostly
    identical to the ones of :func:`stdpipe.pipeline.split_image`.

    In contrast to :func:`stdpipe.utils.crop_image_centered` it accepts output
    width and height as parameters.  These will correspond to the size of the
    output if it is completely inside the original image; if not - they will be
    correspondingly smaller (i.e. it does not pad the data to keep requested
    position exactly at the center).

    :param image: Image to crop
    :param \\*args: Set of additional images, headers, WCS solutions, or tables to split
    :param x0: Pixel `x` coordinate of the cropped image center in the original image
    :param y0: Pixel `y` coordinate of the cropped image center in the original image
    :param width: Pixel width of the sub-image
    :param height: Pixel height of the sub-image, optional. If not provided, assumed to be equal to `width`
    :param get_origin: If set, also return the sub-image origin pixel coordinates
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :param \\**kwargs: Set of images, headers, WCS solutions, or tables to split
    :returns: list of cropped objects corresponding to the sub-image, as well as some sub-image metadata.

    The returned list is constructed from the following elements:

    - `x` and `y` coordinates of current sub-image origin inside the original image
    - Cropped image
    - Cropped additional images, headers, WCS objects or tables in the order of their appearance in the arguments

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
        x1,
        y1,
        x2 - x1 + 1,
        y2 - y1 + 1,
        image,
        *args,
        get_origin=get_origin,
        **kwargs
    )

    return result


def get_detection_limit(obj, sn=5, method='sn', verbose=True):
    """
    Estimate the detection limit using one of several methods.
    The objects table should contain calibrated magnitudes and their errors
    in `mag_calib` and `'mag_calib_err` columns.

    :param obj: astropy.table.Table with calibrated objects
    :param sn: S/N value corresponding to the detection limit
    :param method: Method to use. One of 'sn' (extrapolation S/N vs magnitude) or 'bg' (not yet implemented)
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The magnitude corresponding to the detection limit on a given S/N level.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
