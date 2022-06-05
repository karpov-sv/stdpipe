"""
Module containing higher-level pipeline building blocks, wrapping together lower-level
functionality of STDPipe modules.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import itertools

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

from . import photometry
from . import astrometry
from . import catalogs
from . import psf
from . import utils
from . import cutouts

def refine_astrometry(obj, cat, sr=10/3600, wcs=None, order=0,
                      cat_col_mag='V', cat_col_mag_err=None,
                      cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                      cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000',
                      n_iter=3, use_photometry=True, min_matches=5, method='astropy',
                      update=True, verbose=False, **kwargs):

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
    :param \**kwargs: All other parameters will be directly passed to :func:`~stdpipe.astrometry.refine_wcs_scamp`
    :returns: Refined astrometric solution

    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    log('Astrometric refinement using %.1f arcsec radius, %s matching and %s WCS fitting' %
        (sr*3600, 'photometric' if use_photometry else 'simple positional', method))

    if wcs is not None:
        obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    if method == 'scamp':
        # Fall-through to SCAMP-specific variant
        return astrometry.refine_wcs_scamp(obj, cat, sr=sr, wcs=wcs, order=order,
                                           cat_col_mag=cat_col_mag, cat_col_mag_err=cat_col_mag_err,
                                           cat_col_ra=cat_col_ra, cat_col_dec=cat_col_dec,
                                           cat_col_ra_err=cat_col_ra_err, cat_col_dec_err=cat_col_dec_err,
                                           update=update, verbose=verbose, **kwargs)

    for iter in range(n_iter):
        if use_photometry:
            # Matching involving photometric information
            cat_magerr = cat[cat_col_mag_err] if cat_col_mag_err is not None else None
            m = photometry.match(obj['ra'], obj['dec'], obj['mag'], obj['magerr'], obj['flags'], cat[cat_col_ra], cat[cat_col_dec], cat[cat_col_mag], cat_magerr=cat_magerr, sr=sr)
            if m is None or not m:
                log('Photometric match failed, cannot refine WCS')
                return None
            elif np.sum(m['idx']) < min_matches:
                log('Too few (%d) good photometric matches, cannot refine WCS' % np.sum(m['idx']))
                return None
            else:
                log('Iteration %d: %d matches, %.1f arcsec rms' %
                    (iter, np.sum(m['idx']), np.std(3600*m['dist'][m['idx']])))

            wcs = astrometry.refine_wcs(obj[m['oidx']][m['idx']], cat[m['cidx']][m['idx']], order=order, match=False, method=method)
        else:
            # Simple positional matching
            wcs = astrometry.refine_wcs(obj, cat, order=order, sr=sr, match=True, method=method, **kwargs)

        if update:
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)


    return wcs

def filter_transient_candidates(obj, sr=None, pixscale=None, time=None,
                                obj_col_ra = 'ra', obj_col_dec = 'dec',
                                cat=None, cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                                vizier=['ps1', 'usnob1', 'gsc'], skybot=True, ned=False,
                                flagged=True, flagmask=0xff00,
                                col_id=None, get_candidates=True, remove=True, verbose=False):
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
    :param pixscale: Pixel scale. If provided, and `sr` is not specified, the latter will be set to half median of FWHMs of objects
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
    :param get_candidates: Whether to return the list of candidates, or (if :code:`get_candidates=False`) just a boolean mask of objects surviving the filters
    :param remove: Whether to remove the filtered entries from the returned list. If not, a number of additional columns (all having names `candidate_*`) will be added for every filter used, with `True` set if the filter matches the object. Finally, the boolean column `candidate_good` will have `True` for the objects surviving all the filters
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Either the copy of the list with "good" candidates, or (if :code:`get_candidates=False`) just a boolean mask of objects surviving the filters, with the same size as original objects list.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if sr is None:
        if pixscale is not None:
            # Matching radius of half FWHM
            sr = np.median(obj['fwhm']*pixscale)/2
        else:
            # Fallback value of 1 arcsec, should be sensible for most catalogues
            sr = 1/3600

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

    log('Candidate filtering routine started with %d initial candidates and %.1f arcsec matching radius' % (len(obj), sr*3600))
    cand_idx = np.ones(len(obj), dtype=np.bool)

    # Object flags
    if flagged:
        # Filter out flagged objects (saturated, cosmics, blends, etc)
        cand_idx &= (obj['flags'] & flagmask) == 0

        if remove == False:
            obj_in['candidate_flagged'] = (obj['flags'] & flagmask) > 0

        log(np.sum(cand_idx), 'of them are unflagged')

    # Reference catalogue
    if cat is not None and remove == False:
        obj_in['candidate_refcat'] = False
    if cat is not None and np.any(cand_idx):
        m = astrometry.spherical_match(obj['ra'], obj['dec'], cat[cat_col_ra], cat[cat_col_dec], sr)
        cand_idx[m[0]] = False

        if remove == False:
            obj_in['candidate_refcat'] = False
            obj_in['candidate_refcat'][m[0]] = True

        log(np.sum(cand_idx), 'of them are not matched with reference catalogue')

    # Vizier catalogues
    for catname in (vizier or []):
        if remove == False:
            if 'candidate_vizier_'+catname not in obj_in.keys():
                obj_in['candidate_vizier_'+catname] = False

        if not np.any(cand_idx):
            break

        xcat = catalogs.xmatch_objects(obj[cand_idx], catname, sr)
        if xcat is not None and len(xcat):
            cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

            if remove == False:
                obj_in['candidate_vizier_'+catname][np.in1d(obj[col_id], xcat[col_id])] = True

        log(np.sum(cand_idx), 'remains after matching with', catalogs.catalogs.get(catname, {'name':catname})['name'])

    # SkyBoT
    if skybot and np.any(cand_idx):
        if time is None and 'time' in obj.keys():
            time = obj['time']

        if remove == False:
            if 'candidate_skybot' not in obj_in.keys():
                obj_in['candidate_skybot'] = False

        if time is not None:
            xcat = catalogs.xmatch_skybot(obj[cand_idx], time=time, col_id=col_id)
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

        xcat = catalogs.xmatch_ned(obj[cand_idx], sr, col_id=col_id)
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

def calibrate_photometry(obj, cat, sr=None, pixscale=None, order=0, bg_order=None,
                         obj_col_mag='mag', obj_col_mag_err='magerr',
                         obj_col_ra='ra', obj_col_dec='dec',
                         obj_col_x='x', obj_col_y='y',
                         cat_col_mag='R', cat_col_mag_err=None,
                         cat_col_mag1=None, cat_col_mag2=None,
                         cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                         update=True, verbose=False, **kwargs):

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
    :param \**kwargs: The rest of keyword arguments will be directly passed to :func:`stdpipe.photometry.match`.
    :returns: The dictionary with photometric results, as returned by :func:`stdpipe.photometry.match`.

    """
    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if sr is None:
        if pixscale is not None:
            # Matching radius of half FWHM
            sr = np.median(obj['fwhm']*pixscale)/2
        else:
            # Fallback value of 1 arcsec, should be sensible for most catalogues
            sr = 1/3600

    log('Performing photometric calibration of %d objects vs %d catalogue stars' % (len(obj), len(cat)))
    log('Using %.1f arcsec matching radius, %s magnitude and spatial order %d' % (sr*3600, cat_col_mag, order))
    if cat_col_mag1 and cat_col_mag2:
        log('Using (%s - %s) color for color term' % (cat_col_mag1, cat_col_mag2))

    if cat_col_mag1 and cat_col_mag2:
        color = cat[cat_col_mag1]-cat[cat_col_mag2]
    else:
        color = None

    if cat_col_mag_err:
        cat_magerr = cat[cat_col_mag_err]
    else:
        cat_magerr = None

    m = photometry.match(obj['ra'], obj['dec'], obj[obj_col_mag], obj[obj_col_mag_err], obj['flags'],
                         cat[cat_col_ra], cat[cat_col_dec], cat[cat_col_mag],
                         cat_magerr=cat_magerr,
                         sr=sr, cat_color=color,
                         obj_x=obj['x'], obj_y=obj['y'], spatial_order=order, bg_order=bg_order,
                         verbose=verbose, **kwargs)

    if m:
        log('Photometric calibration finished successfully.')
        # if m['color_term']:
        #     log('Color term is %.2f' % m['color_term'])

        m['cat_col_mag'] = cat_col_mag
        if cat_col_mag1 and cat_col_mag2:
            m['cat_col_mag1'] = cat_col_mag1
            m['cat_col_mag2'] = cat_col_mag2

        if update:
            obj['mag_calib'] = obj[obj_col_mag] + m['zero_fn'](obj['x'], obj['y'], obj['mag'])
            obj['mag_calib_err'] = np.hypot(obj[obj_col_mag_err], m['zero_fn'](obj['x'], obj['y'], obj['mag'], get_err=True))
    else:
        log('Photometric calibration failed')

    return m

def make_random_stars(width=None, height=None, shape=None, nstars=100, minflux=1, maxflux=100000, edge=0, wcs=None, verbose=False):
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
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if (width is None or height is None) and shape is not None:
        height,width = shape

    cat = {
        'x': np.random.uniform(edge, width - 1 - edge, nstars),
        'y': np.random.uniform(edge, height - edge, nstars),
        'flux': 10**np.random.uniform(np.log10(minflux), np.log10(maxflux), nstars)
    }
    cat = Table(cat)

    if wcs is not None and wcs.celestial:
        cat['ra'], cat['dec'] = wcs.all_pix2world(cat['x'], cat['y'], 0)
    else:
        cat['ra'], cat['dec'] = np.nan, np.nan

    cat['mag'] = -2.5*np.log10(cat['flux'])

    return cat

def place_random_stars(image, psf_model, nstars=100, minflux=1, maxflux=100000, gain=None, saturation=65535, edge=0, wcs=None, verbose=False):
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
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    cat = make_random_stars(shape=image.shape, nstars=nstars, minflux=minflux, maxflux=maxflux,
                            edge=edge, wcs=wcs, verbose=verbose)

    for _ in cat:
        psf.place_psf_stamp(image, psf_model, _['x'], _['y'], flux=_['flux'], gain=gain)

    if saturation is not None:
        image[image > saturation] = saturation

    return cat

def split_image(image, nx=1, ny=None, mask=None, bg=None, err=None, header=None, wcs=None, obj=None, cat=None, overlap=0, get_index=False, get_origin=False, verbose=False):
    """
    Generator function to split the image into several (`nx` x `ny`) blocks, while also optionally providing the mask, header, wcs, object list etc for the sub-blocks.
    The blocks may optionally be extended by 'overlap' pixels in all directions, so that at least in some sub-images every part of original image is far from the edge. This parameter may be used e.g. in conjunction with `edge` parameter of :func:`stdpipe.photometry.get_objects_sextractor` to avoid detecting the same object twice.

    :param image: Image to split
    :param nx: Number of sub-images in `x` direction
    :param ny: Number of sub-images in `y` direction
    :param mask: Mask image to split, optional
    :param bg: Background map to split, optional
    :param err: Noise model image to split, optional
    :param header: Image header, optional. If set, the header corresponding to splitted sub-image will be returned, with correctly adjusted WCS information
    :param wcs: WCS solution for the image, optional. If set, the solution for sub-image will be returned
    :param obj: Object list, optional. If provided, the list of objects contained in the sub-image, with accordingly adjusted pixel coordinates, will be returned
    :param cat: Reference catalogue, optional. If provided, the catalogue for the stars on the sub-image will be returned
    :param overlap: If set, defines how much sub-images will overlap, in pixels.
    :param get_index: If set, also returns the number of current sub-image, starting from zero
    :param get_origin: If set, also return the sub-image origin pixel coordinates
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :returns: Every concecutive call to the generator will return the list of cropped objects corresponding to the next sub-image, as well as some sub-image metadata.

    The returned list is constructed from the following elements:

    - Index of current sub-image, if :code:`get_index=True`
    - `x` and `y` coordinates of current sub-image origin inside the original image
    - Current sub-image
    - Cropped mask corresponding to current sub-image, if `mask` is provided
    - Cropped background map, if `bg` is provided
    - Cropped noise model, if `err` is provided
    - FITS header corresponding to the sub-image with correct astrometric solution for it, if `header` is provided
    - WCS astrometric solution for the sub-image, if `wcs` is provided
    - Object list containing only objects that are inside the sub-image with their pixel coordinates adjusted correspondingly, if `obj` is set
    - Reference catalogue containing only stars overlaying the sub-image, if `cat` is provided and `wcs` is set

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if not ny:
        ny = nx

    dx,dy = int(np.floor(image.shape[1]/nx)), int(np.floor(image.shape[0]/ny))

    log('Will split the image (%dx%d) into %dx%d pieces with %dx%d pixels size and %d pix overlap' % (image.shape[1], image.shape[0], nx, ny, dx, dy, overlap))

    for i,(x0,y0) in enumerate(itertools.product(range(0, image.shape[1]-dx+1, dx), range(0, image.shape[0]-dy+1, dy))):
        # Make some overlap
        x1 = max(0, x0 - overlap)
        y1 = max(0, y0 - overlap)
        dx1 = min(x0 - x1 + dx + overlap, image.shape[1] - x1)
        dy1 = min(y0 - y1 + dy + overlap, image.shape[0] - y1)

        _ = cutouts.crop_image(image.astype(np.double), x1, y1, dx1, dy1, header=header)
        if header:
            image1,header1 = _
        else:
            image1,header1 = _,None

        result = []

        if get_index:
            result += [i]

        if get_origin:
            result += [x1, y1]

        result += [image1]

        if mask is not None:
            result += [cutouts.crop_image(mask, x1, y1, dx1, dy1)]

        if bg is not None:
            result += [cutouts.crop_image(bg, x1, y1, dx1, dy1)]

        if err is not None:
            result += [cutouts.crop_image(err, x1, y1, dx1, dy1)]

        if header1 is not None:
            result += [header1]

        if wcs is not None:
            wcs1 = wcs.deepcopy()
            # FIXME: is there any more 'official' way of shifting the WCS?
            wcs1.wcs.crpix[0] -= x1
            wcs1.wcs.crpix[1] -= y1

            wcs1 = WCS(wcs1.to_header(relax=True))

            result += [wcs1]

        if obj is not None:
            oidx = (obj['x'] > x1) & (obj['x'] < x1 + dx1) & (obj['y'] > y1) & (obj['y'] < y1 + dy1)
            obj1 = obj[oidx].copy()
            obj1['x'] -= x1
            obj1['y'] -= y1
            result += [obj1]

        if cat is not None and wcs is not None:
            cx,cy = wcs.all_world2pix(cat['RAJ2000'], cat['DEJ2000'], 0)
            cidx = (cx >= x1) & (cx < x1 + dx1) & (cy >= y1) & (cy < y1 + dy1)
            result += [cat[cidx].copy()]

        log('Block %d: %d %d - %d %d' % (i, x1, y1, x1+dx1, y1+dy1))

        yield result if len(result) > 1 else result[0]
