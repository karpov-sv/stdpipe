from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

from esutil import htm

from . import photometry
from . import astrometry
from . import catalogs
from . import psf
from . import utils

def refine_astrometry(obj, cat, sr=10/3600, wcs=None, order=0,
                      cat_col_mag='V', cat_col_mag_err=None,
                      cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                      cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000',
                      n_iter=3, use_photometry=True, min_matches=5, method='astropy',
                      update=True, verbose=False):
    """
    Higher-level astrometric refinement routine.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

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
                                           update=update, verbose=verbose)

    for iter in range(n_iter):
        if use_photometry:
            # Matching involving photometric information
            cat_magerr = cat[cat_col_mag_err] if cat_col_mag_err is not None else None
            m = photometry.match(obj['ra'], obj['dec'], obj['mag'], obj['magerr'], obj['flags'], cat[cat_col_ra], cat[cat_col_dec], cat[cat_col_mag], cat_magerr=cat_magerr, sr=sr)
            if not m or np.sum(m['idx']) < min_matches:
                log('Too few (%d) good photometric matches, cannot refine WCS' % np.sum(m['idx']))
                return None
            else:
                log('Iteration %d: %d matches, %.1f arcsec rms' %
                    (iter, np.sum(m['idx']), np.std(3600*m['dist'][m['idx']])))

            wcs = astrometry.refine_wcs(obj[m['oidx']][m['idx']], cat[m['cidx']][m['idx']], order=order, match=False, method=method)
        else:
            # Simple positional matching
            wcs = astrometry.refine_wcs(obj, cat, order=order, sr=sr, match=True, method=method)

        if update:
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)


    return wcs

def filter_transient_candidates(obj, sr=None, pixscale=None, time=None,
                                cat=None, cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                                vizier=['ps1', 'usnob1', 'gsc'], skybot=True, ned=False, flagged=True,
                                col_id=None, get_candidates=True, verbose=False):
    """
    Higher-level transient candidate filtering routine.
    """
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

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

    h = htm.HTM(10)

    log('Candidate filtering routine started with %d initial candidates and %.1f arcsec matching radius' % (len(obj), sr*3600))
    cand_idx = np.ones(len(obj), dtype=np.bool)

    if flagged:
        # Filter out flagged objects (saturated, cosmics, blends, etc)
        cand_idx &= obj['flags'] == 0
        print(np.sum(cand_idx), 'of them are unflagged')

    if cat is not None and np.any(cand_idx):
        m = h.match(obj['ra'], obj['dec'], cat[cat_col_ra], cat[cat_col_dec], sr)
        cand_idx[m[0]] = False
        log(np.sum(cand_idx), 'of them are not matched with reference catalogue')

    for catname in vizier:
        if not np.any(cand_idx):
            break

        xcat = catalogs.xmatch_objects(obj[cand_idx], catname, sr)
        if xcat is not None and len(xcat):
            cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])

        log(np.sum(cand_idx), 'remains after matching with', catalogs.catalogs.get(catname, {'name':catname})['name'])

    if skybot and np.any(cand_idx):
        if time is None and 'time' in obj.keys():
            time = obj['time']

        if time is not None:
            xcat = catalogs.xmatch_skybot(obj[cand_idx], time=time, col_id=col_id)
            if xcat is not None and len(xcat):
                cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])
            log(np.sum(cand_idx), 'remains after matching with SkyBot')

    if ned and np.any(cand_idx):
        xcat = catalogs.xmatch_ned(obj[cand_idx], sr, col_id=col_id)
        if xcat is not None and len(xcat):
            cand_idx &= ~np.in1d(obj[col_id], xcat[col_id])
        log(np.sum(cand_idx), 'remains after matching with NED')

    log('%d candidates remaining after filtering' % len(obj[cand_idx]))

    if get_candidates:
        return obj_in[cand_idx].copy()
    else:
        return cand_idx

def calibrate_photometry(obj, cat, sr=None, pixscale=None, order=0,
                         obj_col_mag='mag', obj_col_mag_err='magerr',
                         cat_col_mag='R', cat_col_mag_err=None,
                         cat_col_mag1=None, cat_col_mag2=None,
                         cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                         update=True, verbose=False, **kwargs):
    """
    Higher-level photometric calibration routine
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

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
                         obj_x=obj['x'], obj_y=obj['y'], spatial_order=order,
                         verbose=verbose, **kwargs)

    if m:
        log('Photometric calibration finished successfully.')
        if m['color_term']:
            log('Color term is %.2f' % m['color_term'])

        if update:
            obj['mag_calib'] = obj['mag'] + m['zero_fn'](obj['x'], obj['y'])
    else:
        log('Photometric calibration failed')

    return m

def place_random_stars(image, psf_model, nstars=100, minflux=1, maxflux=100000, gain=1, saturation=65535, wcs=None, verbose=False):
    """
    Randomly place artificial stars into the image.
    Coordinates are distributed uniformly.
    Fluxes are log-uniform between user-provided min and max values.

    Returns: the catalogue of generated stars, with x, y and flux fields set.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    cat = {
        'x': np.random.uniform(0, image.shape[1] - 1, nstars),
        'y': np.random.uniform(0, image.shape[0] - 1, nstars),
        'flux': 10**np.random.uniform(np.log10(minflux), np.log10(maxflux), nstars)
    }
    cat = Table(cat)

    if wcs is not None and wcs.celestial:
        cat['ra'], cat['dec'] = wcs.all_pix2world(cat['x'], cat['y'], 0)
    else:
        cat['ra'], cat['dec'] = np.nan, np.nan

    cat['mag'] = -2.5*np.log10(cat['flux'])

    for _ in cat:
        psf.place_psf_stamp(image, psf_model, _['x'], _['y'], flux=_['flux'], gain=gain)

    if saturation is not None:
        image[image > saturation] = saturation

    return cat
