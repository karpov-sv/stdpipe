from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import os
import json
import tempfile

from urllib.parse import urlencode

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

from esutil import htm

# from scipy.ndimage import binary_dilation
from astropy.convolution import Tophat2DKernel, convolve, convolve_fft

from . import utils
from . import photometry
from . import pipeline
from . import astrometry

# HiPS images

def get_hips_image(hips, ra=None, dec=None, width=None, height=None, fov=None,
                   wcs=None, header=None,
                   asinh=None, normalize=True, upscale=False,
                   get_header=True, verbose=False):
    """
    Load the image from any HiPS survey using CDS hips2fits service.

    The image scale and orientation may be specified by either center coordinates and fov,
    or by directly passing WCS solution or FITS header containing it.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if header is not None:
        wcs = WCS(header)
        width = header['NAXIS1']
        height = header['NAXIS2']

    params = {
        'hips': hips,
        'width': int(width),
        'height': int(height),
        'coordsys': 'icrs',
        'format': 'fits'
    }

    if wcs is not None and wcs.is_celestial:
        # Whether we should upscale the template or not
        if upscale is True:
            # Let's guess the upscaling that will drive pixel scale beyond 1 arcsec/pix
            pixscale = astrometry.get_pixscale(wcs=wcs)*3600
            upscale = int(np.ceil(pixscale / 1))

            if upscale > 1:
                log('Will upscale the image from %.2f to %.2f arcsec/pix' % (pixscale, pixscale/upscale))

        if upscale and upscale > 1:
            wcs = astrometry.upscale_wcs(wcs, upscale, will_rebin=True)
            params['width'] *= upscale
            params['height'] *= upscale

        whdr = wcs.to_header(relax=True)
        if whdr['CTYPE1'] == 'RA---TPV' and 'PV1_0' not in whdr.keys():
            # hips2fits does not like such headers, let's fix it
            whdr['CTYPE1'] = 'RA---TAN'
            whdr['CTYPE2'] = 'DEC--TAN'

        params['wcs'] = json.dumps(dict(whdr))
    elif ra is not None and dec is not None and fov is not None:
        params['ra'] = ra
        params['dec'] = dec
        params['fov'] = fov
    else:
        log('Sky position and size are not provided')
        return None,None

    if width is None or height is None:
        log('Frame size is not provided')
        return None,None

    url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?' + urlencode(params)

    hdu = fits.open(url)

    image = hdu[0].data
    header = hdu[0].header

    # FIXME: hips2fits does not properly return the header for SIP WCS, so we just copy the original WCS over it
    if wcs is not None and wcs.sip is not None:
        wheader = wcs.to_header(relax=True)
        header.update(wheader)

    hdu.close()

    if asinh is None:
        if 'PanSTARRS' in hips:
            asinh = True

    if asinh:
        # Fix asinh flux scaling
        image = np.sinh(image*np.log(10)/2.5)

    if upscale and upscale > 1:
        # We should do downscaling after conversion of the image back to linear flux scaling
        image = utils.rebin_image(image, upscale)

    if normalize:
        # Normalize the image to have median=100 and std=10, corresponding to GAIN=1 assuming Poissonian background
        image -= np.nanmedian(image)
        image *= 10/mad_std(image, ignore_nan=True)
        image += 100

    if get_header:
        return image, header
    else:
        return image

def dilate_mask(mask, dilate=5):
    """
    Dilate binary mask with a given kernel size
    """

    kernel = Tophat2DKernel(dilate).array
    # mask = binary_dilation(mask, kernel)
    if dilate < 10 or True: # it seems convolve is faster than convolve_fft even for 2k x 2k
        mask = convolve(mask, kernel)
    else:
        mask = convolve_fft(mask, kernel)
    mask = mask > 1e-15*np.max(mask) # FIXME: is it correct threshold?..

    return mask

def mask_template(tmpl, cat=None, cat_saturation_mag=None,
                  cat_col_mag='rmag', cat_col_mag_err='e_rmag',
                  cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_sr=1/3600,
                  mask_nans=True, mask_masked=True,
                  mask_photometric=False, aper=2, sn=5,
                  wcs=None, dilate=5, verbose=False, _tmpdir=None):
    """
    Apply various masking heuristics (NaNs, saturated catalogue stars, etc) to the template.

    If `mask_nans` is set, it masks NaN pixels

    If `mask_masked` is set and catalogue is provided, it masks all catalogue stars where
    `cat_col_mag` or `cat_col_mag_err` fields are either masked or NaNs

    If `mask_photometric` is set and catalogue is provided, it detects the objects on the template,
    matches them photometrically, and then rejects all the stars that are not used in photometric fit
    and fainter than the catalogue entries - supposedly, the saturated stars.

    The mask then may be optionally dilated if dilation size is set in `dilate` argument.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if mask_nans:
        tmask = ~np.isfinite(tmpl)
        log(np.sum(tmask), 'template pixels masked after NaN checking')
    else:
        tmask = np.zeros_like(tmpl, dtype=np.bool)

    if cat is not None and wcs is not None:
        # Mask the central pixels of saturated stars
        cx, cy = wcs.all_world2pix(cat[cat_col_ra], cat[cat_col_dec], 0)
        cx = np.round(cx).astype(np.int)
        cy = np.round(cy).astype(np.int)

        tidx = np.zeros(len(cat), dtype=np.bool)

        # First, we select all catalogue objects with masked measurements
        if mask_masked:
            tidx = cat[cat_col_mag].mask == True
            tidx |= ~np.isfinite(cat[cat_col_mag])
            if cat_col_mag_err:
                tidx |= cat[cat_col_mag_err].mask == True
                tidx |= ~np.isfinite(cat[cat_col_mag_err])

        # Next, we also add the ones corresponding to saturation limit
        if cat_saturation_mag is not None:
             tidx |= cat[cat_col_mag] < cat_saturation_mag

        # Also, we may mask photometrically saturated stars
        if mask_photometric:
            # Detect the stars on the template and match with the catalogue
            tobj = photometry.get_objects_sextractor(tmpl, mask=tmask, sn=sn, aper=aper, wcs=wcs, _tmpdir=_tmpdir)
            # tobj = photometry.measure_objects(tobj, tmpl, mask=tmask, fwhm=np.median(tobj['fwhm'][tobj['flags'] == 0]), aper=1)

            tm = pipeline.calibrate_photometry(tobj, cat, sr=cat_sr,
                                               cat_col_mag=cat_col_mag, cat_col_mag_err=cat_col_mag_err,
                                               cat_col_ra=cat_col_ra, cat_col_dec=cat_col_dec,
                                               order=0, accept_flags=0x02, robust=True, scale_noise=True)

            idx = ~tm['idx'] & (tm['zero']-tm['zero_model'] < -0.1)
            tidx[tm['cidx'][idx]] = True

        # ..and keep only the ones inside the image
        tidx &= (cx >= 0) & (cx <= tmpl.shape[1] - 1)
        tidx &= (cy >= 0) & (cy <= tmpl.shape[0] - 1)

        tmask[cy[tidx], cx[tidx]] = True

        log(np.sum(tmask), 'template pixels masked after checking saturated (%s < %.1f) stars' % (cat_col_mag, cat_saturation_mag))

    if dilate and dilate > 0:
        log('Dilating the mask with %d x %d kernel' % (dilate, dilate))
        tmask = dilate_mask(tmask, dilate)
        log(np.sum(tmask), 'template pixels masked after dilation')

    return tmask

# PanSTARRS images

__skycells = None

def find_ps1_skycells(ra, dec, sr, band='r', cell_radius=0.3, fullpath=True):
    global __skycells

    if __skycells is None:
        # Load skycells information and store to global variable
        __skycells = Table.read(utils.get_data_path('ps1skycells.txt'), format='ascii')

    # FIXME: here we may select the cells that are too far from actual footprint
    h = htm.HTM(10)
    _,idx,_ = h.match(ra, dec, __skycells['ra0'], __skycells['dec0'], sr + cell_radius, maxmatch=0)

    if fullpath:
        # Get full path on the server
        return ['rings.v3.skycell/%04d/%03d/rings.v3.skycell.%04d.%03d.stk.%s.unconv.fits' % (_['projectionID'], _['skyCellID'], _['projectionID'], _['skyCellID'], band) for _ in __skycells[idx]]
    else:
        # Get just the file name
        return ['rings.v3.skycell.%04d.%03d.stk.%s.unconv.fits' % (_['projectionID'], _['skyCellID'], band) for _ in __skycells[idx]]

def get_ps1_skycells(ra0, dec0, sr0, band='r', _cachedir='~/.stdpipe-cache/ps1/', normalize=True, verbose=False):
    """
    Get the list of filenames corresponding to skycells in the user-specified sky region.
    The cells are downloaded and stored to the specified cache location.
    Downloaded skycells are (optionally) normalized according to the parameters from their FITS headers.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Normalize _cachedir
    if _cachedir is not None:
        _cachedir = os.path.expanduser(_cachedir)
    else:
        _cachedir = tempfile.gettempdir()
        log('Cache location not specified, falling back to %s', _cachedir)

    filenames = []

    cells = find_ps1_skycells(ra0, dec0, sr0, band=band, fullpath=True)

    for cell in cells:
        cellname = os.path.basename(cell)
        filename = os.path.join(_cachedir, cellname)

        if os.path.exists(filename):
            log('%s already downloaded' % cellname)
        else:
            log('Downloading %s' % cellname)

            url = 'http://ps1images.stsci.edu/' + cell

            if utils.download(url, filename, verbose=verbose):
                if normalize:
                    normalize_ps1_skycell(filename, verbose=verbose)

        if os.path.exists(filename):
            filenames.append(filename)

    return filenames

def normalize_ps1_skycell(filename, verbose=False):
    """
    Normalize PanSTARRS skycell file according to its FITS header
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    header = fits.getheader(filename, -1)

    if 'RADESYS' not in header and 'PC001001' in header:
        # Normalize WCS in the header
        log('Normalizing WCS keywords in %s' % filename)

        header['RADESYS'] = 'FK5'
        header.rename_keyword('PC001001', 'PC1_1')
        header.rename_keyword('PC001002', 'PC1_2')
        header.rename_keyword('PC002001', 'PC2_1')
        header.rename_keyword('PC002002', 'PC2_2')

        data = fits.getdata(filename, -1)

        if 'BSOFTEN' in header and 'BOFFSET' in header:
            # Linearize ASINH scaling
            log('Normalizing ASINH scaling in %s' % filename)

            x = data * 0.4 * np.log(10)
            data = header['BOFFSET'] + header['BSOFTEN'] * (np.exp(x) - np.exp(-x))

            for _ in ['BZERO', 'BSCALE', 'BSOFTEN', 'BOFFSET', 'BLANK']:
                header.remove(_, ignore_missing=True)

        log('Writing data back to %s', filename)
        fits.writeto(filename, data, header, overwrite=True)
