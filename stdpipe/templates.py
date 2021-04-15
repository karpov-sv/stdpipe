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

from scipy.ndimage import binary_dilation
from astropy.convolution import Tophat2DKernel

from . import utils
from . import photometry

# HiPS images

def get_hips_image(hips, ra=None, dec=None, width=None, height=None, fov=None, wcs=None, header=None, asinh=None, normalize=True, get_header=True):
    if header is not None:
        wcs = WCS(header)
        width = header['NAXIS1']
        height = header['NAXIS2']

    params = {
        'hips': hips,
        'width': width,
        'height': height,
        'coordsys': 'icrs',
        'format': 'fits'
    }

    if wcs is not None and wcs.is_celestial:
        params['wcs'] = json.dumps(dict(wcs.to_header(relax=True)))
    elif ra is not None and dec is not None and fov is not None:
        params['ra'] = ra
        params['dec'] = dec
        params['fov'] = fov
    else:
        print('Sky position and size are not provided')
        return None,None

    if width is None or height is None:
        print('Frame size is not provided')
        return None,None

    url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?' + urlencode(params)

    hdu = fits.open(url)

    image = hdu[0].data
    header = hdu[0].header

    hdu.close()

    if asinh is None:
        if 'PanSTARRS' in hips:
            asinh = True

    if asinh:
        # Fix asinh flux scaling
        image = np.sinh(image*np.log(10)/2.5)

    if normalize:
        # Normalize the image to have median=100 and std=10, corresponding to GAIN=1 assuming Poissonian background
        image -= np.nanmedian(image)
        image *= 10/mad_std(image, ignore_nan=True)
        image += 100

    if get_header:
        return image, header
    else:
        return image

def mask_template(tmpl, cat=None, cat_saturation_mag=None,
                  cat_col_mag='rmag', cat_col_mag_err='e_rmag',
                  cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_sr=1/3600,
                  wcs=None, dilate=5, verbose=False):
    """
    Apply various masking heuristics (NaNs, saturated catalogue stars, etc) to the template.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    tmask = ~np.isfinite(tmpl)
    log(np.sum(tmask), 'template pixels masked after NaN checking')

    if cat is not None:
        if cat_saturation_mag is None:
            # Apply photometric match to guess catalogue saturation magnitude
            tobj = photometry.get_objects_sextractor(tmpl, mask=tmask, sn=3, wcs=wcs, aper=3)

            m = photometry.match(tobj['ra'], tobj['dec'], tobj['mag'], tobj['magerr'], tobj['flags'],
                                 cat[cat_col_ra], cat[cat_col_dec], cat[cat_col_mag],
                                 cat_magerr=cat[cat_col_mag_err], cat_saturation=10,
                                 sr=cat_sr, verbose=False)
            if m:
                cat_saturation_mag = np.min(cat[m['cidx']][m['idx']][cat_col_mag])
            else:
                log('Catalogue matching failed, cannot determine saturation level')
                cat_saturation_mag = 10

            log('Catalogue saturates at %s = %.1f' % (cat_col_mag, cat_saturation_mag))

        # Mask the central pixels of saturated stars
        cx, cy = wcs.all_world2pix(cat['RAJ2000'], cat['DEJ2000'], 0)
        cx = np.round(cx).astype(np.int)
        cy = np.round(cy).astype(np.int)

        # First, we select all catalogue objects with masked measurements
        tidx = cat[cat_col_mag].mask == True
        if cat_col_mag_err:
            tidx |= cat[cat_col_mag_err].mask == True

        # Next, we also add the ones corresponding to saturation limit
        tidx |= cat[cat_col_mag] < cat_saturation_mag

        # ..and keep only the ones inside the image
        tidx &= (cx >= 0) & (cx <= tmpl.shape[1] - 1)
        tidx &= (cy >= 0) & (cy <= tmpl.shape[0] - 1)

        tmask[cy[tidx], cx[tidx]] = True

        log(np.sum(tmask), 'template pixels masked after checking saturated (%s < %.1f) stars' % (cat_col_mag, cat_saturation_mag))

    if dilate and dilate > 0:
        log('Dilating the mask with %d x %d kernel' % (dilate, dilate))
        kernel = Tophat2DKernel(dilate).array
        tmask = binary_dilation(tmask, kernel)

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
    log = print if verbose else lambda *args,**kwargs: None

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
    log = print if verbose else lambda *args,**kwargs: None

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
