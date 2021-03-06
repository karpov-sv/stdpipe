from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import json

from urllib.parse import urlencode

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

from esutil import htm

from . import utils

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

# PanSTARRS images

__skycells = None

def find_ps1_skycells(ra, dec, sr, band='r', sr0=0.3, fullpath=True):
    global __skycells

    if __skycells is None:
        print('Loading skycells information')
        __skycells = Table.read(utils.get_data_path('ps1skycells.txt'), format='ascii')

    h = htm.HTM(10)
    _,idx,_ = h.match(ra, dec, __skycells['ra0'], __skycells['dec0'], sr + sr0, maxmatch=0)

    if fullpath:
        # Get full path on the server
        return ['rings.v3.skycell/%04d/%03d/rings.v3.skycell.%04d.%03d.stk.%s.unconv.fits' % (_['projectionID'], _['skyCellID'], _['projectionID'], _['skyCellID'], band) for _ in __skycells[idx]]
    else:
        # Get just the file name
        return ['rings.v3.skycell.%04d.%03d.stk.%s.unconv.fits' % (_['projectionID'], _['skyCellID'], band) for _ in __skycells[idx]]
