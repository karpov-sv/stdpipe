from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

from . import utils

def crop_image(data, x0, y0, r0, header=None):
    x1,x2 = int(np.floor(x0 - r0)), int(np.ceil(x0 + r0))
    y1,y2 = int(np.floor(y0 - r0)), int(np.ceil(y0 + r0))

    src = [min(max(y1, 0), data.shape[0]),
           max(min(y2, data.shape[0]), 0),
           min(max(x1, 0), data.shape[1]),
           max(min(x2, data.shape[1]), 0)]

    dst = [src[0] - y1, src[1] - y1, src[2] - x1, src[3] - x1]

    sub = np.zeros((y2-y1, x2-x1), data.dtype)
    sub.fill(np.nan)
    sub[dst[0]:dst[1], dst[2]:dst[3]] = data[src[0]:src[1], src[2]:src[3]]

    if header is not None:
        subheader = header.copy()

        subheader['NAXIS1'] = sub.shape[1]
        subheader['NAXIS2'] = sub.shape[0]

        # Adjust the WCS keywords if present
        if 'CRPIX1' in subheader and 'CRPIX2' in subheader:
            subheader['CRPIX1'] -= x1
            subheader['CRPIX2'] -= y1

        # FIXME: should we use 0-based or 1-based coordinates here?..

        # Crop target inside cutout
        subheader['CROP_X'] = x0 - x1
        subheader['CROP_Y'] = y0 - y1

        # Crop center inside original frame
        subheader['CROP_X0'] = x0
        subheader['CROP_Y0'] = y0
        subheader['CROP_R0'] = r0

        # Crop position inside original frame
        subheader['CROP_X1'] = x1
        subheader['CROP_X2'] = x2
        subheader['CROP_Y1'] = y1
        subheader['CROP_Y2'] = y2

        return sub, subheader
    else:
        return sub

def get_cutout(image, candidate, size, mask=None, bg=None, diff=None, template=None, err=None, header=None, time=None, filename=None):
    x0, y0 = candidate['x'], candidate['y']

    _ = crop_image(image, x0, y0, size, header=header)
    if header is not None:
        crop,crophead = _
    else:
        crop,crophead = _,None

    cutout = {'image': crop}

    for _ in ['x', 'y', 'ra', 'dec', 'mag', 'magerr', 'mag_calib', 'flags', 'id', 'filename', 'time']:
        if _ in candidate.colnames:
            cutout[_] = candidate[_]

    cutout['name'] = utils.make_jname(candidate['ra'], candidate['dec'])

    if crophead is not None:
        wcs = WCS(crophead)
        cutout['header'] = crophead
        cutout['wcs'] = wcs

    if mask is not None:
        cutout['mask'] = crop_image(mask, x0, y0, size)

    if bg is not None:
        cutout['bg'] = crop_image(bg, x0, y0, size)

    if diff is not None:
        cutout['diff'] = crop_image(diff, x0, y0, size)

    if template is not None:
        cutout['template'] = crop_image(template, x0, y0, size)

    if err is not None:
        cutout['err'] = crop_image(err, x0, y0, size)

    if not 'time' in cutout and time is not None:
        cutout['time'] = Time(time)

    if not 'filename' in cutout and filename is not None:
        cutout['filename'] = filename

    return cutout

def write_cutout(cutout, filename):
    hdus = []

    # Store metadata to primary header
    hdu = fits.PrimaryHDU()

    for _ in ['x', 'y', 'ra', 'dec', 'mag', 'magerr', 'mag_calib', 'flags', 'id', 'time', 'filename']:
        if _ in cutout:
            data = cutout[_]
            # Special handling for unsupported FITS types
            if _ == 'time':
                data = Time(data).to_value('fits')

            hdu.header[_] = data

    hdus.append(hdu)

    # Store imaging data to named extensions
    for _ in ['image', 'template', 'diff', 'mask', 'err', 'bg']:
        if _ in cutout:
            data = cutout[_]

            if data.dtype == np.bool:
                data = data.astype(np.uint16)

            hdu = fits.ImageHDU(data, header=cutout.get('header'), name=_)
            hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=True)

def load_cutout(filename):
    hdus = fits.open(filename)

    cutout = {}
    for _ in ['x', 'y', 'ra', 'dec', 'mag', 'magerr', 'mag_calib', 'flags', 'id', 'filename', 'time']:
        if _ in hdus[0].header:
            data = hdus[0].header[_]

            if _ == 'time':
                data = Time(data)

            cutout[_] = data

    for hdu in hdus[1:]:
        if 'header' not in cutout:
            cutout['header'] = hdu.header

        cutout[hdu.name.lower()] = hdu.data

    hdus.close()

    return cutout
