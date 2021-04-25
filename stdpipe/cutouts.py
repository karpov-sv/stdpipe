from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import datetime

from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift

from . import utils

def crop_image_centered(data, x0, y0, r0, header=None):
    """
    Crops the image to keep only the region with a given radius around the position.
    Also adjusts the FITS header, if provided, to shift the origin of WCS solution
    so that it is still valid for the cutout.

    The size of image is 2*ceil(r0) + 1.
    The original center is inside the pixel at x,y = ceil(r0),ceil(r0)
    """

    # x1,x2 = int(np.floor(x0 - r0)), int(np.ceil(x0 + r0))
    # y1,y2 = int(np.floor(y0 - r0)), int(np.ceil(y0 + r0))
    x1,x2 = int(np.round(x0) - np.ceil(r0)), int(np.round(x0) + np.ceil(r0))
    y1,y2 = int(np.round(y0) - np.ceil(r0)), int(np.round(y0) + np.ceil(r0))

    return crop_image(data, x1, y1, x2 - x1 + 1, y2 - y1 + 1, header=header)

def crop_image(data, x1, y1, width, height, header=None):
    """
    Crops the image to keep only the region with given origin and dimensions.
    Also adjusts the FITS header, if provided, to shift the origin of WCS solution
    so that it is still valid for the cutout.
    """

    x2 = x1 + width
    y2 = y1 + height

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

        # Crop position inside original frame
        subheader['CROP_X1'] = x1
        subheader['CROP_X2'] = x2
        subheader['CROP_Y1'] = y1
        subheader['CROP_Y2'] = y2

        return sub, subheader
    else:
        return sub

def get_cutout(image, candidate, radius, mask=None, background=None, diff=None, template=None, convolved=None, err=None, header=None, time=None, filename=None, name=None):
    x0, y0 = candidate['x'], candidate['y']

    _ = crop_image_centered(image, x0, y0, radius, header=header)
    if header is not None:
        crop,crophead = _
    else:
        crop,crophead = _,None

    cutout = {'image': crop, 'meta': {}}

    if crophead is not None:
        wcs = WCS(crophead)
        cutout['header'] = crophead
        cutout['wcs'] = wcs

    # Image planes
    if mask is not None:
        cutout['mask'] = crop_image_centered(mask, x0, y0, radius)

    if background is not None:
        cutout['background'] = crop_image_centered(background, x0, y0, radius)

    if diff is not None:
        cutout['diff'] = crop_image_centered(diff, x0, y0, radius)

    if template is not None:
        cutout['template'] = crop_image_centered(template, x0, y0, radius)

    if convolved is not None:
        cutout['convolved'] = crop_image_centered(convolved, x0, y0, radius)

    if err is not None:
        cutout['err'] = crop_image_centered(err, x0, y0, radius)

    # Metadata
    for _ in candidate.colnames:
        cutout['meta'][_] = candidate[_]

    # Additional metadata to add or override
    if time is not None:
        cutout['meta']['time'] = Time(time)

    if filename is not None:
        cutout['meta']['filename'] = filename

    if name is not None:
        cutout['meta']['name'] = name
    elif 'name' not in cutout['meta']:
        cutout['meta']['name'] = utils.make_jname(candidate['ra'], candidate['dec'])

    return cutout

def write_cutout(cutout, filename):
    hdus = []

    # Store metadata to primary header
    hdu = fits.PrimaryHDU()

    for _ in cutout['meta']:
        data = cutout['meta'][_]
        # Special handling for unsupported FITS types
        if type(data)  == Time or type(data) == datetime.datetime:
            data = Time(data).to_value('fits')

        hdu.header[_] = data

    for _ in ['x', 'y', 'ra', 'dec', 'mag', 'magerr', 'mag_calib', 'flags', 'id', 'time', 'filename']:
        if _ in cutout:
            data = cutout[_]
            # Special handling for unsupported FITS types
            if _ == 'time':
                data = Time(data).to_value('fits')

            hdu.header[_] = data

    hdus.append(hdu)

    # Store imaging data to named extensions
    for _ in ['image', 'template', 'convolved', 'diff', 'adjusted', 'mask', 'err', 'background']:
        if _ in cutout:
            data = cutout[_]

            if data.dtype == np.bool:
                data = data.astype(np.uint16)

            hdu = fits.ImageHDU(data, header=cutout.get('header'), name=_)
            hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=True)

def load_cutout(filename):
    hdus = fits.open(filename)

    cutout = {'meta': {}}

    for _ in hdus[0].header[4:]:
        name = _.lower()
        data = hdus[0].header[_]

        if name == 'time':
            data = Time(data)

        cutout['meta'][name] = data

    for hdu in hdus[1:]:
        if 'header' not in cutout:
            cutout['header'] = hdu.header

        cutout[hdu.name.lower()] = hdu.data

    hdus.close()

    return cutout

def adjust_cutout(cutout, max_shift=2, bg=None, verbose=False):
    """
    Try to apply some positional adjustment to the cutout in order to minimize the difference.
    It will add one more image plane,
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    if bg is None:
        bg = np.median(cutout['image'])

    def _fn(dx):
        # TODO: only fit central part of cutout
        return np.std((cutout['image'] - bg - shift(cutout['convolved'], dx, mode='reflect'))/cutout['err'])

    res = minimize(_fn, (0, 0), bounds=((-max_shift, max_shift), (-max_shift, max_shift)), method='Powell', options={'disp':False})

    log(res.message)

    if res.success:
        log('Adjustment is: %.2f %.2f' % (res.x[0], res.x[1]))
        log('RMS improvement: %.2f -> %.2f' % (_fn([0, 0]), _fn(res.x)))

        cutout['adjusted'] = cutout['image'] - bg - shift(cutout['convolved'], res.x, mode='reflect')
