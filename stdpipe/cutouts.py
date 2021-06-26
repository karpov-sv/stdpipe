from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import datetime

from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

from scipy.stats import chi2

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

def get_cutout(image, candidate, radius, mask=None, background=None, diff=None, template=None, convolved=None, err=None, footprint=None, header=None, time=None, filename=None, name=None):
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

    if footprint is not None:
        cutout['footprint'] = crop_image_centered(footprint, x0, y0, radius)

    # Metadata
    for _ in candidate.keys():
        cutout['meta'][_] = candidate[_]

    # Additional metadata to add or override
    if time is not None:
        cutout['meta']['time'] = Time(time)

    if filename is not None:
        cutout['meta']['filename'] = filename

    if name is not None:
        cutout['meta']['name'] = name
    elif 'name' not in cutout['meta'] and 'ra' in cutout['meta'].keys() and 'dec' in cutout['meta'].keys():
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
        elif np.isreal(data) and np.isnan(data):
            data = 'NaN'

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
    for _ in ['image', 'template', 'convolved', 'diff', 'adjusted', 'mask', 'err', 'background', 'footprint']:
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
        elif data == 'NaN':
            data = np.nan

        cutout['meta'][name] = data

    for hdu in hdus[1:]:
        if 'header' not in cutout:
            cutout['header'] = hdu.header

        cutout[hdu.name.lower()] = hdu.data

    hdus.close()

    return cutout

def adjust_cutout(cutout, max_shift=2, max_scale=1.1, inner=None, normalize=False, fit_bg=False, verbose=False):
    """
    Try to apply some positional adjustment to the cutout in order to minimize the difference.
    It will add one more image plane, 'adjusted', with the minimized difference between the original image and convolved template.

    If normalize=True, the adjusted image will be divided by the errors.

    If inner is set to an integer value, only the central box with this size will be used for minimization.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    mask = cutout['mask'] if 'mask' in cutout else ~np.isfinite(cutout['image'])
    imask = np.zeros_like(mask)

    # Rough estimation of backgrounds in the image and the template, using SExtractor-like mode estimation
    # bg = np.nanmedian(cutout['image'][~mask])
    bg = 2.5*np.nanmedian(cutout['image'][~mask]) - 1.5*np.nanmean(cutout['image'][~mask])
    # tbg = np.nanmedian(cutout['convolved'][~mask])
    tbg = 2.5*np.nanmedian(cutout['convolved'][~mask]) - 1.5*np.nanmean(cutout['convolved'][~mask])

    if inner is not None and inner > 0:
        # Mask everything outside of a central box with given size
        x,y = np.mgrid[0:mask.shape[1], 0:mask.shape[0]]
        idx = np.abs(x - mask.shape[1]/2 + 0.5) > inner/2
        idx |= np.abs(y - mask.shape[0]/2 + 0.5) > inner/2
        imask[idx] = True

    # Prepare and cleanup the arrays we will use for fitting (as 'shift' does not like nans etc)
    image = cutout['image'].copy()
    tmpl = cutout['convolved'].copy()
    err = cutout['err'].copy()

    image[~np.isfinite(image)] = 0
    tmpl[~np.isfinite(tmpl)] = 0
    err[~np.isfinite(err)] = 0

    def _fn(dx, get_df=False, get_diff=False):
        mask1 = mask | shift(mask, dx[:2], mode='reflect')
        imask1 = mask1 | imask
        diff = image - dx[3] - shift(tmpl - dx[4], dx[:2], mode='reflect')*dx[2]

        chi2_value = np.sum((diff[~imask1]/err[~imask1])**2) # Chi2

        if get_diff:
            return diff,mask1,imask1
        elif get_df:
            return chi2_value, np.sum(~imask1)
        else:
            return chi2_value

    if fit_bg:
        res = minimize(_fn, (0, 0, 1, bg, tbg), bounds=((-max_shift, max_shift), (-max_shift, max_shift), (1/max_scale, max_scale), (None, None), (None, None)), method='Powell', options={'disp':False})
    else:
        res = minimize(_fn, (0, 0, 1, bg, tbg), bounds=((-max_shift, max_shift), (-max_shift, max_shift), (1/max_scale, max_scale), (bg, bg), (tbg, tbg)), method='Powell', options={'disp':False})

    log(res.message)

    if res.success:
        log('Adjustment is: %.2f %.2f bg %.2g tbg %.2g scale %.2f' % (res.x[0], res.x[1], res.x[3], res.x[4], res.x[2]))
        log('Chi2 improvement: %.2f -> %.2f' % (_fn([0, 0, 1, bg, tbg]), _fn(res.x)))

        diff,mask1,_ = _fn(res.x, get_diff=True)
        chi2_0 = _fn([0, 0, 1, bg, tbg])
        chi2_1,df = _fn(res.x, get_df=True)
        log('Final Chi2: %.2f df: %d p-value: %.2g' % (chi2_1, df, chi2.sf(chi2_1, df)))

        # Keep the adjustment results in the metadata
        cutout['meta']['adjust_chi2_0'] = chi2_0
        cutout['meta']['adjust_chi2'] = chi2_1
        cutout['meta']['adjust_df'] = df
        cutout['meta']['adjust_pval'] = chi2.sf(chi2_1, df)
        cutout['meta']['adjust_dx'] = res.x[0]
        cutout['meta']['adjust_dy'] = res.x[1]
        cutout['meta']['adjust_scale'] = res.x[2]
        cutout['meta']['adjust_bg'] = res.x[3]
        cutout['meta']['adjust_tbg'] = res.x[4]

        if normalize:
            diff[~mask1] /= err[~mask1]
            diff[err>1e-30] /= err[err>1e-30]

        # diff[mask1] = 1e-30

        # Add result as a new cutout plane
        cutout['adjusted'] = diff

        return True

    else:
        return False
