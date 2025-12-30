"""
Module for cropping the images and creating image cutouts / postage stamps
"""


import numpy as np
import datetime

from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

from scipy.stats import chi2

from scipy.optimize import minimize
from scipy.ndimage import shift

from . import utils
from . import astrometry


def crop_image_centered(data, x0, y0, r0, header=None):
    """
    Crops the image to keep only the region with a given radius around the position.
    Also adjusts the FITS header, if provided, to shift the origin of WCS solution
    so that it is still valid for the cutout.

    The size of image is :code:`2*ceil(r0) + 1`.
    The original center is inside the pixel at :code:`x,y = ceil(r0),ceil(r0)`
    """

    # x1,x2 = int(np.floor(x0 - r0)), int(np.ceil(x0 + r0))
    # y1,y2 = int(np.floor(y0 - r0)), int(np.ceil(y0 + r0))
    x1, x2 = int(np.round(x0) - np.ceil(r0)), int(np.round(x0) + np.ceil(r0))
    y1, y2 = int(np.round(y0) - np.ceil(r0)), int(np.round(y0) + np.ceil(r0))

    return crop_image(data, x1, y1, x2 - x1 + 1, y2 - y1 + 1, header=header)


def crop_image(data, x1, y1, width, height, header=None):
    """
    Crops the image to keep only the region with given origin and dimensions.
    Also adjusts the FITS header, if provided, to shift the origin of WCS solution
    so that it is still valid for the cutout.
    """

    x2 = x1 + width
    y2 = y1 + height

    src = [
        min(max(y1, 0), data.shape[0]),
        max(min(y2, data.shape[0]), 0),
        min(max(x1, 0), data.shape[1]),
        max(min(x2, data.shape[1]), 0),
    ]

    dst = [src[0] - y1, src[1] - y1, src[2] - x1, src[3] - x1]

    sub = np.zeros((y2 - y1, x2 - x1), data.dtype)
    if isinstance(data[0][0], np.floating):
        # For floating-point we may use NaN as a filler value
        sub.fill(np.nan)
    sub[dst[0] : dst[1], dst[2] : dst[3]] = data[src[0] : src[1], src[2] : src[3]]

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


def get_cutout(
    image,
    candidate,
    radius,
    header=None,
    wcs=None,
    time=None,
    filename=None,
    name=None,
    **kwargs
):
    """Create the cutout from one or more image planes based on the candidate object.

    The object may be either a row from :class:`astropy.table.Table`, or a dictionary with at least
    `x` and `y` keys present and containing the pixel coordinates of the objects in the image.
    The cutout will be centered so that object center is inside its central pixel, and its size will
    be :code:`2*ceil(radius) + 1` pixels.

    :param image: The science image for `image` plane of the cutout
    :param candidate: The candidate object that should at least contain `x` and `y` fields
    :param radius: Cutout radius in pixels. Cutout will have a square shape with :code:`2*ceil(radius) + 1` width
    :param header: The header for original image. It will be copied to `header` field of the cutout and modified accordingly to properly represent the shape and WCS of the cutout. Optional
    :param wcs: WCS for the original image. Will be copied to `wcs` field of the cutout with appropriate adjustment of the center so that it properly represents the astrometric solution on the cutout. Takes precedence over `header` parameter in defining cutout WCS. Optional
    :param time: Time (:class:`astropy.time.Time` or :class:`datetime.datetime` or a string representation compatible with them) of the original image acquisition, to be copied to `time` field of the cutout. Optional
    :param filename: Filename of the original image, to be stored in `filename` field of the cutout. Optional
    :param name: The object name, to be stored in the `name` field of the cutout. Optional
    :param \\**kwargs: All additional keyword arguments are interpreted as additional image planes (e.g. mask, diff, template etc).
    :returns: Cutout structure as described below.

    Cutout is represented by a dictionary with at least the following fields:

    - `image` - primary image plane, centered on the candidate position
    - `mask`, `background`, `diff`, `template`, `convolved`, `err`, `footprint`, etc - corresponding secondary image planes, if provided as parameters to the routine
    - `header` - header of the original image modified to properly represent the cutout WCS, if provided as a parameter to the routine
    - `wcs` - WCS solution for the original image, modified to properly represent the cutout WCS, if provided as a parameter to the routine
    - `meta` - dictionary of additional metadata for the cutout. It will be populated with all fields of the candidate object, plus additionally:

        - `time` - original image timestamp as :class:`astropy.time.Time` object, if provided
        - `filename` - Original image filename, if provided
        - `name` - object name. If not provided, and if the candidate has `ra` and `dec` fields set, the name will be automatically constructed in a JHHMMSS.SS+DDMMSS.S form

    """
    x0, y0 = candidate['x'], candidate['y']

    _ = crop_image_centered(image, x0, y0, radius, header=header)
    if header is not None:
        crop, crophead = _
    else:
        crop, crophead = _, None

    cutout = {'image': crop, 'meta': {}}

    if wcs is not None:
        cutout['wcs'] = wcs

        # Let's update header info with this WCS
        astrometry.clear_wcs(crophead, remove_underscored=True, remove_history=True)
        crophead += wcs.to_header(relax=True)

    if crophead is not None:
        cutout['header'] = crophead

        if wcs is None:
            wcs = WCS(crophead)
            cutout['wcs'] = wcs

    # Image planes
    for pname,plane in kwargs.items():
        if plane is not None and np.ndim(plane) == 2:
            cutout[pname] = crop_image_centered(plane, x0, y0, radius)

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
    elif (
        'name' not in cutout['meta']
        and 'ra' in cutout['meta'].keys()
        and 'dec' in cutout['meta'].keys()
    ):
        cutout['meta']['name'] = utils.make_jname(candidate['ra'], candidate['dec'])

    return cutout


def write_cutout(cutout, filename):
    """Store the cutout as a multi-extension FITS file.

    For every cutout plane, separate FITS extension with corresponding image name will be created.
    The rest of metadata will be stored as FITS keywords in the primary header.

    :param cutout: Cutout structure
    :param filename: Name of FITS file where to store the cutout

    """

    hdus = []

    # Store metadata to primary header
    hdu = fits.PrimaryHDU()

    for _ in cutout['meta']:
        data = cutout['meta'][_]
        # Special handling for unsupported FITS types
        if data is None:
            data = None
        elif type(data) == Time or type(data) == datetime.datetime:
            data = Time(data).to_value('fits')
        elif not np.isscalar(data):
            # NumPy arrays and like
            data = repr(np.ma.filled(data))
        elif np.isreal(data) and np.isnan(data):
            data = 'NaN'
        elif np.isreal(data) and not np.isfinite(data):
            data = 'Inf'

        hdu.header[_] = data

    for _ in [
        'x',
        'y',
        'ra',
        'dec',
        'mag',
        'magerr',
        'mag_calib',
        'flags',
        'id',
        'time',
        'filename',
    ]:
        if _ in cutout:
            data = cutout[_]
            # Special handling for unsupported FITS types
            if _ == 'time':
                data = Time(data).to_value('fits')

            hdu.header[_] = data

    hdus.append(hdu)

    # Store imaging data to named extensions
    for _ in cutout.keys():
        if _ not in ['meta', 'header', 'wcs'] and np.ndim(cutout[_]) == 2:
            data = cutout[_]

            if data.dtype == bool:
                data = data.astype(np.uint16)

            hdu = fits.ImageHDU(data, header=cutout.get('header'), name=_)
            hdus.append(hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=True)


def load_cutout(filename):
    """Restore the cutout from multi-extension FITS file written by :func:`stdpipe.cutouts.write_cutout`.

    :param filename: Name of FITS file containing the cutout
    :returns: Cutout structure

    """

    hdus = fits.open(filename)

    cutout = {'meta': {}}

    for _ in hdus[0].header[4:]:
        name = _.lower()
        data = hdus[0].header[_]

        if name == 'time':
            data = Time(data)
        elif data == 'NaN':
            data = np.nan
        elif data == 'Inf':
            data = np.inf
        elif isinstance(data, str) and data.startswith('array('):
            # FIXME: any safer way to convert it back to array?..
            array = np.array
            data = eval(data)

        cutout['meta'][name] = data

    for hdu in hdus[1:]:
        if 'header' not in cutout:
            cutout['header'] = hdu.header

        name = hdu.name.lower()

        cutout[name] = hdu.data

        # Special handling of mask plane
        if name == 'mask':
            cutout[name] = cutout[name].astype(bool)

    hdus.close()

    if cutout['header']:
        cutout['wcs'] = WCS(cutout['header'])

    return cutout


def adjust_cutout(
    cutout,
    max_shift=2,
    max_scale=1.1,
    inner=None,
    normalize=False,
    fit_bg=False,
    verbose=False,
):
    """Try to apply some positional and scaling adjustment to the cutout in order to minimize the difference between the science image and the template.

    It will add one more image plane, `adjusted`, with the optimized difference between the original image and convolved template.

    If :code:`normalize=True`, the adjusted image will be normalized by the noise model (`err` plane of the cutout).

    If `inner` is set to an integer value, only the central box with this size (in pixels) will be used for the optimization.

    The optimization parameters are bounded by the `max_shift` and `max_scale` parameters.

    If :code:`fit_bg=True`, the difference of image and convolved template background levels will be also fitted for as a free patameter. If not, for both planes the background level will be estimated using SExtractor *mode* algorithm (:code:`2.5*median - 1.5*mean` of unmasked regions) and subtracted prior to fitting using only shift and scale as free parameters.

    :param cutout: Cutout structure as returned by :func:`stdpipe.cutouts.get_cutout`
    :param max_shift: Upper bound for the possible positional adjustment in pixels. The shift will be limited to :code:`(-max_shift, max_shift)` range
    :param max_scale: Upper bound for the possible scaling adjustment. The scale will be limited to :code:`(1/max_scale, max_scale)` range.
    :param inner: If specified, only :code:`inner x inner` innermost pixels of the cutout will be used for the optimization
    :param normalize: Whether to normalize the resulting `adjusted` cutout layer to the noise model (`err` layer)
    :param fit_bg: Whether to include the difference in background levels between the science image and the template as a free parameter in the optimization
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: `True` if optimization succeeded, `False` if it failed.

    On success, an additional plane `adjusted` with optimized difference between the science image and the template will be added to the cutout. Also, the following fields will be added to cutout `meta` dictionary:

    - `adjust_chi2_0` - chi-squared statistics before the optimization, computed inside the `inner` region if requested, or over the whole cutout otherwise
    - `adjust_chi2` - chi-squared statistics after the optimization
    - `adjust_df` - number of degrees of freedom
    - `adjust_pval` - p-value corresponding to `adjust_chi2` and `adjust_df`
    - `adjust_dx` - positional adjustment applied in `x` direction
    - `adjust_dy` - positional adjustment applied in `y` direction
    - `adjust_scale` - scaling adjustment applied
    - `adjust_bg` - the value of science image background. Will be optimized if :code:`fit_bg=True`, or just estimated from the cutout prior to optimization
    - `adjust_tbg` - the value of template background. Will be optimized if :code:`fit_bg=True`, or just estimated from the cutout prior to optimization

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    mask = cutout['mask'] if 'mask' in cutout else ~np.isfinite(cutout['image'])
    imask = np.zeros_like(mask)

    # Rough estimation of backgrounds in the image and the template, using SExtractor-like mode estimation
    # bg = np.nanmedian(cutout['image'][~mask])
    bg = 2.5 * np.nanmedian(cutout['image'][~mask]) - 1.5 * np.nanmean(
        cutout['image'][~mask]
    )
    # tbg = np.nanmedian(cutout['convolved'][~mask])
    tbg = 2.5 * np.nanmedian(cutout['convolved'][~mask]) - 1.5 * np.nanmean(
        cutout['convolved'][~mask]
    )

    if inner is not None and inner > 0:
        # Mask everything outside of a central box with given size
        x, y = np.mgrid[0 : mask.shape[1], 0 : mask.shape[0]]
        idx = np.abs(x - mask.shape[1] / 2 + 0.5) > inner / 2
        idx |= np.abs(y - mask.shape[0] / 2 + 0.5) > inner / 2
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
        diff = image - dx[3] - shift(tmpl - dx[4], dx[:2], mode='reflect') * dx[2]

        chi2_value = np.sum((diff[~imask1] / err[~imask1]) ** 2)  # Chi2

        if get_diff:
            return diff, mask1, imask1
        elif get_df:
            return chi2_value, np.sum(~imask1)
        else:
            return chi2_value

    if fit_bg:
        res = minimize(
            _fn,
            (0, 0, 1, bg, tbg),
            bounds=(
                (-max_shift, max_shift),
                (-max_shift, max_shift),
                (1 / max_scale, max_scale),
                (None, None),
                (None, None),
            ),
            method='Powell',
            options={'disp': False},
        )
    else:
        res = minimize(
            _fn,
            (0, 0, 1, bg, tbg),
            bounds=(
                (-max_shift, max_shift),
                (-max_shift, max_shift),
                (1 / max_scale, max_scale),
                (bg, bg),
                (tbg, tbg),
            ),
            method='Powell',
            options={'disp': False},
        )

    log(res.message)

    if res.success:
        log(
            'Adjustment is: %.2f %.2f bg %.2g tbg %.2g scale %.2f'
            % (res.x[0], res.x[1], res.x[3], res.x[4], res.x[2])
        )
        log('Chi2 improvement: %.2f -> %.2f' % (_fn([0, 0, 1, bg, tbg]), _fn(res.x)))

        diff, mask1, _ = _fn(res.x, get_diff=True)
        chi2_0 = _fn([0, 0, 1, bg, tbg])
        chi2_1, df = _fn(res.x, get_df=True)
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
            diff[err > 1e-30] /= err[err > 1e-30]

        # diff[mask1] = 1e-30

        # Add result as a new cutout plane
        cutout['adjusted'] = diff

        return True

    else:
        return False


def downscale_image(image, scale=1, mode='sum', header=None):
    """
    Downscales the image by an integer factor.
    Also adjusts the FITS header, if provided, so that WCS solution
    is still valid for the result.

    If image shape is not divisible by `scale`, it will be cropped.

    :param image: Image to be rebinned
    :param scale: integer downscaling coefficient
    :param mode: Pixel value reduction mode, one of `sum`, `mean`, `and` or `or`. Default to `sum`
    :param header: If provided, WCS solution in the header will be adjusted, and new header will be returned

    :returns: Rebinned image, and also corrected header if `header` was provided
    """

    # Crop the image if necessary
    maxx = scale * (image.shape[1] // scale)
    maxy = scale * (image.shape[0] // scale)
    image = image[:maxy, :maxx]

    shape = (image.shape[0] // scale, scale, image.shape[1] // scale, scale)

    image1 = image.reshape(shape)

    if mode == 'mean':
        image1 = image1.mean(-1).mean(1)
    elif mode == 'and':
        image1 = np.bitwise_and.reduce(image1, -1)
        image1 = np.bitwise_and.reduce(image1, 1)
    elif mode == 'or':
        image1 = np.bitwise_or.reduce(image1, -1)
        image1 = np.bitwise_or.reduce(image1, 1)
    else:
        # Fallback to mode='sum'
        image1 = image1.sum(-1).sum(1)

    if header is not None:
        wcs = WCS(header)
        wcs = wcs[::scale, ::scale]
        header1 = astrometry.clear_wcs(header, copy=True)
        header1 += wcs.to_header(relax=True)

        return image1, header1
    else:
        return image1
