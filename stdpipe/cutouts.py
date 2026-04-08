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
    """Crop a square region centered on a given pixel position.

    Output size is ``2*ceil(r0) + 1`` pixels.  The original center falls
    inside the pixel at ``x, y = ceil(r0), ceil(r0)``.  If a FITS header is
    provided, it is adjusted so that the WCS solution remains valid for the
    cutout.

    Parameters
    ----------
    data : ndarray
        2D image array to crop.
    x0 : float
        X coordinate of the center (0-based).
    y0 : float
        Y coordinate of the center (0-based).
    r0 : float
        Cutout half-size in pixels.
    header : astropy.io.fits.Header, optional
        FITS header to adjust (``CRPIX`` shifted in-place on a copy).

    Returns
    -------
    ndarray or tuple
        Cropped image, or ``(cropped_image, adjusted_header)`` if ``header`` is provided.
    """

    # x1,x2 = int(np.floor(x0 - r0)), int(np.ceil(x0 + r0))
    # y1,y2 = int(np.floor(y0 - r0)), int(np.ceil(y0 + r0))
    x1, x2 = int(np.round(x0) - np.ceil(r0)), int(np.round(x0) + np.ceil(r0))
    y1, y2 = int(np.round(y0) - np.ceil(r0)), int(np.round(y0) + np.ceil(r0))

    return crop_image(data, x1, y1, x2 - x1 + 1, y2 - y1 + 1, header=header)


def crop_image(data, x1, y1, width, height, header=None):
    """Crop a rectangular region from an image.

    Pixels outside the original image boundary are filled with zeros (or NaN
    for floating-point images).  If a FITS header is provided, ``CRPIX`` is
    adjusted so that the WCS solution remains valid for the cutout.

    Parameters
    ----------
    data : ndarray
        2D image array to crop.
    x1 : int
        Left edge of the crop region (0-based).
    y1 : int
        Bottom edge of the crop region (0-based).
    width : int
        Width of the crop region in pixels.
    height : int
        Height of the crop region in pixels.
    header : astropy.io.fits.Header, optional
        FITS header to adjust (copy is modified, original unchanged).

    Returns
    -------
    ndarray or tuple
        Cropped image, or ``(cropped_image, adjusted_header)`` if ``header`` is provided.
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
    image, candidate, radius, header=None, wcs=None, time=None, filename=None, name=None, **kwargs
):
    """Create a cutout postage stamp from one or more image planes.

    The candidate may be a row from :class:`astropy.table.Table` or a dict with
    at least ``x`` and ``y`` keys.  The cutout is centered so the object falls
    inside the central pixel; size is ``2*ceil(radius) + 1`` pixels.

    Parameters
    ----------
    image : ndarray
        Primary (science) image plane.
    candidate : table row or dict
        Object record containing at least ``x`` and ``y`` pixel coordinates.
    radius : float
        Cutout half-size in pixels.
    header : astropy.io.fits.Header, optional
        Header of the original image.  Copied and adjusted to represent the
        cutout WCS; stored as ``cutout['header']``.
    wcs : astropy.wcs.WCS, optional
        WCS of the original image.  Adjusted and stored as ``cutout['wcs']``.
        Takes precedence over ``header`` for WCS.
    time : astropy.time.Time, datetime, or str, optional
        Observation timestamp; stored in ``cutout['meta']['time']``.
    filename : str, optional
        Source image filename; stored in ``cutout['meta']['filename']``.
    name : str, optional
        Object name; stored in ``cutout['meta']['name']``.  If omitted and
        ``candidate`` has ``ra`` / ``dec``, a J2000 name is constructed.
    **kwargs
        Additional 2D arrays interpreted as extra image planes
        (e.g. ``mask``, ``diff``, ``template``, ``convolved``, ``err``).

    Returns
    -------
    dict
        Cutout dictionary with at least:

        - ``image`` — primary image plane
        - ``mask``, ``diff``, ``template``, etc. — extra planes if provided
        - ``header`` — adjusted FITS header, if provided
        - ``wcs`` — adjusted WCS, if provided
        - ``meta`` — dict with all candidate fields plus ``time``, ``filename``,
          and ``name``
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
    for pname, plane in kwargs.items():
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
    """Store a cutout as a multi-extension FITS file.

    Each image plane becomes a named FITS extension; metadata is stored as
    keywords in the primary header.

    Parameters
    ----------
    cutout : dict
        Cutout structure as returned by :func:`get_cutout`.
    filename : str
        Output FITS filename.
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
    """Restore a cutout from a multi-extension FITS file.

    Parameters
    ----------
    filename : str
        Path to a FITS file written by :func:`write_cutout`.

    Returns
    -------
    dict
        Cutout structure as returned by :func:`get_cutout`.
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
    """Fit a positional and flux-scaling adjustment to minimize the image–template residual.

    Optimizes a shift (dx, dy), scale, and optionally background levels to
    minimize the chi-squared residual between ``cutout['image']`` and
    ``cutout['convolved']``.  On success, adds a ``cutout['adjusted']`` plane
    with the optimized difference.

    Parameters
    ----------
    cutout : dict
        Cutout structure as returned by :func:`get_cutout`. Must contain
        ``image``, ``convolved``, and ``err`` planes.
    max_shift : float, optional
        Maximum allowed positional shift in pixels (symmetric bound).
    max_scale : float, optional
        Maximum allowed flux scale factor; scale is bounded to
        ``(1/max_scale, max_scale)``.
    inner : int, optional
        If set, only the central ``inner × inner`` pixel box is used for
        optimization.
    normalize : bool, optional
        If True, divide the ``adjusted`` plane by the ``err`` plane.
    fit_bg : bool, optional
        If True, fit background levels as free parameters. If False, estimate
        backgrounds from the cutout using SExtractor-mode
        (``2.5*median − 1.5*mean``) and hold them fixed.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    bool
        ``True`` if optimization succeeded, ``False`` otherwise.

    Notes
    -----
    On success the following keys are added to ``cutout['meta']``:
    ``adjust_chi2_0``, ``adjust_chi2``, ``adjust_df``, ``adjust_pval``,
    ``adjust_dx``, ``adjust_dy``, ``adjust_scale``, ``adjust_bg``, ``adjust_tbg``.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    mask = cutout['mask'] if 'mask' in cutout else ~np.isfinite(cutout['image'])
    imask = np.zeros_like(mask)

    # Rough estimation of backgrounds in the image and the template, using SExtractor-like mode estimation
    # bg = np.nanmedian(cutout['image'][~mask])
    bg = 2.5 * np.nanmedian(cutout['image'][~mask]) - 1.5 * np.nanmean(cutout['image'][~mask])
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
    """Downscale an image by an integer factor.

    If the image dimensions are not divisible by ``scale``, the image is
    cropped to the largest divisible size first.  If a FITS header is
    provided, the WCS is adjusted for the downscaled pixel grid.

    Parameters
    ----------
    image : ndarray
        2D image array to downscale.
    scale : int, optional
        Integer downscaling factor.
    mode : str, optional
        Pixel reduction mode: ``'sum'`` (default), ``'mean'``, ``'and'``, or ``'or'``.
    header : astropy.io.fits.Header, optional
        If provided, the WCS is adjusted and a new header is returned alongside
        the downscaled image.

    Returns
    -------
    ndarray or tuple
        Downscaled image, or ``(downscaled_image, adjusted_header)`` if
        ``header`` is provided.
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
