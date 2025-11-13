"""
Routines for object detection and photometry.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, tempfile, shlex
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import Table

import warnings
from astropy.wcs import FITSFixedWarning

# warnings.simplefilter(action='ignore', category=FITSFixedWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

import sep
import photutils
import photutils.background
import photutils.aperture
import photutils.centroids
from photutils.utils import calc_total_error

import statsmodels.api as sm
from scipy.optimize import minimize, least_squares, root_scalar

from . import astrometry

try:
    import cv2

    # Much faster dilation
    dilate = lambda image, mask: cv2.dilate(image.astype(np.uint8), mask).astype(bool)
except:
    from scipy.signal import fftconvolve

    dilate = lambda image, mask: fftconvolve(image, mask, mode='same') > 0.9

from . import utils


def make_kernel(r0=1.0, ext=1.0):
    x, y = np.mgrid[
        np.floor(-ext * r0) : np.ceil(ext * r0 + 1),
        np.floor(-ext * r0) : np.ceil(ext * r0 + 1),
    ]
    r = np.hypot(x, y)
    image = np.exp(-r ** 2 / 2 / r0 ** 2)

    return image


def get_objects_sep(
    image,
    header=None,
    mask=None,
    err=None,
    thresh=4.0,
    aper=3.0,
    bkgann=None,
    r0=0.5,
    gain=1,
    edge=0,
    minnthresh=2,
    minarea=5,
    relfluxradius=2.0,
    wcs=None,
    bg_size=64,
    use_fwhm=False,
    use_mask_large=False,
    subtract_bg=True,
    npix_large=100,
    sn=10.0,
    get_segmentation=False,
    verbose=True,
    **kwargs
):
    """Object detection and simple aperture photometry using `SEP <https://github.com/kbarbary/sep>`_ routines, with the signature as similar as possible to :func:`~stdpipe.photometry.get_objects_sextractor` function.

    Detection flags are documented at https://sep.readthedocs.io/en/v1.1.x/reference.html - they are different from SExtractor ones!

    :param image: Input image as a NumPy array
    :param header: Image header, optional
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param err: Image noise map as a NumPy array, optional
    :param thresh: Detection threshold in sigmas above local background
    :param aper: Circular aperture radius in pixels, to be used for flux measurement
    :param bkgann: Background annulus (tuple with inner and outer radii) to be used for local background estimation. Inside the annulus, simple arithmetic mean of unmasked pixels is used for computing the background, and thus it is subject to some bias in crowded stellar fields. If not set, global background model is used instead.
    :param r0: Smoothing kernel size (sigma) to be used for improving object detection
    :param gain: Image gain, e/ADU
    :param edge: Reject all detected objects closer to image edge than this parameter
    :param minnthresh: Minumal number of pixels above the threshold to be considered a detection
    :param minarea: Minimal number of pixels in the object to be considered a detection
    :param relfluxradius:
    :param wcs: Astrometric solution to be used for assigning sky coordinates (`ra`/`dec`) to detected objects
    :param bg_size: Background grid size in pixels
    :param use_fwhm: If True, the aperture will be set to 1.5*FWHM (if greater than `aper`)
    :param use_mask_large: If True, filter out large objects (with footprints larger than `npix_large` pixels)
    :param npix_large: Threshold for rejecting large objects (if `use_mask_large` is set)
    :param subtract_bg: Whether to subtract the background (default) or not
    :param sn: Minimal S/N ratio for the object to be considered a detection
    :param get_segmentation: If set, segmentation map will also be returned
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: astropy.table.Table object with detected objects
    """

    # Simple Wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    log("Preparing background mask")

    if mask is None:
        mask = np.zeros_like(image, dtype=bool)

    mask_bg = np.zeros_like(mask)
    mask_segm = np.zeros_like(mask)

    log("Building background map")

    bg = sep.Background(image, mask=mask | mask_bg, bw=bg_size, bh=bg_size)
    if subtract_bg:
        image1 = image - bg.back()
    else:
        image1 = image.copy()

    if err is None:
        err = bg.rms()
        err[~np.isfinite(err)] = 1e30
        err[err == 0] = 1e30

    sep.set_extract_pixstack(image.shape[0] * image.shape[1])

    if use_mask_large:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        log("Extracting initial objects")

        obj0, segm = sep.extract(
            image1,
            err=err,
            thresh=thresh,
            minarea=minarea,
            mask=mask | mask_bg,
            filter_kernel=kernel,
            segmentation_map=True,
        )

        log("Dilating large objects")

        mask_segm = np.isin(
            segm, [_ + 1 for _, npix in enumerate(obj0['npix']) if npix > npix_large]
        )
        mask_segm = dilate(mask_segm, np.ones([10, 10]))

    log("Extracting final objects")

    obj0,segm = sep.extract(
        image1,
        err=err,
        thresh=thresh,
        minarea=minarea,
        mask=mask | mask_bg | mask_segm,
        filter_kernel=kernel,
        segmentation_map=True,
        **kwargs
    )

    if use_fwhm:
        # Estimate FHWM and use it to get optimal aperture size
        idx = obj0['flag'] == 0
        fwhm = 2.0 * np.sqrt(np.hypot(obj0['a'][idx], obj0['b'][idx]) * np.log(2))
        fwhm = (
            2.0
            * sep.flux_radius(
                image1,
                obj0['x'][idx],
                obj0['y'][idx],
                relfluxradius * fwhm * np.ones_like(obj0['x'][idx]),
                0.5,
                mask=mask,
            )[0]
        )
        fwhm = np.median(fwhm)

        aper = max(1.5 * fwhm, aper)

        log("FWHM = %.2g, aperture = %.2g" % (fwhm, aper))

    # Windowed positional parameters are often biased in crowded fields, let's avoid them for now
    # xwin,ywin,flag = sep.winpos(image1, obj0['x'], obj0['y'], 0.5, mask=mask)
    xwin, ywin = obj0['x'], obj0['y']

    # Filter out objects too close to frame edges
    idx = (
        (np.round(xwin) > edge)
        & (np.round(ywin) > edge)
        & (np.round(xwin) < image.shape[1] - edge)
        & (np.round(ywin) < image.shape[0] - edge)
    )  # & (obj0['flag'] == 0)

    if minnthresh:
        idx &= obj0['tnpix'] >= minnthresh

    log("Measuring final objects")

    flux, fluxerr, flag = sep.sum_circle(
        image1,
        xwin[idx],
        ywin[idx],
        aper,
        err=err,
        gain=gain,
        mask=mask | mask_bg | mask_segm,
        bkgann=bkgann,
    )
    # For debug purposes, let's make also the same aperture photometry on the background map
    bgflux, bgfluxerr, bgflag = sep.sum_circle(
        bg.back(),
        xwin[idx],
        ywin[idx],
        aper,
        err=bg.rms(),
        gain=gain,
        mask=mask | mask_bg | mask_segm,
    )

    bgnorm = bgflux / np.pi / aper ** 2

    # Fluxes to magnitudes
    mag, magerr = np.zeros_like(flux), np.zeros_like(flux)
    mag[flux > 0] = -2.5 * np.log10(flux[flux > 0])
    # magerr[flux>0] = 2.5*np.log10(1.0 + fluxerr[flux>0]/flux[flux>0])
    magerr[flux > 0] = 2.5 / np.log(10) * fluxerr[flux > 0] / flux[flux > 0]

    # FWHM estimation - FWHM=HFD for Gaussian
    fwhm = (
        2.0
        * sep.flux_radius(
            image1,
            xwin[idx],
            ywin[idx],
            relfluxradius * aper * np.ones_like(xwin[idx]),
            0.5,
            mask=mask,
        )[0]
    )

    flag |= obj0['flag'][idx]

    # Quality cuts
    fidx = (flux > 0) & (magerr < 1.0 / sn)

    if wcs is None and header is not None:
        # If header is provided, we may build WCS from it
        wcs = WCS(header)

    if wcs is not None:
        # If WCS is provided we may convert x,y to ra,dec
        ra, dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        ra, dec = np.zeros_like(obj0['x'][idx]), np.zeros_like(obj0['y'][idx])

    if verbose:
        log("All done")

    obj = Table(
        {
            'x': xwin[idx][fidx],
            'y': ywin[idx][fidx],
            'xerr': np.sqrt(obj0['errx2'][idx][fidx]),
            'yerr': np.sqrt(obj0['erry2'][idx][fidx]),
            'flux': flux[fidx],
            'fluxerr': fluxerr[fidx],
            'mag': mag[fidx],
            'magerr': magerr[fidx],
            'flags': obj0['flag'][idx][fidx] | flag[fidx],
            'ra': ra[fidx],
            'dec': dec[fidx],
            'bg': bgnorm[fidx],
            'fwhm': fwhm[fidx],
            'a': obj0['a'][idx][fidx],
            'b': obj0['b'][idx][fidx],
            'theta': obj0['theta'][idx][fidx],
        }
    )

    obj.meta['aper'] = aper
    obj.meta['bkgann'] = bkgann

    obj.sort('flux', reverse=True)

    if get_segmentation:
        number = np.arange(len(obj0['x'])) + 1 # Running object number
        obj['number'] = number[idx][fidx]

        return obj, segm
    else:
        return obj


def get_objects_sextractor(
    image,
    header=None,
    mask=None,
    err=None,
    thresh=2.0,
    aper=3.0,
    r0=0.0,
    gain=1,
    edge=0,
    minarea=5,
    wcs=None,
    sn=3.0,
    bg_size=None,
    sort=True,
    reject_negative=True,
    mask_to_nans=False,
    checkimages=[],
    extra_params=[],
    extra={},
    psf=None,
    catfile=None,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Thin wrapper around SExtractor binary.

    It processes the image taking into account optional mask and noise map, and returns the list of detected objects and optionally a set of SExtractor-produced checkimages.

    You may check the SExtractor documentation at https://sextractor.readthedocs.io/en/latest/ for more details about possible parameters and general principles of its operation.
    E.g. detection flags (returned in `flags` column of results table) are documented at https://sextractor.readthedocs.io/en/latest/Flagging.html#extraction-flags-flags . In addition to these flags, any object having pixels masked by the input `mask` in its footprint will have :code:`0x100` flag set.

    :param image: Input image as a NumPy array
    :param header: Image header, optional
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param err: Image noise map as a NumPy array, optional
    :param thresh: Detection threshold, in sigmas above local background, to be used for `DETECT_THRESH` parameter of SExtractor call
    :param aper: Circular aperture radius in pixels, to be used for flux measurement. May also be list - then flux will be measured for all apertures from that list.
    :param r0: Smoothing kernel size (sigma, or FWHM/2.355) to be used for improving object detection
    :param gain: Image gain, e/ADU
    :param edge: Reject all detected objects closer to image edge than this parameter
    :param minarea: Minimal number of pixels in the object to be considered a detection (`DETECT_MINAREA` parameter of SExtractor)
    :param wcs: Astrometric solution to be used for assigning sky coordinates (`ra`/`dec`) to detected objects
    :param sn: Minimal S/N ratio for the object to be considered a detection
    :param bg_size: Background grid size in pixels (`BACK_SIZE` SExtractor parameter)
    :param sort: Whether to sort the detections in decreasing brightness or not
    :param reject_negative: Whether to reject the detections with negative fluxes
    :param mask_to_nans: Whether to replace masked image pixels with NaNs prior to running SExtractor on it
    :param checkimages: List of SExtractor checkimages to return along with detected objects. Any SExtractor checkimage type may be used here (e.g. `BACKGROUND`, `BACKGROUND_RMS`, `MINIBACKGROUND`,  `MINIBACK_RMS`, `-BACKGROUND`, `FILTERED`, `OBJECTS`, `-OBJECTS`, `SEGMENTATION`, `APERTURES`). Optional.
    :param extra_params: List of extra object parameters to return for the detection. See :code:`sex -dp` for the full list.
    :param extra: Dictionary of extra configuration parameters to be passed to SExtractor call, with keys as parameter names. See :code:`sex -dd` for the full list.
    :param psf: Path to PSFEx-made PSF model file to be used for PSF photometry. If provided, a set of PSF-measured parameters (`FLUX_PSF`, `MAG_PSF` etc) are added to detected objects. Optional
    :param catfile: If provided, output SExtractor catalogue file will be copied to this location, to be reused by external codes. Optional.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SExtractor. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to SExtractor executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Either the astropy.table.Table object with detected objects, or a list with table of objects (first element) and checkimages (consecutive elements), if checkimages are requested.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    # Find the binary
    binname = None

    if _exe is not None:
        # Check user-provided binary path, and fail if not found
        if os.path.isfile(_exe):
            binname = _exe
    else:
        # Find SExtractor binary in common paths
        for exe in ['sex', 'sextractor', 'source-extractor']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SExtractor binary")
        return None
    # else:
    #     log("Using SExtractor binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='sex', dir=_tmpdir)
    )
    obj = None

    if mask is None:
        # Create minimal mask
        mask = ~np.isfinite(image)
    else:
        # Ensure the mask is boolean array
        mask = mask.astype(bool)

    if mask_to_nans and isinstance(image, np.floating):
        image = image.copy()
        image[mask] = np.nan

    # Prepare
    if type(image) == str:
        # FIXME: this mode of operation is currently broken!
        imagename = image
    else:
        imagename = os.path.join(workdir, 'image.fits')
        fits.writeto(imagename, image, header, overwrite=True)

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'DETECT_MINAREA': minarea,
        'GAIN': gain,
        'DETECT_THRESH': thresh,
        'WEIGHT_TYPE': 'BACKGROUND',
        'MASK_TYPE': 'NONE',  # both 'CORRECT' and 'BLANK' seem to cause systematics?
        'SATUR_LEVEL': np.nanmax(image[~mask]) + 1 # Saturation should be handled in external mask
    }

    if bg_size is not None:
        opts['BACK_SIZE'] = bg_size

    if err is not None:
        # User-provided noise model
        err = err.copy().astype(np.double)
        err[~np.isfinite(err)] = 1e30
        err[err == 0] = 1e30

        errname = os.path.join(workdir, 'errors.fits')
        fits.writeto(errname, err, overwrite=True)
        opts['WEIGHT_IMAGE'] = errname
        opts['WEIGHT_TYPE'] = 'MAP_RMS'

    flagsname = os.path.join(workdir, 'flags.fits')
    fits.writeto(flagsname, mask.astype(np.int16), overwrite=True)
    opts['FLAG_IMAGE'] = flagsname

    if np.isscalar(aper):
        opts['PHOT_APERTURES'] = aper * 2  # SExtractor expects diameters, not radii
        size = ''
    else:
        opts['PHOT_APERTURES'] = ','.join([str(_ * 2) for _ in aper])
        size = '[%d]' % len(aper)

    checknames = [
        os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages
    ]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    params = [
        'MAG_APER' + size,
        'MAGERR_APER' + size,
        'FLUX_APER' + size,
        'FLUXERR_APER' + size,
        'X_IMAGE',
        'Y_IMAGE',
        'ERRX2_IMAGE',
        'ERRY2_IMAGE',
        'A_IMAGE',
        'B_IMAGE',
        'THETA_IMAGE',
        'FLUX_RADIUS',
        'FWHM_IMAGE',
        'FLAGS',
        'IMAFLAGS_ISO',
        'BACKGROUND',
    ]
    params += extra_params

    if psf is not None:
        opts['PSF_NAME'] = psf
        params += [
            'MAG_PSF',
            'MAGERR_PSF',
            'FLUX_PSF',
            'FLUXERR_PSF',
            'XPSF_IMAGE',
            'YPSF_IMAGE',
            'SPREAD_MODEL',
            'SPREADERR_MODEL',
            'CHI2_PSF',
        ]

    paramname = os.path.join(workdir, 'cfg.param')
    with open(paramname, 'w') as paramfile:
        paramfile.write("\n".join(params))
    opts['PARAMETERS_NAME'] = paramname

    catname = os.path.join(workdir, 'out.cat')
    opts['CATALOG_NAME'] = catname
    opts['CATALOG_TYPE'] = 'FITS_LDAC'

    if not r0:
        opts['FILTER'] = 'N'
    else:
        kernel = make_kernel(r0, ext=2.0)
        kernelname = os.path.join(workdir, 'kernel.txt')
        np.savetxt(
            kernelname,
            kernel / np.sum(kernel),
            fmt='%.6f',
            header='CONV NORM',
            comments='',
        )
        opts['FILTER'] = 'Y'
        opts['FILTER_NAME'] = kernelname

    opts.update(extra)

    # Build the command line
    cmd = (
        binname
        + ' '
        + shlex.quote(imagename)
        + ' '
        + utils.format_astromatic_opts(opts)
    )
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log('Will run SExtractor like that:')
    log(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(catname):
        log('SExtractor run succeeded')
        obj = Table.read(catname, hdu=2)
        obj.meta.clear()  # Remove unnecessary entries from the metadata

        idx = (obj['X_IMAGE'] > edge) & (obj['X_IMAGE'] < image.shape[1] - edge)
        idx &= (obj['Y_IMAGE'] > edge) & (obj['Y_IMAGE'] < image.shape[0] - edge)

        if np.isscalar(aper):
            if sn:
                idx &= obj['MAGERR_APER'] < 1.0 / sn
            if reject_negative:
                idx &= obj['FLUX_APER'] > 0
        else:
            if sn:
                idx &= np.all(obj['MAGERR_APER'] < 1.0 / sn, axis=1)
            if reject_negative:
                idx &= np.all(obj['FLUX_APER'] > 0, axis=1)

        obj = obj[idx]

        if wcs is None and header is not None:
            wcs = WCS(header)

        if wcs is not None:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['X_IMAGE'], obj['Y_IMAGE'], 1)
        else:
            obj['ra'], obj['dec'] = (
                np.zeros_like(obj['X_IMAGE']),
                np.zeros_like(obj['Y_IMAGE']),
            )

        obj['FLAGS'][obj['IMAFLAGS_ISO'] > 0] |= 0x100  # Masked pixels in the footprint
        obj.remove_column('IMAFLAGS_ISO')  # We do not need this column

        # Convert variances to rms
        obj['ERRX2_IMAGE'] = np.sqrt(obj['ERRX2_IMAGE'])
        obj['ERRY2_IMAGE'] = np.sqrt(obj['ERRY2_IMAGE'])

        for _, __ in [
            ['X_IMAGE', 'x'],
            ['Y_IMAGE', 'y'],
            ['ERRX2_IMAGE', 'xerr'],
            ['ERRY2_IMAGE', 'yerr'],
            ['FLUX_APER', 'flux'],
            ['FLUXERR_APER', 'fluxerr'],
            ['MAG_APER', 'mag'],
            ['MAGERR_APER', 'magerr'],
            ['BACKGROUND', 'bg'],
            ['FLAGS', 'flags'],
            ['FWHM_IMAGE', 'fwhm'],
            ['A_IMAGE', 'a'],
            ['B_IMAGE', 'b'],
            ['THETA_IMAGE', 'theta'],
        ]:
            obj.rename_column(_, __)

        if psf:
            for _, __ in [
                ['XPSF_IMAGE', 'x_psf'],
                ['YPSF_IMAGE', 'y_psf'],
                ['MAG_PSF', 'mag_psf'],
                ['MAGERR_PSF', 'magerr_psf'],
                ['FLUX_PSF', 'flux_psf'],
                ['FLUXERR_PSF', 'fluxerr_psf'],
                ['CHI2_PSF', 'chi2_psf'],
                ['SPREAD_MODEL', 'spread_model'],
                ['SPREADERR_MODEL', 'spreaderr_model'],
            ]:
                if _ in obj.keys():
                    obj.rename_column(_, __)
                    if 'mag' in __:
                        obj[__][obj[__] == 99] = np.nan  # TODO: use masked column here?

        # SExtractor uses 1-based pixel coordinates
        obj['x'] -= 1
        obj['y'] -= 1

        if 'x_psf' in obj.keys():
            obj['x_psf'] -= 1
            obj['y_psf'] -= 1

        obj.meta['aper'] = aper

        if sort:
            if np.isscalar(aper):
                obj.sort('flux', reverse=True)
            else:
                # Table sorting by vector columns seems to be broken?..
                obj = obj[np.argsort(-obj['flux'][:, 0])]

        if catfile is not None:
            shutil.copyfile(catname, catfile)
            log("Catalogue stored to", catfile)

    else:
        log("Error", res, "running SExtractor")

    result = obj

    if checkimages:
        result = [result]

        for name in checknames:
            if os.path.exists(name):
                result.append(fits.getdata(name))
            else:
                log("Cannot find requested output checkimage file", name)
                result.append(None)

    if _workdir is None:
        shutil.rmtree(workdir)

    return result


def make_series(mul=1.0, x=1.0, y=1.0, order=1, sum=False, zero=True):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if zero:
        res = [np.ones_like(x) * mul]
    else:
        res = []

    for i in range(1, order + 1):
        maxr = i + 1

        for j in range(maxr):
            res.append(mul * x ** (i - j) * y ** j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res


def get_intrinsic_scatter(y, yerr, min=0, max=None):
    def log_likelihood(theta, y, yerr):
        a, b, c = theta
        model = b
        sigma2 = a * yerr ** 2 + c ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    nll = lambda *args: -log_likelihood(*args)
    C = minimize(
        nll,
        [1, 0.0, 0.0],
        args=(y, yerr),
        bounds=[[1, 1], [None, None], [min, max]],
        method='Powell',
    )

    return C.x[2]


def format_color_term(color_term, name=None, color_name=None, fmt='.2f'):
    result = []

    if color_term is None:
        return format(0.0, fmt)

    if name is not None:
        result += [name]

    if isinstance(color_term, float) or isinstance(color_term, int):
        # Scalar?..
        color_term = [color_term]

    # Here we assume it is a list
    for i,val in enumerate(color_term):
        if color_name is not None:
            sign = '-' if val > 0 else '+' # Reverse signs!!!
            sval = format(np.abs(val), fmt)
            deg = '^%d' % (i + 1) if i > 0 else ''
            result += [sign + ' ' + sval + ' (' + color_name + ')' + deg]
        else:
            result += [format(val, fmt)]

    return " ".join(result)


def match(
    obj_ra,
    obj_dec,
    obj_mag,
    obj_magerr,
    obj_flags,
    cat_ra,
    cat_dec,
    cat_mag,
    cat_magerr=None,
    cat_color=None,
    sr=3 / 3600,
    obj_x=None,
    obj_y=None,
    spatial_order=0,
    bg_order=None,
    nonlin=False,
    threshold=5.0,
    niter=10,
    accept_flags=0,
    cat_saturation=None,
    max_intrinsic_rms=0,
    sn=None,
    verbose=False,
    robust=True,
    scale_noise=False,
    use_color=True,
    force_color_term=None,
):
    """Low-level photometric matching routine.

    It tries to build the photometric model for objects detected on the image that includes catalogue magnitude, positionally-dependent zero point, linear color term, optional additive flux term, and also takes into account possible intrinsic magnitude scatter on top of measurement errors.

    :param obj_ra: Array of Right Ascension values for the objects
    :param obj_dec: Array of Declination values for the objects
    :param obj_mag: Array of instrumental magnitude values for the objects
    :param obj_magerr: Array of instrumental magnitude errors for the objects
    :param obj_flags: Array of flags for the objects
    :param cat_ra: Array of catalogue Right Ascension values
    :param cat_dec: Array of catalogue Declination values
    :param cat_mag: Array of catalogue magnitudes
    :param cat_magerr: Array of catalogue magnitude errors
    :param cat_color: Array of catalogue color values, optional
    :param sr: Matching radius, degrees
    :param obj_x: Array of `x` coordinates of objects on the image, optional
    :param obj_y: Array of `y` coordinates of objects on the image, optional
    :param spatial_order: Order of zero point spatial polynomial (0 for constant).
    :param bg_order: Order of additive flux term spatial polynomial (None to disable this term in the model)
    :param nonlin: Whether to fit for simple non-linearity, optional
    :param threshold: Rejection threshold (relative to magnitude errors) for object-catalogue pair to be rejected from the fit
    :param niter: Number of iterations for the fitting
    :param accept_flags: Bitmask for acceptable object flags. Objects having any other bits set will be excluded from the model
    :param cat_saturation: Saturation level for the catalogue - stars brighter than this magnitude will be excluded from the fit
    :param max_intrinsic_rms: Maximal intrinsic RMS to use during the fitting. If set to 0, no intrinsic scatter is included in the noise model.
    :param sn: Minimal acceptable signal to noise ratio (1/obj_magerr) for the objects to be included in the fit
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param robust: Whether to use robust least squares fitting routine instead of weighted least squares
    :param scale_noise: Whether to re-scale the noise model (object and catalogue magnitude errors) to match actual scatter of the data points or not. Intrinsic scatter term is not being scaled this way.
    :param use_color: Whether to use catalogue color for deriving the color term. If integer, it determines the color term order.
    :param force_color_term: Do not fit for the color term, but use this fixed value instead.
    :returns: The dictionary with photometric results, as described below.

    The results of photometric matching are returned in a dictionary with the following fields:

    -  `oidx`, `cidx`, `dist` - indices of positionally matched objects and catalogue stars, as well as their pairwise distances in degrees
    -  `omag`, `omagerr`, `cmag`, `cmagerr` - arrays of object instrumental magnitudes of matched objects, corresponding catalogue magnitudes, and their errors. Array lengths are equal to the number of positional matches.
    -  `color` - catalogue colors corresponding to the matches, or zeros if no color term fitting is requested
    -  `ox`, `oy`, `oflags` - coordinates of matched objects on the image, and their flags
    -  `zero`, `zero_err` - empirical zero points (catalogue - instrumental magnitudes) for every matched object, as well as its errors, derived as a hypotenuse of their corresponding errors.
    -  `zero_model`, `zero_model_err` - modeled "full" zero points (including color terms) for matched objects, and their corresponding errors from the fit
    -  `color_term` - fitted color term. Instrumental photometric system is defined as :code:`obj_mag = cat_mag - color*color_term`
    -  `zero_fn` - function to compute the zero point (without color term) at a given position and for a given instrumental magnitude of object, and optionally its error.
    -  `obj_zero` - zero points for all input objects (not necessarily matched to the catalogue) computed through aforementioned function, i.e. without color term
    -  `params` - Internal parameters of the fittting polynomial
    -  `intrinsic_rms`, `error_scale` - fitted values of intrinsic scatter and noise scaling
    -  `idx` - boolean index of matched objects/catalogue stars used in the final fit (i.e. not rejected during iterative thresholding, and passing initial quality cuts
    -  `idx0` - the same but with just initial quality cuts taken into account

    Returned zero point computation function has the following signature:

    :obj:`zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False)`

    where `xx` and `yy` are coordinates on the image, `mag` is object instrumental magnitude (needed to compute additive flux term). If :code:`get_err=True`, the function returns estimated zero point error instead of zero point, and `add_intrinsic_rms` controls whether this error estimation should also include intrinsic scatter term or not.

    The zero point returned by this function does not include the contribution of color term. Therefore, in order to derive the final calibrated magnitude for the object, you will need to manually add the color contribution: :code:`mag_calibrated = mag_instrumental + color*color_term`, where `color` is a true object color, and `color_term` is reported in the photometric results.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    oidx, cidx, dist = astrometry.spherical_match(obj_ra, obj_dec, cat_ra, cat_dec, sr)

    log(
        len(dist),
        'initial matches between',
        len(obj_ra),
        'objects and',
        len(cat_ra),
        'catalogue stars, sr = %.2f arcsec' % (sr * 3600),
    )
    log('Median separation is %.2f arcsec' % (np.median(dist) * 3600))

    omag = np.ma.filled(obj_mag[oidx], fill_value=np.nan)
    omag_err = np.ma.filled(obj_magerr[oidx], fill_value=np.nan)
    oflags = (
        obj_flags[oidx] if obj_flags is not None else np.zeros_like(omag, dtype=bool)
    )
    cmag = np.ma.filled(cat_mag[cidx], fill_value=np.nan)
    cmag_err = (
        np.ma.filled(cat_magerr[cidx], fill_value=np.nan)
        if cat_magerr is not None
        else np.zeros_like(cmag)
    )

    if obj_x is not None and obj_y is not None:
        x0, y0 = np.mean(obj_x[oidx]), np.mean(obj_y[oidx])
        ox, oy = obj_x[oidx], obj_y[oidx]
        x, y = obj_x[oidx] - x0, obj_y[oidx] - y0
    else:
        x0, y0 = 0, 0
        ox, oy = np.zeros_like(omag), np.zeros_like(omag)
        x, y = np.zeros_like(omag), np.zeros_like(omag)

    # Regressor
    X = make_series(1.0, x, y, order=spatial_order)
    log('Fitting the model with spatial_order =', spatial_order)

    if bg_order is not None:
        # Spatially varying additive flux component, linearized in magnitudes
        X += make_series(-2.5 / np.log(10) / 10 ** (-0.4 * omag), x, y, order=bg_order)
        log('Adjusting background level using polynomial with bg_order =', bg_order)

    if nonlin:
        # Non-linearity
        pos_nonlin = len(X)
        X += make_series(omag, x, y, order=0)
        log('Fitting for simple non-linearity')

    if robust:
        log('Using robust fitting')
    else:
        log('Using weighted fitting')

    if cat_color is not None:
        ccolor = np.ma.filled(cat_color[cidx], fill_value=np.nan)
        if use_color:
            pos_color = len(X)
            for _ in range(int(use_color)):
                X += make_series(ccolor**(_ + 1), x, y, order=0)

            log('Using color term of order', int(use_color))
        elif force_color_term is not None:
            for i,val in enumerate(np.atleast_1d(force_color_term)):
                cmag -= val * ccolor**(i + 1)
            log('Using fixed color term', format_color_term(force_color_term))
    else:
        ccolor = np.zeros_like(cmag)

    Nparams = len(X)  # Number of parameters to be fitted

    X = np.vstack(X).T
    zero = cmag - omag  # We will build a model for this definition of zero point
    zero_err = np.hypot(omag_err, cmag_err)
    # weights = 1.0/zero_err**2

    idx0 = (
        np.isfinite(omag)
        & np.isfinite(omag_err)
        & np.isfinite(cmag)
        & np.isfinite(cmag_err)
        & ((oflags & ~accept_flags) == 0)
    )  # initial mask
    if cat_color is not None and use_color:
        idx0 &= np.isfinite(ccolor)
    if cat_saturation is not None:
        idx0 &= cmag >= cat_saturation
    if sn is not None:
        idx0 &= omag_err < 1 / sn

    log('%d objects pass initial quality cuts' % np.sum(idx0))

    idx = idx0.copy()

    intrinsic_rms = 0
    scale_err = 1
    total_err = zero_err

    for iter in range(niter):
        if np.sum(idx) < Nparams + 1:
            log(
                "Fit failed - %d objects remaining for fitting %d parameters"
                % (np.sum(idx), Nparams)
            )
            return None

        if robust:
            # Rescale the arguments with weights
            C = sm.RLM(zero[idx] / total_err[idx], (X[idx].T / total_err[idx]).T).fit()
        else:
            C = sm.WLS(zero[idx], X[idx], weights=1 / total_err[idx] ** 2).fit()

        zero_model = np.sum(X * C.params, axis=1)
        zero_model_err = np.sqrt(C.cov_params(X).diagonal())

        intrinsic_rms = (
            get_intrinsic_scatter(
                (zero - zero_model)[idx], total_err[idx], max=max_intrinsic_rms
            )
            if max_intrinsic_rms > 0
            else 0
        )

        scale_err = 1 if not scale_noise else np.sqrt(C.scale)  # rms
        total_err = np.hypot(zero_err * scale_err, intrinsic_rms)

        if threshold:
            idx1 = np.abs((zero - zero_model) / total_err)[idx] < threshold
        else:
            idx1 = np.ones_like(idx[idx])

        log(
            'Iteration',
            iter,
            ':',
            np.sum(idx),
            '/',
            len(idx),
            '- rms',
            '%.2f' % np.std((zero - zero_model)[idx0]),
            '%.2f' % np.std((zero - zero_model)[idx]),
            '- normed',
            '%.2f' % np.std((zero - zero_model)[idx] / zero_err[idx]),
            '%.2f' % np.std((zero - zero_model)[idx] / total_err[idx]),
            '- scale %.2f %.2f' % (np.sqrt(C.scale), scale_err),
            '- rms',
            '%.2f' % intrinsic_rms,
        )

        if not np.sum(~idx1):  # and new_intrinsic_rms <= intrinsic_rms:
            log('Fitting converged')
            break
        else:
            idx[idx] &= idx1

    log(np.sum(idx), 'good matches')
    if max_intrinsic_rms > 0:
        log('Intrinsic scatter is %.2f' % intrinsic_rms)

    if nonlin:
        log('Non-linearity term is %.3f' % C.params[pos_nonlin])

    # Export the model
    def zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False):
        if xx is not None and yy is not None:
            x, y = xx - x0, yy - y0
        else:
            x, y = np.zeros_like(omag), np.zeros_like(omag)

        X = make_series(1.0, x, y, order=spatial_order)

        if bg_order is not None and mag is not None:
            X += make_series(
                -2.5 / np.log(10) / 10 ** (-0.4 * np.ma.filled(mag, np.nan)), x, y, order=bg_order
            )

        if nonlin and mag is not None:
            X += make_series(np.ma.filled(mag, np.nan), x, y, order=0)

        X = np.vstack(X).T

        if get_err:
            # It follows the implementation from https://github.com/statsmodels/statsmodels/blob/081fc6e85868308aa7489ae1b23f6e72f5662799/statsmodels/base/model.py#L1383
            # FIXME: crashes on large numbers of stars?..
            if len(x) < 5000:
                err = np.sqrt(np.dot(X, np.dot(C.cov_params()[0:X.shape[1], 0:X.shape[1]], np.transpose(X))).diagonal())
            else:
                err = np.zeros_like(x)
            if add_intrinsic_rms:
                err = np.hypot(err, intrinsic_rms)
            return err
        else:
            return np.sum(X * C.params[0 : X.shape[1]], axis=1)

    if cat_color is not None and (use_color or force_color_term is not None):
        if use_color:
            color_term = list(C.params[pos_color:][:int(use_color)])
            if len(color_term) == 1:
                color_term = color_term[0]

            log('Color term is', format_color_term(color_term))
        elif force_color_term is not None:
            color_term = force_color_term
            log('Color term (fixed) is', format_color_term(color_term))
    else:
        color_term = None

    return {
        'oidx': oidx,
        'cidx': cidx,
        'dist': dist,
        'omag': omag,
        'omag_err': omag_err,
        'cmag': cmag,
        'cmag_err': cmag_err,
        'color': ccolor,
        'color_term': color_term,
        'zero': zero,
        'zero_err': zero_err,
        'zero_model': zero_model,
        'zero_model_err': zero_model_err,
        'zero_fn': zero_fn,
        'params': C.params,
        'error_scale': np.sqrt(C.scale),
        'intrinsic_rms': intrinsic_rms,
        'obj_zero': zero_fn(obj_x, obj_y, mag=obj_mag),
        'ox': ox,
        'oy': oy,
        'oflags': oflags,
        'idx': idx,
        'idx0': idx0,
    }


def get_background(image, mask=None, method='sep', size=128, get_rms=False, **kwargs):
    if method == 'sep':
        bg = sep.Background(image, mask=mask, bw=size, bh=size, **kwargs)

        back, backrms = bg.back(), bg.rms()
    else:  # photutils
        bg = photutils.background.Background2D(image, size, mask=mask, **kwargs)
        back, backrms = bg.background, bg.background_rms

    if get_rms:
        return back, backrms
    else:
        return back


def measure_objects(
    obj,
    image,
    aper=3,
    bkgann=None,
    fwhm=None,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    sn=None,
    centroid_iter=0,
    keep_negative=True,
    get_bg=False,
    verbose=False,
):
    """Aperture photometry at the positions of already detected objects.

    It will estimate and subtract the background unless external background estimation (`bg`) is provided, and use user-provided noise map (`err`) if requested.

    If the `mask` is provided, it will set 0x200 bit in object `flags` if at least one of aperture pixels is masked.

    The results may optionally filtered to drop the detections with low signal to noise ratio if `sn` parameter is set and positive. It will also filter out the events with negative flux.


    :param obj: astropy.table.Table with initial object detections to be measured
    :param image: Input image as a NumPy array
    :param aper: Circular aperture radius in pixels, to be used for flux measurement
    :param bkgann: Background annulus (tuple with inner and outer radii) to be used for local background estimation. If not set, global background model is used instead.
    :param fwhm: If provided, `aper` and `bkgann` will be measured in units of this value (so they will be specified in units of FWHM)
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array with same shape as input image) instead of automatically computed one
    :param err: Image noise map as a NumPy array to be used instead of automatically computed one, optional
    :param gain: Image gain, e/ADU, used to build image noise model
    :param bg_size: Background grid size in pixels
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param centroid_iter: Number of centroiding iterations to run before photometry. If non-zero, will try to improve the aperture placement by finding the centroid of pixels inside the aperture.
    :param keep_negative: If not set, measurements with negative fluxes will be discarded
    :param get_bg: If True, the routine will also return estimated background and background noise images
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The copy of original table with `flux`, `fluxerr`, `mag` and `magerr` columns replaced with the values measured in the routine. If :code:`get_bg=True`, also returns the background and background error images.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if not len(obj):
        log('No objects to measure')
        return obj

    # Operate on the copy of the list
    obj = obj.copy()

    # Sanitize the image and make its copy to safely operate on it
    image1 = image.astype(np.double)
    mask0 = ~np.isfinite(image1)  # Minimal mask
    # image1[mask0] = np.median(image1[~mask0])

    # Ensure that the mask is defined
    if mask is None:
        mask = mask0
    else:
        mask = mask.astype(bool)

    if bg is None or err is None or get_bg:
        log('Estimating global background with %dx%d mesh' % (bg_size, bg_size))
        bg_est = photutils.background.Background2D(
            image1, bg_size, mask=mask | mask0, exclude_percentile=90
        )
        bg_est_bg = bg_est.background
        bg_est_rms = bg_est.background_rms
    else:
        bg_est = None

    if bg is None:
        log(
            'Subtracting global background: median %.1f rms %.2f' % (
                np.median(bg_est_bg), np.std(bg_est_bg)
            )
        )
        image1 -= bg_est_bg
    else:
        log(
            'Subtracting user-provided background: median %.1f rms %.2f' % (
                np.median(bg), np.std(bg)
            )
        )
        image1 -= bg

    image1[mask0] = 0

    if err is None:
        log(
            'Using global background noise map: median %.1f rms %.2f + gain %.1f' % (
                np.median(bg_est_rms),
                np.std(bg_est_rms),
                gain if gain else np.inf,
            )
        )
        err = bg_est_rms
        if gain:
            err = calc_total_error(image1, err, gain)
    else:
        log(
            'Using user-provided noise map: median %.1f rms %.2f' % (
                np.median(err), np.std(err)
            )
        )

    if fwhm is not None and fwhm > 0:
        log('Scaling aperture radii with FWHM %.1f pix' % fwhm)
        aper *= fwhm

    log('Using aperture radius %.1f pixels' % aper)

    if centroid_iter:
        box_size = int(np.ceil(aper))
        if box_size % 2 == 0:
            box_size += 1
        log('Using centroiding routine with %d iterations within %dx%d box' % (centroid_iter, box_size, box_size))
        # Keep original pixel positions
        obj['x_orig'] = obj['x']
        obj['y_orig'] = obj['y']

        for iter in range(centroid_iter):
            obj['x'],obj['y'] = photutils.centroids.centroid_sources(
                image1,
                obj['x'],
                obj['y'],
                box_size=box_size,
                mask=mask
            )

    # FIXME: is there any better way to exclude some positions from photometry?..
    positions = [(_['x'], _['y']) if np.isfinite(_['x']) and np.isfinite(_['y']) else (-1000, -1000) for _ in obj]
    apertures = photutils.aperture.CircularAperture(positions, r=aper)
    # Use just a minimal mask here so that the flux from 'soft-masked' (e.g. saturated) pixels is still counted
    res = photutils.aperture.aperture_photometry(image1, apertures, error=err, mask=mask0)

    obj['flux'] = res['aperture_sum']
    obj['fluxerr'] = res['aperture_sum_err']

    if 'flags' not in obj.keys():
        obj['flags'] = 0

    # Check whether some aperture pixels are masked, and set the flags for that
    mres = photutils.aperture.aperture_photometry(mask | mask0, apertures, method='center')
    obj['flags'][mres['aperture_sum'] > 0] |= 0x200

    # Position-dependent background flux error from global background model, if available
    obj['bg_fluxerr'] = 0.0  # Local background flux error inside the aperture
    if bg_est is not None:
        res = photutils.aperture.aperture_photometry(bg_est_rms**2, apertures)
        obj['bg_fluxerr'] = np.sqrt(res['aperture_sum'])

    # Local background
    if bkgann is not None and len(bkgann) == 2:
        if fwhm is not None and fwhm > 0:
            bkgann = [_ * fwhm for _ in bkgann]
        log(
            'Using local background annulus between %.1f and %.1f pixels' % (
                bkgann[0], bkgann[1]
            )
        )

        # Aperture areas
        image_ones = np.ones_like(image1)
        res_area = photutils.aperture.aperture_photometry(image_ones, apertures, mask=mask0)

        # Local background
        lbg = photutils.background.LocalBackground(
            bkgann[0], bkgann[1],
            bkg_estimator=photutils.background.ModeEstimatorBackground(),
        )

        # Dedicated column for local background on top of global estimation
        obj['bg_local'] = lbg(image1, obj['x'], obj['y'], mask=mask)

        # Sanitize and flag the values where local bg estimation failed
        idx = ~np.isfinite(obj['bg_local'])
        obj['bg_local'][idx] = 0
        obj['flags'][idx] |= 0x400

        obj['flux'] -= obj['bg_local'] * res_area['aperture_sum']

    idx = obj['flux'] > 0
    for _ in ['mag', 'magerr']:
        if _ not in obj.keys():
            obj[_] = np.nan

    obj['mag'][idx] = -2.5 * np.log10(obj['flux'][idx])
    obj['mag'][~idx] = np.nan

    obj['magerr'][idx] = 2.5 / np.log(10) * obj['fluxerr'][idx] / obj['flux'][idx]
    obj['magerr'][~idx] = np.nan

    # Final filtering of properly measured objects
    if sn is not None and sn > 0:
        log('Filtering out measurements with S/N < %.1f' % sn)
        idx = np.isfinite(obj['magerr'])
        idx[idx] &= obj['magerr'][idx] < 1 / sn
        obj = obj[idx]

    if not keep_negative:
        log('Filtering out measurements with negative fluxes')
        idx = obj['flux'] > 0
        obj = obj[idx]

    if get_bg:
        return obj, bg_est_bg, err
    else:
        return obj


def make_sn_model(mag, sn):
    """
    Build a model for signal to noise (S/N) ratio versus magnitude.
    Assumes the noise comes from constant background noise plus Poissonian noise with constant gain.

    :param mag: Array of calibrated magnitudes
    :param sn: Array of S/N values corresponding to them
    :returns: The function that accepts the array of magnitudes and returns the S/N model values for them
    """
    idx = np.isfinite(mag) & np.isfinite(sn)
    mag = mag[idx]
    sn = sn[idx]

    def sn_fn(p, mag):
        return 1 / np.sqrt(p[0] * 10 ** (0.8 * mag) + p[1] * 10 ** (0.4 * mag))

    def lstsq_fn(p, x, y):
        # Minimize residuals in logarithms, for better stability
        return np.log10(y) - np.log10(sn_fn(p, x))

    aidx = np.argsort(sn)

    # Initial params from two limiting cases, one on average and one on brightest point
    x = [
        np.median(10 ** (-0.8 * mag) / sn ** 2),
        10 ** (-0.4 * mag[aidx][-1]) / sn[aidx][-1] ** 2,
    ]

    res = least_squares(lstsq_fn, x, args=(mag, sn), method='lm')

    return lambda mag: sn_fn(res.x, mag)


def get_detection_limit_sn(mag, mag_sn, sn=5, get_model=False, verbose=True):
    """
    Estimate the detection limit using S/N vs magnitude method.

    :param mag: Array of calibrated magnitudes
    :param mag_sn: Array of S/N values corresponding to these magnitudes
    :param sn: S/N level for the detection limit
    :param get_model: If True, also returns the S/N model function
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :returns: The magnitude corresponding to the detection limit on a given S/N level. If :code:`get_model=True`, also returns the function for S/N vs magnitude model
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    mag0 = None

    sn_model = make_sn_model(mag, mag_sn)
    res = root_scalar(
        lambda x: np.log10(sn_model(x)) - np.log10(sn),
        x0=np.nanmax(mag),
        x1=np.nanmax(mag) + 1,
    )
    if res.converged:
        mag0 = res.root
    else:
        log('Cannot determine the root of S/N model function')

    if get_model:
        return mag0, sn_model
    else:
        return mag0
