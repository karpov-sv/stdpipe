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
from . import photometry_model
from .photometry_model import match, make_sn_model, get_detection_limit_sn, format_color_term
from . import photometry_measure
from .photometry_measure import measure_objects

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


