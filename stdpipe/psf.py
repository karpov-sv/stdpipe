"""
Module for working with point-spread function (PSF) models
"""


import os, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData

from scipy import ndimage

import photutils.psf

from . import photometry
from . import utils


def run_psfex(
    image,
    mask=None,
    thresh=2.0,
    aper=None,
    r0=0.0,
    gain=1,
    minarea=5,
    vignet_size=None,
    order=0,
    sex_extra={},
    checkimages=[],
    extra={},
    psffile=None,
    get_obj=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    _sex_exe=None,
    verbose=False,
):
    """Wrapper around PSFEx to help extracting PSF models from images.

    For the details of PSFEx operation we suggest to consult its documentation at https://psfex.readthedocs.io

    :param image: Input image as a NumPy array
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param thresh: Detection threshold in sigmas above local background, for running initial SExtractor object detection
    :param aper: Circular aperture radius in pixels, to be used for PSF normalization. Should contain most of object flux. If not specified, will be estimated as twice the FWHM
    :param r0: Smoothing kernel size (sigma) to be used for improving object detection in initial SExtractor call
    :param gain: Image gain
    :param minarea: Minimal number of pixels in the object to be considered a detection (`DETECT_MINAREA` parameter of SExtractor)
    :param vignet_size: The size of *postage stamps* to be used for PSF model creation
    :param order: Spatial order of PSF model variance
    :param sex_extra: Dictionary of additional options to be passed to SExtractor for initial object detection (`extra` parameter of :func:`stdpipe.photometry.get_objects_sextractor`). Optional
    :param checkimages: List of PSFEx checkimages to return along with PSF model. Optional.
    :param extra: Dictionary of extra configuration parameters to be passed to PSFEx call, with keys as parameter names. See :code:`psfex -dd` for the full list.
    :param psffile: If specified, PSF model file will also be stored under this file name, so that it may e.g. be re-used by SExtractor later. Optional
    :param get_obj: If set, also return the table with SExtractor detected objects.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SExtractor and PSFEx. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to PSFEx executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param _sex_exe: Full path to SExtractor executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: PSF structure corresponding to the built PSFEx model.

    The structure has at least the following fields:

    - `width`, `height` - dimesnions of supersampled PSF stamp
    - `fwhm` - mean full width at half maximum (FWHM) of the images used for building the PSF model
    - `sampling` - conversion factor between PSF stamp (supersampled) pixel size, and original image one (less than unity when supersampled resolution is finer than original image one)
    - `ncoeffs` - number of coefficients pixel polynomials have
    - `degree` - polynomial degree of a spatial variance of PSF model
    - `data` - the data containing per-pixel polynomial coefficients for PSF model
    - `header` - original FITS header of PSF model file, if :code:`get_header=True` parameter was set

    This structure corresponds to the contents of original PSFEx generated output file that
    is documented at https://psfex.readthedocs.io/en/latest/Appendices.html#psf-file-format-description

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
        # Find PSFEx binary in common paths
        for exe in ['psfex']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find PSFEx binary")
        return None
    # else:
    #     log("Using PSFEx binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='psfex', dir=_tmpdir)
    )
    psf = None

    # Estimate image FWHM if aperture radius is not set
    if not aper:
        log('Aperture size not specified, will estimate it from image FWHM')
        obj = photometry.get_objects_sextractor(
            image,
            mask=mask,
            thresh=thresh,
            aper=3.0,
            r0=r0,
            gain=gain,
            minarea=minarea,
            _workdir=workdir,
            _tmpdir=_tmpdir,
            _exe=_sex_exe,
            verbose=verbose,
            extra=sex_extra,
        )
        fwhm = np.median(obj['fwhm'][obj['flags'] == 0])
        aper = 2.0*fwhm
        log('FWHM = %.1f pixels, will use aperture radius %.1f pixels' % (fwhm, aper))

    if vignet_size is None:
        vignet_size = 6 * aper + 1
        log(
            'Extracting PSF using vignette size %d x %d pixels'
            % (vignet_size, vignet_size)
        )

    # Run SExtractor on input image in current workdir so that the LDAC catalogue will be in out.cat there
    obj = photometry.get_objects_sextractor(
        image,
        mask=mask,
        thresh=thresh,
        aper=aper,
        r0=r0,
        gain=gain,
        minarea=minarea,
        _workdir=workdir,
        _tmpdir=_tmpdir,
        _exe=_sex_exe,
        verbose=verbose,
        extra_params=[
            'SNR_WIN',
            'ELONGATION',
            'VIGNET(%d,%d)' % (vignet_size, vignet_size),
        ],
        extra=sex_extra,
    )

    catname = os.path.join(workdir, 'out.cat')
    psfname = os.path.join(workdir, 'out.psf')

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'CHECKPLOT_TYPE': 'NONE',
        'CHECKIMAGE_TYPE': 'NONE',
        'PSFVAR_DEGREES': order,
        'WRITE_XML': 'N',
    }

    checknames = [
        os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages
    ]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    opts.update(extra)

    # Build the command line
    cmd = (
        binname + ' ' + shlex.quote(catname) + ' ' + utils.format_astromatic_opts(opts)
    )
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log('Will run PSFEx like that:')
    log(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(psfname):
        log('PSFEx run succeeded')

        psf = load_psf(psfname, verbose=verbose)

        if psffile is not None:
            shutil.copyfile(psfname, psffile)
            log("PSF model stored to", psffile)

    else:
        log("Error", res, "running PSFEx")

    result = psf

    if checkimages:
        result = [result]

        for name in checknames:
            checkname = os.path.splitext(name)[0] + '_out.fits'
            result.append(fits.getdata(checkname))

    if get_obj:
        if type(result) != list:
            result = [result]

        result.append(obj)

    if _workdir is None:
        shutil.rmtree(workdir)

    return result


def load_psf(filename, get_header=False, verbose=False):
    """Load PSF model from PSFEx file

    The structure may be useful for inspection of PSF model with :func:`stdpipe.psf.get_supersampled_psf_stamp` and :func:`stdpipe.psf.get_psf_stamp`, as well as for injection of PSF instances (fake objects) into the image with :func:`stdpipe.psf.place_psf_stamp`.

    :param filename: Name of a file containing PSF model built by PSFEx
    :param get_header: Whether to return the original FITS header of PSF model file or not. If set, the header will be stored in `header` field of the returned structire
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: PSF structure in the same format as returned from :func:`stdpipe.psf.run_psfex`.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    log('Loading PSF model from %s' % filename)

    data = fits.getdata(filename, 1)
    header = fits.getheader(filename, 1)

    psf = {
        'width': header.get('PSFAXIS1'),
        'height': header.get('PSFAXIS2'),
        'ncoeffs': header.get('PSFAXIS3'),
        'fwhm': header.get('PSF_FWHM'),
        'sampling': header.get('PSF_SAMP'),
        'degree': header.get('POLDEG1', 0),
        'x0': header.get('POLZERO1', 0),
        'sx': header.get('POLSCAL1', 1),
        'y0': header.get('POLZERO2', 0),
        'sy': header.get('POLSCAL2', 1),
    }

    if get_header:
        psf['header'] = header

    psf['data'] = data[0][0]

    log(
        'PSF model %d x %d pixels, FWHM %.1f pixels, sampling %.2f, degree %d'
        % (psf['width'], psf['height'], psf['fwhm'], psf['sampling'], psf['degree'])
    )

    return psf


def bilinear_interpolate(im, x, y):
    """
    Quick and dirty bilinear interpolation
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def get_supersampled_psf_stamp(psf, x=0, y=0, normalize=True):
    """Returns supersampled PSF model for a given position inside the image.

    The returned stamp corresponds to PSF model evaluated at a given position inside the image,
    with its center always in the center of central stamp pixel.
    Every *supersampled* pixel of the stamp corresponds to :code:`psf['sampling']` pixels of the original image.

    :param psf: Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or :func:`stdpipe.psf.load_psf`
    :param x: `x` coordinate of the position inside the original image to evaluate the PSF model
    :param y: `y` coordinate of the position inside the original image to evaluate the PSF model
    :param normalize: Whether to normalize the stamp to have flux exactly equal to unity or not
    :returns: stamp of the PSF model evaluated at the given position inside the image

    """

    dx = 1.0 * (x - psf['x0']) / psf['sx']
    dy = 1.0 * (y - psf['y0']) / psf['sy']

    stamp = np.zeros(psf['data'].shape[1:], dtype=np.double)
    i = 0

    for i2 in range(0, psf['degree'] + 1):
        for i1 in range(0, psf['degree'] + 1 - i2):
            stamp += psf['data'][i] * dx ** i1 * dy ** i2
            i += 1

    if normalize:
        stamp /= np.sum(stamp)

    return stamp


def get_psf_stamp(psf, x=0, y=0, dx=None, dy=None, normalize=True):
    """Returns PSF stamp in original image pixel space with sub-pixel shift applied.

    The PSF model is evaluated at the requested position inside the original image,
    and then downscaled from supersampled pixels of the PSF model to original image pixels,
    and then adjusted to accommodate for requested :code:`(dx, dy)` sub-pixel shift.

    Stamp is odd-sized, with PSF center at::

        x0 = floor(width/2) + dx
        y0 = floor(height/2) + dy

    If :code:`dx=None` or :code:`dy=None`, they are computed directly from the
    floating point parts of the position `x` and `y` arguments::

        dx = x - round(x)
        dy = y - round(y)

    The stamp should directly represent stellar shape at a given position (including sub-pixel
    center shift) inside the image.

    :param psf: Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or :func:`stdpipe.psf.load_psf`
    :param x: `x` coordinate of the position inside the original image to evaluate the PSF model
    :param y: `y` coordinate of the position inside the original image to evaluate the PSF model
    :param dx: Sub-pixel adjustment of PSF position in image space, `x` direction
    :param dy: Sub-pixel adjustment of PSF position in image space, `y` direction
    :param normalize: Whether to normalize the stamp to have flux exactly equal to unity or not
    :returns: Stamp of the PSF model evaluated at the given position inside the image, in original image pixels.

    """

    if dx is None:
        dx = x % 1
    if dy is None:
        dy = y % 1

    supersampled = get_supersampled_psf_stamp(psf, x, y, normalize=normalize)

    # Supersampled stamp center, assuming odd-sized shape
    ssx0 = np.floor(supersampled.shape[1] / 2)
    ssy0 = np.floor(supersampled.shape[0] / 2)

    # Make odd-sized array to hold the result
    x0 = np.floor(psf['width'] * psf['sampling'] / 2)
    y0 = np.floor(psf['height'] * psf['sampling'] / 2)

    width = int(x0) * 2 + 1
    height = int(y0) * 2 + 1

    x0 += dx
    y0 += dy

    # Coordinates in resulting stamp
    y, x = np.mgrid[0:height, 0:width]

    # The same grid in supersampled space, shifted accordingly
    x1 = ssx0 + (x - x0) / psf['sampling']
    y1 = ssy0 + (y - y0) / psf['sampling']

    # FIXME: it should really be Lanczos interpolation here!
    # stamp = bilinear_interpolate(supersampled, x1, y1) / psf['sampling'] ** 2
    stamp = ndimage.map_coordinates(supersampled, [y1, x1], order=3) / psf['sampling'] ** 2

    if normalize:
        stamp /= np.sum(stamp)

    return stamp


def place_psf_stamp(image, psf, x0, y0, flux=1, gain=None):
    """Places PSF stamp, scaled to a given flux, at a given position inside the image.

    PSF stamp is evaluated at a given position, then adjusted to accommodate for
    required sub-pixel shift, and finally scaled to requested flux value. Thus,
    the routine corresponds to injection of an artificial point source into the image.

    The stamp values are added on top of current content of the image.
    If `gain` value is set, the Poissonian noise is applied to the stamp.

    The image is modified in-place.

    :param image: The image where artifi
    :param psf: Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or :func:`stdpipe.psf.load_psf`
    :param x0: `x` coordinate of the position to inject the source
    :param y0: `y` coordinate of the position to inject the source
    :param flux: The source flux in ADU units
    :param gain: Image gain value. If set, used to apply Poissonian noise to the source.

    """

    stamp = get_psf_stamp(psf, x0, y0, normalize=True)
    stamp *= flux

    if gain is not None:
        idx = stamp > 0
        # FIXME: what to do with negative points?..
        stamp[idx] = np.random.poisson(stamp[idx] * gain) / gain

    # Integer coordinates inside the stamp
    y, x = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]]

    # Corresponding image pixels
    y1, x1 = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]]
    x1 += int(np.round(x0) - np.floor(stamp.shape[1] / 2))
    y1 += int(np.round(y0) - np.floor(stamp.shape[0] / 2))

    # Crop the coordinates outside target image
    idx = np.isfinite(stamp)
    idx &= (x1 >= 0) & (x1 < image.shape[1])
    idx &= (y1 >= 0) & (y1 < image.shape[0])

    # Add the stamp to the image!
    image[y1[idx], x1[idx]] += stamp[y[idx], x[idx]]


def create_psf_model(image, obj=None, fwhm=None, size=25, mask=None,
                     oversampling=2, get_raw=False, verbose=False):
    """
    Create an empirical PSF (ePSF) model from stars in the image using photutils.

    This function builds an effective PSF by combining postage stamps of
    isolated stars in the image. This is useful when you don't have a
    PSFEx model or want a purely empirical PSF.

    The returned dictionary structure is compatible with PSFEx output from
    :func:`stdpipe.psf.run_psfex` and can be used with the same evaluation
    functions like :func:`stdpipe.psf.get_psf_stamp`.

    :param image: Input image as a NumPy array, must be background subtracted
    :param obj: Table of star positions. If None, stars will be detected automatically. Should have 'x', 'y' columns and optionally 'flux'.
    :param fwhm: Approximate FWHM of stars in pixels. If None, will be estimated.
    :param size: Size of cutouts to extract around stars (should be odd)
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param oversampling: Oversampling factor for the ePSF (default: 2)
    :param get_raw: If True, returns raw EPSFImage object
    :param verbose: Whether to show verbose messages
    :returns: Dictionary with PSFEx-compatible structure containing the ePSF model

    """

    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if size % 2 == 0:
        size += 1  # Make sure size is odd

    # Detect stars if not provided
    if obj is None:
        log('Detecting stars for ePSF building')
        obj = photometry.get_objects_sep(
            image,
            mask=mask,
            thresh=5.0,
            aper=3.0,
            verbose=verbose
        )

        # Select isolated, bright, non-saturated stars
        # Simple selection: median flux and not too crowded
        if len(obj) == 0:
            raise ValueError("No stars detected for ePSF building")

        flux_median = np.median(obj['flux'])
        flux_std = np.std(obj['flux'])

        # Select stars with flux within reasonable range
        idx = (obj['flux'] > flux_median) & (obj['flux'] < flux_median + 3*flux_std)
        # Remove edge objects
        edge = size
        idx &= (obj['x'] > edge) & (obj['x'] < image.shape[1] - edge)
        idx &= (obj['y'] > edge) & (obj['y'] < image.shape[0] - edge)
        # Remove flagged objects
        if 'flags' in obj.colnames:
            idx &= obj['flags'] == 0

        obj = obj[idx]
        log('Selected %d stars for ePSF building' % len(obj))

    if fwhm is None:
        if 'fwhm' in obj.colnames:
            fwhm = np.median(obj['fwhm'])
            log('Using median FWHM: %.2f pixels' % fwhm)
        else:
            fwhm = 3.0
            log('FWHM not available, using default: %.2f pixels' % fwhm)

    # Extract cutouts
    log('Extracting %dx%d cutouts around %d stars' % (size, size, len(obj)))

    nddata = NDData(data=image, mask=mask)

    # Extract stars using photutils
    stars = photutils.psf.extract_stars(nddata, Table({'x': obj['x'], 'y': obj['y']}), size=size)

    # Build ePSF
    log('Building ePSF model with oversampling=%d' % oversampling)
    epsf_builder = photutils.psf.EPSFBuilder(
        oversampling=oversampling,
        maxiters=10,
        progress_bar=verbose
    )

    epsf, fitted_stars = epsf_builder(stars)

    log('ePSF building complete. PSF shape: %s' % str(epsf.data.shape))

    if get_raw:
        return epsf

    # Build PSFEx-compatible structure
    psf_data = epsf.data

    # Reshape to (ncoeffs, height, width) format
    # ePSF is position-invariant, so ncoeffs=1
    if psf_data.ndim == 2:
        psf_data = psf_data[np.newaxis, :, :]  # Add coefficient dimension

    psf = {
        'width': psf_data.shape[2],
        'height': psf_data.shape[1],
        'fwhm': fwhm,
        'sampling': 1.0 / oversampling,  # PSFEx convention: < 1 means supersampled
        'ncoeffs': 1,
        'degree': 0,  # Constant PSF (no position dependence)
        'x0': 0,
        'y0': 0,
        'sx': 1,
        'sy': 1,
        'data': psf_data,
        'oversampling': oversampling,  # Keep for reference
        'type': 'epsf',  # Identify PSF type
    }

    return psf
