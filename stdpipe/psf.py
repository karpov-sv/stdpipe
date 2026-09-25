"""
Module for working with point-spread function (PSF) models
"""

import os, shutil, tempfile, shlex
import warnings
import numpy as np

from astropy.io import fits
from astropy.table import Table

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
    psf_size=None,
    order=0,
    sex_extra=None,
    checkimages=None,
    extra=None,
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

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a NumPy array.
    mask : numpy.ndarray, optional
        Image mask as a boolean array (True values will be masked).
    thresh : float, optional
        Detection threshold in sigmas above local background, for running initial SExtractor object detection.
    aper : float, optional
        Circular aperture radius in pixels, to be used for PSF normalization. Should contain most of
        object flux. If not specified, will be estimated as twice the FWHM.
    r0 : float, optional
        Smoothing kernel size (sigma) to be used for improving object detection in initial SExtractor call.
    gain : float, optional
        Image gain.
    minarea : int, optional
        Minimal number of pixels in the object to be considered a detection (``DETECT_MINAREA`` parameter
        of SExtractor).
    vignet_size : int, optional
        The size of *postage stamps* to be used for PSF model creation.
    psf_size : int, optional
        The size of the supersampled PSF model.
    order : int, optional
        Spatial order of PSF model variance.
    sex_extra : dict, optional
        Dictionary of additional options to be passed to SExtractor for initial object detection
        (``extra`` parameter of :func:`stdpipe.photometry.get_objects_sextractor`).
    checkimages : list, optional
        List of PSFEx checkimages to return along with PSF model.
    extra : dict, optional
        Dictionary of extra configuration parameters to be passed to PSFEx call, with keys as
        parameter names. See :code:`psfex -dd` for the full list.
    psffile : str, optional
        If specified, PSF model file will also be stored under this file name, so that it may e.g.
        be re-used by SExtractor later.
    get_obj : bool, optional
        If set, also return the table with SExtractor detected objects.
    _workdir : str, optional
        If specified, all temporary files will be created in this directory, and will be kept intact
        after running SExtractor and PSFEx. May be used for debugging exact inputs and outputs of
        the executable.
    _tmpdir : str, optional
        If specified, all temporary files will be created in a dedicated directory (that will be
        deleted after running the executable) inside this path.
    _exe : str, optional
        Full path to PSFEx executable. If not provided, the code tries to locate it automatically
        in your :envvar:`PATH`.
    _sex_exe : str, optional
        Full path to SExtractor executable. If not provided, the code tries to locate it automatically
        in your :envvar:`PATH`.
    verbose : bool or callable, optional
        Whether to show verbose messages during the run of the function or not.

    Returns
    -------
    dict
        PSF structure corresponding to the built PSFEx model.

        The structure has at least the following fields:

        - ``width``, ``height`` - dimensions of supersampled PSF stamp
        - ``fwhm`` - mean full width at half maximum (FWHM) of the images used for building the PSF model
        - ``sampling`` - conversion factor between PSF stamp (supersampled) pixel size, and original image
          one (less than unity when supersampled resolution is finer than original image one)
        - ``ncoeffs`` - number of coefficients pixel polynomials have
        - ``degree`` - polynomial degree of a spatial variance of PSF model
        - ``data`` - the data containing per-pixel polynomial coefficients for PSF model
        - ``header`` - original FITS header of PSF model file, if :code:`get_header=True` parameter was set

        This structure corresponds to the contents of original PSFEx generated output file that
        is documented at https://psfex.readthedocs.io/en/latest/Appendices.html#psf-file-format-description

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='psfex', dir=_tmpdir)
    psf = None

    # Estimate image FWHM if aperture radius is not set
    if sex_extra is None:
        sex_extra = {}
    if checkimages is None:
        checkimages = []
    if extra is None:
        extra = {}

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
        fwhm_vals = obj['fwhm'][obj['flags'] == 0]
        if len(fwhm_vals) > 0:
            fwhm = np.median(fwhm_vals)
        else:
            fwhm = np.nan
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = 3.0
            log('FWHM estimate failed, using default %.1f pixels' % fwhm)
        aper = 2.0 * fwhm
        log('FWHM = %.1f pixels, will use aperture radius %.1f pixels' % (fwhm, aper))

    if vignet_size is None:
        vignet_size = int(np.ceil(6 * aper)) + 1
    else:
        vignet_size = int(np.round(vignet_size))
    if vignet_size % 2 == 0:
        vignet_size += 1
    log('Extracting PSF using vignette size %d x %d pixels' % (vignet_size, vignet_size))

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

    checknames = [os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    opts.update(extra)

    if psf_size is not None:
        opts['PSF_SIZE'] = [psf_size, psf_size]

    # Build the command line
    cmd = binname + ' ' + shlex.quote(catname) + ' ' + utils.format_astromatic_opts(opts)
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log('Will run PSFEx like that:')
    log(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(psfname):
        log('PSFEx run succeeded')

        psf = load_psf(psfname, verbose=verbose)

        # Check whether the PSF model extends beyond the vignet coverage.
        # PSFEx uses Lanczos interpolation with renormalization at boundaries
        # when resampling vignets into the PSF grid. When the PSF model is
        # larger than the vignet (in image pixels), the boundary pixels get
        # spurious non-zero values from the renormalized partial kernel,
        # which biases the PSF wings and corrupts flux measurements.
        psf_extent = psf['width'] * psf['sampling']
        if psf_extent > vignet_size:
            import warnings

            warnings.warn(
                "PSF model extent (%.0f x %.0f pixels = psf_size %d x sampling %.3f) "
                "exceeds vignet size (%d pixels). This causes interpolation artifacts "
                "in the PSF wings that bias flux measurements. "
                "Either increase vignet_size to >= %.0f or decrease psf_size to <= %d."
                % (
                    psf_extent,
                    psf_extent,
                    psf['width'],
                    psf['sampling'],
                    vignet_size,
                    psf_extent,
                    int(vignet_size / psf['sampling']),
                )
            )

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

    Parameters
    ----------
    filename : str
        Name of a file containing PSF model built by PSFEx.
    get_header : bool, optional
        Whether to return the original FITS header of PSF model file or not. If set,
        the header will be stored in the ``header`` field of the returned structure.
    verbose : bool or callable, optional
        Whether to show verbose messages during the run of the function or not.

    Returns
    -------
    dict
        PSF structure in the same format as returned from :func:`stdpipe.psf.run_psfex`.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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
        # PSFEx samples the pixel-integrated PSF, it doesn't integrate over sub-pixels
        'subpixel_integrated': False,
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

    Parameters
    ----------
    psf : dict
        Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or
        :func:`stdpipe.psf.load_psf`.
    x : float, optional
        ``x`` coordinate of the position inside the original image to evaluate the PSF model.
    y : float, optional
        ``y`` coordinate of the position inside the original image to evaluate the PSF model.
    normalize : bool, optional
        Whether to normalize the stamp to have flux exactly equal to unity or not.

    Returns
    -------
    numpy.ndarray
        Stamp of the PSF model evaluated at the given position inside the image.

    """

    dx = 1.0 * (x - psf['x0']) / psf['sx']
    dy = 1.0 * (y - psf['y0']) / psf['sy']

    stamp = np.zeros(psf['data'].shape[1:], dtype=np.double)
    i = 0

    for i2 in range(0, psf['degree'] + 1):
        for i1 in range(0, psf['degree'] + 1 - i2):
            stamp += psf['data'][i] * dx**i1 * dy**i2
            i += 1

    if normalize:
        total = np.sum(stamp)
        if np.isfinite(total) and total > 0:
            stamp /= total

    return stamp


def _psf_center(psf, shape):
    """Center ``(x, y)`` of supersampled PSF model grid of a given shape.

    Point-sampled models (PSFEx convention) have their center at pixel
    ``size // 2`` also for even sizes, while sub-pixel integrated ones
    (and models without declared convention) at ``(size - 1) / 2``.
    """

    h, w = shape
    if psf.get('subpixel_integrated', None) is False:
        return float(w // 2), float(h // 2)
    return (w - 1) / 2.0, (h - 1) / 2.0


def _get_sampled_psf_stamp(psf, x=0, y=0):
    """Supersampled PSF stamp as point samples of the pixel-integrated PSF.

    Sub-pixel integrated models are converted by integrating, for every
    model pixel, the flux over the image pixel centered on it. The cumulative
    flux is known exactly at model pixel edges, and is interpolated with a
    cubic spline where image pixel edges fall inside model pixels (even
    oversampling factors). Returns the stamp normalized to unit sum, and its
    center ``(x, y)``.
    """

    from scipy.interpolate import CubicSpline

    stamp = get_supersampled_psf_stamp(psf, x, y, normalize=True)
    center = _psf_center(psf, stamp.shape)

    N = int(round(1.0 / psf['sampling']))
    if (
        psf.get('subpixel_integrated', False)
        and N > 1
        and np.isclose(N * psf['sampling'], 1.0, rtol=1e-3, atol=0)
    ):
        for axis in (0, 1):
            n = stamp.shape[axis]
            # Cumulative flux at model pixel edges 0..n
            shape = list(stamp.shape)
            shape[axis] = 1
            cumul = np.concatenate([np.zeros(shape), np.cumsum(stamp, axis=axis)], axis=axis)
            spline = CubicSpline(np.arange(n + 1), cumul, axis=axis)
            # Image pixel centered on model pixel k spans k + 0.5 -+ N / 2 in edge coordinates
            centers = np.arange(n) + 0.5
            lo = np.clip(centers - N / 2, 0, n)
            hi = np.clip(centers + N / 2, 0, n)
            stamp = spline(hi) - spline(lo)
        total = np.sum(stamp)
        if np.isfinite(total) and total > 0:
            stamp /= total

    return stamp, center


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

    Supersampled model pixels are interpreted according to the
    ``psf['subpixel_integrated']`` key. If True, every model pixel holds the
    flux integrated over its own (sub-pixel) area, and image pixels are
    obtained by summing blocks of them. If False, model pixels are point
    samples of the PSF already integrated over image pixels (PSFEx and
    :func:`stdpipe.psf.create_psf_model` convention), and image pixels are
    obtained by interpolation. If the key is missing, block summing is used
    whenever the model size is a multiple of an integer oversampling factor.
    The center of point-sampled models of even size is at pixel ``size // 2``
    (PSFEx convention), otherwise at ``(size - 1) / 2``.

    Parameters
    ----------
    psf : dict
        Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or
        :func:`stdpipe.psf.load_psf`.
    x : float, optional
        ``x`` coordinate of the position inside the original image to evaluate the PSF model.
    y : float, optional
        ``y`` coordinate of the position inside the original image to evaluate the PSF model.
    dx : float, optional
        Sub-pixel adjustment of PSF position in image space, ``x`` direction.
    dy : float, optional
        Sub-pixel adjustment of PSF position in image space, ``y`` direction.
    normalize : bool, optional
        Whether to normalize the stamp to have flux exactly equal to unity or not.

    Returns
    -------
    numpy.ndarray
        Stamp of the PSF model evaluated at the given position inside the image, in original
        image pixels.

    """

    if dx is None:
        dx = x - np.round(x)
    if dy is None:
        dy = y - np.round(y)

    supersampled = get_supersampled_psf_stamp(psf, x, y, normalize=normalize)

    # Oversampling factor
    N = int(round(1.0 / psf['sampling']))

    if (
        psf.get('subpixel_integrated', True)
        and N > 1
        # Block-summing N x N subpixels assumes each subpixel is exactly 1/N
        # image pixels; a non-integer 1/sampling (e.g. PSFEx PSF_SAMP=0.53)
        # would silently rescale the PSF by sampling*N
        and np.isclose(N * psf['sampling'], 1.0, rtol=1e-3, atol=0)
        and supersampled.shape[0] % N == 0
        and supersampled.shape[1] % N == 0
    ):
        # Flux-conserving downsampling: shift at oversampled resolution, then sum N×N blocks.
        # The oversampled data stores pixel-integrated flux per subpixel, so the correct
        # way to get image-pixel flux is to sum the subpixels within each image pixel.

        # Apply sub-pixel shift at oversampled resolution via cubic interpolation.
        # At the oversampled resolution the PSF is well-sampled, so cubic interpolation
        # accurately reconstructs the continuous PSF for sub-pixel shifts.
        shift_x_os = dx / psf['sampling']
        shift_y_os = dy / psf['sampling']

        if shift_x_os != 0 or shift_y_os != 0:
            shifted = ndimage.shift(
                supersampled, [shift_y_os, shift_x_os], order=3, mode='constant', cval=0
            )
        else:
            shifted = supersampled

        # Sum N×N blocks
        out_h = supersampled.shape[0] // N
        out_w = supersampled.shape[1] // N
        stamp = shifted[: out_h * N, : out_w * N].reshape(out_h, N, out_w, N).sum(axis=(1, 3))
    else:
        # Fallback for non-integer oversampling or oversampling=1
        ssx0, ssy0 = _psf_center(psf, supersampled.shape)

        x0 = np.floor(psf['width'] * psf['sampling'] / 2)
        y0 = np.floor(psf['height'] * psf['sampling'] / 2)

        width = int(x0) * 2 + 1
        height = int(y0) * 2 + 1

        x0 += dx
        y0 += dy

        y, x = np.mgrid[0:height, 0:width]

        x1 = ssx0 + (x - x0) / psf['sampling']
        y1 = ssy0 + (y - y0) / psf['sampling']

        stamp = ndimage.map_coordinates(supersampled, [y1, x1], order=3) / psf['sampling'] ** 2

    if normalize:
        total = np.sum(stamp)
        if np.isfinite(total) and total > 0:
            stamp /= total

    return stamp


def place_psf_stamp(image, psf, x0, y0, flux=1, gain=None):
    """Places PSF stamp, scaled to a given flux, at a given position inside the image.

    PSF stamp is evaluated at a given position, then adjusted to accommodate for
    required sub-pixel shift, and finally scaled to requested flux value. Thus,
    the routine corresponds to injection of an artificial point source into the image.

    The stamp values are added on top of current content of the image.
    If `gain` value is set, the Poissonian noise is applied to the stamp.

    The image is modified in-place.

    Parameters
    ----------
    image : numpy.ndarray
        The image where the artificial source will be injected (modified in-place).
    psf : dict
        Input PSF structure as returned by :func:`stdpipe.psf.run_psfex` or
        :func:`stdpipe.psf.load_psf`.
    x0 : float
        ``x`` coordinate of the position to inject the source.
    y0 : float
        ``y`` coordinate of the position to inject the source.
    flux : float, optional
        The source flux in ADU units.
    gain : float, optional
        Image gain value. If set, used to apply Poissonian noise to the source.

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


def _estimate_psf_stamp_size(image, obj, fwhm, mask=None, log=None, nstars=100, snr=3.0):
    """Estimate PSF stamp size from the extent of significant stellar wings.

    Stacks radial profiles of the brightest stars, normalized by their core
    flux, and returns an odd size covering the radius where the median
    profile stops being significant, bounded by ``max(15, 5 * fwhm)`` and
    ``10 * fwhm``.
    """

    min_size = max(15, int(np.ceil(5 * fwhm)))
    max_size = max(min_size, int(np.ceil(10 * fwhm)))
    min_size += 1 - min_size % 2
    max_size += 1 - max_size % 2
    half = max_size // 2

    x = np.asarray(obj['x'], dtype=np.float64)
    y = np.asarray(obj['y'], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    ok &= (x >= half) & (x < image.shape[1] - half - 1) & (y >= half) & (y < image.shape[0] - half - 1)
    idx = np.flatnonzero(ok)
    if 'flux' in obj.colnames:
        idx = idx[np.argsort(-np.asarray(obj['flux'], dtype=np.float64)[idx])]
    idx = idx[:nstars]
    if len(idx) < 5:
        return min_size

    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    radii, values = [], []
    for i in idx:
        ix, iy = int(np.round(x[i])), int(np.round(y[i]))
        cutout = image[iy - half : iy + half + 1, ix - half : ix + half + 1].astype(np.float64)
        valid = np.isfinite(cutout)
        if mask is not None:
            valid &= ~mask[iy - half : iy + half + 1, ix - half : ix + half + 1]
        r = np.hypot(xx + ix - x[i], yy + iy - y[i])
        border = valid & (r > half - 1)
        if not np.any(border):
            continue
        cutout = cutout - np.median(cutout[border])
        core = np.sum(cutout[valid & (r < 1.5 * fwhm)])
        if not np.isfinite(core) or core <= 0:
            continue
        radii.append(r[valid])
        values.append(cutout[valid] / core)

    if len(radii) < 5:
        return min_size

    radii = np.concatenate(radii)
    values = np.concatenate(values)
    ibin = np.floor(radii / 0.5).astype(int)
    med, scale, count = _group_robust_stats(ibin, values, int(ibin.max()) + 1)
    centers = (np.arange(len(med)) + 0.5) * 0.5
    with np.errstate(invalid='ignore', divide='ignore'):
        err = 1.2533 * scale / np.sqrt(count)
    faint = (centers > 1.25 * fwhm) & (count >= 5) & ~(med > snr * err)
    radius = centers[np.argmax(faint)] if np.any(faint) else half

    size = int(np.clip(2 * int(np.ceil(radius)) + 1, min_size, max_size))
    if log is not None:
        log(
            'Stacked profile of %d stars significant up to %.1f px, stamp size %d'
            % (len(idx), radius, size)
        )

    return size


def create_psf_model(
    image,
    obj=None,
    fwhm=None,
    size=None,
    mask=None,
    oversampling=None,
    degree=0,
    regularization=1e-6,
    subtract_neighbors=True,
    neighbors_obj=None,
    subtract_background=False,
    isolation=2.0,
    maxiters=5,
    max_degree=3,
    get_raw=False,
    verbose=False,
):
    """
    Create an empirical PSF (ePSF) model from stars in the image.

    Star pixels are placed onto an oversampled grid at their true sub-pixel
    offsets from the star centers (Anderson & King 2000 ePSF approach), and
    per-pixel polynomial coefficients are fitted to them, similarly to PSFEx.
    For ``degree=0`` (default) the model is position-invariant, i.e. a
    weighted mean of the stars; for ``degree > 0`` it is position-dependent:

    .. math::

        PSF(i,j; x,y) = \\sum_k c_k(i,j) \\cdot dx^{p1_k} \\cdot dy^{p2_k}

    where ``(dx, dy)`` are normalized image coordinates and ``(p1_k, p2_k)``
    are polynomial exponents with ``p1_k + p2_k <= degree``.

    The fit is iterated: every star is fitted with the current model for a
    sub-pixel shift and an amplitude, then its pixels are placed again at the
    refined center and normalized by the fitted amplitude. Outliers are
    rejected both per star and per pixel, so that a star with a local defect
    (cosmic ray, poorly subtracted neighbour) still contributes its clean
    pixels. Outer model pixels where the stacked profile is not significant
    are smoothly tapered to zero.

    The returned dictionary structure is compatible with PSFEx output from
    :func:`stdpipe.psf.run_psfex` and can be used with the same evaluation
    functions like :func:`stdpipe.psf.get_psf_stamp`.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a NumPy array, must be background subtracted.
    obj : astropy.table.Table, optional
        Table of star positions. If None, stars will be detected automatically.
        Should have 'x', 'y' columns and optionally 'flux'.
    fwhm : float, optional
        Approximate FWHM of stars in pixels. If None, will be estimated.
    size : int, optional
        Size of cutouts to extract around stars (should be odd). If None, it is
        determined from the radius where the stacked radial profile of the
        brightest stars stops being significant, between
        ``max(15, 5 * fwhm)`` and ``10 * fwhm`` (rounded up to odd).
    mask : numpy.ndarray, optional
        Image mask as a boolean array (True values will be masked).
    oversampling : int, optional
        Oversampling factor for the ePSF. If None (default), it is auto-selected from
        the FWHM: ``1`` when ``fwhm >= 2.5`` image pixels (well-sampled PSF) and
        ``2`` otherwise (under-sampled PSF). Pass an explicit integer to override.
    degree : int or 'auto', optional
        Polynomial degree for spatial PSF variation (default: 0 = constant). Degree 1 =
        linear (3 coefficients), degree 2 = quadratic (6 coefficients), etc. If
        ``'auto'``, the degree (up to ``max_degree``, and to what the number of
        stars supports at 5 stars per coefficient) is selected by spatial
        cross-validation: the lowest degree predicting the shapes of stars left
        out of the fit not worse than the best one by more than one standard
        error. Per-degree scores are then stored in ``degree_selection`` entry
        of the returned dictionary. The choice is conservative: on simulated
        fields it picks the lower of two nearly equivalent degrees (at most
        ~0.2% rms PSF flux accuracy lost), while never selecting degrees that
        the data cannot constrain.
    regularization : float, optional
        Tikhonov regularization parameter for polynomial fitting (default: 1e-6). Only used
        when ``degree > 0``. Set to 0 for unregularized least-squares.
    subtract_neighbors : bool, optional
        If True (default), subtract estimated flux from neighboring stars before extracting
        cutouts. Reduces contamination in crowded fields.
    neighbors_obj : astropy.table.Table, optional
        Full detection catalogue (with 'x', 'y', 'flux' columns) used to model
        neighbour contamination when ``subtract_neighbors=True``. If None, the
        full auto-detected list is used when ``obj`` is None, otherwise ``obj``
        itself. Passing the complete catalogue matters when ``obj`` is a
        pre-selected subset (e.g. from :func:`select_psf_seeds`) — the stars
        that actually contaminate the stamps are usually not in that subset.
    subtract_background : bool or {'none', 'median', 'plane'}, optional
        Local background handling for each training stamp before normalization. ``False`` or
        ``'none'`` leaves the current image values unchanged, ``True`` or ``'median'``
        subtracts the median of the stamp border, and ``'plane'`` fits and subtracts a
        tilted background plane from the border pixels.
        When building from the raw image instead of a background-subtracted one,
        ``'plane'`` is usually the most robust choice.
    isolation : float, optional
        Minimum nearest-neighbor distance in FWHM units for selecting stars for ePSF building
        (default: 2.0). Stars with a neighbor closer than ``isolation * fwhm`` are excluded.
        Combined with ``subtract_neighbors=True``, the lower default value works in both
        sparse and dense fields. Set to 0 or None to disable isolation filtering.
    maxiters : int, optional
        Maximal number of iterations refining star centers and normalizations
        (default: 5). Iterations stop earlier once the RMS center correction
        is below 0.005 pixels. Set to 0 to build the model in a single pass at
        catalogue positions, with stamps normalized by their sums.
    max_degree : int, optional
        Highest polynomial degree considered when ``degree='auto'`` (default: 3).
    get_raw : bool, optional
        If True and ``degree=0``, returns the model as photutils ``ImagePSF``
        object. Ignored when ``degree > 0``.
    verbose : bool or callable, optional
        Whether to show verbose messages.

    Returns
    -------
    dict
        Dictionary with PSFEx-compatible structure containing the PSF model.

    """

    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Detect stars if not provided
    if obj is None:
        log('Detecting stars for ePSF building')
        obj = photometry.get_objects_sep(image, mask=mask, thresh=5.0, aper=3.0, verbose=verbose)

        # Select isolated, bright, non-saturated stars
        # Simple selection: median flux and not too crowded
        if len(obj) == 0:
            raise ValueError("No stars detected for ePSF building")

        # Keep the full detection list for neighbour modelling before the
        # quality/isolation selection below removes the real neighbours
        if neighbors_obj is None:
            neighbors_obj = obj

        # Determine FWHM and stamp size early so we can use them for edge filtering
        if fwhm is None:
            if 'fwhm' in obj.colnames:
                fwhm = np.median(obj['fwhm'])
                log('Using median FWHM: %.2f pixels' % fwhm)
            else:
                fwhm = 3.0
                log('FWHM not available, using default: %.2f pixels' % fwhm)

        flux_median = np.median(obj['flux'])
        # np.std is inflated by the bright tail, so a median + N*std upper
        # bound lets saturated stars through; cut the brightest few percent
        # instead, which is where saturated stars live
        flux_upper = np.percentile(obj['flux'], 98)

        # Select stars with flux within reasonable range
        idx = (obj['flux'] > flux_median) & (obj['flux'] < flux_upper)
        # Remove edge objects
        edge = size if size is not None else max(15, int(np.ceil(5 * fwhm)))
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

    # Auto-pick oversampling from FWHM if not explicitly set.
    # FWHM >= 2.5 image pixels is well-sampled enough that oversampling=1 is
    # adequate (saves model storage and SEP rendering cost). Smaller FWHM
    # (under-sampled PSFs) needs oversampling=2 to avoid pixelisation bias.
    if oversampling is None:
        oversampling = 1 if fwhm >= 2.5 else 2
        log('Auto-selected oversampling=%d (FWHM=%.2f pix)' % (oversampling, fwhm))

    # Filter by isolation: reject stars with a neighbor closer than isolation * fwhm.
    # Neighbor contamination in star stamps is the primary source of ePSF bias
    # in crowded fields; selecting only isolated stars dramatically improves
    # ePSF quality (tested: reduces bias from +30% to -1% at 6 FWHM separation).
    if isolation and isolation > 0 and len(obj) > 1:
        from scipy.spatial import cKDTree

        min_dist = isolation * fwhm
        tree = cKDTree(np.c_[obj['x'], obj['y']])
        nn_dist = tree.query(np.c_[obj['x'], obj['y']], k=2)[0][:, 1]
        isolated = obj[nn_dist > min_dist]
        n_before = len(obj)
        if len(isolated) >= 10:
            obj = isolated
            log(
                'Isolation filter (>%.0f*FWHM = >%.1f px): %d / %d stars selected'
                % (isolation, min_dist, len(obj), n_before)
            )
        else:
            # Not enough isolated stars; fall back to the most isolated ones
            n_fallback = max(10, n_before // 5)
            idx = np.argsort(-nn_dist)[:n_fallback]
            obj = obj[idx]
            log(
                'Isolation filter: only %d stars with >%.1f px separation; '
                'using %d most isolated instead' % (len(isolated), min_dist, len(obj))
            )

    # Auto-size stamps from the extent of significant PSF wings: a model
    # truncated while still carrying flux loses a sub-pixel phase dependent
    # part of it when shifted, and underestimates total fluxes
    if size is None:
        size = _estimate_psf_stamp_size(image, obj, fwhm, mask, log)
    if size % 2 == 0:
        size += 1  # Make sure size is odd
    log('Using stamp size: %d pixels (FWHM=%.1f)' % (size, fwhm))

    background_mode = _normalize_stamp_background_mode(subtract_background)

    psf = _create_psf_model_polynomial(
        image,
        obj,
        fwhm,
        size,
        mask,
        oversampling,
        None if degree == 'auto' else degree,
        regularization,
        subtract_neighbors,
        neighbors_obj if neighbors_obj is not None else obj,
        background_mode,
        maxiters,
        log,
        max_degree=max_degree,
    )

    selection = psf.get('degree_selection')
    if selection is not None and psf['degree'] != selection['registration_degree']:
        # Star centers and normalizations were refined with the highest
        # candidate degree model; rebuild the chosen one from scratch so that
        # it is identical to an explicit build with that degree
        log('Rebuilding PSF model with selected degree %d' % psf['degree'])
        psf = _create_psf_model_polynomial(
            image,
            obj,
            fwhm,
            size,
            mask,
            oversampling,
            psf['degree'],
            regularization,
            subtract_neighbors,
            neighbors_obj if neighbors_obj is not None else obj,
            background_mode,
            maxiters,
            log,
        )
        psf['degree_selection'] = selection

    if get_raw and psf['degree'] == 0:
        # photutils expects oversampled PSF images normalized to oversampling**2
        return photutils.psf.ImagePSF(psf['data'][0] * oversampling**2, oversampling=oversampling)

    return psf


def _build_polynomial_psf_taper(shape, sampling, fwhm, sample_r, sample_v, snr=3.0):
    """Build a smooth radial taper for low-S/N outer PSF pixels.

    The taper starts at the first radius (beyond ``1.25 * fwhm``) where the
    stacked radial profile of the normalized star samples is not
    significantly above zero, so that well-measured extended wings (e.g.
    Moffat) are preserved while noise-dominated outer support is suppressed.
    No taper is applied if the profile stays significant up to the stamp edge.
    """

    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    r = np.sqrt(((x - cx) * sampling) ** 2 + ((y - cy) * sampling) ** 2)

    half_width = 0.5 * (min(h, w) - 1) * sampling
    r_min = max(1.25 * fwhm, sampling)
    r_max = max(r_min, half_width - 2 * sampling)

    # Radial profile of the samples, and the uncertainty of its median
    bin_width = 0.5
    ok = np.isfinite(sample_r) & np.isfinite(sample_v)
    ibin = np.floor(sample_r[ok] / bin_width).astype(int)
    med, scale, count = _group_robust_stats(ibin, sample_v[ok], int(np.max(ibin, initial=0)) + 1)
    centers = (np.arange(len(med)) + 0.5) * bin_width
    with np.errstate(invalid='ignore', divide='ignore'):
        err = 1.2533 * scale / np.sqrt(count)
    faint = (centers > r_min) & (centers < half_width) & (count >= 5) & ~(med > snr * err)
    if not np.any(faint):
        return np.ones(shape), half_width, 0.0

    taper_start = float(np.clip(centers[np.argmax(faint)], r_min, r_max))
    taper_width = max(half_width - taper_start, 0.0)

    if taper_width <= 0:
        return np.ones(shape), taper_start, taper_width

    window = np.ones(shape)
    idx = r > taper_start
    if np.any(idx):
        phase = np.clip((r[idx] - taper_start) / taper_width, 0.0, 1.0)
        window[idx] = 0.5 * (1.0 + np.cos(np.pi * phase))
    window[r >= half_width] = 0.0

    return window, taper_start, taper_width


def _regularize_polynomial_psf(coeffs, sampling, fwhm, sample_r, sample_v):
    """Suppress spurious outer support and restore polynomial PSF normalization."""

    window, taper_start, taper_width = _build_polynomial_psf_taper(
        coeffs.shape[1:], sampling, fwhm, sample_r, sample_v
    )
    coeffs *= window[np.newaxis, :, :]

    # Normalization deficit is redistributed following the constant term
    template = np.clip(coeffs[0], 0.0, None)
    total = float(np.sum(template))
    if not np.isfinite(total) or total <= 0:
        template = window.copy()
        total = float(np.sum(template))
    template /= total

    plane_sums = coeffs.reshape(coeffs.shape[0], -1).sum(axis=1)
    target_sums = np.zeros_like(plane_sums)
    target_sums[0] = 1.0
    coeffs += (target_sums - plane_sums)[:, np.newaxis, np.newaxis] * template[np.newaxis, :, :]

    return coeffs, taper_start, taper_width


def _normalize_stamp_background_mode(mode):
    """Normalize public stamp-background options to internal mode names."""

    if mode is None:
        return 'none'
    if isinstance(mode, (bool, np.bool_)):
        return 'median' if bool(mode) else 'none'
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        aliases = {
            'none': 'none',
            'median': 'median',
            'edge': 'median',
            'constant': 'median',
            'plane': 'plane',
            'tilt': 'plane',
        }
        if normalized in aliases:
            return aliases[normalized]

    raise ValueError(
        "subtract_background should be False/True or one of "
        "{'none', 'median', 'plane'}"
    )


def _subtract_local_stamp_background(cutout, mask_cutout=None, mode='none', border=2):
    """Remove a local constant or tilted background from a stellar cutout."""

    mode = _normalize_stamp_background_mode(mode)
    if mode == 'none':
        return cutout

    h, w = cutout.shape
    border = int(np.clip(border, 1, max(1, min(h, w) // 2)))

    edge = np.zeros_like(cutout, dtype=bool)
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True

    valid = edge & np.isfinite(cutout)
    if mask_cutout is not None:
        valid &= ~np.asarray(mask_cutout, bool)

    if not np.any(valid):
        return cutout

    values = cutout[valid]
    if mode == 'median' or np.sum(valid) < 6:
        cutout -= np.median(values)
        return cutout

    yy, xx = np.mgrid[0:h, 0:w]
    xx0 = xx.astype(np.float64) - 0.5 * (w - 1)
    yy0 = yy.astype(np.float64) - 0.5 * (h - 1)

    x = xx0[valid]
    y = yy0[valid]
    z = values.astype(np.float64)
    keep = np.ones(len(z), dtype=bool)

    for _ in range(2):
        if np.sum(keep) < 3:
            break

        A = np.column_stack([np.ones(np.sum(keep)), x[keep], y[keep]])
        coeffs, _, _, _ = np.linalg.lstsq(A, z[keep], rcond=None)
        model = coeffs[0] + coeffs[1] * x + coeffs[2] * y
        resid = z - model
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med)) * 1.4826

        if not np.isfinite(mad) or mad <= 0:
            break

        new_keep = np.abs(resid - med) <= 3.0 * mad
        if np.sum(new_keep) < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if np.sum(keep) < 3:
        cutout -= np.median(values)
        return cutout

    A = np.column_stack([np.ones(np.sum(keep)), x[keep], y[keep]])
    coeffs, _, _, _ = np.linalg.lstsq(A, z[keep], rcond=None)
    background = coeffs[0] + coeffs[1] * xx0 + coeffs[2] * yy0
    cutout -= background

    return cutout


def _group_layout(group, ngroups, select=None):
    """Precompute placement of grouped values into a NaN-padded table.

    Returns ``(order, rows, cols, shape)`` so that ``table[rows, cols] =
    values[order]`` places every selected value into the row of its group.
    Only values with ``select=True`` (all if None) are placed.
    """

    group = np.asarray(group)
    index = np.arange(len(group)) if select is None else np.flatnonzero(select)
    order = index[np.argsort(group[index], kind='stable')]
    rows = group[order]
    count = np.bincount(rows, minlength=ngroups)[:ngroups]
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    cols = np.arange(len(order)) - start[rows]

    return order, rows, cols, (ngroups, max(int(count.max(initial=0)), 1))


def _nanmedian_rows(table):
    """Median of finite values in every row of a 2-D array, and their number.

    Vectorized replacement for ``np.nanmedian(table, axis=1)``, which falls
    back to a slow per-row loop for wide arrays with NaNs.
    """

    srt = np.sort(table, axis=1)  # NaNs are sorted to the end
    count = np.sum(np.isfinite(table), axis=1)
    lo = np.take_along_axis(srt, np.maximum((count - 1) // 2, 0)[:, np.newaxis], axis=1)[:, 0]
    hi = np.take_along_axis(srt, np.maximum(count // 2, 0)[:, np.newaxis], axis=1)[:, 0]
    med = np.where(count > 0, 0.5 * (lo + hi), np.nan)

    return med, count


def _group_robust_stats(group, values, ngroups, layout=None):
    """Per-group median, robust (MAD-based) scale and number of finite values.

    ``layout`` from :func:`_group_layout` may be passed to avoid re-sorting
    when the grouping stays the same between calls; values not covered by it
    are ignored.
    """

    if layout is None:
        layout = _group_layout(group, ngroups, np.isfinite(values))
    order, rows, cols, shape = layout

    table = np.full(shape, np.nan)
    table[rows, cols] = np.asarray(values, dtype=np.float64)[order]

    with np.errstate(invalid='ignore'):
        med, count = _nanmedian_rows(table)
        mad, _ = _nanmedian_rows(np.abs(table - med[:, np.newaxis]))

    return med, 1.4826 * mad, count


def _prefilter_psf_planes(planes, npad=12):
    """Spline-prefilter PSF coefficient planes for repeated interpolation.

    Uses the same edge handling as ``map_coordinates(mode='nearest')`` with
    prefiltering: planes are padded by ``npad`` edge values before filtering.
    """

    return [
        ndimage.spline_filter(np.pad(plane, npad, mode='edge'), order=3, mode='nearest')
        for plane in planes
    ]


def _interp_psf_planes(filtered, gx, gy, npad=12):
    """Interpolate prefiltered planes at model grid coordinates, one array per plane."""

    coords = [np.ravel(gy) + npad, np.ravel(gx) + npad]
    return [
        ndimage.map_coordinates(f, coords, order=3, mode='nearest', prefilter=False).reshape(
            np.shape(gx)
        )
        for f in filtered
    ]


def _psf_at_offsets(filtered, terms, ux, uy, sampling, os_size):
    """Evaluate PSF model at native pixel offsets from source centers.

    ``filtered`` are prefiltered coefficient planes, ``terms`` the polynomial
    terms of every source (shape ``[..., ncoeffs]``, broadcastable to the
    offsets with a trailing axis). Returns native pixel values for a source
    of unit flux, zero outside the model grid.
    """

    center = (os_size - 1) / 2.0
    gx = ux / sampling + center
    gy = uy / sampling + center
    inside = (gx >= 0) & (gx <= os_size - 1) & (gy >= 0) & (gy <= os_size - 1)

    values = _interp_psf_planes(filtered, gx, gy)
    result = sum(terms[..., k] * values[k] for k in range(len(values)))

    return np.where(inside, result, 0.0) / sampling**2


def _poly_terms(x, y, degree, x0, y0, sx, sy):
    """Polynomial terms of PSF model at given positions, shape ``[n, ncoeffs]``.

    Same term ordering as :func:`get_supersampled_psf_stamp`: i2 outer, i1 inner.
    """

    dx = (np.asarray(x, dtype=np.float64) - x0) / sx
    dy = (np.asarray(y, dtype=np.float64) - y0) / sy

    terms = []
    for i2 in range(degree + 1):
        for i1 in range(degree + 1 - i2):
            terms.append(dx**i1 * dy**i2)

    return np.column_stack(terms)


def _fit_catalogue_fluxes(image, mask, cat_x, cat_y, filtered, terms, sampling, os_size, fwhm):
    """Jointly fit fluxes of catalogue sources with the PSF model.

    Every source is fitted over its core pixels (within ``fwhm`` of its
    center) together with all sources whose model overlaps them, as a single
    sparse linear least-squares problem, so that the wings of a bright star
    do not leak into the flux of a faint neighbour. Returns NaN for sources
    without usable core pixels.
    """

    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    nsrc = len(cat_x)
    reach = (os_size - 1) / 2.0 * sampling + fwhm

    # Core pixels of every source
    R = int(np.ceil(fwhm))
    oy, ox = np.mgrid[-R : R + 1, -R : R + 1]
    px = np.round(cat_x).astype(int)[:, np.newaxis] + ox.ravel()
    py = np.round(cat_y).astype(int)[:, np.newaxis] + oy.ravel()
    good = (px >= 0) & (px < image.shape[1]) & (py >= 0) & (py < image.shape[0])
    good &= np.hypot(px - cat_x[:, np.newaxis], py - cat_y[:, np.newaxis]) < fwhm
    pxc = np.clip(px, 0, image.shape[1] - 1)
    pyc = np.clip(py, 0, image.shape[0] - 1)
    good &= np.isfinite(image[pyc, pxc])
    if mask is not None:
        good &= ~mask[pyc, pxc]

    owner, col = np.nonzero(good)
    row_x = px[owner, col]
    row_y = py[owner, col]
    data = image[row_y, row_x].astype(np.float64)
    nrows = len(data)

    # Sources contributing to the core pixels of every source: itself and
    # all others within the reach of the model
    tree = cKDTree(np.c_[cat_x, cat_y])
    pairs = tree.query_pairs(reach, output_type='ndarray')
    pairs = np.concatenate([pairs, pairs[:, ::-1], np.repeat(np.arange(nsrc), 2).reshape(-1, 2)])
    order = np.argsort(pairs[:, 0], kind='stable')
    pairs = pairs[order]
    start = np.searchsorted(pairs[:, 0], np.arange(nsrc + 1))

    # Expand to (row, contributing source) entries
    rows_start = np.searchsorted(owner, np.arange(nsrc + 1))
    nrows_src = np.diff(rows_start)
    ncontrib = np.diff(start)
    ent_rows = []
    ent_cols = []
    for j in np.flatnonzero(nrows_src > 0):
        r = np.arange(rows_start[j], rows_start[j + 1])
        c = pairs[start[j] : start[j + 1], 1]
        ent_rows.append(np.repeat(r, ncontrib[j]))
        ent_cols.append(np.tile(c, len(r)))

    if not ent_rows:
        return np.full(nsrc, np.nan)

    ent_rows = np.concatenate(ent_rows)
    ent_cols = np.concatenate(ent_cols)
    values = _psf_at_offsets(
        filtered,
        terms[ent_cols],
        row_x[ent_rows] - cat_x[ent_cols],
        row_y[ent_rows] - cat_y[ent_cols],
        sampling,
        os_size,
    )
    nz = values != 0

    A = csr_matrix((values[nz], (ent_rows[nz], ent_cols[nz])), shape=(nrows, nsrc))
    flux = lsqr(A, data, atol=1e-10, btol=1e-10)[0]
    flux[nrows_src == 0] = np.nan

    return flux


def _spline_weights_matrix(samples, os_size, npad=12):
    """Sparse matrix of cubic B-spline interpolation weights for star samples.

    Maps padded, prefiltered per-star model grids (flattened, stacked over
    stars) to the values at sample positions, i.e. it is equivalent to
    ``map_coordinates(order=3, prefilter=False)`` on the output of
    :func:`_prefilter_psf_planes`, but reusable for any model.
    """

    from scipy.sparse import csr_matrix

    nstars, nsamp = samples['node'].shape
    size = os_size + 2 * npad
    gx = samples['gx'] + npad
    gy = samples['gy'] + npad
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)

    def bspline(t):
        return np.stack(
            [
                (1 - t) ** 3 / 6,
                (3 * t**3 - 6 * t**2 + 4) / 6,
                (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6,
                t**3 / 6,
            ],
            axis=-1,
        )

    wx = bspline(gx - ix)  # (nstars, nsamp, 4)
    wy = bspline(gy - iy)
    offsets = np.arange(-1, 3)
    cols_x = np.clip(ix[..., np.newaxis] + offsets, 0, size - 1)
    cols_y = np.clip(iy[..., np.newaxis] + offsets, 0, size - 1)

    star = np.arange(nstars)[:, np.newaxis, np.newaxis, np.newaxis]
    cols = star * size * size + cols_y[..., :, np.newaxis] * size + cols_x[..., np.newaxis, :]
    weights = wy[..., :, np.newaxis] * wx[..., np.newaxis, :]
    rows = np.broadcast_to(np.arange(nstars * nsamp).reshape(nstars, nsamp, 1, 1), cols.shape)

    return csr_matrix(
        (weights.ravel(), (rows.ravel(), cols.ravel())),
        shape=(nstars * nsamp, nstars * size * size),
    )


def _eval_psf_at_samples(coeffs, V, samples, os_size, gradient=False):
    """Evaluate polynomial PSF model for every star at its sample positions.

    Returns an array of shape ``(nstars, nsamp)``, or three of them (value,
    d/dx and d/dy per oversampled pixel) if ``gradient=True``, in which case
    the model of every star is first normalized to unit sum over the grid.

    As spline prefiltering is linear, coefficient planes are prefiltered once
    and combined into per-star grids, which are then interpolated at all
    samples with a single sparse product; the interpolation weights are
    computed once per set of samples and cached in it.
    """

    ncoeffs = V.shape[1]
    nstars, nsamp = samples['node'].shape
    planes = coeffs.reshape(ncoeffs, os_size, os_size)

    if samples.get('_weights') is None:
        samples['_weights'] = _spline_weights_matrix(samples, os_size)
    W = samples['_weights']

    if gradient:
        gy, gx = np.gradient(planes, axis=(1, 2))
        maps = [planes, gx, gy]
    else:
        maps = [planes]

    result = []
    for m in maps:
        filtered = np.array(_prefilter_psf_planes(m)).reshape(ncoeffs, -1)
        # Accelerate BLAS raises spurious floating point flags on finite matmuls
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            grids = V @ filtered
        result.append((W @ grids.ravel()).reshape(nstars, nsamp))

    if gradient:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            totals = V @ planes.reshape(ncoeffs, -1).sum(axis=1)
            result = [r / totals[:, np.newaxis] for r in result]

    return result if gradient else result[0]


def _sample_noise_variance(value, res, fit, keep, norm):
    """Expected variance of normalized star samples, shape ``[nstars, nsamp]``.

    Every star's own robust residual level (background noise) plus a source
    term proportional to the model over star flux, with a global coefficient
    estimated from the residuals of the core samples of kept stars.
    """

    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = np.where(fit, res, np.nan)
        sig = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=1)[:, np.newaxis]), axis=1)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
        model = np.clip(value - res, 0, None) / norm[:, np.newaxis]
        core = fit & keep[:, np.newaxis] & (model > 0.1 * np.nanmax(model))
        # Median of squared normal deviate is 0.455 of its variance
        beta = (
            np.nanmedian(
                (res[core] ** 2 / 0.455 - np.broadcast_to(sig[:, np.newaxis], res.shape)[core] ** 2)
                / model[core]
            )
            if np.any(core)
            else 0.0
        )
        beta = beta if np.isfinite(beta) and beta > 0 else 0.0

        return sig[:, np.newaxis] ** 2 + beta * model


def _solve_polynomial_psf(
    samples,
    V,
    weights,
    regularization,
    os_size,
    coeffs=None,
    clip_sigma=3.0,
    maxiter=30,
    tol=1e-4,
    tol_flip=1e-3,
):
    """Fit per-pixel polynomial PSF model to the star samples.

    Every sample is a star pixel value (normalized to unit star flux) at a
    known offset from the star center. The model is refined by iteratively
    fitting per-node polynomials to the residuals of samples assigned to
    their nearest model node, which converges to a model consistent with the
    exact sample positions (Anderson & King 2000 ePSF approach).

    Outliers are rejected on two levels, from scratch on every iteration:
    whole stars with outlying RMS residuals (galaxies, blends), and
    individual samples deviating from other stars at the same model node
    (cosmic rays, poorly subtracted neighbours), so that a locally
    contaminated star still contributes its clean pixels.

    Iterations stop when the model correction is below ``tol`` relative to
    the model peak and the rejection flipped fewer than ``tol_flip`` of the
    samples; the rejection of a few samples near the clipping threshold may
    oscillate indefinitely, so it can't be required to stay exactly the same.

    Returns ``(coeffs, keep, fit, res)`` where ``coeffs`` is ``[ncoeffs, npix]``,
    ``keep`` flags the stars used, ``fit`` the samples used and ``res`` the
    final residuals.
    """

    nstars, nsamp = samples['value'].shape
    ncoeffs = V.shape[1]
    nnodes = os_size * os_size
    node = samples['node']
    value = samples['value']
    valid = samples['valid']
    norm = samples['norm']

    if coeffs is None:
        coeffs = np.zeros((ncoeffs, nnodes))
        has_model = False
    else:
        coeffs = coeffs.copy()
        has_model = True

    keep = np.ones(nstars, dtype=bool)
    fit = valid.copy()

    node_f = node.ravel()
    star_f = np.repeat(np.arange(nstars), nsamp)
    star_node = star_f * nnodes + node_f
    reg = max(regularization, 1e-12) * np.eye(ncoeffs)[np.newaxis]
    # Node assignment is fixed within the solve, so is the grouping of samples
    layout = _group_layout(node_f, nnodes, valid.ravel())
    nvalid = max(int(valid.sum()), 1)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for iteration in range(maxiter):
            res = value - _eval_psf_at_samples(coeffs, V, samples, os_size)

            if has_model:
                # Star-level rejection on the RMS residual over used samples
                nused = np.maximum(fit.sum(axis=1), 1)
                star_rms = np.sqrt(np.sum(np.where(fit, res, 0) ** 2, axis=1) / nused)
                med_rms = np.median(star_rms[keep])
                mad_rms = np.median(np.abs(star_rms[keep] - med_rms)) * 1.4826
                new_keep = keep
                if mad_rms > 1e-15:
                    new_keep = star_rms < med_rms + clip_sigma * mad_rms
                    if new_keep.sum() < ncoeffs:
                        new_keep = keep

                # Sample-level rejection. Residuals are scaled by their expected
                # noise, as normalized stamps of stars with different brightness
                # have very different noise, and then compared with the spread
                # of all kept stars at the same node. Noise model is the star's
                # own robust background level plus a source term proportional to
                # the model over star flux, with global coefficient estimated
                # from the residuals of the core samples.
                z = res / np.sqrt(_sample_noise_variance(value, res, fit, keep, norm))
                zk = np.where(fit & new_keep[:, np.newaxis], z, np.nan)
                med_n, scale_n, count_n = _group_robust_stats(
                    node_f, zk.ravel(), nnodes, layout=layout
                )
                # Node spread can't be tighter than the noise the stars are scaled by
                scale_n = np.maximum(scale_n, 1.0)
                bad = np.abs(z - med_n[node]) > clip_sigma * scale_n[node]
                bad &= (count_n >= max(5, ncoeffs + 1))[node]
                new_fit = valid & ~bad

                flipped = np.sum(
                    (fit & keep[:, np.newaxis]) != (new_fit & new_keep[:, np.newaxis])
                )
                changed = flipped > tol_flip * nvalid
                keep = new_keep
                fit = new_fit
            else:
                changed = True

            # Per-node weighted least squares for the model correction. The
            # polynomial terms are constant within a star, so weights and
            # weighted residuals are first summed per (star, node)
            w2 = ((weights**2)[:, np.newaxis] * (fit & keep[:, np.newaxis])).ravel()
            wr = w2 * np.where(np.isfinite(res), res, 0).ravel()
            w_sn = np.bincount(star_node, w2, minlength=nstars * nnodes).reshape(nstars, nnodes)
            r_sn = np.bincount(star_node, wr, minlength=nstars * nnodes).reshape(nstars, nnodes)
            VTV = np.einsum('sn,sk,sl->nkl', w_sn, V, V, optimize=True)
            VTr = np.einsum('sn,sk->nk', r_sn, V, optimize=True)
            try:
                corr = np.linalg.solve(VTV + reg, VTr[..., np.newaxis])[..., 0].T
            except np.linalg.LinAlgError:
                corr = np.einsum('pkl,pl->pk', np.linalg.pinv(VTV + reg), VTr).T

            coeffs += corr
            has_model = True

            scale = np.max(np.abs(coeffs[0]))
            if not changed and np.max(np.abs(corr)) < tol * scale:
                break

        res = value - _eval_psf_at_samples(coeffs, V, samples, os_size)

    return coeffs, keep, fit, res


def _fit_stamp_offsets(samples, fit, V, coeffs, os_size, fwhm):
    """Fit sub-pixel shift and amplitude of every star against its PSF model.

    Solves the linearized least-squares problem
    ``sample = a*M - a*dx*dM/dx - a*dy*dM/dy`` over the PSF core, where ``M``
    is the model at the star position normalized to unit sum. Shifts are in
    image pixels, amplitudes are relative to the unit-sum model.
    """

    nstars = V.shape[0]
    sampling = samples['sampling']
    M, gx, gy = _eval_psf_at_samples(coeffs, V, samples, os_size, gradient=True)
    core = fit & (samples['r'] < 2 * fwhm)

    dx = np.zeros(nstars)
    dy = np.zeros(nstars)
    amp = np.ones(nstars)
    ok = np.zeros(nstars, dtype=bool)

    for i in range(nstars):
        use = core[i] & np.isfinite(M[i])
        if np.sum(use) < 6:
            continue

        A = np.column_stack([M[i][use], -gx[i][use] / sampling, -gy[i][use] / sampling])
        sol, _, _, _ = np.linalg.lstsq(A, samples['value'][i][use], rcond=None)
        if not np.all(np.isfinite(sol)) or sol[0] <= 0:
            continue

        amp[i] = sol[0]
        dx[i] = sol[1] / sol[0]
        dy[i] = sol[2] / sol[0]
        ok[i] = True

    return dx, dy, amp, ok


def _subset_samples(samples, idx):
    """Samples of a subset of stars."""

    # Cached entries (leading underscore) belong to the full set of samples
    return {
        key: (value[idx] if isinstance(value, np.ndarray) else value)
        for key, value in samples.items()
        if not key.startswith('_')
    }


def _heldout_star_scores(samples, fit, V, coeffs, var, os_size, fwhm, clip=3.0):
    """Robust goodness of fit of stars against a PSF model not fitted to them.

    Every star gets its own amplitude and sub-pixel shift fitted over the
    core, so only the shape of the model is tested. The score is the mean
    over used core samples (within ``2 * fwhm``, where PSF photometry gets
    its information) of the squared normalized residual, clipped at ``clip``
    sigma. Outer samples are not scored: they are much more numerous, and
    small noise-level differences there would outweigh the core mismatch
    that biases PSF fluxes. Returns NaN for stars that can't be fitted.
    """

    nstars = V.shape[0]
    sampling = samples['sampling']
    M, gx, gy = _eval_psf_at_samples(coeffs, V, samples, os_size, gradient=True)
    core = fit & (samples['r'] < 2 * fwhm)

    scores = np.full(nstars, np.nan)
    for i in range(nstars):
        use = core[i] & np.isfinite(M[i])
        if np.sum(use) < 6:
            continue
        A = np.column_stack([M[i], -gx[i] / sampling, -gy[i] / sampling])
        sol, _, _, _ = np.linalg.lstsq(A[use], samples['value'][i][use], rcond=None)
        if not np.all(np.isfinite(sol)) or sol[0] <= 0:
            continue
        ok = core[i] & np.isfinite(var[i]) & (var[i] > 0)
        if not np.any(ok):
            continue
        z2 = (samples['value'][i][ok] - A[ok] @ sol) ** 2 / var[i][ok]
        scores[i] = np.mean(np.minimum(z2, clip**2))

    return scores


def _select_psf_degree(
    samples,
    fit,
    keep,
    var,
    positions_x,
    positions_y,
    image_shape,
    weights,
    regularization,
    os_size,
    fwhm,
    degrees,
    norm_params,
    coeffs_start,
    log,
    nblocks=4,
    min_scored=10,
):
    """Choose polynomial degree of the PSF model by spatial cross-validation.

    The field is split into ``nblocks x nblocks`` blocks, assigned to
    ``nblocks`` folds so that every fold is spread over the field. For every
    candidate degree, the model is fitted to the stars outside a fold and
    tested on the stars inside it. Degrees are compared by paired per-star
    score differences, and the lowest degree not worse than the best one by
    more than one standard error of the difference is chosen.

    Returns the chosen degree, the full-data solution for it as
    ``(coeffs, keep, fit, res)``, and a dict with per-degree mean score
    differences to the best degree and their errors.
    """

    x0, y0, sx, sy = norm_params
    h, w = image_shape
    bx = np.clip((positions_x / w * nblocks).astype(int), 0, nblocks - 1)
    by = np.clip((positions_y / h * nblocks).astype(int), 0, nblocks - 1)
    folds = (bx + 2 * by) % nblocks

    scores = {}
    solutions = {}
    for degree in degrees:
        ncoeffs = (degree + 1) * (degree + 2) // 2
        V = _poly_terms(positions_x, positions_y, degree, x0, y0, sx, sy)
        start = np.zeros((ncoeffs, coeffs_start.shape[1]))
        start[0] = coeffs_start[0]
        solutions[degree] = _solve_polynomial_psf(
            samples, V, weights, regularization, os_size, coeffs=start
        )

        scores[degree] = np.full(len(positions_x), np.nan)
        for f in range(nblocks):
            train = np.flatnonzero(folds != f)
            test = np.flatnonzero((folds == f) & keep)
            if not len(test) or len(train) < 2 * ncoeffs:
                continue
            coeffs_f, _, _, _ = _solve_polynomial_psf(
                _subset_samples(samples, train),
                V[train],
                weights[train],
                regularization,
                os_size,
                coeffs=solutions[degree][0],
            )
            scores[degree][test] = _heldout_star_scores(
                _subset_samples(samples, test), fit[test], V[test], coeffs_f, var[test], os_size, fwhm
            )

    # Paired comparison on stars scored for all degrees
    good = np.all([np.isfinite(scores[d]) for d in degrees], axis=0)
    ngood = int(np.sum(good))
    means = {d: np.mean(scores[d][good]) if ngood else 0.0 for d in degrees}
    best = min(degrees, key=lambda d: means[d])
    info = {'degrees': list(degrees), 'nstars': ngood, 'registration_degree': degrees[-1]}
    info['score_diff'] = [float(means[d] - means[best]) for d in degrees]
    info['score_diff_err'] = [
        float(np.std(scores[d][good] - scores[best][good]) / np.sqrt(ngood)) if ngood else 0.0
        for d in degrees
    ]

    chosen = best
    for d, diff, err in zip(degrees, info['score_diff'], info['score_diff_err']):
        if diff <= err:
            chosen = d
            break

    if info['nstars'] < min_scored:
        log(
            'Warning: only %d stars could be scored for degree selection, using degree %d'
            % (info['nstars'], degrees[0])
        )
        chosen = degrees[0]

    log(
        'Degree selection on %d stars: '
        % info['nstars']
        + ', '.join(
            'degree %d: %+.4f +- %.4f' % (d, diff, err)
            for d, diff, err in zip(degrees, info['score_diff'], info['score_diff_err'])
        )
        + ' -> degree %d' % chosen
    )

    return chosen, solutions[chosen], info


def _create_psf_model_polynomial(
    image,
    obj,
    fwhm,
    size,
    mask,
    oversampling,
    degree,
    regularization,
    subtract_neighbors,
    neighbors,
    background_mode,
    maxiters,
    log,
    max_degree=None,
):
    """Build PSF model by fitting per-pixel polynomials to star stamps.

    Native star pixels are used directly at their true sub-pixel offsets
    from the star centers, as in Anderson & King (2000) ePSF construction,
    since interpolating the stamps onto a common grid would smooth the PSF
    core. Per-pixel polynomial coefficients of the model are then fitted to
    them; ``degree=0`` reduces this to a (weighted, outlier-clipped) mean.

    The build is iterated similarly to EPIMETHEUS (Benotto et al. 2026):
    every star is fitted with the current model for a sub-pixel shift and an
    amplitude, and its original cutout pixels are placed again at the refined
    center and normalized by the fitted amplitude, so the star pixels are
    never interpolated.
    """

    auto_degree = degree is None
    if auto_degree:
        # Refined below from the number of usable stars
        degree = 0
    ncoeffs = (degree + 1) * (degree + 2) // 2
    if not auto_degree:
        log('Building PSF model: degree=%d (%d coefficients)' % (degree, ncoeffs))
    if background_mode != 'none':
        log('Subtracting local stamp background using %s model' % background_mode)

    if len(obj) < ncoeffs:
        raise ValueError(
            "Need at least %d stars for degree=%d polynomial, got %d" % (ncoeffs, degree, len(obj))
        )

    sampling = 1.0 / oversampling

    # Oversampled stamp size (keep odd for centered stamps)
    os_size = size * oversampling
    if os_size % 2 == 0:
        os_size += 1
    os_center = (os_size - 1) / 2.0

    # Normalization parameters: image center and half-size
    # This maps coordinates to approximately [-1, 1] range
    x0 = image.shape[1] / 2.0
    y0 = image.shape[0] / 2.0
    sx = image.shape[1] / 2.0
    sy = image.shape[0] / 2.0

    # Build neighbor subtraction data if requested
    # We subtract all OTHER detections (from the full catalogue, not just the
    # training stars) from each cutout: first using Gaussian approximations
    # with catalogue fluxes, then with the current PSF model and fluxes
    # jointly fitted with it on every following iteration
    if subtract_neighbors and neighbors is not None and 'flux' in neighbors.colnames:
        sigma = fwhm / 2.3548  # FWHM to sigma
        nb_x = np.array(neighbors['x'], dtype=np.float64)
        nb_y = np.array(neighbors['y'], dtype=np.float64)
        nb_flux = np.array(neighbors['flux'], dtype=np.float64)
        nb_ok = np.isfinite(nb_x) & np.isfinite(nb_y) & np.isfinite(nb_flux) & (nb_flux > 0)
        nb_x, nb_y, nb_flux = nb_x[nb_ok], nb_y[nb_ok], nb_flux[nb_ok]
    else:
        subtract_neighbors = False

    # Cutouts carry a margin around the stamp so that the refined centres
    # still have data to sample from; margin pixels outside the image are masked
    margin = 2
    max_shift = 1.0  # Largest allowed total centre refinement, image pixels
    half = size // 2
    chalf = half + margin
    cut_y, cut_x = np.mgrid[0 : 2 * chalf + 1, 0 : 2 * chalf + 1]

    def _clean_cutout(raw, mask_cutout, nb_image):
        """Cutout with neighbours subtracted, masked pixels zeroed and background removed."""
        cutout = raw - nb_image
        if mask_cutout is not None:
            cutout[mask_cutout] = 0.0
        if background_mode != 'none':
            cutout = _subtract_local_stamp_background(cutout, mask_cutout, mode=background_mode)
        return cutout

    def _sample_star(cutout, mask_cutout, dx, dy):
        """Cutout pixels of a star with sub-pixel offset (dx, dy) from cutout center."""
        # Native pixels at their true offsets from the star, in model grid units
        ux = (cut_x - chalf - dx).ravel()
        uy = (cut_y - chalf - dy).ravel()
        gx = ux / sampling + os_center
        gy = uy / sampling + os_center
        # Only samples within the node grid, as the model can't be
        # interpolated beyond its outermost nodes
        inside = (gx >= 0) & (gx <= os_size - 1) & (gy >= 0) & (gy <= os_size - 1)
        valid = inside.copy()
        if mask_cutout is not None:
            valid &= ~mask_cutout.ravel()
        node = np.round(gy).astype(int) * os_size + np.round(gx).astype(int)

        return {
            # Model grid stores native pixel values per unit grid area
            'value': cutout.ravel() * sampling**2 * inside,
            'valid': valid,
            'node': np.where(inside, node, 0),
            'gx': gx,
            'gy': gy,
            'r': np.hypot(ux, uy),
        }

    # Extract cutouts once; they are sampled again on every iteration
    raw_cutouts = []
    cutouts = []
    cutout_masks = []
    origins = []
    int_x = []
    int_y = []
    positions_x = []
    positions_y = []
    stamp_fluxes = []
    norms = []

    # Plain arrays, as per-element access to (masked) table columns is slow
    obj_x = np.ma.filled(np.ma.asarray(obj['x'], dtype=np.float64), np.nan)
    obj_y = np.ma.filled(np.ma.asarray(obj['y'], dtype=np.float64), np.nan)
    if 'flux' in obj.colnames:
        obj_flux = np.ma.filled(np.ma.asarray(obj['flux'], dtype=np.float64), np.nan)
    else:
        obj_flux = np.ones(len(obj))

    for i in range(len(obj)):
        x_star = obj_x[i]
        y_star = obj_y[i]
        if not np.isfinite(x_star) or not np.isfinite(y_star):
            continue

        # Integer center
        ix = int(np.round(x_star))
        iy = int(np.round(y_star))

        # Skip if the stamp itself extends beyond image
        if (
            ix - half < 0
            or ix + half + 1 > image.shape[1]
            or iy - half < 0
            or iy + half + 1 > image.shape[0]
        ):
            continue

        # Cutout boundaries, with the margin clipped to the image
        x1, x2 = ix - chalf, ix + chalf + 1
        y1, y2 = iy - chalf, iy + chalf + 1
        cx1, cx2 = max(x1, 0), min(x2, image.shape[1])
        cy1, cy2 = max(y1, 0), min(y2, image.shape[0])

        cutout = np.zeros((2 * chalf + 1, 2 * chalf + 1), dtype=np.float64)
        cutout[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1] = image[cy1:cy2, cx1:cx2]

        mask_cutout = np.ones_like(cutout, dtype=bool)
        mask_cutout[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1] = (
            mask[cy1:cy2, cx1:cx2] if mask is not None else False
        )

        # Skip if mask has too many bad pixels inside the stamp
        if np.sum(mask_cutout[margin:-margin, margin:-margin]) > 0.1 * size**2:
            continue

        if not np.any(mask_cutout):
            mask_cutout = None

        # Initial neighbour model
        nb_image = np.zeros_like(cutout)
        if subtract_neighbors:
            # Neighbours close enough to affect this cutout, except the target
            # star itself (matched by position, as the neighbour catalogue may
            # differ from the training list)
            near = (np.abs(nb_x - ix) <= chalf + 5 * sigma) & (
                np.abs(nb_y - iy) <= chalf + 5 * sigma
            )
            near &= ~((np.abs(nb_x - x_star) < 0.5) & (np.abs(nb_y - y_star) < 0.5))
            if np.any(near):
                # Pixel coordinate grids for this cutout
                cy, cx = np.mgrid[y1:y2, x1:x2]
                for j in np.where(near)[0]:
                    # Subtract Gaussian approximation
                    r2 = (cx - nb_x[j]) ** 2 + (cy - nb_y[j]) ** 2
                    amp = nb_flux[j] / (2 * np.pi * sigma**2)
                    nb_image += amp * np.exp(-r2 / (2 * sigma**2))

        raw = cutout
        cutout = _clean_cutout(raw, mask_cutout, nb_image)

        # Initial normalization to unit flux of the samples at catalogue position
        s = _sample_star(cutout, mask_cutout, x_star - ix, y_star - iy)
        total = np.sum(s['value']) / sampling**2
        if not np.isfinite(total) or total <= 0:
            continue

        raw_cutouts.append(raw)
        cutouts.append(cutout)
        cutout_masks.append(mask_cutout)
        origins.append((x1, y1))
        int_x.append(ix)
        int_y.append(iy)
        positions_x.append(x_star)
        positions_y.append(y_star)
        stamp_fluxes.append(obj_flux[i])
        norms.append(total)

    nstars = len(cutouts)
    log('Extracted %d valid stamps for PSF fitting' % nstars)

    # Every model pixel is constrained only by the stars whose pixels fall
    # onto it, i.e. by about nstars / oversampling**2 of them
    min_stars_per_coeff = 5
    samples_per_node = nstars / oversampling**2

    if auto_degree:
        # Highest degree with enough samples per polynomial coefficient; the
        # model is built with it, and the degree is then selected below
        candidates = [
            d
            for d in range(int(max_degree) + 1)
            if samples_per_node >= min_stars_per_coeff * (d + 1) * (d + 2) // 2
        ] or [0]
        degree = candidates[-1]
        ncoeffs = (degree + 1) * (degree + 2) // 2
        log(
            'Building PSF model: automatic degree up to %d (%d stars)' % (degree, nstars)
        )
    elif samples_per_node < min_stars_per_coeff * ncoeffs:
        log(
            'Warning: %d stars at oversampling %d may be too few to constrain '
            'degree %d PSF model (%d coefficients); consider degree=\'auto\''
            % (nstars, oversampling, degree, ncoeffs)
        )

    if nstars < ncoeffs:
        raise ValueError(
            "Only %d valid stamps, need at least %d for degree=%d" % (nstars, ncoeffs, degree)
        )

    positions_x = np.array(positions_x)
    positions_y = np.array(positions_y)
    int_x = np.array(int_x)
    int_y = np.array(int_y)
    norms = np.array(norms)

    # Build Vandermonde matrix [nstars x ncoeffs]
    V = _poly_terms(positions_x, positions_y, degree, x0, y0, sx, sy)
    stamp_fluxes = np.asarray(stamp_fluxes, dtype=float)

    if subtract_neighbors:
        from scipy.spatial import cKDTree

        # Catalogue sources whose model may reach every cutout, except the
        # star itself (matched by position, as the neighbour catalogue may
        # differ from the training list)
        nb_terms = _poly_terms(nb_x, nb_y, degree, x0, y0, sx, sy)
        reach = np.sqrt(2) * chalf + (os_size - 1) / 2.0 * sampling
        nb_lists = cKDTree(np.c_[nb_x, nb_y]).query_ball_point(np.c_[positions_x, positions_y], reach)
        pair_star = np.concatenate(
            [np.full(len(l), k, dtype=int) for k, l in enumerate(nb_lists)] + [np.zeros(0, int)]
        )
        pair_nb = np.concatenate([np.asarray(l, dtype=int) for l in nb_lists] + [np.zeros(0, int)])
        is_self = (np.abs(nb_x[pair_nb] - positions_x[pair_star]) < 0.5) & (
            np.abs(nb_y[pair_nb] - positions_y[pair_star]) < 0.5
        )
        pair_star = pair_star[~is_self]
        pair_nb = pair_nb[~is_self]
        origins = np.array(origins, dtype=np.float64).reshape(-1, 2)

    # Each training stamp is normalized to unit flux, so an unweighted solve
    # over-emphasizes low-S/N outer pixels from fainter stars and broadens the
    # reconstructed wings. Weight by source flux as a proxy for stamp S/N.
    weights = np.ones(nstars, dtype=np.float64)
    valid_flux = np.isfinite(stamp_fluxes) & (stamp_fluxes > 0)
    if np.any(valid_flux):
        flux_ref = np.nanmedian(stamp_fluxes[valid_flux])
        if np.isfinite(flux_ref) and flux_ref > 0:
            weights[valid_flux] = stamp_fluxes[valid_flux] / flux_ref
            weights = np.clip(weights, 0.3, 5.0)
            log(
                'Applying flux weights to PSF fit: median %.0f, range %.2f..%.2f'
                % (flux_ref, np.min(weights), np.max(weights))
            )

    # Iterative refinement of star centres and normalizations
    centers_x = positions_x.copy()
    centers_y = positions_y.copy()
    coeffs = None

    for it in range(maxiters + 1):
        if it > 0 and subtract_neighbors and len(pair_nb):
            # Subtract neighbours using current model and jointly fitted fluxes
            filtered = _prefilter_psf_planes(coeffs.reshape(ncoeffs, os_size, os_size))
            fit_flux = _fit_catalogue_fluxes(
                image, mask, nb_x, nb_y, filtered, nb_terms, sampling, os_size, fwhm
            )
            fit_flux = np.where(np.isfinite(fit_flux), fit_flux, nb_flux)

            ux = (origins[pair_star, 0] - nb_x[pair_nb])[:, np.newaxis, np.newaxis] + cut_x
            uy = (origins[pair_star, 1] - nb_y[pair_nb])[:, np.newaxis, np.newaxis] + cut_y
            nb_values = _psf_at_offsets(
                filtered, nb_terms[pair_nb][:, np.newaxis, np.newaxis, :], ux, uy, sampling, os_size
            )
            nb_images = np.zeros((nstars,) + cut_x.shape)
            np.add.at(nb_images, pair_star, nb_values * fit_flux[pair_nb][:, np.newaxis, np.newaxis])

            cutouts = [
                _clean_cutout(raw_cutouts[k], cutout_masks[k], nb_images[k]) for k in range(nstars)
            ]

        per_star = [
            _sample_star(
                cutouts[k], cutout_masks[k], centers_x[k] - int_x[k], centers_y[k] - int_y[k]
            )
            for k in range(nstars)
        ]
        samples = {key: np.array([s[key] for s in per_star]) for key in per_star[0]}
        samples['value'] /= norms[:, np.newaxis]
        samples['norm'] = norms.copy()
        samples['sampling'] = sampling

        coeffs, keep, fit, res = _solve_polynomial_psf(
            samples, V, weights, regularization, os_size, coeffs=coeffs
        )

        if it == maxiters:
            break

        dx, dy, amp, ok = _fit_stamp_offsets(samples, fit, V, coeffs, os_size, fwhm)

        # Shifts common to all stars (or smoothly varying over the field, up
        # to the polynomial degree) are degenerate with a shift of the model
        # itself, so only keep the part not explained by the polynomial terms.
        # This anchors the model centre to the catalogue centroid convention.
        ref = ok & keep
        if np.sum(ref) > ncoeffs:
            for d in (dx, dy):
                sol, _, _, _ = np.linalg.lstsq(V[ref], d[ref], rcond=None)
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    d -= V @ sol
        dx[~ok] = 0
        dy[~ok] = 0

        new_x = positions_x + np.clip(
            centers_x + np.clip(dx, -0.5, 0.5) - positions_x, -max_shift, max_shift
        )
        new_y = positions_y + np.clip(
            centers_y + np.clip(dy, -0.5, 0.5) - positions_y, -max_shift, max_shift
        )
        step = np.hypot(new_x - centers_x, new_y - centers_y)
        rms_step = np.sqrt(np.mean(step[ref] ** 2)) if np.any(ref) else 0.0

        log(
            'Iteration %d: %d/%d stamps kept, %d pixels clipped, '
            'RMS centre correction %.4f px, amplitude median %.3f'
            % (
                it + 1,
                int(keep.sum()),
                nstars,
                int(np.sum(samples['valid'] & ~fit)),
                rms_step,
                np.median(amp[ok]) if np.any(ok) else np.nan,
            )
        )

        # At least one refinement is always applied, to replace the initial
        # stamp-sum normalization by the fitted amplitudes
        if it > 0 and rms_step < 0.005:
            break

        centers_x, centers_y = new_x, new_y
        norms[ok] *= amp[ok]

    degree_info = None
    if auto_degree and len(candidates) > 1:
        var = _sample_noise_variance(samples['value'], res, fit, keep, samples['norm'])
        degree, (coeffs, keep, fit, res), degree_info = _select_psf_degree(
            samples,
            fit,
            keep,
            var,
            positions_x,
            positions_y,
            image.shape,
            weights,
            regularization,
            os_size,
            fwhm,
            candidates,
            (x0, y0, sx, sy),
            coeffs,
            log,
        )
        ncoeffs = (degree + 1) * (degree + 2) // 2
        V = _poly_terms(positions_x, positions_y, degree, x0, y0, sx, sy)

    # Residual statistics before regularization
    used = fit & keep[:, np.newaxis]
    rms = np.sqrt(np.sum(np.where(used, res, 0) ** 2) / max(int(used.sum()), 1))

    # Suppress low-S/N outer support where the polynomial fit otherwise tends
    # to create square-edge pedestals and ringing.
    psf_data, taper_start, taper_width = _regularize_polynomial_psf(
        coeffs.reshape(ncoeffs, os_size, os_size),
        sampling,
        fwhm,
        samples['r'][used],
        samples['value'][used],
    )
    if taper_width > 0:
        log('Applied outer taper to PSF: start %.2f px, width %.2f px' % (taper_start, taper_width))

    log(
        'PSF fit: %d x %d pixels, %d coefficients, '
        '%d/%d stamps used, %d pixels clipped, RMS residual %.2e (per pixel, normalized)'
        % (
            os_size,
            os_size,
            ncoeffs,
            int(keep.sum()),
            nstars,
            int(np.sum(samples['valid'] & ~fit)),
            rms,
        )
    )

    psf = {
        'width': os_size,
        'height': os_size,
        'fwhm': fwhm,
        'sampling': sampling,
        'ncoeffs': ncoeffs,
        'degree': degree,
        'x0': x0,
        'y0': y0,
        'sx': sx,
        'sy': sy,
        'data': psf_data,
        'oversampling': oversampling,
        'type': 'epsf',
        # Model pixels sample the pixel-integrated PSF (Anderson & King ePSF)
        'subpixel_integrated': False,
    }
    if degree_info is not None:
        psf['degree_selection'] = degree_info

    return psf


def enclosed_psf_fraction(psf, x=0, y=0, radius=None, subpixel=10):
    """Fraction of normalised PSF flux inside a circular aperture.

    Evaluates the (possibly position-dependent) PSF model at ``(x, y)``
    via :func:`get_psf_stamp`, normalises it to unit total flux, and sums
    the pixels lying within ``radius`` of the stamp centre. Useful for
    on-the-fly aperture corrections.

    Pixels are weighted by the fractional coverage of the circular
    aperture using a sub-pixel grid (``subpixel`` × ``subpixel`` samples
    per pixel) — set ``subpixel=1`` to fall back to a hard pixel-centre
    mask, which is faster but has up to ~10 % radius-dependent jitter
    when the aperture edge cuts through individual pixels.

    Parameters
    ----------
    psf : dict
        PSF model from :func:`run_psfex`, :func:`load_psf`, or
        :func:`create_psf_model`.
    x, y : float, optional
        Position in image pixels at which to evaluate a position-dependent
        PSF model. The sub-pixel parts of ``x`` and ``y`` are folded into
        the stamp centre via :func:`get_psf_stamp`.
    radius : float or array-like
        Aperture radius (or radii) in image pixels.
    subpixel : int, optional
        Linear sub-pixel sampling per image pixel for partial-coverage
        weighting. Default 10 (100 sub-samples per pixel).

    Returns
    -------
    float or ndarray
        Enclosed flux fraction at each radius. Scalar when ``radius`` is
        a scalar, otherwise an array of the same shape.
    """
    if radius is None:
        raise TypeError("radius is required")
    radius_arr = np.atleast_1d(np.asarray(radius, float))

    stamp = get_psf_stamp(psf, x, y, normalize=True)
    h, w = stamp.shape
    cx = w // 2 + (x - np.round(x))
    cy = h // 2 + (y - np.round(y))

    if int(subpixel) <= 1:
        yy, xx = np.mgrid[0:h, 0:w]
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        out = np.array([float(np.sum(stamp[rr2 <= r ** 2])) for r in radius_arr])
    else:
        n = int(subpixel)
        # Sub-pixel grid of offsets within a single pixel, centred on 0.
        sub = (np.arange(n) + 0.5) / n - 0.5
        dyy, dxx = np.meshgrid(sub, sub, indexing='ij')
        yy, xx = np.mgrid[0:h, 0:w]
        # For each pixel, distance² from aperture centre at every sub-pixel
        rr2 = (xx[:, :, None, None] - cx + dxx[None, None, :, :]) ** 2 \
            + (yy[:, :, None, None] - cy + dyy[None, None, :, :]) ** 2
        out = np.empty(radius_arr.size, dtype=float)
        for i, r in enumerate(radius_arr):
            frac = np.mean(rr2 <= r ** 2, axis=(2, 3))
            out[i] = float(np.sum(stamp * frac))

    return float(out[0]) if np.isscalar(radius) or np.ndim(radius) == 0 else out


def select_psf_seeds(
    obj,
    image_shape,
    *,
    max_per_tile=25,
    grid=6,
    edge=20,
    obj_col_x='x',
    obj_col_y='y',
    obj_col_flux='flux',
    obj_col_flags='flags',
    accept_flags=0,
):
    """Select bright, edge-clear sources spread uniformly across the field.

    Bins ``obj`` on a regular ``grid_x × grid_y`` spatial grid and returns
    the ``max_per_tile`` brightest sources per cell that pass basic
    quality filters: finite ``x``/``y``/flux, positive flux, optionally a
    flag mask, and inside the image after an ``edge``-pixel margin.

    The returned table is a strict subset of ``obj`` (same columns,
    sub-selected rows) and is the natural input for
    :func:`stdpipe.psf.create_psf_model` and similar PSF-modelling
    routines that benefit from a uniform spatial sampling.

    Parameters
    ----------
    obj : astropy.table.Table
        Source catalogue.
    image_shape : (H, W) tuple
        Image shape used for the spatial grid and edge mask.
    max_per_tile : int
        Maximum number of seeds kept per spatial cell.
    grid : int or (nx, ny) tuple
        Number of cells in the spatial grid. A scalar means a square
        ``grid × grid`` layout.
    edge : float
        Edge margin in pixels; sources within ``edge`` of any image
        boundary are excluded.
    obj_col_x, obj_col_y : str
        Column names for source positions in pixel coordinates.
    obj_col_flux : str
        Column name used to rank candidates within each cell (brightest
        first). Sources with non-positive values are dropped.
    obj_col_flags : str, optional
        Column name for a SExtractor-style integer flag mask. Set to
        ``None`` to skip flag filtering. If the column is missing from
        ``obj``, no flag filter is applied either.
    accept_flags : int
        Bitwise mask of acceptable flags. A source is kept only when
        ``flags & ~accept_flags == 0``.

    Returns
    -------
    astropy.table.Table
        A subset of ``obj`` containing the selected seeds.
    """
    if isinstance(grid, int):
        grid_x = grid_y = grid
    else:
        grid_x, grid_y = grid

    H, W = image_shape

    x = np.asarray(obj[obj_col_x], float)
    y = np.asarray(obj[obj_col_y], float)
    flux = np.asarray(obj[obj_col_flux], float)
    good = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) & (flux > 0)
        & (x > edge) & (x < W - edge) & (y > edge) & (y < H - edge)
    )
    if obj_col_flags is not None and obj_col_flags in obj.colnames:
        flags = np.asarray(obj[obj_col_flags], int)
        good &= (flags & ~int(accept_flags)) == 0

    if not np.any(good):
        return obj[good]  # empty subset, preserves columns

    cand_idx = np.where(good)[0]
    cx = x[cand_idx]; cy = y[cand_idx]; cflux = flux[cand_idx]

    xbin = np.clip((cx / W * grid_x).astype(int), 0, grid_x - 1)
    ybin = np.clip((cy / H * grid_y).astype(int), 0, grid_y - 1)

    keep = []
    for iy in range(grid_y):
        for ix in range(grid_x):
            sel = np.where((xbin == ix) & (ybin == iy))[0]
            if sel.size:
                order = np.argsort(cflux[sel])[::-1][:max_per_tile]
                keep.append(cand_idx[sel[order]])

    if not keep:
        return obj[np.zeros(len(obj), dtype=bool)]
    return obj[np.unique(np.concatenate(keep))]
