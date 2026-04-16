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

    if N > 1 and supersampled.shape[0] % N == 0 and supersampled.shape[1] % N == 0:
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
        ssx0 = (supersampled.shape[1] - 1) / 2.0
        ssy0 = (supersampled.shape[0] - 1) / 2.0

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


def create_psf_model(
    image,
    obj=None,
    fwhm=None,
    size=None,
    mask=None,
    oversampling=2,
    degree=0,
    regularization=1e-6,
    subtract_neighbors=True,
    subtract_background=False,
    isolation=5.0,
    get_raw=False,
    verbose=False,
):
    """
    Create an empirical PSF (ePSF) model from stars in the image.

    For ``degree=0`` (default), builds a position-invariant ePSF using
    photutils ``EPSFBuilder`` (iterative recentering and stacking).

    For ``degree > 0``, builds a position-dependent PSF model by fitting
    per-pixel polynomial coefficients to resampled star stamps, following
    the same approach as PSFEx. The polynomial model is:

    .. math::

        PSF(i,j; x,y) = \\sum_k c_k(i,j) \\cdot dx^{p1_k} \\cdot dy^{p2_k}

    where ``(dx, dy)`` are normalized image coordinates and ``(p1_k, p2_k)``
    are polynomial exponents with ``p1_k + p2_k <= degree``.

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
        Size of cutouts to extract around stars (should be odd). If None, automatically
        determined from FWHM as ``max(25, round_up_to_odd(8 * fwhm))``.
    mask : numpy.ndarray, optional
        Image mask as a boolean array (True values will be masked).
    oversampling : int, optional
        Oversampling factor for the ePSF (default: 2).
    degree : int, optional
        Polynomial degree for spatial PSF variation (default: 0 = constant). Degree 1 =
        linear (3 coefficients), degree 2 = quadratic (6 coefficients), etc.
    regularization : float, optional
        Tikhonov regularization parameter for polynomial fitting (default: 1e-6). Only used
        when ``degree > 0``. Set to 0 for unregularized least-squares.
    subtract_neighbors : bool, optional
        If True (default), subtract estimated flux from neighboring stars before extracting
        cutouts. Reduces contamination in crowded fields. Only used when ``degree > 0``.
    subtract_background : bool or {'none', 'median', 'plane'}, optional
        Local background handling for each training stamp before normalization. ``False`` or
        ``'none'`` leaves the current image values unchanged, ``True`` or ``'median'``
        subtracts the median of the stamp border, and ``'plane'`` fits and subtracts a
        tilted background plane from the border pixels. Only used when ``degree > 0``.
        When building from the raw image instead of a background-subtracted one,
        ``'plane'`` is usually the most robust choice.
    isolation : float, optional
        Minimum nearest-neighbor distance in FWHM units for selecting stars for ePSF building
        (default: 5.0). Stars with a neighbor closer than ``isolation * fwhm`` are excluded.
        Set to 0 or None to disable isolation filtering.
    get_raw : bool, optional
        If True and ``degree=0``, returns raw photutils EPSFModel object. Ignored when
        ``degree > 0``.
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

        # Determine FWHM and stamp size early so we can use them for edge filtering
        if fwhm is None:
            if 'fwhm' in obj.colnames:
                fwhm = np.median(obj['fwhm'])
                log('Using median FWHM: %.2f pixels' % fwhm)
            else:
                fwhm = 3.0
                log('FWHM not available, using default: %.2f pixels' % fwhm)

        if size is None:
            size = max(25, int(np.ceil(8 * fwhm)))
        if size % 2 == 0:
            size += 1

        flux_median = np.median(obj['flux'])
        flux_std = np.std(obj['flux'])

        # Select stars with flux within reasonable range
        idx = (obj['flux'] > flux_median) & (obj['flux'] < flux_median + 3 * flux_std)
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

    # Auto-size stamps based on FWHM if not specified
    if size is None:
        size = max(25, int(np.ceil(8 * fwhm)))
    if size % 2 == 0:
        size += 1  # Make sure size is odd
    log('Using stamp size: %d pixels (FWHM=%.1f)' % (size, fwhm))

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

    background_mode = _normalize_stamp_background_mode(subtract_background)

    if degree == 0:
        if background_mode != 'none':
            log('Ignoring local stamp background mode for position-invariant ePSF builder')
        return _create_psf_model_constant(
            image, obj, fwhm, size, mask, oversampling, get_raw, verbose, log
        )
    else:
        return _create_psf_model_polynomial(
            image,
            obj,
            fwhm,
            size,
            mask,
            oversampling,
            degree,
            regularization,
            subtract_neighbors,
            background_mode,
            log,
        )


def _create_psf_model_constant(image, obj, fwhm, size, mask, oversampling, get_raw, verbose, log):
    """Build position-invariant ePSF using photutils EPSFBuilder."""

    log('Extracting %dx%d cutouts around %d stars' % (size, size, len(obj)))

    nddata = NDData(data=image, mask=mask)

    # Extract stars using photutils
    stars = photutils.psf.extract_stars(nddata, Table({'x': obj['x'], 'y': obj['y']}), size=size)

    # Build ePSF
    log('Building ePSF model with oversampling=%d' % oversampling)
    epsf_builder = photutils.psf.EPSFBuilder(
        oversampling=oversampling, maxiters=10, progress_bar=bool(verbose)
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


def _build_polynomial_psf_taper(reference_stamp, sampling, fwhm):
    """Build a smooth radial taper for low-S/N outer PSF pixels."""

    h, w = reference_stamp.shape
    y, x = np.mgrid[0:h, 0:w]
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    r = np.sqrt(((x - cx) * sampling) ** 2 + ((y - cy) * sampling) ** 2)

    peak = float(np.nanmax(reference_stamp))
    half_width = 0.5 * (min(h, w) - 1) * sampling
    taper_start = 0.75 * half_width
    threshold = 0.02 * peak

    if np.isfinite(peak) and peak > 0:
        edges = np.arange(0, np.max(r) + sampling, sampling)
        centers = 0.5 * (edges[:-1] + edges[1:])
        profile = []
        for r0, r1 in zip(edges[:-1], edges[1:]):
            idx = (r >= r0) & (r < r1)
            profile.append(np.nanmedian(reference_stamp[idx]) if np.any(idx) else np.nan)
        profile = np.asarray(profile, float)
        idx = np.where(np.isfinite(profile) & (profile <= threshold))[0]
        if len(idx):
            taper_start = float(centers[idx[0]])

    taper_start = float(
        np.clip(
            taper_start,
            max(1.25 * fwhm, sampling),
            max(1.25 * fwhm, half_width - 2 * sampling),
        )
    )
    taper_width = max(half_width - taper_start, 0.0)

    if taper_width <= 0:
        return np.ones_like(reference_stamp), taper_start, taper_width

    window = np.ones_like(reference_stamp)
    idx = r > taper_start
    if np.any(idx):
        phase = np.clip((r[idx] - taper_start) / taper_width, 0.0, 1.0)
        window[idx] = 0.5 * (1.0 + np.cos(np.pi * phase))
    window[r >= half_width] = 0.0

    return window, taper_start, taper_width


def _regularize_polynomial_psf(coeffs, stamps_array, keep, sampling, fwhm):
    """Suppress spurious outer support and restore polynomial PSF normalization."""

    reference_stamp = np.nanmedian(stamps_array[keep], axis=0)
    window, taper_start, taper_width = _build_polynomial_psf_taper(reference_stamp, sampling, fwhm)
    coeffs *= window[np.newaxis, :, :]

    template = np.clip(reference_stamp * window, 0.0, None)
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
    background_mode,
    log,
):
    """Build position-dependent PSF by fitting per-pixel polynomials to star stamps.

    Follows the PSFEx algorithm: resamples star cutouts onto an oversampled
    grid, then fits polynomial coefficients per pixel via least squares.
    """

    ncoeffs = (degree + 1) * (degree + 2) // 2
    log('Building position-dependent PSF model: degree=%d (%d coefficients)' % (degree, ncoeffs))
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
    cut_center = (size - 1) / 2.0

    # Normalization parameters: image center and half-size
    # This maps coordinates to approximately [-1, 1] range
    x0 = image.shape[1] / 2.0
    y0 = image.shape[0] / 2.0
    sx = image.shape[1] / 2.0
    sy = image.shape[0] / 2.0

    # Oversampled coordinate grid (computed once, reused for all stamps)
    oy, ox = np.mgrid[0:os_size, 0:os_size]
    ox_rel = (ox - os_center) * sampling  # relative to stamp center, in image pixels
    oy_rel = (oy - os_center) * sampling

    # Build neighbor subtraction image if requested
    # We subtract Gaussian approximations of all OTHER stars from each cutout
    if subtract_neighbors and 'flux' in obj.colnames:
        sigma = fwhm / 2.3548  # FWHM to sigma
        star_x = np.array(obj['x'], dtype=np.float64)
        star_y = np.array(obj['y'], dtype=np.float64)
        star_flux = np.array(obj['flux'], dtype=np.float64)
    else:
        subtract_neighbors = False

    # Extract and resample stamps
    stamps = []
    positions_x = []
    positions_y = []
    stamp_fluxes = []
    half = size // 2

    for i in range(len(obj)):
        x_star = float(obj['x'][i])
        y_star = float(obj['y'][i])

        # Integer center and sub-pixel offset
        ix = int(np.round(x_star))
        iy = int(np.round(y_star))
        dx = x_star - ix
        dy = y_star - iy

        # Cutout boundaries
        x1, x2 = ix - half, ix + half + 1
        y1, y2 = iy - half, iy + half + 1

        # Skip if cutout extends beyond image
        if x1 < 0 or x2 > image.shape[1] or y1 < 0 or y2 > image.shape[0]:
            continue

        cutout = image[y1:y2, x1:x2].astype(np.float64).copy()

        # Subtract estimated neighbor flux from cutout
        if subtract_neighbors:
            # Pixel coordinate grids for this cutout
            cy, cx = np.mgrid[y1:y2, x1:x2]
            for j in range(len(obj)):
                if j == i:
                    continue
                # Only consider neighbors close enough to affect this cutout
                ndx = star_x[j] - ix
                ndy = star_y[j] - iy
                if abs(ndx) > half + 5 * sigma and abs(ndy) > half + 5 * sigma:
                    continue
                # Subtract Gaussian approximation
                r2 = (cx - star_x[j]) ** 2 + (cy - star_y[j]) ** 2
                amp = star_flux[j] / (2 * np.pi * sigma**2)
                cutout -= amp * np.exp(-r2 / (2 * sigma**2))

        # Skip if mask has too many bad pixels in this cutout
        mask_cutout = None
        if mask is not None:
            mask_cutout = mask[y1:y2, x1:x2]
            if np.sum(mask_cutout) > 0.1 * cutout.size:
                continue
            cutout[mask_cutout] = 0.0

        if background_mode != 'none':
            cutout = _subtract_local_stamp_background(cutout, mask_cutout, mode=background_mode)

        # Resample to oversampled grid, centering star at stamp center
        # Map each oversampled pixel back to cutout coordinates
        cutout_x = cut_center + dx + ox_rel
        cutout_y = cut_center + dy + oy_rel

        stamp = ndimage.map_coordinates(
            cutout, [cutout_y, cutout_x], order=3, mode='constant', cval=0.0
        )

        # Normalize to unit flux
        total = np.sum(stamp)
        if not np.isfinite(total) or total <= 0:
            continue
        stamp /= total

        stamps.append(stamp)
        positions_x.append(x_star)
        positions_y.append(y_star)
        stamp_fluxes.append(float(obj['flux'][i]) if 'flux' in obj.colnames else 1.0)

    nstars = len(stamps)
    log('Extracted %d valid stamps for polynomial fitting' % nstars)

    if nstars < ncoeffs:
        raise ValueError(
            "Only %d valid stamps, need at least %d for degree=%d" % (nstars, ncoeffs, degree)
        )

    # Build Vandermonde matrix [nstars x ncoeffs]
    # Same term ordering as get_supersampled_psf_stamp: i2 outer, i1 inner
    x_norm = (np.array(positions_x) - x0) / sx
    y_norm = (np.array(positions_y) - y0) / sy
    stamp_fluxes = np.asarray(stamp_fluxes, dtype=float)

    V = []
    for i2 in range(degree + 1):
        for i1 in range(degree + 1 - i2):
            V.append(x_norm**i1 * y_norm**i2)
    V = np.column_stack(V)

    # Stack stamps as [nstars, npixels]
    stamps_array = np.array(stamps)  # [nstars, os_size, os_size]
    pixels = stamps_array.reshape(nstars, -1)  # [nstars, npixels]

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
                'Applying flux weights to polynomial PSF fit: median %.0f, range %.2f..%.2f'
                % (flux_ref, np.min(weights), np.max(weights))
            )

    # Solve per-pixel polynomial with iterative sigma-clipping.
    # Outlier stamps (from neighbor contamination or artifacts) bias
    # the mean-based least-squares fit; clipping rejects them.
    keep = np.ones(nstars, dtype=bool)
    clip_sigma = 3.0
    max_clip_iters = 3

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for clip_iter in range(max_clip_iters + 1):
            w_k = weights[keep]
            V_k = V[keep] * w_k[:, np.newaxis]
            pixels_k = pixels[keep] * w_k[:, np.newaxis]
            n_keep = int(keep.sum())

            if n_keep < ncoeffs:
                break

            if regularization > 0:
                VTV = V_k.T @ V_k + regularization * np.eye(ncoeffs)
                VTp = V_k.T @ pixels_k
                coeffs = np.linalg.solve(VTV, VTp)
            else:
                coeffs, _, _, _ = np.linalg.lstsq(V_k, pixels_k, rcond=None)

            if clip_iter == max_clip_iters:
                break

            # Compute per-stamp RMS residual
            reconstructed = V @ coeffs  # all stamps, not just kept
            residuals = pixels - reconstructed
            stamp_rms = np.sqrt(np.mean(residuals**2, axis=1))

            med_rms = np.median(stamp_rms[keep])
            mad_rms = np.median(np.abs(stamp_rms[keep] - med_rms)) * 1.4826
            if mad_rms < 1e-15:
                break

            new_keep = stamp_rms < med_rms + clip_sigma * mad_rms
            n_rejected = int(keep.sum() - new_keep.sum())
            if n_rejected == 0:
                break
            if new_keep.sum() < ncoeffs:
                break

            keep = new_keep
            log(
                'Sigma-clip iter %d: rejected %d stamps, %d remaining'
                % (clip_iter + 1, n_rejected, keep.sum())
            )

    # Suppress low-S/N outer support where the polynomial fit otherwise tends
    # to create square-edge pedestals and ringing.
    psf_data, taper_start, taper_width = _regularize_polynomial_psf(
        coeffs.reshape(ncoeffs, os_size, os_size),
        stamps_array,
        keep,
        sampling,
        fwhm,
    )
    coeffs = psf_data.reshape(ncoeffs, -1)
    if taper_width > 0:
        log(
            'Applied outer taper to polynomial PSF: start %.2f px, width %.2f px'
            % (taper_start, taper_width)
        )

    # Compute and report residual statistics
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        reconstructed = V[keep] @ coeffs
        residuals = pixels[keep] - reconstructed
        rms = np.sqrt(np.nanmean(residuals**2))
    log(
        'Polynomial PSF fit: %d x %d pixels, %d coefficients, '
        '%d/%d stamps used, RMS residual %.2e (per pixel, normalized)'
        % (os_size, os_size, ncoeffs, int(keep.sum()), nstars, rms)
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
    }

    return psf
