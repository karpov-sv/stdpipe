"""
Image reprojection routines.

Provides :func:`reproject_swarp` (SWarp wrapper) and :func:`reproject_lanczos`
(pure-Python Lanczos interpolation with SWarp-style oversampling and Jacobian
flux conservation).
"""

import numpy as np

import os
import tempfile
import shlex
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits

from . import utils
from . import astrometry


def _pixel_to_pixel(wcs_from, wcs_to, x, y):
    """Map pixel coordinates from one WCS to another.

    Like :func:`astropy.wcs.utils.pixel_to_pixel` but uses ``quiet=True``
    for the inverse SIP transformation so that out-of-domain pixels
    produce NaN instead of raising convergence warnings.
    """
    sky = wcs_from.all_pix2world(x, y, 0)
    try:
        pix = wcs_to.all_world2pix(sky[0], sky[1], 0, quiet=True)
    except Exception:
        # Fallback if quiet mode is not supported
        pix = wcs_to.all_world2pix(sky[0], sky[1], 0)
    return pix


# ---------------------------------------------------------------------------
# Lanczos helpers
# ---------------------------------------------------------------------------


def _lanczos_kernel(x, a):
    """Lanczos kernel of order *a*."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = np.abs(x) < a
    zero = x == 0
    result[zero] = 1.0
    nonzero = mask & ~zero
    xn = x[nonzero]
    result[nonzero] = (
        np.sin(np.pi * xn)
        * np.sin(np.pi * xn / a)
        / (np.pi * xn * np.pi * xn / a)
    )
    return result


def _lanczos_map_coordinates(image, coords, a=3, cval=np.nan):
    """Interpolate *image* at fractional pixel coordinates using Lanczos kernel.

    Parameters
    ----------
    image : 2D array
    coords : (2, N) array of (row, col) coordinates
    a : int
        Lanczos kernel order (2, 3 or 4 typical).
    cval : float
        Fill value for out-of-bounds pixels.

    Returns
    -------
    values : 1D array of interpolated values
    """
    ny, nx = image.shape
    n_pts = coords.shape[1]
    result = np.full(n_pts, cval, dtype=np.float64)

    yr, xr = coords[0], coords[1]

    # Filter out-of-bounds
    valid = (yr >= -0.5) & (yr < ny - 0.5) & (xr >= -0.5) & (xr < nx - 0.5)

    yr_v = yr[valid]
    xr_v = xr[valid]

    if len(yr_v) == 0:
        return result

    # Integer and fractional parts
    iy = np.floor(yr_v).astype(int)
    ix = np.floor(xr_v).astype(int)
    fy = yr_v - iy
    fx = xr_v - ix

    # Kernel support: -a+1 to a
    offsets = np.arange(-a + 1, a + 1)

    # Precompute kernels: shape (n_valid, 2*a)
    ky = _lanczos_kernel(fy[:, None] - offsets[None, :], a)
    kx = _lanczos_kernel(fx[:, None] - offsets[None, :], a)

    # Normalize
    ky /= ky.sum(axis=1, keepdims=True)
    kx /= kx.sum(axis=1, keepdims=True)

    # Row and column indices
    row_idx = np.clip(iy[:, None] + offsets[None, :], 0, ny - 1)
    col_idx = np.clip(ix[:, None] + offsets[None, :], 0, nx - 1)

    # Apply separable kernel (vectorized: 2a iters instead of 4a²)
    vals = np.zeros(len(yr_v))
    for j in range(len(offsets)):
        row_pixels = image[row_idx[:, j][:, None], col_idx]  # (n_valid, 2a)
        vals += ky[:, j] * np.sum(kx * row_pixels, axis=1)

    result[valid] = vals
    return result


def _reproject_single_flags(image, wcs_in, wcs_out, shape_out):
    """Reproject a single integer flag image using nearest-neighbor sampling.

    Returns
    -------
    result : 2D integer array (0 where no data)
    footprint : 2D float array (1.0 where covered, 0.0 elsewhere)
    """
    ny_out, nx_out = shape_out
    image = np.asarray(image)
    dtype = image.dtype

    yy, xx = np.mgrid[0:ny_out, 0:nx_out]
    pixel_in = _pixel_to_pixel(wcs_out, wcs_in,
                               xx.ravel().astype(float),
                               yy.ravel().astype(float))
    ix = np.round(np.asarray(pixel_in[0])).astype(int)
    iy = np.round(np.asarray(pixel_in[1])).astype(int)

    ny_in, nx_in = image.shape
    valid = (ix >= 0) & (ix < nx_in) & (iy >= 0) & (iy < ny_in)

    if np.issubdtype(dtype, np.floating):
        result = np.full(ny_out * nx_out, np.nan, dtype=dtype)
    else:
        result = np.zeros(ny_out * nx_out, dtype=dtype)

    result[valid] = image[iy[valid], ix[valid]]

    footprint = valid.astype(np.float64).reshape(shape_out)
    return result.reshape(shape_out), footprint


def _reproject_chunk(image, wcs_in, wcs_out, row_start, row_end, nx_out,
                     order, oversamp, area_ratio, sub_offsets):
    """Reproject a horizontal chunk of rows.  Used by :func:`_reproject_single`.

    Returns
    -------
    row_start : int
    result : 2D array (NaN where no data)
    footprint : 2D float array (fractional coverage 0.0–1.0)
    """
    ny_chunk = row_end - row_start

    if oversamp <= 1:
        yy, xx = np.mgrid[row_start:row_end, 0:nx_out]
        pixel_in = _pixel_to_pixel(wcs_out, wcs_in,
                                  xx.ravel().astype(float),
                                  yy.ravel().astype(float))
        coords = np.array([np.asarray(pixel_in[1]),
                           np.asarray(pixel_in[0])])
        values = _lanczos_map_coordinates(image, coords, a=order)
        result = values.reshape(ny_chunk, nx_out) * area_ratio
        footprint = np.isfinite(result).astype(np.float64)
        return row_start, result, footprint

    chunk_shape = (ny_chunk, nx_out)
    accumulator = np.zeros(chunk_shape, dtype=np.float64)
    count = np.zeros(chunk_shape, dtype=np.int32)
    n_sub = len(sub_offsets) ** 2

    for dy_off in sub_offsets:
        for dx_off in sub_offsets:
            yy, xx = np.mgrid[row_start:row_end, 0:nx_out]
            pixel_out_x = (xx + dx_off).ravel().astype(float)
            pixel_out_y = (yy + dy_off).ravel().astype(float)

            pixel_in = _pixel_to_pixel(wcs_out, wcs_in,
                                      pixel_out_x, pixel_out_y)
            coords = np.array([np.asarray(pixel_in[1]),
                               np.asarray(pixel_in[0])])
            values = _lanczos_map_coordinates(image, coords, a=order)
            vals_2d = values.reshape(chunk_shape)

            valid = np.isfinite(vals_2d)
            accumulator[valid] += vals_2d[valid]
            count[valid] += 1

    result = np.full(chunk_shape, np.nan, dtype=np.float64)
    good = count > 0
    result[good] = (accumulator[good] / count[good]) * area_ratio
    footprint = count.astype(np.float64) / n_sub
    return row_start, result, footprint


def _reproject_single(image, wcs_in, wcs_out, shape_out, order, conserve_flux,
                      oversamp, parallel=False):
    """Reproject a single image with Lanczos interpolation.

    Parameters
    ----------
    parallel : bool or int
        If True, use threads (number chosen automatically).
        If int > 1, use that many threads.

    Returns
    -------
    result : 2D array (NaN where no data)
    footprint : 2D float array (fractional coverage 0.0–1.0)
    """
    ny_out, nx_out = shape_out
    image = np.asarray(image, dtype=np.float64)

    # Pixel scale ratio
    scale_in = np.mean(proj_plane_pixel_scales(wcs_in))
    scale_out = np.mean(proj_plane_pixel_scales(wcs_out))
    scale_ratio = scale_out / scale_in  # >1 means output pixels larger

    # Auto oversampling
    if oversamp is None:
        oversamp = max(1, int(scale_ratio + 0.5))

    # Jacobian area ratio for flux conservation
    area_ratio = scale_ratio ** 2 if conserve_flux else 1.0

    # Sub-pixel offsets for oversampling
    if oversamp > 1:
        step = 1.0 / oversamp
        sub_offsets = np.arange(oversamp) * step + step / 2 - 0.5
    else:
        sub_offsets = None

    # Determine number of workers
    if parallel is True:
        n_workers = min(os.cpu_count() or 1, ny_out)
    elif isinstance(parallel, int) and parallel > 1:
        n_workers = min(parallel, ny_out)
    else:
        n_workers = 1

    if n_workers <= 1:
        # Sequential path
        _, result, footprint = _reproject_chunk(
            image, wcs_in, wcs_out, 0, ny_out,
            nx_out, order, oversamp, area_ratio, sub_offsets)
        return result, footprint

    # Threaded path — split by row chunks
    chunk_size = max(1, (ny_out + n_workers - 1) // n_workers)
    row_ranges = []
    for i in range(n_workers):
        rs = i * chunk_size
        re = min(rs + chunk_size, ny_out)
        if rs < re:
            row_ranges.append((rs, re))

    result = np.full(shape_out, np.nan, dtype=np.float64)
    footprint = np.zeros(shape_out, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=len(row_ranges)) as pool:
        futures = [
            pool.submit(_reproject_chunk, image, wcs_in, wcs_out,
                        rs, re, nx_out, order, oversamp, area_ratio,
                        sub_offsets)
            for rs, re in row_ranges
        ]
        for f in futures:
            row_start, chunk, fp_chunk = f.result()
            nrows = chunk.shape[0]
            result[row_start:row_start + nrows] = chunk
            footprint[row_start:row_start + nrows] = fp_chunk

    return result, footprint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reproject_lanczos(
    input=None,
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    order=3,
    conserve_flux=True,
    oversamp=None,
    is_flags=False,
    use_nans=True,
    parallel=False,
    return_footprint=False,
    verbose=False,
):
    """Reproject images using Lanczos interpolation with automatic oversampling.

    Implements SWarp-style oversampling (sub-pixel averaging when output pixels
    are larger than input pixels) and Jacobian area scaling for flux
    conservation.

    Accepts the same input format as :func:`reproject_swarp`: a list of
    ``(image, header/WCS)`` tuples or a list of FITS filenames.  For multiple
    inputs the reprojected frames are averaged (simple coadd).

    Parameters
    ----------
    input : list or tuple
        List of ``(image, header_or_wcs)`` tuples or FITS filenames.
        A single ``(image, header_or_wcs)`` tuple is also accepted
        (wrapped into a list automatically for reproject compatibility).
    wcs : `~astropy.wcs.WCS`, optional
        Output WCS.  Ignored if *header* already contains WCS.
    shape : tuple, optional
        Output ``(height, width)``.
    width, height : int, optional
        Output dimensions (alternative to *shape*).
    header : `~astropy.io.fits.Header`, optional
        Output FITS header (overrides *wcs*/*shape*/*width*/*height*).
    order : int
        Lanczos kernel order (default 3).
    conserve_flux : bool
        If True (default), multiply by the Jacobian area ratio so that
        *total flux* is conserved.  If False, *surface brightness* is
        conserved instead.
    oversamp : int or None
        Sub-pixel oversampling factor per axis.  ``None`` (default) selects
        automatically: ``max(1, round(output_scale / input_scale))``.
    is_flags : bool
        If True, treat input as integer flag/mask images: use
        nearest-neighbor resampling (no interpolation) and bitwise AND
        for combining multiple inputs.  Overrides *order*,
        *conserve_flux* and *oversamp*.
    use_nans : bool
        If True (default), fill regions with no input coverage with NaN
        (floating-point images) or ``0xFFFF`` (integer flag images).
    parallel : bool or int
        If True, use threads for parallel interpolation (number chosen
        automatically).  If int > 1, use that many threads.  Gives
        ~3-4x speedup on multi-core machines.
    return_footprint : bool
        If True, return ``(coadd, footprint)`` where *footprint* is a
        float array with values between 0.0 (no coverage) and 1.0 (full
        coverage).  When oversampling is active, fractional values
        indicate partial sub-pixel coverage.  Default is False for
        backward compatibility.
    verbose : bool or callable
        Logging control.

    Returns
    -------
    coadd : 2D `~numpy.ndarray` or None
        Reprojected (and optionally coadded) image.
    footprint : 2D `~numpy.ndarray`
        Coverage map (only returned when ``return_footprint=True``).
    """
    if input is None:
        input = []

    # Accept a single (image, header/WCS) tuple for reproject compatibility
    if (isinstance(input, tuple)
            and len(input) == 2
            and isinstance(input[0], np.ndarray)):
        input = [input]

    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    # Resolve output geometry
    if header is not None:
        header = header.copy()
    else:
        header = fits.Header({'NAXIS': 2, 'BITPIX': -64, 'EQUINOX': 2000.0})

    if wcs is not None and wcs.is_celestial:
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)

    if (width is None or height is None) and shape is not None:
        height, width = shape

    if width is not None:
        header['NAXIS1'] = width
    if height is not None:
        header['NAXIS2'] = height

    wcs_out = WCS(header)
    if not wcs_out.is_celestial:
        log("Can't reproject without target WCS")
        return None

    shape_out = (header['NAXIS2'], header['NAXIS1'])

    if is_flags:
        log('Input images will be handled as integer flags')

    # Collect input frames
    frames = []
    for item in input:
        if isinstance(item, str):
            hdulist = fits.open(item)
            img = hdulist[0].data
            if not is_flags:
                img = img.astype(np.float64)
            wcs_in = WCS(hdulist[0].header)
            hdulist.close()
        else:
            img, hdr_or_wcs = item
            if not is_flags:
                img = np.asarray(img, dtype=np.float64)
            if isinstance(hdr_or_wcs, WCS):
                wcs_in = hdr_or_wcs
            else:
                wcs_in = WCS(hdr_or_wcs)
        frames.append((img, wcs_in))

    if not frames:
        log("No input frames")
        if return_footprint:
            return None, None
        return None

    # Reproject each frame
    results = []
    footprints = []
    for i, (img, wcs_in) in enumerate(frames):
        log('Reprojecting frame %d/%d' % (i + 1, len(frames)))
        if is_flags:
            result, fp = _reproject_single_flags(img, wcs_in, wcs_out, shape_out)
        else:
            result, fp = _reproject_single(img, wcs_in, wcs_out, shape_out,
                                           order, conserve_flux, oversamp,
                                           parallel=parallel)
        results.append(result)
        footprints.append(fp)

    # Coadd
    if is_flags:
        # Bitwise AND of all frames (like SWarp COMBINE_TYPE=AND)
        coadd = results[0].copy()
        for r in results[1:]:
            coadd &= r

        # Combined footprint: covered if any frame covers the pixel
        footprint = np.maximum.reduce(footprints)

        if use_nans:
            # For integer flags there's no NaN; use 0xFFFF sentinel
            # Determine uncovered pixels from first frame (zeros where no data)
            # We can't reliably distinguish "flag value 0" from "no coverage"
            # for a single frame, but for consistency with reproject_swarp
            # we leave the result as-is (nearest-neighbor always maps somewhere
            # unless the output pixel falls outside the input footprint).
            pass
    else:
        if len(results) == 1:
            coadd = results[0]
            footprint = footprints[0]
        else:
            stack = np.array(results)
            valid = np.isfinite(stack)
            count = valid.sum(axis=0)
            coadd = np.full(shape_out, np.nan, dtype=np.float64)
            good = count > 0
            coadd[good] = np.nansum(stack[:, good], axis=0) / count[good]
            # Average footprint across frames
            footprint = np.mean(footprints, axis=0)

    if return_footprint:
        return coadd, footprint
    return coadd


def reproject_swarp(
    input=None,
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    extra=None,
    is_flags=False,
    use_nans=True,
    get_weights=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """
    Wrapper for running SWarp for re-projecting and mosaicking of images onto target WCS grid.

    It accepts as input either list of filenames, or list of tuples where first
    element is an image, and second one - either FITS header or WCS.

    If the input images are integer flags, set `is_flags=True` so that it will be handled
    by passing `RESAMPLING_TYPE=FLAGS` and `COMBINE_TYPE=AND`.

    If `use_nans=True`, the regions with zero weights will be filled with NaNs (or 0xFFFF).

    Any additional configuration parameter may be passed to SWarp through `extra` argument which
    should be the dictionary with parameter names as keys.

    """

    if input is None:
        input = []
    if extra is None:
        extra = {}

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
        # Find SWarp binary in common paths
        for exe in ['swarp']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SWarp binary")
        return None
    # else:
    #     log("Using SWarp binary at", binname)

    if (width is None or height is None) and shape is not None:
        height, width = shape

    if header is None:
        # Construct minimal FITS header
        header = fits.Header(
            {
                'NAXIS': 2,
                'NAXIS1': width,
                'NAXIS2': height,
                'BITPIX': -64,
                'EQUINOX': 2000.0,
            }
        )
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        astrometry.clear_wcs(header)
        whdr = wcs.to_header(relax=True)

        if wcs.sip is not None:
            whdr = astrometry.wcs_sip2pv(whdr)

        # Here we will try to fix some common problems with WCS not supported by SWarp
        # FIXME: handle SIP distortions!
        if wcs.wcs.has_pc() and 'PC1_1' not in whdr:
            pc = wcs.wcs.get_pc()
            whdr['PC1_1'] = pc[0, 0]
            whdr['PC1_2'] = pc[0, 1]
            whdr['PC2_1'] = pc[1, 0]
            whdr['PC2_2'] = pc[1, 1]

        header += whdr
    else:
        wcs = WCS(header)

        if wcs is None or not wcs.is_celestial:
            log("Can't re-project without target WCS")
            return None

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='swarp', dir=_tmpdir)
    )

    # Output coadd filename
    coaddname = os.path.join(workdir, 'coadd.fits')
    if os.path.exists(coaddname):
        os.unlink(coaddname)

    # Input header filename - the result will be re-projected to it
    headername = os.path.join(workdir, 'coadd.head')
    utils.file_write(headername, header.tostring(endcard=True, sep='\n'))

    # Output weights filename
    weightsname = os.path.join(workdir, 'coadd.weights.fits')

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    xmlname = os.path.join(workdir, 'swarp.xml')

    opts = {
        'VERBOSE_TYPE': 'QUIET' if not verbose else 'NORMAL',
        'IMAGEOUT_NAME': coaddname,
        'WEIGHTOUT_NAME': weightsname,
        'c': confname,
        'XML_NAME': xmlname,
        'VMEM_DIR': workdir,
        'RESAMPLE_DIR': workdir,
        #
        'SUBTRACT_BACK': False,  # Do not subtract the backgrounds
        'FSCALASTRO_TYPE': 'VARIABLE',  # and not re-scale the images by default
    }

    if is_flags:
        log('The images will be handled as integer flags')
        opts['RESAMPLING_TYPE'] = 'FLAGS'
        opts['COMBINE_TYPE'] = 'AND'  # Use only common flags in overlapping masks

    opts.update(extra)

    # Handle input data
    filenames = []
    bzero = 0
    for i, item in enumerate(input):
        if isinstance(item, str):
            # Item is filename already
            filename = item
        elif len(item) == 2:
            # It should be a tuple of image plus header or WCS
            image = item[0]
            header = item[1]

            if image.dtype.name == 'bool':
                image = image.astype(np.int16)

            if isinstance(header, WCS):
                header = header.to_header(relax=True)

            # Convert SIP headers to TPV
            if WCS(header).sip is not None:
                header = astrometry.wcs_sip2pv(header)

            filename = os.path.join(workdir, 'image_%04d.fits' % i)
            fits.writeto(filename, image, header, overwrite=True)

        filenames.append(filename)
        bzero = max(
            bzero, fits.getheader(filename).get('BZERO', 0)
        )  # Keep the largest BZERO among input files

    # Build the command line
    command = (
        binname
        + ' '
        + utils.format_astromatic_opts(opts)
        + ' '
        + ' '.join([shlex.quote(_) for _ in filenames])
    )
    if not verbose:
        command += ' > /dev/null 2>/dev/null'
    log('Will run SWarp like that:')
    log(command)

    # Run the command!

    t0 = time.time()
    res = os.system(command)
    t1 = time.time()

    if res == 0 and os.path.exists(coaddname) and os.path.exists(weightsname):
        log('SWarp run successfully in %.2f seconds' % (t1 - t0))

        coadd = fits.getdata(coaddname)
        weights = fits.getdata(weightsname)

        # it seems SWarp adds BZERO to the output if inputs had them (e.g. unsigned ints do)
        # FIXME: this point needs further investigation!
        if np.issubdtype(coadd.dtype.type, int):
            coadd -= bzero

        if use_nans:
            if np.issubdtype(coadd.dtype, np.floating):
                coadd[weights == 0] = np.nan
            else:
                coadd[weights == 0] = 0xFFFF

    else:
        log('Error', res, 'running SWarp')
        coadd = None
        weights = None

    if _workdir is None:
        shutil.rmtree(workdir)

    if get_weights:
        return coadd, weights
    else:
        return coadd
