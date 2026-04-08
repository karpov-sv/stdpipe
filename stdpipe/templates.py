import numpy as np

import os
import json
import tempfile
import shlex
import time
import shutil

from urllib.parse import urlencode

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table
from astropy.coordinates import SkyCoord

# from scipy.ndimage import binary_dilation
from astropy.convolution import Tophat2DKernel, convolve, convolve_fft

from . import utils
from . import photometry
from . import pipeline
from . import astrometry
from . import cutouts
from .reproject import reproject_swarp, reproject_lanczos  # backward compat

# HiPS images


def get_hips_image(
    hips,
    ra=None,
    dec=None,
    width=None,
    height=None,
    fov=None,
    wcs=None,
    shape=None,
    header=None,
    asinh=None,
    normalize=True,
    upscale=False,
    get_header=True,
    verbose=False,
):
    """Load an image from any HiPS survey using the CDS hips2fits service.

    The pixel grid may be specified by center coordinates and field of view, or
    by passing a WCS solution or FITS header directly.

    Parameters
    ----------
    hips : str
        HiPS survey identifier. See https://aladin.u-strasbg.fr/hips/list for
        the full list.
    ra : float, optional
        Image center Right Ascension in degrees.
    dec : float, optional
        Image center Declination in degrees.
    width : int, optional
        Image width in pixels.
    height : int, optional
        Image height in pixels.
    fov : float, optional
        Field of view (angular size) in degrees.
    wcs : astropy.wcs.WCS, optional
        WCS defining the pixel grid; supersedes ``ra``, ``dec``, ``fov``.
    shape : tuple of int, optional
        Image shape ``(height, width)``; alternative to ``width`` / ``height``.
    header : astropy.io.fits.Header, optional
        Header containing image dimensions and WCS; alternative to providing
        them explicitly.
    asinh : bool or None, optional
        Whether the survey uses non-linear asinh flux scaling. Auto-detected
        for Pan-STARRS surveys; set to ``False`` to override.
    normalize : bool, optional
        If True, pseudo-normalize the image so that background mean ≈ 100 and
        background RMS ≈ 10.
    upscale : bool or int, optional
        If set, request an upscaled image then downscale before returning.
        Useful for Pan-STARRS to reduce asinh photometric errors at coarse
        pixel scales.
    get_header : bool, optional
        If True, also return the FITS header alongside the image.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    ndarray or tuple or None
        Image projected onto the requested pixel grid, or ``(image, header)``
        if ``get_header=True``, or None on failure.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if wcs is None and header is not None:
        wcs = WCS(header)

    if width is None or height is None:
        if shape is not None:
            height, width = shape
        elif header is not None:
            width = header['NAXIS1']
            height = header['NAXIS2']

    params = {
        'hips': hips,
        'width': int(width),
        'height': int(height),
        'coordsys': 'icrs',
        'format': 'fits',
    }

    if wcs is not None and wcs.is_celestial:
        # Whether we should upscale the template or not
        if upscale is True:
            # Let's guess the upscaling that will drive pixel scale beyond 1 arcsec/pix
            pixscale = astrometry.get_pixscale(wcs=wcs) * 3600
            upscale = int(np.ceil(pixscale / 1))

            if upscale > 1:
                log(
                    'Will upscale the image from %.2f to %.2f arcsec/pix'
                    % (pixscale, pixscale / upscale)
                )

        if upscale and upscale > 1:
            log('Upscaling the image %dx' % upscale)
            wcs = astrometry.upscale_wcs(wcs, upscale, will_rebin=True)
            params['width'] *= upscale
            params['height'] *= upscale

        whdr = wcs.to_header(relax=True)
        if whdr['CTYPE1'] == 'RA---TPV' and 'PV1_0' not in whdr.keys():
            # hips2fits does not like such headers, let's fix it
            whdr['CTYPE1'] = 'RA---TAN'
            whdr['CTYPE2'] = 'DEC--TAN'

        params['wcs'] = json.dumps(dict(whdr))
    elif ra is not None and dec is not None and fov is not None:
        params['ra'] = ra
        params['dec'] = dec
        params['fov'] = fov
    else:
        log('Sky position and size are not provided')
        return None, None

    if width is None or height is None:
        log('Frame size is not provided')
        return None, None

    for baseurl in [
        'http://alasky.u-strasbg.fr/hips-image-services/hips2fits',
        'http://alaskybis.u-strasbg.fr/hips-image-services/hips2fits',
    ]:
        url = baseurl + '?' + urlencode(params)

        try:
            t0 = time.time()
            hdu = fits.open(url)
            t1 = time.time()
            log('Downloaded HiPS image in %.2f s' % (t1 - t0))
            break
        except KeyboardInterrupt:
            raise
        except:
            log('Failed downloading HiPS image from', url)
            hdu = None

    if hdu is None:
        log('Cannot download HiPS image!')
        if get_header:
            return None, None
        else:
            return None

    image = hdu[0].data.astype(np.double)
    header = hdu[0].header

    # FIXME: hips2fits does not properly return the header for SIP WCS, so we just copy the original WCS over it
    if wcs is not None and wcs.sip is not None:
        wheader = wcs.to_header(relax=True)
        header.update(wheader)

    hdu.close()

    if asinh is None:
        # All PanSTARRS bands except g are stored in asinh scaling, as of May 2023
        if 'PanSTARRS' in hips and hips != 'PanSTARRS/DR1/g':
            asinh = True

    if asinh:
        # Fix asinh flux scaling
        # image = np.sinh(image*np.log(10)/2.5)
        x = image * 0.4 * np.log(10)
        image = np.exp(x) - np.exp(-x)

    if upscale and upscale > 1:
        # We should do downscaling after conversion of the image back to linear flux scaling
        log('Downscaling the image %dx' % upscale)
        image = utils.rebin_image(image, upscale)

    if normalize:
        # Normalize the image to have median=100 and std=10, corresponding to GAIN=1 assuming Poissonian background
        image -= np.nanmedian(image)
        mad = mad_std(image, ignore_nan=True)
        if mad > 0:
            # We have non-zero background noise, let's normalize it to quasi-Poissonian level
            image *= 10 / mad
            image += 100

    if get_header:
        return image, header
    else:
        return image


def dilate_mask(mask, dilate=5):
    """
    Dilate binary mask with a given kernel size
    """

    kernel = Tophat2DKernel(dilate).array
    # mask = binary_dilation(mask, kernel)
    if dilate < 10 or True:  # it seems convolve is faster than convolve_fft even for 2k x 2k
        mask = convolve(mask, kernel)
    else:
        mask = convolve_fft(mask, kernel)
    mask = mask > 1e-15 * np.max(mask)  # FIXME: is it correct threshold?..

    return mask


def mask_template(
    tmpl,
    cat=None,
    cat_saturation_mag=None,
    cat_col_mag='rmag',
    cat_col_mag_err='e_rmag',
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    cat_sr=1 / 3600,
    mask_nans=True,
    mask_masked=True,
    mask_photometric=False,
    aper=2,
    sn=5,
    wcs=None,
    dilate=5,
    verbose=False,
    _tmpdir=None,
):
    """Apply masking heuristics to a template image.

    The following masking steps are applied (where enabled):

    1. NaN pixels are masked if ``mask_nans=True``.
    2. If ``cat`` is provided and ``wcs`` is set:

       - Stars brighter than ``cat_saturation_mag`` have their central pixel
         masked.
       - Stars with masked or NaN magnitudes are masked if ``mask_masked=True``.
       - Photometrically saturated stars are masked if
         ``mask_photometric=True``.

    3. The mask is optionally dilated by ``dilate`` pixels.

    Parameters
    ----------
    tmpl : ndarray
        Template image array.
    cat : astropy.table.Table, optional
        Reference catalogue for catalogue-based masking.
    cat_saturation_mag : float, optional
        Stars brighter than this magnitude have their central pixel masked.
    cat_col_mag : str, optional
        Catalogue column name for magnitude.
    cat_col_mag_err : str, optional
        Catalogue column name for magnitude error.
    cat_col_ra : str, optional
        Catalogue column name for Right Ascension.
    cat_col_dec : str, optional
        Catalogue column name for Declination.
    cat_sr : float, optional
        Catalogue matching radius in degrees.
    mask_nans : bool, optional
        If True, mask NaN pixels in the template.
    mask_masked : bool, optional
        If True, mask catalogue stars whose magnitudes are masked or NaN.
    mask_photometric : bool, optional
        If True, apply photometric-based masking to identify saturated stars.
    aper : float, optional
        Aperture radius (pixels) for photometry in photometric masking.
    sn : float, optional
        Minimum S/N for star detection in photometric masking.
    wcs : astropy.wcs.WCS, optional
        WCS of the template image.
    dilate : int, optional
        Dilation kernel size in pixels for expanding masked regions.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    _tmpdir : str, optional
        Directory for temporary files created during photometric masking.

    Returns
    -------
    ndarray of bool
        Boolean mask for the template image (``True`` = masked).
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if mask_nans:
        tmask = ~np.isfinite(tmpl)
        log(np.sum(tmask), 'template pixels masked after NaN checking')
    else:
        tmask = np.zeros_like(tmpl, dtype=bool)

    if cat is not None and wcs is not None:
        # Mask the central pixels of saturated stars
        cx, cy = wcs.all_world2pix(cat[cat_col_ra], cat[cat_col_dec], 0)
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)

        tidx = np.zeros(len(cat), dtype=bool)

        # First, we select all catalogue objects with masked measurements
        if mask_masked:
            mag_mask = getattr(cat[cat_col_mag], 'mask', np.zeros(len(cat), dtype=bool))
            tidx = np.asarray(mag_mask, dtype=bool)
            tidx |= ~np.isfinite(cat[cat_col_mag])
            if cat_col_mag_err:
                err_mask = getattr(cat[cat_col_mag_err], 'mask', np.zeros(len(cat), dtype=bool))
                tidx |= np.asarray(err_mask, dtype=bool)
                tidx |= ~np.isfinite(cat[cat_col_mag_err])

        # Next, we also add the ones corresponding to saturation limit
        if cat_saturation_mag is not None:
            tidx |= cat[cat_col_mag] < cat_saturation_mag

        # Also, we may mask photometrically saturated stars
        if mask_photometric:
            # Detect the stars on the template and match with the catalogue
            tobj = photometry.get_objects_sextractor(
                tmpl, mask=tmask, sn=sn, aper=aper, wcs=wcs, _tmpdir=_tmpdir
            )
            # tobj = photometry.measure_objects(tobj, tmpl, mask=tmask, fwhm=np.median(tobj['fwhm'][tobj['flags'] == 0]), aper=1)

            tm = pipeline.calibrate_photometry(
                tobj,
                cat,
                sr=cat_sr,
                cat_col_mag=cat_col_mag,
                cat_col_mag_err=cat_col_mag_err,
                cat_col_ra=cat_col_ra,
                cat_col_dec=cat_col_dec,
                order=0,
                accept_flags=0x02,
                robust=True,
                scale_noise=True,
            )

            idx = ~tm['idx'] & (tm['zero'] - tm['zero_model'] < -0.1)
            tidx[tm['cidx'][idx]] = True

        # ..and keep only the ones inside the image
        tidx &= (cx >= 0) & (cx <= tmpl.shape[1] - 1)
        tidx &= (cy >= 0) & (cy <= tmpl.shape[0] - 1)

        tmask[cy[tidx], cx[tidx]] = True

        if cat_saturation_mag is not None:
            log(
                np.sum(tmask),
                'template pixels masked after checking saturated (%s < %.1f) stars'
                % (cat_col_mag, cat_saturation_mag),
            )
        else:
            log(np.sum(tmask), 'template pixels masked after catalogue checking')

    if dilate and dilate > 0:
        log('Dilating the mask with %d x %d kernel' % (dilate, dilate))
        tmask = dilate_mask(tmask, dilate)
        log(np.sum(tmask), 'template pixels masked after dilation')

    return tmask


# Pan-STARRS images
__ps1_skycells = None


def point_in_ps1(ra, dec):
    """
    Whether the sky point is covered by Pan-STARRS, rough estimation
    """
    return dec > -30


# Legacy Survey images
__ls_skycells = None


def point_in_ls(ra, dec):
    """
    Whether the sky point is covered by Legacy Survey, rough estimation
    """
    sc = SkyCoord(ra, dec, unit='deg')
    res = np.abs(sc.galactic.b.deg) > 20

    return res


def _filter_cells_by_footprint(cell_ra, cell_dec, cell_radius, wcs, width, height):
    """Return boolean mask of cells that overlap the image footprint.

    After the coarse spherical-match preselection, this projects each
    candidate cell center into the image pixel grid and keeps only
    those whose centre falls within ``cell_radius`` (in pixels) of the
    image boundaries.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    pixscale = np.mean(proj_plane_pixel_scales(wcs))  # deg/pixel
    margin = cell_radius / pixscale  # cell radius in pixels

    # Project cell centres to pixel coordinates (may be outside image)
    px, py = wcs.all_world2pix(cell_ra, cell_dec, 0)

    keep = (px > -margin) & (px < width + margin) & (py > -margin) & (py < height + margin)
    return keep


def find_skycells(
    ra, dec, sr, band='r', ext='image', survey='ps1', wcs=None, width=None, height=None
):
    """Find survey skycell URLs covering a sky region.

    Parameters
    ----------
    ra, dec, sr : float
        Centre and radius (degrees) of the search region.
    band : str
        Photometric band.
    ext : str
        ``'image'`` or ``'mask'``.
    survey : str
        ``'ps1'`` or ``'ls'``.
    wcs : `~astropy.wcs.WCS`, optional
        If provided together with *width*/*height*, used to reject
        cells that fall outside the actual rectangular image footprint
        (the default circular search can over-select for elongated
        or rotated fields).
    width, height : int, optional
        Image dimensions in pixels (required when *wcs* is given).
    """
    results = []

    if survey == 'ps1':
        global __ps1_skycells

        if __ps1_skycells is None:
            # Load skycells information and store to global variable
            __ps1_skycells = Table.read(utils.get_data_path('ps1skycells.txt'), format='ascii')

        cell_radius = 0.3

        # Coarse spherical preselection
        _, idx, _ = astrometry.spherical_match(
            ra, dec, __ps1_skycells['ra0'], __ps1_skycells['dec0'], sr + cell_radius
        )

        candidates = __ps1_skycells[idx]

        # Refine against actual rectangular footprint when WCS is available
        if wcs is not None and width is not None and height is not None:
            keep = _filter_cells_by_footprint(
                candidates['ra0'],
                candidates['dec0'],
                cell_radius,
                wcs,
                width,
                height,
            )
            candidates = candidates[keep]

        for cell in candidates:
            url = 'http://ps1images.stsci.edu/'
            url += 'rings.v3.skycell/%04d/%03d/' % (
                cell['projectionID'],
                cell['skyCellID'],
            )
            url += 'rings.v3.skycell.%04d.%03d' % (
                cell['projectionID'],
                cell['skyCellID'],
            )
            url += '.stk.%s.unconv%s.fits' % (band, '.' + ext if ext != 'image' else '')

            results.append(url)

    elif survey == 'ls':
        global __ls_skycells

        if __ls_skycells is None:
            # Load skycells information and store to global variable
            __ls_skycells = Table.read(
                utils.get_data_path('legacysurvey_bricks.fits.gz'), format='fits'
            )

        cell_radius = 0.186

        # Coarse spherical preselection
        _, idx, _ = astrometry.spherical_match(
            ra, dec, __ls_skycells['ra'], __ls_skycells['dec'], sr + cell_radius
        )

        candidates = __ls_skycells[idx]

        # Refine against actual rectangular footprint when WCS is available
        if wcs is not None and width is not None and height is not None:
            keep = _filter_cells_by_footprint(
                candidates['ra'],
                candidates['dec'],
                cell_radius,
                wcs,
                width,
                height,
            )
            candidates = candidates[keep]

        for cell in candidates:
            url = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/'
            url += 'dr10/south/' if (cell['survey'] == 'S') else 'dr9/north/'
            url += 'coadd/%s/%s/' % (cell['brickname'][:3], cell['brickname'])
            if ext == 'mask':
                url += (
                    'legacysurvey-%s-maskbits' % (cell['brickname'])
                )  # Legacy Survey uses single mask for all bands
            else:
                url += 'legacysurvey-%s-image-%s' % (cell['brickname'], band)
            url += '.fits.fz'

            results.append(url)

    else:
        raise RuntimeError('Unsupported survey %s' % survey)

    return results


def fits_open_remote(url, **kwargs):
    """
    Simple wrapper around astropy.io.fits.open() that handles cache corrution errors
    and downloads the image directly if necessary
    """
    try:
        # First try opening using the cache
        hdu = fits.open(url, **kwargs)
    except OSError:
        # If that fails, cache may be corrupted - let's skip it
        kwargs.update({'cache': False})

        try:
            hdu = fits.open(url, **kwargs)
        except:
            import traceback

            traceback.print_exc()

            return None

    return hdu


def get_skycells(
    ra0,
    dec0,
    sr0,
    band='r',
    ext='image',
    survey='ps1',
    normalize=True,
    overwrite=False,
    wcs=None,
    width=None,
    height=None,
    _cachedir=None,
    _cache_downscale=1,
    _tmpdir=None,
    verbose=False,
):
    """
    Get the list of filenames corresponding to skycells in the user-specified sky region.
    The cells are downloaded and stored to the specified cache location.

    To get the masks, use `ext='mask'`.

    Downloaded skycells are (optionally) normalized according to the parameters from their FITS headers.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Normalize _cachedir
    if _cachedir is not None:
        _cachedir = os.path.expanduser(_cachedir)
        log('Cache location is', _cachedir)
    else:
        if _tmpdir is None:
            _tmpdir = tempfile.gettempdir()
        _cachedir = os.path.join(_tmpdir, survey)
        log('Cache location not specified, falling back to %s' % _cachedir)

    # Ensure the cache dir exists
    try:
        os.makedirs(_cachedir)
    except:
        pass

    filenames = []

    cells = find_skycells(
        ra0, dec0, sr0, band=band, ext=ext, survey=survey, wcs=wcs, width=width, height=height
    )

    for cell in cells:
        cellname = os.path.basename(cell)
        filename = os.path.join(_cachedir, cellname)

        if survey == 'ls' and filename.endswith('.fz'):
            filename = os.path.splitext(filename)[0]

        if _cache_downscale > 1:
            filename, fext = os.path.splitext(filename)
            filename = filename + ('.x%d' % _cache_downscale) + fext

        if os.path.exists(filename) and not overwrite:
            log('%s already downloaded' % os.path.split(filename)[-1])
        else:
            log('Downloading %s' % cellname)

            hdu = fits_open_remote(cell)
            if hdu is not None:
                image, header = hdu[1].data, hdu[1].header

                if normalize:
                    if survey == 'ps1':
                        image, header = normalize_ps1_skycell(image, header)

                    if survey == 'ls' and ext == 'image':
                        try:
                            # Get invvar file to mask not covered regions
                            ihdu = fits_open_remote(cell.replace('-image-', '-invvar-'))
                            if ihdu is not None:
                                invvar = ihdu[1].data
                                image[invvar == 0] = np.nan
                                ihdu.close()
                        except:
                            pass

                if _cache_downscale > 1:
                    image, header = cutouts.downscale_image(
                        image,
                        header=header,
                        scale=_cache_downscale,
                        mode='or' if ext == 'mask' else 'sum',
                    )

                    log(
                        "Downscaling the image and storing it as",
                        os.path.split(filename)[-1],
                    )

                fits.writeto(filename, image, header, overwrite=True)

                hdu.close()

        if os.path.exists(filename):
            filenames.append(filename)

    return filenames


def normalize_ps1_skycell(image, header, verbose=False):
    """
    Normalize PanSTARRS skycell file according to its FITS header
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if 'RADESYS' not in header and 'PC001001' in header:
        # Normalize WCS in the header
        log('Normalizing WCS keywords')

        header = header.copy()

        header['RADESYS'] = 'FK5'
        header.rename_keyword('PC001001', 'PC1_1')
        header.rename_keyword('PC001002', 'PC1_2')
        header.rename_keyword('PC002001', 'PC2_1')
        header.rename_keyword('PC002002', 'PC2_2')

        if 'BSOFTEN' in header and 'BOFFSET' in header:
            # Linearize ASINH scaling
            log('Normalizing ASINH scaling')

            x = image * 0.4 * np.log(10)
            image = header['BOFFSET'] + header['BSOFTEN'] * (np.exp(x) - np.exp(-x))
            image /= header['EXPTIME']  # For common photometric zero-point

            for _ in ['BSOFTEN', 'BOFFSET', 'BLANK']:
                header.remove(_, ignore_missing=True)

    return image, header


# Higher level retrieval function
def get_survey_image(
    band='r',
    ext='image',
    survey='ps1',
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    reproject='lanczos',
    extra=None,
    _cachedir=None,
    _cache_downscale=1,
    _tmpdir=None,
    _workdir=None,
    verbose=False,
    **kwargs,
):
    """Download and reproject a Pan-STARRS or Legacy Survey image onto a WCS grid.

    Pan-STARRS images are converted from asinh scaling to linear flux.
    Pan-STARRS mask bits: https://outerspace.stsci.edu/display/PANSTARRS/PS1+Pixel+flags+in+Image+Table+Data
    Legacy Survey mask bits: https://www.legacysurvey.org/dr10/bitmasks

    Parameters
    ----------
    band : str, optional
        Photometric band: ``'g'``, ``'r'``, ``'i'``, ``'z'``, or ``'y'``.
    ext : str, optional
        Data type: ``'image'`` or ``'mask'``.
    survey : str, optional
        Survey name: ``'ps1'`` for Pan-STARRS or ``'ls'`` for Legacy Survey.
    wcs : astropy.wcs.WCS, optional
        Output WCS projection.
    shape : tuple of int, optional
        Output image shape ``(height, width)``; alternative to ``width`` / ``height``.
    width : int, optional
        Output image width in pixels.
    height : int, optional
        Output image height in pixels.
    header : astropy.io.fits.Header, optional
        Header containing WCS and image dimensions; alternative to providing
        them explicitly.
    reproject : str, optional
        Reprojection method: ``'lanczos'`` (default, pure-Python with flux
        conservation) or ``'swarp'`` (external SWarp binary).
    extra : dict, optional
        Extra SWarp parameters (only used when ``reproject='swarp'``).
    _cachedir : str, optional
        Directory for caching downloaded sky cells between calls.
    _cache_downscale : int, optional
        Integer downscale factor for cached images.
    _tmpdir : str, optional
        Directory for temporary files.
    _workdir : str, optional
        If specified, temporary SWarp files are kept for debugging.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.
    **kwargs
        Additional keyword arguments passed to the reprojection function.

    Returns
    -------
    ndarray or None
        Image reprojected onto the requested pixel grid, or None on failure.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Resolve WCS and dimensions early so we can use them for cell filtering
    if wcs is None:
        wcs = WCS(header)
    if shape is not None:
        height, width = shape
    if width is None:
        width = header['NAXIS1']
    if height is None:
        height = header['NAXIS2']

    ra0, dec0, sr0 = astrometry.get_frame_center(wcs=wcs, width=width, height=height)

    cellnames = get_skycells(
        ra0,
        dec0,
        sr0,
        band=band,
        ext=ext,
        survey=survey,
        wcs=wcs,
        width=width,
        height=height,
        _cachedir=_cachedir,
        _cache_downscale=_cache_downscale,
        _tmpdir=_tmpdir,
        verbose=verbose,
    )

    if reproject == 'lanczos':
        coadd = reproject_lanczos(
            cellnames,
            wcs=wcs,
            width=width,
            height=height,
            is_flags=(ext == 'mask'),
            verbose=verbose,
            **kwargs,
        )
    elif reproject == 'swarp':
        coadd = reproject_swarp(
            cellnames,
            wcs=wcs,
            width=width,
            height=height,
            is_flags=(ext == 'mask'),
            extra=extra,
            _tmpdir=_tmpdir,
            _workdir=_workdir,
            verbose=verbose,
            **kwargs,
        )
    else:
        log("Unknown reproject method '%s', use 'lanczos' or 'swarp'" % reproject)
        return None

    if ext == 'mask' and survey == 'ps1':
        coadd &= (
            0xFFFF - 0x8000
        )  # Remove undocumented PS1 'temporary marked' mask bit that is masking seemingly good pixels

    return coadd


def get_survey_image_and_mask(band='r', **kwargs):
    """Download both the image and mask from Pan-STARRS or Legacy Survey.

    Convenience wrapper that calls :func:`get_survey_image` twice.

    Parameters
    ----------
    band : str, optional
        Photometric band: ``'g'``, ``'r'``, ``'i'``, ``'z'``, or ``'y'``.
    **kwargs
        All other keyword arguments are passed to :func:`get_survey_image`.

    Returns
    -------
    tuple of ndarray
        ``(image, mask)`` reprojected onto the requested pixel grid.
    """

    image = get_survey_image(band=band, ext='image', **kwargs)
    mask = get_survey_image(band=band, ext='mask', **kwargs)

    return image, mask


def get_ps1_image(band='r', **kwargs):
    return get_survey_image(band=band, survey='ps1', **kwargs)


def get_ps1_image_and_mask(band='r', **kwargs):
    return get_survey_image_and_mask(band=band, survey='ps1', **kwargs)


def get_ls_image(band='r', **kwargs):
    return get_survey_image(band=band, survey='ls', **kwargs)


def get_ls_image_and_mask(band='r', **kwargs):
    return get_survey_image_and_mask(band=band, survey='ls', **kwargs)
