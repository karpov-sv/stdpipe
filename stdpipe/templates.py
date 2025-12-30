
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
    """Load the image from any HiPS survey using CDS hips2fits service.

    The image scale and orientation may be specified by either center coordinates and fov,
    or by directly passing WCS solution or FITS header containing it.

    :param hips: HiPS identifier of the survey to use. See full list at https://aladin.u-strasbg.fr/hips/list
    :param ra: Image center Right Ascension in degrees, optional
    :param dec: Image center Declination in degrees, optional
    :param width: Image width in pixels, optional
    :param height: Image height in pixels, optional
    :param fov: Image angular size in degrees, optional
    :param wcs: Input WCS as :class:`astropy.wcs.WCS` object. May be used instead of manually specifying `ra`, `dec` and `fov` parameters.
    :param shape: Image shape - tuple of (height, width) values. May be used instead of `width` and `height`
    :param header: Image header containing image dimensions and WCS. May be passed to function instead of manually specifying them
    :param asinh: Whether the HiPS survey is expected to be in non-linear `asinh` scale. For Pan-STARRS which uses it, the scaling will be used automatically unless you specify :code:`asinh=False` here
    :param normalize: Whether to try pseudo-normalizing the image so that its background mean value is 100 and background rms is 10. May be useful if your software fails analyzing small floating-point image levels.
    :param upscale: If set, the routine will request upscaled image, and then will downscale it before returning to user. Useful e.g. for requesting Pan-STARRS images that are stored in non-linear asinh scaling, and thus suffer from photometric errors if you request the image with larger pixel scales. Upscaling (to better than 1''/pixel scale) typically overcomes this problem - the price is increasing network traffic.
    :param get_header: Whether to also return the FITS header alongside with the image.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The image from HiPS survey projected onto requested pixel grid, or image and FITS header if :code:`get_header=True`

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
    if (
        dilate < 10 or True
    ):  # it seems convolve is faster than convolve_fft even for 2k x 2k
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

    """Apply various masking heuristics (NaNs, saturated catalogue stars, etc) to the template image.

    If `mask_nans` is set, it masks all `NaN` pixels in the template

    If catalogue `cat` is provided, it will also try some catalogue-based masking as described below:

    -  If `cat_saturation_mag` is set, it will mask central pixels of all catalogue stars brighter than this magnitude.

    -  If `mask_masked` is set, it also masks all catalogue stars where `cat_col_mag` or `cat_col_mag_err` fields are either masked (using Numpy masked arrays infrastructure) or set to `NaN`

    -  If `mask_photometric` is set and catalogue is provided, it detects the objects on the template, matches them photometrically, and then rejects all the stars that are not used in photometric fit and fainter than the catalogue entries - supposedly, the saturated stars with central pixels either interpolated out or lost some flux due to bleeding.

    Finally, the mask then may be optionally dilated if dilation size is set in `dilate` argument.

    :param tmpl: Input template image as a Numpy array.
    :param cat: Input catalogue to be used for masking, optional
    :param cat_saturation_mag: Saturation level for the catalogue stars.
    :param cat_col_mag: Column name for catalogue magnitudes
    :param cat_col_mag_err: Column name for catalogue magnitude errors
    :param cat_col_ra: Column name for catalogue Right Ascension
    :param cat_col_dec: Column name for catalogue Declination
    :param cat_sr: Catalogue matching radius in degrees
    :param mask_nans: Whether to mask template pixels set to `NaN`
    :param mask_masked: Whether to mask central pixels of catalogue stars masked in the catalogue, or with magnitudes set to `NaN`
    :param mask_photometric: Whether to apply photometric-based masking heuristic as described above
    :param aper: Aperture size for performing photometry on the template for photometric-based masking
    :param sn: Minimal signal to noise ratio for detecting stars in the template for photometric-based masking
    :param wcs: Template WCS as :class:`astropy.wcs.WCS` object.
    :param dilate: Dilation kernel size for dilating the mask to extend masked regions (e.g. to better cover wings of saturated stars)
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :returns: Mask for the template image.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
            tidx = cat[cat_col_mag].mask == True
            tidx |= ~np.isfinite(cat[cat_col_mag])
            if cat_col_mag_err:
                tidx |= cat[cat_col_mag_err].mask == True
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

        log(
            np.sum(tmask),
            'template pixels masked after checking saturated (%s < %.1f) stars'
            % (cat_col_mag, cat_saturation_mag),
        )

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


def find_skycells(ra, dec, sr, band='r', ext='image', survey='ps1'):
    results = []

    if survey == 'ps1':
        global __ps1_skycells

        if __ps1_skycells is None:
            # Load skycells information and store to global variable
            __ps1_skycells = Table.read(
                utils.get_data_path('ps1skycells.txt'), format='ascii'
            )

        cell_radius = 0.3

        # FIXME: here we may select the cells that are too far from actual footprint
        _, idx, _ = astrometry.spherical_match(
            ra, dec, __ps1_skycells['ra0'], __ps1_skycells['dec0'], sr + cell_radius
        )

        for cell in __ps1_skycells[idx]:
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

        # FIXME: here we may select the cells that are too far from actual footprint
        _, idx, _ = astrometry.spherical_match(
            ra, dec, __ls_skycells['ra'], __ls_skycells['dec'], sr + cell_radius
        )

        for cell in __ls_skycells[idx]:
            url = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/'
            url += 'dr10/south/' if (cell['survey'] == 'S') else 'dr9/north/'
            url += 'coadd/%s/%s/' % (cell['brickname'][:3], cell['brickname'])
            if ext == 'mask':
                url += 'legacysurvey-%s-maskbits' % (
                    cell['brickname']
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
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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

    cells = find_skycells(ra0, dec0, sr0, band=band, ext=ext, survey=survey)

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
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
            image /= header['EXPTIME'] # For common photometric zero-point

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
    extra={},
    _cachedir=None,
    _cache_downscale=1,
    _tmpdir=None,
    _workdir=None,
    verbose=False,
    **kwargs
):

    """Downloads the images of specified type (image or mask) from PanSTARRS or Legacy Survey
    and mosaics / re-projects them to requested WCS pixel grid.

    Pan-STARRS images are normalized from original ASINH scaling to common linear scale
    Pan-STARRS mask bits are documented at https://outerspace.stsci.edu/display/PANSTARRS/PS1+Pixel+flags+in+Image+Table+Data

    Legacy Survey mask bits are documented at https://www.legacysurvey.org/dr10/bitmasks

    :param band: Photometric band (one of `g`, `r`, `i`, `z`, or `y`)
    :param ext: Image type - either `image` or `mask`
    :param survey: Survey name, `ps1` for Pan-STARRS or `ls` for Legacy Survey
    :param wcs: Output WCS projection as :class:`astropy.wcs.WCS` object
    :param shape: Output image shape as (height, width) tuple, may be specified instead of `width` and `height`
    :param width: Output image width in pixels, optional
    :param height: Output image height in pixels, optional
    :param header: The header containing image dimensions and WCS, to be used instead of `wcs`, `width` and `height`
    :param extra: Dictionary of extra SWarp parameters to be passed to underlying call to :func:`stdpipe.templates.reproject_swarp`
    :param _cachedir: If specified, this directory will be used as a location to cache downloaded images so that they may be re-used between calls. If not specified, directory with survey name will be created for it in your system temporary directory (:file:`/tmp` on Linux)
    :param _cache_downscale: Downscale integer factor for caching downloaded images in smaller resolution.
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SWarp. May be used for debugging exact inputs and outputs of the executable. Optional
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param \\**kwargs: The rest of parameters will be directly passed to :func:`stdpipe.templates.reproject_swarp`
    :returns: Returns the image re-projected onto requested pixel grid

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    ra0, dec0, sr0 = astrometry.get_frame_center(
        header=header, wcs=wcs, shape=shape, width=width, height=height
    )

    cellnames = get_skycells(
        ra0,
        dec0,
        sr0,
        band=band,
        ext=ext,
        survey=survey,
        _cachedir=_cachedir,
        _cache_downscale=_cache_downscale,
        _tmpdir=_tmpdir,
        verbose=verbose,
    )

    if wcs is None:
        wcs = WCS(header)
    if shape is not None:
        height, width = shape
    if width is None:
        width = header['NAXIS1']
    if height is None:
        height = header['NAXIS2']

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
        **kwargs
    )

    if ext == 'mask' and survey == 'ps1':
        coadd &= (
            ~0x8000
        )  # Remove undocumented PS1 'temporary marked' mask bit that is masking seemingly good pixels

    return coadd


def get_survey_image_and_mask(band='r', **kwargs):
    """Convenience wrapper for simultaneously requesting the image and corresponding mask from Pan-STARRS or Legacy Survey image archive.

    Uses :func:`stdpipe.templates.get_ps1_image` to do the job

    :param band: Photometric band (one of `g`, `r`, `i`, `z`, or `y`)
    :param \\**kwargs: The rest of parameters will be directly passed to :func:`stdpipe.templates.get_survey_image`
    :returns: Image and mask

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


# Image re-projection and mosaicking code
def reproject_swarp(
    input=[],
    wcs=None,
    shape=None,
    width=None,
    height=None,
    header=None,
    extra={},
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

        cheader = fits.getdata(coaddname)
        coadd = fits.getdata(coaddname)
        weights = fits.getdata(weightsname)

        # it seems SWarp adds BZERO to the output if inputs had them (e.g. unsigned ints do)
        # FIXME: this point needs further investigation!
        if np.issubdtype(coadd.dtype.type, int):
            coadd -= bzero

        if use_nans:
            if isinstance(coadd[0][0], np.floating):
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
