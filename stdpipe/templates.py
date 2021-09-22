from __future__ import absolute_import, division, print_function, unicode_literals

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

from esutil import htm

# from scipy.ndimage import binary_dilation
from astropy.convolution import Tophat2DKernel, convolve, convolve_fft

from . import utils
from . import photometry
from . import pipeline
from . import astrometry

# HiPS images

def get_hips_image(hips, ra=None, dec=None, width=None, height=None, fov=None,
                   wcs=None, shape=None, header=None,
                   asinh=None, normalize=True, upscale=False,
                   get_header=True, verbose=False):
    """
    Load the image from any HiPS survey using CDS hips2fits service.

    The image scale and orientation may be specified by either center coordinates and fov,
    or by directly passing WCS solution or FITS header containing it.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if wcs is None and header is not None:
        wcs = WCS(header)

    if width is None or height is None:
        if shape is not None:
            height,width = shape
        elif header is not None:
            width = header['NAXIS1']
            height = header['NAXIS2']

    params = {
        'hips': hips,
        'width': int(width),
        'height': int(height),
        'coordsys': 'icrs',
        'format': 'fits'
    }

    if wcs is not None and wcs.is_celestial:
        # Whether we should upscale the template or not
        if upscale is True:
            # Let's guess the upscaling that will drive pixel scale beyond 1 arcsec/pix
            pixscale = astrometry.get_pixscale(wcs=wcs)*3600
            upscale = int(np.ceil(pixscale / 1))

            if upscale > 1:
                log('Will upscale the image from %.2f to %.2f arcsec/pix' % (pixscale, pixscale/upscale))

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
        return None,None

    if width is None or height is None:
        log('Frame size is not provided')
        return None,None

    url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?' + urlencode(params)

    t0 = time.time()
    hdu = fits.open(url)
    t1 = time.time()

    log('Downloaded HiPS image in %.2f s' % (t1 - t0))

    image = hdu[0].data.astype(np.double)
    header = hdu[0].header

    # FIXME: hips2fits does not properly return the header for SIP WCS, so we just copy the original WCS over it
    if wcs is not None and wcs.sip is not None:
        wheader = wcs.to_header(relax=True)
        header.update(wheader)

    hdu.close()

    if asinh is None:
        if 'PanSTARRS' in hips:
            asinh = True

    if asinh:
        # Fix asinh flux scaling
        # image = np.sinh(image*np.log(10)/2.5)
        x = image * 0.4 * np.log(10)
        image = (np.exp(x) - np.exp(-x))

    if upscale and upscale > 1:
        # We should do downscaling after conversion of the image back to linear flux scaling
        log('Downscaling the image %dx' % upscale)
        image = utils.rebin_image(image, upscale)

    if normalize:
        # Normalize the image to have median=100 and std=10, corresponding to GAIN=1 assuming Poissonian background
        image -= np.nanmedian(image)
        image *= 10/mad_std(image, ignore_nan=True)
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
    if dilate < 10 or True: # it seems convolve is faster than convolve_fft even for 2k x 2k
        mask = convolve(mask, kernel)
    else:
        mask = convolve_fft(mask, kernel)
    mask = mask > 1e-15*np.max(mask) # FIXME: is it correct threshold?..

    return mask

def mask_template(tmpl, cat=None, cat_saturation_mag=None,
                  cat_col_mag='rmag', cat_col_mag_err='e_rmag',
                  cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', cat_sr=1/3600,
                  mask_nans=True, mask_masked=True,
                  mask_photometric=False, aper=2, sn=5,
                  wcs=None, dilate=5, verbose=False, _tmpdir=None):
    """
    Apply various masking heuristics (NaNs, saturated catalogue stars, etc) to the template.

    If `mask_nans` is set, it masks NaN pixels

    If `mask_masked` is set and catalogue is provided, it masks all catalogue stars where
    `cat_col_mag` or `cat_col_mag_err` fields are either masked or NaNs

    If `mask_photometric` is set and catalogue is provided, it detects the objects on the template,
    matches them photometrically, and then rejects all the stars that are not used in photometric fit
    and fainter than the catalogue entries - supposedly, the saturated stars.

    The mask then may be optionally dilated if dilation size is set in `dilate` argument.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    if mask_nans:
        tmask = ~np.isfinite(tmpl)
        log(np.sum(tmask), 'template pixels masked after NaN checking')
    else:
        tmask = np.zeros_like(tmpl, dtype=np.bool)

    if cat is not None and wcs is not None:
        # Mask the central pixels of saturated stars
        cx, cy = wcs.all_world2pix(cat[cat_col_ra], cat[cat_col_dec], 0)
        cx = np.round(cx).astype(np.int)
        cy = np.round(cy).astype(np.int)

        tidx = np.zeros(len(cat), dtype=np.bool)

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
            tobj = photometry.get_objects_sextractor(tmpl, mask=tmask, sn=sn, aper=aper, wcs=wcs, _tmpdir=_tmpdir)
            # tobj = photometry.measure_objects(tobj, tmpl, mask=tmask, fwhm=np.median(tobj['fwhm'][tobj['flags'] == 0]), aper=1)

            tm = pipeline.calibrate_photometry(tobj, cat, sr=cat_sr,
                                               cat_col_mag=cat_col_mag, cat_col_mag_err=cat_col_mag_err,
                                               cat_col_ra=cat_col_ra, cat_col_dec=cat_col_dec,
                                               order=0, accept_flags=0x02, robust=True, scale_noise=True)

            idx = ~tm['idx'] & (tm['zero']-tm['zero_model'] < -0.1)
            tidx[tm['cidx'][idx]] = True

        # ..and keep only the ones inside the image
        tidx &= (cx >= 0) & (cx <= tmpl.shape[1] - 1)
        tidx &= (cy >= 0) & (cy <= tmpl.shape[0] - 1)

        tmask[cy[tidx], cx[tidx]] = True

        log(np.sum(tmask), 'template pixels masked after checking saturated (%s < %.1f) stars' % (cat_col_mag, cat_saturation_mag))

    if dilate and dilate > 0:
        log('Dilating the mask with %d x %d kernel' % (dilate, dilate))
        tmask = dilate_mask(tmask, dilate)
        log(np.sum(tmask), 'template pixels masked after dilation')

    return tmask

# PanSTARRS images

__skycells = None

def find_ps1_skycells(ra, dec, sr, band='r', ext='image', cell_radius=0.3, fullpath=True):
    global __skycells

    if __skycells is None:
        # Load skycells information and store to global variable
        __skycells = Table.read(utils.get_data_path('ps1skycells.txt'), format='ascii')

    # FIXME: here we may select the cells that are too far from actual footprint
    h = htm.HTM(10)
    _,idx,_ = h.match(ra, dec, __skycells['ra0'], __skycells['dec0'], sr + cell_radius, maxmatch=0)

    if fullpath:
        # Get full path on the server
        return ['rings.v3.skycell/%04d/%03d/rings.v3.skycell.%04d.%03d.stk.%s.unconv%s.fits' % (_['projectionID'], _['skyCellID'], _['projectionID'], _['skyCellID'], band, '.' + ext if ext != 'image' else '') for _ in __skycells[idx]]
    else:
        # Get just the file name
        return ['rings.v3.skycell.%04d.%03d.stk.%s.unconv%s.fits' % (_['projectionID'], _['skyCellID'], band, '.' + ext if ext != 'image' else '') for _ in __skycells[idx]]

def get_ps1_skycells(ra0, dec0, sr0, band='r', ext='image', normalize=True, overwrite=False, _cachedir=None, _tmpdir=None, verbose=False):
    """
    Get the list of filenames corresponding to skycells in the user-specified sky region.
    The cells are downloaded and stored to the specified cache location.

    To get the masks, use `ext='mask'`.

    Downloaded skycells are (optionally) normalized according to the parameters from their FITS headers.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Normalize _cachedir
    if _cachedir is not None:
        _cachedir = os.path.expanduser(_cachedir)
        log('Cache location is %s', _cachedir)
    else:
        if _tmpdir is None:
            _tmpdir = tempfile.gettempdir()
        _cachedir = os.path.join(_tmpdir, 'ps1')
        log('Cache location not specified, falling back to %s' % _cachedir)

    # Ensure the cache dir exists
    try:
        os.makedirs(_cachedir)
    except:
        pass

    filenames = []

    cells = find_ps1_skycells(ra0, dec0, sr0, band=band, ext=ext, fullpath=True)

    for cell in cells:
        cellname = os.path.basename(cell)
        filename = os.path.join(_cachedir, cellname)

        if os.path.exists(filename) and not overwrite:
            log('%s already downloaded' % cellname)
        else:
            log('Downloading %s' % cellname)

            url = 'http://ps1images.stsci.edu/' + cell

            if utils.download(url, filename, overwrite=overwrite, verbose=verbose):
                if normalize:
                    normalize_ps1_skycell(filename, verbose=verbose)

        if os.path.exists(filename):
            filenames.append(filename)

    return filenames

def normalize_ps1_skycell(filename, outname=None, verbose=False):
    """
    Normalize PanSTARRS skycell file according to its FITS header
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    header = fits.getheader(filename, -1)

    if 'RADESYS' not in header and 'PC001001' in header:
        # Normalize WCS in the header
        log('Normalizing WCS keywords in %s' % filename)

        header['RADESYS'] = 'FK5'
        header.rename_keyword('PC001001', 'PC1_1')
        header.rename_keyword('PC001002', 'PC1_2')
        header.rename_keyword('PC002001', 'PC2_1')
        header.rename_keyword('PC002002', 'PC2_2')

        data = fits.getdata(filename, -1)

        if 'BSOFTEN' in header and 'BOFFSET' in header:
            # Linearize ASINH scaling
            log('Normalizing ASINH scaling in %s' % filename)

            x = data * 0.4 * np.log(10)
            data = header['BOFFSET'] + header['BSOFTEN'] * (np.exp(x) - np.exp(-x))
            header['FLXSCALE'] = 1/header['BSOFTEN'] # For proper co-adding in SWarp

            for _ in ['BSOFTEN', 'BOFFSET', 'BLANK']:
                header.remove(_, ignore_missing=True)

        if outname is None:
            log('Writing normalized data back to %s' % filename)
            outname = filename
        else:
            log('Writing normalized data to %s' % outname)

        fits.writeto(outname, data, header, overwrite=True)

# PS1 higher level retrieval
def get_ps1_image(band='r', ext='image', wcs=None, shape=None,
                  width=None, height=None, header=None,
                  _cachedir=None, _tmpdir=None, verbose=False):
    """
    Load the images of specified type (image or mask) from PanSTARRS and re-project
    to requested WCS pixel grid.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    ra0,dec0,sr0 = astrometry.get_frame_center(header=header, wcs=wcs, shape=shape, width=width, height=height)

    cellnames = get_ps1_skycells(ra0, dec0, sr0, band=band, ext=ext, _cachedir=_cachedir, _tmpdir=_tmpdir, verbose=verbose)

    if wcs is None:
        wcs = WCS(header)
    if shape is not None:
        height,width = shape
    if width is None:
        width = header['NAXIS1']
    if height is None:
        height = header['NAXIS2']

    coadd = reproject_swarp(cellnames, wcs=wcs, width=width, height=height,
                            is_flags=(ext == 'mask'),
                            _tmpdir=_tmpdir, verbose=verbose)

    return coadd

def get_ps1_image_and_mask(band='r', **kwargs):
    image = get_ps1_image(band=band, ext='image', **kwargs)
    mask = get_ps1_image(band=band, ext='mask', **kwargs)

    return image,mask

# Image re-projection and mosaicking code
def reproject_swarp(input=[], wcs=None, shape=None, width=None, height=None, header=None, extra={},
                    is_flags=False, use_nans=True, get_weights=False,
                    _workdir=None, _tmpdir=None, _exe=None, verbose=False):
    """
    Wrapper for running SWarp for re-projecting and mosaicking of images onto target WCS grid.

    It accepts as input either list of filenames, or list of tuples where first
    element is an image, and second one - either FITS header or WCS.

    If the input images are integer flags, set `is_flags=True` so that it will be handled
    by passing `RESAMPLING_TYPE=FLAGS` and `COMBINE_TYPE=OR`.

    If `use_nans=True`, the regions with zero weights will be filled with NaNs (or 0xFFFF).
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

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
        height,width = shape

    if header is None:
        # Construct minimal FITS header
        header = fits.Header({'NAXIS':2, 'NAXIS1':width, 'NAXIS2':height, 'BITPIX':-64, 'EQUINOX': 2000.0})
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        astrometry.clear_wcs(header)
        header += wcs.to_header(relax=True)
    else:
        wcs = WCS(header)

        if wcs is None or not wcs.is_celestial:
            log("Can't re-project without target WCS")
            return None

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='swarp', dir=_tmpdir)

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
        'SUBTRACT_BACK': False, # Do not subtract the backgrounds
        'FSCALASTRO_TYPE': 'VARIABLE', # and not re-scale the images by default
    }

    if is_flags:
        log('The images will be handled as integer flags')
        opts['RESAMPLING_TYPE'] = 'FLAGS'
        opts['COMBINE_TYPE'] = 'OR'

    opts.update(extra)

    # Handle input data
    filenames = []
    bzero = 0
    for i,item in enumerate(input):
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

            filename = os.path.join(workdir, 'image_%04d.fits' % i)
            fits.writeto(filename, image, header, overwrite=True)

        filenames.append(filename)
        bzero = max(bzero, fits.getheader(filename).get('BZERO', 0)) # Keep the largest BZERO among input files

    # Build the command line
    command = binname + ' ' + utils.format_astromatic_opts(opts) + ' ' + ' '.join([shlex.quote(_) for _ in filenames])
    if not verbose:
        command += ' > /dev/null 2>/dev/null'
    log('Will run SCAMP like that:')
    log(command)

    # Run the command!

    t0 = time.time()
    res = os.system(command)
    t1 = time.time()

    if res == 0 and os.path.exists(coaddname) and os.path.exists(weightsname):
        log('SWarp run successfully in %.2f seconds' % (t1-t0))

        cheader = fits.getdata(coaddname)
        coadd = fits.getdata(coaddname)
        weights = fits.getdata(weightsname)

        # it seems SWarp adds BZERO to the output if inputs had them (e.g. unsigned ints do)
        # FIXME: this point needs further investigation!
        if np.issubdtype(coadd.dtype.type, np.integer):
            coadd -= bzero

        if use_nans:
            if np.issubdtype(coadd.dtype.type, np.floating):
                coadd[weights == 0] = np.nan
            else:
                coadd[weights == 0] = 0xffff

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
