from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.table import Table

from . import photometry

def run_psfex(image, mask=None, thresh=2.0, aper=3.0, r0=0.5, gain=1, minarea=5, vignet_size=None, order=0, sex_opts={}, checkimages=[], extra={}, psffile=None, _workdir=None, _tmpdir=None, verbose=False):
    """
    Wrapper around PSFEx
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    # Find the binary
    binname = None
    for path in ['.', '/usr/bin', '/usr/local/bin', '/opt/local/bin']:
        for exe in ['psfex']:
            if os.path.isfile(os.path.join(path, exe)):
                binname = os.path.join(path, exe)
                break

    if binname is None:
        log("Can't find PSFEx binary")
        return None

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='psfex', dir=_tmpdir)
    psf = None

    if vignet_size is None:
        vignet_size = 6*aper + 1
        log('Extracting PSF using vignette size %d x %d pixels' % (vignet_size, vignet_size))

    # Run SExtractor on input image in current workdir so that the LDAC catalogue will be in out.cat there
    obj = photometry.get_objects_sextractor(image, mask=mask, thresh=thresh, aper=aper, r0=r0, gain=gain, minarea=minarea, _workdir=workdir, _tmpdir=_tmpdir, verbose=verbose, extra_params=['SNR_WIN', 'ELONGATION', 'VIGNET(%d,%d)' % (vignet_size,vignet_size)], extra_opts=sex_opts)

    catname = os.path.join(workdir, 'out.cat')
    psfname = os.path.join(workdir, 'out.psf')

    opts = {
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

    # Build the command line
    cmd = binname + ' ' + shlex.quote(catname) + ' ' + ' '.join(['-%s %s' % (_, shlex.quote(str(opts[_]))) for _ in opts.keys()])
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

    if _workdir is None:
        shutil.rmtree(workdir)

    return result

def load_psf(filename, get_header=False, verbose=False):
    """
    Load PSF model from PSFEx file
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

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

    log('PSF model %d x %d pixels, FWHM %.1f pixels, sampling %.2f, degree %d' % (psf['width'], psf['height'], psf['fwhm'], psf['sampling'], psf['degree']))

    return psf

def bilinear_interpolate(im, x, y):
    """
    Quick and dirty bilinear interpolation
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(np.int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1);
    x1 = np.clip(x1, 0, im.shape[1] - 1);
    y0 = np.clip(y0, 0, im.shape[0] - 1);
    y1 = np.clip(y1, 0, im.shape[0] - 1);

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def get_supersampled_psf_stamp(psf, x=0, y=0, normalize=True):
    """
    Returns supersampled PSF model for a given image position
    """

    dx = 1.0*(x - psf['x0'])/psf['sx']
    dy = 1.0*(y - psf['y0'])/psf['sy']

    stamp = np.zeros(psf['data'].shape[1:], dtype=np.double)
    i = 0

    for i2 in range(0, psf['degree'] + 1):
        for i1 in range(0, psf['degree'] + 1 - i2):
            stamp += psf['data'][i] * dx**i1 * dy**i2
            i += 1

    if normalize:
        stamp /= np.sum(stamp)

    return stamp

def get_psf_stamp(psf, x=0, y=0, dx=None, dy=None, normalize=True):
    """
    Returns PSF stamp in image space with sub-pixel shift applied.
    Stamp is odd-sized, with center at

        x0 = floor(width/2) + dx
        y0 = floor(height/2) + dy

    If dx or dy is None, they are computed as

        dx = x - round(x)
        dy = y - round(y)

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
    x0 = np.floor(psf['width']*psf['sampling']/2)
    y0 = np.floor(psf['height']*psf['sampling']/2)

    width = int(x0) * 2 + 1
    height = int(y0) * 2 + 1

    x0 += dx
    y0 += dy

    # Coordinates in resulting stamp
    y, x = np.mgrid[0:height, 0:width]

    # The same grid in supersampled space, shifted accordingly
    x1 = ssx0 + (x - x0)/psf['sampling']
    y1 = ssy0 + (y - y0)/psf['sampling']

    # FIXME: it should really be Lanczos interpolation here!
    stamp = bilinear_interpolate(supersampled, x1, y1)/psf['sampling']**2

    if normalize:
        stamp /= np.sum(stamp)

    return stamp

def place_psf_stamp(image, psf, x0, y0, flux=1, gain=None):
    """
    Places PSF stamp, scaled to a given flux, at a given position inside the image.
    The stamp values are added to current content of the image.
    If gain value is set, the Poissonian noise is applied to the stamp.
    """

    stamp = get_psf_stamp(psf, x0, y0, normalize=True)
    stamp *= flux

    if gain is not None:
        idx = stamp > 0
        # FIXME: what to do with negative points?..
        stamp[idx] = np.random.poisson(stamp[idx]*gain)/gain

    # Integer coordinates inside the stamp
    y,x = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]

    # Corresponding image pixels
    y1,x1 = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
    x1 += np.int(np.round(x0) - np.floor(stamp.shape[1]/2))
    y1 += np.int(np.round(y0) - np.floor(stamp.shape[0]/2))

    # Crop the coordinates outside target image
    idx = np.isfinite(stamp)
    idx &= (x1 >= 0) & (x1 < image.shape[1])
    idx &= (y1 >= 0) & (y1 < image.shape[0])

    # Add the stamp to the image!
    image[y1[idx], x1[idx]] += stamp[y[idx], x[idx]]
