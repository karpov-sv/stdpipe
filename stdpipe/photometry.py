from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, tempfile, shlex
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

import warnings
from astropy.wcs import FITSFixedWarning
# warnings.simplefilter(action='ignore', category=FITSFixedWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

import sep
import photutils

import statsmodels.api as sm
from esutil import htm

try:
    import cv2
    # Much faster dilation
    dilate = lambda image,mask: cv2.dilate(image.astype(np.uint8), mask).astype(np.bool)
except:
    from scipy.signal import fftconvolve
    dilate = lambda image,mask: fftconvolve(image, mask, mode='same') > 0.9

from . import utils

def make_kernel(r0=1.0, ext=1.0):
    x,y = np.mgrid[np.floor(-ext*r0):np.ceil(ext*r0+1), np.floor(-ext*r0):np.ceil(ext*r0+1)]
    r = np.hypot(x,y)
    image = np.exp(-r**2/2/r0**2)

    return image

def get_objects_sep(image, header=None, mask=None, err=None, thresh=4.0, aper=3.0, bkgann=None, r0=0.5, gain=1, edge=0, minnthresh=2, minarea=5, relfluxradius=2.0, wcs=None, use_fwhm=False, use_mask_large=False, subtract_bg=True, npix_large=100, sn=10.0, verbose=True, **kwargs):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    log("Preparing background mask")

    if mask is None:
        mask = np.zeros_like(image, dtype=np.bool)

    mask_bg = np.zeros_like(mask)
    mask_segm = np.zeros_like(mask)

    log("Building background map")

    bg = sep.Background(image, mask=mask|mask_bg, bw=64, bh=64)
    if subtract_bg:
        image1 = image - bg.back()
    else:
        image1 = image.copy()

    if err is None:
        err = bg.rms()
        err[~np.isfinite(err)] = 1e30
        err[err==0] = 1e30

    sep.set_extract_pixstack(image.shape[0]*image.shape[1])

    if use_mask_large:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        log("Extracting initial objects")

        obj0,segm = sep.extract(image1, err=err, thresh=thresh, minarea=minarea, mask=mask|mask_bg, filter_kernel=kernel, segmentation_map=True)

        log("Dilating large objects")

        mask_segm = np.isin(segm, [_+1 for _,npix in enumerate(obj0['npix']) if npix > npix_large])
        mask_segm = dilate(mask_segm, np.ones([10, 10]))

    log("Extracting final objects")

    obj0 = sep.extract(image1, err=err, thresh=thresh, minarea=minarea, mask=mask|mask_bg|mask_segm, filter_kernel=kernel, **kwargs)

    if use_fwhm:
        # Estimate FHWM and use it to get optimal aperture size
        idx = obj0['flag'] == 0
        fwhm = 2.0*np.sqrt(np.hypot(obj0['a'][idx], obj0['b'][idx])*np.log(2))
        fwhm = 2.0*sep.flux_radius(image1, obj0['x'][idx], obj0['y'][idx], relfluxradius*fwhm*np.ones_like(obj0['x'][idx]), 0.5, mask=mask)[0]
        fwhm = np.median(fwhm)

        aper = max(1.5*fwhm, aper)

        log("FWHM = %.2g, aperture = %.2g" % (fwhm, aper))

    # Windowed positional parameters are often biased in crowded fields, let's avoid them for now
    # xwin,ywin,flag = sep.winpos(image1, obj0['x'], obj0['y'], 0.5, mask=mask)
    xwin,ywin = obj0['x'], obj0['y']

    # Filter out objects too close to frame edges
    idx = (np.round(xwin) > edge) & (np.round(ywin) > edge) & (np.round(xwin) < image.shape[1]-edge) & (np.round(ywin) < image.shape[0]-edge) # & (obj0['flag'] == 0)

    if minnthresh:
        idx &= (obj0['tnpix'] >= minnthresh)

    log("Measuring final objects")

    flux,fluxerr,flag = sep.sum_circle(image1, xwin[idx], ywin[idx], aper, err=err, gain=gain, mask=mask|mask_bg|mask_segm, bkgann=bkgann)
    # For debug purposes, let's make also the same aperture photometry on the background map
    bgflux,bgfluxerr,bgflag = sep.sum_circle(bg.back(), xwin[idx], ywin[idx], aper, err=bg.rms(), gain=gain, mask=mask|mask_bg|mask_segm)

    bgnorm = bgflux/np.pi/aper**2

    # Fluxes to magnitudes
    mag,magerr = np.zeros_like(flux), np.zeros_like(flux)
    mag[flux>0] = -2.5*np.log10(flux[flux>0])
    # magerr[flux>0] = 2.5*np.log10(1.0 + fluxerr[flux>0]/flux[flux>0])
    magerr[flux>0] = 2.5/np.log(10)*fluxerr[flux>0]/flux[flux>0]

    # FWHM estimation - FWHM=HFD for Gaussian
    fwhm = 2.0*sep.flux_radius(image1, xwin[idx], ywin[idx], relfluxradius*aper*np.ones_like(xwin[idx]), 0.5, mask=mask)[0]

    flag |= obj0['flag'][idx]

    # Quality cuts
    fidx = (flux > 0) & (magerr < 1.0/sn)

    if wcs is None and header is not None:
        # If header is provided, we may build WCS from it
        wcs = WCS(header)

    if wcs is not None:
        # If WCS is provided we may convert x,y to ra,dec
        ra,dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        ra,dec = np.zeros_like(obj0['x'][idx]),np.zeros_like(obj0['y'][idx])

    if verbose:
        print("All done")

    obj = Table({'x':xwin[idx][fidx], 'y':ywin[idx][fidx],
                 'xerr': np.sqrt(obj0['errx2'][idx][fidx]), 'yerr': np.sqrt(obj0['erry2'][idx][fidx]),
                 'flux':flux[fidx], 'fluxerr':fluxerr[fidx],
                 'mag':mag[fidx], 'magerr':magerr[fidx],
                 'flags':obj0['flag'][idx][fidx]|flag[fidx],
                 'ra':ra[fidx], 'dec':dec[fidx],
                 'bg':bgnorm[fidx], 'fwhm':fwhm[fidx],
                 'a':obj0['a'][idx][fidx], 'b':obj0['b'][idx][fidx],
                 'theta':obj0['theta'][idx][fidx]})

    obj.meta['aper'] = aper
    obj.meta['bkgann'] = bkgann

    obj.sort('flux', reverse=True)

    return obj

def get_objects_sextractor(image, header=None, mask=None, err=None, thresh=2.0, aper=3.0, r0=0.5, gain=1, edge=0, minarea=5, wcs=None, sn=3.0, sort=True, checkimages=[], extra_params=[], extra={}, psf=None, catfile=None, _workdir=None, _tmpdir=None, verbose=False):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    # Find the binary
    binname = None
    for path in ['.', '/usr/bin', '/usr/local/bin', '/opt/local/bin']:
        for exe in ['sex', 'sextractor', 'source-extractor']:
            if os.path.isfile(os.path.join(path, exe)):
                binname = os.path.join(path, exe)
                break

    if binname is None:
        log("Can't find SExtractor binary")
        return None

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='sex', dir=_tmpdir)
    obj = None

    # Prepare
    if type(image) == str:
        imagename = image
    else:
        imagename = os.path.join(workdir, 'image.fits')
        fits.writeto(imagename, image, header, overwrite=True)

    opts = {
        'VERBOSE_TYPE': 'QUIET',
        'DETECT_MINAREA': minarea,
        'GAIN': gain,
        'DETECT_THRESH': thresh,
        'WEIGHT_TYPE': 'BACKGROUND',
        'MASK_TYPE': 'NONE', # both 'CORRECT' and 'BLANK' seem to cause systematics?
    }

    if mask is None:
        mask = np.zeros_like(image, dtype=np.bool)

    if err is not None:
        # User-provided noise model
        err = err.copy().astype(np.double)
        err[~np.isfinite(err)] = 1e30
        err[err==0] = 1e30

        errname = os.path.join(workdir, 'errors.fits')
        fits.writeto(errname, err)
        opts['WEIGHT_IMAGE'] = errname
        opts['WEIGHT_TYPE'] = 'MAP_RMS'

    flagsname = os.path.join(workdir, 'flags.fits')
    fits.writeto(flagsname, mask.astype(np.int16), overwrite=True)
    opts['FLAG_IMAGE'] = flagsname

    if np.isscalar(aper):
        opts['PHOT_APERTURES'] = aper*2 # SExtractor expects diameters, not radii
        size = ''
    else:
        opts['PHOT_APERTURES'] = ','.join([str(_*2) for _ in aper])
        size = '[%d]' % len(aper)

    checknames = [os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    params = ['MAG_APER'+size, 'MAGERR_APER'+size, 'FLUX_APER'+size, 'FLUXERR_APER'+size, 'X_IMAGE', 'Y_IMAGE', 'ERRX2_IMAGE', 'ERRY2_IMAGE', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'FLUX_RADIUS', 'FWHM_IMAGE', 'FLAGS', 'IMAFLAGS_ISO', 'BACKGROUND']
    params += extra_params

    if psf is not None:
        opts['PSF_NAME'] = psf
        params += ['MAG_PSF', 'MAGERR_PSF', 'FLUX_PSF', 'FLUXERR_PSF', 'XPSF_IMAGE', 'YPSF_IMAGE']

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
        kernel = make_kernel(r0, ext=1.0)
        kernelname = os.path.join(workdir, 'kernel.txt')
        np.savetxt(kernelname, kernel/np.sum(kernel), fmt=b'%.6f', header='CONV NORM', comments='')
        opts['FILTER'] = 'Y'
        opts['FILTER_NAME'] = kernelname

    opts.update(extra)

    # Build the command line
    cmd = binname + ' ' + shlex.quote(imagename) + ' ' + utils.format_astromatic_opts(opts)
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log('Will run SExtractor like that:')
    log(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(catname):
        log('SExtractor run succeeded')
        obj = Table.read(catname, hdu=2)

        idx = (obj['X_IMAGE'] > edge) & (obj['X_IMAGE'] < image.shape[1] - edge)
        idx &= (obj['Y_IMAGE'] > edge) & (obj['Y_IMAGE'] < image.shape[0] - edge)

        if np.isscalar(aper):
            idx &= obj['MAGERR_APER'] < 1.0/sn
            idx &= obj['FLUX_APER'] > 0
        else:
            idx &= np.all(obj['MAGERR_APER'] < 1.0/sn, axis=1)
            idx &= np.all(obj['FLUX_APER'] > 0, axis=1)

        obj = obj[idx]

        if wcs is None and header is not None:
            wcs = WCS(header)

        if wcs is not None:
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['X_IMAGE'], obj['Y_IMAGE'], 1)
        else:
            obj['ra'],obj['dec'] = np.zeros_like(obj['X_IMAGE']), np.zeros_like(obj['Y_IMAGE'])

        obj['FLAGS'][obj['IMAFLAGS_ISO'] > 0] |= 256
        obj.remove_column('IMAFLAGS_ISO') # We do not need this column

        for _,__ in [['X_IMAGE', 'x'],
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
                     ['THETA_IMAGE', 'theta']]:
            obj.rename_column(_, __)

        if psf:
            psf_idx = obj['MAG_PSF'] == 99
            for _,__ in [['XPSF_IMAGE', 'x_psf'],
                         ['YPSF_IMAGE', 'y_psf'],
                         ['MAG_PSF', 'mag_psf'],
                         ['MAGERR_PSF', 'magerr_psf'],
                         ['FLUX_PSF', 'flux_psf'],
                         ['FLUXERR_PSF', 'fluxerr_psf']]:
                obj.rename_column(_, __)
                obj[__][psf_idx] = np.nan # TODO: use masked column here?

        # SExtractor uses 1-based pixel coordinates
        obj['x'] -= 1
        obj['y'] -= 1

        obj.meta['aper'] = aper

        if sort:
            if np.isscalar(aper):
                obj.sort('flux', reverse=True)
            else:
                # Table sorting by vector columns seems to be broken?..
                obj = obj[np.argsort(-obj['flux'][:,0])]

        if catfile is not None:
            shutil.copyfile(catname, catfile)
            log("Catalogue stored to", catfile)

    else:
        log("Error", res, "running SExtractor")

    result = obj

    if checkimages:
        result = [result]

        for name in checknames:
            result.append(fits.getdata(name))

    if _workdir is None:
        shutil.rmtree(workdir)

    return result

def make_series(mul=1.0, x=1.0, y=1.0, order=1, sum=False, zero=True):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if zero:
        res = [np.ones_like(x)*mul]
    else:
        res = []

    for i in range(1,order+1):
        maxr = i+1

        for j in range(maxr):
            res.append(mul * x**(i-j) * y**j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res

def match(obj_ra, obj_dec, obj_mag, obj_magerr, obj_flags, cat_ra, cat_dec, cat_mag, cat_magerr=None, cat_color=None, sr=3/3600, obj_x=None, obj_y=None, spatial_order=0, threshold=5.0, cat_saturation=None, verbose=False, robust=True):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    h = htm.HTM(10)

    oidx,cidx,dist = h.match(obj_ra, obj_dec, cat_ra, cat_dec, sr, maxmatch=0)

    log(len(dist), 'initial matches between', len(obj_ra), 'objects and', len(cat_ra), 'catalogue stars, sr=', sr*3600, 'arcsec')
    log('Median separation is', np.median(dist)*3600, 'arcsec')

    omag, omag_err, oflags = obj_mag[oidx], obj_magerr[oidx], obj_flags[oidx]
    cmag = cat_mag[cidx].filled(fill_value=np.nan)
    cmag_err = cat_magerr[cidx].filled(fill_value=np.nan) if cat_magerr is not None else np.zeros_like(cmag)

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
    if robust:
        log('Using robust fitting')
    else:
        log('Using weighted fitting')

    if cat_color is not None:
        ccolor = cat_color[cidx].filled(fill_value=np.nan)
        X += make_series(ccolor, x, y, order=0)
        log('Using color term')
    else:
        ccolor = np.zeros_like(cmag)

    X = np.vstack(X).T
    zero = cmag - omag # We will build a model for this definition of zero point
    zero_err = np.hypot(omag_err, cmag_err)
    weights = 1.0/zero_err**2

    idx0 = np.isfinite(omag) & np.isfinite(omag_err) & np.isfinite(cmag) & np.isfinite(cmag_err) & (oflags == 0) # initial mask
    if cat_color is not None:
        idx0 &= np.isfinite(ccolor)
    if cat_saturation is not None:
        idx0 &= cmag >= cat_saturation

    idx = idx0.copy()

    for iter in range(3):
        if np.sum(idx) < 3:
            log("Fit failed - %d objects remaining" % np.sum(idx))
            return None

        if robust:
            C = sm.RLM(zero[idx], X[idx]).fit()
        else:
            C = sm.WLS(zero[idx], X[idx], weights=weights[idx]).fit()

        zero_model = np.sum(X*C.params, axis=1)

        if threshold:
            rms = mad_std(((zero - zero_model)/zero_err)[idx])

            if robust:
                idx[idx] &= (np.abs((zero - zero_model)/zero_err)[idx] < threshold*rms)
            else:
                # idx = idx0.copy()
                idx[idx] &= (np.abs((zero - zero_model)/zero_err)[idx] < threshold*rms)

        log('Iteration', iter, ':', np.sum(idx), '/', len(idx), '-', np.std((zero - zero_model)[idx0]), np.std((zero - zero_model)[idx]), '-', np.std((zero - zero_model)[idx]/zero_err[idx]))

        if not threshold:
            break

    log(np.sum(idx), 'good matches')

    # Export the model
    def zero_fn(xx, yy):
        if xx is not None and yy is not None:
            x, y = xx - x0, yy - y0
        else:
            x, y = np.zeros_like(omag), np.zeros_like(omag)

        X = make_series(1.0, x, y, order=spatial_order)
        X = np.vstack(X).T

        return np.sum(X*C.params[0:X.shape[1]], axis=1)

    if cat_color is not None:
        X = make_series(order=spatial_order)
        color_term = C.params[len(X):][0]
        log('Color term is', color_term)
    else:
        color_term = None

    return {'oidx': oidx, 'cidx': cidx, 'dist': dist,
            'omag': omag, 'omag_err': omag_err,
            'cmag': cmag, 'cmag_err': cmag_err,
            'color': ccolor, 'color_term': color_term,
            'zero': zero, 'zero_err': zero_err,
            'zero_model': zero_model, 'zero_fn': zero_fn,
            'obj_zero': zero_fn(obj_x, obj_y),
            'ox': ox, 'oy': oy,
            'idx': idx, 'idx0': idx0}

def get_background(image, mask=None, method='sep', size=128, get_rms=False, **kwargs):
    if method == 'sep':
        bg = sep.Background(image, mask=mask, bw=size, bh=size, **kwargs)

        back,backrms = bg.back(), bg.rms()
    else: # photutils
        bg = photutils.Background2D(image, size, mask=mask, **kwargs)
        back,backrms = bg.background, bg.background_rms

    if get_rms:
        return back, backrms
    else:
        return back
