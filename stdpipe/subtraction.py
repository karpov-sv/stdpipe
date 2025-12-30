
import os, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

from astropy.convolution import convolve, Gaussian2DKernel

# Drop-in replacement for numpy.fft which is supposedly faster
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.)

from scipy import ndimage
from scipy.optimize import least_squares
import statsmodels.api as sm

import sep

from . import astrometry
from . import pipeline
from . import psf


def run_hotpants(
    image,
    template,
    mask=None,
    template_mask=None,
    err=None,
    template_err=None,
    extra=None,
    image_fwhm=None,
    template_fwhm=None,
    image_gain=None,
    template_gain=1000,
    rel_r=3,
    rel_rss=4,
    nx=1,
    ny=None,
    obj=None,
    get_convolved=False,
    get_scaled=False,
    get_noise=False,
    get_kernel=False,
    get_header=False,
    _tmpdir=None,
    _workdir=None,
    _exe=None,
    verbose=False,
):
    """Wrapper for running HOTPANTS to subtract a template from science image.

    To better understand the logic and available subtraction options you may check
    HOTPANTS documentation at https://github.com/acbecker/hotpants

    The routine tries to be clever and select reasonable defaults for running HOTPANTS,
    but you may always directly specify any HOTPANTS option manually through `extra` parameter.

    The noise model for both science and template images may optionally be provided as an
    external parameters (`err` and `template_err`). Moreover, if these parameters are set
    to `True`, the routine will try to build noise models itself. To do so, it will estimate
    the image background and background RMS. Then, this background RMS map is used as a primary
    component of the noise map. On top of that, the contribution of the sources is added by
    subtracting the background from original image, smoothing it a bit to mitigate pixel-level
    noise a bit, and then converting everything above zero to corresponding Poissonian noise
    contribution by dividing with gain value and taking a square root.

    `rel_r` and `rel_rss` define the scaling of corresponding HOTPANTS parameters (which are
    the convolution kernel half width and half-width of the sub-stamps used for kernel fitting)
    with image FWHM.

    The routine also uses "officially recommedned" logic for defining HOTPANTS `ng` parameter,
    that is to set it to :code:`[3, 6, 0.5*sigma_match, 4, 1.0*sigma_match, 2, 2.0*sigma_match]`
    where :code:`sigma_match = np.sqrt(image_fwhm**2 - template_fwhm**2) / 2.35`. This approach
    assumes that :code:`image_fwhm > template_fwhm`, and of course needs `image_fwhm` and
    `template_fwhm` to be provided.

    :param image: Input science image as a Numpy array
    :param template: Input template image, should have the same shape as a science image
    :param mask: Science image mask as a boolean array (True values will be masked), optional
    :param template_mask: Template image mask as a boolean array (True values will be masked), optional
    :param err: Science image error map (expected RMS of every pixel). If set to `True`, the code will try to build the noise map directly from the image and `image_gain` parameter. Optional
    :param template_err: Template image error map. If set to `True`, the code will try to build the noise map directly from the template and `template_gain` parameter. Optional
    :param extra: Extra parameters to be passed to HOTPANTS executable. Should be a dictionary with parameter names as keys. Optional
    :param image_fwhm: FWHM of science image in pixels, optional
    :param template_fwhm: FWHM of template image in pixels, optional
    :param image_gain: Gain of science image
    :param template_gain: Gain of template image
    :param rel_r: If specified, HOTPANTS `r` parameters will be set to :code:`image_fwhm*rel_r`
    :param rel_rss: If specified, HOTPANTS `rss` parameters will be set to :code:`image_fwhm*rel_rss`
    :param nx: Number of image sub-regions in `x` direction
    :param ny: Number of image sub-regions in `y` direction
    :param obj: List of objects detected on science image. If provided, it will be used to better place the sub-stamps used to derive the convolution kernel.
    :param get_convolved: Whether to also return the convolved template image
    :param get_scaled: Whether to also return noise-normalized difference image
    :param get_noise: Whether to also return difference image noise model
    :param get_kernel: Whether to also return convolution kernel used in the subtraction
    :param get_header: Whether to also return the FITS header from HOTPANTS output file
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running HOTPANTS. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to HOTPANTS executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The difference image and, optionally, other kinds of images as requested by the `get_*` options above

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
        # Find HOTPANTS binary in common paths
        for exe in ['hotpants']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find HOTPANTS binary")
        return None
    # else:
    #     log("Using HOTPANTS binary at", binname)

    if mask is None:
        mask = ~np.isfinite(image)
    else:
        mask = mask | ~np.isfinite(image)

    if template_mask is None:
        template_mask = ~np.isfinite(template)
    else:
        template_mask = template_mask | ~np.isfinite(template)

    imin, imax = np.min(image[~mask]), np.max(image[~mask])
    tmin, tmax = np.min(template[~template_mask]), np.max(template[~template_mask])

    # As HOTPANTS uses inclusive checks for high values, let's extend their range just a bit
    imax += 0.01 * (imax - imin)
    tmax += 0.01 * (tmax - tmin)

    # Logic from https://arxiv.org/pdf/1608.01006.pdf
    imin = np.median(image[~mask]) - 10 * mad_std(image[~mask])
    tmin = np.median(template[~template_mask]) - 10 * mad_std(template[~template_mask])

    _nan = 1e-30  #

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='hotpants', dir=_tmpdir)
    )

    image = image.copy()
    image[~np.isfinite(image)] = np.nanmedian(image)  # _nan
    # image[mask] = _nan
    imagename = os.path.join(workdir, 'image.fits')
    fits.writeto(imagename, image, overwrite=True)
    imaskname = os.path.join(workdir, 'imask.fits')
    fits.writeto(imaskname, mask.astype(np.uint16), overwrite=True)

    template = template.copy()
    template[~np.isfinite(template)] = np.nanmedian(template)  # _nan
    # template[template_mask] = _nan
    templatename = os.path.join(workdir, 'template.fits')
    fits.writeto(templatename, template, overwrite=True)
    tmaskname = os.path.join(workdir, 'tmask.fits')
    fits.writeto(tmaskname, template_mask.astype(np.uint16), overwrite=True)

    outname = os.path.join(workdir, 'diff.fits')

    stampname = os.path.join(workdir, 'stamps.reg')

    if not ny:
        ny = nx

    params = {
        'inim': imagename,
        'tmplim': templatename,
        'outim': outname,
        'savexy': stampname,
        # Masks
        'imi': imaskname,
        'tmi': tmaskname,
        # Lower and upper valid values for image
        'il': imin,
        'iu': imax,
        # Lower and upper valid values for template
        'tl': tmin,
        'tu': tmax,
        'tuk': tmax,  # Limit used for kernel estimation, why not the same as 'tu'?..
        # Normalize result to input image
        'n': 'i',
        # Return all possible image planes as a result
        'allm': True,
        # Convolve template image
        'c': 't',
        # Add kernel info to the output
        'hki': True,
        # Number of sub-regions
        'nrx': nx,
        'nry': ny,
        # Disable positional variance of the kernel and background
        'ko': 0,
        'bgo': 0,
        'v': 2,
    }

    # Error map for input image
    if err is not None:
        if type(err) is bool and err == True:
            # err=True means that we should estimate the noise
            log('Building noise model from the image')

            bg = sep.Background(image, mask)
            # image[mask] = bg.back()[mask]
            err = bg.rms()  # Background noise level
            # Noise should be estimated on smoothed image
            kernel = Gaussian2DKernel(1)
            smooth = image.copy()
            smooth[~np.isfinite(smooth)] = np.nanmedian(smooth)
            # smooth = convolve(smooth, kernel, mask=mask, preserve_nan=True)
            smooth[mask] = bg.back()[mask]
            serr = np.abs((smooth - bg.back()))
            if image_gain is not None and image_gain:
                serr /= image_gain
            err = np.sqrt(err ** 2 + serr)  # Contribution from the sources
            # err[mask] = 1/_nan

            # image[mask|template_mask] = np.random.normal(0, bg.rms()[mask|template_mask])
            # fits.writeto(imagename, image, overwrite=True)

        if hasattr(err, 'shape'):
            errname = os.path.join(workdir, 'err.fits')
            params['ini'] = errname
            fits.writeto(errname, err, overwrite=True)

    if template_err is not None:
        if type(template_err) is bool and template_err == True:
            # err=True means that we should estimate the noise
            log('Building noise model from the template')

            bg = sep.Background(template, template_mask)
            template_err = bg.rms()  # Background noise level
            # Noise should be estimated on smoothed image
            kernel = Gaussian2DKernel(1)
            smooth = template.copy()
            smooth[~np.isfinite(smooth)] = np.nanmedian(smooth)
            # smooth = convolve(smooth, kernel, mask=template_mask, preserve_nan=True)
            smooth[template_mask] = bg.back()[template_mask]
            serr = np.abs((smooth - bg.back()))
            if template_gain is not None and template_gain:
                serr /= template_gain
            template_err = np.sqrt(
                template_err ** 2 + serr
            )  # Contribution from the sources
            # template_err[template_mask] = 1/_nan

            # template[mask|template_mask] = np.random.normal(0, bg.rms()[mask|template_mask])
            # fits.writeto(templatename, template, overwrite=True)

        if hasattr(template_err, 'shape'):
            terrname = os.path.join(workdir, 'terr.fits')
            params['tni'] = terrname
            fits.writeto(terrname, template_err, overwrite=True)

    if image_fwhm is not None and template_fwhm is not None:
        # Recommended logic for sigma_match
        if image_fwhm > template_fwhm:
            sigma_match = np.sqrt(image_fwhm ** 2 - template_fwhm ** 2) / 2.35
            sigma_match = max(1.0, sigma_match)
            params['ng'] = [
                3,
                6,
                0.5 * sigma_match,
                4,
                1.0 * sigma_match,
                2,
                2.0 * sigma_match,
            ]

        elif image_fwhm < template_fwhm:
            sigma_match = np.sqrt(template_fwhm ** 2 - image_fwhm ** 2) / 2.35
            sigma_match = max(1.0, sigma_match)
            params['ng'] = [
                3,
                6,
                0.5 * sigma_match,
                4,
                1.0 * sigma_match,
                2,
                2.0 * sigma_match,
            ]
            # TODO: switch to convolve-image mode?..


    if image_fwhm is not None:
        # Logic from https://arxiv.org/pdf/1608.01006.pdf suggests 2.5 and 6 here
        params['r'] = int(np.ceil(image_fwhm * rel_r))
        params['rss'] = int(np.ceil(image_fwhm * rel_rss))

    if image_gain is not None:
        params['ig'] = image_gain

    if template_gain is not None:
        params['tg'] = template_gain

    if extra is not None:
        params.update(extra)

    # Set the stamp locations from detected objects
    if obj is not None and len(obj):
        log('Using %d external positions for substamps' % len(obj))
        xyname = os.path.join(workdir, 'objects.xy')
        np.savetxt(xyname, [[_['x'] + 1, _['y'] + 1] for _ in Table(obj)], fmt='%.1f')
        params['ssf'] = xyname

    # Build command line
    command = [binname]

    for key in params.keys():
        if params[key] is None:
            pass
        elif type(params[key]) == bool:
            if params[key]:
                command.append('-' + key)
            else:
                pass
        else:
            if type(params[key]) == str:
                # Quote string if necessary
                value = shlex.quote(params[key])
            elif hasattr(params[key], '__len__'):
                # List or array, for multi-valued arguments
                value = " ".join([str(_) for _ in params[key]])
            else:
                value = str(params[key])

            command.append('-' + key + ' ' + value)

    command = " ".join(command)

    if not verbose:
        command += " >/dev/null 2>/dev/null"

    log('Will run HOTPANTS like that:')
    log(command)

    if os.path.exists(outname):
        os.unlink(outname)

    # Run the command
    os.system(command)

    if not os.path.exists(outname):
        log('HOTPANTS run failed')
        return None
    elif not np.any(fits.getdata(outname, 0)):
        log('HOTPANTS failed to perform subtraction')
        return None
    else:
        log('HOTPANTS run succeeded')

    # Difference image
    result = [fits.getdata(outname, 0)]

    header = fits.getheader(outname)

    if os.path.exists(stampname):
        with open(stampname, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]

        header['NSTAMPS'] = len(lines)

        log('%d stamps used' % len(lines))

    if get_convolved:
        # Convolved image
        conv = fits.getdata(outname, 1)
        result.append(conv)

    if get_scaled:
        # Noise-scaled difference image
        sdiff = fits.getdata(outname, 2)
        result.append(sdiff)

    if get_noise:
        # Noise image
        noise = fits.getdata(outname, 3)
        result.append(noise)

    if get_kernel:
        # Kernel information
        kernel = Table.read(outname, format='fits', hdu=4)
        result.append(kernel)

    if get_header:
        # Header with metadata
        result.append(header)

    if _workdir is None:
        shutil.rmtree(workdir)

    if len(result) == 1:
        return result[0]
    else:
        return result


def run_zogy(
    image,
    template,
    mask=None,
    template_mask=None,
    err=None,
    template_err=None,
    image_psf=None,
    template_psf=None,
    image_gain=None,
    template_gain=None,
    image_fwhm=None,
    template_fwhm=None,
    scale=None,
    psf_clean=0,
    dx=0.25,
    dy=0.25,
    nx=1,
    ny=None,
    overlap=50,
    fit_scale=True,
    fit_shift=True,
    good_regions=None,
    image_obj=None,
    template_obj=None,
    get_psf=False,
    get_Fpsf=False,
    nthreads=0,
    verbose=False,
    **kwargs
):
    """Image subtraction using Zackay–Ofek–Gal-Yam (ZOGY) algorithm, as described in
    http://dx.doi.org/10.3847/0004-637X/830/1/27

    The science and template images should be already aligned astrometrically, and roughly
    matched photometrically, i.e. have similar flux scale. There is an option to fit for
    the small difference in flux scales (`fit_scale`), which is enabled by default. Also,
    optionally (if `fit_shift` is set) it may fit for a small systematic positional error
    (sub-pixel shift) of the template.

    Required inputs are science and template images (plus optionally their
    masks), their noise maps and PSFs, as well as their relative flux scale. If
    noise maps are not provided, they will be constructed from the input images
    using estimated background noise and gain values for source contributions.
    If PSFs are not specified, they will be also estimated from the images
    using :func:`stdpipe.psf.run_psfex`.  Flux normalization for PSF estimation
    also requires FWHMs of the images, which may either be specified directly,
    or estimated from the images too.

    Relative template image flux scaling may either be provided as `scale`
    argument, or it may be estimated from the lists of objects detected in both
    images. These may either be provided directly (`image_obj` and
    `template_obj`), or derived during PSF estimation. These objects will also
    be used for restricting the scale/shift fitting to their vicinities
    only. Alternatively, the map of good regions may be directly provided as
    `good_regions` argument.

    The image will be optionally split into given number (`nx`x`ny`) of
    sub-images, subtracted independently, and then the results will be stitched
    back. Sub-images will overlap by `overlap` pixels.

    :param image: Input science image as a Numpy array
    :param template: Input template image, should have the same shape as a science image
    :param mask: Science image mask as a boolean array (True values will be masked), optional
    :param template_mask: Template image mask as a boolean array (True values will be masked), optional

    :param err: Science image noise map (expected RMS of every pixel). If not set, the code will try to build the noise map directly from the image and `image_gain` parameter
    :param template_err: Template image noise map. If not set, the code will try to build the noise map directly from the template and `template_gain` parameter
    :param image_gain: Gain of science image, to be used for construction of the noise map
    :param template_gain: Gain of template image, to be used for construction of the noise map

    :param image_psf: PSF for science image. If not provided, will be estimated by :func:`stdpipe.psf.run_psfex`.
    :param template_psf: PSF for template image. If not provided, will be estimated by :func:`stdpipe.psf.run_psfex`.
    :param image_fwhm: FWHM of science image, to be used for estimated PSF normalization
    :param template_fwhm: FWHM of template image, to be used for estimated PSF normalization

    :param scale: Template image flux scale relative to science image

    :param psf_clean: If non-zero, will clean the PSF regions with relative values less than this factor
    :param dx: Astrometric uncertainty (sigma) in x coordinate
    :param dy: Astrometric uncertainty (sigma) in y coordinate

    :param nx: Number of sub-images in `x` direction
    :param ny: Number of sub-images in `y` direction
    :param overlap: If set, defines how much sub-images will overlap, in pixels.

    :param fit_scale: If set, will fit for the difference in flux scales between template and science images
    :param fit_shift: If set, will also fit for the sub-pixel shift between template and science images
    :param good_regions: If set, this boolean map will be used to restrict scale/shift fitting to only these regions
    :param image_obj: List of objects detected in science image. If provided, they will be used for placing good regions
    :param template_obj: List of objects detected in science image. If provided, they will be used for deriving template flux scale
    :param get_psf: If set, will also return the PSF of the difference image
    :param get_Fpsf: If set, will also return the optimal PSF photometry image and its error
    :param nthreads: Set the number of threads to use for FFTW routines (0 for auto)
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param **kwargs: Extra parameters to be passed to :func:`stdpipe.psf.run_psfex` for PSF estimation.
    :returns: The list of the difference image and other subtraction results.

    - D: Subtracted image
    - S_corr: Corrected subtracted image
    - P_D: PSF of subtracted image, if :code:`get_psf=True`
    - Fpsf: optimal PSF photometry image, if :code:`get_Fpsf=True`
    - Fpsf_err: optimal PSF photometry error image, if :code:`get_Fpsf=True`

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if mask is None:
        mask = ~np.isfinite(image)
    else:
        mask = mask | ~np.isfinite(image)

    if template_mask is None:
        template_mask = ~np.isfinite(template)
    else:
        template_mask = template_mask | ~np.isfinite(template)

    # Subtract the background
    image_bg = sep.Background(image, mask)
    template_bg = sep.Background(template.astype(np.double), template_mask)

    N_full = image - image_bg.back()
    R_full = template - template_bg.back()
    # Set to zero the regions where we have no data
    N_full[~np.isfinite(image)] = 0
    R_full[~np.isfinite(template)] = 0

    if err is None:
        log('Building noise model from the image')
        U_N_full = image_bg.rms()
        if image_gain:
            U_N_full = np.sqrt(U_N_full ** 2 + np.abs(N_full) / image_gain)

        SN = image_bg.globalrms
        log('Image mean %.1f rms %.2f ADU' % (image_bg.globalback, image_bg.globalrms))
    else:
        U_N_full = err
        SN = np.median(err)

    if template_err is None:
        log('Building noise model from the template')
        U_R_full = template_bg.rms()
        if template_gain:
            U_R_full = np.sqrt(U_R_full ** 2 + np.abs(R_full) / template_gain)

        SR = template_bg.globalrms
        log('Template mean %.1f rms %.2f ADU' % (template_bg.globalback, template_bg.globalrms))
    else:
        U_R_full = template_err
        SR= np.median(template_err)

    # Artificially assign large uncertainty to the regions we set to zero
    U_N_full[~np.isfinite(image)] = np.nanmax(U_N_full)
    U_R_full[~np.isfinite(template)] = np.nanmax(U_R_full)

    # Estimate image PSF, if not specified
    if image_psf is None:
        log("Estimating image PSF")
        # TODO: should we overwrite image_obj if it was set by the user?..
        image_psf,image_obj = psf.run_psfex(
            image, mask=mask, gain=image_gain,
            aper=2.0*image_fwhm if image_fwhm else None,
            verbose=verbose, get_obj=True,
            **kwargs
        )

    # Estimate template PSF, if not specified
    if template_psf is None:
        log("Estimating template PSF")
        # TODO: should we overwrite image_obj if it was set by the user?..
        template_psf,template_obj = psf.run_psfex(
            template, mask=template_mask, gain=template_gain,
            aper=2.0*template_fwhm if template_fwhm else None,
            verbose=verbose, get_obj=True,
            **kwargs
        )

    # "Good" regions for flux scale refinement
    if good_regions is not None and isinstance(good_regions, np.ndarray):
        good_regions = good_regions.astype(np.bool)
        log("Will use user-provided regions for scale/shift fitting")
    else:
        good_regions = np.zeros_like(image, dtype=bool)

    # Guess initial relative flux scale
    if image_obj and template_obj:
        # Use half FWHM radius for matching
        sr = 0.5*max(image_psf['fwhm'], template_psf['fwhm'])
        iidx,tidx,_ = astrometry.planar_match(
            image_obj['x'], image_obj['y'],
            template_obj['x'], template_obj['y'],
            sr
        )

        if len(iidx):
            # Exclude all flagged objects
            fidx = (image_obj['flags'][iidx] & 0x100) == 0
            fidx &= (template_obj['flags'][tidx] & 0x100) == 0

            if np.sum(fidx) > 5:
                # Compute photometric zero point shift between two object lists
                ZP = template_obj['mag'][tidx][fidx] - image_obj['mag'][iidx][fidx]
                dZP = np.hypot(image_obj['magerr'][iidx][fidx], template_obj['magerr'][tidx][fidx])
                X = np.ones_like(ZP).T
                # Weighted robust regression
                C = sm.RLM(ZP / dZP, (X.T / dZP).T).fit()
                scale = 10**(-0.4*C.params[0])
                log("Estimated initial template flux scale Fr = %.3g" % scale)
            else:
                log("Not enough matched objects to guess template flux scale")

        if len(iidx) > 0 and (fit_scale or fit_shift) and np.sum(good_regions) == 0:
            # Use matched objects (they are all unflagged) to define good regions
            size = int(6*sr) # 3*FWHM
            N = 0
            for i in range(len(iidx)):
                x0 = int(image_obj['x'][iidx][i])
                y0 = int(image_obj['y'][iidx][i])

                # Exclude objects too close to the edges
                if (x0 < size or x0 >= image.shape[1] - size
                    or y0 < size or y0 >= image.shape[0] - size):
                    continue

                # Exclude sub-regions containing masked pixels
                if np.sum((mask|template_mask)[y0 - size: y0 + size, x0 - size:x0 + size]) == 0:
                    good_regions[y0 - size: y0 + size, x0 - size:x0 + size] = True
                    N += 1

            if N > 0:
                log(N, "regions selected for scale/shift fitting")

    else:
        log("Initial template flux scale Fr = %.3g" % scale)

    if (fit_scale or fit_shift) and np.sum(good_regions) == 0:
        good_regions = np.ones_like(good_regions, dtype=bool)
        log("Will fit for scale/shift using whole image")

    # Split the image into sub-images to later stitch it back
    if not ny:
        ny = nx

    D_full = np.zeros_like(image, dtype=np.double)
    S_corr_full = np.zeros_like(image, dtype=np.double)
    D_full = np.zeros_like(image, dtype=np.double)
    Fpsf_full = np.zeros_like(image, dtype=np.double)
    Fpsf_err_full = np.zeros_like(image, dtype=np.double)

    for i, x0, y0, N, R, U_N, U_R, good_reg in pipeline.split_image(
            N_full, R_full, U_N_full, U_R_full, good_regions,
            nx=nx, ny=ny, overlap=overlap,
            get_index=True, get_origin=True,
            verbose=True if nx > 1 and ny > 1 else False
    ):
        # PSF stamps at the centers of sub-image
        P_N_small = psf.get_psf_stamp(
            image_psf,
            x=x0 + N.shape[1]/2, y=y0 + N.shape[0]/2,
            dx=0, dy=0, normalize=True
        )
        P_R_small = psf.get_psf_stamp(
            template_psf,
            x=x0 + N.shape[1]/2, y=y0 + N.shape[0]/2,
            dx=0, dy=0, normalize=True
        )

        if psf_clean:
            # Set to zero the wings with relative amplitude less than psf_clean factor
            P_N_small[P_N_small < psf_clean*np.max(P_N_small)] = 0
            P_N_small /= np.sum(P_N_small)

            P_R_small[P_R_small < psf_clean*np.max(P_R_small)] = 0
            P_R_small /= np.sum(P_R_small)

        Fn = 1 # Fixed, so that the difference will be in science image flux scale
        Fr = scale if scale else 1 # Will be optionally adjusted later

        # Place PSF at the center of image with same size as new / reference
        P_N = np.zeros_like(N)
        P_R = np.zeros_like(R)
        idxN = tuple([slice(int(N.shape[0]/2) - int(P_N_small.shape[0]/2),
                            int(N.shape[0]/2) + int(P_N_small.shape[0]/2) + 1),
                      slice(int(N.shape[1]/2) - int(P_N_small.shape[1]/2),
                            int(N.shape[1]/2) + int(P_N_small.shape[1]/2) + 1)])
        idxR = tuple([slice(int(R.shape[0]/2) - int(P_R_small.shape[0]/2),
                            int(R.shape[0]/2) + int(P_R_small.shape[0]/2) + 1),
                      slice(int(R.shape[1]/2) - int(P_R_small.shape[1]/2),
                            int(R.shape[1]/2) + int(P_R_small.shape[1]/2) + 1)])
        P_N[idxN] = P_N_small
        P_R[idxR] = P_R_small

        # Shift the PSF to the origin so it will not introduce a shift
        P_N = fft.fftshift(P_N)
        P_R = fft.fftshift(P_R)

        # Take all the Fourier Transforms
        N_hat = fft.fft2(N, threads=nthreads)
        R_hat = fft.fft2(R, threads=nthreads)

        P_N_hat = fft.fft2(P_N, threads=nthreads)
        P_R_hat = fft.fft2(P_R, threads=nthreads)

        if fit_scale or fit_shift:
            # Estimate beta=Fn/Fr and shift using Equations 37-39
            def fn(par):
                Fn = 1
                Fr = par[0]

                D_hat_den = np.sqrt(SN**2 * Fr**2 * np.abs(P_R_hat**2) + SR**2 * Fn**2 * np.abs(P_N_hat**2))
                D_hat_n = P_R_hat * N_hat / D_hat_den
                D_hat_r = P_N_hat * R_hat / D_hat_den

                if len(par) > 1:
                    D_hat_r = ndimage.fourier_shift(D_hat_r, par[1:])

                DD_n = np.real(fft.ifft2(D_hat_n, threads=0))
                DD_r = np.real(fft.ifft2(D_hat_r, threads=0))

                return (Fr * DD_n - Fn * DD_r)[good_reg].flatten()

            if np.sum(good_reg) > 10:
                if fit_scale and fit_shift:
                    log('Fitting for flux scale difference and sub-pixel template shift')
                    C = least_squares(fn, [scale, 0, 0], bounds=((0, -0.5, -0.5), (np.inf, 0.5, 0.5)), verbose=0)
                elif fit_scale:
                    log('Fitting for flux scale difference')
                    C = least_squares(fn, [scale], bounds=(0, np.inf), verbose=0)
                else:
                    log('Fitting for sub-pixel template shift')
                    C = least_squares(fn, [scale, 0, 0], bounds=((scale*0.9999, -0.5, -0.5), (scale*1.0001, 0.5, 0.5)), verbose=0)

                if C.success:
                    if fit_scale:
                        Fr = C.x[0]
                        log('Template flux scale Fr = %.3g' % Fr)
                    if len(C.x) > 1:
                        log('Shift is dy = %.2f dx = %.2f' % (C.x[1], C.x[2]))
                        R_hat = ndimage.fourier_shift(R_hat, C.x[1:])
                else:
                    log('Fitting failed')
            else:
                log('Not enough good pixels for fitting')

        # Fourier Transform of Difference Image (Equation 13)
        D_hat_num = (Fr * P_R_hat * N_hat - Fn * P_N_hat * R_hat)
        D_hat_den = np.sqrt(SN**2 * Fr**2 * np.abs(P_R_hat**2) + SR**2 * Fn**2 * np.abs(P_N_hat**2))
        D_hat = D_hat_num / D_hat_den

        # Flux-based zero point (Equation 15)
        FD = Fr * Fn / np.sqrt(SN**2 * Fr**2 + SR**2 * Fn**2)

        # Difference image corrected for correlated noise
        D = np.real(fft.ifft2(D_hat, threads=nthreads)) / FD

        # Fourier Transform of PSF of Subtraction Image (Equation 14)
        P_D_hat = Fr * Fn * P_R_hat * P_N_hat / FD / D_hat_den

        # PSF of Subtraction Image D
        P_D = np.real(fft.ifft2(P_D_hat, threads=nthreads))
        P_D = fft.ifftshift(P_D)
        P_D = P_D[idxN]

        # Fourier Transform of Score Image (Equation 17)
        S_hat = FD * D_hat * np.conj(P_D_hat)

        # Score Image
        S = np.real(fft.ifft2(S_hat, threads=nthreads))

        # Now start calculating Scorr matrix (including all noise terms)

        # Start out with source noise

        # Sigma to variance
        V_N = U_N**2
        V_R = U_R**2

        # Fourier Transform of variance images
        V_N_hat = fft.fft2(V_N, threads=nthreads)
        V_R_hat = fft.fft2(V_R, threads=nthreads)

        # Equation 28
        kr_hat = Fr * Fn**2 * np.conj(P_R_hat) * np.abs(P_N_hat**2) / (D_hat_den**2)
        kr = np.real(fft.ifft2(kr_hat, threads=nthreads))

        # Equation 29
        kn_hat = Fn * Fr**2 * np.conj(P_N_hat) * np.abs(P_R_hat**2) / (D_hat_den**2)
        kn = np.real(fft.ifft2(kn_hat, threads=nthreads))

        # Noise in New Image: Equation 26
        V_S_N = np.real(fft.ifft2(V_N_hat * fft.fft2(kn**2, threads=nthreads), threads=nthreads))

        # Noise in Reference Image: Equation 27
        V_S_R = np.real(fft.ifft2(V_R_hat * fft.fft2(kr**2, threads=nthreads), threads=nthreads))

        # Astrometric Noise
        # Equation 31
        S_N = np.real(fft.ifft2(kn_hat * N_hat, threads=nthreads))
        dSNdx = S_N - np.roll(S_N, 1, axis=1)
        dSNdy = S_N - np.roll(S_N, 1, axis=0)

        # Equation 30
        V_ast_S_N = dx**2 * dSNdx**2 + dy**2 * dSNdy**2

        # Equation 33
        S_R = np.real(fft.ifft2(kr_hat * R_hat))
        dSRdx = S_R - np.roll(S_R, 1, axis=1)
        dSRdy = S_R - np.roll(S_R, 1, axis=0)

        # Equation 32
        V_ast_S_R = dx**2 * dSRdx**2 + dy**2 * dSRdy**2

        # Calculate Scorr
        S_corr = S / np.sqrt(V_S_N + V_S_R + V_ast_S_N + V_ast_S_R)

        # PSF photometry (Equations 41-43)
        F_S = np.sum(Fn**2 * Fr**2 * np.abs(P_N_hat**2) * np.abs(P_R_hat**2) / (D_hat_den**2))
        F_S /= S.shape[1]*S.shape[1] # divide by the number of pixels due to FFT normalization
        Fpsf = S / F_S # optimal PSF photometry, alpha in Equation 41
        Fpsf_err = np.sqrt(V_S_N + V_S_R) / F_S

        # Stitch the piece back to the tapestry
        D_full[y0:y0 + N.shape[0], x0:x0 + N.shape[1]] = D
        S_corr_full[y0:y0 + N.shape[0], x0:x0 + N.shape[1]] = S_corr
        Fpsf_full[y0:y0 + N.shape[0], x0:x0 + N.shape[1]] = Fpsf
        Fpsf_err_full[y0:y0 + N.shape[0], x0:x0 + N.shape[1]] = Fpsf_err

    result = [D_full, S_corr_full]

    if get_psf:
        result += [P_D]

    if get_Fpsf:
        result += [Fpsf_full, Fpsf_err_full]

    return result
