from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table

from astropy.convolution import convolve, Gaussian2DKernel

def run_hotpants(image, template, mask=None, template_mask=None, err=None, template_err=None,
                 extra=None, image_fwhm=None, template_fwhm=None,
                 image_gain=None, template_gain=1000, rel_r=3, rel_rss=4,
                 obj=None,
                 get_convolved=False, get_scaled=False, get_noise=False, get_kernel=False, get_header=False,
                 _tmpdir=None, _workdir=None, _exe=None, verbose=False):
    """Wrapper for running HOTPANTS to subtract a template from science image.

    To better understand the logic and available subtraction options you may check HOTPANTS documentation at https://github.com/acbecker/hotpants

    The routine tries to be clever and select reasonable defaults for running HOTPANTS, but you may always directly specify any HOTPANTS option manually through `extra` parameter.

    The noise model for both science and template images may optionally be provided as an external parameters (`err` and `template_err`). Moreover, if these parameters are set to `True`, the routine will try to build noise models itself. To do so, it will estimate the image background and background RMS. Then, this background RMS map is used as a primary component of the noise map. On top of that, the contribution of the sources is added by subtracting the background from original image, smoothing it a bit to mitigate pixel-level noise a bit, and then converting everything above zero to corresponding Poissonian noise contribution by dividing with gain value and taking a square root.

    `rel_r` and `rel_rss` define the scaling of corresponding HOTPANTS parameters (which are the convolution kernel half width and half-width of the sub-stamps used for kernel fitting) with image FWHM.

    The routine also uses "officially recommedned" logic for defining HOTPANTS `ng` parameter, that is to set it to :code:`[3, 6, 0.5*sigma_match, 4, 1.0*sigma_match, 2, 2.0*sigma_match]` where :code:`sigma_match = np.sqrt(image_fwhm**2 - template_fwhm**2) / 2.35`. This approach assumes that :code:`image_fwhm > template_fwhm`, and of course needs `image_fwhm` and `template_fwhm` to be provided.

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
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

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

    imin,imax = np.min(image[~mask]), np.max(image[~mask])
    tmin,tmax = np.min(template[~template_mask]), np.max(template[~template_mask])

    # As HOTPANTS uses inclusive checks for high values, let's extend their range just a bit
    imax += 0.01*(imax - imin)
    tmax += 0.01*(tmax - tmin)

    # Logic from https://arxiv.org/pdf/1608.01006.pdf
    imin = np.median(image[~mask]) - 10*mad_std(image[~mask])
    tmin = np.median(template[~template_mask]) - 10*mad_std(template[~template_mask])

    _nan = 1e-30 #

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='hotpants', dir=_tmpdir)

    image = image.copy()
    image[~np.isfinite(image)] = np.nanmedian(image) # _nan
    # image[mask] = _nan
    imagename = os.path.join(workdir, 'image.fits')
    fits.writeto(imagename, image, overwrite=True)
    imaskname = os.path.join(workdir, 'imask.fits')
    fits.writeto(imaskname, mask.astype(np.uint16), overwrite=True)

    template = template.copy()
    template[~np.isfinite(template)] = np.nanmedian(template) # _nan
    # template[template_mask] = _nan
    templatename = os.path.join(workdir, 'template.fits')
    fits.writeto(templatename, template, overwrite=True)
    tmaskname = os.path.join(workdir, 'tmask.fits')
    fits.writeto(tmaskname, template_mask.astype(np.uint16), overwrite=True)

    outname = os.path.join(workdir, 'diff.fits')

    stampname = os.path.join(workdir, 'stamps.reg')

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
        'tuk': tmax, # Limit used for kernel estimation, why not the same as 'tu'?..

        # Normalize result to input image
        'n': 'i',

        # Return all possible image planes as a result
        'allm': True,

        # Convolve template image
        'c': 't',

        # Add kernel info to the output
        'hki': True,

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
            import sep
            bg = sep.Background(image, mask)
            # image[mask] = bg.back()[mask]
            err = bg.rms() # Background noise level
            # Noise should be estimated on smoothed image
            kernel = Gaussian2DKernel(1)
            smooth = image.copy()
            smooth[~np.isfinite(smooth)] = np.nanmedian(smooth)
            smooth = convolve(smooth, kernel, mask=mask, preserve_nan=True)
            smooth[mask] = bg.back()[mask]
            serr = np.abs((smooth - bg.back()))
            if image_gain is not None:
                serr /= image_gain
            err = np.sqrt(err**2 + serr) # Contribution from the sources
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
            import sep
            bg = sep.Background(template, template_mask)
            template_err = bg.rms() # Background noise level
            # Noise should be estimated on smoothed image
            kernel = Gaussian2DKernel(1)
            smooth = template.copy()
            smooth[~np.isfinite(smooth)] = np.nanmedian(smooth)
            smooth = convolve(smooth, kernel, mask=template_mask, preserve_nan=True)
            smooth[template_mask] = bg.back()[template_mask]
            serr = np.abs((smooth - bg.back()))
            if template_gain is not None:
                serr /= template_gain
            template_err = np.sqrt(template_err**2 + serr) # Contribution from the sources
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
            sigma_match = np.sqrt(image_fwhm**2 - template_fwhm**2) / 2.35
            sigma_match = max(1.0, sigma_match)
            params['ng'] = [3, 6, 0.5*sigma_match, 4, 1.0*sigma_match, 2, 2.0*sigma_match]

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
