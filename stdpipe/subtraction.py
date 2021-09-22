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
        # Noise-scaled image
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
