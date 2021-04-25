from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.stats import mad_std

from astropy.convolution import convolve, Gaussian2DKernel

def run_hotpants(image, template, mask=None, template_mask=None, err=None,
                 extra=None, image_fwhm=None, template_fwhm=None,
                 image_gain=None, template_gain=1000, rel_r=3, rel_rss=4,
                 get_convolved=False, get_scaled=False, get_noise=False,
                 _tmpdir=None, _workdir=None, verbose=False):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

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

    # Logic from https://arxiv.org/pdf/1608.01006.pdf
    imin = np.median(image[~mask]) - 10*mad_std(image[~mask])
    tmin = np.median(template[~template_mask]) - 10*mad_std(template[~template_mask])

    _nan = imin - 100000

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='hotpants', dir=_tmpdir)

    image = image.copy()
    image[~np.isfinite(image)] = np.nanmedian(image)
    # image[mask] = _nan
    imagename = os.path.join(workdir, 'image.fits')
    fits.writeto(imagename, image, overwrite=True)
    imaskname = os.path.join(workdir, 'imask.fits')
    fits.writeto(imaskname, mask.astype(np.uint16), overwrite=True)

    template = template.copy()
    template[~np.isfinite(template)] = np.nanmedian(template)
    # template[template_mask] = _nan
    templatename = os.path.join(workdir, 'template.fits')
    fits.writeto(templatename, template, overwrite=True)
    tmaskname = os.path.join(workdir, 'tmask.fits')
    fits.writeto(tmaskname, template_mask.astype(np.uint16), overwrite=True)

    outname = os.path.join(workdir, 'diff.fits')

    params = {
        'inim': imagename,
        'tmplim': templatename,
        'outim': outname,

        # Masks
        'imi': imaskname,
        'tmi': tmaskname,

        # Lower and upper valid values for image
        'il': imin,
        'iu': imax,

        # Lower and upper valid values for template
        'tl': tmin,
        'tu': tmax,

        # Normalize result to input image
        'n': 'i',

        # Return all possible image planes as a result
        'allm': True,

        # Convolve template image
        'c': 't',

        # Disable positional variance of the kernel and background
        'ko': 0,
        'bgo': 0,
    }

    # Error map for input image
    if err is not None:
        if type(err) is bool and err == True:
            # err=True means that we should estimate the noise
            log('Building noise model from the image')
            import sep

            if image_gain is None:
                image_gain = 1

            bg = sep.Background(image, mask)
            err = bg.rms() # Background noise level
            # Noise should be estimated on smoothed image
            kernel = Gaussian2DKernel(1 if image_fwhm is None else image_fwhm / 2 / 2.35)
            smooth = convolve(image, kernel, mask=mask)
            err = np.sqrt(err**2 + np.abs((smooth - bg.back()))/image_gain) # Contribution from the sources

        if hasattr(err, 'shape'):
            errname = os.path.join(workdir, 'err.fits')
            params['ini'] = errname
            fits.writeto(errname, err, overwrite=True)

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

    # Build command line
    command = ["hotpants"]

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

    if _workdir is None:
        shutil.rmtree(workdir)

    if len(result) == 1:
        return result[0]
    else:
        return result
