import os, re, shutil, tempfile, shlex
import numpy as np

from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table


# Drop-in replacement for numpy.fft which is supposedly faster
import pyfftw
import pyfftw.interfaces.numpy_fft as fft

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.0)

from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import statsmodels.api as sm

try:
    import sep_x as sep
except ImportError:
    import sep

from . import astrometry
from . import pipeline
from . import psf
from . import sfft as sfft_module


# Minimal S/N of a star in both the image and the template to use it in the stamp check
_STAMP_CHECK_MIN_SN = 10
# Minimal number of such stars in a region for the stamp check
_STAMP_CHECK_MIN_STARS = 5
# Relative deviation of the kernel sum from the flux ratio of the stars that is always tolerated
_STAMP_CHECK_REL_TOL = 0.02


def _read_hotpants_stamps(filename):
    """Centers (0-based) and half-widths of the stamps listed in HOTPANTS `-savexy` region file."""
    xs, ys, hws = [], [], []

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                m = re.search(r'box\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([\d.]+)', line)
                if m:
                    xs.append(float(m.group(1)) - 1)
                    ys.append(float(m.group(2)) - 1)
                    hws.append((float(m.group(3)) - 1) / 2)

    return np.array(xs), np.array(ys), np.array(hws)


def _read_hotpants_regions(header):
    """Regions (0-based inclusive x1, x2, y1, y2) and their kernel sums from HOTPANTS output header."""
    regions = []

    for i in range(header.get('NREGION', 0)):
        reg = header.get('REGION%02d' % i)
        ksum = header.get('KSUM%02d' % i)
        if reg is None or ksum is None:
            continue

        m = re.match(r'\[(\d+):(\d+),(\d+):(\d+)\]', str(reg).strip())
        if m:
            x1, x2, y1, y2 = [int(_) - 1 for _ in m.groups()]
            regions.append((x1, x2, y1, y2, float(ksum)))

    return regions


def _measure_flux_ratios(image, template, good, xs, ys, r, flagged=None):
    """Ratios of image to template fluxes in circular apertures of radius `r`.

    Local background is the median of the annulus between `r` and `1.5*r`, and only the
    pixels valid (`good`) in both images are used. Returns ratios, their errors, S/N in
    the image and the template, all NaN where the aperture is mostly invalid, and the
    numbers of `flagged` pixels inside the apertures.
    """
    n = len(xs)
    ratio, ratio_err = np.full(n, np.nan), np.full(n, np.nan)
    sn1, sn2 = np.full(n, np.nan), np.full(n, np.nan)
    nflagged = np.zeros(n, dtype=int)

    r_out = 1.5 * r
    hw = int(np.ceil(r_out)) + 1

    for i, (x, y) in enumerate(zip(xs, ys)):
        x0, y0 = int(np.round(x)), int(np.round(y))
        sx = slice(max(0, x0 - hw), min(image.shape[1], x0 + hw + 1))
        sy = slice(max(0, y0 - hw), min(image.shape[0], y0 + hw + 1))

        yy, xx = np.mgrid[sy, sx]
        dist = np.hypot(xx - x, yy - y)
        g = good[sy, sx]
        aper = (dist <= r) & g
        ann = (dist > r) & (dist <= r_out) & g

        if flagged is not None:
            nflagged[i] = np.sum(flagged[sy, sx][dist <= r])

        if np.sum(aper) < 0.5 * np.pi * r**2 or np.sum(ann) < 10:
            continue

        fluxes, errs = [], []
        for data in (image, template):
            cut = data[sy, sx]
            bg = np.median(cut[ann])
            rms = mad_std(cut[ann])
            fluxes.append(np.sum(cut[aper] - bg))
            # Background noise inside the aperture plus the error of the median background
            errs.append(rms * np.sqrt(np.sum(aper) * (1 + np.pi / 2 * np.sum(aper) / np.sum(ann))))

        if fluxes[1] == 0 or errs[0] <= 0 or errs[1] <= 0:
            continue

        ratio[i] = fluxes[0] / fluxes[1]
        ratio_err[i] = np.abs(ratio[i]) * np.hypot(errs[0] / fluxes[0], errs[1] / fluxes[1])
        sn1[i], sn2[i] = fluxes[0] / errs[0], fluxes[1] / errs[1]

    return ratio, ratio_err, sn1, sn2, nflagged


def _check_hotpants_stamps(
    image,
    template,
    good,
    stampname,
    header,
    flagged=None,
    obj=None,
    aper=None,
    sigma=3.0,
    log=print,
):
    """Find the stamps that pull HOTPANTS kernel sum away from the flux ratio of the stars.

    HOTPANTS rejects the stamps by their residuals normalized by the expected noise, which
    does not catch a bright star whose profile is distorted in one of the images (e.g.
    saturated in the template) - its noise is large, and it is the one fitted best, while
    driving the kernel normalization for everything else. Here the kernel sum of every
    region is compared with the robust median of image to template flux ratios of the
    stars (from `obj` if given, else from the stamps; only isolated ones, and without
    `flagged` pixels, e.g. outside of HOTPANTS valid ranges), and if they disagree, the stamps
    deviating from the median in the same direction as the kernel sum are found, and
    the most significant one per region is returned.

    Returns the list of (x, y) positions of the template peaks of these stamps.
    """
    sx, sy, shw = _read_hotpants_stamps(stampname)
    regions = _read_hotpants_regions(header)

    if not len(sx) or not regions:
        log('Stamp check: no stamps or kernel sums in HOTPANTS output, skipping')
        return []

    r = aper if aper else np.median(shw)

    if obj is not None and len(obj):
        rx, ry = np.asarray(obj['x'], dtype=float), np.asarray(obj['y'], dtype=float)
        if len(rx) > 1:
            # Isolated stars only, as neighbours in the background annulus are spread
            # differently by the different PSFs, and bias the ratio
            dist, _ = cKDTree(np.c_[rx, ry]).query(np.c_[rx, ry], k=2)
            isolated = dist[:, 1] > 1.5 * r
            rx, ry = rx[isolated], ry[isolated]
    else:
        rx, ry = sx, sy

    rratio, _, rsn1, rsn2, rnflagged = _measure_flux_ratios(
        image, template, good, rx, ry, r, flagged=flagged
    )
    sratio, serr, ssn1, ssn2, _ = _measure_flux_ratios(image, template, good, sx, sy, r)

    rgood = np.isfinite(rratio) & (rsn1 > _STAMP_CHECK_MIN_SN) & (rsn2 > _STAMP_CHECK_MIN_SN)
    rgood &= rnflagged == 0
    sgood = np.isfinite(sratio) & (ssn1 > _STAMP_CHECK_MIN_SN) & (ssn2 > _STAMP_CHECK_MIN_SN)

    positions = []

    for ri, (x1, x2, y1, y2, ksum) in enumerate(regions):
        rin = rgood & (rx >= x1) & (rx <= x2) & (ry >= y1) & (ry <= y2)

        if np.sum(rin) < _STAMP_CHECK_MIN_STARS:
            log(
                'Stamp check: only %d stars with S/N > %g in region %d, skipping'
                % (np.sum(rin), _STAMP_CHECK_MIN_SN, ri)
            )
            continue

        med = np.median(rratio[rin])
        scatter = mad_std(rratio[rin])
        med_err = 1.253 * scatter / np.sqrt(np.sum(rin))

        if not med > 0:
            continue

        dev = ksum - med

        if np.abs(dev) <= max(_STAMP_CHECK_REL_TOL * med, sigma * med_err):
            log(
                'Stamp check: kernel sum %.4g in region %d agrees with the flux ratio %.4g +/- %.2g of %d stars'
                % (ksum, ri, med, med_err, np.sum(rin))
            )
            continue

        log(
            'Stamp check: kernel sum %.4g in region %d deviates from the flux ratio %.4g +/- %.2g of %d stars'
            % (ksum, ri, med, med_err, np.sum(rin))
        )

        sin = sgood & (sx >= x1) & (sx <= x2) & (sy >= y1) & (sy <= y2)
        tol = sigma * np.maximum(serr, max(scatter, _STAMP_CHECK_REL_TOL * med))
        bad = sin & (np.sign(sratio - med) == np.sign(dev)) & (np.abs(sratio - med) > tol)

        if not np.any(bad):
            log('Stamp check: no stamps deviating in the same direction found')
            continue

        # Only the most significant one is masked, the others are re-checked after the
        # re-run, as they may well just follow the biased kernel
        signif = np.where(bad, np.abs(sratio - med) / serr, -np.inf)

        for i in [np.argmax(signif)]:
            # Template peak of the stamp - the culprit is the brightest object there
            x0, y0 = int(np.round(sx[i])), int(np.round(sy[i]))
            hw = int(np.ceil(r))
            ys_ = slice(max(0, y0 - hw), min(template.shape[0], y0 + hw + 1))
            xs_ = slice(max(0, x0 - hw), min(template.shape[1], x0 + hw + 1))
            yy, xx = np.mgrid[ys_, xs_]
            cut = np.where(np.hypot(xx - sx[i], yy - sy[i]) <= r, template[ys_, xs_], -np.inf)
            # Centroid of the pixels above half of the peak, as saturated core may be flat
            inside = np.isfinite(cut)
            bg = np.median(cut[inside])
            core = inside & (cut - bg >= 0.5 * (np.max(cut[inside]) - bg))
            px, py = np.mean(xx[core]), np.mean(yy[core])

            log(
                'Stamp check: stamp at %.0f %.0f has flux ratio %.4g, template peak at %.1f %.1f'
                % (sx[i], sy[i], sratio[i], px, py)
            )
            if (px, py) not in positions:
                positions.append((px, py))

    return positions


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
    check_stamps=False,
    check_stamps_sigma=3.0,
    check_stamps_iter=3,
    check_stamps_radius=None,
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

    Parameters
    ----------
    image : numpy.ndarray
        Input science image as a NumPy array.
    template : numpy.ndarray
        Input template image, should have the same shape as the science image.
    mask : numpy.ndarray, optional
        Science image mask as a boolean array (True values will be masked).
    template_mask : numpy.ndarray, optional
        Template image mask as a boolean array (True values will be masked).
    err : numpy.ndarray or bool, optional
        Science image error map (expected RMS of every pixel). If set to True,
        the code will try to build the noise map directly from the image and
        ``image_gain`` parameter.
    template_err : numpy.ndarray or bool, optional
        Template image error map. If set to True, the code will try to build the
        noise map directly from the template and ``template_gain`` parameter.
    extra : dict, optional
        Extra parameters to be passed to HOTPANTS executable, with parameter names as keys.
    image_fwhm : float, optional
        FWHM of science image in pixels.
    template_fwhm : float, optional
        FWHM of template image in pixels.
    image_gain : float, optional
        Gain of science image.
    template_gain : float, optional
        Gain of template image.
    rel_r : float, optional
        If specified, HOTPANTS ``r`` parameters will be set to :code:`image_fwhm*rel_r`.
    rel_rss : float, optional
        If specified, HOTPANTS ``rss`` parameters will be set to :code:`image_fwhm*rel_rss`.
    nx : int, optional
        Number of image sub-regions in ``x`` direction.
    ny : int, optional
        Number of image sub-regions in ``y`` direction.
    obj : astropy.table.Table, optional
        List of objects detected on science image. If provided, it will be used to
        better place the sub-stamps used to derive the convolution kernel.
    check_stamps : bool, optional
        Whether to check, after the run, that the kernel sum of every region agrees with
        the robust median of image to template flux ratios of the stars (from `obj` if
        given, else from the stamps), measured in apertures with radius of 3 image FWHMs
        (or stamp half-width if FWHM is unknown). If it does not, the stamps deviating
        from the median in the same direction - typically bright stars saturated or
        non-linear in the template, that are not rejected by HOTPANTS as their noise is
        large - are found, the most significant one per region is masked in the template,
        and HOTPANTS is run again, until the kernel sum agrees or `check_stamps_iter`
        re-runs are done. The newly masked regions are then set to the fill value
        (``1e-30`` unless overridden by ``fi`` in `extra`) in the difference and
        noise-scaled difference images, the same as other pixels HOTPANTS considers
        invalid. The number of masked positions is stored in ``NSTMASK`` keyword of
        the output header. The check works best with `obj` given, as only isolated
        stars from it are used as the reference. Without it, the stamps are used
        instead, and these are often too faint - if fewer than 5 of them have S/N > 10
        in both images, the check is skipped.
    check_stamps_sigma : float, optional
        Significance threshold for both the kernel sum deviation and the stamp outliers.
    check_stamps_iter : int, optional
        Maximal number of HOTPANTS re-runs with the stamp check.
    check_stamps_radius : float, optional
        Radius of the template mask around the culprit stamps, in pixels. Default is
        3 template FWHMs, or 3 image FWHMs if the former is unknown.
    get_convolved : bool, optional
        Whether to also return the convolved template image.
    get_scaled : bool, optional
        Whether to also return noise-normalized difference image.
    get_noise : bool, optional
        Whether to also return difference image noise model.
    get_kernel : bool, optional
        Whether to also return convolution kernel used in the subtraction.
    get_header : bool, optional
        Whether to also return the FITS header from HOTPANTS output file.
    _workdir : str, optional
        If specified, all temporary files will be created in this directory, and will
        be kept intact after running HOTPANTS. May be used for debugging.
    _tmpdir : str, optional
        If specified, all temporary files will be created in a dedicated directory
        (that will be deleted after running the executable) inside this path.
    _exe : str, optional
        Full path to HOTPANTS executable. If not provided, the code tries to locate
        it automatically in your :envvar:`PATH`.
    verbose : bool or callable, optional
        Whether to show verbose messages during the run of the function or not.

    Returns
    -------
    numpy.ndarray or list
        The difference image and, optionally, other kinds of images as requested
        by the ``get_*`` options above.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='hotpants', dir=_tmpdir)

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
            err, _ = _build_err_map(image, mask, gain=image_gain, verbose_log=log)

        if hasattr(err, 'shape'):
            errname = os.path.join(workdir, 'err.fits')
            params['ini'] = errname
            fits.writeto(errname, err, overwrite=True)

    if template_err is not None:
        if type(template_err) is bool and template_err == True:
            template_err, _ = _build_err_map(
                template, template_mask, gain=template_gain, verbose_log=log
            )

        if hasattr(template_err, 'shape'):
            terrname = os.path.join(workdir, 'terr.fits')
            params['tni'] = terrname
            fits.writeto(terrname, template_err, overwrite=True)

    if image_fwhm is not None and template_fwhm is not None:
        # Recommended logic for sigma_match
        if image_fwhm > template_fwhm:
            sigma_match = np.sqrt(image_fwhm**2 - template_fwhm**2) / 2.35
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
            sigma_match = np.sqrt(template_fwhm**2 - image_fwhm**2) / 2.35
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

    if obj is not None and len(obj):
        obj = Table(obj)
        log('Using %d external positions for substamps' % len(obj))

    # Template mask, extended by the stamp check
    template_mask_run = template_mask.copy()
    n_stamp_masked = 0

    if check_stamps:
        # Pixels valid in both images, and the ones outside of the ranges valid for the
        # kernel fit, that are measured but excluded from the reference stars
        good = ~mask & ~template_mask
        flagged = (image < params['il']) | (image > params.get('iuk', params['iu']))
        flagged |= (template < params['tl']) | (template > params.get('tuk', params['tu']))

        check_aper = 3 * image_fwhm if image_fwhm else None
        if check_stamps_radius is None:
            fwhm0 = template_fwhm or image_fwhm
            check_stamps_radius = 3 * fwhm0 if fwhm0 else params.get('rss', 15) / 2

    for iteration in range(1 + (max(0, check_stamps_iter) if check_stamps else 0)):
        fits.writeto(tmaskname, template_mask_run.astype(np.uint16), overwrite=True)

        # Set the stamp locations from detected objects, except for the masked ones
        if obj is not None and len(obj):
            xyname = os.path.join(workdir, 'objects.xy')
            ox = np.round(np.asarray(obj['x'])).astype(int)
            oy = np.round(np.asarray(obj['y'])).astype(int)
            inside = (ox >= 0) & (oy >= 0) & (ox < image.shape[1]) & (oy < image.shape[0])
            idx = inside.copy()
            idx[inside] = ~template_mask_run[oy[inside], ox[inside]] | template_mask[oy[inside], ox[inside]]
            np.savetxt(
                xyname, [[_['x'] + 1, _['y'] + 1] for _ in obj[idx]], fmt='%.1f'
            )
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

        if not check_stamps:
            break

        positions = _check_hotpants_stamps(
            image,
            template,
            good,
            stampname,
            fits.getheader(outname),
            flagged=flagged,
            obj=obj,
            aper=check_aper,
            sigma=check_stamps_sigma,
            log=log,
        )

        if not positions:
            break

        if iteration == check_stamps_iter:
            log('Stamp check: maximal number of re-runs reached, keeping the last result')
            break

        yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        for px, py in positions:
            template_mask_run |= (xx - px) ** 2 + (yy - py) ** 2 <= check_stamps_radius**2
        n_stamp_masked += len(positions)

        log(
            'Stamp check: masked %d positions in the template with radius %.1f pixels, re-running HOTPANTS'
            % (len(positions), check_stamps_radius)
        )

    # Regions masked by the stamp check, invalid in the difference images
    stamp_masked = template_mask_run & ~template_mask if n_stamp_masked else None
    fill = params.get('fi', _nan)

    # Difference image
    diff = fits.getdata(outname, 0)
    if stamp_masked is not None:
        diff[stamp_masked] = fill
    result = [diff]

    header = fits.getheader(outname)

    if check_stamps:
        header['NSTMASK'] = n_stamp_masked

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
        if stamp_masked is not None:
            sdiff[stamp_masked] = fill
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
    **kwargs,
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

    Parameters
    ----------
    image : numpy.ndarray
        Input science image as a NumPy array.
    template : numpy.ndarray
        Input template image, should have the same shape as the science image.
    mask : numpy.ndarray, optional
        Science image mask as a boolean array (True values will be masked).
    template_mask : numpy.ndarray, optional
        Template image mask as a boolean array (True values will be masked).
    err : numpy.ndarray, optional
        Science image noise map (expected RMS of every pixel). If not set, the code
        will try to build the noise map directly from the image and ``image_gain`` parameter.
    template_err : numpy.ndarray, optional
        Template image noise map. If not set, the code will try to build the noise map
        directly from the template and ``template_gain`` parameter.
    image_gain : float, optional
        Gain of science image, to be used for construction of the noise map.
    template_gain : float, optional
        Gain of template image, to be used for construction of the noise map.
    image_psf : dict, optional
        PSF for science image. If not provided, will be estimated by
        :func:`stdpipe.psf.run_psfex`.
    template_psf : dict, optional
        PSF for template image. If not provided, will be estimated by
        :func:`stdpipe.psf.run_psfex`.
    image_fwhm : float, optional
        FWHM of science image, to be used for estimated PSF normalization.
    template_fwhm : float, optional
        FWHM of template image, to be used for estimated PSF normalization.
    scale : float, optional
        Template image flux scale relative to science image.
    psf_clean : float, optional
        If non-zero, will clean the PSF regions with relative values less than this factor.
    dx : float, optional
        Astrometric uncertainty (sigma) in x coordinate.
    dy : float, optional
        Astrometric uncertainty (sigma) in y coordinate.
    nx : int, optional
        Number of sub-images in ``x`` direction.
    ny : int, optional
        Number of sub-images in ``y`` direction.
    overlap : int, optional
        If set, defines how much sub-images will overlap, in pixels.
    fit_scale : bool, optional
        If set, will fit for the difference in flux scales between template and science images.
    fit_shift : bool, optional
        If set, will also fit for the sub-pixel shift between template and science images.
    good_regions : numpy.ndarray, optional
        If set, this boolean map will be used to restrict scale/shift fitting to only these regions.
    image_obj : astropy.table.Table, optional
        List of objects detected in science image. If provided, they will be used for placing good regions.
    template_obj : astropy.table.Table, optional
        List of objects detected in template image. If provided, they will be used for deriving
        template flux scale.
    get_psf : bool, optional
        If set, will also return the PSF of the difference image.
    get_Fpsf : bool, optional
        If set, will also return the optimal PSF photometry image and its error.
    nthreads : int, optional
        Set the number of threads to use for FFTW routines (0 for auto).
    verbose : bool or callable, optional
        Whether to show verbose messages during the run of the function or not.
    **kwargs
        Extra parameters to be passed to :func:`stdpipe.psf.run_psfex` for PSF estimation.

    Returns
    -------
    list
        The list of the difference image and other subtraction results:

        - D: Subtracted image
        - S_corr: Corrected subtracted image
        - P_D: PSF of subtracted image, if :code:`get_psf=True`
        - Fpsf: optimal PSF photometry image, if :code:`get_Fpsf=True`
        - Fpsf_err: optimal PSF photometry error image, if :code:`get_Fpsf=True`

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if mask is None:
        mask = ~np.isfinite(image)
    else:
        mask = mask | ~np.isfinite(image)

    if template_mask is None:
        template_mask = ~np.isfinite(template)
    else:
        template_mask = template_mask | ~np.isfinite(template)

    # Build noise models and subtract background
    if err is None:
        U_N_full, image_bg = _build_err_map(image, mask, gain=image_gain, verbose_log=log)
        SN = image_bg.globalrms
    else:
        image_bg = sep.Background(image, mask)
        U_N_full = err
        SN = np.median(err)

    if template_err is None:
        U_R_full, template_bg = _build_err_map(
            template.astype(np.double), template_mask, gain=template_gain, verbose_log=log
        )
        SR = template_bg.globalrms
    else:
        template_bg = sep.Background(template.astype(np.double), template_mask)
        U_R_full = template_err
        SR = np.median(template_err)

    N_full = image - image_bg.back()
    R_full = template - template_bg.back()
    # Set to zero the regions where we have no data
    N_full[~np.isfinite(image)] = 0
    R_full[~np.isfinite(template)] = 0

    # Artificially assign large uncertainty to the regions we set to zero
    U_N_full[~np.isfinite(image)] = np.nanmax(U_N_full)
    U_R_full[~np.isfinite(template)] = np.nanmax(U_R_full)

    # Estimate image PSF, if not specified
    if image_psf is None:
        log("Estimating image PSF")
        # TODO: should we overwrite image_obj if it was set by the user?..
        image_psf, image_obj = psf.run_psfex(
            image,
            mask=mask,
            gain=image_gain,
            aper=2.0 * image_fwhm if image_fwhm else None,
            verbose=verbose,
            get_obj=True,
            **kwargs,
        )

    # Estimate template PSF, if not specified
    if template_psf is None:
        log("Estimating template PSF")
        # TODO: should we overwrite image_obj if it was set by the user?..
        template_psf, template_obj = psf.run_psfex(
            template,
            mask=template_mask,
            gain=template_gain,
            aper=2.0 * template_fwhm if template_fwhm else None,
            verbose=verbose,
            get_obj=True,
            **kwargs,
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
        sr = 0.5 * max(image_psf['fwhm'], template_psf['fwhm'])
        iidx, tidx, _ = astrometry.planar_match(
            image_obj['x'], image_obj['y'], template_obj['x'], template_obj['y'], sr
        )

        if len(iidx):
            # Exclude all flagged objects
            fidx = (image_obj['flags'][iidx] & 0x100) == 0
            fidx &= (template_obj['flags'][tidx] & 0x100) == 0

            # Photometric zero point shift between two object lists
            ZP = np.ma.filled(
                np.ma.asarray(template_obj['mag'][tidx] - image_obj['mag'][iidx], dtype=float),
                np.nan,
            )
            dZP = np.ma.filled(
                np.ma.asarray(
                    np.hypot(image_obj['magerr'][iidx], template_obj['magerr'][tidx]), dtype=float
                ),
                np.nan,
            )
            # Zero or non-finite errors would give infinite weights and a NaN scale
            fidx &= np.isfinite(ZP) & np.isfinite(dZP) & (dZP > 0)

            if np.sum(fidx) > 5:
                ZP, dZP = ZP[fidx], dZP[fidx]
                X = np.ones_like(ZP).T
                # Weighted robust regression
                C = sm.RLM(ZP / dZP, (X.T / dZP).T).fit()
                scale = 10 ** (-0.4 * C.params[0])
                log("Estimated initial template flux scale Fr = %.3g" % scale)
            else:
                log("Not enough matched objects to guess template flux scale")

        if len(iidx) > 0 and (fit_scale or fit_shift) and np.sum(good_regions) == 0:
            # Use matched objects (they are all unflagged) to define good regions
            size = int(6 * sr)  # 3*FWHM
            N = 0
            for i in range(len(iidx)):
                x0 = int(image_obj['x'][iidx][i])
                y0 = int(image_obj['y'][iidx][i])

                # Exclude objects too close to the edges
                if (
                    x0 < size
                    or x0 >= image.shape[1] - size
                    or y0 < size
                    or y0 >= image.shape[0] - size
                ):
                    continue

                # Exclude sub-regions containing masked pixels
                if (
                    np.sum((mask | template_mask)[y0 - size : y0 + size, x0 - size : x0 + size])
                    == 0
                ):
                    good_regions[y0 - size : y0 + size, x0 - size : x0 + size] = True
                    N += 1

            if N > 0:
                log(N, "regions selected for scale/shift fitting")

    else:
        log("Initial template flux scale Fr = %.3g" % scale)

    # A NaN or infinite estimate (e.g. from NaN or zero magnitude errors) makes
    # least_squares fail on the initial point, and spoils the difference image
    if scale is None or not np.isfinite(scale) or scale <= 0:
        if scale is not None:
            log("Initial template flux scale Fr = %.3g is unusable, starting from 1" % scale)
        scale = 1.0

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
        N_full,
        R_full,
        U_N_full,
        U_R_full,
        good_regions,
        nx=nx,
        ny=ny,
        overlap=overlap,
        get_index=True,
        get_origin=True,
        verbose=True if nx > 1 and ny > 1 else False,
    ):
        # PSF stamps at the centers of sub-image
        P_N_small = psf.get_psf_stamp(
            image_psf, x=x0 + N.shape[1] / 2, y=y0 + N.shape[0] / 2, dx=0, dy=0, normalize=True
        )
        P_R_small = psf.get_psf_stamp(
            template_psf, x=x0 + N.shape[1] / 2, y=y0 + N.shape[0] / 2, dx=0, dy=0, normalize=True
        )

        if psf_clean:
            # Set to zero the wings with relative amplitude less than psf_clean factor
            P_N_small[P_N_small < psf_clean * np.max(P_N_small)] = 0
            P_N_small /= np.sum(P_N_small)

            P_R_small[P_R_small < psf_clean * np.max(P_R_small)] = 0
            P_R_small /= np.sum(P_R_small)

        Fn = 1  # Fixed, so that the difference will be in science image flux scale
        Fr = scale if scale else 1  # Will be optionally adjusted later

        # Place PSF at the center of image with same size as new / reference
        P_N = np.zeros_like(N)
        P_R = np.zeros_like(R)
        idxN = tuple(
            [
                slice(
                    int(N.shape[0] / 2) - int(P_N_small.shape[0] / 2),
                    int(N.shape[0] / 2) + int(P_N_small.shape[0] / 2) + 1,
                ),
                slice(
                    int(N.shape[1] / 2) - int(P_N_small.shape[1] / 2),
                    int(N.shape[1] / 2) + int(P_N_small.shape[1] / 2) + 1,
                ),
            ]
        )
        idxR = tuple(
            [
                slice(
                    int(R.shape[0] / 2) - int(P_R_small.shape[0] / 2),
                    int(R.shape[0] / 2) + int(P_R_small.shape[0] / 2) + 1,
                ),
                slice(
                    int(R.shape[1] / 2) - int(P_R_small.shape[1] / 2),
                    int(R.shape[1] / 2) + int(P_R_small.shape[1] / 2) + 1,
                ),
            ]
        )
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

                D_hat_den = np.sqrt(
                    SN**2 * Fr**2 * np.abs(P_R_hat**2) + SR**2 * Fn**2 * np.abs(P_N_hat**2)
                )
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
                    C = least_squares(
                        fn, [scale, 0, 0], bounds=((0, -0.5, -0.5), (np.inf, 0.5, 0.5)), verbose=0
                    )
                elif fit_scale:
                    log('Fitting for flux scale difference')
                    C = least_squares(fn, [scale], bounds=(0, np.inf), verbose=0)
                else:
                    log('Fitting for sub-pixel template shift')
                    C = least_squares(
                        fn,
                        [scale, 0, 0],
                        bounds=((scale * 0.9999, -0.5, -0.5), (scale * 1.0001, 0.5, 0.5)),
                        verbose=0,
                    )

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
        D_hat_num = Fr * P_R_hat * N_hat - Fn * P_N_hat * R_hat
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
        F_S /= S.shape[1] * S.shape[1]  # divide by the number of pixels due to FFT normalization
        Fpsf = S / F_S  # optimal PSF photometry, alpha in Equation 41
        Fpsf_err = np.sqrt(V_S_N + V_S_R) / F_S

        # Stitch the piece back to the tapestry
        D_full[y0 : y0 + N.shape[0], x0 : x0 + N.shape[1]] = D
        S_corr_full[y0 : y0 + N.shape[0], x0 : x0 + N.shape[1]] = S_corr
        Fpsf_full[y0 : y0 + N.shape[0], x0 : x0 + N.shape[1]] = Fpsf
        Fpsf_err_full[y0 : y0 + N.shape[0], x0 : x0 + N.shape[1]] = Fpsf_err

    result = [D_full, S_corr_full]

    if get_psf:
        result += [P_D]

    if get_Fpsf:
        result += [Fpsf_full, Fpsf_err_full]

    return result


def _build_err_map(image, mask, gain=None, verbose_log=None):
    """Build a noise map from image and gain, following the HOTPANTS/ZOGY convention.

    Combines background RMS from SEP with source Poisson noise contribution.

    Returns
    -------
    err : (ny, nx) noise map (per-pixel standard deviation)
    bg : sep.Background object
    """
    log = verbose_log or (lambda *a, **k: None)

    bg = sep.Background(image.astype(np.double), mask)
    err = bg.rms()  # Background noise level

    # Source Poisson noise contribution
    source = np.abs(image - bg.back())
    if gain is not None and gain > 0:
        source /= gain
    else:
        source[:] = 0
    err = np.sqrt(err**2 + source)

    log('Noise model: background %.1f, rms %.2f ADU' % (bg.globalback, bg.globalrms))

    return err, bg


def run_sfft(
    image,
    template,
    mask=None,
    template_mask=None,
    err=None,
    template_err=None,
    image_gain=None,
    template_gain=None,
    kernel_shape=(7, 7),
    kernel_poly_order=2,
    bg_poly_order=2,
    flux_poly_order=1,
    flux_penalty=1e3,
    ridge=1e-6,
    sigma_clip=3.0,
    max_iter=5,
    obj=None,
    obj_size=None,
    obj_grid=4,
    obj_per_cell=10,
    obj_max_masked_fraction=0.5,
    get_convolved=False,
    get_scaled=False,
    get_noise=False,
    get_kernel=False,
    verbose=False,
):
    """Image subtraction using SFFT with spatially varying kernel.

    Wrapper around :func:`stdpipe.sfft.solve` that handles noise model
    construction, difference image noise propagation, and follows the same
    interface conventions as :func:`run_hotpants` and :func:`run_zogy`.

    The noise model for both science and template images may be provided
    directly (as arrays), or built automatically from the images using
    background estimation and gain values (same approach as HOTPANTS).
    If ``err`` or ``template_err`` is set to ``True``, the routine builds
    a noise map from the image using SEP background estimation and the
    corresponding gain parameter.

    Parameters
    ----------
    image : numpy.ndarray
        Input science image as a NumPy array.
    template : numpy.ndarray
        Input template image, same shape as science image.
    mask : numpy.ndarray, optional
        Science image mask (True = bad).
    template_mask : numpy.ndarray, optional
        Template image mask (True = bad).
    err : numpy.ndarray or bool, optional
        Science image noise map (per-pixel RMS). If True, built from the image
        and ``image_gain``.
    template_err : numpy.ndarray or bool, optional
        Template image noise map. If True, built from the template and
        ``template_gain``. Used both for the difference image noise map and
        for fit weighting: the solver performs one reweighting pass with the
        template variance propagated through the fitted kernel (see
        :func:`stdpipe.sfft.solve`).
    image_gain : float, optional
        Gain of science image in e-/ADU.
    template_gain : float, optional
        Gain of template image in e-/ADU.
    kernel_shape : tuple of int, optional
        (ky, kx) kernel size, must be odd. Default (7, 7).
    kernel_poly_order : int, optional
        Polynomial order for kernel spatial variation. Default 2.
    bg_poly_order : int, optional
        Polynomial order for differential background. Default 2.
    flux_poly_order : int, optional
        Polynomial order for kernel-sum (flux scale) constraint. Default 1.
    flux_penalty : float, optional
        Penalty weight for kernel-sum constraint. Default 1e3.
    ridge : float, optional
        Tikhonov regularization. Default 1e-6.
    sigma_clip : float, optional
        Sigma threshold for outlier rejection. Default 3.0.
    max_iter : int, optional
        Maximum sigma-clipping iterations. Default 5.
    obj : astropy.table.Table, optional
        List of detected objects with ``x``, ``y`` columns (and optionally
        ``flux``, ``fluxerr``, ``flags``, ``fwhm``). If provided, kernel
        fitting is restricted to square regions around selected objects,
        analogous to HOTPANTS stamp placement. This concentrates the fit on
        high-S/N regions and can improve kernel quality. Objects with
        nonzero ``flags`` are skipped, and the selection is spatially
        stratified (see ``obj_grid`` / ``obj_per_cell``) so the kernel and
        background spatial polynomials stay constrained across the field.
        If the selected regions provide too few pixels for the model
        parameters, the fit falls back to the whole image.
    obj_size : int, optional
        Half-size in pixels of the fitting region around each object. If None
        (default), derived from the median ``fwhm`` column of *obj*
        (``3 × median(fwhm)``), or 15 pixels if ``fwhm`` is not available.
    obj_grid : int, optional
        Number of grid cells per axis for spatially stratified object
        selection. Default 4.
    obj_per_cell : int, optional
        Maximum number of objects used per grid cell, highest S/N first
        (``flux``/``fluxerr``, or ``flux``, or input order). Default 10.
        Set to None to disable the per-cell cap.
    obj_max_masked_fraction : float, optional
        Maximum fraction of masked pixels allowed in an object region;
        regions above it are skipped. Default 0.5. Isolated masked pixels
        within accepted regions are harmless — the fit excludes them
        individually via the weight map.
    get_convolved : bool, optional
        Whether to also return the convolved template.
    get_scaled : bool, optional
        Whether to also return noise-normalized difference. Pixels where the
        difference is unreliable (image edges and kernel-footprint
        neighbourhoods of template defects, see
        :attr:`stdpipe.sfft.SFFTResult.dmask`) are set to zero.
    get_noise : bool, optional
        Whether to also return the difference image noise map.
    get_kernel : bool, optional
        Whether to also return the :class:`~stdpipe.sfft.SFFTResult` object
        containing kernel coefficients, fit metadata, and the ``dmask``
        attribute flagging unreliable difference pixels.
    verbose : bool or callable, optional
        Whether to show verbose messages.

    Returns
    -------
    numpy.ndarray or list
        The difference image, or a list of requested planes in the same
        order as :func:`run_hotpants`:
        ``[diff, convolved, scaled, noise, kernel]``, with unrequested
        planes omitted.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # --- Masks ---
    if mask is None:
        mask = ~np.isfinite(image)
    else:
        mask = np.asarray(mask, dtype=bool) | ~np.isfinite(image)

    if template_mask is None:
        template_mask = ~np.isfinite(template)
    else:
        template_mask = np.asarray(template_mask, dtype=bool) | ~np.isfinite(template)

    combined_mask = mask | template_mask

    # --- Build noise models ---
    image_f = np.asarray(image, dtype=np.float64)
    template_f = np.asarray(template, dtype=np.float64)

    if err is True:
        log('Building noise model from the image')
        # SEP cannot handle NaN — replace with median for background estimation
        image_clean = image_f.copy()
        image_clean[~np.isfinite(image_clean)] = np.nanmedian(image_clean)
        err, _ = _build_err_map(image_clean, mask, gain=image_gain, verbose_log=log)
    elif err is not None:
        err = np.asarray(err, dtype=np.float64)

    if template_err is True:
        log('Building noise model from the template')
        template_clean = template_f.copy()
        template_clean[~np.isfinite(template_clean)] = np.nanmedian(template_clean)
        template_err, _ = _build_err_map(
            template_clean,
            template_mask,
            gain=template_gain,
            verbose_log=log,
        )
    elif template_err is not None:
        template_err = np.asarray(template_err, dtype=np.float64)

    # SFFT uses inverse-variance weighting internally when err is provided;
    # template_err additionally triggers a reweighting pass with template
    # noise propagated through the fitted kernel
    sfft_err = err  # may be None for uniform weighting

    # --- Build fitting mask from object positions ---
    fitting_mask = combined_mask
    if obj is not None and len(obj):
        obj = Table(obj)
        ny_img, nx_img = image.shape[:2]

        # Determine region half-size
        if obj_size is not None:
            size = int(obj_size)
        elif 'fwhm' in obj.colnames:
            size = int(np.ceil(3 * np.nanmedian(obj['fwhm'])))
        else:
            size = 15

        # Reject objects with detection quality flags (saturation, blending,
        # truncation, ...) — they should not constrain the kernel
        if 'flags' in obj.colnames:
            n_before = len(obj)
            obj = obj[np.asarray(obj['flags']) == 0]
            if len(obj) < n_before:
                log('Rejected %d flagged objects out of %d' % (n_before - len(obj), n_before))

        # Order by S/N, brightest first, for the per-cell selection
        if 'flux' in obj.colnames:
            flux = np.asarray(obj['flux'], dtype=np.float64)
            if 'fluxerr' in obj.colnames:
                with np.errstate(divide='ignore', invalid='ignore'):
                    sn = flux / np.asarray(obj['fluxerr'], dtype=np.float64)
            else:
                sn = flux
            order = np.argsort(-np.nan_to_num(sn, nan=-np.inf))
        else:
            order = np.arange(len(obj))

        # Spatially stratified selection: up to obj_per_cell highest-S/N
        # objects per grid cell, so that the kernel and background spatial
        # polynomials are constrained across the whole field
        grid = max(1, int(obj_grid)) if obj_grid else 1
        cell_counts = np.zeros((grid, grid), dtype=int)

        good_regions = np.zeros(image.shape[:2], dtype=bool)
        n_used = 0

        for i in order:
            row = obj[int(i)]
            x0 = int(round(row['x']))
            y0 = int(round(row['y']))

            # Skip objects too close to edges
            if x0 < size or x0 >= nx_img - size or y0 < size or y0 >= ny_img - size:
                continue

            cell = (min(y0 * grid // ny_img, grid - 1), min(x0 * grid // nx_img, grid - 1))
            if obj_per_cell and cell_counts[cell] >= obj_per_cell:
                continue

            # Skip regions dominated by masked pixels; isolated masked pixels
            # are harmless as the fit excludes them individually via weights
            region_mask = combined_mask[y0 - size : y0 + size + 1, x0 - size : x0 + size + 1]
            if np.mean(region_mask) > obj_max_masked_fraction:
                continue

            good_regions[y0 - size : y0 + size + 1, x0 - size : x0 + size + 1] = True
            cell_counts[cell] += 1
            n_used += 1

        # Restrict the fit only if the regions provide enough pixels for the
        # number of model parameters; otherwise fall back to the whole image
        n_params = kernel_shape[0] * kernel_shape[1] * sfft_module._n_poly(
            kernel_poly_order
        ) + sfft_module._n_poly(bg_poly_order)
        n_region_pixels = int(np.sum(good_regions & ~combined_mask))

        if n_used > 0 and n_region_pixels >= 10 * n_params:
            log(
                'Using %d object regions (%d px half-size, %d good pixels) '
                'for kernel fitting' % (n_used, size, n_region_pixels)
            )
            fitting_mask = combined_mask | ~good_regions
        elif n_used > 0:
            log(
                'Object regions provide only %d good pixels for %d parameters, '
                'fitting on whole image' % (n_region_pixels, n_params)
            )
        else:
            log('No usable object regions, fitting on whole image')

    # --- Run SFFT ---
    log('Running SFFT subtraction')

    sfft_result = sfft_module.solve(
        image_f,
        template_f,
        mask=fitting_mask,
        template_mask=template_mask,
        err=sfft_err,
        template_err=template_err,
        kernel_shape=kernel_shape,
        kernel_poly_order=kernel_poly_order,
        bg_poly_order=bg_poly_order,
        flux_poly_order=flux_poly_order,
        flux_penalty=flux_penalty,
        ridge=ridge,
        sigma_clip=sigma_clip,
        max_iter=max_iter,
        verbose=verbose,
    )

    diff = sfft_result.diff

    result = [diff]

    # --- Convolved template ---
    if get_convolved:
        # model = convolved_template + background
        # Extract background contribution to get convolved template alone
        bg_model = sfft_module.evaluate_background(
            sfft_result,
            *np.meshgrid(np.arange(diff.shape[1]), np.arange(diff.shape[0])),
            diff.shape,
        )
        convolved = sfft_result.model - bg_model
        result.append(convolved)

    # --- Noise map for difference image ---
    noise = None

    if get_noise or get_scaled:
        noise = _compute_diff_noise(sfft_result, err, template_err, combined_mask, log)

    # --- Noise-scaled difference ---
    if get_scaled:
        scaled = np.zeros_like(diff)
        # dmask covers image edges and kernel-footprint neighbourhoods of
        # template defects, where the difference is unreliable
        good = (noise > 0) & np.isfinite(noise) & ~combined_mask & ~sfft_result.dmask
        scaled[good] = diff[good] / noise[good]
        result.append(scaled)

    if get_noise:
        result.append(noise)

    # --- Kernel info ---
    if get_kernel:
        result.append(sfft_result)

    if len(result) == 1:
        return result[0]
    else:
        return result


def _compute_diff_noise(sfft_result, err, template_err, mask, log):
    """Compute per-pixel noise map for the SFFT difference image.

    diff = science - model, where model = Σ_α a_α(x,y) · R_α(x,y) + bg(x,y)

    var_diff(x,y) = var_sci(x,y) + Σ_α a_α²(x,y) · var_tmpl(x+dy, x+dx)

    If template_err is not available, assumes the template contributes no
    noise (deep template limit) and returns just the science noise.
    """

    ny, nx = sfft_result.diff.shape

    # Science noise contribution
    if err is not None and hasattr(err, 'shape'):
        var_sci = err**2
    else:
        # Estimate from difference image residuals
        rms = sfft_result.rms if np.isfinite(sfft_result.rms) else mad_std(sfft_result.diff[~mask])
        var_sci = np.full((ny, nx), rms**2, dtype=np.float64)
        log('No science err provided, using constant RMS = %.2f for noise map' % rms)

    # Template noise contribution (convolved through the kernel)
    if template_err is not None and hasattr(template_err, 'shape'):
        log('Computing convolved template noise contribution')

        var_tmpl = template_err**2

        # var_convolved = Σ_α a_α²(x,y) · var_tmpl_shifted(x,y)
        var_conv = sfft_module._propagate_template_variance(
            sfft_result.kernel_coeffs,
            sfft_result.kernel_shape,
            sfft_result.kernel_poly_order,
            var_tmpl,
        )
        var_total = var_sci + var_conv
    else:
        var_total = var_sci

    noise = np.sqrt(np.clip(var_total, 0, None))

    return noise
