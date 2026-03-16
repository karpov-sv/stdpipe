"""
Routines for object detection and photometry.
"""


import os, shutil, tempfile, shlex
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import Table

import warnings
from astropy.wcs import FITSFixedWarning

# warnings.simplefilter(action='ignore', category=FITSFixedWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

import sep
import photutils
import photutils.background
import photutils.aperture
import photutils.segmentation
import photutils.detection

# Put these to common namespace
from .photometry_model import match, make_sn_model, get_detection_limit_sn, format_color_term
from .photometry_measure import measure_objects

# Conditionally import SEP-based photometry if available
try:
    from .photometry_measure import measure_objects_sep, _HAS_SEP_OPTIMAL
except ImportError:
    _HAS_SEP_OPTIMAL = False

from .photometry_background import (
    get_background,
    get_background_percentile,
    get_background_morphology,
    estimate_background_rms_percentile,
    estimate_background_rms_local,
)

try:
    import cv2

    # Much faster dilation
    dilate = lambda image, mask: cv2.dilate(image.astype(np.uint8), mask).astype(bool)
except:
    from scipy.signal import fftconvolve

    dilate = lambda image, mask: fftconvolve(image, mask, mode='same') > 0.9

from . import utils


def make_kernel(r0=1.0, ext=1.0):
    x, y = np.mgrid[
        np.floor(-ext * r0) : np.ceil(ext * r0 + 1),
        np.floor(-ext * r0) : np.ceil(ext * r0 + 1),
    ]
    r = np.hypot(x, y)
    image = np.exp(-r ** 2 / 2 / r0 ** 2)

    return image


def get_objects_sep(
    image,
    header=None,
    mask=None,
    mask_detect=None,
    err=None,
    thresh=4.0,
    aper=3.0,
    bkgann=None,
    r0=0.5,
    gain=1,
    edge=0,
    minnthresh=2,
    minarea=5,
    relfluxradius=2.0,
    wcs=None,
    bg_size=64,
    fwhm=False,
    optimal=False,
    group_sources=True,
    centroid=False,
    use_mask_large=False,
    subtract_bg=True,
    npix_large=100,
    sn=10.0,
    get_segmentation=False,
    deblend_fwhm=0,
    deblend_method='watershed',
    clip_sigma=3.0,
    clip_iters=5,
    verbose=True,
    **kwargs
):
    """Object detection and aperture/optimal photometry using `SEP <https://github.com/kbarbary/sep>`_ routines, with the signature as similar as possible to :func:`~stdpipe.photometry.get_objects_sextractor` function.

    **Algorithm Overview:**

    1. **Background estimation**: Builds 2D background model using sep.Background() with sigma-clipping
    2. **Object detection**: Runs sep.extract() with optional smoothing kernel and deblending (watershed by default)
    3. **Centroiding** (optional): Refines positions using windowed centroiding (sep.winpos)
    4. **Edge rejection**: Filters objects too close to image edges
    5. **FWHM estimation**: Estimates FWHM per-object (always) or globally (if requested) using flux radius
    6. **Photometry**: Performs aperture photometry (sep.sum_circle) or optimal extraction (sep.sum_circle_optimal with Gaussian PSF)
    7. **Quality cuts**: Applies S/N threshold and sorts by flux

    Detection flags are documented at https://sep.readthedocs.io/en/v1.1.x/reference.html - they are different from SExtractor ones!

    :param image: Input image as a NumPy array
    :param header: Image header, optional
    :param mask: Image mask as a boolean array (True values will be masked), optional. Used for background estimation and photometry. Objects whose detection footprint overlaps this mask are flagged with 0x100 but not excluded from detection.
    :param mask_detect: Optional detection mask as a boolean array (True = exclude from detection). If provided, it is combined (OR) with non-finite pixel mask and used for object detection/deblending. Use this to exclude entire regions (e.g. bad columns, satellite trails) from detection. If None, only non-finite pixels are masked during detection.
    :param err: Image noise map as a NumPy array, optional
    :param thresh: Detection threshold in sigmas above local background
    :param aper: Circular aperture radius in pixels (or in units of FWHM if fwhm parameter is provided/estimated), to be used for flux measurement
    :param bkgann: Background annulus (tuple with inner and outer radii, in pixels or FWHM units if fwhm parameter is provided/estimated) to be used for local background estimation. Inside the annulus, simple arithmetic mean of unmasked pixels is used for computing the background, and thus it is subject to some bias in crowded stellar fields. If not set, global background model is used instead.
    :param r0: Smoothing kernel size (sigma) to be used for improving object detection
    :param gain: Image gain, e/ADU
    :param edge: Reject all detected objects closer to image edge than this parameter
    :param minnthresh: Minumal number of pixels above the threshold to be considered a detection
    :param minarea: Minimal number of pixels in the object to be considered a detection
    :param relfluxradius:
    :param wcs: Astrometric solution to be used for assigning sky coordinates (`ra`/`dec`) to detected objects
    :param bg_size: Background grid size in pixels
    :param fwhm: FWHM handling. If False (default), FWHM is estimated per-object for output but not used for photometry. If True, FWHM is estimated from detected objects and both aper and bkgann are scaled by FWHM (interpreted as being in FWHM units). If numeric, this FWHM value is used directly and aper/bkgann are scaled accordingly.
    :param optimal: If True, use optimal extraction photometry (sep.sum_circle_optimal) instead of aperture photometry. Requires FWHM to be provided or estimated. Only available in SEP 1.4+.
    :param group_sources: If True and optimal=True, use grouped optimal extraction for overlapping sources. Fits nearby sources simultaneously for better accuracy in crowded fields. Default is True (recommended). Only available in SEP 1.4+.
    :param centroid: If True, refine object positions using windowed centroiding (sep.winpos). Default is False as windowed positions can be biased in crowded fields. When SEP 1.4+ is available, uses maxstep=0.2*FWHM to cap position shift per iteration and avoid excessive drift in degenerate cases.
    :param use_mask_large: If True, filter out large objects (with footprints larger than `npix_large` pixels)
    :param npix_large: Threshold for rejecting large objects (if `use_mask_large` is set)
    :param subtract_bg: Whether to subtract the background (default) or not
    :param sn: Minimal S/N ratio for the object to be considered a detection
    :param get_segmentation: If set, segmentation map will also be returned. The output table will include 'seg_id' column with segmentation map IDs (sequential object number + 1).
    :param deblend_fwhm: If > 0, use a fixed circular Gaussian with this FWHM (in pixels) when assigning ambiguous pixels during deblending. This uses a deterministic max-weight assignment and is less sensitive to moment estimates in crowded fields. Default is 0.0 (use adaptive shapes and stochastic assignment). When deblend_method='watershed', this also enforces a minimum peak separation of 0.5*FWHM. Only available in SEP 1.4+.
    :param deblend_method: Deblending algorithm. 'watershed' (default) seeds local maxima and applies watershed assignment within each detection footprint. 'threshold' uses the traditional multi-threshold method. Only available in SEP 1.4+.
    :param clip_sigma: Sigma value for sigma-clipping when ``bkgann`` is provided. Default is 3.0. Only used in SEP 1.4+.
    :param clip_iters: Maximum number of sigma-clipping iterations when ``bkgann`` is provided. Default is 5. Set to 0 to disable clipping. Only used in SEP 1.4+.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: astropy.table.Table object with detected objects, or tuple of (table, segmentation_map) if get_segmentation=True

    **Returned Table Columns:**

    - **x, y**: Object centroid positions (pixels)
    - **xerr, yerr**: Positional uncertainties (pixels)
    - **flux, fluxerr**: Measured flux and error in ADU (aperture or optimal extraction)
    - **mag, magerr**: Instrumental magnitude and error (-2.5*log10(flux))
    - **flags**: Detection quality flags (see SEP documentation for flag meanings)
    - **ra, dec**: Sky coordinates in degrees (if WCS provided, otherwise zeros)
    - **bg**: Local background level per pixel (from aperture or annulus)
    - **fwhm**: Per-object FWHM estimate in pixels (from SEP built-in if available, else 2nd moments)
    - **a, b**: Semi-major and semi-minor axes from 2nd moments (pixels)
    - **theta**: Position angle from 2nd moments (radians, counter-clockwise from +x axis)
    - **seg_id**: Segmentation map object ID (1-based, only if get_segmentation=True)

    **Table Metadata:**

    - **aper**: Aperture radius used for photometry (pixels)
    - **bkgann**: Background annulus used (tuple of inner/outer radii, or None)
    - **optimal**: Whether optimal extraction was used (bool)
    - **group_sources**: Whether grouped optimal extraction was used (bool)
    - **fwhm_phot**: FWHM value used for photometry (if provided or estimated for optimal extraction)

    **Notes:**

    - FWHM estimation prefers SEP's built-in method (ported from SExtractor). Falls back to 2nd moments (FWHM = 2*sqrt(ln(2)*(a**2+b**2))) for older SEP versions - it is exact for Gaussian PSF, approximate for others.
    - Optimal extraction assumes Gaussian PSF. For ground-based data with Moffat PSF, bias varies with crowding.
    - Grouped optimal extraction (enabled by default when optimal=True) fits nearby sources simultaneously for better accuracy in crowded fields. Dramatically improves flux accuracy for close pairs compared to single-source extraction.
    - Watershed deblending (default) provides often better completeness for close pairs (<2 FWHM) compared to threshold method.
    - Object detection is performed on background-subtracted image. Photometry includes local or global background estimation.
    - S/N cut is applied after photometry: only objects with magerr < 1/sn are returned.
    """

    # Simple Wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if r0 > 0.0:
        kernel = make_kernel(r0)
    else:
        kernel = None

    log("Preparing background mask")

    if mask is None:
        mask = np.zeros_like(image, dtype=bool)

    mask_bg = np.zeros_like(mask)
    mask_segm = np.zeros_like(mask)

    # Detection mask: non-finite pixels plus optional user-provided detection
    # mask. The main user mask is not passed to sep.extract() so that bright
    # objects with partially masked cores (e.g. saturated pixels) are detected
    # and deblended as single objects rather than being fragmented into
    # ring-shaped artifacts.  The user mask is still used for background
    # estimation and photometry.  Use mask_detect to exclude entire regions
    # (e.g. bad columns, satellite trails) from detection.
    mask_det = ~np.isfinite(image)
    if mask_detect is not None:
        mask_det = mask_det | mask_detect

    log("Building background map")

    bg = sep.Background(image, mask=mask | mask_bg, bw=bg_size, bh=bg_size)
    if subtract_bg:
        image1 = image - bg.back()
    else:
        image1 = image.copy()

    if err is None:
        err = bg.rms()
        err[~np.isfinite(err)] = 1e30
        err[err == 0] = 1e30

    sep.set_extract_pixstack(image.shape[0] * image.shape[1])

    if use_mask_large:
        # Mask regions around huge objects as they are most probably corrupted by saturation and blooming
        log("Extracting initial objects")

        # Minimal extraction, just to get large objects
        obj0, segm = sep.extract(
            image1,
            err=err,
            thresh=thresh,
            minarea=minarea,
            mask=mask_det | mask_bg,
            filter_kernel=kernel,
            segmentation_map=True,
        )

        log("Dilating large objects")

        mask_segm = np.isin(
            segm, [_ + 1 for _, npix in enumerate(obj0['npix']) if npix > npix_large]
        )
        mask_segm = dilate(mask_segm, np.ones([10, 10]))

    log("Extracting final objects")

    extract_kwargs = {
        'err': err,
        'thresh': thresh,
        'minarea': minarea,
        'mask': mask_det | mask_bg | mask_segm,
        'filter_kernel': kernel,
        'segmentation_map': True,
    }

    # Add deblending parameters if available (SEP 1.4+)
    if _HAS_SEP_OPTIMAL:
        extract_kwargs['deblend_fwhm'] = deblend_fwhm
        extract_kwargs['deblend_method'] = deblend_method

    # Merge with any additional kwargs
    extract_kwargs.update(kwargs)

    obj0, segm = sep.extract(image1, **extract_kwargs)

    # Flag objects whose footprints overlap the user mask.
    # 0x100 = object has masked pixels within its detection footprint.
    if mask.any():
        # For each segmentation label, check if any of its pixels are masked
        # Use fast vectorized approach: find unique labels that overlap mask
        masked_labels = np.unique(segm[mask & (segm > 0)])
        # seg_id for object i is i+1 (1-based)
        obj_seg_ids = np.arange(1, len(obj0) + 1)
        has_masked = np.isin(obj_seg_ids, masked_labels)
        obj0['flag'][has_masked] |= 0x100

    # Handle FWHM parameter
    fwhm_value = None
    scale_with_fwhm = False

    if isinstance(fwhm, (int, float)) and not isinstance(fwhm, bool) and fwhm > 0:
        # Numeric FWHM value provided directly
        fwhm_value = float(fwhm)
        scale_with_fwhm = True
        log("Using provided FWHM = %.2g" % fwhm_value)
    elif fwhm is True or (optimal and fwhm is False):
        # Estimate FWHM from detected objects
        # (either explicitly requested or needed for optimal extraction)
        idx_fwhm = obj0['flag'] == 0
        fwhm_est = 2.0 * np.sqrt(np.hypot(obj0['a'][idx_fwhm], obj0['b'][idx_fwhm]) * np.log(2))
        fwhm_est = (
            2.0
            * sep.flux_radius(
                image1,
                obj0['x'][idx_fwhm],
                obj0['y'][idx_fwhm],
                relfluxradius * fwhm_est * np.ones_like(obj0['x'][idx_fwhm]),
                0.5,
                mask=mask,
            )[0]
        )
        fwhm_value = np.median(fwhm_est)

        if fwhm is True:
            # Scale aperture/annulus when explicitly requested (fwhm=True)
            scale_with_fwhm = True

        log("Estimated FWHM = %.2g" % fwhm_value)

    # Scale aperture and background annulus with FWHM if requested
    if scale_with_fwhm and fwhm_value is not None:
        log("Scaling aperture and background annulus with FWHM")
        aper *= fwhm_value
        if bkgann is not None:
            bkgann = (bkgann[0] * fwhm_value, bkgann[1] * fwhm_value)
        log("  Aperture: %.2g pixels" % aper)
        if bkgann is not None:
            log("  Background annulus: (%.2g, %.2g) pixels" % bkgann)

    # Windowed positional parameters - use if requested
    if centroid:
        log("Refining positions using windowed centroiding")
        # Use sep.winpos for centroiding
        # maxstep parameter is only available in SEP 1.4+
        winpos_kwargs = {'mask': mask | mask_bg | mask_segm}
        if _HAS_SEP_OPTIMAL:
            # SEP 1.4+ has maxstep parameter to cap position shift per iteration
            # Set to 0.2 * FWHM to avoid excessive drift in degenerate cases
            if fwhm_value is not None:
                winpos_kwargs['maxstep'] = 0.2 * fwhm_value
            else:
                # Default to 1.0 pixel if FWHM not available
                winpos_kwargs['maxstep'] = 1.0

        xwin, ywin, flag_win = sep.winpos(image1, obj0['x'], obj0['y'], 0.5, **winpos_kwargs)
    else:
        # Use initial positions from extraction
        xwin, ywin = obj0['x'], obj0['y']
        flag_win = np.zeros(len(obj0), dtype=int)

    # Filter out objects too close to frame edges
    idx = (
        (np.round(xwin) > edge)
        & (np.round(ywin) > edge)
        & (np.round(xwin) < image.shape[1] - edge)
        & (np.round(ywin) < image.shape[0] - edge)
    )  # & (obj0['flag'] == 0)

    if minnthresh:
        idx &= obj0['tnpix'] >= minnthresh

    log("Measuring final objects")

    # Choose photometry method
    if optimal and _HAS_SEP_OPTIMAL:
        if group_sources:
            log("Using grouped optimal extraction")
        else:
            log("Using optimal extraction")
        flux, fluxerr, flag = sep.sum_circle_optimal(
            image1,
            xwin[idx],
            ywin[idx],
            aper,
            fwhm=fwhm_value,
            err=err,
            gain=gain,
            mask=mask | mask_bg | mask_segm,
            bkgann=bkgann,
            grouped=group_sources,
            group_radius_factor=1.2,  # Empirical
            clip_sigma=clip_sigma,
            clip_iters=clip_iters,
        )
    else:
        if optimal and not _HAS_SEP_OPTIMAL:
            log("WARNING: Optimal extraction requested but SEP 1.4+ not available, falling back to aperture photometry")
        flux, fluxerr, flag = sep.sum_circle(
            image1,
            xwin[idx],
            ywin[idx],
            aper,
            err=err,
            gain=gain,
            mask=mask | mask_bg | mask_segm,
            bkgann=bkgann,
            clip_sigma=clip_sigma,
            clip_iters=clip_iters,
        )

    # For debug purposes, let's make also the same aperture photometry on the background map
    bgflux, bgfluxerr, bgflag = sep.sum_circle(
        bg.back(),
        xwin[idx],
        ywin[idx],
        aper,
        err=bg.rms(),
        gain=gain,
        mask=mask | mask_bg | mask_segm,
        clip_sigma=clip_sigma,
        clip_iters=clip_iters,
    )

    bgnorm = bgflux / np.pi / aper ** 2

    # Fluxes to magnitudes
    mag, magerr = np.zeros_like(flux), np.zeros_like(flux)
    mag[flux > 0] = -2.5 * np.log10(flux[flux > 0])
    # magerr[flux>0] = 2.5*np.log10(1.0 + fluxerr[flux>0]/flux[flux>0])
    magerr[flux > 0] = 2.5 / np.log(10) * fluxerr[flux > 0] / flux[flux > 0]

    # FWHM estimation - prefer SEP's built-in FWHM (ported from SExtractor)
    # Fall back to 2nd moments for older SEP versions
    if 'fwhm' in obj0.dtype.names:
        # Use SEP's built-in FWHM (nearly identical to SExtractor)
        fwhm = obj0['fwhm'][idx]
    else:
        # Fall back to 2nd moments: FWHM = 2 * sqrt(ln(2) * (a^2 + b^2))
        # This is exact for Gaussian PSF, approximate for others
        # More robust than flux_radius (5-27x fewer outliers) but less accurate than SEP built-in
        fwhm = 2.0 * np.sqrt(np.log(2) * (obj0['a'][idx]**2 + obj0['b'][idx]**2))

    flag |= obj0['flag'][idx] | flag_win[idx]

    # Quality cuts
    fidx = (flux > 0) & (magerr < 1.0 / sn)

    if wcs is None and header is not None:
        # If header is provided, we may build WCS from it
        wcs = WCS(header)

    if wcs is not None:
        # If WCS is provided we may convert x,y to ra,dec
        ra, dec = wcs.all_pix2world(obj0['x'][idx], obj0['y'][idx], 0)
    else:
        ra, dec = np.zeros_like(obj0['x'][idx]), np.zeros_like(obj0['y'][idx])

    if verbose:
        log("All done")

    obj = Table(
        {
            'x': xwin[idx][fidx],
            'y': ywin[idx][fidx],
            'xerr': np.sqrt(obj0['errx2'][idx][fidx]),
            'yerr': np.sqrt(obj0['erry2'][idx][fidx]),
            'flux': flux[fidx],
            'fluxerr': fluxerr[fidx],
            'mag': mag[fidx],
            'magerr': magerr[fidx],
            'flags': obj0['flag'][idx][fidx] | flag[fidx],
            'ra': ra[fidx],
            'dec': dec[fidx],
            'bg': bgnorm[fidx],
            'fwhm': fwhm[fidx],
            'a': obj0['a'][idx][fidx],
            'b': obj0['b'][idx][fidx],
            'theta': obj0['theta'][idx][fidx],
        }
    )

    obj.meta['aper'] = aper
    obj.meta['bkgann'] = bkgann
    obj.meta['optimal'] = optimal and _HAS_SEP_OPTIMAL
    obj.meta['group_sources'] = group_sources and optimal and _HAS_SEP_OPTIMAL
    if fwhm_value is not None:
        obj.meta['fwhm_phot'] = fwhm_value

    obj.sort('flux', reverse=True)

    if get_segmentation:
        # Segmentation map ID (sequential object number + 1, 1-based indexing)
        seg_id = np.arange(len(obj0['x'])) + 1
        obj['seg_id'] = seg_id[idx][fidx]

        return obj, segm
    else:
        return obj


def get_objects_sextractor(
    image,
    header=None,
    mask=None,
    mask_detect=None,
    err=None,
    thresh=2.0,
    aper=3.0,
    r0=0.0,
    gain=1,
    edge=0,
    minarea=5,
    wcs=None,
    sn=3.0,
    bg_size=None,
    sort=True,
    reject_negative=True,
    mask_to_nans=True,
    checkimages=[],
    extra_params=[],
    extra={},
    psf=None,
    catfile=None,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Thin wrapper around SExtractor binary.

    It processes the image taking into account optional mask and noise map, and returns the list of detected objects and optionally a set of SExtractor-produced checkimages.

    You may check the SExtractor documentation at https://sextractor.readthedocs.io/en/latest/ for more details about possible parameters and general principles of its operation.
    E.g. detection flags (returned in `flags` column of results table) are documented at https://sextractor.readthedocs.io/en/latest/Flagging.html#extraction-flags-flags . In addition to these flags, any object having pixels masked by the input `mask` in its footprint will have :code:`0x100` flag set.

    :param image: Input image as a NumPy array
    :param header: Image header, optional
    :param mask: Image mask as a boolean array (True values will be masked), optional. Used for SExtractor flagging (FLAG_IMAGE / IMAFLAGS_ISO). Objects whose footprint overlaps this mask get flag 0x100.
    :param mask_detect: Optional detection mask as a boolean array (True = exclude from detection). If provided and mask_to_nans is True, these pixels (plus non-finite pixels) are set to NaN before running SExtractor, effectively excluding them from detection. Use this to exclude entire regions (e.g. bad columns, satellite trails) from detection while keeping the user mask only for flagging.
    :param err: Image noise map as a NumPy array, optional
    :param thresh: Detection threshold, in sigmas above local background, to be used for `DETECT_THRESH` parameter of SExtractor call
    :param aper: Circular aperture radius in pixels, to be used for flux measurement. May also be list - then flux will be measured for all apertures from that list.
    :param r0: Smoothing kernel size (sigma, or FWHM/2.355) to be used for improving object detection
    :param gain: Image gain, e/ADU
    :param edge: Reject all detected objects closer to image edge than this parameter
    :param minarea: Minimal number of pixels in the object to be considered a detection (`DETECT_MINAREA` parameter of SExtractor)
    :param wcs: Astrometric solution to be used for assigning sky coordinates (`ra`/`dec`) to detected objects
    :param sn: Minimal S/N ratio for the object to be considered a detection
    :param bg_size: Background grid size in pixels (`BACK_SIZE` SExtractor parameter)
    :param sort: Whether to sort the detections in decreasing brightness or not
    :param reject_negative: Whether to reject the detections with negative fluxes
    :param mask_to_nans: Whether to replace detection-masked pixels with NaNs prior to running SExtractor. When True (default), pixels in mask_detect (if provided) plus non-finite pixels are set to NaN, excluding them from detection while keeping the user mask only for flagging. When False, no pixels are NaN-ed (original behavior).
    :param checkimages: List of SExtractor checkimages to return along with detected objects. Any SExtractor checkimage type may be used here (e.g. `BACKGROUND`, `BACKGROUND_RMS`, `MINIBACKGROUND`,  `MINIBACK_RMS`, `-BACKGROUND`, `FILTERED`, `OBJECTS`, `-OBJECTS`, `SEGMENTATION`, `APERTURES`). Optional.
    :param extra_params: List of extra object parameters to return for the detection. See :code:`sex -dp` for the full list.
    :param extra: Dictionary of extra configuration parameters to be passed to SExtractor call, with keys as parameter names. See :code:`sex -dd` for the full list.
    :param psf: Path to PSFEx-made PSF model file to be used for PSF photometry. If provided, a set of PSF-measured parameters (`FLUX_PSF`, `MAG_PSF` etc) are added to detected objects. Optional
    :param catfile: If provided, output SExtractor catalogue file will be copied to this location, to be reused by external codes. Optional.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SExtractor. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to SExtractor executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Either the astropy.table.Table object with detected objects, or a list with table of objects (first element) and checkimages (consecutive elements), if checkimages are requested.
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
        # Find SExtractor binary in common paths
        for exe in ['sex', 'sextractor', 'source-extractor']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SExtractor binary")
        return None
    # else:
    #     log("Using SExtractor binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='sex', dir=_tmpdir)
    )
    obj = None

    if mask is None:
        # Create minimal mask
        mask = ~np.isfinite(image)
    else:
        # Ensure the mask is boolean array
        mask = mask.astype(bool)

    # Build detection mask: non-finite pixels plus optional user-provided
    # detection mask.  When mask_to_nans is True, these pixels are set to NaN
    # so that SExtractor ignores them during detection and deblending.  The
    # user mask (FLAG_IMAGE) is kept separate for flagging only, so that
    # bright objects with partially masked cores are not fragmented.
    mask_det = ~np.isfinite(image)
    if mask_detect is not None:
        mask_det = mask_det | np.asarray(mask_detect, dtype=bool)

    if mask_to_nans and mask_det.any():
        image = image.copy() if np.issubdtype(image.dtype, np.floating) else image.astype(np.float64)
        image[mask_det] = np.nan

    # Prepare
    if type(image) == str:
        # FIXME: this mode of operation is currently broken!
        imagename = image
    else:
        imagename = os.path.join(workdir, 'image.fits')
        fits.writeto(imagename, image, header, overwrite=True)

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'DETECT_MINAREA': minarea,
        'GAIN': gain,
        'DETECT_THRESH': thresh,
        'WEIGHT_TYPE': 'BACKGROUND',
        'MASK_TYPE': 'NONE',  # both 'CORRECT' and 'BLANK' seem to cause systematics?
        'SATUR_LEVEL': np.nanmax(image[~mask]) + 1 # Saturation should be handled in external mask
    }

    if bg_size is not None:
        opts['BACK_SIZE'] = bg_size

    if err is not None:
        # User-provided noise model
        err = err.copy().astype(np.double)
        err[~np.isfinite(err)] = 1e30
        err[err == 0] = 1e30

        errname = os.path.join(workdir, 'errors.fits')
        fits.writeto(errname, err, overwrite=True)
        opts['WEIGHT_IMAGE'] = errname
        opts['WEIGHT_TYPE'] = 'MAP_RMS'

    flagsname = os.path.join(workdir, 'flags.fits')
    fits.writeto(flagsname, mask.astype(np.int16), overwrite=True)
    opts['FLAG_IMAGE'] = flagsname

    if np.isscalar(aper):
        opts['PHOT_APERTURES'] = aper * 2  # SExtractor expects diameters, not radii
        size = ''
    else:
        opts['PHOT_APERTURES'] = ','.join([str(_ * 2) for _ in aper])
        size = '[%d]' % len(aper)

    checknames = [
        os.path.join(workdir, _.replace('-', 'M_') + '.fits') for _ in checkimages
    ]
    if checkimages:
        opts['CHECKIMAGE_TYPE'] = ','.join(checkimages)
        opts['CHECKIMAGE_NAME'] = ','.join(checknames)

    params = [
        'MAG_APER' + size,
        'MAGERR_APER' + size,
        'FLUX_APER' + size,
        'FLUXERR_APER' + size,
        'X_IMAGE',
        'Y_IMAGE',
        'ERRX2_IMAGE',
        'ERRY2_IMAGE',
        'A_IMAGE',
        'B_IMAGE',
        'THETA_IMAGE',
        'FLUX_RADIUS',
        'FWHM_IMAGE',
        'FLAGS',
        'IMAFLAGS_ISO',
        'BACKGROUND',
    ]
    params += extra_params

    if psf is not None:
        opts['PSF_NAME'] = psf
        params += [
            'MAG_PSF',
            'MAGERR_PSF',
            'FLUX_PSF',
            'FLUXERR_PSF',
            'XPSF_IMAGE',
            'YPSF_IMAGE',
            'SPREAD_MODEL',
            'SPREADERR_MODEL',
            'CHI2_PSF',
        ]

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
        kernel = make_kernel(r0, ext=2.0)
        kernelname = os.path.join(workdir, 'kernel.txt')
        np.savetxt(
            kernelname,
            kernel / np.sum(kernel),
            fmt='%.6f',
            header='CONV NORM',
            comments='',
        )
        opts['FILTER'] = 'Y'
        opts['FILTER_NAME'] = kernelname

    opts.update(extra)

    # Build the command line
    cmd = (
        binname
        + ' '
        + shlex.quote(imagename)
        + ' '
        + utils.format_astromatic_opts(opts)
    )
    if not verbose:
        cmd += ' > /dev/null 2>/dev/null'
    log('Will run SExtractor like that:')
    log(cmd)

    # Run the command!

    res = os.system(cmd)

    if res == 0 and os.path.exists(catname):
        log('SExtractor run succeeded')
        obj = Table.read(catname, hdu=2)
        obj.meta.clear()  # Remove unnecessary entries from the metadata

        idx = (obj['X_IMAGE'] > edge) & (obj['X_IMAGE'] < image.shape[1] - edge)
        idx &= (obj['Y_IMAGE'] > edge) & (obj['Y_IMAGE'] < image.shape[0] - edge)

        if np.isscalar(aper):
            if sn:
                idx &= obj['MAGERR_APER'] < 1.0 / sn
            if reject_negative:
                idx &= obj['FLUX_APER'] > 0
        else:
            if sn:
                idx &= np.all(obj['MAGERR_APER'] < 1.0 / sn, axis=1)
            if reject_negative:
                idx &= np.all(obj['FLUX_APER'] > 0, axis=1)

        obj = obj[idx]

        if wcs is None and header is not None:
            wcs = WCS(header)

        if wcs is not None:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['X_IMAGE'], obj['Y_IMAGE'], 1)
        else:
            obj['ra'], obj['dec'] = (
                np.zeros_like(obj['X_IMAGE']),
                np.zeros_like(obj['Y_IMAGE']),
            )

        obj['FLAGS'][obj['IMAFLAGS_ISO'] > 0] |= 0x100  # Masked pixels in the footprint
        obj.remove_column('IMAFLAGS_ISO')  # We do not need this column

        # Convert variances to rms
        obj['ERRX2_IMAGE'] = np.sqrt(obj['ERRX2_IMAGE'])
        obj['ERRY2_IMAGE'] = np.sqrt(obj['ERRY2_IMAGE'])

        for _, __ in [
            ['X_IMAGE', 'x'],
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
            ['THETA_IMAGE', 'theta'],
        ]:
            obj.rename_column(_, __)

        if psf:
            for _, __ in [
                ['XPSF_IMAGE', 'x_psf'],
                ['YPSF_IMAGE', 'y_psf'],
                ['MAG_PSF', 'mag_psf'],
                ['MAGERR_PSF', 'magerr_psf'],
                ['FLUX_PSF', 'flux_psf'],
                ['FLUXERR_PSF', 'fluxerr_psf'],
                ['CHI2_PSF', 'chi2_psf'],
                ['SPREAD_MODEL', 'spread_model'],
                ['SPREADERR_MODEL', 'spreaderr_model'],
            ]:
                if _ in obj.keys():
                    obj.rename_column(_, __)
                    if 'mag' in __:
                        obj[__][obj[__] == 99] = np.nan  # TODO: use masked column here?

        # SExtractor uses 1-based pixel coordinates
        obj['x'] -= 1
        obj['y'] -= 1

        if 'x_psf' in obj.keys():
            obj['x_psf'] -= 1
            obj['y_psf'] -= 1

        obj.meta['aper'] = aper

        if sort:
            if np.isscalar(aper):
                obj.sort('flux', reverse=True)
            else:
                # Table sorting by vector columns seems to be broken?..
                obj = obj[np.argsort(-obj['flux'][:, 0])]

        if catfile is not None:
            shutil.copyfile(catname, catfile)
            log("Catalogue stored to", catfile)

    else:
        log("Error", res, "running SExtractor")

    result = obj

    if checkimages:
        result = [result]

        for name in checknames:
            if os.path.exists(name):
                result.append(fits.getdata(name))
            else:
                log("Cannot find requested output checkimage file", name)
                result.append(None)

    if _workdir is None:
        shutil.rmtree(workdir)

    return result


def _empty_table(get_segmentation=False):
    """
    Return empty table with correct structure.

    Parameters
    ----------
    get_segmentation : bool
        If True, return tuple (table, None), otherwise just table

    Returns
    -------
    table : astropy.table.Table
        Empty table with standard columns
    segm : None (optional)
        None segmentation map if get_segmentation=True
    """
    obj = Table()
    obj['x'] = []
    obj['y'] = []
    obj['flux'] = []
    obj['fluxerr'] = []
    obj['mag'] = []
    obj['magerr'] = []
    obj['fwhm'] = []
    obj['a'] = []
    obj['b'] = []
    obj['theta'] = []
    obj['bg'] = []
    obj['flags'] = []
    obj['xerr'] = []
    obj['yerr'] = []

    if get_segmentation:
        return obj, None
    else:
        return obj


class ConstantBackground:
    def __init__(self, value, rms, shape):
        self.background = np.full(shape, value)
        self.background_rms = np.full(shape, rms)


def get_value(val):
    """Get array from either Quantity or ndarray"""
    if hasattr(val, 'value'):
        return val.value
    else:
        return np.asarray(val)


def get_objects_photutils(
    image,
    header=None,
    mask=None,
    mask_detect=None,
    err=None,
    # Detection parameters
    thresh=2.0,
    method='segmentation',
    deblend=True,
    minarea=5,
    saturation=None,
    # Segmentation-specific parameters
    npixels=5,
    nlevels=32,
    contrast=0.001,
    connectivity=8,
    # StarFinder-specific parameters
    fwhm=3.0,
    sharplo=0.2,
    sharphi=1.0,
    roundlo=-1.0,
    roundhi=1.0,
    # Photometry parameters
    aper=3.0,
    bkgann=None,
    # Background parameters
    bg_size=64,
    subtract_bg=True,
    # Filtering parameters
    edge=0,
    sn=3.0,
    # Output control
    wcs=None,
    get_segmentation=False,
    verbose=False,
    **kwargs
):
    """
    Detect sources in image using photutils detection algorithms.

    This function provides an alternative to get_objects_sep() using
    photutils instead of SEP. It supports multiple detection methods
    including segmentation-based detection with deblending and
    point source detection using DAOStarFinder or IRAFStarFinder.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array
    header : astropy.io.fits.Header, optional
        FITS header for WCS information
    mask : numpy.ndarray, optional
        Boolean mask (True = masked pixel). Used for background estimation,
        photometry, and flagging (0x100). Not used for detection itself.
    mask_detect : numpy.ndarray, optional
        Boolean mask for detection (True = exclude from detection). OR-ed with
        non-finite pixel mask. If None, only non-finite pixels are excluded
        from detection.
    err : numpy.ndarray, optional
        Error/uncertainty map. If None, uses background RMS
    thresh : float, optional
        Detection threshold in sigma units (default: 2.0)
    method : str, optional
        Detection method: 'segmentation', 'dao', or 'iraf' (default: 'segmentation')
    deblend : bool, optional
        Apply deblending for segmentation method (default: True)
    minarea : int, optional
        Minimum source area in pixels (default: 5)
    saturation : float, optional
        Saturation threshold for flagging saturated sources. Sources with
        peak pixel values exceeding this threshold will have flag 0x004 set.
        If None (default), saturation detection is disabled.
    npixels : int, optional
        Minimum connected pixels for segmentation (default: 5)
    nlevels : int, optional
        Number of deblending levels (default: 32)
    contrast : float, optional
        Deblending contrast threshold (default: 0.001)
    connectivity : int, optional
        Pixel connectivity (4 or 8) for segmentation (default: 8)
    fwhm : float, optional
        FWHM for StarFinder methods in pixels (default: 3.0)
    sharplo : float, optional
        Lower sharpness bound for StarFinder (default: 0.2)
    sharphi : float, optional
        Upper sharpness bound for StarFinder (default: 1.0)
    roundlo : float, optional
        Lower roundness bound for StarFinder (default: -1.0)
    roundhi : float, optional
        Upper roundness bound for StarFinder (default: 1.0)
    aper : float, optional
        Aperture radius in pixels (default: 3.0)
    bkgann : tuple, optional
        Background annulus (inner, outer) radii in pixels
    bg_size : int, optional
        Background estimation box size (default: 64)
    subtract_bg : bool, optional
        Subtract background before detection (default: True)
    edge : int, optional
        Edge exclusion in pixels (default: 0)
    sn : float, optional
        Minimum S/N ratio (default: 3.0)
    wcs : astropy.wcs.WCS, optional
        WCS for coordinate conversion. If None, extracted from header
    get_segmentation : bool, optional
        Return segmentation map (only for method='segmentation') (default: False)
    verbose : bool or callable, optional
        Verbose output (default: False)
    **kwargs
        Additional parameters passed to photutils functions

    Returns
    -------
    table : astropy.table.Table
        Table of detected sources with columns:
        - x, y: pixel coordinates (0-indexed)
        - xerr, yerr: position errors
        - flux, fluxerr: aperture flux and error
        - mag, magerr: instrumental magnitude and error
        - fwhm: FWHM estimate
        - a, b: semi-major and semi-minor axes
        - theta: position angle
        - bg: local background
        - flags: detection flags
        - ra, dec: WCS coordinates (if WCS available)
    segm : photutils.segmentation.SegmentationImage, optional
        Segmentation map (only if get_segmentation=True and method='segmentation')

    Notes
    -----
    The function follows the same workflow as get_objects_sep():
    1. Validate inputs and create mask from NaNs
    2. Estimate background using Background2D
    3. Optionally subtract background
    4. Detect sources using selected method
    5. Measure aperture photometry
    6. Apply quality filters
    7. Add WCS coordinates if available
    8. Return table with standard columns

    Detection methods:
    - 'segmentation': Uses detect_sources() with optional deblending.
      Best for extended sources and crowded fields.
    - 'dao': Uses DAOStarFinder for point sources.
      Best for stellar fields with well-defined PSF.
    - 'iraf': Uses IRAFStarFinder for IRAF-compatible detection.
      Similar to 'dao' but with IRAF-style parameters.

    Examples
    --------
    Basic segmentation detection:

    >>> obj = get_objects_photutils(image, thresh=3.0, aper=5.0)

    Segmentation with deblending:

    >>> obj = get_objects_photutils(
    ...     image, method='segmentation', deblend=True,
    ...     nlevels=32, contrast=0.001
    ... )

    Point source detection:

    >>> obj = get_objects_photutils(
    ...     image, method='dao', fwhm=3.0, thresh=5.0
    ... )

    With background annulus:

    >>> obj = get_objects_photutils(
    ...     image, aper=5.0, bkgann=(10.0, 15.0)
    ... )

    Get segmentation map:

    >>> obj, segm = get_objects_photutils(
    ...     image, method='segmentation', get_segmentation=True
    ... )
    """
    # Validate inputs
    if image.ndim != 2:
        raise ValueError("Image must be 2D array")

    # Setup logging
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    # Create detection mask from NaNs and optional mask_detect
    mask_det = ~np.isfinite(image)
    if mask_detect is not None:
        mask_det = mask_det | np.asarray(mask_detect, dtype=bool)

    if mask is None:
        mask = np.zeros(image.shape, dtype=bool)
    else:
        mask = np.array(mask).astype(bool)

    # Saturation map
    if saturation is not None:
        log(f'Using saturation level {saturation:.2f}')

        smask = image >= saturation

        # Let's subtract saturation mask from user-provided one so that it does not prevent proper segmentation
        mask = mask & (~smask)
    else:
        smask = np.zeros(image.shape, dtype=bool)

    total_mask = mask | mask_det | smask

    # Extract WCS from header if provided
    if wcs is None and header is not None:
        try:
            wcs = WCS(header)
            if not wcs.is_celestial:
                wcs = None
        except Exception:
            wcs = None

    # Estimate background using photutils
    log(f'Estimating background with {bg_size}x{bg_size} grid')

    try:
        bkg = photutils.background.Background2D(
            image,
            box_size=bg_size,
            mask=total_mask,
            bkg_estimator=photutils.background.ModeEstimatorBackground(),
            exclude_percentile=10
        )
    except Exception as e:
        log(f'Warning: Background estimation failed: {e}')
        # Fallback to simple median
        bkg = ConstantBackground(
            np.nanmedian(image[~total_mask]) if np.any(~total_mask) else 0.0,
            np.nanstd(image[~total_mask]) if np.any(~total_mask) else 1.0,
            image.shape
        )

    # These things are not cached internally, so let's avoid re-computing them
    bkg_back = bkg.background
    bkg_rms = bkg.background_rms

    # Optionally subtract background
    if subtract_bg:
        image_bgsub = image - bkg.background
        log(f'Subtracting background: median {np.nanmedian(bkg_back):.2f}, '
            f'rms {np.nanmedian(bkg_rms):.2f}')
    else:
        image_bgsub = image.copy()

    # Create or use error map
    if err is None:
        err = bkg_rms
        log('Using background RMS as error map')
    else:
        log('Using provided error map')

    # Initialize variables for different methods
    segm = None
    catalog = None
    sources = None

    # Source detection based on different methods
    if method == 'segmentation':
        # Detect sources via segmentation
        threshold = thresh * err

        log(f'Detecting sources with threshold {thresh} sigma, at least {npixels} pixels')
        segm = photutils.segmentation.detect_sources(
            image_bgsub,
            threshold=threshold,
            npixels=npixels,
            connectivity=connectivity,
            mask=mask_det
        )

        if segm is None or segm.nlabels == 0:
            log('No sources detected')
            return _empty_table(get_segmentation)

        log(f'Detected {segm.nlabels} initial segments')

        # Optionally deblend sources
        if deblend and segm.nlabels > 0:
            log(f'Deblending with nlevels {nlevels}, contrast {contrast}')
            try:
                segm_deblend = photutils.segmentation.deblend_sources(
                    image_bgsub,
                    segm,
                    npixels=npixels,
                    nlevels=nlevels,
                    contrast=contrast
                )
                segm = segm_deblend
                log(f'Deblended to {segm.nlabels} sources')
            except Exception as e:
                log(f'Warning: Deblending failed: {e}')

        # Extract source properties
        catalog = photutils.segmentation.SourceCatalog(
            image_bgsub,
            segm,
            error=err,
            mask=mask_det,
            background=bkg_back if subtract_bg else None
        )

        # Convert to arrays (handle both Quantity and ndarray)
        x = get_value(catalog.xcentroid)
        y = get_value(catalog.ycentroid)
        area = get_value(catalog.area)

    elif method in ['dao', 'iraf']:
        # Calculate threshold
        threshold_value = thresh * np.nanmedian(err)

        # Select appropriate finder
        if method == 'dao':
            log(f'Using DAOStarFinder with fwhm={fwhm}, threshold={thresh} sigma')
            finder = photutils.detection.DAOStarFinder(
                fwhm=fwhm,
                threshold=threshold_value,
                sharplo=sharplo,
                sharphi=sharphi,
                roundlo=roundlo,
                roundhi=roundhi,
                exclude_border=True
            )
        else:  # method == 'iraf'
            log(f'Using IRAFStarFinder with fwhm={fwhm}, threshold={thresh} sigma')
            finder = photutils.detection.IRAFStarFinder(
                fwhm=fwhm,
                threshold=threshold_value,
                sharplo=sharplo,
                sharphi=sharphi,
                roundlo=roundlo,
                roundhi=roundhi,
                exclude_border=True
            )

        # Find sources
        try:
            sources = finder(image_bgsub, mask=mask_det)
        except Exception as e:
            log(f'Warning: Source detection failed: {e}')
            sources = None

        if sources is None or len(sources) == 0:
            log('No sources detected')
            return _empty_table(get_segmentation=False)

        log(f'Detected {len(sources)} sources')

        # Extract positions (handle both Quantity and ndarray)
        x = get_value(sources['xcentroid'])
        y = get_value(sources['ycentroid'])
        # StarFinder doesn't provide area, use approximate
        area = np.full(len(sources), np.pi * (fwhm / 2)**2)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'segmentation', 'dao', or 'iraf'")

    # Aperture photometry
    log(f'Performing aperture photometry with {aper} pixels aperture')

    # Create apertures at detected positions
    positions = np.column_stack([x, y])
    apertures = photutils.aperture.CircularAperture(positions, r=aper)

    # Optional background annulus
    if bkgann is not None:
        log(f'Using background annulus: {bkgann}')
        bkg_apertures = photutils.aperture.CircularAnnulus(
            positions,
            r_in=bkgann[0],
            r_out=bkgann[1]
        )

        # Measure background
        bkg_phot = photutils.aperture.aperture_photometry(
            image_bgsub,
            bkg_apertures,
            error=err,
            mask=mask
        )
        bkg_mean = bkg_phot['aperture_sum'] / bkg_apertures.area

        # Apply background correction to aperture
        phot_table = photutils.aperture.aperture_photometry(
            image_bgsub,
            apertures,
            error=err,
            mask=mask
        )

        # Handle both Quantity and ndarray
        aper_sum = get_value(phot_table['aperture_sum'])
        aper_sum_err = get_value(phot_table['aperture_sum_err'])
        bkg_mean_arr = get_value(bkg_mean)

        flux = aper_sum - bkg_mean_arr * apertures.area
        fluxerr = aper_sum_err
    else:
        # Simple aperture photometry
        phot_table = photutils.aperture.aperture_photometry(
            image_bgsub,
            apertures,
            error=err,
            mask=mask
        )

        # Handle both Quantity and ndarray
        flux = get_value(phot_table['aperture_sum'])
        fluxerr = get_value(phot_table['aperture_sum_err'])

    # Convert to magnitudes
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5 * np.log10(flux)
        magerr = 2.5 / np.log(10) * fluxerr / flux

    # Build output table
    obj = Table()
    obj['x'] = x
    obj['y'] = y
    obj['flux'] = flux
    obj['fluxerr'] = fluxerr
    obj['mag'] = mag
    obj['magerr'] = magerr
    if segm is not None:
        obj['label'] = catalog.labels

    # Add shape parameters
    if method == 'segmentation':
        # Extract from catalog (handle both Quantity and ndarray)
        semimajor = get_value(catalog.semimajor_sigma)
        semiminor = get_value(catalog.semiminor_sigma)
        orientation = get_value(catalog.orientation)

        obj['a'] = semimajor * 2.355  # Convert sigma to FWHM
        obj['b'] = semiminor * 2.355
        obj['theta'] = orientation
        obj['fwhm'] = (obj['a'] + obj['b']) / 2
        obj['flags'] = np.zeros(len(obj), dtype=int)

        # Set detection flags for segmentation method
        try:
            # 0x002: Deblended sources
            if deblend and segm is not None and hasattr(segm, 'deblended_labels'):
                deblended_labels = set(segm.deblended_labels)
                obj['flags'][np.isin(obj['label'], deblended_labels)] |= 0x002

            # 0x008: Footprint truncated at image boundary
            bbox_xmin = get_value(catalog.bbox_xmin)
            bbox_xmax = get_value(catalog.bbox_xmax)
            bbox_ymin = get_value(catalog.bbox_ymin)
            bbox_ymax = get_value(catalog.bbox_ymax)

            for i in range(len(obj)):
                if (bbox_xmin[i] == 0 or bbox_xmax[i] >= image.shape[1] or
                    bbox_ymin[i] == 0 or bbox_ymax[i] >= image.shape[0]):
                    obj['flags'][i] |= 0x008

            # 0x004: Saturated sources (if saturation threshold provided)
            if saturation is not None:
                max_val = get_value(catalog.max_value)
                obj['flags'][max_val >= saturation] |= 0x004

        except Exception as e:
            log(f'Warning: Flag detection failed for segmentation: {e}')

    else:  # StarFinder
        obj['fwhm'] = np.full(len(obj), fwhm)
        obj['a'] = obj['fwhm']
        obj['b'] = obj['fwhm']
        obj['theta'] = np.zeros(len(obj))
        obj['flags'] = np.zeros(len(obj), dtype=int)

        # Set detection flags for StarFinder methods
        try:
            # 0x010: Poor quality metrics (sharpness/roundness near rejection thresholds)
            margin = 0.1  # 10% margin from thresholds

            if 'sharpness' in sources.colnames:
                sharp = get_value(sources['sharpness'])
                poor_sharp = (sharp < sharplo * (1 + margin)) | (sharp > sharphi * (1 - margin))
                obj['flags'][poor_sharp] |= 0x010

            if 'roundness1' in sources.colnames:
                round1 = get_value(sources['roundness1'])
                poor_round = (round1 < roundlo * (1 + margin)) | (round1 > roundhi * (1 - margin))
                obj['flags'][poor_round] |= 0x010

            # 0x008: Truncated at boundary (near edge, even though exclude_border=True)
            near_edge = (x <= 1) | (x >= image.shape[1] - 2) | (y <= 1) | (y >= image.shape[0] - 2)
            obj['flags'][near_edge] |= 0x008

            # 0x004: Saturated sources
            if saturation is not None and 'peak' in sources.colnames:
                peak = get_value(sources['peak'])
                obj['flags'][peak >= saturation] |= 0x004

        except Exception as e:
            log(f'Warning: Flag detection failed for StarFinder: {e}')

    # 0x100: User-masked pixels in footprint (both methods)
    try:
        if np.any(mask):
            if method == 'segmentation':
                # Use segmentation map
                labels = list(set(segm.data[mask])) # footprints with masked pixels
                obj['flags'][np.isin(obj['label'], labels)] |= 0x100
            else:
                # Check source apertures for masked pixels
                positions = np.column_stack([obj['x'], obj['y']])
                apertures = photutils.aperture.CircularAperture(positions, r=fwhm/2)
                mres = aperture_photometry(mask.astype(int), apertures, method='center')
                obj['flags'][mres['aperture_sum'] > 0] |= 0x100
    except Exception as e:
        log(f'Warning: Masked pixel detection failed: {e}')

    # Add background at each source
    if method == 'segmentation':
        bg_at_centroid = get_value(catalog.background_centroid)
        obj['bg'] = bg_at_centroid
    else:
        obj['bg'] = [
            bkg_back[_y,_x]
            if _x >= 0 and _x < image.shape[1] and _y >= 0 and _y < image.shape[0] else np.nan
            for _x,_y in zip(obj['x'].astype(int),obj['y'].astype(int))
        ]

    # Add position errors (simple estimate based on S/N)
    with np.errstate(divide='ignore', invalid='ignore'):
        positional_error = 0.5 / (flux / fluxerr)
        positional_error = np.clip(positional_error, 0.1, 5.0)

    obj['xerr'] = positional_error
    obj['yerr'] = positional_error

    # Add metadata
    obj.meta['aper'] = aper
    if bkgann is not None:
        obj.meta['bkgann'] = bkgann
    obj.meta['method'] = method
    obj.meta['thresh'] = thresh
    if method == 'segmentation':
        obj.meta['deblend'] = deblend

    # Apply quality filters
    log(f'Applying quality filters: edge={edge}, sn={sn}, minarea={minarea}')

    # Edge filter
    if edge > 0:
        idx = (obj['x'] > edge) & (obj['x'] < image.shape[1] - edge)
        idx &= (obj['y'] > edge) & (obj['y'] < image.shape[0] - edge)
    else:
        idx = np.ones(len(obj), dtype=bool)

    # Area filter (for segmentation)
    if method == 'segmentation':
        idx &= area >= minarea

    # S/N filter
    if sn > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            obj_sn = flux / fluxerr
        idx &= obj_sn > sn

    # Positive flux filter
    idx &= flux > 0

    # Apply filters
    n_before = len(obj)
    obj = obj[idx]

    # Update segmentation map if needed
    if method == 'segmentation' and get_segmentation and segm is not None:
        # Filter segmentation to keep only selected sources
        # Note: This keeps the original segmentation but user can cross-reference with obj table
        pass

    log(f'Filtered {n_before} -> {len(obj)} sources')

    # Sort by brightness (flux, descending)
    if len(obj) > 0:
        obj.sort('flux', reverse=True)

    # Add RA/Dec if WCS available
    if wcs is not None and wcs.has_celestial:
        try:
            coords = wcs.all_pix2world(obj['x'], obj['y'], 0)
            obj['ra'] = coords[0]
            obj['dec'] = coords[1]
            log('Added WCS coordinates (RA, Dec)')
        except Exception as e:
            log(f'Warning: WCS conversion failed: {e}')

    log(f'Returning {len(obj)} sources')

    # Return results
    if get_segmentation:
        if method == 'segmentation':
            return obj, segm.data
        else:
            log('Warning: get_segmentation=True but method is not segmentation')
            return obj, None
    else:
        return obj
