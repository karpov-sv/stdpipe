"""
Aperture photometry using PyRAF DAOPHOT routines.

This module provides photometry functionality compatible with stdpipe.photometry
but using IRAF/DAOPHOT tasks for aperture photometry measurements.
"""


import os
import tempfile
import shutil
import numpy as np

from astropy.io import fits
from astropy.table import Table
import photutils.background

# Check if pyraf is available without importing it yet
# (importing pyraf initializes IRAF which can cause issues with pytest stdin/stdout capturing)
def _check_pyraf_available():
    """Check if pyraf is available without triggering initialization."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("pyraf")
        return spec is not None
    except (ImportError, ValueError):
        return False

PYRAF_AVAILABLE = _check_pyraf_available()

# Global variable to hold iraf module once imported
_iraf = None

def _get_iraf():
    """Lazy import of pyraf.iraf module."""
    global _iraf
    if _iraf is None:
        try:
            from pyraf import iraf
            _iraf = iraf
        except ImportError as e:
            raise ImportError("PyRAF is not available. Please install PyRAF to use this module.") from e
    return _iraf


def _select_psf_stars(obj, fwhm, n_stars=20, isolation_radius=None, max_stars=50, verbose=False):
    """
    Select good PSF stars from object table.

    Selects stars suitable for building a PSF model based on:
    - Brightness (high flux or low magnitude)
    - High S/N ratio (if flux_err available)
    - Isolation (no nearby neighbors)
    - Not saturated (if flags available)
    - Roundness (if shape parameters available)

    :param obj: Object table with detected sources
    :param fwhm: FWHM in pixels for isolation checking
    :param n_stars: Target number of PSF stars to select
    :param isolation_radius: Isolation radius in pixels (default: 10*fwhm)
    :param max_stars: Maximum number of candidates to consider
    :param verbose: Verbose output
    :returns: Array of indices of selected PSF stars
    """
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if len(obj) == 0:
        log("No objects available for PSF star selection")
        return np.array([], dtype=int)

    if isolation_radius is None:
        isolation_radius = 10 * fwhm

    # Start with all objects
    candidates = np.ones(len(obj), dtype=bool)

    # Filter out saturated objects if flags are available
    if 'flags' in obj.colnames:
        # Assume flag 0x04 indicates saturation (common convention)
        saturated = (obj['flags'] & 0x04) != 0
        candidates &= ~saturated
        log(f"Removed {np.sum(saturated)} saturated objects")

    # Prefer objects with good S/N if flux_err is available
    if 'flux' in obj.colnames and 'flux_err' in obj.colnames:
        sn = obj['flux'] / obj['flux_err']
        # Require S/N > 20 for PSF stars
        good_sn = sn > 20
        candidates &= good_sn
        log(f"Filtered to {np.sum(candidates)} objects with S/N > 20")

    # Require positive flux
    if 'flux' in obj.colnames:
        candidates &= obj['flux'] > 0

    if np.sum(candidates) == 0:
        log("Warning: No suitable PSF star candidates found")
        return np.array([], dtype=int)

    # Sort by brightness (flux or magnitude)
    if 'flux' in obj.colnames:
        # Use flux - higher is brighter
        brightness = obj['flux'].copy()
        brightness[~candidates] = -np.inf
        sorted_idx = np.argsort(brightness)[::-1]  # Descending order
    elif 'mag' in obj.colnames:
        # Use magnitude - lower is brighter
        brightness = obj['mag'].copy()
        brightness[~candidates] = np.inf
        sorted_idx = np.argsort(brightness)  # Ascending order
    else:
        log("Warning: No flux or mag column available, cannot sort by brightness")
        sorted_idx = np.arange(len(obj))

    # Consider only the brightest candidates
    sorted_idx = sorted_idx[:max_stars]

    # Select isolated stars
    selected = []
    x_all = obj['x']
    y_all = obj['y']

    for idx in sorted_idx:
        if not candidates[idx]:
            continue

        # Check isolation - no other bright stars nearby
        x, y = x_all[idx], y_all[idx]
        distances = np.sqrt((x_all - x)**2 + (y_all - y)**2)

        # Count neighbors within isolation radius (excluding self)
        neighbors = np.sum((distances < isolation_radius) & (distances > 0) & candidates)

        if neighbors == 0:
            selected.append(idx)
            if len(selected) >= n_stars:
                break

    selected = np.array(selected, dtype=int)
    log(f"Selected {len(selected)} isolated PSF stars from {np.sum(candidates)} candidates")

    return selected


def measure_objects(
    obj,
    image,
    aper=3,
    bkgann=None,
    fwhm=None,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    sn=None,
    centroid_iter=0,
    keep_negative=True,
    get_bg=False,
    _workdir=None,
    _tmpdir=None,
    verbose=False,
):
    """Aperture photometry at the positions of already detected objects using PyRAF DAOPHOT.

    This function provides similar behavior to :func:`stdpipe.photometry.measure_objects`
    but uses IRAF DAOPHOT tasks for photometry measurements.

    It will estimate and subtract the background unless external background estimation (`bg`)
    is provided, and use user-provided noise map (`err`) if requested.

    If the `mask` is provided, it will set 0x200 bit in object `flags` if at least one of
    aperture pixels is masked.

    The results may optionally be filtered to drop the detections with low signal to noise
    ratio if `sn` parameter is set and positive. It will also filter out the events with
    negative flux.

    **Note:** Requires PyRAF and IRAF to be installed and properly configured.

    :param obj: astropy.table.Table with initial object detections to be measured. Must have 'x' and 'y' columns.
    :param image: Input image as a NumPy array
    :param aper: Circular aperture radius in pixels, to be used for flux measurement
    :param bkgann: Background annulus (tuple with inner and outer radii) to be used for local background estimation. If not set, global background model is used instead.
    :param fwhm: If provided, `aper` and `bkgann` will be measured in units of this value (so they will be specified in units of FWHM)
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array with same shape as input image) instead of automatically computed one
    :param err: Image noise map as a NumPy array to be used instead of automatically computed one, optional
    :param gain: Image gain, e/ADU, used for photometry error estimation
    :param bg_size: Background grid size in pixels (used for global background estimation)
    :param sn: Minimal S/N ratio for the object to be considered good. If set, all measurements with magnitude errors exceeding 1/SN will be discarded
    :param centroid_iter: Number of centroiding iterations to run before photometry (currently not implemented for IRAF backend)
    :param keep_negative: If not set, measurements with negative fluxes will be discarded
    :param get_bg: If True, the routine will also return estimated background and background noise images
    :param _workdir: If provided, use this directory for temporary files. If None, create a temporary directory.
    :param _tmpdir: Parent directory for temporary directory creation if _workdir is None
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: The copy of original table with `flux`, `fluxerr`, `mag` and `magerr` columns replaced with the values measured in the routine. If :code:`get_bg=True`, also returns the background and background error images.

    :raises ImportError: If PyRAF is not available
    :raises RuntimeError: If DAOPHOT task fails
    """

    if not PYRAF_AVAILABLE:
        raise ImportError("PyRAF is not available. Please install PyRAF to use this module.")

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if not len(obj):
        log('No objects to measure')
        return obj

    # Check required columns
    if 'x' not in obj.colnames or 'y' not in obj.colnames:
        raise ValueError("Object table must have 'x' and 'y' columns")

    # Operate on the copy of the list
    obj = obj.copy()

    # Sanitize the image and make its copy to safely operate on it
    image1 = image.astype(np.double)
    mask0 = ~np.isfinite(image1)  # Minimal mask

    # Ensure that the mask is defined
    if mask is None:
        mask = mask0
    else:
        mask = mask.astype(bool)

    # Background estimation
    if bg is None or err is None or get_bg:
        log('Estimating global background with %dx%d mesh' % (bg_size, bg_size))
        bg_est = photutils.background.Background2D(
            image1, bg_size, mask=mask | mask0, exclude_percentile=90
        )
        bg_est_bg = bg_est.background
        bg_est_rms = bg_est.background_rms
    else:
        bg_est = None

    if bg is None:
        log(
            'Subtracting global background: median %.1f rms %.2f' % (
                np.median(bg_est_bg), np.std(bg_est_bg)
            )
        )
        image1 = image1 - bg_est_bg
        bg_used = bg_est_bg
    else:
        log(
            'Subtracting user-provided background: median %.1f rms %.2f' % (
                np.median(bg), np.std(bg)
            )
        )
        image1 = image1 - bg
        bg_used = bg

    image1[mask0] = 0

    if err is None:
        if bg_est is not None:
            err = bg_est_rms
            log(
                'Using global background noise map: median %.1f rms %.2f' % (
                    np.median(err), np.std(err)
                )
            )
        else:
            # Estimate simple RMS if no background estimation was done
            err = np.ones_like(image1) * np.nanstd(image1)
            log('Using simple RMS estimate: %.2f' % np.nanstd(image1))
    else:
        log(
            'Using user-provided noise map: median %.1f rms %.2f' % (
                np.median(err), np.std(err)
            )
        )

    # Scale apertures by FWHM if requested
    if fwhm is not None and fwhm > 0:
        log('Scaling aperture radii with FWHM %.1f pix' % fwhm)
        aper_scaled = aper * fwhm
        if bkgann is not None:
            bkgann_scaled = [_ * fwhm for _ in bkgann]
        else:
            bkgann_scaled = None
    else:
        aper_scaled = aper
        bkgann_scaled = bkgann

    log('Using aperture radius %.1f pixels' % aper_scaled)

    if centroid_iter > 0:
        log('Warning: centroid_iter is not implemented for IRAF backend, ignoring')

    # Create or use working directory
    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='iraf', dir=_tmpdir)
    remove_workdir = _workdir is None  # Only remove if we created it

    try:
        # Write image to temporary FITS file
        image_file = os.path.join(workdir, 'image.fits')
        fits.writeto(image_file, image1.astype(np.float32), overwrite=True)

        # Write coordinate file
        coord_file = os.path.join(workdir, 'coords.txt')
        # IRAF uses 1-indexed coordinates
        with open(coord_file, 'w') as f:
            for i, row in enumerate(obj):
                f.write('%.3f %.3f\n' % (row['x'] + 1, row['y'] + 1))

        # Output file for photometry results
        phot_file = os.path.join(workdir, 'phot.mag')

        # Get iraf module (lazy import)
        iraf = _get_iraf()

        # Set up DAOPHOT parameters
        iraf.noao(_doprint=0)
        iraf.digiphot(_doprint=0)
        iraf.daophot(_doprint=0)

        # Configure photometry parameters
        iraf.datapars.fwhmpsf = fwhm if fwhm is not None else 3.0
        iraf.datapars.sigma = np.median(err) if err is not None else 1.0
        iraf.datapars.datamin = 'INDEF'
        iraf.datapars.datamax = 'INDEF'
        iraf.datapars.gain = gain if gain is not None else 1.0
        iraf.datapars.readnoise = 0.0  # Already included in error map

        # Set aperture parameters
        iraf.photpars.apertures = aper_scaled
        iraf.photpars.zmag = 25.0  # Arbitrary zero point

        # Set sky fitting parameters
        if bkgann_scaled is not None:
            log(
                'Using local background annulus between %.1f and %.1f pixels' % (
                    bkgann_scaled[0], bkgann_scaled[1]
                )
            )
            iraf.fitskypars.annulus = bkgann_scaled[0]
            iraf.fitskypars.dannulus = bkgann_scaled[1] - bkgann_scaled[0]
            iraf.fitskypars.salgorithm = 'mode'
        else:
            # Use minimal annulus if no local background requested
            # This effectively uses already subtracted background
            iraf.fitskypars.annulus = aper_scaled + 1
            iraf.fitskypars.dannulus = 1
            iraf.fitskypars.salgorithm = 'constant'
            iraf.fitskypars.skyvalue = 0.0

        # Run DAOPHOT phot task
        log('Running IRAF DAOPHOT phot task')
        iraf.phot(
            image=image_file,
            coords=coord_file,
            output=phot_file,
            interactive='no',
            verify='no',
            verbose='no'
        )

        # Parse DAOPHOT output
        log('Parsing DAOPHOT output')
        flux_list = []
        fluxerr_list = []
        mag_list = []
        magerr_list = []
        sky_list = []

        # Read the output file using IRAF txdump
        txdump_file = os.path.join(workdir, 'txdump.txt')
        iraf.txdump(
            textfiles=phot_file,
            fields='FLUX,MERR,MAG,MSKY',
            expr='yes',
            Stdout=txdump_file
        )

        # Parse txdump output
        with open(txdump_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    flux = float(parts[0]) if parts[0] != 'INDEF' else np.nan
                    magerr = float(parts[1]) if parts[1] != 'INDEF' else np.nan
                    mag = float(parts[2]) if parts[2] != 'INDEF' else np.nan
                    sky = float(parts[3]) if parts[3] != 'INDEF' else np.nan

                    flux_list.append(flux)
                    mag_list.append(mag)
                    magerr_list.append(magerr)
                    sky_list.append(sky)

                    # Calculate flux error from magnitude error
                    if np.isfinite(flux) and np.isfinite(magerr) and flux > 0:
                        fluxerr = flux * magerr * np.log(10) / 2.5
                    else:
                        fluxerr = np.nan
                    fluxerr_list.append(fluxerr)

        # Verify we got results for all objects
        if len(flux_list) != len(obj):
            log(
                'Warning: DAOPHOT returned %d results for %d objects' % (
                    len(flux_list), len(obj)
                )
            )
            # Pad with NaNs if needed
            while len(flux_list) < len(obj):
                flux_list.append(np.nan)
                fluxerr_list.append(np.nan)
                mag_list.append(np.nan)
                magerr_list.append(np.nan)
                sky_list.append(np.nan)

        # Store results in the object table
        obj['flux'] = np.array(flux_list)
        obj['fluxerr'] = np.array(fluxerr_list)
        obj['mag'] = np.array(mag_list)
        obj['magerr'] = np.array(magerr_list)

        if 'flags' not in obj.keys():
            obj['flags'] = 0

        # Flag objects with masked pixels in aperture
        # FIXME: This is a simplified check - just check if object is near masked pixels
        for i, row in enumerate(obj):
            x, y = int(row['x']), int(row['y'])
            # Check pixels in aperture
            y_grid, x_grid = np.ogrid[
                max(0, y - int(aper_scaled) - 1):min(image.shape[0], y + int(aper_scaled) + 2),
                max(0, x - int(aper_scaled) - 1):min(image.shape[1], x + int(aper_scaled) + 2)
            ]
            r = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            if np.any(mask[y_grid, x_grid][r <= aper_scaled]):
                obj['flags'][i] |= 0x200

        # Flag objects with zero flux returned - these are truncated?..
        obj['flags'][obj['flux']==0] &= 0x010 # SExtractor flag for truncated aperture

        # Store local background if available
        if bkgann_scaled is not None:
            obj['bg_local'] = np.array(sky_list)

        # Final filtering of properly measured objects
        if sn is not None and sn > 0:
            log('Filtering out measurements with S/N < %.1f' % sn)
            idx = np.isfinite(obj['magerr'])
            idx[idx] &= obj['magerr'][idx] < 1 / sn
            obj = obj[idx]

        if not keep_negative:
            log('Filtering out measurements with negative fluxes')
            idx = obj['flux'] > 0
            obj = obj[idx]

    finally:
        # Clean up temporary directory only if we created it
        if remove_workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)

    if get_bg:
        return obj, bg_used, err
    else:
        return obj


def measure_objects_psf(
    obj,
    image,
    fwhm=None,
    psf_stars=None,
    n_psf_stars=20,
    mask=None,
    bg=None,
    err=None,
    gain=None,
    bg_size=64,
    sn=None,
    psfrad=None,
    fitrad=None,
    psf_function='auto',
    psf_varorder=0,
    keep_negative=True,
    get_bg=False,
    _workdir=None,
    _tmpdir=None,
    verbose=False,
):
    """PSF photometry using IRAF DAOPHOT (psf + allstar tasks).

    This function performs PSF photometry using the full DAOPHOT workflow:
    1. Select PSF stars (or use provided psf_stars)
    2. Build PSF model using DAOPHOT psf task
    3. Fit all sources using DAOPHOT allstar task
    4. Return results in stdpipe format with quality metrics

    The PSF model is built automatically from bright, isolated stars in the image.
    If psf_stars is provided, those specific stars will be used instead of automatic selection.

    **Note:** Requires PyRAF and IRAF to be installed and properly configured.

    :param obj: astropy.table.Table with initial object detections. Must have 'x' and 'y' columns. Should also have 'flux' or 'mag' for PSF star selection.
    :param image: Input image as a NumPy array
    :param fwhm: FWHM in pixels (required for PSF building and fitting)
    :param psf_stars: Indices of objects to use as PSF stars. If None, stars will be selected automatically.
    :param n_psf_stars: Number of PSF stars to select automatically (default: 20)
    :param mask: Image mask as a boolean array (True values will be masked), optional
    :param bg: If provided, use this background (NumPy array) instead of automatically computed one
    :param err: Image noise map as a NumPy array, optional
    :param gain: Image gain, e/ADU
    :param bg_size: Background grid size in pixels (default: 64)
    :param sn: Minimal S/N ratio for output filtering
    :param psfrad: PSF radius in pixels (default: 3*fwhm). Sets how far the PSF extends.
    :param fitrad: Fitting radius in pixels (default: fwhm). Sets how many pixels to use for fitting each star.
    :param psf_function: PSF function type: 'auto', 'gauss', 'moffat15', 'moffat25', 'lorentz', 'penny1', 'penny2' (default: 'auto')
    :param psf_varorder: PSF variability order: 0=constant, 1=linear, 2=quadratic (default: 0)
    :param keep_negative: If False, discard measurements with negative fluxes
    :param get_bg: If True, also return background and error images
    :param _workdir: Working directory for temporary files
    :param _tmpdir: Parent directory for temporary directory creation
    :param verbose: Verbose output
    :returns: Table with PSF photometry results including flux, mag, x_psf, y_psf, qfit_psf, cfit_psf, flags_psf columns. If get_bg=True, also returns background and error images.

    :raises ImportError: If PyRAF is not available
    :raises ValueError: If required parameters are missing
    :raises RuntimeError: If DAOPHOT tasks fail
    """

    if not PYRAF_AVAILABLE:
        raise ImportError("PyRAF is not available. Please install PyRAF to use this module.")

    # Simple wrapper around print for logging
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    if not len(obj):
        log('No objects to measure')
        return obj

    # Check required parameters
    if fwhm is None or fwhm <= 0:
        raise ValueError("fwhm parameter is required and must be positive for PSF photometry")

    # Check required columns
    if 'x' not in obj.colnames or 'y' not in obj.colnames:
        raise ValueError("Object table must have 'x' and 'y' columns")

    # Operate on a copy
    obj = obj.copy()

    # Sanitize image
    image1 = image.astype(np.double)
    mask0 = ~np.isfinite(image1)

    # Ensure mask is defined
    if mask is None:
        mask = mask0
    else:
        mask = mask.astype(bool)

    # Background estimation
    if bg is None or err is None or get_bg:
        log('Estimating global background with %dx%d mesh' % (bg_size, bg_size))
        bg_est = photutils.background.Background2D(
            image1, bg_size, mask=mask | mask0, exclude_percentile=90
        )
        bg_est_bg = bg_est.background
        bg_est_rms = bg_est.background_rms
    else:
        bg_est = None

    if bg is None:
        log(
            'Subtracting global background: median %.1f rms %.2f' % (
                np.median(bg_est_bg), np.std(bg_est_bg)
            )
        )
        image1 = image1 - bg_est_bg
        bg_used = bg_est_bg
    else:
        log(
            'Subtracting user-provided background: median %.1f rms %.2f' % (
                np.median(bg), np.std(bg)
            )
        )
        image1 = image1 - bg
        bg_used = bg

    image1[mask0] = 0

    if err is None:
        if bg_est is not None:
            err = bg_est_rms
            log(
                'Using global background noise map: median %.1f rms %.2f' % (
                    np.median(err), np.std(err)
                )
            )
        else:
            err = np.ones_like(image1) * np.nanstd(image1)
            log('Using simple RMS estimate: %.2f' % np.nanstd(image1))
    else:
        log(
            'Using user-provided noise map: median %.1f rms %.2f' % (
                np.median(err), np.std(err)
            )
        )

    # Set default PSF and fitting radii
    if psfrad is None:
        psfrad = 3 * fwhm
    if fitrad is None:
        fitrad = fwhm

    log(f'Using PSF radius {psfrad:.1f} pixels, fitting radius {fitrad:.1f} pixels')

    # Select PSF stars if not provided
    if psf_stars is None:
        # Adjust requested number if we have fewer objects
        n_stars_to_select = min(n_psf_stars, max(2, len(obj) // 2))
        log(f'Automatically selecting up to {n_stars_to_select} PSF stars from {len(obj)} objects')
        psf_star_idx = _select_psf_stars(obj, fwhm, n_stars=n_stars_to_select, verbose=verbose)
        if len(psf_star_idx) == 0:
            raise RuntimeError("No suitable PSF stars found. Cannot build PSF model.")
        log(f'Selected {len(psf_star_idx)} PSF stars')
    else:
        psf_star_idx = np.asarray(psf_stars, dtype=int)
        log(f'Using {len(psf_star_idx)} user-provided PSF stars')

    # Create working directory
    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='iraf_psf', dir=_tmpdir)
    remove_workdir = _workdir is None

    try:
        # Write background-subtracted image
        image_file = os.path.join(workdir, 'image.fits')
        fits.writeto(image_file, image1.astype(np.float32), overwrite=True)

        # Write coordinate file for all objects (1-indexed for IRAF)
        coord_file = os.path.join(workdir, 'coords.txt')
        with open(coord_file, 'w') as f:
            for i, row in enumerate(obj):
                f.write('%.3f %.3f\n' % (row['x'] + 1, row['y'] + 1))

        # Get iraf module
        iraf = _get_iraf()

        # Set up DAOPHOT
        iraf.noao(_doprint=0)
        iraf.digiphot(_doprint=0)
        iraf.daophot(_doprint=0)

        # Configure data parameters
        iraf.datapars.fwhmpsf = fwhm
        iraf.datapars.sigma = np.median(err) if err is not None else 1.0
        iraf.datapars.datamin = 'INDEF'
        iraf.datapars.datamax = 'INDEF'
        iraf.datapars.epadu = gain if gain is not None else 1.0
        iraf.datapars.readnoise = 0.0

        # Configure photometry parameters (needed for initial phot)
        iraf.photpars.apertures = fitrad
        iraf.photpars.zmag = 25.0

        # Configure sky parameters (use constant zero - already subtracted)
        iraf.fitskypars.annulus = psfrad + 1.0
        iraf.fitskypars.dannulus = 1.0
        iraf.fitskypars.salgorithm = 'constant'
        iraf.fitskypars.skyvalue = 0.0

        # Step 1: Run phot on all stars to get initial magnitudes
        log('Running DAOPHOT phot task on all objects')
        phot_file = os.path.join(workdir, 'all_stars.mag')
        iraf.phot(
            image=image_file,
            coords=coord_file,
            output=phot_file,
            interactive='no',
            verify='no',
            verbose='no'
        )

        # Step 2: Create PSF star list file using pselect
        log(f'Creating PSF star list with {len(psf_star_idx)} stars')
        pst_file = os.path.join(workdir, 'psf_stars.pst')

        # Build expression to select PSF stars by ID (1-indexed in IRAF)
        # IDs in DAOPHOT are 1-indexed, so we add 1 to our 0-indexed array
        psf_star_ids = [str(idx + 1) for idx in psf_star_idx]
        expr = ' || '.join([f'ID=={id}' for id in psf_star_ids])

        log(f'Selecting PSF stars with IDs: {", ".join(psf_star_ids)}')

        # Use pselect to extract PSF stars from phot output
        iraf.pselect(
            infiles=phot_file,
            outfiles=pst_file,
            expr=expr
        )

        # Step 3: Build PSF model using psf task
        log(f'Building PSF model with function={psf_function}, varorder={psf_varorder}')
        psf_image = os.path.join(workdir, 'psf')  # IRAF will add .fits extension
        pst_out_file = os.path.join(workdir, 'psf_stars_used.pst')
        grp_file = os.path.join(workdir, 'groups.grp')

        # Configure PSF parameters
        iraf.daopars.psfrad = psfrad
        iraf.daopars.fitrad = fitrad
        iraf.daopars.function = psf_function
        iraf.daopars.varorder = psf_varorder

        iraf.psf(
            image=image_file,
            photfile=phot_file,
            pstfile=pst_file,
            psfimage=psf_image,
            opstfile=pst_out_file,
            groupfile=grp_file,
            interactive='no',
            verify='no',
            verbose='no'
        )

        log('PSF model built successfully')

        # Step 4: Run allstar to fit PSF to all objects
        log('Running DAOPHOT allstar task')
        allstar_file = os.path.join(workdir, 'output.als')
        reject_file = os.path.join(workdir, 'rejected.als')
        subimage_file = os.path.join(workdir, 'subtracted.fits')

        # Configure allstar parameters
        iraf.daopars.recenter = 'yes'
        iraf.daopars.fitsky = 'no'  # Sky already subtracted

        iraf.allstar(
            image=image_file,
            photfile=phot_file,
            psfimage=psf_image,
            allstarfile=allstar_file,
            rejfile=reject_file,
            subimage=subimage_file,
            verify='no',
            verbose='no'
        )

        log('PSF fitting completed, parsing results')

        # Step 5: Parse allstar output
        # .als format: ID X Y MAG MERR MSKY NITER SHARPNESS CHI PIER
        txdump_als_file = os.path.join(workdir, 'txdump_als.txt')
        iraf.txdump(
            textfiles=allstar_file,
            fields='ID,XCEN,YCEN,MAG,MERR,SHARPNESS,CHI,PIER',
            expr='yes',
            Stdout=txdump_als_file
        )

        # Store results in object table
        obj['flux'] = np.nan
        obj['fluxerr'] = np.nan
        obj['mag'] = np.nan
        obj['magerr'] = np.nan
        obj['x_psf'] = np.nan
        obj['y_psf'] = np.nan
        obj['qfit_psf'] = np.nan
        obj['cfit_psf'] = np.nan
        obj['flags_psf'] = 0

        # Parse txdump output
        with open(txdump_als_file, 'r') as f:
            txdump_lines = f.readlines()

        log(f'Allstar returned {len(txdump_lines)} fitted objects')

        for line in txdump_lines:
            parts = line.strip().split()
            if len(parts) >= 8:
                obj_id = int(parts[0])
                x_fit = float(parts[1]) - 1 if parts[1] != 'INDEF' else np.nan  # Convert to 0-indexed
                y_fit = float(parts[2]) - 1 if parts[2] != 'INDEF' else np.nan
                mag = float(parts[3]) if parts[3] != 'INDEF' else np.nan
                magerr = float(parts[4]) if parts[4] != 'INDEF' else np.nan
                sharpness = float(parts[5]) if parts[5] != 'INDEF' else np.nan
                chi = float(parts[6]) if parts[6] != 'INDEF' else np.nan
                pier = int(parts[7]) if parts[7] != 'INDEF' else 0

                # Convert mag to flux (using ZP=25.0)
                if np.isfinite(mag):
                    flux = 10**((25.0 - mag) / 2.5)
                    if np.isfinite(magerr) and magerr > 0:
                        fluxerr = flux * magerr * np.log(10) / 2.5
                    else:
                        fluxerr = np.nan
                else:
                    flux = np.nan
                    fluxerr = np.nan

                obj['flux'][obj_id-1] = flux
                obj['fluxerr'][obj_id-1] = fluxerr
                obj['mag'][obj_id-1] = mag
                obj['magerr'][obj_id-1] = magerr
                obj['x_psf'][obj_id-1] = x_fit
                obj['y_psf'][obj_id-1] = y_fit
                obj['qfit_psf'][obj_id-1] = chi
                obj['cfit_psf'][obj_id-1] = sharpness
                obj['flags_psf'][obj_id-1] = pier

        # Initialize flags column if not present
        if 'flags' not in obj.colnames:
            obj['flags'] = 0

        # Flag objects with non-zero pier flags
        obj['flags'][obj['flags_psf'] != 0] |= 0x100
        obj['flags'][~np.isfinite(obj['flux'])] |= 0x100

        # Apply filters
        if sn is not None and sn > 0:
            log(f'Filtering out measurements with S/N < {sn:.1f}')
            idx = np.isfinite(obj['magerr'])
            idx[idx] &= obj['magerr'][idx] < 1 / sn
            obj = obj[idx]

        if not keep_negative:
            log('Filtering out measurements with negative fluxes')
            idx = obj['flux'] > 0
            obj = obj[idx]

    finally:
        # Clean up
        if remove_workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)

    if get_bg:
        return obj, bg_used, err
    else:
        return obj
