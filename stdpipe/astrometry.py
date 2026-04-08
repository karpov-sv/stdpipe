import os, tempfile, shutil, shlex, re, warnings
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astropy import units as u


from scipy.spatial import KDTree

from . import utils

# Put these to common namespace
from .astrometry_quad import refine_wcs_quadhash


def get_frame_center(filename=None, header=None, wcs=None, width=None, height=None, shape=None):
    """Returns image center RA, Dec, and angular radius in degrees.

    Parameters
    ----------
    filename : str, optional
        Path to FITS file.
    header : astropy.io.fits.Header, optional
        FITS header containing WCS keywords.
    wcs : astropy.wcs.WCS, optional
        WCS object.
    width : int, optional
        Image width in pixels. Read from header if not provided.
    height : int, optional
        Image height in pixels. Read from header if not provided.
    shape : tuple of int, optional
        Image shape ``(height, width)``. Used if `width`/`height` not given.

    Returns
    -------
    ra : float or None
        Right Ascension of frame center in degrees.
    dec : float or None
        Declination of frame center in degrees.
    sr : float or None
        Angular radius from center to corner in degrees.
    """
    if not wcs:
        if header:
            wcs = WCS(header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header)

    if width is None or height is None:
        if header is not None:
            width = header['NAXIS1']
            height = header['NAXIS2']
        elif shape is not None:
            height, width = shape

    if not wcs or not wcs.is_celestial:
        return None, None, None

    ra1, dec1 = wcs.all_pix2world(0, 0, 0)
    ra0, dec0 = wcs.all_pix2world(width / 2, height / 2, 0)

    sr = spherical_distance(ra0, dec0, ra1, dec1)

    return ra0.item(), dec0.item(), sr.item()


def get_pixscale(wcs=None, filename=None, header=None):
    """Returns pixel scale of an image in degrees per pixel.

    Parameters
    ----------
    wcs : astropy.wcs.WCS, optional
        WCS object.
    filename : str, optional
        Path to FITS file.
    header : astropy.io.fits.Header, optional
        FITS header containing WCS keywords.

    Returns
    -------
    float
        Pixel scale in degrees per pixel.
    """
    if not wcs:
        if header:
            wcs = WCS(header=header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)

    return np.hypot(wcs.pixel_scale_matrix[0, 0], wcs.pixel_scale_matrix[0, 1])


def radectoxyz(ra, dec):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    xyz = np.array(
        (
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )

    return xyz


def xyztoradec(xyz):
    ra = np.arctan2(xyz[1], xyz[0])
    ra += 2 * np.pi * (ra < 0)
    dec = np.arcsin(xyz[2] / np.linalg.norm(xyz, axis=0))

    return (np.rad2deg(ra), np.rad2deg(dec))


def spherical_distance(ra1, dec1, ra2, dec2):
    """Compute spherical distance between two points or sets of points.

    Parameters
    ----------
    ra1 : float or array_like
        First point or set of points RA in degrees.
    dec1 : float or array_like
        First point or set of points Dec in degrees.
    ra2 : float or array_like
        Second point or set of points RA in degrees.
    dec2 : float or array_like
        Second point or set of points Dec in degrees.

    Returns
    -------
    float or ndarray
        Spherical distance in degrees.
    """

    x = np.sin(np.deg2rad((ra1 - ra2) / 2))
    x *= x
    y = np.sin(np.deg2rad((dec1 - dec2) / 2))
    y *= y

    z = np.cos(np.deg2rad((dec1 + dec2) / 2))
    z *= z

    return np.rad2deg(2 * np.arcsin(np.sqrt(x * (z - y) + y)))


def spherical_match(ra1, dec1, ra2, dec2, sr=1 / 3600):
    """Positional match on the sphere for two lists of coordinates.

    Aimed to be a direct replacement for :func:`esutil.htm.HTM.match` method with :code:`maxmatch=0`.

    Parameters
    ----------
    ra1 : array_like
        First set of points RA in degrees.
    dec1 : array_like
        First set of points Dec in degrees.
    ra2 : array_like
        Second set of points RA in degrees.
    dec2 : array_like
        Second set of points Dec in degrees.
    sr : float, optional
        Maximal acceptable pair distance to be considered a match, in degrees.
        Default is 1/3600 (1 arcsecond).

    Returns
    -------
    idx1 : ndarray of int
        Indices into the first list for matched pairs.
    idx2 : ndarray of int
        Indices into the second list for matched pairs.
    dist : ndarray of float
        Pairwise distances in degrees.
    """

    # Ensure that inputs are arrays, and drop units if any
    ra1 = np.array(np.atleast_1d(ra1))
    dec1 = np.array(np.atleast_1d(dec1))
    ra2 = np.array(np.atleast_1d(ra2))
    dec2 = np.array(np.atleast_1d(dec2))

    idx1, idx2, dist, _ = search_around_sky(
        SkyCoord(ra1, dec1, unit='deg'), SkyCoord(ra2, dec2, unit='deg'), sr * u.deg
    )

    dist = dist.deg  # convert to degrees

    return idx1, idx2, dist


def planar_match(x1, y1, x2, y2, sr=1):
    """Positional match on the plane for two lists of coordinates.

    Parameters
    ----------
    x1 : array_like
        First set of points X coordinates.
    y1 : array_like
        First set of points Y coordinates.
    x2 : array_like
        Second set of points X coordinates.
    y2 : array_like
        Second set of points Y coordinates.
    sr : float, optional
        Maximal acceptable pair distance to be considered a match. Default is 1.

    Returns
    -------
    idx1 : ndarray of int
        Indices into the first list for matched pairs.
    idx2 : ndarray of int
        Indices into the second list for matched pairs.
    dist : ndarray of float
        Pairwise distances.
    """

    kd = KDTree(np.array([x2, y2]).T)
    idx1, idx2 = [], []
    for i, js in enumerate(kd.query_ball_point(np.array([x1, y1]).T, sr)):
        for j in js:
            idx1.append(i)
            idx2.append(j)

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    dist = np.hypot(x1[idx1] - x2[idx2], y1[idx1] - y2[idx2])

    return idx1, idx2, dist


def get_objects_center(obj, col_ra='ra', col_dec='dec'):
    """
    Returns the center RA, Dec, and radius in degrees for a cloud of objects on the sky.
    """
    xyz = radectoxyz(obj[col_ra], obj[col_dec])
    xyz0 = np.mean(xyz, axis=1)
    ra0, dec0 = xyztoradec(xyz0)

    sr0 = np.max(spherical_distance(ra0, dec0, obj[col_ra], obj[col_dec]))

    return ra0, dec0, sr0


def blind_match_objects(
    obj,
    order=2,
    update=False,
    sn=20,
    get_header=False,
    width=None,
    height=None,
    center_ra=None,
    center_dec=None,
    radius=None,
    scale_lower=None,
    scale_upper=None,
    scale_units='arcsecperpix',
    config=None,
    extra={},
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Thin wrapper for blind plate solving using local Astrometry.Net and a list of detected objects.

    It requires `solve-field` binary from Astrometry.Net and some index files to be locally available.

    Parameters
    ----------
    obj : astropy.table.Table
        List of objects on the frame, must contain at least ``x``, ``y``, and ``flux`` columns.
    order : int, optional
        Order for the SIP spatial distortion polynomial.
    update : bool, optional
        If True, the object list will be updated in-place with correct ``ra`` and ``dec`` sky coordinates.
    sn : float, optional
        If provided, only objects with S/N exceeding this value will be used for matching.
    get_header : bool, optional
        If True, return the FITS header object instead of WCS solution.
    width : int, optional
        Image width in pixels, used for guessing the frame center.
    height : int, optional
        Image height in pixels, used for guessing the frame center.
    center_ra : float, optional
        Approximate center RA of the field in degrees.
    center_dec : float, optional
        Approximate center Dec of the field in degrees.
    radius : float, optional
        Search radius in degrees around the specified center.
    scale_lower : float, optional
        Lower limit for the pixel scale.
    scale_upper : float, optional
        Upper limit for the pixel scale.
    scale_units : str, optional
        Units for ``scale_lower``/``scale_upper``. One of ``arcsecperpix``,
        ``arcminwidth``, or ``degwidth``.
    config : str, optional
        Path to config file for ``solve-field``.
    extra : dict, optional
        Additional parameters to pass to ``solve-field`` binary.
    _workdir : str, optional
        If specified, all temporary files will be kept in this directory after the run.
    _tmpdir : str, optional
        If specified, temporary files will be created inside this path.
    _exe : str, optional
        Full path to ``solve-field`` executable. Auto-detected from :envvar:`PATH` if not provided.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.wcs.WCS or astropy.io.fits.Header or None
        Astrometric solution, or FITS header if ``get_header=True``, or None on failure.
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
        # Find it in standard paths - does it even go there on any system?..
        binname = shutil.which('solve-field')

        if binname is None:
            for path in [
                '.',
                '/usr/bin',
                '/usr/local/bin',
                '/opt/local/bin',
                '/usr/astrometry/bin',
                '/usr/local/astrometry/bin',
                '/opt/local/astrometry/bin',
            ]:
                for exe in ['solve-field']:
                    if os.path.isfile(os.path.join(path, exe)):
                        binname = os.path.join(path, exe)
                        break

    if binname is None:
        log("Can't find Astrometry.Net binary")
        return None
    # else:
    #     log("Using Astrometry.Net binary at", binname)

    # Sort objects according to decreasing flux
    aidx = np.argsort(-obj['flux'])

    # Filter out least-significant detections, if SN limit is specified
    if sn is not None and sn > 0:
        aidx = [_ for _ in aidx if obj['flux'][_] / obj['fluxerr'][_] > sn]

    if width is None:
        width = int(np.max(obj['x']))
    if height is None:
        height = int(np.max(obj['y']))

    workdir = (
        _workdir if _workdir is not None else tempfile.mkdtemp(prefix='astrometry', dir=_tmpdir)
    )

    columns = [
        fits.Column(name='XIMAGE', format='1D', array=obj['x'][aidx] + 1),
        fits.Column(name='YIMAGE', format='1D', array=obj['y'][aidx] + 1),
        fits.Column(name='FLUX', format='1D', array=obj['flux'][aidx]),
    ]
    tbhdu = fits.BinTableHDU.from_columns(columns)
    objname = os.path.join(workdir, 'list.fits')
    tbhdu.writeto(objname, overwrite=True)

    tmpname = os.path.join(workdir, 'list.tmp')
    wcsname = os.path.join(workdir, 'list.wcs')

    opts = {
        'x-column': 'XIMAGE',
        'y-column': 'YIMAGE',
        'sort-column': 'FLUX',
        'width': width,
        'height': height,
        'crpix-center': True,
        #
        'overwrite': True,
        'no-plots': True,
        'dir': workdir,
        'verbose': True if verbose else False,
    }

    if config is not None:
        opts['config'] = config

    if order is not None and order > 0:
        opts['tweak-order'] = order
    else:
        opts['no-tweak'] = True

    if scale_lower is not None:
        opts['scale-low'] = scale_lower
    if scale_upper is not None:
        opts['scale-high'] = scale_upper
    if scale_units is not None:
        opts['scale-units'] = scale_units

    if center_ra is not None:
        opts['ra'] = center_ra
    if center_dec is not None:
        opts['dec'] = center_dec
    if radius is not None:
        opts['radius'] = radius

    opts.update(extra)

    wcs = None

    for iter in range(2):
        if os.path.exists(wcsname):
            shutil.move(wcsname, tmpname)
            opts['verify'] = tmpname

        # Build the command line
        command = binname + ' ' + shlex.quote(objname) + ' ' + utils.format_long_opts(opts)
        if not verbose:
            command += ' > /dev/null 2>/dev/null'
        log('Will run iteration %d of Astrometry.Net like that:' % (iter,))
        log(command)

        res = os.system(command)

        if res == 0 and os.path.exists(wcsname):
            log('Successfully run iteration %d' % (iter,))

        else:
            log('Error %s running Astrometry.Net' % res)

    if res == 0 and os.path.exists(wcsname):
        header = fits.getheader(wcsname)
        wcs = WCS(header)

        ra0, dec0, sr0 = get_frame_center(wcs=wcs, width=width, height=height)
        pixscale = get_pixscale(wcs=wcs)

        log(
            'Got WCS solution with center at %.4f %.4f radius %.2f deg and pixel scale %.2f arcsec/pix'
            % (ra0, dec0, sr0, pixscale * 3600)
        )

        if update and wcs:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

    if _workdir is None:
        shutil.rmtree(workdir)

    return wcs


def blind_match_astrometrynet(
    obj,
    order=2,
    update=False,
    sn=20,
    get_header=False,
    width=None,
    height=None,
    solve_timeout=600,
    api_key=None,
    center_ra=None,
    center_dec=None,
    radius=None,
    scale_lower=None,
    scale_upper=None,
    scale_units='arcsecperpix',
    **kwargs,
):
    """Thin wrapper for remote plate solving using Astrometry.Net and a list of detected objects.

    Most parameters are passed directly to
    ``astroquery.astrometrynet.AstrometryNet.solve_from_source_list``.
    API key may be provided as an argument or set in ``~/.astropy/config/astroquery.cfg``.

    Parameters
    ----------
    obj : astropy.table.Table
        List of objects on the frame, must contain at least ``x``, ``y``, and ``flux`` columns.
    order : int, optional
        Order for the SIP spatial distortion polynomial.
    update : bool, optional
        If True, the object list will be updated in-place with correct ``ra`` and ``dec`` sky coordinates.
    sn : float, optional
        If provided, only objects with S/N exceeding this value will be used for matching.
    get_header : bool, optional
        If True, return the FITS header object instead of WCS solution.
    width : int, optional
        Image width in pixels, used for guessing the frame center.
    height : int, optional
        Image height in pixels, used for guessing the frame center.
    solve_timeout : float, optional
        Timeout in seconds to wait for the solution from the remote server.
    api_key : str, optional
        Astrometry.Net API key. If not set, must be configured in
        ``~/.astropy/config/astroquery.cfg``.
    center_ra : float, optional
        Approximate center RA of the field in degrees.
    center_dec : float, optional
        Approximate center Dec of the field in degrees.
    radius : float, optional
        Search radius in degrees around the specified center.
    scale_lower : float, optional
        Lower limit for the pixel scale.
    scale_upper : float, optional
        Upper limit for the pixel scale.
    scale_units : str, optional
        Units for ``scale_lower``/``scale_upper``. One of ``arcsecperpix``,
        ``arcminwidth``, or ``degwidth``.

    Returns
    -------
    astropy.wcs.WCS or astropy.io.fits.Header or None
        Astrometric solution, or FITS header if ``get_header=True``, or None on failure.
    """

    # Import required module - we postpone it until now to hide warnings for API key
    from astroquery.astrometry_net import AstrometryNet

    # Sort objects according to decreasing flux
    aidx = np.argsort(-obj['flux'])

    # Filter out least-significant detections, if SN limit is specified
    if sn is not None and sn > 0:
        aidx = [_ for _ in aidx if obj['flux'][_] / obj['fluxerr'][_] > sn]

    if width is None:
        width = int(np.max(obj['x']))
    if height is None:
        height = int(np.max(obj['y']))

    an = AstrometryNet()
    if api_key is not None:
        an.api_key = api_key

    try:
        header = an.solve_from_source_list(
            obj['x'][aidx] + 1,
            obj['y'][aidx] + 1,
            width,
            height,
            center_ra=center_ra,
            center_dec=center_dec,
            radius=radius,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
            scale_units=scale_units,
            solve_timeout=solve_timeout,
            tweak_order=order,
            **kwargs,
        )
    except:
        import traceback

        traceback.print_exc()

        header = None

    if header is not None:
        wcs = WCS(header)

        if update:
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

        if get_header:
            return header
        else:
            return wcs

    return None


def refine_wcs(
    obj,
    cat,
    order=2,
    match=True,
    sr=3 / 3600,
    update=False,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    method='astropy',
    _tmpdir=None,
    verbose=False,
):
    """Refine the WCS using detected objects and a reference catalogue.

    Parameters
    ----------
    obj : astropy.table.Table
        List of detected objects with at least ``x``, ``y``, ``ra``, ``dec`` columns.
    cat : astropy.table.Table
        Reference astrometric catalogue.
    order : int, optional
        SIP polynomial order for distortion solution.
    match : bool, optional
        If True, perform nearest-neighbor matching between objects and catalogue.
        If False, assume they are already matched line by line.
    sr : float, optional
        Matching radius in degrees. Default is 3/3600 (3 arcseconds).
    update : bool, optional
        If True, update object sky coordinates in-place using the new WCS.
    cat_col_ra : str, optional
        Catalogue column name for Right Ascension.
    cat_col_dec : str, optional
        Catalogue column name for Declination.
    method : str, optional
        Fitting method. Either ``'astropy'`` or ``'astrometrynet'``.
    _tmpdir : str, optional
        If specified, temporary files will be created inside this path.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.wcs.WCS or None
        Refined WCS solution, or None on failure.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    if match:
        # Perform simple nearest-neighbor matching within given radius
        oidx, cidx, dist = spherical_match(
            obj['ra'], obj['dec'], cat[cat_col_ra], cat[cat_col_dec], sr
        )
        _obj = obj[oidx]
        _cat = cat[cidx]
    else:
        # Assume supplied objects and catalogue are already matched line by line
        _obj = obj
        _cat = cat

    wcs = None

    if method == 'astropy':
        wcs = fit_wcs_from_points(
            [_obj['x'], _obj['y']],
            SkyCoord(_cat[cat_col_ra], _cat[cat_col_dec], unit='deg'),
            sip_degree=order,
        )
    elif method == 'astrometrynet':
        binname = None

        # Rough estimate of frame dimensions
        width = np.max(obj['x'])
        height = np.max(obj['y'])

        for path in ['.', '/usr/local', '/opt/local']:
            if os.path.isfile(os.path.join(path, 'astrometry', 'bin', 'fit-wcs')):
                binname = os.path.join(path, 'astrometry', 'bin', 'fit-wcs')
                break

        if binname:
            dirname = tempfile.mkdtemp(prefix='astrometry', dir=_tmpdir)

            columns = [
                fits.Column(name='FIELD_X', format='1D', array=obj['x'] + 1),
                fits.Column(name='FIELD_Y', format='1D', array=obj['y'] + 1),
                fits.Column(name='INDEX_RA', format='1D', array=cat[cat_col_ra]),
                fits.Column(name='INDEX_DEC', format='1D', array=cat[cat_col_dec]),
            ]
            tbhdu = fits.BinTableHDU.from_columns(columns)
            filename = os.path.join(dirname, 'list.fits')
            wcsname = os.path.join(dirname, 'list.wcs')

            tbhdu.writeto(filename, overwrite=True)

            command = "%s -c %s -o %s -W %d -H %d -C -s %d" % (
                binname,
                filename,
                wcsname,
                width,
                height,
                order,
            )

            log('Running Astrometry.Net WCS fitter like that:')
            log(command)

            os.system(command)

            if os.path.isfile(wcsname):
                header = fits.getheader(wcsname)
                wcs = WCS(header)
                log('WCS fitter run succeeded')
            else:
                log('WCS fitter run failed')

            shutil.rmtree(dirname)

        else:
            log("Astrometry.Net fit-wcs binary not found")

    if wcs:
        if update:
            log('Updating object sky coordinates in-place')
            # Update the sky coordinates of objects using new wcs
            obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)
    else:
        log('WCS refinement failed')

    return wcs


def clear_wcs(
    header,
    remove_comments=False,
    remove_history=False,
    remove_underscored=False,
    copy=False,
):
    """Clear WCS-related keywords from a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header to operate on.
    remove_comments : bool, optional
        If True, also remove COMMENT keywords.
    remove_history : bool, optional
        If True, also remove HISTORY keywords.
    remove_underscored : bool, optional
        If True, also remove all keywords starting with underscore (often added by Astrometry.Net).
    copy : bool, optional
        If True, operate on a copy and leave the original unchanged.

    Returns
    -------
    astropy.io.fits.Header
        Modified FITS header.
    """
    if copy:
        header = header.copy()

    wcs_keywords = [
        'WCSAXES',
        'CRPIX1',
        'CRPIX2',
        'PC1_1',
        'PC1_2',
        'PC2_1',
        'PC2_2',
        'CDELT1',
        'CDELT2',
        'CUNIT1',
        'CUNIT2',
        'CTYPE1',
        'CTYPE2',
        'CRVAL1',
        'CRVAL2',
        'LONPOLE',
        'LATPOLE',
        'RADESYS',
        'EQUINOX',
        'B_ORDER',
        'A_ORDER',
        'BP_ORDER',
        'AP_ORDER',
        'CD1_1',
        'CD2_1',
        'CD1_2',
        'CD2_2',
        'IMAGEW',
        'IMAGEH',
    ]

    scamp_keywords = [
        'FGROUPNO',
        'ASTIRMS1',
        'ASTIRMS2',
        'ASTRRMS1',
        'ASTRRMS2',
        'ASTINST',
        'FLXSCALE',
        'MAGZEROP',
        'PHOTIRMS',
        'PHOTINST',
        'PHOTLINK',
    ]

    remove = []

    for key in header.keys():
        if key:
            is_delete = False

            if key in wcs_keywords:
                is_delete = True
            if key in scamp_keywords:
                is_delete = True
            if re.match('^(A|B|AP|BP)_\d+_\d+$', key):
                # SIP
                is_delete = True
            if re.match('^PV_?\d+_\d+$', key):
                # PV
                is_delete = True
            if key[0] == '_' and remove_underscored:
                is_delete = True
            if key == 'COMMENT' and remove_comments:
                is_delete = True
            if key == 'HISTORY' and remove_history:
                is_delete = True

            if is_delete:
                remove.append(key)

    for key in remove:
        header.remove(key, remove_all=True, ignore_missing=True)

    return header


def wcs_pv2sip(header, order=None, accuracy=1e-4):
    """Convert a WCS header from any projection to TAN-SIP representation.

    The original WCS is sampled on a dense pixel grid and SIP
    polynomial coefficients (A, B for the forward distortion and
    AP, BP for the reverse) are fitted to reproduce the same
    pixel→sky mapping via least squares.

    Works for any input projection (TPV, ZPN, ZEA, etc.).

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Input FITS header with WCS keywords.
    order : int or None
        SIP polynomial order.  If ``None`` (default), the order is
        chosen automatically (2 through 6) to achieve *accuracy*
        on the fitting grid.
    accuracy : float
        Target accuracy in arcseconds for automatic order selection
        (default 1e-4, i.e. 0.1 mas).

    Returns
    -------
    header : `~astropy.io.fits.Header`
        New header with TAN-SIP WCS (CTYPE ``RA---TAN-SIP``).
    """

    header = header.copy()

    # Parse the original WCS before modifying the header
    wcs_orig = WCS(header)

    nx = int(header.get('NAXIS1', 1024))
    ny = int(header.get('NAXIS2', 1024))

    # Ensure CD matrix is present (convert PC+CDELT → CD)
    if 'CD1_1' not in header and 'PC1_1' in header:
        cdelt = [header.get('CDELT1'), header.get('CDELT2')]
        header['CD1_1'] = header.pop('PC1_1') * cdelt[0]
        header['CD2_1'] = header.pop('PC2_1') * cdelt[0]
        header['CD1_2'] = header.pop('PC1_2') * cdelt[0]
        header['CD2_2'] = header.pop('PC2_2') * cdelt[0]

    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']

    cd = np.array(
        [
            [header['CD1_1'], header.get('CD1_2', 0)],
            [header.get('CD2_1', 0), header['CD2_2']],
        ]
    )
    cd_inv = np.linalg.inv(cd)

    # Dense pixel grid
    ngrid = int(max(50, min(200, max(nx, ny) // 10)))
    gx = np.linspace(0, nx - 1, ngrid)
    gy = np.linspace(0, ny - 1, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    xf, yf = xx.ravel(), yy.ravel()

    # Sky coordinates from the original WCS
    ra, dec = wcs_orig.all_pix2world(xf, yf, 0)
    good = np.isfinite(ra) & np.isfinite(dec)
    xf, yf, ra, dec = xf[good], yf[good], ra[good], dec[good]

    # TAN inverse projection: (RA, Dec) → (xi, eta) in degrees
    ra0 = np.radians(crval1)
    dec0 = np.radians(crval2)
    ra_r = np.radians(ra)
    dec_r = np.radians(dec)
    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    dra = ra_r - ra0
    denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * np.cos(dra)
    xi = np.degrees(cos_dec * np.sin(dra) / denom)
    eta = np.degrees((sin_dec * cos_dec0 - cos_dec * sin_dec0 * np.cos(dra)) / denom)

    # Pixel offsets from CRPIX (0-based)
    u = xf - (crpix1 - 1)
    v = yf - (crpix2 - 1)

    # Target distorted pixel coords: CD_inv · (xi, eta)
    u_target = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    v_target = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta

    # SIP corrections: A(u, v) = u_target - u, B(u, v) = v_target - v
    du = u_target - u
    dv = v_target - v

    def _sip_terms(sip_order):
        """Return list of (p, q) exponent pairs for SIP order."""
        pq = []
        for total in range(2, sip_order + 1):
            for q in range(total + 1):
                pq.append((total - q, q))
        return pq

    def _sip_basis(u, v, pq, scale):
        """Build normalized SIP polynomial basis matrix.

        Coordinates are divided by *scale* for numerical stability;
        coefficients are rescaled to raw-pixel convention afterwards.
        """
        un, vn = u / scale, v / scale
        return np.column_stack([un**p * vn**q for p, q in pq])

    def _fit_order(u, v, du, dv, sip_order, scale):
        """Fit SIP coefficients at a given order, return residual."""
        pq = _sip_terms(sip_order)
        A_mat = _sip_basis(u, v, pq, scale)
        ca_n, _, _, _ = np.linalg.lstsq(A_mat, du, rcond=None)
        cb_n, _, _, _ = np.linalg.lstsq(A_mat, dv, rcond=None)
        # Evaluate residual in normalized space (numerically stable)
        res_u = du - A_mat @ ca_n
        res_v = dv - A_mat @ cb_n
        pix_scale = np.sqrt(np.abs(np.linalg.det(cd))) * 3600
        max_res = max(np.max(np.abs(res_u)), np.max(np.abs(res_v)))
        max_res *= pix_scale  # arcsec
        # Rescale to raw-pixel convention: coeff_raw = coeff_norm / scale^(p+q)
        ca = ca_n.copy()
        cb = cb_n.copy()
        for i, (p, q) in enumerate(pq):
            s = scale ** (p + q)
            ca[i] /= s
            cb[i] /= s
        return ca, cb, pq, max_res

    # Normalization scale for numerical stability
    scale = max(np.max(np.abs(u)), np.max(np.abs(v)), 1.0)

    if order is not None:
        ca, cb, pq, max_res = _fit_order(u, v, du, dv, order, scale)
    else:
        # Auto-select order: try increasing orders, pick the best one.
        # Cap at 6 — higher orders tend to become numerically unstable
        # when the distortion is inherently non-polynomial (e.g. ZPN).
        best_res = np.inf
        best_result = None
        best_order = 2
        for try_order in range(2, 7):
            ca_t, cb_t, pq_t, max_res_t = _fit_order(u, v, du, dv, try_order, scale)
            if max_res_t < best_res:
                best_res = max_res_t
                best_result = (ca_t, cb_t, pq_t, max_res_t)
                best_order = try_order
            if max_res_t < accuracy:
                break
        ca, cb, pq, max_res = best_result
        order = best_order

    # Strip old distortion keywords
    for key in list(header.keys()):
        if key.startswith(('A_', 'B_', 'AP_', 'BP_', 'PV')):
            del header[key]

    # Write forward SIP coefficients
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    header['A_ORDER'] = order
    header['B_ORDER'] = order

    for i, (p, q) in enumerate(pq):
        if abs(ca[i]) > 1e-20:
            header['A_%d_%d' % (p, q)] = float(ca[i])
        if abs(cb[i]) > 1e-20:
            header['B_%d_%d' % (p, q)] = float(cb[i])

    # Fit reverse SIP (AP, BP): map distorted coords back to undistorted
    du_rev = u - u_target
    dv_rev = v - v_target
    pq_rev = _sip_terms(order)
    A_rev = _sip_basis(u_target, v_target, pq_rev, scale)
    cap, _, _, _ = np.linalg.lstsq(A_rev, du_rev, rcond=None)
    cbp, _, _, _ = np.linalg.lstsq(A_rev, dv_rev, rcond=None)
    for i, (p, q) in enumerate(pq_rev):
        s = scale ** (p + q)
        cap[i] /= s
        cbp[i] /= s

    header['AP_ORDER'] = order
    header['BP_ORDER'] = order
    for i, (p, q) in enumerate(pq_rev):
        if abs(cap[i]) > 1e-20:
            header['AP_%d_%d' % (p, q)] = float(cap[i])
        if abs(cbp[i]) > 1e-20:
            header['BP_%d_%d' % (p, q)] = float(cbp[i])

    return header


def _fit_tpv_coefficients(wcs_orig, header, order=5):
    """Numerically fit TPV polynomial coefficients for a non-TAN WCS.

    Samples the original WCS on a dense pixel grid, projects the sky
    coordinates through a TAN gnomonic projection, and fits TPV
    polynomial coefficients to reproduce the mapping.

    Parameters
    ----------
    wcs_orig : `~astropy.wcs.WCS`
        Original WCS (any projection, with or without SIP).
    header : `~astropy.io.fits.Header`
        Header with CRVAL/CRPIX/CD (will be modified in-place to add
        PV keywords and set CTYPE to TPV).
    order : int
        Maximum polynomial order for the TPV fit (default 5).

    Returns
    -------
    header : modified header with PV keywords and CTYPE set to TPV.
    """
    nx = int(header.get('NAXIS1', 1024))
    ny = int(header.get('NAXIS2', 1024))

    # Dense pixel grid (avoid edges by half-pixel margin)
    ngrid = int(max(50, min(200, max(nx, ny) // 10)))
    gx = np.linspace(0, nx - 1, ngrid)
    gy = np.linspace(0, ny - 1, ngrid)
    xx, yy = np.meshgrid(gx, gy)
    xf, yf = xx.ravel(), yy.ravel()

    # Sky coordinates from the original WCS
    ra, dec = wcs_orig.all_pix2world(xf, yf, 0)

    # Filter out NaN / non-finite (can happen near edges)
    good = np.isfinite(ra) & np.isfinite(dec)
    xf, yf, ra, dec = xf[good], yf[good], ra[good], dec[good]

    # CRVAL and CD matrix from header
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']

    # Build CD matrix
    if 'CD1_1' in header:
        cd = np.array(
            [
                [header['CD1_1'], header.get('CD1_2', 0)],
                [header.get('CD2_1', 0), header['CD2_2']],
            ]
        )
    else:
        pc = np.array(
            [
                [header.get('PC1_1', 1), header.get('PC1_2', 0)],
                [header.get('PC2_1', 0), header.get('PC2_2', 1)],
            ]
        )
        cdelt = np.array([header.get('CDELT1', 1), header.get('CDELT2', 1)])
        cd = pc * cdelt[:, None]

    # Intermediate pixel coords: (u, v) = CD · (x - crpix, y - crpix)
    dx = xf - (crpix1 - 1)  # 0-based pixel coords → 1-based offset
    dy = yf - (crpix2 - 1)
    u = cd[0, 0] * dx + cd[0, 1] * dy
    v = cd[1, 0] * dx + cd[1, 1] * dy

    # TAN gnomonic projection: (RA, Dec) → (xi, eta) in degrees
    ra0 = np.radians(crval1)
    dec0 = np.radians(crval2)
    ra_r = np.radians(ra)
    dec_r = np.radians(dec)

    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    dra = ra_r - ra0

    denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * np.cos(dra)
    xi = np.degrees(cos_dec * np.sin(dra) / denom)
    eta = np.degrees((sin_dec * cos_dec0 - cos_dec * sin_dec0 * np.cos(dra)) / denom)

    # Build TPV polynomial design matrix
    # TPV basis for axis 1: terms in (u, v), axis 2: terms in (v, u)
    # Following the TPV convention, excluding radial terms at
    # indices 3, 11, 23 which are degenerate with polynomial terms
    def _tpv_basis(p, q, max_order):
        """Build TPV polynomial basis.

        p is the "primary" variable (u for axis 1, v for axis 2),
        q is the "secondary" variable.
        """
        terms = []
        indices = []

        # Mapping from (total_order, sub_index) to PV index
        # Order 0: PV_0 = 1
        # Order 1: PV_1 = p, PV_2 = q
        # Order 2: PV_4 = p², PV_5 = pq, PV_6 = q²
        # Order 3: PV_7 = p³, PV_8 = p²q, PV_9 = pq², PV_10 = q³
        # (PV_3=r and PV_11=r³ are radial, skipped)
        # Order 4: PV_12..PV_16
        # Order 5: PV_17..PV_22  (PV_23=r⁵ skipped)
        # Order 6: PV_24..PV_30
        # Order 7: PV_31..PV_38  (PV_39=r⁷ skipped)

        pv_idx = 0
        for total in range(max_order + 1):
            for j in range(total + 1):
                i = total - j
                # p^i * q^j
                terms.append(p**i * q**j)
                indices.append(pv_idx)
                pv_idx += 1

            # Skip radial term after orders 1, 3, 5, 7
            if total in (1, 3, 5, 7):
                pv_idx += 1  # skip PV_3, PV_11, PV_23, PV_39

        return np.column_stack(terms), indices

    A1, idx1 = _tpv_basis(u, v, order)
    A2, idx2 = _tpv_basis(v, u, order)

    # Fit via least squares: xi = A1 @ c1, eta = A2 @ c2
    c1, _, _, _ = np.linalg.lstsq(A1, xi, rcond=None)
    c2, _, _, _ = np.linalg.lstsq(A2, eta, rcond=None)

    # Write PV keywords
    for i, pv_idx in enumerate(idx1):
        if abs(c1[i]) > 1e-16:
            header['PV1_%d' % pv_idx] = float(c1[i])
    for i, pv_idx in enumerate(idx2):
        if abs(c2[i]) > 1e-16:
            header['PV2_%d' % pv_idx] = float(c2[i])

    header['CTYPE1'] = 'RA---TPV'
    header['CTYPE2'] = 'DEC--TPV'

    return header


def wcs_sip2pv(header):
    """Convert a WCS header from SIP (or any projection) to TPV representation.

    The original WCS is sampled on a dense pixel grid and TPV polynomial
    coefficients are fitted to reproduce the same pixel→sky mapping via least
    squares. Works for any projection type (TAN-SIP, ZPN-SIP, etc.) with
    sub-milliarcsecond accuracy.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Input FITS header with WCS keywords.

    Returns
    -------
    astropy.io.fits.Header
        New header with TPV WCS (``CTYPE`` set to ``RA---TPV``).
    """

    header = header.copy()

    # Parse the original WCS before modifying the header
    wcs_orig = WCS(header)

    # Ensure CD matrix is present (convert PC+CDELT → CD)
    if 'CD1_1' not in header and 'PC1_1' in header:
        cdelt = [header.get('CDELT1'), header.get('CDELT2')]

        header['CD1_1'] = header.pop('PC1_1') * cdelt[0]
        header['CD2_1'] = header.pop('PC2_1') * cdelt[0]
        header['CD1_2'] = header.pop('PC1_2') * cdelt[0]
        header['CD2_2'] = header.pop('PC2_2') * cdelt[0]

    # Strip SIP and old PV keywords before fitting new ones
    for key in list(header.keys()):
        if key.startswith(('A_', 'B_', 'AP_', 'BP_', 'PV')):
            del header[key]

    _fit_tpv_coefficients(wcs_orig, header)

    return header


def table_to_ldac(table, header=None, writeto=None):

    primary_hdu = fits.PrimaryHDU()

    header_str = header.tostring(endcard=True)
    # FIXME: this is a quick and dirty hack to preserve final 'END     ' in the string
    # as astropy.io.fits tends to strip trailing whitespaces from string data, and it breaks at least SCAMP
    header_str += fits.Header().tostring(endcard=True)

    header_col = fits.Column(
        name='Field Header Card', format='%dA' % len(header_str), array=[header_str]
    )
    header_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([header_col]))
    header_hdu.header['EXTNAME'] = 'LDAC_IMHEAD'

    data_hdu = fits.table_to_hdu(table)
    data_hdu.header['EXTNAME'] = 'LDAC_OBJECTS'

    hdulist = fits.HDUList([primary_hdu, header_hdu, data_hdu])

    if writeto is not None:
        hdulist.writeto(writeto, overwrite=True)

    return hdulist


def refine_wcs_scamp(
    obj,
    cat=None,
    wcs=None,
    header=None,
    sr=2 / 3600,
    order=3,
    cat_col_ra='RAJ2000',
    cat_col_dec='DEJ2000',
    cat_col_ra_err='e_RAJ2000',
    cat_col_dec_err='e_DEJ2000',
    cat_col_mag='rmag',
    cat_col_mag_err='e_rmag',
    cat_mag_lim=99,
    sn=None,
    position_maxerr=None,
    posangle_maxerr=None,
    pixscale_maxerr=None,
    match_flipped=False,
    extra={},
    get_header=False,
    update=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Wrapper for running SCAMP to get a refined astrometric solution.

    Parameters
    ----------
    obj : astropy.table.Table
        List of objects on the frame, must contain at least ``x``, ``y``, and ``flux`` columns.
    cat : astropy.table.Table or str, optional
        Reference astrometric catalogue, or a SCAMP network catalogue name (e.g. ``'GAIA-DR2'``).
    wcs : astropy.wcs.WCS, optional
        Initial WCS solution.
    header : astropy.io.fits.Header, optional
        FITS header containing the initial astrometric solution.
    sr : float, optional
        Matching radius in degrees. Controls SCAMP's ``CROSSID_RADIUS``.
    order : int, optional
        Polynomial order for PV distortion solution (1 or greater).
    cat_col_ra : str, optional
        Catalogue column name for Right Ascension.
    cat_col_dec : str, optional
        Catalogue column name for Declination.
    cat_col_ra_err : str, optional
        Catalogue column name for Right Ascension error.
    cat_col_dec_err : str, optional
        Catalogue column name for Declination error.
    cat_col_mag : str, optional
        Catalogue column name for magnitude.
    cat_col_mag_err : str, optional
        Catalogue column name for magnitude error.
    cat_mag_lim : float or sequence of float, optional
        Magnitude limit(s) for catalogue stars. A single value is used as an
        upper limit; two values ``[min, max]`` define a range.
    sn : float or sequence of float, optional
        S/N threshold(s) for SCAMP (passed as ``SN_THRESHOLDS``).
    position_maxerr : float, optional
        Maximum positional uncertainty of the initial WCS in arcminutes.
        Controls the search range for field center offset during pattern matching.
        Default is SCAMP's built-in default of 1 arcmin.
    posangle_maxerr : float, optional
        Maximum position angle uncertainty in degrees.
        Default is SCAMP's built-in default of 5 degrees.
    pixscale_maxerr : float, optional
        Maximum pixel scale uncertainty as a multiplicative factor (>1.0).
        E.g. 1.2 means ±20% around the initial scale.
        Default is SCAMP's built-in default of 1.2.
    match_flipped : bool, optional
        If True, also try matching with flipped (mirrored) axes. Default is False.
    extra : dict, optional
        Additional parameters to pass to the SCAMP binary.
    get_header : bool, optional
        If True, return the FITS header object instead of WCS solution.
    update : bool, optional
        If True, update object sky coordinates in-place using the new WCS.
    _workdir : str, optional
        If specified, all temporary files will be kept in this directory after the run.
    _tmpdir : str, optional
        If specified, temporary files will be created inside this path.
    _exe : str, optional
        Full path to the SCAMP executable. Auto-detected from :envvar:`PATH` if not provided.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.wcs.WCS or astropy.io.fits.Header or None
        Refined astrometric solution, or FITS header if ``get_header=True``, or None on failure.
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
        # Find SExtractor binary in common paths
        for exe in ['scamp']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SCAMP binary")
        return None
    # else:
    #     log("Using SCAMP binary at", binname)

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='scamp', dir=_tmpdir)

    if header is None:
        # Construct minimal FITS header covering our data points
        header = fits.Header(
            {
                'NAXIS': 2,
                'NAXIS1': np.max(obj['x'] + 1),
                'NAXIS2': np.max(obj['y'] + 1),
                'BITPIX': -64,
                'EQUINOX': 2000.0,
            }
        )
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        header += wcs.to_header(relax=True)

        # Ensure the header is in TPV convention, as SCAMP does not support SIP
        if wcs.sip is not None:
            log("Converting the header from SIP to TPV convention")
            header = wcs_sip2pv(header)
    else:
        log("Can't operate without initial WCS")
        return None

    # Dummy config filename, to prevent loading from current dir
    confname = os.path.join(workdir, 'empty.conf')
    utils.file_write(confname)

    xmlname = os.path.join(workdir, 'scamp.xml')

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'SOLVE_PHOTOM': 'N',
        'CHECKPLOT_TYPE': 'NONE',
        'WRITE_XML': 'Y',
        'XML_NAME': xmlname,
        'PROJECTION_TYPE': 'TPV',
        'CROSSID_RADIUS': sr * 3600,
        'DISTORT_DEGREES': max(1, order),
        'STABILITY_TYPE': 'EXPOSURE',
        'SN_THRESHOLDS': [3.0, 30.0],
    }

    if sn is not None:
        if np.isscalar(sn):
            opts['SN_THRESHOLDS'] = [sn, 10 * sn]
        else:
            opts['SN_THRESHOLDS'] = [sn[0], sn[1]]

    # Pattern matching search range parameters
    if position_maxerr is not None:
        opts['POSITION_MAXERR'] = position_maxerr
    if posangle_maxerr is not None:
        opts['POSANGLE_MAXERR'] = posangle_maxerr
    if pixscale_maxerr is not None:
        opts['PIXSCALE_MAXERR'] = pixscale_maxerr
    if match_flipped:
        opts['MATCH_FLIPPED'] = 'Y'

    opts.update(extra)

    # Minimal LDAC table with objects
    t_obj = Table(
        data={
            'XWIN_IMAGE': obj['x'] + 1,  # SCAMP uses 1-based coordinates
            'YWIN_IMAGE': obj['y'] + 1,
            'ERRAWIN_IMAGE': obj['xerr'],
            'ERRBWIN_IMAGE': obj['yerr'],
            'FLUX_AUTO': obj['flux'],
            'FLUXERR_AUTO': obj['fluxerr'],
            'MAG_AUTO': obj['mag'],
            'MAGERR_AUTO': obj['magerr'],
            'FLAGS': obj['flags'],
        }
    )

    objname = os.path.join(workdir, 'objects.cat')
    table_to_ldac(t_obj, header, objname)

    hdrname = os.path.join(workdir, 'objects.head')
    opts['HEADER_NAME'] = hdrname
    if os.path.exists(hdrname):
        os.unlink(hdrname)

    if cat:
        if type(cat) == str:
            # Match with network catalogue by name
            opts['ASTREF_CATALOG'] = cat
            log('Using', cat, 'as a network catalogue')
        else:
            # Match with user-provided catalogue
            t_cat = Table(
                data={
                    'X_WORLD': cat[cat_col_ra],
                    'Y_WORLD': cat[cat_col_dec],
                    'ERRA_WORLD': utils.table_get(cat, cat_col_ra_err, 1 / 3600),
                    'ERRB_WORLD': utils.table_get(cat, cat_col_dec_err, 1 / 3600),
                    'MAG': utils.table_get(cat, cat_col_mag, 0),
                    'MAGERR': utils.table_get(cat, cat_col_mag_err, 0.01),
                    'OBSDATE': np.ones_like(cat[cat_col_ra]) * 2000.0,
                    'FLAGS': np.zeros_like(cat[cat_col_ra], dtype=int),
                }
            )

            # Remove masked values
            for _ in t_cat.colnames:
                if np.ma.is_masked(t_cat[_]):
                    t_cat = t_cat[~t_cat[_].mask]

            # Convert units of err columns to degrees, if any
            for _ in ['ERRA_WORLD', 'ERRB_WORLD']:
                if t_cat[_].unit and t_cat[_].unit != 'deg':
                    t_cat[_] = t_cat[_].to('deg')

            # Limit the catalogue to given magnitude range
            if cat_mag_lim is not None:
                if hasattr(cat_mag_lim, '__len__') and len(cat_mag_lim) == 2:
                    # Two elements provided, treat them as lower and upper limits
                    t_cat = t_cat[
                        (t_cat['MAG'] >= cat_mag_lim[0]) & (t_cat['MAG'] <= cat_mag_lim[1])
                    ]
                else:
                    # One element provided, treat it as upper limit
                    t_cat = t_cat[t_cat['MAG'] <= cat_mag_lim]

            catname = os.path.join(workdir, 'catalogue.cat')
            table_to_ldac(t_cat, header, catname)

            opts['ASTREF_CATALOG'] = 'FILE'
            opts['ASTREFCAT_NAME'] = catname
            log('Using user-provided local catalogue')
    else:
        log('Using default settings for network catalogue')

    # Build the command line
    command = binname + ' ' + shlex.quote(objname) + ' ' + utils.format_astromatic_opts(opts)
    if not verbose:
        command += ' > /dev/null 2>/dev/null'
    log('Will run SCAMP like that:')
    log(command)

    # Run the command!

    res = os.system(command)

    wcs = None

    if res == 0 and os.path.exists(hdrname) and os.path.exists(xmlname):
        log('SCAMP run successfully')

        diag = Table.read(xmlname, table_id=0)[0]

        # Log detailed diagnostics from SCAMP XML
        n_matches = int(diag['NDeg_Reference'])
        reduced_chi2 = float(diag['Chi2_Reference'])
        log('Reference matches: %d, reduced chi2: %.2f' % (n_matches, reduced_chi2))

        # Log matching diagnostics if available (pattern matching was performed)
        for key in ['AS_Contrast', 'XY_Contrast', 'DPixelScale', 'DPosAngle', 'Shear']:
            if key in diag.colnames:
                log('  %s: %s' % (key, diag[key]))

        # Log astrometric residuals if available
        for key in ['AstromOffset_Reference', 'AstromSigma_Reference']:
            if key in diag.colnames:
                val = diag[key]
                if hasattr(val, '__len__'):
                    log('  %s: %s arcsec' % (key, ' '.join('%.3f' % v for v in val)))
                else:
                    log('  %s: %.3f arcsec' % (key, val))

        # Validate the solution quality
        # Chi2_Reference from SCAMP is already a reduced chi-squared
        # (sum of chi2 divided by naxis * n_matches), so we check it directly
        if n_matches < 3:
            log('Too few matches (%d), fitting likely failed' % n_matches)
        elif reduced_chi2 > 100:
            log('Reduced chi2 too large (%.1f), fitting likely failed' % reduced_chi2)
        else:
            with open(hdrname, 'r') as f:
                h1 = fits.Header.fromstring(f.read().encode('ascii', 'ignore'), sep='\n')

                # Sometimes SCAMP returns TAN type solution even despite PV keywords present
                if h1['CTYPE1'] != 'RA---TPV' and 'PV1_0' in h1.keys():
                    log(
                        'Got WCS solution with CTYPE1 =',
                        h1['CTYPE1'],
                        ' and PV keywords, fixing it',
                    )
                    h1['CTYPE1'] = 'RA---TPV'
                    h1['CTYPE2'] = 'DEC--TPV'
                # .. while sometimes it does the opposite
                elif h1['CTYPE1'] == 'RA---TPV' and 'PV1_0' not in h1.keys():
                    log(
                        'Got WCS solution with CTYPE1 =',
                        h1['CTYPE1'],
                        ' and without PV keywords, fixing it',
                    )
                    h1['CTYPE1'] = 'RA---TAN'
                    h1['CTYPE2'] = 'DEC--TAN'
                    h1 = WCS(h1).to_header(relax=True)

                if get_header:
                    # FIXME: should we really return raw / unfixed header here?..
                    log('Returning raw header instead of WCS solution')
                    wcs = h1
                else:
                    wcs = WCS(h1)

                    if update:
                        obj['ra'], obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

                    log(
                        'Astrometric accuracy: %.2f" %.2f"'
                        % (h1.get('ASTRRMS1', 0) * 3600, h1.get('ASTRRMS2', 0) * 3600)
                    )

    else:
        log('Error', res, 'running SCAMP')
        wcs = None

    if _workdir is None:
        shutil.rmtree(workdir)

    return wcs


def store_wcs(filename, wcs, overwrite=True):
    """Store WCS information in an (empty) FITS file.

    Parameters
    ----------
    filename : str
        Path to the output FITS file.
    wcs : astropy.wcs.WCS
        WCS object to store.
    overwrite : bool, optional
        If True, overwrite an existing file. Default is True.
    """
    dirname = os.path.split(filename)[0]

    try:
        # For Python3, we may simply use exists_ok=True to avoid wrapping it inside try-cache
        os.makedirs(dirname)
    except:
        pass

    hdu = fits.PrimaryHDU(header=wcs.to_header(relax=True))
    hdu.writeto(filename, overwrite=overwrite)


def upscale_wcs(wcs, scale=2, will_rebin=False):
    """Return a WCS corresponding to a frame upscaled by a given factor.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Input WCS to upscale.
    scale : float, optional
        Upscaling factor (not necessarily integer). Default is 2.
    will_rebin : bool, optional
        If True, adjust CRPIX so that rebinning the result back to original
        resolution with :func:`utils.rebin_image` will not introduce a shift.

    Returns
    -------
    astropy.wcs.WCS
        WCS corresponding to the upscaled frame.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        whdr = wcs.to_header(relax=True)

        for _ in ['CRPIX1', 'CRPIX2']:
            whdr[_] = (whdr[_] - 1) * scale + 1

        for _ in ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
            whdr[_] /= scale

        if 'SIP' in whdr['CTYPE1']:
            # SIP-type distortions
            for i in range(0, whdr.get('A_ORDER', 0) + 1):
                for j in range(0, whdr.get('A_ORDER', 0) + 1):
                    if 'A_%d_%d' % (i, j) in whdr:
                        whdr['A_%d_%d' % (i, j)] /= scale ** (i + j - 1)

            for i in range(0, whdr.get('B_ORDER', 0) + 1):
                for j in range(0, whdr.get('B_ORDER', 0) + 1):
                    if 'B_%d_%d' % (i, j) in whdr:
                        whdr['B_%d_%d' % (i, j)] /= scale ** (i + j - 1)

            for i in range(0, whdr.get('AP_ORDER', 0) + 1):
                for j in range(0, whdr.get('AP_ORDER', 0) + 1):
                    if 'AP_%d_%d' % (i, j) in whdr:
                        whdr['AP_%d_%d' % (i, j)] /= scale ** (i + j - 1)

            for i in range(0, whdr.get('BP_ORDER', 0) + 1):
                for j in range(0, whdr.get('BP_ORDER', 0) + 1):
                    if 'BP_%d_%d' % (i, j) in whdr:
                        whdr['BP_%d_%d' % (i, j)] /= scale ** (i + j - 1)

        new = WCS(whdr)

        if will_rebin:
            # Switch from conserving pixel center position to pixel corner, so
            # that naive downscaling back will give the same image as original
            new.wcs.crpix -= -0.5 * scale + 0.5

    return new
