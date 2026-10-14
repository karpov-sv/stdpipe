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


def refine_wcs_simple(
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
        Setting this to ``'auto'`` results in order 2.
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

    if order == 'auto':
        order = 2

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
    pos_err_floor=0.25,
    max_sigma=None,
    min_matches_per_param=3,
    auto_max_order=5,
    auto_threshold=0.05,
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
    order : int or str, optional
        Polynomial order for PV distortion solution (1 or greater), or ``'auto'``
        to select it based on hold-out residuals, so that higher degrees are only
        used when they actually improve the solution. Automatic selection needs a
        local reference catalogue, and costs one extra SCAMP run per degree tried.
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
    pos_err_floor : float, optional
        Floor for object positional errors, in pixels, added to them in quadrature.
        Centroiding errors are formal photon noise ones and do not include
        atmospheric and other systematic terms, so without such floor SCAMP
        over-weights the brightest stars and reports meaninglessly large chi2.
        Default is 0.25 pixels.
    max_sigma : float, optional
        Maximum acceptable rms of astrometric residuals w.r.t. the reference
        catalogue, in arcsec. Solutions with larger residuals are rejected.
        Default is a half of the matching radius ``sr``.
    min_matches_per_param : float, optional
        Minimal number of matched reference stars per free parameter of the
        distortion polynomial. If the fit has less of them, it is repeated with
        lower degree, down to ``order=1``, as such fits overfit the data and thus
        have deceivingly small residuals. Default is 3.
    auto_max_order : int, optional
        Largest distortion degree to try when ``order='auto'``. Default is 5.
    auto_threshold : float, optional
        Minimal relative improvement of hold-out residuals for accepting the next
        distortion degree when ``order='auto'``. Default is 0.05, i.e. 5 percent.
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

    opts = {
        'c': confname,
        'VERBOSE_TYPE': 'QUIET',
        'SOLVE_PHOTOM': 'N',
        'CHECKPLOT_TYPE': 'NONE',
        'WRITE_XML': 'Y',
        'PROJECTION_TYPE': 'TPV',
        'CROSSID_RADIUS': sr * 3600,
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

    # Object positional errors from centroiding are formal photon noise ones, and
    # thus routinely underestimate the actual scatter w.r.t. the reference catalogue,
    # which also includes atmospheric and other systematic terms. Thus we add a floor
    # to them in quadrature, so that SCAMP weighting and its chi2 stay sensible.
    xerr = np.hypot(np.asarray(obj['xerr'], dtype=float), pos_err_floor)
    yerr = np.hypot(np.asarray(obj['yerr'], dtype=float), pos_err_floor)

    # Minimal LDAC table with objects
    t_obj = Table(
        data={
            'XWIN_IMAGE': obj['x'] + 1,  # SCAMP uses 1-based coordinates
            'YWIN_IMAGE': obj['y'] + 1,
            'ERRAWIN_IMAGE': xerr,
            'ERRBWIN_IMAGE': yerr,
            'FLUX_AUTO': obj['flux'],
            'FLUXERR_AUTO': obj['fluxerr'],
            'MAG_AUTO': obj['mag'],
            'MAGERR_AUTO': obj['magerr'],
            'FLAGS': obj['flags'],
        }
    )

    objname = os.path.join(workdir, 'objects.cat')
    table_to_ldac(t_obj, header, objname)

    def write_catalogue(catalogue, catname):
        """Store the reference catalogue as an LDAC file for SCAMP to use"""
        t_cat = Table(
            data={
                'X_WORLD': catalogue[cat_col_ra],
                'Y_WORLD': catalogue[cat_col_dec],
                'ERRA_WORLD': utils.table_get(catalogue, cat_col_ra_err, 1 / 3600),
                'ERRB_WORLD': utils.table_get(catalogue, cat_col_dec_err, 1 / 3600),
                'MAG': utils.table_get(catalogue, cat_col_mag, 0),
                'MAGERR': utils.table_get(catalogue, cat_col_mag_err, 0.01),
                'OBSDATE': np.ones_like(catalogue[cat_col_ra]) * 2000.0,
                'FLAGS': np.zeros_like(catalogue[cat_col_ra], dtype=int),
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

        table_to_ldac(t_cat, header, catname)

        return catname

    catname = None

    if cat:
        if type(cat) == str:
            # Match with network catalogue by name
            opts['ASTREF_CATALOG'] = cat
            log('Using', cat, 'as a network catalogue')
        else:
            # Match with user-provided catalogue
            catname = write_catalogue(cat, os.path.join(workdir, 'catalogue.cat'))
            opts['ASTREF_CATALOG'] = 'FILE'
            opts['ASTREFCAT_NAME'] = catname
            log('Using user-provided local catalogue')
    else:
        log('Using default settings for network catalogue')

    def run_scamp(order, catname=None, tag='', quiet=False):
        """Run SCAMP for a given distortion degree, and return the solution as a
        FITS header along with the number of matched reference stars and the rms
        of their residuals in arcsec. Header is None if the run failed."""
        # Simple wrapper around log for quiet mode only
        rlog = (lambda *args, **kwargs: None) if quiet else log

        hdrname = os.path.join(workdir, 'objects%s.head' % tag)
        xmlname = os.path.join(workdir, 'scamp%s.xml' % tag)

        for _ in [hdrname, xmlname]:
            if os.path.exists(_):
                os.unlink(_)

        run_opts = dict(opts)
        run_opts['DISTORT_DEGREES'] = max(1, order)
        run_opts['HEADER_NAME'] = hdrname
        run_opts['XML_NAME'] = xmlname
        if catname is not None:
            run_opts['ASTREFCAT_NAME'] = catname

        # Build the command line
        command = (
            binname + ' ' + shlex.quote(objname) + ' ' + utils.format_astromatic_opts(run_opts)
        )
        if not verbose:
            command += ' > /dev/null 2>/dev/null'
        rlog('Will run SCAMP like that:')
        rlog(command)

        # Run the command!

        res = os.system(command)

        if res != 0 or not os.path.exists(hdrname) or not os.path.exists(xmlname):
            rlog('Error', res, 'running SCAMP')
            return None, 0, None

        rlog('SCAMP run successfully')

        diag = Table.read(xmlname, table_id=0)[0]

        # Log detailed diagnostics from SCAMP XML
        n_matches = int(diag['NDeg_Reference'])
        reduced_chi2 = float(diag['Chi2_Reference'])
        rlog('Reference matches: %d, reduced chi2: %.2f' % (n_matches, reduced_chi2))

        # Log matching diagnostics if available (pattern matching was performed)
        for key in ['AS_Contrast', 'XY_Contrast', 'DPixelScale', 'DPosAngle', 'Shear']:
            if key in diag.colnames:
                rlog('  %s: %s' % (key, diag[key]))

        # Log astrometric residuals if available
        sigma = None
        for key in ['AstromOffset_Reference', 'AstromSigma_Reference']:
            if key in diag.colnames:
                val = np.atleast_1d(diag[key]).astype(float)
                rlog('  %s: %s arcsec' % (key, ' '.join('%.3f' % v for v in val)))
                if key == 'AstromSigma_Reference':
                    # Combine per-axis residual rms values into a single number
                    sigma = np.sqrt(np.mean(val**2))

        with open(hdrname, 'r') as f:
            h1 = fits.Header.fromstring(f.read().encode('ascii', 'ignore'), sep='\n')

            # Sometimes SCAMP returns TAN type solution even despite PV keywords present
            if h1['CTYPE1'] != 'RA---TPV' and 'PV1_0' in h1.keys():
                rlog(
                    'Got WCS solution with CTYPE1 =',
                    h1['CTYPE1'],
                    ' and PV keywords, fixing it',
                )
                h1['CTYPE1'] = 'RA---TPV'
                h1['CTYPE2'] = 'DEC--TPV'
            # .. while sometimes it does the opposite
            elif h1['CTYPE1'] == 'RA---TPV' and 'PV1_0' not in h1.keys():
                rlog(
                    'Got WCS solution with CTYPE1 =',
                    h1['CTYPE1'],
                    ' and without PV keywords, fixing it',
                )
                h1['CTYPE1'] = 'RA---TAN'
                h1['CTYPE2'] = 'DEC--TAN'
                h1 = WCS(h1).to_header(relax=True)

        return h1, n_matches, sigma

    def min_matches_for(order):
        """Minimal number of matched reference stars for a given distortion degree.
        The fit should have noticeably more of them than it has free parameters,
        else the residuals get small just due to overfitting. The number below is
        the amount of polynomial terms of a given degree, for both axes."""
        return int(np.ceil(min_matches_per_param * (max(1, order) + 1) * (max(1, order) + 2)))

    def select_order():
        """Choose the distortion degree by fitting one half of the reference
        catalogue and measuring the residuals on the other, unused one. Unlike the
        residuals of the fit itself, these get worse as soon as it starts to
        overfit, so we may just pick the lowest degree that reaches the best of
        them. We have to check all the degrees, as the improvement is not monotonic
        - the distortion is often dominated by odd radial terms, so e.g. order 2
        may add nothing at all over order 1 while order 3 improves a lot."""
        if catname is None:
            # We can only split a local catalogue into fit and test halves
            default_order = min(3, auto_max_order)
            log(
                'Automatic order selection needs a local catalogue, using order %d'
                % default_order
            )
            return default_order

        log('Selecting the distortion degree using half of the catalogue as a test set')

        cat_fit, cat_test = cat[::2], cat[1::2]
        catname_fit = write_catalogue(cat_fit, os.path.join(workdir, 'catalogue_fit.cat'))

        # Trial fits see only a half of the catalogue, and thus get roughly half
        # the matches the final fit will have, so scale the requirement accordingly
        scale = len(cat_fit) / len(cat)

        results = {}
        pairs = None

        for o in range(1, auto_max_order + 1):
            h, n, _ = run_scamp(o, catname=catname_fit, tag='_auto', quiet=True)

            if h is None:
                break

            if n < scale * min_matches_for(o):
                log(
                    '  order %d: %d matches, less than %d needed'
                    % (o, n, int(np.ceil(scale * min_matches_for(o))))
                )
                break

            ra, dec = WCS(h).all_pix2world(obj['x'], obj['y'], 0)

            if pairs is None:
                # Fix the set of object - test star pairs using the lowest degree
                # solution, so that all the degrees are compared on the same stars
                oidx, cidx, dist = spherical_match(
                    ra, dec, cat_test[cat_col_ra], cat_test[cat_col_dec], sr=sr
                )

                if not len(dist):
                    break

                # Keep only the nearest test star for every object
                idx = np.argsort(dist)
                _, first = np.unique(oidx[idx], return_index=True)
                pairs = (oidx[idx][first], cidx[idx][first])

            oidx, cidx = pairs
            res = 3600 * np.median(
                spherical_distance(
                    ra[oidx], dec[oidx],
                    cat_test[cat_col_ra][cidx], cat_test[cat_col_dec][cidx],
                )
            )
            results[o] = res

            log('  order %d: %d matches, test residual %.3f arcsec' % (o, n, res))

        if not results:
            log('Order selection failed, using order 1')
            return 1

        # Lowest degree that is within the threshold from the best residual
        best_res = min(results.values())
        best_order = min(o for o, res in results.items() if res <= best_res * (1 + auto_threshold))

        log('Using distortion degree %d' % best_order)

        return best_order

    if order == 'auto':
        order = select_order()

    h1, n_matches, sigma = run_scamp(order, catname=catname)

    # Degrade the distortion degree if there are not enough reference stars to
    # constrain it, instead of just failing - lower order still beats no solution
    while h1 is not None and n_matches < min_matches_for(order) and order > 1:
        new_order = order - 1
        while new_order > 1 and n_matches < min_matches_for(new_order):
            new_order -= 1

        log(
            'Too few matches (%d < %d) for order %d, repeating with order %d'
            % (n_matches, min_matches_for(order), order, new_order)
        )
        order = new_order
        h1, n_matches, sigma = run_scamp(order, catname=catname)

    wcs = None

    if h1 is not None:
        # Validate the solution quality using the actual residuals w.r.t. the
        # reference catalogue. We do not check Chi2_Reference here - it is a
        # reduced chi2 computed against the quoted per-star positional errors,
        # and thus reflects how well these errors are estimated at least as much
        # as the quality of the fit itself, so it is logged for information only.
        if max_sigma is None:
            max_sigma = 0.5 * sr * 3600

        if n_matches < min_matches_for(order):
            log(
                'Too few matches (%d < %d), fitting likely failed'
                % (n_matches, min_matches_for(order))
            )
        elif sigma is not None and sigma > max_sigma:
            log(
                'Residuals too large (%.2f > %.2f arcsec), fitting likely failed'
                % (sigma, max_sigma)
            )
        elif get_header:
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


def fit_astrometric_residuals(
    x_obs, y_obs, x_pred, y_pred,
    *,
    image_shape=None,
    backend='grid',
    **smoother_kwargs,
):
    """Fit a smooth (dx, dy) correction field for astrometric residuals.

    Given matched pairs of observed and WCS-predicted pixel positions of
    the same sources, fit a smooth two-component field that maps any
    predicted position to the corresponding observed one. The returned
    callable applies the additive correction at arbitrary positions.

    Parameters
    ----------
    x_obs, y_obs : array-like, shape (N,)
        Observed pixel positions of matched sources (e.g. SEP centroids).
    x_pred, y_pred : array-like, shape (N,)
        WCS-predicted pixel positions of the same sources from a reference
        catalogue. ``x_obs - x_pred`` is the residual being modelled.
    image_shape : (H, W), optional
        Forwarded to the grid backend; ignored by LOESS.
    backend : {"grid", "loess"}
        Smoothing backend; see :func:`stdpipe.smoothing.fit_vector_field_2d`.
        Default ``"grid"`` because the typical astrometric-residual workflow
        evaluates the correction at far more positions than were used to
        fit it (e.g. forced-position photometry of large catalogues), where
        grid lookup is ~600--1000x faster than LOESS prediction.
    **smoother_kwargs :
        Forwarded to :func:`fit_vector_field_2d`. Grid backend keys:
        ``grid_shape``, ``min_per_cell``, ``smooth_sigma``. LOESS backend
        keys: ``scales``, ``k``, ``robust_iters``, ...

    Returns
    -------
    correct : callable
        ``correct(x, y) -> (x_corrected, y_corrected)``: adds the smooth
        ``(dx, dy)`` field at ``(x, y)``. Has a ``.smoother`` attribute
        holding the underlying field model for diagnostics.
    """
    from . import smoothing

    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    x_pred = np.asarray(x_pred, float)
    y_pred = np.asarray(y_pred, float)

    smoother = smoothing.fit_vector_field_2d(
        x_pred, y_pred,
        x_obs - x_pred,
        y_obs - y_pred,
        backend=backend,
        image_shape=image_shape,
        **smoother_kwargs,
    )

    def correct(x, y):
        cdx, cdy = smoother(x, y)
        return np.asarray(x, float) + cdx, np.asarray(y, float) + cdy

    correct.smoother = smoother
    return correct


def refine_positions_from_catalog(
    match, obj, cat, wcs,
    *,
    obj_col_x='x', obj_col_y='y',
    cat_col_ra='ra', cat_col_dec='dec',
    image_shape=None,
    backend='grid',
    **smoother_kwargs,
):
    """Build an astrometric residual correction from a match dict.

    Wraps :func:`fit_astrometric_residuals` with the column-extraction
    boilerplate needed when the matches come from
    :func:`stdpipe.pipeline.calibrate_photometry` or
    :func:`stdpipe.photometry.match`. Returns the correction callable
    plus a small dictionary of pre/post-correction residual statistics.

    Parameters
    ----------
    match : dict
        Cross-match dictionary with at least the keys ``idx`` (boolean
        mask of accepted matches), ``oidx`` (object indices into ``obj``),
        and ``cidx`` (catalogue indices into ``cat``).
    obj, cat : astropy.table.Table
        Object and catalogue tables matched by ``match``.
    wcs : astropy.wcs.WCS
        WCS used to project catalogue positions into pixel coordinates.
    obj_col_x, obj_col_y : str
        Column names for observed pixel positions in ``obj``.
    cat_col_ra, cat_col_dec : str
        Column names for sky positions in ``cat``.
    image_shape : (H, W), optional
        Forwarded to the smoothing backend.
    backend : {"grid", "loess"}
        Smoothing backend.
    **smoother_kwargs :
        Forwarded to :func:`fit_astrometric_residuals`.

    Returns
    -------
    correct : callable
        Same as :func:`fit_astrometric_residuals`.
    info : dict
        ``n_matched``, ``n_used``, ``raw_median_dr_pix``,
        ``corrected_median_dr_pix``, ``raw_q90_dr_pix``,
        ``corrected_q90_dr_pix``. The corrected statistics are evaluated
        at the same matched positions used for the fit (in-sample), so
        they over-state the gain; for an unbiased estimate, subset
        ``match['idx']`` into a holdout mask before calling this.
    """
    accepted = np.asarray(match['idx'], bool)
    oidx = np.asarray(match['oidx'])[accepted]
    cidx = np.asarray(match['cidx'])[accepted]

    x_obs = np.asarray(obj[obj_col_x][oidx], float)
    y_obs = np.asarray(obj[obj_col_y][oidx], float)
    x_pred, y_pred = wcs.all_world2pix(
        np.asarray(cat[cat_col_ra][cidx], float),
        np.asarray(cat[cat_col_dec][cidx], float),
        0,
    )

    keep = (
        np.isfinite(x_obs) & np.isfinite(y_obs)
        & np.isfinite(x_pred) & np.isfinite(y_pred)
    )
    x_obs = x_obs[keep]; y_obs = y_obs[keep]
    x_pred = x_pred[keep]; y_pred = y_pred[keep]

    correct = fit_astrometric_residuals(
        x_obs, y_obs, x_pred, y_pred,
        image_shape=image_shape, backend=backend, **smoother_kwargs,
    )

    raw_dx = x_obs - x_pred
    raw_dy = y_obs - y_pred
    pred_dx, pred_dy = correct.smoother(x_pred, y_pred)
    corr_dr = np.hypot(raw_dx - pred_dx, raw_dy - pred_dy)
    raw_dr = np.hypot(raw_dx, raw_dy)

    info = {
        'n_matched': int(np.sum(accepted)),
        'n_used': int(keep.sum()),
        'raw_median_dr_pix': float(np.median(raw_dr)) if raw_dr.size else float('nan'),
        'corrected_median_dr_pix': float(np.median(corr_dr)) if corr_dr.size else float('nan'),
        'raw_q90_dr_pix': float(np.percentile(raw_dr, 90)) if raw_dr.size else float('nan'),
        'corrected_q90_dr_pix': float(np.percentile(corr_dr, 90)) if corr_dr.size else float('nan'),
    }
    return correct, info
