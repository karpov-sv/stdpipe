
import os, tempfile, shutil, shlex, re, warnings
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astropy import units as u

import sip_tpv

from scipy.stats import chi2
from scipy.spatial import KDTree

from . import utils


def get_frame_center(
    filename=None, header=None, wcs=None, width=None, height=None, shape=None
):
    """
    Returns image center RA, Dec, and radius in degrees.
    Accepts either filename, or FITS header, or WCS structure
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
    '''
    Returns pixel scale of an image in degrees per pixel.
    Accepts either WCS structure, or FITS header, or filename
    '''
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
    """Spherical distance.

    :param ra1: First point or set of points RA
    :param dec1: First point or set of points Dec
    :param ra2: Second point or set of points RA
    :param dec2: Second point or set of points Dec
    :returns: Spherical distance in degrees

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

    :param ra1: First set of points RA
    :param dec1: First set of points Dec
    :param ra2: Second set of points RA
    :param dec2: Second set of points Dec
    :param sr: Maximal acceptable pair distance to be considered a match, in degrees
    :returns: Two parallel sets of indices corresponding to matches from first and second lists, along with the pairwise distances in degrees

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


def planar_match(x1, y1, x2, y2, sr = 1):
    """Positional match on the plane for two lists of coordinates.

    :param ra1: First set of points X
    :param dec1: First set of points Y
    :param ra2: Second set of points X
    :param dec2: Second set of points Y
    :param sr: Maximal acceptable pair distance to be considered a match
    :returns: Two parallel sets of indices corresponding to matches from first and second lists, along with the pairwise distances
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

    :param obj: List of objects on the frame that should contain at least `x`, `y` and `flux` columns.
    :param order: Order for the SIP spatial distortion polynomial
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param sn: If provided, only objects with signal to noise ratio exceeding this value will be used for matching.
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param width: Image width, to be used for guessing pixel coordinats of frame center. Optional.
    :param height: Image height, to be used for guessing pixel coordinats of frame center. Optional.
    :param center_ra: Approximate center RA of the field, degrees. Optional.
    :param center_dec: Approximate center Dec of the field, degrees. Optional.
    :param radius: If set, the server will look for solutions only within this radius from the center specified above.
    :param scale_lower: Optional lower limit for the solution scale.
    :param scale_upper: Optional upper limit for the solution scale.
    :param scale_units: Units of the `scale_lower`/`scale_upper` parameters. May be one of `arcsecperpix`, `arcminwidth`, or `degwidth`.
    :param config: Path to config file for `solve-field`, optional.
    :param extra: Dictionary of additional parameters to be passed to `solve-field` binary, optional.
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running `solve-field`. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to `solve-field` executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Either astrometric solution as astropy.wcs.WCS object, or FITS header if :code:`get_header=True`
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
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='astrometry', dir=_tmpdir)
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
    **kwargs
):

    """Thin wrapper for remote plate solving using Astrometry.Net and a list of detected objects.
    Most of the parameters are passed directly to `astroquery.astrometrynet.AstrometryNet.solve_from_source_list` routine.
    API key may either be provided as an argument or specified in ~/.astropy/config/astroquery.cfg

    :param obj: List of objects on the frame that should contain at least `x`, `y` and `flux` columns.
    :param order: Order for the SIP spatial distortion polynomial
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param sn: If provided, only objects with signal to noise ratio exceeding this value will be used for matching.
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param width: Image width, to be used for guessing pixel coordinats of frame center. Optional.
    :param height: Image height, to be used for guessing pixel coordinats of frame center. Optional.
    :param solve_timeout: Timeout in seconds to wait for the solution from remote server.
    :param api_key: API key, optional. If not set, it should be configured in your ~/.astropy/config/astroquery.cfg file
    :param center_ra: Approximate center RA of the field, degrees. Optional.
    :param center_dec: Approximate center Dec of the field, degrees. Optional.
    :param radius: If set, the server will look for solutions only within this radius from the center specified above.
    :param scale_lower: Optional lower limit for the solution scale.
    :param scale_upper: Optional upper limit for the solution scale.
    :param scale_units: Units of the `scale_lower`/`scale_upper` parameters. May be one of `arcsecperpix`, `arcminwidth`, or `degwidth`.
    :returns: Either astrometric solution as astropy.wcs.WCS object, or FITS header if :code:`get_header=True`
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
            **kwargs
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
    '''
    Refine the WCS using detected objects and catalogue.
    '''

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

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
    """Clears WCS related keywords from FITS header

    :param header: Header to operate on
    :param remove_comments: Whether to also remove COMMENT keywords
    :param remove_history: Whether to also remove HISTORY keywords
    :param remove_underscored: Whether to also remove all keywords starting with underscore (often made by e.g. Astrometry.Net)
    :param copy: If True, do not change original FITS header
    :returns: Modified FITS header

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


def wcs_pv2sip(header, method='astrometrynet'):
    """
    TODO
    """
    pass


def wcs_sip2pv(header):
    """
    Convert the WCS header from SIP to TPV representation
    """

    header = header.copy()

    # sip_to_pv expects CD matrix to be present
    if 'CD1_1' not in header and 'PC1_1' in header:
        cdelt = [header.get('CDELT1'), header.get('CDELT2')]

        header['CD1_1'] = header.pop('PC1_1') * cdelt[0]
        header['CD2_1'] = header.pop('PC2_1') * cdelt[0]
        header['CD1_2'] = header.pop('PC1_2') * cdelt[0]
        header['CD2_2'] = header.pop('PC2_2') * cdelt[0]

    sip_tpv.sip_to_pv(header)

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
    extra={},
    get_header=False,
    update=False,
    _workdir=None,
    _tmpdir=None,
    _exe=None,
    verbose=False,
):
    """Wrapper for running SCAMP on user-provided object list and catalogue to get refined astrometric solution.

    :param obj: List of objects on the frame that should contain at least `x`, `y` and `flux` columns.
    :param cat: Reference astrometric catalogue
    :param wcs: Initial WCS
    :param header: FITS header containing initial astrometric solution, optional.
    :param sr: Matching radius in degrees
    :param order: Polynomial order for PV distortion solution (1 or greater)
    :param cat_col_ra: Catalogue column name for Right Ascension
    :param cat_col_dec: Catalogue column name for Declination
    :param cat_col_ra_err: Catalogue column name for Right Ascension error
    :param cat_col_dec_err: Catalogue column name for Declination error
    :param cat_col_mag: Catalogue column name for the magnitude in closest band
    :param cat_col_mag_err: Catalogue column name for the magnitude error
    :param cat_mag_lim: Magnitude limit for catalogue stars
    :param sn: If provided, only objects with signal to noise ratio exceeding this value will be used for matching.
    :param extra: Dictionary of additional parameters to be passed to SCAMP binary, optional.
    :param get_header: If True, function will return the FITS header object instead of WCS solution
    :param update: If set, the object list will be updated in-place to contain correct `ra` and `dec` sky coordinates
    :param _workdir: If specified, all temporary files will be created in this directory, and will be kept intact after running SCAMP. May be used for debugging exact inputs and outputs of the executable. Optional
    :param _tmpdir: If specified, all temporary files will be created in a dedicated directory (that will be deleted after running the executable) inside this path.
    :param _exe: Full path to SCAMP executable. If not provided, the code tries to locate it automatically in your :envvar:`PATH`.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: Refined astrometric solution, or FITS header if :code:`get_header=True`
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
        for exe in ['scamp']:
            binname = shutil.which(exe)
            if binname is not None:
                break

    if binname is None:
        log("Can't find SCAMP binary")
        return None
    # else:
    #     log("Using SCAMP binary at", binname)

    workdir = (
        _workdir
        if _workdir is not None
        else tempfile.mkdtemp(prefix='scamp', dir=_tmpdir)
    )

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
    }

    if sn is not None:
        if np.isscalar(sn):
            opts['SN_THRESHOLDS'] = [sn, 10 * sn]
        else:
            opts['SN_THRESHOLDS'] = [sn[0], sn[1]]

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
                        (t_cat['MAG'] >= cat_mag_lim[0])
                        & (t_cat['MAG'] <= cat_mag_lim[1])
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
    command = (
        binname + ' ' + shlex.quote(objname) + ' ' + utils.format_astromatic_opts(opts)
    )
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
        log('%d matches, chi2 %.1f' % (diag['NDeg_Reference'], diag['Chi2_Reference']))
        # FIXME: is df correct here?..
        if (
            diag['NDeg_Reference'] < 3
            or chi2.sf(diag['Chi2_Reference'], df=diag['NDeg_Reference']) < 1e-3
        ):
            log('It seems the fitting failed')
        else:
            with open(hdrname, 'r') as f:
                h1 = fits.Header.fromstring(
                    f.read().encode('ascii', 'ignore'), sep='\n'
                )

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
    '''
    Auxiliary routine to store WCS information in an (empty) FITS file
    '''
    dirname = os.path.split(filename)[0]

    try:
        # For Python3, we may simply use exists_ok=True to avoid wrapping it inside try-cache
        os.makedirs(dirname)
    except:
        pass

    hdu = fits.PrimaryHDU(header=wcs.to_header(relax=True))
    hdu.writeto(filename, overwrite=overwrite)


def upscale_wcs(wcs, scale=2, will_rebin=False):
    """
    Returns WCS corresponding to the frame upscaled by some (not necessarily integer) factor.

    If you wish to re-bin the image back to original resolution using `utils.rebin_image`,
    you may wish to set `will_rebin` to True, and it will adjust the CRPIX so that
    the result will not be shifted.
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
