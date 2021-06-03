from __future__ import absolute_import, division, print_function, unicode_literals

import os, tempfile, shutil, shlex, re
import numpy as np

from esutil import coords, htm

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.astrometry_net import AstrometryNet

from scipy.stats import chi2

from . import utils

def get_frame_center(filename=None, header=None, wcs=None, width=None, height=None):
    """
    Returns image center RA, Dec, and radius in degrees.
    Accepts either filename, or FITS header, or WCS structure
    """
    if not wcs:
        if header:
            wcs = WCS(header=header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)

    if (not width or not height) and header:
        width = header['NAXIS1']
        height = header['NAXIS2']

    [ra1],[dec1] = wcs.all_pix2world([0], [0], 1)
    [ra0],[dec0] = wcs.all_pix2world([width/2], [height/2], 1)

    sr = coords.sphdist(ra0, dec0, ra1, dec1)[0]

    return ra0, dec0, sr

def get_pixscale(filename=None, header=None, wcs=None):
    '''
    Returns pixel scale of an image in degrees per pixel.
    Accepts either filename, or FITS header, or WCS structure
    '''
    if not wcs:
        if header:
            wcs = WCS(header=header)
        elif filename:
            header = fits.getheader(filename, -1)
            wcs = WCS(header=header)

    return np.hypot(wcs.pixel_scale_matrix[0,0], wcs.pixel_scale_matrix[0,1])

def radectoxyz(ra, dec):
    ra_rad  = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    xyz = np.array((np.cos(dec_rad)*np.cos(ra_rad),
                    np.cos(dec_rad)*np.sin(ra_rad),
                    np.sin(dec_rad)))

    return xyz

def xyztoradec(xyz):
    ra = np.arctan2(xyz[1], xyz[0])
    ra += 2*np.pi * (ra < 0)
    dec = np.arcsin(xyz[2] / np.linalg.norm(xyz, axis=0))

    return (np.rad2deg(ra), np.rad2deg(dec))

def get_objects_center(obj, col_ra='ra', col_dec='dec'):
    """
    Returns the center RA, Dec, and radius in degrees for a cloud of objects on the sky.
    """
    xyz = radectoxyz(obj[col_ra], obj[col_dec])
    xyz0 = np.mean(xyz, axis=1)
    ra0,dec0 = xyztoradec(xyz0)

    sr0 = np.max(coords.sphdist(ra0, dec0, obj[col_ra], obj[col_dec]))

    return ra0, dec0, sr0

def blind_match_objects(obj, order=4, extra="", update=True, sn=20, verbose=False):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    wcs = None
    binname = None
    ext = 0

    for path in ['.', '/usr/local', '/opt/local']:
        if os.path.isfile(os.path.join(path, 'astrometry', 'bin', 'solve-field')):
            binname = os.path.join(path, 'astrometry', 'bin', 'solve-field')
            break

    if binname:
        dirname = tempfile.mkdtemp(prefix='astrometry')

        idx = obj['magerr']<1/sn
        columns = [fits.Column(name='XIMAGE', format='1D', array=obj['x'][idx]+1),
                   fits.Column(name='YIMAGE', format='1D', array=obj['y'][idx]+1),
                   fits.Column(name='FLUX', format='1D', array=obj['flux'][idx])]
        tbhdu = fits.BinTableHDU.from_columns(columns)
        filename = os.path.join(dirname, 'list.fits')
        tbhdu.writeto(filename, overwrite=True)
        extra += " --x-column XIMAGE --y-column YIMAGE --sort-column FLUX --width %d --height %d" % (np.ceil(max(obj['x']+1)), np.ceil(max(obj['y']+1)))

        wcsname = os.path.split(filename)[-1]
        tmpname = os.path.join(dirname, os.path.splitext(wcsname)[0] + '.tmp')
        wcsname = os.path.join(dirname, os.path.splitext(wcsname)[0] + '.wcs')

        command = "%s -D %s --no-verify --overwrite --no-plots -T %s %s" % (binname, dirname, extra, filename)

        log('Running Astrometry.Net first iteration like that:')
        log(command)

        os.system(command)

        if order:
            order_str = "-t %d" % order
        else:
            order_str = "-T"

        if os.path.isfile(wcsname):
            shutil.move(wcsname, tmpname)
            command = "%s -D %s --overwrite --no-plots %s %s --verify %s %s" % (binname, dirname, order_str, extra, tmpname, filename)

            log('Running Astrometry.Net second iteration like that:')
            log(command)

            os.system(command)

            if os.path.isfile(wcsname):
                header = fits.getheader(wcsname)
                wcs = WCS(header)

                if update and wcs:
                    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

                log('Second iteration succeeded')
            else:
                log('Second iteration failed')
        else:
            log('First iteration failed')

        shutil.rmtree(dirname)

    else:
        log("Astrometry.Net binary not found")

    return wcs

def blind_match_astrometrynet(obj, order=2, update=False, sn=20, get_header=False,
                              width=None, height=None,
                              solve_timeout=600, api_key=None,
                              center_ra=None, center_dec=None, radius=None,
                              scale_lower=None, scale_upper=None, scale_units='arcsecperpix', **kwargs):
    '''
    Thin wrapper for remote plate solving using Astrometry.Net and a list of detected objects.
    Most of the parameters are passed directly to astroquery.astrometrynet.AstrometryNet.solve_from_source_list routine.
    API key may either be provided as an argument or specified in ~/.astropy/config/astroquery.cfg
    '''

    # Sort objects according to decreasing flux
    aidx = np.argsort(-obj['flux'])

    # Filter out least-significant detections, if SN limit is specified
    if sn is not None and sn > 0:
        aidx = [_ for _ in aidx if obj['flux'][_]/obj['fluxerr'][_] > sn]

    if width is None:
        width = int(np.max(obj['x']))
    if height is None:
        height = int(np.max(obj['y']))

    an = AstrometryNet()
    if api_key is not None:
        an.api_key = api_key

    try:
        header = an.solve_from_source_list(obj['x'][aidx] + 1, obj['y'][aidx] + 1, width, height,
                                           center_ra=center_ra, center_dec=center_dec, radius=radius,
                                           scale_lower=scale_lower, scale_upper=scale_upper, scale_units=scale_units,
                                           solve_timeout=solve_timeout, tweak_order=order, **kwargs)
    except:
        import traceback
        traceback.print_exc()

        header = None

    if header is not None:
        wcs = WCS(header)

        if update:
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

        if get_header:
            return header
        else:
            return wcs

    return None

def refine_wcs(obj, cat, order=2, match=True, sr=3/3600, update=False,
               cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
               method='astropy', verbose=False):
    '''
    Refine the WCS using detected objects and catalogue.
    '''

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    if match:
        # Perform simple nearest-neighbor matching within given radius
        h = htm.HTM(10)
        oidx,cidx,dist = h.match(obj['ra'], obj['dec'], cat[cat_col_ra], cat[cat_col_dec], sr, maxmatch=0)
        _obj = obj[oidx]
        _cat = cat[cidx]
    else:
        # Assume supplied objects and catalogue are already matched line by line
        _obj = obj
        _cat = cat

    wcs = None

    if method == 'astropy':
        wcs = fit_wcs_from_points([_obj['x'], _obj['y']], SkyCoord(_cat[cat_col_ra], _cat[cat_col_dec], unit='deg'), sip_degree=order)
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
            dirname = tempfile.mkdtemp(prefix='astrometry')

            columns = [fits.Column(name='FIELD_X', format='1D', array=obj['x'] + 1),
                       fits.Column(name='FIELD_Y', format='1D', array=obj['y'] + 1),
                       fits.Column(name='INDEX_RA', format='1D', array=cat[cat_col_ra]),
                       fits.Column(name='INDEX_DEC', format='1D', array=cat[cat_col_dec])]
            tbhdu = fits.BinTableHDU.from_columns(columns)
            filename = os.path.join(dirname, 'list.fits')
            wcsname = os.path.join(dirname, 'list.wcs')

            tbhdu.writeto(filename, overwrite=True)

            command = "%s -c %s -o %s -W %d -H %d -C -s %d" % (binname, filename, wcsname, width, height, order)

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
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)
    else:
        log('WCS refinement failed')

    return wcs

def clear_wcs(header, remove_comments=False, remove_history=False, remove_underscored=False, copy=False):
    if copy:
        header = header.copy()

    wcs_keywords = ['WCSAXES', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'LONPOLE', 'LATPOLE', 'RADESYS', 'EQUINOX', 'B_ORDER', 'A_ORDER', 'BP_ORDER', 'AP_ORDER', 'CD1_1', 'CD2_1', 'CD1_2', 'CD2_2', 'IMAGEW', 'IMAGEH']

    scamp_keywords = ['FGROUPNO', 'ASTIRMS1', 'ASTIRMS2', 'ASTRRMS1', 'ASTRRMS2', 'ASTINST', 'FLXSCALE', 'MAGZEROP', 'PHOTIRMS', 'PHOTINST', 'PHOTLINK']

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

def wcs_sip2pv(header, method='astrometrynet'):
    """
    TODO
    """
    pass

def table_to_ldac(table, header=None, writeto=None):

    primary_hdu = fits.PrimaryHDU()

    header_str = header.tostring(endcard=True)
    # FIXME: this is a quick and dirty hack to preserve final 'END     ' in the string
    # as astropy.io.fits tends to strip trailing whitespaces from string data, and it breaks at least SCAMP
    header_str += fits.Header().tostring(endcard=True)

    header_col = fits.Column(name='Field Header Card', format='%dA' % len(header_str), array=[header_str])
    header_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([header_col]))
    header_hdu.header['EXTNAME'] = 'LDAC_IMHEAD'

    data_hdu = fits.table_to_hdu(table)
    data_hdu.header['EXTNAME'] = 'LDAC_OBJECTS'

    hdulist = fits.HDUList([primary_hdu, header_hdu, data_hdu])

    if writeto is not None:
        hdulist.writeto(writeto, overwrite=True)

    return hdulist

def refine_wcs_scamp(obj, cat=None, wcs=None, header=None, sr=2/3600, order=3,
                     cat_col_ra='RAJ2000', cat_col_dec='DEJ2000',
                     cat_col_ra_err='e_RAJ2000', cat_col_dec_err='e_DEJ2000',
                     cat_col_mag='rmag', cat_col_mag_err='e_rmag',
                     cat_mag_lim=99, extra={},
                     get_header=False, update=False,
                     _workdir=None, _tmpdir=None, verbose=False):
    """
    Wrapper for running SCAMP on user-provided object list and catalogue
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    # Find the binary
    binname = None
    for path in ['.', '/usr/bin', '/usr/local/bin', '/opt/local/bin']:
        for exe in ['scamp']:
            if os.path.isfile(os.path.join(path, exe)):
                binname = os.path.join(path, exe)
                break

    if binname is None:
        log("Can't find SCAMP binary")
        return None

    workdir = _workdir if _workdir is not None else tempfile.mkdtemp(prefix='scamp', dir=_tmpdir)

    if header is None:
        # Construct minimal FITS header covering our data points
        header = fits.Header({'NAXIS':2, 'NAXIS1':np.max(obj['x']+1), 'NAXIS2':np.max(obj['y'] + 1), 'BITPIX':-64, 'EQUINOX': 2000.0})
    else:
        header = header.copy()

    if wcs is not None and wcs.is_celestial:
        # Add WCS information to the header
        header += wcs.to_header(relax=True)

    xmlname = os.path.join(workdir, 'scamp.xml')

    opts = {
        'VERBOSE_TYPE': 'QUIET',
        'SOLVE_PHOTOM': 'N',
        'CHECKPLOT_TYPE': 'NONE',
        'WRITE_XML': 'Y',
        'XML_NAME': xmlname,
        'PROJECTION_TYPE': 'TPV',
        'CROSSID_RADIUS': sr*3600,
        'DISTORT_DEGREES': order,
    }

    opts.update(extra)

    # Minimal LDAC table with objects
    t_obj = Table(data={
        'XWIN_IMAGE': obj['x'] + 1, # SCAMP uses 1-based coordinates
        'YWIN_IMAGE': obj['y'] + 1,

        'ERRAWIN_IMAGE': obj['xerr'],
        'ERRBWIN_IMAGE': obj['yerr'],

        'FLUX_AUTO': obj['flux'],
        'FLUXERR_AUTO': obj['fluxerr'],
        'MAG_AUTO': obj['mag'],
        'MAGERR_AUTO': obj['magerr'],

        'FLAGS': obj['flags'],
    })

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
            t_cat = Table(data={
                'X_WORLD': cat[cat_col_ra],
                'Y_WORLD': cat[cat_col_dec],

                'ERRA_WORLD': utils.table_get(cat, cat_col_ra_err, 1/3600),
                'ERRB_WORLD': utils.table_get(cat, cat_col_dec_err, 1/3600),

                'MAG': cat[cat_col_mag],
                'MAGERR': utils.table_get(cat, cat_col_mag_err, 0.01),
                'OBSDATE': np.ones_like(cat['RAJ2000'])*2000.0
            })

            # Convert units of err columns to degrees, if any
            for _ in ['ERRA_WORLD', 'ERRB_WORLD']:
                if t_cat[_].unit and t_cat[_].unit != 'deg':
                    t_cat[_] = t_cat[_].to('deg')

            # Limit the catalogue to given magnitude range
            if cat_mag_lim is not None:
                if hasattr(cat_mag_lim, '__len__') and len(cat_mag_lim) == 2:
                    # Two elements provided, treat them as lower and upper limits
                    t_cat = t_cat[(t_cat['MAG'] >= cat_mag_lim[0]) & (t_cat['MAG'] <= cat_mag_lim[1])]
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
        log('%d matches, chi2 %.1f' % (diag['NDeg_Reference'], diag['Chi2_Reference']))
        # FIXME: is df correct here?..
        if chi2.sf(diag['Chi2_Reference'], df=diag['NDeg_Reference']) < 1e-3:
            log('It seems the fitting failed')
        else:
            with open(hdrname, 'r') as f:
                h1 = fits.Header.fromstring(f.read().encode('ascii', 'ignore'), sep='\n')

                # Sometimes SCAMP returns TAN type solution even despite PV keywords present
                if h1['CTYPE1'] != 'RA---TPV':
                    log('Got WCS solution with CTYPE1 =', h1['CTYPE1'], ', fixing it')
                    h1['CTYPE1'] = 'RA---TPV'
                    h1['CTYPE2'] = 'DEC--TPV'

                if get_header:
                    log('Returning raw header instead of WCS solution')
                    wcs = h1
                else:
                    wcs = WCS(h1)
                    if update:
                        obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

                    log('Astrometric accuracy: %.2f" %.2f"' % (h1.get('ASTRRMS1', 0)*3600, h1.get('ASTRRMS2', 0)*3600))

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
