from __future__ import absolute_import, division, print_function, unicode_literals

import os, tempfile, posixpath, shutil, re
import numpy as np

from esutil import coords, htm

from astropy.wcs import WCS
from astropy.io import fits

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

def blind_match_objects(obj, order=4, extra="", verbose=False, update=True, sn=20):
    wcs = None
    binname = None
    ext = 0

    for path in ['.', '/usr/local', '/opt/local']:
        if os.path.isfile(posixpath.join(path, 'astrometry', 'bin', 'solve-field')):
            binname = posixpath.join(path, 'astrometry', 'bin', 'solve-field')
            break

    if binname:
        dirname = tempfile.mkdtemp(prefix='astrometry')

        idx = obj['magerr']<1/sn
        columns = [fits.Column(name='XIMAGE', format='1D', array=obj['x'][idx]+1),
                   fits.Column(name='YIMAGE', format='1D', array=obj['y'][idx]+1),
                   fits.Column(name='FLUX', format='1D', array=obj['flux'][idx])]
        tbhdu = fits.BinTableHDU.from_columns(columns)
        filename = posixpath.join(dirname, 'list.fits')
        tbhdu.writeto(filename, overwrite=True)
        extra += " --x-column XIMAGE --y-column YIMAGE --sort-column FLUX --width %d --height %d" % (np.ceil(max(obj['x']+1)), np.ceil(max(obj['y']+1)))

        wcsname = posixpath.split(filename)[-1]
        tmpname = posixpath.join(dirname, posixpath.splitext(wcsname)[0] + '.tmp')
        wcsname = posixpath.join(dirname, posixpath.splitext(wcsname)[0] + '.wcs')

        if verbose:
            print("%s -D %s --no-verify --overwrite --no-plots -T %s %s" % (binname, dir, extra, filename))

        os.system("%s -D %s --no-verify --overwrite --no-plots -T %s %s" % (binname, dir, extra, filename))

        if order:
            order_str = "-t %d" % order
        else:
            order_str = "-T"

        if os.path.isfile(wcsname):
            shutil.move(wcsname, tmpname)
            os.system("%s -D %s --overwrite --no-plots %s %s --verify %s %s" % (binname, dir, order_str, extra, tmpname, filename))

            if os.path.isfile(wcsname):
                header = fits.getheader(wcsname)
                wcs = WCS(header)

                if update and wcs:
                    obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)

        shutil.rmtree(dir)

    else:
        print("Astrometry.Net binary not found")

    return wcs

def refine_wcs(obj, cat, order=2, match=True, sr=3/3600, update=False, cat_col_ra='RAJ2000', cat_col_dec='DEJ2000', method='astropy'):
    '''Refine the WCS using detected objects and catalogue. '''
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
            if os.path.isfile(posixpath.join(path, 'astrometry', 'bin', 'fit-wcs')):
                binname = posixpath.join(path, 'astrometry', 'bin', 'fit-wcs')
                break

        if binname:
            dirname = tempfile.mkdtemp(prefix='astrometry')

            columns = [fits.Column(name='FIELD_X', format='1D', array=obj['x'] + 1),
                       fits.Column(name='FIELD_Y', format='1D', array=obj['y'] + 1),
                       fits.Column(name='INDEX_RA', format='1D', array=cat[cat_col_ra]),
                       fits.Column(name='INDEX_DEC', format='1D', array=cat[cat_col_dec])]
            tbhdu = fits.BinTableHDU.from_columns(columns)
            filename = posixpath.join(dirname, 'list.fits')
            wcsname = posixpath.join(dirname, 'list.wcs')

            tbhdu.writeto(filename, overwrite=True)

            os.system("%s -c %s -o %s -W %d -H %d -C -s %d" % (binname, filename, wcsname, width, height, order))

            if os.path.isfile(wcsname):
                header = fits.getheader(wcsname)
                wcs = WCS(header)

            shutil.rmtree(dirname)

        else:
            print("Astrometry.Net binary not found")

    if wcs:
        if update:
            # Update the sky coordinates of objects using new wcs
            obj['ra'],obj['dec'] = wcs.all_pix2world(obj['x'], obj['y'], 0)
    else:
        print('Could not update wcs')

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
