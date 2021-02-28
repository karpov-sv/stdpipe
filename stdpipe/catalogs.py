from __future__ import absolute_import, division, print_function, unicode_literals

import os, posixpath, shutil, tempfile
import numpy as np

from astropy.table import Table, MaskedColumn, vstack
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astroquery.imcce import Skybot
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from esutil import htm

from . import astrometry

catalogs = {
    'ps1': {'vizier': 'II/349/ps1', 'name': 'PanSTARRS DR1'},
    'gaiadr2': {'vizier': 'I/345/gaia2', 'name': 'Gaia DR2', 'extra': ['RAJ2000', 'DEJ2000']},
    'usnob1': {'vizier': 'I/284/out', 'name': 'USNO-B1'},
    'gsc': {'vizier': 'I/271/out', 'name': 'GSC 2.2'},
    'skymapper': {'vizier': 'II/358/smss', 'name': 'SkyMapper DR1.1', 'extra': ['RAJ2000', 'DEJ2000']},
}

def get_cat_vizier(ra0, dec0, sr0, catalog='ps1', limit=-1, filters={}):
    if catalog not in catalogs:
        print('Unsupported catalogue', catalog)
        return None

    vizier_id = catalogs.get(catalog).get('vizier')
    name = catalogs.get(catalog).get('name')

    columns = ['*'] + catalogs.get(catalog).get('extra', [])

    vizier = Vizier(row_limit=limit, columns=columns, column_filters=filters)
    cats = vizier.query_region(SkyCoord(ra0, dec0, unit='deg'), radius=sr0*u.deg, catalog=vizier_id)

    if not cats or not len(cats) == 1:
        print('Error requesting catalogue', catalog)
        return None

    cat = cats[0]
    cat.meta['vizier_id'] = vizier_id
    cat.meta['name'] = name

    # Fix _RAJ2000/_DEJ2000
    if '_RAJ2000' in cat.keys() and '_DEJ2000' in cat.keys():
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    # Augment catalogue with additional bandpasses

    if catalog == 'ps1':
        # Alternative PS1 transfromation from https://arxiv.org/pdf/1706.06147.pdf, Stetson, seems better with Landolt than official one
        cat['B'] = cat['gmag'] + 0.199 + 0.540*(cat['gmag'] - cat['rmag']) + 0.016*(cat['gmag'] - cat['rmag'])**2
        cat['V'] = cat['gmag'] - 0.020 - 0.498*(cat['gmag'] - cat['rmag']) - 0.008*(cat['gmag'] - cat['rmag'])**2
        cat['R'] = cat['rmag'] - 0.163 - 0.086*(cat['gmag'] - cat['rmag']) - 0.061*(cat['gmag'] - cat['rmag'])**2
        cat['I'] = cat['imag'] - 0.387 - 0.123*(cat['gmag'] - cat['rmag']) - 0.034*(cat['gmag'] - cat['rmag'])**2

        # to SDSS
        # FIXME: add correct zero points and color terms from https://arxiv.org/pdf/1203.0297.pdf
        cat['g_SDSS'] = cat['gmag']
        cat['r_SDSS'] = cat['rmag']
        cat['i_SDSS'] = cat['imag']
        cat['z_SDSS'] = cat['zmag']

    elif catalog == 'gaiadr2':
        # My simple Gaia DR2 to Johnson conversion based on Stetson standards
        pB = [-0.05927724559795761, 0.4224326324292696, 0.626219707920836, -0.011211539139725953]
        pV = [0.0017624722901609662, 0.15671377090187089, 0.03123927839356175, 0.041448557506784556]
        pR = [0.02045449129406191, 0.054005149296716175, -0.3135475489352255, 0.020545083667168156]
        pI = [0.005092289380850884, 0.07027022935721515, -0.7025553064161775, -0.02747532184796779]

        g = cat['Gmag']
        bp_rp = cat['BPmag'] - cat['RPmag']

        # https://www.cosmos.esa.int/web/gaia/dr2-known-issues#PhotometrySystematicEffectsAndResponseCurves
        gcorr = g.copy()
        gcorr[(g>2)&(g<6)] = -0.047344 + 1.16405*g[(g>2)&(g<6)] - 0.046799*g[(g>2)&(g<6)]**2 + 0.0035015*g[(g>2)&(g<6)]**3
        gcorr[(g>6)&(g<16)] = g[(g>6)&(g<16)] - 0.0032*(g[(g>6)&(g<16)] - 6)
        gcorr[g>16] = g[g>16] - 0.032
        g = gcorr

        cat['B'] = g + np.polyval(pB, bp_rp)
        cat['V'] = g + np.polyval(pV, bp_rp)
        cat['R'] = g + np.polyval(pR, bp_rp)
        cat['I'] = g + np.polyval(pI, bp_rp)

        # to SDSS
        cat['g_SDSS'] = g - (0.13518 - 0.46245*bp_rp - 0.25171*bp_rp**2 + 0.021349*bp_rp**3)
        cat['r_SDSS'] = g - (-0.12879 + 0.24662*bp_rp - 0.027464*bp_rp**2 - 0.049465*bp_rp**3)
        cat['i_SDSS'] = g - (-0.29676 + 0.64728*bp_rp - 0.10141*bp_rp**2)

    elif catalog == 'skymapper':
        # to SDSS
        for _ in ['u', 'g', 'r', 'i', 'z']:
            cat[_ + '_PS1'] = cat[_ + 'PSF']

        pass

    return cat

def xmatch_objects(obj, catalog='ps1', sr=3/3600, col_ra='ra', col_dec='dec'):
    if catalog in catalogs:
        vizier_id = catalogs.get(catalog)['vizier']
    else:
        vizier_id = catalog

    xcat = XMatch().query(cat1=obj, cat2='vizier:' + vizier_id, max_distance=sr*u.deg, colRA1=col_ra, colDec1=col_dec)

    return xcat

def xmatch_skybot(obj, sr=10/3600, time=None, location='500', col_ra='ra', col_dec='dec', col_id='id'):
    ra0,dec0,sr0 = astrometry.get_objects_center(obj)

    try:
        # Query SkyBot for (a bit larger than) our FOV at our time
        xcat = Skybot.cone_search(SkyCoord(ra0, dec0, unit='deg'), (sr0 + 2.0*sr)*u.deg, Time(time))
    except RuntimeError:
        # Nothing fount in SkyBot
        return None

    # Cross-match objects
    h = htm.HTM(10)
    oidx,cidx,dist = h.match(obj[col_ra], obj[col_dec], xcat['RA'], xcat['DEC'], 10/3600)

    # Annotate the table with id from objects so that it is possible to identify the matches
    xcat['id'] = MaskedColumn(len(xcat), dtype=np.dtype(obj[col_id][0]))
    xcat['id'].mask = True
    xcat['id'][cidx] = obj[col_id][oidx]
    xcat['id'][cidx].mask = False

    return xcat

def xmatch_ned(obj, sr=3/3600, col_ra='ra', col_dec='dec', col_id='id'):
    xcat = []

    # FIXME: is there more optimal way to query NED for multiple sky positions?..
    for row in obj:
        res = Ned().query_region(SkyCoord(row[col_ra], row[col_dec], unit='deg'), sr*u.deg)

        if res:
            res['id'] = row[col_id]

            xcat.append(res)

    if xcat:
        xcat = vstack(xcat, join_type='exact', metadata_conflicts='silent')

    return xcat
