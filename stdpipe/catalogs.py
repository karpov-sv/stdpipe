"""
Module containing the routines for handling various online catalogues.
"""

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
    'gaiadr2': {'vizier': 'I/345/gaia2', 'name': 'Gaia DR2'},
    'gaiaedr3': {'vizier': 'I/350/gaiaedr3', 'name': 'Gaia EDR3'},
    'usnob1': {'vizier': 'I/284/out', 'name': 'USNO-B1'},
    'gsc': {'vizier': 'I/271/out', 'name': 'GSC 2.2'},
    'skymapper': {'vizier': 'II/358/smss', 'name': 'SkyMapper DR1.1'},
    'vsx': {'vizier': 'B/vsx/vsx', 'name': 'AAVSO VSX'},
    'apass': {'vizier': 'II/336/apass9', 'name': 'APASS DR9'},
    'sdss': {'vizier': 'V/147/sdss12', 'name': 'SDSS DR12', 'extra':['_RAJ2000', '_DEJ2000']},
    'atlas': {'vizier': 'J/ApJ/867/105/refcat2', 'name': 'ATLAS-REFCAT2', 'extra':['_RAJ2000', '_DEJ2000', 'e_Gmag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_Jmag', 'e_Kmag']},
}

def get_cat_vizier(ra0, dec0, sr0, catalog='ps1', limit=-1, filters={}, extra=[]):
    """Download any catalogue from Vizier.

    The catalogue may be anything recognizable by Vizier. For some most popular ones, we have additional support - we try to augment them with photometric measurements not originally present there, based on some analytical magnitude conversion formulae. These catalogues are:

    -  ps1 - Pan-STARRS DR1. We augment it with Johnson-Cousins B, V, R and I magnitudes
    -  gaiadr2 - Gaia DR2. We augment it  with Johnson-Cousins B, V, R and I magnitudes, as well as Pan-STARRS and SDSS ones
    -  gaiaedr3 - Gaia eDR3
    -  skymapper - SkyMapper DR1.1
    -  vsx - AAVSO Variable Stars Index
    -  apass - AAVSO APASS DR9
    -  sdss - SDSS DR12
    -  atlas - ATLAS-RefCat2, compilative all-sky reference catalogue with uniform zero-points in Pan-STARRS-like bands. We augment it with Johnson-Cousins B, V, R and I magnitudes the same way as Pan-STARRS.
    -  usnob1 - USNO-B1
    -  gsc - Guide Star Catalogue 2.2

    :param ra0: Right Ascension of the field center, degrees
    :param dec0: Declination of the field center, degrees
    :param sr0: Field radius, degrees
    :param catalog: Any Vizier catalogue identifier, or a catalogue short name (see above)
    :param limit: Limit for the number of returned rows, optional
    :param filters: Dictionary with column filters to be applied on Vizier side. Dictionary key is the column name, value - filter expression as documented at https://vizier.u-strasbg.fr/vizier/vizHelp/cst.htx
    :param extra: List of extra column names to return in addition to default ones.
    :returns: astropy.table.Table with catalogue as returned by Vizier, with some additional columns added for supported catalogues.
    """

    # TODO: add positional errors

    if catalog in catalogs:
        # For some catalogs we may have some additional information
        vizier_id = catalogs.get(catalog).get('vizier')
        name = catalogs.get(catalog).get('name')

        columns = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] + extra + catalogs.get(catalog).get('extra', [])
    else:
        vizier_id = catalog
        name = catalog
        columns = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] + extra

    vizier = Vizier(row_limit=limit, columns=columns, column_filters=filters)
    cats = vizier.query_region(SkyCoord(ra0, dec0, unit='deg'), radius=sr0*u.deg, catalog=vizier_id)

    if not cats or not len(cats) == 1:
        cats = vizier.query_region(SkyCoord(ra0, dec0, unit='deg'), radius=sr0*u.deg, catalog=vizier_id, cache=False)

        if not cats or not len(cats) == 1:
            print('Error requesting catalogue', catalog)
            return None

    cat = cats[0]
    cat.meta['vizier_id'] = vizier_id
    cat.meta['name'] = name

    # Fix _RAJ2000/_DEJ2000
    if '_RAJ2000' in cat.keys() and '_DEJ2000' in cat.keys() and not 'RAJ2000' in cat.keys():
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    # Augment catalogue with additional bandpasses

    if catalog == 'ps1' or catalog == 'atlas':
        # Alternative PS1 transfromation from https://arxiv.org/pdf/1706.06147.pdf, Stetson, seems better with Landolt than official one
        cat['B'] = cat['gmag'] + 0.199 + 0.540*(cat['gmag'] - cat['rmag']) + 0.016*(cat['gmag'] - cat['rmag'])**2
        cat['V'] = cat['gmag'] - 0.020 - 0.498*(cat['gmag'] - cat['rmag']) - 0.008*(cat['gmag'] - cat['rmag'])**2
        cat['R'] = cat['rmag'] - 0.163 - 0.086*(cat['gmag'] - cat['rmag']) - 0.061*(cat['gmag'] - cat['rmag'])**2
        cat['I'] = cat['imag'] - 0.387 - 0.123*(cat['gmag'] - cat['rmag']) - 0.034*(cat['gmag'] - cat['rmag'])**2

        # to SDSS, zero points and color terms from https://arxiv.org/pdf/1203.0297.pdf
        cat['g_SDSS'] = cat['gmag'] + 0.013 + 0.145*(cat['gmag'] - cat['rmag']) + 0.019*(cat['gmag'] - cat['rmag'])**2
        cat['r_SDSS'] = cat['rmag'] - 0.001 + 0.004*(cat['gmag'] - cat['rmag']) + 0.007*(cat['gmag'] - cat['rmag'])**2
        cat['i_SDSS'] = cat['imag'] - 0.005 + 0.011*(cat['gmag'] - cat['rmag']) + 0.010*(cat['gmag'] - cat['rmag'])**2
        cat['z_SDSS'] = cat['zmag']

        # to SDSS, from Finkbeiner et al. https://arxiv.org/pdf/1512.01214.pdf, valid post-DR13
        cat['g_SDSS'] = cat['gmag'] + 0.01808 + 0.13595*(cat['gmag'] - cat['imag']) - 0.01941*(cat['gmag'] - cat['imag'])**2 + 0.00183*(cat['gmag'] - cat['imag'])**3
        cat['r_SDSS'] = cat['rmag'] + 0.01836 + 0.03577*(cat['gmag'] - cat['imag']) - 0.02612*(cat['gmag'] - cat['imag'])**2 + 0.00558*(cat['gmag'] - cat['imag'])**3
        cat['i_SDSS'] = cat['imag'] - 0.01170 + 0.00400*(cat['gmag'] - cat['imag']) - 0.00066*(cat['gmag'] - cat['imag'])**2 + 0.00058*(cat['gmag'] - cat['imag'])**3

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

        # to PS1 - FIXME: there are some uncorrected color and magnitude trends!
        cat['gmag'] = cat['B'] - 0.108 - 0.485*(cat['B'] - cat['V']) - 0.032*(cat['B'] - cat['V'])**2
        cat['rmag'] = cat['V'] + 0.082 - 0.462*(cat['B'] - cat['V']) + 0.041*(cat['B'] - cat['V'])**2

        # to SDSS, from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
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
    """Cross-match object list with Vizier catalogue using CDS XMatch service.

    Any Vizier catalogue may be used for cross-matching.

    :param obj: astropy.table.Table with objects
    :param catalog: Any Vizier catalogue identifier, or a catalogue short name
    :param sr: Cross-matching radius in degrees
    :param col_ra: Column name in `obj` table containing Right Ascension values
    :param col_dec: Column name in `obj` table containing Declination values
    :returns: The table of matched objects augmented with some fields from the Vizier catalogue.
    """

    if catalog in catalogs:
        vizier_id = catalogs.get(catalog)['vizier']
    else:
        vizier_id = catalog

    xcat = XMatch().query(cat1=obj, cat2='vizier:' + vizier_id, max_distance=sr*u.deg, colRA1=col_ra, colDec1=col_dec)

    return xcat

def xmatch_skybot(obj, sr=10/3600, time=None, col_ra='ra', col_dec='dec', col_id='id'):
    """Cross-match object list with positions of Solar System objects using SkyBoT service

    The routine works by requesting the list of all solar system objects in a cone containing all
    input objects, and then cross-matching them using the radius defined by `sr` parameter. Then it
    returns the table of solar system objects plus a column of unique identifiers corresponding
    to user objects.

    :param obj: astropy.table.Table with objects
    :param sr: Cross-matching radius in degrees
    :param time: Time of the observation corresponding to the objects
    :param col_ra: Column name in `obj` table containing Right Ascension values
    :param col_dec: Column name in `obj` table containing Declination values
    :param col_id: Column name in `obj` table containing some unique object identifier
    :returns: The table of solar system objects augmented with `col_id` column of matched objects.
    """

    ra0,dec0,sr0 = astrometry.get_objects_center(obj)

    try:
        # Query SkyBot for (a bit larger than) our FOV at our time
        xcat = Skybot.cone_search(SkyCoord(ra0, dec0, unit='deg'), (sr0 + 2.0*sr)*u.deg, Time(time))
    except (RuntimeError, KeyError, ConnectionError, OSError):
        # Nothing found in SkyBot
        return None

    # Cross-match objects
    h = htm.HTM(10)
    oidx,cidx,dist = h.match(obj[col_ra], obj[col_dec], xcat['RA'], xcat['DEC'], 10/3600)

    # Annotate the table with id from objects so that it is possible to identify the matches
    xcat[col_id] = MaskedColumn(len(xcat), dtype=np.dtype(obj[col_id][0]))
    xcat[col_id].mask = True
    xcat[col_id][cidx] = obj[col_id][oidx]
    xcat[col_id][cidx].mask = False

    return xcat

def xmatch_ned(obj, sr=3/3600, col_ra='ra', col_dec='dec', col_id='id'):
    """Cross-match object list with NED database entries

    The routine is extremely inefficient as it has to query the objects one by one!

    :param obj: astropy.table.Table with objects
    :param sr: Cross-matching radius in degrees
    :param col_ra: Column name in `obj` table containing Right Ascension values
    :param col_dec: Column name in `obj` table containing Declination values
    :param col_id: Column name in `obj` table containing some unique object identifier
    :returns: The table of NED objects augmented with `id` column containing the identifiers from `col_id` columns of matched objects.

    """

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
