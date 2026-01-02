"""
Module containing the routines for handling various online catalogues.
"""


import os, posixpath, shutil, tempfile
import numpy as np

from astropy.table import Table, MaskedColumn, vstack
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astroquery.imcce import Skybot
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from . import astrometry

catalogs = {
    'ps1': {'vizier': 'II/349/ps1', 'name': 'PanSTARRS DR1'},
    'gaiadr2': {'vizier': 'I/345/gaia2', 'name': 'Gaia DR2', 'extra': ['E(BR/RP)']},
    'gaiaedr3': {'vizier': 'I/350/gaiaedr3', 'name': 'Gaia EDR3'},
    'gaiadr3syn': {
        'vizier': 'I/360/syntphot',
        'name': 'Gaia DR3 synthetic photometry',
        'extra': ['**', '_RAJ2000', '_DEJ2000'],
    },
    'usnob1': {'vizier': 'I/284/out', 'name': 'USNO-B1'},
    'gsc': {'vizier': 'I/271/out', 'name': 'GSC 2.2'},
    'skymapper': {
        'vizier': 'II/379/smssdr4',
        'name': 'SkyMapper DR4',
        'extra': [
            '_RAJ2000',
            '_DEJ2000',
            'e_uPSF',
            'e_vPSF',
            'e_gPSF',
            'e_rPSF',
            'e_iPSF',
            'e_zPSF',
        ],
    },
    'vsx': {'vizier': 'B/vsx/vsx', 'name': 'AAVSO VSX'},
    'apass': {'vizier': 'II/336/apass9', 'name': 'APASS DR9'},
    'sdss': {
        'vizier': 'V/154/sdss16',
        'name': 'SDSS DR16',
        'extra': ['_RAJ2000', '_DEJ2000'],
    },
    'atlas': {
        'vizier': 'J/ApJ/867/105/refcat2',
        'name': 'ATLAS-REFCAT2',
        'extra': [
            '_RAJ2000',
            '_DEJ2000',
            'e_Gmag',
            'e_gmag',
            'e_rmag',
            'e_imag',
            'e_zmag',
            'e_Jmag',
            'e_Kmag',
        ],
    },
}


def _detect_catalog_type(cat):
    """
    Detect catalog type from column names.

    Parameters
    ----------
    cat : astropy.table.Table
        Catalog table to inspect

    Returns
    -------
    catalog_type : str or None
        Detected catalog type ('ps1', 'gaiadr2', 'skymapper', 'apass',
        'gaiadr3syn', 'sdss') or None if cannot detect
    """
    colnames = set(cat.colnames)

    # Check most specific patterns first to avoid ambiguity

    # Gaia DR3 Synthetic - has Fuv, Fg, Fr, Fi, Fz flux columns
    if {'Fuv', 'Fg', 'Fr', 'Fi', 'Fz'}.issubset(colnames):
        return 'gaiadr3syn'

    # Gaia DR2 - has Gmag, BPmag, RPmag
    if {'Gmag', 'BPmag', 'RPmag'}.issubset(colnames):
        return 'gaiadr2'

    # SkyMapper - has gPSF, rPSF, iPSF, zPSF (PSF suffix is distinctive)
    if {'gPSF', 'rPSF', 'iPSF', 'zPSF'}.issubset(colnames):
        return 'skymapper'

    # APASS - has both SDSS-like (g_mag, r_mag, i_mag) AND Johnson (B_mag, V_mag)
    if {'g_mag', 'r_mag', 'i_mag', 'B_mag', 'V_mag'}.issubset(colnames):
        return 'apass'

    # SDSS DR16 - has umag, gmag, rmag, imag, zmag (but NOT ymag)
    # Need to distinguish from PS1 which also has gmag, rmag, imag, zmag
    if {'umag', 'gmag', 'rmag', 'imag', 'zmag'}.issubset(colnames):
        if 'ymag' not in colnames:  # SDSS doesn't have y band
            return 'sdss'

    # Pan-STARRS or ATLAS - both have gmag, rmag, imag, zmag, ymag
    # Cannot distinguish between them from columns alone (identical augmentation anyway)
    if {'gmag', 'rmag', 'imag', 'zmag', 'ymag'}.issubset(colnames):
        return 'ps1'  # Could be 'atlas' too, but augmentation is identical

    # Could not detect
    return None


def _augment_ps1(cat, verbose=False):
    """Augment Pan-STARRS DR1 or ATLAS-RefCat2 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Johnson-Cousins photometry")

        # My own fit on Landolt+Stetson standards from https://arxiv.org/pdf/2205.06186.pdf
        pB1, pB2 = (
            [
                0.10339527794499666,
                -0.492149523946056,
                1.2093816061394638,
                0.061925048331498395,
            ],
            [
                -0.2571974580267897,
                0.9211495207523038,
                -0.8243222108864755,
                0.0619250483314976,
            ],
        )
        pV1, pV2 = (
            [
                -0.011452922062676726,
                -9.949308251868327e-05,
                -0.4650511584366353,
                -0.007076854914511554,
            ],
            [
                0.012749150754020416,
                0.057554580469724864,
                -0.09019328095355343,
                -0.007076854914511329,
            ],
        )
        pR1, pR2 = (
            [
                0.004905242602502597,
                -0.046545625824660514,
                0.07830702317352654,
                -0.08438139204305026,
            ],
            [
                -0.07782426914647306,
                0.14090289318728444,
                -0.3634922073369279,
                -0.08438139204305031,
            ],
        )
        pI1, pI2 = (
            [
                -0.02239162647929074,
                0.04401240100377888,
                -0.038500349283596795,
                -0.19509051168348646
            ],
            [
                0.014586929059030904,
                -0.025228407778416825,
                -0.21476143248697746,
                -0.19509051168348637]
        )

        cat['Bmag'] = (
            cat['gmag']
            + np.polyval(pB1, cat['gmag'] - cat['rmag'])
            + np.polyval(pB2, cat['rmag'] - cat['imag'])
        )
        cat['Vmag'] = (
            cat['gmag']
            + np.polyval(pV1, cat['gmag'] - cat['rmag'])
            + np.polyval(pV2, cat['rmag'] - cat['imag'])
        )
        cat['Rmag'] = (
            cat['rmag']
            + np.polyval(pR1, cat['gmag'] - cat['rmag'])
            + np.polyval(pR2, cat['rmag'] - cat['imag'])
        )
        cat['Imag'] = (
            cat['imag']
            + np.polyval(pI1, cat['gmag'] - cat['rmag'])
            + np.polyval(pI2, cat['rmag'] - cat['imag'])
        )

        cat['e_Bmag'] = cat['e_gmag']
        cat['e_Vmag'] = cat['e_gmag']
        cat['e_Rmag'] = cat['e_rmag']
        cat['e_Imag'] = cat['e_imag']

        # Copies of columns for convenience
        for _ in ['B', 'V', 'R', 'I']:
            cat[_] = cat[_ + 'mag']

        cat['good'] = (cat['gmag'] - cat['rmag'] > -0.5) & (
            cat['gmag'] - cat['rmag'] < 2.5
        )
        cat['good'] &= (cat['rmag'] - cat['imag'] > -0.5) & (
            cat['rmag'] - cat['imag'] < 2.0
        )
        cat['good'] &= (cat['imag'] - cat['zmag'] > -0.5) & (
            cat['imag'] - cat['zmag'] < 1.0
        )

        # to SDSS, zero points and color terms from https://arxiv.org/pdf/1203.0297.pdf
        cat['g_SDSS'] = (
            cat['gmag']
            + 0.013
            + 0.145 * (cat['gmag'] - cat['rmag'])
            + 0.019 * (cat['gmag'] - cat['rmag']) ** 2
        )
        cat['r_SDSS'] = (
            cat['rmag']
            - 0.001
            + 0.004 * (cat['gmag'] - cat['rmag'])
            + 0.007 * (cat['gmag'] - cat['rmag']) ** 2
        )
        cat['i_SDSS'] = (
            cat['imag']
            - 0.005
            + 0.011 * (cat['gmag'] - cat['rmag'])
            + 0.010 * (cat['gmag'] - cat['rmag']) ** 2
        )
        cat['z_SDSS'] = cat['zmag']

    except KeyError as e:
        log(f"Warning: Missing required column for PS1/ATLAS augmentation: {e}")


def _augment_gaiadr2(cat, verbose=False):
    """Augment Gaia DR2 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Johnson-Cousins photometry")

        # My simple Gaia DR2 to Johnson conversion based on Stetson standards
        pB = [
            -0.05927724559795761,
            0.4224326324292696,
            0.626219707920836,
            -0.011211539139725953,
        ]
        pV = [
            0.0017624722901609662,
            0.15671377090187089,
            0.03123927839356175,
            0.041448557506784556,
        ]
        pR = [
            0.02045449129406191,
            0.054005149296716175,
            -0.3135475489352255,
            0.020545083667168156,
        ]
        pI = [
            0.005092289380850884,
            0.07027022935721515,
            -0.7025553064161775,
            -0.02747532184796779,
        ]

        pCB = [876.4047401692277, 5.114021693079334, -2.7332873314449326, 0]
        pCV = [98.03049528983964, 20.582521666713028, 0.8690079603974803, 0]
        pCR = [347.42190542330945, 39.42482430363565, 0.8626828845232541, 0]
        pCI = [79.4028706486939, 9.176899238787003, -0.7826315256072135, 0]

        g = cat['Gmag']
        bp_rp = cat['BPmag'] - cat['RPmag']

        # phot_bp_rp_excess_factor == E(BR/RP) == E_BR_RP_
        Cstar = cat['E_BR_RP_'] - np.polyval(
            [-0.00445024, 0.0570293, -0.02810592, 1.20477819], bp_rp
        )

        # https://www.cosmos.esa.int/web/gaia/dr2-known-issues#PhotometrySystematicEffectsAndResponseCurves
        gcorr = g.copy()
        gcorr[(g > 2) & (g < 6)] = (
            -0.047344
            + 1.16405 * g[(g > 2) & (g < 6)]
            - 0.046799 * g[(g > 2) & (g < 6)] ** 2
            + 0.0035015 * g[(g > 2) & (g < 6)] ** 3
        )
        gcorr[(g > 6) & (g < 16)] = g[(g > 6) & (g < 16)] - 0.0032 * (
            g[(g > 6) & (g < 16)] - 6
        )
        gcorr[g > 16] = g[g > 16] - 0.032
        g = gcorr

        cat['Bmag'] = g + np.polyval(pB, bp_rp) + np.polyval(pCB, Cstar)
        cat['Vmag'] = g + np.polyval(pV, bp_rp) + np.polyval(pCV, Cstar)
        cat['Rmag'] = g + np.polyval(pR, bp_rp) + np.polyval(pCR, Cstar)
        cat['Imag'] = g + np.polyval(pI, bp_rp) + np.polyval(pCI, Cstar)

        # Rough estimation of magnitude error, just from G band
        for _ in ['B', 'V', 'R', 'I', 'g', 'r']:
            cat['e_' + _ + 'mag'] = cat['e_Gmag']

        # Copies of columns for convenience
        for _ in ['B', 'V', 'R', 'I']:
            cat[_] = cat[_ + 'mag']

        # to PS1 - FIXME: there are some uncorrected color and magnitude trends!
        cat['gmag'] = (
            cat['Bmag']
            - 0.108
            - 0.485 * (cat['Bmag'] - cat['Vmag'])
            - 0.032 * (cat['Bmag'] - cat['Vmag']) ** 2
        )
        cat['rmag'] = (
            cat['Vmag']
            + 0.082
            - 0.462 * (cat['Bmag'] - cat['Vmag'])
            + 0.041 * (cat['Bmag'] - cat['Vmag']) ** 2
        )

        # to SDSS, from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        cat['g_SDSS'] = g - (
            0.13518 - 0.46245 * bp_rp - 0.25171 * bp_rp ** 2 + 0.021349 * bp_rp ** 3
        )
        cat['r_SDSS'] = g - (
            -0.12879 + 0.24662 * bp_rp - 0.027464 * bp_rp ** 2 - 0.049465 * bp_rp ** 3
        )
        cat['i_SDSS'] = g - (-0.29676 + 0.64728 * bp_rp - 0.10141 * bp_rp ** 2)

    except KeyError as e:
        log(f"Warning: Missing required column for Gaia DR2 augmentation: {e}")


def _augment_skymapper(cat, verbose=False):
    """Augment SkyMapper DR4 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Pan-STARRS photometry")

        # SkyMapper DR4 to PS1 griz, my fit based on Pancino et al. (2022)
        pg1,pg2 = (
            [
                -0.07715320986152466,
                0.2694597282089696,
                0.04069379065128178,
                0.01396290714542747
            ],
            [
                0.026097008342026252,
                -0.14040957287568073,
                0.133647539780504,
                0.013962907145427432
            ]
        )
        pr1,pr2 = (
            [
                0.08779280979185472,
                -0.23257704629617004,
                0.1890698144343673,
                -0.008125550119663026
            ],
            [
                -0.06273832689338121,
                0.21909317812693613,
                -0.33333867821346764,
                -0.008125550119663026
            ]
        )
        pi1,pi2 = (
            [
                0.07682346433564926,
                -0.25968093729564535,
                0.20326695606024423,
                -0.020866669846663343
            ],
            [
                -0.07053451796616398,
                0.24393073668848025,
                -0.34530693346891785,
                -0.020866669846663343
            ]
        )
        pz1,pz2 = (
            [
                0.09062113394001562,
                -0.2872099858823925,
                0.13842746024394773,
                -0.07120866802036607
            ],
            [
                -0.0872638638913885,
                0.3004063773229889,
                -0.3662066034354688,
                -0.07120866802036607
            ]
        )
        py1,py2 = (
            [
                0.09464754889799834,
                -0.25743991885453054,
                0.023419697082267195,
                -0.14405078638024026
            ],
            [
                -0.13242354061989023,
                0.42697881074088455,
                -0.4516034623969999,
                -0.14405078638024026
            ]
        )

        cat['gmag'] = cat['gPSF'] + np.polyval(pg1, cat['gPSF']-cat['rPSF']) + np.polyval(pg2, cat['rPSF']-cat['iPSF'])
        cat['rmag'] = cat['rPSF'] + np.polyval(pr1, cat['gPSF']-cat['rPSF']) + np.polyval(pr2, cat['rPSF']-cat['iPSF'])
        cat['imag'] = cat['iPSF'] + np.polyval(pi1, cat['gPSF']-cat['rPSF']) + np.polyval(pi2, cat['rPSF']-cat['iPSF'])
        cat['zmag'] = cat['zPSF'] + np.polyval(pz1, cat['gPSF']-cat['rPSF']) + np.polyval(pz2, cat['rPSF']-cat['iPSF'])
        cat['ymag'] = cat['yPSF'] + np.polyval(py1, cat['gPSF']-cat['rPSF']) + np.polyval(py2, cat['rPSF']-cat['iPSF'])

        cat['e_gmag'] = cat['e_gPSF']
        cat['e_rmag'] = cat['e_rPSF']
        cat['e_imag'] = cat['e_iPSF']

        log("Augmenting the catalogue with Johnson-Cousins photometry")

        # Transformation from SkyMapper to Johnson-Cousins, my own fit
        pB1, pB2 = (
            [
                0.10339527794499666,
                -0.4921495239460559,
                1.2093816061394642,
                0.061925048331498395,
            ],
            [
                -0.2571974580267897,
                0.9211495207523038,
                -0.8243222108864753,
                0.0619250483314976,
            ],
        )
        pV1, pV2 = (
            [
                -0.011452922062676726,
                -9.949308251868327e-05,
                -0.46505115843663526,
                -0.007076854914511554,
            ],
            [
                0.012749150754020416,
                0.05755458046972487,
                -0.09019328095355345,
                -0.007076854914511329,
            ],
        )
        pR1, pR2 = (
            [
                0.004905242602502596,
                -0.046545625824660514,
                0.07830702317352654,
                -0.08438139204305026,
            ],
            [
                -0.07782426914647306,
                0.14090289318728444,
                -0.363492207336928,
                -0.08438139204305031,
            ],
        )
        pI1, pI2 = (
            [
                -0.02239162647929074,
                0.04401240100377889,
                -0.03850034928359679,
                -0.19509051168348646,
            ],
            [
                0.014586929059030904,
                -0.025228407778416826,
                -0.21476143248697746,
                -0.19509051168348637,
            ],
        )

        cat['Bmag'] = (
            cat['gmag']
            + np.polyval(pB1, cat['gmag'] - cat['rmag'])
            + np.polyval(pB2, cat['rmag'] - cat['imag'])
        )
        cat['Vmag'] = (
            cat['gmag']
            + np.polyval(pV1, cat['gmag'] - cat['rmag'])
            + np.polyval(pV2, cat['rmag'] - cat['imag'])
        )
        cat['Rmag'] = (
            cat['rmag']
            + np.polyval(pR1, cat['gmag'] - cat['rmag'])
            + np.polyval(pR2, cat['rmag'] - cat['imag'])
        )
        cat['Imag'] = (
            cat['imag']
            + np.polyval(pI1, cat['gmag'] - cat['rmag'])
            + np.polyval(pI2, cat['rmag'] - cat['imag'])
        )

        cat['e_Bmag'] = cat['e_gmag']
        cat['e_Vmag'] = cat['e_gmag']
        cat['e_Rmag'] = cat['e_rmag']
        cat['e_Imag'] = cat['e_imag']

        # Copies of columns for convenience
        for _ in ['B', 'V', 'R', 'I']:
            cat[_] = cat[_ + 'mag']

    except KeyError as e:
        log(f"Warning: Missing required column for SkyMapper augmentation: {e}")


def _augment_apass(cat, verbose=False):
    """Augment APASS DR9 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Cousins R and I")

        # APASS to Cousins R and I, based on Landolt standards
        cat['Rmag'] = (
            cat['r_mag']
            - 0.152
            - 0.1 * (cat['g_mag'] - cat['r_mag'])
            - 0.003 * (cat['g_mag'] - cat['r_mag']) ** 2
        )

        cat['Imag'] = (
            cat['i_mag']
            - 0.364
            - 0.041 * (cat['g_mag'] - cat['r_mag'])
            - 0.027 * (cat['g_mag'] - cat['r_mag']) ** 2
        )

        cat['e_Rmag'] = cat['e_r_mag']
        cat['e_Imag'] = cat['e_i_mag']

        # Copies of columns for convenience
        for _ in ['B', 'V', 'R', 'I']:
            cat[_] = cat[_ + 'mag']

    except KeyError as e:
        log(f"Warning: Missing required column for APASS augmentation: {e}")


def _augment_gaiadr3syn(cat, verbose=False):
    """Augment Gaia DR3 Synthetic Photometry catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        # Magnitude errors from fluxes
        for band in ['uv', 'g', 'r', 'i', 'z']:
            cat['e_' + band + 'mag'] = 2.5 / np.log(10) * cat['e_F' + band] / cat['F' + band]

        log("Augmenting the catalogue with Pan-STARRS photometry")

        # Transformations from SDSS to PS1, my fit on clean Landolt from arxiv:2205.06186
        pg = [-0.011967078086654664, 0.04633087068066124, 0.008034699768453766]
        pr = [0.028943126754064405, -0.12551605085165994, -0.015663503097695073]
        pi = [0.02574683649882956, -0.08834456918373988, -0.03629206234632685]
        pz = [-0.011485506485056892, 0.04695264394997063, -0.0010936924620734057]

        # Store original SDSS magnitudes
        cat['u_SDSS'] = cat['uvmag']
        cat['g_SDSS'] = cat['gmag']
        cat['r_SDSS'] = cat['rmag']
        cat['i_SDSS'] = cat['imag']
        cat['z_SDSS'] = cat['zmag']

        # Convert to PS1
        cat['gmag'] = cat['g_SDSS'] + np.polyval(pg, cat['g_SDSS'] - cat['r_SDSS'])
        cat['rmag'] = cat['r_SDSS'] + np.polyval(pr, cat['g_SDSS'] - cat['r_SDSS'])
        cat['imag'] = cat['i_SDSS'] + np.polyval(pi, cat['g_SDSS'] - cat['r_SDSS'])
        cat['zmag'] = cat['z_SDSS'] + np.polyval(pz, cat['g_SDSS'] - cat['r_SDSS'])

    except KeyError as e:
        log(f"Warning: Missing required column for Gaia DR3 Syn augmentation: {e}")


def _augment_sdss(cat, verbose=False):
    """Augment SDSS DR16 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with AB magnitude corrections")

        # AB magnitude corrections
        cat['umag'] -= 0.04
        cat['zmag'] += 0.02

        # Store as SDSS magnitudes
        cat['u_SDSS'] = cat['umag']
        cat['g_SDSS'] = cat['gmag']
        cat['r_SDSS'] = cat['rmag']
        cat['i_SDSS'] = cat['imag']
        cat['z_SDSS'] = cat['zmag']

    except KeyError as e:
        log(f"Warning: Missing required column for SDSS augmentation: {e}")


def augment_cat_bands(cat, catalog=None, verbose=False):
    """
    Augment catalog with additional photometric bands via magnitude conversions.

    Parameters
    ----------
    cat : astropy.table.Table
        Input catalog to augment. Modified in-place.
    catalog : str, optional
        Catalog type: 'ps1', 'atlas', 'gaiadr2', 'skymapper', 'apass',
        'gaiadr3syn', 'sdss'. If None, auto-detect from columns.
    verbose : bool or callable, optional
        Logging control.

    Returns
    -------
    cat : astropy.table.Table
        The input catalog with additional magnitude columns.
    """
    # Setup logging
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Auto-detect if not specified
    if catalog is None:
        catalog = _detect_catalog_type(cat)
        if catalog is None:
            log("Warning: Could not detect catalog type. Skipping augmentation.")
            return cat
        log(f"Auto-detected catalog type: {catalog}")

    # Dispatch to appropriate helper
    if catalog == 'ps1' or catalog == 'atlas':
        _augment_ps1(cat, verbose=verbose)
    elif catalog == 'gaiadr2':
        _augment_gaiadr2(cat, verbose=verbose)
    elif catalog == 'skymapper':
        _augment_skymapper(cat, verbose=verbose)
    elif catalog == 'apass':
        _augment_apass(cat, verbose=verbose)
    elif catalog == 'gaiadr3syn':
        _augment_gaiadr3syn(cat, verbose=verbose)
    elif catalog == 'sdss':
        _augment_sdss(cat, verbose=verbose)
    else:
        log(f"Warning: Unknown catalog type '{catalog}'. Skipping augmentation.")

    return cat


def get_cat_vizier(
    ra0, dec0, sr0, catalog='ps1', limit=-1, filters={}, extra=[],
    get_distance=False, augment_bands=True, verbose=False
):
    """Download any catalogue from Vizier.

    The catalogue may be anything recognizable by Vizier. For some most popular ones, we have additional support - we try to augment them with photometric measurements not originally present there, based on some analytical magnitude conversion formulae. These catalogues are:

    -  ps1 - Pan-STARRS DR1. We augment it with Johnson-Cousins B, V, R and I magnitudes
    -  gaiadr2 - Gaia DR2. We augment it  with Johnson-Cousins B, V, R and I magnitudes, as well as Pan-STARRS and SDSS ones
    -  gaiaedr3 - Gaia eDR3
    -  gaiadr3syn - Gaia DR3 synthetic photometry based on XP spectra
    -  skymapper - SkyMapper DR1.1
    -  vsx - AAVSO Variable Stars Index
    -  apass - AAVSO APASS DR9
    -  sdss - SDSS DR16
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
    :param get_distance: If set, the distance from the field center will be returned in `_r` column.
    :param augment_bands: If set (default), will augment the photometry in bands not originally present there using conversion formulae.
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :returns: astropy.table.Table with catalogue as returned by Vizier, with some additional columns added for supported catalogues.
    """

    # Simple Wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    # TODO: add positional errors

    if catalog in catalogs:
        # For some catalogs we may have some additional information
        vizier_id = catalogs.get(catalog).get('vizier')
        name = catalogs.get(catalog).get('name')

        columns = (
            ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000']
            + extra
            + catalogs.get(catalog).get('extra', [])
        )
    else:
        vizier_id = catalog
        name = catalog
        columns = ['*', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'] + extra

    log('Requesting from VizieR:', vizier_id, 'columns:', columns)
    log('Center: %.3f %.3f' % (ra0, dec0), 'radius: %.3f' % sr0)
    log('Filters:', filters)

    vizier = Vizier(row_limit=limit, columns=columns, column_filters=filters)
    cats = vizier.query_region(
        SkyCoord(ra0, dec0, unit='deg'), radius=sr0 * u.deg, catalog=vizier_id
    )

    if not cats or not len(cats) == 1:
        cats = vizier.query_region(
            SkyCoord(ra0, dec0, unit='deg'),
            radius=sr0 * u.deg,
            catalog=vizier_id,
            cache=False,
        )

        if not cats or not len(cats) == 1:
            log('Error requesting catalogue', catalog)
            return None

    cat = cats[0]
    cat.meta['vizier_id'] = vizier_id
    cat.meta['name'] = name

    log('Got', len(cat), 'entries with', len(cat.colnames), 'columns')

    # Fix _RAJ2000/_DEJ2000
    if (
        '_RAJ2000' in cat.keys()
        and '_DEJ2000' in cat.keys()
        and not 'RAJ2000' in cat.keys()
    ):
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    if get_distance and 'RAJ2000' in cat.colnames and 'DEJ2000' in cat.colnames:
        log("Augmenting the catalogue with distances from field center")
        cat['_r'] = astrometry.spherical_distance(ra0, dec0, cat['RAJ2000'], cat['DEJ2000'])

    if augment_bands or True:
        # Augment catalogue with additional bandpasses
        augment_cat_bands(cat, catalog=catalog if catalog in catalogs else None, verbose=verbose)

    return cat


def xmatch_objects(obj, catalog='ps1', sr=3 / 3600, col_ra='ra', col_dec='dec', augment_bands=True):
    """Cross-match object list with Vizier catalogue using CDS XMatch service.

    Any Vizier catalogue may be used for cross-matching.

    :param obj: astropy.table.Table with objects
    :param catalog: Any Vizier catalogue identifier, or a catalogue short name
    :param sr: Cross-matching radius in degrees
    :param col_ra: Column name in `obj` table containing Right Ascension values
    :param col_dec: Column name in `obj` table containing Declination values
    :param augment_bands: If set (default), will augment the photometry in bands not originally present there using conversion formulae.
    :returns: The table of matched objects augmented with some fields from the Vizier catalogue.
    """

    if catalog in catalogs:
        vizier_id = catalogs.get(catalog)['vizier']
    else:
        vizier_id = catalog

    xcat = XMatch().query(
        cat1=obj,
        cat2='vizier:' + vizier_id,
        max_distance=sr * u.deg,
        colRA1=col_ra,
        colDec1=col_dec,
    )

    if augment_bands:
        # Augment catalogue with additional bandpasses
        augment_cat_bands(xcat, catalog=catalog if catalog in catalogs else None)

    return xcat


def xmatch_skybot(
    obj, sr=10 / 3600, time=None, col_ra='ra', col_dec='dec', col_id='id'
):
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

    ra0, dec0, sr0 = astrometry.get_objects_center(obj)

    try:
        # Query SkyBot for (a bit larger than) our FOV at our time
        xcat = Skybot.cone_search(
            SkyCoord(ra0, dec0, unit='deg'), (sr0 + 2.0 * sr) * u.deg, Time(time)
        )
    except (RuntimeError, KeyError, ConnectionError, OSError):
        # Nothing found in SkyBot
        return None

    if xcat is None or len(xcat) == 0:
        return None

    # Cross-match objects
    oidx, cidx, dist = astrometry.spherical_match(
        obj[col_ra], obj[col_dec], xcat['RA'], xcat['DEC'], 10 / 3600
    )

    # Annotate the table with id from objects so that it is possible to identify the matches
    xcat[col_id] = MaskedColumn(len(xcat), dtype=np.dtype(obj[col_id][0]))
    xcat[col_id].mask = True
    xcat[col_id][cidx] = obj[col_id][oidx]
    xcat[col_id][cidx].mask = False

    return xcat


def xmatch_ned(obj, sr=3 / 3600, col_ra='ra', col_dec='dec', col_id='id'):
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
        res = Ned().query_region(
            SkyCoord(row[col_ra], row[col_dec], unit='deg'), sr * u.deg
        )

        if res:
            res['id'] = row[col_id]

            xcat.append(res)

    if xcat:
        xcat = vstack(xcat, join_type='exact', metadata_conflicts='silent')

    return xcat
