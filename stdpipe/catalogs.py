"""
Module containing the routines for handling various online catalogues.
"""

import io, os, posixpath, re, shutil, tempfile, time, warnings
import numpy as np

from astropy.table import Table, MaskedColumn, vstack
from astropy.io.votable import parse as parse_votable
import requests

from astroquery.vizier import Vizier
from astroquery.mast import CatalogsClass
from astroquery.exceptions import TableParseError, TimeoutError as AstroqueryTimeoutError
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
    'gaiadr3': {'vizier': 'I/355/gaiadr3', 'name': 'Gaia DR3'},
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
    '2mass': {
        'vizier': 'II/246/out',
        'name': '2MASS PSC',
    },
    'vhs': {
        'vizier': 'II/367/vhs_dr5',
        'name': 'Vista Hemisphere Survey DR5',
        'extra': [
            'e_Jap3',
            'e_Hap3',
            'e_Ksap3',
        ]
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
        Detected catalog type ('ps1', 'gaiadr2', 'gaiaedr3', 'skymapper',
        'apass', 'gaiadr3syn', 'sdss', '2mass', 'vhs') or None if cannot detect
    """
    colnames = set(cat.colnames)

    # Check most specific patterns first to avoid ambiguity

    # Gaia DR3 synthetic photometry - has Fu, Fg, Fr, Fi, Fz flux columns
    if {'Fu', 'Fg', 'Fr', 'Fi', 'Fz'}.issubset(colnames):
        return 'gaiadr3syn'

    # Gaia EDR3 - same magnitudes as DR2, so has to be checked first. RUWE and
    # the corrected G magnitude are not present in DR2
    if {'Gmag', 'BPmag', 'RPmag'}.issubset(colnames) and {'RUWE', 'GmagCorr'} & colnames:
        return 'gaiaedr3'

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

    # 2MASS
    if {'Jmag', 'Hmag', 'Kmag'}.issubset(colnames):
        return '2mass'

    # VHS
    if {'Jap3', 'Hap3', 'Ksap3'}.issubset(colnames):
        return 'vhs'

    # Could not detect
    return None


def _augment_ps1(cat, verbose=False):
    """Augment Pan-STARRS DR1 or ATLAS-RefCat2 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Johnson-Cousins photometry using PS1 mags")

        # Alternative PS1 transfromation from https://scixplorer.org/abs/2018BlgAJ..28....3K, Stetson, seems better with Landolt than official one
        # cat['B'] = cat['gmag'] + 0.199 + 0.540*(cat['gmag'] - cat['rmag']) + 0.016*(cat['gmag'] - cat['rmag'])**2
        # cat['V'] = cat['gmag'] - 0.020 - 0.498*(cat['gmag'] - cat['rmag']) - 0.008*(cat['gmag'] - cat['rmag'])**2
        # cat['R'] = cat['rmag'] - 0.163 - 0.086*(cat['gmag'] - cat['rmag']) - 0.061*(cat['gmag'] - cat['rmag'])**2
        # cat['I'] = cat['imag'] - 0.387 - 0.123*(cat['gmag'] - cat['rmag']) - 0.034*(cat['gmag'] - cat['rmag'])**2

        # My own fit on Landolt+Stetson standards from https://scixplorer.org/abs/2022A%26A...664A.109P
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
                -0.19509051168348646,
            ],
            [
                0.014586929059030904,
                -0.025228407778416825,
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

        cat['good'] = (cat['gmag'] - cat['rmag'] > -0.5) & (cat['gmag'] - cat['rmag'] < 2.5)
        cat['good'] &= (cat['rmag'] - cat['imag'] > -0.5) & (cat['rmag'] - cat['imag'] < 2.0)
        cat['good'] &= (cat['imag'] - cat['zmag'] > -0.5) & (cat['imag'] - cat['zmag'] < 1.0)

        # to SDSS, zero points and color terms from https://scixplorer.org/abs/2012ApJ...750...99T
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
        log(f"Warning: Missing required column for PS1 augmentation: {e}")


def _augment_gaiadr2(cat, verbose=False):
    """Augment Gaia DR2 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Johnson-Cousins photometry using Gaia DR2 mags")

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

        # phot_bp_rp_excess_factor == E(BR/RP). VizieR sends this field with no
        # VOTable ID, so astroquery < 0.4.8, which used the IDs astropy
        # synthesizes from the names, saw it as E_BR_RP_. Since 0.4.8 (#3153) it
        # uses the names, so accept both spellings.
        for _ in ['E(BR/RP)', 'E_BR_RP_']:
            if _ in cat.colnames:
                excess = cat[_]
                break
        else:
            raise KeyError('E(BR/RP)')

        Cstar = excess - np.polyval(
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
        gcorr[(g > 6) & (g < 16)] = g[(g > 6) & (g < 16)] - 0.0032 * (g[(g > 6) & (g < 16)] - 6)
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
        cat['g_SDSS'] = g - (0.13518 - 0.46245 * bp_rp - 0.25171 * bp_rp**2 + 0.021349 * bp_rp**3)
        cat['r_SDSS'] = g - (-0.12879 + 0.24662 * bp_rp - 0.027464 * bp_rp**2 - 0.049465 * bp_rp**3)
        cat['i_SDSS'] = g - (-0.29676 + 0.64728 * bp_rp - 0.10141 * bp_rp**2)

    except KeyError as e:
        log(f"Warning: Missing required column for Gaia DR2 augmentation: {e}")


def _augment_skymapper(cat, verbose=False):
    """Augment SkyMapper DR4 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with Pan-STARRS photometry using SkyMapper DR4 mags")

        # SkyMapper DR4 to PS1 griz, my fit based on Pancino et al. (2022)
        pg1, pg2 = (
            [-0.07715320986152466, 0.2694597282089696, 0.04069379065128178, 0.01396290714542747],
            [0.026097008342026252, -0.14040957287568073, 0.133647539780504, 0.013962907145427432],
        )
        pr1, pr2 = (
            [0.08779280979185472, -0.23257704629617004, 0.1890698144343673, -0.008125550119663026],
            [-0.06273832689338121, 0.21909317812693613, -0.23340488268623696, -0.00812555011966309],
        )
        pi1, pi2 = (
            [
                0.03553380678975111,
                -0.021174189684500792,
                -0.028159666883815007,
                0.0009748746568893062,
            ],
            [0.00911922467970264, -0.0362286983251751, 0.1403094994141109, 0.0009748746568892609],
        )
        pz1, pz2 = (
            [
                0.08071260245520126,
                -0.051693023216670575,
                -0.0739439627982131,
                -0.0020460270205769223,
            ],
            [
                0.09720715174271254,
                -0.32063637962189184,
                0.37918283208242526,
                -0.0020460270205769305,
            ],
        )
        py1, py2 = (
            [0.038781034592287725, -0.11040188064275973, 0.08235396198116865, 0.006980454415779221],
            [-0.0649739656901001, 0.205320995228645, -0.28233276303592, 0.006980454415779424],
        )

        cat['gmag'] = (
            cat['gPSF']
            + np.polyval(pg1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pg2, cat['rPSF'] - cat['iPSF'])
        )
        cat['rmag'] = (
            cat['rPSF']
            + np.polyval(pr1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pr2, cat['rPSF'] - cat['iPSF'])
        )
        cat['imag'] = (
            cat['iPSF']
            + np.polyval(pi1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pi2, cat['rPSF'] - cat['iPSF'])
        )
        cat['zmag'] = (
            cat['zPSF']
            + np.polyval(pz1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pz2, cat['rPSF'] - cat['iPSF'])
        )
        cat['ymag'] = (
            cat['zPSF']
            + np.polyval(py1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(py2, cat['rPSF'] - cat['iPSF'])
        )

        for _ in ['g', 'r', 'i', 'z']:
            cat['e_' + _ + 'mag'] = cat['e_' + _ + 'PSF']
        cat['e_ymag'] = cat['e_zPSF']

        log("Augmenting the catalogue with Johnson-Cousins photometry using SkyMapper DR4 mags")

        # SkyMapper DR4 to Johnson-Cousins BVRI, my fit based on Pancino et al. (2022)
        pB1, pB2 = (
            [-0.22773918482205113, 0.1818124624962873, 1.0021365492384895, 0.10762635377473588],
            [-0.004034933919297649, 0.08214592357213418, -0.07535454054888649, 0.10762635377473558],
        )
        pV1, pV2 = (
            [
                -0.02545732895304914,
                0.03256423830249228,
                -0.33074199873567045,
                -0.002938730214382037,
            ],
            [-0.007342074336918033, 0.08255055271047995, -0.14349325478829064, -0.0029387302143822],
        )
        pR1, pR2 = (
            [0.07296699439306827, -0.1943702618426095, 0.15375263988851387, -0.08547735652048871],
            [-0.07378125129406726, 0.18462924970775316, -0.40720945890364135, -0.08547735652048903],
        )
        pI1, pI2 = (
            [
                -0.00925391710305653,
                0.046223960182760516,
                -0.06889215990613289,
                -0.19321699685334734,
            ],
            [
                0.01197866152020802,
                -0.044370062623186206,
                -0.05231484699406009,
                -0.19321699685334745,
            ],
        )

        cat['Bmag'] = (
            cat['gPSF']
            + np.polyval(pB1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pB2, cat['rPSF'] - cat['iPSF'])
        )
        cat['Vmag'] = (
            cat['gPSF']
            + np.polyval(pV1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pV2, cat['rPSF'] - cat['iPSF'])
        )
        cat['Rmag'] = (
            cat['rPSF']
            + np.polyval(pR1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pR2, cat['rPSF'] - cat['iPSF'])
        )
        cat['Imag'] = (
            cat['iPSF']
            + np.polyval(pI1, cat['gPSF'] - cat['rPSF'])
            + np.polyval(pI2, cat['rPSF'] - cat['iPSF'])
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
        log("Augmenting the catalogue with Cousins R and I using APASS mags")

        # My own fit based on Landolt standards
        cat['Rmag'] = (
            cat['r_mag']
            - 0.157
            - 0.087 * (cat['g_mag'] - cat['r_mag'])
            - 0.014 * (cat['g_mag'] - cat['r_mag']) ** 2
        )
        cat['e_Rmag'] = cat['e_r_mag']

        cat['Imag'] = (
            cat['i_mag']
            - 0.354
            - 0.118 * (cat['g_mag'] - cat['r_mag'])
            - 0.004 * (cat['g_mag'] - cat['r_mag']) ** 2
        )
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
        # Compute magnitude errors from flux errors
        for name in ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'y']:
            cat['e_' + name + 'mag'] = 2.5 / np.log(10) * cat['e_F' + name] / cat['F' + name]

        log("Converting the catalogue Sloan magnitudes to Pan-STARRS ones")

        # umag, gmag, rmag, imag and zmag are Sloan ugriz magnitudes! Let's get PS1 ones instead
        # Fits are based on clean Landolt sample from https://scixplorer.org/abs/2022A%26A...664A.109P
        pg = [-0.030414391501015867, -0.09960002492299584, -0.002910024005294562]
        pr = [-0.009566553708653305, 0.014924591443344211, -0.003928147919030857]
        pi = [-0.010802807724098494, 0.01124900218746879, 0.01274293783734852]
        pz = [-0.0031896767661109523, 0.06537983414287968, 0.007695587806229381]

        for _ in ['u', 'g', 'r', 'i', 'z']:
            cat[_ + '_SDSS'] = cat[_ + 'mag']

        cat['gmag'] = cat['g_SDSS'] + np.polyval(pg, cat['g_SDSS'] - cat['r_SDSS'])
        cat['rmag'] = cat['r_SDSS'] + np.polyval(pr, cat['g_SDSS'] - cat['r_SDSS'])
        cat['imag'] = cat['i_SDSS'] + np.polyval(pi, cat['g_SDSS'] - cat['r_SDSS'])
        cat['zmag'] = cat['z_SDSS'] + np.polyval(pz, cat['g_SDSS'] - cat['r_SDSS'])

    except KeyError as e:
        log(f"Warning: Missing required column for Gaia DR3 synphot augmentation: {e}")


def _augment_sdss(cat, verbose=False):
    """Augment SDSS DR16 catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Converting the catalogue Sloan magnitudes to AB ones")

        # Zero point biases from https://www.sdss4.org/dr16/algorithms/fluxcal/#SDSStoAB
        cat['umag'] -= 0.04
        cat['zmag'] += 0.02

        for _ in ['u', 'g', 'r', 'i', 'z']:
            cat[_ + '_SDSS'] = cat[_ + 'mag']

    except KeyError as e:
        log(f"Warning: Missing required column for SDSS augmentation: {e}")

def _augment_2mass(cat, verbose=False):
    """Augment 2MASS catalog with additional bands."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Renaming the catalogue K magnitudes to Ks ones")

        cat['Ksmag'] = cat['Kmag']
        cat['e_Ksmag'] = cat['e_Kmag']

    except KeyError as e:
        log(f"Warning: Missing required column for 2MASS augmentation: {e}")


def _augment_vhs(cat, verbose=False):
    """Augment VHS catalog with 2MASS-system JHKs magnitudes."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log("Augmenting the catalogue with 2MASS magnitudes")

        for _ in ['J', 'H', 'Ks']:
            cat[f"{_}mag"] = cat[f"{_}ap3"]
            cat[f"e_{_}mag"] = cat[f"e_{_}ap3"]

        def color_term(a, b, coeff):
            # Color-term correction that degrades gracefully to zero (i.e. keeps the
            # uncorrected VISTA magnitude) where the auxiliary band is missing, so that
            # e.g. a field lacking H-band coverage does not wipe out the J magnitudes.
            term = coeff * (np.ma.filled(cat[a].astype(float), np.nan)
                            - np.ma.filled(cat[b].astype(float), np.nan))
            n_bad = int(np.count_nonzero(np.isnan(term)))
            if n_bad:
                log(f"Warning: {n_bad}/{len(cat)} stars lack {a} or {b}; "
                    f"their magnitudes are left uncorrected for the {a}-{b} colour term")
            return np.nan_to_num(term, nan=0.0)

        # VISTA to 2MASS equations from https://scixplorer.org/abs/2013A%26A...552A.101S
        cat['Jmag'] += color_term('Jap3', 'Hap3', 0.070)
        cat['Hmag'] += color_term('Hap3', 'Ksap3', -0.035)
        cat['Ksmag'] += color_term('Jap3', 'Ksap3', -0.011)

    except KeyError as e:
        log(f"Warning: Missing required column for VHS augmentation: {e}")


def augment_cat_bands(cat, catalog=None, verbose=False):
    """
    Augment catalog with additional photometric bands via magnitude conversions.

    Parameters
    ----------
    cat : astropy.table.Table
        Input catalog to augment. Modified in-place.
    catalog : str, optional
        Catalog type. Magnitude conversions are defined for 'ps1', 'atlas',
        'gaiadr2', 'skymapper', 'apass', 'gaiadr3syn', 'sdss', '2mass' and 'vhs';
        other catalogues known to :data:`catalogs` (e.g. 'gaiaedr3', 'usnob1',
        'gsc', 'vsx') are returned unchanged. If None, auto-detect from columns.
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
    elif catalog == '2mass':
        _augment_2mass(cat, verbose=verbose)
    elif catalog == 'vhs':
        _augment_vhs(cat, verbose=verbose)
    elif catalog in catalogs:
        # Known catalogue, we just have no magnitude conversions for it
        log(
            f"No magnitude conversions defined for '{catalog}'"
            f" ({catalogs[catalog]['name']}), skipping augmentation"
        )
    else:
        log(f"Warning: Unknown catalog type '{catalog}'. Skipping augmentation.")

    return cat


# VizieR servers tried, in order, when the default one (astroquery.vizier.conf.server) fails.
# Mirrors may lack data for some catalogues and then report QUERY_STATUS="ERROR" (e.g. CfA has
# no Gaia DR3 synthetic photometry, IDIA lacks PS1, Gaia DR2/EDR3, ATLAS-RefCat2, SkyMapper, VHS),
# or answer cleanly with no rows (e.g. CfA for I/355/gaiadr3), so the next one is tried in
# both cases. Only an empty answer from the default server is trusted as an empty field
vizier_mirrors = ['vizier.cfa.harvard.edu', 'vizier.idia.ac.za']


class VizierQueryError(Exception):
    """VizieR answered, but reported the query as failed."""


def _query_vizier_region(vizier, coord, radius, catalog, cache=True):
    """Region query on Vizier that fails if the server reports an error inside the VOTable.

    astroquery parses a VOTable with QUERY_STATUS="ERROR" (e.g. "NOT AVAILABLE YET" from a
    mirror lacking the catalogue data) into an empty table list, indistinguishable from an
    empty field, so we fetch the raw response and check its status before parsing. Some
    error replies carry only ``<INFO name="Error">`` elements without QUERY_STATUS, so these
    are treated as errors too.

    A result truncated at the row limit is reported as QUERY_STATUS="OVERFLOW"; for large
    catalogues some servers (e.g. IDIA) then also report "(pipe) Request has been interrupted"
    as an ERROR, since the extraction process is killed once enough rows are written. Such a
    response still holds the requested rows, so it is accepted.
    """
    response = vizier.query_region_async(coord, radius=radius, catalog=catalog, cache=cache)

    try:
        response.raise_for_status()

        errors = re.findall(rb'<INFO\b[^>]*\bname="Error"[^>]*\bvalue="([^"]*)"', response.content)
        status_error = re.search(
            rb'<INFO\b[^>]*\bname="QUERY_STATUS"[^>]*\bvalue="ERROR"', response.content
        )
        overflow = re.search(
            rb'<INFO\b[^>]*\bname="QUERY_STATUS"[^>]*\bvalue="OVERFLOW"', response.content
        )

        if (errors or status_error) and not overflow:
            errors = list(dict.fromkeys(_.decode(errors='replace') for _ in errors))
            raise VizierQueryError('; '.join(errors) or 'QUERY_STATUS is ERROR')

        cats = vizier._parse_result(response)
        if cats is None:
            raise VizierQueryError('Unrecognized response format')
    except (requests.exceptions.HTTPError, VizierQueryError, TableParseError):
        # Do not keep the failed answer in the cache
        try:
            vizier._last_query.remove_cache_file(vizier.cache_location)
        except (AttributeError, OSError):
            pass
        raise

    return cats


_FILTER_NUM = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
_FILTER_RE = re.compile(r'^\s*(<=|>=|==|!=|<|>)?\s*(' + _FILTER_NUM + r')\s*$')
_FILTER_RANGE_RE = re.compile(r'^\s*(' + _FILTER_NUM + r')\s*\.\.\s*(' + _FILTER_NUM + r')\s*$')


def _parse_filter(expr):
    """Parse a simple VizieR numeric filter expression into a list of
    ``(op, value)`` conditions, all of which must hold.

    Supports single comparisons (``'<20'``, ``'>=15'``, ``'==3'``, bare ``'3'``)
    and inclusive ranges (``'15..20'``). Returns None if ``expr`` is not one
    of these forms. ``value`` is kept as the original string.
    """
    expr = str(expr)

    m = _FILTER_RANGE_RE.match(expr)
    if m:
        return [('>=', m.group(1)), ('<=', m.group(2))]

    m = _FILTER_RE.match(expr)
    if m:
        return [(m.group(1) or '==', m.group(2))]

    return None


# VizieR filter operators -> ADQL
_ADQL_OPS = {'==': '=', '!=': '<>'}


def _adql_conditions(column, expr):
    """ADQL ``WHERE`` terms for a simple VizieR filter on ``column``, or an
    empty list if ``expr`` cannot be translated."""
    return [f"{column} {_ADQL_OPS.get(op, op)} {value}" for op, value in _parse_filter(expr) or []]

# Page size and maximal number of pages for the paginated MAST fallback, and
# the row cap for the SkyMapper one. Hitting a cap truncates coverage, so it is
# checked for and logged.
_MAST_PAGE_SIZE = 50000
_MAST_MAX_PAGES = 20
_SKYMAPPER_ROW_LIMIT = 100000


def _coerce_numeric(column):
    """Cast a column to float, masking entries that are missing or cannot be
    parsed as a number."""
    try:
        return np.ma.masked_invalid(np.ma.asarray(column, dtype=float))
    except (TypeError, ValueError):
        pass

    values = np.full(len(column), np.nan)
    mask = np.zeros(len(column), dtype=bool)
    for i, v in enumerate(column):
        if v is np.ma.masked or v is None:
            mask[i] = True
            continue
        try:
            values[i] = float(v)
        except (TypeError, ValueError):
            mask[i] = True
    return np.ma.MaskedArray(values, mask=mask)


def _apply_filters(cat, filters, verbose=False):
    """Apply simple VizieR-style numeric column filters (e.g. ``'<20'`` or
    ``'15..20'``) client-side.

    Note: VizieR applies ``filters`` server-side via ``column_filters``
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    ops = {
        '<': np.less, '<=': np.less_equal,
        '>': np.greater, '>=': np.greater_equal,
        '==': np.equal, '!=': np.not_equal,
    }

    for col, expr in (filters or {}).items():
        if col not in cat.colnames:
            log(f"Warning: filter column {col!r} is not in the fallback catalogue, cannot apply {col}{expr!r}")
            continue

        conditions = _parse_filter(expr)
        if not conditions:
            log(f"Warning: cannot apply filter {col}{expr!r} to fallback catalogue, skipping")
            continue

        try:
            column = np.ma.filled(np.ma.asarray(cat[col], dtype=float), np.nan)
        except (TypeError, ValueError):
            log(f"Warning: column {col!r} is not numeric, cannot apply filter {col}{expr!r}")
            continue

        idx = np.ones(len(cat), dtype=bool)
        for op, value in conditions:
            idx &= ops[op](column, float(value))

        cat = cat[idx]

    return cat


def _tap_async_query(base_url, query, timeout=(10, 60), verbose=False):
    """IVOA TAP async (UWS) client: submit a query, poll until it
    completes, and fetch the result.

    Note: unfortunately astroquery and pyvo don't implement a reliable timeout.

    ``timeout`` may be a single value or a ``(connect, read)`` tuple. The 
    longest timeout sets the overall deadline.
    If the query starts towards the end of the timeout, up to 5s extra may be
    spent in attempt to cancel it.
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    connect_cap = timeout[0] if isinstance(timeout, (tuple, list)) else timeout
    total = max(timeout) if isinstance(timeout, (tuple, list)) else timeout
    deadline = time.monotonic() + total

    def remaining():
        return deadline - time.monotonic()

    def request_timeout():
        left = max(remaining(), 0.001)
        return (min(connect_cap, left), left)

    session = requests.Session()
    job_url = None
    try:
        r = session.post(
            f'{base_url}/async',
            data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': query, 'PHASE': 'RUN'},
            timeout=request_timeout(),
            allow_redirects=False,
        )
        r.raise_for_status()
        job_url = r.headers['Location']

        phase = None
        while True:
            if remaining() <= 0:
                raise TimeoutError(f'TAP job did not complete within {total}s')
            r = session.get(f'{job_url}/phase', timeout=request_timeout())
            r.raise_for_status()
            phase = r.text.strip()
            if phase in ('COMPLETED', 'ERROR', 'ABORTED'):
                break
            left = remaining()
            if left <= 0:
                raise TimeoutError(f'TAP job did not complete within {total}s')
            time.sleep(min(1.0, left))  # only sleeps for however much of the budget remains

        if phase != 'COMPLETED':
            try:
                detail = session.get(f'{job_url}/error', timeout=request_timeout()).text[:300]
            except Exception:
                detail = '(could not fetch error details)'
            log(f"Warning: TAP job at {base_url} ended in phase {phase}: {detail}")
            raise RuntimeError(f'TAP job ended in phase {phase}')

        r = session.get(f'{job_url}/results/result', timeout=request_timeout())
        r.raise_for_status()
        return parse_votable(io.BytesIO(r.content)).get_first_table().to_table(use_names_over_ids=True)

    except TimeoutError:
        log(f"Warning: TAP query at {base_url} did not complete within {total}s, giving up")
        if job_url is not None:
            try:
                session.post(f'{job_url}/phase', data={'PHASE': 'ABORT'}, timeout=5)
                log(f"Warning: aborted TAP job {job_url}")
            except Exception as abort_exc:
                log(f"Warning: failed to abort TAP job {job_url}: {str(abort_exc)[:200]}")
        raise
    finally:
        session.close()


def _fallback_ps1(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for Pan-STARRS DR1 via the MAST Catalogs mean-object API."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # VizieR column -> MAST column name, for server-side magnitude-limit filter
    mast_columns = {
        'gmag': 'gMeanPSFMag', 'rmag': 'rMeanPSFMag', 'imag': 'iMeanPSFMag',
        'zmag': 'zMeanPSFMag', 'ymag': 'yMeanPSFMag',
    }
    mast_ops = {'<': 'lt', '<=': 'lte', '>': 'gt', '>=': 'gte', '==': 'eq', '!=': 'neq'}

    params = {
        'ra': ra0, 'dec': dec0, 'radius': sr0,
        # VizieR cuts low detections by default
        'nDetections.gte': 3,
        'pagesize': _MAST_PAGE_SIZE,
    }
    for col, expr in (filters or {}).items():
        mast_col = mast_columns.get(col)
        for op, value in (_parse_filter(expr) or []) if mast_col else []:
            params[f'{mast_col}.{mast_ops[op]}'] = value
            if op in ('<', '<='):
                # MAST encodes a missing value as -999
                params.setdefault(f'{mast_col}.gt', '0')

    # Sorting by the unique objID keeps page offsets stable between requests
    params['sort_by'] = 'objID'

    pages = []
    try:
        log('Trying MAST PanSTARRS DR1 mean-object API as a fallback')
        for page in range(1, _MAST_MAX_PAGES + 1):
            r = requests.get(
                'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv',
                params=dict(params, page=page),
                timeout=timeout,
            )
            r.raise_for_status()
            if not r.text.strip():
                break
            raw = Table.read(io.BytesIO(r.content), format='ascii.csv')
            if len(raw):
                pages.append(raw)
            if len(raw) < params['pagesize']:
                break
        else:
            log(f"Warning: MAST PanSTARRS DR1 fallback hit the {_MAST_MAX_PAGES}-page limit, results may be incomplete")
    except Exception as e:
        # Never return a partial catalogue if a later page fails
        log('MAST PanSTARRS fallback failed:', str(e)[:200])
        return None

    if not pages:
        return None

    raw = vstack(pages, metadata_conflicts='silent') if len(pages) > 1 else pages[0]

    cat = Table()
    cat['RAJ2000'] = raw['raMean']
    cat['DEJ2000'] = raw['decMean']
    cat['e_RAJ2000'] = raw['raMeanErr']
    cat['e_DEJ2000'] = raw['decMeanErr']

    for band in ['g', 'r', 'i', 'z', 'y']:
        mag = _coerce_numeric(raw[band + 'MeanPSFMag'])
        magerr = _coerce_numeric(raw[band + 'MeanPSFMagErr'])
        # MAST encodes a missing value as -999
        cat[band + 'mag'] = np.ma.masked_equal(mag, -999.0)
        cat['e_' + band + 'mag'] = np.ma.masked_equal(magerr, -999.0)

    cat.meta['source'] = 'MAST PanSTARRS DR1 mean-object API (fallback)'
    return cat


_GAIA_TAP_URL = 'https://gea.esac.esa.int/tap-server/tap'


def _fallback_gaiaedr3(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for Gaia eDR3 via the ESA Gaia archive TAP service (DR3 astrometry,
    including early)."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    gaia_columns = {'Gmag': 'phot_g_mean_mag', 'BPmag': 'phot_bp_mean_mag', 'RPmag': 'phot_rp_mean_mag'}
    conditions = [f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0}, {dec0}, {sr0}))"]
    for col, expr in (filters or {}).items():
        gaia_col = gaia_columns.get(col)
        if gaia_col:
            conditions += _adql_conditions(gaia_col, expr)

    try:
        log('Trying ESA Gaia archive (DR3 astrometry) as a fallback for Gaia eDR3')
        # EPOCH_PROP_POS to get positions as of year 2000
        query = f"""
            SELECT source_id, ra_error, dec_error,
                   parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
                   COORD1(EPOCH_PROP_POS(ra, dec, COALESCE(parallax, 0), COALESCE(pmra, 0),
                          COALESCE(pmdec, 0), COALESCE(radial_velocity, 0), ref_epoch, 2000)) AS ra_j2000,
                   COORD2(EPOCH_PROP_POS(ra, dec, COALESCE(parallax, 0), COALESCE(pmra, 0),
                          COALESCE(pmdec, 0), COALESCE(radial_velocity, 0), ref_epoch, 2000)) AS dec_j2000,
                   phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
                   phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error,
                   phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error,
                   ruwe
            FROM gaiadr3.gaia_source
            WHERE {' AND '.join(conditions)}
        """
        raw = _tap_async_query(_GAIA_TAP_URL, query, timeout=timeout, verbose=verbose)
    except Exception as e:
        log('ESA Gaia archive fallback failed:', str(e)[:200])
        return None

    if not len(raw):
        return None

    cat = Table()
    cat['RAJ2000'] = raw['ra_j2000']
    cat['DEJ2000'] = raw['dec_j2000']
    cat['e_RAJ2000'] = raw['ra_error']
    cat['e_DEJ2000'] = raw['dec_error']
    cat['pmRA'] = raw['pmra']
    cat['e_pmRA'] = raw['pmra_error']
    cat['pmDE'] = raw['pmdec']
    cat['e_pmDE'] = raw['pmdec_error']
    cat['Plx'] = raw['parallax']
    cat['e_Plx'] = raw['parallax_error']
    cat['Gmag'] = raw['phot_g_mean_mag']
    cat['BPmag'] = raw['phot_bp_mean_mag']
    cat['RPmag'] = raw['phot_rp_mean_mag']
    cat['RUWE'] = raw['ruwe']

    # Gaia does not publish a magnitude error directly; derive it from the flux error.
    mag_err_factor = 2.5 / np.log(10)
    for mag_err_col, flux_col, flux_err_col in [
        ('e_Gmag', 'phot_g_mean_flux', 'phot_g_mean_flux_error'),
        ('e_BPmag', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error'),
        ('e_RPmag', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error'),
    ]:
        cat[mag_err_col] = mag_err_factor * np.asarray(raw[flux_err_col], dtype=float) / np.asarray(raw[flux_col], dtype=float)

    cat.meta['source'] = 'ESA Gaia archive DR3 astrometry (fallback for eDR3)'
    return cat


def _fallback_gaiadr3syn(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for Gaia DR3 synthetic photometry via the ESA Gaia archive TAP
    service, joining the main source table with ``synthetic_photometry_gspc``.

    Unlike ``I/350/gaiaedr3``, VizieR's ``I/360/syntphot`` does not propagate
    positions to J2000 (its ``_RAJ2000``/``_DEJ2000`` are identical to
    ``RA_ICRS``/``DE_ICRS``).
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # VizieR band name -> ESA synthetic_photometry_gspc column prefix
    bands = {
        'U': 'u_jkc', 'B': 'b_jkc', 'V': 'v_jkc', 'R': 'r_jkc', 'I': 'i_jkc',
        'u': 'u_sdss', 'g': 'g_sdss', 'r': 'r_sdss', 'i': 'i_sdss', 'z': 'z_sdss',
        'y': 'y_ps1',
    }

    conditions = [
        f"1=CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra0}, {dec0}, {sr0}))"
    ]
    for col, expr in (filters or {}).items():
        band = bands.get(col[:-3]) if col.endswith('mag') else None
        if band:
            conditions += _adql_conditions(f"s.{band}_mag", expr)

    try:
        log('Trying ESA Gaia archive (DR3 synthetic photometry) as a fallback')
        select_cols = ', '.join(
            f's.{col}_mag, s.{col}_flux, s.{col}_flux_error' for col in bands.values()
        )
        query = f"""
            SELECT g.source_id, g.ra, g.dec, {select_cols}
            FROM gaiadr3.gaia_source AS g
            JOIN gaiadr3.synthetic_photometry_gspc AS s ON g.source_id = s.source_id
            WHERE {' AND '.join(conditions)}
        """
        raw = _tap_async_query(_GAIA_TAP_URL, query, timeout=timeout, verbose=verbose)
    except Exception as e:
        log('ESA Gaia synthetic photometry fallback failed:', str(e)[:200])
        return None

    if not len(raw):
        return None

    cat = Table()
    cat['RAJ2000'] = raw['ra']
    cat['DEJ2000'] = raw['dec']

    for short, col in bands.items():
        cat[short + 'mag'] = raw[col + '_mag']
        cat['F' + short] = raw[col + '_flux']
        cat['e_F' + short] = raw[col + '_flux_error']

    cat.meta['source'] = 'ESA Gaia archive DR3 synthetic photometry (fallback)'
    return cat


def _fallback_skymapper(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for SkyMapper DR4 via its public IVOA TAP service."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    sm_columns = {
        'uPSF': 'u_psf', 'vPSF': 'v_psf', 'gPSF': 'g_psf', 'rPSF': 'r_psf',
        'iPSF': 'i_psf', 'zPSF': 'z_psf',
    }
    conditions = [
        f"1=CONTAINS(POINT('ICRS', raj2000, dej2000), CIRCLE('ICRS', {ra0}, {dec0}, {sr0}))"
    ]
    for col, expr in (filters or {}).items():
        sm_col = sm_columns.get(col)
        if sm_col:
            conditions += _adql_conditions(sm_col, expr)

    try:
        log('Trying SkyMapper TAP service as a fallback')
        query = f"""
            SELECT TOP {_SKYMAPPER_ROW_LIMIT}
                   raj2000, dej2000, e_raj2000, e_dej2000, flags, class_star,
                   u_psf, e_u_psf, v_psf, e_v_psf,
                   g_psf, e_g_psf, r_psf, e_r_psf, i_psf, e_i_psf, z_psf, e_z_psf
            FROM dr4.master
            WHERE {' AND '.join(conditions)}
        """
        r = requests.get(
            'https://api.skymapper.nci.org.au/public/tap/sync',
            params={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': query},
            timeout=timeout,
        )
        r.raise_for_status()
        if not r.text.strip():
            return None
        raw = Table.read(io.BytesIO(r.content), format='ascii.csv')
    except Exception as e:
        log('SkyMapper TAP fallback failed:', str(e)[:200])
        return None

    if not len(raw):
        return None

    if len(raw) >= _SKYMAPPER_ROW_LIMIT:
        log(f"Warning: SkyMapper TAP fallback hit the {_SKYMAPPER_ROW_LIMIT}-row limit, results may be incomplete")

    cat = Table()
    cat['RAJ2000'] = raw['raj2000']
    cat['DEJ2000'] = raw['dej2000']
    cat['e_RAJ2000'] = raw['e_raj2000']
    cat['e_DEJ2000'] = raw['e_dej2000']
    # Detection/classification flags, mapped to VizieR II/379 names (confirmed
    # against real cached VizieR data).
    cat['flags'] = _coerce_numeric(raw['flags'])
    cat['ClassStar'] = _coerce_numeric(raw['class_star'])

    for band in ['u', 'v', 'g', 'r', 'i', 'z']:
        cat[band + 'PSF'] = _coerce_numeric(raw[band + '_psf'])
        cat['e_' + band + 'PSF'] = _coerce_numeric(raw['e_' + band + '_psf'])

    cat.meta['source'] = 'SkyMapper TAP service (fallback)'
    return cat


_IRSA_TAP_URL = 'https://irsa.ipac.caltech.edu/TAP'


def _fallback_2mass(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for the 2MASS Point Source Catalogue via IRSA."""
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        log('Trying IRSA 2MASS Point Source Catalogue as a fallback')
        query = f"""
            SELECT ra, dec, designation, j_m, j_msigcom, h_m, h_msigcom, k_m, k_msigcom,
                   ph_qual, rd_flg, bl_flg, cc_flg, gal_contam, mp_flg
            FROM fp_psc
            WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0}, {dec0}, {sr0}))=1
        """
        raw = _tap_async_query(_IRSA_TAP_URL, query, timeout=timeout, verbose=verbose)
    except Exception as e:
        log('IRSA 2MASS fallback failed:', str(e)[:200])
        return None

    if not len(raw):
        return None

    cat = Table()
    cat['RAJ2000'] = raw['ra']
    cat['DEJ2000'] = raw['dec']
    cat['2MASS'] = raw['designation']
    cat['Jmag'] = raw['j_m']
    cat['e_Jmag'] = raw['j_msigcom']
    cat['Hmag'] = raw['h_m']
    cat['e_Hmag'] = raw['h_msigcom']
    cat['Kmag'] = raw['k_m']
    cat['e_Kmag'] = raw['k_msigcom']
    # Quality/contamination flags, mapped to VizieR II/246/out names (confirmed
    # live against the real catalogue).
    cat['Qflg'] = raw['ph_qual']
    cat['Rflg'] = raw['rd_flg']
    cat['Bflg'] = raw['bl_flg']
    cat['Cflg'] = raw['cc_flg']
    cat['Xflg'] = raw['gal_contam']
    cat['Aflg'] = raw['mp_flg']

    cat.meta['source'] = 'IRSA 2MASS Point Source Catalogue (fallback)'
    return cat


def _fallback_vhs(ra0, dec0, sr0, filters=None, timeout=(10, 60), verbose=False):
    """Fallback for the VISTA Hemisphere Survey via the VISTA Science Archive.

    Uses DR4, the latest release exposed by astroquery's VSA client (instead of DR5).
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    try:
        from astroquery.vsa import VsaClass

        log('Trying VISTA Science Archive (VHS DR4) as a fallback')
        coord = SkyCoord(ra0, dec0, unit='deg')
        # A local instance to be able to set timeout
        vsa = VsaClass()
        vsa.TIMEOUT = max(timeout) if isinstance(timeout, (tuple, list)) else timeout
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = vsa.query_region(
                coord, radius=sr0 * u.deg, database='VHSDR4', programme_id='VHS'
            )
    except Exception as e:
        log('VSA VHS fallback failed:', str(e)[:200])
        return None

    if not len(raw):
        return None

    cat = Table()
    cat['RAJ2000'] = raw['ra']
    cat['DEJ2000'] = raw['dec']

    for short, col in [('Jap3', 'jAperMag3'), ('Hap3', 'hAperMag3'), ('Ksap3', 'ksAperMag3')]:
        mag = np.asarray(raw[col], dtype=float)
        magerr = np.asarray(raw[col + 'Err'], dtype=float)
        # VSA encodes missing values with a large negative sentinel
        cat[short] = np.ma.masked_less(mag, -1e5)
        cat['e_' + short] = np.ma.masked_less(magerr, -1e5)

    cat['Mstat'] = raw['mergedClassStat']
    cat['Mclass'] = raw['mergedClass']
    cat['pStar'] = raw['pStar']
    cat['pGal'] = raw['pGalaxy']
    cat['pNoise'] = raw['pNoise']
    for band, raw_band in [('Y', 'y'), ('J', 'j'), ('H', 'h'), ('Ks', 'ks')]:
        cat[band + 'errbits'] = raw[raw_band + 'ErrBits']
        cat[band + 'perrbits'] = raw[raw_band + 'ppErrBits']

    cat.meta['name'] = 'VISTA Hemisphere Survey DR4'
    cat.meta['source'] = 'VISTA Science Archive VHS DR4 (fallback)'
    return cat


# Non-VizieR fallback sources, tried when every VizieR server fails or returns
# nothing for a catalogue. Each returns a Table shaped like the corresponding 
# VizieR catalogue (or None on failure).
catalog_fallbacks = {
    'ps1': _fallback_ps1,
    'gaiaedr3': _fallback_gaiaedr3,
    'gaiadr3syn': _fallback_gaiadr3syn,
    'skymapper': _fallback_skymapper,
    '2mass': _fallback_2mass,
    'vhs': _fallback_vhs,
}


def get_cat_vizier(
    ra0,
    dec0,
    sr0,
    catalog='ps1',
    limit=-1,
    filters={},
    extra=[],
    get_distance=False,
    augment_bands=True,
    timeout=(10, 60),
    verbose=False,
):
    """Download any catalogue from Vizier.

    The catalogue may be any Vizier identifier. For a number of popular catalogues,
    additional photometric bands are computed from analytical conversion formulae:

    - ``ps1`` — Pan-STARRS DR1, augmented with Johnson-Cousins B, V, R, I
    - ``gaiadr2`` — Gaia DR2, augmented with Johnson-Cousins B, V, R, I, Pan-STARRS and SDSS
    - ``gaiaedr3`` — Gaia eDR3
    - ``gaiadr3syn`` — Gaia DR3 synthetic photometry from XP spectra
    - ``skymapper`` — SkyMapper DR1.1
    - ``vsx`` — AAVSO Variable Stars Index
    - ``apass`` — AAVSO APASS DR9
    - ``sdss`` — SDSS DR16
    - ``atlas`` — ATLAS-RefCat2, augmented with Johnson-Cousins B, V, R, I
    - ``usnob1`` — USNO-B1
    - ``gsc`` — Guide Star Catalogue 2.2
    - ``2mass`` — 2MASS Point Source Catalogue
    - ``vhs`` — VISTA Hemisphere Survey DR5, augmented with 2MASS-system J, H, Ks

    If the query fails on the default VizieR server with a network error, or the
    server reports it as failed (e.g. lacking the catalogue data), it is retried
    on the servers listed in ``vizier_mirrors``. A clean empty result is not a
    failure and is not retried elsewhere: it is only re-queried once on the same
    server, bypassing the cache.

    If the default VizieR server fails and none of the mirrors returns any data,
    and ``catalog`` is one of ``ps1``, ``gaiaedr3``, ``gaiadr3syn``, ``skymapper``,
    ``2mass`` or ``vhs``, the catalogue is instead fetched from a dedicated
    non-VizieR mirror (MAST, the ESA Gaia archive, the SkyMapper TAP service,
    IRSA, or the VISTA Science Archive, respectively) and reshaped into the
    VizieR format. An empty result confirmed by the default server is trusted
    as is and never triggers this fallback; an empty answer from a mirror is
    not, as mirrors may hold incomplete copies of a catalogue.

    Parameters
    ----------
    ra0 : float
        Right Ascension of the field center in degrees.
    dec0 : float
        Declination of the field center in degrees.
    sr0 : float
        Field radius in degrees.
    catalog : str, optional
        Any Vizier catalogue identifier or a short name from the list above.
    limit : int, optional
        Maximum number of rows to return (``-1`` for unlimited).
    filters : dict, optional
        Column filters applied server-side. Keys are column names; values are
        Vizier filter expressions.
    extra : list of str, optional
        Additional column names to include beyond the defaults.
    get_distance : bool, optional
        If True, include a ``_r`` column with angular distance from field center.
    augment_bands : bool, optional
        If True (default), augment photometry with derived bands where supported.
    timeout : float or tuple of float, optional
        Network timeout in seconds for each server, passed to :mod:`requests`.
        May be a ``(connect, read)`` tuple; the default keeps the connection
        timeout short so that an unreachable server is skipped quickly, while
        allowing slow responses for large queries.
    verbose : bool or callable, optional
        Whether to show verbose messages. May be boolean or a ``print``-like callable.

    Returns
    -------
    astropy.table.Table or None
        Catalogue as returned by Vizier, with additional columns for supported
        catalogues, or None if the query fails.
    """

    # Simple Wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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

    coord = SkyCoord(ra0, dec0, unit='deg')

    cats = None

    for server in [None] + list(vizier_mirrors):
        vizier = Vizier(
            row_limit=limit, columns=columns, column_filters=filters, timeout=timeout
        )
        if server is not None:
            log('Retrying on VizieR mirror', server)
            vizier.VIZIER_SERVER = server

        try:
            cats = _query_vizier_region(vizier, coord, sr0 * u.deg, vizier_id, cache=True)
            if not cats or not len(cats) == 1:
                # Possibly a stale cached answer - re-query the same server
                cats = _query_vizier_region(vizier, coord, sr0 * u.deg, vizier_id, cache=False)
            if not cats and server is not None:
                # An empty answer from a mirror is not trusted: mirrors may hold
                # incomplete or broken copies of a catalogue and still answer cleanly
                log('VizieR mirror', server, 'returned no data for', vizier_id)
                cats = None
                continue
            break
        except (requests.exceptions.RequestException, TableParseError, VizierQueryError) as e:
            # Network-level failure, an unparseable response, or an error reported by the server - try the next server
            log('VizieR query failed on', vizier.VIZIER_SERVER, ':', str(e)[:200])
    else:
        log('All VizieR servers failed, cannot get catalogue', catalog)

    if cats and len(cats) == 1:
        cat = cats[0]
        cat.meta['name'] = name
    else:
        log('Error requesting catalogue', catalog, 'from VizieR')
        if cats is not None:
            return None
        
        cat = None
        fallback = catalog_fallbacks.get(catalog)
        if fallback is not None:
            log(f"Warning: VizieR has no catalogue '{catalog}' available right now, trying a non-VizieR fallback source")
            try:
                cat = fallback(ra0, dec0, sr0, filters=filters, timeout=timeout, verbose=verbose)
            except Exception as e:
                log('Fallback source for', catalog, 'raised an unexpected error:', str(e)[:200])
                cat = None

        if cat is None or not len(cat):
            log('Cannot get catalogue', catalog, 'from any source')
            return None

        cat = _apply_filters(cat, filters, verbose=verbose)
        if limit and limit > 0 and len(cat) > limit:
            # Keep the sources closest to the field center, not an arbitrary
            # subset in whatever order the fallback server returned them
            dist = astrometry.spherical_distance(ra0, dec0, cat['RAJ2000'], cat['DEJ2000'])
            cat = cat[np.argsort(np.ma.filled(np.ma.asarray(dist, dtype=float), np.inf))[:limit]]

        if not len(cat):
            log('Fallback catalogue', catalog, 'is empty after applying filters')
            return None

        cat.meta.setdefault('name', name)

    cat.meta['vizier_id'] = vizier_id

    log('Got', len(cat), 'entries with', len(cat.colnames), 'columns')

    # Fix _RAJ2000/_DEJ2000
    if '_RAJ2000' in cat.keys() and '_DEJ2000' in cat.keys() and not 'RAJ2000' in cat.keys():
        cat.rename_columns(['_RAJ2000', '_DEJ2000'], ['RAJ2000', 'DEJ2000'])

    if get_distance and 'RAJ2000' in cat.colnames and 'DEJ2000' in cat.colnames:
        log("Augmenting the catalogue with distances from field center")
        cat['_r'] = astrometry.spherical_distance(ra0, dec0, cat['RAJ2000'], cat['DEJ2000'])

    if augment_bands:
        # Augment catalogue with additional bandpasses
        augment_cat_bands(cat, catalog=catalog if catalog in catalogs else None, verbose=verbose)

    return cat


def get_cat_panstarrs(
    ra0,
    dec0,
    sr0,
    limit=-1,
    filters=None,
    extra=None,
    get_distance=False,
    augment_bands=True,
    page_size=50000,
    verbose=False,
    **kwargs,
):
    """Download Pan-STARRS DR2 mean PSF photometry directly from MAST.

    This is an independent alternative to ``get_cat_vizier(..., catalog="ps1")``.
    The returned positions and grizy PSF magnitudes use the same column names.
    No photometric quality or star/galaxy cuts are applied automatically.

    Parameters
    ----------
    ra0, dec0 : float
        ICRS field center in degrees.
    sr0 : float
        Cone radius in degrees, greater than zero and at most 1 (the MAST limit).
    limit : int, optional
        Maximum number of rows to retrieve; -1 retrieves all pages.
    filters : dict, optional
        Astroquery MAST column filters, e.g. ``{"rMeanPSFMag": [("lt", 18)],
        "nDetections": [("gte", 2)]}``. These are not VizieR filter expressions.
    extra : list of str, optional
        Additional native MAST columns to return, e.g. ``["rMeanKronMag"]``.
        Their names are retained unchanged.
    get_distance : bool, optional
        Add ``_r``, the angular distance from the query center in degrees.
    augment_bands : bool, optional
        Apply the same PS1 band transformations as :func:`get_cat_vizier`.
    page_size : int, optional
        Maximum rows per request. All pages are retrieved unless limited by limit.
    verbose : bool or callable, optional
        Logging control.

    Returns
    -------
    astropy.table.Table or None
        Catalogue with ``RAJ2000``, ``DEJ2000`` (degrees), ``e_RAJ2000``,
        ``e_DEJ2000`` (arcseconds), ``gmag`` through ``ymag`` and corresponding
        ``e_gmag`` etc. (magnitudes). Missing measurements in these standard
        columns, including MAST's -999 sentinel, are masked. Rows without valid
        positions are discarded.
        A successful empty query returns an empty table; a request or response
        parsing failure returns None, never a partial catalogue.

    Notes
    -----
    Uses :meth:`astroquery.mast.CatalogsClass.query_region`, independently of
    VizieR. Request handling and timeouts follow Astroquery's MAST configuration.
    """
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None
    if not np.all(np.isfinite([ra0, dec0, sr0])) or not 0 < sr0 <= 1:
        raise ValueError("Coordinates must be finite and radius must be in (0, 1] degrees")
    if not isinstance(limit, (int, np.integer)) or limit < -1:
        raise ValueError("limit must be -1 or a non-negative integer")
    if not isinstance(page_size, (int, np.integer)) or page_size < 1:
        raise ValueError("page_size must be a positive integer")
    coord = SkyCoord(ra0, dec0, unit="deg")
    url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
    mapping = {
        "raMean": ("RAJ2000", u.deg),
        "decMean": ("DEJ2000", u.deg),
        "raMeanErr": ("e_RAJ2000", u.arcsec),
        "decMeanErr": ("e_DEJ2000", u.arcsec),
    }
    for band in "grizy":
        mapping[f"{band}MeanPSFMag"] = (f"{band}mag", u.mag)
        mapping[f"{band}MeanPSFMagErr"] = (f"e_{band}mag", u.mag)
    columns = list(dict.fromkeys(["objID", *mapping, *(extra or [])]))
    criteria = dict(filters or {})
    reserved = {"coordinates", "ra", "dec", "radius", "sr", "page", "pagesize", "columns",
                "sort_by", "catalog", "version", "data_release", "table"}
    if any(key.lower() in reserved for key in criteria):
        raise ValueError("filters must contain MAST column filters, not query controls")
    # A fixed page size and unique sort key keep page offsets stable.
    pagesize = min(page_size, limit) if limit > 0 else page_size
    mast = CatalogsClass()
    pages = []
    count = 0
    page_number = 1
    while limit != 0:
        log("Requesting Pan-STARRS DR2 from MAST, page", page_number)
        try:
            page = mast.query_region(
                coord, radius=sr0*u.deg, catalog="Panstarrs", data_release="dr2",
                table="mean", columns=columns, sort_by="objID",
                pagesize=pagesize, page=page_number, **criteria)
            if not len(page):
                break
            if not set(columns).issubset(page.colnames):
                raise ValueError("MAST response is missing requested columns")
        except (requests.exceptions.RequestException, TableParseError,
                AstroqueryTimeoutError, ConnectionError, ValueError) as exc:
            log("Pan-STARRS query failed:", str(exc))
            return None
        pages.append(page)
        count += len(page)
        if len(page) < pagesize or (limit > 0 and count >= limit):
            break
        page_number += 1
    cat = (vstack(pages, metadata_conflicts="silent") if pages else
           Table(names=columns, dtype=[np.int64 if name == "objID" else float for name in columns]))
    if limit >= 0:
        cat = cat[:limit]
    for source, (target, unit) in mapping.items():
        values = np.asarray(np.ma.filled(cat[source].astype(float), np.nan))
        cat.replace_column(source, MaskedColumn(
            values, name=source, mask=~np.isfinite(values) | (values == -999), unit=unit))
        cat.rename_column(source, target)
    cat = cat[~(np.ma.getmaskarray(cat["RAJ2000"]) | np.ma.getmaskarray(cat["DEJ2000"]))]
    cat.meta.update(name="Pan-STARRS DR2", source="MAST", release="dr2",
                    table="mean", url=url)
    if get_distance:
        cat["_r"] = astrometry.spherical_distance(ra0, dec0, cat["RAJ2000"], cat["DEJ2000"])
        cat["_r"].unit = u.deg
    if augment_bands:
        augment_cat_bands(cat, catalog="ps1", verbose=verbose)
    log("Got", len(cat), "entries from MAST")
    return cat


def xmatch_objects(obj, catalog='ps1', sr=3 / 3600, col_ra='ra', col_dec='dec', augment_bands=True):
    """Cross-match object list with a Vizier catalogue using the CDS XMatch service.

    Parameters
    ----------
    obj : astropy.table.Table
        Table of objects to cross-match.
    catalog : str, optional
        Any Vizier catalogue identifier or a short name recognized by :func:`get_cat_vizier`.
    sr : float, optional
        Cross-matching radius in degrees.
    col_ra : str, optional
        Column name in ``obj`` containing Right Ascension values.
    col_dec : str, optional
        Column name in ``obj`` containing Declination values.
    augment_bands : bool, optional
        If True (default), augment photometry with derived bands where supported.

    Returns
    -------
    astropy.table.Table
        Table of matched objects augmented with columns from the Vizier catalogue.
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


def xmatch_skybot(obj, sr=10 / 3600, time=None, col_ra='ra', col_dec='dec', col_id='id'):
    """Cross-match object list with Solar System object positions using SkyBoT.

    Queries SkyBoT for all Solar System objects in a cone covering the input
    field at the specified time, then cross-matches them against the input list
    within ``sr``.

    Parameters
    ----------
    obj : astropy.table.Table
        Table of objects to cross-match.
    sr : float, optional
        Cross-matching radius in degrees.
    time : astropy.time.Time or datetime or str
        Observation time.
    col_ra : str, optional
        Column name in ``obj`` containing Right Ascension values.
    col_dec : str, optional
        Column name in ``obj`` containing Declination values.
    col_id : str, optional
        Column name in ``obj`` containing unique object identifiers.

    Returns
    -------
    astropy.table.Table or None
        Table of matched Solar System objects with a ``col_id`` column referencing
        the matched input objects, or None if no matches are found.
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
    """Cross-match object list with NED database entries.

    .. note::
        Queries NED one object at a time, so this is slow for large lists.

    Parameters
    ----------
    obj : astropy.table.Table
        Table of objects to cross-match.
    sr : float, optional
        Cross-matching radius in degrees.
    col_ra : str, optional
        Column name in ``obj`` containing Right Ascension values.
    col_dec : str, optional
        Column name in ``obj`` containing Declination values.
    col_id : str, optional
        Column name in ``obj`` containing unique object identifiers.

    Returns
    -------
    astropy.table.Table or list
        Table of NED objects with an ``id`` column referencing the matched input
        objects, or an empty list if no matches are found.
    """

    xcat = []

    # FIXME: is there more optimal way to query NED for multiple sky positions?..
    for row in obj:
        res = Ned().query_region(SkyCoord(row[col_ra], row[col_dec], unit='deg'), sr * u.deg)

        if res:
            res['id'] = row[col_id]

            xcat.append(res)

    if xcat:
        xcat = vstack(xcat, join_type='exact', metadata_conflicts='silent')

    return xcat
