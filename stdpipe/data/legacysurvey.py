#!/usr/bin/env python3

# Regenerates legacysurvey_bricks.fits.gz - the list of Legacy Survey DR11
# brick centres used by stdpipe.templates to locate the coadded images.

import numpy as np

from astropy.io import fits
from astropy.table import Table, vstack

base = 'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/'

bricks_south = Table(fits.getdata(base + 'south/survey-bricks-dr11-south.fits.gz'))
bricks_north = Table(fits.getdata(base + 'north/survey-bricks-dr11-north.fits.gz'))


def shorten(bricks, survey):
    # North and south footprints overlap, and the same brick may be present in
    # both. Keeping only the ones primary in their region ensures every brick
    # appears exactly once, and in the region it should be taken from.
    bricks = bricks[bricks['survey_primary'].astype(bool)]

    short = bricks[['brickname', 'ra', 'dec']]
    short['survey'] = survey

    # Per-brick list of bands with actual coverage, e.g. 'grz' - north has no i
    covered = {
        band: np.asarray(bricks['nexp_' + band] > 0)
        if 'nexp_' + band in bricks.colnames
        else np.zeros(len(bricks), dtype=bool)
        for band in 'griz'
    }
    short['bands'] = [
        ''.join([band for band in 'griz' if covered[band][i]]) for i in range(len(short))
    ]

    # Drop the bricks with no coadds at all
    return short[short['bands'] != '']


short = vstack([shorten(bricks_south, 'S'), shorten(bricks_north, 'N')])
short.write('legacysurvey_bricks.fits.gz', format='fits', overwrite=True)
