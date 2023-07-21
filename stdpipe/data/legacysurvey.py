#!/usr/bin/env python3

from astropy.io import fits
from astropy.table import Table, vstack

bricks_dr10_south = Table(fits.getdata('https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/survey-bricks-dr10-south.fits.gz'))

bricks_dr9_north = Table(fits.getdata('https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/survey-bricks-dr9-north.fits.gz'))

s1 = bricks_dr10_south[['brickname', 'ra', 'dec']]
s1['survey'] = 'S'

s2 = bricks_dr9_north[['brickname', 'ra', 'dec']]
s2['survey'] = 'N'

short = vstack([s1, s2])
short.write('legacysurvey_bricks.fits', format='fits')
