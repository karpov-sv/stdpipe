from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

def breakpoint():
    try:
        from IPython.core.debugger import Tracer
        Tracer()()
    except:
        import pdb
        pdb.set_trace()

def make_jname(ra, dec):
    radec = SkyCoord(ra, dec, unit='deg')
    return "J%s%s" % (radec.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True),
                      radec.dec.to_string(sep='', precision=2, alwayssign=True, pad=True))
