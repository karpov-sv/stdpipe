from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import requests
import dateutil

from tqdm.auto import tqdm

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

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

def get_data_path(dataname):
    """
    Returns full path to the data file located in the module data/ folder
    """
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataname)

def download(url, filename=None, overwrite=False, verbose=False):
    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    if not overwrite and filename is not None and os.path.exists(filename):
        log(filename, 'already downloaded')
        return True

    response = requests.get(url, stream=True, allow_redirects=True)
    if response.status_code != 200:
        log('Error %s downloading from %s' % (response.status_code, url))
        return False

    size = int(response.headers.get('Content-Length', 0))

    if filename is None:
        # Some logic to guess filename from headers
        tmp = re.findall("filename=(.+)", response.headers.get('Content-Disposition', ''))
        if len(tmp):
            filename = tmp[1]
        else:
            filename = 'download'

        # TODO: check file existence there again?..

    desc = os.path.basename(filename)

    log('Downloading %s from %s...' % (filename, url))

    with open(filename, 'wb') as file, tqdm(desc=desc, total=size, unit='iB', unit_scale=True, unit_divisor=1024) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            length = file.write(chunk)
            progress.update(length)

    if size and length != size:
        log('Downloaded %d bytes, was expecting %d' % (length, size))
        return False
    else:
        log('Successfully downloaded %d bytes' % length)
        return True

def get_obs_time(header=None, filename=None, string=None, get_datetime=False, verbose=False):
    """
    Extract date and time of observations from FITS headers of common formats.
    Returns astropy Time object.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    # Simple wrapper to display parsed value and convert it as necessary
    def convert_time(time):
        time = Time(time)
        log('Time parsed as:', time)
        if get_datetime:
            return time.datetime
        else:
            return time

    if string:
        log('Parsing user-provided time string:', string)
        try:
            return convert_time(dateutil.parser.parse(string))
        except dateutil.parser.ParserError as err:
            log('Could not parse user-provided string:', err)
            return None

    if header is None:
        log('Loading FITS header from', filename)
        header = fits.getheader(filename)

    for dkey in ['DATE-OBS']:
        if dkey in header:
            log('Found ' + dkey + ':', header[dkey])
            # First try to parse standard ISO time
            try:
                return convert_time(header[dkey])
            except:
                log('Could not parse ' + dkey + ' using Astropy parser')

            for tkey in ['TIME-OBS', 'UT']:
                if tkey in header:
                    log('Found ' + tkey + ':', header[tkey])
                    try:
                        return convert_time(dateutil.parser.parse(header[dkey] + ' ' + header[tkey]))
                    except dateutil.parser.ParserError as err:
                        log('Could not parse ' + dkey + ' + ' + tkey + ':', err)

    log('Unsupported FITS header time format')

    return None

def table_get(table, colname, default=0):
    """
    Simple wrapper to get table column, or default value if it is not present
    """

    if colname in table.colnames:
        return table[colname]
    elif hasattr(default, '__len__'):
        # default value is array, return it
        return default
    else:
        # Broadcast scalar to proper length
        return default*np.ones(len(table), dtype=np.int)
