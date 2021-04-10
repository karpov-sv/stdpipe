from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import requests
import dateutil
import shlex

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
                      radec.dec.to_string(sep='', precision=1, alwayssign=True, pad=True))

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

    # FIXME: if server is compressing the data, Content-Length here will be wrong?..
    size = int(response.headers.get('Content-Length', 0))

    if filename is None:
        # Some logic to guess filename from headers
        tmp = re.findall("filename=(.+)", response.headers.get('Content-Disposition', ''))
        if len(tmp):
            filename = tmp[1]
        else:
            filename = 'download'

        # TODO: check file existence here again?..

    desc = os.path.basename(filename)

    log('Downloading %s from %s...' % (filename, url))

    length = 0

    with open(filename, 'wb') as file, tqdm(desc=desc, total=size, unit='iB', unit_scale=True, unit_divisor=1024) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            chunksize = file.write(chunk)
            progress.update(chunksize)
            length += len(chunk)

    if size and length < size:
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
    elif default is None:
        return None
    elif hasattr(default, '__len__'):
        # default value is array, return it
        return default
    else:
        # Broadcast scalar to proper length
        return default*np.ones(len(table), dtype=np.int)

def format_astromatic_opts(opts):
    """
    Auxiliary function to format dictionary of options into Astromatic compatible command-line string.
    Booleans are converted to Y/N, arrays to comma separated lists, strings are quoted when necessary
    """
    result = []

    for key in opts.keys():
        if opts[key] is None:
            pass
        elif type(opts[key]) == bool:
            result.append('-%s %s' % (key, 'Y' if opts[key] else 'N'))
        else:
            value = opts[key]

            if type(value) == str:
                value = shlex.quote(value)
            elif hasattr(value, '__len__'):
                value = ','.join([str(_) for _ in value])

            result.append('-%s %s' % (key, value))

    result = ' '.join(result)

    return result

# Parsing of DATASEC-like keywords
def parse_det(string):
    """
    Parse DATASEC-like keyword
    """

    x0,x1,y0,y1 = [int(_)-1 for _ in sum([_.split(':') for _ in string[1:-1].split(',')], [])]

    return x0,x1,y0,y1

# Cropping of overscans if any
def crop_overscans(image, header, subtract_bias=True, verbose=False):
    """
    Crop overscans from input image based on its header keywords.
    Also, subtract the 'bias' value estimated from overscan.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = print if verbose else lambda *args,**kwargs: None

    if header is not None:
        header = header.copy()

    bias = 0

    for kw in ['BIASSEC']:
        if kw in header:
            x1,x2,y1,y2 = parse_det(header.get(kw))
            biasimg = image[y1:y2+1, x1:x2+1]

            bias = np.mean(biasimg)
            log('Estimated bias level using %s: %s = %.2f' % (kw, header.get(kw), bias))

    for kw in ['TRIMSEC', 'DATASEC']:
        if kw in header:
            x1,x2,y1,y2 = parse_det(header.get(kw))

            if header is not None and header.get('CRPIX1') is not None:
                header['CRPIX1'] -= x1
                header['CRPIX2'] -= y1

            image = image[y1:y2+1, x1:x2+1]
            log('Cropped image using %s: %s' % (kw, header.get(kw)))

            break

    for kw in ['BIASSEC', 'TRIMSEC', 'DATASEC']:
        if kw in header:
            header[kw + '0'] = header.pop(kw)

    image -= bias

    return image, header
