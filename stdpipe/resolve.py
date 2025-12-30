
import xml.dom.minidom as minidom
import requests
import json
import re
import io

from astropy.coordinates import SkyCoord


def simbadResolve(name='m31'):
    url = 'http://cdsweb.u-strasbg.fr/viz-bin/nph-sesame/-oxpi/SNVA'
    res = requests.get(url, params=requests.utils.quote(name))

    try:
        xml = minidom.parseString(res.content)

        r = xml.getElementsByTagName('Resolver')[0]

        name = r.getElementsByTagName('oname')[0].childNodes[0].nodeValue
        ra = float(r.getElementsByTagName('jradeg')[0].childNodes[0].nodeValue)
        dec = float(r.getElementsByTagName('jdedeg')[0].childNodes[0].nodeValue)

        return name, ra, dec
    except:
        return None, None, None


def parseSexadecimal(string):
    value = 0

    m = re.search(
        "^\s*([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*$", string
    ) or re.search("^\s*([+-])?\s*(\d{1,3})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*$", string)
    if m:
        value = float(m.group(2)) + float(m.group(3)) / 60 + float(m.group(4)) / 3600

        if m.group(1) == '-':
            value = -value

    return value


def tnsResolve(name='AT2023lxx'):
    # Normalize AT name to have whitespace before year
    m = re.match('^(AT)\s*(2\w+)$', name)
    if m:
        name = m[1] + ' ' + m[2]

    for reverse in [False, True]:
        params = {'resolver': 'tns', 'name': name}
        if reverse:
            params.update({'reverse': True})

        r = requests.post("https://api.ztf.fink-portal.org/api/v1/resolver", json=params)

        res = json.loads(r.content)
        if res:
            break

    if res:
        return res[0]['d:fullname'], res[0]['d:ra'], res[0]['d:declination']

    return None, None, None


def resolve(string='M33', verbose=False):
    """
    Resolve the object name (or coordinates string) into proper coordinates on the sky.
    Uses several different algorithms
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    target = None

    if target is None:
        m = re.search("^\s*(\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s*$", string)
        if m:
            log("Resolved as two values in degrees")

            ra = float(m.group(1))
            dec = float(m.group(2))

            target = SkyCoord(ra, dec, unit='deg')

    if target is None:
        m = re.search(
            "^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*$",
            string,
        ) or re.search(
            "^\s*(\d{1,2})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*$",
            string,
        )
        if m:
            log("Resolved as two sexadecimal values, interpreted as hours and degrees")

            ra = (
                float(m.group(1)) + float(m.group(2)) / 60 + float(m.group(3)) / 3600
            ) * 15
            dec = float(m.group(5)) + float(m.group(6)) / 60 + float(m.group(7)) / 3600

            if m.group(4) == '-':
                dec = -dec

            target = SkyCoord(ra, dec, unit='deg')

    if target is None:
        name, ra, dec = simbadResolve(string)

        if name:
            log("Resolved by Simbad as", name)

            target = SkyCoord(ra, dec, unit='deg')

    if target is None:
        name, ra, dec = tnsResolve(string)

        if name:
            log("Resolved by TNS as", name)

            target = SkyCoord(ra, dec, unit='deg')

    if target is None:
        try:
            target = SkyCoord.from_name(string, parse=True)
            log("Resolved by Astropy SkyCoord.from_name()")
        except:
            pass

    if target is not None:
        log(f"RA = {target.ra.deg:.4f} deg, Dec = {target.dec.deg:.4f} deg")
    else:
        log(f"Failed to resolve", string)

    return target
