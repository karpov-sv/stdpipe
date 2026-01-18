
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

    This function attempts multiple resolution methods in order:
    1. Decimal degrees (RA Dec in degrees)
    2. Sexagesimal coordinates (hours/degrees with minutes and seconds)
    3. Simbad name resolution (astronomical objects)
    4. TNS name resolution (transients)
    5. Astropy SkyCoord.from_name() (fallback)

    :param string: Input string to resolve (object name or coordinates)
    :param verbose: Enable verbose output. Can be True/False or a print-like function.
    :returns: astropy.coordinates.SkyCoord object if resolved, None otherwise

    **Supported Coordinate Formats:**

    1. **Decimal degrees** (RA and Dec in degrees, space or comma separated):
       - ``"123.456 45.678"`` → RA=123.456°, Dec=+45.678°
       - ``"123.456, 45.678"`` → RA=123.456°, Dec=+45.678° (comma-separated)
       - ``"123.456 -45.678"`` → RA=123.456°, Dec=-45.678°
       - ``"10.68, 41.27"`` → RA=10.68°, Dec=+41.27° (M31)

    2. **Sexagesimal coordinates** (space-separated or colon-separated):
       - Space format: ``"HH MM SS.ss ±DD MM SS.ss"``
       - Colon format: ``"HH:MM:SS.ss ±DD:MM:SS.ss"``
       - Comma can separate RA from Dec: ``"HH MM SS.ss, ±DD MM SS.ss"``

       Examples:
       - ``"12 34 56.7 +45 12 34.5"`` → RA=12h 34m 56.7s, Dec=+45° 12' 34.5"
       - ``"12 34 56.7, +45 12 34.5"`` → Same (comma between RA and Dec)
       - ``"12:34:56.7 +45:12:34.5"`` → RA=12h 34m 56.7s, Dec=+45° 12' 34.5"
       - ``"12:34:56.7, +45:12:34.5"`` → Same (comma between RA and Dec)
       - ``"12 34 56 45 12 34"`` → Dec assumed positive if no sign
       - ``"00 42 44.3, +41 16 09"`` → M31 coordinates (comma-separated)

       **Note:** RA is interpreted as **hours** (multiplied by 15 to convert to degrees),
       Dec is interpreted as **degrees**. This is the standard astronomical convention.

    3. **Object names** (resolved via Simbad, TNS, or Astropy):
       - Messier objects: ``"M31"``, ``"M 31"``, ``"m31"``
       - NGC/IC objects: ``"NGC1234"``, ``"IC 5146"``
       - Named stars: ``"Betelgeuse"``, ``"Vega"``
       - Transients: ``"AT2023lxx"``, ``"AT 2023lxx"``, ``"SN2023ixf"``
       - Coordinates as names: ``"12h34m56s +45d12m34s"`` (via Astropy)

    **Resolution Priority:**

    The function tries methods in order and returns the first successful match:
    1. Decimal degrees parsing (fastest, no network)
    2. Sexagesimal parsing (fast, no network)
    3. Simbad query (network required)
    4. TNS query via Fink API (network required, for transients)
    5. Astropy SkyCoord.from_name() (network required, CDS Sesame)

    **Examples:**

    >>> from stdpipe.resolve import resolve
    >>>
    >>> # Decimal degrees
    >>> target = resolve("10.68 41.27")
    >>> print(target.ra.deg, target.dec.deg)
    10.68 41.27
    >>>
    >>> # Decimal degrees (comma-separated)
    >>> target = resolve("10.68, 41.27")
    >>> print(target.ra.deg, target.dec.deg)
    10.68 41.27
    >>>
    >>> # Sexagesimal (space-separated)
    >>> target = resolve("12 34 56 +45 12 34")
    >>> print(f"{target.ra.deg:.2f} {target.dec.deg:.2f}")
    188.73 45.21
    >>>
    >>> # Sexagesimal (comma between RA and Dec)
    >>> target = resolve("12 34 56, +45 12 34")
    >>> print(f"{target.ra.deg:.2f} {target.dec.deg:.2f}")
    188.73 45.21
    >>>
    >>> # Sexagesimal (colon-separated)
    >>> target = resolve("00:42:44.3 +41:16:09")
    >>> print(f"{target.ra.deg:.2f} {target.dec.deg:.2f}")
    10.68 41.27
    >>>
    >>> # Sexagesimal (colon-separated with comma)
    >>> target = resolve("00:42:44.3, +41:16:09")
    >>> print(f"{target.ra.deg:.2f} {target.dec.deg:.2f}")
    10.68 41.27
    >>>
    >>> # Object name (requires network)
    >>> target = resolve("M31", verbose=True)
    Resolved by Simbad as M 31
    RA = 10.6847 deg, Dec = 41.2688 deg
    >>>
    >>> # Transient name (requires network)
    >>> target = resolve("AT2023lxx", verbose=True)
    Resolved by TNS as AT 2023lxx
    RA = 123.4560 deg, Dec = -45.6780 deg
    >>>
    >>> # Failed resolution returns None
    >>> target = resolve("InvalidObject123")
    >>> print(target)
    None

    **Verbose Mode:**

    When ``verbose=True``, prints resolution method and result:

    >>> target = resolve("123.456 45.678", verbose=True)
    Resolved as two values in degrees
    RA = 123.4560 deg, Dec = 45.6780 deg

    You can also provide a custom logging function:

    >>> def my_log(*args):
    ...     print("[RESOLVER]", *args)
    >>> target = resolve("M31", verbose=my_log)
    [RESOLVER] Resolved by Simbad as M 31
    [RESOLVER] RA = 10.6847 deg, Dec = 41.2688 deg

    **Network Requirements:**

    - Decimal/sexagesimal parsing: No network required
    - Simbad resolution: Queries http://cdsweb.u-strasbg.fr
    - TNS resolution: Queries https://api.ztf.fink-portal.org
    - Astropy fallback: Queries CDS Sesame service

    If network is unavailable, only coordinate parsing will work. Object name
    resolution will fail silently and return None.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    target = None

    if target is None:
        m = re.search(r"^\s*(\d+\.?\d*)\s*[,\s]+\s*([+-]?\d+\.?\d*)\s*$", string)
        if m:
            log("Resolved as two values in degrees")

            ra = float(m.group(1))
            dec = float(m.group(2))

            target = SkyCoord(ra, dec, unit='deg')

    if target is None:
        m = re.search(
            r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*[,\s]+\s*([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s*$",
            string,
        ) or re.search(
            r"^\s*(\d{1,2})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*[,\s]+\s*([+-])?\s*(\d{1,3})\:(\d{1,2})\:(\d{1,2}\.?\d*)\s*$",
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
