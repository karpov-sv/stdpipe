import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u


def _sky_residuals_arcsec(wcs: WCS, xy: np.ndarray, sky: SkyCoord, center: SkyCoord) -> np.ndarray:
    """
    Residuals in tangent-plane arcsec using a SkyOffsetFrame around `center`.
    Returns concatenated [d_lon_arcsec, d_lat_arcsec] per point.
    """
    ra_dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    model = SkyCoord(ra=ra_dec[0] * u.deg, dec=ra_dec[1] * u.deg, frame=sky.frame)

    off = SkyOffsetFrame(origin=center)
    sky_off = sky.transform_to(off)
    model_off = model.transform_to(off)

    # Use small-angle offsets on the tangent plane (arcsec)
    dlon = (model_off.lon - sky_off.lon).to_value(u.arcsec)
    dlat = (model_off.lat - sky_off.lat).to_value(u.arcsec)
    return np.concatenate([dlon, dlat])


def _pack_params(w: WCS, pv_deg: int) -> np.ndarray:
    # CRPIX, CRVAL, CD(2x2), PV2_0..PV2_pv_deg
    p = []
    p += [float(w.wcs.crpix[0]), float(w.wcs.crpix[1])]
    p += [float(w.wcs.crval[0]), float(w.wcs.crval[1])]
    cd = np.array(w.wcs.cd, dtype=float)
    p += [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]

    pv = np.zeros(pv_deg + 1, dtype=float)
    # astropy stores PV in w.wcs.get_pv() / w.wcs.set_pv(); get_pv returns list of (i, m, value)
    for (i, m, val) in w.wcs.get_pv():
        if i == 2 and 0 <= m <= pv_deg:
            pv[m] = float(val)
    p += pv.tolist()
    return np.array(p, dtype=float)


def _unpack_params_to_wcs(base: WCS, p: np.ndarray, pv_deg: int) -> WCS:
    w = base.deepcopy()
    w.wcs.crpix = [p[0], p[1]]
    w.wcs.crval = [p[2], p[3]]
    w.wcs.cd = np.array([[p[4], p[5]], [p[6], p[7]]], dtype=float)

    pv_vals = p[8: 8 + (pv_deg + 1)]
    # Preserve any PV keywords for other axes (e.g. PV1_*)
    pv_list = [(i, m, val) for (i, m, val) in base.wcs.get_pv() if i != 2]
    pv_list += [(2, m, float(pv_vals[m])) for m in range(pv_deg + 1)]
    w.wcs.set_pv(pv_list)
    return w


def fit_zpn_wcs_from_points(
    xy: np.ndarray,
    sky: SkyCoord,
    wcs_init: WCS,
    pv_deg: int = 5,
    fit_crpix: bool = True,
    fit_crval: bool = True,
    fit_cd: bool = True,
    fit_pv: bool = True,
    robust_loss: str = "soft_l1",
    f_scale_arcsec: float = 2.0,
    max_nfev: int = 200,
):
    """
    Fit a ZPN WCS by optimizing WCS parameters against matched (x,y) <-> (ra,dec).

    Parameters
    ----------
    xy : (N,2) array
        Pixel coordinates (0-based as in astropy WCS, i.e. origin=0).
    sky : SkyCoord (N)
        Reference sky positions.
    wcs_init : astropy.wcs.WCS
        Initial WCS; MUST already be ZPN (RA---ZPN/DEC--ZPN) or at least usable as base.
    pv_deg : int
        Degree for PV2_m coefficients to fit (PV2_0..PV2_pv_deg).
    fit_* : bool
        Toggle which parameter blocks to optimize.
    robust_loss : str
        Passed to scipy.optimize.least_squares(loss=...).
        Good options: 'linear', 'soft_l1', 'huber', 'cauchy'.
    f_scale_arcsec : float
        Robust loss scale in arcsec.
    max_nfev : int
        Optimization iterations (SciPy).

    Returns
    -------
    wcs_best : astropy.wcs.WCS
    result : scipy OptimizeResult (or None if SciPy not available)
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2)")
    if len(sky) != xy.shape[0]:
        raise ValueError("sky and xy must have the same length")

    # Choose a stable center for residuals (near the middle of your matched set)
    center = SkyCoord(
        ra=np.median(sky.ra).to(u.deg),
        dec=np.median(sky.dec).to(u.deg),
        frame=sky.frame
    )

    p0 = _pack_params(wcs_init, pv_deg=pv_deg)

    # Build a mask over parameters to optionally freeze blocks
    mask = np.ones_like(p0, dtype=bool)
    # indices: 0..1 CRPIX, 2..3 CRVAL, 4..7 CD, 8.. PV
    if not fit_crpix:
        mask[0:2] = False
    if not fit_crval:
        mask[2:4] = False
    if not fit_cd:
        mask[4:8] = False
    if not fit_pv:
        mask[8:] = False

    # Reduced parameter vector (only free params)
    free0 = p0[mask]

    # Optional: simple, conservative bounds (you can tighten if you want)
    # CRPIX: unbounded; CRVAL: RA wrap handled by residual frame; CD: unbounded
    # PV: usually PV2_1 > 0 (scale), others modest. We’ll keep wide bounds.
    lb = np.full_like(free0, -np.inf, dtype=float)
    ub = np.full_like(free0, +np.inf, dtype=float)

    # If PV is free, you can enforce PV2_1 > 0 (common sanity constraint)
    # Find where PV2_1 sits in the FREE vector (if present)
    if fit_pv and pv_deg >= 1:
        full_idx_pv1 = 8 + 1
        if mask[full_idx_pv1]:
            free_idx = np.flatnonzero(mask).tolist().index(full_idx_pv1)
            lb[free_idx] = 0.0  # positive radial scale

    def make_wcs_from_free(free: np.ndarray) -> WCS:
        p = p0.copy()
        p[mask] = free
        return _unpack_params_to_wcs(wcs_init, p, pv_deg=pv_deg)

    def fun(free: np.ndarray) -> np.ndarray:
        w = make_wcs_from_free(free)
        return _sky_residuals_arcsec(w, xy, sky, center)

    # Try SciPy; if unavailable, error with a clear message.
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        raise RuntimeError(
            "This fitter needs SciPy (scipy.optimize.least_squares). "
            "Install scipy or tell me and I’ll provide a pure-numpy Gauss-Newton fallback."
        ) from e

    res = least_squares(
        fun,
        free0,
        bounds=(lb, ub),
        loss=robust_loss,
        f_scale=float(f_scale_arcsec),
        max_nfev=int(max_nfev),
        x_scale="jac",
        verbose=0,
    )

    w_best = make_wcs_from_free(res.x)
    return w_best, res


def tan_wcs_to_zpn(
    w_tan: WCS,
    pv_deg: int = 7,
    n_samples: int = 256,
    theta_max_deg: float | None = None,
    drop_sip: bool = True,
) -> WCS:
    """
    Convert a celestial TAN WCS into a ZPN WCS with PV2_m initialized to approximate TAN.

    Notes
    -----
    TAN (gnomonic) radial law: r = tan(theta)  [in radians]
    In "degrees" units (common in FITS WCS plane coordinates), that's:
        r_deg = tan(theta_rad) * (180/pi)

    ZPN radial law: r_deg ≈ sum_{m=0..M} PV2_m * theta_deg^m
    We set PV2_0 = 0 and fit PV2_1..PV2_M to approximate the TAN law
    over theta in [0, theta_max_deg].

    Parameters
    ----------
    w_tan : astropy.wcs.WCS
        Input TAN WCS (2D celestial).
    pv_deg : int
        Highest PV degree to initialize (PV2_0..PV2_pv_deg).
        5–9 is usually plenty; higher can get wiggly.
    n_samples : int
        Samples used for the polynomial fit.
    theta_max_deg : float or None
        Max angular radius (deg) over which to match TAN. If None,
        estimated from image footprint corners using pixel_shape.
    drop_sip : bool
        If True, removes SIP distortions from the returned WCS.

    Returns
    -------
    w_zpn : astropy.wcs.WCS
        A ZPN WCS with same CRVAL/CRPIX/CD and PV2_m initialized.
    """
    # Build a clean WCS with CD only (no PC) to avoid PC/CDELT overriding CD.
    w = WCS(naxis=2)

    # Compute CD from the input WCS
    if w_tan.wcs.has_cd():
        cd = np.array(w_tan.wcs.cd, dtype=float)
    elif w_tan.wcs.has_pc():
        pc = np.array(w_tan.wcs.pc, dtype=float)
        cdelt = np.array(w_tan.wcs.cdelt, dtype=float)
        cd = pc * cdelt[None, :]
    else:
        cdelt = np.array(w_tan.wcs.cdelt, dtype=float)
        cd = np.diag(cdelt)

    w.wcs.crpix = np.array(w_tan.wcs.crpix, dtype=float)
    w.wcs.crval = np.array(w_tan.wcs.crval, dtype=float)
    w.wcs.cd = cd

    # Switch projection to ZPN (keep axis names)
    ctype1, ctype2 = w_tan.wcs.ctype
    if len(ctype1) < 8 or len(ctype2) < 8:
        raise ValueError("Expected CTYPE like 'RA---TAN'/'DEC--TAN'.")
    w.wcs.ctype = (ctype1[:5] + "ZPN", ctype2[:5] + "ZPN")

    # Copy metadata / frame info when available
    try:
        w.wcs.cunit = w_tan.wcs.cunit
    except Exception:
        pass
    try:
        w.wcs.radesys = w_tan.wcs.radesys
    except Exception:
        pass
    try:
        w.wcs.equinox = w_tan.wcs.equinox
    except Exception:
        pass
    try:
        if np.isfinite(w_tan.wcs.lonpole):
            w.wcs.lonpole = float(w_tan.wcs.lonpole)
    except Exception:
        pass
    try:
        if np.isfinite(w_tan.wcs.latpole):
            w.wcs.latpole = float(w_tan.wcs.latpole)
    except Exception:
        pass
    if w_tan.pixel_shape is not None:
        w.pixel_shape = w_tan.pixel_shape

    # SIP/distortion is not standard for ZPN; keep behavior explicit
    if drop_sip:
        w.sip = None
        w.cpdis1 = None
        w.cpdis2 = None
        w.det2im1 = None
        w.det2im2 = None

    # Estimate theta_max from footprint if not provided
    if theta_max_deg is None:
        if w.pixel_shape is None:
            raise ValueError(
                "w_tan.pixel_shape is None; provide theta_max_deg explicitly "
                "or set w_tan.pixel_shape = (nx, ny)."
            )
        nx, ny = w.pixel_shape  # (NAXIS1, NAXIS2)
        crpix = np.array(w.wcs.crpix, dtype=float)

        # Corners in pixel coordinates (origin=0 convention for astropy WCS)
        corners = np.array([
            [0.0, 0.0],
            [nx - 1.0, 0.0],
            [0.0, ny - 1.0],
            [nx - 1.0, ny - 1.0],
        ], dtype=float)

        # Sky positions of corners and center under TAN WCS
        ra_c, dec_c = w_tan.all_pix2world(crpix[0], crpix[1], 0)
        center = SkyCoord(ra_c * u.deg, dec_c * u.deg, frame="icrs")

        ra_k, dec_k = w_tan.all_pix2world(corners[:, 0], corners[:, 1], 0)
        sky_k = SkyCoord(ra_k * u.deg, dec_k * u.deg, frame="icrs")

        theta_max_deg = float(np.max(center.separation(sky_k).to_value(u.deg)))

        # Safety floor
        theta_max_deg = max(theta_max_deg, 0.01)

    # Fit PV2_m so that ZPN radial r(theta) ~ TAN radial r(theta)
    # Use degrees for theta and degrees for r on plane.
    theta = np.linspace(0.0, theta_max_deg, n_samples, dtype=float)
    theta_rad = np.deg2rad(theta)
    r_tan_deg = np.tan(theta_rad) * (180.0 / np.pi)

    # Build design matrix for m=1..pv_deg (PV2_0 fixed to 0)
    # r ≈ sum c[m-1] * theta^m
    A = np.vstack([theta**m for m in range(1, pv_deg + 1)]).T

    # Mild weighting: emphasize central region (helps stability)
    # (You can tune this; it’s just for initialization.)
    wgt = 1.0 / (1.0 + (theta / (0.6 * theta_max_deg))**2)
    Aw = A * wgt[:, None]
    bw = r_tan_deg * wgt

    coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    pv_list = [(2, 0, 0.0)] + [(2, m, float(coeffs[m - 1])) for m in range(1, pv_deg + 1)]
    w.wcs.set_pv(pv_list)

    return w


def fit_wcs_from_points(
    xy,
    world_coords,
    proj_point="center",
    projection=None,
    sip_degree=None,
    pv_deg=5,
):
    """Drop-in wrapper around :func:`astropy.wcs.utils.fit_wcs_from_points`
    that also handles **ZPN** projection (which astropy does not natively fit).

    Parameters
    ----------
    xy : tuple of arrays ``(x, y)`` or ``(2, N)`` array
        Pixel coordinates (same convention as the astropy function).
    world_coords : `~astropy.coordinates.SkyCoord`
        Reference sky positions.
    proj_point : str, optional
        Passed through to astropy for non-ZPN projections.
    projection : `~astropy.wcs.WCS` or other, optional
        Projection template.  If this is a WCS with ``RA---ZPN / DEC--ZPN``
        CTYPEs, :func:`fit_zpn_wcs_from_points` is used instead.
    sip_degree : int or None, optional
        SIP polynomial degree (TAN only).  For ZPN this is ignored;
        use *pv_deg* instead.
    pv_deg : int, optional
        ZPN PV polynomial degree (``PV2_0 … PV2_pv_deg``).  Only used
        when the projection is ZPN.  Default 5.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        Fitted WCS (same return type as the astropy function).
    """
    from astropy.wcs.utils import fit_wcs_from_points as _astropy_fit

    # Detect ZPN projection
    is_zpn = False
    if isinstance(projection, WCS):
        try:
            is_zpn = "ZPN" in projection.wcs.ctype[0]
        except Exception:
            pass

    if is_zpn:
        # Convert xy to (N, 2) array expected by fit_zpn_wcs_from_points
        xy_arr = np.asarray(xy, dtype=float)
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            # (x_array, y_array) form
            xy_arr = np.column_stack(
                [np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)]
            )
        elif xy_arr.ndim == 2 and xy_arr.shape[0] == 2 and xy_arr.shape[1] != 2:
            # (2, N) -> (N, 2)
            xy_arr = xy_arr.T

        wcs_best, _result = fit_zpn_wcs_from_points(
            xy_arr, world_coords, wcs_init=projection, pv_deg=pv_deg
        )
        return wcs_best

    # ---------- Non-ZPN: delegate to astropy ----------
    # SIP only makes sense for TAN-based projections
    effective_sip = sip_degree
    if isinstance(projection, WCS):
        try:
            if "TAN" not in projection.wcs.ctype[0]:
                effective_sip = None
        except Exception:
            pass

    if effective_sip is not None and effective_sip > 0:
        try:
            return _astropy_fit(
                xy, world_coords,
                proj_point=proj_point,
                projection=projection,
                sip_degree=int(effective_sip),
            )
        except TypeError:
            # Older astropy without sip_degree kwarg
            return _astropy_fit(
                xy, world_coords,
                proj_point=proj_point,
                projection=projection,
            )

    return _astropy_fit(
        xy, world_coords,
        proj_point=proj_point,
        projection=projection,
    )
