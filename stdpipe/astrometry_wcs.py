from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u


def _sky_residuals_arcsec(
    wcs: WCS, xy: np.ndarray, frame, off: SkyOffsetFrame, sky_off: SkyCoord, coslat: np.ndarray
) -> np.ndarray:
    """
    Residuals in offset-frame arcsec around the field center.

    The reference positions are precomputed in the offset frame (`sky_off`,
    with `coslat` = cos of their offset latitudes) since they do not change
    during the fit.  Longitude offsets are scaled by cos(lat) so that both
    components are proper angular distances even far from the center.

    Returns concatenated [d_lon_arcsec, d_lat_arcsec] per point.
    """
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    model = SkyCoord(ra * u.deg, dec * u.deg, frame=frame)
    model_off = model.transform_to(off)

    dlon = (model_off.lon - sky_off.lon).to_value(u.arcsec) * coslat
    dlat = (model_off.lat - sky_off.lat).to_value(u.arcsec)
    return np.concatenate([dlon, dlat])


def _spherical_mean(sky: SkyCoord) -> SkyCoord:
    """Direction of the mean unit vector of *sky* (robust to the lon=0/360 wrap)."""
    xyz = sky.cartesian.xyz.value
    mean = np.mean(np.atleast_2d(xyz), axis=1)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot determine field center from reference positions")
    mean = mean / norm
    return SkyCoord(
        np.degrees(np.arctan2(mean[1], mean[0])) * u.deg,
        np.degrees(np.arcsin(np.clip(mean[2], -1.0, 1.0))) * u.deg,
        frame=sky.frame,
    )


def _pack_params(w: WCS, pv_deg: int) -> np.ndarray:
    # CRPIX, CRVAL, CD(2x2), PV2_0..PV2_pv_deg
    p = []
    p += [float(w.wcs.crpix[0]), float(w.wcs.crpix[1])]
    p += [float(w.wcs.crval[0]), float(w.wcs.crval[1])]
    cd = np.array(w.wcs.cd, dtype=float)
    p += [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]

    pv = np.zeros(pv_deg + 1, dtype=float)
    # astropy stores PV in w.wcs.get_pv() / w.wcs.set_pv(); get_pv returns list of (i, m, value)
    for i, m, val in w.wcs.get_pv():
        if i == 2 and 0 <= m <= pv_deg:
            pv[m] = float(val)
    p += pv.tolist()
    return np.array(p, dtype=float)


def _unpack_params_to_wcs(base: WCS, p: np.ndarray, pv_deg: int) -> WCS:
    w = base.deepcopy()
    w.wcs.crpix = [p[0], p[1]]
    w.wcs.crval = [p[2], p[3]]
    w.wcs.cd = np.array([[p[4], p[5]], [p[6], p[7]]], dtype=float)

    pv_vals = p[8 : 8 + (pv_deg + 1)]
    # Preserve any PV keywords for other axes (e.g. PV1_*)
    pv_list = [(i, m, val) for (i, m, val) in base.wcs.get_pv() if i != 2]
    pv_list += [(2, m, float(pv_vals[m])) for m in range(pv_deg + 1)]
    w.wcs.set_pv(pv_list)
    return w


def _normalize_zpn_pv1(w: WCS) -> WCS:
    """Normalize a ZPN WCS so that PV2_1 = 1, absorbing the scale into CD.

    The ZPN projection R = Σ PV2_m·θ^m has a degeneracy: scaling all PV
    by ``c`` and CD by ``c`` gives an identical pixel→sky mapping.  This
    function normalizes PV2_1 to 1 (the FITS standard convention), moving
    any absorbed plate-scale factor into the CD matrix.

    Parameters
    ----------
    w : WCS
        ZPN WCS (modified in place).

    Returns
    -------
    WCS
        The same object, normalized.
    """
    pv_dict = {}
    for i, m, val in w.wcs.get_pv():
        if i == 2:
            pv_dict[m] = val

    pv1 = pv_dict.get(1, 1.0)
    if not np.isfinite(pv1) or abs(pv1) < 1e-15 or abs(pv1 - 1.0) < 1e-12:
        return w  # already normalized or can't normalize

    scale = 1.0 / pv1

    # CD' = CD/pv1 so that intermediate coords scale together with PV.
    cd = np.array(w.wcs.cd, dtype=float)
    w.wcs.cd = cd / pv1

    # Scale all PV2 coefficients
    pv_list = [(i, m, val) for (i, m, val) in w.wcs.get_pv() if i != 2]
    for m, val in pv_dict.items():
        pv_list.append((2, m, float(val * scale)))
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
    verbose: bool = False,
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
    verbose : bool or callable
        Whether to show verbose messages during the run of the function or not.
        May also be a print-like function.

    Returns
    -------
    wcs_best : astropy.wcs.WCS
    result : scipy OptimizeResult (or None if nothing was fitted)

    Notes
    -----
    For stability, the solver runs in two stages when *fit_pv* is True:
    it first fits CRPIX/CRVAL/CD with PV fixed, then fits all free
    parameters (including PV) with conservative bounds to prevent
    invalid projections.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise RuntimeError(
            "fit_zpn_wcs_from_points requires SciPy (scipy.optimize.least_squares)"
        ) from e

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2)")
    if len(sky) != xy.shape[0]:
        raise ValueError("sky and xy must have the same length")

    # Parameter packing assumes the CD convention; fold PC/CDELT into CD if needed
    if not wcs_init.wcs.has_cd():
        cd = np.array(wcs_init.pixel_scale_matrix, dtype=float)
        wcs_init = wcs_init.deepcopy()
        wcs_init.wcs.cdelt = np.ones(2)
        if wcs_init.wcs.has_pc():
            wcs_init.wcs.__delattr__("pc")
        wcs_init.wcs.cd = cd

    if fit_pv:
        pv1_init = 0.0
        for i, m, val in wcs_init.wcs.get_pv():
            if i == 2 and m == 1:
                pv1_init = float(val)
        if not np.isfinite(pv1_init) or pv1_init == 0:
            raise ValueError(
                "Initial ZPN WCS has PV2_1 = 0 or undefined; the linear radial term "
                "must be non-zero to fit PV coefficients "
                "(e.g. initialize the WCS with tan_wcs_to_zpn())"
            )

    # Stable center for residual evaluation: spherical mean of the reference
    # positions (robust to the lon=0/360 wrap, unlike a coordinate-wise median)
    center = _spherical_mean(sky)

    # The offset frame and reference positions in it are constant during the
    # fit, so compute them once here rather than per residual evaluation
    off = SkyOffsetFrame(origin=center)
    sky_off = sky.transform_to(off)
    coslat = np.cos(sky_off.lat.to_value(u.rad))

    def _estimate_theta_max_deg(w: WCS) -> float | None:
        if w.pixel_shape is None:
            return None

        nx, ny = w.pixel_shape
        if nx is None or ny is None:
            return None

        # Prefer a pixel-scale estimate (robust to projection issues)
        pixel_scale_deg = None
        try:
            pscales = w.proj_plane_pixel_scales()
            if hasattr(pscales[0], "to_value"):
                pixel_scale_deg = float(np.mean([s.to_value(u.deg) for s in pscales]))
            else:
                pixel_scale_deg = float(np.mean(pscales))
        except Exception:
            pixel_scale_deg = None

        if pixel_scale_deg is not None and np.isfinite(pixel_scale_deg) and pixel_scale_deg > 0:
            r_pix = 0.5 * np.hypot(nx, ny)
            return max(float(pixel_scale_deg * r_pix), 0.01)

        # Fallback: estimate from footprint if possible
        try:
            crpix = np.array(w.wcs.crpix, dtype=float)
            corners = np.array(
                [[0.0, 0.0], [nx - 1.0, 0.0], [0.0, ny - 1.0], [nx - 1.0, ny - 1.0]],
                dtype=float,
            )
            ra_c, dec_c = w.all_pix2world(crpix[0], crpix[1], 0)
            ra_k, dec_k = w.all_pix2world(corners[:, 0], corners[:, 1], 0)
            if (
                np.isfinite(ra_c)
                and np.isfinite(dec_c)
                and np.all(np.isfinite(ra_k))
                and np.all(np.isfinite(dec_k))
            ):
                center_c = SkyCoord(ra_c * u.deg, dec_c * u.deg, frame="icrs")
                sky_k = SkyCoord(ra_k * u.deg, dec_k * u.deg, frame="icrs")
                theta = float(np.max(center_c.separation(sky_k).to_value(u.deg)))
                return max(theta, 0.01)
        except Exception:
            pass

        return None

    def _make_bounds(p0: np.ndarray, mask: np.ndarray, base: WCS, allow_pv: bool) -> tuple:
        # Build bounds over the full parameter vector, then slice by mask.
        # CRPIX/CRVAL/CD unbounded by default
        lb = np.full_like(p0, -np.inf, dtype=float)
        ub = np.full_like(p0, +np.inf, dtype=float)

        if allow_pv and pv_deg >= 1:
            # PV bounds: keep solution in a physically plausible neighborhood
            pv0 = float(p0[8 + 0])
            pv1 = float(p0[8 + 1])  # non-zero, guaranteed by the check above

            theta_max_deg = _estimate_theta_max_deg(base)
            pv1_abs = abs(pv1)

            # PV2_0 ~ 0 (allow small drift; keep init inside bounds)
            pv0_abs = max(1e-3, abs(pv0) * 2.0)
            lb[8 + 0] = pv0 - pv0_abs
            ub[8 + 0] = pv0 + pv0_abs

            # PV2_1 (linear): allow moderate range, keep the initial sign
            lo = min(max(0.1, pv1_abs * 0.2), pv1_abs)
            hi = pv1_abs * 5.0
            if pv1 > 0:
                lb[8 + 1], ub[8 + 1] = lo, hi
            else:
                lb[8 + 1], ub[8 + 1] = -hi, -lo

            # Higher-order PV terms: limit contribution at field edge
            pv_frac = 0.3  # allow ~30% of linear term at the edge
            for m in range(2, pv_deg + 1):
                pv_m = float(p0[8 + m])
                if theta_max_deg is not None and np.isfinite(theta_max_deg) and theta_max_deg > 0:
                    abs_bound = pv_frac * pv1_abs / (theta_max_deg ** (m - 1))
                else:
                    abs_bound = max(abs(pv_m) * 5.0, 1e-6)
                # Keep initial value inside bounds
                abs_bound = max(abs_bound, abs(pv_m) * 2.0)
                lb[8 + m] = pv_m - abs_bound
                ub[8 + m] = pv_m + abs_bound

        return lb[mask], ub[mask]

    def _fit_with_mask(base: WCS, allow_pv: bool):
        p0 = _pack_params(base, pv_deg=pv_deg)

        # Build a mask over parameters to optionally freeze blocks
        mask = np.ones_like(p0, dtype=bool)
        # indices: 0..1 CRPIX, 2..3 CRVAL, 4..7 CD, 8.. PV
        if not fit_crpix:
            mask[0:2] = False
        if not fit_crval:
            mask[2:4] = False
        if not fit_cd:
            mask[4:8] = False
        if not allow_pv:
            mask[8:] = False
        elif pv_deg >= 1:
            # PV2_1 (linear scale) is degenerate with CD — let it float.
            # Caller normalizes after fit via _normalize_zpn_pv1().
            pass

        if not np.any(mask):
            return base, None

        free0 = p0[mask]
        lb, ub = _make_bounds(p0, mask, base, allow_pv=allow_pv)

        def make_wcs_from_free(free: np.ndarray) -> WCS:
            p = p0.copy()
            p[mask] = free
            return _unpack_params_to_wcs(base, p, pv_deg=pv_deg)

        def fun(free: np.ndarray) -> np.ndarray:
            w = make_wcs_from_free(free)
            try:
                res = _sky_residuals_arcsec(w, xy, sky.frame, off, sky_off, coslat)
            except ValueError:
                # wcslib may reject some in-bounds PV combinations as an
                # invalid projection; report a very poor fit instead of
                # crashing so the optimizer backs off
                return np.full(2 * xy.shape[0], 1e6)
            if not np.all(np.isfinite(res)):
                res = np.nan_to_num(res, nan=1e6, posinf=1e6, neginf=-1e6)
            return res

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

    def _log_residuals(label: str, res) -> None:
        if res is None:
            return
        n = xy.shape[0]
        r = np.hypot(res.fun[:n], res.fun[n:])
        log(
            f"ZPN fit {label}: median residual {np.median(r):.3g} arcsec "
            f"over {n} points, {res.nfev} function evaluations"
        )

    # Two-stage fit: first solve CRPIX/CRVAL/CD with PV fixed,
    # then allow PV with conservative bounds.
    w_curr = wcs_init
    res_last = None

    if fit_pv and (fit_crpix or fit_crval or fit_cd):
        w_curr, res_last = _fit_with_mask(w_curr, allow_pv=False)
        _log_residuals("linear stage (PV fixed)", res_last)

    w_curr, res_last = _fit_with_mask(w_curr, allow_pv=fit_pv)
    _log_residuals("final stage", res_last)

    # Normalize so PV2_1 = 1 (breaks the CD/PV degeneracy)
    if fit_pv:
        w_curr = _normalize_zpn_pv1(w_curr)

    return w_curr, res_last


def _fit_zpn_sip(
    wcs_zpn,
    xy,
    sky,
    sip_degree=2,
    pv_deg=5,
    n_iter=15,
    robust_loss="soft_l1",
    f_scale_arcsec=2.0,
    max_nfev=200,
    verbose=False,
):
    """Add SIP distortion corrections on top of a ZPN WCS.

    The SIP polynomials A(u,v), B(u,v) correct for non-radial distortions
    (coma, astigmatism, etc.) that the ZPN radial model cannot capture.

    Iterates between fitting SIP coefficients (pixel-space residuals) and
    re-optimizing PV parameters with SIP applied. CRPIX is kept fixed in these
    re-fits: moving the projection center is nearly degenerate with the
    quadratic SIP terms on small fields, and lets CRPIX (the center of the SIP
    polynomials) drift far outside the image, where the fit is ill-conditioned.

    Parameters
    ----------
    wcs_zpn : WCS
        ZPN WCS with PV parameters already fitted (no SIP).
    xy : (N, 2) array
        Pixel coordinates of matched sources (0-based).
    sky : SkyCoord
        Reference sky positions.
    sip_degree : int
        SIP polynomial order (default 2).
    pv_deg : int
        ZPN PV polynomial degree for re-fitting.
    n_iter : int
        Maximal number of PV+SIP alternation iterations.
    robust_loss, f_scale_arcsec, max_nfev :
        Passed to ``fit_zpn_wcs_from_points`` for PV re-fitting.
    verbose : bool or callable
        Whether to show verbose messages during the run of the function or not.
        May also be a print-like function.

    Returns
    -------
    wcs_sip : WCS
        ZPN-SIP WCS with both PV and SIP coefficients.
    """
    from astropy.wcs.wcs import Sip

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    xy = np.asarray(xy, dtype=float)
    w = wcs_zpn.deepcopy()

    # SIP term indices: (p, q) with 2 <= p+q <= sip_degree
    pq = []
    for total in range(2, sip_degree + 1):
        for q in range(total + 1):
            pq.append((total - q, q))

    crpix = np.array(w.wcs.crpix, dtype=float)  # 1-based
    prev_q90 = np.inf

    for iteration in range(n_iter):
        # Strip SIP to get the base ZPN-only WCS
        w_nosip = w.deepcopy()
        w_nosip.sip = None
        if '-SIP' in w_nosip.wcs.ctype[0]:
            w_nosip.wcs.ctype = [c.replace('-SIP', '') for c in w_nosip.wcs.ctype]

        # Undistorted pixel coords: where ZPN projection places catalog stars.
        # quiet=True prevents NoConvergence for points near the projection edge;
        # non-finite results are excluded below.
        x_cat, y_cat = w_nosip.all_world2pix(sky.ra.deg, sky.dec.deg, 0, quiet=True)
        u_cat = x_cat - (crpix[0] - 1)
        v_cat = y_cat - (crpix[1] - 1)

        # Observed (distorted) pixel coords
        u_obs = xy[:, 0] - (crpix[0] - 1)
        v_obs = xy[:, 1] - (crpix[1] - 1)

        # SIP convention: u_distorted = u_undistorted + A(u, v)
        # u_obs = undistorted (raw pixel); u_cat = distorted (from ZPN inverse)
        # So: A(u_obs, v_obs) = u_cat - u_obs
        du = u_cat - u_obs
        dv = v_cat - v_obs

        # Filter outliers (use median absolute deviation)
        dist = np.sqrt(du**2 + dv**2)
        finite = np.isfinite(dist)
        if np.sum(finite) < len(pq) + 5:
            log(f"SIP iteration {iteration}: too few finite points ({np.sum(finite)}), stopping")
            break
        med_dist = np.median(dist[finite])
        mad = np.median(np.abs(dist[finite] - med_dist))
        good = finite & (dist < med_dist + 5 * max(mad, 0.5))  # generous 5-sigma clip

        if np.sum(good) < len(pq) + 5:
            log(f"SIP iteration {iteration}: too few points after clipping ({np.sum(good)}), stopping")
            break

        # SIP basis uses undistorted (raw pixel) coordinates
        u_fit = u_obs[good]
        v_fit = v_obs[good]
        du_fit = du[good]
        dv_fit = dv[good]

        # Normalize coordinates to prevent ill-conditioning for high
        # SIP degrees.  Without this, u^5 ~ 2000^5 ~ 3e16 creates a
        # design matrix with condition number > 1e14.
        u_scale = max(np.abs(u_fit).max(), 1.0)
        v_scale = max(np.abs(v_fit).max(), 1.0)
        u_norm = u_fit / u_scale
        v_norm = v_fit / v_scale

        # Build SIP basis matrix in normalized coordinates
        basis = np.column_stack([u_norm**p * v_norm**q for p, q in pq])

        # Fit A and B coefficients via least squares (normalized)
        ca_norm, _, _, _ = np.linalg.lstsq(basis, du_fit, rcond=None)
        cb_norm, _, _, _ = np.linalg.lstsq(basis, dv_fit, rcond=None)

        # Convert back to original (unnormalized) SIP coefficients
        a_vals = np.zeros((sip_degree + 1, sip_degree + 1))
        b_vals = np.zeros((sip_degree + 1, sip_degree + 1))
        for k, (p, q) in enumerate(pq):
            scale_pq = u_scale**p * v_scale**q
            a_vals[p, q] = ca_norm[k] / scale_pq
            b_vals[p, q] = cb_norm[k] / scale_pq

        # Apply SIP to the WCS
        if '-SIP' not in w.wcs.ctype[0]:
            w.wcs.ctype = [c + '-SIP' for c in w.wcs.ctype]

        w.sip = Sip(
            a_vals,
            b_vals,
            np.zeros((sip_degree + 1, sip_degree + 1)),
            np.zeros((sip_degree + 1, sip_degree + 1)),
            crpix,
        )

        # Check convergence: stop when SIP corrections stabilize.
        # Use 90th percentile (sensitive to outer-field + center) rather
        # than median which converges before the center is corrected.
        curr_q90 = np.percentile(dist[good], 90)
        log(f"SIP iteration {iteration}: q90 distortion {curr_q90:.4g} pix over {np.sum(good)} points")
        if iteration >= 2 and prev_q90 < np.inf:
            rel_change = abs(curr_q90 - prev_q90) / max(prev_q90, 1e-6)
            if rel_change < 0.005:
                log(f"SIP converged after {iteration + 1} iterations")
                break
        prev_q90 = curr_q90

        # Re-fit PV with SIP now applied (last iteration skip PV refit)
        if iteration < n_iter - 1:
            w, _ = fit_zpn_wcs_from_points(
                xy,
                sky,
                w,
                pv_deg=pv_deg,
                fit_crpix=False,
                robust_loss=robust_loss,
                f_scale_arcsec=f_scale_arcsec,
                max_nfev=max_nfev,
                verbose=verbose,
            )
            crpix = np.array(w.wcs.crpix, dtype=float)

    return w


def tan_wcs_to_zpn(
    w_tan: WCS,
    pv_deg: int = 5,
    n_samples: int = 256,
    theta_max_deg: float | None = None,
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

    SIP and other pixel-space distortions of the input WCS are dropped
    (they are not representable in the radial ZPN model); refit them on
    top of the result if needed, e.g. with :func:`_fit_zpn_sip`.

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

    Returns
    -------
    w_zpn : astropy.wcs.WCS
        A ZPN WCS with same CRVAL/CRPIX/CD and PV2_m initialized.
    """
    ctype1, ctype2 = w_tan.wcs.ctype
    if len(ctype1) < 8 or len(ctype2) < 8:
        raise ValueError("Expected CTYPE like 'RA---TAN'/'DEC--TAN'.")

    # Clean linear WCS (CRPIX/CRVAL/CD + metadata) with ZPN projection;
    # strips SIP and PV parameters
    w = _wcs_to_linear(w_tan, "ZPN")

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
        corners = np.array(
            [
                [0.0, 0.0],
                [nx - 1.0, 0.0],
                [0.0, ny - 1.0],
                [nx - 1.0, ny - 1.0],
            ],
            dtype=float,
        )

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

    # Mild weighting emphasizing the central region: stabilizes the fit
    # against the steep TAN growth near theta_max; only used for initialization
    wgt = 1.0 / (1.0 + (theta / (0.6 * theta_max_deg)) ** 2)
    Aw = A * wgt[:, None]
    bw = r_tan_deg * wgt

    coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    pv_list = [(2, 0, 0.0)] + [(2, m, float(coeffs[m - 1])) for m in range(1, pv_deg + 1)]
    w.wcs.set_pv(pv_list)

    return w


def convert_wcs_projection(
    wcs_input: WCS,
    target_projection: str,
    pv_deg: int = 5,
) -> WCS:
    """Convert a celestial WCS to a different projection type.

    Parameters
    ----------
    wcs_input : WCS
        Input WCS (any celestial projection).
    target_projection : str
        Target projection code, e.g. ``'TAN'``, ``'ZPN'``, ``'STG'``,
        ``'ARC'``, ``'ZEA'``, ``'SIN'``, etc.
    pv_deg : int, optional
        ZPN PV polynomial degree (only used when *target_projection* is
        ``'ZPN'``).  Default 5.

    Returns
    -------
    WCS
        New WCS with the target projection.  For ZPN the PV coefficients
        are initialised from the TAN radial law — an exact match for TAN
        input, and a generic starting point (meant to be refined by a
        subsequent fit) for other input projections.  For other targets
        CRPIX/CRVAL/CD are copied and CTYPE is replaced, so the mapping
        away from the reference point changes; refit afterwards.
    """
    target = target_projection.upper().strip()

    # Detect current projection code (last 3 chars of CTYPE after the dash)
    try:
        cur_proj = wcs_input.wcs.ctype[0].split('-')[-1]
        # Strip trailing SIP suffix if present
        if cur_proj == 'SIP':
            parts = wcs_input.wcs.ctype[0].replace('-SIP', '').split('-')
            cur_proj = parts[-1]
    except Exception:
        cur_proj = ''

    if cur_proj == target:
        return wcs_input.deepcopy()

    # ZPN needs special initialisation (PV coefficients)
    if target == 'ZPN':
        # tan_wcs_to_zpn works from TAN (with or without SIP)
        # For non-TAN input, we first need a TAN-like WCS base
        if 'TAN' in cur_proj:
            return tan_wcs_to_zpn(wcs_input, pv_deg=pv_deg)
        else:
            # Build a minimal TAN WCS from the input's linear terms,
            # then convert to ZPN
            w_tan = _wcs_to_linear(wcs_input, 'TAN')
            return tan_wcs_to_zpn(w_tan, pv_deg=pv_deg)

    # For all other projections: copy linear WCS with new CTYPE
    return _wcs_to_linear(wcs_input, target)


def _wcs_to_linear(wcs_input: WCS, proj_code: str) -> WCS:
    """Create a clean linear WCS with the same pointing but a new projection.

    Copies CRPIX, CRVAL, CD (or derives CD from PC+CDELT), metadata, and
    pixel_shape.  Strips SIP distortion and PV parameters.
    """
    w = WCS(naxis=2)

    # CD matrix; pixel_scale_matrix correctly folds CDELT_i * PC_ij
    # (row scaling) when the input uses the PC/CDELT convention
    cd = np.array(wcs_input.pixel_scale_matrix, dtype=float)

    w.wcs.crpix = np.array(wcs_input.wcs.crpix, dtype=float)
    w.wcs.crval = np.array(wcs_input.wcs.crval, dtype=float)
    w.wcs.cd = cd

    # Build CTYPE: preserve axis names (e.g. 'RA---' / 'DEC--')
    ctype1, ctype2 = wcs_input.wcs.ctype
    # Strip any existing projection + SIP suffix
    base1 = ctype1.replace('-SIP', '')[:5]
    base2 = ctype2.replace('-SIP', '')[:5]
    w.wcs.ctype = (base1 + proj_code, base2 + proj_code)

    # Copy metadata
    for attr in ('cunit', 'radesys', 'equinox'):
        try:
            setattr(w.wcs, attr, getattr(wcs_input.wcs, attr))
        except Exception:
            pass
    for attr in ('lonpole', 'latpole'):
        try:
            val = getattr(wcs_input.wcs, attr)
            if np.isfinite(val):
                setattr(w.wcs, attr, float(val))
        except Exception:
            pass

    if wcs_input.pixel_shape is not None:
        w.pixel_shape = wcs_input.pixel_shape

    return w


def _fit_tan_sip_robust(
    xy,
    world_coords,
    proj_point="center",
    projection=None,
    sip_degree=2,
    robust_loss="soft_l1",
    f_scale=None,
    verbose=False,
):
    """Fit TAN+SIP WCS with robust loss function.

    Replicates astropy's ``fit_wcs_from_points`` two-stage fitting
    (linear WCS, then joint CD + CRPIX + SIP) but uses a robust loss
    in the linear stage and a two-pass approach for the SIP stage:
    first L2 to capture the distortion pattern, then robust re-fit
    with data-driven ``f_scale`` to suppress outliers.

    Parameters
    ----------
    xy : tuple of arrays ``(x, y)``
        Pixel coordinates.
    world_coords : `~astropy.coordinates.SkyCoord`
        Reference sky positions.
    proj_point : str or SkyCoord
        Projection center ('center' or explicit).
    projection : WCS or None
        Template WCS; used as initial guess for CD matrix.
    sip_degree : int
        SIP polynomial degree.
    robust_loss : str
        Loss function for `scipy.optimize.least_squares`.
    f_scale : float or None
        Soft margin for robust loss in the SIP stage.  If None
        (default), estimated adaptively from the L2 SIP residuals.
        The linear stage always uses a data-driven scale.
    verbose : bool or callable
        Whether to show verbose messages during the run of the function or not.
        May also be a print-like function.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
    """
    from scipy.optimize import least_squares

    from astropy.wcs.utils import celestial_frame_to_wcs
    from astropy.wcs.wcs import Sip

    try:
        # Private astropy helpers (stable since astropy 3.x, but not public API)
        from astropy.wcs.utils import _linear_wcs_fit, _sip_fit
    except ImportError as e:
        raise RuntimeError(
            "astropy internals (_linear_wcs_fit/_sip_fit) are not available in "
            "this astropy version; _fit_tan_sip_robust needs updating"
        ) from e

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    xp, yp = xy
    try:
        lon, lat = world_coords.data.lon.deg, world_coords.data.lat.deg
    except AttributeError:
        unit_sph = world_coords.unit_spherical
        lon, lat = unit_sph.lon.deg, unit_sph.lat.deg

    use_center_as_proj_point = str(proj_point) == "center"

    # Build WCS template
    if isinstance(projection, str) or projection is None:
        proj_code = projection if isinstance(projection, str) else "TAN"
        wcs = celestial_frame_to_wcs(frame=world_coords.frame, projection=proj_code)
    else:
        wcs = projection.deepcopy()
        wcs.sip = None

    if wcs.wcs.has_pc():
        # pixel_scale_matrix folds CDELT_i * PC_ij with correct row scaling
        wcs.wcs.cd = wcs.pixel_scale_matrix
        wcs.wcs.cdelt = (1.0, 1.0)
        wcs.wcs.__delattr__("pc")

    xpmin, xpmax = xp.min(), xp.max()
    ypmin, ypmax = yp.min(), yp.max()

    wcs.pixel_shape = (
        1 if xpmax <= 0.0 else int(np.ceil(xpmax)),
        1 if ypmax <= 0.0 else int(np.ceil(ypmax)),
    )

    if use_center_as_proj_point:
        sc1 = SkyCoord(lon.min() * u.deg, lat.max() * u.deg)
        sc2 = SkyCoord(lon.max() * u.deg, lat.min() * u.deg)
        pa = sc1.position_angle(sc2)
        sep = sc1.separation(sc2)
        midpoint_sc = sc1.directional_offset_by(pa, sep / 2)
        wcs.wcs.crval = (midpoint_sc.data.lon.deg, midpoint_sc.data.lat.deg)
        wcs.wcs.crpix = ((xpmax + xpmin) / 2.0, (ypmax + ypmin) / 2.0)
    else:
        proj_point = proj_point.transform_to(world_coords.frame)
        wcs.wcs.crval = (proj_point.data.lon.deg, proj_point.data.lat.deg)
        close = lambda l, p: p[np.argmin(np.abs(l))]
        wcs.wcs.crpix = (
            close(lon - wcs.wcs.crval[0], xp + 1),
            close(lat - wcs.wcs.crval[1], yp + 1),
        )

    if xpmin == xpmax:
        xpmin, xpmax = xpmin - 0.5, xpmax + 0.5
    if ypmin == ypmax:
        ypmin, ypmax = ypmin - 0.5, ypmax + 0.5

    # --- Stage 1: linear WCS (CD + CRPIX) ---
    # Use robust loss here: linear residuals include distortion as systematic
    # pattern that looks like outliers; robust loss prevents them from biasing
    # the linear fit.
    p0_lin = np.concatenate([wcs.wcs.cd.flatten(), wcs.wcs.crpix.flatten()])
    lin_resids = _linear_wcs_fit(p0_lin, lon, lat, xp, yp, wcs)
    lin_f_scale = max(float(np.median(np.abs(lin_resids)) * 3), 1.0 / 3600)

    fit = least_squares(
        _linear_wcs_fit,
        p0_lin,
        args=(lon, lat, xp, yp, wcs),
        bounds=[
            [-np.inf, -np.inf, -np.inf, -np.inf, xpmin + 1, ypmin + 1],
            [np.inf, np.inf, np.inf, np.inf, xpmax + 1, ypmax + 1],
        ],
        loss=robust_loss,
        f_scale=lin_f_scale,
        method="trf",
    )
    wcs.wcs.crpix = np.array(fit.x[4:6])
    wcs.wcs.cd = np.array(fit.x[0:4].reshape((2, 2)))
    log(
        f"TAN-SIP linear stage: median residual "
        f"{np.median(np.abs(fit.fun)) * 3600:.3g} arcsec"
    )

    # --- Stage 2: joint CD + CRPIX + SIP ---
    if "-SIP" not in wcs.wcs.ctype[0]:
        wcs.wcs.ctype = [x + "-SIP" for x in wcs.wcs.ctype]

    coef_names = [
        f"{i}_{j}"
        for i in range(sip_degree + 1)
        for j in range(sip_degree + 1)
        if (i + j) < (sip_degree + 1) and (i + j) > 1
    ]

    sip_bounds = (
        [xpmin + 1, ypmin + 1] + [-np.inf] * (4 + 2 * len(coef_names)),
        [xpmax + 1, ypmax + 1] + [np.inf] * (4 + 2 * len(coef_names)),
    )
    # Parameter layout of _sip_fit: [CRPIX(2), CD(4), A_coeffs, B_coeffs]
    sip_args = (lon, lat, xp, yp, wcs, sip_degree, coef_names)

    # Pass 1: L2 fit to capture the full distortion pattern (SIP starts at 0)
    p0_sip = np.concatenate(
        (
            np.array(wcs.wcs.crpix),
            wcs.wcs.cd.flatten(),
            np.zeros(2 * len(coef_names)),
        )
    )

    # Compute parameter scales for the optimizer.
    # SIP coefficients A_{p,q} multiply (u-crpix)^p * (v-crpix)^q where
    # pixel offsets can be ~U pixels.  Expected coefficient magnitude is
    # ~1/U^(p+q), which spans many orders of magnitude for high SIP orders.
    # Without proper scaling the optimizer cannot find meaningful SIP≥4
    # coefficients on wide-field images.
    U = max(
        abs(xpmax - wcs.wcs.crpix[0]),
        abs(xpmin - wcs.wcs.crpix[0]),
        abs(ypmax - wcs.wcs.crpix[1]),
        abs(ypmin - wcs.wcs.crpix[1]),
        1.0,
    )
    sip_x_scale = np.ones(len(p0_sip))
    # CRPIX: O(pixels) → scale 1
    # CD: O(deg/pixel) → use current magnitude
    cd_scale = max(np.abs(wcs.wcs.cd).max(), 1e-10)
    sip_x_scale[2:6] = cd_scale
    # SIP coefficients: O(1/U^(p+q))
    for k, coef_name in enumerate(coef_names):
        p_deg, q_deg = int(coef_name[0]), int(coef_name[2])
        order = p_deg + q_deg
        scale = 1.0 / U**order
        sip_x_scale[6 + k] = scale
        sip_x_scale[6 + len(coef_names) + k] = scale

    fit = least_squares(
        _sip_fit,
        p0_sip,
        args=sip_args,
        bounds=sip_bounds,
        x_scale=sip_x_scale,
    )

    # Pass 2: robust re-fit starting from L2 solution
    # f_scale estimated from L2 residuals so distortion signal is inlier
    sip_resids = fit.fun
    if f_scale is None:
        sip_f_scale = max(float(np.median(np.abs(sip_resids)) * 3), 1e-7)
    else:
        sip_f_scale = float(f_scale)

    fit = least_squares(
        _sip_fit,
        fit.x,  # warm-start from L2 solution
        args=sip_args,
        bounds=sip_bounds,
        loss=robust_loss,
        f_scale=sip_f_scale,
        method="trf",
        x_scale=sip_x_scale,
    )
    log(
        f"TAN-SIP robust stage: median residual "
        f"{np.median(np.abs(fit.fun)) * 3600:.3g} arcsec, f_scale {sip_f_scale * 3600:.3g} arcsec"
    )

    coef_fit = (
        list(fit.x[6 : 6 + len(coef_names)]),
        list(fit.x[6 + len(coef_names) :]),
    )

    wcs.wcs.cd = fit.x[2:6].reshape((2, 2))
    wcs.wcs.crpix = fit.x[0:2]

    a_vals = np.zeros((sip_degree + 1, sip_degree + 1))
    b_vals = np.zeros((sip_degree + 1, sip_degree + 1))

    for coef_name in coef_names:
        a_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[0].pop(0)
        b_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[1].pop(0)

    wcs.sip = Sip(
        a_vals,
        b_vals,
        np.zeros((sip_degree + 1, sip_degree + 1)),
        np.zeros((sip_degree + 1, sip_degree + 1)),
        wcs.wcs.crpix,
    )

    return wcs


def fit_wcs_from_points(
    xy,
    world_coords,
    proj_point="center",
    projection=None,
    sip_degree=None,
    pv_deg=5,
    verbose=False,
):
    """Drop-in wrapper around :func:`astropy.wcs.utils.fit_wcs_from_points`
    that also handles **ZPN** projection (which astropy does not natively fit)
    and **ZPN-SIP** (ZPN radial distortion plus SIP polynomial corrections).

    Parameters
    ----------
    xy : tuple of arrays ``(x, y)`` or ``(2, N)`` array
        Pixel coordinates (same convention as the astropy function).
    world_coords : `~astropy.coordinates.SkyCoord`
        Reference sky positions.
    proj_point : str, optional
        Passed through to astropy for non-ZPN projections.  Ignored for ZPN,
        where the projection center comes from the template WCS.
    projection : `~astropy.wcs.WCS` or other, optional
        Projection template.  If this is a WCS with ``RA---ZPN / DEC--ZPN``
        CTYPEs, the ZPN fitter is used instead of astropy's.
    sip_degree : int or None, optional
        SIP polynomial degree.  For TAN projections, controls SIP distortion
        order.  For ZPN projections, if > 0, SIP corrections are fitted on
        top of ZPN PV parameters to capture non-radial distortions, and CRPIX
        of the template is then kept fixed (see :func:`_fit_zpn_sip`).  For
        other projections it is ignored (SIP is not standard there).
    pv_deg : int, optional
        ZPN PV polynomial degree (``PV2_0 … PV2_pv_deg``).  Default 5.
    verbose : bool or callable, optional
        Whether to show verbose messages during the run of the function or not.
        May also be a print-like function.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        Fitted WCS (same return type as the astropy function).
    """
    from astropy.wcs.utils import fit_wcs_from_points as _astropy_fit

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    # Detect ZPN projection
    is_zpn = False
    if isinstance(projection, WCS):
        try:
            is_zpn = "ZPN" in projection.wcs.ctype[0]
        except Exception:
            pass

    if is_zpn:
        # Convert xy to (N, 2) array expected by fit_zpn_wcs_from_points
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            # (x_array, y_array) form
            xy_arr = np.column_stack(
                [np.asarray(xy[0], dtype=float), np.asarray(xy[1], dtype=float)]
            )
        else:
            xy_arr = np.asarray(xy, dtype=float)
            if xy_arr.ndim == 2 and xy_arr.shape[0] == 2 and xy_arr.shape[1] != 2:
                # (2, N) -> (N, 2)
                xy_arr = xy_arr.T

        zpn_deg = int(pv_deg)

        # First fit ZPN PV parameters (radial distortion)
        log(f"Fitting ZPN WCS with pv_deg={zpn_deg} using {len(xy_arr)} points")
        # With SIP following, keep CRPIX fixed: otherwise the ZPN fit absorbs
        # the quadratic distortions by moving the projection center away
        fit_sip = sip_degree is not None and int(sip_degree) > 0
        wcs_best, _result = fit_zpn_wcs_from_points(
            xy_arr,
            world_coords,
            wcs_init=projection,
            pv_deg=zpn_deg,
            fit_crpix=not fit_sip,
            verbose=verbose,
        )

        # Then fit SIP corrections for non-radial distortions
        if fit_sip:
            log(f"Fitting SIP degree {int(sip_degree)} corrections on top of ZPN")
            wcs_best = _fit_zpn_sip(
                wcs_best,
                xy_arr,
                world_coords,
                sip_degree=int(sip_degree),
                pv_deg=zpn_deg,
                verbose=verbose,
            )

        return wcs_best

    # ---------- Non-ZPN ----------
    # SIP only makes sense for TAN-based projections
    effective_sip = sip_degree
    if isinstance(projection, WCS):
        try:
            if "TAN" not in projection.wcs.ctype[0]:
                effective_sip = None
        except Exception:
            pass

    if effective_sip is not None and effective_sip > 0:
        log(f"Fitting TAN-SIP WCS with sip_degree={int(effective_sip)}")
        return _fit_tan_sip_robust(
            xy,
            world_coords,
            proj_point=proj_point,
            projection=projection,
            sip_degree=int(effective_sip),
            verbose=verbose,
        )

    if effective_sip != sip_degree:
        log("SIP is not supported for this projection, ignoring sip_degree")

    log("Fitting WCS using astropy fit_wcs_from_points")
    return _astropy_fit(
        xy,
        world_coords,
        proj_point=proj_point,
        projection=projection,
    )
