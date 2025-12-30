"""
Photometric modeling routines.

This module contains functions for photometric calibration and modeling,
including catalog matching with spatial zero-point models, S/N modeling,
and detection limit estimation.
"""


import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize, least_squares, root_scalar

from . import astrometry


def make_series(mul=1.0, x=1.0, y=1.0, order=1, sum=False, zero=True):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if zero:
        res = [np.ones_like(x) * mul]
    else:
        res = []

    for i in range(1, order + 1):
        maxr = i + 1

        for j in range(maxr):
            res.append(mul * x ** (i - j) * y ** j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res


def get_intrinsic_scatter(y, yerr, min=0, max=None):
    def log_likelihood(theta, y, yerr):
        a, b, c = theta
        model = b
        sigma2 = a * yerr ** 2 + c ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    nll = lambda *args: -log_likelihood(*args)
    C = minimize(
        nll,
        [1, 0.0, 0.0],
        args=(y, yerr),
        bounds=[[1, 1], [None, None], [min, max]],
        method='Powell',
    )

    return C.x[2]


def format_color_term(color_term, name=None, color_name=None, fmt='.2f'):
    result = []

    if color_term is None:
        return format(0.0, fmt)

    if name is not None:
        result += [name]

    if isinstance(color_term, float) or isinstance(color_term, int):
        # Scalar?..
        color_term = [color_term]

    # Here we assume it is a list
    for i,val in enumerate(color_term):
        if color_name is not None:
            sign = '-' if val > 0 else '+'  # Reverse signs!!!
            sval = format(np.abs(val), fmt)
            deg = '^%d' % (i + 1) if i > 0 else ''
            result += [sign + ' ' + sval + ' (' + color_name + ')' + deg]
        else:
            result += [format(val, fmt)]

    return " ".join(result)


def match(
    obj_ra,
    obj_dec,
    obj_mag,
    obj_magerr,
    obj_flags,
    cat_ra,
    cat_dec,
    cat_mag,
    cat_magerr=None,
    cat_color=None,
    sr=3 / 3600,
    obj_x=None,
    obj_y=None,
    spatial_order=0,
    bg_order=None,
    nonlin=False,
    threshold=5.0,
    niter=10,
    accept_flags=0,
    cat_saturation=None,
    max_intrinsic_rms=0,
    sn=None,
    verbose=False,
    robust=True,
    scale_noise=False,
    use_color=True,
    force_color_term=None,
):
    """Low-level photometric matching routine.

    It tries to build the photometric model for objects detected on the image that includes catalogue magnitude, positionally-dependent zero point, linear color term, optional additive flux term, and also takes into account possible intrinsic magnitude scatter on top of measurement errors.

    :param obj_ra: Array of Right Ascension values for the objects
    :param obj_dec: Array of Declination values for the objects
    :param obj_mag: Array of instrumental magnitude values for the objects
    :param obj_magerr: Array of instrumental magnitude errors for the objects
    :param obj_flags: Array of flags for the objects
    :param cat_ra: Array of catalogue Right Ascension values
    :param cat_dec: Array of catalogue Declination values
    :param cat_mag: Array of catalogue magnitudes
    :param cat_magerr: Array of catalogue magnitude errors
    :param cat_color: Array of catalogue color values, optional
    :param sr: Matching radius, degrees
    :param obj_x: Array of `x` coordinates of objects on the image, optional
    :param obj_y: Array of `y` coordinates of objects on the image, optional
    :param spatial_order: Order of zero point spatial polynomial (0 for constant).
    :param bg_order: Order of additive flux term spatial polynomial (None to disable this term in the model)
    :param nonlin: Whether to fit for simple non-linearity, optional
    :param threshold: Rejection threshold (relative to magnitude errors) for object-catalogue pair to be rejected from the fit
    :param niter: Number of iterations for the fitting
    :param accept_flags: Bitmask for acceptable object flags. Objects having any other bits set will be excluded from the model
    :param cat_saturation: Saturation level for the catalogue - stars brighter than this magnitude will be excluded from the fit
    :param max_intrinsic_rms: Maximal intrinsic RMS to use during the fitting. If set to 0, no intrinsic scatter is included in the noise model.
    :param sn: Minimal acceptable signal to noise ratio (1/obj_magerr) for the objects to be included in the fit
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function.
    :param robust: Whether to use robust least squares fitting routine instead of weighted least squares
    :param scale_noise: Whether to re-scale the noise model (object and catalogue magnitude errors) to match actual scatter of the data points or not. Intrinsic scatter term is not being scaled this way.
    :param use_color: Whether to use catalogue color for deriving the color term. If integer, it determines the color term order.
    :param force_color_term: Do not fit for the color term, but use this fixed value instead.
    :returns: The dictionary with photometric results, as described below.

    The results of photometric matching are returned in a dictionary with the following fields:

    -  `oidx`, `cidx`, `dist` - indices of positionally matched objects and catalogue stars, as well as their pairwise distances in degrees
    -  `omag`, `omagerr`, `cmag`, `cmagerr` - arrays of object instrumental magnitudes of matched objects, corresponding catalogue magnitudes, and their errors. Array lengths are equal to the number of positional matches.
    -  `color` - catalogue colors corresponding to the matches, or zeros if no color term fitting is requested
    -  `ox`, `oy`, `oflags` - coordinates of matched objects on the image, and their flags
    -  `zero`, `zero_err` - empirical zero points (catalogue - instrumental magnitudes) for every matched object, as well as its errors, derived as a hypotenuse of their corresponding errors.
    -  `zero_model`, `zero_model_err` - modeled "full" zero points (including color terms) for matched objects, and their corresponding errors from the fit
    -  `color_term` - fitted color term. Instrumental photometric system is defined as :code:`obj_mag = cat_mag - color*color_term`
    -  `zero_fn` - function to compute the zero point (without color term) at a given position and for a given instrumental magnitude of object, and optionally its error.
    -  `obj_zero` - zero points for all input objects (not necessarily matched to the catalogue) computed through aforementioned function, i.e. without color term
    -  `params` - Internal parameters of the fittting polynomial
    -  `intrinsic_rms`, `error_scale` - fitted values of intrinsic scatter and noise scaling
    -  `idx` - boolean index of matched objects/catalogue stars used in the final fit (i.e. not rejected during iterative thresholding, and passing initial quality cuts
    -  `idx0` - the same but with just initial quality cuts taken into account

    Returned zero point computation function has the following signature:

    :obj:`zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False)`

    where `xx` and `yy` are coordinates on the image, `mag` is object instrumental magnitude (needed to compute additive flux term). If :code:`get_err=True`, the function returns estimated zero point error instead of zero point, and `add_intrinsic_rms` controls whether this error estimation should also include intrinsic scatter term or not.

    The zero point returned by this function does not include the contribution of color term. Therefore, in order to derive the final calibrated magnitude for the object, you will need to manually add the color contribution: :code:`mag_calibrated = mag_instrumental + color*color_term`, where `color` is a true object color, and `color_term` is reported in the photometric results.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    oidx, cidx, dist = astrometry.spherical_match(obj_ra, obj_dec, cat_ra, cat_dec, sr)

    log(
        len(dist),
        'initial matches between',
        len(obj_ra),
        'objects and',
        len(cat_ra),
        'catalogue stars, sr = %.2f arcsec' % (sr * 3600),
    )
    log('Median separation is %.2f arcsec' % (np.median(dist) * 3600))

    omag = np.ma.filled(obj_mag[oidx], fill_value=np.nan)
    omag_err = np.ma.filled(obj_magerr[oidx], fill_value=np.nan)
    oflags = (
        obj_flags[oidx] if obj_flags is not None else np.zeros_like(omag, dtype=bool)
    )
    cmag = np.ma.filled(cat_mag[cidx], fill_value=np.nan)
    cmag_err = (
        np.ma.filled(cat_magerr[cidx], fill_value=np.nan)
        if cat_magerr is not None
        else np.zeros_like(cmag)
    )

    if obj_x is not None and obj_y is not None:
        x0, y0 = np.mean(obj_x[oidx]), np.mean(obj_y[oidx])
        ox, oy = obj_x[oidx], obj_y[oidx]
        x, y = obj_x[oidx] - x0, obj_y[oidx] - y0
        x, y = [np.ma.filled(_, fill_value=np.nan) for _ in (x, y)]
    else:
        x0, y0 = 0, 0
        ox, oy = np.zeros_like(omag), np.zeros_like(omag)
        x, y = np.zeros_like(omag), np.zeros_like(omag)

    # Regressor
    X = make_series(1.0, x, y, order=spatial_order)
    log('Fitting the model with spatial_order =', spatial_order)

    if bg_order is not None:
        # Spatially varying additive flux component, linearized in magnitudes
        X += make_series(-2.5 / np.log(10) / 10 ** (-0.4 * omag), x, y, order=bg_order)
        log('Adjusting background level using polynomial with bg_order =', bg_order)

    if nonlin:
        # Non-linearity
        pos_nonlin = len(X)
        X += make_series(omag, x, y, order=0)
        log('Fitting for simple non-linearity')

    if robust:
        log('Using robust fitting')
    else:
        log('Using weighted fitting')

    if cat_color is not None:
        ccolor = np.ma.filled(cat_color[cidx], fill_value=np.nan)
        if use_color:
            pos_color = len(X)
            for _ in range(int(use_color)):
                X += make_series(ccolor**(_ + 1), x, y, order=0)

            log('Using color term of order', int(use_color))
        elif force_color_term is not None:
            for i,val in enumerate(np.atleast_1d(force_color_term)):
                cmag -= val * ccolor**(i + 1)
            log('Using fixed color term', format_color_term(force_color_term))
    else:
        ccolor = np.zeros_like(cmag)

    Nparams = len(X)  # Number of parameters to be fitted

    X = np.vstack(X).T
    zero = cmag - omag  # We will build a model for this definition of zero point
    zero_err = np.hypot(omag_err, cmag_err)
    # weights = 1.0/zero_err**2

    idx0 = (
        np.isfinite(omag)
        & np.isfinite(omag_err)
        & np.isfinite(cmag)
        & np.isfinite(cmag_err)
        & ((oflags & ~accept_flags) == 0)
    )  # initial mask
    if cat_color is not None and use_color:
        idx0 &= np.isfinite(ccolor)
    if cat_saturation is not None:
        idx0 &= cmag >= cat_saturation
    if sn is not None:
        idx0 &= omag_err < 1 / sn

    log('%d objects pass initial quality cuts' % np.sum(idx0))

    idx = idx0.copy()

    intrinsic_rms = 0
    scale_err = 1
    total_err = zero_err

    for iter in range(niter):
        if np.sum(idx) < Nparams + 1:
            log(
                "Fit failed - %d objects remaining for fitting %d parameters"
                % (np.sum(idx), Nparams)
            )
            return None

        if robust:
            # Rescale the arguments with weights
            C = sm.RLM(zero[idx] / total_err[idx], (X[idx].T / total_err[idx]).T).fit()
        else:
            C = sm.WLS(zero[idx], X[idx], weights=1 / total_err[idx] ** 2).fit()

        zero_model = np.sum(X * C.params, axis=1)
        zero_model_err = np.sqrt(C.cov_params(X).diagonal())

        intrinsic_rms = (
            get_intrinsic_scatter(
                (zero - zero_model)[idx], total_err[idx], max=max_intrinsic_rms
            )
            if max_intrinsic_rms > 0
            else 0
        )

        scale_err = 1 if not scale_noise else np.sqrt(C.scale)  # rms
        total_err = np.hypot(zero_err * scale_err, intrinsic_rms)

        if threshold:
            idx1 = np.abs((zero - zero_model) / total_err)[idx] < threshold
        else:
            idx1 = np.ones_like(idx[idx])

        log(
            'Iteration',
            iter,
            ':',
            np.sum(idx),
            '/',
            len(idx),
            '- rms',
            '%.2f' % np.std((zero - zero_model)[idx0]),
            '%.2f' % np.std((zero - zero_model)[idx]),
            '- normed',
            '%.2f' % np.std((zero - zero_model)[idx] / zero_err[idx]),
            '%.2f' % np.std((zero - zero_model)[idx] / total_err[idx]),
            '- scale %.2f %.2f' % (np.sqrt(C.scale), scale_err),
            '- rms',
            '%.2f' % intrinsic_rms,
        )

        if not np.sum(~idx1):  # and new_intrinsic_rms <= intrinsic_rms:
            log('Fitting converged')
            break
        else:
            idx[idx] &= idx1

    log(np.sum(idx), 'good matches')
    if max_intrinsic_rms > 0:
        log('Intrinsic scatter is %.2f' % intrinsic_rms)

    if nonlin:
        log('Non-linearity term is %.3f' % C.params[pos_nonlin])

    # Export the model
    def zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False):
        if xx is not None and yy is not None:
            x, y = xx - x0, yy - y0
        else:
            x, y = np.zeros_like(omag), np.zeros_like(omag)

        # Ensure we do not have MaskedColumns
        x, y = [np.ma.filled(_, fill_value=np.nan) for _ in (x, y)]
        if mag is not None:
            mag = np.ma.filled(mag, np.nan)

        X = make_series(1.0, x, y, order=spatial_order)

        if bg_order is not None and mag is not None:
            X += make_series(
                -2.5 / np.log(10) / 10 ** (-0.4 * mag), x, y, order=bg_order
            )

        if nonlin and mag is not None:
            X += make_series(np.ma.filled(mag, np.nan), x, y, order=0)

        X = np.vstack(X).T

        if get_err:
            # It follows the implementation from https://github.com/statsmodels/statsmodels/blob/081fc6e85868308aa7489ae1b23f6e72f5662799/statsmodels/base/model.py#L1383
            # FIXME: crashes on large numbers of stars?..
            if len(x) < 5000:
                err = np.sqrt(np.dot(X, np.dot(C.cov_params()[0:X.shape[1], 0:X.shape[1]], np.transpose(X))).diagonal())
            else:
                err = np.zeros_like(x)
            if add_intrinsic_rms:
                err = np.hypot(err, intrinsic_rms)
            return err
        else:
            return np.sum(X * C.params[0 : X.shape[1]], axis=1)

    if cat_color is not None and (use_color or force_color_term is not None):
        if use_color:
            color_term = list(C.params[pos_color:][:int(use_color)])
            if len(color_term) == 1:
                color_term = color_term[0]

            log('Color term is', format_color_term(color_term))
        elif force_color_term is not None:
            color_term = force_color_term
            log('Color term (fixed) is', format_color_term(color_term))
    else:
        color_term = None

    return {
        'oidx': oidx,
        'cidx': cidx,
        'dist': dist,
        'omag': omag,
        'omag_err': omag_err,
        'cmag': cmag,
        'cmag_err': cmag_err,
        'color': ccolor,
        'color_term': color_term,
        'zero': zero,
        'zero_err': zero_err,
        'zero_model': zero_model,
        'zero_model_err': zero_model_err,
        'zero_fn': zero_fn,
        'params': C.params,
        'error_scale': np.sqrt(C.scale),
        'intrinsic_rms': intrinsic_rms,
        'obj_zero': zero_fn(obj_x, obj_y, mag=obj_mag),
        'ox': ox,
        'oy': oy,
        'oflags': oflags,
        'idx': idx,
        'idx0': idx0,
    }


def make_sn_model(mag, sn):
    """
    Build a model for signal to noise (S/N) ratio versus magnitude.
    Assumes the noise comes from constant background noise plus Poissonian noise with constant gain.

    :param mag: Array of calibrated magnitudes
    :param sn: Array of S/N values corresponding to them
    :returns: The function that accepts the array of magnitudes and returns the S/N model values for them
    """
    idx = np.isfinite(mag) & np.isfinite(sn)
    mag = mag[idx]
    sn = sn[idx]

    def sn_fn(p, mag):
        return 1 / np.sqrt(p[0] * 10 ** (0.8 * mag) + p[1] * 10 ** (0.4 * mag))

    def lstsq_fn(p, x, y):
        # Minimize residuals in logarithms, for better stability
        return np.log10(y) - np.log10(sn_fn(p, x))

    aidx = np.argsort(sn)

    # Initial params from two limiting cases, one on average and one on brightest point
    x = [
        np.median(10 ** (-0.8 * mag) / sn ** 2),
        10 ** (-0.4 * mag[aidx][-1]) / sn[aidx][-1] ** 2,
    ]

    res = least_squares(lstsq_fn, x, args=(mag, sn), method='lm')

    return lambda mag: sn_fn(res.x, mag)


def get_detection_limit_sn(mag, mag_sn, sn=5, get_model=False, verbose=True):
    """
    Estimate the detection limit using S/N vs magnitude method.

    :param mag: Array of calibrated magnitudes
    :param mag_sn: Array of S/N values corresponding to these magnitudes
    :param sn: S/N level for the detection limit
    :param get_model: If True, also returns the S/N model function
    :param verbose: Whether to show verbose messages during the run of the function or not. May be either boolean, or a `print`-like function
    :returns: The magnitude corresponding to the detection limit on a given S/N level. If :code:`get_model=True`, also returns the function for S/N vs magnitude model
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (
        (verbose if callable(verbose) else print)
        if verbose
        else lambda *args, **kwargs: None
    )

    mag0 = None

    sn_model = make_sn_model(mag, mag_sn)
    res = root_scalar(
        lambda x: np.log10(sn_model(x)) - np.log10(sn),
        x0=np.nanmax(mag),
        x1=np.nanmax(mag) + 1,
    )
    if res.converged:
        mag0 = res.root
    else:
        log('Cannot determine the root of S/N model function')

    if get_model:
        return mag0, sn_model
    else:
        return mag0
