"""
Photometric modeling routines.

This module contains functions for photometric calibration and modeling,
including catalog matching with spatial zero-point models, S/N modeling,
and detection limit estimation.
"""

import warnings

import numpy as np
import statsmodels.api as sm
from scipy import linalg
from scipy.optimize import minimize_scalar, least_squares, root_scalar
from statsmodels.regression import _tools as reg_tools
from statsmodels.robust import robust_linear_model as rlm_model
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
            res.append(mul * x ** (i - j) * y**j)
    if sum:
        return np.sum(res, axis=0)
    else:
        return res


def get_intrinsic_scatter(y, yerr, min=0, max=None):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    good = np.isfinite(y) & np.isfinite(yerr) & (yerr >= 0)
    if np.sum(good) < 2:
        return float(np.maximum(min, 0.0))

    y = y[good]
    yerr = yerr[good]

    c_min = float(np.maximum(min, 0.0))
    if max is None or not np.isfinite(max):
        c_max = float(np.max([np.nanstd(y), np.nanmedian(yerr), c_min + 1e-6]))
    else:
        c_max = float(np.maximum(max, c_min))

    if c_max <= c_min + 1e-12:
        return c_min

    eps = np.finfo(float).eps

    def nll(c):
        sigma2 = np.maximum(yerr**2 + c**2, eps)
        weight = 1.0 / sigma2
        model = np.sum(weight * y) / np.sum(weight)
        return 0.5 * np.sum((y - model) ** 2 * weight + np.log(sigma2))

    result = minimize_scalar(
        nll,
        bounds=(c_min, c_max),
        method='bounded',
    )

    if not result.success or not np.isfinite(result.x):
        return c_min

    return float(np.clip(result.x, c_min, c_max))


def _normalize_spatial_coords(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return x, y, 1.0

    scale = max(float(np.nanmax(np.abs(x[finite]))), float(np.nanmax(np.abs(y[finite]))), 1.0)
    return x / scale, y / scale, scale


def _prepare_parameter_covariance(cov):
    cov = np.asarray(cov, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Parameter covariance must be a square matrix")

    if cov.size == 0:
        return cov

    cov = np.where(np.isfinite(cov), cov, 0.0)
    cov = 0.5 * (cov + cov.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.diag(np.clip(np.diag(cov), 0.0, None))

    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals) @ eigvecs.T


def _evaluate_parameter_covariance(X, cov):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[None, :]

    if X.shape[0] == 0:
        return np.array([], dtype=float)

    err2 = np.full(X.shape[0], np.nan, dtype=float)
    valid = np.all(np.isfinite(X), axis=1)

    if not np.any(valid):
        return np.sqrt(err2)

    quad = np.einsum('ij,jk,ik->i', X[valid], cov, X[valid], optimize=True)
    quad = np.where(np.isfinite(quad), np.maximum(quad, 0.0), np.nan)
    err2[valid] = quad

    return np.sqrt(err2)


def _stable_pinv_exog(exog):
    exog = np.asarray(exog, dtype=float)

    if exog.ndim != 2:
        raise ValueError("Design matrix must be two-dimensional")

    nobs, nparams = exog.shape
    if nparams == 0:
        return np.empty((0, nobs), dtype=float), np.empty((0, 0), dtype=float)

    gram = exog.T @ exog
    rank = np.linalg.matrix_rank(exog)

    if rank == nparams:
        q, r = np.linalg.qr(exog, mode='reduced')
        pinv = np.linalg.solve(r, q.T)
        norm_cov = np.linalg.solve(gram, np.eye(nparams, dtype=float))
    else:
        norm_cov = linalg.pinvh(gram)
        pinv = norm_cov @ exog.T

    norm_cov = 0.5 * (norm_cov + norm_cov.T)
    return pinv, norm_cov


class _StableRLM(sm.RLM):
    def _initialize(self):
        self.pinv_wexog, self.normalized_cov_params = _stable_pinv_exog(self.exog)
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog) - 1)
        self.nobs = float(self.endog.shape[0])

    def fit(
        self,
        maxiter=50,
        tol=1e-8,
        scale_est='mad',
        init=None,
        cov='H1',
        update_scale=True,
        conv='dev',
        start_params=None,
    ):
        if cov.upper() not in ["H1", "H2", "H3"]:
            raise ValueError("Covariance matrix %s not understood" % cov)
        else:
            self.cov = cov.upper()

        conv = conv.lower()
        if conv not in ["weights", "coefs", "dev", "sresid"]:
            raise ValueError("Convergence argument %s not understood" % conv)

        self.scale_est = scale_est

        if start_params is None:
            wls_results = reg_tools._MinimalWLS(
                self.endog,
                self.exog,
                weights=np.ones_like(self.endog),
                check_weights=False,
            ).fit(method='qr')
        else:
            start_params = np.asarray(start_params, dtype=np.double).squeeze()
            if start_params.shape[0] != self.exog.shape[1] or start_params.ndim != 1:
                raise ValueError(
                    'start_params must by a 1-d array with {} values'.format(
                        self.exog.shape[1]
                    )
                )
            fake_wls = reg_tools._MinimalWLS(
                self.endog,
                self.exog,
                weights=np.ones_like(self.endog),
                check_weights=False,
            )
            wls_results = fake_wls.results(start_params)

        if not init:
            self.scale = self._estimate_scale(wls_results.resid)

        history = dict(params=[np.inf], scale=[])
        if conv == 'coefs':
            criterion = history['params']
        elif conv == 'dev':
            history.update(dict(deviance=[np.inf]))
            criterion = history['deviance']
        elif conv == 'sresid':
            history.update(dict(sresid=[np.inf]))
            criterion = history['sresid']
        else:
            history.update(dict(weights=[np.inf]))
            criterion = history['weights']

        history = self._update_history(wls_results, history, conv)
        iteration = 1
        converged = 0
        while not converged:
            if self.scale == 0.0:
                warnings.warn(
                    'Estimated scale is 0.0 indicating that the most last iteration produced '
                    'a perfect fit of the weighted data.',
                    ConvergenceWarning,
                )
                break

            self.weights = self.M.weights(wls_results.resid / self.scale)
            wls_results = reg_tools._MinimalWLS(
                self.endog,
                self.exog,
                weights=self.weights,
                check_weights=True,
            ).fit(method='qr')

            if update_scale is True:
                self.scale = self._estimate_scale(wls_results.resid)

            history = self._update_history(wls_results, history, conv)
            iteration += 1
            converged = rlm_model._check_convergence(criterion, iteration, tol, maxiter)

        results = rlm_model.RLMResults(
            self,
            wls_results.params,
            self.normalized_cov_params,
            self.scale,
        )

        history['iteration'] = iteration
        results.fit_history = history
        results.fit_options = dict(
            cov=cov.upper(),
            scale_est=scale_est,
            norm=self.M.__class__.__name__,
            conv=conv,
        )
        return rlm_model.RLMResultsWrapper(results)


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
    for i, val in enumerate(color_term):
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

    Builds a photometric model for objects detected on the image that includes
    catalogue magnitude, positionally-dependent zero point, linear color term,
    optional additive flux term, and also takes into account possible intrinsic
    magnitude scatter on top of measurement errors.

    Parameters
    ----------
    obj_ra : array-like
        Right Ascension values for the objects.
    obj_dec : array-like
        Declination values for the objects.
    obj_mag : array-like
        Instrumental magnitude values for the objects.
    obj_magerr : array-like
        Instrumental magnitude errors for the objects.
    obj_flags : array-like
        Flags for the objects.
    cat_ra : array-like
        Catalogue Right Ascension values.
    cat_dec : array-like
        Catalogue Declination values.
    cat_mag : array-like
        Catalogue magnitudes.
    cat_magerr : array-like, optional
        Catalogue magnitude errors.
    cat_color : array-like, optional
        Catalogue color values.
    sr : float
        Matching radius in degrees.
    obj_x : array-like, optional
        X coordinates of objects on the image.
    obj_y : array-like, optional
        Y coordinates of objects on the image.
    spatial_order : int
        Order of zero point spatial polynomial (0 for constant).
    bg_order : int or None
        Order of additive flux term spatial polynomial. None to disable this
        term in the model.
    nonlin : bool
        Whether to fit for simple non-linearity.
    threshold : float
        Rejection threshold (relative to magnitude errors) for
        object-catalogue pair to be rejected from the fit.
    niter : int
        Number of iterations for the fitting.
    accept_flags : int
        Bitmask for acceptable object flags. Objects having any other bits
        set will be excluded from the model.
    cat_saturation : float or None
        Saturation level for the catalogue. Stars brighter than this
        magnitude will be excluded from the fit.
    max_intrinsic_rms : float
        Maximal intrinsic RMS to use during the fitting. If set to 0, no
        intrinsic scatter is included in the noise model.
    sn : float or None
        Minimal acceptable signal to noise ratio (1/obj_magerr) for the
        objects to be included in the fit.
    verbose : bool or callable
        Whether to show verbose messages during the run. May be either
        boolean, or a ``print``-like function.
    robust : bool
        Whether to use robust least squares fitting instead of weighted
        least squares.
    scale_noise : bool
        Whether to re-scale the noise model (object and catalogue magnitude
        errors) to match actual scatter of the data points. Intrinsic
        scatter term is not scaled this way.
    use_color : bool or int
        Whether to use catalogue color for deriving the color term. If
        integer, it determines the color term order.
    force_color_term : float or None
        Do not fit for the color term, but use this fixed value instead.

    Returns
    -------
    dict or None
        Dictionary with photometric results, or None if the fit failed.
        Contains the following keys:

        - ``oidx``, ``cidx``, ``dist`` -- indices of positionally matched
          objects and catalogue stars, and their pairwise distances in degrees.
        - ``omag``, ``omag_err``, ``cmag``, ``cmag_err`` -- instrumental
          magnitudes of matched objects, corresponding catalogue magnitudes,
          and their errors. Array lengths equal the number of positional
          matches.
        - ``color`` -- catalogue colors corresponding to the matches, or
          zeros if no color term fitting is requested.
        - ``ox``, ``oy``, ``oflags`` -- coordinates of matched objects on the
          image, and their flags.
        - ``zero``, ``zero_err`` -- empirical zero points (catalogue minus
          instrumental magnitudes) for every matched object, and errors
          derived as a hypotenuse of their corresponding errors.
        - ``zero_model``, ``zero_model_err`` -- modeled "full" zero points
          (including color terms) for matched objects, and their errors from
          the fit.
        - ``color_term`` -- fitted color term. Instrumental photometric system
          is defined as ``obj_mag = cat_mag - color * color_term``.
        - ``zero_fn`` -- function to compute the zero point (without color
          term) at a given position and for a given instrumental magnitude,
          and optionally its error.
        - ``obj_zero`` -- zero points for all input objects (not necessarily
          matched to the catalogue) computed via ``zero_fn``, without color
          term.
        - ``params`` -- internal parameters of the fitting polynomial.
        - ``intrinsic_rms``, ``error_scale`` -- fitted values of intrinsic
          scatter and noise scaling.
        - ``idx`` -- boolean index of matched objects/catalogue stars used in
          the final fit (not rejected during iterative thresholding, and
          passing initial quality cuts).
        - ``idx0`` -- same as ``idx`` but with only initial quality cuts
          applied.

    Notes
    -----
    The returned zero point function ``zero_fn`` has the signature::

        zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False)

    where ``xx`` and ``yy`` are coordinates on the image, ``mag`` is the
    object instrumental magnitude (needed to compute the additive flux term).
    If ``get_err=True``, the function returns estimated zero point error
    instead of zero point, and ``add_intrinsic_rms`` controls whether this
    error estimation should also include the intrinsic scatter term.

    The zero point returned by this function does not include the contribution
    of the color term. To derive the final calibrated magnitude for an object,
    manually add the color contribution::

        mag_calibrated = mag_instrumental + color * color_term

    where ``color`` is the true object color, and ``color_term`` is reported
    in the photometric results.

    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

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

    omag = np.asarray(np.ma.filled(obj_mag[oidx], fill_value=np.nan), dtype=float)
    omag_err = np.asarray(np.ma.filled(obj_magerr[oidx], fill_value=np.nan), dtype=float)
    oflags = np.asarray(obj_flags[oidx]) if obj_flags is not None else np.zeros_like(omag, dtype=bool)
    cmag = np.asarray(np.ma.filled(cat_mag[cidx], fill_value=np.nan), dtype=float)
    cmag_err = (
        np.asarray(np.ma.filled(cat_magerr[cidx], fill_value=np.nan), dtype=float)
        if cat_magerr is not None
        else np.zeros_like(cmag)
    )

    if obj_x is not None and obj_y is not None:
        ox = np.asarray(np.ma.filled(obj_x[oidx], fill_value=np.nan), dtype=float)
        oy = np.asarray(np.ma.filled(obj_y[oidx], fill_value=np.nan), dtype=float)
        x0 = float(np.nanmean(ox)) if np.any(np.isfinite(ox)) else 0.0
        y0 = float(np.nanmean(oy)) if np.any(np.isfinite(oy)) else 0.0
        x = ox - x0
        y = oy - y0
    else:
        x0, y0 = 0, 0
        ox, oy = np.zeros_like(omag), np.zeros_like(omag)
        x, y = np.zeros_like(omag), np.zeros_like(omag)

    x, y, xy_scale = _normalize_spatial_coords(x, y)
    if spatial_order > 0 and xy_scale > 1.0:
        log('Normalizing spatial coordinates by %.1f pixels' % xy_scale)

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

    if force_color_term is not None:
        fit_color_term = False
    elif isinstance(use_color, int) and use_color > 1:
        # use_color=N sets the color term polynomial order
        fit_color_term = use_color
    elif use_color:
        fit_color_term = 1
    else:
        fit_color_term = False

    if cat_color is not None:
        ccolor = np.asarray(np.ma.filled(cat_color[cidx], fill_value=np.nan), dtype=float)
        if fit_color_term:
            pos_color = len(X)
            for _ in range(int(fit_color_term)):
                X += make_series(ccolor ** (_ + 1), x, y, order=0)

            log('Using color term of order', int(fit_color_term))
        elif force_color_term is not None:
            for i, val in enumerate(np.atleast_1d(force_color_term)):
                cmag -= val * ccolor ** (i + 1)
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
    if cat_color is not None and (fit_color_term or force_color_term is not None):
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
            C = _StableRLM(zero[idx] / total_err[idx], (X[idx].T / total_err[idx]).T).fit()
        else:
            C = sm.WLS(zero[idx], X[idx], weights=1 / total_err[idx] ** 2).fit()

        zero_model = np.sum(X * C.params, axis=1)

        scale_err = 1 if not scale_noise else np.sqrt(C.scale)  # rms

        intrinsic_rms = (
            get_intrinsic_scatter((zero - zero_model)[idx], (zero_err * scale_err)[idx], max=max_intrinsic_rms)
            if max_intrinsic_rms > 0
            else 0
        )

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

    cov_p = _prepare_parameter_covariance(C.cov_params())
    zero_model_err = _evaluate_parameter_covariance(X, cov_p)

    # Export the model
    def zero_fn(xx, yy, mag=None, get_err=False, add_intrinsic_rms=False):
        if mag is not None:
            mag = np.atleast_1d(np.ma.filled(mag, np.nan))

        if xx is not None and yy is not None:
            x, y = xx - x0, yy - y0
        else:
            n = len(mag) if mag is not None else 1
            x, y = np.zeros(n), np.zeros(n)

        # Ensure we do not have MaskedColumns
        x, y = [
            np.asarray(np.atleast_1d(np.ma.filled(_, fill_value=np.nan)), dtype=float)
            for _ in (x, y)
        ]
        x /= xy_scale
        y /= xy_scale

        X = make_series(1.0, x, y, order=spatial_order)

        if bg_order is not None and mag is not None:
            X += make_series(-2.5 / np.log(10) / 10 ** (-0.4 * mag), x, y, order=bg_order)

        if nonlin and mag is not None:
            X += make_series(np.ma.filled(mag, np.nan), x, y, order=0)

        X = np.vstack(X).T

        if get_err:
            err = _evaluate_parameter_covariance(X, cov_p[0 : X.shape[1], 0 : X.shape[1]])
            if add_intrinsic_rms:
                err = np.hypot(err, intrinsic_rms)
            return err
        else:
            result = np.sum(X * C.params[0 : X.shape[1]], axis=1)
            return result

    if cat_color is not None and (fit_color_term or force_color_term is not None):
        if fit_color_term:
            color_term = list(C.params[pos_color:][: int(fit_color_term)])
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
    """Build a model for signal to noise (S/N) ratio versus magnitude.

    Assumes the noise comes from constant background noise plus Poissonian
    noise with constant gain.

    Parameters
    ----------
    mag : array-like
        Calibrated magnitudes.
    sn : array-like
        S/N values corresponding to the magnitudes.

    Returns
    -------
    callable
        Function that accepts an array of magnitudes and returns the S/N
        model values for them.

    Raises
    ------
    ValueError
        If no finite positive S/N values are available to build the model.
    """
    idx = np.isfinite(mag) & np.isfinite(sn) & (sn > 0)
    mag = mag[idx]
    sn = sn[idx]

    if mag.size == 0:
        raise ValueError('No finite positive S/N values to build the model')

    def sn_fn(p, mag):
        return 1 / np.sqrt(p[0] * 10 ** (0.8 * mag) + p[1] * 10 ** (0.4 * mag))

    def lstsq_fn(p, x, y):
        # Minimize residuals in logarithms, for better stability
        return np.log10(y) - np.log10(sn_fn(p, x))

    aidx = np.argsort(sn)

    # Initial params from two limiting cases, one on average and one on brightest point
    x = [
        np.median(10 ** (-0.8 * mag) / sn**2),
        10 ** (-0.4 * mag[aidx][-1]) / sn[aidx][-1] ** 2,
    ]

    res = least_squares(lstsq_fn, x, args=(mag, sn), method='lm')

    return lambda mag: sn_fn(res.x, mag)


def get_detection_limit_sn(mag, mag_sn, sn=5, get_model=False, verbose=True):
    """Estimate the detection limit using S/N vs magnitude method.

    Parameters
    ----------
    mag : array-like
        Calibrated magnitudes.
    mag_sn : array-like
        S/N values corresponding to these magnitudes.
    sn : float
        S/N level for the detection limit.
    get_model : bool
        If True, also return the S/N model function.
    verbose : bool or callable
        Whether to show verbose messages during the run. May be either
        boolean, or a ``print``-like function.

    Returns
    -------
    float or None
        Magnitude corresponding to the detection limit at the given S/N
        level. None if the model cannot be built or the root is not found.
    callable or None
        S/N model function (only returned when ``get_model=True``). None
        if the model cannot be built.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args, **kwargs: None

    mag = np.asarray(mag)
    mag_sn = np.asarray(mag_sn)

    idx = np.isfinite(mag) & np.isfinite(mag_sn) & (mag_sn > 0)
    mag = mag[idx]
    mag_sn = mag_sn[idx]

    if mag.size == 0:
        log('No valid magnitudes/SN values for detection limit estimation')
        return (None, None) if get_model else None

    mag0 = None

    try:
        sn_model = make_sn_model(mag, mag_sn)
    except ValueError:
        log('Cannot build S/N model function')
        return (None, None) if get_model else None
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
