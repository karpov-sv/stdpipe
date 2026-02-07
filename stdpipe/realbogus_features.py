"""
Feature-Based Real-Bogus Classification for Astronomical Object Detection

This module provides a feature-based classifier to distinguish real astronomical sources
(stars, galaxies) from artifacts (cosmic rays, hot pixels, satellite trails) without
requiring deep learning dependencies.

Key Features:
- Catalog-only mode (like artefacts.py but more features)
- Cutout-based morphological feature extraction
- Hybrid mode combining both
- Multiple classifiers: IsolationForest, RandomForest, rule-based scoring
- Trend removal for spatially-varying features (generalized from artefacts.py)
- No TensorFlow/Keras dependency (uses sklearn only)

Usage:
    from stdpipe import photometry, realbogus_features as rbf

    # Detect objects
    obj = photometry.get_objects_sep(image, thresh=3.0)

    # Quick catalog-only classification (like artefacts.py)
    good = rbf.classify(obj, method='catalog', classifier='isolation')

    # Cutout-based with scoring (no training needed)
    obj = rbf.classify(obj, image, method='cutout', classifier='scoring',
                       add_score=True)

    # Hybrid with RandomForest
    obj = rbf.classify(obj, image, method='hybrid', classifier='randomforest',
                       model=trained_model)

Author: STDPipe Contributors
"""

import numpy as np
from scipy import ndimage
from astropy.table import Column

from . import smoothing


_VALID_FEATURE_SETS = ("default", "minimal", "extended")
_FEATURE_SET_LEVELS = {
    "minimal": {"minimal"},
    "default": {"minimal", "default"},
    "extended": {"minimal", "default", "extended"},
}


def _normalize_feature_array(arr):
    """Cast feature values to float and replace non-finite values with NaN."""
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        arr = np.where(bad, np.nan, arr)
    return arr


def _safe_ratio(numerator, denominator):
    """Compute ratio with NaN propagation for invalid values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _get_column_array(obj, candidates):
    """Return first available catalog column from candidate list as float array."""
    for colname in candidates:
        if colname in obj.colnames:
            return np.asarray(obj[colname], dtype=float)
    return None


def _get_last_available_pair(obj, pairs):
    """Return the last available (numerator, denominator) pair."""
    selected = None
    for num_col, den_col in pairs:
        if num_col in obj.colnames and den_col in obj.colnames:
            selected = (num_col, den_col)
    return selected


def _compute_catalog_flux_radius(obj, _features):
    return _get_column_array(obj, ["FLUX_RADIUS"])


def _compute_catalog_fwhm(obj, _features):
    return _get_column_array(obj, ["fwhm", "FWHM_IMAGE"])


def _compute_catalog_peakiness(obj, _features):
    pair = _get_last_available_pair(obj, [("FLUX_MAX", "FLUX_AUTO"), ("peak", "flux")])
    if pair is None:
        return None
    return _safe_ratio(
        np.asarray(obj[pair[0]], dtype=float),
        np.asarray(obj[pair[1]], dtype=float),
    )


def _get_catalog_shape_axes(obj):
    pair = _get_last_available_pair(obj, [("a", "b"), ("A_IMAGE", "B_IMAGE")])
    if pair is None:
        return None, None
    return np.asarray(obj[pair[0]], dtype=float), np.asarray(obj[pair[1]], dtype=float)


def _compute_catalog_ellipticity(obj, _features):
    a, b = _get_catalog_shape_axes(obj)
    if a is None:
        return None
    return _safe_ratio(a - b, a + b)


def _compute_catalog_elongation(obj, _features):
    a, b = _get_catalog_shape_axes(obj)
    if a is None:
        return None
    return _safe_ratio(a, b)


def _compute_catalog_fwhm_ratio(_obj, features):
    fwhm = features.get("fwhm")
    if fwhm is None:
        return None
    median_fwhm = np.nanmedian(fwhm)
    if not np.isfinite(median_fwhm) or median_fwhm <= 0:
        return None
    return fwhm / median_fwhm


def _compute_catalog_radius_fwhm_ratio(_obj, features):
    flux_radius = features.get("flux_radius")
    fwhm = features.get("fwhm")
    if flux_radius is None or fwhm is None:
        return None
    return _safe_ratio(flux_radius, fwhm)


def _compute_catalog_snr(obj, _features):
    if "flux" not in obj.colnames or "fluxerr" not in obj.colnames:
        return None
    return _safe_ratio(
        np.asarray(obj["flux"], dtype=float),
        np.asarray(obj["fluxerr"], dtype=float),
    )


_CATALOG_FEATURE_SPECS = [
    ("flux_radius", "minimal", _compute_catalog_flux_radius),
    ("fwhm", "minimal", _compute_catalog_fwhm),
    ("peakiness", "minimal", _compute_catalog_peakiness),
    ("ellipticity", "minimal", _compute_catalog_ellipticity),
    ("elongation", "minimal", _compute_catalog_elongation),
    ("fwhm_ratio", "minimal", _compute_catalog_fwhm_ratio),
    ("radius_fwhm_ratio", "default", _compute_catalog_radius_fwhm_ratio),
    ("snr", "default", _compute_catalog_snr),
]


def _copy_feature_dict(features):
    """Return deep-copied feature arrays keyed by feature name."""
    return {name: np.array(arr, copy=True) for name, arr in features.items()}


def _get_finite_rows(X, context):
    """Return mask of rows with finite feature values, raising if empty."""
    valid = np.all(np.isfinite(X), axis=1)
    if not np.any(valid):
        raise ValueError(f"No finite feature rows available for {context}")
    return valid


def _filter_finite_rows(X, y=None, context="fitting"):
    """Filter feature matrix (and optional labels) to finite rows."""
    valid = _get_finite_rows(X, context=context)
    if y is None:
        return X[valid], valid

    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError("labels length does not match number of feature rows")
    return X[valid], y[valid], valid


# =============================================================================
# Feature Extraction: Catalog-based
# =============================================================================


def extract_catalog_features(obj, feature_set="default"):
    """
    Extract features from object catalog (no image needed).

    These features are derived from detection catalog columns, similar to
    what artefacts.py uses but extended.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with detection columns.
    feature_set : str, optional
        Feature set to extract: 'default', 'minimal', 'extended'.

    Returns
    -------
    features : dict
        Dictionary mapping feature names to arrays.
    feature_names : list
        List of feature names in order.
    """
    if feature_set not in _VALID_FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set: {feature_set}. Expected one of {list(_VALID_FEATURE_SETS)}."
        )

    features = {}
    enabled_levels = _FEATURE_SET_LEVELS[feature_set]
    for name, level, compute_fn in _CATALOG_FEATURE_SPECS:
        if level not in enabled_levels:
            continue

        arr = compute_fn(obj, features)
        if arr is None:
            continue
        features[name] = _normalize_feature_array(arr)

    feature_names = list(features.keys())
    return features, feature_names


# =============================================================================
# Feature Extraction: Cutout-based
# =============================================================================


def _extract_cutout(image, x, y, radius, bg=None, err=None, mask=None):
    """Extract a single cutout centered on (x, y)."""
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))

    x_min = xi - radius
    x_max = xi + radius + 1
    y_min = yi - radius
    y_max = yi + radius + 1

    if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
        return None, None, None, None

    cutout = image[y_min:y_max, x_min:x_max].copy().astype(float)

    if bg is not None:
        if np.isscalar(bg):
            cutout_bg = bg
        else:
            cutout_bg = bg[y_min:y_max, x_min:x_max].copy()
        cutout = cutout - cutout_bg

    cutout_err = None
    if err is not None:
        if np.isscalar(err):
            cutout_err = np.full_like(cutout, float(err))
        else:
            cutout_err = err[y_min:y_max, x_min:x_max].copy().astype(float)

    cutout_mask = None
    if mask is not None:
        cutout_mask = mask[y_min:y_max, x_min:x_max].copy()

    # Sub-pixel offset for accurate centering
    dx = x - xi
    dy = y - yi

    return cutout, cutout_mask, cutout_err, (dx, dy)


def _compute_sharpness(cutout):
    """
    Compute sharpness: central pixel vs neighbors.

    High sharpness indicates cosmic ray or hot pixel.
    """
    if cutout is None or cutout.size < 9:
        return np.nan

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    center = cutout[cy, cx]

    # 8-connected neighbors
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < cutout.shape[0] and 0 <= nx < cutout.shape[1]:
                neighbors.append(cutout[ny, nx])

    if not neighbors:
        return np.nan

    neighbor_mean = np.mean(neighbors)
    if neighbor_mean <= 0:
        return np.nan

    return center / neighbor_mean


def _compute_concentration(cutout, r_inner=2, r_outer=5):
    """
    Compute concentration index: flux ratio in inner/outer apertures.

    Real sources have consistent concentration; artifacts vary.
    """
    if cutout is None:
        return np.nan

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    y, x = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inner_mask = r <= r_inner
    outer_mask = r <= r_outer

    flux_inner = np.sum(cutout[inner_mask])
    flux_outer = np.sum(cutout[outer_mask])

    if flux_outer <= 0:
        return np.nan

    return flux_inner / flux_outer


def _compute_symmetry(cutout):
    """
    Compute 180-degree rotational symmetry.

    Real sources (stars) are symmetric; many artifacts are not.
    Returns residual (lower = more symmetric).
    """
    if cutout is None:
        return np.nan

    rotated = np.rot90(cutout, 2)  # 180 degree rotation

    # Normalize by flux to make scale-independent
    flux = np.sum(np.abs(cutout))
    if flux <= 0:
        return np.nan

    residual = np.sum(np.abs(cutout - rotated)) / flux
    return residual


def _compute_roundness(cutout):
    """
    Compute roundness from image moments.

    Returns value 0-1 where 1 = perfectly round.
    """
    if cutout is None:
        return np.nan

    # Use positive part only for moments
    data = np.maximum(cutout, 0)
    total = np.sum(data)
    if total <= 0:
        return np.nan

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    y, x = np.ogrid[: cutout.shape[0], : cutout.shape[1]]

    # Second moments
    m20 = np.sum(data * (x - cx) ** 2) / total
    m02 = np.sum(data * (y - cy) ** 2) / total
    m11 = np.sum(data * (x - cx) * (y - cy)) / total

    # Ellipse parameters from moments
    diff = m20 - m02
    discriminant = np.sqrt(diff**2 + 4 * m11**2)

    if discriminant == 0:
        return 1.0  # Perfectly round

    # Semi-major and semi-minor axes
    a2 = (m20 + m02 + discriminant) / 2
    b2 = (m20 + m02 - discriminant) / 2

    if a2 <= 0:
        return np.nan

    # Roundness = b/a
    return np.sqrt(max(0, b2) / a2)


def _compute_psf_match(cutout, fwhm):
    """
    Compute how well the cutout matches a Gaussian PSF.

    Returns chi-squared-like metric (lower = better match).
    """
    if cutout is None or fwhm <= 0:
        return np.nan

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    y, x = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    r2 = (x - cx) ** 2 + (y - cy) ** 2

    # Gaussian PSF model
    sigma = fwhm / 2.355
    psf_model = np.exp(-r2 / (2 * sigma**2))

    # Normalize both to unit sum
    cutout_norm = cutout / np.sum(np.abs(cutout)) if np.sum(np.abs(cutout)) > 0 else cutout
    psf_norm = psf_model / np.sum(psf_model)

    # Scale PSF to match cutout amplitude
    scale = np.sum(cutout_norm * psf_norm) / np.sum(psf_norm**2)
    residual = cutout_norm - scale * psf_norm

    # Chi-squared-like metric
    return np.sum(residual**2)


def _compute_peak_offset(cutout):
    """
    Compute distance from center to peak pixel.

    Real centered sources have small offset; artifacts may have large offset.
    """
    if cutout is None:
        return np.nan

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    peak_idx = np.unravel_index(np.argmax(cutout), cutout.shape)
    peak_y, peak_x = peak_idx

    return np.sqrt((peak_x - cx) ** 2 + (peak_y - cy) ** 2)


def _compute_edge_gradient(cutout):
    """
    Compute maximum radial gradient at edge.

    Cosmic rays have sharp edges; real sources have smooth profiles.
    """
    if cutout is None:
        return np.nan

    # Sobel gradient magnitude
    gx = ndimage.sobel(cutout, axis=1)
    gy = ndimage.sobel(cutout, axis=0)
    gradient = np.sqrt(gx**2 + gy**2)

    # Normalize by peak flux
    peak = np.max(np.abs(cutout))
    if peak <= 0:
        return np.nan

    return np.max(gradient) / peak


def _compute_background_consistency(cutout, radius=None):
    """
    Compute consistency of background around the source.

    Real sources have uniform local background; artifacts near edges don't.
    """
    if cutout is None:
        return np.nan

    if radius is None:
        radius = cutout.shape[0] // 2 - 2

    cy, cx = cutout.shape[0] // 2, cutout.shape[1] // 2
    y, x = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Background annulus
    bg_mask = (r > radius) & (r <= radius + 2)
    if np.sum(bg_mask) < 4:
        return np.nan

    bg_pixels = cutout[bg_mask]
    bg_std = np.std(bg_pixels)
    bg_median = np.median(np.abs(bg_pixels))

    if bg_median <= 0:
        return 0.0

    # Coefficient of variation (lower = more consistent)
    return bg_std / bg_median


def _compute_cutout_snr(cutout, cutout_err):
    """Compute integrated SNR in a cutout using provided error map."""
    if cutout is None or cutout_err is None:
        return np.nan

    signal = np.sum(np.maximum(cutout, 0))
    if signal <= 0:
        return np.nan

    err2 = np.sum(np.maximum(cutout_err, 0) ** 2)
    if err2 <= 0:
        return np.nan

    return signal / np.sqrt(err2)


def _get_cutout_feature_specs(include_snr=False):
    """Build cutout feature extraction specs."""
    specs = [
        ("sharpness", lambda cutout, _cutout_err, _fwhm: _compute_sharpness(cutout)),
        ("concentration", lambda cutout, _cutout_err, _fwhm: _compute_concentration(cutout)),
        ("symmetry", lambda cutout, _cutout_err, _fwhm: _compute_symmetry(cutout)),
        ("roundness", lambda cutout, _cutout_err, _fwhm: _compute_roundness(cutout)),
        ("psf_match", lambda cutout, _cutout_err, fwhm: _compute_psf_match(cutout, fwhm)),
        ("peak_offset", lambda cutout, _cutout_err, _fwhm: _compute_peak_offset(cutout)),
        ("edge_gradient", lambda cutout, _cutout_err, _fwhm: _compute_edge_gradient(cutout)),
        (
            "bg_consistency",
            lambda cutout, _cutout_err, _fwhm: _compute_background_consistency(cutout),
        ),
    ]
    if include_snr:
        specs.append(
            (
                "cutout_snr",
                lambda cutout, cutout_err, _fwhm: _compute_cutout_snr(cutout, cutout_err),
            )
        )
    return specs


def extract_cutout_features(
    obj, image, bg=None, err=None, mask=None, fwhm=None, radius=10, verbose=False
):
    """
    Extract morphological features from image cutouts.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x' and 'y' columns.
    image : ndarray
        Science image.
    bg : ndarray or float, optional
        Background map or scalar value.
    err : ndarray or float, optional
        Error/noise map. If provided, `cutout_snr` is computed.
    mask : ndarray, optional
        Boolean mask (True = masked pixels).
    fwhm : float, optional
        Image FWHM for PSF matching. If None, estimated from catalog.
    radius : int, optional
        Cutout radius in pixels. Default: 10.
    verbose : bool, optional
        Print progress. Default: False.

    Returns
    -------
    features : dict
        Dictionary mapping feature names to arrays.
    feature_names : list
        List of feature names in order.
    """
    log = print if verbose else lambda *args, **kwargs: None
    n = len(obj)

    # Estimate FWHM if not provided
    if fwhm is None:
        if "fwhm" in obj.colnames:
            fwhm = np.nanmedian(obj["fwhm"])
        elif "a" in obj.colnames:
            fwhm = np.nanmedian(obj["a"]) * 2.35
        else:
            fwhm = 3.0
        log(f"Estimated FWHM: {fwhm:.2f} pixels")

    feature_specs = _get_cutout_feature_specs(include_snr=err is not None)
    features = {name: np.full(n, np.nan) for name, _fn in feature_specs}

    # Process each object
    for i, row in enumerate(obj):
        x, y = row["x"], row["y"]

        cutout, cutout_mask, cutout_err, _offset = _extract_cutout(
            image, x, y, radius, bg=bg, err=err, mask=mask
        )

        if cutout is None:
            continue

        # Apply mask if present
        if cutout_mask is not None and np.any(cutout_mask):
            if np.sum(~cutout_mask) < 0.5 * cutout.size:
                continue
            median_val = np.nanmedian(cutout[~cutout_mask])
            cutout[cutout_mask] = median_val
            if cutout_err is not None:
                median_err = np.nanmedian(cutout_err[~cutout_mask])
                cutout_err[cutout_mask] = median_err

        # Compute features from declarative spec list.
        for name, compute_fn in feature_specs:
            features[name][i] = compute_fn(cutout, cutout_err, fwhm)

    feature_names = list(features.keys())
    log(f"Extracted {len(feature_names)} cutout features for {n} objects")

    return features, feature_names


def extract_features(
    obj,
    image=None,
    bg=None,
    err=None,
    mask=None,
    fwhm=None,
    radius=10,
    method="auto",
    feature_set="default",
    verbose=False,
):
    """
    Extract features for real-bogus classification.

    Combines catalog-based and cutout-based features depending on method.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog.
    image : ndarray, optional
        Science image (required for 'cutout' and 'hybrid' methods).
    bg : ndarray or float, optional
        Background map or scalar.
    err : ndarray or float, optional
        Error/noise map.
    mask : ndarray, optional
        Boolean mask.
    fwhm : float, optional
        Image FWHM.
    radius : int, optional
        Cutout radius. Default: 10.
    method : str, optional
        Feature extraction method:
        - 'catalog': Catalog features only (no image needed)
        - 'cutout': Cutout features only
        - 'hybrid': Both catalog and cutout features
        - 'auto': 'hybrid' if image provided, else 'catalog'
    feature_set : str, optional
        Catalog feature set to use: 'minimal', 'default', 'extended'.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    features : dict
        Dictionary mapping feature names to arrays.
    feature_names : list
        List of feature names in order.
    """
    valid_methods = ["catalog", "cutout", "hybrid", "auto"]
    if method not in valid_methods:
        raise ValueError(f"Unknown feature extraction method: {method}")

    if method == "auto":
        method = "hybrid" if image is not None else "catalog"

    features = {}
    feature_names = []

    if method in ["catalog", "hybrid"]:
        cat_features, cat_names = extract_catalog_features(obj, feature_set=feature_set)
        features.update(cat_features)
        feature_names.extend(cat_names)

    if method in ["cutout", "hybrid"]:
        if image is None:
            raise ValueError("Image required for cutout/hybrid feature extraction")
        cut_features, cut_names = extract_cutout_features(
            obj, image, bg=bg, err=err, mask=mask, fwhm=fwhm, radius=radius, verbose=verbose
        )
        features.update(cut_features)
        feature_names.extend(cut_names)

    if not feature_names:
        raise ValueError(
            "No features could be extracted from the provided inputs. "
            "Check catalog columns and extraction method."
        )

    return features, feature_names


# =============================================================================
# Trend Removal
# =============================================================================


class TrendModels(dict):
    """Mapping of feature name -> trend model with trend-column metadata."""

    def __init__(self, *args, trend_cols=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_cols = None if trend_cols is None else tuple(trend_cols)


def _coerce_trend_models(trend_models):
    """Normalize legacy/new trend model containers to TrendModels."""
    if trend_models is None:
        return TrendModels()

    if isinstance(trend_models, TrendModels):
        return trend_models

    if isinstance(trend_models, dict):
        legacy_cols = trend_models.get("__trend_cols__")
        models = {k: v for k, v in trend_models.items() if k != "__trend_cols__"}
        return TrendModels(models, trend_cols=legacy_cols)

    raise TypeError("trend_models must be a dict-like object returned by remove_trends()")


def remove_trends(
    features, obj, trend_cols=None, trend_scales=None, k=20, robust_iters=3, verbose=False
):
    """
    Remove smooth spatial/magnitude trends from features.

    This generalizes the trend removal from artefacts.py to work with
    any set of features.

    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    obj : astropy.table.Table
        Object catalog with position/magnitude columns.
    trend_cols : list of str, optional
        Columns to use for trend modeling. Default: ['x', 'y'].
    trend_scales : list of float, optional
        Scaling factors for each trend column. Default: auto-computed.
    k : int, optional
        Number of neighbors for LOESS. Default: 20.
    robust_iters : int, optional
        Robust iteration count. Default: 3.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    detrended : dict
        Dictionary of detrended feature arrays.
    trend_models : TrendModels
        Mapping of trend models keyed by feature name, with trend column metadata.
    """
    log = print if verbose else lambda *args, **kwargs: None

    requested_trend_cols = ["x", "y"] if trend_cols is None else list(trend_cols)
    requested_trend_scales = None if trend_scales is None else list(trend_scales)
    if requested_trend_scales is not None and len(requested_trend_scales) != len(
        requested_trend_cols
    ):
        raise ValueError("trend_scales length must match trend_cols length")

    # Check that trend columns exist
    available_cols = [c for c in requested_trend_cols if c in obj.colnames]
    if not available_cols:
        log("No trend columns available, skipping detrending")
        return _copy_feature_dict(features), TrendModels(trend_cols=())

    trend_cols = available_cols

    # Auto-compute scales if not provided
    if requested_trend_scales is None:
        trend_scales = []
        for col in trend_cols:
            data = np.array(obj[col])
            scale = np.nanstd(data) * 3  # ~3 sigma scale
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            trend_scales.append(scale)
    else:
        scales_by_col = dict(zip(requested_trend_cols, requested_trend_scales))
        trend_scales = []
        for col in trend_cols:
            scale = scales_by_col[col]
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            trend_scales.append(scale)

    log(f"Removing trends using columns: {trend_cols} with scales: {trend_scales}")

    # Build position array
    pos = np.column_stack([np.array(obj[c]) for c in trend_cols])

    # Find valid (unflagged) objects for fitting
    if "flags" in obj.colnames:
        base_valid_idx = (obj["flags"] & 0x7FF) == 0  # Exclude flagged objects
    else:
        base_valid_idx = np.ones(len(obj), dtype=bool)
    base_valid_idx &= np.all(np.isfinite(pos), axis=1)

    detrended = {}
    trend_models = TrendModels(trend_cols=trend_cols)

    for name, arr in features.items():
        arr = np.array(arr)

        # Skip if all NaN
        if np.all(~np.isfinite(arr)):
            detrended[name] = arr.copy()
            trend_models[name] = None
            continue

        valid_idx = base_valid_idx & np.isfinite(arr)
        if np.sum(valid_idx) < max(3, min(k, len(arr))):
            log(f"  {name}: not enough valid points for trend removal, keeping original")
            detrended[name] = arr.copy()
            trend_models[name] = None
            continue

        # Fit LOESS model
        try:
            model = smoothing.ApproxLoessRegressor(
                k=k, scales=trend_scales, robust_iters=robust_iters
            )
            model.fit(pos[valid_idx], arr[valid_idx])

            # Remove trend
            detrended_arr = arr.copy()
            trend_rms = np.nan
            if np.any(base_valid_idx):
                trend = model.predict(pos[base_valid_idx])
                detrended_arr[base_valid_idx] = arr[base_valid_idx] - trend
                trend_rms = np.nanstd(trend)
            detrended[name] = detrended_arr
            trend_models[name] = model

            log(f"  {name}: trend RMS = {trend_rms:.4f}")
        except Exception as e:
            log(f"  {name}: trend removal failed ({e}), keeping original")
            detrended[name] = arr.copy()
            trend_models[name] = None

    return detrended, trend_models


def apply_trend_models(features, obj, trend_models, trend_cols=None):
    """
    Apply pre-fitted trend models to new data.

    Parameters
    ----------
    features : dict
        Dictionary of feature arrays.
    obj : astropy.table.Table
        Object catalog.
    trend_models : TrendModels or dict
        Trend models from remove_trends(). Legacy dicts with ``'__trend_cols__'``
        are also accepted.
    trend_cols : list of str, optional
        Columns used for trend modeling. Must match original.

    Returns
    -------
    detrended : dict
        Dictionary of detrended feature arrays.
    """
    trend_models = _coerce_trend_models(trend_models)
    if not trend_models:
        return _copy_feature_dict(features)

    if not any(model is not None for model in trend_models.values()):
        return _copy_feature_dict(features)

    if trend_models.trend_cols is not None:
        trend_cols = list(trend_models.trend_cols)
    elif trend_cols is None:
        trend_cols = ["x", "y"]

    missing_cols = [c for c in trend_cols if c not in obj.colnames]
    if missing_cols:
        raise ValueError(
            "Missing trend columns required by trend models: " + ", ".join(missing_cols)
        )

    pos = np.column_stack([np.array(obj[c]) for c in trend_cols])
    finite_pos = np.all(np.isfinite(pos), axis=1)

    detrended = {}
    for name, arr in features.items():
        arr = np.array(arr)
        model = trend_models.get(name)

        if model is not None:
            detrended_arr = arr.copy()
            if np.any(finite_pos):
                trend = model.predict(pos[finite_pos])
                detrended_arr[finite_pos] = arr[finite_pos] - trend
            detrended[name] = detrended_arr
        else:
            detrended[name] = arr.copy()

    return detrended


# =============================================================================
# Classifiers
# =============================================================================


def _features_to_array(features, feature_names=None, replace_nonfinite=True, fill_value=-1e6):
    """Convert feature dict to 2D array."""
    if feature_names is None:
        feature_names = list(features.keys())
    if not feature_names:
        raise ValueError("Feature dictionary is empty")

    X = np.column_stack([features[name] for name in feature_names])

    if replace_nonfinite:
        # Replace NaN/inf with extreme values (will be flagged as outliers)
        bad = ~np.isfinite(X)
        X[bad] = fill_value

    return X, feature_names


def _build_classifier_for_inference(classifier, model=None):
    """
    Build classifier object for classify().

    Returns
    -------
    clf : object
        Classifier instance.
    fit_on_current_features : bool
        Whether classify() should fit the classifier on the current feature set.
    """
    if not isinstance(classifier, str):
        return classifier, False

    if classifier == "scoring":
        return ScoringClassifier(), False
    if classifier == "isolation":
        if model is not None:
            return model, False
        return IsolationForestClassifier(), True
    if classifier == "randomforest":
        if model is None:
            raise ValueError("model required for randomforest classifier")
        return model, False

    raise ValueError(f"Unknown classifier: {classifier}")


def _build_classifier_for_training(classifier, random_state=0):
    """Build classifier object for train_classifier()."""
    if classifier == "randomforest":
        return RandomForestClassifier(random_state=random_state)
    if classifier == "isolation":
        return IsolationForestClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier: {classifier}")


def _validate_stratified_split_inputs(y):
    """Validate labels before stratified train/test split."""
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Need at least two label classes for training")
    if np.min(counts) < 2:
        raise ValueError("Each class needs at least 2 samples for stratified train/test split")


class ScoringClassifier:
    """
    Rule-based scoring classifier (no training needed).

    Each feature contributes to a score based on how far it deviates
    from the "typical" real source value.
    """

    # Default weights and thresholds for each feature
    # (feature_name, weight, ideal_value, bad_direction)
    # bad_direction: 'high' means high values are bad, 'low' means low values are bad
    # Default weights and thresholds calibrated on simulated data
    # (feature_name, weight, ideal_value, bad_direction, threshold)
    # bad_direction: 'high' means high values are bad, 'low' means low, 'both' means either
    # threshold: deviation at which penalty is 50% (score_component = 0.5)
    DEFAULT_RULES = {
        # Cutout features - calibrated on simulated astronomical images
        # Thresholds are lenient to avoid over-rejection; tune for your data
        "sharpness": {"weight": 0.15, "ideal": 1.2, "bad": "high", "threshold": 5.0},
        "concentration": {"weight": 0.05, "ideal": 0.25, "bad": "both", "threshold": 0.3},
        "symmetry": {"weight": 0.10, "ideal": 0.05, "bad": "high", "threshold": 0.5},
        "roundness": {"weight": 0.15, "ideal": 0.9, "bad": "low", "threshold": 0.5},
        "psf_match": {"weight": 0.05, "ideal": 0.002, "bad": "high", "threshold": 0.05},
        "peak_offset": {"weight": 0.05, "ideal": 0.0, "bad": "high", "threshold": 8.0},
        "edge_gradient": {"weight": 0.05, "ideal": 1.5, "bad": "high", "threshold": 5.0},
        "bg_consistency": {"weight": 0.05, "ideal": 0.0, "bad": "high", "threshold": 2.0},
        # Catalog features
        "ellipticity": {"weight": 0.15, "ideal": 0.1, "bad": "high", "threshold": 0.6},
        "fwhm_ratio": {"weight": 0.15, "ideal": 1.0, "bad": "both", "threshold": 1.5},
        "peakiness": {"weight": 0.05, "ideal": None, "bad": "high", "threshold": None},
    }

    def __init__(self, rules=None):
        """
        Initialize scoring classifier.

        Parameters
        ----------
        rules : dict, optional
            Custom scoring rules. Default: use DEFAULT_RULES.
        """
        self.rules = rules if rules is not None else self.DEFAULT_RULES.copy()
        self._fitted = True  # No fitting needed

    def fit(self, X, y=None):
        """Fit is a no-op for scoring classifier."""
        return self

    def predict_proba(self, features):
        """
        Compute real-bogus scores.

        Parameters
        ----------
        features : dict
            Feature dictionary.

        Returns
        -------
        scores : ndarray
            Scores from 0 (bogus) to 1 (real).
        """
        if not features:
            raise ValueError("Feature dictionary is empty")

        n = len(list(features.values())[0])
        scores = np.ones(n)
        total_weight = 0

        for name, rule in self.rules.items():
            if name not in features:
                continue

            arr = np.array(features[name])
            weight = rule["weight"]
            ideal = rule["ideal"]
            bad = rule["bad"]
            threshold = rule["threshold"]

            if ideal is None or threshold is None:
                # Use median as ideal
                ideal = np.nanmedian(arr)
                threshold = np.nanstd(arr) * 2
                if threshold <= 0:
                    continue

            # Compute deviation from ideal
            deviation = arr - ideal

            if bad == "high":
                # Penalize values above threshold
                penalty = np.maximum(0, deviation) / threshold
            elif bad == "low":
                # Penalize values below threshold
                penalty = np.maximum(0, -deviation) / threshold
            else:  # 'both'
                penalty = np.abs(deviation) / threshold

            # Clip and convert to score component
            penalty = np.clip(penalty, 0, 2)  # Max 2x threshold penalty
            score_component = 1 - penalty / 2  # Maps [0,2] -> [1,0]

            # Handle NaN
            score_component = np.where(np.isfinite(arr), score_component, 0.5)

            scores *= score_component**weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            scores = scores ** (1 / total_weight)

        return scores

    def predict(self, features, threshold=0.5):
        """
        Predict real (1) or bogus (-1).

        Parameters
        ----------
        features : dict
            Feature dictionary.
        threshold : float
            Score threshold. Default: 0.5.

        Returns
        -------
        predictions : ndarray
            1 for real, -1 for bogus.
        """
        scores = self.predict_proba(features)
        return np.where(scores >= threshold, 1, -1)


class IsolationForestClassifier:
    """
    Wrapper around sklearn IsolationForest for outlier detection.

    This is similar to what artefacts.py does but works with any features.
    """

    def __init__(self, contamination="auto", random_state=0, **kwargs):
        """
        Initialize IsolationForest classifier.

        Parameters
        ----------
        contamination : float or 'auto'
            Expected proportion of outliers.
        random_state : int
            Random seed.
        **kwargs
            Additional arguments for IsolationForest.
        """
        from sklearn.ensemble import IsolationForest

        self.clf = IsolationForest(contamination=contamination, random_state=random_state, **kwargs)
        self._fitted = False
        self.feature_names = None

    def fit(self, features, y=None):
        """
        Fit the isolation forest on features.

        Parameters
        ----------
        features : dict
            Feature dictionary.
        y : ignored
            Not used (unsupervised).
        """
        X, self.feature_names = _features_to_array(features, replace_nonfinite=False)
        X_valid, _ = _filter_finite_rows(X, context="fitting IsolationForest")
        self.clf.fit(X_valid)
        self._fitted = True
        return self

    def predict(self, features):
        """
        Predict inlier (1) or outlier (-1).

        Parameters
        ----------
        features : dict
            Feature dictionary.

        Returns
        -------
        predictions : ndarray
            1 for inlier (real), -1 for outlier (bogus).
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X, _ = _features_to_array(features, self.feature_names)
        return self.clf.predict(X)

    def predict_proba(self, features):
        """
        Compute anomaly scores (higher = more likely real).

        Parameters
        ----------
        features : dict
            Feature dictionary.

        Returns
        -------
        scores : ndarray
            Scores from 0 (anomaly) to 1 (normal).
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X, _ = _features_to_array(features, self.feature_names)

        # decision_function returns values centered at 0
        # positive = inlier, negative = outlier
        raw_scores = self.clf.decision_function(X)

        # Convert to 0-1 range using sigmoid with scaling
        # Typical decision_function range is roughly -0.5 to +0.5
        # Scale by 5 to make sigmoid more discriminative
        scores = 1 / (1 + np.exp(-5 * raw_scores))
        return scores


class RandomForestClassifier:
    """
    Wrapper around sklearn RandomForest for supervised classification.

    Requires labeled training data.
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=0, **kwargs):
        """
        Initialize RandomForest classifier.

        Parameters
        ----------
        n_estimators : int
            Number of trees.
        max_depth : int
            Maximum tree depth.
        random_state : int
            Random seed.
        **kwargs
            Additional arguments for RandomForestClassifier.
        """
        from sklearn.ensemble import RandomForestClassifier as RFC

        self.clf = RFC(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs
        )
        self._fitted = False
        self.feature_names = None

    def fit(self, features, y):
        """
        Fit the random forest on labeled features.

        Parameters
        ----------
        features : dict
            Feature dictionary.
        y : ndarray
            Labels (1 = real, 0 = bogus).
        """
        X, self.feature_names = _features_to_array(features, replace_nonfinite=False)
        X_valid, y_valid, _ = _filter_finite_rows(X, y=y, context="fitting RandomForest")
        self.clf.fit(X_valid, y_valid)
        self._fitted = True
        return self

    def predict(self, features):
        """
        Predict real (1) or bogus (-1).

        Parameters
        ----------
        features : dict
            Feature dictionary.

        Returns
        -------
        predictions : ndarray
            1 for real, -1 for bogus.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X, _ = _features_to_array(features, self.feature_names)
        preds = self.clf.predict(X)
        return np.where(preds == 1, 1, -1)

    def predict_proba(self, features):
        """
        Compute probability of being real.

        Parameters
        ----------
        features : dict
            Feature dictionary.

        Returns
        -------
        scores : ndarray
            Probability of being real (0-1).
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X, _ = _features_to_array(features, self.feature_names)
        probabilities = self.clf.predict_proba(X)
        classes = self.clf.classes_

        # probability of class 1 (real). If class 1 was absent in training,
        # the model can only return class 0.
        if 1 in classes:
            idx = np.where(classes == 1)[0][0]
            return probabilities[:, idx]
        return np.zeros(X.shape[0], dtype=float)

    @property
    def feature_importances_(self):
        """Return feature importances."""
        if not self._fitted:
            return None
        return dict(zip(self.feature_names, self.clf.feature_importances_))


# =============================================================================
# High-Level API
# =============================================================================


def classify(
    obj,
    image=None,
    bg=None,
    err=None,
    mask=None,
    fwhm=None,
    method="auto",
    feature_set="default",
    classifier="scoring",
    model=None,
    threshold=0.5,
    add_score=True,
    flag_bogus=True,
    remove_trend=True,
    trend_cols=None,
    trend_scales=None,
    radius=10,
    verbose=False,
):
    """
    Classify objects as real or bogus.

    This is the main entry point for feature-based real-bogus classification.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x', 'y' columns.
    image : ndarray, optional
        Science image. Required for 'cutout' and 'hybrid' methods.
    bg : ndarray or float, optional
        Background map or scalar.
    err : ndarray or float, optional
        Error/noise map.
    mask : ndarray, optional
        Boolean mask (True = masked).
    fwhm : float, optional
        Image FWHM. If None, estimated from catalog.
    method : str, optional
        Feature extraction method:
        - 'catalog': Catalog features only (no image needed)
        - 'cutout': Cutout features only
        - 'hybrid': Both catalog and cutout features
        - 'auto': 'hybrid' if image provided, else 'catalog'
    feature_set : str, optional
        Catalog feature set for catalog/hybrid methods:
        'minimal', 'default', or 'extended'.
    classifier : str or object, optional
        Classifier to use:
        - 'scoring': Rule-based scoring (no training needed)
        - 'isolation': IsolationForest (unsupervised)
        - 'randomforest': RandomForest (requires model)
        - Custom classifier object with predict/predict_proba methods
    model : object, optional
        Pre-trained classifier model. Required for 'randomforest'.
    threshold : float, optional
        Score threshold for classification. Default: 0.5.
    add_score : bool, optional
        Add 'rb_score' column to output. Default: True.
    flag_bogus : bool, optional
        Set flag 0x800 on bogus objects. Default: True.
    remove_trend : bool, optional
        Remove spatial trends from features. Default: True.
    trend_cols : list of str, optional
        Columns for trend removal. Default: ['x', 'y'].
    trend_scales : list of float, optional
        Scales for trend removal.
    radius : int, optional
        Cutout radius. Default: 10.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    obj : astropy.table.Table
        Input catalog with added 'rb_score' column (if add_score=True)
        and updated flags (if flag_bogus=True).
    """
    log = print if verbose else lambda *args, **kwargs: None

    # Make a copy to avoid modifying input
    obj = obj.copy()

    # Extract features
    log(f"Extracting features using method='{method}'")
    features, feature_names = extract_features(
        obj,
        image=image,
        bg=bg,
        err=err,
        mask=mask,
        fwhm=fwhm,
        radius=radius,
        method=method,
        feature_set=feature_set,
        verbose=verbose,
    )
    log(f"Extracted {len(feature_names)} features: {feature_names}")

    # Remove trends if requested
    if remove_trend:
        log("Removing spatial trends from features")
        features, trend_models = remove_trends(
            features, obj, trend_cols=trend_cols, trend_scales=trend_scales, verbose=verbose
        )

    # Get or create classifier
    clf, fit_on_current_features = _build_classifier_for_inference(classifier, model=model)
    if fit_on_current_features:
        log("Fitting IsolationForest on current data")
        clf.fit(features)

    # Predict
    scores = clf.predict_proba(features)
    predictions = np.where(scores >= threshold, 1, -1)

    n_real = np.sum(predictions > 0)
    n_bogus = np.sum(predictions < 0)
    log(f"Classification: {n_real} real, {n_bogus} bogus")

    # Add score column
    if add_score:
        if "rb_score" in obj.colnames:
            obj["rb_score"] = scores
        else:
            obj.add_column(Column(scores, name="rb_score"))

    # Flag bogus objects
    if flag_bogus:
        if "flags" not in obj.colnames:
            obj.add_column(Column(np.zeros(len(obj), dtype=np.int32), name="flags"))
        # Reset existing bogus bit so output reflects this classification pass.
        obj["flags"] &= ~0x800
        obj["flags"][predictions < 0] |= 0x800

    return obj


def train_classifier(
    training_features,
    labels,
    classifier="randomforest",
    test_size=0.2,
    random_state=0,
    verbose=False,
):
    """
    Train a classifier on labeled data.

    Parameters
    ----------
    training_features : dict
        Feature dictionary from extract_features().
    labels : ndarray
        Labels (1 = real, 0 = bogus).
    classifier : str, optional
        Classifier type: 'randomforest' or 'isolation'.
    test_size : float, optional
        Fraction for validation. Default: 0.2.
    random_state : int, optional
        Random seed.
    verbose : bool, optional
        Print metrics.

    Returns
    -------
    clf : classifier object
        Trained classifier.
    metrics : dict
        Training/validation metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    log = print if verbose else lambda *args, **kwargs: None

    X, feature_names = _features_to_array(training_features, replace_nonfinite=False)
    y = np.asarray(labels)

    # Drop non-finite rows before splitting/training.
    X, y, _ = _filter_finite_rows(X, y=y, context="training")
    _validate_stratified_split_inputs(y)

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to perform stratified train/test split. Check class balance and test_size."
        ) from exc

    # Create and train classifier
    clf = _build_classifier_for_training(classifier, random_state=random_state)

    # Convert back to feature dict for training
    train_features = {name: X_train[:, i] for i, name in enumerate(feature_names)}
    test_features = {name: X_test[:, i] for i, name in enumerate(feature_names)}

    clf.fit(train_features, y_train)

    # Evaluate
    y_pred_train = (clf.predict(train_features) > 0).astype(int)
    y_pred_test = (clf.predict(test_features) > 0).astype(int)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
    }

    if hasattr(clf, "feature_importances_"):
        metrics["feature_importances"] = clf.feature_importances_

    log(f"Training accuracy: {metrics['train_accuracy']:.3f}")
    log(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    log(f"Test precision: {metrics['test_precision']:.3f}")
    log(f"Test recall: {metrics['test_recall']:.3f}")
    log(f"Test F1: {metrics['test_f1']:.3f}")

    return clf, metrics


def generate_training_data(
    n_images=100,
    image_size=(512, 512),
    fwhm_range=(1.5, 6.0),
    n_stars_range=(30, 80),
    n_galaxies_range=(0, 0),
    n_cosmic_rays_range=(5, 20),
    n_hot_pixels_range=(5, 15),
    background_range=(500, 2000),
    add_gradient=False,
    gradient_amplitude=0.3,
    remove_trend=False,
    trend_cols=None,
    verbose=False,
):
    """
    Generate synthetic training data using simulation module.

    Parameters
    ----------
    n_images : int
        Number of images to simulate.
    image_size : tuple
        Image dimensions.
    fwhm_range : tuple
        Range of FWHM values.
    n_stars_range : tuple
        Range of number of stars per image (min, max).
    n_galaxies_range : tuple
        Range of number of galaxies per image (min, max).
    n_cosmic_rays_range : tuple
        Range of number of cosmic rays per image (min, max).
    n_hot_pixels_range : tuple
        Range of number of hot pixels per image (min, max).
    background_range : tuple
        Range of background levels (min, max).
    add_gradient : bool
        Add spatial gradients to simulated images to simulate realistic backgrounds.
    gradient_amplitude : float
        Amplitude of spatial gradient as fraction of background (0-1).
    remove_trend : bool
        Apply trend removal to features during training.
    trend_cols : list of str, optional
        Columns for trend removal. Default: ['x', 'y'].
    verbose : bool
        Print progress.

    Returns
    -------
    features : dict
        Combined features from all images.
    labels : ndarray
        Labels (1 = real, 0 = bogus).
    """
    from . import simulation, photometry

    log = print if verbose else lambda *args, **kwargs: None

    all_features = None
    all_labels = []

    for i in range(n_images):
        fwhm = np.random.uniform(*fwhm_range)
        n_stars = np.random.randint(n_stars_range[0], n_stars_range[1] + 1)
        n_galaxies = np.random.randint(n_galaxies_range[0], n_galaxies_range[1] + 1)
        n_cosmic_rays = np.random.randint(n_cosmic_rays_range[0], n_cosmic_rays_range[1] + 1)
        n_hot_pixels = np.random.randint(n_hot_pixels_range[0], n_hot_pixels_range[1] + 1)
        background = 10 ** np.random.uniform(
            np.log10(background_range[0]), np.log10(background_range[1])
        )

        log(
            f"Generating image {i + 1}/{n_images} (FWHM={fwhm:.1f}, stars={n_stars}, galaxies={n_galaxies}, bg={background:.0f})"
        )

        # Simulate image with known sources
        sim = simulation.simulate_image(
            width=image_size[0],
            height=image_size[1],
            n_stars=n_stars,
            star_fwhm=fwhm,
            n_galaxies=n_galaxies,
            n_cosmic_rays=n_cosmic_rays,
            n_hot_pixels=n_hot_pixels,
            background=background,
            readnoise=10,
            return_catalog=True,
            verbose=False,
        )

        image = sim["image"]
        catalog = sim["catalog"]

        # Add spatial gradient if requested
        if add_gradient:
            # Create smooth gradient across the image
            y_grid, x_grid = np.mgrid[0 : image_size[0], 0 : image_size[1]]
            # Random gradient direction and amplitude
            angle = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0, gradient_amplitude)
            gradient = (
                amplitude
                * background
                * (
                    np.cos(angle) * (x_grid / image_size[1] - 0.5)
                    + np.sin(angle) * (y_grid / image_size[0] - 0.5)
                )
            )
            image = image + gradient

        # Detect objects
        obj = photometry.get_objects_sep(image, thresh=3.0, aper=2 * fwhm, verbose=False)

        if len(obj) == 0:
            continue

        # Extract features
        features, _ = extract_features(
            obj, image, bg=sim.get("background"), fwhm=fwhm, method="hybrid", verbose=False
        )

        # Apply trend removal if requested
        if remove_trend:
            if trend_cols is None:
                trend_cols = ["x", "y"]
            features, _ = remove_trends(features, obj, trend_cols=trend_cols, verbose=False)

        # Match detected objects to catalog to get labels
        labels = np.zeros(len(obj))
        for j, row in enumerate(obj):
            x, y = row["x"], row["y"]
            # Check if near a real source
            for cat_row in catalog:
                if cat_row["type"] in ["star", "galaxy"]:
                    dist = np.sqrt((x - cat_row["x"]) ** 2 + (y - cat_row["y"]) ** 2)
                    if dist < fwhm:
                        labels[j] = 1
                        break

        # Accumulate
        if all_features is None:
            all_features = {k: list(v) for k, v in features.items()}
        else:
            for k, v in features.items():
                all_features[k].extend(v)
        all_labels.extend(labels)

    # Convert to arrays
    if all_features is None:
        log("No detections accumulated from simulations")
        return {}, np.array([], dtype=np.int8)

    all_features = {k: np.array(v) for k, v in all_features.items()}
    all_labels = np.array(all_labels)

    n_real = np.sum(all_labels == 1)
    n_bogus = np.sum(all_labels == 0)
    log(f"Generated {n_real} real and {n_bogus} bogus examples")

    return all_features, all_labels
