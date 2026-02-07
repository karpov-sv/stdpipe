# Various artefact filtering routines
#
# For more advanced real-bogus classification with cutout-based morphological
# features, see the realbogus_features module:
#
#   from stdpipe import realbogus_features as rbf
#   obj = rbf.classify(obj, image, method='hybrid', classifier='scoring')
#
# The realbogus_features module provides:
# - Catalog-only, cutout-only, or hybrid feature extraction
# - Multiple classifiers: scoring (no training), IsolationForest, RandomForest
# - Generalized trend removal for all features
# - Training utilities for custom classifiers

import numpy as np
from sklearn.ensemble import IsolationForest

from . import smoothing


def filter_sextractor_detections(
    obj,
    trend_cols=['x', 'y', 'MAG_AUTO'],
    trend_scales=[1000, 1000, 2],
    return_features=False,
    return_classifier=False,
    random_state=0,
    verbose=True,
    **kwargs
):
    """
    Flag SExtractor detections likely to be artefacts using IsolationForest.

    Builds feature vectors from FLUX_RADIUS, FWHM, and FLUX_MAX/FLUX_AUTO.
    Optionally removes smooth spatial trends (e.g., across x/y/MAG_AUTO) via an
    approximate LOESS regressor before fitting the outlier model.

    Expected columns in `obj`
    -------------------------
    Required:
    - FLUX_RADIUS
    - fwhm
    - FLUX_MAX
    - FLUX_AUTO
    - flags
    Trend columns (when `trend_cols` is set):
    - columns named in `trend_cols` (default: x, y, MAG_AUTO)

    Parameters
    ----------
    obj : array-like / table
        SExtractor catalog with required columns.
    trend_cols : list[str] or None
        Columns used to model smooth trends; set to None/[] to skip detrending.
    trend_scales : list[float]
        Per-dimension scaling for LOESS distances; must match trend_cols length.
    return_features : bool
        If True, return the feature list (arrays + labels) without fitting.
    return_classifier : bool
        If True, return a callable that classifies new catalogs.
    random_state : int
        Random seed for IsolationForest.
    verbose : bool or callable
        Logging control; can be a print-like function.
    **kwargs :
        Extra arguments passed to `ApproxLoessRegressor` (e.g., k, robust_iters).

    Returns
    -------
    good : ndarray[bool] or callable
        Boolean mask of “good” detections, or a classifier callable if
        return_classifier is True.
    """

    # Simple wrapper around print for logging in verbose mode only
    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    def get_features(obj):
        return [
            [obj['FLUX_RADIUS'], 'FLUX_RADIUS'],
            [obj['fwhm'], 'FWHM'],
            [obj['FLUX_MAX'] / obj['FLUX_AUTO'], 'FLUX_MAX / FLUX_AUTO']
        ]

    features = get_features(obj)

    if return_features:
        return features

    log("Using isolation forest outlier detection over columns ({})".format(
        ", ".join([_[1] for _ in features])
    ))

    # Exclude blends etc from the fit, as well as broken measurements
    # also exclude 0x800 that we will use for outliers
    idx = (obj['flags'] & (0x7fff - 0x800)) == 0
    for f in features:
        idx &= np.isfinite(f[0]) & (f[0] > 0)

    if trend_cols:
        log("Removing smooth trends in {} using approximate LOESS".format(
            ", ".join(trend_cols)
        ))

        if trend_scales and len(trend_scales) != len(trend_cols):
            raise ValueError(f"trend_scales length is inconsistent with trend_cols length")

        pos = np.column_stack([np.array(obj[_]) for _ in trend_cols])
        trend_models = []
        X = []

        k = kwargs.pop('k', 20)
        for f in features:
            model = smoothing.ApproxLoessRegressor(k=k, scales=trend_scales, **kwargs)
            model.fit(pos[idx], f[0][idx])
            trend_models.append(model)
            X.append(np.array(f[0]) - model.predict(pos))

    else:
        trend_models = None
        X = [np.array(_[0]) for _ in features]

    X = np.column_stack(X)
    X[~np.isfinite(X)] = -100000 # Definitely outside of the good locus

    clf = IsolationForest(random_state=random_state).fit(X[idx])

    res = clf.predict(X)

    log(f"{np.sum(res > 0)} good, {np.sum(res < 0)} outliers")

    if return_classifier:
        def classifier(obj):
            features = get_features(obj)

            if trend_cols:
                pos = np.column_stack([np.array(obj[_]) for _ in trend_cols])
                X = []

                for f,model in zip(features, trend_models):
                    X.append(np.array(f[0]) - model.predict(pos))
            else:
                X = [np.array(_[0]) for _ in features]

            X = np.column_stack(X)
            X[~np.isfinite(X)] = -100000 # Definitely outside of the good locus

            return clf.predict(X) > 0

        return classifier

    return res > 0


def filter_detections(
    obj,
    image=None,
    bg=None,
    mask=None,
    fwhm=None,
    method='auto',
    classifier='isolation',
    threshold=0.5,
    remove_trend=True,
    trend_cols=None,
    trend_scales=None,
    add_score=False,
    flag_bogus=False,
    verbose=True,
    **kwargs
):
    """
    Filter detections using feature-based real-bogus classification.

    This is a convenience wrapper around realbogus_features.classify() that
    provides a simpler interface similar to filter_sextractor_detections().

    For full control over feature extraction and classification, use
    realbogus_features.classify() directly.

    Parameters
    ----------
    obj : astropy.table.Table
        Object catalog with 'x', 'y' columns.
    image : ndarray, optional
        Science image. If provided, cutout features will be extracted.
    bg : ndarray or float, optional
        Background map or scalar.
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
    classifier : str, optional
        Classifier to use:
        - 'scoring': Rule-based scoring (no training needed)
        - 'isolation': IsolationForest (unsupervised, default)
    threshold : float, optional
        Score threshold for classification. Default: 0.5.
    remove_trend : bool, optional
        Remove spatial trends from features. Default: True.
    trend_cols : list of str, optional
        Columns for trend removal. Default: ['x', 'y'].
    trend_scales : list of float, optional
        Scales for trend removal. Default: auto-computed.
    add_score : bool, optional
        Add 'rb_score' column to output. Default: False.
    flag_bogus : bool, optional
        Set flag 0x800 on bogus objects. Default: False.
    verbose : bool, optional
        Print progress.
    **kwargs
        Additional arguments passed to realbogus_features.classify().

    Returns
    -------
    good : ndarray[bool]
        Boolean mask of "good" (real) detections.

    See Also
    --------
    realbogus_features.classify : Full-featured classification function.
    filter_sextractor_detections : Original SExtractor-specific filter.

    Examples
    --------
    >>> # Simple catalog-only filtering (like original function)
    >>> good = filter_detections(obj, classifier='isolation')

    >>> # Cutout-based filtering with scoring (no training)
    >>> good = filter_detections(obj, image, classifier='scoring')

    >>> # Hybrid with trend removal
    >>> good = filter_detections(obj, image, method='hybrid',
    ...                          remove_trend=True, trend_cols=['x', 'y'])
    """
    from . import realbogus_features as rbf

    log = (verbose if callable(verbose) else print) if verbose else lambda *args,**kwargs: None

    # Call the full classify function
    result = rbf.classify(
        obj,
        image=image,
        bg=bg,
        mask=mask,
        fwhm=fwhm,
        method=method,
        classifier=classifier,
        threshold=threshold,
        add_score=add_score or True,  # Need score to compute mask
        flag_bogus=flag_bogus,
        remove_trend=remove_trend,
        trend_cols=trend_cols,
        trend_scales=trend_scales,
        verbose=verbose,
        **kwargs
    )

    # Return boolean mask
    good = result['rb_score'] >= threshold

    log(f"{np.sum(good)} good, {np.sum(~good)} outliers")

    return good
