# Various artefact filtering routines

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
