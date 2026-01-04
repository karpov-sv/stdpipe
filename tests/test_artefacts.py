import numpy as np

from stdpipe.artefacts import filter_sextractor_detections


def _make_obj(n=50):
    rng = np.random.default_rng(0)
    flux_auto = rng.uniform(1000.0, 2000.0, size=n)
    flux_max = flux_auto * rng.uniform(0.2, 0.6, size=n)
    obj = {
        'FLUX_RADIUS': rng.uniform(1.0, 3.0, size=n),
        'fwhm': rng.uniform(2.0, 4.0, size=n),
        'FLUX_MAX': flux_max,
        'FLUX_AUTO': flux_auto,
        'flags': np.zeros(n, dtype=int),
        'x': rng.uniform(0.0, 2000.0, size=n),
        'y': rng.uniform(0.0, 2000.0, size=n),
        'MAG_AUTO': rng.uniform(15.0, 20.0, size=n),
    }
    return obj


def test_filter_sextractor_returns_features():
    obj = _make_obj(10)
    features = filter_sextractor_detections(obj, return_features=True)

    assert len(features) == 3
    assert [f[1] for f in features] == ['FLUX_RADIUS', 'FWHM', 'FLUX_MAX / FLUX_AUTO']


def test_filter_sextractor_classifier_matches_prediction():
    obj = _make_obj(60)
    # Inject a couple of problematic rows
    obj['FLUX_RADIUS'][0] = np.nan
    obj['FLUX_MAX'][1] = np.nan
    obj['flags'][2] = 1

    res = filter_sextractor_detections(
        obj, trend_cols=None, random_state=0, verbose=False
    )
    clf = filter_sextractor_detections(
        obj, trend_cols=None, return_classifier=True, random_state=0, verbose=False
    )

    res2 = clf(obj)
    assert res.shape == (len(obj['FLUX_AUTO']),)
    assert res2.shape == res.shape
    assert np.array_equal(res, res2)


def test_filter_sextractor_with_trend_cols():
    obj = _make_obj(80)

    res = filter_sextractor_detections(
        obj,
        trend_cols=['x', 'y', 'MAG_AUTO'],
        trend_scales=[1000, 1000, 2],
        random_state=1,
        verbose=False,
        k=15,
        robust_iters=0,
    )
    assert res.shape == (len(obj['FLUX_AUTO']),)
