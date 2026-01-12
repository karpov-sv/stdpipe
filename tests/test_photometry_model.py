"""
Unit tests for stdpipe.photometry_model module.
"""

import numpy as np
import pytest

from stdpipe import photometry_model


def _build_match_data(n=20, zero_point=25.0, color_term=0.12, include_color=False, seed=123):
    rng = np.random.default_rng(seed)

    ra = 10.0 + np.arange(n) * 0.01
    dec = 20.0 + np.arange(n) * 0.01

    obj_mag = rng.normal(15.0, 0.2, n)
    obj_magerr = np.full(n, 0.01)
    obj_flags = np.zeros(n, dtype=int)

    cat_magerr = np.full(n, 0.01)
    obj_x = rng.uniform(0.0, 2048.0, n)
    obj_y = rng.uniform(0.0, 2048.0, n)

    if include_color:
        cat_color = rng.uniform(-0.5, 1.5, n)
        cat_mag = obj_mag + zero_point + color_term * cat_color
    else:
        cat_color = None
        cat_mag = obj_mag + zero_point

    return {
        "obj_ra": ra,
        "obj_dec": dec,
        "obj_mag": obj_mag,
        "obj_magerr": obj_magerr,
        "obj_flags": obj_flags,
        "obj_x": obj_x,
        "obj_y": obj_y,
        "cat_ra": ra.copy(),
        "cat_dec": dec.copy(),
        "cat_mag": cat_mag,
        "cat_magerr": cat_magerr,
        "cat_color": cat_color,
        "zero_point": zero_point,
        "color_term": color_term,
    }


class TestMakeSeries:
    @pytest.mark.unit
    def test_make_series_order_one(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        series = photometry_model.make_series(mul=2.0, x=x, y=y, order=1, sum=False, zero=True)

        assert len(series) == 3
        np.testing.assert_allclose(series[0], np.array([2.0, 2.0]))
        np.testing.assert_allclose(series[1], 2.0 * x)
        np.testing.assert_allclose(series[2], 2.0 * y)

    @pytest.mark.unit
    def test_make_series_sum_matches_list(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.5, 2.5])

        series = photometry_model.make_series(mul=1.0, x=x, y=y, order=2, sum=False, zero=True)
        series_sum = photometry_model.make_series(mul=1.0, x=x, y=y, order=2, sum=True, zero=True)

        assert len(series) == 6
        np.testing.assert_allclose(series_sum, np.sum(series, axis=0))


class TestMatch:
    @pytest.mark.unit
    def test_match_constant_zero_point(self):
        data = _build_match_data(include_color=False)

        result = photometry_model.match(
            data["obj_ra"],
            data["obj_dec"],
            data["obj_mag"],
            data["obj_magerr"],
            data["obj_flags"],
            data["cat_ra"],
            data["cat_dec"],
            data["cat_mag"],
            cat_magerr=data["cat_magerr"],
            sr=1 / 3600,
            obj_x=data["obj_x"],
            obj_y=data["obj_y"],
            spatial_order=0,
            robust=False,
            verbose=False,
        )

        assert result is not None
        assert result["color_term"] is None
        assert np.all(result["idx"])

        np.testing.assert_allclose(result["zero"][result["idx"]], data["zero_point"], atol=1e-6)

        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=1e-6)

    @pytest.mark.unit
    def test_match_constant_zero_point_without_positions(self):
        data = _build_match_data(include_color=False)

        result = photometry_model.match(
            data["obj_ra"],
            data["obj_dec"],
            data["obj_mag"],
            data["obj_magerr"],
            data["obj_flags"],
            data["cat_ra"],
            data["cat_dec"],
            data["cat_mag"],
            cat_magerr=data["cat_magerr"],
            sr=1 / 3600,
            obj_x=None,
            obj_y=None,
            spatial_order=0,
            robust=False,
            verbose=False,
        )

        assert result is not None
        assert result["color_term"] is None
        assert len(result["omag"]) > 0

        zero_eval = result["zero_fn"](None, None)
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=1e-6)
        assert np.shape(zero_eval) == (1,)

        mag_eval = np.array([15.0, 16.0, 17.0])
        zero_eval_mag = result["zero_fn"](None, None, mag=mag_eval)
        np.testing.assert_allclose(zero_eval_mag, data["zero_point"], atol=1e-6)
        assert len(zero_eval_mag) == len(mag_eval)

    @pytest.mark.unit
    def test_match_color_term_fit(self):
        data = _build_match_data(include_color=True, color_term=0.08)

        result = photometry_model.match(
            data["obj_ra"],
            data["obj_dec"],
            data["obj_mag"],
            data["obj_magerr"],
            data["obj_flags"],
            data["cat_ra"],
            data["cat_dec"],
            data["cat_mag"],
            cat_magerr=data["cat_magerr"],
            cat_color=data["cat_color"],
            sr=1 / 3600,
            obj_x=data["obj_x"],
            obj_y=data["obj_y"],
            spatial_order=0,
            robust=False,
            use_color=1,
            verbose=False,
        )

        assert result is not None
        assert result["color_term"] is not None
        assert np.isclose(result["color_term"], data["color_term"], atol=1e-4)

        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=1e-4)


class TestSnModel:
    @pytest.mark.unit
    def test_make_sn_model_recovers_curve(self):
        mag = np.linspace(12.0, 20.0, 50)
        p0 = 1.0e-14
        p1 = 1.0e-8
        sn = 1.0 / np.sqrt(p0 * 10 ** (0.8 * mag) + p1 * 10 ** (0.4 * mag))

        model = photometry_model.make_sn_model(mag, sn)

        np.testing.assert_allclose(model(mag), sn, rtol=5e-2, atol=1e-4)

    @pytest.mark.unit
    def test_get_detection_limit_sn(self):
        mag = np.linspace(12.0, 20.0, 50)
        p0 = 1.0e-14
        p1 = 1.0e-8
        sn = 1.0 / np.sqrt(p0 * 10 ** (0.8 * mag) + p1 * 10 ** (0.4 * mag))

        mag0, sn_model = photometry_model.get_detection_limit_sn(
            mag, sn, sn=5.0, get_model=True, verbose=False
        )

        assert mag0 is not None
        assert np.isfinite(mag0)
        assert np.isclose(sn_model(mag0), 5.0, rtol=1e-2, atol=1e-2)
