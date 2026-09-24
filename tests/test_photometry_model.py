"""
Unit tests for stdpipe.photometry_model module.
"""

import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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


def _add_noise_and_outliers(data, sigma=0.01, n_outliers=0, outlier_offset=1.0, seed=321):
    """Perturb catalogue magnitudes with Gaussian noise and gross outliers.

    Noise keeps the robust fit scale away from zero, so the full IRLS path of
    ``_StableRLM`` is exercised instead of the perfect-fit early exit.
    """
    rng = np.random.default_rng(seed)
    n = len(data["cat_mag"])

    data["cat_mag"] = data["cat_mag"] + rng.normal(0, sigma, n)
    data["obj_magerr"] = np.full(n, sigma / np.sqrt(2))
    data["cat_magerr"] = np.full(n, sigma / np.sqrt(2))

    outliers = np.zeros(n, dtype=bool)
    if n_outliers:
        outliers[rng.choice(n, n_outliers, replace=False)] = True
        data["cat_mag"][outliers] += outlier_offset
    data["outliers"] = outliers

    return data


def _run_match(data, **kwargs):
    return photometry_model.match(
        data["obj_ra"],
        data["obj_dec"],
        data["obj_mag"],
        data["obj_magerr"],
        data["obj_flags"],
        data["cat_ra"],
        data["cat_dec"],
        data["cat_mag"],
        cat_magerr=data["cat_magerr"],
        cat_color=kwargs.pop("cat_color", data["cat_color"]),
        sr=1 / 3600,
        obj_x=data["obj_x"],
        obj_y=data["obj_y"],
        verbose=False,
        **kwargs,
    )


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


class TestIntrinsicScatter:
    @pytest.mark.unit
    def test_get_intrinsic_scatter_recovers_signal(self):
        rng = np.random.default_rng(42)

        yerr = rng.uniform(0.01, 0.03, 400)
        true_scatter = 0.05
        y = rng.normal(0.0, np.sqrt(yerr**2 + true_scatter**2))

        scatter = photometry_model.get_intrinsic_scatter(y, yerr, min=0.0, max=0.2)

        assert np.isfinite(scatter)
        assert np.isclose(scatter, true_scatter, atol=0.01)


class TestStablePinv:
    @pytest.mark.unit
    def test_stable_pinv_exog_matches_pseudoinverse_properties(self):
        rng = np.random.default_rng(123)
        exog = rng.normal(size=(500, 7))
        exog[:, 0] = 1.0
        exog *= np.linspace(0.5, 3.0, exog.shape[0])[:, None]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pinv, norm_cov = photometry_model._stable_pinv_exog(exog)

        assert pinv.shape == (exog.shape[1], exog.shape[0])
        assert norm_cov.shape == (exog.shape[1], exog.shape[1])
        assert np.all(np.isfinite(pinv))
        assert np.all(np.isfinite(norm_cov))

        gram = exog.T @ exog
        rhs = rng.normal(size=exog.shape[0])
        expected = np.linalg.solve(gram, exog.T @ rhs)

        np.testing.assert_allclose(pinv @ rhs, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(norm_cov @ gram, np.eye(exog.shape[1]), rtol=1e-10, atol=1e-10)


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

    @pytest.mark.unit
    def test_match_error_propagation_ignores_invalid_rows(self):
        data = _build_match_data(include_color=True, n=30)
        data["cat_color"][::7] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

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
                spatial_order=2,
                robust=False,
                use_color=1,
                verbose=False,
            )

        assert result is not None
        assert np.all(np.isfinite(result["zero_model_err"][result["idx0"]]))
        assert np.all(np.isnan(result["zero_model_err"][~result["idx0"]]))

        zero_err = result["zero_fn"](data["obj_x"][:5], data["obj_y"][:5], get_err=True)
        assert np.all(np.isfinite(zero_err))
        assert np.all(zero_err >= 0)


class TestStableRLM:
    """Direct tests of the ``_StableRLM`` subclass used by robust ``match()``.

    It re-implements ``RLM.fit()`` on top of private statsmodels helpers, so
    these tests guard against upstream API drift (e.g. the statsmodels 0.15
    ``RLM._estimate_scale(resid, scale_est)`` signature change).
    """

    @staticmethod
    def _make_data(n=200, n_outliers=0, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        exog = np.vstack([np.ones(n), x]).T
        endog = 2.0 + 0.5 * x + rng.normal(0, 0.1, n)
        if n_outliers:
            endog[rng.choice(n, n_outliers, replace=False)] += 5.0
        return endog, exog

    @pytest.mark.unit
    def test_fit_matches_statsmodels_rlm(self):
        endog, exog = self._make_data(n_outliers=10)

        C = photometry_model._StableRLM(endog, exog).fit()
        R = sm.RLM(endog, exog).fit()

        np.testing.assert_allclose(C.params, R.params, rtol=1e-6)
        np.testing.assert_allclose(C.scale, R.scale, rtol=1e-6)
        np.testing.assert_allclose(C.bse, R.bse, rtol=1e-6)

    @pytest.mark.unit
    def test_fit_downweights_outliers(self):
        endog, exog = self._make_data(n_outliers=20)
        outliers = endog - (2.0 + 0.5 * exog[:, 1]) > 2.5

        C = photometry_model._StableRLM(endog, exog).fit()

        np.testing.assert_allclose(C.params, [2.0, 0.5], atol=0.03)
        # Scale is a standard deviation of the clean residuals
        assert 0.07 < C.scale < 0.13
        assert np.all(C.weights[outliers] < 0.1)
        assert np.median(C.weights[~outliers]) == pytest.approx(1.0)
        assert C.fit_history["iteration"] > 1

    @pytest.mark.unit
    def test_fit_perfect_data_returns_exact_params(self):
        x = np.linspace(-1, 1, 50)
        exog = np.vstack([np.ones_like(x), x]).T
        endog = 2.0 + 0.5 * x

        # Depending on roundoff the scale is either exactly zero (early exit
        # with a ConvergenceWarning) or negligible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            C = photometry_model._StableRLM(endog, exog).fit()

        np.testing.assert_allclose(C.params, [2.0, 0.5], atol=1e-10)
        # Treated as a perfect fit by match()
        assert C.scale < photometry_model._MIN_FIT_SCALE

    @pytest.mark.unit
    def test_fit_rank_deficient_design_gives_minimum_norm_solution(self):
        endog, exog = self._make_data(n_outliers=10)
        # Duplicated column makes the design singular; the slope may be split
        # arbitrarily between the two copies, and the minimum-norm solution
        # splits it evenly instead of returning huge cancelling values
        exog = np.hstack([exog, exog[:, 1:]])

        C = photometry_model._StableRLM(endog, exog).fit()

        assert np.all(np.isfinite(C.params))
        assert np.all(np.abs(C.params) < 10)
        np.testing.assert_allclose(C.params[1], C.params[2], rtol=1e-8)
        np.testing.assert_allclose(C.params[0], 2.0, atol=0.03)
        np.testing.assert_allclose(C.params[1] + C.params[2], 0.5, atol=0.03)


class TestMatchRobust:
    """``match()`` with ``robust=True`` (the default), i.e. via ``_StableRLM``."""

    @pytest.mark.unit
    def test_match_robust_is_default(self):
        data = _add_noise_and_outliers(_build_match_data(n=50))

        result = _run_match(data, spatial_order=0)
        result_robust = _run_match(data, spatial_order=0, robust=True)

        assert result is not None
        np.testing.assert_array_equal(result["idx"], result_robust["idx"])
        np.testing.assert_allclose(result["zero_model"], result_robust["zero_model"])

    @pytest.mark.unit
    def test_match_robust_constant_zero_point(self):
        data = _add_noise_and_outliers(_build_match_data(n=50))

        result = _run_match(data, spatial_order=0, robust=True)

        assert result is not None
        assert result["color_term"] is None
        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=5e-3)

        zero_err = result["zero_fn"](data["obj_x"][:5], data["obj_y"][:5], get_err=True)
        assert np.all(np.isfinite(zero_err))
        assert np.all(zero_err > 0)

    @pytest.mark.unit
    def test_match_robust_rejects_outliers(self):
        data = _add_noise_and_outliers(_build_match_data(n=60), n_outliers=8, outlier_offset=0.5)

        result = _run_match(data, spatial_order=0, robust=True, threshold=5.0)

        assert result is not None
        # Outliers are rejected, clean points are kept
        assert not np.any(result["idx"][data["outliers"]])
        assert np.mean(result["idx"][~data["outliers"]]) > 0.9

        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=5e-3)

    @pytest.mark.unit
    def test_match_robust_color_term_and_spatial(self):
        data = _add_noise_and_outliers(
            _build_match_data(n=100, include_color=True, color_term=0.08),
            n_outliers=5,
        )

        result = _run_match(data, spatial_order=1, robust=True, use_color=1)

        assert result is not None
        assert np.isclose(result["color_term"], data["color_term"], atol=5e-3)
        assert np.isfinite(result["color_term_err"]) and result["color_term_err"] > 0

        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=1e-2)

    @pytest.mark.unit
    def test_match_robust_perfect_fit(self):
        # Noiseless data: the robust scale is exactly zero, which must be
        # treated as a perfect fit and not as a degenerate model
        data = _build_match_data(include_color=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = _run_match(data, spatial_order=0, robust=True)

        assert result is not None
        assert np.all(result["idx"])
        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"])
        np.testing.assert_allclose(zero_eval, data["zero_point"], atol=1e-6)

    @pytest.mark.unit
    def test_match_robust_agrees_with_weighted_on_clean_data(self):
        data = _add_noise_and_outliers(_build_match_data(n=80, include_color=True))

        kwargs = dict(spatial_order=1, use_color=1, threshold=None)
        result_rlm = _run_match(data, robust=True, **kwargs)
        result_wls = _run_match(data, robust=False, **kwargs)

        assert result_rlm is not None and result_wls is not None
        np.testing.assert_allclose(result_rlm["color_term"], result_wls["color_term"], atol=5e-3)
        np.testing.assert_allclose(
            result_rlm["zero_fn"](data["obj_x"], data["obj_y"]),
            result_wls["zero_fn"](data["obj_x"], data["obj_y"]),
            atol=5e-3,
        )


class TestMatchRankDeficient:
    """``match()`` with a design matrix not constrained by the data."""

    @staticmethod
    def _constant_color_data(color=0.5):
        # All stars share the same color, so the color term column is
        # proportional to the constant one and only their sum is constrained
        data = _add_noise_and_outliers(_build_match_data(n=60, include_color=True))
        data["cat_color"][:] = color
        rng = np.random.default_rng(1)
        data["cat_mag"] = (
            data["obj_mag"] + data["zero_point"] + data["color_term"] * color + rng.normal(0, 0.01, 60)
        )
        return data

    @pytest.mark.unit
    @pytest.mark.parametrize("robust", [True, False])
    def test_match_constant_color(self, robust):
        data = self._constant_color_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = _run_match(data, spatial_order=0, robust=robust, use_color=1)

        assert result is not None
        # Minimum-norm solution: finite, with the constrained combination correct
        assert np.isfinite(result["color_term"]) and abs(result["color_term"]) < 100
        zero_eval = result["zero_fn"](data["obj_x"], data["obj_y"]) + result["color_term"] * 0.5
        np.testing.assert_allclose(zero_eval, data["zero_point"] + data["color_term"] * 0.5, atol=5e-3)

        # The color term alone, and the zero point without it, are unconstrained
        assert np.isnan(result["color_term_err"])
        assert np.all(np.isnan(result["zero_fn"](data["obj_x"][:5], data["obj_y"][:5], get_err=True)))

        # The full model at the fitted points is constrained
        assert np.all(np.isfinite(result["zero_model_err"][result["idx"]]))
        assert np.all(result["zero_model_err"][result["idx"]] > 0)

    @pytest.mark.unit
    def test_match_constant_color_robust_agrees_with_weighted(self):
        data = self._constant_color_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result_rlm = _run_match(data, spatial_order=0, robust=True, use_color=1)
            result_wls = _run_match(data, spatial_order=0, robust=False, use_color=1)

        np.testing.assert_allclose(result_rlm["color_term"], result_wls["color_term"], rtol=1e-2)

    @pytest.mark.unit
    def test_match_full_rank_errors_stay_finite(self):
        data = _add_noise_and_outliers(_build_match_data(n=60, include_color=True))

        result = _run_match(data, spatial_order=1, robust=True, use_color=1)

        assert np.isfinite(result["color_term_err"]) and result["color_term_err"] > 0
        assert np.all(np.isfinite(result["zero_model_err"][result["idx0"]]))
        assert np.all(np.isfinite(result["zero_fn"](data["obj_x"], data["obj_y"], get_err=True)))


class TestEstimable:
    @pytest.mark.unit
    def test_full_rank_returns_none(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        assert photometry_model._get_row_space(X) is None
        assert np.all(photometry_model._is_estimable(np.eye(3), None))

    @pytest.mark.unit
    def test_collinear_columns(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=20)
        X = np.vstack([np.ones(20), 0.5 * np.ones(20), x]).T

        row_space = photometry_model._get_row_space(X)
        assert row_space is not None
        assert len(row_space[0]) == 2

        rows = np.array(
            [
                [1.0, 0.0, 0.0],  # intercept alone - not estimable
                [0.0, 1.0, 0.0],  # constant color alone - not estimable
                [1.0, 0.5, 0.0],  # their combination at fitted color - estimable
                [0.0, 0.0, 1.0],  # independent slope - estimable
                [1.0, 0.5, 3.0],  # any fitted row - estimable
                [np.nan, 0.0, 0.0],  # invalid rows are left to callers
            ]
        )
        np.testing.assert_array_equal(
            photometry_model._is_estimable(rows, row_space),
            [False, False, True, True, True, True],
        )

    @pytest.mark.unit
    def test_column_scaling_does_not_affect_rank(self):
        rng = np.random.default_rng(0)
        # Widely different but independent column scales, as with bg_order terms
        X = np.vstack([np.ones(30), 1e8 * rng.uniform(1, 2, 30), 1e-6 * rng.normal(size=30)]).T
        assert photometry_model._get_row_space(X) is None


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
    def test_make_sn_model_with_floor(self):
        mag = np.linspace(10.0, 20.0, 100)
        p0 = 1.0e-14
        p1 = 1.0e-8
        p2 = 1.0e-4  # S/N saturates at 100 for bright stars
        sn = 1.0 / np.sqrt(p0 * 10 ** (0.8 * mag) + p1 * 10 ** (0.4 * mag) + p2)

        model = photometry_model.make_sn_model(mag, sn)

        np.testing.assert_allclose(model(mag), sn, rtol=5e-2)
        # Bright end must saturate at 1/sqrt(p2) instead of growing indefinitely
        assert np.isclose(model(5.0), 1.0 / np.sqrt(p2), rtol=5e-2)

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
