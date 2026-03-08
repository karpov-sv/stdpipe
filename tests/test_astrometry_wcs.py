"""
Tests for stdpipe.astrometry_wcs module.

Tests ZPN WCS fitting, ZPN-SIP distortion fitting, TAN-SIP robust fitting,
and the fit_wcs_from_points dispatcher.
"""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcs import Sip
from astropy.coordinates import SkyCoord
import astropy.units as u

from stdpipe.astrometry_wcs import (
    fit_wcs_from_points,
    fit_zpn_wcs_from_points,
    tan_wcs_to_zpn,
    _fit_zpn_sip,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def zpn_wcs():
    """ZPN WCS with radial distortion (PV2_3), 1024x1024, ~1 arcsec/pix."""
    w = WCS(naxis=2)
    w.wcs.crpix = [512, 512]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = np.array([[-3.3e-4, 0], [0, 3.3e-4]])
    w.wcs.ctype = ['RA---ZPN', 'DEC--ZPN']
    w.wcs.set_pv([(2, 0, 0.0), (2, 1, 1.0), (2, 3, 0.5)])
    w.pixel_shape = (1024, 1024)
    return w


@pytest.fixture
def zpn_sip_wcs(zpn_wcs):
    """ZPN WCS with both PV radial distortion and SIP corrections."""
    w = zpn_wcs.deepcopy()
    w.wcs.ctype = ['RA---ZPN-SIP', 'DEC--ZPN-SIP']
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0] = 1e-6
    a[0, 2] = 2e-6
    a[1, 1] = -5e-7
    b[2, 0] = -1e-6
    b[0, 2] = 1e-6
    b[1, 1] = 3e-7
    w.sip = Sip(a, b, np.zeros((3, 3)), np.zeros((3, 3)), w.wcs.crpix)
    return w


@pytest.fixture
def tan_wcs():
    """Simple TAN WCS, 1024x1024, ~1 arcsec/pix."""
    w = WCS(naxis=2)
    w.wcs.crpix = [512, 512]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = np.array([[-3.3e-4, 0], [0, 3.3e-4]])
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.pixel_shape = (1024, 1024)
    return w


def _generate_stars(wcs, n=300, margin=80, seed=42):
    """Generate synthetic star positions for a given WCS."""
    np.random.seed(seed)
    nx, ny = wcs.pixel_shape
    x = np.random.uniform(margin, nx - margin, n)
    y = np.random.uniform(margin, ny - margin, n)
    ra, dec = wcs.all_pix2world(x, y, 0)
    good = np.isfinite(ra) & np.isfinite(dec)
    return x[good], y[good], SkyCoord(ra[good], dec[good], unit='deg')


# ============================================================================
# Tests: tan_wcs_to_zpn
# ============================================================================

class TestTanToZPN:

    def test_basic_conversion(self, tan_wcs):
        """TAN → ZPN conversion produces valid ZPN WCS."""
        w_zpn = tan_wcs_to_zpn(tan_wcs, pv_deg=5)
        assert 'ZPN' in w_zpn.wcs.ctype[0]
        assert 'ZPN' in w_zpn.wcs.ctype[1]
        assert w_zpn.sip is None

    def test_preserves_crval(self, tan_wcs):
        """CRVAL and CRPIX are preserved."""
        w_zpn = tan_wcs_to_zpn(tan_wcs, pv_deg=5)
        np.testing.assert_allclose(w_zpn.wcs.crval, tan_wcs.wcs.crval)
        np.testing.assert_allclose(w_zpn.wcs.crpix, tan_wcs.wcs.crpix)

    def test_sky_agreement(self, tan_wcs):
        """TAN and converted ZPN agree on sky positions to < 0.01 arcsec."""
        w_zpn = tan_wcs_to_zpn(tan_wcs, pv_deg=7)
        x, y, sky_tan = _generate_stars(tan_wcs, n=200)

        ra_zpn, dec_zpn = w_zpn.all_pix2world(x, y, 0)
        sky_zpn = SkyCoord(ra_zpn, dec_zpn, unit='deg')

        sep = sky_tan.separation(sky_zpn).arcsec
        assert np.max(sep) < 0.01, f"Max separation {np.max(sep):.4f}\" > 0.01\""


# ============================================================================
# Tests: fit_zpn_wcs_from_points
# ============================================================================

class TestFitZPNWCS:

    def test_recovers_zpn_from_perturbed_init(self, zpn_wcs):
        """ZPN fitter recovers the correct WCS from a perturbed initial guess."""
        x, y, sky = _generate_stars(zpn_wcs)
        xy = np.column_stack([x, y])

        # Perturb the initial WCS
        w_init = zpn_wcs.deepcopy()
        w_init.wcs.crval[0] += 0.002
        w_init.wcs.crval[1] -= 0.001

        wcs_fit, result = fit_zpn_wcs_from_points(xy, sky, w_init, pv_deg=5)

        ra_fit, dec_fit = wcs_fit.all_pix2world(x, y, 0)
        sep = SkyCoord(ra_fit, dec_fit, unit='deg').separation(sky).arcsec
        assert np.median(sep) < 0.01, f"Median residual {np.median(sep):.4f}\" > 0.01\""

    def test_preserves_zpn_ctype(self, zpn_wcs):
        """Output WCS has ZPN projection type."""
        x, y, sky = _generate_stars(zpn_wcs)
        xy = np.column_stack([x, y])
        wcs_fit, _ = fit_zpn_wcs_from_points(xy, sky, zpn_wcs, pv_deg=3)
        assert 'ZPN' in wcs_fit.wcs.ctype[0]

    def test_noisy_positions(self, zpn_wcs):
        """ZPN fitter handles noisy pixel positions gracefully."""
        x, y, sky = _generate_stars(zpn_wcs, n=200)
        x_noisy = x + np.random.normal(0, 0.3, len(x))
        y_noisy = y + np.random.normal(0, 0.3, len(y))
        xy = np.column_stack([x_noisy, y_noisy])

        wcs_fit, _ = fit_zpn_wcs_from_points(xy, sky, zpn_wcs, pv_deg=3)

        ra_fit, dec_fit = wcs_fit.all_pix2world(x, y, 0)
        sep = SkyCoord(ra_fit, dec_fit, unit='deg').separation(sky).arcsec
        # With 0.3 pixel noise at ~1.2"/pix, expect ~0.3" residuals
        assert np.median(sep) < 0.5

    def test_fit_flags(self, zpn_wcs):
        """Toggling fit_crpix/fit_crval/fit_cd/fit_pv works."""
        x, y, sky = _generate_stars(zpn_wcs)
        xy = np.column_stack([x, y])

        # Fit only CD
        wcs_fit, _ = fit_zpn_wcs_from_points(
            xy, sky, zpn_wcs, pv_deg=3,
            fit_crpix=False, fit_crval=False, fit_pv=False,
        )
        # CRPIX and CRVAL should be unchanged
        np.testing.assert_allclose(wcs_fit.wcs.crpix, zpn_wcs.wcs.crpix)
        np.testing.assert_allclose(wcs_fit.wcs.crval, zpn_wcs.wcs.crval)


# ============================================================================
# Tests: _fit_zpn_sip
# ============================================================================

class TestFitZPNSIP:

    def test_sip_improves_accuracy(self, zpn_sip_wcs):
        """SIP fitting on top of ZPN reduces residuals for ZPN-SIP truth."""
        x, y, sky = _generate_stars(zpn_sip_wcs)
        xy = np.column_stack([x, y])

        # Strip SIP from init
        w_init = zpn_sip_wcs.deepcopy()
        w_init.sip = None
        w_init.wcs.ctype = ['RA---ZPN', 'DEC--ZPN']

        # ZPN-only fit
        wcs_zpn, _ = fit_zpn_wcs_from_points(xy, sky, w_init, pv_deg=5)
        ra1, dec1 = wcs_zpn.all_pix2world(x, y, 0)
        sep_zpn = SkyCoord(ra1, dec1, unit='deg').separation(sky).arcsec

        # ZPN + SIP fit
        wcs_sip = _fit_zpn_sip(wcs_zpn, xy, sky, sip_degree=2, pv_deg=5)
        ra2, dec2 = wcs_sip.all_pix2world(x, y, 0)
        sep_sip = SkyCoord(ra2, dec2, unit='deg').separation(sky).arcsec

        assert np.median(sep_sip) < np.median(sep_zpn), (
            f"SIP should improve: {np.median(sep_sip):.4f}\" >= {np.median(sep_zpn):.4f}\""
        )

    def test_sip_ctype(self, zpn_wcs):
        """Output WCS has ZPN-SIP CTYPE."""
        x, y, sky = _generate_stars(zpn_wcs)
        xy = np.column_stack([x, y])
        wcs_sip = _fit_zpn_sip(zpn_wcs, xy, sky, sip_degree=2)
        assert '-SIP' in wcs_sip.wcs.ctype[0]
        assert '-SIP' in wcs_sip.wcs.ctype[1]
        assert wcs_sip.sip is not None

    def test_sip_no_distortion(self, zpn_wcs):
        """When truth has no SIP, fitted SIP coefficients should be near zero."""
        x, y, sky = _generate_stars(zpn_wcs)
        xy = np.column_stack([x, y])

        wcs_zpn, _ = fit_zpn_wcs_from_points(xy, sky, zpn_wcs, pv_deg=5)
        wcs_sip = _fit_zpn_sip(wcs_zpn, xy, sky, sip_degree=2, pv_deg=5)

        # SIP coefficients should be negligible (< 1e-8)
        assert np.max(np.abs(wcs_sip.sip.a)) < 1e-8, (
            f"SIP A coeffs too large: {np.max(np.abs(wcs_sip.sip.a)):.2e}"
        )
        assert np.max(np.abs(wcs_sip.sip.b)) < 1e-8, (
            f"SIP B coeffs too large: {np.max(np.abs(wcs_sip.sip.b)):.2e}"
        )

    def test_sip_orders(self, zpn_sip_wcs):
        """Higher SIP order should give equal or better residuals."""
        x, y, sky = _generate_stars(zpn_sip_wcs)
        xy = np.column_stack([x, y])

        w_init = zpn_sip_wcs.deepcopy()
        w_init.sip = None
        w_init.wcs.ctype = ['RA---ZPN', 'DEC--ZPN']
        wcs_zpn, _ = fit_zpn_wcs_from_points(xy, sky, w_init, pv_deg=5)

        seps = {}
        for order in [2, 3]:
            wcs_sip = _fit_zpn_sip(wcs_zpn, xy, sky, sip_degree=order, pv_deg=5)
            ra, dec = wcs_sip.all_pix2world(x, y, 0)
            sep = SkyCoord(ra, dec, unit='deg').separation(sky).arcsec
            seps[order] = np.median(sep)

        # Order 3 should be at least as good as order 2
        assert seps[3] <= seps[2] * 1.1, (
            f"SIP3 ({seps[3]:.4f}\") worse than SIP2 ({seps[2]:.4f}\")"
        )

    def test_iteration_convergence(self, zpn_sip_wcs):
        """More iterations should not degrade the solution."""
        x, y, sky = _generate_stars(zpn_sip_wcs)
        xy = np.column_stack([x, y])

        w_init = zpn_sip_wcs.deepcopy()
        w_init.sip = None
        w_init.wcs.ctype = ['RA---ZPN', 'DEC--ZPN']
        wcs_zpn, _ = fit_zpn_wcs_from_points(xy, sky, w_init, pv_deg=5)

        seps = {}
        for n_iter in [1, 3, 5]:
            wcs_sip = _fit_zpn_sip(
                wcs_zpn, xy, sky, sip_degree=2, pv_deg=5, n_iter=n_iter
            )
            ra, dec = wcs_sip.all_pix2world(x, y, 0)
            sep = SkyCoord(ra, dec, unit='deg').separation(sky).arcsec
            seps[n_iter] = np.median(sep)

        # 3 and 5 iterations should be at least as good as 1
        assert seps[3] <= seps[1] * 1.05
        assert seps[5] <= seps[1] * 1.05


# ============================================================================
# Tests: fit_wcs_from_points dispatcher
# ============================================================================

class TestFitWCSFromPoints:

    def test_zpn_no_sip(self, zpn_wcs):
        """Dispatcher routes ZPN without SIP correctly."""
        x, y, sky = _generate_stars(zpn_wcs)

        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=zpn_wcs, sip_degree=0
        )
        assert 'ZPN' in wcs_fit.wcs.ctype[0]
        assert wcs_fit.sip is None or '-SIP' not in wcs_fit.wcs.ctype[0]

    def test_zpn_with_sip(self, zpn_sip_wcs):
        """Dispatcher routes ZPN+SIP correctly and produces ZPN-SIP output."""
        x, y, sky = _generate_stars(zpn_sip_wcs)

        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=zpn_sip_wcs, sip_degree=2
        )
        assert 'ZPN' in wcs_fit.wcs.ctype[0]
        assert '-SIP' in wcs_fit.wcs.ctype[0]
        assert wcs_fit.sip is not None

    def test_zpn_sip_accuracy(self, zpn_sip_wcs):
        """ZPN-SIP fitting via dispatcher achieves good accuracy."""
        x, y, sky = _generate_stars(zpn_sip_wcs)

        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=zpn_sip_wcs, sip_degree=2
        )
        ra, dec = wcs_fit.all_pix2world(x, y, 0)
        sep = SkyCoord(ra, dec, unit='deg').separation(sky).arcsec
        assert np.median(sep) < 0.1, (
            f"Median residual {np.median(sep):.4f}\" > 0.1\""
        )

    def test_tan_sip_routing(self, tan_wcs):
        """TAN with sip_degree > 0 goes through robust SIP fitter."""
        x, y, sky = _generate_stars(tan_wcs, n=100)

        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=tan_wcs, sip_degree=2
        )
        assert 'TAN' in wcs_fit.wcs.ctype[0]
        assert '-SIP' in wcs_fit.wcs.ctype[0]

    def test_tan_no_sip(self, tan_wcs):
        """TAN without SIP goes through astropy's fitter."""
        x, y, sky = _generate_stars(tan_wcs, n=100)

        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=tan_wcs, sip_degree=0
        )
        assert 'TAN' in wcs_fit.wcs.ctype[0]

    def test_pv_deg_independent_of_sip_degree(self, zpn_wcs):
        """pv_deg and sip_degree control separate aspects."""
        x, y, sky = _generate_stars(zpn_wcs)

        # sip_degree=2, pv_deg=3 — PV should have degree 3
        wcs_fit = fit_wcs_from_points(
            [x, y], sky, projection=zpn_wcs, sip_degree=2, pv_deg=3
        )
        pv = dict((m, val) for (i, m, val) in wcs_fit.wcs.get_pv() if i == 2)
        assert 3 in pv, "PV2_3 should be present with pv_deg=3"
        assert 4 not in pv, "PV2_4 should not be present with pv_deg=3"

    def test_xy_format_tuple(self, zpn_wcs):
        """Accepts (x_array, y_array) tuple format."""
        x, y, sky = _generate_stars(zpn_wcs, n=100)
        wcs_fit = fit_wcs_from_points((x, y), sky, projection=zpn_wcs)
        assert wcs_fit is not None

    def test_xy_format_2xN(self, zpn_wcs):
        """Accepts (2, N) array format."""
        x, y, sky = _generate_stars(zpn_wcs, n=100)
        xy_2n = np.array([x, y])
        wcs_fit = fit_wcs_from_points(xy_2n, sky, projection=zpn_wcs)
        assert wcs_fit is not None


# ============================================================================
# Tests: round-trip consistency
# ============================================================================

class TestRoundTrip:

    def test_zpn_sip_forward_inverse(self, zpn_sip_wcs):
        """ZPN-SIP WCS round-trips through pix2world/world2pix."""
        x, y, sky = _generate_stars(zpn_sip_wcs, n=50)
        xy = np.column_stack([x, y])

        w_init = zpn_sip_wcs.deepcopy()
        w_init.sip = None
        w_init.wcs.ctype = ['RA---ZPN', 'DEC--ZPN']

        wcs_zpn, _ = fit_zpn_wcs_from_points(xy, sky, w_init, pv_deg=5)
        wcs_sip = _fit_zpn_sip(wcs_zpn, xy, sky, sip_degree=2, pv_deg=5)

        # Forward: pix → sky
        ra, dec = wcs_sip.all_pix2world(x, y, 0)
        # Inverse: sky → pix
        x_back, y_back = wcs_sip.all_world2pix(ra, dec, 0)

        np.testing.assert_allclose(x, x_back, atol=0.01,
                                   err_msg="Forward-inverse round-trip failed in x")
        np.testing.assert_allclose(y, y_back, atol=0.01,
                                   err_msg="Forward-inverse round-trip failed in y")

    def test_zpn_only_round_trip(self, zpn_wcs):
        """ZPN-only WCS round-trips through pix2world/world2pix."""
        x, y, sky = _generate_stars(zpn_wcs, n=50)
        xy = np.column_stack([x, y])

        wcs_fit, _ = fit_zpn_wcs_from_points(xy, sky, zpn_wcs, pv_deg=3)

        ra, dec = wcs_fit.all_pix2world(x, y, 0)
        x_back, y_back = wcs_fit.all_world2pix(ra, dec, 0)

        np.testing.assert_allclose(x, x_back, atol=0.001)
        np.testing.assert_allclose(y, y_back, atol=0.001)
