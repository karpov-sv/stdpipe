"""
Unit tests for stdpipe.astrometry module.

Tests coordinate transformations, WCS operations, and astrometric utilities.
"""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

from stdpipe import astrometry


class TestCoordinateTransformations:
    """Test coordinate transformation utilities."""

    @pytest.mark.unit
    def test_spherical_distance(self):
        """Test spherical distance calculation."""
        # Distance between two points at the equator, 1 degree apart
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = 1.0, 0.0

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # Should be close to 1 degree
        assert np.abs(dist - 1.0) < 0.001

    @pytest.mark.unit
    def test_spherical_distance_pole(self):
        """Test spherical distance at the pole."""
        # All RAs should give same distance from pole
        ra1, dec1 = 0.0, 90.0
        ra2, dec2 = 180.0, 89.0

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # Should be 1 degree
        assert np.abs(dist - 1.0) < 0.001

    @pytest.mark.unit
    def test_spherical_distance_same_point(self):
        """Test spherical distance for identical points."""
        ra, dec = 123.456, 45.678

        dist = astrometry.spherical_distance(ra, dec, ra, dec)

        # Should be zero
        assert dist < 1e-10

    @pytest.mark.unit
    def test_radectoxyz_xyztoradec_roundtrip(self):
        """Test RA/Dec to XYZ conversion roundtrip."""
        ra_orig = 180.0
        dec_orig = 45.0

        # Convert to XYZ
        xyz = astrometry.radectoxyz(ra_orig, dec_orig)

        # Check unit vector
        assert np.abs(np.linalg.norm(xyz) - 1.0) < 1e-10

        # Convert back to RA/Dec
        ra_back, dec_back = astrometry.xyztoradec(xyz)

        # Should match original
        assert np.abs(ra_back - ra_orig) < 1e-10
        assert np.abs(dec_back - dec_orig) < 1e-10

    @pytest.mark.unit
    def test_radectoxyz_array(self):
        """Test RA/Dec to XYZ conversion with arrays."""
        ra = np.array([0.0, 90.0, 180.0])
        dec = np.array([0.0, 0.0, 0.0])

        xyz = astrometry.radectoxyz(ra, dec)

        # Check shape
        assert xyz.shape == (3, 3)

        # Check that all are unit vectors
        norms = np.linalg.norm(xyz, axis=0)
        assert np.allclose(norms, 1.0)


class TestWCSUtilities:
    """Test WCS-related utilities."""

    @pytest.mark.unit
    def test_get_pixscale_from_wcs(self, simple_wcs):
        """Test pixel scale extraction from WCS."""
        pixscale = astrometry.get_pixscale(wcs=simple_wcs)

        # Should be ~1 arcsec/pixel = 0.0002778 deg/pixel
        expected = 0.0002778
        assert np.abs(pixscale - expected) < 1e-6

    @pytest.mark.unit
    def test_get_pixscale_from_header(self, header_with_wcs):
        """Test pixel scale extraction from header."""
        pixscale = astrometry.get_pixscale(header=header_with_wcs)

        # Should be ~1 arcsec/pixel
        expected = 0.0002778
        assert np.abs(pixscale - expected) < 1e-6

    @pytest.mark.unit
    def test_get_frame_center(self, header_with_wcs):
        """Test frame center calculation."""
        ra, dec, sr = astrometry.get_frame_center(header=header_with_wcs)

        # Center should be at CRVAL
        assert np.abs(ra - 180.0) < 0.1
        assert np.abs(dec - 45.0) < 0.1

        # Radius should be positive
        assert sr > 0

    @pytest.mark.unit
    def test_get_frame_center_with_wcs(self, simple_wcs):
        """Test frame center with WCS and explicit dimensions."""
        ra, dec, sr = astrometry.get_frame_center(
            wcs=simple_wcs,
            width=256,
            height=256
        )

        # Center should be near CRVAL
        assert np.abs(ra - 180.0) < 0.1
        assert np.abs(dec - 45.0) < 0.1

    @pytest.mark.unit
    def test_get_frame_center_no_wcs(self):
        """Test frame center with no WCS returns None."""
        header = fits.Header()
        header['NAXIS1'] = 100
        header['NAXIS2'] = 100

        ra, dec, sr = astrometry.get_frame_center(header=header)

        assert ra is None
        assert dec is None
        assert sr is None


class TestAstrometricMatching:
    """Test astrometric matching and catalog crossmatching."""

    @pytest.mark.unit
    def test_spherical_distance_vectorized(self):
        """Test vectorized spherical distance calculation."""
        # Create a grid of points
        ra1 = np.array([0.0, 1.0, 2.0])
        dec1 = np.array([0.0, 0.0, 0.0])
        ra2 = np.array([0.5, 1.5, 2.5])
        dec2 = np.array([0.0, 0.0, 0.0])

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # All distances should be ~0.5 degrees
        assert dist.shape == (3,)
        assert np.allclose(dist, 0.5, atol=0.01)


# ============================================================================
# Astrometric residual-field correction
# ============================================================================


class TestAstrometricResiduals:
    """Tests for ``fit_astrometric_residuals`` and
    ``refine_positions_from_catalog`` in :mod:`stdpipe.astrometry`."""

    def _smooth_residual_samples(self, N=2000, W=1024, H=1024,
                                  noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        x_pred = rng.uniform(0, W, N)
        y_pred = rng.uniform(0, H, N)
        xn = (x_pred - W / 2) / (W / 2)
        yn = (y_pred - H / 2) / (H / 2)
        true_dx = 0.4 * xn + 0.2 * yn ** 2
        true_dy = 0.3 * yn - 0.25 * xn * yn
        x_obs = x_pred + true_dx + rng.normal(0, noise, N)
        y_obs = y_pred + true_dy + rng.normal(0, noise, N)
        return x_obs, y_obs, x_pred, y_pred, true_dx, true_dy, (H, W)

    @pytest.mark.unit
    @pytest.mark.parametrize("backend", ["grid", "loess"])
    def test_fit_astrometric_residuals_recovers_smooth_field(self, backend):
        from stdpipe.astrometry import fit_astrometric_residuals

        x_obs, y_obs, x_pred, y_pred, true_dx, true_dy, image_shape = \
            self._smooth_residual_samples()

        if backend == "grid":
            correct = fit_astrometric_residuals(
                x_obs, y_obs, x_pred, y_pred,
                backend="grid", image_shape=image_shape,
                grid_shape=(16, 12), min_per_cell=4, smooth_sigma=1.0,
            )
        else:
            correct = fit_astrometric_residuals(
                x_obs, y_obs, x_pred, y_pred,
                backend="loess",
                scales=(image_shape[1] / 12.0, image_shape[0] / 8.0),
                k=120, robust_iters=0,
            )

        x_corr, y_corr = correct(x_pred, y_pred)
        # The corrected predicted positions should land near the observed
        # ones to within the noise level.
        assert np.std(x_corr - x_obs) < 0.04
        assert np.std(y_corr - y_obs) < 0.04
        # And the smoother model should track the noiseless truth.
        cdx, cdy = correct.smoother(x_pred, y_pred)
        assert np.std(cdx - true_dx) < 0.04
        assert np.std(cdy - true_dy) < 0.04

    @pytest.mark.unit
    def test_refine_positions_from_catalog_returns_info(
        self, header_with_wcs, simple_wcs,
    ):
        """End-to-end on a synthetic match: build obj/cat with a known
        smooth pixel-space distortion, run a position match outside, then
        check the function reduces the in-sample residuals."""
        from astropy.table import Table
        from stdpipe.astrometry import refine_positions_from_catalog

        rng = np.random.default_rng(1)
        wcs = simple_wcs
        H = header_with_wcs['NAXIS2']; W = header_with_wcs['NAXIS1']
        N = 400
        x_pred = rng.uniform(20, W - 20, N)
        y_pred = rng.uniform(20, H - 20, N)
        ra_cat, dec_cat = wcs.all_pix2world(x_pred, y_pred, 0)
        # Smooth pixel-space distortion + noise: this is what obj will
        # see relative to the catalogue projection.
        xn = (x_pred - W / 2) / (W / 2); yn = (y_pred - H / 2) / (H / 2)
        true_dx = 0.5 * xn + 0.3 * yn
        true_dy = 0.4 * yn - 0.2 * xn * yn
        x_obs = x_pred + true_dx + rng.normal(0, 0.02, N)
        y_obs = y_pred + true_dy + rng.normal(0, 0.02, N)

        obj = Table({'x': x_obs, 'y': y_obs})
        cat = Table({'ra': ra_cat, 'dec': dec_cat})
        match = {
            'idx': np.ones(N, dtype=bool),
            'oidx': np.arange(N),
            'cidx': np.arange(N),
        }
        correct, info = refine_positions_from_catalog(
            match, obj, cat, wcs,
            image_shape=(H, W), grid_shape=(8, 8),
            min_per_cell=4, smooth_sigma=1.0,
        )

        assert info['n_matched'] == N
        assert info['n_used'] == N
        # The correction should bring the mean residual well below the raw level.
        assert info['corrected_median_dr_pix'] < 0.5 * info['raw_median_dr_pix']
        # And `correct` should evaluate the field anywhere.
        xq = np.array([100.0, 200.0]); yq = np.array([100.0, 200.0])
        cx, cy = correct(xq, yq)
        assert cx.shape == (2,) and cy.shape == (2,)


class TestRefineAstrometryResidualField:
    """``pipeline.refine_astrometry`` ``refine_residual_field=True`` path."""

    @pytest.mark.unit
    def test_apply_residual_field_correction_updates_obj_xy(
        self, header_with_wcs, simple_wcs,
    ):
        """The ``_apply_residual_field_correction`` helper subtracts the
        smooth correction from ``obj['x']``/``['y']`` in place when
        ``update=True`` and shrinks the catalogue residual."""
        from astropy.table import Table
        from stdpipe.pipeline import _apply_residual_field_correction

        rng = np.random.default_rng(2)
        wcs = simple_wcs
        H = header_with_wcs['NAXIS2']; W = header_with_wcs['NAXIS1']
        N = 400
        x_pred = rng.uniform(20, W - 20, N)
        y_pred = rng.uniform(20, H - 20, N)
        ra_cat, dec_cat = wcs.all_pix2world(x_pred, y_pred, 0)
        xn = (x_pred - W / 2) / (W / 2); yn = (y_pred - H / 2) / (H / 2)
        true_dx = 0.5 * xn + 0.3 * yn
        true_dy = 0.4 * yn - 0.2 * xn * yn
        x_obs = x_pred + true_dx + rng.normal(0, 0.02, N)
        y_obs = y_pred + true_dy + rng.normal(0, 0.02, N)

        obj = Table({
            'x': x_obs.copy(), 'y': y_obs.copy(),
            'ra': ra_cat.copy(), 'dec': dec_cat.copy(),
        })
        cat = Table({'ra': ra_cat, 'dec': dec_cat})

        log_msgs = []
        _apply_residual_field_correction(
            obj, cat, wcs, sr=10 / 3600,
            cat_col_ra='ra', cat_col_dec='dec',
            update=True,
            residual_field_kwargs=dict(
                image_shape=(H, W), grid_shape=(8, 8),
                min_per_cell=4, smooth_sigma=1.0,
            ),
            log=log_msgs.append,
        )

        # x/y must have moved.
        assert not np.allclose(obj['x'], x_obs)
        assert not np.allclose(obj['y'], y_obs)
        # And they must lie closer to x_pred than the originals did.
        before = np.median(np.hypot(x_obs - x_pred, y_obs - y_pred))
        after = np.median(np.hypot(np.asarray(obj['x']) - x_pred,
                                    np.asarray(obj['y']) - y_pred))
        assert after < 0.3 * before
        # ra/dec are re-derived from the new x/y.
        ra_new, dec_new = wcs.all_pix2world(obj['x'], obj['y'], 0)
        np.testing.assert_allclose(obj['ra'], ra_new)
        np.testing.assert_allclose(obj['dec'], dec_new)
        assert any('Residual field' in m for m in log_msgs)

    @pytest.mark.unit
    def test_apply_residual_field_skips_with_too_few_matches(
        self, header_with_wcs, simple_wcs,
    ):
        """Skip message + no-op when fewer than 50 matches survive."""
        from astropy.table import Table
        from stdpipe.pipeline import _apply_residual_field_correction

        wcs = simple_wcs
        N = 5
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 256, N); y = rng.uniform(0, 256, N)
        obj = Table({'x': x.copy(), 'y': y.copy()})
        cat = Table({'ra': np.array([0.0]), 'dec': np.array([-89.0])})
        log_msgs = []
        _apply_residual_field_correction(
            obj, cat, wcs, sr=1 / 3600,
            cat_col_ra='ra', cat_col_dec='dec',
            update=True,
            residual_field_kwargs={},
            log=log_msgs.append,
        )
        np.testing.assert_array_equal(obj['x'], x)
        np.testing.assert_array_equal(obj['y'], y)
        assert any('skipped' in m for m in log_msgs)


# ============================================================================
# Integration tests requiring external tools
# ============================================================================

class TestAstrometryNetIntegration:
    """Integration tests for Astrometry.Net wrapper."""

    @pytest.mark.integration
    @pytest.mark.requires_astrometry_net
    @pytest.mark.slow
    def test_astrometry_net_basic(self, image_with_sources, temp_dir):
        """Test basic Astrometry.Net plate solving."""
        # This would require actual astrometry.net to be installed
        # and configured, so marking as integration test
        pytest.skip("Requires full astrometry.net setup")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
