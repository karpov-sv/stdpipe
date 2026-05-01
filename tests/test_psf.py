"""Tests for PSF helper utilities in :mod:`stdpipe.psf`."""

import numpy as np
import pytest
from astropy.table import Table

from stdpipe.psf import enclosed_psf_fraction, select_psf_seeds


def _gaussian_psf(sigma=2.0, size=51):
    """Build a position-invariant PSFEx-style dict holding a unit-flux,
    centred Gaussian. ``sampling=1`` so 1 supersampled pixel == 1 image
    pixel, which keeps the analytic comparison clean."""
    half = size // 2
    y, x = np.mgrid[0:size, 0:size]
    g = np.exp(-((x - half) ** 2 + (y - half) ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return {
        'data': g[None, :, :],
        'width': size,
        'height': size,
        'sampling': 1.0,
        'degree': 0,
        'ncoeffs': 1,
        'x0': 0.0, 'y0': 0.0,
        'sx': 1.0, 'sy': 1.0,
        'type': 'epsf',
    }


# ---------------------------------------------------------------------------
# enclosed_psf_fraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnclosedPsfFraction:
    def test_matches_analytic_gaussian(self):
        sigma = 2.0
        psf = _gaussian_psf(sigma=sigma, size=61)
        radii = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        out = enclosed_psf_fraction(psf, x=0.0, y=0.0, radius=radii, subpixel=10)
        analytic = 1.0 - np.exp(-(radii ** 2) / (2 * sigma ** 2))
        # Sub-pixel weighting should land within ~1% of the analytic curve.
        assert np.all(np.abs(out - analytic) < 0.01)

    def test_scalar_radius_returns_scalar(self):
        psf = _gaussian_psf(sigma=2.0)
        out = enclosed_psf_fraction(psf, radius=3.0)
        assert isinstance(out, float)

    def test_array_radius_returns_array_same_shape(self):
        psf = _gaussian_psf(sigma=2.0)
        out = enclosed_psf_fraction(psf, radius=[1.0, 2.0, 4.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert np.all(np.diff(out) > 0)  # monotone increasing in r

    def test_large_radius_captures_all_flux(self):
        psf = _gaussian_psf(sigma=2.0, size=51)
        out = enclosed_psf_fraction(psf, radius=20.0)
        # Stamp is normalised; aperture covers the whole stamp, so ~1.
        assert out > 0.999

    def test_subpixel_one_falls_back_to_pixel_centre_mask(self):
        psf = _gaussian_psf(sigma=2.0, size=51)
        # Hard-mask result quantises in pixel-count steps and can drift
        # from the analytic curve by several percent.
        sub10 = enclosed_psf_fraction(psf, radius=2.7, subpixel=10)
        sub1 = enclosed_psf_fraction(psf, radius=2.7, subpixel=1)
        analytic = 1.0 - np.exp(-(2.7 ** 2) / (2 * 2.0 ** 2))
        assert abs(sub10 - analytic) < abs(sub1 - analytic)

    def test_radius_required(self):
        psf = _gaussian_psf()
        with pytest.raises(TypeError, match="radius is required"):
            enclosed_psf_fraction(psf)


# ---------------------------------------------------------------------------
# select_psf_seeds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectPsfSeeds:
    def _make_obj(self, n=400, seed=0, image_shape=(256, 256)):
        rng = np.random.default_rng(seed)
        H, W = image_shape
        t = Table()
        t['x'] = rng.uniform(0, W, n)
        t['y'] = rng.uniform(0, H, n)
        t['flux'] = rng.uniform(100, 10000, n)
        t['flags'] = np.zeros(n, dtype=int)
        return t

    def test_grid_cell_capacity_is_respected(self):
        obj = self._make_obj(n=2000, image_shape=(256, 256))
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), grid=4, max_per_tile=3, edge=10,
        )
        # 4×4 grid × 3 per tile = 48 maximum.
        assert len(seeds) <= 16 * 3
        # Per-cell capacity check.
        gx = (np.asarray(seeds['x'], float) / 256 * 4).astype(int)
        gy = (np.asarray(seeds['y'], float) / 256 * 4).astype(int)
        for ix in range(4):
            for iy in range(4):
                assert int(np.sum((gx == ix) & (gy == iy))) <= 3

    def test_picks_brightest_within_each_cell(self):
        # Place 5 sources in one cell; max_per_tile=2 should keep the two brightest.
        obj = Table()
        obj['x'] = np.full(5, 30.0)        # all in cell (0, 0) of a 4×4 grid on 256×256
        obj['y'] = np.full(5, 30.0)
        obj['flux'] = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        obj['flags'] = np.zeros(5, dtype=int)
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), grid=4, max_per_tile=2, edge=10,
        )
        kept_flux = sorted(np.asarray(seeds['flux'], float).tolist())
        assert kept_flux == [5.0, 8.0]

    def test_edge_filter_drops_border_sources(self):
        obj = Table()
        obj['x'] = np.array([5.0, 50.0, 250.0])
        obj['y'] = np.array([50.0, 50.0, 50.0])
        obj['flux'] = np.ones(3)
        obj['flags'] = np.zeros(3, dtype=int)
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), edge=20, grid=2, max_per_tile=10,
        )
        kept_x = sorted(np.asarray(seeds['x'], float).tolist())
        assert kept_x == [50.0]   # 5 and 250 are within the edge band

    def test_flag_filter_drops_flagged_sources(self):
        obj = Table()
        obj['x'] = np.array([50.0, 50.0])
        obj['y'] = np.array([50.0, 60.0])
        obj['flux'] = np.array([5.0, 10.0])
        obj['flags'] = np.array([0, 1])
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), edge=10, accept_flags=0,
        )
        assert len(seeds) == 1
        assert float(seeds['flux'][0]) == 5.0

    def test_accept_flags_passes_specific_bits(self):
        obj = Table()
        obj['x'] = np.array([50.0, 50.0])
        obj['y'] = np.array([50.0, 60.0])
        obj['flux'] = np.array([5.0, 10.0])
        # Both flagged with bit 0x1; with accept_flags=0x1, both should pass.
        obj['flags'] = np.array([1, 1])
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), edge=10, accept_flags=1,
        )
        assert len(seeds) == 2

    def test_missing_flag_column_skips_flag_filter(self):
        obj = Table()
        obj['x'] = np.array([50.0, 60.0])
        obj['y'] = np.array([50.0, 60.0])
        obj['flux'] = np.array([5.0, 10.0])
        # No 'flags' column; should silently skip flag filtering.
        seeds = select_psf_seeds(obj, image_shape=(256, 256), edge=10)
        assert len(seeds) == 2

    def test_drops_nan_and_nonpositive_flux(self):
        obj = Table()
        obj['x'] = np.array([50.0, 60.0, 70.0, 80.0])
        obj['y'] = np.array([50.0, 60.0, 70.0, 80.0])
        obj['flux'] = np.array([5.0, np.nan, -1.0, 10.0])
        obj['flags'] = np.zeros(4, dtype=int)
        seeds = select_psf_seeds(obj, image_shape=(256, 256), edge=10)
        kept_flux = sorted(np.asarray(seeds['flux'], float).tolist())
        assert kept_flux == [5.0, 10.0]

    def test_rectangular_grid(self):
        obj = self._make_obj(n=2000, image_shape=(256, 512))
        seeds = select_psf_seeds(
            obj, image_shape=(256, 512), grid=(8, 4), max_per_tile=2, edge=10,
        )
        assert len(seeds) <= 8 * 4 * 2

    def test_custom_column_names(self):
        obj = Table()
        obj['X_IMAGE'] = np.array([50.0, 100.0])
        obj['Y_IMAGE'] = np.array([50.0, 100.0])
        obj['FLUX_AUTO'] = np.array([1.0, 2.0])
        seeds = select_psf_seeds(
            obj, image_shape=(256, 256), edge=10,
            obj_col_x='X_IMAGE', obj_col_y='Y_IMAGE',
            obj_col_flux='FLUX_AUTO', obj_col_flags=None,
        )
        assert len(seeds) == 2
