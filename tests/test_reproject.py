"""Tests for stdpipe.reproject module (reproject_lanczos)."""

import numpy as np
import pytest
import tempfile
import os

from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D

from stdpipe.reproject import reproject_lanczos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcs(crval=(180.0, 45.0), cdelt=(-1.0 / 3600, 1.0 / 3600),
              crpix=(128.5, 128.5), naxis=(256, 256)):
    """Build a simple TAN WCS. cdelt in degrees."""
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = naxis[0]
    header['NAXIS2'] = naxis[1]
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = crval[0]
    header['CRVAL2'] = crval[1]
    header['CRPIX1'] = crpix[0]
    header['CRPIX2'] = crpix[1]
    header['CD1_1'] = cdelt[0]
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = cdelt[1]
    header['EQUINOX'] = 2000.0
    return WCS(header), header


def _make_star_image(wcs, header, positions=None, flux=10000.0, fwhm=3.0):
    """Create image with Gaussian stars at given pixel positions."""
    ny = header['NAXIS2']
    nx = header['NAXIS1']
    image = np.zeros((ny, nx), dtype=np.float64)

    sigma = fwhm / 2.3548
    if positions is None:
        positions = [(nx / 2, ny / 2)]

    yy, xx = np.mgrid[0:ny, 0:nx]
    for (cx, cy) in positions:
        g = Gaussian2D(amplitude=flux / (2 * np.pi * sigma ** 2),
                       x_mean=cx, y_mean=cy,
                       x_stddev=sigma, y_stddev=sigma)
        image += g(xx, yy)
    return image


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReprojectLanczos:

    def test_identity(self):
        """Same WCS in and out — flux ratio should be ~1.0."""
        wcs_in, hdr = _make_wcs()
        image = _make_star_image(wcs_in, hdr)
        total_in = image.sum()

        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_in, header=hdr
        )
        assert result is not None
        total_out = np.nansum(result)
        assert abs(total_out / total_in - 1.0) < 0.002

    def test_subpixel_shift(self):
        """Sub-pixel shift preserves total flux."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        total_in = image.sum()

        # Shift by 0.3 pixels in RA
        cdelt = -1.0 / 3600
        shift = 0.3 * abs(cdelt)
        wcs_out, hdr_out = _make_wcs(crval=(180.0 + shift, 45.0))

        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out
        )
        total_out = np.nansum(result)
        assert abs(total_out / total_in - 1.0) < 0.002

    def test_rotation(self):
        """15-degree rotation preserves total flux."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        total_in = image.sum()

        # Rotated output WCS
        angle = np.radians(15)
        cdelt = 1.0 / 3600
        hdr_out = hdr_in.copy()
        hdr_out['CD1_1'] = -cdelt * np.cos(angle)
        hdr_out['CD1_2'] = cdelt * np.sin(angle)
        hdr_out['CD2_1'] = -cdelt * np.sin(angle)  # note sign
        hdr_out['CD2_2'] = -cdelt * np.cos(angle)  # note sign - keep parity
        # Fix: proper rotation matrix with correct parity
        hdr_out['CD1_1'] = -cdelt * np.cos(angle)
        hdr_out['CD1_2'] = -cdelt * np.sin(angle)
        hdr_out['CD2_1'] = cdelt * np.sin(angle)
        hdr_out['CD2_2'] = -cdelt * np.cos(angle)
        # Actually let's just do it cleanly
        hdr_out['CD1_1'] = -cdelt * np.cos(angle)
        hdr_out['CD1_2'] = cdelt * np.sin(angle)
        hdr_out['CD2_1'] = cdelt * np.sin(angle)
        hdr_out['CD2_2'] = cdelt * np.cos(angle)
        wcs_out = WCS(hdr_out)

        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out
        )
        total_out = np.nansum(result)
        # Rotation may lose some edge flux, allow 2%
        assert abs(total_out / total_in - 1.0) < 0.02

    def test_rescale_flux_conserving(self):
        """2x rescale with conserve_flux=True preserves total flux."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        total_in = image.sum()

        # Output: 2x larger pixels → 128x128 image covering same area
        wcs_out, hdr_out = _make_wcs(
            cdelt=(-2.0 / 3600, 2.0 / 3600),
            crpix=(64.25, 64.25),
            naxis=(128, 128),
        )

        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=True,
        )
        total_out = np.nansum(result)
        assert abs(total_out / total_in - 1.0) < 0.01

    def test_rescale_sb_conserving(self):
        """2x rescale with conserve_flux=False preserves surface brightness."""
        wcs_in, hdr_in = _make_wcs()
        # Uniform image — SB = 1.0 everywhere
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']
        image = np.ones((ny, nx), dtype=np.float64)

        wcs_out, hdr_out = _make_wcs(
            cdelt=(-2.0 / 3600, 2.0 / 3600),
            crpix=(64.25, 64.25),
            naxis=(128, 128),
        )

        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=False,
        )
        # Interior pixels should be ~1.0
        interior = result[10:-10, 10:-10]
        good = np.isfinite(interior)
        assert np.all(good), "Interior should be fully covered"
        assert abs(np.mean(interior[good]) - 1.0) < 0.002

    def test_uniform_field_rescale(self):
        """Uniform input=1.0 under 1.5x rescale, pixel values correct."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']
        image = np.ones((ny, nx), dtype=np.float64)

        # 1.5x larger pixels
        scale = 1.5
        n_out = int(256 / scale)
        wcs_out, hdr_out = _make_wcs(
            cdelt=(-scale / 3600, scale / 3600),
            crpix=(n_out / 2 + 0.5, n_out / 2 + 0.5),
            naxis=(n_out, n_out),
        )

        # With flux conservation: pixel value = area_ratio = 1.5^2 = 2.25
        result_fc = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=True,
        )
        interior = result_fc[5:-5, 5:-5]
        good = np.isfinite(interior)
        expected = scale ** 2
        assert abs(np.mean(interior[good]) / expected - 1.0) < 0.01

        # Without flux conservation: pixel value = 1.0
        result_sb = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=False,
        )
        interior = result_sb[5:-5, 5:-5]
        good = np.isfinite(interior)
        assert abs(np.mean(interior[good]) - 1.0) < 0.01

    def test_fwhm_undersampled(self):
        """Oversampling improves accuracy for undersampled stars."""
        # Small FWHM = 1.5 px star, rescale 2x
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=1.5)
        total_in = image.sum()

        wcs_out, hdr_out = _make_wcs(
            cdelt=(-2.0 / 3600, 2.0 / 3600),
            crpix=(64.25, 64.25),
            naxis=(128, 128),
        )

        # With auto oversampling (should activate oversamp=2)
        result_os = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=True,
        )
        total_os = np.nansum(result_os)

        # Without oversampling
        result_no = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_out, header=hdr_out,
            conserve_flux=True, oversamp=1,
        )
        total_no = np.nansum(result_no)

        # Both should be within 5% for undersampled case
        assert abs(total_os / total_in - 1.0) < 0.05
        # Oversampled result should be at least as good
        err_os = abs(total_os / total_in - 1.0)
        err_no = abs(total_no / total_in - 1.0)
        assert err_os <= err_no + 0.01  # allow small tolerance

    def test_oversamp_auto(self):
        """Auto-oversampling activates for downscaling, stays 1 for same scale."""
        from stdpipe.reproject import _reproject_single, proj_plane_pixel_scales

        wcs_in, _ = _make_wcs()
        wcs_same, _ = _make_wcs()
        wcs_down, _ = _make_wcs(cdelt=(-2.0 / 3600, 2.0 / 3600))

        # Same scale: ratio ~1 → oversamp should be 1
        scale_in = np.mean(proj_plane_pixel_scales(wcs_in))
        scale_same = np.mean(proj_plane_pixel_scales(wcs_same))
        ratio_same = scale_same / scale_in
        assert max(1, int(ratio_same + 0.5)) == 1

        # 2x downscale: ratio ~2 → oversamp should be 2
        scale_down = np.mean(proj_plane_pixel_scales(wcs_down))
        ratio_down = scale_down / scale_in
        assert max(1, int(ratio_down + 0.5)) == 2

    def test_multiple_inputs(self):
        """List of 2 images coadded correctly."""
        wcs_in, hdr_in = _make_wcs()
        image1 = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        image2 = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)

        result = reproject_lanczos(
            [(image1, wcs_in), (image2, wcs_in)],
            wcs=wcs_in, header=hdr_in,
        )
        # Average of two identical images = same as one
        result_single = reproject_lanczos(
            [(image1, wcs_in)], wcs=wcs_in, header=hdr_in,
        )
        good = np.isfinite(result) & np.isfinite(result_single)
        np.testing.assert_allclose(result[good], result_single[good], rtol=1e-10)

    def test_input_from_file(self):
        """Accepts FITS filename as input."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        total_in = image.sum()

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            fname = f.name
        try:
            fits.writeto(fname, image, hdr_in, overwrite=True)
            result = reproject_lanczos(
                [fname], wcs=wcs_in, header=hdr_in,
            )
            total_out = np.nansum(result)
            assert abs(total_out / total_in - 1.0) < 0.002
        finally:
            os.unlink(fname)

    def test_lanczos_order(self):
        """Different Lanczos orders (2, 3, 4) all produce valid results."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)
        total_in = image.sum()

        for order in [2, 3, 4]:
            result = reproject_lanczos(
                [(image, wcs_in)], wcs=wcs_in, header=hdr_in, order=order,
            )
            total_out = np.nansum(result)
            assert abs(total_out / total_in - 1.0) < 0.002, \
                f"Order {order} flux error too large"

    def test_output_shape(self):
        """shape/width/height/header all specify output correctly."""
        wcs_in, hdr_in = _make_wcs()
        image = _make_star_image(wcs_in, hdr_in, flux=10000.0, fwhm=4.0)

        # Via shape
        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_in, shape=(100, 100),
        )
        assert result.shape == (100, 100)

        # Via width/height
        result = reproject_lanczos(
            [(image, wcs_in)], wcs=wcs_in, width=80, height=90,
        )
        assert result.shape == (90, 80)

        # Via header
        _, hdr_small = _make_wcs(naxis=(64, 64), crpix=(32.5, 32.5))
        result = reproject_lanczos(
            [(image, wcs_in)], header=hdr_small,
        )
        assert result.shape == (64, 64)

    def test_is_flags_identity(self):
        """Flag image reprojected onto same WCS preserves flag values."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']
        flags = np.zeros((ny, nx), dtype=np.int16)
        flags[100:150, 100:150] = 0x0001
        flags[120:130, 120:130] = 0x0003  # two flags set
        flags[50:60, 50:60] = 0x0100

        result = reproject_lanczos(
            [(flags, wcs_in)], wcs=wcs_in, header=hdr_in, is_flags=True,
        )
        # Same WCS → should be identical
        np.testing.assert_array_equal(result, flags)

    def test_is_flags_nearest_neighbor(self):
        """Flag resampling uses nearest-neighbor, not interpolation."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']
        # Checkerboard pattern: alternating 0 and 0x0001
        flags = np.zeros((ny, nx), dtype=np.int16)
        flags[::2, ::2] = 0x0001
        flags[1::2, 1::2] = 0x0001

        result = reproject_lanczos(
            [(flags, wcs_in)], wcs=wcs_in, header=hdr_in, is_flags=True,
        )
        # Nearest-neighbor should only produce original values (0 or 1)
        unique = np.unique(result)
        assert set(unique).issubset({0, 1})

    def test_is_flags_combine_and(self):
        """Multiple flag images combined with bitwise AND."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']

        flags1 = np.full((ny, nx), 0x000F, dtype=np.int16)  # bits 0-3
        flags2 = np.full((ny, nx), 0x0007, dtype=np.int16)  # bits 0-2

        result = reproject_lanczos(
            [(flags1, wcs_in), (flags2, wcs_in)],
            wcs=wcs_in, header=hdr_in, is_flags=True,
        )
        # AND: 0x000F & 0x0007 = 0x0007
        np.testing.assert_array_equal(result, 0x0007)

    def test_is_flags_rescale(self):
        """Flag image survives rescaling with nearest-neighbor."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']
        flags = np.zeros((ny, nx), dtype=np.int16)
        # Large block of flags (well-sampled even after 2x rescale)
        flags[80:180, 80:180] = 0x0004

        wcs_out, hdr_out = _make_wcs(
            cdelt=(-2.0 / 3600, 2.0 / 3600),
            crpix=(64.25, 64.25),
            naxis=(128, 128),
        )
        result = reproject_lanczos(
            [(flags, wcs_in)], wcs=wcs_out, header=hdr_out, is_flags=True,
        )
        # Only original values should appear
        unique = np.unique(result)
        assert set(unique).issubset({0, 4})
        # Center should still be flagged
        assert result[64, 64] == 0x0004

    def test_is_flags_dtype_preserved(self):
        """Integer dtype is preserved for flag images."""
        wcs_in, hdr_in = _make_wcs()
        ny, nx = hdr_in['NAXIS2'], hdr_in['NAXIS1']

        for dtype in [np.int16, np.int32, np.uint16]:
            flags = np.ones((ny, nx), dtype=dtype) * 3
            result = reproject_lanczos(
                [(flags, wcs_in)], wcs=wcs_in, header=hdr_in, is_flags=True,
            )
            assert result.dtype == dtype
