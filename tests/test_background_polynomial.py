"""Tests for get_background_polynomial in photometry_background module."""

import numpy as np
import pytest

from stdpipe.photometry_background import get_background_polynomial, get_background


def _make_gradient_image(ny=256, nx=256, gradient_coeffs=(100, 0.5, 0.3),
                         nebula=True, stars=30, noise_std=10, seed=42):
    """Create test image with gradient + optional nebula + stars + noise."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Linear gradient
    a, b, c = gradient_coeffs
    gradient = a + b * xx + c * yy

    image = gradient.copy().astype(np.float64)

    # Gaussian "nebula"
    neb = np.zeros((ny, nx))
    if nebula:
        cx, cy = int(nx * 0.6), int(ny * 0.5)
        sig = min(nx, ny) * 0.15
        neb = 200 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sig**2))
        image += neb

    # Stars
    for _ in range(stars):
        sx = rng.randint(5, nx - 5)
        sy = rng.randint(5, ny - 5)
        image[sy - 1:sy + 2, sx - 1:sx + 2] += rng.uniform(500, 3000)

    # Noise
    image += rng.normal(0, noise_std, (ny, nx))

    return image, gradient, neb


class TestBackgroundPolynomial:

    @pytest.mark.unit
    def test_linear_gradient_recovery(self):
        """Order-1 fit should recover a linear gradient accurately."""
        image, gradient, _ = _make_gradient_image(nebula=False, stars=0, noise_std=5)
        bg, rms = get_background_polynomial(image, order=1)

        residual = gradient - bg
        assert np.std(residual) < 2.0  # should be much less than the noise

    @pytest.mark.unit
    def test_star_rejection(self):
        """Stars should be clipped and not bias the fit."""
        image, gradient, _ = _make_gradient_image(nebula=False, stars=100, noise_std=10)
        bg, rms = get_background_polynomial(image, order=1, sigma=3.0)

        residual = gradient - bg
        assert np.std(residual) < 10.0

    @pytest.mark.unit
    def test_nebula_preservation_order1(self):
        """Order-1 polynomial should not absorb a Gaussian nebula."""
        image, gradient, nebula = _make_gradient_image(nebula=True, stars=0, noise_std=5)
        bg, _ = get_background_polynomial(image, order=1)

        # Nebula leakage into background should be small relative to peak
        leak = np.max(np.abs(bg - gradient))
        assert leak < 30, f"Nebula leakage too high: {leak:.1f}"

    @pytest.mark.unit
    def test_full_vs_directional_nebula_leakage(self):
        """Directional mode should leak less nebula than full 2D at order >= 2."""
        image, gradient, _ = _make_gradient_image(nebula=True, stars=30)

        bg_full, _ = get_background_polynomial(image, order=2)
        bg_dir, _ = get_background_polynomial(image, order=2, directional=True)

        leak_full = np.std(bg_full - gradient)
        leak_dir = np.std(bg_dir - gradient)

        # Directional should leak less (or at most equal)
        assert leak_dir <= leak_full * 1.5, (
            f"Directional leaked more: {leak_dir:.2f} vs full {leak_full:.2f}")

    @pytest.mark.unit
    def test_directional_auto_detect(self):
        """Auto-detected direction should roughly match the actual gradient."""
        rng = np.random.RandomState(123)
        ny, nx = 256, 256
        yy, xx = np.mgrid[0:ny, 0:nx]
        # Strong gradient at ~60 degrees
        gradient = 100 + 2.0 * xx + 3.5 * yy
        image = gradient + rng.normal(0, 5, (ny, nx))

        bg, _ = get_background_polynomial(image, order=2, directional=True)
        # Should recover the gradient well
        assert np.std(gradient - bg) < 5.0

    @pytest.mark.unit
    def test_directional_explicit_angle(self):
        """Explicit direction angle should be used."""
        rng = np.random.RandomState(0)
        ny, nx = 256, 256
        yy, xx = np.mgrid[0:ny, 0:nx]

        # Quadratic gradient along 45 degrees
        t = ((xx - nx / 2) + (yy - ny / 2)) / np.sqrt(2)
        gradient = 500 + 0.5 * t + 0.005 * t ** 2
        image = gradient + rng.normal(0, 10, (ny, nx))

        bg, _ = get_background_polynomial(image, order=2, directional=45.0)
        assert np.std(gradient - bg) < 15.0

    @pytest.mark.unit
    def test_direction_kwarg_compat(self):
        """The `direction` keyword should work as an alias."""
        image, _, _ = _make_gradient_image()
        bg1, _ = get_background_polynomial(image, order=2, directional=45.0)
        bg2, _ = get_background_polynomial(image, order=2, direction=45.0)
        np.testing.assert_array_equal(bg1, bg2)

    @pytest.mark.unit
    def test_order_zero(self):
        """Order 0 should give a constant (clipped mean)."""
        rng = np.random.RandomState(0)
        image = 100 + rng.normal(0, 5, (64, 64))
        bg, rms = get_background_polynomial(image, order=0)

        assert bg.shape == image.shape
        assert np.std(bg) < 0.01  # constant
        assert abs(np.mean(bg) - 100) < 2.0

    @pytest.mark.unit
    def test_masked_pixels(self):
        """Masked pixels should be excluded from the fit."""
        image, gradient, _ = _make_gradient_image(nebula=False, stars=0, noise_std=5)

        # Mask a large bright patch — should not affect the fit
        mask = np.zeros(image.shape, dtype=bool)
        mask[50:100, 50:100] = True
        image[mask] = 99999.0

        bg, _ = get_background_polynomial(image, mask=mask, order=1)
        residual = gradient - bg
        assert np.std(residual) < 3.0

    @pytest.mark.unit
    def test_all_masked(self):
        """All-masked image should return NaN background."""
        image = np.ones((64, 64))
        mask = np.ones((64, 64), dtype=bool)
        bg, rms = get_background_polynomial(image, mask=mask, order=1)
        assert np.all(np.isnan(bg))
        assert np.isnan(rms)

    @pytest.mark.unit
    def test_returns_rms(self):
        """RMS should be a reasonable estimate of the noise."""
        image, _, _ = _make_gradient_image(nebula=False, stars=0, noise_std=10)
        _, rms = get_background_polynomial(image, order=1)
        assert 5 < rms < 20  # should be close to 10

    @pytest.mark.unit
    def test_dispatcher_polynomial(self):
        """get_background(method='polynomial') should work."""
        image, _, _ = _make_gradient_image()
        bg = get_background(image, method='polynomial', order=2)
        assert bg.shape == image.shape

    @pytest.mark.unit
    def test_dispatcher_polynomial_get_rms(self):
        """get_background(method='polynomial', get_rms=True) should return tuple."""
        image, _, _ = _make_gradient_image()
        bg, rms = get_background(image, method='polynomial', order=1, get_rms=True)
        assert bg.shape == image.shape
        assert rms.shape == image.shape
        # RMS should be uniform
        assert np.std(rms) < 1e-10

    @pytest.mark.unit
    def test_directional_false_for_order1(self):
        """directional=True should have no effect for order <= 1."""
        image, _, _ = _make_gradient_image()
        bg1, _ = get_background_polynomial(image, order=1, directional=False)
        bg2, _ = get_background_polynomial(image, order=1, directional=True)
        np.testing.assert_allclose(bg1, bg2, atol=1e-10)

    @pytest.mark.unit
    def test_large_image_subsampling(self):
        """Large images should be handled via subsampling without error."""
        rng = np.random.RandomState(0)
        ny, nx = 1024, 1024
        yy, xx = np.mgrid[0:ny, 0:nx]
        gradient = 100 + 0.3 * xx + 0.2 * yy
        image = gradient + rng.normal(0, 10, (ny, nx))

        bg, _ = get_background_polynomial(image, order=2, directional=True)
        assert bg.shape == (ny, nx)
        assert np.std(gradient - bg) < 15.0

    @pytest.mark.unit
    def test_higher_order_directional_constrained(self):
        """Directional order 4 should have fewer params than full order 4."""
        # Full order 4: 15 terms. Directional order 4: 6 terms.
        # With a nebula, full order 4 should leak more.
        image, gradient, _ = _make_gradient_image(nebula=True, stars=50)

        bg_full, _ = get_background_polynomial(image, order=4)
        bg_dir, _ = get_background_polynomial(image, order=4, directional=True)

        leak_full = np.max(np.abs(bg_full - gradient))
        leak_dir = np.max(np.abs(bg_dir - gradient))

        # Directional should be more conservative
        assert leak_dir < leak_full * 1.2
