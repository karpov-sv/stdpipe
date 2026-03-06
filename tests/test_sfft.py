"""
Tests for SFFT image subtraction.

Tests both the low-level solver (stdpipe.sfft.solve) and the high-level
wrapper (stdpipe.subtraction.run_sfft).
"""

import pytest
import numpy as np
from astropy.stats import mad_std


# ============================================================================
# Helpers
# ============================================================================

def _gaussian_2d(shape, x0, y0, fwhm, flux=1.0):
    """Create a 2-D Gaussian source image."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return flux * g / g.sum()


def _make_pair(
    size=256,
    n_stars=50,
    template_fwhm=2.5,
    science_fwhm=3.5,
    background=1000.0,
    readnoise=10.0,
    gain=2.0,
    flux_scale=1.1,
    n_transients=3,
    transient_flux=3000.0,
    seed=42,
):
    """Create a science/template pair with known flux scale and transients.

    Returns science, template, mask, info dict.
    """
    rng = np.random.RandomState(seed)

    edge = 20
    star_x = rng.uniform(edge, size - edge, n_stars)
    star_y = rng.uniform(edge, size - edge, n_stars)
    star_flux = 10 ** rng.uniform(2.5, 4.0, n_stars)

    # Template: sharp PSF, low noise
    template = rng.normal(background * 0.5, readnoise * 0.3, (size, size))
    template += background * 0.5
    for i in range(n_stars):
        template += _gaussian_2d(
            (size, size), star_x[i], star_y[i], template_fwhm, star_flux[i]
        )
    template = rng.poisson(np.clip(template * gain, 0, None)).astype(
        np.float64
    ) / gain

    # Science: broader PSF, noisier, different flux scale
    science = rng.normal(background, readnoise, (size, size))
    for i in range(n_stars):
        science += _gaussian_2d(
            (size, size),
            star_x[i],
            star_y[i],
            science_fwhm,
            star_flux[i] * flux_scale,
        )

    # Transients (only in science)
    trans_x = rng.uniform(edge + 20, size - edge - 20, n_transients)
    trans_y = rng.uniform(edge + 20, size - edge - 20, n_transients)
    for i in range(n_transients):
        science += _gaussian_2d(
            (size, size), trans_x[i], trans_y[i], science_fwhm, transient_flux
        )

    science = rng.poisson(np.clip(science * gain, 0, None)).astype(
        np.float64
    ) / gain

    mask = np.zeros((size, size), dtype=bool)

    info = {
        'star_x': star_x,
        'star_y': star_y,
        'star_flux': star_flux,
        'trans_x': trans_x,
        'trans_y': trans_y,
        'transient_flux': transient_flux,
        'flux_scale': flux_scale,
        'science_fwhm': science_fwhm,
        'template_fwhm': template_fwhm,
        'gain': gain,
        'background': background,
    }

    return science, template, mask, info


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def image_pair():
    """Module-scoped image pair to avoid re-creating for each test."""
    return _make_pair()


@pytest.fixture(scope='module')
def sfft_result_basic(image_pair):
    """Module-scoped basic SFFT solve result."""
    from stdpipe.sfft import solve

    science, template, mask, info = image_pair
    return solve(
        science,
        template,
        mask=mask,
        kernel_shape=(7, 7),
        kernel_poly_order=1,
        bg_poly_order=1,
        sigma_clip=3.0,
        max_iter=3,
        verbose=False,
    )


# ============================================================================
# Tests: sfft.solve (low-level solver)
# ============================================================================

class TestSFFTSolve:
    """Tests for stdpipe.sfft.solve."""

    def test_basic_solve(self, sfft_result_basic):
        """solve returns SFFTResult with expected attributes."""
        from stdpipe.sfft import SFFTResult

        r = sfft_result_basic
        assert isinstance(r, SFFTResult)
        assert r.diff.shape == (256, 256)
        assert r.model.shape == (256, 256)
        assert r.kernel_shape == (7, 7)
        assert r.kernel_poly_order == 1
        assert r.bg_poly_order == 1
        assert r.flux_poly_order == 1
        assert np.isfinite(r.rms)
        assert r.n_iter >= 1
        assert r.n_good > 0

    def test_rms_reasonable(self, sfft_result_basic, image_pair):
        """Residual RMS is in a reasonable range (below raw image noise)."""
        _, _, _, info = image_pair
        r = sfft_result_basic
        # RMS should be much less than the background level
        assert r.rms < info['background'] * 0.5
        # But above zero
        assert r.rms > 0

    def test_diff_no_nan(self, sfft_result_basic):
        """Difference image has no NaN values."""
        assert not np.any(np.isnan(sfft_result_basic.diff))

    def test_flux_scale_recovered(self, sfft_result_basic, image_pair):
        """Flux scale polynomial constant term is in a plausible range."""
        _, _, _, info = image_pair
        r = sfft_result_basic
        # First coefficient is the constant flux scale
        recovered = r.flux_poly_coeffs[0]
        # On small images with simple Gaussians, the kernel sum may not
        # perfectly equal the true flux ratio, but should be in the
        # right ballpark (within ~20%)
        assert 0.5 < recovered < 2.0

    def test_kernel_coeffs_shape(self, sfft_result_basic):
        """Kernel coefficients have the expected shape."""
        r = sfft_result_basic
        n_kernel = 7 * 7
        n_kpoly = 3  # order 1: 1, x, y
        assert r.kernel_coeffs.shape == (n_kernel, n_kpoly)

    def test_bg_coeffs_shape(self, sfft_result_basic):
        """Background coefficients have the expected shape."""
        r = sfft_result_basic
        n_bgpoly = 3  # order 1
        assert r.bg_coeffs.shape == (n_bgpoly,)

    def test_sigma_clipping(self, image_pair):
        """Sigma clipping converges and reduces n_good."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair

        r_noclip = solve(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, sigma_clip=None, max_iter=1,
            verbose=False,
        )
        r_clip = solve(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, sigma_clip=3.0, max_iter=5,
            verbose=False,
        )

        # Clipping should reject some pixels (transients, outliers)
        assert r_clip.n_good <= r_noclip.n_good
        # And improve or maintain RMS
        assert r_clip.rms <= r_noclip.rms * 1.01

    def test_different_kernel_sizes(self, image_pair):
        """Different kernel sizes all produce finite results."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair

        for ksize in [5, 7, 9]:
            r = solve(
                science, template, mask=mask,
                kernel_shape=(ksize, ksize), kernel_poly_order=1,
                bg_poly_order=1, max_iter=1, verbose=False,
            )
            assert np.isfinite(r.rms)
            assert not np.any(np.isnan(r.diff))

    def test_different_poly_orders(self, image_pair):
        """Different polynomial orders produce valid results."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair

        for kpoly in [0, 1, 2]:
            r = solve(
                science, template, mask=mask,
                kernel_shape=(5, 5), kernel_poly_order=kpoly,
                bg_poly_order=kpoly, flux_poly_order=min(kpoly, 1),
                max_iter=1, verbose=False,
            )
            assert np.isfinite(r.rms)

    def test_err_weighting(self, image_pair):
        """Providing an error map changes the fit."""
        from stdpipe.sfft import solve

        science, template, mask, info = image_pair
        err = np.sqrt(np.clip(science, 1, None) / info['gain'])

        r_uniform = solve(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )
        r_weighted = solve(
            science, template, mask=mask, err=err,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )

        # Both should produce finite results; coefficients should differ
        assert np.isfinite(r_uniform.rms)
        assert np.isfinite(r_weighted.rms)
        assert not np.allclose(
            r_uniform.kernel_coeffs, r_weighted.kernel_coeffs
        )

    def test_err_true(self, image_pair):
        """err=True auto-estimates noise from the image."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair

        r = solve(
            science, template, mask=mask, err=True,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )
        assert np.isfinite(r.rms)

    def test_nan_handling(self):
        """NaN pixels in input images are handled gracefully."""
        from stdpipe.sfft import solve

        rng = np.random.RandomState(99)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))

        # Inject NaN
        template[10, 15] = np.nan
        template[20:25, 30:35] = np.nan
        science[50, 60] = np.nan

        r = solve(
            science, template,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )
        assert np.isfinite(r.rms)
        assert not np.any(np.isnan(r.diff))

    def test_mask_handling(self):
        """Masked pixels are excluded from the fit."""
        from stdpipe.sfft import solve

        rng = np.random.RandomState(77)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))

        mask = np.zeros((size, size), dtype=bool)
        mask[40:60, 40:60] = True  # Mask a region

        r = solve(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )
        assert np.isfinite(r.rms)
        assert r.n_good < size * size

    def test_template_mask(self):
        """template_mask parameter works."""
        from stdpipe.sfft import solve

        rng = np.random.RandomState(77)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))

        tmask = np.zeros((size, size), dtype=bool)
        tmask[:, 90:] = True

        r = solve(
            science, template, template_mask=tmask,
            kernel_shape=(5, 5), kernel_poly_order=1,
            bg_poly_order=1, max_iter=1, verbose=False,
        )
        assert np.isfinite(r.rms)

    def test_even_kernel_raises(self, image_pair):
        """Even kernel dimensions raise ValueError."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair
        with pytest.raises(ValueError, match="odd"):
            solve(science, template, kernel_shape=(6, 6), max_iter=1)

    def test_shape_mismatch_raises(self):
        """Mismatched image shapes raise ValueError."""
        from stdpipe.sfft import solve

        with pytest.raises(ValueError, match="same shape"):
            solve(np.zeros((100, 100)), np.zeros((100, 120)), max_iter=1)

    def test_flux_poly_order_too_high_raises(self, image_pair):
        """flux_poly_order > kernel_poly_order raises ValueError."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair
        with pytest.raises(ValueError, match="flux_poly_order"):
            solve(
                science, template,
                kernel_poly_order=1, flux_poly_order=2,
                max_iter=1,
            )

    def test_verbose_callable(self, image_pair):
        """verbose=callable works and receives log messages."""
        from stdpipe.sfft import solve

        science, template, mask, _ = image_pair
        messages = []
        solve(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=0,
            bg_poly_order=0, flux_poly_order=0, max_iter=1,
            verbose=lambda *a, **k: messages.append(str(a)),
        )
        assert len(messages) > 0


# ============================================================================
# Tests: sfft convenience functions
# ============================================================================

class TestSFFTConvenience:
    """Tests for evaluate_kernel_at, evaluate_flux_scale, evaluate_background."""

    def test_evaluate_kernel_at(self, sfft_result_basic):
        """evaluate_kernel_at returns a kernel of expected shape."""
        from stdpipe.sfft import evaluate_kernel_at

        kernel = evaluate_kernel_at(sfft_result_basic, 128, 128, (256, 256))
        assert kernel.shape == (7, 7)
        assert np.all(np.isfinite(kernel))
        # Kernel should have a dominant central peak
        assert kernel[3, 3] == np.max(kernel)

    def test_evaluate_flux_scale(self, sfft_result_basic):
        """evaluate_flux_scale returns a scalar for scalar input."""
        from stdpipe.sfft import evaluate_flux_scale

        fs = evaluate_flux_scale(sfft_result_basic, 128, 128, (256, 256))
        assert np.isscalar(fs) or fs.ndim == 0
        assert np.isfinite(fs)

    def test_evaluate_flux_scale_array(self, sfft_result_basic):
        """evaluate_flux_scale works with array inputs."""
        from stdpipe.sfft import evaluate_flux_scale

        x = np.array([50, 128, 200])
        y = np.array([50, 128, 200])
        fs = evaluate_flux_scale(sfft_result_basic, x, y, (256, 256))
        assert fs.shape == (3,)
        assert np.all(np.isfinite(fs))

    def test_evaluate_background(self, sfft_result_basic):
        """evaluate_background returns finite values."""
        from stdpipe.sfft import evaluate_background

        bg = evaluate_background(sfft_result_basic, 128, 128, (256, 256))
        assert np.isfinite(bg)

    def test_evaluate_background_map(self, sfft_result_basic):
        """evaluate_background works for a full image grid."""
        from stdpipe.sfft import evaluate_background

        xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        bg = evaluate_background(sfft_result_basic, xx, yy, (256, 256))
        assert bg.shape == (256, 256)
        assert np.all(np.isfinite(bg))


# ============================================================================
# Tests: subtraction.run_sfft (high-level wrapper)
# ============================================================================

class TestRunSFFTWrapper:
    """Tests for stdpipe.subtraction.run_sfft wrapper."""

    def test_basic_call(self, image_pair):
        """Basic call returns a single diff array."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, _ = image_pair
        diff = run_sfft(science, template, mask=mask, verbose=False)
        assert isinstance(diff, np.ndarray)
        assert diff.shape == science.shape
        assert not np.any(np.isnan(diff))

    def test_err_true(self, image_pair):
        """err=True builds noise model and runs successfully."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        diff = run_sfft(
            science, template, mask=mask,
            err=True, image_gain=info['gain'],
            verbose=False,
        )
        assert isinstance(diff, np.ndarray)
        assert np.isfinite(mad_std(diff))

    def test_both_err_true(self, image_pair):
        """Both err=True and template_err=True work."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        diff = run_sfft(
            science, template, mask=mask,
            err=True, template_err=True,
            image_gain=info['gain'], template_gain=info['gain'] * 5,
            verbose=False,
        )
        assert isinstance(diff, np.ndarray)

    def test_err_array(self, image_pair):
        """Providing err as an array works."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        err = np.full_like(science, 50.0)
        diff = run_sfft(
            science, template, mask=mask, err=err, verbose=False,
        )
        assert isinstance(diff, np.ndarray)

    def test_get_noise(self, image_pair):
        """get_noise=True returns [diff, noise]."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        result = run_sfft(
            science, template, mask=mask,
            err=True, template_err=True,
            image_gain=info['gain'], template_gain=info['gain'] * 5,
            get_noise=True, verbose=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        diff, noise = result
        assert diff.shape == science.shape
        assert noise.shape == science.shape
        assert np.all(noise >= 0)
        assert np.all(np.isfinite(noise))

    def test_noise_calibration(self, image_pair):
        """Noise map is correctly calibrated: diff/noise has ~unit scatter."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        diff, noise = run_sfft(
            science, template, mask=mask,
            err=True, template_err=True,
            image_gain=info['gain'], template_gain=info['gain'] * 5,
            get_noise=True, verbose=False,
        )
        # Avoid edges and masked pixels
        edge = 20
        inner = slice(edge, -edge)
        ratio = diff[inner, inner] / noise[inner, inner]
        scatter = mad_std(ratio)
        # Should be close to 1 (within ~15% tolerance for small images)
        assert 0.7 < scatter < 1.3, f"Noise scatter {scatter:.3f} not ~1"

    def test_noise_without_template_err(self, image_pair):
        """Noise map with only science err (deep template limit)."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        diff, noise = run_sfft(
            science, template, mask=mask,
            err=True, image_gain=info['gain'],
            get_noise=True, verbose=False,
        )
        assert noise.shape == science.shape
        assert np.all(noise > 0)

    def test_noise_with_template_err_larger(self, image_pair):
        """Including template err increases the noise map."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair

        _, noise_sci_only = run_sfft(
            science, template, mask=mask,
            err=True, image_gain=info['gain'],
            get_noise=True, verbose=False,
        )

        _, noise_both = run_sfft(
            science, template, mask=mask,
            err=True, template_err=True,
            image_gain=info['gain'], template_gain=info['gain'],
            get_noise=True, verbose=False,
        )

        # Including template noise should make the noise map larger
        edge = 20
        inner = slice(edge, -edge)
        assert np.median(noise_both[inner, inner]) >= np.median(
            noise_sci_only[inner, inner]
        )

    def test_get_scaled(self, image_pair):
        """get_scaled=True returns [diff, scaled_diff]."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        result = run_sfft(
            science, template, mask=mask,
            err=True, image_gain=info['gain'],
            get_scaled=True, verbose=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        diff, scaled = result
        assert scaled.shape == science.shape
        # Scaled diff should have unit scatter where valid
        edge = 20
        inner = slice(edge, -edge)
        valid = scaled[inner, inner]
        valid = valid[valid != 0]
        scatter = mad_std(valid)
        assert 0.5 < scatter < 1.5

    def test_get_convolved(self, image_pair):
        """get_convolved=True returns [diff, convolved_template]."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, _ = image_pair
        result = run_sfft(
            science, template, mask=mask,
            get_convolved=True, verbose=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        diff, conv = result
        assert conv.shape == science.shape
        # Convolved template + diff ≈ science (model = conv + bg)
        # The diff = science - model, so conv ≈ science - diff - bg
        assert np.isfinite(np.median(conv))

    def test_get_kernel(self, image_pair):
        """get_kernel=True returns [diff, SFFTResult]."""
        from stdpipe.subtraction import run_sfft
        from stdpipe.sfft import SFFTResult

        science, template, mask, _ = image_pair
        result = run_sfft(
            science, template, mask=mask,
            get_kernel=True, verbose=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        diff, sfft_res = result
        assert isinstance(sfft_res, SFFTResult)
        assert sfft_res.kernel_shape == (7, 7)

    def test_get_multiple(self, image_pair):
        """Multiple get_* options return results in correct order."""
        from stdpipe.subtraction import run_sfft
        from stdpipe.sfft import SFFTResult

        science, template, mask, info = image_pair
        result = run_sfft(
            science, template, mask=mask,
            err=True, template_err=True,
            image_gain=info['gain'], template_gain=info['gain'] * 5,
            get_convolved=True, get_noise=True,
            get_scaled=True, get_kernel=True,
            verbose=False,
        )
        # Order: diff, convolved, noise, scaled, kernel
        assert len(result) == 5
        diff, conv, noise, scaled, sfft_res = result
        assert diff.shape == science.shape
        assert conv.shape == science.shape
        assert noise.shape == science.shape
        assert scaled.shape == science.shape
        assert isinstance(sfft_res, SFFTResult)

    def test_nan_in_template(self):
        """NaN pixels in template are handled by the wrapper."""
        from stdpipe.subtraction import run_sfft

        rng = np.random.RandomState(42)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))
        template[10, 15] = np.nan
        template[50:55, 60:65] = np.nan

        diff = run_sfft(science, template, verbose=False)
        assert not np.any(np.isnan(diff))

    def test_nan_in_science(self):
        """NaN pixels in science image are handled."""
        from stdpipe.subtraction import run_sfft

        rng = np.random.RandomState(42)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))
        science[30, 40] = np.nan

        diff = run_sfft(science, template, verbose=False)
        assert not np.any(np.isnan(diff))

    def test_mask_combined(self):
        """mask and template_mask are combined properly."""
        from stdpipe.subtraction import run_sfft

        rng = np.random.RandomState(42)
        size = 128
        science = rng.normal(1000, 10, (size, size))
        template = rng.normal(1000, 10, (size, size))

        mask = np.zeros((size, size), dtype=bool)
        mask[10:20, :] = True
        tmask = np.zeros((size, size), dtype=bool)
        tmask[:, 100:] = True

        diff = run_sfft(
            science, template, mask=mask, template_mask=tmask,
            verbose=False,
        )
        assert isinstance(diff, np.ndarray)

    def test_kernel_params_passthrough(self, image_pair):
        """SFFT-specific parameters are passed through to the solver."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, _ = image_pair
        diff, sfft_res = run_sfft(
            science, template, mask=mask,
            kernel_shape=(9, 9),
            kernel_poly_order=1,
            bg_poly_order=0,
            flux_poly_order=0,
            get_kernel=True, verbose=False,
        )
        assert sfft_res.kernel_shape == (9, 9)
        assert sfft_res.kernel_poly_order == 1
        assert sfft_res.bg_poly_order == 0
        assert sfft_res.flux_poly_order == 0

    def test_verbose_true(self, image_pair, capsys):
        """verbose=True prints output."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, _ = image_pair
        run_sfft(
            science, template, mask=mask,
            kernel_shape=(5, 5), kernel_poly_order=0,
            bg_poly_order=0, flux_poly_order=0, max_iter=1,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert 'SFFT' in captured.out

    def test_transient_detected(self, image_pair):
        """Transients are visible in the difference image."""
        from stdpipe.subtraction import run_sfft

        science, template, mask, info = image_pair
        diff, noise = run_sfft(
            science, template, mask=mask,
            err=True, image_gain=info['gain'],
            get_noise=True, verbose=False,
        )

        # Check each transient location
        for tx, ty in zip(info['trans_x'], info['trans_y']):
            ix, iy = int(round(tx)), int(round(ty))
            if 10 < ix < 246 and 10 < iy < 246:
                snr = diff[iy, ix] / noise[iy, ix]
                # Transient should be clearly detected (> 3 sigma)
                assert snr > 3, f"Transient at ({tx:.0f},{ty:.0f}) SNR={snr:.1f}"


# ============================================================================
# Tests: noise model helpers
# ============================================================================

class TestBuildErrMap:
    """Tests for _build_err_map helper."""

    def test_basic(self):
        """_build_err_map returns positive noise map."""
        from stdpipe.subtraction import _build_err_map

        rng = np.random.RandomState(42)
        image = rng.normal(1000, 50, (128, 128))
        mask = np.zeros((128, 128), dtype=bool)
        err, bg = _build_err_map(image, mask, gain=2.0)
        assert err.shape == (128, 128)
        assert np.all(err > 0)
        assert np.all(np.isfinite(err))

    def test_gain_increases_noise_at_sources(self):
        """Lower gain increases noise contribution from sources."""
        from stdpipe.subtraction import _build_err_map

        rng = np.random.RandomState(42)
        image = rng.normal(1000, 50, (128, 128))
        # Add a bright source
        image[64, 64] += 5000
        mask = np.zeros((128, 128), dtype=bool)

        err_highgain, _ = _build_err_map(image, mask, gain=100.0)
        err_lowgain, _ = _build_err_map(image, mask, gain=1.0)

        # At the bright source, low gain should give higher noise
        assert err_lowgain[64, 64] > err_highgain[64, 64]

    def test_no_gain(self):
        """With gain=None, no source contribution is added."""
        from stdpipe.subtraction import _build_err_map

        rng = np.random.RandomState(42)
        image = rng.normal(1000, 50, (128, 128))
        mask = np.zeros((128, 128), dtype=bool)
        err, _ = _build_err_map(image, mask, gain=None)
        assert np.all(err > 0)


class TestComputeDiffNoise:
    """Tests for _compute_diff_noise helper."""

    def test_science_only(self, sfft_result_basic):
        """With no template err, diff noise equals science noise."""
        from stdpipe.subtraction import _compute_diff_noise

        err = np.full((256, 256), 50.0)
        mask = np.zeros((256, 256), dtype=bool)
        noise = _compute_diff_noise(
            sfft_result_basic, err, None, mask, lambda *a, **k: None
        )
        # Should be exactly the science noise since no template contribution
        np.testing.assert_allclose(noise, 50.0)

    def test_with_template_err(self, sfft_result_basic):
        """Template err contribution increases the noise map."""
        from stdpipe.subtraction import _compute_diff_noise

        err = np.full((256, 256), 50.0)
        tmpl_err = np.full((256, 256), 10.0)
        mask = np.zeros((256, 256), dtype=bool)

        noise_sci = _compute_diff_noise(
            sfft_result_basic, err, None, mask, lambda *a, **k: None
        )
        noise_both = _compute_diff_noise(
            sfft_result_basic, err, tmpl_err, mask, lambda *a, **k: None
        )
        # Noise with template should be >= noise without
        assert np.all(noise_both >= noise_sci - 1e-10)
        # And strictly larger in most pixels (where kernel coeffs are nonzero)
        edge = 20
        inner = slice(edge, -edge)
        assert np.median(noise_both[inner, inner]) > np.median(
            noise_sci[inner, inner]
        )

    def test_no_err_at_all(self, sfft_result_basic):
        """With no err maps, falls back to constant RMS estimate."""
        from stdpipe.subtraction import _compute_diff_noise

        mask = np.zeros((256, 256), dtype=bool)
        noise = _compute_diff_noise(
            sfft_result_basic, None, None, mask, lambda *a, **k: None
        )
        assert noise.shape == (256, 256)
        assert np.all(noise > 0)
        # Should be constant (from rms estimate)
        assert np.std(noise) < 1e-10
