"""
Unit tests for global background estimation with get_background().

Tests the trade-off between mesh size (size parameter) and background model accuracy:
- Too small: stars contaminate mesh cells
- Too large: fails to capture gradients

Tests different background models:
- Flat background
- Linear gradients
- Quadratic gradients
- Sinusoidal patterns (realistic vignetting)
"""

import pytest
import numpy as np
from scipy.special import erf

from stdpipe import photometry


def create_gaussian_star(size, x, y, flux, fwhm):
    """
    Create a single Gaussian star with pixel-integrated PSF.

    Parameters
    ----------
    size : int
        Image size (square)
    x, y : float
        Star position
    flux : float
        Total flux
    fwhm : float
        Full-width at half-maximum

    Returns
    -------
    image : ndarray
        Star image
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x_edges = np.arange(size + 1) - 0.5
    y_edges = np.arange(size + 1) - 0.5
    sqrt2_sigma = np.sqrt(2) * sigma
    cdf_x = 0.5 * (1 + erf((x_edges - x) / sqrt2_sigma))
    cdf_y = 0.5 * (1 + erf((y_edges - y) / sqrt2_sigma))
    flux_x = np.diff(cdf_x)
    flux_y = np.diff(cdf_y)
    image = flux * np.outer(flux_y, flux_x)
    return image


def create_flat_background(size, level=100.0):
    """
    Create flat background.

    Parameters
    ----------
    size : int
        Image size
    level : float
        Background level in ADU

    Returns
    -------
    bg : ndarray
        Background image
    """
    return np.full((size, size), level)


def create_linear_gradient(size, amplitude, angle_deg=0):
    """
    Create linear gradient background.

    Parameters
    ----------
    size : int
        Image size
    amplitude : float
        Gradient amplitude (ADU change across image)
    angle_deg : float
        Gradient direction in degrees (0 = horizontal, 90 = vertical)

    Returns
    -------
    bg : ndarray
        Background image with linear gradient
    """
    yy, xx = np.mgrid[:size, :size]
    center = size / 2

    # Rotate coordinates
    angle_rad = np.radians(angle_deg)
    xx_rot = (xx - center) * np.cos(angle_rad) + (yy - center) * np.sin(angle_rad)

    # Linear gradient from -amplitude/2 to +amplitude/2
    bg = amplitude * (xx_rot / size)

    return bg


def create_quadratic_gradient(size, amplitude, center_offset=0):
    """
    Create quadratic (parabolic) gradient background.

    Parameters
    ----------
    size : int
        Image size
    amplitude : float
        Gradient amplitude (ADU difference between center and edge)
    center_offset : float
        Offset to add to center value (allows brighter center or edges)

    Returns
    -------
    bg : ndarray
        Background image with quadratic gradient
    """
    yy, xx = np.mgrid[:size, :size]
    center = size / 2

    # Radial distance from center
    r = np.sqrt((xx - center)**2 + (yy - center)**2)
    r_max = np.sqrt(2) * size / 2  # Maximum distance (corner)

    # Quadratic gradient
    bg = center_offset + amplitude * (r / r_max)**2

    return bg


def create_sinusoidal_background(size, amplitude, wavelength_fraction=1.0):
    """
    Create sinusoidal background (simulates vignetting or other periodic patterns).

    Parameters
    ----------
    size : int
        Image size
    amplitude : float
        Amplitude of sinusoidal variation
    wavelength_fraction : float
        Wavelength as fraction of image size (1.0 = one full period)

    Returns
    -------
    bg : ndarray
        Background image with sinusoidal pattern
    """
    yy, xx = np.mgrid[:size, :size]
    center = size / 2

    # Radial distance from center
    r = np.sqrt((xx - center)**2 + (yy - center)**2)
    r_max = np.sqrt(2) * size / 2

    # Sinusoidal variation
    bg = amplitude * np.sin(2 * np.pi * (r / r_max) / wavelength_fraction)

    return bg


def add_random_stars(image, nstars, flux_range, fwhm, rng, edge_buffer=20):
    """
    Add random stars to an image.

    Parameters
    ----------
    image : ndarray
        Background image to add stars to (modified in place)
    nstars : int
        Number of stars to add
    flux_range : tuple
        (min_flux, max_flux) range for star fluxes
    fwhm : float
        FWHM of stars
    rng : np.random.RandomState
        Random number generator
    edge_buffer : int
        Keep stars away from edges by this many pixels

    Returns
    -------
    stars : list of dict
        List of star parameters: {'x', 'y', 'flux'}
    """
    size = image.shape[0]
    stars = []

    for _ in range(nstars):
        # Random position (avoid edges)
        x = rng.uniform(edge_buffer, size - edge_buffer)
        y = rng.uniform(edge_buffer, size - edge_buffer)

        # Random flux
        flux = rng.uniform(*flux_range)

        # Add star
        star_image = create_gaussian_star(size, x, y, flux, fwhm)
        image += star_image

        stars.append({'x': x, 'y': y, 'flux': flux})

    return stars


def compute_background_rms(estimated_bg, true_bg, mask=None):
    """
    Compute RMS difference between estimated and true background.

    Parameters
    ----------
    estimated_bg : ndarray
        Estimated background
    true_bg : ndarray
        True background model
    mask : ndarray, optional
        Mask of pixels to exclude

    Returns
    -------
    rms : float
        RMS difference in ADU
    """
    diff = estimated_bg - true_bg
    if mask is not None:
        diff = diff[~mask]
    return np.sqrt(np.mean(diff**2))


def compute_background_bias(estimated_bg, true_bg, mask=None):
    """
    Compute mean bias between estimated and true background.

    Parameters
    ----------
    estimated_bg : ndarray
        Estimated background
    true_bg : ndarray
        True background model
    mask : ndarray, optional
        Mask of pixels to exclude

    Returns
    -------
    bias : float
        Mean bias in ADU
    """
    diff = estimated_bg - true_bg
    if mask is not None:
        diff = diff[~mask]
    return np.mean(diff)


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def test_config():
    """Common test configuration."""
    return {
        'size': 512,
        'fwhm': 3.0,
        'noise_std': 2.0,
        'nstars': 100,
        'flux_range': (1000, 10000),
        'edge_buffer': 30
    }


@pytest.mark.unit
def test_flat_background_no_stars(test_config, rng):
    """
    Test background estimation on flat background without stars.

    Should work perfectly regardless of mesh size.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create flat background with noise
    true_bg = create_flat_background(size, bg_level)
    image = true_bg + rng.normal(0, noise_std, (size, size))

    # Test different mesh sizes
    for mesh_size in [32, 64, 128, 256]:
        estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

        # Compute RMS and bias
        rms = compute_background_rms(estimated_bg, true_bg)
        bias = compute_background_bias(estimated_bg, true_bg)

        # Should be very accurate (close to noise level)
        assert rms < 5 * noise_std, (
            f"size={mesh_size}: RMS too high ({rms:.2f} ADU) for flat background"
        )
        assert abs(bias) < noise_std, (
            f"size={mesh_size}: Bias too high ({bias:.2f} ADU) for flat background"
        )


@pytest.mark.unit
@pytest.mark.parametrize("mesh_size", [32, 64, 128, 256])
def test_flat_background_with_stars(test_config, rng, mesh_size):
    """
    Test background estimation on flat background with stars.

    Larger mesh sizes should be more robust to star contamination.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create flat background
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()

    # Add noise
    image += rng.normal(0, noise_std, (size, size))

    # Add stars
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

    # Compute RMS and bias
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # For flat background, all mesh sizes should work reasonably well
    # But smaller sizes may be more contaminated by stars
    if mesh_size <= 64:
        # Small mesh: may have star contamination
        assert rms < 20.0, (
            f"size={mesh_size}: RMS too high ({rms:.2f} ADU)"
        )
    else:
        # Large mesh: should be quite accurate
        assert rms < 10.0, (
            f"size={mesh_size}: RMS too high ({rms:.2f} ADU)"
        )

    # Bias should be small (stars should be clipped out by sigma-clipping)
    assert abs(bias) < 5.0, (
        f"size={mesh_size}: Bias too high ({bias:.2f} ADU)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("mesh_size", [64, 128, 256])
@pytest.mark.parametrize("gradient_amp", [100, 500, 1000])
def test_linear_gradient_with_stars(test_config, rng, mesh_size, gradient_amp):
    """
    Test background estimation with linear gradient and stars.

    Trade-off between mesh size and gradient capture:
    - Small size: better gradient capture but more star contamination
    - Large size: less contamination but misses fine gradient structure
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    # Create linear gradient background
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=45)
    image = true_bg.copy()

    # Add noise
    image += rng.normal(0, noise_std, (size, size))

    # Add stars
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

    # Compute RMS and bias
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # Expected performance depends on gradient amplitude and mesh size
    if gradient_amp <= 100:
        # Weak gradient: should work well for all sizes
        assert rms < 20.0, (
            f"size={mesh_size}, gradient={gradient_amp}: "
            f"RMS too high ({rms:.2f} ADU)"
        )
    elif gradient_amp <= 500:
        # Moderate gradient: intermediate sizes work best
        if mesh_size == 128:
            assert rms < 30.0, (
                f"size={mesh_size}, gradient={gradient_amp}: "
                f"RMS too high ({rms:.2f} ADU)"
            )
    else:
        # Strong gradient: smaller sizes needed but allow for more variation
        if mesh_size == 64:
            assert rms < 50.0, (
                f"size={mesh_size}, gradient={gradient_amp}: "
                f"RMS too high ({rms:.2f} ADU)"
            )
        elif mesh_size == 128:
            assert rms < 60.0, (
                f"size={mesh_size}, gradient={gradient_amp}: "
                f"RMS too high ({rms:.2f} ADU)"
            )

    # Just check finite values
    assert np.isfinite(bias), f"size={mesh_size}: Bias is not finite"


@pytest.mark.unit
@pytest.mark.parametrize("mesh_size", [64, 128, 256])
@pytest.mark.parametrize("gradient_amp", [100, 500, 1000])
def test_quadratic_gradient_with_stars(test_config, rng, mesh_size, gradient_amp):
    """
    Test background estimation with quadratic gradient and stars.

    Quadratic gradients are harder to capture than linear ones.
    Requires smaller mesh sizes.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    # Create quadratic gradient background
    true_bg = create_quadratic_gradient(size, gradient_amp, center_offset=100)
    image = true_bg.copy()

    # Add noise
    image += rng.normal(0, noise_std, (size, size))

    # Add stars
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

    # Compute RMS and bias
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # Quadratic gradients are challenging
    # Smaller mesh sizes needed but more susceptible to star contamination
    if gradient_amp <= 100:
        # Weak quadratic: should work reasonably
        assert rms < 30.0, (
            f"size={mesh_size}, gradient={gradient_amp}: "
            f"RMS too high ({rms:.2f} ADU)"
        )
    else:
        # Strong quadratic: difficult for all mesh sizes
        # Just verify it doesn't crash and produces reasonable values
        assert np.isfinite(rms), f"size={mesh_size}: RMS is not finite"
        assert rms < 250.0, (
            f"size={mesh_size}, gradient={gradient_amp}: "
            f"RMS unreasonably high ({rms:.2f} ADU)"
        )


@pytest.mark.unit
@pytest.mark.parametrize("mesh_size", [64, 128])
def test_sinusoidal_background_with_stars(test_config, rng, mesh_size):
    """
    Test background estimation with sinusoidal pattern (simulates vignetting).

    This is a challenging case that requires moderate mesh sizes.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    amplitude = 200.0

    # Create sinusoidal background
    true_bg = create_sinusoidal_background(size, amplitude, wavelength_fraction=1.0)
    image = true_bg.copy()

    # Add constant offset
    image += 100.0
    true_bg += 100.0

    # Add noise
    image += rng.normal(0, noise_std, (size, size))

    # Add stars
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

    # Compute RMS and bias
    rms = compute_background_rms(estimated_bg, true_bg)

    # Sinusoidal patterns are challenging
    # Just verify reasonable behavior
    assert np.isfinite(rms), f"size={mesh_size}: RMS is not finite"
    assert rms < 200.0, (
        f"size={mesh_size}: RMS unreasonably high ({rms:.2f} ADU) "
        f"for sinusoidal background"
    )


@pytest.mark.unit
def test_compare_mesh_sizes_linear_gradient(test_config, rng):
    """
    Compare different mesh sizes on the same linear gradient image.

    Demonstrates the trade-off: smaller sizes capture gradients better
    but are more susceptible to star contamination.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    gradient_amp = 500.0

    # Create image with linear gradient and stars
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=0)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test multiple mesh sizes
    mesh_sizes = [32, 64, 128, 256]
    results = {}

    for mesh_size in mesh_sizes:
        estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)
        rms = compute_background_rms(estimated_bg, true_bg)
        bias = compute_background_bias(estimated_bg, true_bg)

        results[mesh_size] = {'rms': rms, 'bias': bias}

    # All should produce finite results
    for mesh_size, metrics in results.items():
        assert np.isfinite(metrics['rms']), f"size={mesh_size}: RMS is not finite"
        assert np.isfinite(metrics['bias']), f"size={mesh_size}: Bias is not finite"

    # Check that we're in a reasonable range
    # Optimal size should be around 64-128 for this gradient
    best_rms = min(r['rms'] for r in results.values())
    assert best_rms < 50.0, f"Best RMS ({best_rms:.2f}) too high for this gradient"


@pytest.mark.unit
def test_sep_vs_photutils_methods(test_config, rng):
    """
    Compare SEP and photutils background estimation methods.

    Both should give similar results on the same image.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = 50  # Fewer stars for faster test
    flux_range = test_config['flux_range']
    gradient_amp = 300.0
    mesh_size = 128

    # Create image with linear gradient and stars
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=45)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate with SEP
    bg_sep = photometry.get_background(image, method='sep', size=mesh_size)

    # Estimate with photutils
    bg_photutils = photometry.get_background(image, method='photutils', size=mesh_size)

    # Both should be finite
    assert np.all(np.isfinite(bg_sep)), "SEP background has non-finite values"
    assert np.all(np.isfinite(bg_photutils)), "Photutils background has non-finite values"

    # They should be similar (but not identical due to different algorithms)
    diff_rms = np.sqrt(np.mean((bg_sep - bg_photutils)**2))
    assert diff_rms < 50.0, (
        f"SEP and photutils differ too much (RMS={diff_rms:.2f} ADU)"
    )


@pytest.mark.unit
def test_background_with_mask(test_config, rng):
    """
    Test that masked regions are handled correctly.

    Masked pixels should not contaminate background estimation.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0
    mesh_size = 128

    # Create flat background
    true_bg = create_flat_background(size, bg_level)
    image = true_bg + rng.normal(0, noise_std, (size, size))

    # Create mask with a large contaminated region
    mask = np.zeros((size, size), dtype=bool)
    mask[200:300, 200:300] = True

    # Add high values in masked region (should be ignored)
    image[mask] += 10000.0

    # Estimate background with mask
    estimated_bg = photometry.get_background(image, mask=mask, method='sep', size=mesh_size)

    # Background should still be accurate outside masked region
    rms_unmasked = compute_background_rms(estimated_bg, true_bg, mask=mask)

    assert rms_unmasked < 10.0, (
        f"RMS in unmasked region too high ({rms_unmasked:.2f} ADU)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("mesh_size", [64, 128, 256])
def test_extreme_star_contamination(test_config, rng, mesh_size):
    """
    Test with very high star density (extreme contamination).

    Background estimation should still be robust (stars clipped out).
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = 500  # Very high density
    flux_range = (500, 5000)
    bg_level = 100.0

    # Create flat background
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))

    # Add many stars
    stars = add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='sep', size=mesh_size)

    # With sigma-clipping, should still recover background reasonably
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # Allow more error due to extreme contamination
    assert np.isfinite(rms), "RMS is not finite with extreme contamination"
    assert rms < 50.0, (
        f"size={mesh_size}: RMS too high ({rms:.2f} ADU) "
        f"even with sigma-clipping"
    )


@pytest.mark.unit
def test_get_rms_option(test_config, rng):
    """
    Test that get_rms=True returns both background and RMS maps.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0
    mesh_size = 128

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Get both background and RMS
    bg, bg_rms = photometry.get_background(
        image, method='sep', size=mesh_size, get_rms=True
    )

    # Both should be finite
    assert np.all(np.isfinite(bg)), "Background has non-finite values"
    assert np.all(np.isfinite(bg_rms)), "Background RMS has non-finite values"

    # RMS should be close to noise level
    mean_rms = np.mean(bg_rms)
    assert 0.5 * noise_std < mean_rms < 5 * noise_std, (
        f"Background RMS ({mean_rms:.2f}) far from noise level ({noise_std:.2f})"
    )
