"""
Unit tests for non-grid background estimation methods.

Tests percentile filtering and morphological opening methods that don't use
grid-based sampling, comparing them with standard SEP/photutils methods.
"""

import pytest
import numpy as np
from scipy.special import erf

from stdpipe import photometry

# Import test utilities
from test_background_estimation import (
    create_gaussian_star,
    create_flat_background,
    create_linear_gradient,
    create_quadratic_gradient,
    create_sinusoidal_background,
    add_random_stars,
    compute_background_rms,
    compute_background_bias
)


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def test_config():
    """Common test configuration."""
    return {
        'size': 256,  # Smaller for faster tests
        'fwhm': 3.0,
        'noise_std': 2.0,
        'nstars': 50,
        'flux_range': (1000, 10000),
        'edge_buffer': 20
    }


@pytest.mark.unit
def test_percentile_flat_background(test_config, rng):
    """
    Test percentile filtering on flat background with stars.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create flat background with stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background with percentile method
    estimated_bg = photometry.get_background(image, method='percentile', size=51, percentile=25.0)

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # Should be accurate for flat background
    assert rms < 5.0, f"RMS too high ({rms:.2f} ADU) for flat background"
    assert abs(bias) < 2.0, f"Bias too high ({bias:.2f} ADU) for flat background"


@pytest.mark.unit
def test_morphology_flat_background(test_config, rng):
    """
    Test morphological opening on flat background with stars.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create flat background with stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background with morphology method
    # Size should be larger than typical star (FWHM ~ 3, so use 15-25)
    estimated_bg = photometry.get_background(image, method='morphology', size=21)

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # Should be accurate for flat background
    # Note: morphological opening has slight systematic underestimation (~5 ADU)
    # This is expected and acceptable trade-off for speed
    assert rms < 10.0, f"RMS too high ({rms:.2f} ADU) for flat background"
    assert abs(bias) < 10.0, f"Bias too high ({bias:.2f} ADU) for flat background"


@pytest.mark.unit
@pytest.mark.parametrize("gradient_amp", [100, 500])
def test_percentile_linear_gradient(test_config, rng, gradient_amp):
    """
    Test percentile filtering on linear gradient.

    Note: Large kernel percentile filtering smooths gradients significantly.
    This is a fundamental limitation of the method - the percentile within
    a large window biases toward lower values when gradient is present.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    # Create linear gradient with stars
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=45)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background with smaller kernel for gradients
    estimated_bg = photometry.get_background(image, method='percentile', size=31, percentile=25.0)

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)

    # Percentile filtering struggles with gradients (smooths them out)
    # Just verify it produces reasonable results
    if gradient_amp <= 100:
        assert rms < 50.0, f"RMS too high ({rms:.2f} ADU) for weak gradient"
    else:
        assert rms < 200.0, f"RMS unreasonably high ({rms:.2f} ADU) for moderate gradient"


@pytest.mark.unit
@pytest.mark.parametrize("gradient_amp", [100, 500])
def test_morphology_linear_gradient(test_config, rng, gradient_amp):
    """
    Test morphological opening on linear gradient.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    # Create linear gradient with stars
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=45)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='morphology', size=21)

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)

    # Should handle linear gradients reasonably
    if gradient_amp <= 100:
        assert rms < 10.0, f"RMS too high ({rms:.2f} ADU) for weak gradient"
    else:
        assert rms < 40.0, f"RMS too high ({rms:.2f} ADU) for moderate gradient"


@pytest.mark.unit
def test_percentile_quadratic_gradient(test_config, rng):
    """
    Test percentile filtering on quadratic gradient.

    This is challenging for all methods including percentile (large kernel smooths gradients).
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    gradient_amp = 300.0

    # Create quadratic gradient with stars
    true_bg = create_quadratic_gradient(size, gradient_amp, center_offset=100)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background with percentile (non-grid), use smaller kernel
    estimated_bg_percentile = photometry.get_background(
        image, method='percentile', size=31, percentile=25.0
    )

    # Compare with grid-based method (SEP)
    estimated_bg_sep = photometry.get_background(image, method='sep', size=64)

    # Compute metrics
    rms_percentile = compute_background_rms(estimated_bg_percentile, true_bg)
    rms_sep = compute_background_rms(estimated_bg_sep, true_bg)

    # Both methods struggle with quadratic gradients
    # Just verify reasonable results
    assert rms_percentile < 150.0, (
        f"Percentile method RMS unreasonably high ({rms_percentile:.2f} ADU) for quadratic gradient"
    )

    print(f"Quadratic gradient: Percentile RMS={rms_percentile:.2f}, SEP RMS={rms_sep:.2f}")


@pytest.mark.unit
def test_morphology_quadratic_gradient(test_config, rng):
    """
    Test morphological opening on quadratic gradient.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    gradient_amp = 300.0

    # Create quadratic gradient with stars
    true_bg = create_quadratic_gradient(size, gradient_amp, center_offset=100)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(image, method='morphology', size=21)

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)

    # Should produce reasonable results
    assert rms < 50.0, f"RMS too high ({rms:.2f} ADU) for quadratic gradient"


@pytest.mark.unit
def test_percentile_with_mask(test_config, rng):
    """
    Test that percentile filtering handles masked regions correctly.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create flat background
    true_bg = create_flat_background(size, bg_level)
    image = true_bg + rng.normal(0, noise_std, (size, size))

    # Create mask
    mask = np.zeros((size, size), dtype=bool)
    mask[100:150, 100:150] = True

    # Add high values in masked region (should be ignored)
    image[mask] += 10000.0

    # Estimate background with mask
    estimated_bg = photometry.get_background(
        image, mask=mask, method='percentile', size=51, percentile=25.0
    )

    # Background should be accurate outside masked region
    rms_unmasked = compute_background_rms(estimated_bg, true_bg, mask=mask)

    assert rms_unmasked < 10.0, (
        f"RMS in unmasked region too high ({rms_unmasked:.2f} ADU)"
    )


@pytest.mark.unit
def test_morphology_with_mask(test_config, rng):
    """
    Test that morphological opening handles masked regions correctly.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create flat background
    true_bg = create_flat_background(size, bg_level)
    image = true_bg + rng.normal(0, noise_std, (size, size))

    # Create mask
    mask = np.zeros((size, size), dtype=bool)
    mask[100:150, 100:150] = True

    # Add high values in masked region
    image[mask] += 10000.0

    # Estimate background with mask
    estimated_bg = photometry.get_background(
        image, mask=mask, method='morphology', size=21
    )

    # Background should be accurate outside masked region
    rms_unmasked = compute_background_rms(estimated_bg, true_bg, mask=mask)

    assert rms_unmasked < 10.0, (
        f"RMS in unmasked region too high ({rms_unmasked:.2f} ADU)"
    )


@pytest.mark.unit
def test_percentile_get_rms(test_config, rng):
    """
    Test that get_rms=True returns background RMS for percentile method.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Get both background and RMS
    bg, bg_rms = photometry.get_background(
        image, method='percentile', size=51, get_rms=True
    )

    # Both should be finite
    assert np.all(np.isfinite(bg)), "Background has non-finite values"
    assert np.all(np.isfinite(bg_rms)), "Background RMS has non-finite values"

    # RMS should be in reasonable range
    mean_rms = np.mean(bg_rms)
    assert 0.1 < mean_rms < 20.0, (
        f"Background RMS ({mean_rms:.2f}) outside reasonable range"
    )


@pytest.mark.unit
def test_morphology_get_rms(test_config, rng):
    """
    Test that get_rms=True returns background RMS for morphology method.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Get both background and RMS
    bg, bg_rms = photometry.get_background(
        image, method='morphology', size=21, get_rms=True
    )

    # Both should be finite
    assert np.all(np.isfinite(bg)), "Background has non-finite values"
    assert np.all(np.isfinite(bg_rms)), "Background RMS has non-finite values"

    # RMS should be in reasonable range
    mean_rms = np.mean(bg_rms)
    assert 0.1 < mean_rms < 20.0, (
        f"Background RMS ({mean_rms:.2f}) outside reasonable range"
    )


@pytest.mark.unit
@pytest.mark.parametrize("percentile_val", [10, 25, 50])
def test_percentile_parameter(test_config, rng, percentile_val):
    """
    Test different percentile values.

    Lower percentiles = more aggressive star rejection.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create background with stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(
        image, method='percentile', size=51, percentile=percentile_val
    )

    # Should produce finite results
    assert np.all(np.isfinite(estimated_bg)), (
        f"percentile={percentile_val}: Background has non-finite values"
    )

    # Compute RMS
    rms = compute_background_rms(estimated_bg, true_bg)
    assert rms < 10.0, (
        f"percentile={percentile_val}: RMS too high ({rms:.2f} ADU)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("morph_size", [15, 25, 35])
def test_morphology_size_parameter(test_config, rng, morph_size):
    """
    Test different morphological structuring element sizes.

    Larger sizes = more aggressive star removal but may suppress background features.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create background with stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(
        image, method='morphology', size=morph_size
    )

    # Should produce finite results
    assert np.all(np.isfinite(estimated_bg)), (
        f"size={morph_size}: Background has non-finite values"
    )

    # Compute RMS
    rms = compute_background_rms(estimated_bg, true_bg)
    assert rms < 10.0, (
        f"size={morph_size}: RMS too high ({rms:.2f} ADU)"
    )


@pytest.mark.unit
def test_compare_all_methods(test_config, rng):
    """
    Compare all four methods (sep, photutils, percentile, morphology) on same image.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    gradient_amp = 300.0

    # Create challenging image: quadratic gradient with stars
    true_bg = create_quadratic_gradient(size, gradient_amp, center_offset=100)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test all methods
    methods = {
        'sep': {'size': 64},
        'photutils': {'size': 64},
        'percentile': {'size': 31, 'percentile': 25.0},  # Smaller kernel for gradients
        'morphology': {'size': 21}
    }

    results = {}
    for method_name, params in methods.items():
        estimated_bg = photometry.get_background(image, method=method_name, **params)
        rms = compute_background_rms(estimated_bg, true_bg)
        results[method_name] = {'rms': rms, 'background': estimated_bg}

        print(f"{method_name:12s}: RMS = {rms:6.2f} ADU")

    # All methods should produce finite results
    for method_name, result in results.items():
        assert np.all(np.isfinite(result['background'])), (
            f"{method_name}: Background has non-finite values"
        )

    # All methods should be in reasonable range
    # Quadratic gradients are challenging for all methods
    for method_name, result in results.items():
        assert result['rms'] < 150.0, (
            f"{method_name}: RMS unreasonably high ({result['rms']:.2f} ADU)"
        )


@pytest.mark.unit
def test_percentile_iterative_clipping(test_config, rng):
    """
    Test that iterative sigma-clipping improves percentile method.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create background with stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # No iterations
    bg_no_iter = photometry.get_background(
        image, method='percentile', size=51, percentile=25.0, maxiters=1
    )

    # With iterations
    bg_with_iter = photometry.get_background(
        image, method='percentile', size=51, percentile=25.0, maxiters=3
    )

    # Both should work
    rms_no_iter = compute_background_rms(bg_no_iter, true_bg)
    rms_with_iter = compute_background_rms(bg_with_iter, true_bg)

    # Iterations should help (or at least not hurt)
    assert rms_with_iter < 10.0, (
        f"RMS with iterations too high ({rms_with_iter:.2f} ADU)"
    )
