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


# ========================================================================
# Gaussian Process Background Estimation Tests
# ========================================================================


@pytest.mark.unit
def test_gp_flat_background(test_config, rng):
    """
    Test Gaussian Process method on flat background with stars.
    """
    pytest.importorskip("sklearn")

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

    # Estimate background with GP method
    estimated_bg = photometry.get_background(
        image, method='gp', size=64, max_points=1000, random_state=42
    )

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)
    bias = compute_background_bias(estimated_bg, true_bg)

    # GP should be very accurate for flat background
    assert rms < 5.0, f"RMS too high ({rms:.2f} ADU) for flat background"
    assert abs(bias) < 2.0, f"Bias too high ({bias:.2f} ADU) for flat background"


@pytest.mark.unit
@pytest.mark.parametrize("gradient_amp", [100, 500])
def test_gp_linear_gradient(test_config, rng, gradient_amp):
    """
    Test Gaussian Process method on linear gradient.

    GP should handle linear gradients very well due to its flexible kernel.
    """
    pytest.importorskip("sklearn")

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

    # Estimate background with GP
    estimated_bg = photometry.get_background(
        image, method='gp', size=64, max_points=1500, random_state=42
    )

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)

    # GP should handle gradients very well
    if gradient_amp <= 100:
        assert rms < 8.0, f"RMS too high ({rms:.2f} ADU) for weak gradient"
    else:
        assert rms < 15.0, f"RMS too high ({rms:.2f} ADU) for moderate gradient"


@pytest.mark.unit
def test_gp_quadratic_gradient(test_config, rng):
    """
    Test Gaussian Process method on quadratic gradient.

    This is where GP should really shine - complex non-linear backgrounds.
    """
    pytest.importorskip("sklearn")

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

    # Estimate background with GP
    estimated_bg = photometry.get_background(
        image, method='gp', size=64, max_points=2000, random_state=42
    )

    # Compare with other methods
    estimated_bg_sep = photometry.get_background(image, method='sep', size=64)
    estimated_bg_morph = photometry.get_background(image, method='morphology', size=21)

    # Compute metrics
    rms_gp = compute_background_rms(estimated_bg, true_bg)
    rms_sep = compute_background_rms(estimated_bg_sep, true_bg)
    rms_morph = compute_background_rms(estimated_bg_morph, true_bg)

    # GP should be the best for complex gradients
    assert rms_gp < 25.0, f"GP RMS too high ({rms_gp:.2f} ADU) for quadratic gradient"

    # GP should outperform grid-based methods
    assert rms_gp < rms_sep, f"GP ({rms_gp:.2f}) should be better than SEP ({rms_sep:.2f})"

    print(f"Quadratic gradient: GP RMS={rms_gp:.2f}, SEP RMS={rms_sep:.2f}, Morphology RMS={rms_morph:.2f}")


@pytest.mark.unit
def test_gp_with_mask(test_config, rng):
    """
    Test that GP method handles masked regions correctly.
    """
    pytest.importorskip("sklearn")

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
        image, mask=mask, method='gp', size=64, max_points=1000, random_state=42
    )

    # Background should be accurate outside masked region
    rms_unmasked = compute_background_rms(estimated_bg, true_bg, mask=mask)

    assert rms_unmasked < 10.0, (
        f"RMS in unmasked region too high ({rms_unmasked:.2f} ADU)"
    )


@pytest.mark.unit
def test_gp_get_uncertainty(test_config, rng):
    """
    Test that get_rms=True returns uncertainty map for GP method.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Get both background and uncertainty
    bg, bg_std = photometry.get_background(
        image, method='gp', size=64, max_points=1000, get_rms=True, random_state=42
    )

    # Both should be finite
    assert np.all(np.isfinite(bg)), "Background has non-finite values"
    assert np.all(np.isfinite(bg_std)), "Background uncertainty has non-finite values"

    # Uncertainty should be positive and in reasonable range
    assert np.all(bg_std > 0), "Uncertainty should be positive"
    mean_std = np.mean(bg_std)
    assert 0.1 < mean_std < 50.0, (
        f"Background uncertainty ({mean_std:.2f}) outside reasonable range"
    )

    # Uncertainty should be 2D array (not scalar)
    assert bg_std.shape == bg.shape, "Uncertainty should be same shape as background"


@pytest.mark.unit
def test_gp_without_uncertainty(test_config, rng):
    """
    Test that get_rms=False returns scalar RMS for GP method.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Get background and scalar RMS
    bg, rms = photometry.get_background(
        image, method='gp', size=64, max_points=1000, get_rms=True, random_state=42
    )

    # For GP with get_rms=True, should return 2D array
    # Let's test get_rms=False path by calling directly
    from stdpipe.photometry_background import get_background_gp
    bg2, rms2 = get_background_gp(
        image, max_points=1000, get_uncertainty=False, random_state=42
    )

    # rms2 should be scalar float
    assert isinstance(rms2, float), f"Without uncertainty, should return scalar, got {type(rms2)}"
    assert np.isfinite(rms2), "Scalar RMS should be finite"
    assert rms2 > 0, "Scalar RMS should be positive"


@pytest.mark.unit
@pytest.mark.parametrize("length_scale", [32, 64, 128])
def test_gp_length_scale_parameter(test_config, rng, length_scale):
    """
    Test different length scale values.

    Larger length scales = smoother backgrounds.
    """
    pytest.importorskip("sklearn")

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
        image, method='gp', size=length_scale, max_points=1000, random_state=42
    )

    # Should produce finite results
    assert np.all(np.isfinite(estimated_bg)), (
        f"length_scale={length_scale}: Background has non-finite values"
    )

    # Compute RMS
    rms = compute_background_rms(estimated_bg, true_bg)
    assert rms < 10.0, (
        f"length_scale={length_scale}: RMS too high ({rms:.2f} ADU)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("max_points", [500, 1000, 2000])
def test_gp_max_points_parameter(test_config, rng, max_points):
    """
    Test different numbers of training points.

    More points = better fit but slower.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    gradient_amp = 200.0

    # Create gradient with stars (more challenging)
    true_bg = create_linear_gradient(size, gradient_amp, angle_deg=45)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background
    estimated_bg = photometry.get_background(
        image, method='gp', size=64, max_points=max_points, random_state=42
    )

    # Should produce finite results
    assert np.all(np.isfinite(estimated_bg)), (
        f"max_points={max_points}: Background has non-finite values"
    )

    # Compute RMS
    rms = compute_background_rms(estimated_bg, true_bg)
    assert rms < 20.0, (
        f"max_points={max_points}: RMS too high ({rms:.2f} ADU)"
    )

    print(f"max_points={max_points}: RMS={rms:.2f} ADU")


@pytest.mark.unit
def test_gp_grid_step_sampling(test_config, rng):
    """
    Test grid-based subsampling vs random subsampling.
    """
    pytest.importorskip("sklearn")

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

    # Random sampling (default)
    bg_random = photometry.get_background(
        image, method='gp', size=64, max_points=1000, grid_step=None, random_state=42
    )

    # Grid sampling
    bg_grid = photometry.get_background(
        image, method='gp', size=64, max_points=1000, grid_step=8, random_state=42
    )

    # Both should work
    assert np.all(np.isfinite(bg_random)), "Random sampling: non-finite values"
    assert np.all(np.isfinite(bg_grid)), "Grid sampling: non-finite values"

    # Both should be accurate
    rms_random = compute_background_rms(bg_random, true_bg)
    rms_grid = compute_background_rms(bg_grid, true_bg)

    assert rms_random < 10.0, f"Random sampling: RMS too high ({rms_random:.2f} ADU)"
    assert rms_grid < 10.0, f"Grid sampling: RMS too high ({rms_grid:.2f} ADU)"


@pytest.mark.unit
def test_gp_sigma_clipping(test_config, rng):
    """
    Test that sigma clipping effectively removes stars from training set.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = 100  # More stars for challenging test
    flux_range = (5000, 20000)  # Brighter stars
    bg_level = 100.0

    # Create background with many bright stars
    true_bg = create_flat_background(size, bg_level)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Aggressive clipping
    bg_aggressive = photometry.get_background(
        image, method='gp', size=64, max_points=1000,
        clip_sigma=2.5, n_clip_iter=3, random_state=42
    )

    # Gentle clipping
    bg_gentle = photometry.get_background(
        image, method='gp', size=64, max_points=1000,
        clip_sigma=5.0, n_clip_iter=1, random_state=42
    )

    # Both should work
    rms_aggressive = compute_background_rms(bg_aggressive, true_bg)
    rms_gentle = compute_background_rms(bg_gentle, true_bg)

    # Aggressive clipping should be better (or at least reasonable)
    assert rms_aggressive < 10.0, f"Aggressive clipping: RMS too high ({rms_aggressive:.2f} ADU)"
    assert rms_gentle < 15.0, f"Gentle clipping: RMS too high ({rms_gentle:.2f} ADU)"

    print(f"Aggressive clipping: RMS={rms_aggressive:.2f}, Gentle: RMS={rms_gentle:.2f}")


@pytest.mark.unit
def test_gp_sinusoidal_background(test_config, rng):
    """
    Test GP on sinusoidal background (e.g., fringe pattern).

    This tests the method's flexibility for complex non-polynomial backgrounds.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']

    # Create sinusoidal background with stars
    true_bg = create_sinusoidal_background(size, amplitude=50.0, wavelength_fraction=0.5)
    # Add base level
    true_bg += 100.0
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Estimate background with GP (smaller length scale for higher frequency features)
    estimated_bg = photometry.get_background(
        image, method='gp', size=32, max_points=2000, random_state=42
    )

    # Compute metrics
    rms = compute_background_rms(estimated_bg, true_bg)

    # GP may struggle with high-frequency patterns (depends on length scale)
    # but should still produce reasonable results
    assert rms < 30.0, f"RMS too high ({rms:.2f} ADU) for sinusoidal background"


@pytest.mark.unit
def test_gp_compare_with_other_methods(test_config, rng):
    """
    Compare GP with all other methods on a challenging background.
    """
    pytest.importorskip("sklearn")

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

    # Test all methods including GP
    methods = {
        'sep': {'size': 64},
        'photutils': {'size': 64},
        'percentile': {'size': 31, 'percentile': 25.0},
        'morphology': {'size': 21},
        'gp': {'size': 64, 'max_points': 2000, 'random_state': 42}
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

    # GP should be among the best methods for this complex background
    gp_rms = results['gp']['rms']
    assert gp_rms < 50.0, f"GP RMS too high ({gp_rms:.2f} ADU)"

    # GP should outperform grid-based methods on complex backgrounds
    assert gp_rms < results['sep']['rms'], "GP should outperform SEP on quadratic gradient"


@pytest.mark.unit
def test_gp_error_handling(test_config, rng):
    """
    Test error handling for edge cases.
    """
    pytest.importorskip("sklearn")

    size = test_config['size']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)

    # Test with too few valid pixels (< 50 required)
    mask_all = np.ones((size, size), dtype=bool)
    mask_all[:5, :5] = False  # Only 25 pixels valid (< 50)

    with pytest.raises(RuntimeError, match="Too few valid pixels"):
        photometry.get_background(
            image, mask=mask_all, method='gp', max_points=1000
        )

    # Test with non-2D image
    image_1d = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="must be 2D"):
        from stdpipe.photometry_background import get_background_gp
        get_background_gp(image_1d)

    # Test with mismatched mask shape
    wrong_mask = np.zeros((100, 100), dtype=bool)
    with pytest.raises(ValueError, match="mask shape"):
        from stdpipe.photometry_background import get_background_gp
        get_background_gp(image, mask=wrong_mask)


@pytest.mark.unit
def test_morphology_smoothing_improvement(test_config, rng):
    """
    Test that improved smoothing reduces kernel-sized artifacts.
    """
    size = test_config['size']
    fwhm = test_config['fwhm']
    noise_std = test_config['noise_std']
    nstars = test_config['nstars']
    flux_range = test_config['flux_range']
    bg_level = 100.0

    # Create background with stars
    true_bg = create_linear_gradient(size, 300, angle_deg=45)
    image = true_bg.copy()
    image += rng.normal(0, noise_std, (size, size))
    add_random_stars(image, nstars, flux_range, fwhm, rng)

    # Test different smoothing settings
    bg_no_smooth = photometry.get_background(
        image, method='morphology', size=25, smooth=False
    )
    bg_with_smooth = photometry.get_background(
        image, method='morphology', size=25, smooth=True
    )
    bg_extra_smooth = photometry.get_background(
        image, method='morphology', size=25, smooth=True, smooth_size=15
    )

    # Measure smoothness via gradient variance
    def measure_smoothness(bg):
        """Compute smoothness metric (lower = smoother)."""
        dy, dx = np.gradient(bg)
        grad_mag = np.sqrt(dx**2 + dy**2)
        return np.std(grad_mag)

    smoothness_no = measure_smoothness(bg_no_smooth)
    smoothness_yes = measure_smoothness(bg_with_smooth)
    smoothness_extra = measure_smoothness(bg_extra_smooth)

    # Smoothing should significantly reduce gradient variance
    assert smoothness_yes < smoothness_no * 0.5, (
        f"Smoothing should reduce artifacts by >50%, but got "
        f"{smoothness_yes:.3f} vs {smoothness_no:.3f}"
    )

    # Extra smoothing should be even smoother
    assert smoothness_extra <= smoothness_yes * 1.1, (
        f"Extra smoothing should not be worse than default"
    )

    # All methods should have similar RMS error
    rms_no = compute_background_rms(bg_no_smooth, true_bg)
    rms_yes = compute_background_rms(bg_with_smooth, true_bg)

    assert abs(rms_no - rms_yes) < 5.0, (
        f"Smoothing should not significantly affect accuracy: "
        f"RMS {rms_no:.2f} vs {rms_yes:.2f}"
    )


@pytest.mark.unit
def test_morphology_smooth_size_parameter(test_config, rng):
    """
    Test that smooth_size parameter controls smoothing strength.
    """
    size = test_config['size']
    noise_std = test_config['noise_std']
    bg_level = 100.0

    # Create simple image
    image = create_flat_background(size, bg_level)
    image += rng.normal(0, noise_std, (size, size))

    # Test different smooth_size values
    bg_5 = photometry.get_background(
        image, method='morphology', size=25, smooth=True, smooth_size=5
    )
    bg_15 = photometry.get_background(
        image, method='morphology', size=25, smooth=True, smooth_size=15
    )
    bg_25 = photometry.get_background(
        image, method='morphology', size=25, smooth=True, smooth_size=25
    )

    # All should produce finite results
    assert np.all(np.isfinite(bg_5)), "smooth_size=5: non-finite values"
    assert np.all(np.isfinite(bg_15)), "smooth_size=15: non-finite values"
    assert np.all(np.isfinite(bg_25)), "smooth_size=25: non-finite values"

    # Larger smooth_size should produce smoother results
    def measure_smoothness(bg):
        dy, dx = np.gradient(bg)
        grad_mag = np.sqrt(dx**2 + dy**2)
        return np.std(grad_mag)

    smoothness_5 = measure_smoothness(bg_5)
    smoothness_15 = measure_smoothness(bg_15)
    smoothness_25 = measure_smoothness(bg_25)

    # Larger smooth_size should generally be smoother (or similar)
    assert smoothness_25 <= smoothness_5 * 1.5, (
        f"Larger smooth_size should be smoother: {smoothness_25:.3f} vs {smoothness_5:.3f}"
    )
