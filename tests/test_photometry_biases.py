"""
Unit tests for photometry biases and systematic errors.

Tests for:
- FWHM-dependent bias in PSF photometry
- Crowding effects with grouped vs ungrouped fitting
- Pixel integration correctness
- Comparison between optimal extraction and PSF photometry backends
"""

import pytest
import numpy as np
from scipy.special import erf
from astropy.table import Table

from stdpipe import photometry_measure, photometry_psf


def create_pixel_integrated_image(size, x, y, flux, fwhm):
    """
    Create an image with pixel-integrated Gaussian PSF.

    This simulates how real CCDs work: photons arrive randomly within pixels
    and the detector integrates over the pixel area. Uses the error function
    to compute the integral of a Gaussian over each pixel.

    Parameters
    ----------
    size : int
        Image size in pixels
    x, y : float
        Source position (sub-pixel precision)
    flux : float
        Total source flux in ADU
    fwhm : float
        PSF FWHM in pixels

    Returns
    -------
    image : ndarray
        2D image with pixel-integrated Gaussian PSF
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Pixel edges (pixels span from i-0.5 to i+0.5)
    x_edges = np.arange(size + 1) - 0.5
    y_edges = np.arange(size + 1) - 0.5

    # Integrate Gaussian over pixel boundaries using CDF (error function)
    sqrt2_sigma = np.sqrt(2) * sigma
    cdf_x = 0.5 * (1 + erf((x_edges - x) / sqrt2_sigma))
    cdf_y = 0.5 * (1 + erf((y_edges - y) / sqrt2_sigma))

    # Flux in each pixel = difference of CDFs at edges
    flux_x = np.diff(cdf_x)
    flux_y = np.diff(cdf_y)

    # 2D PSF (Gaussian is separable)
    image = flux * np.outer(flux_y, flux_x)

    return image


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.mark.unit
@pytest.mark.parametrize("fwhm", [1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
def test_optimal_extraction_no_fwhm_bias(fwhm, rng):
    """
    Test that optimal extraction has no FWHM-dependent bias after pixel integration fix.

    Before the fix, point-sampled Gaussian PSF had systematic bias:
    - FWHM=1.5: +11.6% error
    - FWHM=3.0: +2.7% error
    - FWHM=6.0: +0.8% error

    After the fix using pixel-integrated Gaussian (erf-based):
    - All FWHM values: < 1% error
    """
    size = 301
    flux_true = 3000.0
    noise_std = 2.0
    x_true, y_true = 150.0, 150.0

    # Create pixel-integrated image (realistic)
    image = create_pixel_integrated_image(size, x_true, y_true, flux_true, fwhm)
    image += rng.normal(0, noise_std, (size, size))

    obj = Table()
    obj['x'] = [x_true]
    obj['y'] = [y_true]
    obj['flux'] = [flux_true]
    obj['fluxerr'] = [noise_std]

    # Measure with optimal extraction (uses pixel-integrated PSF)
    result = photometry_measure.measure_objects(
        obj, image,
        aper=5.0,
        fwhm=fwhm,
        optimal=True,
        group_sources=False,  # Single source, no grouping needed
        verbose=False
    )

    flux_meas = result['flux'][0]
    flux_err_percent = (flux_meas - flux_true) / flux_true * 100

    # After pixel integration fix, should have < 1% error for all FWHM
    assert np.isfinite(flux_meas), f"FWHM={fwhm}: Flux is NaN/Inf"
    assert abs(flux_err_percent) < 1.0, (
        f"FWHM={fwhm}: Expected < 1% error, got {flux_err_percent:.2f}% "
        f"(measured {flux_meas:.1f} vs true {flux_true:.1f})"
    )


@pytest.mark.unit
@pytest.mark.parametrize("fwhm", [1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
def test_psf_photometry_no_fwhm_bias(fwhm, rng):
    """
    Test that PSF photometry has no FWHM-dependent bias.

    photutils CircularGaussianSigmaPRF is already pixel-integrated (it's a PRF),
    so should not have FWHM-dependent bias even before any fixes.
    """
    size = 301
    flux_true = 3000.0
    noise_std = 2.0
    x_true, y_true = 150.0, 150.0

    # Create pixel-integrated image (realistic)
    image = create_pixel_integrated_image(size, x_true, y_true, flux_true, fwhm)
    image += rng.normal(0, noise_std, (size, size))

    obj = Table()
    obj['x'] = [x_true]
    obj['y'] = [y_true]
    obj['flux'] = [flux_true]

    # Measure with PSF photometry (uses CircularGaussianSigmaPRF)
    result = photometry_psf.measure_objects_psf(
        obj, image,
        fwhm=fwhm,
        group_sources=False,  # Single source, no grouping needed
        verbose=False
    )

    flux_meas = result['flux'][0]
    flux_err_percent = (flux_meas - flux_true) / flux_true * 100

    # CircularGaussianSigmaPRF is already pixel-integrated, should have < 1% error
    assert np.isfinite(flux_meas), f"FWHM={fwhm}: Flux is NaN/Inf"
    assert abs(flux_err_percent) < 1.0, (
        f"FWHM={fwhm}: Expected < 1% error, got {flux_err_percent:.2f}% "
        f"(measured {flux_meas:.1f} vs true {flux_true:.1f})"
    )


@pytest.mark.unit
@pytest.mark.parametrize("separation_fwhm", [0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
def test_optimal_extraction_crowding_grouped_vs_ungrouped(separation_fwhm, rng):
    """
    Test that grouped optimal extraction dramatically improves accuracy in crowded fields.

    At close separations (< 1.5 FWHM), ungrouped fitting has catastrophic errors:
    - 0.75 FWHM: ~48% error (ungrouped) vs < 1% (grouped)
    - 1.00 FWHM: ~27% error (ungrouped) vs < 1% (grouped)
    - 1.25 FWHM: ~13% error (ungrouped) vs < 1% (grouped)
    - 1.50 FWHM: ~6% error (ungrouped) vs < 10% (grouped) - transition zone

    At wide separations (> 3 FWHM), both methods give identical results.
    """
    size = 301
    fwhm = 3.0
    flux_true = 3000.0
    noise_std = 2.0

    # Create two sources at specified separation
    x1, y1 = 150.0, 150.0
    sep_pix = separation_fwhm * fwhm
    x2, y2 = x1 + sep_pix, y1

    # Create pixel-integrated image (realistic)
    image = create_pixel_integrated_image(size, x1, y1, flux_true, fwhm)
    image += create_pixel_integrated_image(size, x2, y2, flux_true, fwhm)
    image += rng.normal(0, noise_std, (size, size))

    obj = Table()
    obj['x'] = [x1, x2]
    obj['y'] = [y1, y2]
    obj['flux'] = [flux_true, flux_true]
    obj['fluxerr'] = [noise_std, noise_std]

    # Measure with grouped extraction (new default)
    result_grouped = photometry_measure.measure_objects(
        obj.copy(), image,
        aper=5.0,
        fwhm=fwhm,
        optimal=True,
        group_sources=True,
        verbose=False
    )

    # Measure with ungrouped extraction (old default)
    result_ungrouped = photometry_measure.measure_objects(
        obj.copy(), image,
        aper=5.0,
        fwhm=fwhm,
        optimal=True,
        group_sources=False,
        verbose=False
    )

    # Calculate mean absolute errors
    flux_errs_grouped = [
        abs((result_grouped['flux'][i] - flux_true) / flux_true * 100)
        for i in range(2)
    ]
    flux_errs_ungrouped = [
        abs((result_ungrouped['flux'][i] - flux_true) / flux_true * 100)
        for i in range(2)
    ]
    mean_err_grouped = np.mean(flux_errs_grouped)
    mean_err_ungrouped = np.mean(flux_errs_ungrouped)

    # Grouped should always be good (< 10% at all separations, < 1% at > 1.5 FWHM)
    # At 1.5 FWHM we're in the transition zone, so allow higher error
    max_err = 10.0 if separation_fwhm <= 1.5 else 1.0
    assert mean_err_grouped < max_err, (
        f"Sep={separation_fwhm:.2f} FWHM: Grouped extraction error {mean_err_grouped:.2f}% > {max_err}%"
    )

    # At close separations, ungrouped should be much worse
    if separation_fwhm < 1.5:
        assert mean_err_ungrouped > 5.0, (
            f"Sep={separation_fwhm:.2f} FWHM: Ungrouped should have > 5% error, got {mean_err_ungrouped:.2f}%"
        )
        # Grouped should be dramatically better
        improvement_factor = mean_err_ungrouped / mean_err_grouped
        assert improvement_factor > 5.0, (
            f"Sep={separation_fwhm:.2f} FWHM: Expected > 5× improvement, got {improvement_factor:.1f}×"
        )

    # At wide separations (> 3 FWHM), both should be similar
    if separation_fwhm >= 3.0:
        assert abs(mean_err_grouped - mean_err_ungrouped) < 0.5, (
            f"Sep={separation_fwhm:.2f} FWHM: At wide separation, grouped ({mean_err_grouped:.2f}%) "
            f"and ungrouped ({mean_err_ungrouped:.2f}%) should be similar"
        )


@pytest.mark.unit
@pytest.mark.parametrize("separation_fwhm", [0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
def test_psf_photometry_crowding_grouped_vs_ungrouped(separation_fwhm, rng):
    """
    Test that grouped PSF fitting dramatically improves accuracy in crowded fields.

    Similar to optimal extraction test, but for photutils PSF photometry backend.
    """
    size = 301
    fwhm = 3.0
    flux_true = 3000.0
    noise_std = 2.0

    # Create two sources at specified separation
    x1, y1 = 150.0, 150.0
    sep_pix = separation_fwhm * fwhm
    x2, y2 = x1 + sep_pix, y1

    # Create pixel-integrated image (realistic)
    image = create_pixel_integrated_image(size, x1, y1, flux_true, fwhm)
    image += create_pixel_integrated_image(size, x2, y2, flux_true, fwhm)
    image += rng.normal(0, noise_std, (size, size))

    obj = Table()
    obj['x'] = [x1, x2]
    obj['y'] = [y1, y2]
    obj['flux'] = [flux_true, flux_true]

    # Measure with grouped PSF fitting (new default)
    result_grouped = photometry_psf.measure_objects_psf(
        obj.copy(), image,
        fwhm=fwhm,
        group_sources=True,
        verbose=False
    )

    # Measure with ungrouped PSF fitting (old default)
    result_ungrouped = photometry_psf.measure_objects_psf(
        obj.copy(), image,
        fwhm=fwhm,
        group_sources=False,
        verbose=False
    )

    # Calculate mean absolute errors
    flux_errs_grouped = [
        abs((result_grouped['flux'][i] - flux_true) / flux_true * 100)
        for i in range(2)
    ]
    flux_errs_ungrouped = [
        abs((result_ungrouped['flux'][i] - flux_true) / flux_true * 100)
        for i in range(2)
    ]
    mean_err_grouped = np.mean(flux_errs_grouped)
    mean_err_ungrouped = np.mean(flux_errs_ungrouped)

    # Grouped should always be good (< 2% at practical separations, allowing some tolerance)
    assert mean_err_grouped < 2.0, (
        f"Sep={separation_fwhm:.2f} FWHM: Grouped PSF fitting error {mean_err_grouped:.2f}% > 2%"
    )

    # At close separations, ungrouped should be much worse
    if separation_fwhm < 1.5:
        assert mean_err_ungrouped > 5.0, (
            f"Sep={separation_fwhm:.2f} FWHM: Ungrouped should have > 5% error, got {mean_err_ungrouped:.2f}%"
        )
        # Grouped should be dramatically better
        improvement_factor = mean_err_ungrouped / mean_err_grouped
        assert improvement_factor > 5.0, (
            f"Sep={separation_fwhm:.2f} FWHM: Expected > 5× improvement, got {improvement_factor:.1f}×"
        )

    # At wide separations (> 3 FWHM), both should be similar
    if separation_fwhm >= 3.0:
        assert abs(mean_err_grouped - mean_err_ungrouped) < 1.0, (
            f"Sep={separation_fwhm:.2f} FWHM: At wide separation, grouped ({mean_err_grouped:.2f}%) "
            f"and ungrouped ({mean_err_ungrouped:.2f}%) should be similar"
        )


@pytest.mark.unit
@pytest.mark.parametrize("fwhm", [1.5, 3.0, 6.0])
def test_psf_stamp_pixel_integration(fwhm):
    """
    Test that _get_psf_stamp_at_position correctly implements pixel integration.

    The pixel-integrated PSF should have a lower peak than point-sampled PSF:
    - FWHM=1.5: ~18% peak reduction
    - FWHM=3.0: ~5% peak reduction
    - FWHM=6.0: ~1% peak reduction
    """
    from stdpipe.photometry_measure import _get_psf_stamp_at_position

    # Create PSF stamp using pixel-integrated function
    psf_integrated = _get_psf_stamp_at_position(fwhm, 100.0, 100.0)

    # Create old-style point-sampled PSF for comparison
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    size = int(np.ceil(fwhm * 3)) * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size]
    psf_point_sampled = np.exp(-((xx - size//2)**2 + (yy - size//2)**2) / (2 * sigma**2))
    psf_point_sampled /= np.sum(psf_point_sampled)

    # Compare peaks
    peak_integrated = np.max(psf_integrated)
    peak_point_sampled = np.max(psf_point_sampled)
    peak_diff_percent = (peak_integrated - peak_point_sampled) / peak_point_sampled * 100

    # Expected peak reductions (from theory)
    expected_reductions = {1.5: -17.8, 3.0: -5.0, 6.0: -1.3}
    expected = expected_reductions[fwhm]

    # Check that peak reduction is as expected (within 1%)
    assert abs(peak_diff_percent - expected) < 1.0, (
        f"FWHM={fwhm}: Expected {expected:.1f}% peak reduction, got {peak_diff_percent:.1f}%"
    )

    # Check normalization
    total = np.sum(psf_integrated)
    assert abs(total - 1.0) < 1e-6, (
        f"FWHM={fwhm}: PSF should sum to 1.0, got {total:.10f}"
    )


@pytest.mark.unit
def test_both_backends_comparison_isolated(rng):
    """
    Test that both backends (optimal extraction and PSF photometry) give similar results
    for isolated sources across different FWHM values.

    After the fixes, both should have:
    - No FWHM-dependent bias (< 1% error)
    - Similar accuracy to each other
    """
    size = 301
    flux_true = 3000.0
    noise_std = 2.0
    x_true, y_true = 150.0, 150.0
    fwhm_values = [1.5, 3.0, 6.0]

    for fwhm in fwhm_values:
        # Create pixel-integrated image (realistic)
        image = create_pixel_integrated_image(size, x_true, y_true, flux_true, fwhm)
        image += rng.normal(0, noise_std, (size, size))

        obj = Table()
        obj['x'] = [x_true]
        obj['y'] = [y_true]
        obj['flux'] = [flux_true]
        obj['fluxerr'] = [noise_std]

        # Optimal extraction
        result_opt = photometry_measure.measure_objects(
            obj.copy(), image,
            aper=5.0,
            fwhm=fwhm,
            optimal=True,
            group_sources=False,
            verbose=False
        )

        # PSF photometry
        result_psf = photometry_psf.measure_objects_psf(
            obj.copy(), image,
            fwhm=fwhm,
            group_sources=False,
            verbose=False
        )

        flux_opt = result_opt['flux'][0]
        flux_psf = result_psf['flux'][0]
        err_opt_percent = (flux_opt - flux_true) / flux_true * 100
        err_psf_percent = (flux_psf - flux_true) / flux_true * 100

        # Both should have < 1% error
        assert abs(err_opt_percent) < 1.0, (
            f"FWHM={fwhm}: Optimal extraction error {err_opt_percent:.2f}% > 1%"
        )
        assert abs(err_psf_percent) < 1.0, (
            f"FWHM={fwhm}: PSF photometry error {err_psf_percent:.2f}% > 1%"
        )

        # Both should be similar to each other (within 1%)
        assert abs(err_opt_percent - err_psf_percent) < 1.0, (
            f"FWHM={fwhm}: Backends differ by {abs(err_opt_percent - err_psf_percent):.2f}% "
            f"(optimal: {err_opt_percent:.2f}%, PSF: {err_psf_percent:.2f}%)"
        )


@pytest.mark.unit
def test_both_backends_comparison_crowded(rng):
    """
    Test that both backends give excellent results in crowded fields with grouped fitting.

    At moderate separations (1.0-2.0 FWHM), both backends with grouped fitting should have good accuracy.
    At 1.5 FWHM (transition zone), optimal extraction may have ~10% error, but PSF photometry remains < 2%.
    At wider separations (2.0 FWHM), both should have < 1% error.
    """
    size = 301
    fwhm = 3.0
    flux_true = 3000.0
    noise_std = 2.0
    separation_fwhm_values = [1.0, 1.5, 2.0]

    for separation_fwhm in separation_fwhm_values:
        # Create two sources at specified separation
        x1, y1 = 150.0, 150.0
        sep_pix = separation_fwhm * fwhm
        x2, y2 = x1 + sep_pix, y1

        # Create pixel-integrated image (realistic)
        image = create_pixel_integrated_image(size, x1, y1, flux_true, fwhm)
        image += create_pixel_integrated_image(size, x2, y2, flux_true, fwhm)
        image += rng.normal(0, noise_std, (size, size))

        obj = Table()
        obj['x'] = [x1, x2]
        obj['y'] = [y1, y2]
        obj['flux'] = [flux_true, flux_true]
        obj['fluxerr'] = [noise_std, noise_std]

        # Optimal extraction with grouped fitting
        result_opt = photometry_measure.measure_objects(
            obj.copy(), image,
            aper=5.0,
            fwhm=fwhm,
            optimal=True,
            group_sources=True,
            verbose=False
        )

        # PSF photometry with grouped fitting
        result_psf = photometry_psf.measure_objects_psf(
            obj.copy(), image,
            fwhm=fwhm,
            group_sources=True,
            verbose=False
        )

        # Calculate mean absolute errors
        mean_err_opt = np.mean([
            abs((result_opt['flux'][i] - flux_true) / flux_true * 100)
            for i in range(2)
        ])
        mean_err_psf = np.mean([
            abs((result_psf['flux'][i] - flux_true) / flux_true * 100)
            for i in range(2)
        ])

        # Both should have excellent accuracy with grouped fitting
        # At 1.5 FWHM we're in the transition zone, so allow higher error
        max_err_opt = 10.0 if separation_fwhm <= 1.5 else 1.0
        max_err_psf = 2.0
        assert mean_err_opt < max_err_opt, (
            f"Sep={separation_fwhm:.2f} FWHM: Optimal extraction (grouped) error {mean_err_opt:.2f}% > {max_err_opt}%"
        )
        assert mean_err_psf < max_err_psf, (
            f"Sep={separation_fwhm:.2f} FWHM: PSF photometry (grouped) error {mean_err_psf:.2f}% > {max_err_psf}%"
        )


@pytest.mark.unit
def test_default_is_grouped_extraction():
    """
    Test that grouped fitting is the default for optimal extraction.

    This is a regression test to ensure the safe default is maintained.
    """
    import inspect

    # Get the default value of group_sources parameter
    sig = inspect.signature(photometry_measure.measure_objects)
    group_sources_default = sig.parameters['group_sources'].default

    assert group_sources_default is True, (
        "group_sources should default to True for safe behavior in crowded fields"
    )


@pytest.mark.unit
def test_default_is_grouped_psf():
    """
    Test that grouped fitting is the default for PSF photometry.

    This is a regression test to ensure the safe default is maintained.
    """
    import inspect

    # Get the default value of group_sources parameter
    sig = inspect.signature(photometry_psf.measure_objects_psf)
    group_sources_default = sig.parameters['group_sources'].default

    assert group_sources_default is True, (
        "group_sources should default to True for safe behavior in crowded fields"
    )


# ============================================================================
# Aperture vs optimal vs deblended-aperture on Moffat profiles
# ============================================================================
#
# Sources are drawn as pixel-integrated Moffat profiles (β=2.5), which is a
# realistic stellar profile and DOES NOT match the Gaussian PSF that
# optimal extraction and ``measure_aperture_deblended`` (when given a
# Gaussian psf-dict) assume internally. The tests sweep FWHM and pair
# separation and report the bias of all three methods.

def _moffat_image(positions, fluxes, size, fwhm, beta=2.5, oversample=8):
    """Pixel-integrated Moffat image with one or more sources."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    dyy, dxx = np.meshgrid(sub, sub, indexing='ij')
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=np.float64)
    norm = (beta - 1.0) / (np.pi * alpha ** 2)
    for (x_src, y_src), flux in zip(positions, fluxes):
        rr2 = (
            (xx[:, :, None, None] - x_src + dxx[None, None, :, :]) ** 2
            + (yy[:, :, None, None] - y_src + dyy[None, None, :, :]) ** 2
        )
        moffat = norm * (1.0 + rr2 / alpha ** 2) ** (-beta)
        image += flux * moffat.mean(axis=(2, 3))
    return image


def _gaussian_psf_dict(fwhm, size=51):
    """Position-invariant unit-flux Gaussian PSF dict, sampling=1."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-((xx - half) ** 2 + (yy - half) ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return {
        'data': g[None, :, :], 'width': size, 'height': size,
        'sampling': 1.0, 'degree': 0, 'ncoeffs': 1,
        'x0': 0.0, 'y0': 0.0, 'sx': 1.0, 'sy': 1.0, 'type': 'epsf',
    }


def _measure_three_ways(image, x, y, *, aper_fwhm, fwhm, target_idx,
                        flux_seed, flux_psf, flux_psf_err):
    """Run aperture / optimal / deblended on the same detection list and
    return the three measured fluxes for ``target_idx``. ``aper_fwhm`` is
    in FWHM units; both ``measure_objects`` and
    ``measure_aperture_deblended`` multiply by ``fwhm`` internally."""
    obj = Table()
    obj['x'] = np.asarray(x, float)
    obj['y'] = np.asarray(y, float)
    obj['flux'] = np.asarray(flux_seed, float)
    obj['fluxerr'] = np.full(len(x), 1.0)

    res_aper = photometry_measure.measure_objects(
        obj, image, aper=aper_fwhm, fwhm=fwhm,
        optimal=False, group_sources=False, verbose=False,
    )
    res_opt = photometry_measure.measure_objects(
        obj, image, aper=aper_fwhm, fwhm=fwhm,
        optimal=True, group_sources=True, verbose=False,
    )
    psf = _gaussian_psf_dict(fwhm)
    res_deb = photometry_measure.measure_aperture_deblended(
        image, x, y, aper=aper_fwhm, fwhm=fwhm, psf=psf,
        flux_seed=flux_seed, flux_psf=flux_psf, flux_psf_err=flux_psf_err,
        target=np.array([i in target_idx for i in range(len(x))]),
        aperture_correction='ratio_field' if len(x) >= 6 else 'fixed',
        propagate_neighbour_errors=False,
    )
    return (
        np.asarray(res_aper['flux'])[list(target_idx)],
        np.asarray(res_opt['flux'])[list(target_idx)],
        np.asarray(res_deb['flux']),
    )


@pytest.mark.unit
@pytest.mark.parametrize("fwhm", [2.0, 3.0, 4.0, 6.0])
def test_moffat_isolated_bias_three_methods(fwhm):
    """ISOLATED Moffat source: report bias of aperture, optimal extraction
    (Gaussian-PSF assumption) and deblended (Gaussian-PSF assumption).
    All three must give finite results; aperture and deblended must agree
    closely (the deblending terms cancel for a single source); optimal
    will pick up FWHM-dependent PSF-mismatch bias."""
    size = 81
    flux_true = 5000.0
    x_true = y_true = size / 2.0
    image = _moffat_image([(x_true, y_true)], [flux_true], size, fwhm)

    f_ap, f_opt, f_deb = _measure_three_ways(
        image, np.array([x_true]), np.array([y_true]),
        aper_fwhm=2.0, fwhm=fwhm, target_idx=(0,),
        flux_seed=np.array([float(flux_true)]),
        flux_psf=np.array([float(flux_true)]),
        flux_psf_err=np.array([1.0]),
    )

    bias_ap = (f_ap[0] - flux_true) / flux_true
    bias_opt = (f_opt[0] - flux_true) / flux_true
    bias_deb = (f_deb[0] - flux_true) / flux_true
    print(f"\n[isolated, FWHM={fwhm}] aper={f_ap[0]:.1f}  "
          f"opt={f_opt[0]:.1f}  deb={f_deb[0]:.1f}  truth={flux_true}")
    print(f"  bias [%] aper={100*bias_ap:+.2f}  opt={100*bias_opt:+.2f}  "
          f"deb={100*bias_deb:+.2f}")

    assert np.isfinite(f_ap[0])
    assert np.isfinite(f_opt[0])
    assert np.isfinite(f_deb[0])

    # Deblended ≈ aperture for an isolated source (model-correction terms
    # cancel up to sub-pixel rounding noise between sep.sum_circle and
    # our enclosed_psf_fraction grid).
    np.testing.assert_allclose(f_deb[0], f_ap[0], rtol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize("separation_fwhm", [1.0, 1.5, 2.0, 3.0, 5.0])
def test_moffat_pair_bias_three_methods(separation_fwhm):
    """CROWDED Moffat pair: report bias of all three methods on the
    target star (which has a brighter neighbour at ``separation_fwhm``
    FWHMs along x). The aperture sum is contaminated; optimal extraction
    has both contamination and PSF-mismatch effects; deblended subtracts
    the neighbour and should land much closer to the isolated baseline."""
    size = 121
    fwhm = 3.0
    flux_target = 3000.0
    flux_neighbour = 6000.0
    x1, y1 = 60.0, 60.0
    x2, y2 = x1 + separation_fwhm * fwhm, y1
    image = _moffat_image(
        [(x1, y1), (x2, y2)], [flux_target, flux_neighbour], size, fwhm,
    )
    image_iso = _moffat_image([(x1, y1)], [flux_target], size, fwhm)

    f_ap, f_opt, f_deb = _measure_three_ways(
        image,
        np.array([x1, x2]), np.array([y1, y2]),
        aper_fwhm=1.5, fwhm=fwhm, target_idx=(0,),
        flux_seed=np.array([flux_target, flux_neighbour]),
        flux_psf=np.array([flux_target, flux_neighbour]),
        flux_psf_err=np.array([10.0, 10.0]),
    )
    f_iso_ap, _, _ = _measure_three_ways(
        image_iso,
        np.array([x1]), np.array([y1]),
        aper_fwhm=1.5, fwhm=fwhm, target_idx=(0,),
        flux_seed=np.array([flux_target]),
        flux_psf=np.array([flux_target]),
        flux_psf_err=np.array([10.0]),
    )
    baseline = float(f_iso_ap[0])

    bias_ap = (float(f_ap[0]) - baseline) / baseline
    bias_opt = (float(f_opt[0]) - baseline) / baseline
    bias_deb = (float(f_deb[0]) - baseline) / baseline

    print(f"\n[pair sep={separation_fwhm} FWHM, FWHM={fwhm}, "
          f"target={flux_target}, neighbour={flux_neighbour}]")
    print(f"  baseline (isolated aperture) = {baseline:.1f}")
    print(f"  measured aper={f_ap[0]:.1f} ({100*bias_ap:+.2f}%)  "
          f"opt={f_opt[0]:.1f} ({100*bias_opt:+.2f}%)  "
          f"deb={f_deb[0]:.1f} ({100*bias_deb:+.2f}%)")

    assert all(np.isfinite([f_ap[0], f_opt[0], f_deb[0]]))

    if separation_fwhm <= 2.0:
        # In the crowded regime the deblended flux must be at least
        # twice as close to the isolated baseline as the plain aperture.
        assert abs(bias_deb) < 0.5 * abs(bias_ap), (
            f"deblending should beat aperture in the crowded regime "
            f"(sep={separation_fwhm} FWHM): aper bias {100*bias_ap:.2f}% "
            f"vs deb bias {100*bias_deb:.2f}%"
        )
    else:
        # In the uncrowded regime the deblended flux should track the
        # plain aperture (no measurable advantage, no harm done).
        np.testing.assert_allclose(f_deb[0], f_ap[0], rtol=0.02)
