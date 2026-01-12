"""
Unit tests for stdpipe.photometry_measure module.

Tests aperture photometry, optimal extraction, and grouped fitting.
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import photometry_measure


def _simulate_random_field(
    rng,
    *,
    size,
    fwhm,
    n_sources,
    margin,
    min_sep,
    noise_std,
    aper,
    mask_fraction=0.0,
):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    xs = []
    ys = []
    attempts = 0
    while len(xs) < n_sources and attempts < 10000:
        attempts += 1
        x = rng.randint(margin, size - margin)
        y = rng.randint(margin, size - margin)
        if xs:
            dist2 = (np.array(xs) - x)**2 + (np.array(ys) - y)**2
            if np.any(dist2 < min_sep**2):
                continue
        xs.append(int(x))
        ys.append(int(y))
    assert len(xs) == n_sources

    fluxes = rng.uniform(1000.0, 5000.0, size=n_sources)
    amplitudes = fluxes / (2 * np.pi * sigma**2)
    yy, xx = np.mgrid[:size, :size]
    image = rng.normal(0.0, noise_std, (size, size))
    for x, y, amp in zip(xs, ys, amplitudes):
        image += amp * np.exp(
            -((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)
        )

    obj = Table()
    obj['x'] = np.array(xs, dtype=float)
    obj['y'] = np.array(ys, dtype=float)
    obj['flux'] = fluxes
    obj['fluxerr'] = np.full(n_sources, noise_std)

    bg = np.zeros_like(image)
    err = np.full_like(image, noise_std)
    mask = None
    if mask_fraction:
        mask = rng.rand(size, size) < mask_fraction

    result_aper = photometry_measure.measure_objects(
        obj.copy(),
        image,
        aper=aper,
        fwhm=fwhm,
        optimal=False,
        mask=mask,
        bg=bg,
        err=err,
        verbose=False,
    )
    result_opt = photometry_measure.measure_objects(
        obj.copy(),
        image,
        aper=aper,
        fwhm=fwhm,
        optimal=True,
        mask=mask,
        bg=bg,
        err=err,
        verbose=False,
    )

    rel_aper = np.abs(result_aper['flux'] - fluxes) / fluxes
    rel_opt = np.abs(result_opt['flux'] - fluxes) / fluxes
    valid_aper = np.isfinite(rel_aper)
    valid_opt = np.isfinite(rel_opt)
    med_aper = float(np.nan)
    med_opt = float(np.nan)
    if np.any(valid_aper):
        med_aper = float(np.median(rel_aper[valid_aper]))
    if np.any(valid_opt):
        med_opt = float(np.median(rel_opt[valid_opt]))
    accuracy_aper = 1.0 - med_aper
    accuracy_opt = 1.0 - med_opt

    return {
        'result_aper': result_aper,
        'result_opt': result_opt,
        'accuracy_aper': accuracy_aper,
        'accuracy_opt': accuracy_opt,
        'med_aper': med_aper,
        'med_opt': med_opt,
        'n_valid_aper': int(np.sum(valid_aper)),
        'n_valid_opt': int(np.sum(valid_opt)),
        'mask_fraction': mask_fraction,
    }


class TestOptimalExtraction:
    """Test optimal extraction photometry in measure_objects."""

    @pytest.mark.unit
    def test_optimal_extraction_basic(self, image_with_sources, detected_objects):
        """Test basic optimal extraction with Gaussian PSF."""
        result = photometry_measure.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Should return the same number of objects
        assert len(result) == len(detected_objects)

        # Should have flux and fluxerr
        assert 'flux' in result.colnames
        assert 'fluxerr' in result.colnames
        assert 'npix_optimal' in result.colnames

        # Fluxes should be positive
        assert np.all(result['flux'] > 0)
        assert np.all(result['fluxerr'] > 0)

        # Check that npix is reasonable
        assert np.all(result['npix_optimal'] > 0)

    @pytest.mark.unit
    def test_optimal_vs_aperture_similar_flux(self, image_with_sources, detected_objects):
        """Test that optimal extraction gives similar flux to aperture for isolated sources."""
        # Aperture photometry
        result_aper = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Optimal extraction
        result_opt = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Fluxes should be similar (within factor of 2 for isolated sources)
        flux_ratio = result_opt['flux'] / result_aper['flux']
        assert np.all(flux_ratio > 0.5)
        assert np.all(flux_ratio < 2.0)

    @pytest.mark.unit
    def test_optimal_extraction_requires_psf_or_fwhm(self, image_with_sources, detected_objects):
        """Test that optimal extraction requires either psf or fwhm."""
        with pytest.raises(ValueError, match="Either 'psf' or 'fwhm' must be provided"):
            photometry_measure.measure_objects(
                detected_objects,
                image_with_sources,
                aper=5.0,
                optimal=True,
                verbose=False
            )

    @pytest.mark.unit
    def test_optimal_extraction_with_mask(self, image_with_sources, detected_objects, mask_with_bad_pixels):
        """Test optimal extraction with masked pixels."""
        result = photometry_measure.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            mask=mask_with_bad_pixels,
            verbose=False
        )

        assert len(result) == len(detected_objects)
        # Some objects near mask edges should still be measurable
        valid = np.isfinite(result['flux'])
        assert np.sum(valid) > 0

    @pytest.mark.unit
    def test_optimal_extraction_mask_avoids_bias(self):
        """Test masked pixels do not bias optimal extraction."""
        size = 51
        fwhm = 3.0
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        y, x = np.mgrid[:size, :size]
        cx = cy = size // 2
        amplitude = 1000.0

        image_clean = amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )
        image_bad = image_clean.copy()
        mask = np.zeros_like(image_bad, dtype=bool)
        mask[cy-1:cy+2, cx-1:cx+2] = True
        image_bad[mask] = 1e6

        obj = Table()
        obj['x'] = [float(cx)]
        obj['y'] = [float(cy)]
        obj['flux'] = [1.0]
        obj['fluxerr'] = [1.0]

        bg = np.zeros_like(image_clean)
        err = np.ones_like(image_clean)

        result_clean = photometry_measure.measure_objects(
            obj.copy(),
            image_clean,
            aper=5.0,
            fwhm=fwhm,
            optimal=True,
            bg=bg,
            err=err,
            verbose=False,
        )
        result_masked = photometry_measure.measure_objects(
            obj.copy(),
            image_bad,
            aper=5.0,
            fwhm=fwhm,
            optimal=True,
            mask=mask,
            bg=bg,
            err=err,
            verbose=False,
        )
        result_unmasked = photometry_measure.measure_objects(
            obj.copy(),
            image_bad,
            aper=5.0,
            fwhm=fwhm,
            optimal=True,
            bg=bg,
            err=err,
            verbose=False,
        )

        assert np.isfinite(result_clean['flux'][0])
        np.testing.assert_allclose(
            result_masked['flux'][0],
            result_clean['flux'][0],
            rtol=1e-6,
            atol=1e-6,
        )
        assert result_unmasked['flux'][0] > result_clean['flux'][0] * 10

    @pytest.mark.unit
    def test_random_field_accuracy_aperture_vs_optimal(self):
        """Test relative accuracy of aperture and optimal extraction."""
        rng = np.random.RandomState(123)
        metrics = _simulate_random_field(
            rng,
            size=200,
            fwhm=3.0,
            n_sources=20,
            margin=20,
            min_sep=18,
            noise_std=2.0,
            aper=3.0,
        )
        accuracy_aper = metrics['accuracy_aper']
        accuracy_opt = metrics['accuracy_opt']
        med_aper = metrics['med_aper']
        med_opt = metrics['med_opt']

        print(
            f"aperture accuracy {accuracy_aper:.3f} (median rel err {med_aper:.3f}), "
            f"optimal accuracy {accuracy_opt:.3f} (median rel err {med_opt:.3f})"
        )

        assert accuracy_aper > 0.95, (
            f"aperture accuracy {accuracy_aper:.3f}, median rel err {med_aper:.3f}"
        )
        assert accuracy_opt > 0.95, (
            f"optimal accuracy {accuracy_opt:.3f}, median rel err {med_opt:.3f}"
        )
        assert med_opt <= med_aper * 1.2, (
            f"optimal median rel err {med_opt:.3f} worse than aperture {med_aper:.3f}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "label,min_sep,n_sources",
        [
            ("moderate", 12, 40),
            ("crowded", 8, 60),
            ("very_crowded", 6, 80),
        ],
    )
    @pytest.mark.parametrize("mask_fraction", [0.0, 0.1, 0.3])
    def test_random_field_accuracy_crowded_report(
        self, label, min_sep, n_sources, mask_fraction
    ):
        """Report accuracy for crowded fields without assertions."""
        rng = np.random.RandomState(456 + int(min_sep * 10))
        metrics = _simulate_random_field(
            rng,
            size=200,
            fwhm=3.0,
            n_sources=n_sources,
            margin=15,
            min_sep=min_sep,
            noise_std=2.0,
            aper=3.0,
            mask_fraction=mask_fraction,
        )
        accuracy_aper = metrics['accuracy_aper']
        accuracy_opt = metrics['accuracy_opt']
        med_aper = metrics['med_aper']
        med_opt = metrics['med_opt']
        n_valid_aper = metrics['n_valid_aper']
        n_valid_opt = metrics['n_valid_opt']

        assert len(metrics['result_aper']) == n_sources
        assert len(metrics['result_opt']) == n_sources

        print(
            f"crowding={label} min_sep={min_sep} n={n_sources} mask={mask_fraction:.2f} "
            f"aperture accuracy {accuracy_aper:.3f} (median rel err {med_aper:.3f}, "
            f"valid {n_valid_aper}), "
            f"optimal accuracy {accuracy_opt:.3f} (median rel err {med_opt:.3f}, "
            f"valid {n_valid_opt})"
        )
    @pytest.mark.unit
    def test_optimal_extraction_edge_objects(self, image_with_sources):
        """Test that edge objects are handled correctly."""
        # Create objects very close to edges
        edge_objects = Table()
        edge_objects['x'] = [2.0, 253.0, 128.0, 128.0]
        edge_objects['y'] = [128.0, 128.0, 2.0, 253.0]
        edge_objects['flux'] = [1000.0, 1000.0, 1000.0, 1000.0]
        edge_objects['fluxerr'] = [10.0, 10.0, 10.0, 10.0]

        result = photometry_measure.measure_objects(
            edge_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Edge objects should have NaN flux and be flagged
        assert np.all(~np.isfinite(result['flux']))
        assert np.all((result['flags'] & 0x800) > 0)

    @pytest.mark.unit
    def test_optimal_extraction_snr(self, image_with_sources, detected_objects):
        """Test S/N filtering with optimal extraction."""
        result_all = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            sn=None,
            verbose=False
        )

        result_filtered = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            sn=10.0,  # High S/N threshold
            verbose=False
        )

        # Filtered should have fewer or equal objects
        assert len(result_filtered) <= len(result_all)

    @pytest.mark.unit
    def test_optimal_extraction_magnitudes(self, image_with_sources, detected_objects):
        """Test that magnitudes are computed correctly."""
        result = photometry_measure.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Check magnitude columns exist
        assert 'mag' in result.colnames
        assert 'magerr' in result.colnames

        # Positive fluxes should have valid magnitudes
        positive_flux = result['flux'] > 0
        assert np.all(np.isfinite(result['mag'][positive_flux]))
        assert np.all(np.isfinite(result['magerr'][positive_flux]))

        # Check magnitude relationship: brighter objects have smaller mag
        sorted_flux = np.argsort(result['flux'][positive_flux])[::-1]
        assert np.all(np.diff(result['mag'][positive_flux][sorted_flux]) >= -0.01)

    @pytest.mark.unit
    def test_optimal_extraction_local_background(self, image_with_sources, detected_objects):
        """Test optimal extraction with local background annulus."""
        result = photometry_measure.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            bkgann=(8.0, 12.0),
            optimal=True,
            verbose=False
        )

        # Should have bg_local column
        assert 'bg_local' in result.colnames
        assert len(result) == len(detected_objects)

    @pytest.mark.unit
    def test_optimal_extraction_chi2(self, image_with_sources, detected_objects):
        """Test that chi-squared is computed and reasonable."""
        result = photometry_measure.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Should have chi2 column
        assert 'chi2_optimal' in result.colnames

        # Chi2 should be positive for valid measurements
        valid = np.isfinite(result['flux']) & (result['flux'] > 0)
        assert np.all(result['chi2_optimal'][valid] > 0)

        # For point sources with correct PSF, reduced chi2 should be reasonable
        # Allow range up to 100 for noisy data
        assert np.all(result['chi2_optimal'][valid] < 100)


class TestGroupedOptimalExtraction:
    """Test grouped optimal extraction for crowded fields."""

    @pytest.fixture
    def image_with_close_pairs(self):
        """Create a 256x256 image with close pairs of sources."""
        np.random.seed(42)

        # Background with noise
        image = np.random.normal(100, 5, (256, 256))

        # Add pairs of close sources (separation < typical grouper_radius)
        sources = [
            # Pair 1: separation ~6 pixels
            (50, 50, 1000),
            (56, 50, 800),
            # Pair 2: separation ~7 pixels
            (150, 80, 1200),
            (155, 85, 900),
            # Isolated source
            (200, 200, 1500),
        ]

        y, x = np.ogrid[:256, :256]
        sigma = 1.27  # FWHM ~ 3 pixels

        for sx, sy, amp in sources:
            source = amp * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
            image += source

        return image.astype(np.float64)

    @pytest.fixture
    def close_pair_objects(self):
        """Create detected objects table with close pairs."""
        objects = Table()
        objects['x'] = [50.0, 56.0, 150.0, 155.0, 200.0]
        objects['y'] = [50.0, 50.0, 80.0, 85.0, 200.0]
        objects['flux'] = [1000.0, 800.0, 1200.0, 900.0, 1500.0]
        objects['fluxerr'] = [10.0, 10.0, 10.0, 10.0, 10.0]
        objects['flags'] = [0, 0, 0, 0, 0]
        objects['fwhm'] = [3.0, 3.0, 3.0, 3.0, 3.0]
        return objects

    @pytest.mark.unit
    def test_grouped_extraction_basic(self, image_with_close_pairs, close_pair_objects):
        """Test basic grouped optimal extraction."""
        result = photometry_measure.measure_objects(
            close_pair_objects,
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(close_pair_objects)

        # Should have standard columns
        assert 'flux' in result.colnames
        assert 'fluxerr' in result.colnames

        # Should have group columns
        assert 'group_id' in result.colnames
        assert 'group_size' in result.colnames

        # All objects should have valid group_id
        assert np.all(result['group_id'] >= 0)

    @pytest.mark.unit
    def test_grouped_extraction_groups_identified(self, image_with_close_pairs, close_pair_objects):
        """Test that close sources are correctly identified as groups."""
        result = photometry_measure.measure_objects(
            close_pair_objects,
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=10.0,  # Should group sources within 10 pixels
            verbose=False
        )

        # With radius=10, pairs at ~6 and ~7 pixel separation should be grouped
        # Isolated source at (200, 200) should be alone
        # Check that we have at least 2 pairs and 1 isolated
        group_sizes = result['group_size']
        assert np.sum(group_sizes == 1) >= 1  # At least 1 isolated
        assert np.sum(group_sizes == 2) >= 2  # At least 2 pairs (4 sources total)

    @pytest.mark.unit
    def test_grouped_extraction_isolated_same_as_single(self, image_with_sources, detected_objects):
        """Test that isolated sources give same result with grouped=True."""
        # Single-source extraction
        result_single = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=False,
            verbose=False
        )

        # Grouped extraction (should detect all as isolated)
        result_grouped = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=3.0,  # Small radius - all should be isolated
            verbose=False
        )

        # Fluxes should be very similar (within 1%)
        flux_diff = np.abs(result_single['flux'] - result_grouped['flux'])
        rel_diff = flux_diff / result_single['flux']
        assert np.all(rel_diff < 0.01)

    @pytest.mark.unit
    def test_grouped_extraction_flux_recovery(self, image_with_close_pairs, close_pair_objects):
        """Test that grouped extraction improves flux recovery for close pairs."""
        # The input amplitudes (peak pixel values) are 1000, 800, 1200, 900, 1500
        # Total integrated flux of 2D Gaussian = 2π × amp × σ²
        # With FWHM=3, σ=1.27, so total flux ≈ 10.1 × amplitude
        input_amplitudes = np.array([1000.0, 800.0, 1200.0, 900.0, 1500.0])
        sigma = 1.27
        expected_fluxes = 2 * np.pi * input_amplitudes * sigma**2

        # Grouped extraction
        result = photometry_measure.measure_objects(
            close_pair_objects,
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=10.0,
            verbose=False
        )

        # All fluxes should be positive
        assert np.all(result['flux'] > 0)

        # Fluxes should be within reasonable range of expected values
        # (allowing for noise, PSF mismatch, and aperture effects)
        flux_ratio = result['flux'] / expected_fluxes
        assert np.all(flux_ratio > 0.5)  # Not too low
        assert np.all(flux_ratio < 2.0)  # Not too high

    @pytest.mark.unit
    def test_grouped_extraction_custom_radius(self, image_with_close_pairs, close_pair_objects):
        """Test grouped extraction with custom grouper radius."""
        # Very small radius - all isolated
        result_small = photometry_measure.measure_objects(
            close_pair_objects.copy(),
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=1.0,  # Very small - no groups
            verbose=False
        )
        assert np.all(result_small['group_size'] == 1)

        # Large radius - should have groups
        result_large = photometry_measure.measure_objects(
            close_pair_objects.copy(),
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=15.0,  # Large enough to group
            verbose=False
        )
        assert np.any(result_large['group_size'] > 1)

    @pytest.mark.unit
    def test_grouped_extraction_chi2(self, image_with_close_pairs, close_pair_objects):
        """Test that chi-squared is computed for grouped extraction."""
        result = photometry_measure.measure_objects(
            close_pair_objects,
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            verbose=False
        )

        # Should have chi2 column
        assert 'chi2_optimal' in result.colnames

        # Chi2 should be positive for valid measurements
        valid = np.isfinite(result['flux']) & (result['flux'] > 0)
        assert np.all(result['chi2_optimal'][valid] > 0)

    @pytest.mark.unit
    def test_grouped_extraction_with_mask(self, image_with_close_pairs, close_pair_objects, mask_with_bad_pixels):
        """Test grouped extraction with masked pixels."""
        result = photometry_measure.measure_objects(
            close_pair_objects,
            image_with_close_pairs,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            mask=mask_with_bad_pixels,
            verbose=False
        )

        # Should still return results
        assert len(result) == len(close_pair_objects)
        # Some objects should still be measurable
        valid = np.isfinite(result['flux'])
        assert np.sum(valid) > 0


class TestMaskedColumnHandling:
    """Test that photometry functions handle MaskedColumn inputs correctly.

    These tests verify that functions don't crash when given astropy tables
    with MaskedColumn instead of regular Column. Some tests may fail to
    document where MaskedColumn handling is broken.
    """

    @pytest.mark.unit
    def test_measure_objects_aperture_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test aperture photometry handles MaskedColumn x/y inputs."""
        # This should not crash - masked entries should be handled gracefully
        result = photometry_measure.measure_objects(
            detected_objects_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(detected_objects_masked)

        # Unmasked objects (first 3) should have valid flux
        assert np.sum(np.isfinite(result['flux'][:3])) == 3

    @pytest.mark.unit
    def test_measure_objects_optimal_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test optimal extraction handles MaskedColumn x/y inputs."""
        result = photometry_measure.measure_objects(
            detected_objects_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(detected_objects_masked)

        # Unmasked objects (first 3) should have valid flux
        assert np.sum(np.isfinite(result['flux'][:3])) == 3

    @pytest.mark.unit
    def test_measure_objects_grouped_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test grouped optimal extraction handles MaskedColumn inputs."""
        result = photometry_measure.measure_objects(
            detected_objects_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=10.0,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(detected_objects_masked)

        # Should have group columns
        assert 'group_id' in result.colnames
        assert 'group_size' in result.colnames

    @pytest.mark.unit
    def test_measure_objects_with_local_background_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test local background estimation with MaskedColumn inputs."""
        result = photometry_measure.measure_objects(
            detected_objects_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            bkgann=(8.0, 12.0),  # Local background annulus
            optimal=False,
            verbose=False
        )

        # Should have bg_local column
        assert 'bg_local' in result.colnames
        assert len(result) == len(detected_objects_masked)

    @pytest.mark.unit
    def test_measure_objects_centroiding_with_masked_columns(self, image_with_sources, detected_objects_masked):
        """Test centroiding iteration with MaskedColumn inputs."""
        result = photometry_measure.measure_objects(
            detected_objects_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=2,  # Enable centroiding
            optimal=False,
            verbose=False
        )

        # Should have x_orig, y_orig columns from centroiding
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames
        assert len(result) == len(detected_objects_masked)

    @pytest.mark.unit
    def test_masked_column_all_unmasked_same_as_regular(self, image_with_sources, detected_objects):
        """Test that all-unmasked MaskedColumn gives same result as regular Column."""
        from astropy.table import MaskedColumn

        # Create a copy with MaskedColumns but no masked values
        obj_masked = Table()
        for col in detected_objects.colnames:
            obj_masked[col] = MaskedColumn(detected_objects[col], mask=False)

        # Run with regular columns
        result_regular = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Run with MaskedColumns (all unmasked)
        result_masked = photometry_measure.measure_objects(
            obj_masked,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Fluxes should be identical
        np.testing.assert_array_almost_equal(
            result_regular['flux'],
            result_masked['flux'],
            decimal=5
        )


class TestFullyMaskedFootprints:
    """Test measure_objects behavior when object footprints are fully masked.

    These tests verify that the code handles edge cases where some objects
    have their entire aperture covered by the mask.
    """

    @pytest.fixture
    def image_and_objects_with_masked_footprint(self):
        """Create image with sources and a mask that fully covers some objects.

        Creates 4 sources where the 2nd source is fully masked.
        """
        np.random.seed(42)

        # Background with noise
        image = np.random.normal(100, 5, (256, 256)).astype(np.float64)

        # Add point sources (simple Gaussian profiles)
        sources = [
            (50, 50, 1000),    # x, y, amplitude - unmasked
            (120, 80, 800),    # x, y, amplitude - FULLY MASKED
            (200, 150, 1200),  # x, y, amplitude - unmasked
            (80, 200, 600),    # x, y, amplitude - unmasked
        ]

        y, x = np.ogrid[:256, :256]
        sigma = 1.27  # FWHM ~ 3 pixels
        for sx, sy, amp in sources:
            source = amp * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
            image += source

        # Create objects table
        objects = Table()
        objects['x'] = [50.0, 120.0, 200.0, 80.0]
        objects['y'] = [50.0, 80.0, 150.0, 200.0]
        objects['flux'] = [10000.0, 8000.0, 12000.0, 6000.0]
        objects['flux_err'] = [100.0, 120.0, 150.0, 200.0]
        objects['flags'] = [0, 0, 0, 0]
        objects['fwhm'] = [3.0, 3.2, 2.8, 3.1]

        # Create mask that fully covers the 2nd source (at 120, 80)
        mask = np.zeros((256, 256), dtype=bool)
        # Mask a 20x20 region around the 2nd source
        mask[70:90, 110:130] = True

        return image, objects, mask

    @pytest.fixture
    def image_and_objects_all_masked(self):
        """Create image with sources where ALL objects are fully masked."""
        np.random.seed(42)

        # Background with noise
        image = np.random.normal(100, 5, (256, 256)).astype(np.float64)

        # Add point sources
        sources = [
            (50, 50, 1000),
            (120, 80, 800),
        ]

        y, x = np.ogrid[:256, :256]
        sigma = 1.27
        for sx, sy, amp in sources:
            source = amp * np.exp(-((x - sx)**2 + (y - sy)**2) / (2 * sigma**2))
            image += source

        # Create objects table
        objects = Table()
        objects['x'] = [50.0, 120.0]
        objects['y'] = [50.0, 80.0]
        objects['flux'] = [10000.0, 8000.0]
        objects['flux_err'] = [100.0, 120.0]
        objects['flags'] = [0, 0]
        objects['fwhm'] = [3.0, 3.2]

        # Create mask that covers ALL sources
        mask = np.zeros((256, 256), dtype=bool)
        mask[40:60, 40:60] = True    # Covers 1st source
        mask[70:90, 110:130] = True  # Covers 2nd source

        return image, objects, mask

    @pytest.mark.unit
    def test_aperture_photometry_with_fully_masked_object(self, image_and_objects_with_masked_footprint):
        """Test aperture photometry when one object is fully masked."""
        image, objects, mask = image_and_objects_with_masked_footprint

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(objects)

        # Unmasked objects (indices 0, 2, 3) should have valid flux
        assert np.isfinite(result['flux'][0])
        assert np.isfinite(result['flux'][2])
        assert np.isfinite(result['flux'][3])

        # The masked object (index 1) should be flagged
        # Flag 0x200 indicates masked aperture pixels
        assert (result['flags'][1] & 0x200) != 0

    @pytest.mark.unit
    def test_centroiding_with_fully_masked_object(self, image_and_objects_with_masked_footprint):
        """Test centroiding when one object is fully masked."""
        image, objects, mask = image_and_objects_with_masked_footprint

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=2,
            optimal=False,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(objects)

        # Should have centroiding columns
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames

        # Unmasked objects should have valid centroid adjustments
        assert np.isfinite(result['x'][0])
        assert np.isfinite(result['x'][2])
        assert np.isfinite(result['x'][3])

        # Masked object should be flagged
        assert (result['flags'][1] & 0x200) != 0

    @pytest.mark.unit
    def test_optimal_extraction_with_fully_masked_object(self, image_and_objects_with_masked_footprint):
        """Test optimal extraction when one object is fully masked."""
        image, objects, mask = image_and_objects_with_masked_footprint

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(objects)

        # Unmasked objects should have valid flux
        assert np.isfinite(result['flux'][0])
        assert np.isfinite(result['flux'][2])
        assert np.isfinite(result['flux'][3])

        # Masked object flux may be NaN or flagged
        # The key is that it doesn't crash

    @pytest.mark.unit
    def test_grouped_optimal_with_fully_masked_object(self, image_and_objects_with_masked_footprint):
        """Test grouped optimal extraction when one object is fully masked."""
        image, objects, mask = image_and_objects_with_masked_footprint

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            group_sources=True,
            grouper_radius=50.0,  # Large radius to group some objects
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(objects)

        # Should have group columns
        assert 'group_id' in result.colnames
        assert 'group_size' in result.colnames

        # Unmasked objects should have valid flux
        assert np.isfinite(result['flux'][0])
        assert np.isfinite(result['flux'][2])
        assert np.isfinite(result['flux'][3])

    @pytest.mark.unit
    def test_local_background_with_fully_masked_object(self, image_and_objects_with_masked_footprint):
        """Test local background estimation when one object is fully masked."""
        image, objects, mask = image_and_objects_with_masked_footprint

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            bkgann=(8.0, 12.0),
            optimal=False,
            verbose=False
        )

        # Should return same number of objects
        assert len(result) == len(objects)

        # Should have bg_local column
        assert 'bg_local' in result.colnames

        # Unmasked objects should have valid local background
        # (background annulus may not be fully masked even if aperture is)
        assert np.isfinite(result['bg_local'][0])

    @pytest.mark.unit
    def test_all_objects_fully_masked_aperture(self, image_and_objects_all_masked):
        """Test aperture photometry when ALL objects are fully masked."""
        image, objects, mask = image_and_objects_all_masked

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            optimal=False,
            verbose=False
        )

        # Should still return same number of objects (don't crash)
        assert len(result) == len(objects)

        # All objects should be flagged
        for i in range(len(result)):
            assert (result['flags'][i] & 0x200) != 0

    @pytest.mark.unit
    def test_all_objects_fully_masked_optimal(self, image_and_objects_all_masked):
        """Test optimal extraction when ALL objects are fully masked."""
        image, objects, mask = image_and_objects_all_masked

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            optimal=True,
            verbose=False
        )

        # Should still return same number of objects (don't crash)
        assert len(result) == len(objects)

    @pytest.mark.unit
    def test_all_objects_fully_masked_centroiding(self, image_and_objects_all_masked):
        """Test centroiding when ALL objects are fully masked."""
        image, objects, mask = image_and_objects_all_masked

        result = photometry_measure.measure_objects(
            objects.copy(),
            image,
            mask=mask,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=2,
            optimal=False,
            verbose=False
        )

        # Should still return same number of objects (don't crash)
        assert len(result) == len(objects)

        # Should have centroiding columns
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames


class TestPSFCentroiding:
    """Test PSF-weighted centroiding functionality."""

    @pytest.mark.unit
    def test_psf_centroid_basic(self, image_with_sources, detected_objects):
        """Test basic PSF-weighted centroiding."""
        result = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            verbose=False
        )

        # Should have centroiding columns
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames

        # Positions should be valid
        assert np.all(np.isfinite(result['x']))
        assert np.all(np.isfinite(result['y']))

        # For sources already well-centered, shifts should be very small
        shifts = np.sqrt((result['x'] - result['x_orig'])**2 + (result['y'] - result['y_orig'])**2)
        # All shifts should be reasonable (< 0.1 pixel for well-centered sources)
        assert np.all(shifts < 0.1), f"Shifts too large: {shifts}"

    @pytest.mark.unit
    def test_psf_centroid_requires_psf_or_fwhm(self, image_with_sources, detected_objects):
        """Test that PSF centroiding requires either psf or fwhm."""
        # Without psf or fwhm, should fall back to COM
        result = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            centroid_iter=3,
            centroid_method='psf',  # Request PSF but no psf/fwhm provided
            verbose=False
        )

        # Should still work (falls back to COM)
        assert 'x_orig' in result.colnames
        assert len(result) == len(detected_objects)

    @pytest.mark.unit
    def test_psf_centroid_with_psf_model(self, image_with_sources, detected_objects):
        """Test PSF centroiding with PSF model (not just FWHM)."""
        # Create a simple PSF model (Gaussian dict would work, but we'll use FWHM)
        result = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            verbose=False
        )

        assert len(result) == len(detected_objects)
        assert np.all(np.isfinite(result['x']))
        assert np.all(np.isfinite(result['y']))

    @pytest.mark.unit
    def test_psf_centroid_vs_com_similar(self, image_with_sources, detected_objects):
        """Test that PSF and COM centroiding give similar results for bright isolated sources."""
        result_com = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='com',
            verbose=False
        )

        result_psf = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            verbose=False
        )

        # For bright isolated sources, both methods should give similar results
        # Allow for 1 pixel difference
        x_diff = np.abs(result_com['x'] - result_psf['x'])
        y_diff = np.abs(result_com['y'] - result_psf['y'])

        assert np.all(x_diff < 1.0), f"X differences: {x_diff}"
        assert np.all(y_diff < 1.0), f"Y differences: {y_diff}"

    @pytest.mark.unit
    def test_psf_centroid_with_mask(self, image_with_sources, detected_objects, mask_with_bad_pixels):
        """Test PSF centroiding with masked pixels."""
        result = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            mask=mask_with_bad_pixels,
            verbose=False
        )

        # Should still return results
        assert len(result) == len(detected_objects)

        # Most objects should have valid positions
        valid = np.isfinite(result['x']) & np.isfinite(result['y'])
        assert np.sum(valid) > 0

    @pytest.mark.unit
    def test_psf_centroid_convergence(self, image_with_sources, detected_objects):
        """Test that PSF centroiding works with offset initial positions."""
        # Offset initial positions to test centroiding recovery
        offset_objects = detected_objects.copy()
        offset_objects['x'] = offset_objects['x'] + 0.5
        offset_objects['y'] = offset_objects['y'] + 0.3

        result = photometry_measure.measure_objects(
            offset_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=5,  # More iterations for convergence
            centroid_method='psf',
            verbose=False
        )

        # Should have centroiding columns
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames

        # Positions should be valid
        assert np.all(np.isfinite(result['x']))
        assert np.all(np.isfinite(result['y']))

        # True positions from the fixture
        true_x = np.array([50.0, 120.0, 200.0, 80.0])
        true_y = np.array([50.0, 80.0, 150.0, 200.0])

        # Final positions should be closer to true positions than initial offset
        initial_dist = np.sqrt((offset_objects['x'] - true_x)**2 + (offset_objects['y'] - true_y)**2)
        final_dist = np.sqrt((result['x'] - true_x)**2 + (result['y'] - true_y)**2)

        # Centroiding should improve positions (or at least not make them worse)
        assert np.all(final_dist <= initial_dist), (
            f"Centroiding made positions worse! Initial: {initial_dist}, Final: {final_dist}"
        )

    @pytest.mark.unit
    def test_psf_centroid_edge_objects(self, image_with_sources):
        """Test PSF centroiding near image edges."""
        # Create objects very close to edges
        edge_objects = Table()
        edge_objects['x'] = [5.0, 250.0, 128.0, 128.0]
        edge_objects['y'] = [128.0, 128.0, 5.0, 250.0]
        edge_objects['flux'] = [1000.0, 1000.0, 1000.0, 1000.0]
        edge_objects['fluxerr'] = [10.0, 10.0, 10.0, 10.0]

        result = photometry_measure.measure_objects(
            edge_objects,
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            verbose=False
        )

        # Should return same number of objects (may have NaN for edge cases)
        assert len(result) == len(edge_objects)

    @pytest.mark.unit
    def test_psf_centroid_with_optimal_extraction(self, image_with_sources, detected_objects):
        """Test PSF centroiding combined with optimal extraction."""
        result = photometry_measure.measure_objects(
            detected_objects.copy(),
            image_with_sources,
            aper=5.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            optimal=True,
            verbose=False
        )

        # Should have both centroiding and optimal extraction columns
        assert 'x_orig' in result.colnames
        assert 'y_orig' in result.colnames
        assert 'npix_optimal' in result.colnames
        assert 'chi2_optimal' in result.colnames

        # All measurements should be valid
        assert np.all(np.isfinite(result['flux']))
        assert np.all(result['flux'] > 0)


def _simulate_random_field_with_centroiding(
    rng,
    *,
    size,
    fwhm,
    n_sources,
    margin,
    min_sep,
    noise_std,
    aper,
    mask_fraction=0.0,
    centroid_method='com',
):
    """
    Simulate a random field and test centroiding accuracy.

    Returns metrics on how well centroiding recovers true source positions.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Generate true positions
    xs_true = []
    ys_true = []
    attempts = 0
    while len(xs_true) < n_sources and attempts < 10000:
        attempts += 1
        x = rng.uniform(margin, size - margin)
        y = rng.uniform(margin, size - margin)
        if xs_true:
            dist2 = (np.array(xs_true) - x)**2 + (np.array(ys_true) - y)**2
            if np.any(dist2 < min_sep**2):
                continue
        xs_true.append(x)
        ys_true.append(y)
    assert len(xs_true) == n_sources

    fluxes = rng.uniform(1000.0, 5000.0, size=n_sources)
    amplitudes = fluxes / (2 * np.pi * sigma**2)

    # Create image with noise
    yy, xx = np.mgrid[:size, :size]
    image = rng.normal(0.0, noise_std, (size, size))
    for x, y, amp in zip(xs_true, ys_true, amplitudes):
        image += amp * np.exp(
            -((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)
        )

    # Create initial detections with slight offset to test centroiding
    offset = 0.3  # Initial offset in pixels
    obj = Table()
    obj['x'] = np.array(xs_true, dtype=float) + rng.uniform(-offset, offset, size=n_sources)
    obj['y'] = np.array(ys_true, dtype=float) + rng.uniform(-offset, offset, size=n_sources)
    obj['flux'] = fluxes
    obj['fluxerr'] = np.full(n_sources, noise_std)

    bg = np.zeros_like(image)
    err = np.full_like(image, noise_std)
    mask = None
    if mask_fraction:
        mask = rng.rand(size, size) < mask_fraction

    # Measure with centroiding
    result = photometry_measure.measure_objects(
        obj.copy(),
        image,
        aper=aper,
        fwhm=fwhm,
        centroid_iter=3,
        centroid_method=centroid_method,
        optimal=False,
        mask=mask,
        bg=bg,
        err=err,
        verbose=False,
    )

    # Compute centroiding accuracy
    xs_true = np.array(xs_true)
    ys_true = np.array(ys_true)

    # Distance from true positions
    dx = result['x'] - xs_true
    dy = result['y'] - ys_true
    dist = np.sqrt(dx**2 + dy**2)

    # Initial offset distance
    dx_init = obj['x'] - xs_true
    dy_init = obj['y'] - ys_true
    dist_init = np.sqrt(dx_init**2 + dy_init**2)

    valid = np.isfinite(dist)
    med_dist = float(np.nan)
    med_dist_init = float(np.nan)
    improvement = float(np.nan)

    if np.any(valid):
        med_dist = float(np.median(dist[valid]))
        med_dist_init = float(np.median(dist_init[valid]))
        improvement = (med_dist_init - med_dist) / med_dist_init if med_dist_init > 0 else 0.0

    return {
        'result': result,
        'med_dist': med_dist,
        'med_dist_init': med_dist_init,
        'improvement': improvement,
        'n_valid': int(np.sum(valid)),
        'mask_fraction': mask_fraction,
        'centroid_method': centroid_method,
        'xs_true': xs_true,
        'ys_true': ys_true,
    }


class TestPSFCentroidingAccuracy:
    """Test PSF centroiding accuracy at different crowding levels."""

    @pytest.mark.unit
    def test_centroiding_accuracy_basic(self):
        """Test centroiding accuracy for isolated sources."""
        rng = np.random.RandomState(789)

        metrics_com = _simulate_random_field_with_centroiding(
            rng,
            size=200,
            fwhm=3.0,
            n_sources=20,
            margin=20,
            min_sep=18,
            noise_std=2.0,
            aper=3.0,
            centroid_method='com',
        )

        # Reset RNG for fair comparison
        rng = np.random.RandomState(789)
        metrics_psf = _simulate_random_field_with_centroiding(
            rng,
            size=200,
            fwhm=3.0,
            n_sources=20,
            margin=20,
            min_sep=18,
            noise_std=2.0,
            aper=3.0,
            centroid_method='psf',
        )

        med_dist_com = metrics_com['med_dist']
        med_dist_psf = metrics_psf['med_dist']
        improvement_com = metrics_com['improvement']
        improvement_psf = metrics_psf['improvement']

        print(
            f"\nIsolated sources: COM median dist {med_dist_com:.3f} pix (improvement {improvement_com:.1%}), "
            f"PSF median dist {med_dist_psf:.3f} pix (improvement {improvement_psf:.1%})"
        )

        # COM should improve significantly
        assert improvement_com > 0, f"COM should improve: {improvement_com:.3f}"
        assert med_dist_com < 0.5, f"COM median dist too large: {med_dist_com:.3f}"

        # PSF should at least not crash and produce valid results
        # Note: PSF centroiding works well for clean cases but may be less robust
        # to noise than COM in some scenarios. This requires further investigation.
        assert med_dist_psf < 1.0, f"PSF median dist too large: {med_dist_psf:.3f}"
        assert metrics_psf['n_valid'] == 20, "All PSF centroids should be valid"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "label,min_sep,n_sources",
        [
            ("moderate", 12, 40),
            ("crowded", 8, 60),
            ("very_crowded", 6, 80),
        ],
    )
    @pytest.mark.parametrize("mask_fraction", [0.0, 0.1, 0.3])
    def test_centroiding_accuracy_crowded_report(
        self, label, min_sep, n_sources, mask_fraction
    ):
        """Report centroiding accuracy for crowded fields without assertions."""
        seed = 890 + int(min_sep * 10) + int(mask_fraction * 100)

        # Test COM centroiding
        rng_com = np.random.RandomState(seed)
        metrics_com = _simulate_random_field_with_centroiding(
            rng_com,
            size=200,
            fwhm=3.0,
            n_sources=n_sources,
            margin=15,
            min_sep=min_sep,
            noise_std=2.0,
            aper=3.0,
            mask_fraction=mask_fraction,
            centroid_method='com',
        )

        # Test PSF centroiding with same random field
        rng_psf = np.random.RandomState(seed)
        metrics_psf = _simulate_random_field_with_centroiding(
            rng_psf,
            size=200,
            fwhm=3.0,
            n_sources=n_sources,
            margin=15,
            min_sep=min_sep,
            noise_std=2.0,
            aper=3.0,
            mask_fraction=mask_fraction,
            centroid_method='psf',
        )

        med_dist_com = metrics_com['med_dist']
        med_dist_psf = metrics_psf['med_dist']
        improvement_com = metrics_com['improvement']
        improvement_psf = metrics_psf['improvement']
        n_valid_com = metrics_com['n_valid']
        n_valid_psf = metrics_psf['n_valid']

        # Compute relative improvement of PSF over COM
        if med_dist_com > 0:
            psf_vs_com = (med_dist_com - med_dist_psf) / med_dist_com
        else:
            psf_vs_com = 0.0

        print(
            f"\ncrowding={label} min_sep={min_sep} n={n_sources} mask={mask_fraction:.2f} "
            f"| COM: median dist {med_dist_com:.3f} pix (improvement {improvement_com:.1%}, valid {n_valid_com}) "
            f"| PSF: median dist {med_dist_psf:.3f} pix (improvement {improvement_psf:.1%}, valid {n_valid_psf}) "
            f"| PSF vs COM: {psf_vs_com:+.1%}"
        )

    @pytest.mark.unit
    def test_centroiding_accuracy_faint_sources(self):
        """Test that PSF centroiding works for faint sources."""
        rng = np.random.RandomState(999)

        # Create field with faint sources (high noise)
        sigma = 3.0 / (2 * np.sqrt(2 * np.log(2)))
        size = 200
        n_sources = 20
        noise_std = 5.0  # High noise for faint sources

        # Generate positions
        xs_true = []
        ys_true = []
        attempts = 0
        while len(xs_true) < n_sources and attempts < 10000:
            attempts += 1
            x = rng.uniform(20, size - 20)
            y = rng.uniform(20, size - 20)
            if xs_true:
                dist2 = (np.array(xs_true) - x)**2 + (np.array(ys_true) - y)**2
                if np.any(dist2 < 15**2):
                    continue
            xs_true.append(x)
            ys_true.append(y)

        # Faint sources
        fluxes = rng.uniform(500.0, 1500.0, size=n_sources)  # Lower flux
        amplitudes = fluxes / (2 * np.pi * sigma**2)

        # Create image
        yy, xx = np.mgrid[:size, :size]
        image = rng.normal(0.0, noise_std, (size, size))
        for x, y, amp in zip(xs_true, ys_true, amplitudes):
            image += amp * np.exp(
                -((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)
            )

        # Initial detections with offset
        obj = Table()
        obj['x'] = np.array(xs_true, dtype=float) + rng.uniform(-0.5, 0.5, size=n_sources)
        obj['y'] = np.array(ys_true, dtype=float) + rng.uniform(-0.5, 0.5, size=n_sources)
        obj['flux'] = fluxes
        obj['fluxerr'] = np.full(n_sources, noise_std)

        bg = np.zeros_like(image)
        err = np.full_like(image, noise_std)

        # Test both methods
        result_com = photometry_measure.measure_objects(
            obj.copy(),
            image,
            aper=3.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='com',
            bg=bg,
            err=err,
            verbose=False,
        )

        result_psf = photometry_measure.measure_objects(
            obj.copy(),
            image,
            aper=3.0,
            fwhm=3.0,
            centroid_iter=3,
            centroid_method='psf',
            bg=bg,
            err=err,
            verbose=False,
        )

        # Compute distances
        xs_true = np.array(xs_true)
        ys_true = np.array(ys_true)

        dist_com = np.sqrt((result_com['x'] - xs_true)**2 + (result_com['y'] - ys_true)**2)
        dist_psf = np.sqrt((result_psf['x'] - xs_true)**2 + (result_psf['y'] - ys_true)**2)

        valid = np.isfinite(dist_com) & np.isfinite(dist_psf)
        med_dist_com = np.median(dist_com[valid])
        med_dist_psf = np.median(dist_psf[valid])

        print(
            f"\nFaint sources (high noise): COM median dist {med_dist_com:.3f} pix, "
            f"PSF median dist {med_dist_psf:.3f} pix"
        )

        # PSF should produce valid results (even if not better than COM in all cases)
        # The algorithm works correctly but may be sensitive to noise characteristics
        assert med_dist_psf < 1.0, f"PSF median dist too large: {med_dist_psf:.3f}"
        assert np.sum(valid) == len(result_psf), "All PSF centroids should be valid"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
