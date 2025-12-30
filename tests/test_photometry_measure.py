"""
Unit tests for stdpipe.photometry_measure module.

Tests aperture photometry, optimal extraction, and grouped fitting.
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import photometry_measure


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
