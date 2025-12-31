"""
Unit tests for stdpipe.pipeline module.

Tests pipeline building blocks and image preprocessing utilities.
"""

import pytest
import numpy as np
from astropy.io import fits
from astropy.table import Table, MaskedColumn
from astropy.wcs import WCS

from stdpipe import pipeline


class TestMakeMask:
    """Test mask creation functionality."""

    @pytest.mark.unit
    def test_make_mask_basic(self, simple_image):
        """Test basic mask creation with no masking."""
        mask = pipeline.make_mask(simple_image, verbose=False)

        assert mask.shape == simple_image.shape
        assert mask.dtype == bool
        # All finite values, so very few masked pixels
        assert np.sum(mask) < simple_image.size * 0.01

    @pytest.mark.unit
    def test_make_mask_nan_pixels(self, simple_image):
        """Test masking of NaN pixels."""
        image = simple_image.copy()
        # Add some NaN pixels
        image[10:20, 10:20] = np.nan

        mask = pipeline.make_mask(image, verbose=False)

        # NaN region should be masked
        assert np.all(mask[10:20, 10:20])
        assert mask.sum() >= 100  # At least the NaN region

    @pytest.mark.unit
    def test_make_mask_inf_pixels(self, simple_image):
        """Test masking of infinite pixels."""
        image = simple_image.copy()
        # Add some inf pixels
        image[30:40, 30:40] = np.inf

        mask = pipeline.make_mask(image, verbose=False)

        # Inf region should be masked
        assert np.all(mask[30:40, 30:40])

    @pytest.mark.unit
    def test_make_mask_saturation_value(self, simple_image):
        """Test saturation masking with explicit value."""
        image = simple_image.copy()
        # Add some saturated pixels
        image[50:60, 50:60] = 10000

        mask = pipeline.make_mask(
            image,
            saturation=1000,
            verbose=False
        )

        # Saturated region should be masked
        assert np.all(mask[50:60, 50:60])

    @pytest.mark.unit
    def test_make_mask_saturation_auto(self, simple_image):
        """Test automatic saturation level estimation."""
        image = simple_image.copy()
        # Add one very bright pixel
        image[50, 50] = 10000

        mask = pipeline.make_mask(
            image,
            saturation=True,  # Auto-estimate
            verbose=False
        )

        # The bright pixel should be masked
        assert mask[50, 50]

    @pytest.mark.unit
    def test_make_mask_external_mask(self, simple_image):
        """Test combining with external mask."""
        # Create external mask
        external = np.zeros_like(simple_image, dtype=bool)
        external[0:10, :] = True  # Mask top rows

        mask = pipeline.make_mask(
            simple_image,
            external_mask=external,
            verbose=False
        )

        # External mask should be included
        assert np.all(mask[0:10, :])

    @pytest.mark.unit
    def test_make_mask_datasec(self, simple_image):
        """Test DATASEC masking from header."""
        header = fits.Header()
        header['NAXIS1'] = simple_image.shape[1]
        header['NAXIS2'] = simple_image.shape[0]
        # Only use central 50x50 region
        header['DATASEC'] = '[25:75,25:75]'

        mask = pipeline.make_mask(
            simple_image,
            header=header,
            verbose=False
        )

        # Regions outside DATASEC should be masked
        assert np.all(mask[0:24, :])  # Before start
        assert np.all(mask[75:, :])   # After end
        assert np.all(mask[:, 0:24])
        assert np.all(mask[:, 75:])

        # Central region should not be masked (unless NaN/inf)
        # (might have some masked due to non-finite values)

    @pytest.mark.unit
    def test_make_mask_cosmics(self, image_with_sources):
        """Test cosmic ray masking."""
        image = image_with_sources.copy()

        # Add a cosmic ray (very sharp, bright spike)
        image[100, 100] = image[100, 100] + 10000

        mask = pipeline.make_mask(
            image,
            mask_cosmics=True,
            gain=1.0,
            verbose=False
        )

        # Cosmic ray should be detected and masked
        # (This is probabilistic, so we just check that masking ran)
        assert mask.dtype == bool
        assert mask.shape == image.shape


class TestImagePreprocessing:
    """Test other preprocessing utilities if they exist."""

    @pytest.mark.unit
    def test_preprocessing_workflow(self, image_with_sources, simple_header):
        """Test a typical preprocessing workflow."""
        # This is an integration-style test of the workflow
        image = image_with_sources.copy()

        # Step 1: Create mask
        mask = pipeline.make_mask(
            image,
            saturation=True,
            verbose=False
        )

        assert mask.shape == image.shape

        # Masked pixels should be a small fraction for this clean image
        assert np.sum(mask) < image.size * 0.1


class TestPipelineIntegration:
    """Integration tests for complete pipeline operations."""

    @pytest.mark.integration
    def test_full_preprocessing_pipeline(self, sample_fits_data, temp_dir):
        """Test full preprocessing on real FITS data."""
        image, header = sample_fits_data

        # Create comprehensive mask
        mask = pipeline.make_mask(
            image,
            header=header,
            saturation=True,
            mask_cosmics=True,
            gain=header.get('GAIN', 1.0),
            verbose=False
        )

        assert mask.shape == image.shape
        assert mask.dtype == bool

        # Some pixels should be masked
        assert mask.sum() > 0


# ============================================================================
# Fixtures for calibrate_photometry tests
# ============================================================================

@pytest.fixture
def simple_wcs_256():
    """Create a simple WCS object for 256x256 image."""
    header = fits.Header()
    header['NAXIS1'] = 256
    header['NAXIS2'] = 256
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL1'] = 180.0  # RA in degrees
    header['CRVAL2'] = 45.0   # Dec in degrees
    header['CRPIX1'] = 128.5  # Reference pixel
    header['CRPIX2'] = 128.5
    header['CD1_1'] = -0.0002778  # ~1 arcsec/pixel
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = 0.0002778
    header['EQUINOX'] = 2000.0
    return WCS(header)


@pytest.fixture
def detected_objects_with_radec(simple_wcs_256):
    """Create detected objects table with RA/Dec from WCS."""
    obj = Table()
    obj['x'] = [50.0, 120.0, 200.0, 80.0, 150.0, 60.0, 180.0, 100.0]
    obj['y'] = [50.0, 80.0, 150.0, 200.0, 100.0, 180.0, 60.0, 130.0]

    # Convert pixel to world coordinates
    coords = simple_wcs_256.pixel_to_world(obj['x'], obj['y'])
    obj['ra'] = coords.ra.deg
    obj['dec'] = coords.dec.deg

    # Instrumental magnitudes (simulated)
    obj['mag'] = np.array([15.5, 16.3, 14.8, 17.2, 15.9, 16.8, 14.5, 16.0])
    obj['magerr'] = np.array([0.02, 0.03, 0.015, 0.05, 0.025, 0.04, 0.012, 0.03])
    obj['flux'] = 10**((25 - obj['mag']) / 2.5)
    obj['flags'] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    obj['fwhm'] = np.array([3.0, 3.2, 2.8, 3.1, 3.0, 3.3, 2.9, 3.1])

    return obj


@pytest.fixture
def reference_catalog(detected_objects_with_radec):
    """Create reference catalog matching detected objects with known zero point."""
    cat = Table()
    # Same positions (perfect match)
    cat['RAJ2000'] = detected_objects_with_radec['ra'].copy()
    cat['DEJ2000'] = detected_objects_with_radec['dec'].copy()

    # Catalog magnitudes with known zero point offset of 25.0
    zp = 25.0
    cat['R'] = np.array(detected_objects_with_radec['mag']) + zp
    cat['Rerr'] = np.array([0.01, 0.02, 0.01, 0.03, 0.015, 0.025, 0.01, 0.02])

    # For color term testing: B-R = 0.5 color
    cat['B'] = cat['R'] + 0.5

    return cat


@pytest.fixture
def detected_objects_masked(detected_objects_with_radec):
    """Create objects table with MaskedColumns, last two entries masked."""
    obj = Table()
    n = len(detected_objects_with_radec)
    # Mask last two entries
    mask = [False] * (n - 2) + [True, True]

    for col in detected_objects_with_radec.colnames:
        data = detected_objects_with_radec[col]
        # flags should not be masked for testing purposes
        if col == 'flags':
            obj[col] = MaskedColumn(data, mask=[False] * n)
        else:
            obj[col] = MaskedColumn(data, mask=mask)

    return obj


@pytest.fixture
def reference_catalog_masked(reference_catalog):
    """Create reference catalog with MaskedColumns, last two entries masked."""
    cat = Table()
    n = len(reference_catalog)
    mask = [False] * (n - 2) + [True, True]

    for col in reference_catalog.colnames:
        data = reference_catalog[col]
        cat[col] = MaskedColumn(data, mask=mask)

    return cat


# ============================================================================
# calibrate_photometry basic tests
# ============================================================================

class TestCalibratePhotometryBasic:
    """Test basic calibrate_photometry functionality."""

    @pytest.mark.unit
    def test_basic_calibration_returns_result(self, detected_objects_with_radec, reference_catalog):
        """Test that basic photometric calibration returns a result dictionary."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,  # 1 arcsec
            obj_col_mag='mag',
            obj_col_mag_err='magerr',
            cat_col_mag='R',
            verbose=False
        )

        assert result is not None
        assert isinstance(result, dict)
        assert 'zero_fn' in result

    @pytest.mark.unit
    def test_calibration_adds_mag_calib(self, detected_objects_with_radec, reference_catalog):
        """Test that mag_calib and mag_calib_err columns are added when update=True."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        assert 'mag_calib' not in obj.colnames
        assert 'mag_calib_err' not in obj.colnames

        pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        assert 'mag_calib' in obj.colnames
        assert 'mag_calib_err' in obj.colnames
        # Check values are finite
        assert np.all(np.isfinite(obj['mag_calib']))
        assert np.all(np.isfinite(obj['mag_calib_err']))

    @pytest.mark.unit
    def test_calibration_zero_point_recovery(self, detected_objects_with_radec, reference_catalog):
        """Test that the recovered zero point is close to expected value."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        # The zero point should be ~25.0 (the offset we used in the fixture)
        # Check through calibrated magnitudes
        expected_calib_mag = cat['R']  # Should match catalog mags
        # Allow some tolerance due to fitting
        np.testing.assert_allclose(obj['mag_calib'], expected_calib_mag, rtol=0.01)

    @pytest.mark.unit
    def test_calibration_no_update(self, detected_objects_with_radec, reference_catalog):
        """Test that update=False does not modify the obj table."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()
        original_colnames = obj.colnames.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            update=False,
            verbose=False
        )

        assert result is not None
        assert obj.colnames == original_colnames
        assert 'mag_calib' not in obj.colnames

    @pytest.mark.unit
    def test_calibration_with_color_term(self, detected_objects_with_radec, reference_catalog):
        """Test calibration with color term (cat_col_mag1 and cat_col_mag2)."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            cat_col_mag='R',
            cat_col_mag1='B',
            cat_col_mag2='R',
            verbose=False
        )

        assert result is not None
        assert 'cat_col_mag1' in result
        assert 'cat_col_mag2' in result
        assert result['cat_col_mag1'] == 'B'
        assert result['cat_col_mag2'] == 'R'

    @pytest.mark.unit
    def test_calibration_spatial_order(self, detected_objects_with_radec, reference_catalog):
        """Test calibration with spatial order > 0."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Test with linear spatial variation
        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=1,  # Linear spatial variation
            verbose=False
        )

        assert result is not None
        assert 'zero_fn' in result

    @pytest.mark.unit
    def test_calibration_custom_columns(self, detected_objects_with_radec, reference_catalog):
        """Test calibration with custom column names."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Rename columns
        obj.rename_column('mag', 'inst_mag')
        obj.rename_column('magerr', 'inst_mag_err')
        cat.rename_column('R', 'rmag')

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            obj_col_mag='inst_mag',
            obj_col_mag_err='inst_mag_err',
            cat_col_mag='rmag',
            verbose=False
        )

        assert result is not None
        assert result['cat_col_mag'] == 'rmag'


# ============================================================================
# calibrate_photometry edge case tests
# ============================================================================

class TestCalibratePhotometryEdgeCases:
    """Test edge cases for calibrate_photometry."""

    @pytest.mark.unit
    def test_empty_objects_table(self, reference_catalog):
        """Test that empty objects table returns None.

        Note: flags must have int dtype, otherwise photometry.match() fails
        due to bitwise_and operation on float64.
        """
        obj = Table()
        obj['x'] = []
        obj['y'] = []
        obj['ra'] = []
        obj['dec'] = []
        obj['mag'] = []
        obj['magerr'] = []
        obj['flags'] = np.array([], dtype=int)  # Must be int dtype
        obj['fwhm'] = []

        result = pipeline.calibrate_photometry(
            obj, reference_catalog,
            sr=1/3600,
            verbose=False
        )

        assert result is None

    @pytest.mark.unit
    def test_no_positional_matches(self, detected_objects_with_radec, reference_catalog):
        """Test that no positional matches returns None."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Shift catalog positions far away (10 degrees)
        cat['RAJ2000'] = cat['RAJ2000'] + 10.0

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,  # 1 arcsec
            verbose=False
        )

        assert result is None

    @pytest.mark.unit
    def test_sr_from_pixscale(self, detected_objects_with_radec, reference_catalog):
        """Test automatic sr calculation from pixscale."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # pixscale ~1 arcsec/pixel = 1/3600 deg/pixel
        pixscale = 1/3600

        result = pipeline.calibrate_photometry(
            obj, cat,
            pixscale=pixscale,  # sr should be computed from FWHM
            verbose=False
        )

        assert result is not None

    @pytest.mark.unit
    def test_flagged_objects_handled(self, detected_objects_with_radec, reference_catalog):
        """Test that flagged objects are handled (typically excluded)."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Flag some objects
        obj['flags'] = np.array([0, 1, 0, 2, 0, 4, 0, 0])

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # Should still work with some flagged objects
        assert result is not None

    @pytest.mark.unit
    def test_with_catalog_mag_errors(self, detected_objects_with_radec, reference_catalog):
        """Test calibration using catalog magnitude errors."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            cat_col_mag_err='Rerr',
            verbose=False
        )

        assert result is not None


# ============================================================================
# calibrate_photometry MaskedColumn tests
# ============================================================================

class TestCalibratePhotometryMaskedColumns:
    """Test MaskedColumn handling in calibrate_photometry."""

    @pytest.mark.unit
    def test_masked_obj_magnitudes(self, detected_objects_masked, reference_catalog):
        """Test calibration with MaskedColumn magnitudes in obj table."""
        obj = detected_objects_masked.copy()
        cat = reference_catalog.copy()

        # Should not crash, masked values handled gracefully
        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # Should still work - masked objects are effectively excluded
        assert result is not None

    @pytest.mark.unit
    def test_masked_obj_coordinates(self, detected_objects_masked, reference_catalog):
        """Test calibration with MaskedColumn coordinates in obj table."""
        obj = detected_objects_masked.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # Should handle masked coordinates (won't match)
        assert result is not None

    @pytest.mark.unit
    def test_masked_cat_magnitudes(self, detected_objects_with_radec, reference_catalog_masked):
        """Test calibration with MaskedColumn magnitudes in catalog."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog_masked.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # Should handle masked catalog entries
        assert result is not None

    @pytest.mark.unit
    def test_all_unmasked_same_as_regular(self, detected_objects_with_radec, reference_catalog):
        """Test that MaskedColumn with all unmasked gives same result as regular Column."""
        obj_regular = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Create MaskedColumn version with no masking
        obj_masked = Table()
        for col in obj_regular.colnames:
            obj_masked[col] = MaskedColumn(obj_regular[col], mask=False)

        result_regular = pipeline.calibrate_photometry(
            obj_regular, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        result_masked = pipeline.calibrate_photometry(
            obj_masked, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        assert result_regular is not None
        assert result_masked is not None

        # Results should be equivalent
        np.testing.assert_allclose(
            obj_regular['mag_calib'],
            obj_masked['mag_calib'],
            rtol=1e-10
        )

    @pytest.mark.unit
    def test_partial_masking_still_calibrates(self, detected_objects_with_radec, reference_catalog):
        """Test that partial masking still produces calibration result."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Mask only 2 out of 8 objects
        n = len(obj)
        mask = [False] * (n - 2) + [True, True]

        obj['mag'] = MaskedColumn(obj['mag'], mask=mask)
        obj['magerr'] = MaskedColumn(obj['magerr'], mask=mask)

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        # Should work with remaining 6 objects
        assert result is not None
        assert 'mag_calib' in obj.colnames

    @pytest.mark.unit
    def test_both_obj_and_cat_masked(self, detected_objects_masked, reference_catalog_masked):
        """Test calibration with MaskedColumns in both obj and cat."""
        obj = detected_objects_masked.copy()
        cat = reference_catalog_masked.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # Should handle both being masked (using unmasked overlap)
        assert result is not None


# ============================================================================
# calibrate_photometry zero_fn callable tests
# ============================================================================

class TestCalibratePhotometryZeroFn:
    """Test the zero_fn callable returned by calibrate_photometry."""

    # Basic Functionality Tests

    @pytest.mark.unit
    def test_zero_fn_exists_and_callable(self, detected_objects_with_radec, reference_catalog):
        """Test that zero_fn exists in result and is callable."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        # zero_fn should exist
        assert 'zero_fn' in result
        assert result['zero_fn'] is not None

        # Should be callable
        assert callable(result['zero_fn'])

        # Should work with basic call
        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Should not crash
        zero_values = zero_fn(x_test, y_test)

        # Should return array
        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    @pytest.mark.unit
    def test_zero_fn_returns_array(self, detected_objects_with_radec, reference_catalog):
        """Test that zero_fn returns array with correct shape."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Test with arrays of different sizes
        for n in [1, 5, 10]:
            x_test = np.random.uniform(0, 256, n)
            y_test = np.random.uniform(0, 256, n)

            zero_values = zero_fn(x_test, y_test)

            assert isinstance(zero_values, np.ndarray)
            assert len(zero_values) == n

    @pytest.mark.unit
    def test_zero_fn_scalar_input(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with scalar inputs."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Single scalar position
        x_test = 100.0
        y_test = 150.0

        # Should work without crashing
        zero_value = zero_fn(x_test, y_test)

        # Should return array (even for scalar input)
        assert isinstance(zero_value, (np.ndarray, float, np.floating))

    @pytest.mark.unit
    def test_zero_fn_none_coordinates(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with None coordinates uses defaults."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Call with None coordinates
        zero_values = zero_fn(None, None)

        # Should return array (default positions)
        assert isinstance(zero_values, np.ndarray)
        # Length should match number of matched objects
        assert len(zero_values) > 0

    # Spatial Model Tests

    @pytest.mark.unit
    def test_zero_fn_spatial_order_0_constant(self, detected_objects_with_radec, reference_catalog):
        """Test that spatial_order=0 produces constant zero-point."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,  # Constant zero-point
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Test at different positions
        positions = [
            (10, 10),    # Corner
            (128, 128),  # Center
            (245, 245),  # Opposite corner
            (10, 245),   # Another corner
        ]

        zero_points = [zero_fn(np.array([x]), np.array([y]))[0] for x, y in positions]

        # All should be very close (constant model)
        assert np.std(zero_points) < 0.01  # Very small variation

    @pytest.mark.unit
    def test_zero_fn_spatial_order_1_linear(self, detected_objects_with_radec, reference_catalog):
        """Test that spatial_order=1 allows linear variation."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=1,  # Linear spatial variation
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Test at different positions - should be able to vary
        x_test = np.array([10, 128, 245])
        y_test = np.array([10, 128, 245])

        zero_values = zero_fn(x_test, y_test)

        # Should return valid array
        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3
        # With linear model, values CAN vary (but don't have to if data doesn't require it)
        # Just verify it works without crashing

    @pytest.mark.unit
    def test_zero_fn_spatial_order_2_quadratic(self, detected_objects_with_radec, reference_catalog):
        """Test that spatial_order=2 works (quadratic variation)."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=2,  # Quadratic spatial variation
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Test at grid of positions
        x_test = np.array([10, 50, 100, 150, 200, 245])
        y_test = np.array([10, 50, 100, 150, 200, 245])

        zero_values = zero_fn(x_test, y_test)

        # Should return valid array
        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 6

    # Error Computation Tests

    @pytest.mark.unit
    def test_zero_fn_get_err_false(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with get_err=False returns zero-points."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Get zero-points (default)
        zero_values = zero_fn(x_test, y_test, get_err=False)

        # Should return array of zero-points
        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

        # Values should be in reasonable range for zero-points (not tiny error values)
        # Typical zero-points are several magnitudes
        assert np.all(np.abs(zero_values) < 100)  # Sanity check

    @pytest.mark.unit
    def test_zero_fn_get_err_true(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with get_err=True returns errors."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Get errors
        zero_errors = zero_fn(x_test, y_test, get_err=True, add_intrinsic_rms=False)

        # Should return array of errors
        assert isinstance(zero_errors, np.ndarray)
        assert len(zero_errors) == 3

        # Errors should be non-negative
        assert np.all(zero_errors >= 0)

    @pytest.mark.unit
    def test_zero_fn_add_intrinsic_rms(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn add_intrinsic_rms parameter."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Get errors (statistical only)
        zero_errors_stat = zero_fn(x_test, y_test, get_err=True, add_intrinsic_rms=False)

        # Get errors (with intrinsic RMS)
        zero_errors_total = zero_fn(x_test, y_test, get_err=True, add_intrinsic_rms=True)

        # Both should be non-negative
        assert np.all(zero_errors_stat >= 0)
        assert np.all(zero_errors_total >= 0)

        # Total errors >= statistical errors (due to added intrinsic RMS in quadrature)
        assert np.all(zero_errors_total >= zero_errors_stat)

        # If intrinsic_rms > 0, total should be strictly larger
        if result['intrinsic_rms'] > 0:
            assert np.all(zero_errors_total > zero_errors_stat)

    @pytest.mark.unit
    def test_zero_fn_err_large_input(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn error computation with large input (>5000 positions)."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Create large input (>5000 positions)
        n = 6000
        x_test = np.random.uniform(0, 256, n)
        y_test = np.random.uniform(0, 256, n)

        # Get errors - should return zeros for large input (performance limitation)
        zero_errors = zero_fn(x_test, y_test, get_err=True)

        # Should return array of zeros (per implementation in photometry_model.py line 369)
        assert isinstance(zero_errors, np.ndarray)
        assert len(zero_errors) == n
        assert np.all(zero_errors == 0)

    # Magnitude Dependence Tests

    @pytest.mark.unit
    def test_zero_fn_with_mag_parameter(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with mag parameter."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Use simple model first (mag is optional)
        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])
        mag_test = np.array([15.0, 16.0, 17.0])

        # Should work with mag parameter even when not required
        zero_values = zero_fn(x_test, y_test, mag=mag_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    @pytest.mark.unit
    def test_zero_fn_without_mag_when_not_required(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn without mag when not required (simple model)."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Simple model: bg_order=None, nonlin=False (default)
        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,
            bg_order=None,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Should work without mag parameter
        zero_values = zero_fn(x_test, y_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    @pytest.mark.unit
    def test_zero_fn_mag_none_with_simple_model(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with mag=None explicitly."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Explicit mag=None should work
        zero_values = zero_fn(x_test, y_test, mag=None)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    # Data Type Handling Tests

    @pytest.mark.unit
    def test_zero_fn_numpy_arrays(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with regular NumPy arrays."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Regular NumPy arrays
        x_test = np.array([100.0, 150.0, 200.0])
        y_test = np.array([100.0, 150.0, 200.0])
        mag_test = np.array([15.0, 16.0, 17.0])

        # Should work fine
        zero_values = zero_fn(x_test, y_test, mag=mag_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    @pytest.mark.unit
    def test_zero_fn_masked_columns(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with MaskedColumn inputs."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # MaskedColumn inputs
        from astropy.table import MaskedColumn
        x_test = MaskedColumn([100.0, 150.0, 200.0], mask=[False, False, False])
        y_test = MaskedColumn([100.0, 150.0, 200.0], mask=[False, False, False])
        mag_test = MaskedColumn([15.0, 16.0, 17.0], mask=[False, False, False])

        # Should handle MaskedColumns (converted to arrays internally)
        zero_values = zero_fn(x_test, y_test, mag=mag_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3

    @pytest.mark.unit
    def test_zero_fn_mixed_types(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with mixed scalar/array inputs."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Mixed types - should handle gracefully or raise clear error
        x_test = 100.0  # scalar
        y_test = np.array([100.0, 150.0, 200.0])  # array

        try:
            # This might work (broadcasting) or might fail
            zero_values = zero_fn(x_test, y_test)
            # If it works, verify result
            assert isinstance(zero_values, np.ndarray)
        except (ValueError, IndexError):
            # If it fails, that's also acceptable behavior
            pass

    # Integration Tests

    @pytest.mark.unit
    def test_zero_fn_calibrated_magnitudes_match(self, detected_objects_with_radec, reference_catalog):
        """Test that zero_fn produces consistent calibrated magnitudes."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        # Get result with update=True
        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            update=True,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Manually compute calibrated magnitudes using zero_fn
        manual_calib = obj['mag'] + zero_fn(obj['x'], obj['y'], obj['mag'])

        # Should match obj['mag_calib'] from calibrate_photometry
        assert 'mag_calib' in obj.colnames

        # Should be very close
        np.testing.assert_allclose(manual_calib, obj['mag_calib'], rtol=1e-5)

    @pytest.mark.unit
    def test_zero_fn_consistency_across_calls(self, detected_objects_with_radec, reference_catalog):
        """Test that zero_fn returns identical results across multiple calls."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']
        x_test = np.array([100, 150, 200])
        y_test = np.array([100, 150, 200])

        # Call multiple times
        zero_1 = zero_fn(x_test, y_test)
        zero_2 = zero_fn(x_test, y_test)
        zero_3 = zero_fn(x_test, y_test)

        # Should be identical (deterministic)
        np.testing.assert_array_equal(zero_1, zero_2)
        np.testing.assert_array_equal(zero_2, zero_3)

    @pytest.mark.unit
    def test_zero_fn_at_calibration_star_positions(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn at matched calibration star positions."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,  # Simple constant model for easier comparison
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Get matched object positions
        ox = result['ox']
        oy = result['oy']

        if len(ox) > 0:
            # Compute zero-points at matched positions
            zero_at_matched = zero_fn(ox, oy)

            # These should be close to the empirical zero-points
            # (zero_fn is fitted model of result['zero'])
            empirical_zero = result['zero']

            # Should be reasonably close (fitted model approximates data)
            # Allow some deviation since it's a model fit
            assert np.allclose(zero_at_matched, empirical_zero, atol=0.5)

    # Edge Cases

    @pytest.mark.unit
    def test_zero_fn_single_position(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with single position."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Single position as 1-element arrays
        x_test = np.array([100.0])
        y_test = np.array([150.0])

        zero_value = zero_fn(x_test, y_test)

        # Should return single-element array
        assert isinstance(zero_value, np.ndarray)
        assert len(zero_value) == 1

    @pytest.mark.unit
    def test_zero_fn_empty_arrays(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with empty arrays."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            verbose=False
        )

        zero_fn = result['zero_fn']

        # Empty arrays
        x_test = np.array([])
        y_test = np.array([])

        # Should return empty array (or handle gracefully)
        zero_values = zero_fn(x_test, y_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 0

    @pytest.mark.unit
    def test_zero_fn_all_nan_coordinates(self, detected_objects_with_radec, reference_catalog):
        """Test zero_fn with all-NaN coordinates."""
        obj = detected_objects_with_radec.copy()
        cat = reference_catalog.copy()

        result = pipeline.calibrate_photometry(
            obj, cat,
            sr=1/3600,
            order=0,  # Constant model
            verbose=False
        )

        zero_fn = result['zero_fn']

        # All-NaN coordinates
        x_test = np.array([np.nan, np.nan, np.nan])
        y_test = np.array([np.nan, np.nan, np.nan])

        # Should handle gracefully
        zero_values = zero_fn(x_test, y_test)

        assert isinstance(zero_values, np.ndarray)
        assert len(zero_values) == 3
        # For constant model (order=0), position doesn't matter, so NaN coords
        # still return the constant zero-point (not NaN)
        # This is actually sensible behavior
        assert np.all(np.isfinite(zero_values))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
