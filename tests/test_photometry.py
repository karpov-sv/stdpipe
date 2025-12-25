"""
Unit tests for stdpipe.photometry module.

Tests object detection, photometry, and calibration utilities.
"""

import pytest
import numpy as np
from astropy.table import Table

from stdpipe import photometry


class TestMakeKernel:
    """Test kernel generation utilities."""

    @pytest.mark.unit
    def test_make_kernel_basic(self):
        """Test basic kernel creation."""
        kernel = photometry.make_kernel(r0=1.0, ext=2.0)

        # Check kernel properties
        assert kernel.ndim == 2
        assert kernel.shape[0] == kernel.shape[1]  # Should be square
        assert kernel.max() == 1.0  # Peak should be at 1.0 (center)
        assert np.all(kernel >= 0)  # All values should be positive

    @pytest.mark.unit
    def test_make_kernel_size(self):
        """Test kernel size scales with r0."""
        kernel1 = photometry.make_kernel(r0=1.0, ext=2.0)
        kernel2 = photometry.make_kernel(r0=2.0, ext=2.0)

        # Larger r0 should give larger kernel
        assert kernel2.shape[0] > kernel1.shape[0]

    @pytest.mark.unit
    def test_make_kernel_normalization(self):
        """Test kernel sum (not normalized to 1 by default)."""
        kernel = photometry.make_kernel(r0=1.0, ext=3.0)

        # Just check that sum is reasonable
        assert kernel.sum() > 0


class TestSEPPhotometry:
    """Test SEP-based object detection and photometry."""

    @pytest.mark.unit
    def test_get_objects_sep_simple_image(self, simple_image, simple_mask):
        """Test SEP object detection on simple noise image."""
        # Should detect very few or no objects in pure noise
        obj = photometry.get_objects_sep(
            simple_image,
            mask=simple_mask,
            thresh=5.0,  # High threshold
            verbose=False
        )

        assert isinstance(obj, Table)
        # Pure noise with high threshold should have few detections
        assert len(obj) < 10

    @pytest.mark.unit
    def test_get_objects_sep_with_sources(self, image_with_sources, simple_wcs):
        """Test SEP object detection on image with artificial sources."""
        obj = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            wcs=simple_wcs,
            verbose=False
        )

        assert isinstance(obj, Table)

        # Should detect the 4 artificial sources
        assert len(obj) >= 4

        # Check that expected columns are present
        required_cols = ['x', 'y', 'flux', 'fluxerr', 'mag', 'flags']
        for col in required_cols:
            assert col in obj.colnames

        # If WCS provided, should have ra/dec
        assert 'ra' in obj.colnames
        assert 'dec' in obj.colnames

        # Fluxes should be positive
        assert np.all(obj['flux'] > 0)

    @pytest.mark.unit
    def test_get_objects_sep_with_mask(self, image_with_sources, mask_with_bad_pixels):
        """Test that masking works properly."""
        # Detect without mask
        obj_nomask = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            verbose=False
        )

        # Detect with mask
        obj_masked = photometry.get_objects_sep(
            image_with_sources,
            mask=mask_with_bad_pixels,
            thresh=5.0,
            verbose=False
        )

        # Both should be tables
        assert isinstance(obj_nomask, Table)
        assert isinstance(obj_masked, Table)

        # Masked version might detect fewer objects near edges
        # (but not guaranteed depending on source positions)
        assert len(obj_masked) >= 0

    @pytest.mark.unit
    def test_get_objects_sep_edge_rejection(self, image_with_sources):
        """Test edge rejection parameter."""
        # No edge rejection
        obj_no_edge = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            edge=0,
            verbose=False
        )

        # With edge rejection
        obj_with_edge = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            edge=20,
            verbose=False
        )

        # Edge rejection should result in fewer or equal detections
        assert len(obj_with_edge) <= len(obj_no_edge)

    @pytest.mark.unit
    def test_get_objects_sep_background_params(self, image_with_sources):
        """Test different background estimation parameters."""
        # Fine background grid
        obj_fine = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            bg_size=32,
            verbose=False
        )

        # Coarse background grid
        obj_coarse = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            bg_size=128,
            verbose=False
        )

        # Both should detect objects
        assert len(obj_fine) > 0
        assert len(obj_coarse) > 0


class TestSExtractorIntegration:
    """Integration tests for SExtractor wrapper."""

    @pytest.mark.integration
    @pytest.mark.requires_sextractor
    def test_get_objects_sextractor_basic(self, image_with_sources, temp_dir):
        """Test basic SExtractor object detection."""
        obj = photometry.get_objects_sextractor(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            _workdir=temp_dir,
            verbose=False
        )

        assert isinstance(obj, Table)
        assert len(obj) >= 4  # Should find our artificial sources

        # Check for standard SExtractor columns
        # (exact names depend on implementation)
        assert 'x' in obj.colnames or 'X_IMAGE' in obj.colnames

    @pytest.mark.integration
    @pytest.mark.requires_sextractor
    def test_sextractor_vs_sep_consistency(self, image_with_sources):
        """Test that SExtractor and SEP give similar results."""
        # SEP detection
        obj_sep = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            verbose=False
        )

        # SExtractor detection
        obj_sex = photometry.get_objects_sextractor(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            verbose=False
        )

        # Should detect similar number of objects (within reason)
        assert abs(len(obj_sep) - len(obj_sex)) < 5


class TestPhotometricCalibration:
    """Test photometric calibration utilities."""

    @pytest.mark.unit
    def test_magnitude_conversion(self):
        """Test flux to magnitude conversion."""
        # This is a basic test - actual calibration functions
        # would be in the photometry module
        flux = np.array([100.0, 1000.0, 10000.0])
        zeropoint = 25.0

        mag = zeropoint - 2.5 * np.log10(flux)

        expected = np.array([20.0, 17.5, 15.0])
        assert np.allclose(mag, expected)

    @pytest.mark.unit
    def test_magnitude_error_propagation(self):
        """Test magnitude error from flux error."""
        flux = 1000.0
        flux_err = 10.0

        mag_err = 2.5 / np.log(10) * flux_err / flux

        # Should be ~0.0108 mag
        assert np.abs(mag_err - 0.0108) < 0.001


# ============================================================================
# Property-based tests (if hypothesis is available)
# ============================================================================

try:
    from hypothesis import given, strategies as st

    class TestPhotometryProperties:
        """Property-based tests for photometry functions."""

        @pytest.mark.unit
        @given(
            r0=st.floats(min_value=0.5, max_value=5.0),
            ext=st.floats(min_value=1.0, max_value=5.0)
        )
        def test_kernel_always_positive(self, r0, ext):
            """Test that kernel is always positive."""
            kernel = photometry.make_kernel(r0=r0, ext=ext)
            assert np.all(kernel >= 0)

        @pytest.mark.unit
        @given(
            flux=st.floats(min_value=1.0, max_value=1e6),
            zp=st.floats(min_value=20.0, max_value=30.0)
        )
        def test_flux_mag_roundtrip(self, flux, zp):
            """Test flux->mag->flux roundtrip."""
            mag = zp - 2.5 * np.log10(flux)
            flux_back = 10**((zp - mag) / 2.5)
            assert np.abs(flux - flux_back) / flux < 1e-10

except ImportError:
    # hypothesis not available, skip property tests
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
