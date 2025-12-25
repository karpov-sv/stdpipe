"""
Unit tests for stdpipe.astrometry module.

Tests coordinate transformations, WCS operations, and astrometric utilities.
"""

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

from stdpipe import astrometry


class TestCoordinateTransformations:
    """Test coordinate transformation utilities."""

    @pytest.mark.unit
    def test_spherical_distance(self):
        """Test spherical distance calculation."""
        # Distance between two points at the equator, 1 degree apart
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = 1.0, 0.0

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # Should be close to 1 degree
        assert np.abs(dist - 1.0) < 0.001

    @pytest.mark.unit
    def test_spherical_distance_pole(self):
        """Test spherical distance at the pole."""
        # All RAs should give same distance from pole
        ra1, dec1 = 0.0, 90.0
        ra2, dec2 = 180.0, 89.0

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # Should be 1 degree
        assert np.abs(dist - 1.0) < 0.001

    @pytest.mark.unit
    def test_spherical_distance_same_point(self):
        """Test spherical distance for identical points."""
        ra, dec = 123.456, 45.678

        dist = astrometry.spherical_distance(ra, dec, ra, dec)

        # Should be zero
        assert dist < 1e-10

    @pytest.mark.unit
    def test_radectoxyz_xyztoradec_roundtrip(self):
        """Test RA/Dec to XYZ conversion roundtrip."""
        ra_orig = 180.0
        dec_orig = 45.0

        # Convert to XYZ
        xyz = astrometry.radectoxyz(ra_orig, dec_orig)

        # Check unit vector
        assert np.abs(np.linalg.norm(xyz) - 1.0) < 1e-10

        # Convert back to RA/Dec
        ra_back, dec_back = astrometry.xyztoradec(xyz)

        # Should match original
        assert np.abs(ra_back - ra_orig) < 1e-10
        assert np.abs(dec_back - dec_orig) < 1e-10

    @pytest.mark.unit
    def test_radectoxyz_array(self):
        """Test RA/Dec to XYZ conversion with arrays."""
        ra = np.array([0.0, 90.0, 180.0])
        dec = np.array([0.0, 0.0, 0.0])

        xyz = astrometry.radectoxyz(ra, dec)

        # Check shape
        assert xyz.shape == (3, 3)

        # Check that all are unit vectors
        norms = np.linalg.norm(xyz, axis=0)
        assert np.allclose(norms, 1.0)


class TestWCSUtilities:
    """Test WCS-related utilities."""

    @pytest.mark.unit
    def test_get_pixscale_from_wcs(self, simple_wcs):
        """Test pixel scale extraction from WCS."""
        pixscale = astrometry.get_pixscale(wcs=simple_wcs)

        # Should be ~1 arcsec/pixel = 0.0002778 deg/pixel
        expected = 0.0002778
        assert np.abs(pixscale - expected) < 1e-6

    @pytest.mark.unit
    def test_get_pixscale_from_header(self, header_with_wcs):
        """Test pixel scale extraction from header."""
        pixscale = astrometry.get_pixscale(header=header_with_wcs)

        # Should be ~1 arcsec/pixel
        expected = 0.0002778
        assert np.abs(pixscale - expected) < 1e-6

    @pytest.mark.unit
    def test_get_frame_center(self, header_with_wcs):
        """Test frame center calculation."""
        ra, dec, sr = astrometry.get_frame_center(header=header_with_wcs)

        # Center should be at CRVAL
        assert np.abs(ra - 180.0) < 0.1
        assert np.abs(dec - 45.0) < 0.1

        # Radius should be positive
        assert sr > 0

    @pytest.mark.unit
    def test_get_frame_center_with_wcs(self, simple_wcs):
        """Test frame center with WCS and explicit dimensions."""
        ra, dec, sr = astrometry.get_frame_center(
            wcs=simple_wcs,
            width=256,
            height=256
        )

        # Center should be near CRVAL
        assert np.abs(ra - 180.0) < 0.1
        assert np.abs(dec - 45.0) < 0.1

    @pytest.mark.unit
    def test_get_frame_center_no_wcs(self):
        """Test frame center with no WCS returns None."""
        header = fits.Header()
        header['NAXIS1'] = 100
        header['NAXIS2'] = 100

        ra, dec, sr = astrometry.get_frame_center(header=header)

        assert ra is None
        assert dec is None
        assert sr is None


class TestAstrometricMatching:
    """Test astrometric matching and catalog crossmatching."""

    @pytest.mark.unit
    def test_spherical_distance_vectorized(self):
        """Test vectorized spherical distance calculation."""
        # Create a grid of points
        ra1 = np.array([0.0, 1.0, 2.0])
        dec1 = np.array([0.0, 0.0, 0.0])
        ra2 = np.array([0.5, 1.5, 2.5])
        dec2 = np.array([0.0, 0.0, 0.0])

        dist = astrometry.spherical_distance(ra1, dec1, ra2, dec2)

        # All distances should be ~0.5 degrees
        assert dist.shape == (3,)
        assert np.allclose(dist, 0.5, atol=0.01)


# ============================================================================
# Integration tests requiring external tools
# ============================================================================

class TestAstrometryNetIntegration:
    """Integration tests for Astrometry.Net wrapper."""

    @pytest.mark.integration
    @pytest.mark.requires_astrometry_net
    @pytest.mark.slow
    def test_astrometry_net_basic(self, image_with_sources, temp_dir):
        """Test basic Astrometry.Net plate solving."""
        # This would require actual astrometry.net to be installed
        # and configured, so marking as integration test
        pytest.skip("Requires full astrometry.net setup")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
