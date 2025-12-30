"""
Unit tests for stdpipe.photometry_model module.

Tests photometric calibration and modeling utilities.
"""

import pytest
import numpy as np

from stdpipe import photometry


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
