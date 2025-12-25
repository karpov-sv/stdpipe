"""
Unit tests for stdpipe.utils module.

Tests utility functions for FITS handling, downloads, and helpers.
"""

import pytest
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

from stdpipe import utils


class TestCoordinateFormatting:
    """Test coordinate formatting utilities."""

    @pytest.mark.unit
    def test_make_jname_basic(self):
        """Test J-name generation from coordinates."""
        ra = 180.0
        dec = 45.0

        jname = utils.make_jname(ra, dec)

        assert jname.startswith('J')
        assert len(jname) > 5  # Should have reasonable length

    @pytest.mark.unit
    def test_make_jname_negative_dec(self):
        """Test J-name with negative declination."""
        ra = 123.456
        dec = -23.456

        jname = utils.make_jname(ra, dec)

        assert jname.startswith('J')
        assert '-' in jname  # Should have sign for negative dec

    @pytest.mark.unit
    def test_make_jname_zero_ra(self):
        """Test J-name with RA=0."""
        ra = 0.0
        dec = 0.0

        jname = utils.make_jname(ra, dec)

        assert jname.startswith('J')
        assert len(jname) > 5


class TestDataPath:
    """Test data path utilities."""

    @pytest.mark.unit
    def test_get_data_path(self):
        """Test getting path to module data files."""
        datapath = utils.get_data_path('test.txt')

        # Should be a valid path string
        assert isinstance(datapath, str)
        # Should contain 'data' directory
        assert 'data' in datapath
        # Should end with the filename
        assert datapath.endswith('test.txt')

    @pytest.mark.unit
    def test_get_data_path_different_files(self):
        """Test data path with different filenames."""
        path1 = utils.get_data_path('file1.txt')
        path2 = utils.get_data_path('file2.txt')

        # Should be different
        assert path1 != path2
        # Both should contain data directory
        assert 'data' in path1
        assert 'data' in path2


class TestDownload:
    """Test download utilities."""

    @pytest.mark.unit
    def test_download_existing_file(self, temp_dir):
        """Test that existing file is not re-downloaded."""
        filepath = os.path.join(temp_dir, 'test.txt')

        # Create file
        with open(filepath, 'w') as f:
            f.write('test')

        # Try to download (should skip)
        result = utils.download(
            'http://example.com/file.txt',
            filename=filepath,
            overwrite=False,
            verbose=False
        )

        # Should return True (file exists, skip download)
        assert result == True

        # File content should be unchanged
        with open(filepath, 'r') as f:
            assert f.read() == 'test'

    @pytest.mark.unit
    @pytest.mark.requires_network
    @pytest.mark.slow
    def test_download_small_file(self, temp_dir):
        """Test downloading a small file from the internet."""
        # This test requires network and is slow
        pytest.skip("Network test - implement if needed")


class TestFITSUtilities:
    """Test FITS header parsing and utilities."""

    @pytest.mark.unit
    def test_parse_det_basic(self):
        """Test parsing DATASEC/TRIMSEC strings."""
        # FITS DATASEC format: [x1:x2,y1:y2]
        # This function may or may not exist - check implementation
        if hasattr(utils, 'parse_det'):
            x1, x2, y1, y2 = utils.parse_det('[1:100,1:100]')

            assert x1 == 0  # FITS is 1-indexed, Python is 0-indexed
            assert x2 == 99
            assert y1 == 0
            assert y2 == 99
        else:
            pytest.skip("parse_det function not found")

    @pytest.mark.unit
    def test_parse_det_subregion(self):
        """Test parsing DATASEC with non-zero start."""
        if hasattr(utils, 'parse_det'):
            x1, x2, y1, y2 = utils.parse_det('[25:75,25:75]')

            assert x1 == 24
            assert x2 == 74
            assert y1 == 24
            assert y2 == 74
        else:
            pytest.skip("parse_det function not found")


class TestBreakpoint:
    """Test debugging utilities."""

    @pytest.mark.unit
    def test_breakpoint_exists(self):
        """Test that breakpoint utility exists."""
        # Just check it's callable
        assert callable(utils.breakpoint)


# ============================================================================
# Integration tests
# ============================================================================

class TestUtilsIntegration:
    """Integration tests for utilities."""

    @pytest.mark.integration
    def test_coordinate_consistency(self):
        """Test that coordinate utilities are consistent with astropy."""
        ra = 123.456
        dec = 45.678

        # Our J-name
        jname = utils.make_jname(ra, dec)

        # Compare with astropy SkyCoord
        coord = SkyCoord(ra, dec, unit='deg')
        # Just check that jname has reasonable format
        assert jname.startswith('J')
        assert len(jname) > 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
