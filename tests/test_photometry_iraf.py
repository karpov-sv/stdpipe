"""
Unit tests for stdpipe.photometry_iraf module.

Tests aperture photometry using PyRAF DAOPHOT routines.
"""

import pytest
import numpy as np
from astropy.table import Table
import shutil

# Import the module - should work now with lazy loading
from stdpipe import photometry_iraf

# Check if PyRAF is actually available
PYRAF_AVAILABLE = photometry_iraf.PYRAF_AVAILABLE

# Skip all tests if PyRAF is not available
pytestmark = pytest.mark.skipif(
    not PYRAF_AVAILABLE,
    reason="PyRAF not available"
)


class TestMeasureObjectsIRAF:
    """Test IRAF DAOPHOT-based aperture photometry."""

    @pytest.mark.unit
    def test_measure_objects_basic(self, image_with_sources, detected_objects):
        """Test basic aperture photometry with IRAF DAOPHOT."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            verbose=False
        )

        # Check that result is a Table
        assert isinstance(result, Table)

        # Should have the same number of objects
        assert len(result) == len(detected_objects)

        # Check that required columns are present
        required_cols = ['flux', 'fluxerr', 'mag', 'magerr', 'flags']
        for col in required_cols:
            assert col in result.colnames

        # Check that fluxes are finite and positive
        assert np.all(np.isfinite(result['flux']))
        assert np.all(result['flux'] > 0)

        # Check that errors are positive
        assert np.all(result['fluxerr'] > 0)
        assert np.all(result['magerr'] > 0)

    @pytest.mark.unit
    def test_measure_objects_empty_table(self, image_with_sources):
        """Test with empty object table."""
        empty_objects = Table()
        empty_objects['x'] = []
        empty_objects['y'] = []

        result = photometry_iraf.measure_objects(
            empty_objects,
            image_with_sources,
            verbose=False
        )

        # Should return empty table
        assert len(result) == 0

    @pytest.mark.unit
    def test_measure_objects_with_fwhm_scaling(self, image_with_sources, detected_objects):
        """Test aperture scaling with FWHM."""
        fwhm = 3.0
        aper_in_fwhm = 2.0

        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=aper_in_fwhm,
            fwhm=fwhm,
            verbose=False
        )

        # Should complete successfully
        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)
        assert np.all(np.isfinite(result['flux']))

    @pytest.mark.unit
    def test_measure_objects_with_background_annulus(self, image_with_sources, detected_objects):
        """Test with local background annulus."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            bkgann=(5.0, 8.0),
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)

        # Should have local background column
        assert 'bg_local' in result.colnames

        # Background should be finite
        assert np.all(np.isfinite(result['bg_local']))

    @pytest.mark.unit
    def test_measure_objects_with_mask(self, image_with_sources, detected_objects, mask_with_bad_pixels):
        """Test photometry with image mask."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            mask=mask_with_bad_pixels,
            verbose=False
        )

        assert isinstance(result, Table)

        # Some objects might have flags set due to masked pixels
        # Check that flags column exists
        assert 'flags' in result.colnames

    @pytest.mark.unit
    def test_measure_objects_with_external_background(self, image_with_sources, detected_objects):
        """Test with external background array."""
        # Create simple background
        bg = np.ones_like(image_with_sources) * 100.0

        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            bg=bg,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)

    @pytest.mark.unit
    def test_measure_objects_with_error_map(self, image_with_sources, detected_objects):
        """Test with external error map."""
        # Create simple error map
        err = np.ones_like(image_with_sources) * 5.0

        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            err=err,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)
        assert np.all(np.isfinite(result['fluxerr']))

    @pytest.mark.unit
    def test_measure_objects_with_gain(self, image_with_sources, detected_objects):
        """Test photometry with gain parameter."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            gain=1.5,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)

    @pytest.mark.unit
    def test_measure_objects_sn_filter(self, image_with_sources, detected_objects):
        """Test S/N filtering."""
        # Use high S/N threshold to filter some objects
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            sn=100.0,  # Very high threshold
            verbose=False
        )

        # Should filter out some or all objects
        assert len(result) <= len(detected_objects)

        # All remaining objects should have good S/N
        if len(result) > 0:
            assert np.all(result['magerr'] < 1/100.0)

    @pytest.mark.unit
    def test_measure_objects_keep_negative(self, image_with_sources, detected_objects):
        """Test negative flux filtering."""
        # Test with keep_negative=False
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            keep_negative=False,
            verbose=False
        )

        # All fluxes should be positive
        assert np.all(result['flux'] > 0)

    @pytest.mark.unit
    def test_measure_objects_get_bg(self, image_with_sources, detected_objects):
        """Test returning background maps."""
        result, bg, err = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            get_bg=True,
            verbose=False
        )

        assert isinstance(result, Table)
        assert isinstance(bg, np.ndarray)
        assert isinstance(err, np.ndarray)

        # Background and error should have same shape as image
        assert bg.shape == image_with_sources.shape
        assert err.shape == image_with_sources.shape

    @pytest.mark.unit
    def test_measure_objects_workdir(self, image_with_sources, detected_objects, temp_dir):
        """Test using custom working directory."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            _workdir=temp_dir,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)

        # Working directory should still exist (not cleaned up)
        import os
        assert os.path.exists(temp_dir)

        # Should contain IRAF output files
        files = os.listdir(temp_dir)
        assert any('image.fits' in f for f in files)
        assert any('coords.txt' in f for f in files)

    @pytest.mark.unit
    def test_measure_objects_tmpdir(self, image_with_sources, detected_objects, temp_dir):
        """Test using custom temp directory for temp file creation."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            _tmpdir=temp_dir,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) == len(detected_objects)

    @pytest.mark.unit
    def test_measure_objects_missing_columns(self, image_with_sources):
        """Test error handling for missing x/y columns."""
        bad_objects = Table()
        bad_objects['flux'] = [100.0, 200.0]

        with pytest.raises(ValueError, match="x.*y"):
            photometry_iraf.measure_objects(
                bad_objects,
                image_with_sources,
                verbose=False
            )

    @pytest.mark.unit
    def test_measure_objects_verbose(self, image_with_sources, detected_objects, capsys):
        """Test verbose output."""
        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            verbose=True
        )

        # Check that some output was produced
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    @pytest.mark.unit
    def test_measure_objects_callable_verbose(self, image_with_sources, detected_objects):
        """Test verbose with callable function."""
        messages = []

        def log_function(*args, **kwargs):
            messages.append(' '.join(str(arg) for arg in args))

        result = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            verbose=log_function
        )

        # Should have collected some messages
        assert len(messages) > 0


class TestIRAFPhotometryIntegration:
    """Integration tests for IRAF photometry."""

    @pytest.mark.unit
    def test_photometry_workflow(self, image_with_sources):
        """Test complete photometry workflow."""
        # First detect objects using SEP
        from stdpipe import photometry

        obj = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            verbose=False
        )

        # Then measure with IRAF
        result = photometry_iraf.measure_objects(
            obj,
            image_with_sources,
            aper=3.0,
            bkgann=(5.0, 8.0),
            verbose=False
        )

        # Should have detected the artificial sources
        assert len(result) >= 4

        # All should have valid photometry
        assert np.all(np.isfinite(result['flux']))
        assert np.all(result['flux'] > 0)

    @pytest.mark.unit
    def test_compare_apertures(self, image_with_sources, detected_objects):
        """Test that different aperture sizes give different fluxes."""
        result_small = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=2.0,
            verbose=False
        )

        result_large = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=5.0,
            verbose=False
        )

        # Larger aperture should generally give more flux
        # (at least for some objects)
        assert np.any(result_large['flux'] > result_small['flux'])

    @pytest.mark.unit
    def test_background_subtraction_effect(self, image_with_sources, detected_objects):
        """Test that background annulus affects measurements."""
        # No local background
        result_nobg = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            bkgann=None,
            verbose=False
        )

        # With local background
        result_withbg = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            bkgann=(5.0, 8.0),
            verbose=False
        )

        # Results should differ (local bg subtraction changes fluxes)
        assert not np.allclose(result_nobg['flux'], result_withbg['flux'])


# Add fixture for checking PyRAF availability
@pytest.fixture
def has_pyraf():
    """Check if PyRAF is available."""
    try:
        import pyraf
        return True
    except ImportError:
        return False


# Mark tests that require PyRAF
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_pyraf: mark test as requiring PyRAF/IRAF"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests requiring PyRAF if not available."""
    for item in items:
        if "test_photometry_iraf" in str(item.fspath):
            if not PYRAF_AVAILABLE:
                item.add_marker(pytest.mark.skip(reason="PyRAF not available"))


class TestMeasureObjectsPSFIRAF:
    """Test IRAF DAOPHOT PSF photometry."""

    @pytest.mark.unit
    def test_measure_objects_psf_basic(self, image_with_sources, detected_objects):
        """Test basic PSF photometry with automatic PSF star selection."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Check that result is a Table
        assert isinstance(result, Table)

        # Should have some objects (some might be filtered)
        assert len(result) > 0

        # Check required columns are present
        required_cols = ['flux', 'fluxerr', 'mag', 'magerr', 'x_psf', 'y_psf',
                        'qfit_psf', 'cfit_psf', 'flags_psf']
        for col in required_cols:
            assert col in result.colnames

        # Note: DAOPHOT allstar may not successfully fit all objects
        # Some objects may have NaN values if the fit failed
        # Check that at least one object was fitted successfully
        valid_fits = np.isfinite(result['flux'])
        assert np.sum(valid_fits) > 0, "No objects were successfully fitted"

        # For successfully fitted objects, check that values are reasonable
        if np.sum(valid_fits) > 0:
            assert np.all(result['flux'][valid_fits] > 0)
            assert np.all(result['fluxerr'][valid_fits] > 0)
            assert np.all(result['magerr'][valid_fits] > 0)
            assert np.all(np.isfinite(result['x_psf'][valid_fits]))
            assert np.all(np.isfinite(result['y_psf'][valid_fits]))

    @pytest.mark.unit
    def test_measure_objects_psf_with_selected_stars(self, image_with_sources, detected_objects):
        """Test PSF photometry with pre-selected PSF stars."""
        # Use first two objects as PSF stars
        psf_star_idx = [0, 1]

        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            psf_stars=psf_star_idx,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0
        assert 'x_psf' in result.colnames
        assert 'y_psf' in result.colnames

    @pytest.mark.unit
    def test_measure_objects_psf_missing_fwhm(self, image_with_sources, detected_objects):
        """Test that missing FWHM raises ValueError."""
        with pytest.raises(ValueError, match="fwhm"):
            photometry_iraf.measure_objects_psf(
                detected_objects,
                image_with_sources,
                verbose=False
            )

    @pytest.mark.unit
    def test_measure_objects_psf_with_background(self, image_with_sources, detected_objects):
        """Test PSF photometry with external background."""
        bg = np.ones_like(image_with_sources) * 100.0

        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            bg=bg,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

    @pytest.mark.unit
    def test_measure_objects_psf_with_error_map(self, image_with_sources, detected_objects):
        """Test PSF photometry with external error map."""
        err = np.ones_like(image_with_sources) * 5.0

        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            err=err,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

        # Check that fitted objects have finite flux errors
        valid_fits = np.isfinite(result['flux'])
        if np.sum(valid_fits) > 0:
            assert np.all(np.isfinite(result['fluxerr'][valid_fits]))

    @pytest.mark.unit
    def test_measure_objects_psf_with_gain(self, image_with_sources, detected_objects):
        """Test PSF photometry with gain parameter."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            gain=1.5,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

    @pytest.mark.unit
    def test_measure_objects_psf_sn_filter(self, image_with_sources, detected_objects):
        """Test S/N filtering in PSF photometry."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            sn=100.0,  # Very high threshold
            verbose=False
        )

        # Should filter out some or all objects
        assert len(result) <= len(detected_objects)

    @pytest.mark.unit
    def test_measure_objects_psf_keep_negative(self, image_with_sources, detected_objects):
        """Test negative flux filtering in PSF photometry."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            keep_negative=False,
            verbose=False
        )

        # All fluxes should be positive
        assert np.all(result['flux'] > 0)

    @pytest.mark.unit
    def test_measure_objects_psf_custom_radii(self, image_with_sources, detected_objects):
        """Test PSF photometry with custom PSF and fitting radii."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            psfrad=10.0,
            fitrad=4.0,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

    @pytest.mark.unit
    def test_measure_objects_psf_function_types(self, image_with_sources, detected_objects):
        """Test different PSF function types."""
        for psf_func in ['gauss', 'moffat15']:
            result = photometry_iraf.measure_objects_psf(
                detected_objects,
                image_with_sources,
                fwhm=3.0,
                psf_function=psf_func,
                verbose=False
            )

            assert isinstance(result, Table)
            assert len(result) > 0

    @pytest.mark.unit
    def test_measure_objects_psf_quality_columns(self, image_with_sources, detected_objects):
        """Test that quality metric columns are present and valid."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Check quality columns
        assert 'qfit_psf' in result.colnames  # chi
        assert 'cfit_psf' in result.colnames  # sharpness
        assert 'flags_psf' in result.colnames  # pier flags

        # For successfully fitted objects, check quality metrics
        valid_fits = np.isfinite(result['flux'])
        if np.sum(valid_fits) > 0:
            assert np.all(np.isfinite(result['qfit_psf'][valid_fits]))
            assert np.all(np.isfinite(result['cfit_psf'][valid_fits]))

    @pytest.mark.unit
    def test_measure_objects_psf_get_bg(self, image_with_sources, detected_objects):
        """Test returning background maps from PSF photometry."""
        result, bg, err = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            get_bg=True,
            verbose=False
        )

        assert isinstance(result, Table)
        assert isinstance(bg, np.ndarray)
        assert isinstance(err, np.ndarray)

        # Background and error should have same shape as image
        assert bg.shape == image_with_sources.shape
        assert err.shape == image_with_sources.shape

    @pytest.mark.unit
    def test_measure_objects_psf_workdir(self, image_with_sources, detected_objects, temp_dir):
        """Test PSF photometry with custom working directory."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            _workdir=temp_dir,
            verbose=False
        )

        assert isinstance(result, Table)
        assert len(result) > 0

        # Check that PSF-related files were created
        import os
        files = os.listdir(temp_dir)
        assert any('psf' in f.lower() for f in files)

    @pytest.mark.unit
    def test_measure_objects_psf_empty_table(self, image_with_sources):
        """Test PSF photometry with empty object table."""
        empty_objects = Table()
        empty_objects['x'] = []
        empty_objects['y'] = []

        result = photometry_iraf.measure_objects_psf(
            empty_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Should return empty table
        assert len(result) == 0

    @pytest.mark.unit
    def test_measure_objects_psf_verbose(self, image_with_sources, detected_objects, capsys):
        """Test verbose output in PSF photometry."""
        result = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=True
        )

        # Check that some output was produced
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert 'PSF' in captured.out or 'phot' in captured.out or 'allstar' in captured.out


class TestPSFPhotometryWorkflow:
    """Integration tests for complete PSF photometry workflow."""

    @pytest.mark.unit
    def test_full_psf_workflow(self, image_with_sources):
        """Test complete PSF photometry workflow from detection to fitting."""
        from stdpipe import photometry

        # Step 1: Detect objects
        obj = photometry.get_objects_sep(
            image_with_sources,
            thresh=5.0,
            aper=3.0,
            verbose=False
        )

        assert len(obj) >= 4  # Should detect the artificial sources

        # Step 2: PSF photometry
        result = photometry_iraf.measure_objects_psf(
            obj,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Should have detected and fitted at least some sources
        valid_fits = np.isfinite(result['flux'])
        assert np.sum(valid_fits) >= 1, "At least one source should be fitted"

        # Successfully fitted objects should have valid PSF photometry
        assert np.all(result['flux'][valid_fits] > 0)
        assert np.all(np.isfinite(result['x_psf'][valid_fits]))
        assert np.all(np.isfinite(result['y_psf'][valid_fits]))

    @pytest.mark.unit
    def test_compare_aperture_psf(self, image_with_sources, detected_objects):
        """Compare aperture and PSF photometry results."""
        # Aperture photometry
        result_aper = photometry_iraf.measure_objects(
            detected_objects,
            image_with_sources,
            aper=3.0,
            verbose=False
        )

        # PSF photometry
        result_psf = photometry_iraf.measure_objects_psf(
            detected_objects,
            image_with_sources,
            fwhm=3.0,
            verbose=False
        )

        # Both should return results
        assert len(result_aper) > 0
        assert len(result_psf) > 0

        # PSF photometry should provide fitted positions
        assert 'x_psf' in result_psf.colnames
        assert 'y_psf' in result_psf.colnames

        # Both should have flux measurements
        assert 'flux' in result_aper.colnames
        assert 'flux' in result_psf.colnames
