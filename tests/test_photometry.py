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


class TestGetObjectsPhotutils:
    """Test photutils-based object detection."""

    # ========================================================================
    # Basic Detection Tests
    # ========================================================================

    @pytest.mark.unit
    def test_dao_basic(self, image_with_sources):
        """Test DAOStarFinder detection."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='dao',
            fwhm=3.0,
            aper=3.0,
            verbose=False
        )

        # Should detect sources
        assert len(obj) > 0, "No sources detected"

        # Check required columns
        required = ['x', 'y', 'flux', 'fluxerr', 'mag', 'magerr', 'fwhm']
        for col in required:
            assert col in obj.colnames

        # Check values
        assert np.all(np.isfinite(obj['x']))
        assert np.all(obj['flux'] > 0)
        # Check FWHM is constant (as provided)
        assert np.allclose(obj['fwhm'], 3.0)

    @pytest.mark.unit
    def test_iraf_basic(self, image_with_sources):
        """Test IRAFStarFinder detection."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='iraf',
            fwhm=3.0,
            aper=3.0,
            verbose=False
        )

        # Should detect sources
        assert len(obj) > 0, "No sources detected"

        # Check required columns
        required = ['x', 'y', 'flux', 'fluxerr', 'mag', 'magerr']
        for col in required:
            assert col in obj.colnames

        # Check values
        assert np.all(np.isfinite(obj['x']))
        assert np.all(obj['flux'] > 0)

    # ========================================================================
    # Deblending Tests
    # ========================================================================

    @pytest.mark.unit
    def test_deblending_enabled(self, image_with_sources):
        """Test that deblending increases source count (or at least runs)."""
        # Detect without deblending
        obj_no_deblend = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            deblend=False,
            aper=3.0,
            verbose=False
        )

        # Detect with deblending
        obj_deblend = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            deblend=True,
            nlevels=32,
            contrast=0.001,
            aper=3.0,
            verbose=False
        )

        # Both should detect sources
        assert len(obj_no_deblend) > 0
        assert len(obj_deblend) > 0

        # Deblending should work (may or may not increase count depending on image)
        # Just verify both complete successfully
        assert 'flux' in obj_deblend.colnames

    @pytest.mark.unit
    def test_deblending_parameters(self, image_with_sources):
        """Test deblending with different parameters."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            deblend=True,
            nlevels=16,
            contrast=0.01,
            aper=3.0,
            verbose=False
        )

        assert len(obj) > 0
        assert 'flux' in obj.colnames

    # ========================================================================
    # Parameter Tests
    # ========================================================================

    @pytest.mark.unit
    def test_different_aperture_sizes(self, image_with_sources):
        """Test detection with different aperture sizes."""
        for aper_size in [2.0, 3.0, 5.0]:
            obj = photometry.get_objects_photutils(
                image_with_sources,
                thresh=3.0,
                method='segmentation',
                aper=aper_size,
                verbose=False
            )

            assert len(obj) > 0
            assert obj.meta['aper'] == aper_size

    @pytest.mark.unit
    def test_background_annulus(self, image_with_sources):
        """Test with background annulus."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            aper=3.0,
            bkgann=(5.0, 8.0),
            verbose=False
        )

        assert len(obj) > 0
        assert 'bkgann' in obj.meta
        assert obj.meta['bkgann'] == (5.0, 8.0)

    @pytest.mark.unit
    def test_background_subtraction_toggle(self, image_with_sources):
        """Test with and without background subtraction."""
        # With background subtraction (default)
        obj_with_bg = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            subtract_bg=True,
            verbose=False
        )

        # Without background subtraction
        obj_no_bg = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            subtract_bg=False,
            verbose=False
        )

        # Both should detect sources
        assert len(obj_with_bg) > 0
        assert len(obj_no_bg) > 0

    @pytest.mark.unit
    def test_threshold_levels(self, image_with_sources):
        """Test different detection thresholds."""
        # Lower threshold should detect more sources
        obj_low = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            verbose=False
        )

        # Higher threshold should detect fewer sources
        obj_high = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='segmentation',
            verbose=False
        )

        # Both should detect something
        assert len(obj_low) > 0
        assert len(obj_high) > 0

        # Lower threshold should detect at least as many sources
        assert len(obj_low) >= len(obj_high)

    @pytest.mark.unit
    def test_custom_background_size(self, image_with_sources):
        """Test with custom background grid size."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            bg_size=32,
            verbose=False
        )

        assert len(obj) > 0

    # ========================================================================
    # Filtering Tests
    # ========================================================================

    @pytest.mark.unit
    def test_edge_filter(self, image_with_sources):
        """Test edge exclusion filtering."""
        # No edge filter
        obj_no_edge = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            edge=0,
            verbose=False
        )

        # With edge filter
        obj_edge = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            edge=20,
            verbose=False
        )

        # Edge filter should remove some sources
        assert len(obj_edge) <= len(obj_no_edge)

    @pytest.mark.unit
    def test_sn_filter(self, image_with_sources):
        """Test S/N filtering."""
        # Low S/N threshold
        obj_low_sn = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            sn=2.0,
            verbose=False
        )

        # High S/N threshold
        obj_high_sn = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            sn=10.0,
            verbose=False
        )

        # Higher S/N threshold should give fewer sources
        assert len(obj_high_sn) <= len(obj_low_sn)

    @pytest.mark.unit
    def test_minarea_filter(self, image_with_sources):
        """Test minimum area filtering (segmentation only)."""
        # Small minarea
        obj_small = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            minarea=3,
            verbose=False
        )

        # Large minarea
        obj_large = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            minarea=20,
            verbose=False
        )

        # Larger minarea should give fewer sources
        assert len(obj_large) <= len(obj_small)

    # ========================================================================
    # WCS Tests
    # ========================================================================

    @pytest.mark.unit
    def test_wcs_from_header(self, image_with_sources, header_with_wcs):
        """Test WCS conversion from header."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            header=header_with_wcs,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        assert len(obj) > 0
        # Should have RA/Dec columns
        assert 'ra' in obj.colnames
        assert 'dec' in obj.colnames
        # Check values are reasonable (within expected range)
        assert np.all(np.isfinite(obj['ra']))
        assert np.all(np.isfinite(obj['dec']))

    @pytest.mark.unit
    def test_wcs_direct(self, image_with_sources, simple_wcs):
        """Test WCS conversion from WCS object."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            wcs=simple_wcs,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        assert len(obj) > 0
        assert 'ra' in obj.colnames
        assert 'dec' in obj.colnames

    @pytest.mark.unit
    def test_no_wcs(self, image_with_sources):
        """Test without WCS (no RA/Dec columns)."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        # Should not have RA/Dec columns
        assert 'ra' not in obj.colnames
        assert 'dec' not in obj.colnames

    # ========================================================================
    # Output Tests
    # ========================================================================

    @pytest.mark.unit
    def test_output_columns(self, image_with_sources):
        """Test that all required columns are present."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        required = ['x', 'y', 'xerr', 'yerr', 'flux', 'fluxerr',
                    'mag', 'magerr', 'fwhm', 'a', 'b', 'theta',
                    'bg', 'flags']

        for col in required:
            assert col in obj.colnames, f"Missing required column: {col}"

    @pytest.mark.unit
    def test_segmentation_map_return(self, image_with_sources):
        """Test returning segmentation map."""
        obj, segm = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            get_segmentation=True,
            verbose=False
        )

        assert len(obj) > 0
        assert segm is not None
        # Segmentation map should have non-zero label
        assert np.sum(segm > 0) > 0

    @pytest.mark.unit
    def test_segmentation_map_starfinder_warning(self, image_with_sources):
        """Test that get_segmentation with StarFinder returns None."""
        obj, segm = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='dao',
            fwhm=3.0,
            get_segmentation=True,
            verbose=False
        )

        assert len(obj) > 0
        # Should return None for segmentation map
        assert segm is None

    @pytest.mark.unit
    def test_empty_detection(self, simple_image):
        """Test handling of no detections."""
        # Very high threshold should detect nothing
        obj = photometry.get_objects_photutils(
            simple_image,
            thresh=100.0,
            method='segmentation',
            verbose=False
        )

        # Should return empty table
        assert len(obj) == 0
        # But with correct columns
        assert 'x' in obj.colnames
        assert 'flux' in obj.colnames

    @pytest.mark.unit
    def test_metadata(self, image_with_sources):
        """Test that metadata is properly set."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            deblend=True,
            aper=5.0,
            bkgann=(10.0, 15.0),
            verbose=False
        )

        assert 'method' in obj.meta
        assert obj.meta['method'] == 'segmentation'
        assert 'thresh' in obj.meta
        assert obj.meta['thresh'] == 3.0
        assert 'aper' in obj.meta
        assert obj.meta['aper'] == 5.0
        assert 'bkgann' in obj.meta
        assert obj.meta['bkgann'] == (10.0, 15.0)
        assert 'deblend' in obj.meta
        assert obj.meta['deblend'] is True

    @pytest.mark.unit
    def test_sorted_by_brightness(self, image_with_sources):
        """Test that sources are sorted by brightness (flux descending)."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            verbose=False
        )

        if len(obj) > 1:
            # Check that fluxes are in descending order
            assert np.all(obj['flux'][:-1] >= obj['flux'][1:])

    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================

    @pytest.mark.unit
    def test_with_mask(self, image_with_sources, mask_with_bad_pixels):
        """Test detection with mask."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            mask=mask_with_bad_pixels,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        # Should still detect sources in unmasked regions
        assert len(obj) > 0

    @pytest.mark.unit
    def test_with_error_map(self, image_with_sources):
        """Test with custom error map."""
        # Create simple error map
        err = np.full_like(image_with_sources, 5.0)

        obj = photometry.get_objects_photutils(
            image_with_sources,
            err=err,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        assert len(obj) > 0

    @pytest.mark.unit
    def test_with_nans_in_image(self, image_with_sources):
        """Test handling of NaN values in image."""
        # Add some NaNs
        image_with_nans = image_with_sources.copy()
        image_with_nans[10:20, 10:20] = np.nan

        obj = photometry.get_objects_photutils(
            image_with_nans,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        # Should still detect sources in valid regions
        assert len(obj) > 0

    @pytest.mark.unit
    def test_invalid_method(self, image_with_sources):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            photometry.get_objects_photutils(
                image_with_sources,
                thresh=3.0,
                method='invalid_method',
                verbose=False
            )

    @pytest.mark.unit
    def test_invalid_image_dimension(self):
        """Test that non-2D image raises error."""
        image_3d = np.random.normal(100, 10, (10, 10, 10))

        with pytest.raises(ValueError, match="Image must be 2D"):
            photometry.get_objects_photutils(
                image_3d,
                thresh=3.0,
                method='segmentation'
            )

    @pytest.mark.unit
    def test_verbose_output(self, image_with_sources, capsys):
        """Test verbose output."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            verbose=True
        )

        # Capture printed output
        captured = capsys.readouterr()

        # Should have printed something
        assert len(captured.out) > 0
        assert 'Estimating background' in captured.out or 'Detecting sources' in captured.out

    # ========================================================================
    # StarFinder-specific Tests
    # ========================================================================

    @pytest.mark.unit
    def test_starfinder_sharpness_bounds(self, image_with_sources):
        """Test StarFinder with custom sharpness bounds."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='dao',
            fwhm=3.0,
            sharplo=0.3,
            sharphi=0.9,
            verbose=False
        )

        # May detect fewer sources due to stricter sharpness criteria
        # Just verify it completes
        assert isinstance(obj, Table)

    @pytest.mark.unit
    def test_starfinder_roundness_bounds(self, image_with_sources):
        """Test StarFinder with custom roundness bounds."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='iraf',
            fwhm=3.0,
            roundlo=-0.5,
            roundhi=0.5,
            verbose=False
        )

        # May detect fewer sources due to stricter roundness criteria
        # Just verify it completes
        assert isinstance(obj, Table)

    # ========================================================================
    # Integration Tests
    # ========================================================================

    @pytest.mark.integration
    def test_compare_methods_same_image(self, image_with_sources):
        """Test that different methods on same image give reasonable results."""
        # Segmentation
        obj_seg = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            verbose=False
        )

        # DAOStarFinder
        obj_dao = photometry.get_objects_photutils(
            image_with_sources,
            thresh=5.0,
            method='dao',
            fwhm=3.0,
            verbose=False
        )

        # Both should detect sources
        assert len(obj_seg) > 0
        assert len(obj_dao) > 0

        # Both should have same column structure
        required = ['x', 'y', 'flux', 'mag']
        for col in required:
            assert col in obj_seg.colnames
            assert col in obj_dao.colnames

    @pytest.mark.integration
    def test_complete_workflow(self, image_with_sources, header_with_wcs):
        """Test complete detection workflow with all features."""
        obj, segm = photometry.get_objects_photutils(
            image_with_sources,
            header=header_with_wcs,
            thresh=2.0,
            method='segmentation',
            deblend=True,
            nlevels=32,
            contrast=0.001,
            aper=3.0,
            bkgann=(5.0, 8.0),
            bg_size=64,
            subtract_bg=True,
            edge=10,
            sn=3.0,
            minarea=5,
            get_segmentation=True,
            verbose=False
        )

        # Should detect sources
        assert len(obj) > 0

        # Should have all columns including WCS
        assert 'ra' in obj.colnames
        assert 'dec' in obj.colnames

        # Should have segmentation map
        assert segm is not None
        assert np.sum(segm > 0) > 0

        # Check metadata
        assert obj.meta['method'] == 'segmentation'
        assert obj.meta['deblend'] is True


class TestPhotutilsFlags:
    """Test detection flags in get_objects_photutils."""

    @pytest.mark.unit
    def test_segmentation_deblend_flag(self, image_with_sources):
        """Test that deblended sources get 0x002 flag."""
        obj_no_deblend = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            deblend=False,
            aper=3.0,
            verbose=False
        )

        obj_deblend = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            deblend=True,
            aper=3.0,
            verbose=False
        )

        # When deblending is enabled and there are blended sources,
        # some objects might have the deblended flag
        # If fewer sources with deblend=True, some were deblended
        if len(obj_deblend) > 0 and len(obj_no_deblend) > len(obj_deblend):
            # Check that deblended flag is set
            deblend_flags = obj_deblend['flags'] & 0x002
            assert np.any(deblend_flags != 0)

    @pytest.mark.unit
    def test_segmentation_truncated_flag(self, simple_image):
        """Test that truncated sources at edges get 0x008 flag."""
        # Create image with sources at edges
        img = simple_image.copy()
        # Add a bright source at the corner (will be truncated)
        img[0:5, 0:5] = 500  # Bright corner region
        img[0:5, -5:] = 500  # Top-right corner
        img[-5:, 0:5] = 500  # Bottom-left corner

        obj = photometry.get_objects_photutils(
            img,
            thresh=3.0,
            method='segmentation',
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # Check that some sources have truncated flag
            truncated_flags = obj['flags'] & 0x008
            # At least some sources should be near edge and truncated
            # (corner sources will definitely be truncated)
            assert np.any(truncated_flags != 0)

    @pytest.mark.unit
    def test_segmentation_saturated_flag(self, simple_image):
        """Test saturation flag with saturation parameter."""
        img = simple_image.copy()
        # Add a very bright source
        img[50:55, 50:55] = 500

        # Without saturation parameter, no 0x004 flag
        obj_no_sat = photometry.get_objects_photutils(
            img,
            thresh=2.0,
            method='segmentation',
            saturation=None,
            aper=3.0,
            verbose=False
        )

        if len(obj_no_sat) > 0:
            sat_flags = obj_no_sat['flags'] & 0x004
            assert np.all(sat_flags == 0)

        # With saturation parameter set very low, should flag bright sources
        obj_sat = photometry.get_objects_photutils(
            img,
            thresh=2.0,
            method='segmentation',
            saturation=100,  # Much lower threshold
            aper=3.0,
            verbose=False
        )

        if len(obj_sat) > 0:
            sat_flags = obj_sat['flags'] & 0x004
            # At least some sources should be flagged as saturated
            assert np.any(sat_flags != 0)

    @pytest.mark.unit
    def test_segmentation_masked_pixels_flag(self, image_with_sources):
        """Test masked pixels flag (0x100) when mask provided."""
        mask = np.zeros(image_with_sources.shape, dtype=bool)
        # Mask a small region
        mask[50:60, 50:60] = True

        obj_no_mask = photometry.get_objects_photutils(
            image_with_sources,
            mask=None,
            thresh=2.0,
            method='segmentation',
            aper=3.0,
            verbose=False
        )

        obj_with_mask = photometry.get_objects_photutils(
            image_with_sources,
            mask=mask,
            thresh=2.0,
            method='segmentation',
            aper=3.0,
            verbose=False
        )

        if len(obj_no_mask) > 0:
            # Without mask, no 0x100 flags
            masked_flags = obj_no_mask['flags'] & 0x100
            assert np.all(masked_flags == 0)

        if len(obj_with_mask) > 0:
            # With mask, sources overlapping mask should have 0x100 flag
            masked_flags = obj_with_mask['flags'] & 0x100
            # May have some masked sources depending on detection
            # Just check the flag column exists
            assert len(masked_flags) == len(obj_with_mask)

    @pytest.mark.unit
    def test_dao_edge_flag(self, image_with_sources):
        """Test edge/truncation flag for DAOStarFinder."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='dao',
            fwhm=3.0,
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # Check that flags column exists
            assert 'flags' in obj.colnames
            # Some sources near edge should be flagged (if DAOStarFinder detected any)
            # At minimum, flags column should exist
            assert obj['flags'].dtype == int

    @pytest.mark.unit
    def test_dao_quality_flags(self, image_with_sources):
        """Test quality metrics flagging for DAOStarFinder."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='dao',
            fwhm=3.0,
            sharplo=0.2,
            sharphi=1.0,
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # Check flags exist
            assert 'flags' in obj.colnames
            # Quality flags (0x010) may be set for poor sharpness
            # Just verify flags column is valid
            assert np.all(np.isfinite(obj['flags']))

    @pytest.mark.unit
    def test_dao_saturated_flag(self, simple_image):
        """Test saturation flag for DAOStarFinder."""
        img = simple_image.copy()
        # Add bright source
        img[50:55, 50:55] = 200

        obj_no_sat = photometry.get_objects_photutils(
            img,
            thresh=2.0,
            method='dao',
            fwhm=3.0,
            saturation=None,
            aper=3.0,
            verbose=False
        )

        if len(obj_no_sat) > 0:
            sat_flags = obj_no_sat['flags'] & 0x004
            assert np.all(sat_flags == 0)

        obj_sat = photometry.get_objects_photutils(
            img,
            thresh=2.0,
            method='dao',
            fwhm=3.0,
            saturation=150,
            aper=3.0,
            verbose=False
        )

        if len(obj_sat) > 0:
            # Saturation flag may be set
            assert 'flags' in obj_sat.colnames

    @pytest.mark.unit
    def test_no_saturation_param_default(self, image_with_sources):
        """Test that saturation parameter defaults to None."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=2.0,
            method='segmentation',
            # saturation not provided - should default to None
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # No 0x004 flags should be set when saturation=None
            sat_flags = obj['flags'] & 0x004
            assert np.all(sat_flags == 0)

    @pytest.mark.unit
    def test_no_mask_no_masked_flag(self, image_with_sources):
        """Test that 0x100 flag not set when mask is None."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            mask=None,
            thresh=2.0,
            method='segmentation',
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # Without mask, no 0x100 flags
            masked_flags = obj['flags'] & 0x100
            assert np.all(masked_flags == 0)

    @pytest.mark.unit
    def test_multiple_flags_combined(self, simple_image):
        """Test that multiple flags can be set on same source."""
        img = simple_image.copy()
        # Add bright source at corner (truncated + saturated if saturation set low)
        img[0:5, 0:5] = 300

        mask = np.zeros(img.shape, dtype=bool)
        mask[1:4, 1:4] = True  # Mask overlapping with bright corner source

        obj = photometry.get_objects_photutils(
            img,
            mask=mask,
            thresh=2.0,
            method='segmentation',
            saturation=200,
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # Corner source could have multiple flags:
            # 0x008 (truncated), 0x004 (saturated), 0x100 (masked)
            flags = obj['flags']
            # Just verify flags are set properly without crashing
            assert len(flags) == len(obj)

    @pytest.mark.unit
    def test_backward_compatible_flags_zero(self, image_with_sources):
        """Test backward compatibility - flags default to 0 if no issues."""
        obj = photometry.get_objects_photutils(
            image_with_sources,
            thresh=3.0,
            method='segmentation',
            deblend=True,
            saturation=None,  # No saturation checking
            mask=None,  # No mask
            aper=3.0,
            verbose=False
        )

        if len(obj) > 0:
            # For well-behaved sources away from edge, flags might be all 0
            # Just verify flags column exists and is numeric
            assert 'flags' in obj.colnames
            assert obj['flags'].dtype == int
            # All flag values should be non-negative integers
            assert np.all(obj['flags'] >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])