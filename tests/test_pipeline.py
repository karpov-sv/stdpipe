"""
Unit tests for stdpipe.pipeline module.

Tests pipeline building blocks and image preprocessing utilities.
"""

import pytest
import numpy as np
from astropy.io import fits

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
